// src/io/event_monitor_actor/writer.rs

//! PTY writer actor.
//!
//! Owns the primary PTY FD (RAII: dropping it hangs up the child). Bytes for
//! the shell arrive on the data lane; `Resize` arrives on the control lane and
//! therefore preempts any queued bulk writes — the scheduler drains Control
//! before Data by construction.
//!
//! Writes are nonblocking. If the kernel PTY buffer fills (a stopped shell mid
//! large paste), the unwritten tail is queued in `pending` and `handle_os()` waits
//! for `EPOLLOUT` — with the waker registered alongside, so Resize and
//! Shutdown still get through while the queue drains.
//!
//! # Pre-bind buffering
//!
//! The app holds the writer handle before the troupe's `Bind` lands, and a
//! `Resize` (Control) actually outranks `Bind` (Management) in a scheduler
//! wake. So the writer accepts Data and Control while unbound: writes queue in
//! `pending`, and a resize is coalesced into `pending_resize` and applied when
//! `Bind` arrives. (The PTY is already sized at spawn, so a pre-bind resize is
//! only ever a refinement.)

use super::{Directory, WriterControl, WriterManagement};
use crate::io::event::{Event, EventFlags, EventMonitor};
use crate::io::pty::{NixPty, PtyChannel};
use crate::io::waker::FdWaker;
use crate::io::Resize;
use actor_scheduler::{
    Actor, ActorStatus, ActorTypes, HandlerError, HandlerResult, SystemStatus, TroupeActor,
};
use log::*;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::Arc;

const TOKEN_PTY: u64 = 0;
const TOKEN_WAKER: u64 = 1;

pub(super) struct PtyWriter {
    /// PTY + monitor + waker, present once `Bind` has been handled.
    bound: Option<BoundWriter>,
    /// Bytes accepted but not yet written. Accumulates even while unbound.
    pending: VecDeque<Vec<u8>>,
    /// Offset of the first unwritten byte in `pending.front()`.
    cursor: usize,
    /// Latest resize requested while unbound, applied at `Bind`.
    pending_resize: Option<Resize>,
    /// Set on unrecoverable write errors (child gone); further writes drop.
    broken: bool,
}

struct BoundWriter {
    pty: NixPty,
    monitor: EventMonitor,
    waker: Arc<FdWaker>,
    events: Vec<Event>,
}

impl BoundWriter {
    fn bind(pty: NixPty, waker: Arc<FdWaker>) -> anyhow::Result<Self> {
        let monitor = EventMonitor::new()?;
        // Registered permanently; handle_os() only polls while `pending` is
        // non-empty, so level-triggered "always writable" costs nothing.
        monitor.add(&pty, TOKEN_PTY, EventFlags::EPOLLOUT)?;
        monitor.add(&*waker, TOKEN_WAKER, EventFlags::EPOLLIN)?;
        Ok(Self {
            pty,
            monitor,
            waker,
            events: Vec::with_capacity(16),
        })
    }
}

impl PtyWriter {
    /// Write queued bytes until done or the kernel buffer is full. No-op while
    /// unbound (bytes stay queued for the post-`Bind` flush).
    fn flush_pending(&mut self) {
        let Some(bound) = self.bound.as_mut() else {
            return;
        };
        while let Some(front) = self.pending.front() {
            match bound.pty.write(&front[self.cursor..]) {
                Ok(n) => {
                    self.cursor += n;
                    if self.cursor == front.len() {
                        self.pending.pop_front();
                        self.cursor = 0;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return,
                Err(e) => {
                    error!(
                        "PTY write error, dropping {} queued buffers: {}",
                        self.pending.len(),
                        e
                    );
                    self.pending.clear();
                    self.cursor = 0;
                    self.broken = true;
                    return;
                }
            }
        }
    }

    fn apply_resize(&mut self, resize: Resize) {
        let Some(bound) = self.bound.as_ref() else {
            // Not bound yet: remember the latest requested size.
            self.pending_resize = Some(resize);
            return;
        };
        // Resize failures are non-fatal (child may have exited).
        if let Err(e) = bound.pty.resize(resize.cols, resize.rows) {
            warn!(
                "Failed to resize PTY to {}x{}: {}",
                resize.cols, resize.rows, e
            );
        } else {
            debug!("PTY resized to {}x{}", resize.cols, resize.rows);
        }
    }
}

impl ActorTypes for PtyWriter {
    type Data = Vec<u8>;
    type Control = WriterControl;
    type Management = WriterManagement;
}

impl TroupeActor<Directory> for PtyWriter {
    fn new(_dir: Directory) -> Self {
        Self {
            bound: None,
            pending: VecDeque::new(),
            cursor: 0,
            pending_resize: None,
            broken: false,
        }
    }
}

impl Actor<Vec<u8>, WriterControl, WriterManagement> for PtyWriter {
    fn handle_data(&mut self, bytes: Vec<u8>) -> HandlerResult {
        if self.broken || bytes.is_empty() {
            return Ok(());
        }
        self.pending.push_back(bytes);
        // Common case (bound, queue was empty, PTY writable): flushes inline
        // and handle_os() never needs to poll. Unbound: stays queued.
        self.flush_pending();
        Ok(())
    }

    fn handle_control(&mut self, msg: WriterControl) -> HandlerResult {
        let WriterControl::Resize(resize) = msg;
        self.apply_resize(resize);
        Ok(())
    }

    fn handle_management(&mut self, msg: WriterManagement) -> HandlerResult {
        let WriterManagement::Bind { pty, waker } = msg;
        match BoundWriter::bind(pty, waker) {
            Ok(bound) => self.bound = Some(bound),
            Err(e) => {
                return Err(HandlerError::recoverable(format!(
                    "PTY writer bind failed: {e}"
                )))
            }
        }
        // Apply anything queued before the bind.
        if let Some(resize) = self.pending_resize.take() {
            self.apply_resize(resize);
        }
        self.flush_pending();
        Ok(())
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        if self.broken || self.pending.is_empty() {
            // Message-driven: let the scheduler block on the doorbell. With
            // the waker unarmed, sends to this actor cost no wake syscall —
            // this is the common state (keystrokes flush inline).
            return Ok(ActorStatus::Idle);
        }
        let Some(bound) = self.bound.as_mut() else {
            return Ok(ActorStatus::Idle); // unbound: nothing to poll on yet
        };

        // Kernel buffer was full; wait until it drains or a message arrives.
        if !bound.waker.arm() {
            return Ok(ActorStatus::Busy); // raced-in messages: drain first
        }

        let poll = bound.monitor.events(&mut bound.events, -1);
        bound.waker.disarm();
        poll.map_err(|e| HandlerError::recoverable(format!("PTY writer poll failed: {e}")))?;

        let mut pty_writable = false;
        for event in &bound.events {
            match event.token {
                TOKEN_WAKER => {} // already drained by disarm()
                TOKEN_PTY => pty_writable = true,
                other => debug!("PTY writer: unexpected poll token {}", other),
            }
        }

        if pty_writable {
            self.flush_pending();
        }

        Ok(ActorStatus::Busy)
    }
}

// Coverage for pre-bind write queueing, pre-bind resize coalescing, and
// bound writes/resizes lives in `event_monitor_actor::tests`, exercised
// through the real actor message-passing surface (`PtyTroupe`/`writer_handle`)
// rather than by constructing this private struct directly.
