// src/io/event_monitor_actor/mod.rs

//! Platform-agnostic PTY I/O actor that coordinates read and write operations.
//!
//! The EventMonitorActor internally spawns two threads:
//! - Read thread: Uses kqueue/epoll for efficient PTY read polling and ANSI parsing
//! - Write thread: Handles PTY writes and resize commands
//!
//! This architecture provides true parallelism between read and write operations,
//! preventing blocking writes from delaying read polling. The external API remains
//! unchanged - callers see a single EventMonitorActor.

mod read_thread;
mod write_thread;

use crate::io::pty::{NixPty, PtyReader, PtyWriter, PtyChannel};
use crate::orchestrator::OrchestratorSender;
use crate::platform::actions::PlatformAction;
use anyhow::{Context, Result};
use log::*;
use read_thread::run_read_loop;
use std::os::unix::io::AsFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use write_thread::run_write_loop;

/// EventMonitor actor that manages PTY I/O across two dedicated threads.
///
/// Internally spawns a supervisor thread which uses scoped threads to run:
/// - Read thread: Polls PTY for data, parses ANSI, sends to orchestrator
/// - Write thread: Receives write/resize commands, executes on PTY
///
/// External callers see a single unified actor.
pub struct EventMonitorActor {
    supervisor_handle: Option<JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,
}

impl EventMonitorActor {
    /// Spawns the EventMonitor actor with two dedicated I/O threads.
    ///
    /// # Arguments
    ///
    /// * `pty` - The PTY to monitor (owned by supervisor thread)
    /// * `orchestrator_tx` - Channel to send events to Orchestrator (used by read thread)
    /// * `pty_action_rx` - Channel to receive PlatformActions (used by write thread)
    ///
    /// # Returns
    ///
    /// Returns `Self` (handle to both threads for cleanup)
    pub fn spawn(
        pty: NixPty,
        orchestrator_tx: OrchestratorSender,
        pty_action_rx: std::sync::mpsc::Receiver<PlatformAction>,
    ) -> Result<Self> {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_signal_clone = stop_signal.clone();

        let supervisor_handle = thread::Builder::new()
            .name("io-supervisor".to_string())
            .spawn(move || {
                let pty_ref = &pty;
                let stop_signal_read = stop_signal_clone.clone();
                let stop_signal_write = stop_signal_clone.clone();

                thread::scope(|s| {
                    // Spawn read thread
                    s.spawn(move || {
                        // Create PtyReader borrowing the PTY FD
                        let reader = PtyReader::new(pty_ref.as_fd());
                        if let Err(e) = run_read_loop(reader, orchestrator_tx, stop_signal_read) {
                            error!("PTY read thread error: {:#}", e);
                        }
                    });

                    // Spawn write thread
                    s.spawn(move || {
                        // Create PtyWriter borrowing the PTY FD
                        let writer = PtyWriter::new(pty_ref.as_fd(), Some(pty_ref.child_pid()));
                        if let Err(e) = run_write_loop(writer, pty_action_rx, stop_signal_write) {
                            error!("PTY write thread error: {:#}", e);
                        }
                    });
                });

                // When scope returns, threads are joined.
                // pty is dropped here, closing the FD.
                debug!("IO supervisor thread finished, PTY dropped");
            })
            .context("Failed to spawn IO supervisor thread")?;

        info!("EventMonitorActor spawned with IO supervisor");

        Ok(Self {
            supervisor_handle: Some(supervisor_handle),
            stop_signal,
        })
    }
}

impl Drop for EventMonitorActor {
    fn drop(&mut self) {
        debug!("EventMonitorActor dropped, signaling threads to stop");

        // Signal threads to stop
        self.stop_signal.store(true, Ordering::Relaxed);

        // Join supervisor thread
        if let Some(handle) = self.supervisor_handle.take() {
            if let Err(e) = handle.join() {
                error!("IO supervisor thread panicked: {:?}", e);
            }
        }

        debug!("EventMonitorActor cleanup complete");
    }
}
