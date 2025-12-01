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

use crate::ansi::AnsiCommand;
use crate::io::pty::NixPty;
use anyhow::{Context, Result};
use log::*;
use read_thread::ReadThread;
use std::sync::mpsc::{Receiver, SyncSender};
use write_thread::WriteThread;

/// EventMonitor actor that manages PTY I/O across two dedicated threads.
///
/// Internally spawns:
/// - Read thread: Polls PTY for data, parses ANSI, sends to app
/// - Write thread: Receives write commands, executes on PTY
///
/// External callers see a single unified actor.
pub struct EventMonitorActor {
    read_thread: Option<ReadThread>,
    write_thread: Option<WriteThread>,
}

impl EventMonitorActor {
    /// Spawns the EventMonitor actor with two dedicated I/O threads.
    ///
    /// # Arguments
    ///
    /// * `pty` - The PTY to monitor (owned by write thread)
    /// * `pty_cmd_tx` - Channel to send parsed ANSI commands to app
    /// * `pty_write_rx` - Channel to receive bytes to write to PTY
    ///
    /// # Returns
    ///
    /// Returns `Self` (handle to both threads for cleanup)
    pub fn spawn(
        pty: NixPty,
        pty_cmd_tx: SyncSender<Vec<AnsiCommand>>,
        pty_write_rx: Receiver<Vec<u8>>,
    ) -> Result<Self> {
        // Clone PTY for the read thread (shared ownership of FD)
        let pty_read = pty
            .try_clone()
            .context("Failed to clone PTY for read thread")?;

        // Spawn read thread (uses independent PTY clone)
        let read_thread =
            ReadThread::spawn(pty_read, pty_cmd_tx).context("Failed to spawn PTY read thread")?;

        // Spawn write thread (owns primary PTY for writes and lifecycle management)
        let write_thread =
            WriteThread::spawn(pty, pty_write_rx).context("Failed to spawn PTY write thread")?;

        info!("EventMonitorActor spawned with read and write threads");

        Ok(Self {
            read_thread: Some(read_thread),
            write_thread: Some(write_thread),
        })
    }
}

impl Drop for EventMonitorActor {
    fn drop(&mut self) {
        debug!("EventMonitorActor dropped, cleaning up I/O threads");

        // Drop write thread first (owns the PTY, will close the FD)
        if let Some(write_thread) = self.write_thread.take() {
            drop(write_thread);
        }

        // Then drop read thread (will exit when PTY FD is closed)
        if let Some(read_thread) = self.read_thread.take() {
            drop(read_thread);
        }

        debug!("EventMonitorActor cleanup complete");
    }
}
