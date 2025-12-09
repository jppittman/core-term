// src/io/event_monitor_actor/mod.rs

//! Platform-agnostic PTY I/O actor that coordinates read, parse, and write operations.
//!
//! The EventMonitorActor internally spawns three threads:
//! - Read thread: Uses kqueue/epoll for efficient PTY read polling, sends raw bytes to parser
//! - Parser thread: Receives raw bytes, parses ANSI commands, sends to worker
//! - Write thread: Handles PTY writes and resize commands
//!
//! This architecture provides true parallelism: read thread maximizes PTY bandwidth,
//! parser thread handles CPU-intensive parsing, write thread handles writes.
//! External callers see a single unified actor.

mod parser_thread;
mod read_thread;
mod write_thread;

use crate::ansi::AnsiCommand;
use crate::io::pty::NixPty;
use anyhow::{Context, Result};
use log::*;
// ParserThread no longer needed - using actor_scheduler::spawn() instead
use read_thread::ReadThread;
use std::sync::mpsc::{Receiver, SyncSender};
use write_thread::WriteThread;

/// EventMonitor actor that manages PTY I/O across three dedicated threads.
///
/// Internally spawns:
/// - Read thread: Polls PTY for data, sends raw bytes to parser
/// - Parser thread: Parses ANSI commands, sends to worker
/// - Write thread: Receives write commands, executes on PTY
///
/// External callers see a single unified actor.
pub struct EventMonitorActor {
    read_thread: Option<ReadThread>,
    write_thread: Option<WriteThread>,
    // Parser actor is spawned and managed by actor-scheduler, no need to track it
}

impl EventMonitorActor {
    /// Spawns the EventMonitor actor with three dedicated threads.
    ///
    /// # Arguments
    ///
    /// * `pty` - The PTY to monitor (owned by write thread)
    /// * `cmd_tx` - Channel to send parsed ANSI commands to app
    /// * `pty_write_rx` - Channel to receive bytes to write to PTY
    ///
    /// # Returns
    ///
    /// Returns `Self` (handle to all threads for cleanup)
    pub fn spawn(
        pty: NixPty,
        cmd_tx: SyncSender<Vec<AnsiCommand>>,
        pty_write_rx: Receiver<Vec<u8>>,
    ) -> Result<Self> {
        // Create parser actor and spawn it using new API
        let parser_state = parser_thread::ParserState::new(cmd_tx);
        let parser_tx = actor_scheduler::spawn_with_config(
            parser_state,
            10,  // burst limit: max 10 byte batches per wake
            64,  // buffer size: 64 byte batches
            None,
        );

        // Clone PTY for the read thread (shared ownership of FD)
        let pty_read = pty
            .try_clone()
            .context("Failed to clone PTY for read thread")?;

        // Spawn read thread: PTY → sends raw bytes to parser via ActorHandle
        let read_thread =
            ReadThread::spawn(pty_read, parser_tx).context("Failed to spawn PTY read thread")?;

        // Spawn write thread (owns primary PTY for writes and lifecycle management)
        let write_thread =
            WriteThread::spawn(pty, pty_write_rx).context("Failed to spawn PTY write thread")?;

        info!("EventMonitorActor spawned with read, parser, and write threads");

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

        // Then drop read thread (will exit when PTY FD is closed, closes parser_tx handle)
        if let Some(read_thread) = self.read_thread.take() {
            drop(read_thread);
        }

        // Parser actor will exit when all handles are dropped (managed by actor-scheduler)

        debug!("EventMonitorActor cleanup complete");
    }
}
