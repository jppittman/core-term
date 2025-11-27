// src/io/event_monitor_actor/write_thread.rs

//! Internal write thread for EventMonitorActor.
//!
//! This module handles all PTY write operations and resize commands
//! in a dedicated thread.

use crate::io::pty::{NixPty, PtyChannel};
use crate::platform::actions::PlatformAction;
use anyhow::{Context, Result};
use log::*;
use std::io::Write;
use std::sync::mpsc::Receiver;
use std::thread::{self, JoinHandle};

/// Internal write thread handle.
pub(super) struct WriteThread {
    thread_handle: Option<JoinHandle<()>>,
}

impl WriteThread {
    /// Spawns the write thread.
    ///
    /// # Arguments
    ///
    /// * `pty` - The PTY (write thread owns it for proper FD lifecycle)
    /// * `command_rx` - Channel to receive write/resize commands
    pub(super) fn spawn(mut pty: NixPty, command_rx: Receiver<PlatformAction>) -> Result<Self> {
        let thread_handle = thread::Builder::new()
            .name("pty-write".to_string())
            .spawn(move || {
                if let Err(e) = Self::write_loop(&mut pty, command_rx) {
                    error!("PTY write thread error: {:#}", e);
                }
            })
            .context("Failed to spawn PTY write thread")?;

        debug!("PTY write thread spawned");

        Ok(Self {
            thread_handle: Some(thread_handle),
        })
    }

    /// Main write loop.
    ///
    /// Polls for commands from the orchestrator with a timeout, then executes
    /// write or resize operations on the PTY. Uses timeout to allow graceful shutdown.
    fn write_loop(pty: &mut NixPty, command_rx: Receiver<PlatformAction>) -> Result<()> {
        debug!("PTY write thread starting");

        loop {
            match command_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(action) => {
                    if !Self::handle_command(pty, action)? {
                        // Shutdown requested
                        debug!("Write thread received shutdown signal");
                        return Ok(());
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No commands available, loop again
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    info!("Write thread: command channel disconnected, shutting down");
                    return Ok(());
                }
            }
        }
    }

    /// Handles a single command.
    ///
    /// Returns `false` if the thread should shut down, `true` otherwise.
    fn handle_command(pty: &mut NixPty, action: PlatformAction) -> Result<bool> {
        match action {
            PlatformAction::Write(data) => {
                trace!("Write thread: Writing {} bytes to PTY", data.len());
                pty.write_all(&data).context("Failed to write to PTY")?;
            }
            PlatformAction::ResizePty { cols, rows } => {
                debug!("Write thread: Resizing PTY to {}x{}", cols, rows);
                pty.resize(cols, rows).context("Failed to resize PTY")?;
            }
            _ => {
                // Other actions should not be sent to this thread
                warn!(
                    "Write thread: Received unexpected action (ignoring): {:?}",
                    action
                );
            }
        }
        Ok(true)
    }
}

impl Drop for WriteThread {
    fn drop(&mut self) {
        debug!("WriteThread dropped");
        // NixPty will be dropped here, cleaning up the FD and child process
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                error!("Write thread panicked: {:?}", e);
            }
        }
    }
}
