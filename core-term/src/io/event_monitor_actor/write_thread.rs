// src/io/event_monitor_actor/write_thread.rs

//! Internal write thread for EventMonitorActor.
//!
//! This module handles all PTY write operations and resize commands
//! in a dedicated thread.

use crate::io::pty::PtyWriter;
use crate::platform::actions::PlatformAction;
use anyhow::{Context, Result};
use log::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

/// Main write loop.
///
/// Polls for commands from the orchestrator with a timeout, then executes
/// write or resize operations on the PTY. Uses timeout to allow graceful shutdown.
pub(super) fn run_write_loop(
    mut pty: PtyWriter<'_>,
    command_rx: Receiver<PlatformAction>,
    stop_signal: Arc<AtomicBool>,
) -> Result<()> {
    debug!("PTY write thread starting");

    loop {
        if stop_signal.load(Ordering::Relaxed) {
            debug!("Write thread received stop signal");
            break;
        }

        match command_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(action) => {
                if !handle_command(&mut pty, action)? {
                    // Shutdown requested
                    debug!("Write thread received shutdown signal from command");
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
    Ok(())
}

/// Handles a single command.
///
/// Returns `false` if the thread should shut down, `true` otherwise.
fn handle_command(pty: &mut PtyWriter<'_>, action: PlatformAction) -> Result<bool> {
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
