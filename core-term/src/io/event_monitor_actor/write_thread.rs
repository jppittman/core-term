// src/io/event_monitor_actor/write_thread.rs

//! Internal write thread for EventMonitorActor.
//!
//! This module handles all PTY write operations in a dedicated thread.

use crate::io::pty::NixPty;
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
    /// * `pty_write_rx` - Channel to receive bytes to write to PTY
    pub(super) fn spawn(mut pty: NixPty, pty_write_rx: Receiver<Vec<u8>>) -> Result<Self> {
        let thread_handle = thread::Builder::new()
            .name("pty-write".to_string())
            .spawn(move || {
                if let Err(e) = Self::write_loop(&mut pty, pty_write_rx) {
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
    /// Receives bytes from the app and writes them to the PTY.
    /// Uses timeout to allow graceful shutdown when channel closes.
    fn write_loop(pty: &mut NixPty, pty_write_rx: Receiver<Vec<u8>>) -> Result<()> {
        debug!("PTY write thread starting");

        loop {
            match pty_write_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(data) => {
                    trace!("Write thread: Writing {} bytes to PTY", data.len());
                    pty.write_all(&data).context("Failed to write to PTY")?;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No data available, loop again
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    info!("Write thread: channel disconnected, shutting down");
                    return Ok(());
                }
            }
        }
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
