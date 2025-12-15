// src/io/event_monitor_actor/write_thread/mod.rs

use crate::io::pty::NixPty;
use log::*;
use std::io::Write;
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

pub struct WriteThread {
    join_handle: Option<JoinHandle<()>>,
}

impl WriteThread {
    /// Spawns the write thread.
    ///
    /// # Arguments
    ///
    /// * `mut pty` - The primary PTY handle (will be closed when thread exits)
    /// * `rx` - Channel to receive data to write
    pub fn spawn(mut pty: NixPty, rx: Receiver<Vec<u8>>) -> anyhow::Result<Self> {
        let handle = std::thread::Builder::new()
            .name("pty-writer".to_string())
            .spawn(move || {
                debug!("Write thread started");
                while let Ok(data) = rx.recv() {
                    if let Err(e) = pty.write_all(&data) {
                        error!("Failed to write to PTY: {}", e);
                        break;
                    }
                    if let Err(e) = pty.flush() {
                        error!("Failed to flush PTY: {}", e);
                        break;
                    }
                }
                debug!("Write thread exited - closing PTY");
                // PTY is dropped here, closing the FD
            })?;

        Ok(Self {
            join_handle: Some(handle),
        })
    }
}

impl Drop for WriteThread {
    fn drop(&mut self) {
        // Closing the channel (which happens when SyncSender is dropped by owner)
        // will cause the loop to exit.
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}
