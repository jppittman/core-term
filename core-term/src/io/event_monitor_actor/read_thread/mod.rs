// src/io/event_monitor_actor/read_thread/mod.rs

use crate::io::pty::NixPty;
use crate::io::event_monitor_actor::parser_thread::{NoControl, NoManagement};
use actor_scheduler::{ActorHandle, Message};
use log::*;
use std::io::Read;
use std::thread::JoinHandle;

pub struct ReadThread {
    join_handle: Option<JoinHandle<()>>,
}

impl ReadThread {
    /// Spawns the read thread.
    ///
    /// # Arguments
    ///
    /// * `mut pty` - Clone of the PTY for reading
    /// * `parser_tx` - Actor handle to send raw bytes to parser thread
    pub fn spawn(
        mut pty: NixPty,
        parser_tx: ActorHandle<Vec<u8>, NoControl, NoManagement>,
    ) -> anyhow::Result<Self> {
        let handle = std::thread::Builder::new()
            .name("pty-reader".to_string())
            .spawn(move || {
                debug!("Read thread started");
                let mut buf = [0u8; 4096];

                loop {
                    // Blocking read on PTY (or use epoll/kqueue for better efficiency)
                    // For now, simple blocking read is fine as this is a dedicated thread.
                    match pty.read(&mut buf) {
                        Ok(0) => {
                            info!("PTY returned EOF");
                            break;
                        }
                        Ok(n) => {
                            let data = buf[..n].to_vec();
                            // Send to parser actor
                            if let Err(e) = parser_tx.send(Message::Data(data)) {
                                warn!("Failed to send bytes to parser (channel closed): {}", e);
                                break;
                            }
                        }
                        Err(e) => {
                            error!("PTY read error: {}", e);
                            break;
                        }
                    }
                }
                debug!("Read thread exited");
            })?;

        Ok(Self {
            join_handle: Some(handle),
        })
    }
}

impl Drop for ReadThread {
    fn drop(&mut self) {
        // We cannot easily interrupt the blocking read unless we close the FD.
        // The WriteThread owns the PTY and will close it on drop, which should
        // wake up this read().
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}
