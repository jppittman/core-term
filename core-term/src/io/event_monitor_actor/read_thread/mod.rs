// src/io/event_monitor_actor/read_thread/mod.rs

use crate::io::event_monitor_actor::parser_thread::{NoControl, NoManagement};
use crate::io::pty::NixPty;
use actor_scheduler::{ActorHandle, Message};
use log::*;
use std::io::Read;
use std::sync::mpsc::Receiver;
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
    /// * `recycler_rx` - Receiver for recycled buffers from parser thread
    pub fn spawn(
        mut pty: NixPty,
        parser_tx: ActorHandle<Vec<u8>, NoControl, NoManagement>,
        recycler_rx: Receiver<Vec<u8>>,
    ) -> anyhow::Result<Self> {
        let handle = std::thread::Builder::new()
            .name("pty-reader".to_string())
            .spawn(move || {
                debug!("Read thread started");
                eprintln!("DEBUG: Read thread started!");
                let mut buf = [0u8; 4096];
                let mut recycled_bufs: Vec<Vec<u8>> = Vec::with_capacity(8);

                // Create EventMonitor
                let monitor = match crate::io::event::EventMonitor::new() {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Failed to create EventMonitor for PTY read thread: {}", e);
                        return;
                    }
                };

                // Register PTY FD
                #[cfg(target_os = "macos")]
                {
                    use crate::io::event::KqueueFlags;
                    if let Err(e) = monitor.add(&pty, 0, KqueueFlags::EPOLLIN) {
                        error!("Failed to register PTY fd with kqueue: {}", e);
                        return;
                    }
                }

                #[cfg(target_os = "linux")]
                {
                    use crate::io::event::EpollFlags;
                    if let Err(e) = monitor.add(&pty, 0, EpollFlags::EPOLLIN) {
                        error!("Failed to register PTY fd with epoll: {}", e);
                        return;
                    }
                }

                #[cfg(target_os = "macos")]
                let mut events = Vec::<crate::io::event::KqueueEvent>::with_capacity(16);
                #[cfg(target_os = "linux")]
                let mut events = Vec::<crate::io::event::EpollEvent>::with_capacity(16);

                loop {
                    // Wait for events (infinite timeout)
                    if let Err(e) = monitor.events(&mut events, -1) {
                        error!("EventMonitor polling error: {}", e);
                        break;
                    }

                    // If we woke up, try to read
                    // We don't strictly need to check which event it was because we only registered one.
                    // But we should loop through events if we had multiple sources.

                    // Check if PTY is readable
                    match pty.read(&mut buf) {
                        Ok(0) => {
                            info!("PTY returned EOF");
                            break;
                        }
                        Ok(n) => {
                            // Check for recycled buffers
                            while let Ok(v) = recycler_rx.try_recv() {
                                recycled_bufs.push(v);
                            }

                            // Reuse buffer if available, otherwise allocate new
                            let mut data =
                                recycled_bufs.pop().unwrap_or_else(|| Vec::with_capacity(n));
                            data.clear();
                            data.extend_from_slice(&buf[..n]);

                            // Send to parser actor
                            if let Err(e) = parser_tx.send(Message::Data(data)) {
                                warn!("Failed to send bytes to parser (channel closed): {}", e);
                                break;
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            // Spurious wakeup, just continue
                            continue;
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
            if let Err(panic_payload) = handle.join() {
                if std::thread::panicking() {
                    eprintln!("Read thread panicked (during unwind): {:?}", panic_payload);
                } else {
                    std::panic::resume_unwind(panic_payload);
                }
            }
        }
    }
}
