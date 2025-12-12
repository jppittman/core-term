// src/io/event_monitor_actor/read_thread.rs

//! Internal read thread for EventMonitorActor.
//!
//! This module handles all PTY read operations in a dedicated thread,
//! using kqueue/epoll for efficient event-driven I/O.

use crate::io::event::{EventMonitor, KqueueFlags};
use crate::io::event_monitor_actor::parser_thread::{NoControl, NoManagement};
use crate::io::traits::EventSource;
use actor_scheduler::{ActorHandle, Message};
use anyhow::{Context, Result};
use log::*;
use std::thread::{self, JoinHandle};

/// Token for identifying PTY events in the EventMonitor.
const PTY_TOKEN: u64 = 1;

/// Base buffer size for PTY reads (4KB).
const PTY_BASE_BUFFER_SIZE: usize = 4096;

/// Multiplier for read buffer size to handle bursts of output.
const PTY_READ_BUFFER_SIZE_MULTIPLIER: usize = 4;

/// Total buffer size for PTY reads (16KB).
const PTY_READ_BUFFER_SIZE: usize = PTY_READ_BUFFER_SIZE_MULTIPLIER * PTY_BASE_BUFFER_SIZE;

/// Internal read thread handle.
pub(super) struct ReadThread {
    thread_handle: Option<JoinHandle<()>>,
}

impl ReadThread {
    /// Spawns the read thread.
    ///
    /// # Arguments
    ///
    /// * `source` - The event source (e.g., PTY) to monitor and read from
    /// * `parser_tx` - Actor handle to send raw bytes to parser thread
    pub(super) fn spawn<S>(
        source: S,
        parser_tx: ActorHandle<Vec<u8>, NoControl, NoManagement>,
    ) -> Result<Self>
    where
        S: EventSource + 'static,
    {
        let thread_handle = thread::Builder::new()
            .name("pty-read".to_string())
            .spawn(move || {
                if let Err(e) = Self::read_loop(source, parser_tx) {
                    error!("PTY read thread error: {:#}", e);
                }
            })
            .context("Failed to spawn PTY read thread")?;

        debug!("PTY read thread spawned");

        Ok(Self {
            thread_handle: Some(thread_handle),
        })
    }

    /// Main read loop.
    ///
    /// Polls the PTY for read events, reads raw bytes, and sends them to parser thread via ActorHandle.
    /// Focus is on maximizing PTY read bandwidth - no parsing here!
    fn read_loop<S>(
        mut source: S,
        parser_tx: ActorHandle<Vec<u8>, NoControl, NoManagement>,
    ) -> Result<()>
    where
        S: EventSource,
    {
        let fd = source.as_raw_fd();
        debug!("PTY read thread starting (fd: {})", fd);

        let event_monitor =
            EventMonitor::new().context("Failed to create EventMonitor in read thread")?;

        event_monitor
            .add(&source, PTY_TOKEN, KqueueFlags::EPOLLIN)
            .context("Failed to register PTY with EventMonitor")?;

        debug!("Read thread registered PTY fd {} for EPOLLIN", fd);

        let mut events_buffer = Vec::with_capacity(8);
        let mut read_buffer = vec![0u8; PTY_READ_BUFFER_SIZE];

        loop {
            // Poll for PTY events with a 100ms timeout
            event_monitor
                .events(&mut events_buffer, 100)
                .context("EventMonitor polling failed")?;

            // Process PTY read events
            for event in &events_buffer {
                if event.token == PTY_TOKEN && event.flags.contains(KqueueFlags::EPOLLIN) {
                    let should_continue =
                        Self::handle_pty_readable(&mut source, &mut read_buffer, &parser_tx)?;

                    if !should_continue {
                        info!("Read thread: PTY closed, stopping read loop");
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Handles PTY readable events - drains all available bytes and sends to parser via ActorHandle.
    /// Returns Ok(true) to continue, Ok(false) to stop (EOF).
    fn handle_pty_readable<S>(
        source: &mut S,
        read_buffer: &mut [u8],
        parser_tx: &ActorHandle<Vec<u8>, NoControl, NoManagement>,
    ) -> Result<bool>
    where
        S: EventSource,
    {
        // Accumulate raw bytes - read ALL available data from PTY first
        let mut total_bytes_read = 0;
        let mut raw_data_buffer = Vec::new();

        loop {
            // Read from PTY using safe trait method
            let bytes_read = match source.read(read_buffer) {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No more data available
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                    continue;
                }
                Err(e) => {
                    return Err(e).context("Failed to read from PTY");
                }
            };

            if bytes_read == 0 {
                // EOF - PTY closed
                info!("Read thread: PTY returned EOF");
                return Ok(false);
            }

            // Append raw bytes to buffer (keep reading - no parsing!)
            raw_data_buffer.extend_from_slice(&read_buffer[..bytes_read]);
            total_bytes_read += bytes_read;
        }

        // Send all accumulated bytes to parser thread via Data lane
        if total_bytes_read > 0 {
            trace!(
                "Read thread: Read {} bytes from PTY, sending to parser via Data lane",
                total_bytes_read
            );

            match parser_tx.send(Message::Data(raw_data_buffer)) {
                Ok(()) => {}
                Err(_) => {
                    return Err(anyhow::anyhow!("Parser thread disconnected"));
                }
            }
        }

        Ok(true)
    }
}

impl Drop for ReadThread {
    fn drop(&mut self) {
        debug!("ReadThread dropped");
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                error!("Read thread panicked: {:?}", e);
            }
        }
    }
}
