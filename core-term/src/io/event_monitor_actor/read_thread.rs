// src/io/event_monitor_actor/read_thread.rs

//! Internal read thread for EventMonitorActor.
//!
//! This module handles all PTY read operations in a dedicated thread,
//! using kqueue/epoll for efficient event-driven I/O.

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
#[cfg(target_os = "linux")]
use crate::io::event::{EpollFlags as KqueueFlags, EventMonitor};
#[cfg(target_os = "macos")]
use crate::io::event::{EventMonitor, KqueueFlags};
use crate::io::traits::EventSource;
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::thread::{self, JoinHandle};

/// Token for identifying PTY events in the EventMonitor.
const PTY_TOKEN: u64 = 1;

/// Base buffer size for PTY reads (4KB).
const PTY_BASE_BUFFER_SIZE: usize = 4096;

/// Multiplier for read buffer size to handle bursts of output.
const PTY_READ_BUFFER_SIZE_MULTIPLIER: usize = 4;

/// Total buffer size for PTY reads (16KB).
const PTY_READ_BUFFER_SIZE: usize = PTY_READ_BUFFER_SIZE_MULTIPLIER * PTY_BASE_BUFFER_SIZE;

/// Maximum commands per IOEvent to allow rendering to interleave.
/// Small enough to prevent stalling rendering, large enough to be efficient.
const MAX_COMMANDS_PER_IOEVENT: usize = 1000;

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
    /// * `pty_cmd_tx` - Channel to send parsed ANSI commands to app
    pub(super) fn spawn<S>(source: S, pty_cmd_tx: SyncSender<Vec<AnsiCommand>>) -> Result<Self>
    where
        S: EventSource + 'static,
    {
        let thread_handle = thread::Builder::new()
            .name("pty-read".to_string())
            .spawn(move || {
                if let Err(e) = Self::read_loop(source, pty_cmd_tx) {
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
    /// Polls the PTY for read events, reads data, parses ANSI commands,
    /// and sends them to the app. Engine polls at vsync rate, no doorbell needed.
    fn read_loop<S: EventSource>(
        mut source: S,
        pty_cmd_tx: SyncSender<Vec<AnsiCommand>>,
    ) -> Result<()> {
        let fd = source.as_raw_fd();
        debug!("PTY read thread starting (fd: {})", fd);

        let event_monitor =
            EventMonitor::new().context("Failed to create EventMonitor in read thread")?;

        event_monitor
            .add(&source, PTY_TOKEN, KqueueFlags::EPOLLIN)
            .context("Failed to register PTY with EventMonitor")?;

        debug!("Read thread registered PTY fd {} for EPOLLIN", fd);

        let mut ansi_parser = AnsiProcessor::new();
        let mut events_buffer = Vec::with_capacity(8);
        let mut read_buffer = vec![0u8; PTY_READ_BUFFER_SIZE];

        // Buffer for accumulating commands when channel is full
        let mut command_buffer: Vec<AnsiCommand> = Vec::new();

        loop {
            // Poll for PTY events with a 100ms timeout
            event_monitor
                .events(&mut events_buffer, 100)
                .context("EventMonitor polling failed")?;

            // Process PTY read events
            for event in &events_buffer {
                if event.token == PTY_TOKEN && event.flags.contains(KqueueFlags::EPOLLIN) {
                    let should_continue = Self::handle_pty_readable(
                        &mut source,
                        &mut read_buffer,
                        &mut ansi_parser,
                        &pty_cmd_tx,
                        &mut command_buffer,
                    )?;
                    
                    if !should_continue {
                        info!("Read thread: PTY closed, stopping read loop");
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Handles PTY readable events.
    /// Returns Ok(true) to continue, Ok(false) to stop (EOF).
    fn handle_pty_readable<S: EventSource>(
        source: &mut S,
        read_buffer: &mut [u8],
        ansi_parser: &mut AnsiProcessor,
        pty_cmd_tx: &SyncSender<Vec<AnsiCommand>>,
        command_buffer: &mut Vec<AnsiCommand>,
    ) -> Result<bool> {
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

            trace!("Read thread: Read {} bytes from PTY", bytes_read);

            // Parse bytes into ANSI commands
            let mut ansi_commands = ansi_parser.process_bytes(&read_buffer[..bytes_read]);

            if !ansi_commands.is_empty() {
                debug!("Read thread: Parsed {} ANSI commands", ansi_commands.len());

                // Add new commands to buffer
                command_buffer.append(&mut ansi_commands);

                // Try to send buffered commands in chunks using try_send
                while command_buffer.len() >= MAX_COMMANDS_PER_IOEVENT {
                    // Take a chunk from the buffer
                    let chunk: Vec<_> = command_buffer.drain(..MAX_COMMANDS_PER_IOEVENT).collect();

                    match pty_cmd_tx.try_send(chunk.clone()) {
                        Ok(()) => {
                            trace!("Read thread: Sent {} commands to app", chunk.len());
                        }
                        Err(TrySendError::Full(_)) => {
                            // Channel is full, put chunk back and stop trying
                            debug!(
                                "Read thread: Channel full, keeping {} buffered commands",
                                chunk.len() + command_buffer.len()
                            );
                            command_buffer.splice(0..0, chunk);
                            break;
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            return Err(anyhow::anyhow!("App command channel disconnected"));
                        }
                    }
                }

                // If buffer is too large, force a few blocking sends to drain it
                if command_buffer.len() > 10000 {
                    warn!(
                        "Read thread: Command buffer exceeded 10,000 commands ({}), forcing blocking send",
                        command_buffer.len()
                    );
                    // Send a few chunks with blocking send (not all - app might catch up)
                    let chunks_to_force = 3; // Send 3 chunks (~3000 commands) to give app a chance
                    for _ in 0..chunks_to_force {
                        if command_buffer.is_empty() {
                            break;
                        }
                        let chunk_size = command_buffer.len().min(MAX_COMMANDS_PER_IOEVENT);
                        let chunk: Vec<_> = command_buffer.drain(..chunk_size).collect();
                        pty_cmd_tx
                            .send(chunk)
                            .context("Failed to send buffered commands")?;
                    }
                }
            }
        }

        // Try to flush buffer if we have any pending commands
        if !command_buffer.is_empty() {
            match pty_cmd_tx.try_send(std::mem::take(command_buffer)) {
                Ok(()) => {
                    trace!("Read thread: Flushed command buffer successfully");
                }
                Err(TrySendError::Full(commands)) => {
                    // Put commands back in buffer
                    *command_buffer = commands;
                    debug!(
                        "Read thread: Channel still full, keeping {} commands in buffer",
                        command_buffer.len()
                    );
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(anyhow::anyhow!("App command channel disconnected"));
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
