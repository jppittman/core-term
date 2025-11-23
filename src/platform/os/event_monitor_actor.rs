// src/platform/os/event_monitor_actor.rs

//! Platform-agnostic PTY I/O actor that runs in a dedicated background thread.
//!
//! The EventMonitorActor wraps an EventMonitor (kqueue on macOS, epoll on Linux)
//! and a PTY channel, providing a clean message-passing interface for PTY I/O.
//! This allows the main thread to focus on UI events while PTY I/O happens
//! asynchronously in the background.
//!
//! This actor owns the AnsiProcessor, parsing PTY output into ANSI commands
//! in parallel with the Orchestrator thread for better performance.

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use crate::orchestrator::OrchestratorSender;
use crate::platform::actions::PlatformAction;
use crate::platform::os::event::{EventMonitor, KqueueFlags};
use crate::platform::os::pty::{NixPty, PtyChannel};
use crate::platform::BackendEvent;
use anyhow::{Context, Result};
use log::*;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd;
use std::sync::mpsc::{TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};

/// Token for identifying PTY events in the EventMonitor.
const PTY_TOKEN: u64 = 1;

/// Base buffer size for PTY reads (4KB).
const PTY_BASE_BUFFER_SIZE: usize = 4096;

/// Multiplier for read buffer size to handle bursts of output.
const PTY_READ_BUFFER_SIZE_MULTIPLIER: usize = 4;

/// Total buffer size for PTY reads (16KB).
const PTY_READ_BUFFER_SIZE: usize = PTY_READ_BUFFER_SIZE_MULTIPLIER * PTY_BASE_BUFFER_SIZE;

/// EventMonitor actor that manages PTY I/O in a background thread.
///
/// This actor runs an event loop that:
/// - Polls the PTY file descriptor for readable events
/// - Reads data from the PTY and sends it to the main thread
/// - Receives write/resize commands from the main thread
pub struct EventMonitorActor {
    thread_handle: Option<JoinHandle<()>>,
}

impl EventMonitorActor {
    /// Spawns the EventMonitor actor in a background thread.
    ///
    /// # Arguments
    ///
    /// * `pty` - The PTY to monitor
    /// * `orchestrator_tx` - Unified channel to send events to Orchestrator
    /// * `pty_action_rx` - Channel to receive PlatformActions (Write, ResizePty) from Orchestrator
    ///
    /// # Returns
    ///
    /// Returns `Self` (handle to the actor for cleanup)
    pub fn spawn(
        pty: NixPty,
        orchestrator_tx: OrchestratorSender,
        pty_action_rx: std::sync::mpsc::Receiver<PlatformAction>,
    ) -> Result<Self> {
        let thread_handle = thread::Builder::new()
            .name("event-monitor".to_string())
            .spawn(move || {
                if let Err(e) = Self::actor_thread_main(pty, orchestrator_tx, pty_action_rx) {
                    error!("EventMonitor actor thread error: {:#}", e);
                }
            })
            .context("Failed to spawn EventMonitor actor thread")?;

        info!("EventMonitor actor thread spawned");

        Ok(Self {
            thread_handle: Some(thread_handle),
        })
    }

    /// Main loop for the EventMonitor actor thread.
    ///
    /// This function:
    /// 1. Creates an EventMonitor (kqueue/epoll)
    /// 2. Registers the PTY fd for read events
    /// 3. Creates an AnsiProcessor for parsing PTY output
    /// 4. Polls for events with a timeout
    /// 5. Reads PTY data when available, parses it into ANSI commands
    /// 6. Processes commands from the Orchestrator thread
    fn actor_thread_main(
        mut pty: NixPty,
        orchestrator_tx: OrchestratorSender,
        pty_action_rx: std::sync::mpsc::Receiver<PlatformAction>,
    ) -> Result<()> {
        debug!(
            "EventMonitor actor thread starting (PTY fd: {})",
            pty.as_raw_fd()
        );

        let event_monitor =
            EventMonitor::new().context("Failed to create EventMonitor in actor thread")?;

        let pty_fd = pty.as_raw_fd();
        event_monitor
            .add(pty_fd, PTY_TOKEN, KqueueFlags::EPOLLIN)
            .context("Failed to register PTY with EventMonitor")?;

        debug!(
            "EventMonitor registered PTY fd {} with token {}",
            pty_fd, PTY_TOKEN
        );

        let mut ansi_parser = AnsiProcessor::new();
        let mut events_buffer = Vec::with_capacity(8);
        let mut read_buffer = vec![0u8; PTY_READ_BUFFER_SIZE];

        // Buffer for accumulating commands when channel is full
        let mut command_buffer: Vec<AnsiCommand> = Vec::new();

        loop {
            // Poll for PTY events with a 100ms timeout to allow checking for commands
            event_monitor
                .events(&mut events_buffer, 100)
                .context("EventMonitor polling failed")?;

            // Process PTY events (data available to read)
            for event in &events_buffer {
                if event.token == PTY_TOKEN && event.flags.contains(KqueueFlags::EPOLLIN) {
                    Self::handle_pty_readable(
                        &mut pty,
                        &mut read_buffer,
                        &mut ansi_parser,
                        &orchestrator_tx,
                        &mut command_buffer,
                    )?;
                }
            }

            // Process commands from orchestrator thread (non-blocking)
            loop {
                match pty_action_rx.try_recv() {
                    Ok(action) => {
                        if !Self::handle_command(&mut pty, action)? {
                            // Command returned false = shutdown requested
                            debug!("EventMonitor actor received shutdown signal");
                            return Ok(());
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        info!("EventMonitor actor: command channel disconnected, shutting down");
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Handles PTY readable events by reading data, parsing ANSI, and sending to Orchestrator.
    /// Uses try_send with buffering to avoid blocking when the orchestrator channel is full.
    fn handle_pty_readable(
        pty: &mut NixPty,
        read_buffer: &mut [u8],
        ansi_parser: &mut AnsiProcessor,
        orchestrator_tx: &OrchestratorSender,
        command_buffer: &mut Vec<AnsiCommand>,
    ) -> Result<()> {
        loop {
            match pty.read(read_buffer) {
                Ok(0) => {
                    // EOF - PTY closed
                    info!("EventMonitor: PTY returned EOF, sending CloseRequested");
                    orchestrator_tx
                        .send(BackendEvent::CloseRequested)
                        .context("Failed to send CloseRequested event")?;
                    break;
                }
                Ok(bytes_read) => {
                    trace!("EventMonitor: Read {} bytes from PTY", bytes_read);

                    // Parse bytes into ANSI commands
                    let mut ansi_commands = ansi_parser.process_bytes(&read_buffer[..bytes_read]);

                    if !ansi_commands.is_empty() {
                        debug!("EventMonitor: Parsed {} ANSI commands", ansi_commands.len());

                        // If we have buffered commands, add new ones to buffer
                        if !command_buffer.is_empty() {
                            command_buffer.append(&mut ansi_commands);
                            debug!(
                                "EventMonitor: Added to buffer, total buffered: {}",
                                command_buffer.len()
                            );
                        } else {
                            // Try to send without blocking
                            use crate::orchestrator::OrchestratorEvent;
                            match orchestrator_tx.try_send(OrchestratorEvent::IOEvent {
                                commands: ansi_commands.clone(),
                            }) {
                                Ok(()) => {
                                    trace!("EventMonitor: Sent IOEvent successfully");
                                }
                                Err(TrySendError::Full(_)) => {
                                    // Channel is full, start buffering
                                    debug!(
                                        "EventMonitor: Channel full, buffering {} commands",
                                        ansi_commands.len()
                                    );
                                    command_buffer.append(&mut ansi_commands);
                                }
                                Err(TrySendError::Disconnected(_)) => {
                                    return Err(anyhow::anyhow!(
                                        "Orchestrator channel disconnected"
                                    ));
                                }
                            }
                        }

                        // If buffer is too large, warn and use blocking send
                        if command_buffer.len() > 10_000 {
                            warn!(
                                "EventMonitor: Command buffer exceeded 10,000 commands ({}), using blocking send",
                                command_buffer.len()
                            );
                            let commands_to_send = std::mem::take(command_buffer);
                            use crate::orchestrator::OrchestratorEvent;
                            orchestrator_tx
                                .send(OrchestratorEvent::IOEvent {
                                    commands: commands_to_send,
                                })
                                .context("Failed to send buffered IOEvent")?;
                        }
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    // No more data available right now
                    trace!("EventMonitor: PTY read would block, no more data");

                    // Try to flush buffer if we have any pending commands
                    if !command_buffer.is_empty() {
                        use crate::orchestrator::OrchestratorEvent;
                        match orchestrator_tx.try_send(OrchestratorEvent::IOEvent {
                            commands: std::mem::take(command_buffer),
                        }) {
                            Ok(()) => {
                                trace!("EventMonitor: Flushed command buffer successfully");
                            }
                            Err(TrySendError::Full(event)) => {
                                // Put commands back in buffer
                                if let OrchestratorEvent::IOEvent { commands } = event {
                                    *command_buffer = commands;
                                    debug!(
                                        "EventMonitor: Channel still full, keeping {} commands in buffer",
                                        command_buffer.len()
                                    );
                                }
                            }
                            Err(TrySendError::Disconnected(_)) => {
                                return Err(anyhow::anyhow!("Orchestrator channel disconnected"));
                            }
                        }
                    }
                    break;
                }
                Err(e) => {
                    return Err(e).context("Failed to read from PTY");
                }
            }
        }

        Ok(())
    }

    /// Handles commands from the main thread.
    ///
    /// Returns `false` if the actor should shut down, `true` otherwise.
    fn handle_command(pty: &mut NixPty, action: PlatformAction) -> Result<bool> {
        match action {
            PlatformAction::Write(data) => {
                trace!("EventMonitor: Writing {} bytes to PTY", data.len());
                pty.write_all(&data).context("Failed to write to PTY")?;
            }
            PlatformAction::ResizePty { cols, rows } => {
                debug!("EventMonitor: Resizing PTY to {}x{}", cols, rows);
                pty.resize(cols, rows).context("Failed to resize PTY")?;
            }
            _ => {
                // Other actions (Render, SetTitle, etc.) should not be sent to this actor
                warn!(
                    "EventMonitor: Received unexpected action (ignoring): {:?}",
                    action
                );
            }
        }
        Ok(true)
    }
}

impl Drop for EventMonitorActor {
    fn drop(&mut self) {
        debug!("EventMonitorActor dropped");
        // Thread will exit when command channel is closed (which happens when command_tx is dropped)
        // We could optionally join the thread here, but it's not strictly necessary
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                error!("EventMonitor actor thread panicked: {:?}", e);
            }
        }
    }
}
