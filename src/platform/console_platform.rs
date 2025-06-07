// src/platform/console_platform.rs
//
// Console platform implementation.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd;

use anyhow::{Context, Result};
use log::{debug, error, info, trace};

use crate::platform::actions::PlatformAction;
use crate::platform::backends::console::ConsoleDriver;
use crate::platform::backends::{BackendEvent, Driver, PlatformState};
use crate::platform::os::epoll::{EpollFlags, EventMonitor};
use crate::platform::os::pty::{NixPty, PtyChannel, PtyConfig};
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;

use super::os::epoll;

const PTY_EPOLL_TOKEN: u64 = 1;
// Buffer size for reading from PTY to align with common page sizes and buffer practices.
const PTY_READ_BUFFER_SIZE: usize = 4096;
// Index for clipboard selection, using 2 as a placeholder based on XDriver's PRIMARY selection.
// This might be specific to certain environments or backend driver expectations.
const CLIPBOARD_SELECTION_INDEX: u32 = 2;

pub struct ConsolePlatform {
    pty: NixPty,
    driver: ConsoleDriver,
    event_monitor: EventMonitor,
    shutdown_requested: bool,
    event_buffer: Vec<BackendEvent>,
    pty_event_buffer: Vec<epoll::epoll_event>,
}

impl ConsolePlatform {
    pub fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)> {
        info!(
            "Initializing ConsolePlatform with PTY size {}x{}",
            initial_pty_cols, initial_pty_rows
        );
        info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

        let shell_args_refs: Vec<&str> = shell_args.iter().map(String::as_str).collect();

        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args_refs,
            initial_cols: initial_pty_cols,
            initial_rows: initial_pty_rows,
        };

        let pty = NixPty::spawn_with_config(&pty_config)
            .context("Failed to create NixPty for ConsolePlatform")?;
        let driver = ConsoleDriver::new().context("Failed to create ConsoleDriver")?;

        let event_monitor =
            EventMonitor::new().context("Failed to create EventMonitor for ConsolePlatform PTY")?;
        let pty_fd = pty.as_raw_fd();
        event_monitor
            .add(pty_fd, PTY_EPOLL_TOKEN, EpollFlags::EPOLLIN)
            .context("Failed to add PTY FD to event monitor for ConsolePlatform")?;
        debug!(
            "PTY FD {} added to event monitor for ConsolePlatform",
            pty_fd
        );

        let initial_platform_state = driver.get_platform_state();
        info!(
            "Initial platform state for ConsolePlatform: {:?}",
            initial_platform_state
        );

        Ok((
            Self {
                pty,
                driver,
                event_monitor,
                shutdown_requested: false,
                event_buffer: Vec::new(),
                pty_event_buffer: Vec::new(),
            },
            initial_platform_state,
        ))
    }
}

impl Platform for ConsolePlatform {
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized,
    {
        ConsolePlatform::new(
            initial_pty_cols,
            initial_pty_rows,
            shell_command,
            shell_args,
        )
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        let mut collected_events = Vec::new();

        if self.shutdown_requested {
            if self.event_buffer.is_empty() {
                collected_events.push(BackendEvent::CloseRequested.into());
            }
            // Drain any remaining buffered events even if shutdown is requested.
            // This ensures that any events generated before shutdown is fully processed
            // are not lost.
            for backend_event in self.event_buffer.drain(..) {
                if matches!(backend_event, BackendEvent::CloseRequested) {
                    // Defensive check: if CloseRequested is in the buffer, ensure shutdown state is true.
                    self.shutdown_requested = true;
                }
                collected_events.push(backend_event.into());
            }
            return Ok(collected_events);
        }

        // Drain buffered events from previous polls
        for backend_event in self.event_buffer.drain(..) {
            if matches!(backend_event, BackendEvent::CloseRequested) {
                info!("ConsolePlatform: CloseRequested event drained from buffer, initiating shutdown.");
                self.shutdown_requested = true;
            }
            collected_events.push(backend_event.into());
        }

        self.pty_event_buffer.drain(..);
        // PTY Events
        // Poll for PTY events with a non-blocking timeout (0)
        if let Err(e) = self.event_monitor.events(&mut self.pty_event_buffer, 0) {
            if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                if *nix_err == nix::Error::EINTR {
                    // It's a recoverable interruption. Log it and continue execution.
                    // The operation will be retried on the next poll.
                    debug!("ConsolePlatform: PTY event monitor poll interrupted by EINTR; will retry on next poll.");
                } else {
                    // It's a different, unrecoverable Nix error. Add context and return from the function.
                    return Err(e)
                        .context("ConsolePlatform: Error polling PTY event monitor (Nix error)");
                }
            } else {
                // It's a non-Nix, unrecoverable error. Add context and return from the function.
                return Err(e)
                    .context("ConsolePlatform: Error polling PTY event monitor (Non-Nix error)");
            }
        }
        for event_ref in &mut self.pty_event_buffer {
            if event_ref.u64 == PTY_EPOLL_TOKEN {
                let mut buf = [0u8; PTY_READ_BUFFER_SIZE];
                match self.pty.read(&mut buf) {
                    Ok(0) => {
                        info!("ConsolePlatform: PTY read EOF, initiating shutdown.");
                        self.shutdown_requested = true;
                        collected_events.push(BackendEvent::CloseRequested.into());
                    }
                    Ok(count) => {
                        trace!("ConsolePlatform: Read {} bytes from PTY", count);
                        collected_events.push(PlatformEvent::IOEvent {
                            data: buf[..count].to_vec(),
                        });
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        // No data available from PTY at this moment.
                    }
                    Err(e) => {
                        // An actual error occurred during PTY read.
                        return Err(e).context("ConsolePlatform: Failed to read from PTY");
                    }
                }
            }
        }

        // Driver Events (UI)
        // Process events from the underlying console driver.
        let driver_events_result = self.driver.process_events();
        match driver_events_result {
            Ok(driver_events) => {
                if !driver_events.is_empty() {
                    trace!(
                        "ConsolePlatform: Received {} backend events from ConsoleDriver",
                        driver_events.len()
                    );
                }
                for backend_event in driver_events {
                    if matches!(backend_event, BackendEvent::CloseRequested) {
                        info!("ConsolePlatform: CloseRequested event received from driver, initiating shutdown.");
                        self.shutdown_requested = true;
                    }
                    collected_events.push(backend_event.into());
                }
            }
            Err(e) => {
                return Err(e).context("ConsolePlatform: Failed to process ConsoleDriver events");
            }
        }

        Ok(collected_events)
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        for action in actions {
            trace!("ConsolePlatform: Dispatching action: {:?}", action);
            match action {
                PlatformAction::Write(data) => {
                    self.pty
                        .write_all(&data)
                        .context("ConsolePlatform: Failed to write to PTY")?;
                }
                PlatformAction::ResizePty { cols, rows } => {
                    self.pty
                        .resize(cols, rows)
                        .context("ConsolePlatform: Failed to resize PTY")?;
                }
                PlatformAction::Render(commands) => {
                    self.driver
                        .execute_render_commands(commands)
                        .context("ConsolePlatform: Failed to execute render commands")?;
                    self.driver
                        .present()
                        .context("ConsolePlatform: Failed to present frame via ConsoleDriver")?;
                }
                PlatformAction::SetTitle(title) => {
                    self.driver.set_title(&title);
                }
                PlatformAction::RingBell => {
                    self.driver.bell();
                }
                PlatformAction::CopyToClipboard(text) => {
                    // The `own_selection` method in `ConsoleDriver` is expected to be a no-op
                    // or log this, as console environments typically don't manage system clipboards
                    // in the same way GUI environments do. CLIPBOARD_SELECTION_INDEX is based on
                    // conventions from X11 (PRIMARY selection), passed for trait compatibility.
                    let text_len = text.len(); // Get length before moving
                    self.driver
                        .own_selection(CLIPBOARD_SELECTION_INDEX.into(), text); // text is moved here
                    debug!("ConsolePlatform: CopyToClipboard action processed (expected no-op for ConsoleDriver). Text length: {}", text_len);
                }
                PlatformAction::SetCursorVisibility(visible) => {
                    // ConsoleDriver currently does not implement cursor visibility control.
                    // This action is a no-op for ConsolePlatform to prevent panics.
                    debug!("ConsolePlatform: SetCursorVisibility action processed (no-op for ConsolePlatform). Visible: {}", visible);
                }
                PlatformAction::RequestPaste => {
                    unimplemented!("paste for console backend unimplemented");
                }
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
    }
}

// Inherent methods
impl ConsolePlatform {
    pub fn shutdown(&mut self) -> Result<()> {
        if self.shutdown_requested {
            info!("ConsolePlatform: Shutdown already requested or in progress.");
            return Ok(());
        }
        info!("ConsolePlatform: Shutting down...");
        self.driver
            .cleanup()
            .context("ConsolePlatform: Failed to cleanup ConsoleDriver")?;
        self.shutdown_requested = true;
        info!("ConsolePlatform: Shutdown complete.");
        Ok(())
    }
}

impl Drop for ConsolePlatform {
    fn drop(&mut self) {
        info!("ConsolePlatform: Dropping...");
        if let Err(e) = self.shutdown() {
            error!("ConsolePlatform: Error during shutdown in Drop: {:?}", e);
        }
        info!("ConsolePlatform: Dropped.");
    }
}
