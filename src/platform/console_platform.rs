// src/platform/console_platform.rs
//
// Console platform implementation.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd;

use anyhow::{Context, Result};
use log::{debug, error, info, trace}; // Removed warn

use crate::platform::actions::{PtyActionCommand, UiActionCommand};
use crate::platform::backends::console::ConsoleDriver;
use crate::platform::backends::{BackendEvent, Driver, PlatformState};
use crate::platform::os::epoll::{EpollFlags, EventMonitor};
use crate::platform::os::pty::{NixPty, PtyConfig, PtyChannel};
use crate::platform::platform_trait::Platform;

// use libc::epoll_event as LibcEpollEvent; // Removed due to warning

const PTY_EPOLL_TOKEN: u64 = 1;

pub struct ConsolePlatform {
    pty: NixPty,
    driver: ConsoleDriver,
    event_monitor: EventMonitor,
    shutdown_requested: bool,
    event_buffer: Vec<BackendEvent>,
}

// Inherent implementation block
impl ConsolePlatform {
    // This is the inherent, public `new` method.
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

        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty for ConsolePlatform")?;
        let driver = ConsoleDriver::new().context("Failed to create ConsoleDriver")?;

        let event_monitor = EventMonitor::new().context("Failed to create EventMonitor for ConsolePlatform PTY")?;
        let pty_fd = pty.as_raw_fd();
        event_monitor
            .add(pty_fd, PTY_EPOLL_TOKEN, EpollFlags::EPOLLIN)
            .context("Failed to add PTY FD to event monitor for ConsolePlatform")?;
        debug!("PTY FD {} added to event monitor for ConsolePlatform", pty_fd);

        let initial_platform_state = driver.get_platform_state();
        info!("Initial platform state for ConsolePlatform: {:?}", initial_platform_state);

        Ok((
            Self {
                pty,
                driver,
                event_monitor,
                shutdown_requested: false,
                event_buffer: Vec::new(),
            },
            initial_platform_state,
        ))
    }
}

impl Platform for ConsolePlatform {
    // The trait's new method now calls the inherent public new method.
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized,
    {
        ConsolePlatform::new(initial_pty_cols, initial_pty_rows, shell_command, shell_args)
    }

    fn poll_pty_data(&mut self) -> Result<Option<Vec<u8>>> {
        if self.shutdown_requested {
            return Ok(None);
        }

        match self.event_monitor.events(0) {
            Ok(events_slice) => {
                for event_ref in events_slice {
                    let event_data_token = event_ref.u64;
                    if event_data_token == PTY_EPOLL_TOKEN {
                        let mut buf = [0u8; 4096];
                        match self.pty.read(&mut buf) {
                            Ok(0) => {
                                info!("ConsolePlatform: PTY read EOF, requesting shutdown.");
                                self.shutdown_requested = true;
                                return Ok(None);
                            }
                            Ok(count) => {
                                trace!("ConsolePlatform: Read {} bytes from PTY", count);
                                return Ok(Some(buf[..count].to_vec()));
                            }
                            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                                return Ok(None);
                            }
                            Err(e) => {
                                return Err(e).context("ConsolePlatform: Failed to read from PTY");
                            }
                        }
                    }
                }
            }
            Err(e) => {
                if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                    if *nix_err == nix::Error::EINTR {
                        debug!("ConsolePlatform: PTY Event monitor poll interrupted by EINTR, retrying.");
                        return Ok(None);
                    }
                }
                return Err(e).context("ConsolePlatform: Failed to poll event monitor for PTY data");
            }
        }
        Ok(None)
    }

    fn poll_ui_event(&mut self) -> Result<Option<BackendEvent>> {
        if let Some(event) = self.event_buffer.pop() {
            trace!("ConsolePlatform: Popped event from buffer: {:?}", event);
            return Ok(Some(event));
        }

        if self.shutdown_requested {
            info!("ConsolePlatform: Shutdown requested, returning CloseRequested event if buffer empty.");
            self.event_buffer.push(BackendEvent::CloseRequested);
            return Ok(self.event_buffer.pop());
        }

        match self.driver.process_events() {
            Ok(driver_events) => {
                if !driver_events.is_empty() {
                    trace!("ConsolePlatform: Received {} events from ConsoleDriver", driver_events.len());
                    for event_in_vec in &driver_events {
                        if matches!(event_in_vec, BackendEvent::CloseRequested) {
                            info!("ConsolePlatform: CloseRequested event received from driver, requesting shutdown.");
                            self.shutdown_requested = true;
                        }
                    }
                    self.event_buffer.extend(driver_events.into_iter().rev());
                    if let Some(event_to_ret) = self.event_buffer.pop() {
                        trace!("ConsolePlatform: Popped event from buffer after processing ConsoleDriver events: {:?}", event_to_ret);
                        return Ok(Some(event_to_ret));
                    }
                }
            }
            Err(e) => {
                return Err(e).context("ConsolePlatform: Failed to process ConsoleDriver events");
            }
        }
        Ok(None)
    }

    fn dispatch_pty_action(&mut self, action: PtyActionCommand) -> Result<()> {
        trace!("ConsolePlatform: Dispatching PTY action: {:?}", action);
        match action {
            PtyActionCommand::Write(data) => {
                self.pty.write_all(&data).context("ConsolePlatform: Failed to write to PTY")?;
            }
            PtyActionCommand::ResizePty { cols, rows } => {
                self.pty.resize(cols, rows).context("ConsolePlatform: Failed to resize PTY")?;
            }
        }
        Ok(())
    }

    fn dispatch_ui_action(&mut self, action: UiActionCommand) -> Result<()> {
        trace!("ConsolePlatform: Dispatching UI action: {:?}", action);
        match action {
            UiActionCommand::Render(commands) => {
                self.driver.execute_render_commands(commands).context("ConsolePlatform: Failed to execute render commands")?;
                self.driver.present().context("ConsolePlatform: Failed to present frame via ConsoleDriver")?;
            }
            UiActionCommand::SetTitle(title) => {
                self.driver.set_title(&title);
            }
            UiActionCommand::RingBell => {
                self.driver.bell();
            }
            UiActionCommand::CopyToClipboard(text) => {
                debug!("ConsolePlatform: CopyToClipboard called (no-op for ConsoleDriver). Text length: {}", text.len());
            }
            UiActionCommand::SetCursorVisibility(visible) => {
                debug!("ConsolePlatform: SetCursorVisibility called (no-op in ConsolePlatform to avoid panic from unimplemented ConsoleDriver method). Visible: {}", visible);
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.shutdown_requested {
            info!("ConsolePlatform: Shutdown already requested or in progress.");
            return Ok(());
        }
        info!("ConsolePlatform: Shutting down...");
        self.driver.cleanup().context("ConsolePlatform: Failed to cleanup ConsoleDriver")?;
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
