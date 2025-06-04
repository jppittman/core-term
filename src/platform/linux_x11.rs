// src/platform/linux_x11.rs
//
// Linux X11 platform implementation.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd; // Added for NixPty.as_raw_fd()

use anyhow::{Context, Result};
use log::*;

use crate::platform::actions::{PtyActionCommand, UiActionCommand};
use crate::platform::backends::x11::window::CursorVisibility; // Used for converting bool
use crate::platform::backends::{BackendEvent, Driver, PlatformState};
use crate::platform::os::epoll::{EpollFlags, EventMonitor};
use crate::platform::os::pty::{NixPty, PtyConfig, PtyChannel};
use crate::platform::platform_trait::Platform;
use crate::platform::backends::x11::XDriver;

// use libc::epoll_event as LibcEpollEvent; // Removed due to warning


/// EPOLL token for PTY events.
const PTY_EPOLL_TOKEN: u64 = 1;
/// EPOLL token for X11 driver events.
const DRIVER_EPOLL_TOKEN: u64 = 2;

pub struct LinuxX11Platform {
    pty: NixPty,
    driver: XDriver,
    event_monitor: EventMonitor,
    event_buffer: Vec<BackendEvent>,
    shutdown_requested: bool,
}

// Inherent implementation block
impl LinuxX11Platform {
    // This is the inherent, public `new` method that can be called directly.
    pub fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    // No `where Self: Sized` needed for inherent methods like this.
    {
        info!(
            "Initializing LinuxX11Platform with PTY size {}x{}",
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

        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty")?;
        let driver = XDriver::new().context("Failed to create XDriver")?;
        let event_monitor = EventMonitor::new().context("Failed to create EventMonitor")?;

        let pty_fd = pty.as_raw_fd();
        event_monitor
            .add(pty_fd, PTY_EPOLL_TOKEN, EpollFlags::EPOLLIN)
            .context("Failed to add PTY FD to event monitor")?;
        debug!("PTY FD {} added to event monitor", pty_fd);

        if let Some(driver_fd) = driver.get_event_fd() {
            event_monitor
                .add(driver_fd, DRIVER_EPOLL_TOKEN, EpollFlags::EPOLLIN)
                .context("Failed to add X11 driver FD to event monitor")?;
            debug!("X11 Driver FD {} added to event monitor", driver_fd);
        } else {
            info!("X11 Driver does not provide an event FD for polling.");
        }

        let initial_platform_state = driver.get_platform_state();
        info!("Initial platform state: {:?}", initial_platform_state);

        Ok((
            Self {
                pty,
                driver,
                event_monitor,
                event_buffer: Vec::new(),
                shutdown_requested: false,
            },
            initial_platform_state,
        ))
    }
}

impl Platform for LinuxX11Platform {
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
        // Call the inherent new method
        LinuxX11Platform::new(initial_pty_cols, initial_pty_rows, shell_command, shell_args)
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
                                info!("PTY read EOF, requesting shutdown.");
                                self.shutdown_requested = true;
                                return Ok(None);
                            }
                            Ok(count) => {
                                trace!("Read {} bytes from PTY", count);
                                return Ok(Some(buf[..count].to_vec()));
                            }
                            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                                return Ok(None);
                            }
                            Err(e) => {
                                return Err(e).context("Failed to read from PTY");
                            }
                        }
                    }
                }
            }
            Err(e) => {
                 if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                    if *nix_err == nix::Error::EINTR {
                        debug!("Event monitor poll interrupted by EINTR, retrying.");
                        return Ok(None);
                    }
                }
                return Err(e).context("Failed to poll event monitor for PTY data");
            }
        }
        Ok(None)
    }

    fn poll_ui_event(&mut self) -> Result<Option<BackendEvent>> {
        if let Some(event) = self.event_buffer.pop() {
            trace!("Popped event from buffer: {:?}", event);
            return Ok(Some(event));
        }

        if self.shutdown_requested {
            info!("Shutdown requested, returning CloseRequested event.");
            self.event_buffer.push(BackendEvent::CloseRequested);
            return Ok(self.event_buffer.pop());
        }

        match self.event_monitor.events(0) {
            Ok(events_slice) => {
                for event_ref in events_slice {
                    let event_data_token = event_ref.u64;
                    if event_data_token == DRIVER_EPOLL_TOKEN {
                        trace!("Processing X11 driver events");
                        let driver_events = self.driver.process_events().context("Failed to process X11 driver events")?;
                        for driver_event in &driver_events {
                            if matches!(driver_event, BackendEvent::CloseRequested) {
                                info!("CloseRequested event received from driver, requesting shutdown.");
                                self.shutdown_requested = true;
                            }
                        }
                        self.event_buffer.extend(driver_events.into_iter().rev());
                        if let Some(event) = self.event_buffer.pop() {
                             trace!("Popped event from buffer after processing driver events: {:?}", event);
                            return Ok(Some(event));
                        }
                    }
                }
            }
             Err(e) => {
                 if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                    if *nix_err == nix::Error::EINTR {
                        debug!("Event monitor poll interrupted by EINTR, retrying.");
                        return Ok(None);
                    }
                }
                return Err(e).context("Failed to poll event monitor for UI events");
            }
        }
        Ok(None)
    }

    fn dispatch_pty_action(&mut self, action: PtyActionCommand) -> Result<()> {
        trace!("Dispatching PTY action: {:?}", action);
        match action {
            PtyActionCommand::Write(data) => {
                self.pty.write_all(&data).context("Failed to write to PTY")?;
            }
            PtyActionCommand::ResizePty { cols, rows } => {
                self.pty.resize(cols, rows).context("Failed to resize PTY")?;
            }
        }
        Ok(())
    }

    fn dispatch_ui_action(&mut self, action: UiActionCommand) -> Result<()> {
        trace!("Dispatching UI action: {:?}", action);
        match action {
            UiActionCommand::Render(commands) => {
                self.driver.execute_render_commands(commands.clone()).context("Failed to execute render commands")?;
                self.driver.present().context("Failed to present frame")?;
            }
            UiActionCommand::SetTitle(title) => {
                self.driver.set_title(&title);
            }
            UiActionCommand::RingBell => {
                self.driver.bell();
            }
            UiActionCommand::CopyToClipboard(text) => {
                self.driver.own_selection(2, text);
            }
            UiActionCommand::SetCursorVisibility(visible) => {
                let cursor_visibility = if visible {
                    CursorVisibility::Shown
                } else {
                    CursorVisibility::Hidden
                };
                self.driver.set_cursor_visibility(cursor_visibility);
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.shutdown_requested {
            info!("Shutdown already requested or in progress.");
            return Ok(());
        }
        info!("Shutting down LinuxX11Platform...");
        self.driver.cleanup().context("Failed to cleanup XDriver")?;
        self.shutdown_requested = true;
        info!("LinuxX11Platform shutdown complete.");
        Ok(())
    }
}

impl Drop for LinuxX11Platform {
    fn drop(&mut self) {
        info!("Dropping LinuxX11Platform...");
        if let Err(e) = self.shutdown() {
            error!("Error during shutdown in Drop: {:?}", e);
        }
        info!("LinuxX11Platform dropped.");
    }
}
