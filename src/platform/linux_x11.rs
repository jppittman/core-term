// src/platform/linux_x11.rs
//
// Linux X11 platform implementation.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd; // Added for NixPty.as_raw_fd()

use anyhow::{Context, Result};
use log::*;

use crate::config::CONFIG;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::x11::window::CursorVisibility; // Used for converting bool
use crate::platform::backends::x11::XDriver;
use crate::platform::backends::{BackendEvent, Driver, PlatformState};
use crate::platform::os::epoll::{EpollFlags, EventMonitor};
use crate::platform::os::pty::{NixPty, PtyChannel, PtyConfig};
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;

// use libc::epoll_event as LibcEpollEvent; // Removed due to warning

/// EPOLL token for PTY events.
const PTY_EPOLL_TOKEN: u64 = 1;
/// EPOLL token for X11 driver events.
const DRIVER_EPOLL_TOKEN: u64 = 2;

pub struct LinuxX11Platform {
    pty: NixPty,
    driver: XDriver,
    event_monitor: EventMonitor,
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
                shutdown_requested: false,
            },
            initial_platform_state,
        ))
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
        LinuxX11Platform::new(
            initial_pty_cols,
            initial_pty_rows,
            shell_command,
            shell_args,
        )
    }

    fn poll_events(&mut self) -> Result<Vec<super::PlatformEvent>> {
        if self.shutdown_requested {
            info!("Shutdown requested, returning CloseRequested event.");
            return Ok(vec![BackendEvent::CloseRequested.into()])
        }

        let timeout = (CONFIG.performance.max_draw_latency_ms
            - CONFIG.performance.min_draw_latency_ms)
            .as_millis() as u8;
        match self.event_monitor.events(timeout.into()) {
            Ok(events_slice) => {
                for event_ref in events_slice {
                    match event_ref.u64 {
                        DRIVER_EPOLL_TOKEN => {
                            trace!("Processing X11 driver events");
                            let driver_events = self
                                .driver
                                .process_events()
                                .context("Failed to process X11 driver events")?;
                            for driver_event in &driver_events {
                                if matches!(driver_event, BackendEvent::CloseRequested) {
                                    info!("CloseRequested event received from driver, requesting shutdown.");
                                    self.shutdown_requested = true;
                                    return Ok(vec![BackendEvent::CloseRequested.into()]);
                                }
                            }
                            return Ok(driver_events.into_iter().map(|e| PlatformEvent::from(e)).collect())
                        }
                        PTY_EPOLL_TOKEN => {
                            const ST_BUFFER: usize = 4096;
                            // magic number: arbitrary. More than st. Should be plenty. Maybe this
                            // belongs in config?
                            let mut buf = [0u8; 4 * ST_BUFFER];
                            match self.pty.read(&mut buf) {
                                Ok(0) => {
                                    info!("PTY read EOF, requesting shutdown.");
                                    self.shutdown_requested = true;
                                    return Ok(vec![BackendEvent::CloseRequested.into()]);
                                }
                                Ok(count) => {
                                    trace!("Read {} bytes from PTY", count);
                                    return Ok(vec![PlatformEvent::IOEvent {
                                        data: buf[..count].to_vec(),
                                    }]);
                                }
                                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                                    return Ok(vec![]);
                                }
                                Err(e) => {
                                    return Err(e).context("Failed to read from PTY");
                                }
                            }
                        }
                        _ => {
                            unimplemented!("unrecognized event type: {:?}", event_ref);
                        }
                    }
                }
                return Ok(vec![]);
            }
            Err(e) => {
                if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                    if *nix_err == nix::Error::EINTR {
                        debug!("Event monitor poll interrupted by EINTR, retrying.");
                        return Ok(vec![]);
                    }
                }
                return Err(e).context("Failed to poll event monitor for UI events");
            }
        }
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        for action in actions {
            match action {
                PlatformAction::Write(data) => self
                    .pty
                    .write_all(&data)
                    .context("Failed to write to PTY")?,
                PlatformAction::ResizePty { cols, rows } => {
                    self.pty
                        .resize(cols, rows)
                        .context("Failed to resize PTY")?;
                }
                PlatformAction::Render(commands) => {
                    self.driver
                        .execute_render_commands(commands.clone())
                        .context("Failed to execute render commands")?;
                    self.driver.present().context("Failed to present frame")?;
                }
                PlatformAction::SetTitle(title) => {
                    self.driver.set_title(&title);
                }
                PlatformAction::RingBell => {
                    self.driver.bell();
                }
                PlatformAction::CopyToClipboard(text) => {
                    self.driver.own_selection(2, text);
                }
                PlatformAction::SetCursorVisibility(visible) => {
                    let cursor_visibility = if visible {
                        CursorVisibility::Shown
                    } else {
                        CursorVisibility::Hidden
                    };
                    self.driver.set_cursor_visibility(cursor_visibility);
                }
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
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
