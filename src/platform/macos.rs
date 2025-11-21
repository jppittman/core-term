use crate::platform::actions::PlatformAction;
use crate::platform::backends::{
    cocoa::CocoaDriver, BackendEvent, CursorVisibility, Driver, PlatformState,
};
use crate::platform::os::event::{EventMonitor, KqueueEvent, KqueueFlags};
use crate::platform::os::pty::{NixPty, PtyChannel, PtyConfig};
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;
use anyhow::{Context, Result};
use log::*;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd;

/// kqueue token for PTY events.
const PTY_TOKEN: u64 = 1;
/// kqueue token for Cocoa driver events.
const DRIVER_TOKEN: u64 = 2;
/// Base buffer size for PTY reads
const PTY_BASE_BUFFER_SIZE: usize = 4096;
const PTY_READ_BUFFER_SIZE_MULTIPLIER: usize = 4;
const PTY_READ_BUFFER_SIZE: usize = PTY_READ_BUFFER_SIZE_MULTIPLIER * PTY_BASE_BUFFER_SIZE;

pub struct MacosPlatform {
    pty: NixPty,
    driver: CocoaDriver,
    event_monitor: EventMonitor,
    shutdown_requested: bool,
    event_buffer: Vec<KqueueEvent>,
    platform_events: Vec<PlatformEvent>,
}

impl Platform for MacosPlatform {
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized,
    {
        info!(
            "Initializing MacosPlatform with PTY size {}x{}",
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
        let driver = CocoaDriver::new().context("Failed to create CocoaDriver")?;
        let event_monitor = EventMonitor::new().context("Failed to create EventMonitor")?;

        let pty_fd = pty.as_raw_fd();
        event_monitor
            .add(pty_fd, PTY_TOKEN, KqueueFlags::EPOLLIN)
            .context("Failed to add PTY FD to event monitor")?;
        debug!("PTY FD {} added to event monitor", pty_fd);

        if let Some(driver_fd) = driver.get_event_fd() {
            event_monitor
                .add(driver_fd, DRIVER_TOKEN, KqueueFlags::EPOLLIN)
                .context("Failed to add Cocoa driver FD to event monitor")?;
            debug!("Cocoa Driver FD {} added to event monitor", driver_fd);
        } else {
            info!("Cocoa Driver does not provide an event FD for polling.");
        }

        let initial_platform_state = driver.get_platform_state();
        info!("Initial platform state: {:?}", initial_platform_state);

        Ok((
            Self {
                pty,
                event_buffer: Vec::with_capacity(PTY_READ_BUFFER_SIZE),
                driver,
                event_monitor,
                shutdown_requested: false,
                platform_events: Vec::with_capacity(PTY_READ_BUFFER_SIZE),
            },
            initial_platform_state,
        ))
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        self.platform_events.clear();

        // Poll for events with a small timeout
        self.event_monitor
            .events(&mut self.event_buffer, 10)
            .context("Failed to poll events")?;

        let mut accumulated_pty_data = Vec::new();

        for event in &self.event_buffer {
            match event.token {
                DRIVER_TOKEN => {
                    let driver_events = self
                        .driver
                        .process_events()
                        .context("Driver event processing failed")?;
                    self.platform_events
                        .extend(driver_events.into_iter().map(PlatformEvent::from));
                }
                PTY_TOKEN => {
                    // Read from PTY
                    let mut pty_read_buf = [0u8; PTY_READ_BUFFER_SIZE];
                    loop {
                        match self.pty.read(&mut pty_read_buf) {
                            Ok(0) => {
                                // PTY EOF
                                self.platform_events
                                    .push(BackendEvent::CloseRequested.into());
                                return Ok(std::mem::take(&mut self.platform_events));
                            }
                            Ok(n) => {
                                accumulated_pty_data.extend_from_slice(&pty_read_buf[..n]);
                            }
                            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                                break;
                            }
                            Err(e) => {
                                return Err(e).context("Failed to read from PTY");
                            }
                        }
                    }
                }
                _ => {
                    warn!("Unknown event token: {}", event.token);
                }
            }
        }

        // If we read any PTY data, generate an IOEvent
        if !accumulated_pty_data.is_empty() {
            self.platform_events
                .push(PlatformEvent::IOEvent { data: accumulated_pty_data });
        }

        Ok(std::mem::take(&mut self.platform_events))
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        for action in actions {
            match action {
                PlatformAction::Write(data) => {
                    self.pty
                        .write_all(&data)
                        .context("Failed to write to PTY")?;
                }
                PlatformAction::ResizePty { cols, rows } => {
                    self.pty
                        .resize(cols, rows)
                        .context("Failed to resize PTY")?;
                }
                PlatformAction::Render(render_commands) => {
                    self.driver
                        .execute_render_commands(render_commands)
                        .context("MacosPlatform: Failed to dispatch Render to driver")?;
                    // As per previous note, PresentFrame might be implicitly handled by driver
                    // or explicitly called. Let's add an explicit PresentFrame after Render.
                    self.driver
                        .present()
                        .context("MacosPlatform: Failed to dispatch PresentFrame to driver")?;
                }
                PlatformAction::SetTitle(title) => {
                    self.driver.set_title(&title);
                    // .context("MacosPlatform: Failed to dispatch SetTitle to driver")?; // set_title doesn't return Result
                }
                PlatformAction::RingBell => {
                    self.driver.bell();
                    // .context("MacosPlatform: Failed to dispatch RingBell to driver")?; // bell doesn't return Result
                }
                PlatformAction::CopyToClipboard(text) => {
                    // Using a placeholder atom (0) for now. A real implementation would use a Cocoa-specific way.
                    self.driver.own_selection(0, text);
                    // .context("MacosPlatform: Failed to dispatch CopyToClipboard to driver")?; // own_selection doesn't return Result
                }
                PlatformAction::SetCursorVisibility(visible) => {
                    let visibility = if visible {
                        CursorVisibility::Shown
                    } else {
                        CursorVisibility::Hidden
                    };
                    self.driver.set_cursor_visibility(visibility);
                    // .context("MacosPlatform: Failed to dispatch SetCursorVisibility to driver")?; // set_cursor_visibility doesn't return Result
                }
                PlatformAction::RequestPaste => {
                    // Using placeholder atoms (0, 0) for now.
                    // A real implementation would use Cocoa-specific APIs for pasteboard interaction.
                    self.driver.request_selection_data(0, 0);
                    // .context("MacosPlatform: Failed to dispatch RequestPaste to driver")?; // request_selection_data doesn't return Result
                }
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        // println!("MacosPlatform: Getting current platform state from driver");
        self.driver.get_platform_state()
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("MacosPlatform: cleanup() called. Cleaning up driver...");
        self.driver.cleanup()
    }
}

impl Drop for MacosPlatform {
    fn drop(&mut self) {
        debug!("MacosPlatform dropped");
        // NixPty will handle PTY cleanup in its own Drop implementation
    }
}
