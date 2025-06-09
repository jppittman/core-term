// src/platform/linux_wayland.rs

use anyhow::{Context, Result};
use log::*;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd;
use std::time::Duration;

use crate::config::CONFIG;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::wayland::WaylandDriver;
// Ensure correct import for CursorVisibility, it's likely under backends or backends::x11 (if re-exported)
// For Wayland, it might be a specific enum or a bool directly.
// Assuming CursorVisibility is available via crate::platform::backends::CursorVisibility
use crate::platform::PlatformEvent;
use crate::platform::backends::{BackendEvent, CursorVisibility, Driver, PlatformState};
use crate::platform::os::epoll::{EpollFlags, EventMonitor};
use crate::platform::os::pty::{NixPty, PtyConfig};
use crate::platform::platform_trait::Platform;

const PTY_EPOLL_TOKEN: u64 = 1;
const DRIVER_EPOLL_TOKEN: u64 = 2;
const PTY_READ_BUFFER_SIZE: usize = 4096 * 4;

pub struct LinuxWaylandPlatform {
    pty: NixPty,
    driver: WaylandDriver,
    event_monitor: EventMonitor,
    shutdown_requested: bool,
    epoll_event_buffer: Vec<nix::sys::epoll::epoll_event>,
    platform_events: Vec<PlatformEvent>,
}

impl LinuxWaylandPlatform {
    pub fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)> {
        info!(
            "Initializing LinuxWaylandPlatform with PTY size {}x{}",
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
            .context("Failed to create NixPty for Wayland")?;
        let mut driver = WaylandDriver::new().context("Failed to create WaylandDriver")?;
        let event_monitor =
            EventMonitor::new().context("Failed to create EventMonitor for Wayland")?;

        let pty_fd = pty.as_raw_fd();
        event_monitor
            .add(pty_fd, PTY_EPOLL_TOKEN, EpollFlags::EPOLLIN)
            .context("Failed to add PTY FD to event monitor (Wayland)")?;
        debug!("Wayland: PTY FD {} added to event monitor", pty_fd);

        if let Some(driver_fd) = driver.get_event_fd() {
            event_monitor
                .add(driver_fd, DRIVER_EPOLL_TOKEN, EpollFlags::EPOLLIN)
                .context("Failed to add Wayland driver FD to event monitor")?;
            debug!("Wayland: Driver FD {} added to event monitor", driver_fd);
        } else {
            warn!(
                "Wayland Driver does not provide an event FD for polling. UI event processing might be impacted."
            );
        }

        let initial_platform_state = driver.get_platform_state();
        info!(
            "Wayland: Initial platform state: {:?}",
            initial_platform_state
        );

        Ok((
            Self {
                pty,
                driver,
                event_monitor,
                shutdown_requested: false,
                epoll_event_buffer: Vec::with_capacity(10),
                platform_events: Vec::new(),
            },
            initial_platform_state,
        ))
    }

    fn process_epoll_batch(&mut self, accumulated_pty_data: &mut Vec<u8>) -> Result<bool> {
        if self.epoll_event_buffer.is_empty() {
            return Ok(false);
        }

        let mut pty_bytes_read_this_batch = 0;
        let mut had_driver_event_this_batch = false;
        const SMALL_PTY_READ_THRESHOLD: usize = 16;

        for event_ref in self.epoll_event_buffer.iter() {
            let event_token = event_ref.data();
            match event_token {
                DRIVER_EPOLL_TOKEN => {
                    had_driver_event_this_batch = true;
                    match self.driver.process_events() {
                        Ok(driver_backend_events) => {
                            for backend_event in driver_backend_events {
                                self.platform_events
                                    .push(PlatformEvent::from(backend_event));
                            }
                        }
                        Err(e) => {
                            error!(
                                "Wayland driver event processing failed: {:?}. Requesting close.",
                                e
                            );
                            self.platform_events
                                .push(PlatformEvent::from(BackendEvent::CloseRequested));
                            return Ok(false); // Indicate failure to stop polling loop
                        }
                    }
                }
                PTY_EPOLL_TOKEN => {
                    let mut pty_read_chunk_buf = [0u8; PTY_READ_BUFFER_SIZE];
                    loop {
                        match self.pty.read(&mut pty_read_chunk_buf) {
                            Ok(0) => {
                                info!("Wayland: PTY EOF detected.");
                                self.platform_events
                                    .push(PlatformEvent::from(BackendEvent::CloseRequested));
                                return Ok(false); // Indicate PTY closed to stop polling loop
                            }
                            Ok(count) => {
                                if count > 0 {
                                    accumulated_pty_data
                                        .extend_from_slice(&pty_read_chunk_buf[..count]);
                                    pty_bytes_read_this_batch += count;
                                } else {
                                    break;
                                } // No more data in this chunk read
                            }
                            Err(e) if e.kind() == ErrorKind::WouldBlock => break, // Done reading for now
                            Err(e) if e.kind() == ErrorKind::Interrupted => continue, // Retry read
                            Err(e) => {
                                error!(
                                    "Wayland: Critical PTY read error: {:?}. Requesting close.",
                                    e
                                );
                                self.platform_events
                                    .push(PlatformEvent::from(BackendEvent::CloseRequested));
                                return Ok(false); // Indicate failure
                            }
                        }
                    }
                }
                _ => warn!("Wayland: Unrecognized epoll token: {}", event_token),
            }
        }
        // Continue polling if driver had events or if a significant amount of PTY data was read
        Ok(had_driver_event_this_batch || pty_bytes_read_this_batch >= SMALL_PTY_READ_THRESHOLD)
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.shutdown_requested {
            info!("Wayland: Shutdown already requested or in progress.");
            return Ok(());
        }
        info!("Shutting down LinuxWaylandPlatform...");
        self.driver
            .cleanup()
            .context("Failed to cleanup WaylandDriver during shutdown")?;
        self.shutdown_requested = true;
        info!("LinuxWaylandPlatform shutdown complete.");
        Ok(())
    }
}

impl Platform for LinuxWaylandPlatform {
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized,
    {
        LinuxWaylandPlatform::new(
            initial_pty_cols,
            initial_pty_rows,
            shell_command,
            shell_args,
        )
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        let mut accumulated_pty_data: Vec<u8> = Vec::new();
        self.platform_events.clear();

        let min_latency_ms = CONFIG.performance.min_draw_latency_ms.as_millis() as isize;
        let max_burst_duration = CONFIG.performance.max_draw_latency_ms;
        let burst_start_time = std::time::Instant::now();

        loop {
            let elapsed_in_burst = burst_start_time.elapsed();
            if elapsed_in_burst >= max_burst_duration {
                break;
            }

            let remaining_time_in_burst = max_burst_duration.saturating_sub(elapsed_in_burst);
            let epoll_timeout_ms = std::cmp::min(
                min_latency_ms,
                remaining_time_in_burst
                    .as_millis()
                    .try_into()
                    .unwrap_or(min_latency_ms),
            )
            .max(0); // Ensure timeout is not negative

            self.epoll_event_buffer.clear(); // Clear before waiting for new events
            match self
                .event_monitor
                .events(&mut self.epoll_event_buffer, epoll_timeout_ms)
            {
                Ok(()) => {
                    // Ok(()) means epoll_wait returned successfully, number of events is in buffer len
                    if self.epoll_event_buffer.is_empty() {
                        // Timeout occurred, no events
                        break;
                    }
                    if !self.process_epoll_batch(&mut accumulated_pty_data)? {
                        // process_epoll_batch returns false if polling should stop (e.g. EOF, error, or minimal activity)
                        break;
                    }
                }
                Err(e) => {
                    if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                        if *nix_err == nix::Error::EINTR {
                            trace!(
                                "Wayland: epoll_wait interrupted (EINTR), continuing poll loop."
                            );
                            continue; // Loop again if interrupted
                        }
                    }
                    // For other errors, push any accumulated data and then return the error
                    if !accumulated_pty_data.is_empty() {
                        self.platform_events.push(PlatformEvent::IOEvent {
                            data: std::mem::take(&mut accumulated_pty_data),
                        });
                    }
                    return Err(e).context("Wayland: epoll_wait failed in polling loop");
                }
            }
        }

        if !accumulated_pty_data.is_empty() {
            self.platform_events.push(PlatformEvent::IOEvent {
                data: accumulated_pty_data,
            });
        }
        Ok(std::mem::take(&mut self.platform_events))
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        for action in actions {
            match action {
                PlatformAction::Write(data) => {
                    self.pty
                        .write_all(&data)
                        .context("Wayland: Failed to write to PTY")?;
                }
                PlatformAction::ResizePty { cols, rows } => {
                    self.pty
                        .resize(cols, rows)
                        .context("Wayland: Failed to resize PTY")?;
                }
                PlatformAction::Render(commands) => {
                    self.driver
                        .execute_render_commands(commands)
                        .context("Wayland: Failed to execute render commands")?;
                    self.driver
                        .present()
                        .context("Wayland: Failed to present frame")?;
                }
                PlatformAction::SetTitle(title) => {
                    self.driver.set_title(&title);
                }
                PlatformAction::RingBell => {
                    self.driver.bell();
                }
                PlatformAction::CopyToClipboard(text) => {
                    debug!(
                        "Wayland: PlatformAction::CopyToClipboard: '{}'. (Not fully implemented)",
                        text
                    );
                    // Actual Wayland clipboard requires data device manager interaction
                    // Using placeholder values for atoms. These would need to be actual atoms
                    // obtained from the Wayland compositor for PRIMARY or CLIPBOARD.
                    self.driver.own_selection(0, text); // TODO: Use actual selection atoms
                }
                PlatformAction::SetCursorVisibility(visible) => {
                    self.driver.set_cursor_visibility(if visible {
                        CursorVisibility::Shown
                    } else {
                        CursorVisibility::Hidden
                    });
                }
                PlatformAction::RequestPaste => {
                    debug!("Wayland: PlatformAction::RequestPaste. (Not fully implemented)");
                    // Actual Wayland paste requires data device manager interaction
                    // Using placeholder values for atoms.
                    self.driver.request_selection_data(0, 0); // TODO: Use actual selection and target atoms
                }
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("WaylandPlatform: cleanup() called. Attempting to cleanup driver.");
        self.driver.cleanup()
    }
}

impl Drop for LinuxWaylandPlatform {
    fn drop(&mut self) {
        info!("Dropping LinuxWaylandPlatform...");
        if !self.shutdown_requested {
            if let Err(e) = self.shutdown() {
                error!("Wayland: Error during implicit shutdown in Drop: {:?}", e);
            }
        }
        info!("LinuxWaylandPlatform dropped.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::platform_trait::Platform; // To use Platform::new
    use anyhow::Result;

    // This test requires a running Wayland compositor and a functional PTY environment.
    // It should be marked #[ignore] for typical test runs.
    #[test]
    #[ignore] // Ignoring by default
    fn test_linux_wayland_platform_new() -> Result<()> {
        // env_logger::builder().is_test(true).try_init().ok(); // Optional: for verbose test logging

        let shell_command = "/bin/sh".to_string(); // Use a simple shell for testing
        let shell_args: Vec<String> = vec!["-c".to_string(), "echo test".to_string()]; // Simple command

        // Use the Platform trait's new method
        match LinuxWaylandPlatform::new(80, 24, shell_command, shell_args) {
            Ok((mut platform, initial_state)) => {
                info!("LinuxWaylandPlatform::new() test successful, platform instance created.");
                info!("Initial platform state: {:?}", initial_state);

                // Perform a minimal operation, e.g., getting current state again
                let current_state = platform.get_current_platform_state();
                assert_eq!(
                    current_state, initial_state,
                    "Initial and current states should match before any operations."
                );

                // Call cleanup to release resources
                platform.cleanup()?;
                Ok(())
            }
            Err(e) => {
                error!("LinuxWaylandPlatform::new() test failed: {:?}", e);
                Err(e)
            }
        }
    }
}
