// src/platform/linux_x11.rs
//
// Linux X11 platform implementation.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::AsRawFd;

use anyhow::{Context, Result};
use log::*;

use crate::config::CONFIG;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::x11::window::CursorVisibility; // Used for converting bool
use crate::platform::backends::x11::{XDriver, TRAIT_ATOM_ID_CLIPBOARD};
use crate::platform::backends::{BackendEvent, Driver, PlatformState};
use crate::platform::os::epoll::{EpollFlags, EventMonitor};
use crate::platform::os::pty::{NixPty, PtyChannel, PtyConfig};
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;

use super::os::epoll;

/// EPOLL token for PTY events.
const PTY_EPOLL_TOKEN: u64 = 1;
/// EPOLL token for X11 driver events.
const DRIVER_EPOLL_TOKEN: u64 = 2;
/// Base buffer size for PTY reads, often related to system page sizes.
const PTY_BASE_BUFFER_SIZE: usize = 4096;
/// Multiplier for the base PTY buffer size. This is somewhat arbitrary but chosen to be
/// generous to capture large bursts of PTY output.
const PTY_READ_BUFFER_SIZE_MULTIPLIER: usize = 4;
/// Total buffer size for PTY reads.
const PTY_READ_BUFFER_SIZE: usize = PTY_READ_BUFFER_SIZE_MULTIPLIER * PTY_BASE_BUFFER_SIZE;
/// Index for clipboard selection, using 2 for PRIMARY selection, common in X11.
const CLIPBOARD_SELECTION_INDEX: u32 = 2;

pub struct LinuxX11Platform {
    pty: NixPty,
    driver: XDriver,
    event_monitor: EventMonitor,
    shutdown_requested: bool,
    pty_event_buffer: Vec<epoll::epoll_event>,
    platform_events: Vec<PlatformEvent>,
}

impl LinuxX11Platform {
    pub fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)> {
        // No `where Self: Sized` needed for inherent methods like this.
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
                pty_event_buffer: Vec::with_capacity(PTY_READ_BUFFER_SIZE),
                driver,
                event_monitor,
                shutdown_requested: false,
                platform_events: Vec::with_capacity(PTY_READ_BUFFER_SIZE),
            },
            initial_platform_state,
        ))
    }

    /// Processes a batch of epoll events and returns a boolean indicating if polling should continue.
    ///
    /// This helper is called after `epoll_wait` returns. It translates kernel-level event readiness
    /// into `PlatformEvent`s and provides a heuristic signal.
    ///
    /// # Returns
    /// * `Ok(true)`: If significant activity occurred (substantial PTY read or any X11 UI event),
    ///   suggesting the main polling loop should continue if its budget allows.
    /// * `Ok(false)`: If activity was minimal (small PTY read with no UI events), if `epoll_wait`
    ///   timed out (empty `events_slice`), or if a PTY EOF/critical error occurred.
    fn process_epoll_batch(
        &mut self,
        accumulated_pty_data: &mut Vec<u8>,
    ) -> Result<bool> {
        if self.pty_event_buffer.is_empty() {
            // No events from epoll_wait (likely timed out); signal to stop polling for this cycle.
            return Ok(false);
        }

        let mut pty_bytes_read_this_batch = 0;
        let mut had_x11_event_this_batch = false;
        // PTY reads below this threshold, without other UI events, are considered "small"
        // and might signal an end to the current polling/coalescing attempt.
        const SMALL_PTY_READ_THRESHOLD: usize = 16;

        for event_ref in &self.pty_event_buffer {
            let event_token = unsafe { std::ptr::addr_of!(event_ref.u64).read_unaligned() };
            match event_token {
                DRIVER_EPOLL_TOKEN => {
                    had_x11_event_this_batch = true;
                    let driver_events = self
                        .driver
                        .process_events()
                        .context("Driver event processing failed during batch")?;
                    // The orchestrator is responsible for interpreting any CloseRequested events.
                   self.platform_events
                        .extend(driver_events.into_iter().map(PlatformEvent::from));
                }
                PTY_EPOLL_TOKEN => {
                    // Inner loop to attempt to drain the PTY for this specific epoll signal.
                    // This is useful because level-triggered epoll will keep signaling if data remains.
                    let mut pty_read_chunk_buf = [0u8; PTY_READ_BUFFER_SIZE];
                    loop {
                        match self.pty.read(&mut pty_read_chunk_buf) {
                            Ok(0) => {
                                // PTY EOF
                                self.platform_events.push(BackendEvent::CloseRequested.into());
                                // PTY EOF is a definitive reason to stop further polling in this poll_events call.
                                return Ok(false);
                            }
                            Ok(count) => {
                                if count > 0 {
                                    accumulated_pty_data
                                        .extend_from_slice(&pty_read_chunk_buf[..count]);
                                    pty_bytes_read_this_batch += count;
                                }
                            }
                            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                                // No more data to read *right now* from the PTY for this epoll signal.
                                break;
                            }
                            Err(e) if e.kind() == ErrorKind::Interrupted => {
                                continue;
                            }
                            Err(e) => {
                                // Any other PTY read error.
                                log::error!("Critical PTY read error in batch: {:?}. Forwarding as CloseRequested.", e);
                                self.platform_events.push(BackendEvent::CloseRequested.into());
                                // Critical error; stop further polling.
                                return Ok(false);
                            }
                        }
                    }
                }
                _ => unreachable!("unrecognized epoll token"),
            }
        }

        // Heuristic: Determine if the main poll_events loop should continue based on this batch.
        if had_x11_event_this_batch {
            // Any UI activity suggests a "burst" is ongoing or starting.
            return Ok(true);
        }
        if pty_bytes_read_this_batch >= SMALL_PTY_READ_THRESHOLD {
            // A significant PTY read also suggests a burst.
            return Ok(true);
        }

        // Otherwise (only small/no PTY read, no UI events), it's likely interactive.
        // Signal to stop polling for this `poll_events` call to prioritize latency.
        Ok(false)
    }

    // Made public so it can be called from main.rs
    pub fn shutdown(&mut self) -> Result<()> {
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

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        let mut accumulated_pty_data: Vec<u8> = Vec::new();
        self.pty_event_buffer.clear();

        let min_latency = CONFIG.performance.min_draw_latency_ms.as_millis() as isize;
        let max_latency = CONFIG.performance.max_draw_latency_ms;

        let burst_start_time = std::time::Instant::now();
        let remaining_time = max_latency - burst_start_time.elapsed();
        loop {
            if remaining_time.is_zero() {
                break;
            }

            match self.event_monitor.events(
                &mut self.pty_event_buffer,
                std::cmp::min(min_latency, remaining_time.as_millis() as isize),
            ) {
                Ok(()) => {
                    // This is the success path, formerly the code after the `?`
                    let should_continue = self.process_epoll_batch(
                        &mut accumulated_pty_data,
                    )?;

                    if !should_continue {
                        break;
                    }
                }
                Err(e) => {
                    if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                        if *nix_err == nix::Error::EINTR {
                            // EINTR is a recoverable interruption. Continue the main polling loop.
                            log::trace!("epoll_wait interrupted (EINTR), continuing poll loop.");
                            continue;
                        }
                    }

                    if !accumulated_pty_data.is_empty() {
                        self.platform_events.push(PlatformEvent::IOEvent {
                            data: std::mem::take(&mut accumulated_pty_data),
                        });
                    }
                    return Err(e).context("epoll_wait failed in coalescing loop");
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
                    // Log length before text is moved to own_selection
                    debug!("LinuxX11Platform: CopyToClipboard action, text length: {}. Dispatching to XDriver.", text.len());
                    self.driver
                        .own_selection(CLIPBOARD_SELECTION_INDEX.into(), text); // Cast to u64
                }
                PlatformAction::SetCursorVisibility(visible) => {
                    let cursor_visibility = if visible {
                        CursorVisibility::Shown
                    } else {
                        CursorVisibility::Hidden
                    };
                    self.driver.set_cursor_visibility(cursor_visibility);
                }
                PlatformAction::RequestPaste => {
                    self.driver.request_selection_data(CLIPBOARD_SELECTION_INDEX.into(), TRAIT_ATOM_ID_CLIPBOARD);
                }
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
    }

    fn cleanup(&mut self) -> Result<()> {
        log::info!("LinuxX11Platform: cleanup() called. Cleaning up XDriver...");
        self.driver.cleanup()
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
