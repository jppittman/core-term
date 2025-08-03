// src/platform/linux_x11.rs
//
// Linux X11 platform implementation with a hybrid threading model.
// The PTY is handled in a background thread, while X11 remains on the main thread.

use std::io::{ErrorKind, Read, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use log::*;

use crate::config::CONFIG;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::x11::window::CursorVisibility;
use crate::platform::backends::x11::{XDriver, TRAIT_ATOM_ID_CLIPBOARD};
use crate::platform::backends::{BackendEvent, Driver, PlatformState};
use crate::platform::os::epoll::{self, EpollFlags, EventMonitor};
use crate::platform::os::pty::{NixPty, PtyChannel, PtyConfig};
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;

const CLIPBOARD_SELECTION_INDEX: u32 = 2;

/// Actions sent from the main thread to the PTY actor thread.
#[derive(Debug)]
enum PtyActorAction {
    Write(Vec<u8>),
    Resize { cols: u16, rows: u16 },
    Shutdown,
}

/// The actor that runs in a background thread to handle PTY I/O.
struct PtyActor {
    pty: NixPty,
}

impl PtyActor {
    /// Runs the actor's main loop.
    fn run(
        mut self,
        action_rx: Receiver<PtyActorAction>,
        event_tx: Sender<PlatformEvent>,
    ) -> Result<()> {
        let mut pty_read_buf = [0u8; 4096];
        self.pty.set_nonblocking(true)?;

        loop {
            // Check for actions from the main thread.
            if let Ok(action) = action_rx.try_recv() {
                match action {
                    PtyActorAction::Write(data) => self.pty.write_all(&data)?,
                    PtyActorAction::Resize { cols, rows } => self.pty.resize(cols, rows)?,
                    PtyActorAction::Shutdown => {
                        info!("PTY actor received shutdown signal.");
                        break;
                    }
                }
            }

            // Read from the PTY.
            match self.pty.read(&mut pty_read_buf) {
                Ok(0) => {
                    info!("PTY actor read EOF.");
                    let _ = event_tx.send(PlatformEvent::BackendEvent(BackendEvent::CloseRequested));
                    break;
                }
                Ok(count) => {
                    let data = pty_read_buf[..count].to_vec();
                    if event_tx.send(PlatformEvent::IOEvent { data }).is_err() {
                        // Main thread hung up.
                        break;
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    // No data, sleep briefly to prevent busy-looping.
                    thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => return Err(e).context("PTY actor failed to read from PTY"),
            }
        }
        Ok(())
    }
}

pub struct LinuxX11Platform {
    driver: XDriver,
    event_monitor: EventMonitor,
    shutdown_requested: bool,
    pty_action_tx: Sender<PtyActorAction>,
    pty_event_rx: Receiver<PlatformEvent>,
    pty_thread_handle: Option<JoinHandle<()>>,
    // Buffer for epoll events from the driver
    driver_event_buffer: Vec<epoll::epoll_event>,
}

impl Platform for LinuxX11Platform {
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)> {
        info!("Initializing LinuxX11Platform with hybrid threading model.");

        // --- PTY Actor Setup ---
        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args.iter().map(String::as_str).collect::<Vec<_>>(),
            initial_cols: initial_pty_cols,
            initial_rows: initial_pty_rows,
        };
        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty")?;
        let pty_actor = PtyActor { pty };
        let (pty_action_tx, pty_action_rx) = mpsc::channel();
        let (pty_event_tx, pty_event_rx) = mpsc::channel();

        let pty_thread_handle = thread::Builder::new()
            .name("pty_actor".to_string())
            .spawn(move || {
                if let Err(e) = pty_actor.run(pty_action_rx, pty_event_tx) {
                    error!("PTY actor thread exited with error: {:?}", e);
                }
            })
            .context("Failed to spawn PTY actor thread")?;

        // --- X11 Driver Setup (Main Thread) ---
        let driver = XDriver::new().context("Failed to create XDriver")?;
        let event_monitor = EventMonitor::new().context("Failed to create EventMonitor")?;

        if let Some(driver_fd) = driver.get_event_fd() {
            event_monitor
                .add(driver_fd, DRIVER_EPOLL_TOKEN, EpollFlags::EPOLLIN)
                .context("Failed to add X11 driver FD to event monitor")?;
        }

        let initial_platform_state = driver.get_platform_state();

        Ok((
            Self {
                driver,
                event_monitor,
                shutdown_requested: false,
                pty_action_tx,
                pty_event_rx,
                pty_thread_handle: Some(pty_thread_handle),
                driver_event_buffer: Vec::with_capacity(16),
            },
            initial_platform_state,
        ))
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        let mut platform_events = Vec::new();

        // 1. Drain any pending events from the PTY actor thread.
        for pty_event in self.pty_event_rx.try_iter() {
            platform_events.push(pty_event);
        }

        // 2. Poll for X11 driver events.
        // Use a timeout to ensure we remain responsive and can process PTY events
        // that might arrive while we are waiting.
        let timeout = if platform_events.is_empty() {
            CONFIG.performance.min_draw_latency_ms.as_millis() as isize
        } else {
            0 // If we already have PTY events, don't block.
        };

        self.driver_event_buffer.clear();
        match self.event_monitor.events(&mut self.driver_event_buffer, timeout) {
            Ok(()) => {
                for _ in &self.driver_event_buffer {
                    let driver_events = self.driver.process_events()?;
                    platform_events.extend(driver_events.into_iter().map(PlatformEvent::from));
                }
            }
            Err(e) => {
                if let Some(nix_err) = e.downcast_ref::<nix::Error>() {
                    if *nix_err == nix::Error::EINTR {
                        return Ok(platform_events); // Interrupted, return what we have.
                    }
                }
                return Err(e).context("epoll_wait failed in main thread");
            }
        }

        Ok(platform_events)
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        for action in actions {
            match action {
                PlatformAction::Write(data) => self.pty_action_tx.send(PtyActorAction::Write(data))?,
                PlatformAction::ResizePty { cols, rows } => self.pty_action_tx.send(PtyActorAction::Resize { cols, rows })?,
                // UI actions are handled directly on the main thread.
                PlatformAction::Render(commands) => {
                    self.driver.execute_render_commands(commands.clone())?;
                    self.driver.present()?;
                }
                PlatformAction::SetTitle(title) => self.driver.set_title(&title),
                PlatformAction::RingBell => self.driver.bell(),
                PlatformAction::CopyToClipboard(text) => {
                    self.driver.own_selection(CLIPBOARD_SELECTION_INDEX.into(), text);
                }
                PlatformAction::SetCursorVisibility(visible) => {
                    self.driver.set_cursor_visibility(if visible {
                        CursorVisibility::Shown
                    } else {
                        CursorVisibility::Hidden
                    });
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
        if self.shutdown_requested {
            return Ok(());
        }
        self.shutdown_requested = true;
        info!("Cleaning up LinuxX11Platform...");

        // Shut down PTY actor thread
        if let Some(handle) = self.pty_thread_handle.take() {
            info!("Sending shutdown signal to PTY actor...");
            let _ = self.pty_action_tx.send(PtyActorAction::Shutdown);
            info!("Waiting for PTY actor thread to join...");
            if let Err(e) = handle.join() {
                 error!("Failed to join PTY actor thread: {:?}", e);
            }
        }

        self.driver.cleanup()
    }
}

impl Drop for LinuxX11Platform {
    fn drop(&mut self) {
        if !self.shutdown_requested {
            if let Err(e) = self.cleanup() {
                error!("Error during platform cleanup in Drop: {:?}", e);
            }
        }
    }
}

// Dummy constant for the epoll token, as it's only used for the driver now.
const DRIVER_EPOLL_TOKEN: u64 = 1;
