// In src/main.rs

// Declare modules
pub mod ansi;
pub mod backends;
pub mod color;
pub mod glyph;
pub mod os; // Contains `pub mod pty;` and now `pub mod epoll;`
pub mod orchestrator;
pub mod renderer;
pub mod term;

// Use statements for items needed in main.rs
use crate::{
    backends::{console::ConsoleDriver, x11::XDriver, Driver},
    orchestrator::{AppOrchestrator, OrchestratorStatus},
    os::{
        epoll::{EpollFlags, EventMonitor}, // Use EventMonitor, EpollEvent is used via the slice
        pty::{NixPty, PtyConfig, PtyChannel},      // PtyChannel is not directly used here now
    },
    renderer::Renderer,
    term::{AnsiProcessor, TerminalEmulator}, // TerminalInterface is not directly used here now, AnsiProcessor imported
};
use std::os::unix::io::AsRawFd; // For FD handling

// Logging
use log::{debug, error, info, trace, warn};

// Constants for epoll tokens.
// These values are arbitrary but must be unique for each FD monitored.
const PTY_EPOLL_TOKEN: u64 = 1;
const DRIVER_EPOLL_TOKEN: u64 = 2;

// Timeout for epoll_wait in milliseconds. -1 means block indefinitely.
const EPOLL_TIMEOUT_INDEFINITE: isize = -1;
// const EPOLL_TIMEOUT_SHORT_MS: isize = 16; // Example for a ~60Hz polling if driver needs it

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_micros()
        .init();

    info!("Starting myterm orchestrator...");

    // --- Configuration ---
    let shell_path = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
    let shell_args: Vec<String> = Vec::new();
    let shell_args_str: Vec<&str> = shell_args.iter().map(AsRef::as_ref).collect();

    let mut initial_cols = backends::DEFAULT_WINDOW_WIDTH_CHARS as u16;
    let mut initial_rows = backends::DEFAULT_WINDOW_HEIGHT_CHARS as u16;

    // --- Setup PTY ---
    let pty_config = PtyConfig {
        command_executable: &shell_path,
        args: &shell_args_str,
        initial_cols,
        initial_rows,
    };
    let mut pty_channel = NixPty::spawn_with_config(&pty_config).unwrap_or_else(|e| {
        eprintln!("Fatal: Failed to spawn PTY: {}", e);
        std::process::exit(1); // Adhering to style guide for critical setup failure
    });
    let pty_fd = pty_channel.as_raw_fd();
    log::info!("PTY spawned with master fd: {}", pty_fd);

    // --- Setup Driver ---
    let mut driver: Box<dyn Driver> = match XDriver::new() {
        Ok(d) => Box::new(d),
        Err(e) => {
            warn!(
                "Failed to initialize X11 driver: {}. Falling back to ConsoleDriver.",
                e
            );
            Box::new(ConsoleDriver::new().unwrap_or_else(|ce| {
                eprintln!("Fatal: Failed to initialize ConsoleDriver: {}", ce);
                std.process::exit(1);
            }))
        }
    };
    info!("Driver initialized.");

    // Update initial dimensions from driver
    let (display_width_px, display_height_px) = driver.get_display_dimensions_pixels();
    let (char_width_px, char_height_px) = driver.get_font_dimensions();
    if char_width_px > 0 && char_height_px > 0 {
        initial_cols = (display_width_px as usize / char_width_px).max(1) as u16;
        initial_rows = (display_height_px as usize / char_height_px).max(1) as u16;
        if let Err(e) = pty_channel.resize(initial_cols, initial_rows) {
            warn!(
                "Failed to resize PTY to initial driver dimensions: {}x{} cells. Error: {}",
                initial_cols, initial_rows, e
            );
        }
    }
    info!(
        "Initial terminal dimensions set to: {}x{} cells",
        initial_cols, initial_rows
    );

    // --- Setup Terminal Emulator, Parser, Renderer ---
    let mut term_emulator = TerminalEmulator::new(
        initial_cols as usize,
        initial_rows as usize,
        1000, // Scrollback limit (though North Star says no scrollback, this is a common param)
    );
    let mut ansi_parser = AnsiProcessor::new();
    let mut renderer = Renderer::new();

    // --- Create AppOrchestrator ---
    let mut orchestrator = AppOrchestrator::new(
        &mut pty_channel,
        &mut term_emulator,
        &mut ansi_parser, // Pass mut ref to AnsiProcessor
        renderer,
        &mut *driver,
    );
    info!("AppOrchestrator created.");

    // --- Setup EventMonitor (formerly epoll directly) ---
    let mut event_monitor = EventMonitor::new()
        .map_err(|e| anyhow::anyhow!("Failed to create event monitor: {}", e))?;

    event_monitor
        .add(pty_fd, PTY_EPOLL_TOKEN, EpollFlags::EPOLLIN)
        .map_err(|e| anyhow::anyhow!("Failed to add PTY FD {} to event monitor: {}", pty_fd, e))?;
    log::trace!(
        "PTY FD {} added to event monitor for read events.",
        pty_fd
    );

    if let Some(driver_event_fd) = orchestrator.driver.get_event_fd() {
        event_monitor
            .add(
                driver_event_fd,
                DRIVER_EPOLL_TOKEN,
                EpollFlags::EPOLLIN,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to add Driver FD {} to event monitor: {}",
                    driver_event_fd,
                    e
                )
            })?;
        log::trace!(
            "Driver FD {} added to event monitor for read events.",
            driver_event_fd
        );
    } else {
        log::info!("Driver does not provide an event FD. Main loop will not poll it via epoll.");
    }

    info!("Starting main event loop...");
    'main_loop: loop {
        // The event_monitor.events() call now handles the epoll_wait internally
        // and returns a slice of events that occurred.
        match event_monitor.events(EPOLL_TIMEOUT_INDEFINITE) {
            Ok(events_slice) => {
                log::trace!("Event monitor returned {} events.", events_slice.len());
                if events_slice.is_empty() {
                    // This case typically happens if epoll_wait times out with 0 events.
                    // If timeout is indefinite (-1), this should ideally not happen unless interrupted.
                    // If a timeout is used (e.g., for periodic tasks), this is normal.
                    log::trace!("Event monitor timed out or returned no events, continuing loop.");
                    // Add any periodic tasks here if needed when using a timeout.
                    // continue 'main_loop; // Explicitly continue if no events and using a timeout
                }

                for event in events_slice {
                    let event_token = event.data();
                    match event_token {
                        PTY_EPOLL_TOKEN => {
                            log::trace!("Event on PTY FD (token {}).", PTY_EPOLL_TOKEN);
                            match orchestrator.process_pty_events() {
                                Ok(OrchestratorStatus::Running) => {}
                                Ok(OrchestratorStatus::Shutdown) => {
                                    info!("PTY indicated shutdown. Exiting main loop.");
                                    break 'main_loop;
                                }
                                Err(e) => {
                                    error!("Error processing PTY events: {}. Exiting.", e);
                                    break 'main_loop;
                                }
                            }
                        }
                        DRIVER_EPOLL_TOKEN => {
                            log::trace!("Event on Driver FD (token {}).", DRIVER_EPOLL_TOKEN);
                            match orchestrator.process_driver_events() {
                                Ok(OrchestratorStatus::Running) => {}
                                Ok(OrchestratorStatus::Shutdown) => {
                                    info!("Driver indicated shutdown. Exiting main loop.");
                                    break 'main_loop;
                                }
                                Err(msg) => {
                                    error!("Error processing driver events: {}. Exiting.", msg);
                                    break 'main_loop;
                                }
                            }
                        }
                        _ => {
                            warn!("Unknown epoll token: {}", event_token);
                        }
                    }
                }
            }
            Err(e) => {
                // Check if the error is EINTR (interrupted system call)
                if let Some(nix_err) = e.root_cause().downcast_ref::<nix::Error>() {
                    if nix_err.as_errno() == Some(nix::errno::Errno::EINTR) {
                        log::trace!("Event monitor wait interrupted by signal, continuing.");
                        continue 'main_loop; // Interrupted by signal, retry.
                    }
                }
                error!("Event monitor wait failed: {}. Root cause: {:?}", e, e.root_cause());
                break 'main_loop; // Exit on unrecoverable epoll error.
            }
        }

        if let Err(e) = orchestrator.render_if_needed() {
            error!("Rendering failed: {}. Exiting.", e);
            break 'main_loop;
        }
    }

    info!("Exiting myterm orchestrator. Cleaning up...");
    orchestrator.driver.cleanup()?;
    info!("Cleanup complete.");
    Ok(())
}
