// In src/main.rs

// Declare modules
pub mod ansi;
pub mod color;
pub mod config;
pub mod glyph;
pub mod keys;
pub mod orchestrator;
pub mod platform; // This will now contain os and backends
pub mod renderer;
pub mod term;

// Use statements for items needed in main.rs
use crate::{
    ansi::AnsiProcessor, // AnsiProcessor is in the ansi module
    orchestrator::{AppOrchestrator, OrchestratorStatus},
    platform::backends::{console::ConsoleDriver, x11::XDriver, Driver},
    platform::os::{
        epoll::{EpollFlags, EventMonitor}, // Using EventMonitor for epoll management
        pty::{NixPty, PtyConfig}, // NixPty for PTY channel, PtyConfig for its setup. Removed PtyChannel.
    },
    renderer::Renderer,
    term::TerminalEmulator, // The core terminal state machine
};
use std::os::unix::io::AsRawFd; // For getting raw file descriptors

// Logging
use log::{error, info, trace, warn};

// Constants for epoll tokens.
// These values are arbitrary but must be unique for each FD monitored.
const PTY_EPOLL_TOKEN: u64 = 1;
const DRIVER_EPOLL_TOKEN: u64 = 2;

// Timeout for epoll_wait in milliseconds. -1 means block indefinitely.
const EPOLL_TIMEOUT_INDEFINITE: isize = -1;
// const EPOLL_TIMEOUT_SHORT_MS: isize = 16; // Example for a ~60Hz polling if driver needs it

/// Default scrollback limit for the terminal emulator.
const DEFAULT_SCROLLBACK_LIMIT: usize = 1000;

/// Main entry point for the `myterm` application.
///
/// Orchestrates the setup and event loop for the terminal emulator, PTY,
/// backend driver, and renderer.
fn main() -> anyhow::Result<()> {
    // Initialize the logger. Default filter is "info" if RUST_LOG is not set.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_micros()
        .init();

    info!("Starting myterm orchestrator...");

    // --- Configuration ---
    let shell_path = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
    let shell_args: Vec<String> = Vec::new(); // No specific args by default
    let shell_args_str: Vec<&str> = shell_args.iter().map(AsRef::as_ref).collect();

    // Initial dimensions, these are default and might be updated by AppOrchestrator based on driver.
    let initial_cols = platform::backends::DEFAULT_WINDOW_WIDTH_CHARS as u16;
    let initial_rows = platform::backends::DEFAULT_WINDOW_HEIGHT_CHARS as u16;

    // --- Setup PTY ---
    let pty_config = PtyConfig {
        command_executable: &shell_path,
        args: &shell_args_str,
        initial_cols,
        initial_rows,
    };
    let mut pty_channel = NixPty::spawn_with_config(&pty_config).unwrap_or_else(|e| {
        eprintln!("Fatal: Failed to spawn PTY: {}", e);
        std::process::exit(1); // Exit on critical PTY setup failure
    });
    let pty_fd = pty_channel.as_raw_fd();
    info!("PTY spawned with master fd: {}", pty_fd);

    // --- Setup Driver ---
    // Attempt to initialize X11 driver, fall back to ConsoleDriver on error.
    let mut driver: Box<dyn Driver> = match XDriver::new() {
        Ok(d) => Box::new(d),
        Err(e) => {
            warn!(
                "Failed to initialize X11 driver: {}. Falling back to ConsoleDriver.",
                e
            );
            Box::new(ConsoleDriver::new().unwrap_or_else(|ce| {
                eprintln!("Fatal: Failed to initialize ConsoleDriver: {}", ce);
                std::process::exit(1); // Exit on critical ConsoleDriver setup failure
            }))
        }
    };
    info!("Driver initialized.");

    // Note: Initial dimensions (initial_cols, initial_rows) are passed to PtyConfig.
    // AppOrchestrator::new will now query the driver for its actual PlatformState
    // (including font and display pixel dimensions), calculate the resulting grid size,
    // and then resize both the PTY and the TerminalEmulator instance accordingly.
    // So, the explicit dimension query and PTY resize previously done here in main.rs
    // are no longer needed.

    // --- Setup Terminal Emulator, Parser, Renderer ---
    // TerminalEmulator is initialized with initial_cols and initial_rows,
    // but AppOrchestrator::new will immediately send a Resize ControlEvent
    // to update it based on actual driver metrics.
    trace!(
        "Initializing TerminalEmulator with default scrollback limit: {}",
        DEFAULT_SCROLLBACK_LIMIT
    );
    let mut term_emulator = TerminalEmulator::new(
        initial_cols as usize, // These might be default, orchestrator will resize
        initial_rows as usize, // These might be default, orchestrator will resize
        // DEFAULT_SCROLLBACK_LIMIT, // Removed, TerminalEmulator::new now gets it from CONFIG
    );
    let mut ansi_parser = AnsiProcessor::new();
    let renderer = Renderer::new();

    // --- Create AppOrchestrator ---
    // The AppOrchestrator takes mutable references to the main components.
    let mut orchestrator = AppOrchestrator::new(
        &mut pty_channel,
        &mut term_emulator,
        &mut ansi_parser,
        renderer, // renderer is moved into the orchestrator
        &mut *driver,
    );
    info!("AppOrchestrator created.");

    // --- Setup EventMonitor (epoll wrapper) ---
    let mut event_monitor = EventMonitor::new()
        .map_err(|e| anyhow::anyhow!("Failed to create event monitor: {}", e))?;

    // Add PTY file descriptor for monitoring read events.
    event_monitor
        .add(pty_fd, PTY_EPOLL_TOKEN, EpollFlags::EPOLLIN)
        .map_err(|e| anyhow::anyhow!("Failed to add PTY FD {} to event monitor: {}", pty_fd, e))?;
    trace!("PTY FD {} added to event monitor for read events.", pty_fd);

    // Add driver's event file descriptor, if available.
    if let Some(driver_event_fd) = orchestrator.driver.get_event_fd() {
        event_monitor
            .add(driver_event_fd, DRIVER_EPOLL_TOKEN, EpollFlags::EPOLLIN)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to add Driver FD {} to event monitor: {}",
                    driver_event_fd,
                    e
                )
            })?;
        trace!(
            "Driver FD {} added to event monitor for read events.",
            driver_event_fd
        );
    } else {
        info!("Driver does not provide an event FD. Main loop will not poll it via epoll.");
    }

    info!("Starting main event loop...");
    'main_loop: loop {
        // Wait for events on monitored file descriptors.
        match event_monitor.events(EPOLL_TIMEOUT_INDEFINITE) {
            Ok(events_slice) => {
                log::trace!("Event monitor returned {} events.", events_slice.len());
                if events_slice.is_empty() {
                    // This case typically happens if epoll_wait times out with 0 events.
                    // If timeout is indefinite (-1), this should ideally not happen unless interrupted.
                    log::trace!("Event monitor timed out or returned no events, continuing loop.");
                }

                for event in events_slice {
                    let event_token = event.u64; // Retrieve the token associated with the event.
                    match event_token {
                        PTY_EPOLL_TOKEN => {
                            log::trace!("Event on PTY FD (token {}).", PTY_EPOLL_TOKEN);
                            // Process data from the PTY.
                            match orchestrator.process_pty_events() {
                                Ok(OrchestratorStatus::Running) => { /* Continue */ }
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
                            // Process events from the backend driver.
                            match orchestrator.process_driver_events() {
                                Ok(OrchestratorStatus::Running) => { /* Continue */ }
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
                // Handle errors from epoll_wait, specifically EINTR (interrupted system call).
                if let Some(nix_err) = e.root_cause().downcast_ref::<nix::Error>() {
                    if *nix_err == nix::Error::EINTR {
                        trace!("Event monitor wait interrupted by EINTR, continuing.");
                        continue 'main_loop;
                    }
                }
                // For other epoll errors, log and exit.
                error!(
                    "Event monitor wait failed: {}. Root cause: {:?}",
                    e,
                    e.root_cause()
                );
                break 'main_loop;
            }
        }

        // Render the terminal if any state changes necessitate it.
        if let Err(e) = orchestrator.render_if_needed() {
            error!("Rendering failed: {}. Exiting.", e);
            break 'main_loop;
        }
    }
    Ok(())
}
