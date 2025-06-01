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
    config::CONFIG, // Use the global configuration
    orchestrator::{AppOrchestrator, OrchestratorStatus},
    platform::Platform, // Use the Platform enum
    platform::SystemEvent, // Import SystemEvent
    renderer::Renderer,
    term::TerminalEmulator, // The core terminal state machine
};
// use std::os::unix::io::AsRawFd; // No longer needed directly in main for FDs

// Logging
use log::{error, info, trace, warn};

// Timeout for platform event polling.
const PLATFORM_EVENT_POLL_TIMEOUT_MS: i32 = 16; // Approx 60 FPS for Tick events

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

    info!("Starting myterm application...");

    // --- Configuration ---
    // Access the globally initialized configuration.
    // CONFIG is a Lazy<Config>, so dereferencing it gives access to the Config struct.
    let config = &*CONFIG;
    info!("Configuration loaded/initialized globally.");

    // --- Setup Platform (replaces PTY and Driver setup) ---
    let mut platform = {
        if cfg!(feature = "x11") {
            Platform::new_x11(config).map_err(|e| {
                error!("Failed to initialize X11 platform: {}", e);
                e // Propagate error to main's Result
            })?
        } else if cfg!(feature = "console") {
            Platform::new_console(config).map_err(|e| {
                error!("Failed to initialize Console platform: {}", e);
                e
            })?
        } else {
            // This case should ideally be caught by feature assertions at build time if possible,
            // or handled by a default feature in Cargo.toml.
            return Err(anyhow::anyhow!("No platform backend feature (x11 or console) enabled. Please compile with --features x11 or --features console."));
        }
    };
    info!("Platform initialized successfully.");


    // --- Setup Terminal Emulator, Parser, Renderer ---
    // Initial dimensions are now derived by AppOrchestrator::new from platform state.
    // So, TerminalEmulator can be initialized with defaults that Orchestrator will override.
    let initial_cols = config.appearance.columns as usize;
    let initial_rows = config.appearance.rows as usize;

    trace!(
        "Initializing TerminalEmulator with default scrollback limit: {} and configured dimensions: {}x{}",
        DEFAULT_SCROLLBACK_LIMIT, initial_cols, initial_rows
    );
    let mut term_emulator = TerminalEmulator::new(
        initial_cols,
        initial_rows,
        DEFAULT_SCROLLBACK_LIMIT,
    );
    let mut ansi_parser = AnsiProcessor::new();
    let renderer = Renderer::new();

    // --- Create AppOrchestrator ---
    let mut orchestrator = AppOrchestrator::new(
        // &mut pty_channel, // Replaced by platform
        &mut term_emulator,
        &mut ansi_parser,
        renderer, // renderer is moved into the orchestrator
        // &mut *driver, // Replaced by platform
        platform, // Pass the concrete Platform enum instance
    );
    info!("AppOrchestrator created.");

    // --- Main Event Loop (simplified using Platform::poll_system_events) ---
    info!("Starting main event loop...");
    'main_loop: loop {
        match orchestrator.platform.poll_system_events(Some(PLATFORM_EVENT_POLL_TIMEOUT_MS)) {
            Ok(system_events) => {
                if system_events.is_empty() {
                    // This can happen if poll_system_events had a timeout of 0ms and no events were pending.
                    // Or if an internal interruption (like EINTR for epoll) occurred which poll_system_events handled.
                    // If PLATFORM_EVENT_POLL_TIMEOUT_MS was > 0, an empty vec implies no specific FD events but not necessarily a Tick.
                    // Tick is explicitly returned by poll_system_events if the timeout expires.
                    // So, an empty vec here means "no specific I/O or UI event, and no Tick either".
                    // We might not need to do anything special other than re-rendering if needed.
                    trace!("poll_system_events returned no specific events this cycle.");
                }

                for event in system_events {
                    trace!("Processing SystemEvent: {:?}", event);
                    match event {
                        SystemEvent::PrimaryIoReady => {
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
                        SystemEvent::UiInputReady => {
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
                        SystemEvent::Tick => {
                            // Handle tick events (e.g., for cursor blinking or other periodic tasks)
                            // For now, just log it. Orchestrator might handle it internally if needed.
                            trace!("Tick event received.");
                            // Orchestrator's render_if_needed might handle cursor blinking based on Tick.
                        }
                        SystemEvent::Error(e) => {
                            error!("System error from platform: {}. Exiting.", e);
                            break 'main_loop;
                        }
                        SystemEvent::ShutdownAdvised => {
                            info!("Platform advised shutdown. Exiting main loop.");
                            break 'main_loop;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to poll platform system events: {}. Exiting.", e);
                break 'main_loop;
            }
        }

        // Render the terminal if any state changes necessitate it.
        if let Err(e) = orchestrator.render_if_needed() {
            error!("Rendering failed: {}. Exiting.", e);
            break 'main_loop;
        }
    }

    // Cleanup platform resources before exiting
    if let Err(e) = orchestrator.platform.cleanup() {
        error!("Error during platform cleanup: {}", e);
    } else {
        info!("Platform cleanup successful.");
    }

    info!("MyTerm application terminated.");
    Ok(())
}
