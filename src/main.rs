// In src/main.rs

// Declare modules
pub mod ansi;
pub mod color;
pub mod config;
pub mod glyph;
pub mod keys;
pub mod orchestrator;
pub mod platform;
pub mod renderer;
pub mod term;

// Use statements for items needed in main.rs
use crate::{
    ansi::AnsiProcessor,
    config::CONFIG, // Using Config directly
    orchestrator::{AppOrchestrator, OrchestratorStatus},
    platform::actions::PtyActionCommand, // For initial PTY resize
    platform::linux_x11::LinuxX11Platform, // Specific platform implementation
    platform::platform_trait::Platform,  // Trait needed for platform methods
    renderer::Renderer,
    term::TerminalEmulator,
};

// Logging
use anyhow::Context; // For context on Results
use log::{error, info, warn}; // Removed trace as it's not used in main

// Default initial PTY dimensions (hints for Platform::new)
const DEFAULT_INITIAL_PTY_COLS: u16 = 80;
const DEFAULT_INITIAL_PTY_ROWS: u16 = 24;

/// Main entry point for the `myterm` application.
fn main() -> anyhow::Result<()> {
    // Initialize the logger. Default filter is "info" if RUST_LOG is not set.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_micros()
        .init();

    info!("Starting myterm...");

    // --- Configuration ---
    // Load application config (using default for now as per plan)
    // In future, this might be: let config = Config::load_or_default();
    info!("Configuration loaded (using default).");

    let shell_command = std::env::var("SHELL").unwrap_or_else(|_| {
        warn!("SHELL environment variable not set, defaulting to /bin/bash");
        "/bin/bash".to_string()
    });
    let shell_args: Vec<String> = Vec::new(); // No specific args by default for now

    info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

    // --- Instantiate Concrete Platform ---
    // These are hints for the platform's PTY initialization.
    // The actual terminal dimensions will be derived from PlatformState later.
    // The Platform::new method is expected to be available via the Platform trait,
    // so if LinuxX11Platform implements Platform, this should work.
    // The error "no function or associated item named `new` found for struct `LinuxX11Platform`"
    // suggests the 'use crate::platform::platform_trait::Platform;' might be needed if calling as Platform::new,
    // or that the inherent new method on LinuxX11Platform wasn't found due to other issues.
    // The previous fixes to LinuxX11Platform ensured its `new` method is inherent.
    let (mut platform, initial_platform_state) = LinuxX11Platform::new(
        DEFAULT_INITIAL_PTY_COLS,
        DEFAULT_INITIAL_PTY_ROWS,
        shell_command,
        shell_args,
    )
    .context("Failed to initialize LinuxX11Platform")?;

    info!(
        "Platform initialized. Initial state: {:?}",
        initial_platform_state
    );

    // --- Initialize Core Components ---
    // Calculate terminal dimensions based on actual font and display metrics from the platform.
    let term_cols = (initial_platform_state.display_width_px as usize
        / initial_platform_state.font_cell_width_px.max(1) as usize)
        .max(1);
    let term_rows = (initial_platform_state.display_height_px as usize
        / initial_platform_state.font_cell_height_px.max(1) as usize)
        .max(1);

    info!(
        "Calculated initial terminal dimensions: {} cols, {} rows",
        term_cols, term_rows
    );

    let mut term_emulator = TerminalEmulator::new(
        term_cols,
        term_rows,
        // config.scrollback_limit.unwrap_or(DEFAULT_SCROLLBACK_LIMIT) // Assuming config provides this
    );
    info!("TerminalEmulator initialized.");

    let mut ansi_parser = AnsiProcessor::new();
    info!("AnsiProcessor initialized.");

    // Renderer::new() takes no arguments.
    let renderer = Renderer::new();
    info!("Renderer initialized.");

    // --- Initial Resize Synchronization ---
    // Ensure the PTY's dimensions match the TerminalEmulator's calculated dimensions.
    platform
        .dispatch_pty_action(PtyActionCommand::ResizePty {
            cols: term_cols as u16,
            rows: term_rows as u16,
        })
        .context("Failed to dispatch initial PTY resize command")?;
    info!(
        "Dispatched initial PTY resize to {}x{}",
        term_cols, term_rows
    );

    // --- Instantiate AppOrchestrator ---
    let mut orchestrator = AppOrchestrator::new(
        &mut platform,
        &mut term_emulator,
        &mut ansi_parser,
        renderer,
        // &config, // If AppOrchestrator takes config directly
    );
    info!("AppOrchestrator created and initialized.");

    // --- Main Event Loop ---
    info!("Starting main event loop...");
    loop {
        match orchestrator.process_event_cycle() {
            Ok(OrchestratorStatus::Running) => {
                std::thread::sleep(std::time::Duration::from_millis(
                    CONFIG.performance.min_draw_latency_ms as u64,
                ));
            }
            Ok(OrchestratorStatus::Shutdown) => {
                info!("Orchestrator requested shutdown. Exiting main loop.");
                break;
            }
            Err(e) => {
                error!(
                    "Error in orchestrator event cycle: {:#}. Root cause: {:?}. Exiting.",
                    e,
                    e.root_cause()
                );
                break;
            }
        }
    }

    // --- Cleanup ---
    info!("Shutting down platform...");
    platform.shutdown().context("Failed to shutdown platform")?;
    info!("myterm exited successfully.");

    Ok(())
}
