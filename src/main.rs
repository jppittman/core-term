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
    orchestrator::{AppOrchestrator, OrchestratorStatus},
    platform::actions::PlatformAction,
    platform::platform_trait::Platform, // Trait needed for platform methods
    renderer::Renderer,
    term::TerminalEmulator,
};

// Platform-specific imports
#[cfg(all(target_os = "linux", feature = "wayland"))]
use crate::platform::linux_wayland::WaylandPlatform;
#[cfg(all(target_os = "linux", not(feature = "wayland")))]
use crate::platform::linux_x11::LinuxX11Platform;

// Logging
use anyhow::Context;
use log::{error, info, warn};

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

    info!("Configuration loaded (using default).");

    let shell_command = std::env::var("SHELL").unwrap_or_else(|_| {
        warn!("SHELL environment variable not set, defaulting to /bin/bash");
        "/bin/bash".to_string()
    });
    let shell_args: Vec<String> = Vec::new();

    info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

    // --- Instantiate Concrete Platform ---
    let (mut platform_instance, initial_platform_state): (Box<dyn Platform>, _) = {
        #[cfg(all(target_os = "linux", feature = "wayland"))]
        {
            info!("Initializing WaylandPlatform...");
            let (platform, state) = WaylandPlatform::new(
                DEFAULT_INITIAL_PTY_COLS,
                DEFAULT_INITIAL_PTY_ROWS,
                shell_command.clone(),
                shell_args.clone(),
            )
            .context("Failed to initialize WaylandPlatform")?;
            (Box::new(platform), state)
        }
        #[cfg(all(target_os = "linux", not(feature = "wayland")))]
        {
            info!("Initializing LinuxX11Platform...");
            let (platform, state) = LinuxX11Platform::new(
                DEFAULT_INITIAL_PTY_COLS,
                DEFAULT_INITIAL_PTY_ROWS,
                shell_command.clone(),
                shell_args.clone(),
            )
            .context("Failed to initialize LinuxX11Platform")?;
            (Box::new(platform), state)
        }
        #[cfg(not(target_os = "linux"))]
        {
            // Placeholder for other OSes or a compile error
            panic!("Unsupported OS: This application currently only supports Linux with X11 or Wayland.");
            // Or, to make it a compile error:
            // compile_error!("Unsupported OS: This application currently only supports Linux with X11 or Wayland.");
        }
    };

    // The `platform_instance` is now a Box<dyn Platform>.
    // The rest of the code will use `platform_instance` via the `Platform` trait.
    let platform: &mut dyn Platform = platform_instance.as_mut();


    info!(
        "Platform initialized. Initial state: {:?}",
        initial_platform_state
    );

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

    let mut term_emulator = TerminalEmulator::new(term_cols, term_rows);
    info!("TerminalEmulator initialized.");

    let mut ansi_parser = AnsiProcessor::new();
    info!("AnsiProcessor initialized.");

    let renderer = Renderer::new();
    info!("Renderer initialized.");

    platform
        .dispatch_actions(vec![PlatformAction::ResizePty {
            cols: term_cols as u16,
            rows: term_rows as u16,
        }])
        .context("Failed to dispatch initial PTY resize command")?;
    info!(
        "Dispatched initial PTY resize to {}x{}",
        term_cols, term_rows
    );

    let mut orchestrator = AppOrchestrator::new(
        platform, // Pass the &mut dyn Platform
        &mut term_emulator,
        &mut ansi_parser,
        renderer,
    );
    info!("AppOrchestrator created and initialized.");

    info!("Starting main event loop...");
    loop {
        match orchestrator.process_event_cycle() {
            Ok(OrchestratorStatus::Running) => {
                continue;
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

    info!("Shutting down platform...");
    // platform_instance still owns the Box, use it to call shutdown.
    platform_instance.shutdown().context("Failed to shutdown platform")?;
    info!("myterm exited successfully.");

    Ok(())
}
