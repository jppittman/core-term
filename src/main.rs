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
    platform::backends::PlatformState,
    platform::platform_trait::Platform,
    renderer::Renderer,
    term::TerminalEmulator,
};

// Logging
use anyhow::Context;
use log::{error, info, warn};

const DEFAULT_INITIAL_PTY_COLS: u16 = 80;
const DEFAULT_INITIAL_PTY_ROWS: u16 = 24;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_micros()
        .init();

    info!("Starting core...");
    info!("Configuration loaded (using default).");

    if std::env::var_os("TERM").is_none() {
        std::env::set_var("TERM", "screen-256color");
    }

    let shell_command = std::env::var("SHELL").unwrap_or_else(|_| {
        warn!("SHELL environment variable not set, defaulting to /bin/bash");
        "/bin/bash".to_string()
    });
    let shell_args: Vec<String> = Vec::new();
    info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

    let (mut platform, initial_platform_state): (Box<dyn Platform>, PlatformState);

    #[cfg(all(target_os = "linux", not(feature = "wayland")))]
    {
        use crate::platform::linux_x11::LinuxX11Platform;
        info!("Initializing LinuxX11Platform...");
        let (linux_platform, state) = LinuxX11Platform::new(
            DEFAULT_INITIAL_PTY_COLS,
            DEFAULT_INITIAL_PTY_ROWS,
            shell_command,
            shell_args,
        )
        .context("Failed to initialize LinuxX11Platform")?;
        platform = Box::new(linux_platform);
        initial_platform_state = state;
    }
    #[cfg(all(target_os = "linux", feature = "wayland"))]
    {
        use crate::platform::linux_wayland::LinuxWaylandPlatform;
        info!("Initializing LinuxWaylandPlatform...");
        let (wayland_platform, state) = LinuxWaylandPlatform::new(
            DEFAULT_INITIAL_PTY_COLS,
            DEFAULT_INITIAL_PTY_ROWS,
            shell_command,
            shell_args,
        )
        .context("Failed to initialize LinuxWaylandPlatform")?;
        platform = Box::new(wayland_platform);
        initial_platform_state = state;
    }
    #[cfg(target_os = "macos")]
    {
        use crate::platform::macos::MacosPlatform;
        info!("Initializing MacosPlatform...");
        let (macos_platform, state) = MacosPlatform::new(
            DEFAULT_INITIAL_PTY_COLS,
            DEFAULT_INITIAL_PTY_ROWS,
            shell_command,
            shell_args,
        )
        .context("Failed to initialize MacosPlatform")?;
        platform = Box::new(macos_platform);
        initial_platform_state = state;
    }
    #[cfg(not(any(
        all(target_os = "linux", not(feature = "wayland")),
        all(target_os = "linux", feature = "wayland"),
        target_os = "macos"
    )))]
    {
        // This will cause a compile error if no platform is selected,
        // which is good because we need `platform` and `initial_platform_state` to be initialized.
        // However, to be more explicit and avoid "use of uninitialized variable" errors
        // in some analysis tools, we can panic here.
        panic!(
            "Unsupported target OS or feature combination. Only Linux (X11/Wayland with 'wayland' feature) and macOS are currently supported."
        );
    }

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
        platform.as_mut(),
        &mut term_emulator,
        &mut ansi_parser,
        renderer,
    );
    info!("AppOrchestrator created and initialized.");

    info!("Starting main event loop...");
    loop {
        match orchestrator.process_event_cycle() {
            Ok(OrchestratorStatus::Running) => { /* Continue */ }
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
    if let Err(e) = platform.cleanup() {
        error!("Error during platform cleanup: {:?}", e);
    } else {
        info!("Platform cleanup successful.");
    }

    info!("myterm exited successfully.");

    Ok(())
}
