// In src/main.rs

//! Main entry point and module declarations for `core-term`.

// Declare modules
pub mod ansi;
pub mod app;
pub mod color;
pub mod config;
pub mod glyph;
pub mod io;
pub mod keys;
pub mod orchestrator;
pub mod pixels;
pub mod platform;
pub mod renderer;
pub mod surface;
pub mod term;

// Use statements for items needed in main.rs
use crate::config::CONFIG;
use anyhow::Context;
use log::{info, warn};

/// Main entry point for the `myterm` application.
fn main() -> anyhow::Result<()> {
    use std::fs::OpenOptions;

    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("/tmp/core-term.log")
        .expect("Failed to open log file");

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_micros()
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();

    info!("Starting core-term...");

    if std::env::var_os("TERM").is_none() {
        std::env::set_var("TERM", "screen-256color");
    }

    let shell_command = std::env::var("SHELL").unwrap_or_else(|_| {
        warn!("SHELL environment variable not set, defaulting to /bin/bash");
        "/bin/bash".to_string()
    });
    let shell_args: Vec<String> = Vec::new();

    info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

    use crate::orchestrator::orchestrator_actor::{OrchestratorActor, OrchestratorArgs};
    use crate::orchestrator::orchestrator_channel::create_orchestrator_channels;
    use crate::term::TerminalEmulator;

    let term_cols = CONFIG.appearance.columns as usize;
    let term_rows = CONFIG.appearance.rows as usize;
    info!("Terminal dimensions: {}x{} cells", term_cols, term_rows);

    let (orchestrator_sender, ui_rx, pty_rx) = create_orchestrator_channels(128);
    let (display_action_tx, display_action_rx) = std::sync::mpsc::sync_channel(128);
    let (pty_action_tx, pty_action_rx) = std::sync::mpsc::sync_channel(128);

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let _event_monitor_actor = {
        use crate::io::event_monitor_actor::EventMonitorActor;
        use crate::io::pty::{NixPty, PtyConfig};

        let shell_args_refs: Vec<&str> = shell_args.iter().map(String::as_str).collect();
        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args_refs,
            initial_cols: CONFIG.appearance.columns,
            initial_rows: CONFIG.appearance.rows,
        };
        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty")?;
        info!("Spawned PTY");

        EventMonitorActor::spawn(pty, orchestrator_sender.clone(), pty_action_rx)
            .context("Failed to spawn EventMonitorActor")?
    };
    info!("EventMonitorActor spawned successfully");

    use crate::io::vsync_actor::VsyncActor;
    let target_fps = CONFIG.performance.target_fps;
    let _vsync_actor = VsyncActor::spawn(orchestrator_sender.clone(), target_fps)
        .context("Failed to spawn VsyncActor")?;
    info!("VsyncActor spawned successfully");

    // Engine Initialization
    use pixelflow_engine::{EngineConfig, EnginePlatform, WindowConfig};

    let engine_config = EngineConfig {
        window: WindowConfig {
            title: CONFIG.appearance.default_title.clone(),
            columns: CONFIG.appearance.columns,
            rows: CONFIG.appearance.rows,
            cell_width_px: CONFIG.appearance.cell_width_px,
            cell_height_px: CONFIG.appearance.cell_height_px,
            initial_x: 100.0,
            initial_y: 100.0,
        },
        performance: CONFIG.performance.clone(),
    };

    let platform =
        EnginePlatform::new(engine_config.into()).context("Failed to initialize EnginePlatform")?;

    let waker = platform.create_waker();

    // Spawn Orchestrator
    let term_emulator = TerminalEmulator::new(term_cols, term_rows);
    let _orchestrator_actor = OrchestratorActor::spawn(
        term_emulator,
        OrchestratorArgs {
            ui_rx,
            pty_rx,
            display_action_tx,
            pty_action_tx,
            waker,
        },
    )
    .context("Failed to spawn OrchestratorActor")?;
    info!("OrchestratorActor spawned successfully");

    // Create App
    use crate::app::CoreTermApp;
    let app = CoreTermApp::new(
        orchestrator_sender.clone(),
        display_action_rx,
        crate::config::Config::default(),
    );

    info!("Platform initialized. Starting main event loop...");

    platform.run(app).context("Platform event loop failed")?;

    info!("core-term exited successfully.");

    Ok(())
}
