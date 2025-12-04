// In src/main.rs

//! Main entry point and module declarations for `core-term`.

// Declare modules
pub mod ansi;
pub mod color;
pub mod config;
pub mod glyph;
pub mod io;
pub mod keys;
pub mod messages;
pub mod surface;
pub mod term;
pub mod terminal_app;

// Use statements for items needed in main.rs
use crate::config::CONFIG;
use anyhow::Context;
use log::{info, warn};

/// Main entry point for the `myterm` application.
fn main() -> anyhow::Result<()> {
    use std::fs::OpenOptions;

    // Start CPU profiler if feature enabled
    #[cfg(feature = "profiling")]
    let profiler_guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libsystem", "pthread", "vdso"])
        .build()
        .expect("Failed to start profiler");

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
    #[cfg(feature = "profiling")]
    info!("CPU profiling enabled - flamegraph.svg will be written on exit");

    if std::env::var_os("TERM").is_none() {
        std::env::set_var("TERM", "screen-256color");
    }

    let shell_command = std::env::var("SHELL").unwrap_or_else(|_| {
        warn!("SHELL environment variable not set, defaulting to /bin/bash");
        "/bin/bash".to_string()
    });
    let shell_args: Vec<String> = Vec::new();

    info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

    use crate::term::TerminalEmulator;

    let term_cols = CONFIG.appearance.columns as usize;
    let term_rows = CONFIG.appearance.rows as usize;
    info!("Terminal dimensions: {}x{} cells", term_cols, term_rows);

    // Create channels for PTY communication
    // pty_cmd: read thread → app (parsed ANSI commands)
    // pty_write: app → write thread (raw bytes to write to PTY)
    let (pty_cmd_tx, pty_cmd_rx) = std::sync::mpsc::sync_channel(128);
    let (pty_write_tx, pty_write_rx) = std::sync::mpsc::sync_channel(128);

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
        EnginePlatform::new(engine_config).context("Failed to initialize EnginePlatform")?;

    // Spawn PTY I/O actor
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

        EventMonitorActor::spawn(pty, pty_cmd_tx, pty_write_rx)
            .context("Failed to spawn EventMonitorActor")?
    };
    info!("EventMonitorActor spawned successfully");

    // Create terminal emulator
    let term_emulator = TerminalEmulator::new(term_cols, term_rows);

    info!("Platform initialized. Starting main event loop...");

    // Spawn app worker in its own thread, get proxy for engine
    // Platform-specific pixel format
    #[cfg(target_os = "macos")]
    let _worker_handle = {
        use crate::terminal_app::spawn_terminal_app;
        use pixelflow_render::CocoaPixel;

        let (proxy, handle) = spawn_terminal_app::<CocoaPixel>(
            term_emulator,
            pty_cmd_rx,
            pty_write_tx,
            crate::config::Config::default(),
        )
        .context("Failed to spawn terminal app worker")?;

        platform.run(proxy).context("Platform event loop failed")?;
        handle
    };

    #[cfg(target_os = "linux")]
    let _worker_handle = {
        use crate::terminal_app::spawn_terminal_app;
        use pixelflow_render::X11Pixel;

        let (proxy, handle) = spawn_terminal_app::<X11Pixel>(
            term_emulator,
            pty_cmd_rx,
            pty_write_tx,
            crate::config::Config::default(),
        )
        .context("Failed to spawn terminal app worker")?;

        platform.run(proxy).context("Platform event loop failed")?;
        handle
    };

    info!("core-term exited successfully.");

    // Write flamegraph on exit if profiling enabled
    #[cfg(feature = "profiling")]
    {
        let path = "/tmp/core-term-flamegraph.svg";
        info!("Writing flamegraph to {}...", path);
        match profiler_guard.report().build() {
            Ok(report) => {
                match std::fs::File::create(path) {
                    Ok(file) => {
                        if let Err(e) = report.flamegraph(file) {
                            warn!("Failed to write flamegraph: {}", e);
                        } else {
                            info!("Flamegraph written to {}", path);
                        }
                    }
                    Err(e) => warn!("Failed to create {}: {}", path, e),
                }
            }
            Err(e) => warn!("Failed to build profiler report: {:?}", e),
        }
    }

    Ok(())
}
