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
use clap::Parser;
use log::{info, warn};

/// core-term: A modern terminal emulator
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Execute a command instead of launching a shell
    #[arg(short = 'c', long = "command")]
    command: Option<String>,

    /// Additional arguments to pass to the command
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

/// Main entry point for the `myterm` application.
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
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

    // Determine command to execute based on -c flag
    let (shell_command, shell_args) = if let Some(command) = args.command {
        // Execute command with -c flag
        let shell = std::env::var("SHELL").unwrap_or_else(|_| {
            warn!("SHELL environment variable not set, defaulting to /bin/bash");
            "/bin/bash".to_string()
        });

        // Use shell to execute the command string
        let mut cmd_args = vec!["-c".to_string(), command];

        // If there are additional args, pass them as positional arguments
        // In shell -c mode, $0 will be "sh", and $1, $2, etc. will be the args
        if !args.args.is_empty() {
            cmd_args.push("--".to_string());
            cmd_args.extend(args.args);
        }

        info!("Executing command with -c flag: {} {:?}", shell, cmd_args);
        (shell, cmd_args)
    } else {
        // Launch interactive shell
        let shell = std::env::var("SHELL").unwrap_or_else(|_| {
            warn!("SHELL environment variable not set, defaulting to /bin/bash");
            "/bin/bash".to_string()
        });
        info!("Launching interactive shell: {}", shell);
        (shell, Vec::new())
    };

    info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

    use crate::term::TerminalEmulator;

    let term_cols = CONFIG.appearance.columns as usize;
    let term_rows = CONFIG.appearance.rows as usize;
    info!("Terminal dimensions: {}x{} cells", term_cols, term_rows);

    // Create channel for PTY writes
    // pty_write: app → write thread (raw bytes to write to PTY)
    // PTY reads now go through priority channel created by spawn_terminal_app
    let (pty_write_tx, pty_write_rx) = std::sync::mpsc::sync_channel(128);

    // Create terminal emulator
    let term_emulator = TerminalEmulator::new(term_cols, term_rows);

    // Engine Initialization with new API
    use pixelflow_engine::{EngineConfig, WindowConfig};

    // Calculate window size in pixels from terminal cell dimensions
    let window_width = (CONFIG.appearance.columns as u32) * (CONFIG.appearance.cell_width_px as u32);
    let window_height = (CONFIG.appearance.rows as u32) * (CONFIG.appearance.cell_height_px as u32);

    let engine_config = EngineConfig {
        window: WindowConfig {
            title: CONFIG.appearance.default_title.clone(),
            width: window_width,
            height: window_height,
            initial_x: 100.0,
            initial_y: 100.0,
        },
        performance: CONFIG.performance.clone(),
    };

    info!("Engine config created. Starting main event loop...");

    // Platform-specific pixel format
    #[cfg(target_os = "macos")]
    {
        use crate::io::event_monitor_actor::EventMonitorActor;
        use crate::io::pty::{NixPty, PtyConfig};
        use crate::terminal_app::spawn_terminal_app;
        use pixelflow_render::CocoaPixel;
        use pixelflow_engine::EnginePlatform;

        // Create PTY command channel
        let (pty_cmd_tx, pty_cmd_rx) = std::sync::mpsc::sync_channel(128);

        // 1. Create platform (no spawning)
        let platform = EnginePlatform::new(engine_config)
            .context("Failed to create platform")?;

        // 2. Init platform (spawns engine, returns handle)
        let (platform, engine_handle) = platform.init()
            .context("Failed to initialize platform")?;

        // 3. Spawn terminal app WITH engine handle
        let app_handle = spawn_terminal_app::<CocoaPixel>(
            term_emulator,
            pty_write_tx,
            pty_cmd_rx,
            crate::config::Config::default(),
            engine_handle.clone(),
        );

        // Spawn PTY
        let shell_args_refs: Vec<&str> = shell_args.iter().map(String::as_str).collect();
        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args_refs,
            initial_cols: CONFIG.appearance.columns,
            initial_rows: CONFIG.appearance.rows,
        };
        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty")?;
        info!("Spawned PTY");

        let _event_monitor_actor = EventMonitorActor::spawn(pty, pty_cmd_tx, pty_write_rx)
            .context("Failed to spawn EventMonitorActor")?;
        info!("EventMonitorActor spawned successfully");

        // 4. Run platform (registers app, blocks until quit)
        platform.run(app_handle, &engine_handle)
            .context("Engine run failed")?;
    }

    #[cfg(target_os = "linux")]
    {
        use crate::io::event_monitor_actor::EventMonitorActor;
        use crate::io::pty::{NixPty, PtyConfig};
        use crate::terminal_app::spawn_terminal_app;
        use pixelflow_render::X11Pixel;
        use pixelflow_engine::EnginePlatform;

        // Create PTY command channel
        let (pty_cmd_tx, pty_cmd_rx) = std::sync::mpsc::sync_channel(128);

        // 1. Create platform (no spawning)
        let platform = EnginePlatform::new(engine_config)
            .context("Failed to create platform")?;

        // 2. Init platform (spawns engine, returns handle)
        let (platform, engine_handle) = platform.init()
            .context("Failed to initialize platform")?;

        // 3. Spawn terminal app WITH engine handle
        let app_handle = spawn_terminal_app::<X11Pixel>(
            term_emulator,
            pty_write_tx,
            pty_cmd_rx,
            crate::config::Config::default(),
            engine_handle.clone(),
        );

        // Spawn PTY
        let shell_args_refs: Vec<&str> = shell_args.iter().map(String::as_str).collect();
        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args_refs,
            initial_cols: CONFIG.appearance.columns,
            initial_rows: CONFIG.appearance.rows,
        };
        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty")?;
        info!("Spawned PTY");

        let _event_monitor_actor = EventMonitorActor::spawn(pty, pty_cmd_tx, pty_write_rx)
            .context("Failed to spawn EventMonitorActor")?;
        info!("EventMonitorActor spawned successfully");

        // 4. Run platform (registers app, blocks until quit)
        platform.run(app_handle, &engine_handle)
            .context("Engine run failed")?;
    }

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
