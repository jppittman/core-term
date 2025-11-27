// In src/main.rs

//! Main entry point and module declarations for `core-term`.

// Declare modules
/// ANSI escape sequence parsing and handling.
pub mod ansi;
/// Color types and definitions.
pub mod color;
/// Configuration management.
pub mod config;
/// Display driver interfaces and implementations.
pub mod display;
/// Glyph definitions.
pub mod glyph;
/// Input/Output subsystems (PTY, event loop).
pub mod io;
/// Keyboard input handling.
pub mod keys;
/// Core logic orchestration (actors).
pub mod orchestrator;
/// Pixel coordinate types.
pub mod pixels;
/// Platform abstraction layer.
pub mod platform;
/// Rendering subsystem.
pub mod renderer;
/// Terminal emulator state machine.
pub mod term;

// Use statements for items needed in main.rs

// Logging
use crate::config::CONFIG;
use anyhow::Context;
use log::{info, warn};

/// Main entry point for the `myterm` application.
fn main() -> anyhow::Result<()> {
    // Initialize the logger to write to /tmp/core-term.log
    // Default filter is "info" if RUST_LOG is not set.
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

    // Set TERM environment variable
    if std::env::var_os("TERM").is_none() {
        std::env::set_var("TERM", "screen-256color");
    }

    let shell_command = std::env::var("SHELL").unwrap_or_else(|_| {
        warn!("SHELL environment variable not set, defaulting to /bin/bash");
        "/bin/bash".to_string()
    });
    let shell_args: Vec<String> = Vec::new();

    info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

    // =========================================================================
    // Platform-Agnostic Initialization (at main scope - actors live here)
    // =========================================================================

    use crate::orchestrator::orchestrator_actor::{OrchestratorActor, OrchestratorArgs};
    use crate::orchestrator::orchestrator_channel::create_orchestrator_channels;
    use crate::term::TerminalEmulator;

    let term_cols = CONFIG.appearance.columns as usize;
    let term_rows = CONFIG.appearance.rows as usize;
    info!("Terminal dimensions: {}x{} cells", term_cols, term_rows);

    // 1. Create unified orchestrator channel (all actors send to same channel)
    let (orchestrator_sender, ui_rx, pty_rx) = create_orchestrator_channels(128);

    // Orchestrator → Platform: Display actions including RequestRedraw (buffered to prevent blocking)
    let (display_action_tx, display_action_rx) = std::sync::mpsc::sync_channel(128);

    // Orchestrator → PTY: Write and ResizePty (buffered to prevent blocking)
    let (pty_action_tx, pty_action_rx) = std::sync::mpsc::sync_channel(128);

    // 3. Spawn PTY EventMonitor (platform-specific implementation, but owned at main scope)
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

    // 4. Spawn VsyncActor (platform-agnostic)
    use crate::io::vsync_actor::VsyncActor;
    let target_fps = CONFIG.performance.target_fps;
    let _vsync_actor = VsyncActor::spawn(orchestrator_sender.clone(), target_fps)
        .context("Failed to spawn VsyncActor")?;
    info!("VsyncActor spawned successfully");

    // =========================================================================
    // Render Thread Initialization
    // =========================================================================

    info!("Spawning render thread...");
    use crate::config::Config;
    use crate::renderer::{spawn_render_thread, Renderer};

    let config = Config::default();
    let renderer = Renderer::new();

    // FIXME: Scale factor should come from display initialization, not hardcoded
    // For now, assume 2.0 scale factor (Retina displays) on macOS with Cocoa
    // X11 (including XQuartz) doesn't do HiDPI scaling, so use 1.0
    #[cfg(all(target_os = "macos", use_cocoa_display))]
    let assumed_scale_factor = 2.0;
    #[cfg(not(all(target_os = "macos", use_cocoa_display)))]
    let assumed_scale_factor = 1.0;

    let cell_width_px = (config.appearance.cell_width_px as f64 * assumed_scale_factor) as usize;
    let cell_height_px = (config.appearance.cell_height_px as f64 * assumed_scale_factor) as usize;

    let render_channels = spawn_render_thread(renderer, cell_width_px, cell_height_px, config)
        .context("Failed to spawn render thread")?;
    info!("Render thread spawned successfully");

    // =========================================================================
    // Platform-Specific Initialization (windowing/display only)
    // =========================================================================

    // Initialize GenericPlatform (works with any display driver)
    let platform = {
        use crate::platform::GenericPlatform;
        info!("Initializing GenericPlatform...");

        // Create platform channels struct
        let platform_channels = crate::platform::PlatformChannels {
            display_action_rx,
            platform_event_tx: orchestrator_sender.clone(),
        };

        GenericPlatform::new(platform_channels, render_channels)
            .context("Failed to initialize GenericPlatform")?
    };

    // Create waker based on display driver
    // CocoaDisplayDriver needs macOS event loop waker; others use no-op
    #[cfg(use_cocoa_display)]
    let waker = {
        use crate::platform::waker::CocoaWaker;
        Box::new(CocoaWaker::new()) as Box<dyn crate::platform::waker::EventLoopWaker>
    };

    #[cfg(not(use_cocoa_display))]
    let waker = Box::new(crate::platform::waker::NoOpWaker)
        as Box<dyn crate::platform::waker::EventLoopWaker>;

    // 5. Spawn OrchestratorActor (platform-agnostic hub)
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

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        panic!("Unsupported target OS. Only Linux and macOS are currently supported.");
    }

    info!("Platform initialized. Starting main event loop...");

    // Run the platform's event loop - blocks until shutdown
    #[cfg(feature = "profiling")]
    {
        info!("Profiling enabled - flamegraph will be generated on exit");
        eprintln!("Profiling enabled - flamegraph will be generated on exit");
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(1000)
            .blocklist(&["libc", "libsystem", "libdyld"])
            .build()
            .expect("Failed to create profiler guard");

        let result = platform.run().context("Platform event loop failed");

        info!("Platform run completed, building profiler report...");
        eprintln!("Platform run completed, building profiler report...");

        match guard.report().build() {
            Ok(report) => {
                // Save as SVG flamegraph
                let flamegraph_path = "/tmp/flamegraph.svg";
                info!("Writing flamegraph to {}", flamegraph_path);
                eprintln!("Writing flamegraph to {}", flamegraph_path);

                match std::fs::File::create(flamegraph_path) {
                    Ok(file) => match report.flamegraph(file) {
                        Ok(_) => {
                            info!("Flamegraph saved to {}", flamegraph_path);
                            eprintln!("Flamegraph saved to {}", flamegraph_path);
                        }
                        Err(e) => {
                            eprintln!("Failed to write flamegraph: {}", e);
                        }
                    },
                    Err(e) => {
                        eprintln!("Failed to create flamegraph file: {}", e);
                    }
                }

                // Save as text format for manual inspection
                let text_path = "/tmp/profile.txt";
                info!("Writing text profile to {}", text_path);
                eprintln!("Writing text profile to {}", text_path);

                match std::fs::File::create(text_path) {
                    Ok(file) => {
                        use std::io::Write;
                        let mut writer = std::io::BufWriter::new(file);
                        match write!(&mut writer, "{:#?}", report) {
                            Ok(_) => {
                                info!("Text profile saved to {}", text_path);
                                eprintln!("Text profile saved to {}", text_path);
                            }
                            Err(e) => {
                                eprintln!("Failed to write text profile: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to create text profile file: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to build profiler report: {:?}", e);
            }
        }

        result?;
    }

    #[cfg(not(feature = "profiling"))]
    platform.run().context("Platform event loop failed")?;

    info!("core-term exited successfully.");

    Ok(())
}
