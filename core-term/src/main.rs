// In src/main.rs

//! Main entry point for `core-term`.

use anyhow::Context;
use core_term::config::CONFIG;
use log::{info, warn};

struct Args {
    command: Option<String>,
    args: Vec<String>,
}

fn parse_args() -> Args {
    let mut args_iter = std::env::args().skip(1);
    let mut command = None;
    let mut trailing_args = Vec::new();

    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "-c" | "--command" => {
                command = args_iter.next();
            }
            "-h" | "--help" => {
                eprintln!("core-term: A modern terminal emulator\n");
                eprintln!("Usage: core-term [OPTIONS]\n");
                eprintln!("Options:");
                eprintln!("  -c, --command <CMD>  Execute a command instead of launching a shell");
                eprintln!("  -h, --help           Print help");
                eprintln!("  -V, --version        Print version");
                std::process::exit(0);
            }
            "-V" | "--version" => {
                eprintln!("core-term {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            _ => {
                trailing_args.push(arg);
                trailing_args.extend(args_iter);
                break;
            }
        }
    }

    Args {
        command,
        args: trailing_args,
    }
}

/// Get secure log file path using user-specific runtime directory.
/// Falls back to user's cache directory to avoid world-readable /tmp.
fn get_secure_log_path() -> std::path::PathBuf {
    // Prefer XDG_RUNTIME_DIR (user-specific, restricted permissions)
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        return std::path::PathBuf::from(runtime_dir).join("core-term.log");
    }

    // Fall back to user's cache directory
    if let Some(home) = std::env::var_os("HOME") {
        let cache_dir = std::path::PathBuf::from(home)
            .join(".cache")
            .join("core-term");
        // Create directory with restricted permissions if it doesn't exist
        if !cache_dir.exists() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                let _ = std::fs::DirBuilder::new().mode(0o700).create(&cache_dir);
            }
            #[cfg(not(unix))]
            {
                let _ = std::fs::create_dir_all(&cache_dir);
            }
        }
        return cache_dir.join("core-term.log");
    }

    // Last resort: use /tmp with PID to reduce collision risk
    std::path::PathBuf::from(format!("/tmp/core-term-{}.log", std::process::id()))
}

/// Main entry point for the `myterm` application.
fn main() -> anyhow::Result<()> {
    let args = parse_args();
    use std::fs::OpenOptions;

    // Start CPU profiler if feature enabled
    #[cfg(feature = "profiling")]
    let profiler_guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libsystem", "pthread", "vdso"])
        .build()
        .expect("Failed to start profiler");

    let log_path = get_secure_log_path();

    // Open log file securely - avoid following symlinks on Unix
    #[cfg(unix)]
    let log_file = {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            // O_NOFOLLOW prevents symlink attacks
            .custom_flags(libc::O_NOFOLLOW)
            .open(&log_path)
            .expect("Failed to open log file")
    };

    #[cfg(not(unix))]
    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
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

    use core_term::term::TerminalEmulator;

    let term_cols = CONFIG.appearance.columns as usize;
    let term_rows = CONFIG.appearance.rows as usize;
    info!("Terminal dimensions: {}x{} cells", term_cols, term_rows);

    // Create channel for PTY commands (writes and resizes)
    // pty_cmd: app → write thread (PtyCommand: Write or Resize)
    // PTY reads now go through priority channel created by spawn_terminal_app
    use core_term::io::PtyCommand;
    let (pty_cmd_tx, pty_cmd_rx): (
        std::sync::mpsc::SyncSender<PtyCommand>,
        std::sync::mpsc::Receiver<PtyCommand>,
    ) = std::sync::mpsc::sync_channel(128);

    // Create terminal emulator
    let term_emulator = TerminalEmulator::new(term_cols, term_rows);

    // Engine Initialization with EngineTroupe
    use pixelflow_runtime::{EngineConfig, EngineTroupe, WindowConfig};

    let engine_config = EngineConfig {
        window: WindowConfig {
            title: CONFIG.appearance.default_title.clone(),
            width: CONFIG.appearance.columns as u32 * CONFIG.appearance.cell_width_px as u32,
            height: CONFIG.appearance.rows as u32 * CONFIG.appearance.cell_height_px as u32,
        },
        performance: CONFIG.performance.clone(),
    };

    info!("Engine config created. Creating EngineTroupe...");

    // Phase 1: Create troupe (channels ready, no threads spawned yet)
    let troupe =
        EngineTroupe::with_config(engine_config.clone()).context("Failed to create EngineTroupe")?;

    // Phase 2: Get unregistered engine handle
    let unregistered_handle = troupe.engine_handle();

    {
        use core_term::io::event_monitor_actor::EventMonitorActor;
        use core_term::io::pty::{NixPty, PtyConfig};
        use core_term::terminal_app::{spawn_terminal_app, TerminalAppParams, TerminalAppSender};

        // Phase 3: Spawn terminal app with UNREGISTERED handle
        // The app will call register() during its initialization
        let params = TerminalAppParams {
            emulator: term_emulator,
            pty_tx: pty_cmd_tx,
            config: core_term::config::Config::default(),
            unregistered_engine: unregistered_handle,
            window_config: engine_config.window.clone(),
        };
        let (app_handle, _app_thread_handle) =
            spawn_terminal_app(params).context("Failed to spawn terminal app")?;

        // Create adapter for PTY parser to send to app
        let app_sender = Box::new(TerminalAppSender::new(app_handle.clone()));

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

        let _event_monitor_actor = EventMonitorActor::spawn(pty, app_sender, pty_cmd_rx)
            .context("Failed to spawn EventMonitorActor")?;
        info!("EventMonitorActor spawned successfully");

        // Phase 3: Run troupe (blocks on main thread)
        // The _app_handle keeps the terminal app channel alive
        let _ = app_handle; // Keep app_handle alive until troupe completes
        troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    info!("core-term exited successfully.");

    // Write flamegraph on exit if profiling enabled
    #[cfg(feature = "profiling")]
    {
        // Use secure path in same directory as log file
        let path = get_secure_log_path()
            .parent()
            .map(|p| p.join("core-term-flamegraph.svg"))
            .unwrap_or_else(|| {
                std::path::PathBuf::from(format!(
                    "/tmp/core-term-flamegraph-{}.svg",
                    std::process::id()
                ))
            });
        info!("Writing flamegraph to {}...", path.display());
        match profiler_guard.report().build() {
            Ok(report) => match std::fs::File::create(&path) {
                Ok(file) => {
                    if let Err(e) = report.flamegraph(file) {
                        warn!("Failed to write flamegraph: {}", e);
                    } else {
                        info!("Flamegraph written to {}", path.display());
                    }
                }
                Err(e) => warn!("Failed to create {}: {}", path.display(), e),
            },
            Err(e) => warn!("Failed to build profiler report: {:?}", e),
        }
    }

    Ok(())
}
