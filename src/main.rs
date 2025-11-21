// In src/main.rs

// Declare modules
pub mod ansi;
pub mod color;
pub mod config;
pub mod glyph;
pub mod keys;
pub mod orchestrator;
pub mod platform;
pub mod rasterizer;
pub mod renderer;
pub mod term;

// Use statements for items needed in main.rs
use crate::platform::platform_trait::Platform;

// Logging
use anyhow::Context;
use log::{info, warn};

// Default initial PTY dimensions (hints for Platform::new)
const DEFAULT_INITIAL_PTY_COLS: u16 = 80;
const DEFAULT_INITIAL_PTY_ROWS: u16 = 24;

/// Main entry point for the `myterm` application.
fn main() -> anyhow::Result<()> {
    // Initialize the logger. Default filter is "info" if RUST_LOG is not set.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_micros()
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

    // Create platform - it spawns all actors internally
    #[cfg(target_os = "macos")]
    let platform = {
        use crate::platform::macos::MacosPlatform;
        info!("Initializing MacosPlatform...");
        let (platform, _initial_state) = MacosPlatform::new(
            DEFAULT_INITIAL_PTY_COLS,
            DEFAULT_INITIAL_PTY_ROWS,
            shell_command,
            shell_args,
        )
        .context("Failed to initialize MacosPlatform")?;
        platform
    };

    #[cfg(target_os = "linux")]
    let platform = {
        use crate::platform::backends::x11::XDriver;
        use crate::platform::linux_x11::LinuxX11Platform;
        info!("Initializing LinuxX11Platform...");
        let (platform, _initial_state) = LinuxX11Platform::<XDriver>::new(
            DEFAULT_INITIAL_PTY_COLS,
            DEFAULT_INITIAL_PTY_ROWS,
            shell_command,
            shell_args,
        )
        .context("Failed to initialize LinuxX11Platform")?;
        platform
    };

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        panic!("Unsupported target OS. Only Linux and macOS are currently supported.");
    }

    info!("Platform initialized. Starting main event loop...");

    // Run the platform's event loop - blocks until shutdown
    platform.run().context("Platform event loop failed")?;

    info!("core-term exited successfully.");

    Ok(())
}
