// src/config.rs

//! Defines configuration structures and provides global access to the loaded configuration.
//!
//! The application's configuration is loaded once and made available globally
//! via a lazily initialized static variable `CONFIG`.

// --- Crates and Modules ---
use crate::{
    color::{Color, NamedColor},
    term::cursor::CursorShape, // Assumes CursorShape is in `crate::term::modes`
};
use log::{error, info}; // Added warn, info
use serde::{Deserialize, Serialize};
use std::{sync::LazyLock as Lazy, path::PathBuf};

// --- Global Configuration Access ---

/// Lazily initialized global static storage for the application's configuration.
pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    // This closure is executed once.
    // It attempts to load the configuration and falls back to defaults if necessary.
    match load_config_from_file_or_defaults() {
        Ok(cfg) => {
            info!("Configuration loaded successfully (or defaults used).");
            cfg
        }
        Err(e) => {
            // If load_config_from_file_or_defaults itself can return an error
            // (e.g., for critical unrecoverable parsing issues not handled by defaulting internally),
            // we log it here and still proceed with hardcoded defaults.
            // The current placeholder always returns Ok(Config::default()).
            error!(
                "Critical error during configuration loading: {:?}. Using emergency default configuration.",
                e
            );
            Config::default()
        }
    }
});

/// Placeholder function representing the logic to load configuration.
///
/// In a real application, this would:
/// 1. Determine the configuration file path(s).
/// 2. Read the file content.
/// 3. Deserialize it (e.g., from TOML) into the `Config` struct.
/// 4. If any step fails (file not found, parse error), it could log a warning
///    and return `Ok(Config::default())`, or return an `Err` for more critical issues.
///
/// For simplicity, this version now directly returns `Ok(Config::default())`
/// or could return `anyhow::Result<Config>`.
fn load_config_from_file_or_defaults() -> anyhow::Result<Config> {
    // This function is now expected to handle its own errors internally if it wants
    // to try loading and then fall back to defaults, or it can propagate an error
    // if the loading process itself is critically unrecoverable.

    // Example: If you were loading from "core-term.toml"
    // let config_path = PathBuf::from("core-term.toml");
    // match std::fs::read_to_string(&config_path) {
    //     Ok(content) => {
    //         match toml::from_str(&content) {
    //             Ok(cfg) => {
    //                 info!("Successfully loaded configuration from {:?}.", config_path);
    //                 Ok(cfg)
    //             }
    //             Err(e) => {
    //                 warn!("Failed to parse config file {:?}: {}. Using default configuration.", config_path, e);
    //                 Ok(Config::default()) // Fallback to defaults on parse error
    //             }
    //         }
    //     }
    //     Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
    //         info!("Config file not found at {:?}. Using default configuration.", config_path);
    //         Ok(Config::default()) // Fallback to defaults if file not found
    //     }
    //     Err(e) => {
    //         // For other I/O errors, you might want to propagate them
    //         Err(anyhow::Error::from(e).context(format!("Failed to read config file {:?}", config_path)))
    //     }
    // }

    // Current placeholder behavior: always succeeds with defaults.
    info!("Placeholder: `load_config_from_file_or_defaults` called. Returning default config.");
    Ok(Config::default())
}

// --- Configuration Structures ---

/// Represents the complete configuration for the terminal emulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub appearance: AppearanceConfig,
    pub behavior: BehaviorConfig,
    pub performance: PerformanceConfig,
    pub colors: ColorScheme,
    pub shell: ShellConfig,
    pub mouse: MouseConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            appearance: AppearanceConfig::default(),
            behavior: BehaviorConfig::default(),
            performance: PerformanceConfig::default(),
            colors: ColorScheme::default(),
            shell: ShellConfig::default(),
            mouse: MouseConfig::default(),
        }
    }
}

/// Defines settings related to the visual appearance of the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppearanceConfig {
    pub font: FontConfig,
    pub columns: u16,
    pub rows: u16,
    pub border_pixels: u16,
    pub cursor: CursorConfig,
}

impl Default for AppearanceConfig {
    fn default() -> Self {
        AppearanceConfig {
            font: FontConfig::default(),
            columns: 80,
            rows: 24,
            border_pixels: 2,
            cursor: CursorConfig::default(),
        }
    }
}

/// Font configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FontConfig {
    pub normal: String,
    pub cw_scale: f32,
    pub ch_scale: f32,
}

impl Default for FontConfig {
    fn default() -> Self {
        FontConfig {
            normal: "Liberation Mono:pixelsize=12:antialias=true:autohint=true".to_string(),
            cw_scale: 1.0,
            ch_scale: 1.0,
        }
    }
}

/// Cursor appearance settings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct CursorConfig {
    pub shape: CursorShape,
    pub thickness: u16,
    pub blink_timeout_ms: u32,
}

impl Default for CursorConfig {
    fn default() -> Self {
        CursorConfig {
            shape: CursorShape::SteadyBlock,
            thickness: 2,
            blink_timeout_ms: 800,
        }
    }
}

/// Defines settings related to the operational behavior of the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BehaviorConfig {
    pub scrollback_lines: usize,
    pub tabspaces: u8,
    pub word_delimiters: String,
    pub double_click_timeout_ms: u32,
    pub triple_click_timeout_ms: u32,
    pub bell_volume: i8,
    pub term_env_var: String,
    pub allow_alt_screen: bool,
    pub allow_window_ops: bool,
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        BehaviorConfig {
            scrollback_lines: 1000,
            tabspaces: 8,
            word_delimiters: " `\"'()[]{}<>".to_string(),
            double_click_timeout_ms: 300,
            triple_click_timeout_ms: 600,
            bell_volume: 0,
            term_env_var: "core-256color".to_string(),
            allow_alt_screen: true,
            allow_window_ops: false,
        }
    }
}

/// Defines settings related to performance and rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    pub min_draw_latency_ms: f64,
    pub max_draw_latency_ms: f64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        PerformanceConfig {
            min_draw_latency_ms: 2.0,
            max_draw_latency_ms: 33.0,
        }
    }
}

/// Defines the color scheme for the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorScheme {
    pub foreground: Color,
    pub background: Color,
    pub cursor: Color,
    pub reverse_cursor: Color,
    pub ansi: [Color; 16],
}

impl Default for ColorScheme {
    fn default() -> Self {
        ColorScheme {
            foreground: Color::Named(NamedColor::White),
            background: Color::Named(NamedColor::Black),
            cursor: Color::Named(NamedColor::White),
            reverse_cursor: Color::Named(NamedColor::Black),
            ansi: [
                Color::Named(NamedColor::Black),
                Color::Named(NamedColor::Red),
                Color::Named(NamedColor::Green),
                Color::Named(NamedColor::Yellow),
                Color::Named(NamedColor::Blue),
                Color::Named(NamedColor::Magenta),
                Color::Named(NamedColor::Cyan),
                Color::Named(NamedColor::White),
                Color::Named(NamedColor::BrightBlack),
                Color::Named(NamedColor::BrightRed),
                Color::Named(NamedColor::BrightGreen),
                Color::Named(NamedColor::BrightYellow),
                Color::Named(NamedColor::BrightBlue),
                Color::Named(NamedColor::BrightMagenta),
                Color::Named(NamedColor::BrightCyan),
                Color::Named(NamedColor::BrightWhite),
            ],
        }
    }
}

/// Defines settings related to the shell and its execution environment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ShellConfig {
    pub program: Option<PathBuf>,
    pub args: Vec<String>,
    pub working_directory: Option<PathBuf>,
}

/// Defines settings related to mouse behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MouseConfig {
    pub cursor_shape: String,
    pub force_modifier: String,
}

impl Default for MouseConfig {
    fn default() -> Self {
        MouseConfig {
            cursor_shape: "xterm".to_string(),
            force_modifier: "ShiftMask".to_string(),
        }
    }
}
