// src/config.rs

//! Defines the configuration structures for the `core` terminal emulator.
//!
//! This module provides a set of structs that can be deserialized from a
//! configuration file (e.g., TOML, JSON, YAML) to customize the terminal's
//! appearance, behavior, and other settings.
//!
//! The design aims for clarity and practicality, allowing users to easily
//! understand and modify settings. Default values are provided for most options,
//! inspired by sensible defaults like those found in `st` and other common
//! terminal emulators.

// Serde is used for deserializing the configuration from a file.
// The `Serialize` trait is also derived for convenience, allowing the current
// configuration to be exported if needed.
use serde::{Deserialize, Serialize};
use std::path::PathBuf; // For paths, like shell path.

// Import color definitions from the main color module.
// This ensures consistency in how colors are represented throughout the application.
use crate::color::{Color, NamedColor};
use crate::backends::{KeySymbol, Modifiers}; // For KeybindingConfig

// --- Top-Level Configuration Structure ---

/// Represents the complete configuration for the terminal emulator.
///
/// This struct is the root of the configuration and is intended to be
/// deserialized from a configuration file. It groups settings into logical
/// categories like appearance, behavior, and keybindings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)] // Apply default values for the entire struct if a field is missing.
pub struct Config {
    /// Appearance-related settings.
    pub appearance: AppearanceConfig,
    /// Behavior-related settings.
    pub behavior: BehaviorConfig,
    /// Performance-related settings.
    pub performance: PerformanceConfig,
    /// Color scheme configuration.
    pub colors: ColorScheme,
    /// Shell and execution environment settings.
    pub shell: ShellConfig,
    /// Mouse behavior settings.
    pub mouse: MouseConfig,
    /// Keybinding configurations.
    pub keybindings: KeybindingsConfig,
}

// --- Keybinding Configuration ---

/// Represents a combination of a key and modifiers for keybindings.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct KeyCombination {
    #[serde(default)] // Use default for symbol if missing in config
    pub symbol: KeySymbol,
    #[serde(default)] // Use default for modifiers if missing in config
    pub modifiers: Modifiers,
}

/// Defines keybindings for various actions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct KeybindingsConfig {
    /// Example: Keybinding for copying text.
    pub copy: Option<KeyCombination>,
    /// Example: Keybinding for pasting text.
    pub paste: Option<KeyCombination>,
    // Add other actions and their KeyCombination here.
}


// --- Appearance Configuration ---

/// Defines settings related to the visual appearance of the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppearanceConfig {
    /// Font configuration.
    pub font: FontConfig,
    /// Default number of columns for the terminal.
    /// This might be overridden by the backend driver based on window size.
    pub columns: u16,
    /// Default number of rows for the terminal.
    /// This might be overridden by the backend driver based on window size.
    pub rows: u16,
    /// Thickness of the window border in pixels.
    pub border_pixels: u16,
    /// Cursor appearance settings.
    pub cursor: CursorConfig,
}

impl Default for AppearanceConfig {
    fn default() -> Self {
        AppearanceConfig {
            font: FontConfig::default(),
            columns: 80,
            rows: 24,
            border_pixels: 2, // Consistent with st's default
            cursor: CursorConfig::default(),
        }
    }
}

/// Font configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FontConfig {
    /// Primary font string (e.g., "Liberation Mono:pixelsize=12:antialias=true:autohint=true").
    /// The format is typically Pango-style or Fontconfig-style.
    pub normal: String,
    // TODO: Consider adding separate bold/italic font strings if advanced font handling is desired.
    // pub bold: Option<String>,
    // pub italic: Option<String>,
    // pub bold_italic: Option<String>,
    /// Scale factor for character width.
    pub cw_scale: f32,
    /// Scale factor for character height.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CursorConfig {
    /// Default cursor shape.
    /// For example, 2 for Block, 4 for Underline, 6 for Bar.
    /// (Values inspired by st's cursorshape).
    pub shape: u16,
    /// Thickness of underline and bar cursors in pixels.
    pub thickness: u16,
    /// Blinking cursor timeout in milliseconds. Set to 0 to disable blinking.
    pub blink_timeout_ms: u32,
}

impl Default for CursorConfig {
    fn default() -> Self {
        CursorConfig {
            shape: 2, // Block cursor
            thickness: 2,
            blink_timeout_ms: 800,
        }
    }
}

// --- Behavior Configuration ---

/// Defines settings related to the operational behavior of the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BehaviorConfig {
    /// Number of lines to keep in scrollback.
    /// Note: The core `TerminalEmulator` as per NORTH_STAR.md does not implement
    /// scrollback itself. This setting would be for a component that wraps it or
    /// for a potential future extension. For now, it's included for completeness
    /// as it's a very common terminal setting.
    pub scrollback_lines: usize,
    /// Number of spaces per tab character.
    pub tabspaces: u8,
    /// Characters considered as word delimiters for selection purposes.
    pub word_delimiters: String,
    /// Timeout in milliseconds for double-click selection.
    pub double_click_timeout_ms: u32,
    /// Timeout in milliseconds for triple-click selection.
    pub triple_click_timeout_ms: u32,
    /// Bell volume (-100 to 100). 0 disables the bell.
    pub bell_volume: i8,
    /// Value for the TERM environment variable.
    pub term_env_var: String,
    /// If true, allows the use of alternate screen buffer.
    pub allow_alt_screen: bool,
    /// If true, allows certain non-interactive window operations (e.g., setting clipboard via OSC).
    /// This is potentially insecure and should be used with caution.
    pub allow_window_ops: bool,
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        BehaviorConfig {
            scrollback_lines: 1000, // A common default
            tabspaces: 8,
            word_delimiters: " `\"'()[]{}<>".to_string(), // Similar to st
            double_click_timeout_ms: 300,
            triple_click_timeout_ms: 600,
            bell_volume: 0, // Disabled by default, as per st
            term_env_var: "core-256color".to_string(), // Custom TERM name
            allow_alt_screen: true,
            allow_window_ops: false, // Secure default
        }
    }
}

// --- Performance Configuration ---

/// Defines settings related to performance and rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    /// Minimum draw latency in milliseconds.
    /// Within the latency range, drawing occurs when content stops arriving (idle).
    /// Lower values can increase responsiveness but may cause more tearing/flicker.
    pub min_draw_latency_ms: f64,
    /// Maximum draw latency in milliseconds.
    /// The terminal will draw even if not idle after this period.
    pub max_draw_latency_ms: f64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        PerformanceConfig {
            min_draw_latency_ms: 2.0,  // Inspired by st
            max_draw_latency_ms: 33.0, // Inspired by st
        }
    }
}

// --- Color Scheme Configuration ---

/// Defines the color scheme for the terminal.
/// Uses the `crate::color::Color` enum for color representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorScheme {
    /// Default foreground color.
    pub foreground: Color,
    /// Default background color.
    pub background: Color,
    /// Cursor color.
    pub cursor: Color,
    /// Reverse cursor color (used when cursor is over selected text, for example).
    /// Note: The actual rendering of a "reverse cursor" might involve swapping
    /// the underlying cell's fg/bg rather than using this color directly.
    /// This color is provided for themes that might want a specific reverse cursor color.
    pub reverse_cursor: Color,

    /// The 16 standard ANSI colors (normal and bright).
    /// Index 0-7: Normal colors (Black, Red, Green, Yellow, Blue, Magenta, Cyan, White)
    /// Index 8-15: Bright colors (BrightBlack, BrightRed, etc.)
    pub ansi: [Color; 16],
    // Optional: Configuration for the 256-color palette (indices 16-255).
    // Most terminals derive these algorithmically. Providing overrides might be complex
    // to represent in a config file unless done carefully (e.g., a map of index to Color).
    // For now, we'll assume the standard algorithmic generation for 256 colors.
    // pub extended_palette_overrides: Option<std::collections::HashMap<u8, Color>>,
}

impl Default for ColorScheme {
    fn default() -> Self {
        // Defaults inspired by st's color scheme
        ColorScheme {
            foreground: Color::Named(NamedColor::White), // Typically a light gray or white
            background: Color::Named(NamedColor::Black),
            cursor: Color::Named(NamedColor::White), // Often same as foreground or a distinct color
            reverse_cursor: Color::Named(NamedColor::Black), // Often same as background

            ansi: [
                // Normal
                Color::Named(NamedColor::Black),   // 0
                Color::Named(NamedColor::Red),     // 1
                Color::Named(NamedColor::Green),   // 2
                Color::Named(NamedColor::Yellow),  // 3
                Color::Named(NamedColor::Blue),    // 4
                Color::Named(NamedColor::Magenta), // 5
                Color::Named(NamedColor::Cyan),    // 6
                Color::Named(NamedColor::White),   // 7
                // Bright
                Color::Named(NamedColor::BrightBlack),   // 8
                Color::Named(NamedColor::BrightRed),     // 9
                Color::Named(NamedColor::BrightGreen),   // 10
                Color::Named(NamedColor::BrightYellow),  // 11
                Color::Named(NamedColor::BrightBlue),    // 12
                Color::Named(NamedColor::BrightMagenta), // 13
                Color::Named(NamedColor::BrightCyan),    // 14
                Color::Named(NamedColor::BrightWhite),   // 15
            ],
        }
    }
}

// --- Shell Configuration ---

/// Defines settings related to the shell and its execution environment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ShellConfig {
    /// Path to the shell executable (e.g., "/bin/bash").
    /// If `None`, the system's default shell (from SHELL env var or /etc/passwd) is used.
    pub program: Option<PathBuf>,
    /// Arguments to pass to the shell program.
    pub args: Vec<String>,
    /// Optional working directory for the shell. If `None`, defaults to user's home or current dir.
    pub working_directory: Option<PathBuf>,
}

// --- Mouse Configuration ---
/// Defines settings related to mouse behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MouseConfig {
    /// Default shape of the mouse cursor (X11 cursor name or an enum).
    /// Example: "left_ptr" for X11, or an internal enum value.
    /// For simplicity, using a string here that the X11 backend can interpret.
    pub cursor_shape: String,
    /// Modifier key (e.g., "ShiftMask", "Mod1Mask") that, when active,
    /// forces mouse selection/shortcuts even when a mouse reporting mode is active.
    /// Set to empty string or "None" to disable.
    pub force_modifier: String,
    // TODO: Define mouse shortcuts if they are to be configurable.
    // pub shortcuts: Vec<MouseShortcutConfig>,
}

impl Default for MouseConfig {
    fn default() -> Self {
        MouseConfig {
            cursor_shape: "xterm".to_string(), // XC_xterm from X11/cursorfont.h
            force_modifier: "ShiftMask".to_string(), // Consistent with st
        }
    }
}
