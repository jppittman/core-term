// src/color.rs

//! Defines color-related enums (`NamedColor`, `Color`) and conversion functions.

use log::warn;
use serde::{Deserialize, Serialize}; // For logging warnings if needed (e.g., invalid index)

/// Standard ANSI named colors (indices 0-15).
/// These are the 8 normal and 8 bright colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum NamedColor {
    Black = 0,
    Red = 1,
    Green = 2,
    Yellow = 3,
    Blue = 4,
    Magenta = 5,
    Cyan = 6,
    White = 7,       // Also known as Grey
    BrightBlack = 8, // Also known as Dark Grey / Bright Grey
    BrightRed = 9,
    BrightGreen = 10,
    BrightYellow = 11,
    BrightBlue = 12,
    BrightMagenta = 13,
    BrightCyan = 14,
    BrightWhite = 15,
}

impl NamedColor {
    /// Converts a u8 index (0-15) to a `NamedColor`.
    ///
    /// # Panics
    /// Panics if the index is out of the valid range (0-15).
    pub fn from_index(idx: u8) -> Self {
        if idx > 15 {
            panic!("Invalid NamedColor index: {}. Must be 0-15.", idx);
        }
        // SAFETY: The check above ensures idx is within the valid range for NamedColor's repr(u8).
        unsafe { std::mem::transmute(idx) }
    }

    /// Returns the `Color::Rgb` representation of this named color.
    /// These are common sRGB values used by many terminals.
    pub fn to_rgb_color(&self) -> Color {
        // Changed return type to Color
        match self {
            NamedColor::Black => Color::Rgb(0, 0, 0),
            NamedColor::Red => Color::Rgb(205, 0, 0),
            NamedColor::Green => Color::Rgb(0, 205, 0),
            NamedColor::Yellow => Color::Rgb(205, 205, 0),
            NamedColor::Blue => Color::Rgb(0, 0, 238),
            NamedColor::Magenta => Color::Rgb(205, 0, 205),
            NamedColor::Cyan => Color::Rgb(0, 205, 205),
            NamedColor::White => Color::Rgb(229, 229, 229),
            NamedColor::BrightBlack => Color::Rgb(127, 127, 127),
            NamedColor::BrightRed => Color::Rgb(255, 0, 0),
            NamedColor::BrightGreen => Color::Rgb(0, 255, 0),
            NamedColor::BrightYellow => Color::Rgb(255, 255, 0),
            NamedColor::BrightBlue => Color::Rgb(92, 92, 255),
            NamedColor::BrightMagenta => Color::Rgb(255, 0, 255),
            NamedColor::BrightCyan => Color::Rgb(0, 255, 255),
            NamedColor::BrightWhite => Color::Rgb(255, 255, 255),
        }
    }
}

/// Represents a color value used in the terminal.
/// Can be a default placeholder, a standard named ANSI color,
/// an indexed color from the 256-color palette, or an RGB true color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Color {
    /// Default foreground or background color, to be resolved by the renderer
    /// or backend based on its own defaults.
    Default,
    /// A standard named ANSI color (indices 0-15).
    Named(NamedColor),
    /// An indexed color from the 256-color palette (indices 0-255).
    /// Note: Indices 0-15 can also be represented via `Color::Named`.
    Indexed(u8),
    /// An RGB true color, with each component from 0 to 255.
    Rgb(u8, u8, u8),
}

impl Default for Color {
    /// Returns `Color::Default` as the default color.
    fn default() -> Self {
        Color::Default
    }
}

// Constants for 256-color palette indexing
const ANSI_NAMED_COLOR_COUNT: u8 = 16;
const COLOR_CUBE_OFFSET: u8 = 16;
const COLOR_CUBE_SIZE: u8 = 6; // 6x6x6 cube
const COLOR_CUBE_TOTAL_COLORS: u8 = COLOR_CUBE_SIZE * COLOR_CUBE_SIZE * COLOR_CUBE_SIZE; // 216
const GRAYSCALE_OFFSET: u8 = COLOR_CUBE_OFFSET + COLOR_CUBE_TOTAL_COLORS; // 16 + 216 = 232
const GRAYSCALE_LEVELS: u8 = 24; // Indices 232-255

/// Converts an input `Color` to its approximate `Color::Rgb` representation if it's indexed.
/// If the input is already `Color::Rgb` or `Color::Named`, it's converted appropriately or returned.
/// `Color::Default` is returned as `Color::Rgb(0,0,0)` (black) with a warning, as this function
/// is for resolving to a concrete RGB.
///
/// # Arguments
/// * `color_input` - The `Color` enum variant to convert.
///
/// # Returns
/// A `Color::Rgb(r, g, b)` enum variant.
pub fn convert_to_rgb_color(color_input: Color) -> Color {
    match color_input {
        Color::Indexed(idx) => {
            if idx < ANSI_NAMED_COLOR_COUNT {
                // Standard and bright ANSI colors (0-15)
                return NamedColor::from_index(idx).to_rgb_color();
            }
            if idx < GRAYSCALE_OFFSET {
                // 6x6x6 Color Cube (indices 16-231)
                let cube_idx = idx - COLOR_CUBE_OFFSET;
                let r_comp = (cube_idx / (COLOR_CUBE_SIZE * COLOR_CUBE_SIZE)) % COLOR_CUBE_SIZE;
                let g_comp = (cube_idx / COLOR_CUBE_SIZE) % COLOR_CUBE_SIZE;
                let b_comp = cube_idx % COLOR_CUBE_SIZE;
                let r_val = if r_comp == 0 { 0 } else { r_comp * 40 + 55 };
                let g_val = if g_comp == 0 { 0 } else { g_comp * 40 + 55 };
                let b_val = if b_comp == 0 { 0 } else { b_comp * 40 + 55 };
                return Color::Rgb(r_val, g_val, b_val);
            }
            // Grayscale ramp (indices 232-255)
            let gray_idx = idx - GRAYSCALE_OFFSET;
            let level = gray_idx * 10 + 8;
            Color::Rgb(level, level, level)
        }
        Color::Named(named_color) => {
            // Convert named color to its RGB representation
            named_color.to_rgb_color()
        }
        Color::Rgb(r, g, b) => {
            // Already an RGB color, return as is
            Color::Rgb(r, g, b)
        }
        Color::Default => {
            warn!(
                "convert_to_rgb_color received Color::Default. This function expects concrete colors or indexed colors. Returning black."
            );
            Color::Rgb(0, 0, 0) // Fallback for Default
        }
    }
}
