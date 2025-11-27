//! Common types used throughout pixelflow-render.

use bitflags::bitflags;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Standard ANSI named colors (indices 0-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum NamedColor {
    /// ANSI Black.
    Black = 0,
    /// ANSI Red.
    Red = 1,
    /// ANSI Green.
    Green = 2,
    /// ANSI Yellow.
    Yellow = 3,
    /// ANSI Blue.
    Blue = 4,
    /// ANSI Magenta.
    Magenta = 5,
    /// ANSI Cyan.
    Cyan = 6,
    /// ANSI White.
    White = 7,
    /// ANSI Bright Black.
    BrightBlack = 8,
    /// ANSI Bright Red.
    BrightRed = 9,
    /// ANSI Bright Green.
    BrightGreen = 10,
    /// ANSI Bright Yellow.
    BrightYellow = 11,
    /// ANSI Bright Blue.
    BrightBlue = 12,
    /// ANSI Bright Magenta.
    BrightMagenta = 13,
    /// ANSI Bright Cyan.
    BrightCyan = 14,
    /// ANSI Bright White.
    BrightWhite = 15,
}

impl NamedColor {
    /// Convert a u8 index (0-15) to a NamedColor.
    ///
    /// # Parameters
    /// * `idx` - The color index (must be < 16).
    ///
    /// # Panics
    /// Panics if `idx` >= 16.
    pub fn from_index(idx: u8) -> Self {
        assert!(idx < 16, "Invalid NamedColor index: {}. Must be 0-15.", idx);
        // SAFETY: The check above ensures idx is within the valid range
        unsafe { core::mem::transmute(idx) }
    }

    /// Returns the RGB representation of this named color.
    ///
    /// These are common sRGB values used by many terminals.
    ///
    /// # Returns
    /// * A tuple `(r, g, b)`.
    pub fn to_rgb(self) -> (u8, u8, u8) {
        match self {
            NamedColor::Black => (0, 0, 0),
            NamedColor::Red => (205, 0, 0),
            NamedColor::Green => (0, 205, 0),
            NamedColor::Yellow => (205, 205, 0),
            NamedColor::Blue => (0, 0, 238),
            NamedColor::Magenta => (205, 0, 205),
            NamedColor::Cyan => (0, 205, 205),
            NamedColor::White => (229, 229, 229),
            NamedColor::BrightBlack => (127, 127, 127),
            NamedColor::BrightRed => (255, 0, 0),
            NamedColor::BrightGreen => (0, 255, 0),
            NamedColor::BrightYellow => (255, 255, 0),
            NamedColor::BrightBlue => (92, 92, 255),
            NamedColor::BrightMagenta => (255, 0, 255),
            NamedColor::BrightCyan => (0, 255, 255),
            NamedColor::BrightWhite => (255, 255, 255),
        }
    }
}

/// Represents a color value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Color {
    /// Default foreground or background color.
    Default,
    /// A standard named ANSI color (indices 0-15).
    Named(NamedColor),
    /// An indexed color from the 256-color palette (indices 0-255).
    Indexed(u8),
    /// An RGB true color.
    Rgb(u8, u8, u8),
}

impl Default for Color {
    fn default() -> Self {
        Color::Default
    }
}

bitflags! {
    /// Text attribute flags (bold, underline, etc.).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct AttrFlags: u16 {
        /// Bold text.
        const BOLD          = 1 << 0;
        /// Faint (dim) text.
        const FAINT         = 1 << 1;
        /// Italic text.
        const ITALIC        = 1 << 2;
        /// Underlined text.
        const UNDERLINE     = 1 << 3;
        /// Blinking text.
        const BLINK         = 1 << 4;
        /// Reverse video (foreground and background swapped).
        const REVERSE       = 1 << 5;
        /// Hidden text (not visible).
        const HIDDEN        = 1 << 6;
        /// Strikethrough text.
        const STRIKETHROUGH = 1 << 7;
    }
}

// Constants for 256-color palette
const ANSI_NAMED_COLOR_COUNT: u8 = 16;
const COLOR_CUBE_OFFSET: u8 = 16;
const COLOR_CUBE_SIZE: u8 = 6; // 6x6x6 cube
const COLOR_CUBE_TOTAL_COLORS: u8 = COLOR_CUBE_SIZE * COLOR_CUBE_SIZE * COLOR_CUBE_SIZE; // 216
const GRAYSCALE_OFFSET: u8 = COLOR_CUBE_OFFSET + COLOR_CUBE_TOTAL_COLORS; // 16 + 216 = 232

impl From<Color> for u32 {
    /// Convert a Color enum to a u32 pixel value (RGBA format: 0xAABBGGRR).
    ///
    /// # Returns
    /// * A `u32` value representing the color in RGBA layout (little-endian).
    fn from(color: Color) -> u32 {
        let (r, g, b) = match color {
            Color::Default => (0, 0, 0), // Default to black
            Color::Named(named) => named.to_rgb(),
            Color::Indexed(idx) => {
                if idx < ANSI_NAMED_COLOR_COUNT {
                    // Standard and bright ANSI colors (0-15)
                    NamedColor::from_index(idx).to_rgb()
                } else if idx < GRAYSCALE_OFFSET {
                    // 6x6x6 Color Cube (indices 16-231)
                    let cube_idx = idx - COLOR_CUBE_OFFSET;
                    let r_comp = (cube_idx / (COLOR_CUBE_SIZE * COLOR_CUBE_SIZE)) % COLOR_CUBE_SIZE;
                    let g_comp = (cube_idx / COLOR_CUBE_SIZE) % COLOR_CUBE_SIZE;
                    let b_comp = cube_idx % COLOR_CUBE_SIZE;
                    let r_val = if r_comp == 0 { 0 } else { r_comp * 40 + 55 };
                    let g_val = if g_comp == 0 { 0 } else { g_comp * 40 + 55 };
                    let b_val = if b_comp == 0 { 0 } else { b_comp * 40 + 55 };
                    (r_val, g_val, b_val)
                } else {
                    // Grayscale ramp (indices 232-255)
                    let gray_idx = idx - GRAYSCALE_OFFSET;
                    let level = gray_idx * 10 + 8;
                    (level, level, level)
                }
            }
            Color::Rgb(r, g, b) => (r, g, b),
        };

        // Convert to u32 in RGBA format (0xAABBGGRR with full alpha)
        u32::from_le_bytes([r, g, b, 255])
    }
}
