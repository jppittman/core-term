// pixelflow-render/src/color.rs
//! Unified color types for terminal rendering.
//!
//! This module provides:
//! - **Semantic colors**: `Color` enum for high-level specification
//! - **Pixel formats**: `Rgba8`, `Bgra8` for framebuffer storage
//!
//! For color manifolds, use `pixelflow_core::{Rgba, Red, Green, Blue, Alpha}`.

use bitflags::bitflags;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Re-export the Pixel trait from the local pixel module
pub use super::pixel::Pixel;

// =============================================================================
// Semantic Color Types (The "User Input" tier)
// =============================================================================

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
    pub fn from_index(idx: u8) -> Self {
        assert!(idx < 16, "Invalid NamedColor index: {}. Must be 0-15.", idx);
        unsafe { core::mem::transmute(idx) }
    }

    /// Returns the RGB representation of this named color.
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

/// Represents a semantic color value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Color {
    /// Default foreground or background color.
    #[default]
    Default,
    /// A standard named ANSI color (indices 0-15).
    Named(NamedColor),
    /// An indexed color from the 256-color palette (indices 0-255).
    Indexed(u8),
    /// An RGB true color.
    Rgb(u8, u8, u8),
}

impl Color {
    /// Convert to an Rgba8 pixel.
    #[inline]
    pub fn to_rgba8(self) -> Rgba8 {
        Rgba8(u32::from(self))
    }

    /// Convert to a Bgra8 pixel.
    #[inline]
    pub fn to_bgra8(self) -> Bgra8 {
        Bgra8::from(self.to_rgba8())
    }
}

// Constants for 256-color palette conversion
const ANSI_NAMED_COLOR_COUNT: u8 = 16;
const COLOR_CUBE_OFFSET: u8 = 16;
const COLOR_CUBE_SIZE: u8 = 6;
const COLOR_CUBE_TOTAL_COLORS: u8 = COLOR_CUBE_SIZE * COLOR_CUBE_SIZE * COLOR_CUBE_SIZE;
const GRAYSCALE_OFFSET: u8 = COLOR_CUBE_OFFSET + COLOR_CUBE_TOTAL_COLORS;

impl From<Color> for u32 {
    /// Convert a Color to a u32 pixel value (RGBA format: 0xAABBGGRR).
    fn from(color: Color) -> u32 {
        let (r, g, b) = match color {
            Color::Default => (0, 0, 0),
            Color::Named(named) => named.to_rgb(),
            Color::Indexed(idx) => {
                if idx < ANSI_NAMED_COLOR_COUNT {
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
        u32::from_le_bytes([r, g, b, 255])
    }
}

// =============================================================================
// Text Attributes
// =============================================================================

bitflags! {
    /// Text attribute flags (bold, underline, etc.).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct AttrFlags: u16 {
        const BOLD          = 1 << 0;
        const FAINT         = 1 << 1;
        const ITALIC        = 1 << 2;
        const UNDERLINE     = 1 << 3;
        const BLINK         = 1 << 4;
        const REVERSE       = 1 << 5;
        const HIDDEN        = 1 << 6;
        const STRIKETHROUGH = 1 << 7;
    }
}

// =============================================================================
// Pixel Format Types (Storage types)
// =============================================================================

/// Rgba8 pixel: bytes are [R, G, B, A] in memory order.
/// As a u32 on little-endian: 0xAABBGGRR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Rgba8(pub u32);

/// Bgra8 pixel: bytes are [B, G, R, A] in memory order.
/// As a u32 on little-endian: 0xAARRGGBB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Bgra8(pub u32);

impl Rgba8 {
    /// Creates a new RGBA pixel from component values.
    #[inline]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([r, g, b, a]))
    }

    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

impl Bgra8 {
    /// Creates a new BGRA pixel from component values.
    #[inline]
    pub fn new(b: u8, g: u8, r: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([b, g, r, a]))
    }

    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

// Swizzle: swap bytes 0 and 2 (R and B)
#[inline]
fn swizzle_rb(v: u32) -> u32 {
    (v & 0xFF00FF00) | ((v >> 16) & 0x000000FF) | ((v & 0x000000FF) << 16)
}

impl From<Bgra8> for Rgba8 {
    #[inline]
    fn from(bgra: Bgra8) -> Rgba8 {
        Rgba8(swizzle_rb(bgra.0))
    }
}

impl From<Rgba8> for Bgra8 {
    #[inline]
    fn from(rgba: Rgba8) -> Bgra8 {
        Bgra8(swizzle_rb(rgba.0))
    }
}

// =============================================================================
// Pixel Trait Implementations
// =============================================================================

impl Pixel for Rgba8 {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }
    #[inline]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a = (a * 255.0).clamp(0.0, 255.0) as u8;
        Self::new(r, g, b, a)
    }
}

impl Pixel for Bgra8 {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }
    #[inline]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a = (a * 255.0).clamp(0.0, 255.0) as u8;
        Self::new(b, g, r, a)
    }
}

// =============================================================================
// Platform-specific type aliases
// =============================================================================

/// Pixel format for X11 (XImage with ZPixmap on little-endian).
pub type X11Pixel = Bgra8;

/// Pixel format for Cocoa (CGImage with kCGImageAlphaPremultipliedLast).
pub type CocoaPixel = Rgba8;

/// Pixel format for Web (ImageData).
pub type WebPixel = Rgba8;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba8_components() {
        let c = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_bgra8_components() {
        let c = Bgra8::new(0x33, 0x22, 0x11, 0xFF);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_rgba8_to_bgra8() {
        let rgba = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        let bgra = Bgra8::from(rgba);
        assert_eq!(bgra.r(), 0x11);
        assert_eq!(bgra.g(), 0x22);
        assert_eq!(bgra.b(), 0x33);
        assert_eq!(bgra.a(), 0xFF);
    }
}
