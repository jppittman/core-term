// pixelflow-render/src/color.rs
//! Unified color types for terminal rendering.
//!
//! This module provides:
//! - **Semantic colors**: `Color` enum for high-level specification
//! - **Pixel formats**: `Rgba8`, `Bgra8` for framebuffer storage
//!
//! A colour *output* is not here: it is four channel kernels packed by
//! [`Pixel::packed_shifts`] inside the compiled scene
//! (`render::scene::compile_packed_for`). This module is the semantic input
//! side and the framebuffer storage side, and nothing between them.

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
    #[must_use]
    pub fn from_index(idx: u8) -> Self {
        assert!(idx < 16, "Invalid NamedColor index: {}. Must be 0-15.", idx);
        unsafe { core::mem::transmute(idx) }
    }

    /// Returns the RGB representation of this named color.
    #[must_use]
    pub fn to_rgb(self) -> (u8, u8, u8) {
        ANSI_COLORS_RGB[self as usize]
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
    #[must_use]
    pub fn to_rgba8(self) -> Rgba8 {
        Rgba8(u32::from(self))
    }

    /// Convert to a Bgra8 pixel.
    #[inline]
    #[must_use]
    pub fn to_bgra8(self) -> Bgra8 {
        Bgra8::from(self.to_rgba8())
    }

    /// Convert to normalized f32 RGBA components.
    #[inline]
    #[must_use]
    pub fn to_f32_rgba(self) -> (f32, f32, f32, f32) {
        let rgba = self.to_rgba8();
        (
            rgba.r() as f32 / 255.0,
            rgba.g() as f32 / 255.0,
            rgba.b() as f32 / 255.0,
            rgba.a() as f32 / 255.0,
        )
    }
}

// Constants for 256-color palette conversion
const ANSI_NAMED_COLOR_COUNT: u8 = 16;
const COLOR_CUBE_OFFSET: u8 = 16;
const COLOR_CUBE_SIZE: u8 = 6;
const COLOR_CUBE_TOTAL_COLORS: u8 = COLOR_CUBE_SIZE * COLOR_CUBE_SIZE * COLOR_CUBE_SIZE;
const GRAYSCALE_OFFSET: u8 = COLOR_CUBE_OFFSET + COLOR_CUBE_TOTAL_COLORS;

const CUBE_SCALE_FACTOR: u8 = 40;
const CUBE_BASE_OFFSET: u8 = 55;
const GRAYSCALE_STEP: u8 = 10;
const GRAYSCALE_BASE: u8 = 8;

const ANSI_COLORS_RGB: [(u8, u8, u8); 16] = [
    (0, 0, 0),       // Black
    (205, 0, 0),     // Red
    (0, 205, 0),     // Green
    (205, 205, 0),   // Yellow
    (0, 0, 238),     // Blue
    (205, 0, 205),   // Magenta
    (0, 205, 205),   // Cyan
    (229, 229, 229), // White
    (127, 127, 127), // BrightBlack
    (255, 0, 0),     // BrightRed
    (0, 255, 0),     // BrightGreen
    (255, 255, 0),   // BrightYellow
    (92, 92, 255),   // BrightBlue
    (255, 0, 255),   // BrightMagenta
    (0, 255, 255),   // BrightCyan
    (255, 255, 255), // BrightWhite
];

/// Precomputed 256-color palette lookup table.
/// Stores packed RGBA (0xAABBGGRR) values for O(1) conversion.
static PALETTE: [u32; 256] = generate_palette();

/// Generates the 256-color palette at compile/link time.
const fn generate_palette() -> [u32; 256] {
    let mut palette = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let idx = i as u8;
        let (r, g, b) = if idx < ANSI_NAMED_COLOR_COUNT {
            // Named colors
            ANSI_COLORS_RGB[idx as usize]
        } else if idx < GRAYSCALE_OFFSET {
            // 6x6x6 Color Cube (indices 16-231)
            let cube_idx = idx - COLOR_CUBE_OFFSET;
            let r_comp = (cube_idx / (COLOR_CUBE_SIZE * COLOR_CUBE_SIZE)) % COLOR_CUBE_SIZE;
            let g_comp = (cube_idx / COLOR_CUBE_SIZE) % COLOR_CUBE_SIZE;
            let b_comp = cube_idx % COLOR_CUBE_SIZE;
            let r_val = if r_comp == 0 {
                0
            } else {
                r_comp * CUBE_SCALE_FACTOR + CUBE_BASE_OFFSET
            };
            let g_val = if g_comp == 0 {
                0
            } else {
                g_comp * CUBE_SCALE_FACTOR + CUBE_BASE_OFFSET
            };
            let b_val = if b_comp == 0 {
                0
            } else {
                b_comp * CUBE_SCALE_FACTOR + CUBE_BASE_OFFSET
            };
            (r_val, g_val, b_val)
        } else {
            // Grayscale ramp (indices 232-255)
            let gray_idx = idx - GRAYSCALE_OFFSET;
            let level = gray_idx * GRAYSCALE_STEP + GRAYSCALE_BASE;
            (level, level, level)
        };

        // Pack into u32 (RGBA little-endian: 0xAABBGGRR)
        // u32::from_le_bytes is const-stable since 1.32
        palette[i] = u32::from_le_bytes([r, g, b, 255]);
        i += 1;
    }
    palette
}

impl From<Color> for u32 {
    /// Convert a Color to a u32 pixel value (RGBA format: 0xAABBGGRR).
    #[inline] // Hot path!
    fn from(color: Color) -> u32 {
        match color {
            Color::Default => u32::from_le_bytes([0, 0, 0, 255]),

            // Optimized lookup for Named and Indexed
            Color::Named(named) => PALETTE[named as usize],
            Color::Indexed(idx) => PALETTE[idx as usize],

            // Fallback for TrueColor
            Color::Rgb(r, g, b) => u32::from_le_bytes([r, g, b, 255]),
        }
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
    #[must_use]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([r, g, b, a]))
    }

    #[inline]
    #[must_use]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    #[must_use]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    #[must_use]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    #[must_use]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

impl Bgra8 {
    /// Creates a new BGRA pixel from component values.
    #[inline]
    #[must_use]
    pub fn new(b: u8, g: u8, r: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([b, g, r, a]))
    }

    #[inline]
    #[must_use]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    #[must_use]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    #[must_use]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    #[must_use]
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
    /// `Rgba8::new` stores `[r, g, b, a]`, so r is byte 0.
    fn packed_shifts() -> Option<[u32; 4]> {
        Some([0, 8, 16, 24])
    }

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
    /// `Bgra8::new` stores `[b, g, r, a]`, so r is byte 2.
    fn packed_shifts() -> Option<[u32; 4]> {
        Some([16, 8, 0, 24])
    }

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

/// The platform's framebuffer pixel format, and with it the byte order every
/// packed kernel packs for ([`Pixel::packed_shifts`]).
#[cfg(target_os = "macos")]
pub type PlatformPixel = Rgba8;

/// See the macOS variant.
#[cfg(target_os = "linux")]
pub type PlatformPixel = Bgra8;

/// See the macOS variant.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub type PlatformPixel = Rgba8;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {

    /// [`Pixel::packed_shifts`] and the scalar [`Pixel::from_rgba`] are the
    /// same byte-order fact stated twice; this pins them to each other so the
    /// JIT pack (which consumes the shifts) can never drift from the scalar
    /// pack.
    #[test]
    fn packed_shifts_agree_with_pixel_from_rgba() {
        fn pack(shifts: [u32; 4], r: f32, g: f32, b: f32, a: f32) -> u32 {
            let q = |x: f32| (x * 255.0).clamp(0.0, 255.0) as u32;
            (q(r) << shifts[0]) | (q(g) << shifts[1]) | (q(b) << shifts[2]) | (q(a) << shifts[3])
        }
        let samples = [
            (0.0, 0.25, 0.5, 1.0),
            (1.0, 0.0, 0.999, 0.004),
            (0.7, 0.7, 0.7, 0.0),
        ];
        for (r, g, b, a) in samples {
            assert_eq!(
                Rgba8::from_rgba(r, g, b, a).0,
                pack(
                    Rgba8::packed_shifts().expect("Rgba8 is packed RGBA"),
                    r,
                    g,
                    b,
                    a
                ),
                "Rgba8 disagrees with its own packed_shifts"
            );
            assert_eq!(
                Bgra8::from_rgba(r, g, b, a).0,
                pack(
                    Bgra8::packed_shifts().expect("Bgra8 is packed RGBA"),
                    r,
                    g,
                    b,
                    a
                ),
                "Bgra8 disagrees with its own packed_shifts"
            );
            assert_eq!(
                <u32 as Pixel>::from_rgba(r, g, b, a),
                pack(
                    <u32 as Pixel>::packed_shifts().expect("u32 is packed RGBA"),
                    r,
                    g,
                    b,
                    a
                ),
                "u32 disagrees with its own packed_shifts"
            );
        }
    }
    use super::*;

    #[test]
    fn rgba8_components() {
        let c = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn bgra8_components() {
        let c = Bgra8::new(0x33, 0x22, 0x11, 0xFF);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn rgba8_to_bgra8() {
        let rgba = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        let bgra = Bgra8::from(rgba);
        assert_eq!(bgra.r(), 0x11);
        assert_eq!(bgra.g(), 0x22);
        assert_eq!(bgra.b(), 0x33);
        assert_eq!(bgra.a(), 0xFF);
    }

    /// A named colour's `[0, 1]` channels are what a scene's four constant
    /// channel kernels are built from, so this is the number the pack sees.
    /// That those channels reach the frame as bytes is
    /// `tests/rendering_contract.rs`.
    #[test]
    fn named_color_channels_are_its_ansi_rgb() {
        let (r, g, b, a) = Color::Named(NamedColor::Red).to_f32_rgba();
        // ANSI red is (205, 0, 0), opaque.
        assert!((r - 205.0 / 255.0).abs() < 1e-6);
        assert_eq!((g, b, a), (0.0, 0.0, 1.0));
    }

    #[test]
    fn true_color_channels_are_its_bytes() {
        let (r, g, b, a) = Color::Rgb(10, 20, 30).to_f32_rgba();
        assert!((r - 10.0 / 255.0).abs() < 1e-6);
        assert!((g - 20.0 / 255.0).abs() < 1e-6);
        assert!((b - 30.0 / 255.0).abs() < 1e-6);
        assert_eq!(a, 1.0);
    }
}
