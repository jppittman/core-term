// pixelflow-render/src/color.rs
//! Unified color types for terminal rendering.
//!
//! This module provides:
//! - **Semantic colors**: `Color`, `NamedColor` for terminal color specification
//! - **Text attributes**: `AttrFlags` for bold, italic, underline, etc.
//! - **Pixel formats**: `Rgba`, `Bgra` for framebuffer pixel representation
//!
//! # Design
//!
//! Colors flow through the system as follows:
//! 1. Terminal escape codes specify semantic `Color` values (Named, Indexed, RGB)
//! 2. At render time, `Color` is resolved to a concrete pixel value
//! 3. Pixel formats (`Rgba`/`Bgra`) handle platform-specific byte ordering
//!
//! # Platform Format Requirements
//!
//! Different display drivers expect different pixel formats:
//!
//! | Platform | Expected Format | Type Alias |
//! |----------|-----------------|------------|
//! | X11      | BGRA            | [`X11Pixel`] |
//! | Cocoa    | RGBA            | [`CocoaPixel`] |
//! | Web      | RGBA            | [`WebPixel`] |
//!
//! The pixel format is monomorphized at compile time - no runtime conversion needed.

use bitflags::bitflags;
use pixelflow_core::backend::{BatchArithmetic, SimdBatch};
use pixelflow_core::batch::Batch;
use pixelflow_core::traits::Manifold;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Re-export the Pixel trait from the local pixel module
pub use super::pixel::Pixel;

// =============================================================================
// Semantic Color Types
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
    ///
    /// # Panics
    /// Panics if `idx` >= 16.
    pub fn from_index(idx: u8) -> Self {
        assert!(idx < 16, "Invalid NamedColor index: {}. Must be 0-15.", idx);
        // SAFETY: The check above ensures idx is within the valid range
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
    /// Convert to an Rgba pixel.
    ///
    /// Uses the `From<Color> for u32` implementation which produces RGBA format.
    #[inline]
    pub fn to_rgba(self) -> Rgba {
        Rgba(u32::from(self))
    }

    /// Convert to a Bgra pixel.
    ///
    /// Converts via Rgba, then swizzles R and B channels.
    #[inline]
    pub fn to_bgra(self) -> Bgra {
        Bgra::from(self.to_rgba())
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

// =============================================================================
// Pixel Format Types
// =============================================================================

/// RGBA pixel: bytes are [R, G, B, A] in memory order.
/// As a u32 on little-endian: 0xAABBGGRR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Rgba(pub u32);

/// BGRA pixel: bytes are [B, G, R, A] in memory order.
/// As a u32 on little-endian: 0xAARRGGBB
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Bgra(pub u32);

impl Rgba {
    /// Creates a new RGBA pixel from component values.
    #[inline]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([r, g, b, a]))
    }

    /// Returns the red component.
    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    /// Returns the green component.
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    /// Returns the blue component.
    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    /// Returns the alpha component.
    #[inline]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

impl Bgra {
    /// Creates a new BGRA pixel from component values.
    #[inline]
    pub fn new(b: u8, g: u8, r: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([b, g, r, a]))
    }

    /// Returns the blue component.
    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    /// Returns the green component.
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    /// Returns the red component.
    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    /// Returns the alpha component.
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

impl From<Bgra> for Rgba {
    #[inline]
    fn from(bgra: Bgra) -> Rgba {
        Rgba(swizzle_rb(bgra.0))
    }
}

impl From<Rgba> for Bgra {
    #[inline]
    fn from(rgba: Rgba) -> Bgra {
        Bgra(swizzle_rb(rgba.0))
    }
}

// =============================================================================
// Pixel Trait Implementations
// =============================================================================

impl Pixel for Rgba {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }

    #[inline(always)]
    fn batch_to_u32(batch: Batch<Self>) -> Batch<u32> {
        // SAFETY: Rgba is repr(transparent) over u32
        unsafe { core::mem::transmute_copy(&batch) }
    }

    #[inline(always)]
    fn batch_from_u32(batch: Batch<u32>) -> Batch<Self> {
        // SAFETY: Rgba is repr(transparent) over u32
        unsafe { core::mem::transmute_copy(&batch) }
    }

    #[inline(always)]
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self> {
        // Gather u32 values, then transmute
        let u32_slice: &[u32] = unsafe { core::mem::transmute(slice) };
        let gathered = BatchArithmetic::gather(u32_slice, indices);
        unsafe { core::mem::transmute_copy(&gathered) }
    }

    #[inline(always)]
    fn batch_red(batch: Batch<u32>) -> Batch<u32> {
        batch & Batch::<u32>::splat(0xFF)
    }

    #[inline(always)]
    fn batch_green(batch: Batch<u32>) -> Batch<u32> {
        (batch >> 8) & Batch::<u32>::splat(0xFF)
    }

    #[inline(always)]
    fn batch_blue(batch: Batch<u32>) -> Batch<u32> {
        (batch >> 16) & Batch::<u32>::splat(0xFF)
    }

    #[inline(always)]
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32> {
        batch >> 24
    }

    #[inline(always)]
    fn batch_from_channels(
        r: Batch<u32>,
        g: Batch<u32>,
        b: Batch<u32>,
        a: Batch<u32>,
    ) -> Batch<u32> {
        r | (g << 8) | (b << 16) | (a << 24)
    }

    #[inline(always)]
    fn batch_store(batch: Batch<Self>, slice: &mut [Self]) {
        // SAFETY: Rgba is repr(transparent) over u32, same size and alignment
        let u32_slice: &mut [u32] = unsafe { core::mem::transmute(slice) };
        let u32_batch: Batch<u32> = unsafe { core::mem::transmute_copy(&batch) };
        SimdBatch::store(&u32_batch, u32_slice);
    }
}

impl<C> Manifold<Rgba, C> for Rgba
where
    C: Copy + core::fmt::Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, _: Batch<C>, _: Batch<C>, _: Batch<C>, _: Batch<C>) -> Batch<Rgba> {
        Batch::<Rgba>::splat(*self)
    }
}

impl Pixel for Bgra {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }

    #[inline(always)]
    fn batch_to_u32(batch: Batch<Self>) -> Batch<u32> {
        // SAFETY: Bgra is repr(transparent) over u32
        unsafe { core::mem::transmute_copy(&batch) }
    }

    #[inline(always)]
    fn batch_from_u32(batch: Batch<u32>) -> Batch<Self> {
        // SAFETY: Bgra is repr(transparent) over u32
        unsafe { core::mem::transmute_copy(&batch) }
    }

    #[inline(always)]
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self> {
        // Gather u32 values, then transmute
        let u32_slice: &[u32] = unsafe { core::mem::transmute(slice) };
        let gathered = BatchArithmetic::gather(u32_slice, indices);
        unsafe { core::mem::transmute_copy(&gathered) }
    }

    #[inline(always)]
    fn batch_red(batch: Batch<u32>) -> Batch<u32> {
        (batch >> 16) & Batch::<u32>::splat(0xFF)
    }

    #[inline(always)]
    fn batch_green(batch: Batch<u32>) -> Batch<u32> {
        (batch >> 8) & Batch::<u32>::splat(0xFF)
    }

    #[inline(always)]
    fn batch_blue(batch: Batch<u32>) -> Batch<u32> {
        batch & Batch::<u32>::splat(0xFF)
    }

    #[inline(always)]
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32> {
        batch >> 24
    }

    #[inline(always)]
    fn batch_from_channels(
        r: Batch<u32>,
        g: Batch<u32>,
        b: Batch<u32>,
        a: Batch<u32>,
    ) -> Batch<u32> {
        b | (g << 8) | (r << 16) | (a << 24)
    }

    #[inline(always)]
    fn batch_store(batch: Batch<Self>, slice: &mut [Self]) {
        // SAFETY: Bgra is repr(transparent) over u32, same size and alignment
        let u32_slice: &mut [u32] = unsafe { core::mem::transmute(slice) };
        let u32_batch: Batch<u32> = unsafe { core::mem::transmute_copy(&batch) };
        SimdBatch::store(&u32_batch, u32_slice);
    }
}

impl<C> Manifold<Bgra, C> for Bgra
where
    C: Copy + core::fmt::Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, _: Batch<C>, _: Batch<C>, _: Batch<C>, _: Batch<C>) -> Batch<Bgra> {
        Batch::<Bgra>::splat(*self)
    }
}

// Note: Surface<Rgba> for Rgba and Surface<Bgra> for Bgra are provided by
// the blanket impl `Surface<P> for P where P: Pixel` in pixelflow-core/src/pixel.rs

// =============================================================================
// Platform-specific type aliases
// =============================================================================

/// Pixel format for X11 (XImage with ZPixmap on little-endian).
pub type X11Pixel = Bgra;

/// Pixel format for Cocoa (CGImage with kCGImageAlphaPremultipliedLast).
pub type CocoaPixel = Rgba;

/// Pixel format for Web (ImageData).
pub type WebPixel = Rgba;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_components() {
        let c = Rgba::new(0x11, 0x22, 0x33, 0xFF);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_bgra_components() {
        let c = Bgra::new(0x33, 0x22, 0x11, 0xFF);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_rgba_to_bgra() {
        let rgba = Rgba::new(0x11, 0x22, 0x33, 0xFF);
        let bgra = Bgra::from(rgba);
        assert_eq!(bgra.r(), 0x11);
        assert_eq!(bgra.g(), 0x22);
        assert_eq!(bgra.b(), 0x33);
        assert_eq!(bgra.a(), 0xFF);
    }

    #[test]
    fn test_bgra_to_rgba() {
        let bgra = Bgra::new(0x33, 0x22, 0x11, 0xFF);
        let rgba = Rgba::from(bgra);
        assert_eq!(rgba.r(), 0x11);
        assert_eq!(rgba.g(), 0x22);
        assert_eq!(rgba.b(), 0x33);
        assert_eq!(rgba.a(), 0xFF);
    }

    #[test]
    fn test_roundtrip() {
        let original = Rgba::new(0xAA, 0xBB, 0xCC, 0xDD);
        let converted = Rgba::from(Bgra::from(original));
        assert_eq!(original, converted);
    }

    // Note: Batch tests removed - they need to be rewritten for new Backend-generic API

    #[test]
    fn test_color_to_rgba() {
        let color = Color::Rgb(0xAA, 0xBB, 0xCC);
        let rgba = color.to_rgba();
        assert_eq!(rgba.r(), 0xAA);
        assert_eq!(rgba.g(), 0xBB);
        assert_eq!(rgba.b(), 0xCC);
        assert_eq!(rgba.a(), 0xFF);
    }

    #[test]
    fn test_color_to_bgra() {
        let color = Color::Rgb(0xAA, 0xBB, 0xCC);
        let bgra = color.to_bgra();
        // BGRA memory order: [B, G, R, A]
        assert_eq!(bgra.r(), 0xAA);
        assert_eq!(bgra.g(), 0xBB);
        assert_eq!(bgra.b(), 0xCC);
        assert_eq!(bgra.a(), 0xFF);
    }

    #[test]
    fn test_named_color_to_rgba() {
        let color = Color::Named(NamedColor::Red);
        let rgba = color.to_rgba();
        // NamedColor::Red = (205, 0, 0)
        assert_eq!(rgba.r(), 205);
        assert_eq!(rgba.g(), 0);
        assert_eq!(rgba.b(), 0);
    }
}
