// pixelflow-render/src/color.rs
//! Unified color types for terminal rendering.
//!
//! This module provides:
//! - **Semantic colors**: `Color` enum for high-level specification
//! - **Pixel formats**: `Rgba8`, `Bgra8` for framebuffer storage
//! - **Color manifolds**: `ColorManifold`, `Lift`, `ColorMap` for functional color composition
//!
//! For color manifolds, use `pixelflow_core::{Rgba, Red, Green, Blue, Alpha}`.

use bitflags::bitflags;
use pixelflow_core::{Discrete, Field, Manifold};

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

// Make NamedColor a manifold - an infinite field of that ANSI color
impl pixelflow_core::Manifold for NamedColor {
    type Output = pixelflow_core::Discrete;

    #[inline(always)]
    fn eval_raw(
        &self,
        _x: pixelflow_core::Field,
        _y: pixelflow_core::Field,
        _z: pixelflow_core::Field,
        _w: pixelflow_core::Field,
    ) -> pixelflow_core::Discrete {
        let (r, g, b) = self.to_rgb();
        pixelflow_core::pack_rgba(
            pixelflow_core::Field::from(r as f32 / 255.0),
            pixelflow_core::Field::from(g as f32 / 255.0),
            pixelflow_core::Field::from(b as f32 / 255.0),
            pixelflow_core::Field::from(1.0),
        )
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

    /// Convert to normalized f32 RGBA components.
    #[inline]
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

// Make Color a manifold - an infinite field of that color
impl pixelflow_core::Manifold for Color {
    type Output = pixelflow_core::Discrete;

    #[inline(always)]
    fn eval_raw(
        &self,
        _x: pixelflow_core::Field,
        _y: pixelflow_core::Field,
        _z: pixelflow_core::Field,
        _w: pixelflow_core::Field,
    ) -> pixelflow_core::Discrete {
        let (r, g, b, a) = self.to_f32_rgba();
        pixelflow_core::pack_rgba(
            pixelflow_core::Field::from(r),
            pixelflow_core::Field::from(g),
            pixelflow_core::Field::from(b),
            pixelflow_core::Field::from(a),
        )
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
// Color Manifolds
// =============================================================================

/// The RGBA color cube as a manifold.
///
/// `ColorCube` is the terminal object for color: it interprets its input
/// coordinates as RGBA channels and packs them to `Discrete`.
///
/// - X = Red   (0.0 to 1.0)
/// - Y = Green (0.0 to 1.0)
/// - Z = Blue  (0.0 to 1.0)
/// - W = Alpha (0.0 to 1.0)
///
/// # Philosophy
///
/// Colors ARE coordinates. Use `At` (the universal contramap) to navigate
/// the color cube:
///
/// ```ignore
/// use pixelflow_core::{At, X, Y};
/// use pixelflow_graphics::ColorCube;
///
/// // Solid red
/// let red = At { inner: ColorCube, x: 1.0, y: 0.0, z: 0.0, w: 1.0 };
///
/// // Gradient: red varies with screen X
/// let gradient = At { inner: ColorCube, x: X / 255.0, y: 0.5, z: 0.5, w: 1.0 };
///
/// // Grayscale: same value for R, G, B
/// let gray = At { inner: ColorCube, x: v, y: v, z: v, w: 1.0 };
///
/// // Blend two colors: coordinate arithmetic before At
/// let blended = At {
///     inner: ColorCube,
///     x: t * r1 + (1.0 - t) * r2,
///     y: t * g1 + (1.0 - t) * g2,
///     z: t * b1 + (1.0 - t) * b2,
///     w: t * a1 + (1.0 - t) * a2,
/// };
/// ```
///
/// # ColorCube vs ColorManifold
///
/// - Use `ColorCube` with `At` when channel values are scalar expressions
/// - Use `ColorManifold` when channels come from separate manifold trees
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorCube;

impl Manifold for ColorCube {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, r: Field, g: Field, b: Field, a: Field) -> Discrete {
        pixelflow_core::pack_rgba(r, g, b, a)
    }
}

/// Grayscale: lifts a scalar to R=G=B, A=1.
///
/// Convenience for the common pattern:
/// ```ignore
/// At { inner: ColorCube, x: v, y: v, z: v, w: 1.0 }
/// ```
///
/// # Example
/// ```ignore
/// use pixelflow_graphics::Grayscale;
/// use pixelflow_core::X;
///
/// let gradient = Grayscale(X / 256.0);  // Black to white
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Grayscale<M>(pub M);

impl<M: Manifold<Output = Field>> Manifold for Grayscale<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let v = self.0.eval_raw(x, y, z, w);
        pixelflow_core::pack_rgba(v, v, v, Field::from(1.0))
    }
}

/// Composes 4 Field manifolds into a single RGBA output.
///
/// Unlike `ColorCube` which interprets input coordinates as colors,
/// `ColorManifold` evaluates separate manifolds for each channel
/// and packs the results.
///
/// # Use Cases
///
/// - Terminal grids with Select trees per channel
/// - Any case where R, G, B, A come from separate computation trees
///
/// # Example
/// ```ignore
/// use pixelflow_graphics::ColorManifold;
/// use pixelflow_core::X;
///
/// // Red channel varies with X, others constant
/// let grad = ColorManifold::new(
///     X / 255.0,    // R: gradient
///     0.5f32,       // G: constant
///     0.5f32,       // B: constant
///     1.0f32,       // A: opaque
/// );
/// ```
#[derive(Clone, Debug)]
pub struct ColorManifold<R, G, B, A> {
    r: R,
    g: G,
    b: B,
    a: A,
}

impl<R, G, B, A> ColorManifold<R, G, B, A> {
    /// Create a new color manifold from four channel manifolds.
    pub fn new(r: R, g: G, b: B, a: A) -> Self {
        Self { r, g, b, a }
    }
}

impl<R, G, B, A> Manifold for ColorManifold<R, G, B, A>
where
    R: Manifold<Output = Field>,
    G: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
    A: Manifold<Output = Field>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let r = self.r.eval_raw(x, y, z, w);
        let g = self.g.eval_raw(x, y, z, w);
        let b = self.b.eval_raw(x, y, z, w);
        let a = self.a.eval_raw(x, y, z, w);
        pixelflow_core::pack_rgba(r, g, b, a)
    }
}

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
