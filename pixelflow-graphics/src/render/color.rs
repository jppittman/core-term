//! Color type definitions and operations.
//!
//! Provides the `Color` enum for defining colors in the scene, including
//! standard RGB, named colors (ANSI/X11), and grayscale.
//!
//! `Color` implements `Manifold<Output = Discrete>`, so it can be used directly
//! in scene composition.

use pixelflow_core::{Discrete, Field, Manifold, RgbaComponents};

/// A color value in the scene.
///
/// Colors are manifolds that evaluate to `Discrete` (packed RGBA u32 pixels).
/// They are constant manifolds (independent of coordinates) unless composed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Color {
    /// 24-bit RGB color (0-255). Alpha is 255.
    Rgb(u8, u8, u8),
    /// 32-bit RGBA color (0-255).
    Rgba(u8, u8, u8, u8),
    /// A named color from the standard palette.
    Named(NamedColor),
    /// Grayscale value (0-255).
    Gray(u8),
}

/// Standard named colors (ANSI 16 + extended).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NamedColor {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    // Bright variants
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

impl NamedColor {
    /// Get the RGB components of the named color.
    pub const fn to_rgb(self) -> (u8, u8, u8) {
        match self {
            NamedColor::Black => (0, 0, 0),
            NamedColor::Red => (205, 49, 49),
            NamedColor::Green => (13, 188, 121),
            NamedColor::Yellow => (229, 229, 16),
            NamedColor::Blue => (36, 114, 200),
            NamedColor::Magenta => (188, 63, 188),
            NamedColor::Cyan => (17, 168, 205),
            NamedColor::White => (229, 229, 229),
            NamedColor::BrightBlack => (102, 102, 102),
            NamedColor::BrightRed => (241, 76, 76),
            NamedColor::BrightGreen => (35, 209, 139),
            NamedColor::BrightYellow => (245, 245, 67),
            NamedColor::BrightBlue => (59, 142, 234),
            NamedColor::BrightMagenta => (214, 112, 214),
            NamedColor::BrightCyan => (41, 184, 219),
            NamedColor::BrightWhite => (255, 255, 255),
        }
    }
}

impl Color {
    /// Convert color to packed u32 RGBA.
    pub fn to_u32(&self) -> u32 {
        let (r, g, b, a) = match self {
            Color::Rgb(r, g, b) => (*r, *g, *b, 255),
            Color::Rgba(r, g, b, a) => (*r, *g, *b, *a),
            Color::Named(n) => {
                let (r, g, b) = n.to_rgb();
                (r, g, b, 255)
            }
            Color::Gray(v) => (*v, *v, *v, 255),
        };
        u32::from_be_bytes([a, b, g, r]) // Little endian: R G B A in memory -> A B G R in u32
    }

    /// Evaluate the color as a Field (0.0 - 1.0) for each channel.
    ///
    /// This is a helper for the Manifold implementation.
    #[inline(always)]
    fn as_fields(&self) -> (Field, Field, Field, Field) {
        let (r, g, b, a) = match self {
            Color::Rgb(r, g, b) => (*r, *g, *b, 255),
            Color::Rgba(r, g, b, a) => (*r, *g, *b, *a),
            Color::Named(n) => {
                let (r, g, b) = n.to_rgb();
                (r, g, b, 255)
            }
            Color::Gray(v) => (*v, *v, *v, 255),
        };

        // Convert u8 [0, 255] to f32 [0.0, 1.0]
        // Pre-multiply by 1/255
        const INV_255: f32 = 1.0 / 255.0;
        (
            Field::from(r as f32 * INV_255),
            Field::from(g as f32 * INV_255),
            Field::from(b as f32 * INV_255),
            Field::from(a as f32 * INV_255),
        )
    }
}

// Implement Manifold for Color.
// This allows Colors to be leaves in the scene graph.
impl<I: pixelflow_core::Computational> Manifold<I> for Color {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> Discrete {
        let (r, g, b, a) = self.as_fields();
        Discrete::pack(RgbaComponents {
            r,
            g,
            b,
            a,
        })
    }
}

// Re-export common pixel types
pub use pixelflow_core::Discrete as PackedPixel;
// Re-export Pixel trait so it's available as render::color::Pixel
pub use crate::render::pixel::Pixel;

/// Standard RGBA8888 pixel (used by macOS/Web).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rgba8(pub u32);

impl Rgba8 {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([r, g, b, a]))
    }
    pub fn r(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }
    pub fn g(&self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }
    pub fn b(&self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }
    pub fn a(&self) -> u8 {
        ((self.0 >> 24) & 0xFF) as u8
    }
}

impl Pixel for Rgba8 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self.0
    }
    #[inline(always)]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r_u8 = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g_u8 = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b_u8 = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a_u8 = (a * 255.0).clamp(0.0, 255.0) as u8;
        Self(u32::from_le_bytes([r_u8, g_u8, b_u8, a_u8]))
    }
}

/// Standard BGRA8888 pixel (used by Linux/X11).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Bgra8(pub u32);

impl Bgra8 {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        // B G R A
        Self(u32::from_le_bytes([b, g, r, a]))
    }
    pub fn r(&self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }
    pub fn g(&self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }
    pub fn b(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }
    pub fn a(&self) -> u8 {
        ((self.0 >> 24) & 0xFF) as u8
    }
}

impl Pixel for Bgra8 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        // v comes from Discrete which is RGBA (little endian: R G B A)
        // We want BGRA (little endian: B G R A)
        // Swap R and B
        let r = (v) & 0xFF;
        let g = (v >> 8) & 0xFF;
        let b = (v >> 16) & 0xFF;
        let a = (v >> 24) & 0xFF;
        Self(b | (g << 8) | (r << 16) | (a << 24))
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        // Convert back to RGBA for internal use if needed
        let b = (self.0) & 0xFF;
        let g = (self.0 >> 8) & 0xFF;
        let r = (self.0 >> 16) & 0xFF;
        let a = (self.0 >> 24) & 0xFF;
        r | (g << 8) | (b << 16) | (a << 24)
    }
    #[inline(always)]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r_u8 = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g_u8 = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b_u8 = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a_u8 = (a * 255.0).clamp(0.0, 255.0) as u8;
        // B G R A order
        Self(u32::from_le_bytes([b_u8, g_u8, r_u8, a_u8]))
    }
}

impl From<Rgba8> for Bgra8 {
    fn from(rgba: Rgba8) -> Self {
        let r = (rgba.0) & 0xFF;
        let g = (rgba.0 >> 8) & 0xFF;
        let b = (rgba.0 >> 16) & 0xFF;
        let a = (rgba.0 >> 24) & 0xFF;
        Self(b | (g << 8) | (r << 16) | (a << 24))
    }
}

// Grayscale manifold wrapper.
#[derive(Clone, Copy, Debug)]
pub struct Grayscale<M>(pub M);

impl<M> Manifold for Grayscale<M>
where
    M: Manifold<Output = Field>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let val = self.0.eval_raw(x, y, z, w);
        Discrete::pack(RgbaComponents {
            r: val,
            g: val,
            b: val,
            a: Field::from(1.0),
        })
    }
}

// Stubs for other types mentioned in errors/exports to fix build
bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct AttrFlags: u8 {
        const BOLD = 0x01;
        const DIM = 0x02;
        const ITALIC = 0x04;
        const UNDERLINE = 0x08;
        const BLINK = 0x10;
        const INVERSE = 0x20;
        const HIDDEN = 0x40;
        const STRIKETHROUGH = 0x80;
    }
}

// Stubs/Aliases
pub type CocoaPixel = Rgba8;
pub type WebPixel = Rgba8;
pub type X11Pixel = Bgra8;

// Implementations for ColorCubes
#[derive(Copy, Clone, Debug, Default)]
pub struct ColorCube;

impl Manifold<Field> for ColorCube {
    type Output = Discrete;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        Discrete::pack(RgbaComponents { r: x, g: y, b: z, a: w })
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct RgbaColorCube;

impl Manifold<Field> for RgbaColorCube {
    type Output = Discrete;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        Discrete::pack(RgbaComponents { r: x, g: y, b: z, a: w })
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BgraColorCube;

impl Manifold<Field> for BgraColorCube {
    type Output = Discrete;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        Discrete::pack(RgbaComponents { r: z, g: y, b: x, a: w })
    }
}

// PlatformColorCube alias or struct
#[derive(Copy, Clone, Debug, Default)]
pub struct PlatformColorCube;

impl Manifold<Field> for PlatformColorCube {
    type Output = Discrete;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        RgbaColorCube.eval_raw(x, y, z, w)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ColorManifold; // Still just a placeholder if needed
