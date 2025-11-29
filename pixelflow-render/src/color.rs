// pixelflow-render/src/color.rs
//! Color format types for framebuffer pixels.
//!
//! Uses newtypes for type-safe color format handling with zero runtime overhead.
//! The From trait provides conversion between formats.
//!
//! The `Pixel` trait provides SIMD-friendly batch operations for channel access,
//! enabling zero-cost format abstraction in the render pipeline.
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
//! When building render pipelines, use the appropriate pixel format type
//! with combinators like [`Over`](pixelflow_core::ops::Over):
//!
//! ```ignore
//! // For X11:
//! let blend = mask.over::<Bgra, _, _>(fg, bg);
//!
//! // For Cocoa:
//! let blend = mask.over::<Rgba, _, _>(fg, bg);
//! ```
//!
//! The pixel format is monomorphized at compile time - no runtime conversion needed.

use pixelflow_core::Batch;
// Re-export the Pixel trait from pixelflow-core
pub use pixelflow_core::Pixel;

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
    pub fn r(self) -> u8 { self.0.to_le_bytes()[0] }
    /// Returns the green component.
    #[inline]
    pub fn g(self) -> u8 { self.0.to_le_bytes()[1] }
    /// Returns the blue component.
    #[inline]
    pub fn b(self) -> u8 { self.0.to_le_bytes()[2] }
    /// Returns the alpha component.
    #[inline]
    pub fn a(self) -> u8 { self.0.to_le_bytes()[3] }
}

impl Bgra {
    /// Creates a new BGRA pixel from component values.
    #[inline]
    pub fn new(b: u8, g: u8, r: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([b, g, r, a]))
    }

    /// Returns the blue component.
    #[inline]
    pub fn b(self) -> u8 { self.0.to_le_bytes()[0] }
    /// Returns the green component.
    #[inline]
    pub fn g(self) -> u8 { self.0.to_le_bytes()[1] }
    /// Returns the red component.
    #[inline]
    pub fn r(self) -> u8 { self.0.to_le_bytes()[2] }
    /// Returns the alpha component.
    #[inline]
    pub fn a(self) -> u8 { self.0.to_le_bytes()[3] }
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

impl Pixel for Rgba {
    #[inline]
    fn from_u32(v: u32) -> Self { Self(v) }
    #[inline]
    fn to_u32(self) -> u32 { self.0 }

    #[inline(always)]
    fn batch_red(batch: Batch<u32>) -> Batch<u32> {
        // RGBA: R is byte 0 (bits 0-7)
        batch & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_green(batch: Batch<u32>) -> Batch<u32> {
        // RGBA: G is byte 1 (bits 8-15)
        (batch >> 8) & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_blue(batch: Batch<u32>) -> Batch<u32> {
        // RGBA: B is byte 2 (bits 16-23)
        (batch >> 16) & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32> {
        // RGBA: A is byte 3 (bits 24-31)
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
}

impl Pixel for Bgra {
    #[inline]
    fn from_u32(v: u32) -> Self { Self(v) }
    #[inline]
    fn to_u32(self) -> u32 { self.0 }

    #[inline(always)]
    fn batch_red(batch: Batch<u32>) -> Batch<u32> {
        // BGRA: R is byte 2 (bits 16-23)
        (batch >> 16) & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_green(batch: Batch<u32>) -> Batch<u32> {
        // BGRA: G is byte 1 (bits 8-15)
        (batch >> 8) & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_blue(batch: Batch<u32>) -> Batch<u32> {
        // BGRA: B is byte 0 (bits 0-7)
        batch & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32> {
        // BGRA: A is byte 3 (bits 24-31)
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
}

// =============================================================================
// Platform-specific type aliases
// =============================================================================

/// Pixel format for X11 (XImage with ZPixmap on little-endian).
/// X11 expects BGRA byte order.
pub type X11Pixel = Bgra;

/// Pixel format for Cocoa (CGImage with kCGImageAlphaPremultipliedLast).
/// Cocoa expects RGBA byte order.
pub type CocoaPixel = Rgba;

/// Pixel format for Web (ImageData).
/// Web/Canvas expects RGBA byte order.
pub type WebPixel = Rgba;

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

    #[test]
    fn test_rgba_batch_channels() {
        // Create 4 RGBA pixels with different values
        let p0 = Rgba::new(0x10, 0x20, 0x30, 0x40);
        let p1 = Rgba::new(0x11, 0x21, 0x31, 0x41);
        let p2 = Rgba::new(0x12, 0x22, 0x32, 0x42);
        let p3 = Rgba::new(0x13, 0x23, 0x33, 0x43);

        let batch = Batch::new(p0.0, p1.0, p2.0, p3.0);

        let r = Rgba::batch_red(batch);
        let g = Rgba::batch_green(batch);
        let b = Rgba::batch_blue(batch);
        let a = Rgba::batch_alpha(batch);

        assert_eq!(r.to_array_usize(), [0x10, 0x11, 0x12, 0x13]);
        assert_eq!(g.to_array_usize(), [0x20, 0x21, 0x22, 0x23]);
        assert_eq!(b.to_array_usize(), [0x30, 0x31, 0x32, 0x33]);
        assert_eq!(a.to_array_usize(), [0x40, 0x41, 0x42, 0x43]);
    }

    #[test]
    fn test_bgra_batch_channels() {
        // Create 4 BGRA pixels with different values
        // Bgra::new takes (b, g, r, a)
        let p0 = Bgra::new(0x30, 0x20, 0x10, 0x40);
        let p1 = Bgra::new(0x31, 0x21, 0x11, 0x41);
        let p2 = Bgra::new(0x32, 0x22, 0x12, 0x42);
        let p3 = Bgra::new(0x33, 0x23, 0x13, 0x43);

        let batch = Batch::new(p0.0, p1.0, p2.0, p3.0);

        let r = Bgra::batch_red(batch);
        let g = Bgra::batch_green(batch);
        let b = Bgra::batch_blue(batch);
        let a = Bgra::batch_alpha(batch);

        // Even though stored as BGRA, batch_red should return R values
        assert_eq!(r.to_array_usize(), [0x10, 0x11, 0x12, 0x13]);
        assert_eq!(g.to_array_usize(), [0x20, 0x21, 0x22, 0x23]);
        assert_eq!(b.to_array_usize(), [0x30, 0x31, 0x32, 0x33]);
        assert_eq!(a.to_array_usize(), [0x40, 0x41, 0x42, 0x43]);
    }

    #[test]
    fn test_rgba_batch_roundtrip() {
        let p0 = Rgba::new(0xAA, 0xBB, 0xCC, 0xDD);
        let p1 = Rgba::new(0x11, 0x22, 0x33, 0x44);
        let p2 = Rgba::new(0x55, 0x66, 0x77, 0x88);
        let p3 = Rgba::new(0x99, 0x00, 0xFF, 0xEE);

        let batch = Batch::new(p0.0, p1.0, p2.0, p3.0);

        let r = Rgba::batch_red(batch);
        let g = Rgba::batch_green(batch);
        let b = Rgba::batch_blue(batch);
        let a = Rgba::batch_alpha(batch);

        let reconstructed = Rgba::batch_from_channels(r, g, b, a);
        assert_eq!(reconstructed.to_array_usize(), batch.to_array_usize());
    }

    #[test]
    fn test_bgra_batch_roundtrip() {
        let p0 = Bgra::new(0xCC, 0xBB, 0xAA, 0xDD);
        let p1 = Bgra::new(0x33, 0x22, 0x11, 0x44);
        let p2 = Bgra::new(0x77, 0x66, 0x55, 0x88);
        let p3 = Bgra::new(0xFF, 0x00, 0x99, 0xEE);

        let batch = Batch::new(p0.0, p1.0, p2.0, p3.0);

        let r = Bgra::batch_red(batch);
        let g = Bgra::batch_green(batch);
        let b = Bgra::batch_blue(batch);
        let a = Bgra::batch_alpha(batch);

        let reconstructed = Bgra::batch_from_channels(r, g, b, a);
        assert_eq!(reconstructed.to_array_usize(), batch.to_array_usize());
    }
}
