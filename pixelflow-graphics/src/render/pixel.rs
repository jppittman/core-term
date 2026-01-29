// pixelflow-graphics/src/render/pixel.rs
//! Pixel format trait for generic color operations.

use core::fmt::Debug;

/// Trait for pixel types that can be used in surfaces and frames.
///
/// Pixels must be POD-like (Copy + 'static) and convertible to/from u32.
/// This trait assumes the underlying storage is effectively an array of u32s.
pub trait Pixel: Copy + Default + Debug + PartialEq + 'static + Send + Sync {
    /// Create from raw u32 value.
    fn from_u32(v: u32) -> Self;

    /// Convert to raw u32 value.
    fn to_u32(self) -> u32;

    /// Create from normalized RGBA components [0, 1].
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self;

    // Note: No batch methods here. The Engine knows how to load
    // pixel types into SIMD lanes (assuming valid layout).
}

impl Pixel for u8 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        v as u8
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn from_rgba(r: f32, _g: f32, _b: f32, _a: f32) -> Self {
        (r * 255.0).clamp(0.0, 255.0) as u8
    }
}

impl Pixel for u32 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        v
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self
    }
    #[inline(always)]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r_u8 = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g_u8 = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b_u8 = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a_u8 = (a * 255.0).clamp(0.0, 255.0) as u8;
        u32::from_le_bytes([r_u8, g_u8, b_u8, a_u8])
    }
}
