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
}
