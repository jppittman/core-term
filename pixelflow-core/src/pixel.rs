// pixelflow-core/src/pixel.rs
//! Pixel format trait for generic color operations.
//!
//! The `Pixel` trait provides SIMD-friendly batch operations for channel access,
//! enabling zero-cost format abstraction in the render pipeline.

use crate::Batch;

/// Trait for pixel types that can be used in surfaces and frames.
///
/// Provides both scalar and SIMD batch operations for channel access.
/// All batch methods should be `#[inline(always)]` in implementations
/// for zero-cost abstraction.
///
/// # Channel Semantics
/// The batch methods extract/reconstruct logical R, G, B, A channels
/// regardless of the physical byte order. This allows format-agnostic
/// color math in combinators like `Over`.
pub trait Pixel: Copy + Default + 'static + Send + Sync {
    /// Create from raw u32 value.
    fn from_u32(v: u32) -> Self;

    /// Convert to raw u32 value.
    fn to_u32(self) -> u32;

    /// Extract red channel from a batch of 4 pixels.
    /// Returns values in range 0-255.
    fn batch_red(batch: Batch<u32>) -> Batch<u32>;

    /// Extract green channel from a batch of 4 pixels.
    /// Returns values in range 0-255.
    fn batch_green(batch: Batch<u32>) -> Batch<u32>;

    /// Extract blue channel from a batch of 4 pixels.
    /// Returns values in range 0-255.
    fn batch_blue(batch: Batch<u32>) -> Batch<u32>;

    /// Extract alpha channel from a batch of 4 pixels.
    /// Returns values in range 0-255.
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32>;

    /// Reconstruct a batch of pixels from individual channel batches.
    /// Each channel batch should contain values in the range 0-255.
    fn batch_from_channels(
        r: Batch<u32>,
        g: Batch<u32>,
        b: Batch<u32>,
        a: Batch<u32>,
    ) -> Batch<u32>;
}

// Note: Surface<P> for concrete pixel types (Rgba, Bgra) is implemented
// in pixelflow-render/src/color.rs to avoid conflicting with the closure
// blanket impl in pipe.rs.
