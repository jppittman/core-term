// pixelflow-core/src/pixel.rs
//! Pixel format trait for generic color operations.

use crate::batch::{Batch, NativeBackend};
use crate::backend::{Backend, BatchArithmetic, SimdBatch};
use crate::pipe::Surface;
use core::fmt::Debug;

/// Trait for pixel types that can be used in surfaces and frames.
pub trait Pixel: Copy + Default + Debug + 'static + Send + Sync {
    /// Create from raw u32 value.
    fn from_u32(v: u32) -> Self;

    /// Convert to raw u32 value.
    fn to_u32(self) -> u32;

    /// Convert a batch of pixels to a batch of u32 (packed representation).
    fn batch_to_u32(batch: Batch<Self>) -> Batch<u32>;

    /// Convert a batch of u32 (packed representation) to a batch of pixels.
    fn batch_from_u32(batch: Batch<u32>) -> Batch<Self>;

    /// Gather pixels from a slice using indices.
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self>;

    /// Extract red channel from a batch of pixels.
    fn batch_red(batch: Batch<u32>) -> Batch<u32>;

    /// Extract green channel from a batch of pixels.
    fn batch_green(batch: Batch<u32>) -> Batch<u32>;

    /// Extract blue channel from a batch of pixels.
    fn batch_blue(batch: Batch<u32>) -> Batch<u32>;

    /// Extract alpha channel from a batch of pixels.
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32>;

    /// Reconstruct a batch of pixels from individual channel batches.
    fn batch_from_channels(r: Batch<u32>, g: Batch<u32>, b: Batch<u32>, a: Batch<u32>) -> Batch<u32>;
}

// Implement Surface for any Pixel type (Constant Surface)
impl<P: Pixel> Surface<P> for P {
    #[inline(always)]
    fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<P> {
        Batch::splat(*self)
    }
}

impl Pixel for u8 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self { v as u8 }
    #[inline(always)]
    fn to_u32(self) -> u32 { self as u32 }

    #[inline(always)]
    fn batch_to_u32(batch: Batch<Self>) -> Batch<u32> {
        NativeBackend::upcast_u8_to_u32(batch)
    }

    #[inline(always)]
    fn batch_from_u32(batch: Batch<u32>) -> Batch<Self> {
        NativeBackend::downcast_u32_to_u8(batch)
    }

    #[inline(always)]
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self> {
        let gathered = BatchArithmetic::gather_u8(slice, indices);
        NativeBackend::downcast_u32_to_u8(gathered)
    }

    #[inline(always)]
    fn batch_red(batch: Batch<u32>) -> Batch<u32> { batch }
    #[inline(always)]
    fn batch_green(batch: Batch<u32>) -> Batch<u32> { batch }
    #[inline(always)]
    fn batch_blue(batch: Batch<u32>) -> Batch<u32> { batch }
    #[inline(always)]
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32> { batch }

    #[inline(always)]
    fn batch_from_channels(r: Batch<u32>, _: Batch<u32>, _: Batch<u32>, _: Batch<u32>) -> Batch<u32> { r }
}

impl Pixel for u32 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self { v }
    #[inline(always)]
    fn to_u32(self) -> u32 { self }

    #[inline(always)]
    fn batch_to_u32(batch: Batch<Self>) -> Batch<u32> {
        batch
    }

    #[inline(always)]
    fn batch_from_u32(batch: Batch<u32>) -> Batch<Self> {
        batch
    }

    #[inline(always)]
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self> {
        BatchArithmetic::gather(slice, indices)
    }

    #[inline(always)]
    fn batch_red(batch: Batch<u32>) -> Batch<u32> { batch & Batch::splat(0xFF) }
    #[inline(always)]
    fn batch_green(batch: Batch<u32>) -> Batch<u32> { (batch >> 8) & Batch::splat(0xFF) }
    #[inline(always)]
    fn batch_blue(batch: Batch<u32>) -> Batch<u32> { (batch >> 16) & Batch::splat(0xFF) }
    #[inline(always)]
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32> { (batch >> 24) & Batch::splat(0xFF) }

    #[inline(always)]
    fn batch_from_channels(r: Batch<u32>, g: Batch<u32>, b: Batch<u32>, a: Batch<u32>) -> Batch<u32> {
        r | (g << 8) | (b << 16) | (a << 24)
    }
}
