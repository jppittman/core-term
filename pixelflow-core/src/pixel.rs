// pixelflow-core/src/pixel.rs
//! Pixel format trait for generic color operations.

use crate::backend::{Backend, BatchArithmetic, SimdBatch};
use crate::pipe::Surface;
use core::fmt::Debug;

/// Trait for pixel types that can be used in surfaces and frames.
pub trait Pixel: Copy + Default + Debug + 'static + Send + Sync {
    /// Create from raw u32 value.
    fn from_u32(v: u32) -> Self;

    /// Convert to raw u32 value.
    fn to_u32(self) -> u32;

    /// Extract red channel from a batch of 4 pixels.
    fn batch_red<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>;

    /// Extract green channel from a batch of 4 pixels.
    fn batch_green<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>;

    /// Extract blue channel from a batch of 4 pixels.
    fn batch_blue<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>;

    /// Extract alpha channel from a batch of 4 pixels.
    fn batch_alpha<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>;

    /// Reconstruct a batch of pixels from individual channel batches.
    fn batch_from_channels<B: Backend>(
        r: B::Batch<u32>,
        g: B::Batch<u32>,
        b: B::Batch<u32>,
        a: B::Batch<u32>,
    ) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>;
}

// Implement Surface for any Pixel type (Constant Surface)
impl<P: Pixel> Surface<P> for P {
    #[inline(always)]
    fn eval<B: Backend>(&self, _x: B::Batch<u32>, _y: B::Batch<u32>) -> B::Batch<P>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        SimdBatch::splat(*self)
    }
}

impl Pixel for u8 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self { v as u8 }
    #[inline(always)]
    fn to_u32(self) -> u32 { self as u32 }

    #[inline(always)]
    fn batch_red<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { batch }
    #[inline(always)]
    fn batch_green<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { batch }
    #[inline(always)]
    fn batch_blue<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { batch }
    #[inline(always)]
    fn batch_alpha<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { batch }

    #[inline(always)]
    fn batch_from_channels<B: Backend>(r: B::Batch<u32>, _: B::Batch<u32>, _: B::Batch<u32>, _: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { r }
}

impl Pixel for u32 {
    #[inline(always)]
    fn from_u32(v: u32) -> Self { v }
    #[inline(always)]
    fn to_u32(self) -> u32 { self }

    #[inline(always)]
    fn batch_red<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { batch & SimdBatch::splat(0xFF) }
    #[inline(always)]
    fn batch_green<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { (batch >> 8) & SimdBatch::splat(0xFF) }
    #[inline(always)]
    fn batch_blue<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { (batch >> 16) & SimdBatch::splat(0xFF) }
    #[inline(always)]
    fn batch_alpha<B: Backend>(batch: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> { (batch >> 24) & SimdBatch::splat(0xFF) }

    #[inline(always)]
    fn batch_from_channels<B: Backend>(r: B::Batch<u32>, g: B::Batch<u32>, b: B::Batch<u32>, a: B::Batch<u32>) -> B::Batch<u32> where B::Batch<u32>: BatchArithmetic<u32> {
        r | (g << 8) | (b << 16) | (a << 24)
    }
}
