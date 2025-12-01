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

    /// Convert a batch of pixels to a batch of u32 (packed representation).
    fn batch_to_u32<B: Backend>(batch: B::Batch<Self>) -> B::Batch<u32>
    where B::Batch<Self>: SimdBatch<Self>;

    /// Convert a batch of u32 (packed representation) to a batch of pixels.
    fn batch_from_u32<B: Backend>(batch: B::Batch<u32>) -> B::Batch<Self>
    where B::Batch<Self>: SimdBatch<Self>;

    /// Gather pixels from a slice using indices.
    fn batch_gather<B: Backend>(slice: &[Self], indices: B::Batch<u32>) -> B::Batch<Self>
    where
        B::Batch<Self>: SimdBatch<Self>,
        B::Batch<u32>: BatchArithmetic<u32>;

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
    fn batch_to_u32<B: Backend>(batch: B::Batch<Self>) -> B::Batch<u32> {
        B::upcast_u8_to_u32(batch)
    }

    #[inline(always)]
    fn batch_from_u32<B: Backend>(batch: B::Batch<u32>) -> B::Batch<Self> {
        B::downcast_u32_to_u8(batch)
    }

    #[inline(always)]
    fn batch_gather<B: Backend>(slice: &[Self], indices: B::Batch<u32>) -> B::Batch<Self>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let gathered = <B::Batch<u32> as BatchArithmetic<u32>>::gather_u8(slice, indices);
        B::downcast_u32_to_u8(gathered)
    }

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
    fn batch_to_u32<B: Backend>(batch: B::Batch<Self>) -> B::Batch<u32> {
        batch
    }

    #[inline(always)]
    fn batch_from_u32<B: Backend>(batch: B::Batch<u32>) -> B::Batch<Self> {
        batch
    }

    #[inline(always)]
    fn batch_gather<B: Backend>(slice: &[Self], indices: B::Batch<u32>) -> B::Batch<Self>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        <B::Batch<u32> as BatchArithmetic<u32>>::gather(slice, indices)
    }

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
