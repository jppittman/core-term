use crate::backend::{Backend, BatchArithmetic, SimdBatch};
use crate::pipe::Surface;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

/// Pixel format descriptor.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// R, G, B, A order (0xAABBGGRR in little endian).
    Rgba,
    /// B, G, R, A order (0xAARRGGBB in little endian).
    Bgra,
}

impl PixelFormat {
    /// Swizzles a generic RGBA batch to this format.
    ///
    /// Assumes input `batch` is in RGBA format (0xAABBGGRR).
    #[inline(always)]
    pub fn swizzle<B: Backend>(self, batch: B::Batch<u32>) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        match self {
            PixelFormat::Rgba => batch,
            PixelFormat::Bgra => {
                // Swap R and B
                // R is at 0, B is at 16
                let mask_ga = SimdBatch::splat(0xFF00FF00);
                let mask_r = SimdBatch::splat(0x000000FF);
                let mask_b = SimdBatch::splat(0x00FF0000);

                let ga = batch & mask_ga;
                let r = batch & mask_r;
                let b = batch & mask_b;

                let new_b = r << 16;
                let new_r = b >> 16;

                ga | new_b | new_r
            }
        }
    }
}

/// A container for a rendered buffer.
pub struct Buffer<T> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
}

/// The Platform combinator that binds a backend to a surface.
pub struct Platform<B: Backend> {
    pub scale: u32,
    pub format: PixelFormat,
    pub _backend: PhantomData<B>,
}

impl<B: Backend> Platform<B> {
    /// Creates a new Platform configuration.
    pub fn new() -> Self {
        Self {
            scale: 1,
            format: PixelFormat::Rgba,
            _backend: PhantomData,
        }
    }

    /// Sets the scale factor.
    pub fn with_scale(mut self, scale: u32) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the output pixel format.
    pub fn with_format(mut self, format: PixelFormat) -> Self {
        self.format = format;
        self
    }

    /// Materializes a surface into a buffer.
    ///
    /// This runs the render loop using the bound Backend `B`.
    pub fn materialize<S: Surface<u32>>(
        &self,
        surface: &S,
        width: usize,
        height: usize,
    ) -> Buffer<u32>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let mut data = vec![0u32; width * height];
        let lanes = B::LANES;

        for y in 0..height {
            let row_start = y * width;
            let row_end = row_start + width;
            let row_slice = &mut data[row_start..row_end];

            let mut x = 0;
            // Hot path: SIMD loop
            while x + lanes <= width {
                let bx = B::Batch::<u32>::sequential_from(x as u32);
                let by = B::Batch::<u32>::splat(y as u32);

                // Apply scale (logical = physical / scale)
                // Use SimdBatch::splat via full path if inference ambiguous, but SimdBatch is imported.
                let scale_factor = B::Batch::<u32>::splat(self.scale);
                // Div requires BatchArithmetic.
                let sx = bx / scale_factor;
                let sy = by / scale_factor;

                // Evaluate surface
                let rgba = surface.eval::<B>(sx, sy);

                // Swizzle
                let out = self.format.swizzle::<B>(rgba);

                // Store
                // Store trait method takes &self.
                SimdBatch::store(&out, &mut row_slice[x..x + lanes]);

                x += lanes;
            }

            // Cold path: Scalar fallback
            if x < width {
                while x < width {
                   use crate::backends::scalar::{Scalar, ScalarBatch};

                   let bx = ScalarBatch(x as u32);
                   let by = ScalarBatch(y as u32);

                   let scale = ScalarBatch(self.scale);
                   let sx = bx / scale;
                   let sy = by / scale;

                   let rgba = surface.eval::<Scalar>(sx, sy);

                   // Scalar backend doesn't need swizzle call via platform format if we call Scalar directly?
                   // format.swizzle::<Scalar> works.
                   let out = self.format.swizzle::<Scalar>(rgba);

                   // Store ScalarBatch
                   SimdBatch::store(&out, &mut row_slice[x..x+1]);
                   x += 1;
                }
            }
        }

        Buffer { data, width, height }
    }
}
