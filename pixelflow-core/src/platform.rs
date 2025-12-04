use crate::backend::SimdBatch;
use crate::batch::{Batch, LANES};
use crate::traits::Surface;
use alloc::vec;
use alloc::vec::Vec;

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
    #[inline(always)]
    pub fn swizzle(self, batch: Batch<u32>) -> Batch<u32> {
        match self {
            PixelFormat::Rgba => batch,
            PixelFormat::Bgra => {
                let mask_ga = Batch::<u32>::splat(0xFF00FF00u32);
                let mask_r = Batch::<u32>::splat(0x000000FFu32);
                let mask_b = Batch::<u32>::splat(0x00FF0000u32);

                let ga = batch & mask_ga;
                let r = batch & mask_r;
                let b = batch & mask_b;

                ga | (r << 16) | (b >> 16)
            }
        }
    }
}

/// A container for a rendered buffer.
pub struct Buffer<T> {
    /// The linear buffer data.
    pub data: Vec<T>,
    /// The width of the buffer in pixels.
    pub width: usize,
    /// The height of the buffer in pixels.
    pub height: usize,
}

/// The Platform configuration for rendering.
pub struct Platform {
    /// The scale factor (integer scaling).
    pub scale: u32,
    /// The output pixel format.
    pub format: PixelFormat,
}

impl Platform {
    /// Creates a new Platform configuration.
    pub fn new() -> Self {
        Self {
            scale: 1,
            format: PixelFormat::Rgba,
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
    pub fn materialize<S: Surface<u32>>(
        &self,
        surface: &S,
        width: usize,
        height: usize,
    ) -> Buffer<u32> {
        let mut data = vec![0u32; width * height];

        for y in 0..height {
            let row_start = y * width;
            let row_slice = &mut data[row_start..row_start + width];

            let mut x = 0;
            // Hot path: SIMD loop
            while x + LANES <= width {
                let bx = Batch::<u32>::sequential_from(x as u32);
                let by = Batch::<u32>::splat(y as u32);

                let scale_factor = Batch::<u32>::splat(self.scale);
                let sx = bx / scale_factor;
                let sy = by / scale_factor;

                let rgba = surface.eval(sx, sy);
                let out = self.format.swizzle(rgba);

                SimdBatch::store(&out, &mut row_slice[x..x + LANES]);
                x += LANES;
            }

            // Cold path: remainder pixels one at a time
            while x < width {
                let sx = (x as u32) / self.scale;
                let sy = (y as u32) / self.scale;
                let rgba = surface.eval_one(sx, sy);
                // Need to swizzle single pixel - just use batch with first lane
                let out = self.format.swizzle(Batch::<u32>::splat(rgba));
                row_slice[x] = out.first();
                x += 1;
            }
        }

        Buffer {
            data,
            width,
            height,
        }
    }
}

impl Default for Platform {
    fn default() -> Self {
        Self::new()
    }
}
