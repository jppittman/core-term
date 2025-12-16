use crate::backend::{Backend, BatchArithmetic, SimdBatch};
use crate::batch::{Batch, LANES};
use crate::traits::Surface;
use core::fmt::Debug;
use core::ops::{BitAnd, BitOr, Not, Shl, Shr};

// ============================================================================
// Tensor Views
// ============================================================================

/// A read-only view into a 2D tensor (e.g. image buffer).
#[derive(Copy, Clone)]
pub struct TensorView<'a, T> {
    /// Slice of data.
    pub data: &'a [T],
    /// Width of the tensor.
    pub width: usize,
    /// Height of the tensor.
    pub height: usize,
    /// Stride (elements per row).
    pub stride: usize,
}

impl<'a, T> TensorView<'a, T> {
    /// Creates a new `TensorView`.
    #[inline(always)]
    pub const fn new(data: &'a [T], width: usize, height: usize, stride: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride,
        }
    }
}

impl<'a> TensorView<'a, u32> {
    /// Gathers 2D data from the tensor using generic backend indices.
    ///
    /// # Safety
    /// This function is unsafe because it performs gather operations which might
    /// exceed bounds if logic is incorrect, although it includes bounds checks.
    #[inline(always)]
    pub unsafe fn gather_2d<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<u32>
    where
        B::Batch<u32>: BatchArithmetic<u32> + BitAnd<Output = B::Batch<u32>>,
    {
        let w = <B::Batch<u32> as SimdBatch<u32>>::splat(self.width as u32);
        let h = <B::Batch<u32> as SimdBatch<u32>>::splat(self.height as u32);
        let zero = <B::Batch<u32> as SimdBatch<u32>>::splat(0);

        // Strict 2D bounds check: x < width && y < height
        // This handles large wrapped negative values because they appear as large u32 > width.
        let in_bounds = x.cmp_lt(w) & y.cmp_lt(h);

        // Use 0 for coordinate calculation if out of bounds to prevent overflow/wrapping issues
        // in the index calculation itself.
        let safe_x = in_bounds.select(x, zero);
        let safe_y = in_bounds.select(y, zero);

        let stride_vec = <B::Batch<u32> as SimdBatch<u32>>::splat(self.stride as u32);
        let idx_vec = (safe_y * stride_vec) + safe_x;

        // gather handles slice bounds safety (returns 0 if idx >= len),
        // but we also mask with in_bounds to ensure 2D geometric correctness.
        let val = <B::Batch<u32> as BatchArithmetic<u32>>::gather(self.data, idx_vec);

        in_bounds.select(val, zero)
    }
}

impl<'a> TensorView<'a, u8> {
    /// Gathers 2D data from a u8 tensor (e.g. alpha mask) using generic backend indices.
    ///
    /// # Safety
    /// Unsafe due to raw gather operations.
    #[inline(always)]
    pub unsafe fn gather_2d<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<u32>
    where
        B::Batch<u32>: BatchArithmetic<u32> + BitAnd<Output = B::Batch<u32>>,
    {
        let w = <B::Batch<u32> as SimdBatch<u32>>::splat(self.width as u32);
        let h = <B::Batch<u32> as SimdBatch<u32>>::splat(self.height as u32);
        let zero = <B::Batch<u32> as SimdBatch<u32>>::splat(0);

        let in_bounds = x.cmp_lt(w) & y.cmp_lt(h);

        let safe_x = in_bounds.select(x, zero);
        let safe_y = in_bounds.select(y, zero);

        let stride_vec = <B::Batch<u32> as SimdBatch<u32>>::splat(self.stride as u32);
        let idx_vec = (safe_y * stride_vec) + safe_x;

        let val = <B::Batch<u32> as BatchArithmetic<u32>>::gather_u8(self.data, idx_vec);

        in_bounds.select(val, zero)
    }

    /// Gathers 4-bit packed data (e.g. font atlas) as u32 values.
    ///
    /// # Safety
    /// Unsafe due to raw gather operations.
    #[inline(always)]
    pub unsafe fn gather_4bit<B: Backend>(
        &self,
        x: B::Batch<u32>,
        y: B::Batch<u32>,
    ) -> B::Batch<u32>
    where
        B::Batch<u32>: BatchArithmetic<u32>
            + BitAnd<Output = B::Batch<u32>>
            + BitOr<Output = B::Batch<u32>>
            + Not<Output = B::Batch<u32>>
            + Shr<i32, Output = B::Batch<u32>>
            + Shl<i32, Output = B::Batch<u32>>,
    {
        // Strict bounds check on PIXEL coordinates
        let w = <B::Batch<u32> as SimdBatch<u32>>::splat(self.width as u32);
        let h = <B::Batch<u32> as SimdBatch<u32>>::splat(self.height as u32);
        let in_bounds = x.cmp_lt(w) & y.cmp_lt(h);
        let zero = <B::Batch<u32> as SimdBatch<u32>>::splat(0);

        // Use safe coordinates for calculation
        let safe_x = in_bounds.select(x, zero);
        let safe_y = in_bounds.select(y, zero);

        let byte_x = safe_x >> 1;
        let is_odd = safe_x & <B::Batch<u32> as SimdBatch<u32>>::splat(1);

        // gather_2d will perform its own check, but safe_x/safe_y ensure we are within valid range
        let packed = unsafe { self.gather_2d::<B>(byte_x, safe_y) };

        let high_nibble = (packed >> 4) & <B::Batch<u32> as SimdBatch<u32>>::splat(0x0F);
        let low_nibble = packed & <B::Batch<u32> as SimdBatch<u32>>::splat(0x0F);

        let all_ones = <B::Batch<u32> as SimdBatch<u32>>::splat(0xFFFFFFFF);
        let mask = is_odd * all_ones;
        // Need explicit !mask because Not isn't always inferred on B::Batch<u32>
        let nibble = (high_nibble & !mask) | (low_nibble & mask);
        let val = (nibble << 4) | nibble;

        in_bounds.select(val, zero)
    }

    /// Samples the surface using bilinear interpolation on 4-bit packed data.
    ///
    /// # Safety
    /// Unsafe due to raw gather operations.
    #[inline(always)]
    pub unsafe fn sample_4bit_bilinear<B: Backend>(
        &self,
        u_fp: B::Batch<u32>,
        v_fp: B::Batch<u32>,
    ) -> B::Batch<u32>
    where
        B::Batch<u32>: BatchArithmetic<u32>
            + BitAnd<Output = B::Batch<u32>>
            + Shr<i32, Output = B::Batch<u32>>
            + BitOr<Output = B::Batch<u32>>
            + Not<Output = B::Batch<u32>>
            + Shl<i32, Output = B::Batch<u32>>,
    {
        let u0_raw = u_fp >> 16;
        let v0_raw = v_fp >> 16;
        let max_x = <B::Batch<u32> as SimdBatch<u32>>::splat((self.width - 1) as u32);
        let max_y = <B::Batch<u32> as SimdBatch<u32>>::splat((self.height - 1) as u32);

        let u0 = u0_raw.min(max_x);
        let v0 = v0_raw.min(max_y);
        let u1 = (u0 + <B::Batch<u32> as SimdBatch<u32>>::splat(1)).min(max_x);
        let v1 = (v0 + <B::Batch<u32> as SimdBatch<u32>>::splat(1)).min(max_y);

        let du = (u_fp >> 8) & <B::Batch<u32> as SimdBatch<u32>>::splat(0xFF);
        let dv = (v_fp >> 8) & <B::Batch<u32> as SimdBatch<u32>>::splat(0xFF);
        let inv_du = <B::Batch<u32> as SimdBatch<u32>>::splat(256) - du;
        let inv_dv = <B::Batch<u32> as SimdBatch<u32>>::splat(256) - dv;

        let p00 = unsafe { self.gather_4bit::<B>(u0, v0) };
        let p10 = unsafe { self.gather_4bit::<B>(u1, v0) };
        let p01 = unsafe { self.gather_4bit::<B>(u0, v1) };
        let p11 = unsafe { self.gather_4bit::<B>(u1, v1) };

        let top = (p00 * inv_du + p10 * du) >> 8;
        let bot = (p01 * inv_du + p11 * du) >> 8;

        (top * inv_dv + bot * dv) >> 8
    }
}

/// A mutable view into a 2D tensor (e.g. image buffer).
pub struct TensorViewMut<'a, T> {
    /// Mutable slice of data.
    pub data: &'a mut [T],
    /// Width of the tensor.
    pub width: usize,
    /// Height of the tensor.
    pub height: usize,
    /// Stride (elements per row).
    pub stride: usize,
}

/// Rasterizes a surface into a target buffer.
///
/// This function iterates over the target buffer, evaluating the surface
/// for each pixel using SIMD batches where possible.
pub fn execute<T, S>(surface: &S, target: &mut [T], width: usize, height: usize)
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T> + ?Sized,
{
    // Early return for zero-size targets to avoid UB with from_raw_parts_mut
    if width == 0 || height == 0 {
        return;
    }

    render_stripe(surface, target, width, 0, height);
}

/// Render a horizontal stripe of rows [start_y, end_y)
#[inline(always)]
fn render_stripe<T, S>(
    surface: &S,
    target: &mut [T],
    width: usize,
    start_y: usize,
    end_y: usize,
) where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T> + ?Sized,
{
    for (row_idx, y) in (start_y..end_y).enumerate() {
        let row_start = row_idx * width;
        let y_batch = Batch::<u32>::splat(y as u32);

        let mut x = 0;
        // Hot path: SIMD loop
        while x + LANES <= width {
            let x_batch = Batch::<u32>::sequential_from(x as u32);
            let result = surface.eval(x_batch, y_batch);

            // Store result. Note: generic Batch<T> doesn't expose inherent `store` cleanly
            // without SimdBatch trait bound on Batch<T>, but we know Batch<T> is SimdBatch<T>
            // due to backend definition.
            // Using SimdBatch::store trait method.
            SimdBatch::store(&result, &mut target[row_start + x..row_start + x + LANES]);

            x += LANES;
        }

        // Cold path: remainder pixels
        while x < width {
            target[row_start + x] = surface.eval_one(x as u32, y as u32);
            x += 1;
        }
    }
}

/// Render a specific row range [start_y, end_y) into the target buffer.
///
/// This is exposed for parallel rendering: external code can partition
/// the framebuffer into stripes and call this function from multiple threads,
/// as long as the stripes don't overlap.
pub fn execute_stripe<T, S>(
    surface: &S,
    target: &mut [T],
    width: usize,
    start_y: usize,
    end_y: usize,
) where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T> + ?Sized,
{
    render_stripe(surface, target, width, start_y, end_y);
}
