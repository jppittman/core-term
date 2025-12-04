//! # PixelFlow Core
//!
//! A zero-cost, type-driven SIMD abstraction for pixel operations.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod backend;
pub mod backends;
pub mod batch;
/// Curve and matrix primitives for vector graphics.
pub mod curve;
/// Downcasting support for trait objects.
pub mod downcast;
/// Domain-specific language extensions (fluent combinators).
pub mod dsl;
/// Surface combinators and operations (Scale, Offset, Over, etc.).
pub mod ops;
/// The core Surface trait and pipeline definitions.
pub mod pipe;
pub mod pixel;
/// Platform configuration and rendering utilities.
pub mod platform;

pub use backend::{FloatBatchOps, SimdBatch}; // Export SimdBatch and FloatBatchOps
pub use batch::{Batch, SHUFFLE_RGBA_BGRA, SimdOps};
pub use ops::Scale;
pub use pixel::Pixel;
pub use platform::{PixelFormat, Platform};

use crate::backend::{Backend, BatchArithmetic};
use core::fmt::Debug;
use core::ops::{BitAnd, BitOr, Not, Shl, Shr};

// ============================================================================
// Tensor Macros
// ============================================================================

macro_rules! define_tensor {
    ($name:ident, $rows:literal, $cols:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Copy, Clone)]
        pub struct $name<T, B: Backend>
        where
            T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
        {
            /// The underlying elements of the tensor.
            pub elements: [B::Batch<T>; $rows * $cols],
        }
        impl<T, B: Backend> $name<T, B>
        where
            T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
        {
            /// Creates a new tensor from an array of batches.
            #[inline(always)]
            pub fn new(elements: [B::Batch<T>; $rows * $cols]) -> Self {
                Self { elements }
            }

            /// Gets the element at the specified row and column.
            #[inline(always)]
            pub fn get(&self, row: usize, col: usize) -> B::Batch<T> {
                self.elements[row * $cols + col]
            }

            /// Maps a function over all elements of the tensor.
            #[inline(always)]
            pub fn map<U, F>(self, mut f: F) -> $name<U, B>
            where
                U: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
                F: FnMut(B::Batch<T>) -> B::Batch<U>,
            {
                let elements = core::array::from_fn(|i| f(self.elements[i]));
                $name { elements }
            }
        }
    };
}

macro_rules! impl_matmul {
    ($left:ident, $right:ident, $output:ident, $m:literal, $k:literal, $n:literal) => {
        impl<T, B: Backend> $left<T, B>
        where
            T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            B::Batch<T>: BatchArithmetic<T>,
        {
            /// Performs matrix multiplication.
            #[inline(always)]
            #[allow(clippy::modulo_one)]
            pub fn matmul(&self, other: &$right<T, B>) -> $output<T, B> {
                let elements = core::array::from_fn(|i| {
                    let r = i / $n;
                    let c = i % $n;
                    let mut sum = self.get(r, 0) * other.get(0, c);
                    let mut k = 1;
                    while k < $k {
                        sum = sum + (self.get(r, k) * other.get(k, c));
                        k += 1;
                    }
                    sum
                });
                $output { elements }
            }
        }
        impl<T, B: Backend> core::ops::Mul<$right<T, B>> for $left<T, B>
        where
            T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            B::Batch<T>: BatchArithmetic<T>,
        {
            type Output = $output<T, B>;
            #[inline(always)]
            fn mul(self, other: $right<T, B>) -> Self::Output {
                self.matmul(&other)
            }
        }
    };
}

define_tensor!(Tensor2x2, 2, 2, "A 2x2 tensor.");
define_tensor!(Tensor2x1, 2, 1, "A 2x1 tensor (column vector).");
define_tensor!(Tensor1x2, 1, 2, "A 1x2 tensor (row vector).");
define_tensor!(Tensor1x1, 1, 1, "A 1x1 tensor (scalar wrapper).");

impl_matmul!(Tensor2x2, Tensor2x1, Tensor2x1, 2, 2, 1);
impl_matmul!(Tensor1x2, Tensor2x1, Tensor1x1, 1, 2, 1);

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

    /// Gathers a 2x2 neighborhood of pixels.
    ///
    /// # Safety
    /// Unsafe due to raw gather operations.
    #[inline(always)]
    pub unsafe fn gather_tensor2x2<B: Backend>(
        &self,
        x0: B::Batch<u32>,
        x1: B::Batch<u32>,
        y0: B::Batch<u32>,
        y1: B::Batch<u32>,
    ) -> Tensor2x2<u32, B>
    where
        B::Batch<u32>: BatchArithmetic<u32> + BitAnd<Output = B::Batch<u32>>,
    {
        Tensor2x2::new([
            unsafe { self.gather_2d::<B>(x0, y0) },
            unsafe { self.gather_2d::<B>(x1, y0) },
            unsafe { self.gather_2d::<B>(x0, y1) },
            unsafe { self.gather_2d::<B>(x1, y1) },
        ])
    }

    /// Gathers a 2x2 neighborhood of 4-bit packed pixels.
    ///
    /// # Safety
    /// Unsafe due to raw gather operations.
    #[inline(always)]
    pub unsafe fn gather_tensor2x2_4bit<B: Backend>(
        &self,
        x0: B::Batch<u32>,
        x1: B::Batch<u32>,
        y0: B::Batch<u32>,
        y1: B::Batch<u32>,
    ) -> Tensor2x2<u32, B>
    where
        B::Batch<u32>: BatchArithmetic<u32>
            + BitAnd<Output = B::Batch<u32>>
            + BitOr<Output = B::Batch<u32>>
            + Not<Output = B::Batch<u32>>
            + Shr<i32, Output = B::Batch<u32>>
            + Shl<i32, Output = B::Batch<u32>>,
    {
        Tensor2x2::new([
            unsafe { self.gather_4bit::<B>(x0, y0) },
            unsafe { self.gather_4bit::<B>(x1, y0) },
            unsafe { self.gather_4bit::<B>(x0, y1) },
            unsafe { self.gather_4bit::<B>(x1, y1) },
        ])
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
pub fn execute<P, S>(surface: &S, target: &mut [P], width: usize, height: usize)
where
    P: pixel::Pixel,
    S: pipe::Surface<P> + ?Sized,
{
    use crate::batch::{Batch, LANES};

    // Early return for zero-size targets to avoid UB with from_raw_parts_mut
    if width == 0 || height == 0 {
        return;
    }

    for y in 0..height {
        let row_start = y * width;
        let y_batch = Batch::<u32>::splat(y as u32);

        let mut x = 0;
        // Hot path: SIMD loop
        while x + LANES <= width {
            let x_batch = Batch::<u32>::sequential_from(x as u32);
            let result = surface.eval(x_batch, y_batch);
            P::batch_store(result, &mut target[row_start + x..row_start + x + LANES]);
            x += LANES;
        }

        // Cold path: remainder pixels
        while x < width {
            target[row_start + x] = surface.eval_one(x as u32, y as u32);
            x += 1;
        }
    }
}