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
pub mod dsl;
pub mod ops;
pub mod pipe;
pub mod pixel;
pub mod platform;

pub use batch::{Batch, SimdOps, SHUFFLE_RGBA_BGRA};
pub use backend::SimdBatch; // Export SimdBatch
pub use ops::Scale;
pub use pixel::Pixel;
pub use platform::{Platform, PixelFormat};

use crate::backend::{Backend, BatchArithmetic};
use core::fmt::Debug;

// ============================================================================
// Tensor Macros
// ============================================================================

macro_rules! define_tensor {
    ($name:ident, $rows:literal, $cols:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Copy, Clone)]
        pub struct $name<T, B: Backend>
        where T: Copy + Debug + Default + Send + Sync + 'static
        {
            pub elements: [B::Batch<T>; $rows * $cols],
        }
        impl<T, B: Backend> $name<T, B>
        where T: Copy + Debug + Default + Send + Sync + 'static
        {
            #[inline(always)]
            pub fn new(elements: [B::Batch<T>; $rows * $cols]) -> Self {
                Self { elements }
            }

            #[inline(always)]
            pub fn get(&self, row: usize, col: usize) -> B::Batch<T> {
                self.elements[row * $cols + col]
            }

            #[inline(always)]
            pub fn map<U, F>(self, mut f: F) -> $name<U, B>
            where
                U: Copy + Debug + Default + Send + Sync + 'static,
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
            T: Copy + Debug + Default + Send + Sync + 'static,
            B::Batch<T>: BatchArithmetic<T>
        {
            #[inline(always)]
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
            T: Copy + Debug + Default + Send + Sync + 'static,
            B::Batch<T>: BatchArithmetic<T>
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

#[derive(Copy, Clone)]
pub struct TensorView<'a, T> {
    pub data: &'a [T],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<'a, T> TensorView<'a, T> {
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
    #[inline(always)]
    pub unsafe fn gather_2d<B: Backend>(
        &self,
        x: B::Batch<u32>,
        y: B::Batch<u32>,
    ) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let stride_vec = <B::Batch<u32> as SimdBatch<u32>>::splat(self.stride as u32);
        let idx_vec = (y * stride_vec) + x;
        let max_idx = self.data.len().saturating_sub(1) as u32;
        let max_idx_vec = <B::Batch<u32> as SimdBatch<u32>>::splat(max_idx);
        let clamped_idx_vec = idx_vec.min(max_idx_vec);
        <B::Batch<u32> as BatchArithmetic<u32>>::gather(self.data, clamped_idx_vec)
    }
}

impl<'a> TensorView<'a, u8> {
    #[inline(always)]
    pub unsafe fn gather_2d<B: Backend>(
        &self,
        x: B::Batch<u32>,
        y: B::Batch<u32>,
    ) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let stride_vec = <B::Batch<u32> as SimdBatch<u32>>::splat(self.stride as u32);
        let idx_vec = (y * stride_vec) + x;
        let max_idx = self.data.len().saturating_sub(1) as u32;
        let max_idx_vec = <B::Batch<u32> as SimdBatch<u32>>::splat(max_idx);
        let clamped_idx_vec = idx_vec.min(max_idx_vec);
        <B::Batch<u32> as BatchArithmetic<u32>>::gather_u8(self.data, clamped_idx_vec)
    }

    #[inline(always)]
    pub unsafe fn gather_4bit<B: Backend>(
        &self,
        x: B::Batch<u32>,
        y: B::Batch<u32>,
    ) -> B::Batch<u32>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let byte_x = x >> 1;
        let is_odd = x & <B::Batch<u32> as SimdBatch<u32>>::splat(1);
        let packed = unsafe { self.gather_2d::<B>(byte_x, y) };
        let high_nibble = (packed >> 4) & <B::Batch<u32> as SimdBatch<u32>>::splat(0x0F);
        let low_nibble = packed & <B::Batch<u32> as SimdBatch<u32>>::splat(0x0F);
        let all_ones = <B::Batch<u32> as SimdBatch<u32>>::splat(0xFFFFFFFF);
        let mask = is_odd * all_ones;
        let nibble = (high_nibble & !mask) | (low_nibble & mask);
        (nibble << 4) | nibble
    }

    #[inline(always)]
    pub unsafe fn gather_tensor2x2<B: Backend>(
        &self,
        x0: B::Batch<u32>,
        x1: B::Batch<u32>,
        y0: B::Batch<u32>,
        y1: B::Batch<u32>,
    ) -> Tensor2x2<u32, B>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        Tensor2x2::new([
            unsafe { self.gather_2d::<B>(x0, y0) },
            unsafe { self.gather_2d::<B>(x1, y0) },
            unsafe { self.gather_2d::<B>(x0, y1) },
            unsafe { self.gather_2d::<B>(x1, y1) },
        ])
    }

    #[inline(always)]
    pub unsafe fn gather_tensor2x2_4bit<B: Backend>(
        &self,
        x0: B::Batch<u32>,
        x1: B::Batch<u32>,
        y0: B::Batch<u32>,
        y1: B::Batch<u32>,
    ) -> Tensor2x2<u32, B>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        Tensor2x2::new([
            unsafe { self.gather_4bit::<B>(x0, y0) },
            unsafe { self.gather_4bit::<B>(x1, y0) },
            unsafe { self.gather_4bit::<B>(x0, y1) },
            unsafe { self.gather_4bit::<B>(x1, y1) },
        ])
    }

    #[inline(always)]
    pub unsafe fn sample_4bit_bilinear<B: Backend>(
        &self,
        u_fp: B::Batch<u32>,
        v_fp: B::Batch<u32>,
    ) -> B::Batch<u32>
    where
        B::Batch<u32>: BatchArithmetic<u32>
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
        let pixels = unsafe { self.gather_tensor2x2_4bit::<B>(u0, u1, v0, v1) };
        let weights_x = Tensor2x1::new([inv_du, du]);
        let weights_y = Tensor1x2::new([inv_dv, dv]);

        let horizontal: Tensor2x1<u32, B> =
            (pixels * weights_x).map(|v| v >> 8);

        let result: Tensor1x1<u32, B> =
            (weights_y * horizontal).map(|v| v >> 8);

        result.get(0, 0)
    }
}

pub struct TensorViewMut<'a, T> {
    pub data: &'a mut [T],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

pub fn execute<P, S>(surface: &S, target: &mut [P], width: usize, height: usize)
where
    P: pixel::Pixel,
    S: pipe::Surface<P> + ?Sized,
{
    use crate::batch::{NativeBackend, LANES};

    for y in 0..height {
        let row_start = y * width;
        let y_batch = <NativeBackend as Backend>::Batch::<u32>::splat(y as u32);

        let mut x = 0;
        while x + LANES <= width {
            let x_batch = <NativeBackend as Backend>::Batch::<u32>::sequential_from(x as u32);

            let result = surface.eval::<NativeBackend>(x_batch, y_batch);

            // Convert result (Batch<P>) to Batch<u32> for store
            // NativeBackend::Batch<u32> implies arithmetic
            let result_u32 = P::batch_to_u32::<NativeBackend>(result);

            unsafe {
                let ptr = target.as_mut_ptr().add(row_start + x) as *mut u32;
                let slice = core::slice::from_raw_parts_mut(ptr, LANES);
                SimdBatch::store(&result_u32, slice);
            }

            x += LANES;
        }

        while x < width {
             use crate::backends::scalar::{Scalar, ScalarBatch};
             let x_scalar = ScalarBatch(x as u32);
             let y_scalar = ScalarBatch(y as u32);
             let result = surface.eval::<Scalar>(x_scalar, y_scalar);
             // Downcast result to u32? No result is Batch<P>.
             // P::batch_to_u32::<Scalar>(result) returns ScalarBatch<u32>
             let res_u32 = P::batch_to_u32::<Scalar>(result);
             // Scalar store writes 1 u32.
             // Target is [P]. P::from_u32.
             target[row_start + x] = P::from_u32(res_u32.0);
             x += 1;
        }
    }
}
