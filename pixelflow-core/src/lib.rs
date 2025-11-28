//! # PixelFlow Core
//!
//! A zero-cost, type-driven SIMD abstraction for pixel operations.
//!
//! ## Design Philosophy
//!
//! **The type IS the instruction selector.**
//!
//! - `Batch<u32>`: 32-bit operations (4 lanes)
//! - `Batch<u16>`: 16-bit operations (8 lanes)
//! - `Batch<u8>`: 8-bit operations (16 lanes)

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod backends;
pub mod batch;
pub mod simd;
pub mod dsl;
pub mod ops;
pub mod pipe;

pub use batch::{Batch, Batch256};
use simd::{Simd, SimdElement};

// ============================================================================
// Tensor Macros
// ============================================================================

macro_rules! define_tensor {
    ($name:ident, $rows:literal, $cols:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Copy, Clone)]
        pub struct $name<T: Copy + crate::simd::SimdElement, V: crate::simd::Simd = crate::batch::Batch<u32>> {
            /// The flattened elements of the tensor.
            pub elements: [V::Cast<T>; $rows * $cols],
        }
        impl<T: Copy + crate::simd::SimdElement, V: crate::simd::Simd> $name<T, V> {
            /// Creates a new tensor from an array of batches.
            #[inline(always)]
            pub fn new(elements: [V::Cast<T>; $rows * $cols]) -> Self {
                Self { elements }
            }

            /// Retrieves a batch from the tensor at the specified row and column.
            #[inline(always)]
            pub fn get(&self, row: usize, col: usize) -> V::Cast<T> {
                self.elements[row * $cols + col]
            }

            /// Applies a function to every batch in the tensor.
            #[inline(always)]
            pub fn map<U: Copy + crate::simd::SimdElement, F>(self, mut f: F) -> $name<U, V>
            where
                F: FnMut(V::Cast<T>) -> V::Cast<U>,
            {
                let elements = core::array::from_fn(|i| f(self.elements[i]));
                $name { elements }
            }
        }
    };
}

macro_rules! impl_matmul {
    ($left:ident, $right:ident, $output:ident, $m:literal, $k:literal, $n:literal) => {
        impl<T: Copy + crate::simd::SimdElement, V: crate::simd::Simd> $left<T, V> {
            /// Performs matrix multiplication with another tensor.
            #[inline(always)]
            pub fn matmul(&self, other: &$right<T, V>) -> $output<T, V> {
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
        impl<T: Copy + crate::simd::SimdElement, V: crate::simd::Simd> core::ops::Mul<$right<T, V>> for $left<T, V> {
            type Output = $output<T, V>;
            #[inline(always)]
            fn mul(self, other: $right<T, V>) -> Self::Output {
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

/// A read-only view into a 2D tensor (image/grid).
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

impl<'a> TensorView<'a, u8> {
    #[inline(always)]
    pub unsafe fn gather_2d<V: Simd>(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<u32> {
        let stride = <V::Cast<u32> as Simd>::splat(self.stride as u32);
        let idx_vec = (y * stride) + x;
        let max_idx = <V::Cast<u32> as Simd>::splat(self.data.len().saturating_sub(1) as u32);
        let clamped = idx_vec.min(max_idx);

        const LANES: usize = <V::Cast<u32> as Simd>::LANES;
        let mut idx_arr = [0u32; 64];
        let mut res_arr = [0u32; 64];

        unsafe { clamped.store(idx_arr.as_mut_ptr()) };
        for i in 0..LANES {
            res_arr[i] = unsafe { *self.data.get_unchecked(idx_arr[i] as usize) as u32 };
        }

        unsafe { <V::Cast<u32> as Simd>::load(res_arr.as_ptr()) }
    }

    #[inline(always)]
    pub unsafe fn gather_4bit<V: Simd>(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<u32> {
        let byte_x = x >> 1;
        let is_odd = x & <V::Cast<u32> as Simd>::splat(1);
        let packed = unsafe { self.gather_2d::<V>(byte_x, y) };
        let high_nibble = (packed >> 4) & <V::Cast<u32> as Simd>::splat(0x0F);
        let low_nibble = packed & <V::Cast<u32> as Simd>::splat(0x0F);
        let all_ones = <V::Cast<u32> as Simd>::splat(0xFFFFFFFF);
        let mask = is_odd * all_ones;
        let nibble = (high_nibble & !mask) | (low_nibble & mask);
        (nibble << 4) | nibble
    }

    #[inline(always)]
    pub unsafe fn gather_tensor2x2<V: Simd>(
        &self,
        x0: V::Cast<u32>,
        x1: V::Cast<u32>,
        y0: V::Cast<u32>,
        y1: V::Cast<u32>,
    ) -> Tensor2x2<u32, V> {
        Tensor2x2::<u32, V>::new([
            unsafe { self.gather_2d::<V>(x0, y0) },
            unsafe { self.gather_2d::<V>(x1, y0) },
            unsafe { self.gather_2d::<V>(x0, y1) },
            unsafe { self.gather_2d::<V>(x1, y1) },
        ])
    }

    #[inline(always)]
    pub unsafe fn gather_tensor2x2_4bit<V: Simd>(
        &self,
        x0: V::Cast<u32>,
        x1: V::Cast<u32>,
        y0: V::Cast<u32>,
        y1: V::Cast<u32>,
    ) -> Tensor2x2<u32, V> {
        Tensor2x2::<u32, V>::new([
            unsafe { self.gather_4bit::<V>(x0, y0) },
            unsafe { self.gather_4bit::<V>(x1, y0) },
            unsafe { self.gather_4bit::<V>(x0, y1) },
            unsafe { self.gather_4bit::<V>(x1, y1) },
        ])
    }

    #[inline(always)]
    pub unsafe fn sample_4bit_bilinear<V: Simd>(
        &self,
        u_fp: V::Cast<u32>,
        v_fp: V::Cast<u32>,
    ) -> V::Cast<u32> {
        let u0_raw = u_fp >> 16;
        let v0_raw = v_fp >> 16;
        let max_x = <V::Cast<u32> as Simd>::splat((self.width - 1) as u32);
        let max_y = <V::Cast<u32> as Simd>::splat((self.height - 1) as u32);
        let u0 = u0_raw.min(max_x);
        let v0 = v0_raw.min(max_y);
        let u1 = (u0 + <V::Cast<u32> as Simd>::splat(1)).min(max_x);
        let v1 = (v0 + <V::Cast<u32> as Simd>::splat(1)).min(max_y);
        let du = (u_fp >> 8) & <V::Cast<u32> as Simd>::splat(0xFF);
        let dv = (v_fp >> 8) & <V::Cast<u32> as Simd>::splat(0xFF);
        let inv_du = <V::Cast<u32> as Simd>::splat(256) - du;
        let inv_dv = <V::Cast<u32> as Simd>::splat(256) - dv;
        let pixels = unsafe { self.gather_tensor2x2_4bit::<V>(u0, u1, v0, v1) };

        let weights_x = Tensor2x1::<u32, V>::new([inv_du, du]);
        let weights_y = Tensor1x2::<u32, V>::new([inv_dv, dv]);

        let horizontal: Tensor2x1<u16, V> =
            (pixels.map(|p| p.cast::<u16>()) * weights_x.map(|w| w.cast::<u16>())).map(|v| v >> 8);
        let result: Tensor1x1<u16, V> =
            (weights_y.map(|w| w.cast::<u16>()) * horizontal).map(|v| v >> 8);
        unsafe {
            let tmp = result.get(0, 0).cast::<u32>();
            core::ptr::read(&tmp as *const _ as *const V::Cast<u32>)
        }
    }

    #[inline(always)]
    pub unsafe fn sample_4bit_nearest<V: Simd>(
        &self,
        u: V::Cast<u32>,
        v: V::Cast<u32>,
    ) -> V::Cast<u32> {
        let clamped_u = u.min(<V::Cast<u32> as Simd>::splat((self.width - 1) as u32));
        let clamped_v = v.min(<V::Cast<u32> as Simd>::splat((self.height - 1) as u32));
        unsafe { self.gather_4bit::<V>(clamped_u, clamped_v) }
    }
}

// --- TensorViewMut ---

pub struct TensorViewMut<'a, T> {
    pub data: &'a mut [T],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<'a, T> TensorViewMut<'a, T> {
    #[inline(always)]
    pub fn new(data: &'a mut [T], width: usize, height: usize, stride: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    #[inline(always)]
    pub unsafe fn sub_view(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> TensorViewMut<'_, T> {
        let start_offset = y * self.stride + x;
        let ptr = unsafe { self.data.as_mut_ptr().add(start_offset) };
        let len = self.data.len().saturating_sub(start_offset);
        let slice = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
        TensorViewMut {
            data: slice,
            width,
            height,
            stride: self.stride,
        }
    }
}

// Define a trait to expose map_pixels generically
/// Trait for types that can map a function over pixels in a batch-wise manner.
pub trait MapPixels<T: Copy + SimdElement> {
    /// Maps a function over the pixels.
    fn map_pixels<F, V: Simd>(&mut self, f: F)
    where
        F: FnMut(V::Cast<u32>, V::Cast<u32>) -> V::Cast<T>;
}

impl<'a> MapPixels<u8> for TensorViewMut<'a, u8> {
    #[inline(always)]
    fn map_pixels<F, V: Simd>(&mut self, mut f: F)
    where
        F: FnMut(V::Cast<u32>, V::Cast<u32>) -> V::Cast<u8>,
    {
        const LANES: usize = <V::Cast<u32> as Simd>::LANES;
        let mut res_buf = [0u8; 64];

        for y in 0..self.height {
            let y_vec = <V::Cast<u32> as Simd>::splat(y as u32);
            let row_offset = y * self.stride;
            let mut x = 0;
            let iota = <V::Cast<u32> as Simd>::iota();

            while x + LANES <= self.width {
                let x_vec = <V::Cast<u32> as Simd>::splat(x as u32) + iota;
                let result = f(x_vec, y_vec);

                // Generic store for u8
                unsafe { result.store(res_buf.as_mut_ptr()) };
                for i in 0..LANES {
                    self.data[row_offset + x + i] = res_buf[i];
                }
                x += LANES;
            }
            if x < self.width {
                let x_vec = <V::Cast<u32> as Simd>::splat(x as u32) + iota;
                let result = f(x_vec, y_vec);
                unsafe { result.store(res_buf.as_mut_ptr()) };
                let count = (self.width - x).min(LANES);
                for i in 0..count {
                     self.data[row_offset + x + i] = res_buf[i];
                }
            }
        }
    }
}

impl<'a> MapPixels<u32> for TensorViewMut<'a, u32> {
    #[inline(always)]
    fn map_pixels<F, V: Simd>(&mut self, mut f: F)
    where
        F: FnMut(V::Cast<u32>, V::Cast<u32>) -> V::Cast<u32>,
    {
        const LANES: usize = <V::Cast<u32> as Simd>::LANES;
        for y in 0..self.height {
            let y_vec = <V::Cast<u32> as Simd>::splat(y as u32);
            let row_offset = y * self.stride;
            let mut x = 0;
            let iota = <V::Cast<u32> as Simd>::iota();

            while x + LANES <= self.width {
                let x_vec = <V::Cast<u32> as Simd>::splat(x as u32) + iota;
                let result = f(x_vec, y_vec);
                unsafe {
                    result.store(self.data.as_mut_ptr().add(row_offset + x));
                }
                x += LANES;
            }

            if x < self.width {
                let x_vec = <V::Cast<u32> as Simd>::splat(x as u32) + iota;
                let result = f(x_vec, y_vec);
                let tail_slice = &mut self.data[row_offset + x..];

                // Safe generic partial store
                let mut buf = [0u32; 64];
                unsafe { result.store(buf.as_mut_ptr()) };

                let write_len = (self.width - x).min(tail_slice.len());
                for i in 0..write_len {
                    tail_slice[i] = buf[i];
                }
            }
        }
    }
}

// ============================================================================
// Execution
// ============================================================================

/// Executes a pipeline operation onto a target tensor.
///
/// # Parameters
/// * `pipe` - The pipeline to evaluate.
/// * `target` - The target mutable tensor view.
///
/// # Type Parameters
/// * `T` - The pixel type (e.g., `u8`, `u32`).
/// * `P` - The pipeline type which implements `pipe::Surface`.
pub fn execute<T, P>(pipe: P, target: &mut TensorViewMut<T>)
where
    T: Copy + SimdElement,
    P: pipe::Surface<Batch256<u32>, T>,
    for<'a> TensorViewMut<'a, T>: MapPixels<T>, // Constrain T to types that support mapping
{
    // Opt-in to 256-bit SIMD (8 lanes)
    target.map_pixels::<_, Batch256<u32>>(|x, y| pipe.eval(x, y));
}

#[cfg(test)]
mod tests {
    use super::*;
    use batch::Batch;
    #[test]
    fn test_gather_4bit_correctness() {
        let packed = [0x12u8, 0x34, 0x56, 0x78];
        let view = TensorView::new(&packed, 8, 1, 4);
        let x = Batch::<u32>::new(0, 1, 2, 3);
        let y = Batch::<u32>::splat(0);
        let result = unsafe { view.gather_4bit(x, y) };
        let expected = [1 * 17, 2 * 17, 3 * 17, 4 * 17];
        assert_eq!(result.to_array_usize(), expected);
    }
}
