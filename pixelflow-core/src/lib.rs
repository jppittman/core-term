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
/// SIMD batch processing types and operations.
pub mod batch;
/// Domain-Specific Language (DSL) extensions for building pipelines.
pub mod dsl;
/// Core operations and surface implementations.
pub mod ops;
/// Pixel format trait for generic color operations.
pub mod pixel;
/// Pipeline abstractions and surface traits.
pub mod pipe;

pub use batch::{Batch, SimdOpsU8, SHUFFLE_RGBA_BGRA};
pub use pixel::Pixel;

// ============================================================================
// Tensor Macros
// ============================================================================

macro_rules! define_tensor {
    ($name:ident, $rows:literal, $cols:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Copy, Clone)]
        pub struct $name<T: Copy> {
            /// The flattened elements of the tensor.
            pub elements: [crate::batch::Batch<T>; $rows * $cols],
        }
        impl<T: Copy> $name<T> {
            /// Creates a new tensor from an array of batches.
            ///
            /// # Parameters
            /// * `elements` - The array of batches representing the tensor.
            ///
            /// # Returns
            /// * A new tensor instance.
            #[inline(always)]
            pub fn new(elements: [crate::batch::Batch<T>; $rows * $cols]) -> Self {
                Self { elements }
            }

            /// Retrieves a batch from the tensor at the specified row and column.
            ///
            /// # Parameters
            /// * `row` - The row index.
            /// * `col` - The column index.
            ///
            /// # Returns
            /// * The batch at the specified position.
            #[inline(always)]
            pub fn get(&self, row: usize, col: usize) -> crate::batch::Batch<T> {
                self.elements[row * $cols + col]
            }

            /// Applies a function to every batch in the tensor.
            ///
            /// # Parameters
            /// * `f` - The function to apply to each batch.
            ///
            /// # Returns
            /// * A new tensor with the transformed elements.
            #[inline(always)]
            pub fn map<U: Copy, F>(self, mut f: F) -> $name<U>
            where
                F: FnMut(crate::batch::Batch<T>) -> crate::batch::Batch<U>,
            {
                let elements = core::array::from_fn(|i| f(self.elements[i]));
                $name { elements }
            }
        }
    };
}

macro_rules! impl_matmul {
    ($left:ident, $right:ident, $output:ident, $m:literal, $k:literal, $n:literal) => {
        impl<T: Copy> $left<T>
        where
            crate::batch::SimdVec<T>: crate::batch::SimdOps<T>,
        {
            /// Performs matrix multiplication with another tensor.
            ///
            /// # Parameters
            /// * `other` - The right-hand side tensor.
            ///
            /// # Returns
            /// * The result of the matrix multiplication.
            #[inline(always)]
            pub fn matmul(&self, other: &$right<T>) -> $output<T> {
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
        impl<T: Copy> core::ops::Mul<$right<T>> for $left<T>
        where
            crate::batch::SimdVec<T>: crate::batch::SimdOps<T>,
        {
            type Output = $output<T>;
            /// Multiplies this tensor with another tensor.
            #[inline(always)]
            fn mul(self, other: $right<T>) -> Self::Output {
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
///
/// This struct provides efficient access to pixel data, supporting strided access
/// and various sampling operations (gather, bilinear interpolation).
#[derive(Copy, Clone)]
pub struct TensorView<'a, T> {
    /// The raw data slice.
    pub data: &'a [T],
    /// The width of the tensor in elements.
    pub width: usize,
    /// The height of the tensor in elements.
    pub height: usize,
    /// The stride (number of elements) between rows.
    pub stride: usize,
}

impl<'a, T> TensorView<'a, T> {
    /// Creates a new `TensorView`.
    ///
    /// # Parameters
    /// * `data` - The raw data slice.
    /// * `width` - The width of the view.
    /// * `height` - The height of the view.
    /// * `stride` - The stride (row pitch) of the view.
    ///
    /// # Returns
    /// * A new `TensorView` instance.
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

impl<'a, T: batch::Gatherable> TensorView<'a, T> {
    /// Gathers 32-bit values from the tensor at the specified 2D coordinates.
    ///
    /// # Parameters
    /// * `x` - A batch of X coordinates.
    /// * `y` - A batch of Y coordinates.
    ///
    /// # Returns
    /// * A batch containing the gathered values.
    ///
    /// # Safety
    /// This function uses `gather` which may be unsafe if indices are out of bounds,
    /// though this implementation clamps indices to the tensor dimensions (saturating).
    #[inline(always)]
    pub unsafe fn gather_2d(
        &self,
        x: batch::Batch<u32>,
        y: batch::Batch<u32>,
    ) -> batch::Batch<u32> {
        // Vectorized Index Calculation: idx_vec = y * stride + x
        let stride_vec = batch::Batch::splat(self.stride as u32);
        let idx_vec = (y * stride_vec) + x;

        // Vectorized Clamping
        let max_idx = self.data.len().saturating_sub(1) as u32;
        let max_idx_vec = batch::Batch::splat(max_idx);
        let clamped_idx_vec = idx_vec.min(max_idx_vec);

        // Extract for Scalar Gather (Bridge to Hardware)
        let indices = clamped_idx_vec.to_array_usize();
        unsafe { batch::Batch::gather(self.data, indices) }
    }
}

impl<'a> TensorView<'a, u8> {
    /// Gathers 4-bit packed values from the tensor.
    ///
    /// Assumes pixels are packed 2 per byte (4 bits each).
    ///
    /// # Parameters
    /// * `x` - A batch of X coordinates.
    /// * `y` - A batch of Y coordinates.
    ///
    /// # Returns
    /// * A batch containing the gathered 4-bit values, expanded to 8-bit by replication (e.g., 0xA -> 0xAA).
    ///
    /// # Safety
    /// Relies on `gather_2d` which is unsafe.
    #[inline(always)]
    pub unsafe fn gather_4bit(
        &self,
        x: batch::Batch<u32>,
        y: batch::Batch<u32>,
    ) -> batch::Batch<u32> {
        let byte_x = x >> 1;
        let is_odd = x & batch::Batch::splat(1);
        let packed = unsafe { self.gather_2d(byte_x, y) };
        let high_nibble = (packed >> 4) & batch::Batch::splat(0x0F);
        let low_nibble = packed & batch::Batch::splat(0x0F);
        let all_ones = batch::Batch::splat(0xFFFFFFFF);
        let mask = is_odd * all_ones;
        let nibble = (high_nibble & !mask) | (low_nibble & mask);
        (nibble << 4) | nibble
    }

    /// Gathers a 2x2 tensor of values.
    ///
    /// # Parameters
    /// * `x0` - X coordinates for the left column.
    /// * `x1` - X coordinates for the right column.
    /// * `y0` - Y coordinates for the top row.
    /// * `y1` - Y coordinates for the bottom row.
    ///
    /// # Returns
    /// * A `Tensor2x2` containing the gathered values.
    ///
    /// # Safety
    /// Relies on `gather_2d`.
    #[inline(always)]
    pub unsafe fn gather_tensor2x2(
        &self,
        x0: batch::Batch<u32>,
        x1: batch::Batch<u32>,
        y0: batch::Batch<u32>,
        y1: batch::Batch<u32>,
    ) -> Tensor2x2<u32> {
        Tensor2x2::new([
            unsafe { self.gather_2d(x0, y0) },
            unsafe { self.gather_2d(x1, y0) },
            unsafe { self.gather_2d(x0, y1) },
            unsafe { self.gather_2d(x1, y1) },
        ])
    }

    /// Gathers a 2x2 tensor of 4-bit packed values.
    ///
    /// # Parameters
    /// * `x0` - X coordinates for the left column.
    /// * `x1` - X coordinates for the right column.
    /// * `y0` - Y coordinates for the top row.
    /// * `y1` - Y coordinates for the bottom row.
    ///
    /// # Returns
    /// * A `Tensor2x2` containing the gathered values (expanded to 8-bit).
    ///
    /// # Safety
    /// Relies on `gather_4bit`.
    #[inline(always)]
    pub unsafe fn gather_tensor2x2_4bit(
        &self,
        x0: batch::Batch<u32>,
        x1: batch::Batch<u32>,
        y0: batch::Batch<u32>,
        y1: batch::Batch<u32>,
    ) -> Tensor2x2<u32> {
        Tensor2x2::new([
            unsafe { self.gather_4bit(x0, y0) },
            unsafe { self.gather_4bit(x1, y0) },
            unsafe { self.gather_4bit(x0, y1) },
            unsafe { self.gather_4bit(x1, y1) },
        ])
    }

    /// Samples 4-bit packed values using bilinear interpolation.
    ///
    /// # Parameters
    /// * `u_fp` - Fixed-point X coordinates (16.16 format).
    /// * `v_fp` - Fixed-point Y coordinates (16.16 format).
    ///
    /// # Returns
    /// * A batch of interpolated values.
    ///
    /// # Safety
    /// Relies on `gather_tensor2x2_4bit`.
    #[inline(always)]
    pub unsafe fn sample_4bit_bilinear(
        &self,
        u_fp: batch::Batch<u32>,
        v_fp: batch::Batch<u32>,
    ) -> batch::Batch<u32> {
        let u0_raw = u_fp >> 16;
        let v0_raw = v_fp >> 16;
        let max_x = batch::Batch::splat((self.width - 1) as u32);
        let max_y = batch::Batch::splat((self.height - 1) as u32);
        let u0 = u0_raw.min(max_x);
        let v0 = v0_raw.min(max_y);
        let u1 = (u0 + batch::Batch::splat(1)).min(max_x);
        let v1 = (v0 + batch::Batch::splat(1)).min(max_y);
        let du = (u_fp >> 8) & batch::Batch::splat(0xFF);
        let dv = (v_fp >> 8) & batch::Batch::splat(0xFF);
        let inv_du = batch::Batch::splat(256) - du;
        let inv_dv = batch::Batch::splat(256) - dv;
        let pixels = unsafe { self.gather_tensor2x2_4bit(u0, u1, v0, v1) };
        let weights_x = Tensor2x1::new([inv_du, du]);
        let weights_y = Tensor1x2::new([inv_dv, dv]);
        let horizontal: Tensor2x1<u16> =
            (pixels.map(|p| p.cast::<u16>()) * weights_x.map(|w| w.cast::<u16>())).map(|v| v >> 8);
        let result: Tensor1x1<u16> =
            (weights_y.map(|w| w.cast::<u16>()) * horizontal).map(|v| v >> 8);
        result.get(0, 0).cast::<u32>()
    }

    /// Samples 4-bit packed values using nearest neighbor interpolation.
    ///
    /// # Parameters
    /// * `u` - X coordinates.
    /// * `v` - Y coordinates.
    ///
    /// # Returns
    /// * A batch of sampled values.
    ///
    /// # Safety
    /// Relies on `gather_4bit`.
    #[inline(always)]
    pub unsafe fn sample_4bit_nearest(
        &self,
        u: batch::Batch<u32>,
        v: batch::Batch<u32>,
    ) -> batch::Batch<u32> {
        let clamped_u = u.min(batch::Batch::splat((self.width - 1) as u32));
        let clamped_v = v.min(batch::Batch::splat((self.height - 1) as u32));
        unsafe { self.gather_4bit(clamped_u, clamped_v) }
    }
}

// --- TensorViewMut ---

/// A mutable view into a 2D tensor (image/grid).
///
/// Allows for modification of pixel data and creating sub-views.
pub struct TensorViewMut<'a, T> {
    /// The mutable raw data slice.
    pub data: &'a mut [T],
    /// The width of the view.
    pub width: usize,
    /// The height of the view.
    pub height: usize,
    /// The stride of the view.
    pub stride: usize,
}

impl<'a, T> TensorViewMut<'a, T> {
    /// Creates a new `TensorViewMut`.
    ///
    /// # Parameters
    /// * `data` - The mutable raw data slice.
    /// * `width` - The width of the view.
    /// * `height` - The height of the view.
    /// * `stride` - The stride of the view.
    ///
    /// # Returns
    /// * A new `TensorViewMut` instance.
    #[inline(always)]
    pub fn new(data: &'a mut [T], width: usize, height: usize, stride: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    /// Creates a mutable sub-view from this view.
    ///
    /// # Parameters
    /// * `x` - The x-coordinate of the top-left corner of the sub-view.
    /// * `y` - The y-coordinate of the top-left corner of the sub-view.
    /// * `width` - The width of the sub-view.
    /// * `height` - The height of the sub-view.
    ///
    /// # Returns
    /// * A new `TensorViewMut` representing the sub-view.
    ///
    /// # Safety
    /// Uses unsafe pointer arithmetic to create the slice.
    #[inline(always)]
    pub unsafe fn sub_view(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> TensorViewMut<'_, T> {
        let start_offset = y * self.stride + x;
        // Wrapped unsafe calls in unsafe blocks as required
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
pub trait MapPixels<T: Copy> {
    /// Maps a function over the pixels.
    ///
    /// # Parameters
    /// * `f` - A function taking X and Y coordinate batches and returning a pixel batch.
    fn map_pixels<F>(&mut self, f: F)
    where
        F: FnMut(batch::Batch<u32>, batch::Batch<u32>) -> batch::Batch<T>;
}

impl<'a> MapPixels<u8> for TensorViewMut<'a, u8> {
    #[inline(always)]
    fn map_pixels<F>(&mut self, mut f: F)
    where
        F: FnMut(batch::Batch<u32>, batch::Batch<u32>) -> batch::Batch<u8>,
    {
        const LANES: usize = 4;
        for y in 0..self.height {
            let y_vec = batch::Batch::splat(y as u32);
            let row_offset = y * self.stride;
            let mut x = 0;
            while x + LANES <= self.width {
                let x_vec =
                    batch::Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);
                let result = f(x_vec, y_vec);
                let bytes = result.cast::<u32>().to_bytes_packed();
                self.data[row_offset + x] = bytes[0];
                self.data[row_offset + x + 1] = bytes[1];
                self.data[row_offset + x + 2] = bytes[2];
                self.data[row_offset + x + 3] = bytes[3];
                x += LANES;
            }
            if x < self.width {
                let x_vec =
                    batch::Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);
                let result = f(x_vec, y_vec);
                let bytes = result.cast::<u32>().to_bytes_packed();
                let tail_slice = &mut self.data[row_offset + x..];
                let count = (self.width - x).min(4).min(tail_slice.len());
                for i in 0..count {
                    tail_slice[i] = bytes[i];
                }
            }
        }
    }
}

impl<'a> MapPixels<u32> for TensorViewMut<'a, u32> {
    #[inline(always)]
    fn map_pixels<F>(&mut self, mut f: F)
    where
        F: FnMut(batch::Batch<u32>, batch::Batch<u32>) -> batch::Batch<u32>,
    {
        const LANES: usize = 4;
        for y in 0..self.height {
            let y_vec = batch::Batch::splat(y as u32);
            let row_offset = y * self.stride;
            let mut x = 0;

            // 1. Hot Path: Unchecked SIMD (99% of pixels)
            while x + LANES <= self.width {
                let x_vec =
                    batch::Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);
                let result = f(x_vec, y_vec);
                unsafe {
                    result.store(self.data.as_mut_ptr().add(row_offset + x));
                }
                x += LANES;
            }

            // 2. Cold Path: Safe Partial Store (Right Edge)
            if x < self.width {
                let x_vec =
                    batch::Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);
                let result = f(x_vec, y_vec);
                let tail_slice = &mut self.data[row_offset + x..];
                let write_len = (self.width - x).min(tail_slice.len());
                result.store_into_slice(&mut tail_slice[..write_len]);
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
    T: Copy,
    P: pipe::Surface<T>,
    for<'a> TensorViewMut<'a, T>: MapPixels<T>, // Constrain T to types that support mapping
{
    target.map_pixels(|x, y| pipe.eval(x, y));
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
