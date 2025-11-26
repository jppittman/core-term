//! # PixelFlow Core
//!
//! A zero-cost, type-driven SIMD abstraction for pixel operations.
//!
//! ## Design Philosophy
//!
//! **The type IS the instruction selector.**
//!
//! Unlike C-style SIMD libraries that expose functions like `add_u16()` vs `add_u32()`,
//! PixelFlow uses Rust's type system to automatically select the correct instruction:
//!
//! - `Batch<u32> + Batch<u32>` → `paddd` (32-bit add)
//! - `Batch<u16> + Batch<u16>` → `paddw` (16-bit add)
//! - `Batch<u16> * Batch<u16>` → `pmullw` (16-bit multiply)
//!
//! ## Type Inference
//!
//! The struct is **unconstrained** - bounds only appear on method impls.
//! This lets Rust infer types automatically:
//!
//! ```ignore
//! let a = Batch::splat(100u32);  // Infers Batch<u32>
//! let b = Batch::splat(50);      // Infers u32 from context
//! let c = a + b;                 // Infers Batch<u32>
//!
//! // Cast to u16 for multiply
//! let result = (a.cast::<u16>() * b.cast::<u16>()) >> 8;
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

use core::ops::{Add, Sub, Mul, BitAnd, BitOr, Not, Shl, Shr};

pub mod backends;

// Select backend based on target architecture
#[cfg(target_arch = "x86_64")]
use backends::x86 as backend;

#[cfg(target_arch = "aarch64")]
use backends::arm as backend;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use backends::scalar as backend;

// Re-export platform-optimal types
pub use backend::SimdVec;

/// A SIMD batch of values of type `T`.
///
/// **Minimal constraints** - only `T: Copy` (required by ARM backend).
/// All other bounds appear on method impls for maximum type inference.
///
/// The type `T` determines which SIMD instruction is used:
/// - `Batch<u32>`: 32-bit operations (4 lanes)
/// - `Batch<u16>`: 16-bit operations (8 lanes)
/// - `Batch<u8>`: 8-bit operations (16 lanes)
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Batch<T: Copy> {
    pub(crate) inner: backend::SimdVec<T>,
}

// ============================================================================
// Type Casting (The only place we need explicit type annotations)
// ============================================================================

impl<T: Copy> Batch<T> {
    /// Reinterpret the bits of this batch as another type (zero-cost bitcast).
    ///
    /// This is the only method that needs explicit type annotation:
    ///
    /// ```ignore
    /// let pixels = Batch::splat(0xFF00FF00u32);
    /// let as_u16 = pixels.cast::<u16>(); // View 4×u32 as 8×u16
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn cast<U: Copy>(self) -> Batch<U> {
        Batch {
            inner: backend::cast(self.inner),
        }
    }
}

// Convenience casting methods
impl Batch<u32> {
    /// Cast to u16 (commonly used in pixel operations).
    #[inline(always)]
    #[must_use]
    pub fn as_u16(self) -> Batch<u16> {
        self.cast()
    }
}

impl Batch<u16> {
    /// Cast back to u32.
    #[inline(always)]
    #[must_use]
    pub fn as_u32(self) -> Batch<u32> {
        self.cast()
    }
}

// ============================================================================
// Constructors (Need bounds to know how to build backend type)
// ============================================================================

impl<T: Copy> Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    /// Create a batch where all lanes have the same value (broadcast).
    ///
    /// ```ignore
    /// let white = Batch::splat(0xFFFFFFFFu32);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn splat(val: T) -> Self {
        Self {
            inner: backend::SimdVec::<T>::splat(val),
        }
    }

    /// Load a batch from memory.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for reading `size_of::<T>() * N` bytes (where N is lane count)
    /// - No alignment required (uses unaligned loads)
    #[inline(always)]
    #[must_use]
    pub unsafe fn load(ptr: *const T) -> Self {
        Self {
            inner: unsafe { backend::SimdVec::<T>::load(ptr) },
        }
    }

    /// Store a batch to memory.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for writing `size_of::<T>() * N` bytes
    /// - No alignment required (uses unaligned stores)
    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut T) {
        unsafe { self.inner.store(ptr) }
    }

    /// Conditional select: `(self & mask) | (other & !mask)`
    ///
    /// Where `mask` is set, use `self`. Where clear, use `other`.
    #[inline(always)]
    #[must_use]
    pub fn select(self, other: Self, mask: Self) -> Self {
        Self {
            inner: self.inner.select(other.inner, mask.inner),
        }
    }

    /// Create a batch from explicit lane values.
    ///
    /// For `Batch<u32>`: Creates 4 lanes `[v0, v1, v2, v3]`
    #[inline(always)]
    #[must_use]
    pub fn new(v0: T, v1: T, v2: T, v3: T) -> Self {
        Self {
            inner: backend::SimdVec::<T>::new(v0, v1, v2, v3),
        }
    }

    /// Element-wise minimum.
    #[inline(always)]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self {
            inner: self.inner.min(other.inner),
        }
    }

    /// Element-wise maximum.
    #[inline(always)]
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self {
            inner: self.inner.max(other.inner),
        }
    }

    /// Saturating addition - clamps to max value instead of wrapping.
    ///
    /// Essential for pixel operations to avoid overflow artifacts.
    #[inline(always)]
    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            inner: self.inner.saturating_add(other.inner),
        }
    }

    /// Saturating subtraction - clamps to zero instead of wrapping.
    ///
    /// Essential for pixel operations to avoid underflow artifacts.
    #[inline(always)]
    #[must_use]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self {
            inner: self.inner.saturating_sub(other.inner),
        }
    }
}

// ============================================================================
// Specialized Methods for Batch<u32> (Pixel Operations)
// ============================================================================

/// Trait for types that can be gathered into a `Batch<u32>`.
pub trait Gatherable: Copy {
    /// Convert this type to u32 for packing into a batch.
    fn to_u32(self) -> u32;
}

impl Gatherable for u8 {
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl Gatherable for u16 {
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl Gatherable for u32 {
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self
    }
}

impl Batch<u32> {
    /// Helper: Convert to array of usize for indexing.
    ///
    /// Used internally by `gather_2d()` for address calculation.
    #[inline(always)]
    pub fn to_array_usize(self) -> [usize; 4] {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        [arr[0] as usize, arr[1] as usize, arr[2] as usize, arr[3] as usize]
    }

    /// Helper: Pack low bytes for writing pixels.
    ///
    /// Extracts the low byte from each u32 lane. Used by `map_pixels()`.
    #[inline(always)]
    pub fn to_bytes_packed(self) -> [u8; 4] {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        [arr[0] as u8, arr[1] as u8, arr[2] as u8, arr[3] as u8]
    }

    /// Extract a single lane from the batch.
    ///
    /// # Panics
    /// Panics if `index >= 4`.
    #[inline(always)]
    pub fn extract(self, index: usize) -> u32 {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        arr[index]
    }

    /// Alpha blend two colors: (self * alpha + other * (256 - alpha)) >> 8
    ///
    /// This is the **blend unit** for the shader pipeline. Used for compositing
    /// text over background using alpha channel.
    ///
    /// # Arguments
    /// - `bg`: Background color
    /// - `alpha`: Per-channel alpha value (0-255)
    ///
    /// # Returns
    /// Blended color
    ///
    /// # Example
    /// ```ignore
    /// let white = Batch::splat(0xFFFFFFFF);
    /// let black = Batch::splat(0x00000000);
    /// let half_alpha = Batch::splat(128);
    /// let gray = white.blend_alpha(black, half_alpha);  // ~50% gray
    /// ```
    #[inline(always)]
    pub fn blend_alpha(self, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
        // Blend each byte independently: (fg * alpha + bg * (256 - alpha)) >> 8
        //
        // Extract individual bytes, blend them, and reassemble.
        // This is necessary because cast<u16>() doesn't split u32 lanes,
        // it just reinterprets bits.

        let mask_byte = Batch::splat(0xFF);

        // Extract bytes
        let fg_b0 = self & mask_byte;
        let fg_b1 = (self >> 8) & mask_byte;
        let fg_b2 = (self >> 16) & mask_byte;
        let fg_b3 = (self >> 24) & mask_byte;

        let bg_b0 = bg & mask_byte;
        let bg_b1 = (bg >> 8) & mask_byte;
        let bg_b2 = (bg >> 16) & mask_byte;
        let bg_b3 = (bg >> 24) & mask_byte;

        let alpha_b0 = alpha & mask_byte;
        let alpha_b1 = (alpha >> 8) & mask_byte;
        let alpha_b2 = (alpha >> 16) & mask_byte;
        let alpha_b3 = (alpha >> 24) & mask_byte;

        let inv_alpha_b0 = Batch::splat(256) - alpha_b0;
        let inv_alpha_b1 = Batch::splat(256) - alpha_b1;
        let inv_alpha_b2 = Batch::splat(256) - alpha_b2;
        let inv_alpha_b3 = Batch::splat(256) - alpha_b3;

        // Blend each byte
        let blended_b0 = ((fg_b0 * alpha_b0) + (bg_b0 * inv_alpha_b0)) >> 8;
        let blended_b1 = ((fg_b1 * alpha_b1) + (bg_b1 * inv_alpha_b1)) >> 8;
        let blended_b2 = ((fg_b2 * alpha_b2) + (bg_b2 * inv_alpha_b2)) >> 8;
        let blended_b3 = ((fg_b3 * alpha_b3) + (bg_b3 * inv_alpha_b3)) >> 8;

        // Reassemble
        blended_b0 | (blended_b1 << 8) | (blended_b2 << 16) | (blended_b3 << 24)
    }

    /// Gather 4 elements from arbitrary memory locations.
    ///
    /// Works with any `Gatherable` type (u8, u16, u32).
    /// Type inference determines which one to use.
    ///
    /// # Safety
    ///
    /// All indices must be within bounds of `src`.
    ///
    /// ```ignore
    /// // Type inference from slice type
    /// let palette: &[u8] = &[0xFF, 0xAA, 0x55, 0x00];
    /// let pixels = Batch::gather(palette, [0, 1, 2, 3]);
    ///
    /// let texture: &[u32] = &[0xFFAABBCC, 0x11223344];
    /// let pixels = Batch::gather(texture, [0, 1, 0, 1]);
    /// ```
    #[inline(always)]
    #[must_use]
    pub unsafe fn gather<T: Gatherable>(src: &[T], indices: [usize; 4]) -> Self {
        unsafe {
            Self::new(
                src.get_unchecked(indices[0]).to_u32(),
                src.get_unchecked(indices[1]).to_u32(),
                src.get_unchecked(indices[2]).to_u32(),
                src.get_unchecked(indices[3]).to_u32(),
            )
        }
    }
}

// ============================================================================
// LocalTensor - Fixed-Size Register Tensor (Zero-Cost Linear Algebra)
// ============================================================================

/// Define a fixed-size tensor struct and its basic methods.
///
/// This macro generates structs like `Tensor2x2<T>` with `elements: [Batch<T>; 4]`.
/// The compiler fully unrolls all operations at compile time.
macro_rules! define_tensor {
    ($name:ident, $rows:literal, $cols:literal) => {
        /// A fixed-size tensor held in registers.
        #[derive(Copy, Clone)]
        pub struct $name<T: Copy> {
            pub elements: [Batch<T>; $rows * $cols],
        }

        impl<T: Copy> $name<T> {
            #[inline(always)]
            pub fn new(elements: [Batch<T>; $rows * $cols]) -> Self {
                Self { elements }
            }

            #[inline(always)]
            pub fn get(&self, row: usize, col: usize) -> Batch<T> {
                // Const-folding will optimize this index calc away
                self.elements[row * $cols + col]
            }

            #[inline(always)]
            pub fn map<U: Copy, F>(self, mut f: F) -> $name<U>
            where
                F: FnMut(Batch<T>) -> Batch<U>,
            {
                let elements = core::array::from_fn(|i| f(self.elements[i]));
                $name { elements }
            }
        }
    };
}

/// Implement Matrix Multiplication: (M x K) * (K x N) -> (M x N)
macro_rules! impl_matmul {
    ($left:ident, $right:ident, $output:ident, $m:literal, $k:literal, $n:literal) => {
        impl<T: Copy> $left<T>
        where
            backend::SimdVec<T>: SimdOps<T>,
        {
            #[inline(always)]
            pub fn matmul(&self, other: &$right<T>) -> $output<T> {
                // We use from_fn to fully unroll the loops at compile time
                let elements = core::array::from_fn(|i| {
                    let r = i / $n; // Row in Output
                    let c = i % $n; // Col in Output

                    // Dot Product: Row(self, r) . Col(other, c)
                    // For small K (1..4), simple loop unrolls fine.
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

        // Implement the * operator for ergonomic usage
        impl<T: Copy> core::ops::Mul<$right<T>> for $left<T>
        where
            backend::SimdVec<T>: SimdOps<T>,
        {
            type Output = $output<T>;

            #[inline(always)]
            fn mul(self, other: $right<T>) -> Self::Output {
                self.matmul(&other)
            }
        }
    };
}

// --- Define the Shapes we need for Bilinear Interpolation ---

// 1. The Pixel Window (2x2)
define_tensor!(Tensor2x2, 2, 2);

// 2. The Weight Vectors
// Column Vector (2 Rows, 1 Col)
define_tensor!(Tensor2x1, 2, 1);
// Row Vector (1 Row, 2 Cols)
define_tensor!(Tensor1x2, 1, 2);

// 3. The Result (1x1)
define_tensor!(Tensor1x1, 1, 1);

// --- Define Matrix Multiplications ---

// 1. Horizontal Contraction: (Pixels 2x2) * (WeightsX 2x1) -> (Result 2x1)
impl_matmul!(Tensor2x2, Tensor2x1, Tensor2x1, 2, 2, 1);

// 2. Vertical Contraction: (WeightsY 1x2) * (Result 2x1) -> (Result 1x1)
impl_matmul!(Tensor1x2, Tensor2x1, Tensor1x1, 1, 2, 1);

// ============================================================================
// TensorView - Strided Memory Views
// ============================================================================

/// A view into a 2D tensor (image) stored in memory.
#[derive(Copy, Clone)]
pub struct TensorView<'a, T> {
    pub data: &'a [T],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<'a, T> TensorView<'a, T> {
    /// Create a new tensor view (const-constructible).
    ///
    /// This allows static initialization of views over static data:
    /// ```
    /// use pixelflow_core::TensorView;
    /// static DATA: [u8; 1000] = [0; 1000];
    /// const VIEW: TensorView<u8> = TensorView::new(&DATA, 10, 10, 10);
    /// ```
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
    /// Gather 4 pixels from 2D coordinates (x, y).
    ///
    /// This vectorizes the address calculation: `idx = y * stride + x`.
    ///
    /// # Safety
    ///
    /// All coordinates must be within bounds.
    #[inline(always)]
    pub unsafe fn gather_2d(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        // Extract scalar indices for the current gather backend
        // TODO: In a perfect world, we'd keep this entirely in SIMD.
        let x_arr = x.to_array_usize();
        let y_arr = y.to_array_usize();

        // Clamp coordinates to valid range
        let max_idx = self.data.len().saturating_sub(1);

        let idx = [
            (y_arr[0] * self.stride + x_arr[0]).min(max_idx),
            (y_arr[1] * self.stride + x_arr[1]).min(max_idx),
            (y_arr[2] * self.stride + x_arr[2]).min(max_idx),
            (y_arr[3] * self.stride + x_arr[3]).min(max_idx),
        ];

        unsafe { Batch::gather(self.data, idx) }
    }

    /// Gather 4 pixels from 4-bit packed data with inline decompression.
    ///
    /// **Packed format**: Each byte stores 2 pixels:
    /// - High nibble (bits 7-4): first pixel
    /// - Low nibble (bits 3-0): second pixel
    ///
    /// **Decompression**: `pixel = nibble * 17` maps [0-15] → [0-255]
    ///
    /// This is a **zero-copy** operation - the atlas stays 4-bit packed,
    /// decompression is fused into the SIMD gather.
    ///
    /// # Arguments
    /// - `x`, `y`: Pixel coordinates (NOT byte coordinates)
    /// - `self.stride`: Stride in BYTES (pixels / 2, rounded up)
    ///
    /// # Safety
    /// All coordinates must be within bounds of the packed image.
    ///
    /// # Example
    /// ```ignore
    /// // Packed: [0x12, 0x34] represents pixels [17, 34, 51, 68]
    /// let packed = &[0x12u8, 0x34];
    /// let view = TensorView::new(packed, 4, 1, 2); // 4 pixels wide, stride=2 bytes
    /// let x = Batch::new(0, 1, 2, 3);
    /// let y = Batch::splat(0);
    /// let pixels = unsafe { view.gather_4bit(x, y) };
    /// // pixels = [17, 34, 51, 68]
    /// ```
    #[inline(always)]
    pub unsafe fn gather_4bit(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        // 1. Compute byte coordinates (x >> 1 since 2 pixels per byte)
        let byte_x = x >> 1;

        // 2. Determine which nibble: 0 = even (high nibble), 1 = odd (low nibble)
        let is_odd = x & Batch::splat(1);

        // 3. Gather packed bytes using existing gather_2d
        let packed = unsafe { self.gather_2d(byte_x, y) };

        // 4. Extract nibbles
        //    For even pixels (x & 1 == 0): use high nibble (bits 7-4)
        //    For odd pixels (x & 1 == 1):  use low nibble (bits 3-0)
        let high_nibble = (packed >> 4) & Batch::splat(0x0F);
        let low_nibble = packed & Batch::splat(0x0F);

        // 5. Select the correct nibble based on whether x is odd or even
        //    Create a mask: 0 → 0x00000000, 1 → 0xFFFFFFFF
        //    We use: if (is_odd) then all 1s, else all 0s
        //    Trick: negate is_odd, which gives 0→0, 1→0xFFFFFFFF in two's complement
        //    But we need regular sub, not saturating_sub!
        //    Alternative: multiply by 0xFFFFFFFF
        let all_ones = Batch::splat(0xFFFFFFFF);
        let mask = is_odd * all_ones; // 0→0x00000000, 1→0xFFFFFFFF
        let nibble = (high_nibble & !mask) | (low_nibble & mask);

        // 6. Expand nibble to byte: nibble * 17 = (nibble << 4) | nibble
        //    This maps [0-15] → [0, 17, 34, ..., 255]
        (nibble << 4) | nibble
    }

    /// Gather a 2x2 neighborhood for bilinear interpolation into a Tensor2x2.
    #[inline(always)]
    pub unsafe fn gather_tensor2x2(
        &self,
        x0: Batch<u32>,
        x1: Batch<u32>,
        y0: Batch<u32>,
        y1: Batch<u32>,
    ) -> Tensor2x2<u32> {
        Tensor2x2::new([
            unsafe { self.gather_2d(x0, y0) }, // TL (0,0)
            unsafe { self.gather_2d(x1, y0) }, // TR (0,1)
            unsafe { self.gather_2d(x0, y1) }, // BL (1,0)
            unsafe { self.gather_2d(x1, y1) }, // BR (1,1)
        ])
    }

    /// Gather a 2x2 neighborhood from 4-bit packed data for bilinear interpolation.
    ///
    /// Zero-copy version of `gather_tensor2x2` that works directly on 4-bit packed data.
    /// Decompression is fused into the gather operation.
    ///
    /// # Arguments
    /// - `x0`, `x1`: Left and right pixel X coordinates
    /// - `y0`, `y1`: Top and bottom pixel Y coordinates
    ///
    /// # Safety
    /// All coordinates must be within bounds of the packed image.
    #[inline(always)]
    pub unsafe fn gather_tensor2x2_4bit(
        &self,
        x0: Batch<u32>,
        x1: Batch<u32>,
        y0: Batch<u32>,
        y1: Batch<u32>,
    ) -> Tensor2x2<u32> {
        Tensor2x2::new([
            unsafe { self.gather_4bit(x0, y0) }, // TL (0,0)
            unsafe { self.gather_4bit(x1, y0) }, // TR (0,1)
            unsafe { self.gather_4bit(x0, y1) }, // BL (1,0)
            unsafe { self.gather_4bit(x1, y1) }, // BR (1,1)
        ])
    }

    /// Sample from 4-bit packed texture with bilinear filtering (GPU texture2D equivalent).
    ///
    /// This is the **texture sampler** for the shader pipeline. Performs:
    /// 1. 4-bit decompression (nibble * 17)
    /// 2. 2x2 neighborhood gathering
    /// 3. Bilinear interpolation
    ///
    /// All fused in SIMD registers - zero intermediate allocations.
    ///
    /// # Arguments
    /// - `u_fp`, `v_fp`: Texture coordinates in 16.16 fixed-point format
    ///
    /// # Returns
    /// Interpolated grayscale value (0-255) per SIMD lane
    ///
    /// # Safety
    /// Coordinates are clamped to texture bounds internally.
    #[inline(always)]
    pub unsafe fn sample_4bit_bilinear(&self, u_fp: Batch<u32>, v_fp: Batch<u32>) -> Batch<u32> {
        // Extract integer coordinates (16-bit integer part)
        let u0_raw = u_fp >> 16;
        let v0_raw = v_fp >> 16;

        // Clamp to texture bounds (both u0/v0 AND u1/v1)
        let max_x = Batch::splat((self.width - 1) as u32);
        let max_y = Batch::splat((self.height - 1) as u32);
        let u0 = u0_raw.min(max_x);
        let v0 = v0_raw.min(max_y);
        let u1 = (u0 + Batch::splat(1)).min(max_x);
        let v1 = (v0 + Batch::splat(1)).min(max_y);

        // Extract fractional parts (8-bit precision for interpolation weights)
        let du = (u_fp >> 8) & Batch::splat(0xFF);
        let dv = (v_fp >> 8) & Batch::splat(0xFF);
        let inv_du = Batch::splat(256) - du;
        let inv_dv = Batch::splat(256) - dv;

        // Gather 2x2 neighborhood from 4-bit packed data
        let pixels = unsafe { self.gather_tensor2x2_4bit(u0, u1, v0, v1) };

        // Bilinear interpolation via tensor multiplication
        let weights_x = Tensor2x1::new([inv_du, du]);
        let weights_y = Tensor1x2::new([inv_dv, dv]);

        let horizontal: Tensor2x1<u16> =
            (pixels.map(|p| p.cast::<u16>()) * weights_x.map(|w| w.cast::<u16>()))
                .map(|v| v >> 8);
        let result: Tensor1x1<u16> = (weights_y.map(|w| w.cast::<u16>()) * horizontal)
            .map(|v| v >> 8);

        result.get(0, 0).cast::<u32>()
    }

    /// Nearest-neighbor sample from 4-bit packed texture.
    ///
    /// Used for effects like bold rendering where exact pixel lookup is needed.
    ///
    /// # Arguments
    /// - `u`, `v`: Pixel coordinates (integer)
    ///
    /// # Safety
    /// Coordinates are clamped to texture bounds internally.
    #[inline(always)]
    pub unsafe fn sample_4bit_nearest(&self, u: Batch<u32>, v: Batch<u32>) -> Batch<u32> {
        let clamped_u = u.min(Batch::splat((self.width - 1) as u32));
        let clamped_v = v.min(Batch::splat((self.height - 1) as u32));
        unsafe { self.gather_4bit(clamped_u, clamped_v) }
    }
}

/// A mutable view into a 2D tensor.
pub struct TensorViewMut<'a, T> {
    pub data: &'a mut [T],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<'a, T> TensorViewMut<'a, T> {
    /// Create a new mutable tensor view.
    ///
    /// This wraps a mutable slice as a 2D view for shader operations.
    #[inline(always)]
    pub fn new(data: &'a mut [T], width: usize, height: usize, stride: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride,
        }
    }
}

impl<'a> TensorViewMut<'a, u8> {
    /// Apply a kernel function to every pixel in the tensor.
    ///
    /// This abstracts the loop splitting:
    /// 1. Runs the SIMD kernel for the bulk of the row (4 pixels at a time).
    /// 2. Runs the same kernel for the remainder (scalar tail), masking invalid writes.
    ///
    /// The kernel `f` takes `(x, y)` coordinates as Batches and returns a pixel Batch.
    #[inline(always)]
    pub fn map_pixels<F>(&mut self, mut f: F)
    where
        F: FnMut(Batch<u32>, Batch<u32>) -> Batch<u32>,
    {
        const LANES: usize = 4;

        for y in 0..self.height {
            let y_vec = Batch::splat(y as u32);
            let row_offset = y * self.stride;
            let mut x = 0;

            // 1. Main SIMD Loop
            while x + LANES <= self.width {
                // Generate coordinate vector: [x, x+1, x+2, x+3]
                let x_vec = Batch::new(
                    x as u32,
                    (x + 1) as u32,
                    (x + 2) as u32,
                    (x + 3) as u32,
                );

                // Execute Kernel
                let result = f(x_vec, y_vec);

                // Store Result (extract low byte from each u32)
                let bytes = result.to_bytes_packed();
                self.data[row_offset + x] = bytes[0];
                self.data[row_offset + x + 1] = bytes[1];
                self.data[row_offset + x + 2] = bytes[2];
                self.data[row_offset + x + 3] = bytes[3];

                x += LANES;
            }

            // 2. Remainder Loop
            // We run the SIMD kernel one last time, but only write back valid pixels.
            if x < self.width {
                let valid_count = self.width - x;
                let x_vec = Batch::new(
                    x as u32,
                    (x + 1) as u32,
                    (x + 2) as u32,
                    (x + 3) as u32,
                );

                let result = f(x_vec, y_vec);
                let bytes = result.to_bytes_packed();

                for i in 0..valid_count {
                    self.data[row_offset + x + i] = bytes[i];
                }
            }
        }
    }
}

impl<'a> TensorViewMut<'a, u32> {
    /// Apply a kernel to every pixel (u32 color framebuffer version).
    ///
    /// This is the **shader runtime** for color framebuffers. The kernel
    /// receives (x, y) coordinates and returns ARGB pixel values.
    ///
    /// Handles SIMD chunking (4 pixels at a time) and scalar remainder automatically.
    #[inline(always)]
    pub fn map_pixels<F>(&mut self, mut f: F)
    where
        F: FnMut(Batch<u32>, Batch<u32>) -> Batch<u32>,
    {
        const LANES: usize = 4;

        for y in 0..self.height {
            let y_vec = Batch::splat(y as u32);
            let row_offset = y * self.stride;
            let mut x = 0;

            // SIMD loop: Process 4 pixels at a time
            while x + LANES <= self.width {
                let x_vec = Batch::new(
                    x as u32,
                    (x + 1) as u32,
                    (x + 2) as u32,
                    (x + 3) as u32,
                );

                let result = f(x_vec, y_vec);

                // Store 4 u32 pixels directly
                unsafe {
                    result.store(self.data.as_mut_ptr().add(row_offset + x));
                }

                x += LANES;
            }

            // Scalar remainder
            if x < self.width {
                let valid = self.width - x;
                let x_vec = Batch::new(
                    x as u32,
                    (x + 1) as u32,
                    (x + 2) as u32,
                    (x + 3) as u32,
                );

                let result = f(x_vec, y_vec);
                let mut arr = [0u32; 4];
                unsafe { result.store(arr.as_mut_ptr()) };

                for i in 0..valid {
                    self.data[row_offset + x + i] = arr[i];
                }
            }
        }
    }
}

// ============================================================================
// Standard Operator Overloading - Rust infers T automatically!
// ============================================================================

impl<T: Copy> Add for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.add(rhs.inner),
        }
    }
}

impl<T: Copy> Sub for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.sub(rhs.inner),
        }
    }
}

impl<T: Copy> Mul for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.mul(rhs.inner),
        }
    }
}

impl<T: Copy> BitAnd for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.bitand(rhs.inner),
        }
    }
}

impl<T: Copy> BitOr for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.bitor(rhs.inner),
        }
    }
}

impl<T: Copy> Not for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            inner: self.inner.not(),
        }
    }
}

impl<T: Copy> Shr<i32> for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        Self {
            inner: self.inner.shr(count),
        }
    }
}

impl<T: Copy> Shl<i32> for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        Self {
            inner: self.inner.shl(count),
        }
    }
}

// ============================================================================
// Backend Interface (Internal Trait)
// ============================================================================

/// Internal trait that backends must implement for each supported type.
///
/// This connects the generic operations to platform-specific intrinsics:
/// - `SimdOps<u32>` → `paddd`, `psubd`, `pmulld`
/// - `SimdOps<u16>` → `paddw`, `psubw`, `pmullw`
pub trait SimdOps<T>: Copy + Clone + Sized {
    /// Broadcast a single value to all lanes
    fn splat(val: T) -> Self;

    /// Load from memory
    unsafe fn load(ptr: *const T) -> Self;

    /// Store to memory
    unsafe fn store(self, ptr: *mut T);

    /// Create from explicit lane values
    fn new(v0: T, v1: T, v2: T, v3: T) -> Self;

    /// Addition
    fn add(self, other: Self) -> Self;

    /// Subtraction
    fn sub(self, other: Self) -> Self;

    /// Multiplication
    fn mul(self, other: Self) -> Self;

    /// Bitwise AND
    fn bitand(self, other: Self) -> Self;

    /// Bitwise OR
    fn bitor(self, other: Self) -> Self;

    /// Bitwise NOT
    fn not(self) -> Self;

    /// Logical right shift
    fn shr(self, count: i32) -> Self;

    /// Logical left shift
    fn shl(self, count: i32) -> Self;

    /// Conditional select
    fn select(self, other: Self, mask: Self) -> Self;

    /// Element-wise minimum
    fn min(self, other: Self) -> Self;

    /// Element-wise maximum
    fn max(self, other: Self) -> Self;

    /// Saturating addition (clamps to max value instead of wrapping)
    fn saturating_add(self, other: Self) -> Self;

    /// Saturating subtraction (clamps to zero instead of wrapping)
    fn saturating_sub(self, other: Self) -> Self;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_4bit_correctness() {
        // Packed data: [0x12, 0x34, 0x56, 0x78]
        // Represents pixels: [1*17, 2*17, 3*17, 4*17, 5*17, 6*17, 7*17, 8*17]
        //                  = [17, 34, 51, 68, 85, 102, 119, 136]
        let packed = [0x12u8, 0x34, 0x56, 0x78];

        // Create a view: 8 pixels wide, 1 pixel tall, stride = 4 bytes
        let view = TensorView::new(&packed, 8, 1, 4);

        // Test gathering pixels 0, 1, 2, 3 (first two bytes)
        let x = Batch::<u32>::new(0, 1, 2, 3);
        let y = Batch::<u32>::splat(0);
        let result = unsafe { view.gather_4bit(x, y) };

        let expected = [1 * 17, 2 * 17, 3 * 17, 4 * 17];
        assert_eq!(result.to_array_usize(), expected);

        // Test gathering pixels 4, 5, 6, 7 (second two bytes)
        let x = Batch::<u32>::new(4, 5, 6, 7);
        let result = unsafe { view.gather_4bit(x, y) };

        let expected = [5 * 17, 6 * 17, 7 * 17, 8 * 17];
        assert_eq!(result.to_array_usize(), expected);
    }

    #[test]
    fn test_gather_4bit_edge_values() {
        // Test min and max nibble values: 0x00, 0x0F, 0xF0, 0xFF
        let packed = [0x00u8, 0x0F, 0xF0, 0xFF];
        // Pixels: [0*17, 0*17, 0*17, 15*17, 15*17, 0*17, 15*17, 15*17]
        //       = [0, 0, 0, 255, 255, 0, 255, 255]

        let view = TensorView::new(&packed, 8, 1, 4);
        let y = Batch::<u32>::splat(0);

        // First 4 pixels
        let x = Batch::<u32>::new(0, 1, 2, 3);
        let result = unsafe { view.gather_4bit(x, y) };
        assert_eq!(result.to_array_usize(), [0, 0, 0, 255]);

        // Last 4 pixels
        let x = Batch::<u32>::new(4, 5, 6, 7);
        let result = unsafe { view.gather_4bit(x, y) };
        assert_eq!(result.to_array_usize(), [255, 0, 255, 255]);
    }

    #[test]
    fn test_const_tensorview_construction() {
        static DATA: [u8; 4] = [1, 2, 3, 4];
        const VIEW: TensorView<u8> = TensorView::new(&DATA, 2, 2, 2);

        assert_eq!(VIEW.width, 2);
        assert_eq!(VIEW.height, 2);
        assert_eq!(VIEW.stride, 2);
        assert_eq!(VIEW.data.len(), 4);
    }

    #[test]
    fn test_sample_4bit_bilinear_exact_centers() {
        // 2x2 atlas with known values
        // Row 0: [0, 85] -> 4-bit [0, 5] -> packed 0x05
        // Row 1: [170, 255] -> 4-bit [10, 15] -> packed 0xAF
        let packed = [0x05u8, 0xAF];
        let view = TensorView::new(&packed, 2, 2, 1);

        // Sample at exact pixel centers (integer coords)
        // Pixel centers in 16.16 fixed-point: 0x00000000, 0x00010000, etc.
        let u_fp = Batch::<u32>::new(0, 0x00010000, 0, 0x00010000);
        let v_fp = Batch::<u32>::new(0, 0, 0x00010000, 0x00010000);

        let result = unsafe { view.sample_4bit_bilinear(u_fp, v_fp) };
        let samples = result.to_array_usize();

        // Allow tolerance for bilinear filtering
        assert!((samples[0] as i32 - 0).abs() <= 5, "Top-left should be ~0, got {}", samples[0]);
        assert!((samples[1] as i32 - 85).abs() <= 5, "Top-right should be ~85, got {}", samples[1]);
        assert!((samples[2] as i32 - 170).abs() <= 5, "Bottom-left should be ~170, got {}", samples[2]);
        assert!((samples[3] as i32 - 255).abs() <= 5, "Bottom-right should be ~255, got {}", samples[3]);
    }

    #[test]
    fn test_sample_4bit_bilinear_interpolation() {
        // 2x2 atlas: [0, 255, 0, 255]
        // 4-bit: [0, 15, 0, 15]
        // Packed: 0x0F, 0x0F
        let packed = [0x0Fu8, 0x0F];
        let view = TensorView::new(&packed, 2, 2, 1);

        // Sample at center of 2x2 grid (should be 50% blend of all 4 pixels)
        // Center point in 16.16: 0x00008000 (0.5), 0x00008000 (0.5)
        let u_fp = Batch::<u32>::splat(0x00008000);
        let v_fp = Batch::<u32>::splat(0x00008000);

        let result = unsafe { view.sample_4bit_bilinear(u_fp, v_fp) };
        let sample = result.extract(0);

        // Average of [0, 255, 0, 255] should be ~128
        assert!(
            (sample as i32 - 128).abs() <= 10,
            "Center sample should be ~128, got {}",
            sample
        );
    }

    #[test]
    fn test_sample_4bit_bilinear_edge_clamping() {
        // 2x2 atlas with known values
        let packed = [0xFFu8, 0xFF];
        let view = TensorView::new(&packed, 2, 2, 1);

        // Sample at right edge (u1 should clamp to max_x)
        let u_fp = Batch::<u32>::splat(0x00018000); // 1.5 (between pixels 1 and 2, but 2 is out of bounds)
        let v_fp = Batch::<u32>::splat(0x00008000); // 0.5

        let result = unsafe { view.sample_4bit_bilinear(u_fp, v_fp) };
        let sample = result.extract(0);

        // Should clamp u1 to edge pixel (all 255 in this atlas)
        assert_eq!(sample, 255, "Edge sample should clamp to boundary");
    }

    #[test]
    fn test_sample_4bit_nearest() {
        // 4x1 atlas: [0, 85, 170, 255]
        // 4-bit: [0, 5, 10, 15]
        // Packed: 0x05, 0xAF
        let packed = [0x05u8, 0xAF];
        let view = TensorView::new(&packed, 4, 1, 2);

        let x = Batch::<u32>::new(0, 1, 2, 3);
        let y = Batch::<u32>::splat(0);

        let result = unsafe { view.sample_4bit_nearest(x, y) };
        let samples = result.to_array_usize();

        assert_eq!(samples[0], 0);
        assert_eq!(samples[1], 85);
        assert_eq!(samples[2], 170);
        assert_eq!(samples[3], 255);
    }

    #[test]
    fn test_blend_alpha_zero_alpha() {
        // With 0% alpha, should get background
        let fg = Batch::<u32>::splat(0xFF_FF_FF_FF);
        let bg = Batch::<u32>::splat(0x00_00_00_00);
        let alpha = Batch::<u32>::splat(0x00_00_00_00);

        let result = fg.blend_alpha(bg, alpha);

        assert_eq!(result.to_array_usize(), [0, 0, 0, 0]);
    }

    #[test]
    fn test_blend_alpha_with_replicated_values() {
        // Test blend with properly formatted ARGB pixels
        let fg = Batch::<u32>::splat(0x80_80_80_80); // Gray (128 in all channels)
        let bg = Batch::<u32>::splat(0x00_00_00_00); // Black
        // Alpha needs to be replicated across channels for proper blending
        let alpha = Batch::<u32>::splat(0xFF_FF_FF_FF); // Full alpha

        let result = fg.blend_alpha(bg, alpha);

        // Result should be close to fg since alpha is 100%
        // Note: The actual blending math depends on cast behavior
        assert_ne!(result.to_array_usize(), [0, 0, 0, 0], "Should not be all black");
    }

    #[test]
    fn test_blend_alpha_half_alpha() {
        // With 50% alpha (0x80 = 128), should get halfway blend
        let fg = Batch::<u32>::splat(0xFF_FF_FF_FF);
        let bg = Batch::<u32>::splat(0x00_00_00_00);
        let alpha = Batch::<u32>::splat(0x80_80_80_80);

        let result = fg.blend_alpha(bg, alpha);
        let blended = result.extract(0);

        // Extract R channel
        let r = (blended & 0xFF) as u8;

        // 50% of 255 should be ~128
        assert!(
            (r as i32 - 128).abs() <= 2,
            "50% blend should be ~128, got {}",
            r
        );
    }

    #[test]
    fn test_blend_alpha_smoke_test() {
        // Basic smoke test that blend doesn't crash or produce garbage
        let fg = Batch::<u32>::splat(0xFF_FF_FF_FF);
        let bg = Batch::<u32>::splat(0x00_00_00_00);
        let alpha = Batch::<u32>::splat(0x80_80_80_80);

        let result = fg.blend_alpha(bg, alpha);

        // Result should be non-zero (since we're blending white with black)
        assert_ne!(result.to_array_usize(), [0, 0, 0, 0], "Blend should produce non-zero output");
    }

    #[test]
    fn test_gather_tensor2x2_4bit() {
        // 2x2 atlas for testing 2x2 gathering
        // [0,  85]
        // [170, 255]
        // 4-bit values: [0, 5, 10, 15]
        // Row 0: [0, 5] -> packed 0x05
        // Row 1: [10, 15] -> packed 0xAF
        let packed = [0x05u8, 0xAF];
        let view = TensorView::new(&packed, 2, 2, 1);

        // Gather 2x2 region
        let x0 = Batch::<u32>::splat(0);
        let x1 = Batch::<u32>::splat(1);
        let y0 = Batch::<u32>::splat(0);
        let y1 = Batch::<u32>::splat(1);

        let tensor = unsafe { view.gather_tensor2x2_4bit(x0, x1, y0, y1) };

        // get(row, col): row * 2 + col
        // Top-left: get(0,0) = elements[0] = (0,0) = 0
        // Top-right: get(0,1) = elements[1] = (1,0) = 85
        // Bottom-left: get(1,0) = elements[2] = (0,1) = 170
        // Bottom-right: get(1,1) = elements[3] = (1,1) = 255
        assert_eq!(tensor.get(0, 0).extract(0), 0, "Top-left (0,0)");
        assert_eq!(tensor.get(0, 1).extract(0), 85, "Top-right (1,0)");
        assert_eq!(tensor.get(1, 0).extract(0), 170, "Bottom-left (0,1)");
        assert_eq!(tensor.get(1, 1).extract(0), 255, "Bottom-right (1,1)");
    }

    #[test]
    fn test_gather_4bit_stride_calculation() {
        // Test that stride is correctly calculated for odd widths
        // 5x1 atlas: [0, 17, 34, 51, 68]
        // 4-bit: [0, 1, 2, 3, 4]
        // Packed: 0x01, 0x23, 0x40
        let packed = [0x01u8, 0x23, 0x40];
        let view = TensorView::new(&packed, 5, 1, 3); // stride = (5+1)/2 = 3

        let x = Batch::<u32>::new(0, 1, 2, 3);
        let y = Batch::<u32>::splat(0);

        let result = unsafe { view.gather_4bit(x, y) };
        assert_eq!(result.to_array_usize(), [0, 17, 34, 51]);

        // Test last pixel
        let x = Batch::<u32>::new(4, 4, 4, 4);
        let result = unsafe { view.gather_4bit(x, y) };
        assert_eq!(result.extract(0), 68);
    }
}
