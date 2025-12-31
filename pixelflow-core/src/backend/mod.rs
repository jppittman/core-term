//! Backend trait and SIMD operations.
//!
//! This module provides the abstraction layer over platform-specific SIMD.
//! All types and traits here are private - only `Field` and `Discrete` are exposed publicly.
//!
//! ## Native Mask Types
//!
//! Each backend provides a native mask type optimized for its architecture:
//! - AVX-512: `Mask16(__mmask16)` - uses dedicated k-registers, ~0 cycles for mask ops
//! - SSE2: `Mask4(__m128)` - float-based mask (no separate mask unit)
//! - NEON: `Mask4(uint32x4_t)` - integer mask
//! - Scalar: `MaskScalar(bool)`

use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Shl, Shr, Sub};

/// A backend provides the SIMD implementation for a specific platform.
pub trait Backend: 'static + Copy + Clone + Send + Sync + Debug {
    /// Number of lanes in the SIMD vector.
    const LANES: usize;

    /// The SIMD vector type for f32.
    type F32: SimdOps;

    /// The SIMD vector type for u32 (for packed pixels).
    type U32: SimdU32Ops;
}

/// Operations on native mask types.
///
/// Masks represent boolean conditions per-lane. On AVX-512, this uses the
/// dedicated k-register file which runs on a separate execution unit,
/// effectively making mask operations free (parallel with float ALU).
pub trait MaskOps:
    Copy
    + Clone
    + Debug
    + Default
    + Send
    + Sync
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Not<Output = Self>
{
    /// Check if any lane is true (non-zero).
    /// On AVX-512, this compiles to `kortestw` on the mask unit (~0 cycles).
    fn any(self) -> bool;

    /// Check if all lanes are true (non-zero).
    /// On AVX-512, this compiles to a mask equality check (~0 cycles).
    fn all(self) -> bool;
}

/// All SIMD operations for f32.
pub trait SimdOps:
    Copy
    + Clone
    + Debug
    + Default
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Not<Output = Self>
{
    /// Native mask type for this SIMD width.
    /// On AVX-512, this is `Mask16(__mmask16)` using k-registers.
    /// On SSE2/NEON, this is a float/int vector used as a mask.
    type Mask: MaskOps;

    /// Number of lanes.
    const LANES: usize;

    /// Splat a scalar across all lanes.
    fn splat(val: f32) -> Self;

    /// Create sequential values [start, start+1, start+2, ...].
    fn sequential(start: f32) -> Self;

    /// Store to a slice.
    fn store(&self, out: &mut [f32]);

    /// Less than comparison (returns native mask).
    fn cmp_lt(self, rhs: Self) -> Self::Mask;
    /// Less than or equal comparison (returns native mask).
    fn cmp_le(self, rhs: Self) -> Self::Mask;
    /// Greater than comparison (returns native mask).
    fn cmp_gt(self, rhs: Self) -> Self::Mask;
    /// Greater than or equal comparison (returns native mask).
    fn cmp_ge(self, rhs: Self) -> Self::Mask;

    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Element-wise minimum.
    fn min(self, rhs: Self) -> Self;
    /// Element-wise maximum.
    fn max(self, rhs: Self) -> Self;

    /// Conditional select using native mask.
    /// On AVX-512, uses `vblendmps` with k-register directly.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    /// Load from a slice.
    fn from_slice(slice: &[f32]) -> Self;

    /// Gather: load from slice at indices specified by self.
    fn gather(slice: &[f32], indices: Self) -> Self;

    /// Floor (round toward negative infinity).
    fn floor(self) -> Self;

    /// Fused multiply-add: (self * b) + c
    fn mul_add(self, b: Self, c: Self) -> Self;

    /// Masked add using native mask: self + (mask ? val : 0)
    /// On AVX-512, uses masked `vaddps` with k-register.
    fn add_masked(self, val: Self, mask: Self::Mask) -> Self;

    /// Approximate reciprocal (1/x). ~12-14 bits accuracy.
    fn recip(self) -> Self;

    /// Approximate reciprocal square root (1/sqrt(x)). ~12-14 bits accuracy.
    fn rsqrt(self) -> Self;

    // =========================================================================
    // Mask Conversion (for Field API compatibility)
    // =========================================================================

    /// Convert native mask to float representation (all-1s or all-0s per lane).
    /// Used by Field to maintain its "Field as mask" API.
    fn mask_to_float(mask: Self::Mask) -> Self;

    /// Convert float representation to native mask.
    /// Used by Field to convert float masks to native for select/add_masked.
    fn float_to_mask(self) -> Self::Mask;

    // =========================================================================
    // Bit Manipulation (for transcendentals)
    // =========================================================================

    /// Splat u32 bit pattern as float (BITCAST - no conversion).
    /// The u32 value is reinterpreted as IEEE 754 float bits.
    fn from_u32_bits(bits: u32) -> Self;

    /// Shift bits right treating as u32 (BITCAST in, BITCAST out).
    /// Reinterprets float bits as u32, shifts, reinterprets result as float bits.
    fn shr_u32(self, n: u32) -> Self;

    /// Interpret bits as i32, convert to f32 (SEMANTIC conversion).
    /// Reinterprets float bits as i32, then converts that integer to f32.
    /// Example: bits 0x00000005 → i32 value 5 → f32 value 5.0
    fn i32_to_f32(self) -> Self;

    /// Base-2 logarithm.
    ///
    /// Computes log2(x) for positive finite x.
    /// Uses hardware getexp/getmant on AVX-512, bit manipulation + polynomial elsewhere.
    /// Accuracy: ~10^-7 relative error (24-bit mantissa precision).
    fn log2(self) -> Self;
}

/// SIMD operations for u32 (packed pixels).
pub trait SimdU32Ops:
    Copy
    + Clone
    + Debug
    + Default
    + Send
    + Sync
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
{
    /// Number of lanes.
    const LANES: usize;

    /// Splat a scalar across all lanes.
    fn splat(val: u32) -> Self;

    /// Store to a slice.
    ///
    /// # Internal Use Only
    ///
    /// **If you're reading this, you're trying to use the library wrong.**
    ///
    /// This method is part of the internal SIMD backend and should remain
    /// private to the crate. Users should not directly extract values from
    /// SIMD types - the library is designed around declarative manifold
    /// composition, not imperative value extraction.
    ///
    /// **The function you're looking for is [`materialize`](crate::materialize) in lib.rs.**
    fn store(&self, out: &mut [u32]);

    /// Convert from f32 SIMD (clamp, scale by 255, truncate).
    fn from_f32_scaled<F: SimdOps>(f: F) -> Self;
}

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "x86_64")]
pub mod x86;

pub mod scalar;

pub mod fastmath;
