//! Backend trait and SIMD operations.
//!
//! This module provides the abstraction layer over platform-specific SIMD.
//! All types and traits here are private - only `Field` and `Discrete` are exposed publicly.

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
    /// Number of lanes.
    const LANES: usize;

    /// Splat a scalar across all lanes.
    fn splat(val: f32) -> Self;

    /// Create sequential values [start, start+1, start+2, ...].
    fn sequential(start: f32) -> Self;

    /// Store to a slice.
    fn store(&self, out: &mut [f32]);

    /// Check if any lane is non-zero (for early exit).
    fn any(&self) -> bool;

    /// Check if all lanes are non-zero.
    fn all(&self) -> bool;

    // Comparisons (return masks)
    fn cmp_lt(self, rhs: Self) -> Self;
    fn cmp_le(self, rhs: Self) -> Self;
    fn cmp_gt(self, rhs: Self) -> Self;
    fn cmp_ge(self, rhs: Self) -> Self;

    // Float operations
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;

    /// Conditional select: if mask bit is 1, take from if_true, else if_false.
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self;
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
    fn store(&self, out: &mut [u32]);

    /// Convert from f32 SIMD (clamp, scale by 255, truncate).
    fn from_f32_scaled<F: SimdOps>(f: F) -> Self;
}

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "x86_64")]
pub mod x86;

pub mod scalar;
