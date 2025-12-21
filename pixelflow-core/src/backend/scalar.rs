//! Scalar fallback backend for non-SIMD platforms.

use super::{Backend, SimdOps};
use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Sub};

/// Scalar fallback backend (1 lane - no SIMD).
#[derive(Copy, Clone, Debug, Default)]
pub struct Scalar;

impl Backend for Scalar {
    const LANES: usize = 1;
    type F32 = ScalarF32;
}

/// Scalar f32 wrapper that implements all required ops.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct ScalarF32(f32);

// ============================================================================
// SimdOps for ScalarF32
// ============================================================================

impl SimdOps for ScalarF32 {
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(val)
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self(start)
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        out[0] = self.0;
    }

    #[inline(always)]
    fn any(&self) -> bool {
        self.0.to_bits() != 0
    }

    #[inline(always)]
    fn all(&self) -> bool {
        self.0.to_bits() != 0
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Self {
        Self(if self.0 < rhs.0 {
            f32::from_bits(!0u32)
        } else {
            0.0
        })
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Self {
        Self(if self.0 <= rhs.0 {
            f32::from_bits(!0u32)
        } else {
            0.0
        })
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Self {
        Self(if self.0 > rhs.0 {
            f32::from_bits(!0u32)
        } else {
            0.0
        })
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        Self(if self.0 >= rhs.0 {
            f32::from_bits(!0u32)
        } else {
            0.0
        })
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Self(libm::sqrtf(self.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(libm::fabsf(self.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(if self.0 < rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(if self.0 > rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(if mask.0.to_bits() != 0 {
            if_true.0
        } else {
            if_false.0
        })
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl Add for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl BitAnd for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(f32::from_bits(self.0.to_bits() & rhs.0.to_bits()))
    }
}

impl BitOr for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(f32::from_bits(self.0.to_bits() | rhs.0.to_bits()))
    }
}

impl Not for ScalarF32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(f32::from_bits(!self.0.to_bits()))
    }
}
