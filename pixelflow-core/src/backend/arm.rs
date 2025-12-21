//! ARM NEON backend (4 lanes for f32).

use super::{Backend, SimdOps};
use core::arch::aarch64::*;
use core::fmt::{Debug, Formatter};
use core::ops::*;

/// NEON Backend (4 lanes).
#[derive(Copy, Clone, Debug, Default)]
pub struct Neon;

impl Backend for Neon {
    const LANES: usize = 4;
    type F32 = F32x4;
}

/// 4-lane f32 SIMD vector for ARM NEON.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4(float32x4_t);

impl Default for F32x4 {
    fn default() -> Self {
        unsafe { Self(vdupq_n_f32(0.0)) }
    }
}

impl Debug for F32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        write!(f, "F32x4({:?})", arr)
    }
}

impl F32x4 {
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut arr = [0.0f32; 4];
        unsafe { vst1q_f32(arr.as_mut_ptr(), self.0) };
        arr
    }
}

// ============================================================================
// SimdOps Implementation
// ============================================================================

impl SimdOps for F32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { Self(vdupq_n_f32(val)) }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        unsafe {
            let arr = [start, start + 1.0, start + 2.0, start + 3.0];
            Self(vld1q_f32(arr.as_ptr()))
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        unsafe { vst1q_f32(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        unsafe {
            // Convert to u32 mask, check if max > 0
            let as_u32: uint32x4_t = vreinterpretq_u32_f32(self.0);
            vmaxvq_u32(as_u32) != 0
        }
    }

    #[inline(always)]
    fn all(&self) -> bool {
        unsafe {
            let as_u32: uint32x4_t = vreinterpretq_u32_f32(self.0);
            vminvq_u32(as_u32) != 0
        }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Self {
        unsafe {
            let mask = vcltq_f32(self.0, rhs.0);
            Self(vreinterpretq_f32_u32(mask))
        }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Self {
        unsafe {
            let mask = vcleq_f32(self.0, rhs.0);
            Self(vreinterpretq_f32_u32(mask))
        }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Self {
        unsafe {
            let mask = vcgtq_f32(self.0, rhs.0);
            Self(vreinterpretq_f32_u32(mask))
        }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        unsafe {
            let mask = vcgeq_f32(self.0, rhs.0);
            Self(vreinterpretq_f32_u32(mask))
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe { Self(vsqrtq_f32(self.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(vminq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(vmaxq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            let mask_u32 = vreinterpretq_u32_f32(mask.0);
            let result = vbslq_f32(mask_u32, if_true.0, if_false.0);
            Self(result)
        }
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }
}

impl Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }
}

impl Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }
}

impl Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(vdivq_f32(self.0, rhs.0)) }
    }
}

impl BitAnd for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0);
            let b = vreinterpretq_u32_f32(rhs.0);
            Self(vreinterpretq_f32_u32(vandq_u32(a, b)))
        }
    }
}

impl BitOr for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0);
            let b = vreinterpretq_u32_f32(rhs.0);
            Self(vreinterpretq_f32_u32(vorrq_u32(a, b)))
        }
    }
}

impl Not for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0);
            Self(vreinterpretq_f32_u32(vmvnq_u32(a)))
        }
    }
}
