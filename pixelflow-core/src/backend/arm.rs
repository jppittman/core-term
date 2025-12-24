//! ARM NEON backend (4 lanes for f32).

use super::{Backend, SimdOps, SimdU32Ops};
use core::arch::aarch64::*;
use core::fmt::{Debug, Formatter};
use core::ops::*;

/// NEON Backend (4 lanes).
#[derive(Copy, Clone, Debug, Default)]
pub struct Neon;

impl Backend for Neon {
    const LANES: usize = 4;
    type F32 = F32x4;
    type U32 = U32x4;
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

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(vld1q_f32(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // NEON doesn't have gather - do scalar loads
        let idx = indices.to_array();
        let len = slice.len();
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let ix = (libm::floorf(idx[i]) as isize).clamp(0, len as isize - 1) as usize;
            out[i] = slice[ix];
        }
        Self::from_slice(&out)
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

// ============================================================================
// U32x4 - 4-lane u32 SIMD for packed RGBA pixels
// ============================================================================

/// 4-lane u32 SIMD vector for ARM NEON (packed RGBA pixels).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x4(uint32x4_t);

impl Default for U32x4 {
    fn default() -> Self {
        unsafe { Self(vdupq_n_u32(0)) }
    }
}

impl Debug for U32x4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        write!(f, "U32x4({:?})", arr)
    }
}

impl U32x4 {
    #[inline(always)]
    fn to_array(self) -> [u32; 4] {
        let mut arr = [0u32; 4];
        unsafe { vst1q_u32(arr.as_mut_ptr(), self.0) };
        arr
    }
}

impl SimdU32Ops for U32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        unsafe { Self(vdupq_n_u32(val)) }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        unsafe { vst1q_u32(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        // This is a placeholder - actual impl needs F to be F32x4
        // We'll handle this differently
        Self::default()
    }
}

impl BitAnd for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(vandq_u32(self.0, rhs.0)) }
    }
}

impl BitOr for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(vorrq_u32(self.0, rhs.0)) }
    }
}

impl Shl<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        unsafe {
            let shift = vdupq_n_s32(rhs as i32);
            Self(vshlq_u32(self.0, shift))
        }
    }
}

impl Shr<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        unsafe {
            let shift = vdupq_n_s32(-(rhs as i32));
            Self(vshlq_u32(self.0, shift))
        }
    }
}

impl U32x4 {
    /// Pack 4 f32 Fields (RGBA) into packed u32 pixels.
    #[inline(always)]
    pub fn pack_rgba(r: F32x4, g: F32x4, b: F32x4, a: F32x4) -> Self {
        unsafe {
            // Clamp to [0, 1] and scale to [0, 255]
            let scale = vdupq_n_f32(255.0);
            let zero = vdupq_n_f32(0.0);
            let one = vdupq_n_f32(1.0);

            let r_clamped = vminq_f32(vmaxq_f32(r.0, zero), one);
            let g_clamped = vminq_f32(vmaxq_f32(g.0, zero), one);
            let b_clamped = vminq_f32(vmaxq_f32(b.0, zero), one);
            let a_clamped = vminq_f32(vmaxq_f32(a.0, zero), one);

            let r_scaled = vmulq_f32(r_clamped, scale);
            let g_scaled = vmulq_f32(g_clamped, scale);
            let b_scaled = vmulq_f32(b_clamped, scale);
            let a_scaled = vmulq_f32(a_clamped, scale);

            // Convert to u32
            let r_u32 = vcvtq_u32_f32(r_scaled);
            let g_u32 = vcvtq_u32_f32(g_scaled);
            let b_u32 = vcvtq_u32_f32(b_scaled);
            let a_u32 = vcvtq_u32_f32(a_scaled);

            // Pack: R | (G << 8) | (B << 16) | (A << 24)
            let g_shifted = vshlq_n_u32(g_u32, 8);
            let b_shifted = vshlq_n_u32(b_u32, 16);
            let a_shifted = vshlq_n_u32(a_u32, 24);

            let packed = vorrq_u32(vorrq_u32(r_u32, g_shifted), vorrq_u32(b_shifted, a_shifted));
            Self(packed)
        }
    }
}
