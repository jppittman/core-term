//! x86_64 backend.

use super::{Backend, SimdOps};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::fmt::{Debug, Formatter};
use core::ops::*;

// ============================================================================
// SSE2 Backend
// ============================================================================

/// SSE2 Backend (4 lanes).
#[derive(Copy, Clone, Debug, Default)]
pub struct Sse2;

impl Backend for Sse2 {
    const LANES: usize = 4;
    type F32 = F32x4;
}

/// 4-lane f32 SIMD vector for SSE2.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4(__m128);

impl Default for F32x4 {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_ps()) }
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
        unsafe { _mm_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }
}

impl SimdOps for F32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { Self(_mm_set1_ps(val)) }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        unsafe {
            // _mm_set_ps args are in reverse order: e3, e2, e1, e0
            Self(_mm_set_ps(start + 3.0, start + 2.0, start + 1.0, start))
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        unsafe {
            let int_vec = _mm_castps_si128(self.0);
            let zero = _mm_setzero_si128();
            // Check if any bit is set (bitwise non-zero)
            let eq = _mm_cmpeq_epi32(int_vec, zero);
            // If any lane is non-zero, eq for that lane is 0x00000000
            // If all lanes are zero, eq is all 0xFFFFFFFF
            // movemask takes MSB of each byte.
            // If any lane is non-zero, at least one byte in eq is 0.
            // So movemask != 0xFFFF.
            _mm_movemask_epi8(eq) != 0xFFFF
        }
    }

    #[inline(always)]
    fn all(&self) -> bool {
        unsafe {
            let int_vec = _mm_castps_si128(self.0);
            let zero = _mm_setzero_si128();
            // Check if all lanes are non-zero (bitwise)
            let eq = _mm_cmpeq_epi32(int_vec, zero);
            // If all are non-zero, eq is all 0x00000000
            // movemask == 0
            _mm_movemask_epi8(eq) == 0
        }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Self {
        unsafe { Self(_mm_cmplt_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Self {
        unsafe { Self(_mm_cmple_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Self {
        unsafe { Self(_mm_cmpgt_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        unsafe { Self(_mm_cmpge_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe { Self(_mm_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            // Mask off sign bit (bit 31)
            let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
            Self(_mm_and_ps(self.0, mask))
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(_mm_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(_mm_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            // (mask & if_true) | (!mask & if_false)
            // _mm_andnot_ps(a, b) computes (!a) & b
            let t = _mm_and_ps(mask.0, if_true.0);
            let f = _mm_andnot_ps(mask.0, if_false.0);
            Self(_mm_or_ps(t, f))
        }
    }
}

// Operators for F32x4
impl Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(_mm_div_ps(self.0, rhs.0)) }
    }
}

impl BitAnd for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm_and_ps(self.0, rhs.0)) }
    }
}

impl BitOr for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm_or_ps(self.0, rhs.0)) }
    }
}

impl Not for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            // XOR with all 1s
            let all_ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
            Self(_mm_xor_ps(self.0, all_ones))
        }
    }
}

// ============================================================================
// AVX512 Backend
// ============================================================================

/// AVX512 Backend (16 lanes).
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone, Debug, Default)]
pub struct Avx512;

#[cfg(target_feature = "avx512f")]
impl Backend for Avx512 {
    const LANES: usize = 16;
    type F32 = F32x16;
}

/// 16-lane f32 SIMD vector for AVX512.
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x16(__m512);

#[cfg(target_feature = "avx512f")]
impl Default for F32x16 {
    fn default() -> Self {
        unsafe { Self(_mm512_setzero_ps()) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Debug for F32x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let arr = self.to_array();
        write!(f, "F32x16({:?})", arr)
    }
}

#[cfg(target_feature = "avx512f")]
impl F32x16 {
    #[inline(always)]
    fn to_array(self) -> [f32; 16] {
        let mut arr = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    unsafe fn from_mask(mask: __mmask16) -> Self {
        let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        let all_zeros = _mm512_setzero_ps();
        Self(_mm512_mask_blend_ps(mask, all_zeros, all_ones))
    }
}

#[cfg(target_feature = "avx512f")]
impl SimdOps for F32x16 {
    const LANES: usize = 16;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { Self(_mm512_set1_ps(val)) }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        unsafe {
            // Construct sequentially.
            let mut arr = [0.0; 16];
            for i in 0..16 {
                arr[i] = start + i as f32;
            }
            Self(_mm512_loadu_ps(arr.as_ptr()))
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [f32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        unsafe {
            let as_int = _mm512_castps_si512(self.0);
            let mask = _mm512_cmp_epi32_mask(as_int, _mm512_setzero_si512(), _MM_CMPINT_NE);
            mask != 0
        }
    }

    #[inline(always)]
    fn all(&self) -> bool {
        unsafe {
            let as_int = _mm512_castps_si512(self.0);
            let mask = _mm512_cmp_epi32_mask(as_int, _mm512_setzero_si512(), _MM_CMPINT_NE);
            mask == 0xFFFF
        }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Self {
        unsafe {
            let mask = _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LT_OQ);
            Self::from_mask(mask)
        }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Self {
        unsafe {
            let mask = _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LE_OQ);
            Self::from_mask(mask)
        }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Self {
        unsafe {
            let mask = _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GT_OQ);
            Self::from_mask(mask)
        }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Self {
        unsafe {
            let mask = _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GE_OQ);
            Self::from_mask(mask)
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe { Self(_mm512_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
            Self(_mm512_and_ps(self.0, mask))
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            // (mask & if_true) | (!mask & if_false)
            Self(_mm512_or_ps(
                _mm512_and_ps(mask.0, if_true.0),
                _mm512_andnot_ps(mask.0, if_false.0)
            ))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl Add for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_add_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Sub for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_sub_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Mul for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_mul_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Div for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_div_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl BitAnd for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_and_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl BitOr for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_or_ps(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Not for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            Self(_mm512_xor_ps(self.0, all_ones))
        }
    }
}
