//! x86_64 backend.

use super::{Backend, SimdOps, SimdU32Ops};
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
    type U32 = U32x4;
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

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(_mm_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // SSE2 doesn't have gather - do scalar loads
        let idx = indices.to_array();
        let len = slice.len();
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let ix = (libm::floorf(idx[i]) as isize).clamp(0, len as isize - 1) as usize;
            out[i] = slice[ix];
        }
        Self::from_slice(&out)
    }

    #[inline(always)]
    fn floor(self) -> Self {
        unsafe {
            // SSE2 floor emulation:
            // 1. Truncate toward zero
            let trunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(self.0));
            // 2. For negative non-integers, truncation rounds toward zero (wrong direction)
            //    Need to subtract 1 where self < trunc
            let mask = _mm_cmplt_ps(self.0, trunc);
            let one = _mm_set1_ps(1.0);
            let correction = _mm_and_ps(mask, one);
            Self(_mm_sub_ps(trunc, correction))
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
// U32x4 - 4-lane u32 SIMD for packed RGBA pixels (SSE2)
// ============================================================================

/// 4-lane u32 SIMD vector for SSE2 (packed RGBA pixels).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x4(__m128i);

impl Default for U32x4 {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_si128()) }
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
        unsafe { _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, self.0) };
        arr
    }
}

impl SimdU32Ops for U32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        unsafe { Self(_mm_set1_epi32(val as i32)) }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        // Placeholder - actual packing is done via pack_rgba
        Self::default()
    }
}

impl BitAnd for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, rhs.0)) }
    }
}

impl BitOr for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, rhs.0)) }
    }
}

impl Shl<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        unsafe {
            // _mm_sll_epi32 takes shift count in lower 64-bits of __m128i
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm_sll_epi32(self.0, shift))
        }
    }
}

impl Shr<u32> for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm_srl_epi32(self.0, shift))
        }
    }
}

impl U32x4 {
    /// Pack 4 f32 Fields (RGBA) into packed u32 pixels.
    #[inline(always)]
    pub fn pack_rgba(r: F32x4, g: F32x4, b: F32x4, a: F32x4) -> Self {
        unsafe {
            // Clamp to [0, 1] and scale to [0, 255]
            let scale = _mm_set1_ps(255.0);
            let zero = _mm_setzero_ps();
            let one = _mm_set1_ps(1.0);

            let r_clamped = _mm_min_ps(_mm_max_ps(r.0, zero), one);
            let g_clamped = _mm_min_ps(_mm_max_ps(g.0, zero), one);
            let b_clamped = _mm_min_ps(_mm_max_ps(b.0, zero), one);
            let a_clamped = _mm_min_ps(_mm_max_ps(a.0, zero), one);

            let r_scaled = _mm_mul_ps(r_clamped, scale);
            let g_scaled = _mm_mul_ps(g_clamped, scale);
            let b_scaled = _mm_mul_ps(b_clamped, scale);
            let a_scaled = _mm_mul_ps(a_clamped, scale);

            // Convert to i32 (SSE2 cvttps converts to signed)
            let r_i32 = _mm_cvttps_epi32(r_scaled);
            let g_i32 = _mm_cvttps_epi32(g_scaled);
            let b_i32 = _mm_cvttps_epi32(b_scaled);
            let a_i32 = _mm_cvttps_epi32(a_scaled);

            // Pack: R | (G << 8) | (B << 16) | (A << 24)
            let g_shifted = _mm_slli_epi32(g_i32, 8);
            let b_shifted = _mm_slli_epi32(b_i32, 16);
            let a_shifted = _mm_slli_epi32(a_i32, 24);

            let packed = _mm_or_si128(
                _mm_or_si128(r_i32, g_shifted),
                _mm_or_si128(b_shifted, a_shifted),
            );
            Self(packed)
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
    type U32 = U32x16;
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
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            let all_zeros = _mm512_setzero_ps();
            Self(_mm512_mask_blend_ps(mask, all_zeros, all_ones))
        }
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

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(_mm512_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // Scalar fallback - could use _mm512_i32gather_ps for perf
        let idx = indices.to_array();
        let len = slice.len();
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            let ix = (libm::floorf(idx[i]) as isize).clamp(0, len as isize - 1) as usize;
            out[i] = slice[ix];
        }
        Self::from_slice(&out)
    }

    #[inline(always)]
    fn floor(self) -> Self {
        unsafe {
            // AVX-512: use roundscale with floor mode (1 = floor, bit 3 = suppress exceptions)
            // imm8 = 0b1001 = 9
            Self(_mm512_roundscale_ps::<9>(self.0))
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

// ============================================================================
// U32x16 - 16-lane u32 SIMD for packed RGBA pixels (AVX512)
// ============================================================================

/// 16-lane u32 SIMD vector for AVX512 (packed RGBA pixels).
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x16(__m512i);

#[cfg(target_feature = "avx512f")]
impl Default for U32x16 {
    fn default() -> Self {
        unsafe { Self(_mm512_setzero_si512()) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Debug for U32x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut arr = [0u32; 16];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut __m512i, self.0) };
        write!(f, "U32x16({:?})", arr)
    }
}

#[cfg(target_feature = "avx512f")]
impl SimdU32Ops for U32x16 {
    const LANES: usize = 16;

    #[inline(always)]
    fn splat(val: u32) -> Self {
        unsafe { Self(_mm512_set1_epi32(val as i32)) }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u32]) {
        assert!(out.len() >= Self::LANES);
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) }
    }

    #[inline(always)]
    fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
        // Placeholder
        Self::default()
    }
}

#[cfg(target_feature = "avx512f")]
impl BitAnd for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl BitOr for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_or_si512(self.0, rhs.0)) }
    }
}

#[cfg(target_feature = "avx512f")]
impl Shl<u32> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm512_sll_epi32(self.0, shift))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl Shr<u32> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        unsafe {
            let shift = _mm_cvtsi32_si128(rhs as i32);
            Self(_mm512_srl_epi32(self.0, shift))
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl U32x16 {
    /// Pack 16 f32 Fields (RGBA) into packed u32 pixels.
    #[inline(always)]
    pub(crate) fn pack_rgba(r: F32x16, g: F32x16, b: F32x16, a: F32x16) -> Self {
        unsafe {
            let scale = _mm512_set1_ps(255.0);
            let zero = _mm512_setzero_ps();
            let one = _mm512_set1_ps(1.0);

            let r_clamped = _mm512_min_ps(_mm512_max_ps(r.0, zero), one);
            let g_clamped = _mm512_min_ps(_mm512_max_ps(g.0, zero), one);
            let b_clamped = _mm512_min_ps(_mm512_max_ps(b.0, zero), one);
            let a_clamped = _mm512_min_ps(_mm512_max_ps(a.0, zero), one);

            let r_scaled = _mm512_mul_ps(r_clamped, scale);
            let g_scaled = _mm512_mul_ps(g_clamped, scale);
            let b_scaled = _mm512_mul_ps(b_clamped, scale);
            let a_scaled = _mm512_mul_ps(a_clamped, scale);

            let r_i32 = _mm512_cvttps_epi32(r_scaled);
            let g_i32 = _mm512_cvttps_epi32(g_scaled);
            let b_i32 = _mm512_cvttps_epi32(b_scaled);
            let a_i32 = _mm512_cvttps_epi32(a_scaled);

            let g_shifted = _mm512_slli_epi32(g_i32, 8);
            let b_shifted = _mm512_slli_epi32(b_i32, 16);
            let a_shifted = _mm512_slli_epi32(a_i32, 24);

            let packed = _mm512_or_si512(
                _mm512_or_si512(r_i32, g_shifted),
                _mm512_or_si512(b_shifted, a_shifted),
            );
            Self(packed)
        }
    }
}