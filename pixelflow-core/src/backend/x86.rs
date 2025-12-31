//! x86_64 backend.

use super::{Backend, MaskOps, SimdOps, SimdU32Ops};
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

// ============================================================================
// Mask4 - 4-lane mask for SSE2 (float-based, no separate mask unit)
// ============================================================================

/// 4-lane mask for SSE2.
///
/// On SSE2, there's no separate mask register file like AVX-512's k-registers.
/// Masks are stored as float vectors where each lane is either all-1s (true)
/// or all-0s (false).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Mask4(__m128);

impl Default for Mask4 {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_ps()) }
    }
}

impl Debug for Mask4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mask4({:04b})", unsafe { _mm_movemask_ps(self.0) })
    }
}

impl MaskOps for Mask4 {
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) == 0xF }
    }
}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm_and_ps(self.0, rhs.0)) }
    }
}

impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm_or_ps(self.0, rhs.0)) }
    }
}

impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
            Self(_mm_xor_ps(self.0, all_ones))
        }
    }
}

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
    type Mask = Mask4;
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
    fn cmp_lt(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmplt_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmple_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmpgt_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Mask4 {
        unsafe { Mask4(_mm_cmpge_ps(self.0, rhs.0)) }
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
    fn select(mask: Mask4, if_true: Self, if_false: Self) -> Self {
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
            // Indices are pre-floored by Texture::eval_raw, so truncation is safe and faster.
            // Even for general use, truncation matches floor for positive indices.
            let ix = (idx[i] as isize).clamp(0, len as isize - 1) as usize;
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
            let cmp_mask = _mm_cmplt_ps(self.0, trunc);
            let one = _mm_set1_ps(1.0);
            let correction = _mm_and_ps(cmp_mask, one);
            Self(_mm_sub_ps(trunc, correction))
        }
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            Self(_mm_fmadd_ps(self.0, b.0, c.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            // Fallback: separate mul + add (two roundings)
            self * b + c
        }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Mask4) -> Self {
        // SSE2: no native masked add, emulate with select
        // self + (mask ? val : 0)
        unsafe {
            let masked_val = _mm_and_ps(mask.0, val.0);
            Self(_mm_add_ps(self.0, masked_val))
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // Use fast reciprocal approximation (~12 bits accuracy)
        unsafe { Self(_mm_rcp_ps(self.0)) }
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        // Fast reciprocal square root (~12 bits accuracy)
        // Sufficient for AA coverage calculations
        unsafe { Self(_mm_rsqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn mask_to_float(mask: Mask4) -> Self {
        // Mask4 already stores float representation
        Self(mask.0)
    }

    #[inline(always)]
    fn float_to_mask(self) -> Mask4 {
        // Float representation is already a valid mask
        Mask4(self.0)
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

impl core::ops::Neg for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            // Flip sign bit via XOR with -0.0 (0x80000000 = min i32 value)
            let neg_zero = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
            Self(_mm_xor_ps(self.0, neg_zero))
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

impl Not for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // XOR with all 1s
        unsafe {
            let ones = _mm_set1_epi32(-1);
            Self(_mm_xor_si128(self.0, ones))
        }
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

// ============================================================================
// Mask16 - 16-lane native k-register mask for AVX-512
// ============================================================================

/// 16-lane mask using AVX-512 k-registers.
///
/// This is the big win: AVX-512 has dedicated 8 mask registers (k0-k7) that
/// run on a separate execution unit from the float ALU. Mask operations
/// (and/or/not/any/all) are effectively free - they execute in parallel
/// with float work.
///
/// - `kortestw k1, k1` for any() - ~0 cycles (mask unit)
/// - `kand/kor/knot` for mask logic - ~0-1 cycles (mask unit)
/// - `vblendmps` uses k-register directly - no conversion overhead
#[cfg(target_feature = "avx512f")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Mask16(pub(crate) __mmask16);

#[cfg(target_feature = "avx512f")]
impl Default for Mask16 {
    fn default() -> Self {
        Self(0)
    }
}

#[cfg(target_feature = "avx512f")]
impl Debug for Mask16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mask16({:016b})", self.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl MaskOps for Mask16 {
    #[inline(always)]
    fn any(self) -> bool {
        // kortestw k1, k1 - runs on mask unit, ~0 cycles
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        // Mask equality check - runs on mask unit, ~0 cycles
        self.0 == 0xFFFF
    }
}

#[cfg(target_feature = "avx512f")]
impl BitAnd for Mask16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // kand - runs on mask unit, ~0-1 cycles
        Self(self.0 & rhs.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl BitOr for Mask16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // kor - runs on mask unit, ~0-1 cycles
        Self(self.0 | rhs.0)
    }
}

#[cfg(target_feature = "avx512f")]
impl Not for Mask16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // knot - runs on mask unit, ~0-1 cycles
        Self(!self.0)
    }
}

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
    type Mask = Mask16;
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
    fn cmp_lt(self, rhs: Self) -> Mask16 {
        // Returns native k-register mask directly - no conversion!
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LT_OQ)) }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Mask16 {
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LE_OQ)) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Mask16 {
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GT_OQ)) }
    }

    #[inline(always)]
    fn cmp_ge(self, rhs: Self) -> Mask16 {
        unsafe { Mask16(_mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GE_OQ)) }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe { Self(_mm512_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
            Self(_mm512_and_ps(self.0, abs_mask))
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
    fn select(mask: Mask16, if_true: Self, if_false: Self) -> Self {
        // Native k-register blend - no conversion needed!
        // vblendmps uses k-register directly
        unsafe { Self(_mm512_mask_blend_ps(mask.0, if_false.0, if_true.0)) }
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= Self::LANES);
        unsafe { Self(_mm512_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn gather(slice: &[f32], indices: Self) -> Self {
        // Native AVX-512 gather - 16 floats in one instruction
        // Precondition: indices must be valid (0..slice.len()), already ensured by Texture::eval_raw
        unsafe {
            // Convert float indices to i32 (truncate - indices are already integral from floor)
            let idx_i32 = _mm512_cvttps_epi32(indices.0);
            // Gather 16 f32 values; scale=4 bytes per element
            Self(_mm512_i32gather_ps::<4>(idx_i32, slice.as_ptr()))
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        unsafe {
            // AVX-512: use roundscale with floor mode (1 = floor, bit 3 = suppress exceptions)
            // imm8 = 0b1001 = 9
            Self(_mm512_roundscale_ps::<9>(self.0))
        }
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // AVX-512F always includes FMA
        unsafe { Self(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Mask16) -> Self {
        // Native masked add using k-register directly - no conversion!
        // This is the hot path for winding number accumulation
        unsafe { Self(_mm512_mask_add_ps(self.0, mask.0, self.0, val.0)) }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // AVX-512 has higher accuracy reciprocal (~14 bits)
        unsafe { Self(_mm512_rcp14_ps(self.0)) }
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        // Fast reciprocal square root (~14 bits accuracy)
        // Sufficient for AA coverage calculations
        unsafe { Self(_mm512_rsqrt14_ps(self.0)) }
    }

    #[inline(always)]
    fn mask_to_float(mask: Mask16) -> Self {
        // Convert k-register to float representation (all-1s or all-0s)
        // This is the "expensive" direction - try to avoid in hot paths
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            let all_zeros = _mm512_setzero_ps();
            Self(_mm512_mask_blend_ps(mask.0, all_zeros, all_ones))
        }
    }

    #[inline(always)]
    fn float_to_mask(self) -> Mask16 {
        // Convert float representation to k-register mask
        // Check if each lane is non-zero
        unsafe {
            let as_int = _mm512_castps_si512(self.0);
            Mask16(_mm512_cmp_epi32_mask(
                as_int,
                _mm512_setzero_si512(),
                _MM_CMPINT_NE,
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

#[cfg(target_feature = "avx512f")]
impl core::ops::Neg for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            // Flip sign bit via XOR with -0.0 (0x80000000 = min i32 value)
            let neg_zero = _mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN));
            Self(_mm512_xor_ps(self.0, neg_zero))
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
impl Not for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // XOR with all 1s
        unsafe {
            let ones = _mm512_set1_epi32(-1);
            Self(_mm512_xor_si512(self.0, ones))
        }
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
