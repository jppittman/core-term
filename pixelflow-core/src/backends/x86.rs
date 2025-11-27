//! x86_64 SSE backend using `std::arch` intrinsics.
//!
//! This backend uses SSE2 instructions (guaranteed on all x86_64 CPUs).
//!
//! ## Type-Driven Instruction Selection
//!
//! On x86, all SIMD registers are just `__m128i` (128 bits of untyped data).
//! We use `PhantomData<T>` to track the logical type at compile time:
//!
//! - `SimdVec<u32>` → `paddd`, `pmulld` (32-bit ops, 4 lanes)
//! - `SimdVec<u16>` → `paddw`, `pmullw` (16-bit ops, 8 lanes)
//! - `SimdVec<u8>` → `paddb`, `pmullb` (8-bit ops, 16 lanes)
//!
//! The type `T` is purely a compile-time marker - bitcasting is free.

use core::arch::x86_64::*;
use core::marker::PhantomData;
use crate::batch::SimdOps;

/// Platform-specific SIMD vector wrapper.
///
/// On x86: All types are just `__m128i` with a `PhantomData<T>` marker.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SimdVec<T>(pub(crate) __m128i, PhantomData<T>);

// ============================================================================
// u32 Implementation (4 lanes, 32-bit operations)
// ============================================================================

impl SimdOps<u32> for SimdVec<u32> {
    #[inline(always)]
    fn splat(val: u32) -> Self {
        // SAFETY: _mm_set1_epi32 is always safe
        unsafe { Self(_mm_set1_epi32(val as i32), PhantomData) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes
        unsafe { Self(_mm_loadu_si128(ptr as *const __m128i), PhantomData) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u32) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.0) }
    }

    #[inline(always)]
    fn new(v0: u32, v1: u32, v2: u32, v3: u32) -> Self {
        // SAFETY: _mm_set_epi32 is always safe
        // Note: SSE uses reverse order (e3, e2, e1, e0)
        unsafe {
            Self(
                _mm_set_epi32(v3 as i32, v2 as i32, v1 as i32, v0 as i32),
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: paddd (32-bit add)
        unsafe { Self(_mm_add_epi32(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: psubd (32-bit subtract)
        unsafe { Self(_mm_sub_epi32(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: pmulld (32-bit multiply, SSE4.1)
        // For SSE2-only CPUs, this would need emulation
        unsafe { Self(_mm_mullo_epi32(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        // SAFETY: pand (bitwise AND)
        unsafe { Self(_mm_and_si128(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        // SAFETY: por (bitwise OR)
        unsafe { Self(_mm_or_si128(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: XOR with all-ones produces NOT
        unsafe {
            let all_ones = _mm_set1_epi32(-1);
            Self(_mm_xor_si128(self.0, all_ones), PhantomData)
        }
    }

    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        // SAFETY: psrld (32-bit logical right shift)
        unsafe {
            let count_vec = _mm_cvtsi32_si128(count);
            Self(_mm_srl_epi32(self.0, count_vec), PhantomData)
        }
    }

    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        // SAFETY: pslld (32-bit logical left shift)
        unsafe {
            let count_vec = _mm_cvtsi32_si128(count);
            Self(_mm_sll_epi32(self.0, count_vec), PhantomData)
        }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // Implements: (self & mask) | (other & !mask)
        // SAFETY: All bitwise operations are safe
        unsafe {
            let masked_self = _mm_and_si128(self.0, mask.0);
            let not_mask = _mm_xor_si128(mask.0, _mm_set1_epi32(-1));
            let masked_other = _mm_and_si128(other.0, not_mask);
            Self(_mm_or_si128(masked_self, masked_other), PhantomData)
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SSE2 compatible unsigned min
        unsafe {
            let mask = cmp_gt_u32(self.0, other.0); // mask = self > other
            // if self > other, min is other. else self.
            // select(other, self, mask)
            let masked_other = _mm_and_si128(other.0, mask);
            let not_mask = _mm_xor_si128(mask, _mm_set1_epi32(-1));
            let masked_self = _mm_and_si128(self.0, not_mask);
            Self(_mm_or_si128(masked_self, masked_other), PhantomData)
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SSE2 compatible unsigned max
        unsafe {
            let mask = cmp_gt_u32(self.0, other.0); // mask = self > other
            // if self > other, max is self. else other.
            // select(self, other, mask)
            let masked_self = _mm_and_si128(self.0, mask);
            let not_mask = _mm_xor_si128(mask, _mm_set1_epi32(-1));
            let masked_other = _mm_and_si128(other.0, not_mask);
            Self(_mm_or_si128(masked_self, masked_other), PhantomData)
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            let sum = _mm_add_epi32(self.0, other.0);
            // Overflow if sum < self (unsigned)
            // mask = self > sum
            let mask = cmp_gt_u32(self.0, sum);
            // if mask (overflow), return MAX (-1), else sum
            // sum | mask
            Self(_mm_or_si128(sum, mask), PhantomData)
        }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            let diff = _mm_sub_epi32(self.0, other.0);
            // Underflow if self < other (unsigned) -> other > self
            // mask = other > self
            let mask = cmp_gt_u32(other.0, self.0);
            // if mask (underflow), return 0, else diff
            // diff & !mask
            let not_mask = _mm_xor_si128(mask, _mm_set1_epi32(-1));
            Self(_mm_and_si128(diff, not_mask), PhantomData)
        }
    }
}

// Helper: Unsigned Greater Than for u32 (SSE2)
#[inline(always)]
unsafe fn cmp_gt_u32(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sign_flip = _mm_set1_epi32(0x80000000u32 as i32);
        let a_flipped = _mm_xor_si128(a, sign_flip);
        let b_flipped = _mm_xor_si128(b, sign_flip);
        _mm_cmpgt_epi32(a_flipped, b_flipped)
    }
}

// ============================================================================
// u16 Implementation (8 lanes, 16-bit operations)
// ============================================================================

impl SimdOps<u16> for SimdVec<u16> {
    #[inline(always)]
    fn splat(val: u16) -> Self {
        // SAFETY: _mm_set1_epi16 is always safe
        unsafe { Self(_mm_set1_epi16(val as i16), PhantomData) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u16) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes (8×u16)
        unsafe { Self(_mm_loadu_si128(ptr as *const __m128i), PhantomData) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u16) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.0) }
    }

    #[inline(always)]
    fn new(v0: u16, v1: u16, v2: u16, v3: u16) -> Self {
        // SAFETY: Create 8×u16 vector (only setting first 4, rest are zero)
        // For full 8-lane construction, would need 8 parameters
        unsafe {
            Self(
                _mm_set_epi16(0, 0, 0, 0, v3 as i16, v2 as i16, v1 as i16, v0 as i16),
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: paddw (16-bit add)
        // This is the instruction that `add_u16()` was trying to expose!
        unsafe { Self(_mm_add_epi16(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: psubw (16-bit subtract)
        unsafe { Self(_mm_sub_epi16(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: pmullw (16-bit multiply, keeps low 16 bits)
        // This is the instruction that `mullo_u16()` was trying to expose!
        unsafe { Self(_mm_mullo_epi16(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        // SAFETY: pand (bitwise AND, type-agnostic)
        unsafe { Self(_mm_and_si128(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        // SAFETY: por (bitwise OR, type-agnostic)
        unsafe { Self(_mm_or_si128(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: XOR with all-ones
        unsafe {
            let all_ones = _mm_set1_epi16(-1);
            Self(_mm_xor_si128(self.0, all_ones), PhantomData)
        }
    }

    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        // SAFETY: psrlw (16-bit logical right shift)
        // This is the instruction that `shift_right_u16()` was trying to expose!
        unsafe {
            let count_vec = _mm_cvtsi32_si128(count);
            Self(_mm_srl_epi16(self.0, count_vec), PhantomData)
        }
    }

    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        // SAFETY: psllw (16-bit logical left shift)
        unsafe {
            let count_vec = _mm_cvtsi32_si128(count);
            Self(_mm_sll_epi16(self.0, count_vec), PhantomData)
        }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // SAFETY: Same bitwise logic as u32
        unsafe {
            let masked_self = _mm_and_si128(self.0, mask.0);
            let not_mask = _mm_xor_si128(mask.0, _mm_set1_epi16(-1));
            let masked_other = _mm_and_si128(other.0, not_mask);
            Self(_mm_or_si128(masked_self, masked_other), PhantomData)
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: _mm_min_epu16 (unsigned 16-bit min, requires SSE4.1)
        unsafe { Self(_mm_min_epu16(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: _mm_max_epu16 (unsigned 16-bit max, requires SSE4.1)
        unsafe { Self(_mm_max_epu16(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        // SAFETY: _mm_adds_epu16 (native u16 saturating add in SSE2!)
        unsafe { Self(_mm_adds_epu16(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        // SAFETY: _mm_subs_epu16 (native u16 saturating sub in SSE2!)
        unsafe { Self(_mm_subs_epu16(self.0, other.0), PhantomData) }
    }
}

// ============================================================================
// u8 Implementation (16 lanes, 8-bit operations)
// ============================================================================

impl SimdOps<u8> for SimdVec<u8> {
    #[inline(always)]
    fn splat(val: u8) -> Self {
        // SAFETY: _mm_set1_epi8 is always safe
        unsafe { Self(_mm_set1_epi8(val as i8), PhantomData) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes (16×u8)
        unsafe { Self(_mm_loadu_si128(ptr as *const __m128i), PhantomData) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.0) }
    }

    #[inline(always)]
    fn new(v0: u8, v1: u8, v2: u8, v3: u8) -> Self {
        // SAFETY: Create 16×u8 vector (only setting first 4, rest are zero)
        unsafe {
            Self(
                _mm_set_epi8(
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, v3 as i8, v2 as i8, v1 as i8, v0 as i8,
                ),
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: paddb (8-bit add)
        unsafe { Self(_mm_add_epi8(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: psubb (8-bit subtract)
        unsafe { Self(_mm_sub_epi8(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn mul(self, _other: Self) -> Self {
        // SAFETY: SSE2 does NOT have pmullb (8-bit multiply)
        // Would need emulation or SSE4.1's pmaddubsw
        unimplemented!("8-bit multiply not supported in SSE2")
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm_set1_epi8(-1);
            Self(_mm_xor_si128(self.0, all_ones), PhantomData)
        }
    }

    #[inline(always)]
    fn shr(self, _count: i32) -> Self {
        // SAFETY: SSE2 does not have psrlb (8-bit shift)
        // Would need emulation with masking tricks
        unimplemented!("8-bit shift not supported in SSE2")
    }

    #[inline(always)]
    fn shl(self, _count: i32) -> Self {
        unimplemented!("8-bit shift not supported in SSE2")
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        unsafe {
            let masked_self = _mm_and_si128(self.0, mask.0);
            let not_mask = _mm_xor_si128(mask.0, _mm_set1_epi8(-1));
            let masked_other = _mm_and_si128(other.0, not_mask);
            Self(_mm_or_si128(masked_self, masked_other), PhantomData)
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: _mm_min_epu8 (unsigned 8-bit min, requires SSE2)
        unsafe { Self(_mm_min_epu8(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: _mm_max_epu8 (unsigned 8-bit max, requires SSE2)
        unsafe { Self(_mm_max_epu8(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        // SAFETY: _mm_adds_epu8 (native u8 saturating add in SSE2!)
        unsafe { Self(_mm_adds_epu8(self.0, other.0), PhantomData) }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        // SAFETY: _mm_subs_epu8 (native u8 saturating sub in SSE2!)
        unsafe { Self(_mm_subs_epu8(self.0, other.0), PhantomData) }
    }
}

// ============================================================================
// Bitcasting (Zero-Cost Type Conversion)
// ============================================================================

/// Bitcast between SIMD types.
///
/// On x86, this is completely free - same register, different interpretation.
///
/// ```ignore
/// let pixels = SimdVec::<u32>::splat(0xFF00FF00); // 4×u32
/// let as_u16: SimdVec<u16> = cast(pixels);         // View as 8×u16
/// ```
#[inline(always)]
pub fn cast<T, U>(v: SimdVec<T>) -> SimdVec<U> {
    // SAFETY: On x86, all types are just __m128i
    // The PhantomData type marker changes, but the bits are identical
    SimdVec(v.0, PhantomData)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_add() {
        let a = SimdVec::<u32>::splat(100);
        let b = SimdVec::<u32>::splat(50);
        let c = a.add(b);

        let mut output = [0u32; 4];
        unsafe { c.store(output.as_mut_ptr()) };
        assert_eq!(output, [150; 4]);
    }

    #[test]
    fn test_u16_multiply() {
        // This is the key operation for bilinear interpolation
        let a = SimdVec::<u16>::splat(100);
        let b = SimdVec::<u16>::splat(2);
        let c = a.mul(b); // Uses pmullw!

        let mut output = [0u16; 8];
        unsafe { c.store(output.as_mut_ptr()) };
        assert_eq!(output, [200; 8]);
    }

    #[test]
    fn test_bitcast() {
        let pixels = SimdVec::<u32>::new(0x12345678, 0x9ABCDEF0, 0, 0);
        let as_u16: SimdVec<u16> = cast(pixels);

        let mut output = [0u16; 8];
        unsafe { as_u16.store(output.as_mut_ptr()) };

        // Little-endian byte order
        assert_eq!(output[0], 0x5678); // Low 16 bits of 0x12345678
        assert_eq!(output[1], 0x1234); // High 16 bits
    }
}
