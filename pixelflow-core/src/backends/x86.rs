//! x86_64 SSE backend using `std::arch` intrinsics.
//!
//! This backend uses SSE2 instructions, which are guaranteed to be available
//! on all x86_64 CPUs.

use core::arch::x86_64::*;
use crate::PixelBatch;

/// A batch of 4 pixels (128 bits) using SSE registers.
///
/// Wraps `__m128i` in a newtype for safety and to implement traits.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Batch(__m128i);

impl PixelBatch for Batch {
    #[inline(always)]
    fn splat(val: u32) -> Self {
        // SAFETY: _mm_set1_epi32 is always safe, just broadcasts a value
        unsafe { Self(_mm_set1_epi32(val as i32)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes
        // _mm_loadu_si128 handles unaligned loads
        unsafe { Self(_mm_loadu_si128(ptr as *const __m128i)) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u32) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        // _mm_storeu_si128 handles unaligned stores
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.0) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        // SAFETY: Bitwise OR is always safe
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        // SAFETY: Bitwise AND is always safe
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: XOR with all-ones produces NOT
        unsafe {
            let all_ones = _mm_set1_epi32(-1);
            Self(_mm_xor_si128(self.0, all_ones))
        }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // Implements: (self & mask) | (other & !mask)
        //
        // This is the "Jeff Dean optimization" - use SSE instructions
        // to perform conditional selection in one or two cycles.
        //
        // SAFETY: All these operations are safe bitwise ops
        unsafe {
            let masked_self = _mm_and_si128(self.0, mask.0);
            let not_mask = _mm_xor_si128(mask.0, _mm_set1_epi32(-1));
            let masked_other = _mm_and_si128(other.0, not_mask);
            Self(_mm_or_si128(masked_self, masked_other))
        }
    }

    #[inline(always)]
    fn to_bytes(self) -> [[u8; 4]; 4] {
        let mut output = [0u32; 4];
        unsafe { self.store(output.as_mut_ptr()) };
        [
            output[0].to_le_bytes(),
            output[1].to_le_bytes(),
            output[2].to_le_bytes(),
            output[3].to_le_bytes(),
        ]
    }

    #[inline(always)]
    fn from_bytes(bytes: [[u8; 4]; 4]) -> Self {
        let pixels = [
            u32::from_le_bytes(bytes[0]),
            u32::from_le_bytes(bytes[1]),
            u32::from_le_bytes(bytes[2]),
            u32::from_le_bytes(bytes[3]),
        ];
        unsafe { Self::load(pixels.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splat() {
        let batch = Batch::splat(0x12345678);
        let mut output = [0u32; 4];
        unsafe { batch.store(output.as_mut_ptr()) };
        assert_eq!(output, [0x12345678, 0x12345678, 0x12345678, 0x12345678]);
    }

    #[test]
    fn test_load_store() {
        let input = [0x11111111u32, 0x22222222, 0x33333333, 0x44444444];
        let batch = unsafe { Batch::load(input.as_ptr()) };
        let mut output = [0u32; 4];
        unsafe { batch.store(output.as_mut_ptr()) };
        assert_eq!(output, input);
    }

    #[test]
    fn test_or() {
        let a = Batch::splat(0x0F0F0F0F);
        let b = Batch::splat(0xF0F0F0F0);
        let c = a.or(b);
        let mut output = [0u32; 4];
        unsafe { c.store(output.as_mut_ptr()) };
        assert_eq!(output, [0xFFFFFFFF; 4]);
    }

    #[test]
    fn test_and() {
        let a = Batch::splat(0xFF00FF00);
        let b = Batch::splat(0xFFFF0000);
        let c = a.and(b);
        let mut output = [0u32; 4];
        unsafe { c.store(output.as_mut_ptr()) };
        assert_eq!(output, [0xFF000000; 4]);
    }

    #[test]
    fn test_not() {
        let a = Batch::splat(0x00000000);
        let b = a.not();
        let mut output = [0u32; 4];
        unsafe { b.store(output.as_mut_ptr()) };
        assert_eq!(output, [0xFFFFFFFF; 4]);
    }

    #[test]
    fn test_select() {
        let fg = Batch::splat(0xFFFFFFFF); // White
        let bg = Batch::splat(0x00000000); // Black
        let mask = unsafe {
            Batch::load([0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000].as_ptr())
        };
        let result = fg.select(bg, mask);
        let mut output = [0u32; 4];
        unsafe { result.store(output.as_mut_ptr()) };
        assert_eq!(output, [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000]);
    }
}
