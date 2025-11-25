//! ARM NEON backend using `std::arch` intrinsics.
//!
//! This backend uses NEON instructions for AArch64 (64-bit ARM),
//! which includes Apple Silicon (M1/M2/M3) and modern ARM Linux systems.

use core::arch::aarch64::*;
use crate::PixelBatch;

/// A batch of 4 pixels (128 bits) using NEON registers.
///
/// Wraps `uint32x4_t` in a newtype for safety and to implement traits.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Batch(uint32x4_t);

impl PixelBatch for Batch {
    #[inline(always)]
    fn splat(val: u32) -> Self {
        // SAFETY: vdupq_n_u32 is always safe, just broadcasts a value
        unsafe { Self(vdupq_n_u32(val)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes
        // vld1q_u32 handles unaligned loads
        unsafe { Self(vld1q_u32(ptr)) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u32) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        // vst1q_u32 handles unaligned stores
        unsafe { vst1q_u32(ptr, self.0) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        // SAFETY: Bitwise OR is always safe
        unsafe { Self(vorrq_u32(self.0, other.0)) }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        // SAFETY: Bitwise AND is always safe
        unsafe { Self(vandq_u32(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: Bitwise NOT (MVN instruction)
        unsafe { Self(vmvnq_u32(self.0)) }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // Implements: (self & mask) | (other & !mask)
        //
        // NEON has a dedicated instruction for this: BSL (Bit Select)
        // BSL: result = (mask & a) | (!mask & b)
        //
        // But vbslq_u32 takes arguments in a different order: vbslq(mask, a, b)
        // means: result = (mask & a) | (!mask & b)
        //
        // SAFETY: All these operations are safe bitwise ops
        unsafe { Self(vbslq_u32(mask.0, self.0, other.0)) }
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
