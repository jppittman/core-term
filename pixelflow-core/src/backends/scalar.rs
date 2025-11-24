//! Scalar fallback backend for platforms without SIMD support.
//!
//! This backend processes 4 pixels using regular scalar operations.
//! It's slower than SIMD but guarantees correctness on all platforms.

use crate::PixelBatch;

/// A batch of 4 pixels using scalar operations.
#[derive(Copy, Clone)]
pub struct Batch([u32; 4]);

impl PixelBatch for Batch {
    #[inline(always)]
    fn splat(val: u32) -> Self {
        Self([val, val, val, val])
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 4 u32 values
        unsafe {
            Self([
                *ptr.offset(0),
                *ptr.offset(1),
                *ptr.offset(2),
                *ptr.offset(3),
            ])
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u32) {
        // SAFETY: Caller guarantees ptr is valid for writing 4 u32 values
        unsafe {
            *ptr.offset(0) = self.0[0];
            *ptr.offset(1) = self.0[1];
            *ptr.offset(2) = self.0[2];
            *ptr.offset(3) = self.0[3];
        }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // (self & mask) | (other & !mask)
        Self([
            (self.0[0] & mask.0[0]) | (other.0[0] & !mask.0[0]),
            (self.0[1] & mask.0[1]) | (other.0[1] & !mask.0[1]),
            (self.0[2] & mask.0[2]) | (other.0[2] & !mask.0[2]),
            (self.0[3] & mask.0[3]) | (other.0[3] & !mask.0[3]),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splat() {
        let batch = Batch::splat(0x12345678);
        assert_eq!(batch.0, [0x12345678; 4]);
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
    fn test_select() {
        let fg = Batch::splat(0xFFFFFFFF);
        let bg = Batch::splat(0x00000000);
        let mask = Batch([0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000]);
        let result = fg.select(bg, mask);
        assert_eq!(result.0, [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000]);
    }
}
