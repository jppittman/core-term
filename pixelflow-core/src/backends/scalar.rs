//! Scalar fallback backend for platforms without SIMD support.
//!
//! This backend processes data using regular scalar operations in arrays.
//! It's slower than SIMD but guarantees correctness on all platforms.
//!
//! ## Type-Driven Operations
//!
//! Like the SIMD backends, this uses the type to determine the operation:
//! - `SimdVec<u32>` → Array of 4×u32
//! - `SimdVec<u16>` → Array of 8×u16
//! - `SimdVec<u8>` → Array of 16×u8

use core::marker::PhantomData;
use crate::batch::SimdOps;

// ============================================================================
// Platform-Specific Types
// ============================================================================

/// Platform-specific SIMD vector wrapper.
///
/// On scalar: Uses fixed-size arrays matching SIMD lane counts.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SimdVec<T>(pub(crate) ScalarReg<T>);

/// The actual storage types (platform-specific).
#[derive(Copy, Clone)]
pub union ScalarReg<T> {
    pub u32: [u32; 4],
    pub u16: [u16; 8],
    pub u8: [u8; 16],
    _marker: PhantomData<T>,
}

// ============================================================================
// u32 Implementation (4 lanes)
// ============================================================================

impl SimdOps<u32> for SimdVec<u32> {
    #[inline(always)]
    fn splat(val: u32) -> Self {
        Self(ScalarReg { u32: [val; 4] })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 4 u32 values
        unsafe {
            Self(ScalarReg {
                u32: [
                    *ptr.offset(0),
                    *ptr.offset(1),
                    *ptr.offset(2),
                    *ptr.offset(3),
                ],
            })
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u32) {
        // SAFETY: Caller guarantees ptr is valid for writing 4 u32 values
        unsafe {
            let arr = self.0.u32;
            *ptr.offset(0) = arr[0];
            *ptr.offset(1) = arr[1];
            *ptr.offset(2) = arr[2];
            *ptr.offset(3) = arr[3];
        }
    }

    #[inline(always)]
    fn new(v0: u32, v1: u32, v2: u32, v3: u32) -> Self {
        Self(ScalarReg { u32: [v0, v1, v2, v3] })
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0].wrapping_add(b[0]),
                    a[1].wrapping_add(b[1]),
                    a[2].wrapping_add(b[2]),
                    a[3].wrapping_add(b[3]),
                ],
            })
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0].wrapping_sub(b[0]),
                    a[1].wrapping_sub(b[1]),
                    a[2].wrapping_sub(b[2]),
                    a[3].wrapping_sub(b[3]),
                ],
            })
        }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0].wrapping_mul(b[0]),
                    a[1].wrapping_mul(b[1]),
                    a[2].wrapping_mul(b[2]),
                    a[3].wrapping_mul(b[3]),
                ],
            })
        }
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]],
            })
        }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]],
            })
        }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = self.0.u32;
            Self(ScalarReg {
                u32: [!a[0], !a[1], !a[2], !a[3]],
            })
        }
    }

    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0] >> count,
                    a[1] >> count,
                    a[2] >> count,
                    a[3] >> count,
                ],
            })
        }
    }

    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0] << count,
                    a[1] << count,
                    a[2] << count,
                    a[3] << count,
                ],
            })
        }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // (self & mask) | (other & !mask)
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            let m = mask.0.u32;
            Self(ScalarReg {
                u32: [
                    (a[0] & m[0]) | (b[0] & !m[0]),
                    (a[1] & m[1]) | (b[1] & !m[1]),
                    (a[2] & m[2]) | (b[2] & !m[2]),
                    (a[3] & m[3]) | (b[3] & !m[3]),
                ],
            })
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].min(b[2]),
                    a[3].min(b[3]),
                ],
            })
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0].max(b[0]),
                    a[1].max(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ],
            })
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0].saturating_add(b[0]),
                    a[1].saturating_add(b[1]),
                    a[2].saturating_add(b[2]),
                    a[3].saturating_add(b[3]),
                ],
            })
        }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u32;
            let b = other.0.u32;
            Self(ScalarReg {
                u32: [
                    a[0].saturating_sub(b[0]),
                    a[1].saturating_sub(b[1]),
                    a[2].saturating_sub(b[2]),
                    a[3].saturating_sub(b[3]),
                ],
            })
        }
    }
}

// ============================================================================
// u16 Implementation (8 lanes)
// ============================================================================

impl SimdOps<u16> for SimdVec<u16> {
    #[inline(always)]
    fn splat(val: u16) -> Self {
        Self(ScalarReg { u16: [val; 8] })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u16) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 8 u16 values
        unsafe {
            Self(ScalarReg {
                u16: [
                    *ptr.offset(0),
                    *ptr.offset(1),
                    *ptr.offset(2),
                    *ptr.offset(3),
                    *ptr.offset(4),
                    *ptr.offset(5),
                    *ptr.offset(6),
                    *ptr.offset(7),
                ],
            })
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u16) {
        // SAFETY: Caller guarantees ptr is valid for writing 8 u16 values
        unsafe {
            let arr = self.0.u16;
            *ptr.offset(0) = arr[0];
            *ptr.offset(1) = arr[1];
            *ptr.offset(2) = arr[2];
            *ptr.offset(3) = arr[3];
            *ptr.offset(4) = arr[4];
            *ptr.offset(5) = arr[5];
            *ptr.offset(6) = arr[6];
            *ptr.offset(7) = arr[7];
        }
    }

    #[inline(always)]
    fn new(v0: u16, v1: u16, v2: u16, v3: u16) -> Self {
        Self(ScalarReg {
            u16: [v0, v1, v2, v3, 0, 0, 0, 0],
        })
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0].wrapping_add(b[0]),
                    a[1].wrapping_add(b[1]),
                    a[2].wrapping_add(b[2]),
                    a[3].wrapping_add(b[3]),
                    a[4].wrapping_add(b[4]),
                    a[5].wrapping_add(b[5]),
                    a[6].wrapping_add(b[6]),
                    a[7].wrapping_add(b[7]),
                ],
            })
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0].wrapping_sub(b[0]),
                    a[1].wrapping_sub(b[1]),
                    a[2].wrapping_sub(b[2]),
                    a[3].wrapping_sub(b[3]),
                    a[4].wrapping_sub(b[4]),
                    a[5].wrapping_sub(b[5]),
                    a[6].wrapping_sub(b[6]),
                    a[7].wrapping_sub(b[7]),
                ],
            })
        }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0].wrapping_mul(b[0]),
                    a[1].wrapping_mul(b[1]),
                    a[2].wrapping_mul(b[2]),
                    a[3].wrapping_mul(b[3]),
                    a[4].wrapping_mul(b[4]),
                    a[5].wrapping_mul(b[5]),
                    a[6].wrapping_mul(b[6]),
                    a[7].wrapping_mul(b[7]),
                ],
            })
        }
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0] & b[0],
                    a[1] & b[1],
                    a[2] & b[2],
                    a[3] & b[3],
                    a[4] & b[4],
                    a[5] & b[5],
                    a[6] & b[6],
                    a[7] & b[7],
                ],
            })
        }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0] | b[0],
                    a[1] | b[1],
                    a[2] | b[2],
                    a[3] | b[3],
                    a[4] | b[4],
                    a[5] | b[5],
                    a[6] | b[6],
                    a[7] | b[7],
                ],
            })
        }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = self.0.u16;
            Self(ScalarReg {
                u16: [!a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7]],
            })
        }
    }

    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0] >> count,
                    a[1] >> count,
                    a[2] >> count,
                    a[3] >> count,
                    a[4] >> count,
                    a[5] >> count,
                    a[6] >> count,
                    a[7] >> count,
                ],
            })
        }
    }

    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0] << count,
                    a[1] << count,
                    a[2] << count,
                    a[3] << count,
                    a[4] << count,
                    a[5] << count,
                    a[6] << count,
                    a[7] << count,
                ],
            })
        }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            let m = mask.0.u16;
            Self(ScalarReg {
                u16: [
                    (a[0] & m[0]) | (b[0] & !m[0]),
                    (a[1] & m[1]) | (b[1] & !m[1]),
                    (a[2] & m[2]) | (b[2] & !m[2]),
                    (a[3] & m[3]) | (b[3] & !m[3]),
                    (a[4] & m[4]) | (b[4] & !m[4]),
                    (a[5] & m[5]) | (b[5] & !m[5]),
                    (a[6] & m[6]) | (b[6] & !m[6]),
                    (a[7] & m[7]) | (b[7] & !m[7]),
                ],
            })
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].min(b[2]),
                    a[3].min(b[3]),
                    a[4].min(b[4]),
                    a[5].min(b[5]),
                    a[6].min(b[6]),
                    a[7].min(b[7]),
                ],
            })
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0].max(b[0]),
                    a[1].max(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                    a[4].max(b[4]),
                    a[5].max(b[5]),
                    a[6].max(b[6]),
                    a[7].max(b[7]),
                ],
            })
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0].saturating_add(b[0]),
                    a[1].saturating_add(b[1]),
                    a[2].saturating_add(b[2]),
                    a[3].saturating_add(b[3]),
                    a[4].saturating_add(b[4]),
                    a[5].saturating_add(b[5]),
                    a[6].saturating_add(b[6]),
                    a[7].saturating_add(b[7]),
                ],
            })
        }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u16;
            let b = other.0.u16;
            Self(ScalarReg {
                u16: [
                    a[0].saturating_sub(b[0]),
                    a[1].saturating_sub(b[1]),
                    a[2].saturating_sub(b[2]),
                    a[3].saturating_sub(b[3]),
                    a[4].saturating_sub(b[4]),
                    a[5].saturating_sub(b[5]),
                    a[6].saturating_sub(b[6]),
                    a[7].saturating_sub(b[7]),
                ],
            })
        }
    }
}

// ============================================================================
// u8 Implementation (16 lanes)
// ============================================================================

impl SimdOps<u8> for SimdVec<u8> {
    #[inline(always)]
    fn splat(val: u8) -> Self {
        Self(ScalarReg { u8: [val; 16] })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 u8 values
        unsafe {
            Self(ScalarReg {
                u8: [
                    *ptr.offset(0),
                    *ptr.offset(1),
                    *ptr.offset(2),
                    *ptr.offset(3),
                    *ptr.offset(4),
                    *ptr.offset(5),
                    *ptr.offset(6),
                    *ptr.offset(7),
                    *ptr.offset(8),
                    *ptr.offset(9),
                    *ptr.offset(10),
                    *ptr.offset(11),
                    *ptr.offset(12),
                    *ptr.offset(13),
                    *ptr.offset(14),
                    *ptr.offset(15),
                ],
            })
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 u8 values
        unsafe {
            let arr = self.0.u8;
            for i in 0..16 {
                *ptr.offset(i) = arr[i as usize];
            }
        }
    }

    #[inline(always)]
    fn new(v0: u8, v1: u8, v2: u8, v3: u8) -> Self {
        Self(ScalarReg {
            u8: [v0, v1, v2, v3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_add(b[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_sub(b[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_mul(b[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i] & b[i];
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i] | b[i];
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = self.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = !a[i];
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i] >> count;
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i] << count;
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let m = mask.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = (a[i] & m[i]) | (b[i] & !m[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].min(b[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].max(b[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].saturating_add(b[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            let a = self.0.u8;
            let b = other.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].saturating_sub(b[i]);
            }
            Self(ScalarReg { u8: result })
        }
    }
}

// ============================================================================
// Bitcasting (Zero-Cost Type Conversion)
// ============================================================================

/// Bitcast between scalar types.
///
/// This just reinterprets the union field.
#[inline(always)]
pub fn cast<T, U>(v: SimdVec<T>) -> SimdVec<U> {
    SimdVec(v.0)
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
        let a = SimdVec::<u16>::splat(100);
        let b = SimdVec::<u16>::splat(2);
        let c = a.mul(b);

        let mut output = [0u16; 8];
        unsafe { c.store(output.as_mut_ptr()) };
        assert_eq!(output, [200; 8]);
    }

    #[test]
    fn test_select() {
        let fg = SimdVec::<u32>::splat(0xFFFFFFFF);
        let bg = SimdVec::<u32>::splat(0x00000000);
        let mask = SimdVec::<u32>::new(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000);
        let result = fg.select(bg, mask);

        let mut output = [0u32; 4];
        unsafe { result.store(output.as_mut_ptr()) };
        assert_eq!(output, [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000]);
    }
}
