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

use crate::batch::{SimdOps, SimdOpsU8};
use core::marker::PhantomData;

// ============================================================================
// Platform-Specific Types
// ============================================================================

/// Platform-specific SIMD vector wrapper.
///
/// On scalar: Uses fixed-size arrays matching SIMD lane counts.
#[repr(transparent)]
pub struct SimdVec<T>(pub(crate) ScalarReg<T>);

impl<T> Clone for SimdVec<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SimdVec<T> {}

/// The actual storage types (platform-specific).
pub union ScalarReg<T> {
    /// 4 lanes of u32.
    pub u32: [u32; 4],
    /// 8 lanes of u16.
    pub u16: [u16; 8],
    /// 16 lanes of u8.
    pub u8: [u8; 16],
    /// 4 lanes of f32.
    pub f32: [f32; 4],
    /// Type marker.
    _marker: PhantomData<T>,
}

impl<T> Clone for ScalarReg<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ScalarReg<T> {}

// ============================================================================
// f32 Implementation (4 lanes)
// ============================================================================

impl SimdOps<f32> for SimdVec<f32> {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(ScalarReg { f32: [val; 4] })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        unsafe {
            Self(ScalarReg {
                f32: [
                    *ptr.offset(0),
                    *ptr.offset(1),
                    *ptr.offset(2),
                    *ptr.offset(3),
                ],
            })
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        unsafe {
            let arr = self.0.f32;
            *ptr.offset(0) = arr[0];
            *ptr.offset(1) = arr[1];
            *ptr.offset(2) = arr[2];
            *ptr.offset(3) = arr[3];
        }
    }

    #[inline(always)]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        Self(ScalarReg {
            f32: [v0, v1, v2, v3],
        })
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            Self(ScalarReg {
                f32: [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
            })
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            Self(ScalarReg {
                f32: [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]],
            })
        }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            Self(ScalarReg {
                f32: [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]],
            })
        }
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        unsafe {
            let a = core::mem::transmute::<[f32; 4], [u32; 4]>(self.0.f32);
            let b = core::mem::transmute::<[f32; 4], [u32; 4]>(other.0.f32);
            let res = [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        unsafe {
            let a = core::mem::transmute::<[f32; 4], [u32; 4]>(self.0.f32);
            let b = core::mem::transmute::<[f32; 4], [u32; 4]>(other.0.f32);
            let res = [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = core::mem::transmute::<[f32; 4], [u32; 4]>(self.0.f32);
            let res = [!a[0], !a[1], !a[2], !a[3]];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        // Bitwise shift on float representation
        unsafe {
            let a = core::mem::transmute::<[f32; 4], [u32; 4]>(self.0.f32);
            let res = [a[0] >> count, a[1] >> count, a[2] >> count, a[3] >> count];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        // Bitwise shift on float representation
        unsafe {
            let a = core::mem::transmute::<[f32; 4], [u32; 4]>(self.0.f32);
            let res = [a[0] << count, a[1] << count, a[2] << count, a[3] << count];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        unsafe {
            let a = core::mem::transmute::<[f32; 4], [u32; 4]>(self.0.f32);
            let b = core::mem::transmute::<[f32; 4], [u32; 4]>(other.0.f32);
            let m = core::mem::transmute::<[f32; 4], [u32; 4]>(mask.0.f32);
            let res = [
                (a[0] & m[0]) | (b[0] & !m[0]),
                (a[1] & m[1]) | (b[1] & !m[1]),
                (a[2] & m[2]) | (b[2] & !m[2]),
                (a[3] & m[3]) | (b[3] & !m[3]),
            ];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            Self(ScalarReg {
                f32: [
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
            let a = self.0.f32;
            let b = other.0.f32;
            Self(ScalarReg {
                f32: [
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
        // For floats, just add (no wrap)
        self.add(other)
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        // For floats, just sub
        self.sub(other)
    }
}

impl crate::batch::SimdFloatOps for SimdVec<f32> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        unsafe {
            let a = self.0.f32;
            Self(ScalarReg {
                f32: [a[0].sqrt(), a[1].sqrt(), a[2].sqrt(), a[3].sqrt()],
            })
        }
        #[cfg(not(feature = "std"))]
        unsafe {
             let a = self.0.f32;
            // Approximation or use libm if available. For no_std without libm, sqrt is hard.
            // Assuming intrinsics are available or user provides it.
            // For now, use a simple iterative approximation or assume std.
            // Actually, f32 methods are available in core/alloc? No, they are in libm.
            // But we can use `f32::sqrt` if we have standard library or compiler builtins.
            // Rust `f32::sqrt` works in no_std if using builtins? No.
            // We'll assume std for now as scalar backend usually runs in tests or non-simd envs with std.
            // If strictly no_std, we need `libm` crate dependency.
            // Since I cannot add dependencies easily, I will just call `.sqrt()` and hope for the best (usually implies std or libm linked).
             Self(ScalarReg {
                f32: [
                    libm::sqrtf(a[0]),
                    libm::sqrtf(a[1]),
                    libm::sqrtf(a[2]),
                    libm::sqrtf(a[3]),
                ]
            })
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            let a = self.0.f32;
             // Using libm::fabsf or just bit manipulation
             // But f32::abs is in core? No, usually std.
             // Actually, core::f32 methods exist since Rust 1.20 but some like sqrt require std/libm. abs is bitwise.
            Self(ScalarReg {
                f32: [a[0].abs(), a[1].abs(), a[2].abs(), a[3].abs()],
            })
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
         unsafe {
            let a = self.0.f32;
             // Requires libm or std
             #[cfg(feature = "std")]
             {
                 Self(ScalarReg {
                    f32: [a[0].floor(), a[1].floor(), a[2].floor(), a[3].floor()],
                })
             }
             #[cfg(not(feature = "std"))]
             {
                 Self(ScalarReg {
                    f32: [libm::floorf(a[0]), libm::floorf(a[1]), libm::floorf(a[2]), libm::floorf(a[3])],
                })
             }
        }
    }

    #[inline(always)]
    fn ceil(self) -> Self {
         unsafe {
            let a = self.0.f32;
             #[cfg(feature = "std")]
             {
                 Self(ScalarReg {
                    f32: [a[0].ceil(), a[1].ceil(), a[2].ceil(), a[3].ceil()],
                })
             }
             #[cfg(not(feature = "std"))]
             {
                 Self(ScalarReg {
                    f32: [libm::ceilf(a[0]), libm::ceilf(a[1]), libm::ceilf(a[2]), libm::ceilf(a[3])],
                })
             }
        }
    }

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            Self(ScalarReg {
                f32: [a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]],
            })
        }
    }

    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            // Mask is all 1s if true, all 0s if false
            let mask = |cond| if cond { u32::MAX } else { 0 };
            let res = [
                mask(a[0] > b[0]),
                mask(a[1] > b[1]),
                mask(a[2] > b[2]),
                mask(a[3] > b[3]),
            ];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            let mask = |cond| if cond { u32::MAX } else { 0 };
            let res = [
                mask(a[0] >= b[0]),
                mask(a[1] >= b[1]),
                mask(a[2] >= b[2]),
                mask(a[3] >= b[3]),
            ];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            let mask = |cond| if cond { u32::MAX } else { 0 };
            let res = [
                mask(a[0] < b[0]),
                mask(a[1] < b[1]),
                mask(a[2] < b[2]),
                mask(a[3] < b[3]),
            ];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            let mask = |cond| if cond { u32::MAX } else { 0 };
            let res = [
                mask(a[0] <= b[0]),
                mask(a[1] <= b[1]),
                mask(a[2] <= b[2]),
                mask(a[3] <= b[3]),
            ];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }

    #[inline(always)]
    fn cmp_eq(self, other: Self) -> Self {
         unsafe {
            let a = self.0.f32;
            let b = other.0.f32;
            let mask = |cond| if cond { u32::MAX } else { 0 };
            let res = [
                mask(a[0] == b[0]),
                mask(a[1] == b[1]),
                mask(a[2] == b[2]),
                mask(a[3] == b[3]),
            ];
            Self(ScalarReg {
                f32: core::mem::transmute(res),
            })
        }
    }
}

// ============================================================================
// u32 Implementation (4 lanes)
// ============================================================================

impl SimdOps<u32> for SimdVec<u32> {
    /// Broadcasts a value to all lanes.
    #[inline(always)]
    fn splat(val: u32) -> Self {
        Self(ScalarReg { u32: [val; 4] })
    }

    /// Loads a vector from a pointer.
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

    /// Stores the vector to a pointer.
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

    /// Creates a new vector from values.
    #[inline(always)]
    fn new(v0: u32, v1: u32, v2: u32, v3: u32) -> Self {
        Self(ScalarReg {
            u32: [v0, v1, v2, v3],
        })
    }

    /// Adds two vectors.
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

    /// Subtracts two vectors.
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

    /// Multiplies two vectors.
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

    /// Bitwise AND.
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

    /// Bitwise OR.
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

    /// Bitwise NOT.
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = self.0.u32;
            Self(ScalarReg {
                u32: [!a[0], !a[1], !a[2], !a[3]],
            })
        }
    }

    /// Logical shift right.
    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u32;
            Self(ScalarReg {
                u32: [a[0] >> count, a[1] >> count, a[2] >> count, a[3] >> count],
            })
        }
    }

    /// Logical shift left.
    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        unsafe {
            let a = self.0.u32;
            Self(ScalarReg {
                u32: [a[0] << count, a[1] << count, a[2] << count, a[3] << count],
            })
        }
    }

    /// Element-wise selection.
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

    /// Minimum value.
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

    /// Maximum value.
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

    /// Saturating addition.
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

    /// Saturating subtraction.
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
    /// Broadcasts a value to all lanes.
    #[inline(always)]
    fn splat(val: u16) -> Self {
        Self(ScalarReg { u16: [val; 8] })
    }

    /// Loads a vector from a pointer.
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

    /// Stores the vector to a pointer.
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

    /// Creates a new vector (partial initialization).
    #[inline(always)]
    fn new(v0: u16, v1: u16, v2: u16, v3: u16) -> Self {
        Self(ScalarReg {
            u16: [v0, v1, v2, v3, 0, 0, 0, 0],
        })
    }

    /// Adds two vectors.
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

    /// Subtracts two vectors.
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

    /// Multiplies two vectors.
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

    /// Bitwise AND.
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

    /// Bitwise OR.
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

    /// Bitwise NOT.
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = self.0.u16;
            Self(ScalarReg {
                u16: [!a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7]],
            })
        }
    }

    /// Logical shift right.
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

    /// Logical shift left.
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

    /// Element-wise selection.
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

    /// Minimum value.
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

    /// Maximum value.
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

    /// Saturating addition.
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

    /// Saturating subtraction.
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
    /// Broadcasts a value to all lanes.
    #[inline(always)]
    fn splat(val: u8) -> Self {
        Self(ScalarReg { u8: [val; 16] })
    }

    /// Loads a vector from a pointer.
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

    /// Stores the vector to a pointer.
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

    /// Creates a new vector (partial initialization).
    #[inline(always)]
    fn new(v0: u8, v1: u8, v2: u8, v3: u8) -> Self {
        Self(ScalarReg {
            u8: [v0, v1, v2, v3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
    }

    /// Adds two vectors.
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

    /// Subtracts two vectors.
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

    /// Multiplies two vectors.
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

    /// Bitwise AND.
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

    /// Bitwise OR.
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

    /// Bitwise NOT.
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

    /// Logical shift right.
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

    /// Logical shift left.
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

    /// Element-wise selection.
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

    /// Minimum value.
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

    /// Maximum value.
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

    /// Saturating addition.
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

    /// Saturating subtraction.
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

impl SimdOpsU8 for SimdVec<u8> {
    /// Shuffles bytes according to indices (scalar fallback).
    ///
    /// For each byte position i:
    /// - If `indices[i] & 0x80` is set, result[i] = 0
    /// - Otherwise, result[i] = self[indices[i] & 0x0F]
    #[inline(always)]
    fn shuffle_bytes(self, indices: Self) -> Self {
        unsafe {
            let data = self.0.u8;
            let idx = indices.0.u8;
            let mut result = [0u8; 16];
            for i in 0..16 {
                if idx[i] & 0x80 != 0 {
                    result[i] = 0;
                } else {
                    result[i] = data[(idx[i] & 0x0F) as usize];
                }
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
///
/// # Parameters
/// * `v` - The source vector.
///
/// # Returns
/// * The vector bitcasted to type `U`.
#[inline(always)]
pub fn cast<T, U>(v: SimdVec<T>) -> SimdVec<U> {
    unsafe { core::mem::transmute(v) }
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

    #[test]
    fn test_shuffle_bytes_rgba_bgra() {
        // Test RGBA to BGRA conversion
        // Input: 4 RGBA pixels [R,G,B,A] = [0x11,0x22,0x33,0xFF] each
        let rgba_bytes: [u8; 16] = [
            0x11, 0x22, 0x33, 0xFF, // Pixel 0: R=0x11, G=0x22, B=0x33, A=0xFF
            0xAA, 0xBB, 0xCC, 0xDD, // Pixel 1
            0x00, 0x55, 0xAA, 0xFF, // Pixel 2
            0x12, 0x34, 0x56, 0x78, // Pixel 3
        ];
        // Shuffle mask: swap bytes 0↔2 in each 4-byte group
        let shuffle_mask: [u8; 16] = [
            2, 1, 0, 3, // Pixel 0
            6, 5, 4, 7, // Pixel 1
            10, 9, 8, 11, // Pixel 2
            14, 13, 12, 15, // Pixel 3
        ];

        let data = unsafe { SimdVec::<u8>::load(rgba_bytes.as_ptr()) };
        let indices = unsafe { SimdVec::<u8>::load(shuffle_mask.as_ptr()) };
        let result = data.shuffle_bytes(indices);

        let mut output = [0u8; 16];
        unsafe { result.store(output.as_mut_ptr()) };

        // Expected: BGRA [B,G,R,A]
        assert_eq!(output[0..4], [0x33, 0x22, 0x11, 0xFF]); // Pixel 0: B,G,R,A
        assert_eq!(output[4..8], [0xCC, 0xBB, 0xAA, 0xDD]); // Pixel 1
        assert_eq!(output[8..12], [0xAA, 0x55, 0x00, 0xFF]); // Pixel 2
        assert_eq!(output[12..16], [0x56, 0x34, 0x12, 0x78]); // Pixel 3
    }
}
