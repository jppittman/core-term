//! ARM NEON backend (4 lanes for u32/f32).

use crate::backend::{Backend, BatchArithmetic, FloatBatchOps, SimdBatch};
use core::any;
use core::arch::aarch64::*;
use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::ops::*;

/// NEON Backend (4 lanes for u32/f32).
#[derive(Copy, Clone, Debug, Default)]
pub struct Neon;

impl Backend for Neon {
    const LANES: usize = 4;
    type Batch<T: Copy + Debug + Default + Send + Sync + 'static + core::cmp::PartialEq> = SimdVec<T>;

    #[inline(always)]
    fn downcast_u32_to_u8(b: SimdVec<u32>) -> SimdVec<u8> {
        unsafe {
            // Pack u32x4 -> u16x4 -> u8x8 -> u8x16
            // 1. Narrow u32x4 to u16x4 (saturating)
            let u16_low = vqmovn_u32(b.0.u32);
            // 2. Expand to u16x8 (low half valid, high half zero)
            let u16_combined = vcombine_u16(u16_low, vdup_n_u16(0));
            // 3. Narrow u16x8 to u8x8 (saturating)
            let u8_low = vqmovn_u16(u16_combined);
            // 4. Expand to u8x16 (low half valid)
            let u8_combined = vcombine_u8(u8_low, vdup_n_u8(0));

            SimdVec(NeonReg { u8: u8_combined }, PhantomData)
        }
    }

    #[inline(always)]
    fn upcast_u8_to_u32(b: SimdVec<u8>) -> SimdVec<u32> {
        // Zero-extend the low 4 bytes of the u8x16 to u32x4.
        // u8x16 → u16x8 (low half) → u32x4 (low half)
        unsafe {
            let u8x16 = b.0.u8;
            let u16x8 = vmovl_u8(vget_low_u8(u8x16)); // u8x8 → u16x8
            let u32x4 = vmovl_u16(vget_low_u16(u16x8)); // u16x4 → u32x4
            SimdVec(NeonReg { u32: u32x4 }, PhantomData)
        }
    }

    #[inline(always)]
    fn u32_to_f32(b: SimdVec<u32>) -> SimdVec<f32> {
        unsafe {
            SimdVec(
                NeonReg {
                    f32: vcvtq_f32_u32(b.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn f32_to_u32(b: SimdVec<f32>) -> SimdVec<u32> {
        unsafe {
            SimdVec(
                NeonReg {
                    u32: vcvtq_u32_f32(b.0.f32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn transmute_u32_to_f32(b: SimdVec<u32>) -> SimdVec<f32> {
        unsafe { b.transmute() }
    }

    #[inline(always)]
    fn transmute_f32_to_u32(b: SimdVec<f32>) -> SimdVec<u32> {
        unsafe { b.transmute() }
    }
}

/// Platform-specific SIMD vector wrapper.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SimdVec<T: Copy>(pub(crate) NeonReg<T>, PhantomData<T>);

/// The actual NEON register - uses different types for different element sizes.
#[derive(Copy, Clone)]
#[repr(C)]
pub union NeonReg<T> {
    pub u32: uint32x4_t,
    pub u16: uint16x8_t,
    pub u8: uint8x16_t,
    pub f32: float32x4_t,
    pub i32: int32x4_t,
    _marker: PhantomData<T>,
}

impl<T: Copy> Default for SimdVec<T> {
    fn default() -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vdupq_n_u32(0),
                },
                PhantomData,
            )
        }
    }
}

impl<T: Copy> Debug for SimdVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "SimdVec({:?})", any::type_name::<T>())
    }
}

// Inherent methods for compatibility
impl<T: Copy> SimdVec<T> {
    #[inline(always)]
    pub unsafe fn transmute<U: Copy>(self) -> SimdVec<U> {
        unsafe { core::mem::transmute_copy(&self) }
    }

    /// Reinterpret the bits of this batch as another type of the same size.
    /// Preserves the bit pattern, not the numeric value.
    #[inline(always)]
    pub fn bitcast<U: Copy>(self) -> SimdVec<U> {
        unsafe { self.transmute() }
    }

    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut T) {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                vst1q_u32(ptr as *mut u32, self.0.u32)
            } else if core::mem::size_of::<T>() == 2 {
                // Store 4x u16 (64 bits) from low half
                let low = vget_low_u16(self.0.u16);
                vst1_u16(ptr as *mut u16, low)
            } else if core::mem::size_of::<T>() == 1 {
                // Store 4x u8 (32 bits) -> interpreted as 1x u32
                // We use lane 0 of the u32 view, which corresponds to bytes 0..3
                vst1q_lane_u32(ptr as *mut u32, self.0.u32, 0)
            }
        }
    }
}

impl SimdVec<u32> {
    #[inline(always)]
    pub fn splat(val: u32) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vdupq_n_u32(val),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    pub fn new(v0: u32, v1: u32, v2: u32, v3: u32) -> Self {
        unsafe {
            let arr = [v0, v1, v2, v3];
            Self(
                NeonReg {
                    u32: vld1q_u32(arr.as_ptr()),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        unsafe { core::mem::transmute(self) }
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 4]) -> Self {
        unsafe { core::mem::transmute(arr) }
    }

    /// Convert to f32, preserving the numeric value.
    #[inline(always)]
    pub fn to_f32(self) -> SimdVec<f32> {
        Neon::u32_to_f32(self)
    }
}

impl SimdVec<u16> {
    #[inline(always)]
    pub fn splat(val: u16) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vdupq_n_u16(val),
                },
                PhantomData,
            )
        }
    }
}

impl SimdVec<u8> {
    #[inline(always)]
    pub fn splat(val: u8) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vdupq_n_u8(val),
                },
                PhantomData,
            )
        }
    }
}

impl SimdVec<f32> {
    #[inline(always)]
    pub fn splat(val: f32) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vdupq_n_f32(val),
                },
                PhantomData,
            )
        }
    }

    /// Convert to u32, preserving the numeric value (truncating).
    #[inline(always)]
    pub fn to_u32(self) -> SimdVec<u32> {
        Neon::f32_to_u32(self)
    }
}

// ============================================================================
// u32 Implementation
// ============================================================================

impl Add for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vaddq_u32(self.0.u32, rhs.0.u32),
                },
                PhantomData,
            )
        }
    }
}

impl Add<u32> for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u32) -> Self {
        self + Self::splat(rhs)
    }
}

impl Sub for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vsubq_u32(self.0.u32, rhs.0.u32),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vmulq_u32(self.0.u32, rhs.0.u32),
                },
                PhantomData,
            )
        }
    }
}

impl Mul<u32> for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: u32) -> Self {
        self * Self::splat(rhs)
    }
}

impl Div for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // NEON doesn't have integer division - fall back to scalar
        unsafe {
            let mut a = [0u32; 4];
            let mut b = [0u32; 4];
            vst1q_u32(a.as_mut_ptr(), self.0.u32);
            vst1q_u32(b.as_mut_ptr(), rhs.0.u32);
            let res = [
                if b[0] != 0 { a[0] / b[0] } else { 0 },
                if b[1] != 0 { a[1] / b[1] } else { 0 },
                if b[2] != 0 { a[2] / b[2] } else { 0 },
                if b[3] != 0 { a[3] / b[3] } else { 0 },
            ];
            Self(
                NeonReg {
                    u32: vld1q_u32(res.as_ptr()),
                },
                PhantomData,
            )
        }
    }
}

impl BitAnd for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vandq_u32(self.0.u32, rhs.0.u32),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vorrq_u32(self.0.u32, rhs.0.u32),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: veorq_u32(self.0.u32, rhs.0.u32),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vmvnq_u32(self.0.u32),
                },
                PhantomData,
            )
        }
    }
}

impl Shl<i32> for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        unsafe {
            let shift = vdupq_n_s32(rhs);
            Self(
                NeonReg {
                    u32: vshlq_u32(self.0.u32, shift),
                },
                PhantomData,
            )
        }
    }
}

impl Shr<i32> for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        unsafe {
            let shift = vdupq_n_s32(-rhs);
            Self(
                NeonReg {
                    u32: vshlq_u32(self.0.u32, shift),
                },
                PhantomData,
            )
        }
    }
}

// Generic SimdBatch implementation
impl<T: Copy + Debug + Default + PartialEq + Send + Sync + 'static> SimdBatch<T> for SimdVec<T> {
    const LANES: usize = 4;

    fn splat(val: T) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let v: u32 = core::mem::transmute_copy(&val);
                Self(
                    NeonReg {
                        u32: vdupq_n_u32(v),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 2 {
                let v: u16 = core::mem::transmute_copy(&val);
                Self(
                    NeonReg {
                        u16: vdupq_n_u16(v),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 1 {
                let v: u8 = core::mem::transmute_copy(&val);
                Self(NeonReg { u8: vdupq_n_u8(v) }, PhantomData)
            } else {
                Self(
                    NeonReg {
                        u32: vdupq_n_u32(0),
                    },
                    PhantomData,
                )
            }
        }
    }

    fn sequential_from(start: T) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let s: u32 = core::mem::transmute_copy(&start);
                let arr = [s, s + 1, s + 2, s + 3];
                Self(
                    NeonReg {
                        u32: vld1q_u32(arr.as_ptr()),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 1 {
                let s: u8 = core::mem::transmute_copy(&start);
                let mut arr = [0u8; 16];
                for i in 0..16 {
                    arr[i] = s.wrapping_add(i as u8);
                }
                Self(
                    NeonReg {
                        u8: vld1q_u8(arr.as_ptr()),
                    },
                    PhantomData,
                )
            } else {
                Self(
                    NeonReg {
                        u32: vdupq_n_u32(0),
                    },
                    PhantomData,
                )
            }
        }
    }

    #[inline(always)]
    fn load(slice: &[T]) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                Self(
                    NeonReg {
                        u32: vld1q_u32(slice.as_ptr() as *const u32),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 2 {
                // Load 4x u16 (64 bits) -> low half of vector
                let ptr = slice.as_ptr() as *const u16;
                let low = vld1_u16(ptr);
                // Combine with zero high half
                let zero = vdup_n_u16(0);
                Self(
                    NeonReg {
                        u16: vcombine_u16(low, zero),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 1 {
                // Load 4x u8 (32 bits) -> interpreted as 1x u32
                let ptr = slice.as_ptr() as *const u32;
                // Load single u32 into lane 0, other lanes zero
                let zero = vdupq_n_u32(0);
                let val = vld1q_lane_u32(ptr, zero, 0);
                Self(NeonReg { u32: val }, PhantomData)
            } else {
                Self(
                    NeonReg {
                        u32: vdupq_n_u32(0),
                    },
                    PhantomData,
                )
            }
        }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [T]) {
        // Delegate to inherent store which handles safe partial storage
        unsafe {
            SimdVec::store(*self, slice.as_mut_ptr());
        }
    }

    fn first(&self) -> T {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let v = vgetq_lane_u32(self.0.u32, 0);
                core::mem::transmute_copy(&v)
            } else if core::mem::size_of::<T>() == 2 {
                let v = vgetq_lane_u16(self.0.u16, 0);
                core::mem::transmute_copy(&v)
            } else if core::mem::size_of::<T>() == 1 {
                let v = vgetq_lane_u8(self.0.u8, 0);
                core::mem::transmute_copy(&v)
            } else {
                T::default()
            }
        }
    }

    fn any(&self) -> bool {
        // Use max across lanes to check if any is non-zero
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let max = vmaxvq_u32(self.0.u32);
                max != 0
            } else if core::mem::size_of::<T>() == 2 {
                // maxv for u16 requires wrapping because NEON intrinsic naming is inconsistent
                // vmaxvq_u16 doesn't exist in Rust's aarch64 module? It should.
                // Checking docs... neon::vmaxvq_u16
                // Assuming it exists or we use fallback
                let vec_u16: SimdVec<u16> = core::mem::transmute_copy(self);
                let arr = vec_u16.to_array_u16();
                arr.iter().any(|&x| x != 0)
            } else if core::mem::size_of::<T>() == 1 {
                let max = vmaxvq_u8(self.0.u8);
                max != 0
            } else {
                false
            }
        }
    }

    fn all(&self) -> bool {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let min = vminvq_u32(self.0.u32);
                min != 0
            } else if core::mem::size_of::<T>() == 2 {
                let vec_u16: SimdVec<u16> = core::mem::transmute_copy(self);
                let arr = vec_u16.to_array_u16();
                arr.iter().all(|&x| x != 0)
            } else if core::mem::size_of::<T>() == 1 {
                let min = vminvq_u8(self.0.u8);
                min != 0
            } else {
                false
            }
        }
    }

    fn extract_lane(&self, lane: usize) -> T {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let mut arr = [0u32; 4];
                vst1q_u32(arr.as_mut_ptr(), self.0.u32);
                let v = arr[lane];
                core::mem::transmute_copy(&v)
            } else if core::mem::size_of::<T>() == 2 {
                let mut arr = [0u16; 8];
                vst1q_u16(arr.as_mut_ptr(), self.0.u16);
                let v = arr[lane];
                core::mem::transmute_copy(&v)
            } else if core::mem::size_of::<T>() == 1 {
                let mut arr = [0u8; 16];
                vst1q_u8(arr.as_mut_ptr(), self.0.u8);
                let v = arr[lane];
                core::mem::transmute_copy(&v)
            } else {
                T::default()
            }
        }
    }
}

impl SimdVec<u16> {
    #[inline(always)]
    fn to_array_u16(self) -> [u16; 8] {
        unsafe { core::mem::transmute(self) }
    }
}

impl BatchArithmetic<u32> for SimdVec<u32> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vbslq_u32(self.0.u32, if_true.0.u32, if_false.0.u32),
                },
                PhantomData,
            )
        }
    }

    fn gather(base: &[u32], indices: Self) -> Self {
        let idx = indices.to_array();
        Self::from_array(idx.map(|i| base.get(i as usize).copied().unwrap_or(0)))
    }

    fn gather_u8(base: &[u8], indices: Self) -> Self {
        let idx = indices.to_array();
        Self::from_array(idx.map(|i| base.get(i as usize).copied().unwrap_or(0) as u32))
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vminq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vmaxq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vqaddq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vqsubq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vceqq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let eq = vceqq_u32(self.0.u32, other.0.u32);
            Self(NeonReg { u32: vmvnq_u32(eq) }, PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vcltq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vcleq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vcgtq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u32: vcgeq_u32(self.0.u32, other.0.u32),
                },
                PhantomData,
            )
        }
    }
}

// ============================================================================
// u16 Implementation
// ============================================================================

impl Add for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vaddq_u16(self.0.u16, rhs.0.u16),
                },
                PhantomData,
            )
        }
    }
}

impl Sub for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vsubq_u16(self.0.u16, rhs.0.u16),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vmulq_u16(self.0.u16, rhs.0.u16),
                },
                PhantomData,
            )
        }
    }
}

impl Div for SimdVec<u16> {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        unimplemented!("u16 div")
    }
}

impl BitAnd for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vandq_u16(self.0.u16, rhs.0.u16),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vorrq_u16(self.0.u16, rhs.0.u16),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: veorq_u16(self.0.u16, rhs.0.u16),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vmvnq_u16(self.0.u16),
                },
                PhantomData,
            )
        }
    }
}

impl Shl<i32> for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        unsafe {
            let shift = vdupq_n_s16(rhs as i16);
            Self(
                NeonReg {
                    u16: vshlq_u16(self.0.u16, shift),
                },
                PhantomData,
            )
        }
    }
}

impl Shr<i32> for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        unsafe {
            let shift = vdupq_n_s16(-rhs as i16);
            Self(
                NeonReg {
                    u16: vshlq_u16(self.0.u16, shift),
                },
                PhantomData,
            )
        }
    }
}

impl BatchArithmetic<u16> for SimdVec<u16> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vbslq_u16(self.0.u16, if_true.0.u16, if_false.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn gather(_base: &[u16], _indices: Self) -> Self {
        unimplemented!("u16 gather")
    }

    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vminq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vmaxq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vqaddq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vqsubq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vceqq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let eq = vceqq_u16(self.0.u16, other.0.u16);
            Self(NeonReg { u16: vmvnq_u16(eq) }, PhantomData)
        }
    }

    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vcltq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vcleq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vcgtq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }

    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u16: vcgeq_u16(self.0.u16, other.0.u16),
                },
                PhantomData,
            )
        }
    }
}

// ============================================================================
// u8 Implementation
// ============================================================================

impl Add for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vaddq_u8(self.0.u8, rhs.0.u8),
                },
                PhantomData,
            )
        }
    }
}

impl Sub for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vsubq_u8(self.0.u8, rhs.0.u8),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vmulq_u8(self.0.u8, rhs.0.u8),
                },
                PhantomData,
            )
        }
    }
}

impl Div for SimdVec<u8> {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        unimplemented!("u8 div")
    }
}

impl BitAnd for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vandq_u8(self.0.u8, rhs.0.u8),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vorrq_u8(self.0.u8, rhs.0.u8),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: veorq_u8(self.0.u8, rhs.0.u8),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vmvnq_u8(self.0.u8),
                },
                PhantomData,
            )
        }
    }
}

impl Shl<i32> for SimdVec<u8> {
    type Output = Self;
    fn shl(self, _rhs: i32) -> Self {
        unimplemented!("u8 shl")
    }
}

impl Shr<i32> for SimdVec<u8> {
    type Output = Self;
    fn shr(self, _rhs: i32) -> Self {
        unimplemented!("u8 shr")
    }
}

impl BatchArithmetic<u8> for SimdVec<u8> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vbslq_u8(self.0.u8, if_true.0.u8, if_false.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn gather(base: &[u8], indices: Self) -> Self {
        unsafe {
            let mut idx = [0u8; 16];
            vst1q_u8(idx.as_mut_ptr(), indices.0.u8);
            let mut res = [0u8; 16];
            for i in 0..16 {
                res[i] = base.get(idx[i] as usize).copied().unwrap_or(0);
            }
            Self(
                NeonReg {
                    u8: vld1q_u8(res.as_ptr()),
                },
                PhantomData,
            )
        }
    }

    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vminq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vmaxq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vqaddq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vqsubq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vceqq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let eq = vceqq_u8(self.0.u8, other.0.u8);
            Self(NeonReg { u8: vmvnq_u8(eq) }, PhantomData)
        }
    }

    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vcltq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vcleq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vcgtq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }

    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    u8: vcgeq_u8(self.0.u8, other.0.u8),
                },
                PhantomData,
            )
        }
    }
}

// ============================================================================
// f32 Implementation
// ============================================================================

impl Add for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vaddq_f32(self.0.f32, rhs.0.f32),
                },
                PhantomData,
            )
        }
    }
}

impl Sub for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vsubq_f32(self.0.f32, rhs.0.f32),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vmulq_f32(self.0.f32, rhs.0.f32),
                },
                PhantomData,
            )
        }
    }
}

impl Div for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vdivq_f32(self.0.f32, rhs.0.f32),
                },
                PhantomData,
            )
        }
    }
}

impl BitAnd for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0.f32);
            let b = vreinterpretq_u32_f32(rhs.0.f32);
            Self(
                NeonReg {
                    f32: vreinterpretq_f32_u32(vandq_u32(a, b)),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0.f32);
            let b = vreinterpretq_u32_f32(rhs.0.f32);
            Self(
                NeonReg {
                    f32: vreinterpretq_f32_u32(vorrq_u32(a, b)),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0.f32);
            let b = vreinterpretq_u32_f32(rhs.0.f32);
            Self(
                NeonReg {
                    f32: vreinterpretq_f32_u32(veorq_u32(a, b)),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0.f32);
            Self(
                NeonReg {
                    f32: vreinterpretq_f32_u32(vmvnq_u32(a)),
                },
                PhantomData,
            )
        }
    }
}

impl Shl<i32> for SimdVec<f32> {
    type Output = Self;
    fn shl(self, _rhs: i32) -> Self {
        unimplemented!()
    }
}

impl Shr<i32> for SimdVec<f32> {
    type Output = Self;
    fn shr(self, _rhs: i32) -> Self {
        unimplemented!()
    }
}

impl BatchArithmetic<f32> for SimdVec<f32> {
    /// Selects values based on a mask.
    ///
    /// The mask (self) contains comparison results where each lane is either:
    /// - `0xFFFFFFFF` (all bits set) for true
    /// - `0x00000000` (all bits clear) for false
    ///
    /// These mask bits are stored in the f32 field of the union (reinterpreted from u32),
    /// allowing type-safe integration with comparison operations.
    ///
    /// # Behavior
    /// For each SIMD lane:
    /// - If mask bit is 1, select from `if_true`
    /// - If mask bit is 0, select from `if_false`
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            let mask = vreinterpretq_u32_f32(self.0.f32);
            let t = vreinterpretq_u32_f32(if_true.0.f32);
            let f = vreinterpretq_u32_f32(if_false.0.f32);
            Self(
                NeonReg {
                    f32: vreinterpretq_f32_u32(vbslq_u32(mask, t, f)),
                },
                PhantomData,
            )
        }
    }

    fn gather(_base: &[f32], _indices: Self) -> Self {
        unimplemented!()
    }

    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vminq_f32(self.0.f32, other.0.f32),
                },
                PhantomData,
            )
        }
    }

    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vmaxq_f32(self.0.f32, other.0.f32),
                },
                PhantomData,
            )
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        self + other
    }
    fn saturating_sub(self, other: Self) -> Self {
        self - other
    }

    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            let mask = vceqq_f32(self.0.f32, other.0.f32);
            Self(
                NeonReg {
                    u32: mask,  // Store mask in u32 field, not f32!
                },
                PhantomData,
            )
        }
    }

    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let eq = vceqq_f32(self.0.f32, other.0.f32);
            Self(
                NeonReg {
                    u32: vmvnq_u32(eq),  // Store mask in u32 field, not f32!
                },
                PhantomData,
            )
        }
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane contains:
    /// - `0xFFFFFFFF` (all bits set) if `self[i] < other[i]`
    /// - `0x00000000` (all bits clear) if `self[i] >= other[i]`
    ///
    /// # Implementation Note
    /// The ARM NEON `vcltq_f32` intrinsic returns a `uint32x4_t` mask. We store this
    /// mask in the u32 field of our union. When cast to `Batch<f32>`, the entire union
    /// is transmuted, preserving these mask bits in the f32 representation. This allows
    /// type-safe chaining of comparisons with select operations.
    ///
    /// # Example
    /// ```ignore
    /// let a = Batch::<f32>::splat(1.0);
    /// let b = Batch::<f32>::splat(2.0);
    /// let mask = a.cmp_lt(b);  // Returns all-ones mask
    /// let result = mask.select(white, black);  // Selects white
    /// ```
    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            let mask = vcltq_f32(self.0.f32, other.0.f32);
            Self(
                NeonReg {
                    u32: mask,  // Store mask in u32 field, not f32!
                },
                PhantomData,
            )
        }
    }

    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            let mask = vcleq_f32(self.0.f32, other.0.f32);
            Self(
                NeonReg {
                    u32: mask,  // Store mask in u32 field, not f32!
                },
                PhantomData,
            )
        }
    }

    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            let mask = vcgtq_f32(self.0.f32, other.0.f32);
            Self(
                NeonReg {
                    u32: mask,  // Store mask in u32 field, not f32!
                },
                PhantomData,
            )
        }
    }

    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            let mask = vcgeq_f32(self.0.f32, other.0.f32);
            Self(
                NeonReg {
                    u32: mask,  // Store mask in u32 field, not f32!
                },
                PhantomData,
            )
        }
    }
}

impl FloatBatchOps for SimdVec<f32> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vsqrtq_f32(self.0.f32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            Self(
                NeonReg {
                    f32: vabsq_f32(self.0.f32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        // IEEE 754 bit hack + polynomial correction
        // log2(x) = exponent + log2(mantissa) where mantissa in [1, 2)
        unsafe {
            let bits = vreinterpretq_u32_f32(self.0.f32);

            // Extract exponent: ((bits >> 23) & 0xFF) - 127
            let exp_bits = vshrq_n_u32(bits, 23);
            let exp_biased = vandq_u32(exp_bits, vdupq_n_u32(0xFF));
            let exp_i32 = vsubq_s32(vreinterpretq_s32_u32(exp_biased), vdupq_n_s32(127));
            let exp_f32 = vcvtq_f32_s32(exp_i32);

            // Extract mantissa, normalize to [1, 2): (bits & 0x7FFFFF) | 0x3F800000
            let mantissa_bits = vorrq_u32(
                vandq_u32(bits, vdupq_n_u32(0x007FFFFF)),
                vdupq_n_u32(0x3F800000),
            );
            let m = vreinterpretq_f32_u32(mantissa_bits);

            // Polynomial approximation for log2(m) where m in [1, 2)
            // log2(m) ≈ x * P(x) where x = m - 1, x ∈ [0, 1)
            // Using degree-5 minimax polynomial for <0.1% error
            let x = vsubq_f32(m, vdupq_n_f32(1.0));

            // Minimax coefficients for log2(1+x) ≈ x * (c0 + x*(c1 + x*(c2 + x*(c3 + x*c4))))
            let c0 = vdupq_n_f32(1.4426950216);
            let c1 = vdupq_n_f32(-0.7211452817);
            let c2 = vdupq_n_f32(0.4778098810);
            let c3 = vdupq_n_f32(-0.3473593140);
            let c4 = vdupq_n_f32(0.1533194870);

            // Horner's method: c0 + x*(c1 + x*(c2 + x*(c3 + x*c4)))
            let poly = vfmaq_f32(c3, x, c4);
            let poly = vfmaq_f32(c2, x, poly);
            let poly = vfmaq_f32(c1, x, poly);
            let poly = vfmaq_f32(c0, x, poly);
            let log2_mantissa = vmulq_f32(x, poly);

            // Result = exponent + log2(mantissa)
            let result = vaddq_f32(exp_f32, log2_mantissa);

            Self(NeonReg { f32: result }, PhantomData)
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        // 2^x = 2^floor(x) * 2^frac(x)
        // 2^floor(x) via bit manipulation, 2^frac(x) via polynomial
        unsafe {
            let x = self.0.f32;

            // Clamp to avoid overflow/underflow
            let x = vmaxq_f32(x, vdupq_n_f32(-126.0));
            let x = vminq_f32(x, vdupq_n_f32(127.0));

            // Split into integer and fractional parts
            let floor_x = vrndmq_f32(x); // floor
            let frac = vsubq_f32(x, floor_x);

            // 2^frac using polynomial (frac in [0, 1))
            // 2^f ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
            // Degree-5 minimax polynomial for <0.1% error
            let c1 = vdupq_n_f32(0.6931471806);
            let c2 = vdupq_n_f32(0.2402265070);
            let c3 = vdupq_n_f32(0.0555041086);
            let c4 = vdupq_n_f32(0.0096181291);
            let c5 = vdupq_n_f32(0.0013333558);

            // Horner's method
            let poly = vfmaq_f32(c4, frac, c5);
            let poly = vfmaq_f32(c3, frac, poly);
            let poly = vfmaq_f32(c2, frac, poly);
            let poly = vfmaq_f32(c1, frac, poly);
            let poly = vfmaq_f32(vdupq_n_f32(1.0), frac, poly);

            // 2^floor(x) by adding floor(x) to exponent bits
            // (floor_x + 127) << 23
            let floor_i32 = vcvtq_s32_f32(floor_x);
            let exp_bits = vshlq_n_u32(
                vreinterpretq_u32_s32(vaddq_s32(floor_i32, vdupq_n_s32(127))),
                23,
            );
            let pow2_int = vreinterpretq_f32_u32(exp_bits);

            // Result = 2^floor(x) * 2^frac(x)
            let result = vmulq_f32(pow2_int, poly);

            Self(NeonReg { f32: result }, PhantomData)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimdBatch;

    #[test]
    fn test_u32_add() {
        let a = SimdVec::<u32>::splat(100);
        let b = SimdVec::<u32>::splat(50);
        let c = a + b;

        let mut output = [0u32; 4];
        SimdBatch::store(&c, &mut output);
        assert_eq!(output, [150; 4]);
    }

    #[test]
    fn test_f32_sqrt() {
        let a = SimdVec::<f32>::splat(4.0);
        let b = a.sqrt();

        let mut output = [0.0f32; 4];
        SimdBatch::store(&b, &mut output);
        assert_eq!(output, [2.0; 4]);
    }

    #[test]
    fn test_sequential_from() {
        let seq = <SimdVec<u32> as SimdBatch<u32>>::sequential_from(10);
        let mut output = [0u32; 4];
        SimdBatch::store(&seq, &mut output);
        assert_eq!(output, [10, 11, 12, 13]);
    }

    #[test]
    fn test_log2_powers_of_two() {
        use crate::backend::FloatBatchOps;
        // log2(2^n) should equal n
        for n in 0..10 {
            let x = SimdVec::<f32>::splat(2.0f32.powi(n));
            let result = x.log2();
            let mut output = [0.0f32; 4];
            SimdBatch::store(&result, &mut output);
            let error = (output[0] - n as f32).abs();
            assert!(error < 0.01, "log2(2^{}) = {}, expected {}", n, output[0], n);
        }
    }

    #[test]
    fn test_exp2_integers() {
        use crate::backend::FloatBatchOps;
        // 2^n should be exact for small integers
        for n in 0..10 {
            let x = SimdVec::<f32>::splat(n as f32);
            let result = x.exp2();
            let mut output = [0.0f32; 4];
            SimdBatch::store(&result, &mut output);
            let expected = 2.0f32.powi(n);
            let error = (output[0] - expected).abs() / expected;
            assert!(error < 0.01, "2^{} = {}, expected {}", n, output[0], expected);
        }
    }

    #[test]
    fn test_pow_accuracy() {
        use crate::backend::FloatBatchOps;
        // Test various pow combinations
        let test_cases = [
            (2.0f32, 3.0f32, 8.0f32),
            (3.0, 2.0, 9.0),
            (4.0, 0.5, 2.0),
            (8.0, 1.0 / 3.0, 2.0),
            (10.0, 2.0, 100.0),
        ];
        for (base, exp, expected) in test_cases {
            let x = SimdVec::<f32>::splat(base);
            let y = SimdVec::<f32>::splat(exp);
            let result = x.pow(y);
            let mut output = [0.0f32; 4];
            SimdBatch::store(&result, &mut output);
            let error = (output[0] - expected).abs() / expected;
            assert!(error < 0.02, "{}^{} = {}, expected {}", base, exp, output[0], expected);
        }
    }

    #[test]
    fn test_gamma_correction_roundtrip() {
        use crate::backend::FloatBatchOps;
        // sRGB gamma is ~2.2, test that x^2.2^(1/2.2) ≈ x
        let gamma = 2.2f32;
        let inv_gamma = 1.0 / gamma;

        for i in 1..10 {
            let x = i as f32 / 10.0; // 0.1 to 0.9
            let input = SimdVec::<f32>::splat(x);
            let gamma_exp = SimdVec::<f32>::splat(gamma);
            let inv_gamma_exp = SimdVec::<f32>::splat(inv_gamma);

            let result = input.pow(gamma_exp).pow(inv_gamma_exp);
            let mut output = [0.0f32; 4];
            SimdBatch::store(&result, &mut output);
            let error = (output[0] - x).abs();
            assert!(error < 0.01, "Gamma roundtrip: {} -> {} (error {})", x, output[0], error);
        }
    }

    /// Helper to extract u32 bits from f32 batch (for testing mask values)
    fn extract_u32_bits(batch: SimdVec<f32>) -> [u32; 4] {
        unsafe {
            let mut output = [0u32; 4];
            // Read the f32 field as u32 to see mask bits
            vst1q_u32(output.as_mut_ptr(), vreinterpretq_u32_f32(batch.0.f32));
            output
        }
    }

    #[test]
    fn test_f32_cmp_eq() {
        use crate::backend::BatchArithmetic;

        // Test equality
        let a = SimdVec::<f32>::splat(1.0);
        let b = SimdVec::<f32>::splat(1.0);
        let result = a.cmp_eq(b);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "1.0 == 1.0 should produce all-ones mask");

        // Test inequality
        let c = SimdVec::<f32>::splat(1.0);
        let d = SimdVec::<f32>::splat(2.0);
        let result = c.cmp_eq(d);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "1.0 == 2.0 should produce all-zeros mask");
    }

    #[test]
    fn test_f32_cmp_ne() {
        use crate::backend::BatchArithmetic;

        // Test inequality
        let a = SimdVec::<f32>::splat(1.0);
        let b = SimdVec::<f32>::splat(2.0);
        let result = a.cmp_ne(b);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "1.0 != 2.0 should produce all-ones mask");

        // Test equality
        let c = SimdVec::<f32>::splat(1.0);
        let d = SimdVec::<f32>::splat(1.0);
        let result = c.cmp_ne(d);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "1.0 != 1.0 should produce all-zeros mask");
    }

    #[test]
    fn test_f32_cmp_lt() {
        use crate::backend::BatchArithmetic;

        // Test less than (true)
        let a = SimdVec::<f32>::splat(1.0);
        let b = SimdVec::<f32>::splat(2.0);
        let result = a.cmp_lt(b);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "1.0 < 2.0 should produce all-ones mask");

        // Test less than (false - equal)
        let c = SimdVec::<f32>::splat(2.0);
        let d = SimdVec::<f32>::splat(2.0);
        let result = c.cmp_lt(d);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "2.0 < 2.0 should produce all-zeros mask");

        // Test less than (false - greater)
        let e = SimdVec::<f32>::splat(3.0);
        let f = SimdVec::<f32>::splat(2.0);
        let result = e.cmp_lt(f);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "3.0 < 2.0 should produce all-zeros mask");
    }

    #[test]
    fn test_f32_cmp_le() {
        use crate::backend::BatchArithmetic;

        // Test less or equal (true - less)
        let a = SimdVec::<f32>::splat(1.0);
        let b = SimdVec::<f32>::splat(2.0);
        let result = a.cmp_le(b);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "1.0 <= 2.0 should produce all-ones mask");

        // Test less or equal (true - equal)
        let c = SimdVec::<f32>::splat(2.0);
        let d = SimdVec::<f32>::splat(2.0);
        let result = c.cmp_le(d);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "2.0 <= 2.0 should produce all-ones mask");

        // Test less or equal (false)
        let e = SimdVec::<f32>::splat(3.0);
        let f = SimdVec::<f32>::splat(2.0);
        let result = e.cmp_le(f);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "3.0 <= 2.0 should produce all-zeros mask");
    }

    #[test]
    fn test_f32_cmp_gt() {
        use crate::backend::BatchArithmetic;

        // Test greater than (true)
        let a = SimdVec::<f32>::splat(2.0);
        let b = SimdVec::<f32>::splat(1.0);
        let result = a.cmp_gt(b);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "2.0 > 1.0 should produce all-ones mask");

        // Test greater than (false - equal)
        let c = SimdVec::<f32>::splat(2.0);
        let d = SimdVec::<f32>::splat(2.0);
        let result = c.cmp_gt(d);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "2.0 > 2.0 should produce all-zeros mask");

        // Test greater than (false - less)
        let e = SimdVec::<f32>::splat(1.0);
        let f = SimdVec::<f32>::splat(2.0);
        let result = e.cmp_gt(f);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "1.0 > 2.0 should produce all-zeros mask");
    }

    #[test]
    fn test_f32_cmp_ge() {
        use crate::backend::BatchArithmetic;

        // Test greater or equal (true - greater)
        let a = SimdVec::<f32>::splat(2.0);
        let b = SimdVec::<f32>::splat(1.0);
        let result = a.cmp_ge(b);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "2.0 >= 1.0 should produce all-ones mask");

        // Test greater or equal (true - equal)
        let c = SimdVec::<f32>::splat(2.0);
        let d = SimdVec::<f32>::splat(2.0);
        let result = c.cmp_ge(d);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "2.0 >= 2.0 should produce all-ones mask");

        // Test greater or equal (false)
        let e = SimdVec::<f32>::splat(1.0);
        let f = SimdVec::<f32>::splat(2.0);
        let result = e.cmp_ge(f);
        let bits = extract_u32_bits(result);
        assert_eq!(bits, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "1.0 >= 2.0 should produce all-zeros mask");
    }

    #[test]
    fn test_f32_select_with_comparison() {
        use crate::backend::BatchArithmetic;

        // Test select with true mask
        let a = SimdVec::<f32>::splat(1.0);
        let b = SimdVec::<f32>::splat(2.0);
        let mask = a.cmp_lt(b); // true mask
        let if_true = SimdVec::<f32>::splat(10.0);
        let if_false = SimdVec::<f32>::splat(20.0);
        let result = mask.select(if_true, if_false);

        let mut output = [0.0f32; 4];
        SimdBatch::store(&result, &mut output);
        assert_eq!(output, [10.0, 10.0, 10.0, 10.0],
                   "select with true mask should pick if_true");

        // Test select with false mask
        let c = SimdVec::<f32>::splat(2.0);
        let d = SimdVec::<f32>::splat(1.0);
        let mask = c.cmp_lt(d); // false mask
        let result = mask.select(if_true, if_false);

        let mut output = [0.0f32; 4];
        SimdBatch::store(&result, &mut output);
        assert_eq!(output, [20.0, 20.0, 20.0, 20.0],
                   "select with false mask should pick if_false");
    }

    #[test]
    fn test_f32_comparison_cast_select() {
        use crate::backend::BatchArithmetic;

        // This tests the actual usage pattern from the raymarching code:
        // Compare f32 values, cast to u32, then select u32 values
        let x = SimdVec::<f32>::splat(0.1);
        let threshold = SimdVec::<f32>::splat(0.2);

        let in_range = x.cmp_lt(threshold);

        // Cast the mask from f32 to u32 for selecting u32 colors
        let mask_u32 = in_range.cast::<u32>();

        let white = SimdVec::<u32>::splat(0xFFFFFFFF);
        let black = SimdVec::<u32>::splat(0x00000000);
        let result = mask_u32.select(white, black);

        let mut output = [0u32; 4];
        SimdBatch::store(&result, &mut output);
        assert_eq!(output, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                   "0.1 < 0.2 should select white (0xFFFFFFFF)");

        // Test the false case
        let y = SimdVec::<f32>::splat(0.3);
        let out_of_range = y.cmp_lt(threshold);
        let mask_u32 = out_of_range.cast::<u32>();
        let result = mask_u32.select(white, black);

        let mut output = [0u32; 4];
        SimdBatch::store(&result, &mut output);
        assert_eq!(output, [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                   "0.3 < 0.2 should select black (0x00000000)");
    }
}
