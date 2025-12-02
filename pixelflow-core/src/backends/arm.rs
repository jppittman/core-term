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
    type Batch<T: Copy + Debug + Default + Send + Sync + 'static> = SimdVec<T>;

    #[inline(always)]
    fn downcast_u32_to_u8(b: SimdVec<u32>) -> SimdVec<u8> {
        unsafe { b.transmute() }
    }

    #[inline(always)]
    fn upcast_u8_to_u32(b: SimdVec<u8>) -> SimdVec<u32> {
        unsafe { b.transmute() }
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

    #[inline(always)]
    pub fn cast<U: Copy>(self) -> SimdVec<U> {
        unsafe { self.transmute() }
    }

    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut T) {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                vst1q_u32(ptr as *mut u32, self.0.u32)
            } else if core::mem::size_of::<T>() == 2 {
                vst1q_u16(ptr as *mut u16, self.0.u16)
            } else if core::mem::size_of::<T>() == 1 {
                vst1q_u8(ptr as *mut u8, self.0.u8)
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
impl<T: Copy + Debug + Default + Send + Sync + 'static> SimdBatch<T> for SimdVec<T> {
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
                Self(
                    NeonReg {
                        u16: vld1q_u16(slice.as_ptr() as *const u16),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 1 {
                Self(
                    NeonReg {
                        u8: vld1q_u8(slice.as_ptr() as *const u8),
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

    fn store(&self, slice: &mut [T]) {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                vst1q_u32(slice.as_mut_ptr() as *mut u32, self.0.u32)
            } else if core::mem::size_of::<T>() == 2 {
                vst1q_u16(slice.as_mut_ptr() as *mut u16, self.0.u16)
            } else if core::mem::size_of::<T>() == 1 {
                vst1q_u8(slice.as_mut_ptr() as *mut u8, self.0.u8)
            }
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
        unsafe {
            let mut idx = [0u32; 4];
            vst1q_u32(idx.as_mut_ptr(), indices.0.u32);
            let len = base.len();
            let v0 = if (idx[0] as usize) < len { base[idx[0] as usize] } else { 0 };
            let v1 = if (idx[1] as usize) < len { base[idx[1] as usize] } else { 0 };
            let v2 = if (idx[2] as usize) < len { base[idx[2] as usize] } else { 0 };
            let v3 = if (idx[3] as usize) < len { base[idx[3] as usize] } else { 0 };
            let arr = [v0, v1, v2, v3];
            Self(
                NeonReg {
                    u32: vld1q_u32(arr.as_ptr()),
                },
                PhantomData,
            )
        }
    }

    fn gather_u8(base: &[u8], indices: Self) -> Self {
        unsafe {
            let mut idx = [0u32; 4];
            vst1q_u32(idx.as_mut_ptr(), indices.0.u32);
            let len = base.len();
            let v0 = if (idx[0] as usize) < len { base[idx[0] as usize] as u32 } else { 0 };
            let v1 = if (idx[1] as usize) < len { base[idx[1] as usize] as u32 } else { 0 };
            let v2 = if (idx[2] as usize) < len { base[idx[2] as usize] as u32 } else { 0 };
            let v3 = if (idx[3] as usize) < len { base[idx[3] as usize] as u32 } else { 0 };
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
                    f32: vreinterpretq_f32_u32(mask),
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
                    f32: vreinterpretq_f32_u32(vmvnq_u32(eq)),
                },
                PhantomData,
            )
        }
    }

    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            let mask = vcltq_f32(self.0.f32, other.0.f32);
            Self(
                NeonReg {
                    f32: vreinterpretq_f32_u32(mask),
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
                    f32: vreinterpretq_f32_u32(mask),
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
                    f32: vreinterpretq_f32_u32(mask),
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
                    f32: vreinterpretq_f32_u32(mask),
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
}
