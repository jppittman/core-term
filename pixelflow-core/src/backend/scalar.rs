//! Scalar backend implementation (LANES=1).

use crate::backend::{Backend, BatchArithmetic, FloatBatchOps, SimdBatch};
use core::fmt::Debug;
use core::ops::*;

/// A scalar backend with 1 lane.
#[derive(Copy, Clone, Debug, Default)]
pub struct Scalar;

impl Backend for Scalar {
    const LANES: usize = 1;
    type Batch<T: Copy + Debug + Default + Send + Sync + 'static + core::cmp::PartialEq> =
        ScalarBatch<T>;

    #[inline(always)]
    fn downcast_u32_to_u8(b: ScalarBatch<u32>) -> ScalarBatch<u8> {
        ScalarBatch(b.0 as u8)
    }

    #[inline(always)]
    fn upcast_u8_to_u32(b: ScalarBatch<u8>) -> ScalarBatch<u32> {
        ScalarBatch(b.0 as u32)
    }

    #[inline(always)]
    fn u32_to_f32(b: ScalarBatch<u32>) -> ScalarBatch<f32> {
        ScalarBatch(b.0 as f32)
    }

    #[inline(always)]
    fn f32_to_u32(b: ScalarBatch<f32>) -> ScalarBatch<u32> {
        ScalarBatch(b.0 as u32)
    }

    #[inline(always)]
    fn transmute_u32_to_f32(b: ScalarBatch<u32>) -> ScalarBatch<f32> {
        ScalarBatch(f32::from_bits(b.0))
    }

    #[inline(always)]
    fn transmute_f32_to_u32(b: ScalarBatch<f32>) -> ScalarBatch<u32> {
        ScalarBatch(b.0.to_bits())
    }
}

/// A batch containing a single value.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct ScalarBatch<T>(pub T);

// PartialEq only for types that support it
impl<T: PartialEq> PartialEq for ScalarBatch<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

// Integer arithmetic uses wrapping to match SIMD semantics
macro_rules! impl_arithmetic_int {
    ($t:ty) => {
        impl Add for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                ScalarBatch(self.0.wrapping_add(rhs.0))
            }
        }
        impl Add<$t> for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: $t) -> Self {
                self + Self::splat(rhs)
            }
        }
        impl Mul<$t> for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: $t) -> Self {
                self * Self::splat(rhs)
            }
        }
        impl Sub for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                ScalarBatch(self.0.wrapping_sub(rhs.0))
            }
        }
        impl Mul for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                ScalarBatch(self.0.wrapping_mul(rhs.0))
            }
        }
        impl Div for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                ScalarBatch(self.0.wrapping_div(rhs.0))
            }
        }
    };
}

// Float arithmetic (no wrapping needed)
macro_rules! impl_arithmetic_float {
    ($t:ty) => {
        impl Add for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                ScalarBatch(self.0 + rhs.0)
            }
        }
        impl Sub for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                ScalarBatch(self.0 - rhs.0)
            }
        }
        impl Mul for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                ScalarBatch(self.0 * rhs.0)
            }
        }
        impl Div for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                ScalarBatch(self.0 / rhs.0)
            }
        }
    };
}

macro_rules! impl_bitwise_int {
    ($t:ty) => {
        impl BitAnd for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                ScalarBatch(self.0 & rhs.0)
            }
        }
        impl BitOr for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                ScalarBatch(self.0 | rhs.0)
            }
        }
        impl BitXor for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                ScalarBatch(self.0 ^ rhs.0)
            }
        }
        impl Not for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn not(self) -> Self {
                ScalarBatch(!self.0)
            }
        }
        impl Shl<i32> for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn shl(self, rhs: i32) -> Self {
                ScalarBatch(self.0 << rhs)
            }
        }
        impl Shr<i32> for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn shr(self, rhs: i32) -> Self {
                ScalarBatch(self.0 >> rhs)
            }
        }
    };
}

macro_rules! impl_bitwise_float {
    ($t:ty, $u:ty) => {
        impl BitAnd for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                let a: $u = f32::to_bits(self.0);
                let b: $u = f32::to_bits(rhs.0);
                ScalarBatch(f32::from_bits(a & b))
            }
        }
        impl BitOr for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                let a: $u = f32::to_bits(self.0);
                let b: $u = f32::to_bits(rhs.0);
                ScalarBatch(f32::from_bits(a | b))
            }
        }
        impl BitXor for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                let a: $u = f32::to_bits(self.0);
                let b: $u = f32::to_bits(rhs.0);
                ScalarBatch(f32::from_bits(a ^ b))
            }
        }
        impl Not for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn not(self) -> Self {
                let a: $u = f32::to_bits(self.0);
                ScalarBatch(f32::from_bits(!a))
            }
        }
        impl Shl<i32> for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn shl(self, _rhs: i32) -> Self {
                unimplemented!("Shift not supported for float batches")
            }
        }
        impl Shr<i32> for ScalarBatch<$t> {
            type Output = Self;
            #[inline(always)]
            fn shr(self, _rhs: i32) -> Self {
                unimplemented!("Shift not supported for float batches")
            }
        }
    };
}

impl_arithmetic_int!(u8);
impl_bitwise_int!(u8);
impl_arithmetic_int!(u16);
impl_bitwise_int!(u16);
impl_arithmetic_int!(u32);
impl_bitwise_int!(u32);
impl_arithmetic_int!(i32);
impl_bitwise_int!(i32);

impl_arithmetic_float!(f32);
impl_bitwise_float!(f32, u32);

impl<T: Copy + Send + Sync + Debug + Default + PartialEq + 'static> SimdBatch<T>
    for ScalarBatch<T>
{
    const LANES: usize = 1;

    fn splat(val: T) -> Self {
        ScalarBatch(val)
    }
    fn sequential_from(start: T) -> Self {
        ScalarBatch(start)
    }
    fn load(slice: &[T]) -> Self {
        ScalarBatch(slice[0])
    }
    fn store(&self, slice: &mut [T]) {
        slice[0] = self.0;
    }
    fn first(&self) -> T {
        self.0
    }

    fn any(&self) -> bool {
        self.0 != T::default()
    }

    fn all(&self) -> bool {
        self.0 != T::default()
    }

    fn extract_lane(&self, _lane: usize) -> T {
        self.0
    }
}

impl<T: Copy> ScalarBatch<T> {
    /// Convert the batch to an array.
    pub fn to_array(self) -> [T; 1] {
        [self.0]
    }
}

impl BatchArithmetic<u32> for ScalarBatch<u32> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        (if_true & self) | (if_false & !self)
    }
    fn gather(base: &[u32], indices: Self) -> Self {
        let idx = indices.0 as usize;
        ScalarBatch(if idx < base.len() { base[idx] } else { 0 })
    }
    fn gather_u8(base: &[u8], indices: Self) -> Self {
        let idx = indices.0 as usize;
        ScalarBatch(if idx < base.len() {
            base[idx] as u32
        } else {
            0
        })
    }
    fn min(self, other: Self) -> Self {
        ScalarBatch(self.0.min(other.0))
    }
    fn max(self, other: Self) -> Self {
        ScalarBatch(self.0.max(other.0))
    }
    fn saturating_add(self, other: Self) -> Self {
        ScalarBatch(self.0.saturating_add(other.0))
    }
    fn saturating_sub(self, other: Self) -> Self {
        ScalarBatch(self.0.saturating_sub(other.0))
    }

    fn cmp_eq(self, other: Self) -> Self {
        if self.0 == other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_ne(self, other: Self) -> Self {
        if self.0 != other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_lt(self, other: Self) -> Self {
        if self.0 < other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_le(self, other: Self) -> Self {
        if self.0 <= other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_gt(self, other: Self) -> Self {
        if self.0 > other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_ge(self, other: Self) -> Self {
        if self.0 >= other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
}

impl BatchArithmetic<u8> for ScalarBatch<u8> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        (if_true & self) | (if_false & !self)
    }
    fn gather(base: &[u8], indices: Self) -> Self {
        let idx = indices.0 as usize;
        ScalarBatch(if idx < base.len() { base[idx] } else { 0 })
    }
    fn min(self, other: Self) -> Self {
        ScalarBatch(self.0.min(other.0))
    }
    fn max(self, other: Self) -> Self {
        ScalarBatch(self.0.max(other.0))
    }
    fn saturating_add(self, other: Self) -> Self {
        ScalarBatch(self.0.saturating_add(other.0))
    }
    fn saturating_sub(self, other: Self) -> Self {
        ScalarBatch(self.0.saturating_sub(other.0))
    }

    fn cmp_eq(self, other: Self) -> Self {
        if self.0 == other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_ne(self, other: Self) -> Self {
        if self.0 != other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_lt(self, other: Self) -> Self {
        if self.0 < other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_le(self, other: Self) -> Self {
        if self.0 <= other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_gt(self, other: Self) -> Self {
        if self.0 > other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_ge(self, other: Self) -> Self {
        if self.0 >= other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
}

impl BatchArithmetic<u16> for ScalarBatch<u16> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        (if_true & self) | (if_false & !self)
    }
    fn gather(base: &[u16], indices: Self) -> Self {
        ScalarBatch(base[indices.0 as usize])
    }
    fn min(self, other: Self) -> Self {
        ScalarBatch(self.0.min(other.0))
    }
    fn max(self, other: Self) -> Self {
        ScalarBatch(self.0.max(other.0))
    }
    fn saturating_add(self, other: Self) -> Self {
        ScalarBatch(self.0.saturating_add(other.0))
    }
    fn saturating_sub(self, other: Self) -> Self {
        ScalarBatch(self.0.saturating_sub(other.0))
    }

    fn cmp_eq(self, other: Self) -> Self {
        if self.0 == other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_ne(self, other: Self) -> Self {
        if self.0 != other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_lt(self, other: Self) -> Self {
        if self.0 < other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_le(self, other: Self) -> Self {
        if self.0 <= other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_gt(self, other: Self) -> Self {
        if self.0 > other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
    fn cmp_ge(self, other: Self) -> Self {
        if self.0 >= other.0 {
            ScalarBatch(!0)
        } else {
            ScalarBatch(0)
        }
    }
}

impl BatchArithmetic<f32> for ScalarBatch<f32> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        (if_true & self) | (if_false & !self)
    }
    fn gather(_base: &[f32], _indices: Self) -> Self {
        unimplemented!("Cannot gather float indices")
    }
    fn min(self, other: Self) -> Self {
        ScalarBatch(self.0.min(other.0))
    }
    fn max(self, other: Self) -> Self {
        ScalarBatch(self.0.max(other.0))
    }
    fn saturating_add(self, other: Self) -> Self {
        self + other
    }
    fn saturating_sub(self, other: Self) -> Self {
        self - other
    }

    fn cmp_eq(self, other: Self) -> Self {
        let mask = if self.0 == other.0 { !0u32 } else { 0 };
        ScalarBatch(f32::from_bits(mask))
    }
    fn cmp_ne(self, other: Self) -> Self {
        let mask = if self.0 != other.0 { !0u32 } else { 0 };
        ScalarBatch(f32::from_bits(mask))
    }
    fn cmp_lt(self, other: Self) -> Self {
        let mask = if self.0 < other.0 { !0u32 } else { 0 };
        ScalarBatch(f32::from_bits(mask))
    }
    fn cmp_le(self, other: Self) -> Self {
        let mask = if self.0 <= other.0 { !0u32 } else { 0 };
        ScalarBatch(f32::from_bits(mask))
    }
    fn cmp_gt(self, other: Self) -> Self {
        let mask = if self.0 > other.0 { !0u32 } else { 0 };
        ScalarBatch(f32::from_bits(mask))
    }
    fn cmp_ge(self, other: Self) -> Self {
        let mask = if self.0 >= other.0 { !0u32 } else { 0 };
        ScalarBatch(f32::from_bits(mask))
    }
}

impl FloatBatchOps for ScalarBatch<f32> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        ScalarBatch(libm::sqrtf(self.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        ScalarBatch(libm::fabsf(self.0))
    }

    #[inline(always)]
    fn log2(self) -> Self {
        ScalarBatch(libm::log2f(self.0))
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        ScalarBatch(libm::exp2f(self.0))
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use super::*;

    #[test]
    fn gather_out_of_bounds_returns_zero() {
        // Bug: gather with out-of-bounds indices should return 0, not panic
        // or return garbage. This is important because coordinate calculations
        // can overflow, producing invalid indices.
        let data = vec![0xDEADBEEFu32, 0xCAFEBABE, 0x12345678, 0xABCDEF00];

        // Index 100 is out of bounds
        let idx = ScalarBatch::<u32>(100);
        let result = ScalarBatch::<u32>::gather(&data, idx);

        // Should return 0 for out-of-bounds, not panic
        assert_eq!(result.0, 0, "Out-of-bounds gather should return 0");
    }

    #[test]
    fn gather_u8_out_of_bounds_returns_zero() {
        let data = vec![0xAAu8, 0xBB, 0xCC, 0xDD];

        let idx = ScalarBatch::<u32>(100);
        let result = ScalarBatch::<u32>::gather_u8(&data, idx);

        assert_eq!(result.0, 0, "Out-of-bounds gather_u8 should return 0");
    }
}
