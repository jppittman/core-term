//! Backend trait and Batch operations.

use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Shl, Shr, Sub};

/// A backend provides the SIMD implementation.
pub trait Backend: 'static + Copy + Clone + Send + Sync + Debug {
    /// Number of lanes in the SIMD vector.
    const LANES: usize;
    /// Whether gather operations are slow on this backend.
    const GATHER_IS_SLOW: bool = false;

    /// The SIMD vector type.
    type Batch<T: Copy + Debug + Default + PartialEq + Send + Sync + 'static>: SimdBatch<T>;

    /// Casts a batch of u32 to u8 (narrowing/bitcast depending on backend).
    /// Used for adapting Surface<u32> logic to Surface<u8> interfaces.
    fn downcast_u32_to_u8(b: Self::Batch<u32>) -> Self::Batch<u8>;

    /// Casts a batch of u8 to u32 (widening/bitcast depending on backend).
    /// Used for adapting Surface<u8> masks to u32 operations.
    fn upcast_u8_to_u32(b: Self::Batch<u8>) -> Self::Batch<u32>;

    /// Converts a batch of u32 to f32 (integer to float).
    fn u32_to_f32(b: Self::Batch<u32>) -> Self::Batch<f32>;

    /// Converts a batch of f32 to u32 (float to integer, truncating).
    fn f32_to_u32(b: Self::Batch<f32>) -> Self::Batch<u32>;

    /// Bitcast/transmute between u32 and f32 (reinterpret bits).
    fn transmute_u32_to_f32(b: Self::Batch<u32>) -> Self::Batch<f32>;

    /// Bitcast/transmute between f32 and u32 (reinterpret bits).
    fn transmute_f32_to_u32(b: Self::Batch<f32>) -> Self::Batch<u32>;

    /// Computes the inverse of a 3x3 matrix.
    fn inverse_mat3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        let det_abs = if det < 0.0 { -det } else { det };
        if det_abs < 1e-6 {
            return None;
        }

        let inv_det = 1.0 / det;

        let mut inv = [[0.0; 3]; 3];

        inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

        inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;

        inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

        Some(inv)
    }
}

/// Basic operations supported by any SIMD batch (storage/movement).
pub trait SimdBatch<T: Copy + Debug + Default + PartialEq + Send + Sync + 'static>: Copy + Clone + Debug + Default + Send + Sync {
    /// Number of lanes in this batch.
    const LANES: usize;

    /// Create a batch with all lanes set to `val`.
    fn splat(val: T) -> Self;

    /// Create a batch with sequential values starting from `start`.
    /// E.g. [start, start+1, start+2, ...]
    fn sequential_from(start: T) -> Self;

    /// Load a batch from a slice.
    /// The slice must have at least `LANES` elements.
    fn load(slice: &[T]) -> Self;

    /// Store the batch to a slice.
    /// The slice must have at least `LANES` elements.
    fn store(&self, slice: &mut [T]);

    /// Extract the first lane.
    fn first(&self) -> T;

    /// Returns true if any lane is non-zero.
    fn any(&self) -> bool;

    /// Returns true if all lanes are non-zero.
    fn all(&self) -> bool;

    /// Church boolean / conditional execution optimization.
    fn church<F1, F2>(self, t: F1, f: F2) -> Self
    where
        Self: BatchArithmetic<T>,
        F1: FnOnce() -> Self,
        F2: FnOnce() -> Self,
    {
        if self.all() {
            t()
        } else if !self.any() {
            f()
        } else {
            self.select(t(), f())
        }
    }
}

/// Arithmetic operations for batches.
pub trait BatchArithmetic<T>:
    SimdBatch<T> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Select values from `if_true` or `if_false` based on the mask (self).
    /// If a bit is 1, take from `if_true`. If 0, take from `if_false`.
    /// Note: `self` acts as the mask.
    fn select(self, if_true: Self, if_false: Self) -> Self;

    /// Gather values from a slice using indices in this batch.
    fn gather(base: &[T], indices: Self) -> Self;

    /// Gather u8 values from a slice using indices in this batch.
    /// This is used when T is u32 but we want to gather bytes (e.g. from an atlas).
    fn gather_u8(_base: &[u8], _indices: Self) -> Self {
        unimplemented!("gather_u8 not implemented for this type")
    }

    /// Lane-wise minimum.
    fn min(self, other: Self) -> Self;

    /// Lane-wise maximum.
    fn max(self, other: Self) -> Self;

    /// Saturating addition.
    fn saturating_add(self, other: Self) -> Self;

    /// Saturating subtraction.
    fn saturating_sub(self, other: Self) -> Self;

    /// Compare equal. Returns a mask.
    fn cmp_eq(self, other: Self) -> Self;

    /// Compare not equal. Returns a mask.
    fn cmp_ne(self, other: Self) -> Self;

    /// Compare less than. Returns a mask.
    fn cmp_lt(self, other: Self) -> Self;

    /// Compare less than or equal. Returns a mask.
    fn cmp_le(self, other: Self) -> Self;

    /// Compare greater than. Returns a mask.
    fn cmp_gt(self, other: Self) -> Self;

    /// Compare greater than or equal. Returns a mask.
    fn cmp_ge(self, other: Self) -> Self;
}

/// Float-specific batch operations (sqrt, abs, transcendentals).
pub trait FloatBatchOps: BatchArithmetic<f32> {
    /// Square root of each element.
    fn sqrt(self) -> Self;

    /// Absolute value of each element.
    fn abs(self) -> Self;

    /// Base-2 logarithm of each element.
    /// Uses IEEE 754 bit extraction + polynomial correction for accuracy.
    fn log2(self) -> Self;

    /// Base-2 exponential of each element (2^x).
    /// Uses range reduction + polynomial approximation.
    fn exp2(self) -> Self;

    /// Natural logarithm of each element.
    fn ln(self) -> Self {
        // ln(x) = log2(x) / log2(e)
        self.log2() * Self::splat(1.0 / core::f32::consts::LOG2_E)
    }

    /// Natural exponential of each element (e^x).
    fn exp(self) -> Self {
        // e^x = 2^(x * log2(e))
        (self * Self::splat(core::f32::consts::LOG2_E)).exp2()
    }

    /// Power function: self^exponent for each element.
    fn pow(self, exponent: Self) -> Self {
        // x^y = 2^(y * log2(x))
        (exponent * self.log2()).exp2()
    }
}

pub use BatchArithmetic as BatchOps;

#[cfg(target_arch = "x86_64")]
pub mod x86;
pub mod scalar;
#[cfg(target_arch = "aarch64")]
pub mod arm;
