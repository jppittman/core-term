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
    type Batch<T: Copy + Debug + Default + Send + Sync + 'static>: SimdBatch<T>;

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
pub trait SimdBatch<T: Copy>: Copy + Clone + Debug + Default + Send + Sync {
    fn splat(val: T) -> Self;
    fn sequential_from(start: T) -> Self;
    fn load(slice: &[T]) -> Self;
    fn store(&self, slice: &mut [T]);
    /// Extract the first lane.
    ///
    /// Named `first` rather than `scalar` to be honest: when you splat(x)
    /// and call first(), you're doing a point sample. The lanes could be
    /// used for supersampling (different subpixel offsets) with a different
    /// reduction (average, min, max, etc).
    fn first(&self) -> T;
}

/// Arithmetic operations supported by numeric SIMD batches.
pub trait BatchArithmetic<T: Copy>:
    SimdBatch<T>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Shl<i32, Output = Self>
    + Shr<i32, Output = Self>
{
    fn select(self, if_true: Self, if_false: Self) -> Self;
    fn gather(base: &[T], indices: Self) -> Self;
    fn gather_u8(_base: &[u8], _indices: Self) -> Self {
        unimplemented!("gather_u8 not implemented for this type")
    }

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn saturating_add(self, other: Self) -> Self;
    fn saturating_sub(self, other: Self) -> Self;

    fn cmp_eq(self, other: Self) -> Self;
    fn cmp_ne(self, other: Self) -> Self;
    fn cmp_lt(self, other: Self) -> Self;
    fn cmp_le(self, other: Self) -> Self;
    fn cmp_gt(self, other: Self) -> Self;
    fn cmp_ge(self, other: Self) -> Self;
}

/// Float-specific batch operations (sqrt, abs, etc.).
pub trait FloatBatchOps: BatchArithmetic<f32> {
    /// Square root of each element.
    fn sqrt(self) -> Self;

    /// Absolute value of each element.
    fn abs(self) -> Self;
}

pub use BatchArithmetic as BatchOps;
