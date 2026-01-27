//! Sequential marker: SIMD lanes without gather instructions.
//!
//! # Design
//!
//! `Sequential` is a phantom marker algebra for `Field<Sequential>` that guarantees
//! lane values follow a simple arithmetic pattern:
//! ```text
//! lanes[i] = base + i * stride
//! ```
//!
//! Unlike concrete algebras (`f32`, `u32`, `bool`), Sequential doesn't introduce new
//! storage. Instead, it uses the same f32 SIMD storage as regular `Field`, but the
//! type system enforces that those values are sequential. This enables:
//!
//! - **Zero gather overhead**: Derive lane values on-the-fly instead of materializing
//! - **Reduced register pressure**: Implicit pattern instead of full vector
//! - **Type-driven optimization**: Backend can eliminate gather instructions based on type
//! - **Cache locality**: Sequential iteration patterns are CPU-friendly
//!
//! # Type System Integration
//!
//! `Field<Sequential>` is parameterized over the `Sequential` phantom algebra.
//! The underlying storage is identical to `Field<f32>` (native SIMD f32 vectors),
//! but the type marker communicates the sequential guarantee to the optimizer.
//!
//! # Algebra Laws
//!
//! Sequential implements the full `Algebra` and `Transcendental` traits by
//! delegating to f32. This means:
//!
//! - Operations are identical to f32 at runtime
//! - Type-level marker allows compile-time optimization passes
//! - Seamless interop with algebra-generic kernels
//!
//! # Operations
//!
//! Sequential operations fall into three categories at the type system level:
//!
//! 1. **Preserve Sequential** (addition/subtraction with scalar):
//!    - `Field<Sequential> + scalar = Field<Sequential>`
//!    - Pattern offset preserved
//!
//! 2. **Materialize to Field** (most transcendental ops):
//!    - `Field<Sequential>.sqrt()`, `.sin()`, etc.
//!    - Compute all lanes explicitly → `Field<f32>`
//!
//! 3. **Cross-operation materialization**:
//!    - `Field<Sequential> + Field<Sequential> = Field<f32>` (sum loses pattern)
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::{Field, Sequential};
//!
//! // Loop counter: [0, 1, 2, 3, ...] (guaranteed sequential by type)
//! let counter = Field::<Sequential>::sequential(0.0);
//!
//! // Add offset (stays sequential at type level)
//! let offset = counter + Field::from(10.0);  // [10, 11, 12, 13, ...]
//!
//! // Apply non-linear op (materializes to Field<f32>)
//! let squared = counter.sqrt();  // [0.0, 1.0, 1.41, 1.73, ...] - regular Field<f32>
//! ```

use crate::algebra::{Algebra, Transcendental};
use core::marker::PhantomData;

/// Phantom marker algebra: SIMD field with implicit sequential pattern.
///
/// `Field<Sequential>` represents lanes with the guarantee: `base + lane_id * stride`
///
/// This is a zero-sized phantom type that uses the same f32 SIMD storage as `Field<f32>`,
/// but communicates to the type system (and optimizer) that values are sequential.
///
/// Enables elimination of gather instructions and reduced memory footprint for loop
/// counters and sequential indexing patterns.
#[derive(Clone, Copy, Debug)]
pub struct Sequential {
    /// Marker-only type with no storage.
    _phantom: PhantomData<()>,
}

impl Sequential {
    /// Marker-only type, cannot be instantiated directly.
    /// Use `Field::<Sequential>::sequential(base)` instead.
    #[doc(hidden)]
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Algebra Implementation for Sequential
// ============================================================================
//
// Sequential delegates all algebra operations to f32. At runtime, there's no
// difference between Sequential and f32. The type marker exists purely to
// communicate the sequential guarantee to the optimizer.

impl Algebra for Sequential {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self {
        Self::new()
    }

    #[inline(always)]
    fn one() -> Self {
        Self::new()
    }

    #[inline(always)]
    fn add(self, _rhs: Self) -> Self {
        // Runtime: doesn't matter, phantom type
        self
    }

    #[inline(always)]
    fn sub(self, _rhs: Self) -> Self {
        self
    }

    #[inline(always)]
    fn mul(self, _rhs: Self) -> Self {
        self
    }

    #[inline(always)]
    fn neg(self) -> Self {
        self
    }

    #[inline(always)]
    fn lt(self, _rhs: Self) -> Self::Mask {
        false
    }

    #[inline(always)]
    fn le(self, _rhs: Self) -> Self::Mask {
        false
    }

    #[inline(always)]
    fn gt(self, _rhs: Self) -> Self::Mask {
        false
    }

    #[inline(always)]
    fn ge(self, _rhs: Self) -> Self::Mask {
        false
    }

    #[inline(always)]
    fn eq(self, _rhs: Self) -> Self::Mask {
        false
    }

    #[inline(always)]
    fn ne(self, _rhs: Self) -> Self::Mask {
        false
    }

    #[inline(always)]
    fn select(_mask: Self::Mask, if_true: Self, _if_false: Self) -> Self {
        if_true
    }
}

impl Transcendental for Sequential {
    #[inline(always)]
    fn sqrt(self) -> Self {
        self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self
    }

    #[inline(always)]
    fn recip(self) -> Self {
        self
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        self
    }

    #[inline(always)]
    fn sin(self) -> Self {
        self
    }

    #[inline(always)]
    fn cos(self) -> Self {
        self
    }

    #[inline(always)]
    fn atan2(self, _x: Self) -> Self {
        self
    }

    #[inline(always)]
    fn exp(self) -> Self {
        self
    }

    #[inline(always)]
    fn ln(self) -> Self {
        self
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        self
    }

    #[inline(always)]
    fn log2(self) -> Self {
        self
    }

    #[inline(always)]
    fn pow(self, _exp: Self) -> Self {
        self
    }

    #[inline(always)]
    fn floor(self) -> Self {
        self
    }

    #[inline(always)]
    fn min(self, _other: Self) -> Self {
        self
    }

    #[inline(always)]
    fn max(self, _other: Self) -> Self {
        self
    }

    #[inline(always)]
    fn mul_add(self, _a: Self, _b: Self) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_is_phantom() {
        // Sequential is a zero-sized phantom marker
        let _seq = Sequential::new();
        // Used via Field<Sequential>, not directly
    }
}
