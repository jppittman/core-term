//! # Numeric Traits
//!
//! Five-tier trait design:
//! - `Computational`: Public trait for types that support constant embedding
//! - `Coordinate`: Marker for types usable as domain coordinates (Field, Jet*, PathJet*)
//! - `AsField`: Extract the scalar Field value from any coordinate type
//! - `Selectable`: Internal trait for types that support branchless selection
//! - `Numeric`: Internal trait with full SIMD operations

/// Marker trait for types that can be used as domain coordinates.
///
/// Types implementing this trait can appear in kernel domains like `(T, T, T, T)`.
/// This includes Field, Jet2, Jet3, PathJet, etc.
///
/// Types like `Discrete` implement `Computational` (for constant embedding) but
/// NOT `Coordinate` because they are output-only types, not coordinate types.
pub trait Coordinate: Computational {}

/// Public trait for user-facing manifold bounds.
///
/// Provides arithmetic operators and constant creation. Users implement
/// `Manifold<I: Computational>` for their types. The library's combinators
/// (Sqrt, Abs, Select, etc.) handle the actual SIMD operations internally.
///
/// This trait intentionally hides SIMD internals like `sqrt`, `select_raw`,
/// and mask introspection. Use the provided combinators instead.
///
/// Arithmetic operators return Manifold types (AST nodes) for algebraic optimization.
pub trait Computational:
    Copy
    + Send
    + Sync
    + Sized
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitOr<Output = Self>
    + core::ops::Not<Output = Self>
{
    /// Create from f32 constant (broadcasts to all lanes).
    fn from_f32(val: f32) -> Self;

    /// Create sequential values [start, start+1, start+2, ...].
    ///
    /// For SIMD types, creates a vector with lane values:
    /// `[start, start+1, start+2, ..., start+(LANES-1)]`
    fn sequential(start: f32) -> Self;
}

/// Internal trait for types that support branchless selection.
///
/// Much weaker than `Numeric` - only requires bitwise blending capability.
/// This allows `Discrete` (packed RGBA) to participate in Select combinators
/// without implementing nonsensical math operations.
pub trait Selectable: Copy + Send + Sync {
    /// Raw conditional select - always blends both values.
    ///
    /// For each SIMD lane, picks `if_true` or `if_false` based on mask.
    /// The mask is a `Field` where each lane is either all-1s (true) or all-0s (false).
    fn select_raw(mask: crate::Field, if_true: Self, if_false: Self) -> Self;
}

/// Internal trait with full SIMD operations.
///
/// Extends `Computational` with methods users shouldn't call directly.
/// These are used by the library's combinators (Sqrt, Select, etc.).
pub trait Numeric: Computational {
    /// Square root.
    fn sqrt(self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// Element-wise minimum.
    fn min(self, rhs: Self) -> Self;

    /// Element-wise maximum.
    fn max(self, rhs: Self) -> Self;

    /// Less than comparison (returns mask).
    fn lt(self, rhs: Self) -> Self;

    /// Less than or equal (returns mask).
    fn le(self, rhs: Self) -> Self;

    /// Greater than comparison (returns mask).
    fn gt(self, rhs: Self) -> Self;

    /// Greater than or equal (returns mask).
    fn ge(self, rhs: Self) -> Self;

    /// Conditional select with early-exit optimization.
    /// Returns if_true where mask is set, if_false elsewhere.
    /// Short-circuits: if all lanes true, returns if_true without evaluating if_false's blend.
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self;

    /// Raw conditional select - always blends both values.
    /// Use this in hot loops where you've already computed both branches.
    fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self;

    /// Check if any lane/component is non-zero.
    /// For SIMD types, checks if any lane is true.
    /// For scalar/jet types, can return false to disable short-circuit optimization.
    fn any(&self) -> bool;

    /// Check if all lanes/components are non-zero.
    /// For SIMD types, checks if all lanes are true.
    /// For scalar/jet types, can return false to disable short-circuit optimization.
    fn all(&self) -> bool;

    /// Create from i32 scalar.
    fn from_i32(val: i32) -> Self;

    /// Create from Field (zero-cost for Field, constant jet for Jet2).
    fn from_field(field: crate::Field) -> Self;

    // ========================================================================
    // Trigonometric Operations (for Spherical Harmonics)
    // ========================================================================

    /// Sine.
    fn sin(self) -> Self;

    /// Cosine.
    fn cos(self) -> Self;

    /// Two-argument arctangent (atan2(y, x)).
    fn atan2(self, x: Self) -> Self;

    /// Raise to a power.
    fn pow(self, exp: Self) -> Self;

    /// Exponential function.
    fn exp(self) -> Self;

    /// Base-2 logarithm.
    fn log2(self) -> Self;

    /// Base-2 exponential (2^x).
    fn exp2(self) -> Self;

    /// Floor (round toward negative infinity).
    fn floor(self) -> Self;

    /// Fused multiply-add: self * b + c
    /// Uses FMA instruction when available (single rounding).
    fn mul_add(self, b: Self, c: Self) -> Self;

    /// Fast approximate reciprocal (1/x).
    /// Uses SIMD reciprocal instruction when available (~12-14 bits accuracy).
    fn recip(self) -> Self;

    /// Fast approximate reciprocal square root (1/sqrt(x)).
    /// Uses SIMD rsqrt instruction when available (~12-14 bits accuracy).
    ///
    /// This is much faster than `sqrt` followed by division:
    /// - rsqrt: ~5 cycles
    /// - sqrt + div: ~25 cycles
    fn rsqrt(self) -> Self;

    // ========================================================================
    // Raw Arithmetic (SIMD operations, used by combinator eval_raw)
    // ========================================================================

    /// Raw addition - direct SIMD operation, no AST building.
    fn raw_add(self, rhs: Self) -> Self;

    /// Raw subtraction - direct SIMD operation, no AST building.
    fn raw_sub(self, rhs: Self) -> Self;

    /// Raw multiplication - direct SIMD operation, no AST building.
    fn raw_mul(self, rhs: Self) -> Self;

    /// Raw division - direct SIMD operation, no AST building.
    fn raw_div(self, rhs: Self) -> Self;

    /// Masked add: self + (mask ? val : 0)
    /// Optimized for winding accumulation patterns.
    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        // Default implementation - backends can override with masked instructions
        self.raw_add(mask & val)
    }
}
