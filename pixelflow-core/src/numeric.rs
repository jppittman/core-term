//! # Numeric Trait
//!
//! The computational substrate - types that operations can be performed on.
//! This trait is internal and abstracts Field and Jet2.

/// The computational substrate for manifolds.
///
/// Types implementing this trait represent values that can flow through
/// manifold computations. Currently implemented by:
/// - `Field`: Concrete SIMD batches (the default)
/// - `Jet2`: AD-enhanced values tracking derivatives
///
/// This trait is intentionally private to maintain API flexibility.
pub trait Numeric:
    Copy
    + Send
    + Sync
    + Sized
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitOr<Output = Self>
{
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

    /// Conditional select.
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self;

    /// Check if any lane/component is non-zero.
    /// For SIMD types, checks if any lane is true.
    /// For scalar/jet types, can return false to disable short-circuit optimization.
    fn any(&self) -> bool;

    /// Check if all lanes/components are non-zero.
    /// For SIMD types, checks if all lanes are true.
    /// For scalar/jet types, can return false to disable short-circuit optimization.
    fn all(&self) -> bool;

    /// Create from f32 scalar.
    fn from_f32(val: f32) -> Self;

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
}
