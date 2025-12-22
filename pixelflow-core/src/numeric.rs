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
pub(crate) trait Numeric:
    Copy
    + Send
    + Sync
    + Sized
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
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

    /// Create from f32 scalar.
    fn from_f32(val: f32) -> Self;

    /// Create from i32 scalar.
    fn from_i32(val: i32) -> Self;
}
