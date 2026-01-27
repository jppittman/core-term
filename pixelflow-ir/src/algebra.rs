//! Algebraic structures for Field carriers.
//!
//! The `Algebra` trait defines what operations a type supports when lifted
//! into a `Field`. This creates a unified type system where:
//!
//! - `Field<f32>` = SIMD floats (the default)
//! - `Field<bool>` = native masks (k-registers on AVX-512)
//! - `Field<u32>` = packed discrete values
//! - `Field<Dual<2>>` = 2D automatic differentiation
//!
//! The key insight: SIMD-ification is orthogonal to the algebra.

/// An algebraic structure that can be carried by Field.
///
/// This trait defines the operations available on a type when it's
/// SIMD-ified into a Field. Implementations exist for scalars (f32, bool, u32)
/// and composite types (Dual<N, A>).
pub trait Algebra: Sized + Copy + Send + Sync + 'static {
    /// What comparisons return.
    /// - For `f32`/integers: `bool` (hard select)
    /// - For `Dual<N, A>`: `A` (signed distance for soft select)
    type Mask: Algebra;

    /// Zero element (additive identity).
    fn zero() -> Self;

    /// One element (multiplicative identity).
    fn one() -> Self;

    /// Create from f32 (for constants). Identity for f32, truncates for integers.
    fn from_f32(val: f32) -> Self;

    // ========================================================================
    // Ring operations
    // ========================================================================

    /// Addition.
    fn add(self, rhs: Self) -> Self;

    /// Subtraction.
    fn sub(self, rhs: Self) -> Self;

    /// Multiplication.
    fn mul(self, rhs: Self) -> Self;

    /// Negation.
    fn neg(self) -> Self;

    // ========================================================================
    // Comparison → Mask
    // ========================================================================

    /// Less than.
    fn lt(self, rhs: Self) -> Self::Mask;

    /// Less than or equal.
    fn le(self, rhs: Self) -> Self::Mask;

    /// Greater than.
    fn gt(self, rhs: Self) -> Self::Mask;

    /// Greater than or equal.
    fn ge(self, rhs: Self) -> Self::Mask;

    /// Equality.
    fn eq(self, rhs: Self) -> Self::Mask;

    /// Not equal.
    fn ne(self, rhs: Self) -> Self::Mask;

    // ========================================================================
    // Selection (branchless)
    // ========================================================================

    /// Branchless select: returns `if_true` where mask is true, `if_false` otherwise.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
}

/// Extended operations for differentiable algebras.
///
/// These operations require calculus to differentiate, which is why
/// they're separated from the base `Algebra` trait.
pub trait Transcendental: Algebra {
    /// Division.
    fn div(self, rhs: Self) -> Self;

    /// Reciprocal (1/x).
    fn recip(self) -> Self;

    /// Square root.
    fn sqrt(self) -> Self;

    /// Reciprocal square root (1/√x).
    fn rsqrt(self) -> Self;

    /// Sine.
    fn sin(self) -> Self;

    /// Cosine.
    fn cos(self) -> Self;

    /// Exponential (e^x).
    fn exp(self) -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// Base-2 exponential.
    fn exp2(self) -> Self;

    /// Base-2 logarithm.
    fn log2(self) -> Self;

    /// Power (x^y).
    fn pow(self, exp: Self) -> Self;

    /// Two-argument arctangent.
    fn atan2(self, x: Self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// Floor.
    fn floor(self) -> Self;

    /// Minimum.
    fn min(self, rhs: Self) -> Self;

    /// Maximum.
    fn max(self, rhs: Self) -> Self;

    /// Fused multiply-add: a * b + c
    fn mul_add(self, b: Self, c: Self) -> Self;
}

// ============================================================================
// Scalar implementations
// ============================================================================

impl Algebra for f32 {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self { 0.0 }

    #[inline(always)]
    fn one() -> Self { 1.0 }

    #[inline(always)]
    fn from_f32(val: f32) -> Self { val }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self { self + rhs }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { self - rhs }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { self * rhs }

    #[inline(always)]
    fn neg(self) -> Self { -self }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask { self < rhs }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self::Mask { self <= rhs }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask { self > rhs }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self::Mask { self >= rhs }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask { self == rhs }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self::Mask { self != rhs }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
}

impl Transcendental for f32 {
    #[inline(always)]
    fn div(self, rhs: Self) -> Self { self / rhs }

    #[inline(always)]
    fn recip(self) -> Self { 1.0 / self }

    #[inline(always)]
    fn sqrt(self) -> Self { libm::sqrtf(self) }

    #[inline(always)]
    fn rsqrt(self) -> Self { 1.0 / libm::sqrtf(self) }

    #[inline(always)]
    fn sin(self) -> Self { libm::sinf(self) }

    #[inline(always)]
    fn cos(self) -> Self { libm::cosf(self) }

    #[inline(always)]
    fn exp(self) -> Self { libm::expf(self) }

    #[inline(always)]
    fn ln(self) -> Self { libm::logf(self) }

    #[inline(always)]
    fn exp2(self) -> Self { libm::exp2f(self) }

    #[inline(always)]
    fn log2(self) -> Self { libm::log2f(self) }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self { libm::powf(self, exp) }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self { libm::atan2f(self, x) }

    #[inline(always)]
    fn abs(self) -> Self { libm::fabsf(self) }

    #[inline(always)]
    fn floor(self) -> Self { libm::floorf(self) }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self { libm::fminf(self, rhs) }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self { libm::fmaxf(self, rhs) }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self { libm::fmaf(self, b, c) }
}

impl Algebra for bool {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self { false }

    #[inline(always)]
    fn one() -> Self { true }

    #[inline(always)]
    fn from_f32(val: f32) -> Self { val != 0.0 }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self { self | rhs }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { self & !rhs }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { self & rhs }

    #[inline(always)]
    fn neg(self) -> Self { !self }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask { !self & rhs }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self::Mask { !self | rhs }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask { self & !rhs }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self::Mask { self | !rhs }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask { self == rhs }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self::Mask { self != rhs }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        (mask & if_true) | (!mask & if_false)
    }
}

impl Algebra for u32 {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self { 0 }

    #[inline(always)]
    fn one() -> Self { 1 }

    #[inline(always)]
    fn from_f32(val: f32) -> Self { val as u32 }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { self.wrapping_mul(rhs) }

    #[inline(always)]
    fn neg(self) -> Self { self.wrapping_neg() }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask { self < rhs }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self::Mask { self <= rhs }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask { self > rhs }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self::Mask { self >= rhs }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask { self == rhs }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self::Mask { self != rhs }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
}

impl Algebra for i32 {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self { 0 }

    #[inline(always)]
    fn one() -> Self { 1 }

    #[inline(always)]
    fn from_f32(val: f32) -> Self { val as i32 }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { self.wrapping_mul(rhs) }

    #[inline(always)]
    fn neg(self) -> Self { self.wrapping_neg() }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask { self < rhs }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self::Mask { self <= rhs }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask { self > rhs }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self::Mask { self >= rhs }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask { self == rhs }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self::Mask { self != rhs }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
}
