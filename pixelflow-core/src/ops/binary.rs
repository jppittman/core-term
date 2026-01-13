//! # Binary Operations
//!
//! AST nodes for arithmetic: Add, Sub, Mul, Div, MulAdd.
//!
//! ## Automatic Optimization: Div by Sqrt → Mul by Rsqrt
//!
//! When dividing by `Sqrt<R>`, the type system automatically fuses this into
//! `Mul<L, Rsqrt<R>>`, using the fast SIMD rsqrt instruction instead of
//! separate sqrt and divide (~8 cycles vs ~25 cycles).
//!
//! ```ignore
//! // These are semantically equivalent but the second is automatically optimized:
//! let slow = x / y.sqrt();           // Div<X, Sqrt<Y>> - uses sqrt + div
//! let fast = x * Rsqrt(y);           // Mul<X, Rsqrt<Y>> - uses rsqrt + mul
//!
//! // The operator overloading makes this automatic:
//! let auto = x / Sqrt(y);            // Becomes Mul<X, Rsqrt<Y>> at compile time!
//! ```

use crate::Manifold;

/// Addition: L + R
#[derive(Clone, Debug)]
pub struct Add<L, R>(pub L, pub R);

/// Subtraction: L - R
#[derive(Clone, Debug)]
pub struct Sub<L, R>(pub L, pub R);

/// Multiplication: L * R
#[derive(Clone, Debug)]
pub struct Mul<L, R>(pub L, pub R);

/// Division: L / R
#[derive(Clone, Debug)]
pub struct Div<L, R>(pub L, pub R);

/// Fused Multiply-Add: A * B + C
///
/// Uses FMA instruction when available (single rounding).
/// This is automatically generated when `Mul + Rhs` or `Lhs + Mul` is written,
/// enabling zero-cost compile-time fusion.
#[derive(Clone, Debug)]
pub struct MulAdd<A, B, C>(pub A, pub B, pub C);

/// Multiply by precomputed reciprocal: M * (1/divisor)
///
/// Optimizes division by constants. The reciprocal is computed once at
/// construction time, turning expensive divisions into fast multiplies.
/// Automatically generated when `Manifold / f32` is written.
#[derive(Clone, Debug)]
pub struct MulRecip<M> {
    /// The inner manifold to evaluate
    pub inner: M,
    /// Precomputed 1/divisor
    pub recip: f32,
}

/// Masked Add: Acc + (Mask ? Val : 0)
///
/// Optimized winding number accumulation. On AVX-512, uses masked add
/// instruction for true single-instruction operation.
#[derive(Clone, Debug)]
pub struct AddMasked<Acc, Val, Mask> {
    /// Accumulator value
    pub acc: Acc,
    /// Value to conditionally add
    pub val: Val,
    /// Mask determining which lanes to add
    pub mask: Mask,
}

impl<L, R, I> Manifold<I> for Add<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0
            .eval_raw(x, y, z, w)
            .raw_add(self.1.eval_raw(x, y, z, w))
    }
}

impl<L, R, I> Manifold<I> for Sub<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0
            .eval_raw(x, y, z, w)
            .raw_sub(self.1.eval_raw(x, y, z, w))
    }
}

impl<L, R, I> Manifold<I> for Mul<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0
            .eval_raw(x, y, z, w)
            .raw_mul(self.1.eval_raw(x, y, z, w))
    }
}

impl<L, R, I> Manifold<I> for Div<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0
            .eval_raw(x, y, z, w)
            .raw_div(self.1.eval_raw(x, y, z, w))
    }
}

impl<A, B, C, I> Manifold<I> for MulAdd<A, B, C>
where
    I: crate::numeric::Numeric,
    A: Manifold<I, Output = I>,
    B: Manifold<I, Output = I>,
    C: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let a = self.0.eval_raw(x, y, z, w);
        let b = self.1.eval_raw(x, y, z, w);
        let c = self.2.eval_raw(x, y, z, w);
        a.mul_add(b, c)
    }
}

impl<M, I> Manifold<I> for MulRecip<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        // Multiply by precomputed reciprocal - avoids slow division
        self.inner
            .eval_raw(x, y, z, w)
            .raw_mul(I::from_f32(self.recip))
    }
}

impl<Acc, Val, Mask, I> Manifold<I> for AddMasked<Acc, Val, Mask>
where
    I: crate::numeric::Numeric,
    Acc: Manifold<I, Output = I>,
    Val: Manifold<I, Output = I>,
    Mask: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let acc = self.acc.eval_raw(x, y, z, w);
        let val = self.val.eval_raw(x, y, z, w);
        let mask = self.mask.eval_raw(x, y, z, w);
        acc.add_masked(val, mask)
    }
}

// ============================================================================
// Automatic Fusion: L / Sqrt(R) → L * Rsqrt(R)
// ============================================================================

/// Multiply by reciprocal square root: L * rsqrt(R).
///
/// This is the optimized form of `L / sqrt(R)`, using fast SIMD rsqrt
/// instruction with Newton-Raphson refinement instead of separate
/// sqrt and divide operations (~8 cycles vs ~25 cycles).
///
/// Created automatically when dividing by `Sqrt<R>`.
#[derive(Clone, Debug)]
pub struct MulRsqrt<L, R>(pub L, pub R);

impl<L, R, I> Manifold<I> for MulRsqrt<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        // L * rsqrt(R) = L / sqrt(R) but faster
        self.0
            .eval_raw(x, y, z, w)
            .raw_mul(self.1.eval_raw(x, y, z, w).rsqrt())
    }
}
