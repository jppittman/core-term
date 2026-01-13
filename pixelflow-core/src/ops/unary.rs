//! # Unary Operations
//!
//! AST nodes for unary ops: Sqrt, Abs, Floor, Sin, Min, Max.

use crate::Manifold;
use crate::numeric::{Computational, Numeric};

/// Square root.
#[derive(Clone, Debug)]
pub struct Sqrt<M>(pub M);

/// Negation: -M
#[derive(Clone, Debug)]
pub struct Neg<M>(pub M);

/// Absolute value.
#[derive(Clone, Debug)]
pub struct Abs<M>(pub M);

/// Floor (round toward negative infinity).
#[derive(Clone, Debug)]
pub struct Floor<M>(pub M);

/// Reciprocal square root: 1/sqrt(M).
/// Uses fast SIMD rsqrt with Newton-Raphson refinement.
#[derive(Clone, Debug)]
pub struct Rsqrt<M>(pub M);

/// Sine function.
#[derive(Clone, Debug)]
pub struct Sin<M>(pub M);

/// Cosine function.
#[derive(Clone, Debug)]
pub struct Cos<M>(pub M);

/// Base-2 logarithm.
#[derive(Clone, Debug)]
pub struct Log2<M>(pub M);

/// Base-2 exponential.
#[derive(Clone, Debug)]
pub struct Exp2<M>(pub M);

/// Element-wise maximum.
#[derive(Clone, Debug)]
pub struct Max<L, R>(pub L, pub R);

/// Element-wise minimum.
#[derive(Clone, Debug)]
pub struct Min<L, R>(pub L, pub R);

impl<M, I> Manifold<I> for Sqrt<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).sqrt()
    }
}

impl<M, I> Manifold<I> for Neg<M>
where
    I: Numeric,
    M: Manifold<I>,
    M::Output: Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        <M::Output as Computational>::from_f32(0.0).raw_sub(self.0.eval_raw(x, y, z, w))
    }
}

impl<M, I> Manifold<I> for Abs<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).abs()
    }
}

impl<M, I> Manifold<I> for Floor<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).floor()
    }
}

impl<M, I> Manifold<I> for Rsqrt<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).rsqrt()
    }
}

impl<M, I> Manifold<I> for Sin<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).sin()
    }
}

impl<M, I> Manifold<I> for Cos<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).cos()
    }
}

impl<M, I> Manifold<I> for Log2<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).log2()
    }
}

impl<M, I> Manifold<I> for Exp2<M>
where
    I: crate::numeric::Numeric,
    M: Manifold<I>,
    M::Output: crate::numeric::Numeric,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).exp2()
    }
}

impl<L, R, I, O> Manifold<I> for Max<L, R>
where
    I: crate::numeric::Numeric,
    O: crate::numeric::Numeric,
    L: Manifold<I, Output = O>,
    R: Manifold<I, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).max(self.1.eval_raw(x, y, z, w))
    }
}

impl<L, R, I, O> Manifold<I> for Min<L, R>
where
    I: crate::numeric::Numeric,
    O: crate::numeric::Numeric,
    L: Manifold<I, Output = O>,
    R: Manifold<I, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w).min(self.1.eval_raw(x, y, z, w))
    }
}
