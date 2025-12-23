//! # Unary Operations
//!
//! AST nodes for unary ops: Sqrt, Abs, Min, Max.

use crate::Manifold;
use crate::numeric::Numeric;

/// Square root.
#[derive(Clone, Copy, Debug)]
pub struct Sqrt<M>(pub M);

/// Absolute value.
#[derive(Clone, Copy, Debug)]
pub struct Abs<M>(pub M);

/// Element-wise maximum.
#[derive(Clone, Copy, Debug)]
pub struct Max<L, R>(pub L, pub R);

/// Element-wise minimum.
#[derive(Clone, Copy, Debug)]
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
