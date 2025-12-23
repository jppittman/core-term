//! # Binary Operations
//!
//! AST nodes for arithmetic: Add, Sub, Mul, Div.

use crate::Manifold;

/// Addition: L + R
#[derive(Clone, Copy, Debug)]
pub struct Add<L, R>(pub L, pub R);

/// Subtraction: L - R
#[derive(Clone, Copy, Debug)]
pub struct Sub<L, R>(pub L, pub R);

/// Multiplication: L * R
#[derive(Clone, Copy, Debug)]
pub struct Mul<L, R>(pub L, pub R);

/// Division: L / R
#[derive(Clone, Copy, Debug)]
pub struct Div<L, R>(pub L, pub R);

impl<L, R, I> Manifold<I> for Add<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I>,
    R: Manifold<I>,
    L::Output: core::ops::Add<R::Output>,
{
    type Output = <L::Output as core::ops::Add<R::Output>>::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w) + self.1.eval_raw(x, y, z, w)
    }
}

impl<L, R, I> Manifold<I> for Sub<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I>,
    R: Manifold<I>,
    L::Output: core::ops::Sub<R::Output>,
{
    type Output = <L::Output as core::ops::Sub<R::Output>>::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w) - self.1.eval_raw(x, y, z, w)
    }
}

impl<L, R, I> Manifold<I> for Mul<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I>,
    R: Manifold<I>,
    L::Output: core::ops::Mul<R::Output>,
{
    type Output = <L::Output as core::ops::Mul<R::Output>>::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w) * self.1.eval_raw(x, y, z, w)
    }
}

impl<L, R, I> Manifold<I> for Div<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I>,
    R: Manifold<I>,
    L::Output: core::ops::Div<R::Output>,
{
    type Output = <L::Output as core::ops::Div<R::Output>>::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.0.eval_raw(x, y, z, w) / self.1.eval_raw(x, y, z, w)
    }
}
