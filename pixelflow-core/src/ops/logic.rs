//! # Logic Operations
//!
//! AST nodes for bitwise logic: And, Or, Not.

use crate::Manifold;
use crate::numeric::Numeric;
use core::ops::{BitAnd, BitOr, Not};

/// Bitwise AND.
#[derive(Clone, Copy, Debug)]
pub struct And<L, R>(pub L, pub R);

/// Bitwise OR.
#[derive(Clone, Copy, Debug)]
pub struct Or<L, R>(pub L, pub R);

/// Bitwise NOT.
#[derive(Clone, Copy, Debug)]
pub struct BNot<M>(pub M);

impl<L, R, I> Manifold<I> for And<L, R>
where
    I: Numeric + BitAnd<Output = I>,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0.eval_raw(x, y, z, w) & self.1.eval_raw(x, y, z, w)
    }
}

impl<L, R, I> Manifold<I> for Or<L, R>
where
    I: Numeric + BitOr<Output = I>,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0.eval_raw(x, y, z, w) | self.1.eval_raw(x, y, z, w)
    }
}

impl<M, I> Manifold<I> for BNot<M>
where
    I: Numeric + Not<Output = I>,
    M: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        !self.0.eval_raw(x, y, z, w)
    }
}
