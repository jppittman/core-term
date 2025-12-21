//! # Logic Operations
//!
//! AST nodes for bitwise logic: And, Or, Not.

use crate::{Field, Manifold};

/// Bitwise AND.
#[derive(Clone, Copy, Debug)]
pub struct And<L, R>(pub L, pub R);

/// Bitwise OR.
#[derive(Clone, Copy, Debug)]
pub struct Or<L, R>(pub L, pub R);

/// Bitwise NOT.
#[derive(Clone, Copy, Debug)]
pub struct BNot<M>(pub M);

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for And<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w) & self.1.eval_raw(x, y, z, w)
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Or<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w) | self.1.eval_raw(x, y, z, w)
    }
}

impl<M: Manifold<Output = Field>> Manifold for BNot<M> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        !self.0.eval_raw(x, y, z, w)
    }
}
