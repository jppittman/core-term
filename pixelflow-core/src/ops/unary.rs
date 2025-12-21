//! # Unary Operations
//!
//! AST nodes for unary ops: Sqrt, Abs, Min, Max.

use crate::{Field, Manifold};

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

impl<M: Manifold<Output = Field>> Manifold for Sqrt<M> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).sqrt()
    }
}

impl<M: Manifold<Output = Field>> Manifold for Abs<M> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).abs()
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Max<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).max(self.1.eval_raw(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Min<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).min(self.1.eval_raw(x, y, z, w))
    }
}
