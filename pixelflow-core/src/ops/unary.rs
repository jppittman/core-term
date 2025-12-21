//! # Unary Operations
//!
//! AST nodes for math functions: Sqrt, Abs, Max, Min.

use crate::backend::{BatchArithmetic, FloatBatchOps};
use crate::{Field, Manifold};

/// Square root.
#[derive(Clone, Copy, Debug)]
pub struct Sqrt<T>(pub T);

/// Absolute value.
#[derive(Clone, Copy, Debug)]
pub struct Abs<T>(pub T);

/// Maximum of two values.
#[derive(Clone, Copy, Debug)]
pub struct Max<L, R>(pub L, pub R);

/// Minimum of two values.
#[derive(Clone, Copy, Debug)]
pub struct Min<L, R>(pub L, pub R);

impl<T: Manifold<Output = Field>> Manifold for Sqrt<T> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).sqrt()
    }
}

impl<T: Manifold<Output = Field>> Manifold for Abs<T> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).abs()
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Max<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).max(self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Min<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).min(self.1.eval(x, y, z, w))
    }
}
