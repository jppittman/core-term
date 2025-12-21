//! # Binary Operations
//!
//! AST nodes for arithmetic: Add, Sub, Mul, Div.

use crate::{Field, Manifold};

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

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Add<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w) + self.1.eval_raw(x, y, z, w)
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Sub<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w) - self.1.eval_raw(x, y, z, w)
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Mul<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w) * self.1.eval_raw(x, y, z, w)
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Div<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w) / self.1.eval_raw(x, y, z, w)
    }
}
