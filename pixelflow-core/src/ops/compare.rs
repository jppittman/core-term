//! # Comparison Operations
//!
//! AST nodes for comparisons: Lt, Gt, Le, Ge.
//! These produce masks that can be used with Select.

use crate::backend::BatchArithmetic;
use crate::{Field, Manifold};

/// Less than: L < R
#[derive(Clone, Copy, Debug)]
pub struct Lt<L, R>(pub L, pub R);

/// Greater than: L > R
#[derive(Clone, Copy, Debug)]
pub struct Gt<L, R>(pub L, pub R);

/// Less than or equal: L <= R
#[derive(Clone, Copy, Debug)]
pub struct Le<L, R>(pub L, pub R);

/// Greater than or equal: L >= R
#[derive(Clone, Copy, Debug)]
pub struct Ge<L, R>(pub L, pub R);

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Lt<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_lt(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Gt<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_gt(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Le<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_le(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Ge<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_ge(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}
