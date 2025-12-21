//! # Project Combinator
//!
//! Extracts a component from a Vector Manifold.
//! Effectively `Dot(M, BasisSet[D])`.

use crate::{Field, Manifold, ops::Vector, variables::Dimension};
use core::marker::PhantomData;

/// Project a Vector Manifold onto a Dimension basis.
///
/// If `M` produces a Vector (like `ColorVector`), `Project<M, X>` extracts the
/// first component (Red).
#[derive(Clone, Copy, Debug)]
pub struct Project<M, D>(pub M, pub PhantomData<D>);

impl<M, D> Project<M, D> {
    /// Create a new projection.
    #[inline(always)]
    pub fn new(manifold: M) -> Self {
        Self(manifold, PhantomData)
    }
}

impl<M, D> Manifold for Project<M, D>
where
    M: Manifold,
    M::Output: Vector,
    D: Dimension + Send + Sync + 'static,
{
    // The output type is the Component type of the Vector (usually Field)
    type Output = <M::Output as Vector>::Component;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        self.0.eval_raw(x, y, z, w).get(D::AXIS)
    }
}
