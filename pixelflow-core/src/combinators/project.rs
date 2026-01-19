//! # Project Combinator
//!
//! Extracts a component from a projectable type.

use crate::{Field, Manifold, variables::Dimension};
use core::marker::PhantomData;

type Field4 = (Field, Field, Field, Field);

/// A trait for types that can be projected onto a dimension to yield a component manifold.
///
/// Instead of evaluating to a Vector and then extracting, implementors provide
/// direct access to their component manifolds.
pub trait Projectable<D: Dimension>: Sized {
    /// The manifold type for this dimension's component.
    type Component: Manifold<Field4, Output = Field>;

    /// Get the component manifold for this dimension.
    fn project(&self) -> Self::Component;
}

/// Project a composite type onto a Dimension to get a scalar manifold.
///
/// If `M` is `Projectable<D>`, `Project<M, D>` evaluates the component for dimension D.
#[derive(Clone, Debug)]
pub struct Project<M, D>(pub M, pub PhantomData<D>);

impl<M, D> Project<M, D> {
    /// Create a new projection.
    #[inline(always)]
    pub fn new(source: M) -> Self {
        Self(source, PhantomData)
    }
}

impl<M, D> Manifold<Field4> for Project<M, D>
where
    M: Projectable<D> + Send + Sync,
    D: Dimension + Send + Sync + 'static,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        self.0.project().eval(p)
    }
}
