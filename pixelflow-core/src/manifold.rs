//! # Manifold Trait
//!
//! The core abstraction: a function from 4D coordinates to values.

use crate::Field;

/// A manifold is a function from coordinates to a value.
///
/// The `Output` associated type allows manifolds to produce different
/// types: scalar manifolds return `Field`, color manifolds return
/// `(Field, Field, Field, Field)`, etc.
pub trait Manifold: Send + Sync {
    /// The type this manifold produces when evaluated.
    type Output;

    /// Evaluate at the given coordinates.
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output;
}

// Scalar constants are manifolds
impl Manifold for f32 {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Field {
        Field::from(*self)
    }
}

impl Manifold for i32 {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Field {
        Field::from(*self)
    }
}

// Field itself is a manifold (identity)
impl Manifold for Field {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Field {
        *self
    }
}
