//! # Manifold Trait
//!
//! The core abstraction: a function from 4D coordinates to values.

use crate::Field;

/// A manifold is a function from coordinates to a value.
///
/// The `Output` associated type allows manifolds to produce different
/// types: scalar manifolds return `Field`, color manifolds return
/// `(Field, Field, Field, Field)`, etc.
///
/// The `I` type parameter represents the input coordinate type.
/// By default it's `Field`, but can be `Jet2` for automatic differentiation.
pub trait Manifold<I: crate::numeric::Numeric = Field>: Send + Sync {
    /// The type this manifold produces when evaluated.
    type Output;

    /// Evaluate at the given coordinates.
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output;
}

// Scalar constants are manifolds - they promote to the input Numeric type
impl<I: crate::numeric::Numeric> Manifold<I> for f32 {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        I::from_f32(*self)
    }
}

impl<I: crate::numeric::Numeric> Manifold<I> for i32 {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        I::from_i32(*self)
    }
}

// Field is a constant manifold - promotes to the input Numeric type
impl<I: crate::numeric::Numeric> Manifold<I> for Field {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        // Convert Field to f32, then promote to I
        // For Field inputs this is identity via from_f32(self as f32) == self
        // For Jet2 inputs this creates a constant jet
        let mut temp = [0.0f32];
        self.store(&mut temp);
        I::from_f32(temp[0])
    }
}
