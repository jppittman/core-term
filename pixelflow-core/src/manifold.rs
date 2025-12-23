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

// Arc<M> is a manifold if M is (allows Arc<dyn Manifold> to work)
impl<M: Manifold + ?Sized> Manifold for alloc::sync::Arc<M> {
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        (**self).eval_raw(x, y, z, w)
    }
}

// Box<M> is a manifold if M is
impl<M: Manifold + ?Sized> Manifold for alloc::boxed::Box<M> {
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        (**self).eval_raw(x, y, z, w)
    }
}

// &M is a manifold if M is
impl<M: Manifold + ?Sized> Manifold for &M {
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        (**self).eval_raw(x, y, z, w)
    }
}

// ============================================================================
// Scale Combinator
// ============================================================================

/// A combinator that scales the coordinate space of an inner manifold.
///
/// When evaluated at (x, y), it evaluates the inner manifold at (x/scale, y/scale).
#[derive(Clone, Copy, Debug)]
pub struct Scale<M> {
    inner: M,
    scale: f32,
}

impl<M> Scale<M> {
    /// Create a new Scale combinator with uniform scaling.
    pub fn uniform(inner: M, scale: f64) -> Self {
        Self {
            inner,
            scale: scale as f32,
        }
    }
}

impl<M: Manifold> Manifold for Scale<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        let inv_scale = Field::from(1.0 / self.scale);
        self.inner
            .eval_raw(x * inv_scale, y * inv_scale, z, w)
    }
}
