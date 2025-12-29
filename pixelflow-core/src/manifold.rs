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
/// The bound is `Computational`, which provides arithmetic operators and
/// constant creation methods (`from_f32`, `sequential`).
pub trait Manifold<I: crate::numeric::Computational = Field>: Send + Sync {
    /// The type this manifold produces when evaluated.
    type Output;

    /// Evaluate at the given coordinates.
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output;
}

// Scalar constants are manifolds - they promote to the input Computational type
impl<I: crate::numeric::Computational> Manifold<I> for f32 {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        I::from_f32(*self)
    }
}

// i32 needs Numeric for from_i32 (internal use)
impl<I: crate::numeric::Numeric> Manifold<I> for i32 {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        I::from_i32(*self)
    }
}

// Field needs Numeric for from_field (internal use)
impl<I: crate::numeric::Numeric> Manifold<I> for Field {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        // Zero-cost conversion: identity for Field, constant jet for Jet2
        I::from_field(*self)
    }
}

// Jet2 is a constant manifold - ignores input, always returns itself
impl Manifold<crate::jet::Jet2> for crate::jet::Jet2 {
    type Output = crate::jet::Jet2;
    #[inline(always)]
    fn eval_raw(&self, _x: crate::jet::Jet2, _y: crate::jet::Jet2, _z: crate::jet::Jet2, _w: crate::jet::Jet2) -> crate::jet::Jet2 {
        *self
    }
}

// Jet3 is a constant manifold - ignores input, always returns itself
impl Manifold<crate::jet::Jet3> for crate::jet::Jet3 {
    type Output = crate::jet::Jet3;
    #[inline(always)]
    fn eval_raw(&self, _x: crate::jet::Jet3, _y: crate::jet::Jet3, _z: crate::jet::Jet3, _w: crate::jet::Jet3) -> crate::jet::Jet3 {
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

// &M is a manifold if M is (works for any Computational input type)
impl<I: crate::numeric::Computational, M: Manifold<I> + ?Sized> Manifold<I> for &M {
    type Output = M::Output;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        (**self).eval_raw(x, y, z, w)
    }
}

// ============================================================================
// Thunk: Lazy Manifold Construction
// ============================================================================

/// Lazy manifold construction wrapper.
///
/// Wraps a closure that produces a manifold. The closure is called on each
/// evaluation, constructing the inner manifold fresh. Since manifolds are
/// typically zero-sized types built from combinators, LLVM inlines everything
/// and the thunk disappears at compile time.
///
/// This enables compositional manifold definitions using functions:
///
/// ```ignore
/// fn circle_sdf() -> Thunk<impl Fn() -> impl Manifold> {
///     Thunk(|| (X * X + Y * Y).sqrt() - 1.0f32)
/// }
///
/// // Compose with other manifolds
/// let scene = circle_sdf() + 0.5f32;
/// ```
#[derive(Clone, Copy)]
pub struct Thunk<F>(pub F);

impl<I, F, M> Manifold<I> for Thunk<F>
where
    I: crate::numeric::Numeric,
    F: Fn() -> M + Send + Sync,
    M: Manifold<I>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        (self.0)().eval_raw(x, y, z, w)
    }
}

// ============================================================================
// Scale Combinator
// ============================================================================

/// A combinator that scales the coordinate space of an inner manifold.
///
/// When evaluated at (x, y), it evaluates the inner manifold at (x/scale, y/scale).
/// This is purely compositional - uses At with coordinate manifolds.
///
/// Type alias for `At<Mul<Field, X>, Mul<Field, Y>, Z, W, M>`.
pub type Scale<M> = crate::combinators::At<
    crate::ops::Mul<Field, crate::X>,
    crate::ops::Mul<Field, crate::Y>,
    crate::Z,
    crate::W,
    M,
>;

/// Create a Scale combinator with uniform scaling.
///
/// Uses fast reciprocal instruction for the 1/scale computation.
pub fn scale<M>(inner: M, scale_factor: f64) -> Scale<M> {
    // Use fast reciprocal - this is a bespoke SIMD instruction
    let inv_scale = Field::from(scale_factor as f32).recip();
    crate::combinators::At {
        inner,
        x: inv_scale * crate::X,
        y: inv_scale * crate::Y,
        z: crate::Z,
        w: crate::W,
    }
}
