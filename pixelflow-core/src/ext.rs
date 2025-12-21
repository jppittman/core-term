//! # DSL Extension Trait
//!
//! Provides fluent method-chaining API for manifolds.

use crate::Manifold;
use crate::combinators::Select;
use crate::ops::{Abs, Add, Div, Ge, Gt, Le, Lt, Max, Min, Mul, Sqrt, Sub};

use alloc::boxed::Box;

/// Type-erased manifold (returning a Field).
pub type BoxedManifold = Box<dyn Manifold<Output = crate::Field>>;

impl Manifold for BoxedManifold {
    type Output = crate::Field;
    #[inline(always)]
    fn eval_raw(
        &self,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        (**self).eval_raw(x, y, z, w)
    }
}

/// Extension methods for composing manifolds.
pub trait ManifoldExt: Manifold<Output = crate::Field> + Sized {
    /// Evaluate the manifold at the given coordinates.
    ///
    /// This is a convenience wrapper around `eval_raw` that accepts any type
    /// that can be converted into `Field` (e.g. `f32`, `i32`, `Field`).
    #[inline(always)]
    fn eval<
        A: Into<crate::Field>,
        B: Into<crate::Field>,
        C: Into<crate::Field>,
        D: Into<crate::Field>,
    >(
        &self,
        x: A,
        y: B,
        z: C,
        w: D,
    ) -> crate::Field {
        self.eval_raw(x.into(), y.into(), z.into(), w.into())
    }

    /// Add two manifolds.
    fn add<R: Manifold>(self, rhs: R) -> Add<Self, R> {
        Add(self, rhs)
    }
    /// Subtract two manifolds.
    fn sub<R: Manifold>(self, rhs: R) -> Sub<Self, R> {
        Sub(self, rhs)
    }
    /// Multiply two manifolds.
    fn mul<R: Manifold>(self, rhs: R) -> Mul<Self, R> {
        Mul(self, rhs)
    }
    /// Divide two manifolds.
    fn div<R: Manifold>(self, rhs: R) -> Div<Self, R> {
        Div(self, rhs)
    }
    /// Square root.
    fn sqrt(self) -> Sqrt<Self> {
        Sqrt(self)
    }
    /// Absolute value.
    fn abs(self) -> Abs<Self> {
        Abs(self)
    }
    /// Maximum of two manifolds.
    fn max<R: Manifold>(self, rhs: R) -> Max<Self, R> {
        Max(self, rhs)
    }
    /// Minimum of two manifolds.
    fn min<R: Manifold>(self, rhs: R) -> Min<Self, R> {
        Min(self, rhs)
    }

    // Comparisons
    /// Less than.
    fn lt<R: Manifold>(self, rhs: R) -> Lt<Self, R> {
        Lt(self, rhs)
    }
    /// Greater than.
    fn gt<R: Manifold>(self, rhs: R) -> Gt<Self, R> {
        Gt(self, rhs)
    }
    /// Less than or equal.
    fn le<R: Manifold>(self, rhs: R) -> Le<Self, R> {
        Le(self, rhs)
    }
    /// Greater than or equal.
    fn ge<R: Manifold>(self, rhs: R) -> Ge<Self, R> {
        Ge(self, rhs)
    }

    /// Conditional select. If self (as mask), use if_true; else if_false.
    fn select<T: Manifold, F: Manifold>(self, if_true: T, if_false: F) -> Select<Self, T, F> {
        Select {
            cond: self,
            if_true,
            if_false,
        }
    }

    /// Type-erase this manifold into a boxed trait object.
    fn boxed(self) -> BoxedManifold
    where
        Self: 'static,
    {
        Box::new(self)
    }
}

/// Blanket implementation: every scalar Manifold gets ManifoldExt.
impl<T: Manifold<Output = crate::Field> + Sized> ManifoldExt for T {}
