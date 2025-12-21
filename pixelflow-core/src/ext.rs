//! # DSL Extension Trait
//!
//! Provides fluent method-chaining API for manifolds.

use crate::Manifold;
use crate::combinators::Select;
use crate::ops::{Abs, Add, Div, Ge, Gt, Le, Lt, Max, Min, Mul, Sqrt, Sub};

use alloc::boxed::Box;

/// Type-erased manifold.
pub type BoxedManifold = Box<dyn Manifold>;

impl Manifold for BoxedManifold {
    #[inline(always)]
    fn eval(
        &self,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        (**self).eval(x, y, z, w)
    }
}

/// Extension methods for composing manifolds.
pub trait ManifoldExt: Manifold + Sized {
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

/// Blanket implementation: every Manifold gets ManifoldExt.
impl<T: Manifold + Sized> ManifoldExt for T {}
