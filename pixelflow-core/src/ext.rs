//! # DSL Extension Trait
//!
//! Provides fluent method-chaining API for manifolds.

use crate::Manifold;
use crate::combinators::select::Select;
use crate::ops::{Abs, Add, Div, Ge, Gt, Le, Lt, Max, Min, Mul, Sqrt, Sub};

use alloc::sync::Arc;

/// Type-erased manifold (returning a Field), wrapped in a struct to allow trait implementations.
#[derive(Clone)]
pub struct BoxedManifold(pub Arc<dyn Manifold<Output = crate::Field>>);

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
        self.0.eval_raw(x, y, z, w)
    }
}

// Operator Implementations for BoxedManifold
// This allows writing `a + b` where a or b are BoxedManifolds.

impl<R: Manifold> core::ops::Add<R> for BoxedManifold {
    type Output = Add<Self, R>;
    fn add(self, rhs: R) -> Self::Output {
        Add(self, rhs)
    }
}

impl<R: Manifold> core::ops::Sub<R> for BoxedManifold {
    type Output = Sub<Self, R>;
    fn sub(self, rhs: R) -> Self::Output {
        Sub(self, rhs)
    }
}

impl<R: Manifold> core::ops::Mul<R> for BoxedManifold {
    type Output = Mul<Self, R>;
    fn mul(self, rhs: R) -> Self::Output {
        Mul(self, rhs)
    }
}

impl<R: Manifold> core::ops::Div<R> for BoxedManifold {
    type Output = Div<Self, R>;
    fn div(self, rhs: R) -> Self::Output {
        Div(self, rhs)
    }
}

/// Extension methods for composing manifolds.
pub trait ManifoldExt: Manifold<Output = crate::Field> + Sized {
    /// Evaluate the manifold at the given coordinates.
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

    /// Add this manifold to another.
    fn add<R: Manifold>(self, rhs: R) -> Add<Self, R> {
        Add(self, rhs)
    }

    /// Subtract another manifold from this one.
    fn sub<R: Manifold>(self, rhs: R) -> Sub<Self, R> {
        Sub(self, rhs)
    }

    /// Multiply this manifold by another.
    fn mul<R: Manifold>(self, rhs: R) -> Mul<Self, R> {
        Mul(self, rhs)
    }

    /// Divide this manifold by another.
    fn div<R: Manifold>(self, rhs: R) -> Div<Self, R> {
        Div(self, rhs)
    }

    /// Take the square root of this manifold.
    fn sqrt(self) -> Sqrt<Self> {
        Sqrt(self)
    }

    /// Take the absolute value of this manifold.
    fn abs(self) -> Abs<Self> {
        Abs(self)
    }

    /// Take the element-wise maximum of two manifolds.
    fn max<R: Manifold>(self, rhs: R) -> Max<Self, R> {
        Max(self, rhs)
    }

    /// Take the element-wise minimum of two manifolds.
    fn min<R: Manifold>(self, rhs: R) -> Min<Self, R> {
        Min(self, rhs)
    }

    /// Less than comparison (returns 0 or -1 mask).
    fn lt<R: Manifold>(self, rhs: R) -> Lt<Self, R> {
        Lt(self, rhs)
    }

    /// Greater than comparison (returns 0 or -1 mask).
    fn gt<R: Manifold>(self, rhs: R) -> Gt<Self, R> {
        Gt(self, rhs)
    }

    /// Less than or equal comparison (returns 0 or -1 mask).
    fn le<R: Manifold>(self, rhs: R) -> Le<Self, R> {
        Le(self, rhs)
    }

    /// Greater than or equal comparison (returns 0 or -1 mask).
    fn ge<R: Manifold>(self, rhs: R) -> Ge<Self, R> {
        Ge(self, rhs)
    }

    /// Select between two manifolds based on this manifold (as a condition mask).
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
        BoxedManifold(Arc::new(self))
    }
}

impl<T: Manifold<Output = crate::Field> + Sized> ManifoldExt for T {}
