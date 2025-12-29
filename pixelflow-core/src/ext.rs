//! # DSL Extension Trait
//!
//! Provides fluent method-chaining API for manifolds.
//!
//! `ManifoldExt` enables building expressions via method chaining. While the
//! trait uses `Field` for type inference during expression construction, the
//! resulting expression trees are **fully generic** over the `Numeric` type
//! and can be evaluated with either `Field` (for concrete SIMD computation)
//! or `Jet2` (for automatic differentiation).
//!
//! # Expression Building vs. Evaluation
//!
//! - **Building**: Uses `ManifoldExt` methods with implicit `Field` type inference
//! - **Evaluation**: The resulting expression implements `Manifold<N>` for any `N: Numeric`
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::{ManifoldExt, X, Y, Jet2, Field, Manifold};
//!
//! // Build expression using ManifoldExt
//! let circle = (X * X + Y * Y).sqrt();
//!
//! // Evaluate with Field (concrete values)
//! let val = circle.eval(3.0, 4.0, 0.0, 0.0); // Returns 5.0
//!
//! // Evaluate with Jet2 (automatic differentiation)
//! let x_jet = Jet2::x(3.0.into());
//! let y_jet = Jet2::y(4.0.into());
//! let zero = Jet2::constant(0.0.into());
//! let result = circle.eval_raw(x_jet, y_jet, zero, zero);
//! // result.val = 5.0, result.dx = 0.6, result.dy = 0.8 (normalized gradient)
//! ```

use crate::Manifold;
use crate::combinators::{At, Map, Select};
use crate::ops::{Abs, Add, Atan2, Cos, Div, Floor, Ge, Gt, Le, Lt, Max, Min, Mul, Sin, Sqrt, Sub};

use alloc::sync::Arc;

/// Type-erased manifold (returning Field), wrapped in a struct to allow trait implementations.
///
/// Note: `BoxedManifold` is Field-specific because trait objects require a concrete type.
/// For generic numeric contexts, use static dispatch instead.
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
///
/// This trait provides a fluent API for building manifold expressions. The trait
/// is bound to `Manifold<Output = Field>` for type inference during expression
/// construction, but the resulting expression trees implement `Manifold<N>` for
/// any `N: Numeric`, enabling evaluation with both `Field` and `Jet2`.
///
/// # Genericity of Expression Trees
///
/// While `ManifoldExt` uses `Field` for type inference, the expression types
/// returned by its methods (like `Sqrt<M>`, `Add<L, R>`, etc.) are fully generic:
///
/// ```ignore
/// let expr = X.sqrt();  // expr has type Sqrt<X>
/// // Sqrt<X> implements:
/// //   - Manifold<Field, Output = Field>
/// //   - Manifold<Jet2, Output = Jet2>
/// ```
///
/// # Example with Automatic Differentiation
///
/// ```ignore
/// use pixelflow_core::{ManifoldExt, X, Y, Jet2, Manifold, Numeric};
///
/// // Build expression (Field is used for type inference)
/// let expr = X * X + Y;
///
/// // Evaluate with Jet2 for automatic differentiation
/// let x = Jet2::x(5.0.into());
/// let y = Jet2::y(3.0.into());
/// let zero = Jet2::constant(0.0.into());
/// let result = expr.eval_raw(x, y, zero, zero);
/// // result.val = 28, result.dx = 10 (∂/∂x of x² + y), result.dy = 1 (∂/∂y)
/// ```
pub trait ManifoldExt: Manifold<Output = crate::Field> + Sized {
    /// Evaluate the manifold at the given coordinates.
    ///
    /// This convenience method accepts types that convert to `Field`.
    /// For evaluation with other numeric types (like `Jet2`), use `eval_raw` directly.
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

    /// Map a function over this manifold's output (functor fmap).
    fn map<F>(self, func: F) -> Map<Self, F>
    where
        F: Fn(crate::Field) -> crate::Field + Send + Sync,
    {
        Map::new(self, func)
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

    /// Floor (round toward negative infinity).
    fn floor(self) -> Floor<Self> {
        Floor(self)
    }

    /// Sine function.
    fn sin(self) -> Sin<Self> {
        Sin(self)
    }

    /// Cosine function.
    fn cos(self) -> Cos<Self> {
        Cos(self)
    }

    /// Arctangent of two arguments: atan2(self, rhs).
    ///
    /// Computes the arc tangent of self/rhs, using the signs of both arguments
    /// to determine the correct quadrant.
    fn atan2<R: Manifold>(self, rhs: R) -> Atan2<Self, R> {
        Atan2(self, rhs)
    }

    /// Element-wise maximum.
    fn max<R: Manifold>(self, rhs: R) -> Max<Self, R> {
        Max(self, rhs)
    }

    /// Element-wise minimum.
    fn min<R: Manifold>(self, rhs: R) -> Min<Self, R> {
        Min(self, rhs)
    }

    /// Less than comparison.
    fn lt<R: Manifold>(self, rhs: R) -> Lt<Self, R> {
        Lt(self, rhs)
    }

    /// Greater than comparison.
    fn gt<R: Manifold>(self, rhs: R) -> Gt<Self, R> {
        Gt(self, rhs)
    }

    /// Less than or equal comparison.
    fn le<R: Manifold>(self, rhs: R) -> Le<Self, R> {
        Le(self, rhs)
    }

    /// Greater than or equal comparison.
    fn ge<R: Manifold>(self, rhs: R) -> Ge<Self, R> {
        Ge(self, rhs)
    }

    /// Conditional select: if self is non-zero, return if_true, else if_false.
    fn select<T: Manifold, F: Manifold>(self, if_true: T, if_false: F) -> Select<Self, T, F> {
        Select {
            cond: self,
            if_true,
            if_false,
        }
    }

    /// Pin this manifold to manifold-computed coordinates.
    ///
    /// Returns a new manifold that evaluates the inner manifold at the given
    /// coordinate manifolds (which can be constants or expressions).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Evaluate at constant field coordinates
    /// let at_origin = manifold.at(0.0.into(), 0.0.into(), 0.0.into(), 0.0.into());
    /// let result = at_origin.eval_raw(Field::from(100.0), Field::from(100.0), Field::from(100.0), Field::from(100.0));
    /// // result is the same as manifold.eval_raw(0.0, 0.0, 0.0, 0.0)
    /// ```
    fn at(self, x: crate::Field, y: crate::Field, z: crate::Field, w: crate::Field) -> At<crate::Field, crate::Field, crate::Field, crate::Field, Self> {
        At { inner: self, x, y, z, w }
    }

    /// Type-erase this manifold into a boxed trait object.
    ///
    /// Note: Boxing erases the static type and fixes evaluation to `Field`.
    /// For `Jet2` evaluation, keep the expression statically typed.
    fn boxed(self) -> BoxedManifold
    where
        Self: 'static,
    {
        BoxedManifold(Arc::new(self))
    }
}

impl<T: Manifold<Output = crate::Field> + Sized> ManifoldExt for T {}
