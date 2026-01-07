//! # DSL Extension Trait: Fluent Manifold Building
//!
//! Provides a fluent method-chaining API for composing manifold expressions.
//!
//! ## Overview
//!
//! `ManifoldExt` is the primary way users build manifold expressions. It provides:
//!
//! - **Method-chaining API**: `x.sqrt().abs().max(y)`
//! - **Operator overloading**: `x * x + y * y`
//! - **Polymorphic evaluation**: Build once, evaluate with `Field` or `Jet2`
//! - **Type-safe composition**: Expression types are capture compute graphs
//!
//! While `ManifoldExt` methods use `Field` for **type inference during construction**,
//! the resulting expression trees are **fully generic** over any `Computational` input type
//! (`Field`, `Jet2`, `Jet3`). This allows the same expression to evaluate both:
//!
//! - As concrete SIMD values (for rendering)
//! - As automatic derivatives (for gradients, antialiasing)
//!
//! ## Expression Building vs. Evaluation
//!
//! PixelFlow separates these two phases:
//!
//! ### Building Phase
//! ```ignore
//! let circle = (X * X + Y * Y).sqrt() - 1.0;  // Just builds a type tree
//! ```
//!
//! **No computation happens.** The type `Sqrt<Sub<Add<Mul<X,X>, Mul<Y,Y>>, f32>>` is
//! the abstract syntax tree (AST) that represents the computation. The expression tree
//! is a first-class value you can pass around.
//!
//! ### Evaluation Phase
//! ```ignore
//! // Concrete SIMD evaluation
//! let distance = circle.eval(3.0, 4.0, 0.0, 0.0);  // Returns Field (SIMD batch)
//!
//! // Automatic differentiation
//! let x = Jet2::x(3.0);
//! let y = Jet2::y(4.0);
//! let zero = Jet2::new(0.0f32, 0.0f32, 0.0f32);
//! let result = circle.eval_raw(x, y, zero, ...);
//! // result contains: value, ∂/∂x, ∂/∂y
//! ```
//!
//! ## Key Design Principles
//!
//! 1. **Static Typing**: Expression structure is known at compile time
//! 2. **Zero-Cost Abstractions**: All composition overhead erased by monomorphization
//! 3. **Polymorphic by Default**: Same code works with any `Computational` type
//! 4. **Declarative**: Express *what* to compute, not *how*
//!
//! ## Example: Building a Circle Signed Distance Field
//!
//! ```ignore
//! use pixelflow_core::{ManifoldExt, X, Y, Jet2, Field, Manifold};
//!
//! // Build expression
//! let circle = (X * X + Y * Y).sqrt() - 1.0;
//!
//! // Evaluate with Field (normal rendering)
//! let val = circle.eval(3.0, 4.0, 0.0, 0.0);  // Returns Field ~= 4.0
//!
//! // Evaluate with Jet2 (automatic differentiation)
//! let x_jet = Jet2::x(3.0);
//! let y_jet = Jet2::y(4.0);
//! let zero = Jet2::new(0.0f32, 0.0f32, 0.0f32);
//! let result = circle.eval_raw(x_jet, y_jet, zero, zero);
//! // result.val = 4.0 (distance)
//! // result.dx ≈ 0.6 (∂/∂x gradient)
//! // result.dy ≈ 0.8 (∂/∂y gradient)
//! ```
//!
//! ## Method Organization
//!
//! `ManifoldExt` methods fall into three categories:
//!
//! 1. **Evaluation**: `eval`, `eval_at`, `eval_raw`, `constant`
//! 2. **Unary Operations**: `sqrt`, `abs`, `sin`, `cos`, `floor`, `rsqrt`
//! 3. **Binary Operations**: `add`, `sub`, `mul`, `div`, `min`, `max`
//! 4. **Comparisons**: `lt`, `le`, `gt`, `ge`, `select`
//! 5. **Coordinate Transform**: `at` (remap coordinate space)
//! 6. **Functor Operations**: `map` (apply a function to output)

use crate::Manifold;
use crate::combinators::{At, ClosureMap, Map, Select};
use crate::ops::{Abs, Add, Cos, Div, Floor, Ge, Gt, Le, Lt, Max, Min, Mul, Rsqrt, Sin, Sqrt, Sub};

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
/// works with any manifold output type, with some methods specialized for `Field`.
/// The resulting expression trees implement `Manifold<N>` for any `N: Numeric`,
/// enabling evaluation with both `Field` and `Jet2`.
///
/// # Genericity of Expression Trees
///
/// Expression types returned by methods (like `Sqrt<M>`, `Add<L, R>`, etc.) are
/// fully generic and work with any numeric type:
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
/// // Build expression (works with any output type)
/// let expr = X * X + Y;
///
/// // Evaluate with Jet2 for automatic differentiation
/// let x = Jet2::x(5.0.into());
/// let y = Jet2::y(3.0.into());
/// let zero = Jet2::new(0.0f32, 0.0f32, 0.0f32);
/// let result = expr.eval_raw(x, y, zero, zero);
/// // result.val = 28, result.dx = 10 (∂/∂x of x² + y), result.dy = 1 (∂/∂y)
/// ```
pub trait ManifoldExt: Manifold + Sized {
    /// Evaluate the manifold at the given coordinates.
    ///
    /// This convenience method accepts types that convert to `Field`.
    /// For evaluation with other numeric types (like `Jet2`), use `eval_raw` directly.
    ///
    /// Note: Only available for manifolds that output `Field`.
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
    ) -> crate::Field
    where
        Self: Manifold<crate::Field, Output = crate::Field>,
    {
        self.eval_raw(x.into(), y.into(), z.into(), w.into())
    }

    /// Evaluate the manifold at manifold-computed coordinates.
    ///
    /// Takes coordinate expressions (manifolds), evaluates them at origin,
    /// then evaluates self at those coordinates. Immediate execution, no AST.
    ///
    /// Note: Only available for manifolds that output `Field`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Translate: evaluate inner at (x - dx, y - dy)
    /// inner.eval_at(x - dx, y - dy, z, w)
    /// ```
    #[inline(always)]
    fn eval_at<Cx, Cy, Cz, Cw>(&self, x: Cx, y: Cy, z: Cz, w: Cw) -> crate::Field
    where
        Self: Manifold<crate::Field, Output = crate::Field>,
        Cx: Manifold<crate::Field, Output = crate::Field>,
        Cy: Manifold<crate::Field, Output = crate::Field>,
        Cz: Manifold<crate::Field, Output = crate::Field>,
        Cw: Manifold<crate::Field, Output = crate::Field>,
    {
        let zero = crate::Field::from(0.0);
        let new_x = x.eval_raw(zero, zero, zero, zero);
        let new_y = y.eval_raw(zero, zero, zero, zero);
        let new_z = z.eval_raw(zero, zero, zero, zero);
        let new_w = w.eval_raw(zero, zero, zero, zero);
        self.eval_raw(new_x, new_y, new_z, new_w)
    }

    /// Transform the output of this manifold using another manifold.
    ///
    /// This is the algebraic map operation. It applies `transform` to the output of `self`.
    /// `self` output becomes the `X` coordinate for `transform`.
    /// The `Y`, `Z`, and `W` coordinates are passed through unchanged.
    ///
    /// # Arguments
    ///
    /// - `transform`: A manifold to apply to the output. `X` in `transform` refers to `self` output.
    ///
    /// # Returns
    ///
    /// A new manifold representing the transformation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pixelflow_core::{ManifoldExt, X};
    ///
    /// // Double the output
    /// let doubled = X.map(X * 2.0);
    /// ```
    fn map<T>(self, transform: T) -> Map<Self, T>
    where
        Self: Manifold<crate::Field, Output = crate::Field>,
        T: Manifold<crate::Field, Output = crate::Field>,
    {
        Map::new(self, transform)
    }

    /// Lift this manifold's output to ray space via a covariant map.
    ///
    /// This enables conversion from point space (Field) to ray space (PathJet).
    /// The function transforms each Field output into a PathJet.
    ///
    /// Note: Only available for manifolds that output `Field`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pixelflow_core::{ManifoldExt, X, Y, jet::PathJet};
    ///
    /// // Convert screen coordinates to rays from origin
    /// let ray_x = X.lift(PathJet::from_slope);  // origin=0, direction=X
    /// let ray_y = Y.lift(PathJet::from_slope);  // origin=0, direction=Y
    /// let ray_z = 1.0f32.lift(PathJet::from_slope);  // origin=0, direction=1 (forward)
    /// ```
    fn lift<F>(self, func: F) -> ClosureMap<Self, F>
    where
        Self: Manifold<crate::Field, Output = crate::Field>,
        F: Fn(crate::Field) -> crate::jet::PathJet<crate::Field> + Send + Sync,
    {
        ClosureMap::new(self, func)
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

    /// Reciprocal square root (1/sqrt(x)).
    fn rsqrt(self) -> Rsqrt<Self> {
        Rsqrt(self)
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

    /// Branchless conditional select between two manifolds.
    ///
    /// Returns `if_true` where `self` is non-zero (treats as true), `if_false` elsewhere.
    /// This is a **branchless** operation—both branches are evaluated, then one is selected
    /// per SIMD lane. No control flow.
    ///
    /// # Arguments
    ///
    /// - `if_true`: Manifold to use where condition is true
    /// - `if_false`: Manifold to use where condition is false
    ///
    /// # Returns
    ///
    /// A new manifold that computes the conditional selection.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pixelflow_core::{ManifoldExt, X, Y};
    ///
    /// // Checkerboard pattern
    /// let inside_circle = ((X * X + Y * Y) - 1.0).sqrt().lt(0.1);
    /// let pattern = inside_circle.select(0.0, 1.0);
    /// ```
    ///
    /// # Performance
    ///
    /// Both branches are always evaluated. For complex branches, this is more expensive
    /// than a scalar `if` statement, but matches typical shader execution models where
    /// lanes follow independent code paths.
    fn select<T: Manifold, F: Manifold>(self, if_true: T, if_false: F) -> Select<Self, T, F> {
        Select {
            cond: self,
            if_true,
            if_false,
        }
    }

    /// Remap coordinate space before evaluating this manifold.
    ///
    /// Creates a new manifold that first remaps the input coordinates, then evaluates
    /// `self` at the remapped coordinates. This is the mechanism for coordinate transforms
    /// like scaling, translation, and rotation.
    ///
    /// # Arguments
    ///
    /// The coordinate arguments can be:
    /// - **Constants**: `0.0`, `1.5`
    /// - **Coordinate variables**: `X`, `Y`, `Z`, `W`
    /// - **Expressions**: `X / scale`, `X - offset`, `sqrt(X * X + Y * Y)`
    ///
    /// # Returns
    ///
    /// A new manifold that first computes the coordinate transforms, then evaluates
    /// the inner manifold at those coordinates.
    ///
    /// # Example: Scale
    ///
    /// ```ignore
    /// use pixelflow_core::{ManifoldExt, X, Y};
    ///
    /// let circle = (X * X + Y * Y).sqrt() - 1.0;
    ///
    /// // Scale by 2: sample at (x/2, y/2) instead of (x, y)
    /// let scale_factor = 2.0;
    /// let scaled = circle.at(
    ///     X / scale_factor,
    ///     Y / scale_factor,
    ///     Z,
    ///     W,
    /// );
    /// ```
    ///
    /// # Example: Polar Coordinates
    ///
    /// ```ignore
    /// // Convert to polar, then evaluate manifold in polar space
    /// let radius = (X * X + Y * Y).sqrt();
    /// let angle = (Y.atan2(X)) / std::f32::consts::TAU;
    /// let warped = manifold.at(radius, angle, Z, W);
    /// ```
    ///
    /// # Implementation Note
    ///
    /// The coordinate transforms are themselves manifolds (expressions built from operators).
    /// When you call `at`, you're composing two manifolds: the coordinate transform and
    /// the inner manifold. The resulting type captures both.
    fn at<Cx, Cy, Cz, Cw>(self, x: Cx, y: Cy, z: Cz, w: Cw) -> At<Cx, Cy, Cz, Cw, Self>
    where
        Cx: Manifold,
        Cy: Manifold,
        Cz: Manifold,
        Cw: Manifold,
    {
        At {
            inner: self,
            x,
            y,
            z,
            w,
        }
    }

    /// Collapse this manifold expression to a concrete Field value.
    ///
    /// Evaluates the manifold at the origin (0, 0, 0, 0) and returns the result.
    /// This is used internally within PixelFlow to materialize intermediate
    /// AST nodes into concrete Field values.
    ///
    /// Note: Only available for manifolds that output `Field`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pixelflow_core::{ManifoldExt, Field};
    ///
    /// let expression = Field::from(2.0) * Field::from(3.0);
    /// let result: Field = expression.constant();  // result = 6.0
    /// ```
    #[inline(always)]
    fn constant(self) -> crate::Field
    where
        Self: Manifold<crate::Field, Output = crate::Field>,
    {
        self.eval_raw(
            crate::Field::from(0.0),
            crate::Field::from(0.0),
            crate::Field::from(0.0),
            crate::Field::from(0.0),
        )
    }

    /// Type-erase this manifold into a boxed trait object.
    ///
    /// Note: Boxing erases the static type and fixes evaluation to `Field`.
    /// For `Jet2` evaluation, keep the expression statically typed.
    ///
    /// Only available for manifolds that output `Field`.
    fn boxed(self) -> BoxedManifold
    where
        Self: Manifold<crate::Field, Output = crate::Field> + 'static,
    {
        BoxedManifold(Arc::new(self))
    }
}

/// Blanket implementation for all manifolds.
///
/// This makes the DSL methods available for any manifold, regardless of output type.
/// Field-specific methods (eval, constant, map, etc.) are only available when the
/// manifold outputs `Field`, enforced by where clauses on those methods.
impl<T: Manifold + Sized> ManifoldExt for T {}
