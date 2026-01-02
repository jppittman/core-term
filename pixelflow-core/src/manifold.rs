//! # Manifold Trait
//!
//! The core abstraction of pixelflow-core: **a function from 4D coordinates to values**.
//!
//! ## What Is a Manifold?
//!
//! In pixelflow, a manifold is:
//!
//! - **A functor**: Maps coordinate space (scalars in a field) to output values
//! - **Compositional**: Manifolds combine via operators (`+`, `*`, `/`) to build larger manifolds
//! - **A compile-time graph**: The type system IS the compute graph
//! - **Declarative**: Users express *what* to compute, not *how* to compute it
//!
//! ## The Manifold Philosophy
//!
//! PixelFlow turns **algebraic expressions into fused SIMD kernels** without runtime dispatch.
//!
//! ### Example: Building a Circle
//! ```ignore
//! use pixelflow_core::{X, Y, Manifold};
//!
//! // Type: Sqrt<Add<Mul<X, X>, Mul<Y, Y>>>
//! let distance_from_origin = (X * X + Y * Y).sqrt();
//!
//! // This is NOT computation - it's an AST. Evaluation is decoupled from definition.
//! ```
//!
//! ### Evaluation is Polymorphic
//! The same expression can evaluate with different numeric backends:
//!
//! ```ignore
//! // Concrete evaluation with SIMD vectors
//! let val = distance_from_origin.eval_raw(Field::from(3.0), Field::from(4.0), ...);
//! // Returns Field = 5.0 (SIMD batch)
//!
//! // Automatic differentiation
//! let x = Jet2::x(3.0);
//! let y = Jet2::y(4.0);
//! let result = distance_from_origin.eval_raw(x, y, ...);
//! // Returns Jet2 with value 5.0 and gradients (∂/∂x, ∂/∂y)
//! ```
//!
//! ## The Type System as a Compute Graph
//!
//! In pixelflow, **types are not just information—they are the compile-time representation
//! of a computation graph**. Each operator (`Add`, `Mul`, `Sqrt`, etc.) corresponds to a
//! type that:
//!
//! 1. Captures the structure of the computation
//! 2. Is monomorphized into a fused kernel with zero runtime dispatch
//! 3. Can be evaluated with any `Computational` input type
//!
//! The compiler sees:
//! ```ignore
//! Add<Mul<X, X>, Mul<Y, Y>>  // Type = Compute graph
//! ```
//!
//! And generates:
//! ```text
//! Single fused SIMD loop: (x₀² + y₀, x₁² + y₁, ..., x₁₅² + y₁₅)
//! ```
//!
//! No vtable, no if-statements, no branching. **Just algebra**.
//!
//! ## Manifolds vs. Fields/Jets (The IR)
//!
//! **Users should NOT write code that directly manipulates Fields or Jets.**
//!
//! - **Field** and **Jet2/Jet3** are the **intermediate representation (IR)**
//! - **Manifolds** are the **high-level abstraction** users compose
//!
//! Think of it like this:
//! - A **Manifold** is like a mathematical expression or equation
//! - **Field/Jet** is like LLVM IR—powerful but detailed
//!
//! Users should:
//! - ✅ Write manifolds: `(X * X + Y * Y).sqrt()`
//! - ❌ NOT write IR directly: `Field::raw_mul(...)`

use crate::Field;

/// A manifold is a function from coordinates to a value.
///
/// # Overview
///
/// `Manifold` is a function that takes 4D coordinates (typically x, y, z, w) and returns
/// some output value. It's the foundational abstraction for **declarative, composable
/// computation** in pixelflow.
///
/// # Key Properties
///
/// - **Functor**: Maps the input coordinate type (field) to an output type
/// - **Generic over evaluation strategy**: Via the type parameter `I`, supports:
///   - Concrete SIMD evaluation (`Field`)
///   - Automatic differentiation (`Jet2` or `Jet3`)
/// - **Zero-cost**: Polymorphic evaluation is resolved at compile time via monomorphization
/// - **Composable**: Manifolds combine via operators to build larger manifolds
///
/// # Associated Type: `Output`
///
/// The `Output` associated type specifies what the manifold produces when evaluated.
///
/// Common examples:
/// - **Scalar manifold**: `Output = Field` — produces single values per coordinate
/// - **Color manifold**: `Output = Discrete` — produces packed RGBA u32 pixels
/// - **Vector manifold**: `Output = (Field, Field, Field, Field)` — produces 4D vectors
///
/// # Type Parameter: `I` (Input Type)
///
/// The `I` type parameter determines *how* the manifold is evaluated, not *what* it
/// computes. All manifold expressions are generic over `I`, so the same expression
/// can be evaluated with:
///
/// - `I = Field`: Concrete SIMD evaluation (e.g., 16 values per SIMD lane on AVX-512)
/// - `I = Jet2`: Automatic differentiation (value + 2 partial derivatives)
/// - `I = Jet3`: Extended AD (value + 3 partial derivatives)
///
/// The bound `Computational` ensures `I` supports:
/// - Arithmetic operators (`+`, `-`, `*`, `/`)
/// - Constant creation (`from_f32`, `sequential`)
///
/// # Composition
///
/// Manifolds compose via operator overloading. The type system captures the structure:
///
/// ```ignore
/// use pixelflow_core::{X, Y};
///
/// let m1 = X * X;           // Type: Mul<X, X>
/// let m2 = Y * Y;           // Type: Mul<Y, Y>
/// let m3 = m1 + m2;         // Type: Add<Mul<X, X>, Mul<Y, Y>>
/// let m4 = m3.sqrt();       // Type: Sqrt<Add<...>>
/// ```
///
/// The type tree is the compute graph. When `eval_raw` is called, the compiler
/// monomorphizes and inlines the entire tree into a single fused kernel.
///
/// # Philosophy: Expression Trees, Not Evaluation Loops
///
/// PixelFlow separates **expression definition** from **evaluation**:
///
/// ```ignore
/// // Define: Build the expression tree (no computation yet)
/// let circle = (X * X + Y * Y).sqrt() - 1.0;
///
/// // Evaluate: Run the computation with concrete coordinates
/// let result = circle.eval_raw(
///     Field::from(3.0),
///     Field::from(4.0),
///     Field::from(0.0),
///     Field::from(0.0),
/// );
/// ```
///
/// This design enables:
/// - **Monomorphic kernels**: No vtable dispatch, one specialized kernel per expression
/// - **Automatic differentiation**: Same expression works with `Jet2` for gradients
/// - **Compiler optimization**: LLVM sees the full expression, inlines everything
///
/// # Internal Representation
///
/// The `eval_raw` method's raw SIMD argument types (`Field`, `Jet2`, etc.) are **not
/// intended for direct code**. These are the **intermediate representation (IR)**.
/// Users compose at the manifold level; the library handles IR internally.
///
/// If you find yourself writing `Field::...` or using jet methods directly,
/// you're likely using the library wrong. Use manifold operators instead.
///
/// ## For Custom Manifold Implementers
///
/// If you're implementing a custom `Manifold`, you'll receive `Field` or `Jet2` values
/// in `eval_raw`. **You must compose these using operators**, not call low-level methods:
///
/// ```ignore
/// fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
///     // ✅ Correct: Use operators (they're polymorphic)
///     (x * x + y * y).sqrt()
///
///     // ❌ Wrong: Don't call Field methods directly
///     // Field::raw_mul(x, x)
/// }
/// ```
///
/// Operators like `+`, `*`, `sqrt` are polymorphic (work with any `Computational` type)
/// and compose properly with automatic differentiation (`Jet2`, `Jet3`).
pub trait Manifold<I: crate::numeric::Computational = Field>: Send + Sync {
    /// The type this manifold produces when evaluated.
    type Output;

    /// Evaluate the manifold at the given 4D coordinates.
    ///
    /// # Arguments
    ///
    /// - `x`, `y`, `z`, `w`: Coordinate values of type `I`
    ///
    /// The input type `I` determines the evaluation mode:
    /// - `I = Field`: Evaluates with concrete SIMD values (normal rendering)
    /// - `I = Jet2`: Evaluates with automatic differentiation (gradients)
    /// - `I = Jet3`: Evaluates with extended automatic differentiation
    ///
    /// # Returns
    ///
    /// An instance of `Self::Output` computed at the given coordinates.
    ///
    /// # Note
    ///
    /// This method is the **only** way to extract values from a manifold.
    /// Users typically call this indirectly via higher-level functions:
    /// - [`crate::materialize_discrete`] for color manifolds
    /// - [`crate::materialize`] for vector manifolds
    /// - [`ManifoldExt::eval`](crate::ManifoldExt::eval) for convenience evaluation
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pixelflow_core::{X, Y, Manifold, Field};
    ///
    /// let m = X + Y;
    /// let result = m.eval_raw(
    ///     Field::from(3.0),
    ///     Field::from(4.0),
    ///     Field::from(0.0),
    ///     Field::from(0.0),
    /// );
    /// ```
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
    fn eval_raw(
        &self,
        _x: crate::jet::Jet2,
        _y: crate::jet::Jet2,
        _z: crate::jet::Jet2,
        _w: crate::jet::Jet2,
    ) -> crate::jet::Jet2 {
        *self
    }
}

// Jet3 is a constant manifold - ignores input, always returns itself
impl Manifold<crate::jet::Jet3> for crate::jet::Jet3 {
    type Output = crate::jet::Jet3;
    #[inline(always)]
    fn eval_raw(
        &self,
        _x: crate::jet::Jet3,
        _y: crate::jet::Jet3,
        _z: crate::jet::Jet3,
        _w: crate::jet::Jet3,
    ) -> crate::jet::Jet3 {
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

// ============================================================================
// Differentiable Trait
// ============================================================================

/// A manifold that can provide analytical gradients.
///
/// **Key insight:** Differentiable manifolds of Field ↔ Manifolds of Jet2
///
/// - All differentiable manifolds can return Jet2 (value + gradients)
/// - All manifolds returning Jet2 are differentiable
///
/// This trait uses associated types to specify which coordinates the manifold
/// differentiates with respect to, enabling efficient analytical derivatives
/// without wasteful autodiff propagation.
///
/// # Example
///
/// ```ignore
/// // A quadratic curve with analytical derivatives
/// impl<K, D> Differentiable for Quad<K, D> {
///     type DiffWrt = (X, Y);  // Differentiable w.r.t. spatial coordinates
/// }
/// ```
pub trait Differentiable {
    /// Which coordinates this manifold is differentiable with respect to.
    ///
    /// Common values:
    /// - `(X, Y)` — Spatial derivatives (most common for 2D graphics)
    /// - `(X, Y, Z)` — 3D spatial derivatives
    /// - `(X, Y, Z, W)` — Full 4D derivatives
    type DiffWrt;
}
