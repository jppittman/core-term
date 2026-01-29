//! # Manifold Trait
//!
//! The core abstraction of pixelflow-core: **a function from a domain to values**.
//!
//! ## What Is a Manifold?
//!
//! In pixelflow, a manifold is:
//!
//! - **A functor**: Maps a domain `P` to output values
//! - **Compositional**: Manifolds combine via operators (`+`, `*`, `/`) to build larger manifolds
//! - **A compile-time graph**: The type system IS the compute graph
//! - **Declarative**: Users express *what* to compute, not *how* to compute it
//!
//! ## The Domain-Generic Design
//!
//! The manifold trait is parameterized by a domain type `P`:
//!
//! ```ignore
//! trait Manifold<P> {
//!     type Output;
//!     fn eval(&self, p: P) -> Self::Output;
//! }
//! ```
//!
//! This design eliminates the "tax of 4 dimensions" - a 2D kernel only pays for 2D:
//!
//! - **2D kernel**: `P = (Field, Field)` - just x and y
//! - **3D kernel**: `P = (Field, Field, Field)` - x, y, and z
//! - **With let binding**: `P = (V, Rest)` - bound value prepended to domain
//!
//! ## Example: Building a Circle
//!
//! ```ignore
//! use pixelflow_core::{X, Y, Manifold};
//!
//! // Type: Sqrt<Add<Mul<X, X>, Mul<Y, Y>>>
//! let distance_from_origin = (X * X + Y * Y).sqrt();
//!
//! // Evaluate on 2D domain
//! let val = distance_from_origin.eval((Field::from(3.0), Field::from(4.0)));
//! // Returns Field = 5.0
//! ```
//!
//! ## Let Bindings as Domain Extension
//!
//! Let bindings naturally extend the domain:
//!
//! ```ignore
//! // let dist = sqrt(x² + y²); dist - 1.0
//! let circle = Let(
//!     (X * X + Y * Y).sqrt(),  // compute distance
//!     Var::<N0> - 1.0          // use it: Var<0> reads head of domain
//! );
//!
//! // Domain flows:
//! // (x, y) → Let → LetExtended(dist, (x, y)) → body
//! ```
//!
//! ## Automatic Differentiation
//!
//! The same expression works with different numeric backends:
//!
//! ```ignore
//! // With Field: concrete SIMD values
//! let val = circle.eval((Field::from(3.0), Field::from(4.0)));
//!
//! // With Jet2: automatic differentiation
//! let x = Jet2::x(Field::from(3.0));
//! let y = Jet2::y(Field::from(4.0));
//! let result = circle.eval((x, y));
//! // result.val = 4.0, result.dx ≈ 0.6, result.dy ≈ 0.8
//! ```

use crate::{Discrete, Field};

/// A manifold is a function from a domain to a value.
///
/// # Overview
///
/// `Manifold<P>` is a function that takes a domain `P` and returns some output value.
/// It's the foundational abstraction for **declarative, composable computation** in pixelflow.
///
/// # Key Properties
///
/// - **Domain-generic**: Works with any domain type (2D, 3D, 4D, let-extended)
/// - **Functor**: Maps the input domain to an output type
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
/// # Type Parameter: `P` (Domain)
///
/// The `P` type parameter is the domain the manifold operates on:
///
/// - `P = (Field, Field)`: 2D spatial domain
/// - `P = (Field, Field, Field, Field)`: 4D spatial domain
/// - `P = (Jet2, Jet2)`: 2D with automatic differentiation
/// - `P = LetExtended<V, Rest>`: Domain extended with a let binding
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
/// The type tree is the compute graph. When `eval` is called, the compiler
/// monomorphizes and inlines the entire tree into a single fused kernel.
pub trait Manifold<P = (Field, Field, Field, Field)>: Send + Sync {
    /// The type this manifold produces when evaluated.
    type Output;

    /// Evaluate the manifold at the given domain point.
    ///
    /// # Arguments
    ///
    /// - `p`: The domain point to evaluate at
    ///
    /// # Returns
    ///
    /// An instance of `Self::Output` computed at the given domain point.
    fn eval(&self, p: P) -> Self::Output;
}

// ============================================================================
// Legacy Compatibility: eval_raw adapter
// ============================================================================

/// Extension trait providing the legacy `eval_raw` interface.
///
/// This allows existing code to continue using the 4-argument signature
/// while internally delegating to the new domain-generic `eval`.
pub trait ManifoldCompat<I>: Manifold<(I, I, I, I)> {
    /// Evaluate using the legacy 4-coordinate signature.
    ///
    /// This is a compatibility shim that constructs a 4-tuple and calls `eval`.
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output
    where
        I: Copy,
    {
        self.eval((x, y, z, w))
    }
}

// Blanket impl for all manifolds on 4D domains
impl<M, I> ManifoldCompat<I> for M where M: Manifold<(I, I, I, I)> {}

// ============================================================================
// Scalar Constants as Manifolds
// ============================================================================

// f32 constants are manifolds - they ignore the domain and return themselves
impl<P> Manifold<P> for f32
where
    P: Send + Sync,
{
    type Output = Field;
    #[inline(always)]
    fn eval(&self, _p: P) -> Field {
        Field::from(*self)
    }
}

// i32 constants are manifolds - they ignore the domain and return as Field
impl<P> Manifold<P> for i32
where
    P: Send + Sync,
{
    type Output = Field;
    #[inline(always)]
    fn eval(&self, _p: P) -> Field {
        Field::from(*self)
    }
}

// ============================================================================
// Computational Types as Constant Manifolds
// ============================================================================
//
// All Computational types (Field, Jet2, Jet3, etc.) are constant manifolds.
// They ignore the input domain and return themselves unchanged.
// This allows them to be used in Let bindings as bound values.

macro_rules! impl_constant_manifold {
    ($($ty:ty),* $(,)?) => {
        $(
            impl<P> Manifold<P> for $ty
            where
                P: Send + Sync,
            {
                type Output = $ty;
                #[inline(always)]
                fn eval(&self, _p: P) -> $ty {
                    *self
                }
            }
        )*
    };
}

impl_constant_manifold!(
    Field,
    Discrete,
    crate::jet::Jet2,
    crate::jet::Jet3,
);

// ============================================================================
// Smart Pointer Implementations
// ============================================================================

// Arc<M> is a manifold if M is
impl<P, M: Manifold<P> + ?Sized> Manifold<P> for alloc::sync::Arc<M>
where
    P: Copy,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        (**self).eval(p)
    }
}

// Box<M> is a manifold if M is
impl<P, M: Manifold<P> + ?Sized> Manifold<P> for alloc::boxed::Box<M>
where
    P: Copy,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        (**self).eval(p)
    }
}

// &M is a manifold if M is
impl<P, M: Manifold<P> + ?Sized> Manifold<P> for &M
where
    P: Copy,
{
    type Output = M::Output;
    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        (**self).eval(p)
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

impl<P, F, M> Manifold<P> for Thunk<F>
where
    P: Copy,
    F: Fn() -> M + Send + Sync,
    M: Manifold<P>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        (self.0)().eval(p)
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
