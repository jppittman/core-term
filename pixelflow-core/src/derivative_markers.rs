//! # Type-Level Derivative Markers
//!
//! This module defines a type-level system for automatic differentiation where
//! derivatives are types, not runtime data. The egraph can then apply calculus
//! rules as rewrite rules.
//!
//! ## Design Philosophy
//!
//! Instead of:
//! ```ignore
//! struct Dual<const N: usize> { val: f32, partials: [f32; N] }
//! ```
//!
//! We have:
//! ```ignore
//! Field<Jet<(D<X>, D<Y>)>>  // 2D gradient
//! Field<Jet<(D<X>, D<Y>, D<D<X>>, D<X,Y>, D<D<Y>>)>>  // 2D + Hessian
//! ```
//!
//! The partials are named by their derivative type, not by array index.
//! This enables:
//! 1. Type-safe derivative access (can't mix up dx and dy)
//! 2. Egraph rewrite rules for calculus
//! 3. Composable higher-order derivatives
//! 4. Zero runtime overhead (all resolved at compile time)

use core::marker::PhantomData;

// ============================================================================
// Coordinate Markers (What we differentiate with respect to)
// ============================================================================

/// X coordinate marker
pub struct X;

/// Y coordinate marker
pub struct Y;

/// Z coordinate marker
pub struct Z;

/// Time coordinate (for animations)
pub struct T;

/// Generic parameter marker (for parametric derivatives)
pub struct P<const I: usize>;

// ============================================================================
// Derivative Type Constructor
// ============================================================================

/// Derivative marker: `D<X>` means "derivative with respect to X"
///
/// This is a type constructor, not a value. It marks what derivative
/// a partial represents.
///
/// ## Examples
/// - `D<X>` — first derivative w.r.t. X (∂/∂x)
/// - `D<Y>` — first derivative w.r.t. Y (∂/∂y)
/// - `D<D<X>>` — second derivative w.r.t. X (∂²/∂x²)
pub struct D<V>(PhantomData<V>);

/// Mixed partial derivative marker
///
/// `Mixed<X, Y>` represents ∂²/∂x∂y
///
/// Note: Mixed<X, Y> = Mixed<Y, X> by Schwarz's theorem (for smooth functions)
pub struct Mixed<V1, V2>(PhantomData<(V1, V2)>);

// ============================================================================
// Jet Type - Collection of Tracked Derivatives
// ============================================================================

/// A jet tracking a specific set of derivatives.
///
/// The type parameter `Partials` is a tuple of derivative markers.
///
/// ## Examples
/// ```ignore
/// Jet<()>                           // Just value, no derivatives
/// Jet<(D<X>,)>                      // 1D: value + dx
/// Jet<(D<X>, D<Y>)>                 // 2D: value + dx, dy
/// Jet<(D<X>, D<Y>, D<Z>)>           // 3D: value + dx, dy, dz
/// Jet<(D<X>, D<Y>, D<D<X>>, Mixed<X,Y>, D<D<Y>>)>  // 2D + Hessian
/// ```
pub struct Jet<Partials>(PhantomData<Partials>);

// ============================================================================
// Type Aliases for Common Jet Configurations
// ============================================================================

/// No derivatives tracked (just the value)
pub type Jet0 = Jet<()>;

/// 1D autodiff: value + ∂/∂x
pub type Jet1D = Jet<(D<X>,)>;

/// 2D autodiff: value + ∂/∂x, ∂/∂y
pub type Jet2D = Jet<(D<X>, D<Y>)>;

/// 3D autodiff: value + ∂/∂x, ∂/∂y, ∂/∂z
pub type Jet3D = Jet<(D<X>, D<Y>, D<Z>)>;

/// 2D with Hessian: value + gradient + second derivatives
/// Tracks: ∂/∂x, ∂/∂y, ∂²/∂x², ∂²/∂x∂y, ∂²/∂y²
pub type Jet2DH = Jet<(D<X>, D<Y>, D<D<X>>, Mixed<X, Y>, D<D<Y>>)>;

/// 3D with Hessian (6 second derivatives due to symmetry)
pub type Jet3DH = Jet<(
    D<X>, D<Y>, D<Z>,
    D<D<X>>, Mixed<X, Y>, Mixed<X, Z>,
    D<D<Y>>, Mixed<Y, Z>,
    D<D<Z>>,
)>;

// ============================================================================
// Traits for Derivative Access
// ============================================================================

/// Trait for types that have a derivative with respect to V
pub trait HasPartial<V> {
    /// The storage type for this partial
    type Storage;

    /// Get the partial derivative with respect to V
    fn partial(&self) -> &Self::Storage;
}

/// Trait for types that have a value component
pub trait HasValue {
    type Storage;
    fn value(&self) -> &Self::Storage;
}

// ============================================================================
// Egraph Rewrite Rules (Conceptual)
// ============================================================================
//
// These would be implemented in the egraph, not as Rust traits.
// Shown here for documentation.
//
// ## Linearity Rules
// D<V>[f + g] → D<V>[f] + D<V>[g]
// D<V>[c * f] → c * D<V>[f]  (where c is constant w.r.t. V)
// D<V>[c] → 0                 (where c is constant w.r.t. V)
//
// ## Product Rule
// D<V>[f * g] → D<V>[f] * g + f * D<V>[g]
//
// ## Quotient Rule
// D<V>[f / g] → (D<V>[f] * g - f * D<V>[g]) / (g * g)
//
// ## Chain Rule
// D<V>[f(g)] → D[f](g) * D<V>[g]
//
// ## Common Functions
// D<V>[sqrt(f)] → D<V>[f] / (2 * sqrt(f))
// D<V>[sin(f)] → cos(f) * D<V>[f]
// D<V>[cos(f)] → -sin(f) * D<V>[f]
// D<V>[exp(f)] → exp(f) * D<V>[f]
// D<V>[ln(f)] → D<V>[f] / f
// D<V>[f^n] → n * f^(n-1) * D<V>[f]
//
// ## Second Derivatives (Hessian elements)
// D<V>[D<W>[f]] = D<W>[D<V>[f]]  (Schwarz's theorem)
//
// ============================================================================

// ============================================================================
// Storage Mapping
// ============================================================================

/// Maps a Jet type to its concrete SIMD storage layout.
///
/// This is how `Field<Jet<...>>` knows what memory to allocate.
pub trait JetStorage {
    /// The concrete storage type (tuple of SIMD vectors)
    type Storage: Copy + Clone + Send + Sync;

    /// Number of f32 components (1 + number of partials)
    const COMPONENTS: usize;
}

// Implementation for common cases
impl JetStorage for Jet<()> {
    type Storage = (); // No extra storage, just the value in Field
    const COMPONENTS: usize = 1;
}

impl JetStorage for Jet<(D<X>,)> {
    // Storage: (value, dx)
    // But value is in Field itself, so just dx
    type Storage = (); // Single partial
    const COMPONENTS: usize = 2;
}

impl JetStorage for Jet<(D<X>, D<Y>)> {
    // Storage: (dx, dy)
    type Storage = (); // Two partials
    const COMPONENTS: usize = 3;
}

impl JetStorage for Jet<(D<X>, D<Y>, D<Z>)> {
    type Storage = (); // Three partials
    const COMPONENTS: usize = 4;
}

impl JetStorage for Jet<(D<X>, D<Y>, D<D<X>>, Mixed<X, Y>, D<D<Y>>)> {
    type Storage = (); // 2 first derivs + 3 second derivs
    const COMPONENTS: usize = 6;
}

// ============================================================================
// Example: How a 2D Jet Field Would Work
// ============================================================================
//
// ```ignore
// // Define a 2D scalar field
// fn circle_sdf<J: JetStorage>(p: (Field<J>, Field<J>)) -> Field<J>
// where
//     J: HasPartial<X> + HasPartial<Y>,
// {
//     let (x, y) = p;
//     (x * x + y * y).sqrt() - 1.0
// }
//
// // The egraph sees:
// //   sqrt(x*x + y*y) - 1.0
// //
// // And can derive (via rewrite rules):
// //   D<X>[sqrt(x*x + y*y) - 1.0]
// //   = D<X>[sqrt(x*x + y*y)] - D<X>[1.0]
// //   = D<X>[sqrt(x*x + y*y)] - 0
// //   = (D<X>[x*x + y*y]) / (2 * sqrt(x*x + y*y))
// //   = (D<X>[x*x] + D<X>[y*y]) / (2 * sqrt(x*x + y*y))
// //   = (2*x*D<X>[x] + 2*y*D<X>[y]) / (2 * sqrt(x*x + y*y))
// //   = (2*x*1 + 2*y*0) / (2 * sqrt(x*x + y*y))
// //   = x / sqrt(x*x + y*y)
// //
// // This is the gradient! The egraph derived it symbolically.
// ```
//
// ============================================================================

// ============================================================================
// Lifting: Promoting Field to Field<Jet<...>>
// ============================================================================

/// Trait for lifting a value to a jet with specific derivatives seeded.
pub trait Lift<J: JetStorage> {
    type Output;

    /// Lift to a jet where this is treated as a constant (all partials = 0)
    fn constant(self) -> Self::Output;

    /// Lift to a jet where this is treated as a variable w.r.t. the given coordinate
    fn variable<V>(self) -> Self::Output
    where
        J: HasPartial<V>;
}

// ============================================================================
// The Key Insight: Egraph Calculus
// ============================================================================
//
// Traditional autodiff (forward mode):
//   - Dual numbers carry derivatives at runtime
//   - Chain rule is imperative code in each operation
//   - Arrays of partials, closures, runtime overhead
//
// Type-level derivatives with egraph:
//   - Derivatives are requested via types
//   - Egraph applies calculus rules as rewrites
//   - Final expression is optimized THEN executed
//   - No runtime derivative tracking, just the result
//
// The egraph doesn't "run" the derivative - it rewrites the expression
// to produce the derivative symbolically, then optimizes the combined
// expression (original + all requested derivatives) before codegen.
//
// This is why pixelflow uses an AST/IR instead of immediate evaluation.
// The egraph can see the whole computation graph and apply algebraic
// simplifications that would be invisible to runtime autodiff.
//
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_types_are_zero_sized() {
        assert_eq!(core::mem::size_of::<X>(), 0);
        assert_eq!(core::mem::size_of::<D<X>>(), 0);
        assert_eq!(core::mem::size_of::<D<D<X>>>(), 0);
        assert_eq!(core::mem::size_of::<Mixed<X, Y>>(), 0);
        assert_eq!(core::mem::size_of::<Jet<(D<X>, D<Y>)>>(), 0);
    }
}
