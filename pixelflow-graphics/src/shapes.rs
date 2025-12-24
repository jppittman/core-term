//! Basic shapes as manifold stencils.
//!
//! Each shape is a function that takes foreground and background manifolds
//! and returns a composed manifold using Select. This enables:
//! - Natural composition via nesting
//! - Automatic bounds checking (outer shapes clip inner)
//! - Short-circuit evaluation via Select's all/any checks

use pixelflow_core::{And, Field, Ge, Le, Manifold, ManifoldExt, Select, X, Y};

// ============================================================================
// Type Aliases
// ============================================================================

/// The unit square condition: (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
pub type UnitSquareCond = And<And<And<Ge<X, f32>, Le<X, f32>>, Ge<Y, f32>>, Le<Y, f32>>;

/// A manifold bounded to the unit square.
pub type Bounded<M> = Select<UnitSquareCond, M, f32>;

// ============================================================================
// Constants
// ============================================================================

/// Empty/transparent - evaluates to 0.0 everywhere.
pub const EMPTY: f32 = 0.0;

/// Solid/opaque - evaluates to 1.0 everywhere.
pub const SOLID: f32 = 1.0;

// ============================================================================
// Unit Shapes
// ============================================================================

/// Unit circle centered at origin with radius 1.
/// Bounding box: [-1, -1] to [1, 1]
///
/// Returns fg where x² + y² < 1, bg elsewhere.
pub fn circle<F: Manifold<Output = Field>, B: Manifold<Output = Field>>(
    fg: F,
    bg: B,
) -> impl Manifold<Output = Field> {
    (X * X + Y * Y).lt(1.0f32).select(fg, bg)
}

/// Unit square from [0,0] to [1,1].
///
/// Returns fg where 0 ≤ x ≤ 1 and 0 ≤ y ≤ 1, bg elsewhere.
///
/// Works with both Field and Jet2 for anti-aliased rendering.
/// Returns concrete type for use in type aliases.
pub fn square<F, B>(fg: F, bg: B) -> Select<UnitSquareCond, F, B> {
    let cond = Ge(X, 0.0f32) & Le(X, 1.0f32) & Ge(Y, 0.0f32) & Le(Y, 1.0f32);
    Select { cond, if_true: fg, if_false: bg }
}

/// Half-plane: x ≥ 0
///
/// Returns fg where x ≥ 0, bg elsewhere.
pub fn half_plane_x<F: Manifold<Output = Field>, B: Manifold<Output = Field>>(
    fg: F,
    bg: B,
) -> impl Manifold<Output = Field> {
    X.ge(0.0f32).select(fg, bg)
}

/// Half-plane: y ≥ 0
///
/// Returns fg where y ≥ 0, bg elsewhere.
pub fn half_plane_y<F: Manifold<Output = Field>, B: Manifold<Output = Field>>(
    fg: F,
    bg: B,
) -> impl Manifold<Output = Field> {
    Y.ge(0.0f32).select(fg, bg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_at_origin() {
        // Point at origin should be inside
        // Point at (2, 0) should be outside
    }

    #[test]
    fn composition_works() {
        // circle inside square
        let _scene = square(circle(SOLID, 0.5f32), EMPTY);
    }
}
