//! Basic shapes as manifold stencils.
//!
//! Each shape is a function that takes foreground and background manifolds
//! and returns a composed manifold using Select. This enables:
//! - Natural composition via nesting
//! - Automatic bounds checking (outer shapes clip inner)
//! - Short-circuit evaluation via Select's all/any checks
//!
//! Note: Requires core to support operator overloading for manifolds.

use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y};

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
pub fn square<F: Manifold<Output = Field>, B: Manifold<Output = Field> + Clone>(
    fg: F,
    bg: B,
) -> impl Manifold<Output = Field> {
    // Use bitwise AND to combine conditions
    let mask = X.ge(0.0f32) & X.le(1.0f32) & Y.ge(0.0f32) & Y.le(1.0f32);
    mask.select(fg, bg)
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
