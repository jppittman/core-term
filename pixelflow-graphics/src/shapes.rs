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
#[derive(Clone, Debug)]
pub struct Square<F, B> {
    pub fg: F,
    pub bg: B,
}

impl<F, B> Manifold for Square<F, B>
where
    F: Manifold<Output = Field> + Clone,
    B: Manifold<Output = Field> + Clone,
{
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // DEBUG: Check coordinates
        // use std::io::Write;
        // let _ = std::io::stdout().flush();
        // println!("Square check");
        let mask = X.ge(0.0) & X.le(1.0) & Y.ge(0.0) & Y.le(1.0);
        mask.select(self.fg.clone(), self.bg.clone())
            .eval_raw(x, y, z, w)
    }
}

/// Helper to create a Square manifold.
pub fn square<F, B>(fg: F, bg: B) -> Square<F, B>
where
    F: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
{
    Square { fg, bg }
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
