//! Basic shapes as manifold stencils.
//!
//! Each shape is a function that takes foreground and background manifolds
//! and returns a composed manifold using Select. This enables:
//! - Natural composition via nesting
//! - Automatic bounds checking (outer shapes clip inner)
//! - Short-circuit evaluation via Select's all/any checks
//!
//! All shapes follow the idiomatic PixelFlow pattern: compose manifolds,
//! don't compute fields directly. Shapes use coordinate variables (X, Y)
//! and comparison operators to build conditional evaluation trees.

use pixelflow_core::{And, Field, Ge, Le, Manifold, ManifoldExt, Select, X, Y};

// ============================================================================
// Type Aliases
// ============================================================================

/// The unit square condition: (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
pub type UnitSquareCond = And<And<And<Ge<X, f32>, Le<X, f32>>, Ge<Y, f32>>, Le<Y, f32>>;

/// A manifold bounded to the unit square.
pub type Bounded<M> = Select<UnitSquareCond, M, f32>;

/// Half-plane x ≥ 0 selecting between two manifolds.
pub type HalfPlaneX<F, B> = Select<Ge<X, f32>, F, B>;

/// Half-plane y ≥ 0 selecting between two manifolds.
pub type HalfPlaneY<F, B> = Select<Ge<Y, f32>, F, B>;

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
pub fn half_plane_x<F, B>(fg: F, bg: B) -> HalfPlaneX<F, B>
where
    F: Manifold,
    B: Manifold,
{
    Select {
        cond: Ge(X, 0.0f32),
        if_true: fg,
        if_false: bg,
    }
}

/// Half-plane: y ≥ 0
///
/// Returns fg where y ≥ 0, bg elsewhere.
pub fn half_plane_y<F, B>(fg: F, bg: B) -> HalfPlaneY<F, B>
where
    F: Manifold,
    B: Manifold,
{
    Select {
        cond: Ge(Y, 0.0f32),
        if_true: fg,
        if_false: bg,
    }
}

// ============================================================================
// Extended Shapes
// ============================================================================

/// Rectangle from [0, 0] to [width, height].
///
/// Returns fg where 0 ≤ x ≤ width and 0 ≤ y ≤ height, bg elsewhere.
pub fn rectangle<F: Manifold<Output = Field>, B: Manifold<Output = Field>>(
    width: f32,
    height: f32,
    fg: F,
    bg: B,
) -> impl Manifold<Output = Field> {
    let w_check = Ge(X, 0.0f32) & Le(X, width);
    let h_check = Ge(Y, 0.0f32) & Le(Y, height);
    (w_check & h_check).select(fg, bg)
}

/// Ellipse centered at origin with semi-axes rx, ry.
///
/// Returns fg where (x/rx)² + (y/ry)² < 1, bg elsewhere.
pub fn ellipse<F: Manifold<Output = Field>, B: Manifold<Output = Field>>(
    rx: f32,
    ry: f32,
    fg: F,
    bg: B,
) -> impl Manifold<Output = Field> {
    let rx_sq = rx * rx;
    let ry_sq = ry * ry;
    let normalized = (X * X) / Field::from(rx_sq) + (Y * Y) / Field::from(ry_sq);
    normalized.lt(1.0f32).select(fg, bg)
}

/// Annulus (ring) centered at origin with inner and outer radius.
///
/// Returns fg where r_inner ≤ sqrt(x² + y²) ≤ r_outer, bg elsewhere.
pub fn annulus<F: Manifold<Output = Field>, B: Manifold<Output = Field>>(
    r_inner: f32,
    r_outer: f32,
    fg: F,
    bg: B,
) -> impl Manifold<Output = Field> {
    let r_sq = X * X + Y * Y;
    let r_inner_sq = r_inner * r_inner;
    let r_outer_sq = r_outer * r_outer;
    let inside_outer = r_sq.le(r_outer_sq);
    let outside_inner = r_sq.ge(r_inner_sq);
    (inside_outer & outside_inner).select(fg, bg)
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
