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
    Select {
        cond,
        if_true: fg,
        if_false: bg,
    }
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

// ============================================================================
// Static Manifold Definitions (const-constructible)
// ============================================================================
//
// Since manifold types are just AST nodes (zero-sized or small structs),
// they can be constructed as `const`. This enables reusable, static definitions.
//
// Note: We can't use operator overloading in const context (that requires
// nightly's `const_trait_impl`), but struct literal construction works fine.

use pixelflow_core::ops::{Add, Mul, Sqrt, Sub};
use pixelflow_core::Lt;

/// Unit circle distance squared: x² + y²
/// Type encodes the computation; no runtime cost until evaluated.
pub const UNIT_CIRCLE_DIST_SQ: Add<Mul<X, X>, Mul<Y, Y>> = Add(Mul(X, X), Mul(Y, Y));

/// Unit circle SDF: sqrt(x² + y²) - 1.0
/// Negative inside, zero on boundary, positive outside.
pub const UNIT_CIRCLE_SDF: Sub<Sqrt<Add<Mul<X, X>, Mul<Y, Y>>>, f32> =
    Sub(Sqrt(Add(Mul(X, X), Mul(Y, Y))), 1.0);

/// Unit circle condition: x² + y² < 1.0
/// Returns mask (all-1s inside, all-0s outside).
pub const UNIT_CIRCLE_COND: Lt<Add<Mul<X, X>, Mul<Y, Y>>, f32> = Lt(Add(Mul(X, X), Mul(Y, Y)), 1.0);

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::combinators::{At, Texture};

    /// Evaluate a scalar manifold at a point (via texture materialization).
    fn eval_scalar<M: Manifold<Output = Field>>(m: &M, x: f32, y: f32) -> f32 {
        // Bind coordinates with At, then materialize via 1x1 texture
        let bound = At {
            inner: m,
            x,
            y,
            z: 0.0f32,
            w: 0.0f32,
        };
        let tex = Texture::from_manifold(&bound, 1, 1);
        tex.data()[0]
    }

    #[test]
    fn circle_should_contain_origin() {
        let c = circle(SOLID, EMPTY);
        let res = eval_scalar(&c, 0.0, 0.0);
        assert_eq!(res, 1.0, "Origin should be inside circle");
    }

    #[test]
    fn circle_should_exclude_outside_points() {
        let c = circle(SOLID, EMPTY);
        let res = eval_scalar(&c, 1.1, 0.0);
        assert_eq!(res, 0.0, "Point (1.1, 0) should be outside circle");
    }

    #[test]
    fn circle_should_exclude_boundary() {
        // circle is strict < 1.0
        let c = circle(SOLID, EMPTY);
        let res = eval_scalar(&c, 1.0, 0.0);
        assert_eq!(res, 0.0, "Boundary point (1.0, 0) should be outside circle (strict inequality)");
    }

    #[test]
    fn square_should_include_boundaries() {
        let s = square(SOLID, EMPTY);
        // Test corners
        assert_eq!(eval_scalar(&s, 0.0, 0.0), 1.0, "Top-left (0,0) inside");
        assert_eq!(eval_scalar(&s, 1.0, 0.0), 1.0, "Top-right (1,0) inside");
        assert_eq!(eval_scalar(&s, 0.0, 1.0), 1.0, "Bottom-left (0,1) inside");
        assert_eq!(eval_scalar(&s, 1.0, 1.0), 1.0, "Bottom-right (1,1) inside");
    }

    #[test]
    fn square_should_exclude_outside() {
        let s = square(SOLID, EMPTY);
        assert_eq!(eval_scalar(&s, -0.001, 0.5), 0.0, "Left of 0.0 should be outside");
        assert_eq!(eval_scalar(&s, 1.001, 0.5), 0.0, "Right of 1.0 should be outside");
        assert_eq!(eval_scalar(&s, 0.5, -0.001), 0.0, "Above 0.0 should be outside");
        assert_eq!(eval_scalar(&s, 0.5, 1.001), 0.0, "Below 1.0 should be outside");
    }

    #[test]
    fn rectangle_should_respect_dimensions() {
        let width = 2.0;
        let height = 3.0;
        let r = rectangle(width, height, SOLID, EMPTY);

        assert_eq!(eval_scalar(&r, 0.0, 0.0), 1.0, "Origin inside");
        assert_eq!(eval_scalar(&r, width, height), 1.0, "Corner inside");
        assert_eq!(eval_scalar(&r, width + 0.1, 1.0), 0.0, "Outside width");
        assert_eq!(eval_scalar(&r, 1.0, height + 0.1), 0.0, "Outside height");
    }

    #[test]
    fn annulus_should_include_boundaries() {
        let inner = 0.5;
        let outer = 1.0;
        let a = annulus(inner, outer, SOLID, EMPTY);

        assert_eq!(eval_scalar(&a, 0.5, 0.0), 1.0, "Inner radius inclusive");
        assert_eq!(eval_scalar(&a, 1.0, 0.0), 1.0, "Outer radius inclusive");
        assert_eq!(eval_scalar(&a, 0.49, 0.0), 0.0, "Inside hole exclusive");
        assert_eq!(eval_scalar(&a, 1.01, 0.0), 0.0, "Outside ring exclusive");
    }

    #[test]
    fn composition_should_combine_shape_logics() {
        // circle inside square.
        // circle is radius 1, centered at origin.
        // square is [0,0] to [1,1].
        // Overlap is the quarter circle in the first quadrant.

        // Define: Square containing (Circle containing 1.0 else 0.5) else 0.0
        // If inside square:
        //    If inside circle: 1.0
        //    Else: 0.5
        // Else: 0.0

        let scene = square(circle(SOLID, Field::from(0.5)), EMPTY);

        // Point inside both (0.1, 0.1) -> 1.0
        assert_eq!(eval_scalar(&scene, 0.1, 0.1), 1.0, "Inside both");

        // Point inside square, outside circle (0.9, 0.9) -> 0.9^2 + 0.9^2 = 0.81 + 0.81 = 1.62 > 1 -> 0.5
        assert_eq!(eval_scalar(&scene, 0.9, 0.9), 0.5, "Inside square, outside circle");

        // Point outside square, inside circle (-0.1, 0.1) -> 0.0
        assert_eq!(eval_scalar(&scene, -0.1, 0.1), 0.0, "Outside square");
    }

    #[test]
    fn const_manifold_eval() {
        // First test the distance squared (no sqrt)
        let dist_sq_origin = eval_scalar(&UNIT_CIRCLE_DIST_SQ, 0.0, 0.0);
        assert!(
            dist_sq_origin.abs() < 0.001,
            "dist_sq at origin should be 0.0, got {}",
            dist_sq_origin
        );

        let dist_sq_edge = eval_scalar(&UNIT_CIRCLE_DIST_SQ, 1.0, 0.0);
        assert!(
            (dist_sq_edge - 1.0).abs() < 0.001,
            "dist_sq at (1,0) should be 1.0, got {}",
            dist_sq_edge
        );

        // Now test the full SDF (using relaxed tolerance for fast sqrt approximation)
        let at_edge = eval_scalar(&UNIT_CIRCLE_SDF, 1.0, 0.0);
        assert!(
            at_edge.abs() < 0.01,
            "SDF at edge should be ~0.0, got {}",
            at_edge
        );

        let outside = eval_scalar(&UNIT_CIRCLE_SDF, 2.0, 0.0);
        assert!(
            (outside - 1.0).abs() < 0.01,
            "SDF at (2,0) should be ~1.0, got {}",
            outside
        );

        // Test sqrt(0) - this was a bug (returned NaN before fix)
        let at_origin = eval_scalar(&UNIT_CIRCLE_SDF, 0.0, 0.0);
        assert!(
            (at_origin - (-1.0)).abs() < 0.01,
            "SDF at origin should be -1.0, got {}",
            at_origin
        );
    }

    #[test]
    fn const_condition_with_select() {
        // Use the const condition with select to get 1.0 inside, 0.0 outside
        let selected = UNIT_CIRCLE_COND.select(1.0f32, 0.0f32);

        let inside = eval_scalar(&selected, 0.0, 0.0);
        assert!(
            (inside - 1.0).abs() < 0.001,
            "Origin should be inside (1.0), got {}",
            inside
        );

        let outside = eval_scalar(&selected, 2.0, 0.0);
        assert!(
            outside.abs() < 0.001,
            "Point (2,0) should be outside (0.0), got {}",
            outside
        );
    }
}
