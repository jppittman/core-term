//! Loop-Blinn curve rendering using pure Manifold algebra.
//!
//! The key insight: barycentric coordinates are LINEAR in (X, Y),
//! so the entire Loop-Blinn implicit u² - v is just polynomial composition.
//! No per-lane extraction needed—it's manifolds all the way down.

use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// ============================================================================
// Smooth Step (AA helper) - Pure Manifold Version
// ============================================================================

/// Smooth step interpolation for anti-aliasing.
///
/// Returns 0 when x < edge0, 1 when x > edge1, and smoothly
/// interpolates between for values in between.
///
/// This is implemented as pure manifold composition:
/// t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
/// result = t² * (3 - 2*t)
pub fn smooth_step<E0, E1, M>(edge0: E0, edge1: E1, value: M) -> impl Manifold
where
    E0: Manifold + Copy,
    E1: Manifold + Copy,
    M: Manifold + Copy,
{
    // t = (value - edge0) / (edge1 - edge0)
    let t_unclamped = (value - edge0) / (edge1 - edge0);

    // clamp to [0, 1] using manifold max/min
    let t_clamped = t_unclamped.max(0.0f32).min(1.0f32);

    // Hermite interpolation: t² * (3 - 2*t)
    t_clamped * t_clamped * (3.0f32 - 2.0f32 * t_clamped)
}

/// Extension trait for smooth_step on manifolds.
pub trait SmoothStepExt: Manifold + Sized + Copy {
    /// Apply smooth step interpolation for anti-aliasing.
    ///
    /// Returns 0 when self < edge0, 1 when self > edge1.
    fn smooth_step<E0: Manifold + Copy, E1: Manifold + Copy>(
        self,
        edge0: E0,
        edge1: E1,
    ) -> impl Manifold {
        smooth_step(edge0, edge1, self)
    }
}

impl<M: Manifold + Sized + Copy> SmoothStepExt for M {}

// ============================================================================
// Loop-Blinn Quadratic Curve (Pure Manifold Algebra)
// ============================================================================

/// A quadratic Bézier curve using Loop-Blinn rendering.
///
/// The curve is defined by control points P0, P1, P2.
/// This struct holds the coefficients for the implicit equation,
/// which are computed from the control points at construction time.
///
/// The implicit equation is: u² - v
/// where (u, v) are texture coordinates that are LINEAR in screen (x, y).
#[derive(Clone, Copy, Debug)]
pub struct LoopBlinnQuad {
    // Coefficients for u = a*X + b*Y + c (linear in screen coords)
    u_a: f32,
    u_b: f32,
    u_c: f32,
    // Coefficients for v = d*X + e*Y + f (linear in screen coords)
    v_d: f32,
    v_e: f32,
    v_f: f32,
}

impl LoopBlinnQuad {
    /// Create a Loop-Blinn curve from control points.
    ///
    /// P0 and P2 are endpoints, P1 is the control point.
    /// Returns None if the triangle is degenerate.
    pub fn new(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> Option<Self> {
        let (x0, y0) = (p0[0], p0[1]);
        let (x1, y1) = (p1[0], p1[1]);
        let (x2, y2) = (p2[0], p2[1]);

        // Area of triangle P0, P1, P2 (twice the signed area)
        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);

        if area.abs() < 1e-6 {
            return None; // Degenerate triangle
        }

        let inv_area = 1.0 / area;

        // Barycentric coords as linear functions of X, Y:
        // α(X,Y) = ((x1-X)(y2-Y) - (x2-X)(y1-Y)) / area
        //        = (x1*y2 - x2*y1 + X*(y1-y2) + Y*(x2-x1)) / area
        //
        // α = αx*X + αy*Y + αc

        let alpha_x = (y1 - y2) * inv_area;
        let alpha_y = (x2 - x1) * inv_area;
        let alpha_c = (x1 * y2 - x2 * y1) * inv_area;

        let beta_x = (y2 - y0) * inv_area;
        let beta_y = (x0 - x2) * inv_area;
        let beta_c = (x2 * y0 - x0 * y2) * inv_area;

        // γ = 1 - α - β
        let gamma_x = -alpha_x - beta_x;
        let gamma_y = -alpha_y - beta_y;
        let gamma_c = 1.0 - alpha_c - beta_c;

        // Loop-Blinn texture coordinates:
        // P0 -> (u=0, v=0)
        // P1 -> (u=0.5, v=0)
        // P2 -> (u=1, v=1)
        //
        // u = 0*α + 0.5*β + 1*γ = 0.5*β + γ
        // v = 0*α + 0*β + 1*γ = γ

        let u_a = 0.5 * beta_x + gamma_x;
        let u_b = 0.5 * beta_y + gamma_y;
        let u_c = 0.5 * beta_c + gamma_c;

        let v_d = gamma_x;
        let v_e = gamma_y;
        let v_f = gamma_c;

        Some(Self {
            u_a,
            u_b,
            u_c,
            v_d,
            v_e,
            v_f,
        })
    }

    /// Returns the implicit manifold: u² - v
    ///
    /// Negative = inside curve, positive = outside, zero = on curve.
    pub fn implicit(&self) -> impl Manifold + Copy {
        // u = u_a*X + u_b*Y + u_c
        let u = X * self.u_a + Y * self.u_b + self.u_c;
        // v = v_d*X + v_e*Y + v_f
        let v = X * self.v_d + Y * self.v_e + self.v_f;
        // implicit = u² - v
        u * u - v
    }

    /// Returns a manifold for anti-aliased coverage.
    ///
    /// The result is 0 outside, 1 inside, with smooth transition
    /// over half a pixel at the boundary.
    pub fn coverage(&self) -> impl Manifold {
        // Implicit is negative inside, positive outside
        // We want: inside = 1, outside = 0
        // smooth_step(edge0, edge1, value) returns 0 when value < edge0, 1 when value > edge1
        // So smooth_step(0.5, -0.5, implicit) gives: 1 inside (implicit < -0.5), 0 outside (implicit > 0.5)
        smooth_step(0.5f32, -0.5f32, self.implicit())
    }
}

// ============================================================================
// Line Segment (also pure manifold)
// ============================================================================

/// A line segment for curve rendering.
///
/// Lines contribute to winding but have no curvature.
/// The implicit is the signed distance to the line (linear).
#[derive(Clone, Copy, Debug)]
pub struct LineSegment {
    // Line equation: a*X + b*Y + c = 0
    // (a, b) is the normal, c is the offset
    a: f32,
    b: f32,
    c: f32,
}

impl LineSegment {
    /// Create a line from two endpoints.
    pub fn new(p0: [f32; 2], p1: [f32; 2]) -> Self {
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];

        // Normal is perpendicular to direction
        let len = (dx * dx + dy * dy).sqrt();
        let (a, b) = if len > 1e-6 {
            (-dy / len, dx / len)
        } else {
            (0.0, 1.0) // Degenerate
        };

        // c = -(a*x0 + b*y0)
        let c = -(a * p0[0] + b * p0[1]);

        Self { a, b, c }
    }

    /// Returns the signed distance manifold.
    /// Positive on one side, negative on the other.
    pub fn signed_distance(&self) -> impl Manifold + Copy {
        X * self.a + Y * self.b + self.c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_valid_loop_blinn_quad() {
        let quad = LoopBlinnQuad::new([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]);
        assert!(quad.is_some());
    }

    #[test]
    fn rejects_degenerate_collinear_points() {
        // Collinear points should return None
        let quad = LoopBlinnQuad::new([0.0, 0.0], [0.5, 0.0], [1.0, 0.0]);
        assert!(quad.is_none());
    }
}
