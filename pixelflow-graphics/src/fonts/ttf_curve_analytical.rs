//! Loop-Blinn analytical curve rendering (2005 paper technique).
//!
//! Uses barycentric coordinate transformation where curves are represented by
//! implicit functions f(u,v,w) = 0. The barycentric coords vary linearly with
//! screen position, enabling analytical gradient computation for coverage.
//!
//! For winding numbers: we compute signed coverage at curve boundaries and
//! accumulate fractional winding contributions for antialiased fill.

use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y};
use pixelflow_macros::kernel;

type Field4 = (Field, Field, Field, Field);

/// Line segment using Loop-Blinn implicit function.
///
/// Implicit: f = a*x + b*y + c (perpendicular distance, normalized)
/// Gradient: ∇f = [a, b] with magnitude = 1 (already normalized)
/// Coverage: clamp(0.5 - f/|∇f|, 0, 1) = clamp(0.5 - f, 0, 1)
#[derive(Clone)]
pub struct AnalyticalLine {
    // Line equation: a*x + b*y + c = 0 (normalized: √(a²+b²) = 1)
    a: f32,
    b: f32,
    c: f32,
    // Winding direction based on line orientation
    dir: f32,
    // Bounding box for early rejection
    y_min: f32,
    y_max: f32,
}

impl AnalyticalLine {
    #[inline]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2]) -> Option<Self> {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();

        if len < 1e-6 {
            return None;
        }

        // Perpendicular direction (rotate 90°), normalized
        let a = dy / len;
        let b = -dx / len;
        let c = -(a * x0 + b * y0);

        // Winding direction
        let dir = if dy > 0.0 { -1.0 } else { 1.0 };

        let y_min = y0.min(y1);
        let y_max = y0.max(y1);

        Some(Self {
            a,
            b,
            c,
            dir,
            y_min,
            y_max,
        })
    }
}

impl Manifold<Field4> for AnalyticalLine {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let k = kernel!(|a: f32, b: f32, c: f32, dir: f32, y_min: f32, y_max: f32| {
            // Early rejection
            let in_y_range = (Y >= y_min) & (Y < y_max);

            // Implicit function: f = a*x + b*y + c
            let f = X * a + Y * b + c;

            // Gradient magnitude = 1 (normalized in constructor)
            // Coverage = clamp(0.5 - f, 0, 1) for 1-pixel filter width
            let coverage = (f * -1.0 + 0.5).max(0.0).min(1.0);

            // Fractional winding contribution
            let winding = dir * coverage;

            in_y_range.select(winding, 0.0)
        });

        k(self.a, self.b, self.c, self.dir, self.y_min, self.y_max).eval(p)
    }
}

/// Quadratic Bézier using Loop-Blinn barycentric implicit function.
///
/// Barycentric coords: u, v, w vary linearly with screen position
/// Implicit: f = u² - vw (for serpentine/cusp curves)
/// Gradient: ∇f = [2u·u_x - v·w_x - w·v_x, 2u·u_y - v·w_y - w·v_y]
///
/// For winding: check ray crossing at curve boundary (f≈0) with fractional coverage
#[derive(Clone)]
pub struct AnalyticalQuad {
    // Barycentric coordinate coefficients: u = u_x*X + u_y*Y + u_c
    u_x: f32, u_y: f32, u_c: f32,
    v_x: f32, v_y: f32, v_c: f32,
    w_x: f32, w_y: f32, w_c: f32,
    // Bounding box for early rejection (and correct finite segment rendering)
    y_min: f32, y_max: f32,
    // Curve classification
    is_linear: bool,
    // For winding: need to know curve orientation
    orientation: f32,  // +1 or -1
}

impl AnalyticalQuad {
    #[inline]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2], [x2, y2]: [f32; 2]) -> Self {
        // Check if curve is linear (degenerate)
        let dx01 = x1 - x0;
        let dy01 = y1 - y0;
        let dx12 = x2 - x1;
        let dy12 = y2 - y1;
        let cross = dx01 * dy12 - dy01 * dx12;

        let y_min = y0.min(y1).min(y2);
        let y_max = y0.max(y1).max(y2);

        if cross.abs() < 1e-6 {
            // Degenerate linear case
            let dx = x2 - x0;
            let dy = y2 - y0;
            let len = (dx * dx + dy * dy).sqrt();

            if len < 1e-6 {
                // Completely degenerate
                return Self {
                    u_x: 0.0, u_y: 0.0, u_c: 1.0,
                    v_x: 0.0, v_y: 0.0, v_c: 0.0,
                    w_x: 0.0, w_y: 0.0, w_c: 0.0,
                    y_min, y_max,
                    is_linear: true,
                    orientation: 1.0,
                };
            }

            // Linear implicit: f = perpendicular distance
            let a = dy / len;
            let b = -dx / len;
            let c = -(a * x0 + b * y0);

            Self {
                u_x: a, u_y: b, u_c: c,
                v_x: 0.0, v_y: 0.0, v_c: 0.0,
                w_x: 0.0, w_y: 0.0, w_c: 0.0,
                y_min, y_max,
                is_linear: true,
                orientation: if dy > 0.0 { -1.0 } else { 1.0 },
            }
        } else {
            // True quadratic - set up Loop-Blinn barycentric coordinates
            // Standard serpentine/cusp setup from the paper:
            // At P0: (u,v,w) = (0, 1, 0)
            // At P1: (u,v,w) = (0.5, 0, 0)
            // At P2: (u,v,w) = (1, 0, 1)
            //
            // This gives implicit function f = u² - vw that equals 0 on the curve

            // Solve for barycentric coordinate coefficients
            // We have 3 constraints from the 3 control points
            // Build system: [x,y,1] * M = [u,v,w]

            let det = x0*(y1 - y2) - y0*(x1 - x2) + (x1*y2 - x2*y1);

            if det.abs() < 1e-10 {
                // Numerically unstable - fallback to linear
                return Self {
                    u_x: 0.0, u_y: 0.0, u_c: 1.0,
                    v_x: 0.0, v_y: 0.0, v_c: 0.0,
                    w_x: 0.0, w_y: 0.0, w_c: 0.0,
                    y_min, y_max,
                    is_linear: true,
                    orientation: 1.0,
                };
            }

            let inv_det = 1.0 / det;

            // Solve for u coefficients: u(P0)=0, u(P1)=0.5, u(P2)=1
            // Using Cramer's rule on the system
            let u_x = inv_det * ((0.5 - 0.0)*(y2 - y0) + (1.0 - 0.5)*(y0 - y1));
            let u_y = inv_det * ((0.0 - 0.5)*(x2 - x0) + (0.5 - 1.0)*(x0 - x1));
            let u_c = 0.0 - u_x * x0 - u_y * y0;

            // Solve for v coefficients: v(P0)=1, v(P1)=0, v(P2)=0
            let v_x = inv_det * ((0.0 - 1.0)*(y2 - y0) + (0.0 - 0.0)*(y0 - y1));
            let v_y = inv_det * ((1.0 - 0.0)*(x2 - x0) + (0.0 - 0.0)*(x0 - x1));
            let v_c = 1.0 - v_x * x0 - v_y * y0;

            // Solve for w coefficients: w(P0)=0, w(P1)=0, w(P2)=1
            let w_x = inv_det * ((0.0 - 0.0)*(y2 - y0) + (1.0 - 0.0)*(y0 - y1));
            let w_y = inv_det * ((0.0 - 0.0)*(x2 - x0) + (0.0 - 1.0)*(x0 - x1));
            let w_c = 0.0 - w_x * x0 - w_y * y0;

            // Determine orientation for winding
            // Cross product of tangent vectors at endpoints
            let orientation = if cross > 0.0 { -1.0 } else { 1.0 };

            Self {
                u_x, u_y, u_c,
                v_x, v_y, v_c,
                w_x, w_y, w_c,
                y_min, y_max,
                is_linear: false,
                orientation,
            }
        }
    }
}

impl Manifold<Field4> for AnalyticalQuad {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        if self.is_linear {
            // Linear case: f = u (signed distance)
            let k = kernel!(|u_x: f32, u_y: f32, u_c: f32, orientation: f32, y_min: f32, y_max: f32| {
                // Early rejection
                let in_y_range = (Y >= y_min) & (Y < y_max);
                let f = X * u_x + Y * u_y + u_c;
                // Coverage with 1-pixel filter
                let coverage = (f * -1.0 + 0.5).max(0.0).min(1.0);
                let winding = orientation * coverage;
                in_y_range.select(winding, 0.0)
            });
            k(self.u_x, self.u_y, self.u_c, self.orientation, self.y_min, self.y_max).eval(p)
        } else {
            // Quadratic Loop-Blinn case: Compose Manifolds directly, no big kernel

            // Build barycentric coordinates as Manifold expressions
            let u = X * self.u_x + Y * self.u_y + self.u_c;
            let v = X * self.v_x + Y * self.v_y + self.v_c;
            let w = X * self.w_x + Y * self.w_y + self.w_c;

            // Loop-Blinn implicit function: f = u² - vw
            let f = u.clone() * u.clone() - v.clone() * w.clone();

            // Analytical gradient components
            let grad_x = u.clone() * (2.0 * self.u_x)
                       - v.clone() * self.w_x
                       - w.clone() * self.v_x;
            let grad_y = u.clone() * (2.0 * self.u_y)
                       - v.clone() * self.w_y
                       - w.clone() * self.v_y;

            // Gradient magnitude
            let grad_mag = (grad_x.clone() * grad_x + grad_y.clone() * grad_y).sqrt();

            // Coverage from signed distance
            let scaled_f = f / grad_mag.max(1e-6);
            let coverage = (scaled_f * -1.0 + 0.5).max(0.0).min(1.0);

            // Apply orientation for winding and Y-bounds check
            // Even for curves, we need to bound the vertical extent to prevent
            // infinite strips from filling the bounding box.
            let in_y_range = Y.ge(self.y_min) & Y.lt(self.y_max);
            in_y_range.select(coverage * self.orientation, 0.0).eval(p)
        }
    }
}
