//! Analytical Loop-Blinn curve kernels.
//!
//! Minimal version for initial bring-up. AA can be added later.

use pixelflow_core::{Field, Manifold, W, X, Y, Z};
use pixelflow_macros::kernel;

type Field4 = (Field, Field, Field, Field);

/// Simple line segment (no AA for now - keep kernel simple).
#[derive(Clone)]
pub struct AnalyticalLine {
    y_min: f32,
    y_max: f32,
    slope_x_over_y: f32,
    intercept_x: f32,
    dir: f32,
}

impl AnalyticalLine {
    #[inline]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2]) -> Option<Self> {
        let dy = y1 - y0;
        if dy.abs() < 1e-6 {
            return None;
        }

        let dx = x1 - x0;
        let dir = if dy > 0.0 { -1.0 } else { 1.0 };

        Some(Self {
            y_min: y0.min(y1),
            y_max: y0.max(y1),
            slope_x_over_y: dx / dy,
            intercept_x: x0 - (dx / dy) * y0,
            dir,
        })
    }
}

impl Manifold<Field4> for AnalyticalLine {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        // Line winding: check if Y is in range, compute X crossing, apply direction
        let k = kernel!(|y_min: f32, y_max: f32, slope: f32, intercept: f32, dir: f32| {
            let in_y_range = (Y >= y_min) & (Y < y_max);
            let x_at_y = Y * slope + intercept;
            let crosses_right = X < x_at_y;
            in_y_range.select(crosses_right.select(dir, 0.0), 0.0)
        });

        k(self.y_min, self.y_max, self.slope_x_over_y, self.intercept_x, self.dir).eval(p)
    }
}

/// Simple quadratic Bézier (no AA for now - keep kernel simple).
#[derive(Clone)]
pub struct AnalyticalQuad {
    // Quadratic coefficients: x(t) = ax*t² + bx*t + cx, y(t) = ay*t² + by*t + cy
    ax: f32, bx: f32, cx: f32,
    ay: f32, by: f32, cy: f32,
    // Precomputed for discriminant: disc = disc_slope * Y + disc_const
    disc_slope: f32,
    disc_const: f32,
    inv_2ay: f32,
    neg_by_over_2ay: f32,
    is_linear: bool,
}

impl AnalyticalQuad {
    #[inline]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2], [x2, y2]: [f32; 2]) -> Self {
        let ay = y0 - 2.0 * y1 + y2;
        let by = 2.0 * (y1 - y0);
        let ax = x0 - 2.0 * x1 + x2;
        let bx = 2.0 * (x1 - x0);

        let is_linear = ay.abs() < 1e-6;
        let inv_2ay = if is_linear { 0.0 } else { 0.5 / ay };

        Self {
            ax, bx, cx: x0,
            ay, by, cy: y0,
            disc_slope: 4.0 * ay,
            disc_const: by * by - 4.0 * ay * y0,
            inv_2ay,
            neg_by_over_2ay: -by * inv_2ay,
            is_linear,
        }
    }
}

impl Manifold<Field4> for AnalyticalQuad {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        use pixelflow_core::{X, Y, Z, ManifoldExt};

        // Linear degenerate case (5 parameters - now works with WithContext!)
        if self.is_linear {
            let dir = if self.by > 0.0 { -1.0 } else { 1.0 };
            let k = kernel!(|bx: f32, cx: f32, by: f32, cy: f32, dir: f32| {
                // For linear: t = (Y - cy) / by, x = bx * t + cx
                // Recompute t inline in each expression to avoid moves

                // Check if t in [0, 1]  (recompute t for each comparison)
                let in_range = (((Y - cy) / by) >= 0.0) & (((Y - cy) / by) <= 1.0);

                // Compute x at t: x = bx * ((Y - cy) / by) + cx
                let x_at_t = bx * ((Y - cy) / by) + cx;

                // Check if crossing is to the right
                let crosses = X < x_at_t;

                // Only contribute if in range and crosses
                in_range.select(crosses.select(dir, 0.0), 0.0)
            });
            return k(self.bx, self.cx, self.by, self.cy, dir).eval(p);
        }

        // Quadratic case: Use layered coordinate transforms like the original.
        // Layer 3 (innermost): X=screen_x, Y=t_plus, Z=t_minus
        // In this space, we can compute dy/dt = Y * (2*ay) + by where Y is the parameter t!
        let winding = {
            let x_plus = Y * Y * self.ax + Y * self.bx + self.cx;
            let x_minus = Z * Z * self.ax + Z * self.bx + self.cx;
            let dy_plus = Y * (2.0 * self.ay) + self.by;
            let dy_minus = Z * (2.0 * self.ay) + self.by;

            let valid_plus = Y.ge(0.0) & Y.le(1.0) & X.lt(x_plus);
            let valid_minus = Z.ge(0.0) & Z.le(1.0) & X.lt(x_minus);

            let sign_plus = dy_plus.gt(0.0).select(-1.0, 1.0);
            let sign_minus = dy_minus.gt(0.0).select(-1.0, 1.0);

            valid_plus.select(sign_plus, 0.0) + valid_minus.select(sign_minus, 0.0)
        };

        // Layer 2: X=screen_x, Y=screen_y, Z=sqrt_disc
        // Map Y → t_plus, Z → t_minus
        let with_roots = winding.at(
            X,
            Z * self.inv_2ay + self.neg_by_over_2ay,  // t_plus
            Z * -self.inv_2ay + self.neg_by_over_2ay, // t_minus
            W,
        );

        // Layer 1 (outermost): screen coords
        // Compute discriminant and check if roots exist
        let disc = Y * self.disc_slope + self.disc_const;
        disc.clone()
            .ge(0.0)
            .select(with_roots.at(X, Y, disc.max(0.0).sqrt(), W), 0.0)
            .eval(p)
    }
}
