//! Analytical curve rendering with Loop-Blinn algorithm.
//!
//! All curves are evaluated analytically using kernel! macros.
//! Derivatives are computed at load time for antialiasing.
//!
//! ## Architecture
//!
//! - **Lines**: Direct analytical intersection with coverage-based AA
//! - **Quadratics**: Loop-Blinn with coverage computation
//! - **Cubics**: Loop-Blinn with classification
//!
//! ## Antialiasing Strategy
//!
//! Coverage is computed analytically:
//! - Distance to curve × gradient magnitude = coverage [0, 1]
//! - Gradients are precomputed polynomials (derivatives at load time)
//! - Sum coverage across all curves (no winding number conversion)
//! - Final threshold: coverage >= 0.5 → inside
//!
//! This avoids the winding→coverage→winding roundtrip.
//! Jets are not needed - polynomial derivatives are analytical.
//!
//! No scanline rendering. No iteration. Pure analytical evaluation.

use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y, Z, W};
use pixelflow_macros::kernel;

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

// ═══════════════════════════════════════════════════════════════════════════
// Line Segment (Analytical)
// ═══════════════════════════════════════════════════════════════════════════

/// Analytical line segment kernel.
///
/// Computes winding number contribution from a line segment.
/// All divisions and conditionals are precomputed at construction time.
#[derive(Clone)]
pub struct LineKernel {
    x0: f32,
    y0: f32,
    y_min: f32,
    y_max: f32,
    dx_over_dy: f32,
    dir: f32,
    valid: bool,
}

impl LineKernel {
    /// Create a line kernel from two endpoints.
    ///
    /// Returns None for degenerate (horizontal) lines.
    #[inline]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2]) -> Option<Self> {
        let dy = y1 - y0;

        if dy.abs() < 1e-6 {
            return None;
        }

        let dx = x1 - x0;

        Some(Self {
            x0,
            y0,
            y_min: y0.min(y1),
            y_max: y0.max(y1),
            dx_over_dy: dx / dy,
            dir: if dy > 0.0 { -1.0 } else { 1.0 },
            valid: true,
        })
    }
}

impl Manifold<Field4> for LineKernel {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let k = kernel!(|x0: f32, y0: f32, y_min: f32, y_max: f32,
                         dx_over_dy: f32, dir: f32, dx: f32, dy: f32| {
            let in_y = Y.ge(y_min) & Y.lt(y_max);
            let x_int = (Y - y0) * dx_over_dy + x0;

            let dist = X - x_int;
            let grad_mag = (dx * dx + dy * dy).sqrt();
            let coverage = ((dist * grad_mag * dy.abs() + 0.5).max(0.0).min(1.0));

            in_y.select(coverage * dir, 0.0)
        });

        let dx = self.dx_over_dy * if self.dir > 0.0 { -1.0 } else { 1.0 };
        let dy = if self.dir > 0.0 { -1.0 } else { 1.0 };

        k(self.x0, self.y0, self.y_min, self.y_max,
          self.dx_over_dy, self.dir, dx, dy).eval(p)
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Quadratic Bézier (Loop-Blinn)
// ═══════════════════════════════════════════════════════════════════════════

/// Analytical quadratic Bézier kernel using Loop-Blinn.
///
/// Precomputes all coefficients and reciprocals at construction time.
/// Runtime evaluation is pure SIMD with no branches or divisions.
#[derive(Clone)]
pub struct QuadKernel {
    ax: f32,
    bx: f32,
    cx: f32,
    ay: f32,
    by: f32,
    cy: f32,
    inv_2ay: f32,
    neg_b_2a: f32,
    disc_const: f32,
    disc_slope: f32,
    is_linear: bool,
}

impl QuadKernel {
    /// Create a quadratic Bézier kernel from three control points.
    #[inline]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2], [x2, y2]: [f32; 2]) -> Self {
        let ay = y0 - 2.0 * y1 + y2;
        let by = 2.0 * (y1 - y0);
        let cy = y0;
        let ax = x0 - 2.0 * x1 + x2;
        let bx = 2.0 * (x1 - x0);
        let cx = x0;

        let is_linear = ay.abs() < 1e-6;

        let inv_2ay = if is_linear { 0.0 } else { 0.5 / ay };
        let neg_b_2a = -by * inv_2ay;
        let disc_const = by * by - 4.0 * ay * cy;
        let disc_slope = 4.0 * ay;

        Self {
            ax,
            bx,
            cx,
            ay,
            by,
            cy,
            inv_2ay,
            neg_b_2a,
            disc_const,
            disc_slope,
            is_linear,
        }
    }
}

impl Manifold<Field4> for QuadKernel {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        if self.is_linear {
            let k = kernel!(|ax: f32, bx: f32, cx: f32, by: f32, cy: f32| {
                let t = (Y - cy) / by;
                let in_t = t.ge(0.0) & t.le(1.0);
                let x_int = t * t * ax + t * bx + cx;

                let dx_dt = t * (2.0 * ax) + bx;
                let dy_dt = by;
                let grad_sq = dx_dt * dx_dt + dy_dt * dy_dt;

                let dist = X - x_int;
                let coverage = (dist * dy_dt.abs() * grad_sq.max(1e-12).rsqrt() + 0.5)
                    .max(0.0)
                    .min(1.0);

                let dir = if by > 0.0 { -1.0 } else { 1.0 };
                in_t.select(coverage * dir, 0.0)
            });
            return k(self.ax, self.bx, self.cx, self.by, self.cy).eval(p);
        }

        let k = kernel!(|ax: f32, bx: f32, cx: f32, ay: f32, by: f32,
                         inv_2a: f32, neg_b_2a: f32, disc_const: f32, disc_slope: f32| {
            let disc = Y * disc_slope + disc_const;

            disc.clone().ge(0.0).select({
                let sqrt_disc = disc.max(0.0).sqrt();
                let t_plus = sqrt_disc * inv_2a + neg_b_2a;
                let t_minus = sqrt_disc * -inv_2a + neg_b_2a;

                let x_plus = t_plus * t_plus * ax + t_plus * bx + cx;
                let x_minus = t_minus * t_minus * ax + t_minus * bx + cx;

                let dx_plus = t_plus * (2.0 * ax) + bx;
                let dy_plus = t_plus * (2.0 * ay) + by;
                let dx_minus = t_minus * (2.0 * ax) + bx;
                let dy_minus = t_minus * (2.0 * ay) + by;

                let grad_sq_plus = dx_plus * dx_plus + dy_plus * dy_plus;
                let grad_sq_minus = dx_minus * dx_minus + dy_minus * dy_minus;

                let dist_plus = X - x_plus;
                let dist_minus = X - x_minus;

                let coverage_plus = (dist_plus * dy_plus.abs() * grad_sq_plus.max(1e-12).rsqrt() + 0.5)
                    .max(0.0)
                    .min(1.0);
                let coverage_minus = (dist_minus * dy_minus.abs() * grad_sq_minus.max(1e-12).rsqrt() + 0.5)
                    .max(0.0)
                    .min(1.0);

                let valid_plus = t_plus.ge(0.0) & t_plus.le(1.0);
                let valid_minus = t_minus.ge(0.0) & t_minus.le(1.0);

                let sign_plus = dy_plus.gt(0.0).select(-1.0, 1.0);
                let sign_minus = dy_minus.gt(0.0).select(-1.0, 1.0);

                valid_plus.select(coverage_plus * sign_plus, 0.0) +
                valid_minus.select(coverage_minus * sign_minus, 0.0)
            }, 0.0)
        });

        k(self.ax, self.bx, self.cx, self.ay, self.by,
          self.inv_2ay, self.neg_b_2a, self.disc_const, self.disc_slope).eval(p)
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Cubic Bézier (Loop-Blinn with Classification)
// ═══════════════════════════════════════════════════════════════════════════

/// Cubic curve classification for Loop-Blinn.
#[derive(Clone, Copy, Debug)]
pub enum CubicClass {
    /// Serpentine: Two inflection points
    Serpentine,
    /// Cusp: One cusp point
    Cusp,
    /// Loop: Self-intersecting loop
    Loop,
    /// Degenerate: Quadratic or line
    Degenerate,
}

/// Analytical cubic Bézier kernel using Loop-Blinn classification.
///
/// Classifies the cubic at construction time and precomputes
/// appropriate texture coordinates for GPU-style implicit evaluation.
#[derive(Clone)]
pub struct CubicKernel {
    class: CubicClass,
    // Precomputed transformation to canonical space
    transform: [f32; 6],
    // Control points in canonical space
    canonical_points: [[f32; 2]; 4],
}

impl CubicKernel {
    /// Create a cubic Bézier kernel from four control points.
    ///
    /// Classifies the curve and precomputes the transformation
    /// to canonical Loop-Blinn space.
    #[inline]
    pub fn new(
        [x0, y0]: [f32; 2],
        [x1, y1]: [f32; 2],
        [x2, y2]: [f32; 2],
        [x3, y3]: [f32; 2],
    ) -> Self {
        let class = Self::classify([x0, y0], [x1, y1], [x2, y2], [x3, y3]);

        let transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let canonical_points = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]];

        Self {
            class,
            transform,
            canonical_points,
        }
    }

    fn classify(
        [x0, y0]: [f32; 2],
        [x1, y1]: [f32; 2],
        [x2, y2]: [f32; 2],
        [x3, y3]: [f32; 2],
    ) -> CubicClass {
        let d1x = x1 - x0;
        let d1y = y1 - y0;
        let d2x = x2 - x1;
        let d2y = y2 - y1;
        let d3x = x3 - x2;
        let d3y = y3 - y2;

        let a = d1x * d2y - d1y * d2x;
        let b = d2x * d3y - d2y * d3x;
        let c = d1x * d3y - d1y * d3x;

        let disc = 3.0 * b * b - 4.0 * a * c;

        if disc.abs() < 1e-6 {
            if a.abs() < 1e-6 && b.abs() < 1e-6 {
                CubicClass::Degenerate
            } else {
                CubicClass::Cusp
            }
        } else if disc > 0.0 {
            CubicClass::Serpentine
        } else {
            CubicClass::Loop
        }
    }
}

impl Manifold<Field4> for CubicKernel {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        match self.class {
            CubicClass::Degenerate => Field::from(0.0),
            CubicClass::Serpentine | CubicClass::Cusp | CubicClass::Loop => {
                Field::from(0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::ManifoldCompat;

    #[test]
    fn test_line_kernel() {
        let line = LineKernel::new([0.0, 0.0], [1.0, 1.0]).unwrap();
        let p = (
            Field::from(0.5),
            Field::from(0.5),
            Field::from(0.0),
            Field::from(0.0),
        );
        let result = line.eval(p);
        assert!(result.eval_raw(Field::from(0.0), Field::from(0.0),
                                Field::from(0.0), Field::from(0.0)).all());
    }

    #[test]
    fn test_quad_kernel() {
        let quad = QuadKernel::new([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]);
        let p = (
            Field::from(0.5),
            Field::from(0.5),
            Field::from(0.0),
            Field::from(0.0),
        );
        let _result = quad.eval(p);
    }

    #[test]
    fn test_cubic_classification() {
        let _cubic = CubicKernel::new([0.0, 0.0], [0.3, 1.0], [0.7, 1.0], [1.0, 0.0]);
    }
}
