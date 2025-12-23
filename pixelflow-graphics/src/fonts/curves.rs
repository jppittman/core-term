//! Curve primitives for glyph rendering.
//!
//! This module defines the fundamental curve types used for representing
//! glyph outlines: lines and quadratic Bézier curves.

/// A line segment from p0 to p1.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
}

impl Line {
    /// Create a new line segment.
    pub fn new(p0: [f32; 2], p1: [f32; 2]) -> Self {
        Self { p0, p1 }
    }
}

/// Minimum triangle area for valid quadratic curves.
const MIN_TRIANGLE_AREA: f32 = 1e-6;

/// A quadratic Bézier curve with pre-computed barycentric coefficients.
///
/// The curve is defined by three control points: p0 (start), p1 (control), p2 (end).
/// The barycentric coefficients are pre-computed for efficient inside/outside testing.
#[derive(Clone, Copy, Debug)]
pub struct Quadratic {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    // Loop-Blinn style UV coefficients for curve evaluation
    pub u_a: f32,
    pub u_b: f32,
    pub u_c: f32,
    pub v_d: f32,
    pub v_e: f32,
    pub v_f: f32,
}

impl Quadratic {
    /// Try to create a quadratic curve from three points.
    ///
    /// Returns `None` if the points are collinear (degenerate curve).
    pub fn try_new(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> Option<Self> {
        let (x0, y0) = (p0[0], p0[1]);
        let (x1, y1) = (p1[0], p1[1]);
        let (x2, y2) = (p2[0], p2[1]);

        // Signed area of triangle p0-p1-p2
        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if area.abs() < MIN_TRIANGLE_AREA {
            return None;
        }
        let inv_area = 1.0 / area;

        // Barycentric coordinate coefficients
        let alpha_x = (y1 - y2) * inv_area;
        let alpha_y = (x2 - x1) * inv_area;
        let alpha_c = (x1 * y2 - x2 * y1) * inv_area;

        let beta_x = (y2 - y0) * inv_area;
        let beta_y = (x0 - x2) * inv_area;
        let beta_c = (x2 * y0 - x0 * y2) * inv_area;

        let gamma_x = -alpha_x - beta_x;
        let gamma_y = -alpha_y - beta_y;
        let gamma_c = 1.0 - alpha_c - beta_c;

        // Loop-Blinn UV mapping
        let u_a = 0.5 * beta_x + gamma_x;
        let u_b = 0.5 * beta_y + gamma_y;
        let u_c = 0.5 * beta_c + gamma_c;

        let v_d = gamma_x;
        let v_e = gamma_y;
        let v_f = gamma_c;

        Some(Self {
            p0,
            p1,
            p2,
            u_a,
            u_b,
            u_c,
            v_d,
            v_e,
            v_f,
        })
    }
}

/// A curve segment, either a line or a quadratic Bézier.
#[derive(Clone, Copy, Debug)]
pub enum Segment {
    Line(Line),
    Quad(Quadratic),
}
