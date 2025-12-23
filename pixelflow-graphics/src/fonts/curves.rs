//! Curve primitives for glyph rendering.
//!
//! `Curve<N>` is the atomic building block - N control points defining
//! a Bezier curve of degree N-1.

use pixelflow_core::{Field, Manifold, Numeric};

/// N control points. The atom.
#[derive(Clone, Copy, Debug)]
pub struct Curve<const N: usize>(pub [[f32; 2]; N]);

pub type Line = Curve<2>;
pub type Quad = Curve<3>;
pub type Cubic = Curve<4>;

/// Mixed-degree sum.
#[derive(Clone, Copy, Debug)]
pub enum Segment {
    Line(Line),
    Quad(Quad),
    Cubic(Cubic),
}

// ============================================================================
// Manifold Implementations
// ============================================================================

impl Manifold for Line {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        let [[x0, y0], [x1, y1]] = self.0;

        let p0_y = Field::from(y0);
        let p1_y = Field::from(y1);
        let dy = p1_y - p0_y;

        // Skip near-horizontal lines
        let dy_abs = Numeric::abs(dy);
        let valid = Numeric::gt(dy_abs, Field::from(1e-6));

        let p0_x = Field::from(x0);
        let p1_x = Field::from(x1);
        let dx = p1_x - p0_x;

        // Y-range check
        let y_min = Numeric::min(p0_y, p1_y);
        let y_max = Numeric::max(p0_y, p1_y);
        let geq_min = Numeric::ge(y, y_min);
        let leq_max = Numeric::le(y, y_max);
        let in_y = geq_min * leq_max;

        // X-intersection
        let x_int = p0_x + (y - p0_y) * (dx / dy);
        let is_left = Numeric::lt(x, x_int);

        // Direction: +1 up, -1 down
        let going_up = Numeric::gt(dy, Field::from(0.0));
        let dir = Numeric::select(going_up, Field::from(1.0), Field::from(-1.0));

        // Combine
        let contrib = Numeric::select(is_left, dir, Field::from(0.0));
        let result = Numeric::select(in_y, contrib, Field::from(0.0));
        Numeric::select(valid, result, Field::from(0.0))
    }
}

impl Manifold for Quad {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        // For winding number, we only care about the endpoints
        // The control point affects shape but not winding contribution
        let [p0, _p1, p2] = self.0;
        let line: Line = Curve([p0, p2]);
        line.eval_raw(x, y, _z, _w)
    }
}

impl Manifold for Cubic {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        // Same as quad - winding only cares about endpoints
        let [p0, _p1, _p2, p3] = self.0;
        let line: Line = Curve([p0, p3]);
        line.eval_raw(x, y, _z, _w)
    }
}

impl Manifold for Segment {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        match self {
            Self::Line(c) => c.eval_raw(x, y, z, w),
            Self::Quad(c) => c.eval_raw(x, y, z, w),
            Self::Cubic(c) => c.eval_raw(x, y, z, w),
        }
    }
}
