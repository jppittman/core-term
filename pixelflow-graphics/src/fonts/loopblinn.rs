//! Loop-Blinn curve rendering using pure Manifold algebra.

use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y};

// ============================================================================
// Smooth Step (AA helper) - Pure Manifold Version
// ============================================================================

pub fn smooth_step<E0, E1, M>(edge0: E0, edge1: E1, value: M) -> impl Manifold<Output = Field>
where
    E0: Manifold<Output = Field> + Copy,
    E1: Manifold<Output = Field> + Copy,
    M: Manifold<Output = Field> + Copy,
{
    // t = (value - edge0) / (edge1 - edge0)
    let range = edge1.sub(edge0);
    // Explicit fluent method chaining avoids operator overload ambiguity
    let t_unclamped = value.sub(edge0).div(range);

    // clamp to [0, 1] using ManifoldExt methods
    let t_clamped = t_unclamped.max(0.0f32).min(1.0f32);

    // Hermite interpolation: t² * (3 - 2*t)
    let t_sq = t_clamped.mul(t_clamped);
    let two_t = t_clamped.mul(2.0f32);
    let term = (3.0f32).sub(two_t);

    t_sq.mul(term)
}

/// Extension trait for smooth_step on manifolds.
pub trait SmoothStepExt: Manifold + Sized + Copy {
    fn smooth_step<E0, E1>(self, edge0: E0, edge1: E1) -> impl Manifold<Output = Field>
    where
        E0: Manifold<Output = Field> + Copy,
        E1: Manifold<Output = Field> + Copy,
        Self: Manifold<Output = Field>,
    {
        smooth_step(edge0, edge1, self)
    }
}

impl<M: Manifold<Output = Field> + Sized + Copy> SmoothStepExt for M {}

// ============================================================================
// Loop-Blinn Quadratic Curve
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct LoopBlinnQuad {
    u_a: f32,
    u_b: f32,
    u_c: f32,
    v_d: f32,
    v_e: f32,
    v_f: f32,
}

impl LoopBlinnQuad {
    pub fn new(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> Option<Self> {
        let (x0, y0) = (p0[0], p0[1]);
        let (x1, y1) = (p1[0], p1[1]);
        let (x2, y2) = (p2[0], p2[1]);

        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);

        if area.abs() < 1e-6 {
            return None;
        }

        let inv_area = 1.0 / area;

        // ... math implementation ...
        // Re-implementing correctly:
        let alpha_x = (y1 - y2) * inv_area;
        let alpha_y = (x2 - x1) * inv_area;
        let alpha_c = (x1 * y2 - x2 * y1) * inv_area;

        let beta_x = (y2 - y0) * inv_area;
        let beta_y = (x0 - x2) * inv_area;
        let beta_c = (x2 * y0 - x0 * y2) * inv_area;

        let gamma_x = -alpha_x - beta_x;
        let gamma_y = -alpha_y - beta_y;
        let gamma_c = 1.0 - alpha_c - beta_c;

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

    pub fn implicit(&self) -> impl Manifold<Output = Field> + Copy {
        // u = aX + bY + c
        // v = dX + eY + f
        // result = u^2 - v
        let u = X.mul(self.u_a).add(Y.mul(self.u_b)).add(self.u_c);
        let v = X.mul(self.v_d).add(Y.mul(self.v_e)).add(self.v_f);
        u.mul(u).sub(v)
    }

    pub fn coverage(&self) -> impl Manifold<Output = Field> {
        // Using ManifoldExt method on the implicit result
        self.implicit().smooth_step(0.5f32, -0.5f32)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LineSegment {
    a: f32,
    b: f32,
    c: f32,
}

impl LineSegment {
    pub fn new(p0: [f32; 2], p1: [f32; 2]) -> Self {
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        let (a, b) = if len > 1e-6 {
            (-dy / len, dx / len)
        } else {
            (0.0, 1.0)
        };
        let c = -(a * p0[0] + b * p0[1]);
        Self { a, b, c }
    }

    // Explicit return type bound helps compiler
    pub fn signed_distance(&self) -> impl Manifold<Output = Field> + Copy {
        X.mul(self.a).add(Y.mul(self.b)).add(self.c)
    }
}

pub type Point = [f32; 2];

#[derive(Clone, Copy, Debug)]
pub enum Segment {
    Line(LineSegment),
    Quad(LoopBlinnQuad),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphBounds {
    pub width: u32,
    pub height: u32,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

#[derive(Clone, Debug)]
pub struct Glyph {
    pub segments: std::sync::Arc<[Segment]>,
    pub bounds: GlyphBounds,
    pub advance: f32,
}

impl Glyph {
    // Helper to evaluate segment coverage
    fn eval_segment(&self, segment: &Segment, x: Field, y: Field, z: Field, w: Field) -> Field {
        match segment {
            Segment::Line(l) => {
                let val = l.signed_distance().eval_raw(x, y, z, w);
                val.smooth_step(0.5f32, -0.5f32).eval_raw(x, y, z, w)
            }
            Segment::Quad(q) => q.coverage().eval_raw(x, y, z, w),
        }
    }
}

impl Manifold for Glyph {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Iterate and Sum
        let mut sum = Field::from(0.0);

        for segment in self.segments.iter() {
            let val = self.eval_segment(segment, x, y, z, w);
            // field + field
            sum = sum + val;
        }

        // Clamp result to [0, 1] using ManifoldExt
        sum.max(0.0f32).min(1.0f32).eval_raw(x, y, z, w)
    }
}
