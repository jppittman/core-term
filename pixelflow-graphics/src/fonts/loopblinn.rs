//! Loop-Blinn curve rendering using pure Manifold algebra.
//!
//! The key insight: barycentric coordinates are LINEAR in (X, Y),
//! so the entire Loop-Blinn implicit u² - v is just polynomial composition.
//! No per-lane extraction needed—it's manifolds all the way down.

use pixelflow_core::{BoxedManifold, Field, Manifold, ManifoldExt, X, Y};

// ============================================================================
// Smooth Step (AA helper) - Pure Manifold Version
// ============================================================================

pub fn smooth_step<E0, E1, M>(edge0: E0, edge1: E1, value: M) -> BoxedManifold
where
    E0: Manifold<Output = Field> + Copy + 'static,
    E1: Manifold<Output = Field> + Copy + 'static,
    M: Manifold<Output = Field> + Copy + 'static,
{
    let range = edge1.sub(edge0);
    let t_unclamped = value.sub(edge0).div(range);
    let t_clamped = t_unclamped.max(0.0f32).min(1.0f32);

    let t2 = t_clamped.mul(t_clamped);
    let term = t_clamped.mul(-2.0).add(3.0);
    t2.mul(term).boxed()
}

pub trait SmoothStepExt: Manifold + Sized + Copy {
    fn smooth_step<E0, E1>(self, edge0: E0, edge1: E1) -> BoxedManifold
    where
        E0: Manifold<Output = Field> + Copy + 'static,
        E1: Manifold<Output = Field> + Copy + 'static,
        Self: Manifold<Output = Field> + 'static,
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
    alpha_x: f32,
    alpha_y: f32,
    alpha_c: f32,
    beta_x: f32,
    beta_y: f32,
    beta_c: f32,
    ay: f32,
    by: f32,
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

        let ay = y0 - 2.0 * y1 + y2;
        let by = 2.0 * (y1 - y0);

        Some(Self {
            u_a,
            u_b,
            u_c,
            v_d,
            v_e,
            v_f,
            alpha_x,
            alpha_y,
            alpha_c,
            beta_x,
            beta_y,
            beta_c,
            ay,
            by,
        })
    }

    pub fn implicit(&self) -> BoxedManifold {
        let u = X * self.u_a + Y * self.u_b + self.u_c;
        let v = X * self.v_d + Y * self.v_e + self.v_f;
        (u * u - v).boxed()
    }

    pub fn winding(&self) -> BoxedManifold {
        let u_k = Y * self.u_b + self.u_c;
        let v_k = Y * self.v_e + self.v_f;

        let a_coeff = self.u_a * self.u_a;
        let b_coeff = u_k * (2.0 * self.u_a) - self.v_d;
        let c_coeff = u_k * u_k - v_k;

        let disc = b_coeff.clone() * b_coeff.clone() - c_coeff.clone() * (4.0 * a_coeff);
        let valid_disc = disc.clone().ge(0.0);
        let sqrt_disc = disc.abs().sqrt();

        let sign_b = b_coeff.clone().ge(0.0).select(1.0, -1.0);
        let q = (b_coeff + sign_b * sqrt_disc) * -0.5;

        let x1 = (q.clone() / a_coeff).boxed();
        let x2 = (c_coeff / q).boxed();

        let check_root = move |x_root: BoxedManifold| -> BoxedManifold {
            let is_left = X.lt(x_root.clone());
            let alpha = x_root.clone() * self.alpha_x + Y * self.alpha_y + self.alpha_c;
            let beta = x_root.clone() * self.beta_x + Y * self.beta_y + self.beta_c;

            let v_at_root = x_root * self.v_d + Y * self.v_e + self.v_f;
            let in_triangle = alpha.clone().ge(0.0).select(
                beta.clone().ge(0.0).select(
                    (alpha + beta)
                        .le(1.0)
                        .select(v_at_root.clone().lt(1.0), 0.0),
                    0.0,
                ),
                0.0,
            );

            let t = v_at_root.abs().sqrt();
            let y_prime = t * (2.0 * self.ay) + self.by;
            let dir = y_prime.ge(0.0).select(1.0, -1.0);

            in_triangle.select(is_left.select(dir, 0.0), 0.0).boxed()
        };

        valid_disc
            .select(check_root(x1) + check_root(x2), 0.0)
            .boxed()
    }
}

// ============================================================================
// Line Segment
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct LineSegment {
    p0: [f32; 2],
    p1: [f32; 2],
}

impl LineSegment {
    pub fn new(p0: [f32; 2], p1: [f32; 2]) -> Self {
        Self { p0, p1 }
    }

    pub fn winding(&self) -> BoxedManifold {
        let (x0, y0) = (self.p0[0], self.p0[1]);
        let (x1, y1) = (self.p1[0], self.p1[1]);

        let dy = y1 - y0;
        let in_y = if dy >= 0.0 {
            Y.ge(y0).select(Y.lt(y1), 0.0)
        } else {
            Y.ge(y1).select(Y.lt(y0), 0.0)
        };

        let dx_dy = (x1 - x0) / (if dy.abs() < 1e-6 { 1.0 } else { dy });
        let x_int = (Y - y0) * dx_dy + x0;
        let is_left = X.lt(x_int);
        let dir = if dy >= 0.0 { 1.0 } else { -1.0 };

        in_y.select(is_left.select(dir, 0.0), 0.0).boxed()
    }

    pub fn signed_distance(&self) -> BoxedManifold {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        let (a, b) = if len > 1e-6 {
            (-dy / len, dx / len)
        } else {
            (0.0, 1.0)
        };
        let c = -(a * self.p0[0] + b * self.p0[1]);
        (X * a + Y * b + c).boxed()
    }
}

// ============================================================================
// Shared Types
// ============================================================================

pub type Point = [f32; 2];

#[derive(Clone, Copy, Debug)]
pub enum Segment {
    Line(LineSegment),
    Quad(LoopBlinnQuad),
}

impl Segment {
    pub fn winding(&self) -> BoxedManifold {
        match self {
            Segment::Line(l) => l.winding(),
            Segment::Quad(q) => q.winding(),
        }
    }
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

impl Manifold for Glyph {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Build the winding sum manifold
        let mut sum_manifold: Option<BoxedManifold> = None;
        for segment in self.segments.iter() {
            let winding = segment.winding();
            sum_manifold = match sum_manifold {
                Some(acc) => Some((acc + winding).boxed()),
                None => Some(winding),
            };
        }
        let winding = sum_manifold.unwrap_or_else(|| 0.0.boxed());
        let coverage = winding.abs().gt(0.5).select(1.0, 0.0);

        // TODO: Add AABB clipping once glyphs are in unit space
        // For now, segments are pre-scaled to pixel space, so AABB doesn't work
        coverage.eval_raw(x, y, z, w)
    }
}
