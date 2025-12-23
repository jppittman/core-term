//! Loop-Blinn curve rendering with smooth winding for Jet2.

use pixelflow_core::{Field, Jet2, Manifold, Numeric};

const MIN_TRIANGLE_AREA: f32 = 1e-6;

// ============================================================================
// Loop-Blinn Quadratic Curve
// ============================================================================

/// Loop-Blinn quadratic curve parameters (WIP - fields used in future curve impl).
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
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
        if area.abs() < MIN_TRIANGLE_AREA {
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

        Some(Self {
            u_a,
            u_b,
            u_c,
            v_d,
            v_e,
            v_f,
        })
    }
}

// ============================================================================
// Line Segment
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct LineSegment {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
}

impl LineSegment {
    pub fn new(p0: [f32; 2], p1: [f32; 2]) -> Self {
        Self { p0, p1 }
    }
}

// Field implementation: hard step comparisons
impl Manifold<Field> for LineSegment {
    type Output = Field;
    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        let p0_y = Field::from_f32(self.p0[1]);
        let p1_y = Field::from_f32(self.p1[1]);
        let dy = p1_y - p0_y;

        // Skip near-horizontal
        let dy_abs = dy.abs();
        let valid = dy_abs.gt(Field::from_f32(1e-6));

        let p0_x = Field::from_f32(self.p0[0]);
        let p1_x = Field::from_f32(self.p1[0]);
        let dx = p1_x - p0_x;

        // Y-range check: y >= y_min AND y <= y_max
        let y_min = p0_y.min(p1_y);
        let y_max = p0_y.max(p1_y);

        let geq_min = y.ge(y_min);
        let leq_max = y.le(y_max);
        // Both must be true: use select(cond1, cond2, 0) to AND them
        let in_y = Field::select(geq_min, leq_max, Field::from_f32(0.0));

        // X-intersection
        let x_int = p0_x + (y - p0_y) * (dx / dy);
        let is_left = x.lt(x_int);

        // Direction
        let going_up = dy.gt(Field::from_f32(0.0));
        let dir = Field::select(going_up, Field::from_f32(1.0), Field::from_f32(-1.0));

        // Combine: valid * in_y * is_left * dir
        let contrib = Field::select(is_left, dir, Field::from_f32(0.0));
        let result = Field::select(in_y, contrib, Field::from_f32(0.0));
        Field::select(valid, result, Field::from_f32(0.0))
    }
}

// Jet2 implementation: soft differentiable comparisons
impl Manifold<Jet2> for LineSegment {
    type Output = Jet2;
    fn eval_raw(&self, x: Jet2, y: Jet2, _z: Jet2, _w: Jet2) -> Jet2 {
        let p0_y = Jet2::from_f32(self.p0[1]);
        let p1_y = Jet2::from_f32(self.p1[1]);
        let dy = p1_y - p0_y;

        // Hard validity check (doesn't need to be smooth)
        let dy_abs = dy.abs();
        let valid = dy_abs.gt(Jet2::from_f32(1e-6));

        let p0_x = Jet2::from_f32(self.p0[0]);
        let p1_x = Jet2::from_f32(self.p1[0]);
        let dx = p1_x - p0_x;

        // Soft y-range check using inline hermite smooth_step
        let y_min = p0_y.min(p1_y);
        let y_max = p0_y.max(p1_y);

        let k = Jet2::from_f32(2.0); // Sharpness

        // y >= y_min: soft_gt(y, y_min)
        let diff_min = (y - y_min) / k;
        let t_min = ((diff_min) + Jet2::from_f32(1.0)) / Jet2::from_f32(2.0);
        let t_min = t_min.max(Jet2::from_f32(0.0)).min(Jet2::from_f32(1.0));
        let t2_min = t_min * t_min;
        let t3_min = t2_min * t_min;
        let geq_min = t3_min * Jet2::from_f32(-2.0) + t2_min * Jet2::from_f32(3.0);

        // y <= y_max: soft_lt(y, y_max)
        let diff_max = (y_max - y) / k;
        let t_max = ((diff_max) + Jet2::from_f32(1.0)) / Jet2::from_f32(2.0);
        let t_max = t_max.max(Jet2::from_f32(0.0)).min(Jet2::from_f32(1.0));
        let t2_max = t_max * t_max;
        let t3_max = t2_max * t_max;
        let leq_max = t3_max * Jet2::from_f32(-2.0) + t2_max * Jet2::from_f32(3.0);

        let in_y = geq_min * leq_max;

        // X-intersection
        let x_int = p0_x + (y - p0_y) * (dx / dy);

        // x < x_int: soft_lt(x, x_int)
        let diff_x = (x_int - x) / k;
        let t_x = ((diff_x) + Jet2::from_f32(1.0)) / Jet2::from_f32(2.0);
        let t_x = t_x.max(Jet2::from_f32(0.0)).min(Jet2::from_f32(1.0));
        let t2_x = t_x * t_x;
        let t3_x = t2_x * t_x;
        let is_left = t3_x * Jet2::from_f32(-2.0) + t2_x * Jet2::from_f32(3.0);

        // Direction (hard is ok)
        let going_up = dy.gt(Jet2::from_f32(0.0));
        let dir = Jet2::select(going_up, Jet2::from_f32(1.0), Jet2::from_f32(-1.0));

        // Combine
        let result = in_y * is_left * dir;
        Jet2::select(valid, result, Jet2::from_f32(0.0))
    }
}

// Quads: stub for now
impl<I: Numeric> Manifold<I> for LoopBlinnQuad {
    type Output = I;
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        I::from_f32(0.0)
    }
}

// ============================================================================
// AlgebraicGlyph
// ============================================================================

#[derive(Clone, Debug)]
pub struct AlgebraicGlyph {
    pub line_segments: std::sync::Arc<[LineSegment]>,
    pub quad_segments: std::sync::Arc<[LoopBlinnQuad]>,
}

impl AlgebraicGlyph {
    pub fn new(
        line_segments: std::sync::Arc<[LineSegment]>,
        quad_segments: std::sync::Arc<[LoopBlinnQuad]>,
    ) -> Self {
        Self {
            line_segments,
            quad_segments,
        }
    }
}

// Field winding sum with Jet2-based gradient AA
impl Manifold<Field> for AlgebraicGlyph {
    type Output = Field;
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Construct Jet2 to get gradients
        let x_jet = Jet2::x(x);
        let y_jet = Jet2::y(y);
        let z_jet = Jet2 {
            val: z,
            dx: Field::from_f32(0.0),
            dy: Field::from_f32(0.0),
        };
        let w_jet = Jet2 {
            val: w,
            dx: Field::from_f32(0.0),
            dy: Field::from_f32(0.0),
        };

        // Get winding + gradients from Jet2 impl
        let winding_jet = Manifold::<Jet2>::eval_raw(self, x_jet, y_jet, z_jet, w_jet);

        // Approximate signed distance using gradients
        let grad_sq = winding_jet.dx * winding_jet.dx + winding_jet.dy * winding_jet.dy;
        let grad_mag = (grad_sq + Field::from_f32(1e-8)).sqrt();

        // Normalize: (winding - threshold) / |grad|
        // Threshold at 0.5 for non-zero winding rule
        let dist = (winding_jet.val - Field::from_f32(0.5)) / grad_mag;

        // Hard threshold on distance (dist > 0 means inside)
        // For soft AA, could use smooth_step on dist, but hard is fine with gradient normalization
        let inside = dist.gt(Field::from_f32(0.0));
        Field::select(inside, Field::from_f32(1.0), Field::from_f32(0.0))
    }
}

// Jet2 winding sum
impl Manifold<Jet2> for AlgebraicGlyph {
    type Output = Jet2;
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        let mut winding = Jet2::from_f32(0.0);

        for seg in self.line_segments.iter() {
            winding = winding + Manifold::<Jet2>::eval_raw(seg, x, y, z, w);
        }

        for seg in self.quad_segments.iter() {
            winding = winding + Manifold::<Jet2>::eval_raw(seg, x, y, z, w);
        }

        winding.abs()
    }
}

// ============================================================================
// Support Types
// ============================================================================

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

use crate::shapes::Square;
use crate::transform::{Scale, Translate};

pub type UnitGlyph = Square<Scale<AlgebraicGlyph>, f32>;

#[derive(Clone, Debug)]
pub struct Glyph {
    pub advance: f32,
    pub manifold: Translate<Scale<UnitGlyph>>,
}

impl Glyph {
    pub fn translate(&mut self, delta: [f32; 2]) {
        self.manifold.offset[0] += delta[0];
        self.manifold.offset[1] += delta[1];
    }

    pub fn set_position(&mut self, pos: [f32; 2]) {
        self.manifold.offset = pos;
    }

    pub fn set_size(&mut self, size: f32) {
        self.manifold.manifold.factor = size;
    }
}

impl Manifold for Glyph {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.manifold.eval_raw(x, y, z, w)
    }
}
