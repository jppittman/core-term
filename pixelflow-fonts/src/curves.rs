use pixelflow_core::{Batch, SimdFloatOps, SimdOps};
use std::ops::{Add, Div, Mul, Sub};

pub type Point = [f32; 2];

#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub p0: Point,
    pub p1: Point,
}

#[derive(Clone, Copy, Debug)]
pub struct Quadratic {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    // Precomputed projection matrix for Loop-Blinn
    pub projection: [[f32; 3]; 2],
}

impl Quadratic {
    pub fn try_new(p0: Point, p1: Point, p2: Point) -> Option<Self> {
        let det = p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]);

        if det.abs() < 1e-6 {
            return None;
        }

        let inv_det = 1.0 / det;

        let a = (0.0 * (p1[1] - p2[1]) + 0.5 * (p2[1] - p0[1]) + 1.0 * (p0[1] - p1[1])) * inv_det;
        let b = (0.0 * (p2[0] - p1[0]) + 0.5 * (p0[0] - p2[0]) + 1.0 * (p1[0] - p0[0])) * inv_det;
        let c = (0.0 * (p1[0] * p2[1] - p2[0] * p1[1])
            + 0.5 * (p2[0] * p0[1] - p0[0] * p2[1])
            + 1.0 * (p0[0] * p1[1] - p1[0] * p0[1]))
            * inv_det;

        let d = (0.0 * (p1[1] - p2[1]) + 0.0 * (p2[1] - p0[1]) + 1.0 * (p0[1] - p1[1])) * inv_det;
        let e = (0.0 * (p2[0] - p1[0]) + 0.0 * (p0[0] - p2[0]) + 1.0 * (p1[0] - p0[0])) * inv_det;
        let f = (0.0 * (p1[0] * p2[1] - p2[0] * p1[1])
            + 0.0 * (p2[0] * p0[1] - p0[0] * p2[1])
            + 1.0 * (p0[0] * p1[1] - p1[0] * p0[1]))
            * inv_det;

        Some(Self {
            p0,
            p1,
            p2,
            projection: [[a, b, c], [d, e, f]],
        })
    }
}

#[derive(Clone, Copy)]
pub enum Segment {
    Line(Line),
    Quad(Quadratic),
}

impl Segment {
    /// Computes the winding number contribution of a ray cast from (x, y) to (+inf, y).
    /// Returns 1, -1, or 0 in each lane (as i32 bits).
    #[inline(always)]
    pub fn winding_batch(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<u32> {
        match self {
            Segment::Line(l) => {
                let p0y = Batch::splat(l.p0[1]);
                let p1y = Batch::splat(l.p1[1]);
                let p0x = Batch::splat(l.p0[0]);
                let p1x = Batch::splat(l.p1[0]);

                let c1 = p0y.cmp_le(y) & y.cmp_lt(p1y); // p0 <= y < p1
                let c2 = p1y.cmp_le(y) & y.cmp_lt(p0y); // p1 <= y < p0
                let in_y = c1 | c2;

                // t = (y - p0y) / (p1y - p0y)
                let dy = p1y - p0y;
                let t = (y - p0y) / dy;
                let x_int = p0x + t * (p1x - p0x);

                let is_left = x.cmp_lt(x_int);
                let mask = in_y & is_left;

                let dir = if l.p0[1] < l.p1[1] { 1i32 } else { -1i32 };
                let dir_batch = Batch::splat(dir as u32);

                dir_batch.select(Batch::splat(0), mask)
            }
            Segment::Quad(q) => {
                let a_val = q.p0[1] - 2.0 * q.p1[1] + q.p2[1];
                let b_val = 2.0 * (q.p1[1] - q.p0[1]);
                // c_val depends on y: c = p0y - y

                let p0y = Batch::splat(q.p0[1]);
                let p1y = Batch::splat(q.p1[1]);
                // let p2y = Batch::splat(q.p2[1]);
                let p0x = Batch::splat(q.p0[0]);
                let p1x = Batch::splat(q.p1[0]);
                let p2x = Batch::splat(q.p2[0]);

                let mut winding = Batch::splat(0u32);

                if a_val.abs() < 1e-6 {
                    // Linear approximation
                    if b_val.abs() > 1e-6 {
                        let b = Batch::splat(b_val);
                        // t = -c / b = -(p0y - y) / b = (y - p0y) / b
                        let t = (y - p0y) / b;
                        let valid_t = t.cmp_ge(Batch::splat(0.0)) & t.cmp_lt(Batch::splat(1.0));

                        let one = Batch::splat(1.0);
                        let omt = one - t;
                        let xt = (omt * omt) * p0x
                            + (Batch::splat(2.0) * omt * t) * p1x
                            + (t * t) * p2x;

                        let is_left = x.cmp_lt(xt);
                        let mask = valid_t & is_left;

                        let dir = if b_val > 0.0 { 1i32 } else { -1i32 };
                        winding = winding + Batch::splat(dir as u32).select(Batch::splat(0), mask);
                    }
                } else {
                    let a = Batch::splat(a_val);
                    let b = Batch::splat(b_val);
                    let c = p0y - y; // c = p0y - y

                    let disc = b * b - Batch::splat(4.0) * a * c;
                    let has_roots = disc.cmp_ge(Batch::splat(0.0));

                    // We can't easily branch on `has_roots` per pixel inside batch.
                    // We execute for all, mask at end.
                    let sqrt_disc = disc.sqrt(); // NaN if disc < 0, but we mask it

                    let two_a = Batch::splat(2.0) * a;
                    let t1 = (Batch::splat(0.0) - b - sqrt_disc) / two_a;
                    let t2 = (Batch::splat(0.0) - b + sqrt_disc) / two_a;

                    for t in [t1, t2] {
                        let valid_t = t.cmp_ge(Batch::splat(0.0)) & t.cmp_lt(Batch::splat(1.0));

                        let one = Batch::splat(1.0);
                        let omt = one - t;
                        let xt = (omt * omt) * p0x
                            + (Batch::splat(2.0) * omt * t) * p1x
                            + (t * t) * p2x;

                        let is_left = x.cmp_lt(xt);

                        // dy/dt = 2at + b
                        let dy_dt = Batch::splat(2.0) * a * t + b;
                        let increasing = dy_dt.cmp_gt(Batch::splat(0.0));

                        let val = increasing.select(Batch::splat(1u32), Batch::splat((-1i32) as u32));

                        let mask = has_roots & valid_t & is_left;
                        winding = winding + val.select(Batch::splat(0), mask);
                    }
                }
                winding
            }
        }
    }

    /// Computes the signed pseudo-distance from (x, y) to the segment.
    #[inline(always)]
    pub fn min_distance_batch(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<f32> {
        match self {
            Segment::Line(l) => {
                let dx = l.p1[0] - l.p0[0];
                let dy = l.p1[1] - l.p0[1];
                let len_sq = dx * dx + dy * dy;
                let len = len_sq.sqrt();

                // If len < epsilon, return large distance
                if len < 1e-6 {
                    return Batch::splat(1000.0);
                }

                let p0x = Batch::splat(l.p0[0]);
                let p0y = Batch::splat(l.p0[1]);
                let dx_batch = Batch::splat(dx);
                let dy_batch = Batch::splat(dy);
                let len_batch = Batch::splat(len);

                // ((x - l.p0[0]) * -dy + (y - l.p0[1]) * dx) / len
                let dist = ((x - p0x) * (Batch::splat(0.0) - dy_batch) + (y - p0y) * dx_batch) / len_batch;
                dist
            }
            Segment::Quad(q) => {
                // let u = x * q.projection[0][0] + y * q.projection[0][1] + q.projection[0][2];
                let u = x * Batch::splat(q.projection[0][0])
                      + y * Batch::splat(q.projection[0][1])
                      + Batch::splat(q.projection[0][2]);

                // let v = x * q.projection[1][0] + y * q.projection[1][1] + q.projection[1][2];
                let v = x * Batch::splat(q.projection[1][0])
                      + y * Batch::splat(q.projection[1][1])
                      + Batch::splat(q.projection[1][2]);

                // let f = u * u - v;
                let f = u * u - v;

                let du_dx = Batch::splat(q.projection[0][0]);
                let du_dy = Batch::splat(q.projection[0][1]);
                let dv_dx = Batch::splat(q.projection[1][0]);
                let dv_dy = Batch::splat(q.projection[1][1]);

                // let df_dx = 2.0 * u * du_dx - dv_dx;
                let df_dx = Batch::splat(2.0) * u * du_dx - dv_dx;

                // let df_dy = 2.0 * u * du_dy - dv_dy;
                let df_dy = Batch::splat(2.0) * u * du_dy - dv_dy;

                // let grad_len = (df_dx * df_dx + df_dy * df_dy).sqrt();
                let grad_sq = df_dx * df_dx + df_dy * df_dy;
                let grad_len = grad_sq.sqrt();

                // if grad_len < 1e-6 { 1000.0 } else { f / grad_len }
                let zero_grad = grad_len.cmp_lt(Batch::splat(1e-6));

                let dist = f / grad_len;
                // zero_grad is Batch<u32>. Transmute to Batch<f32> for select mask
                Batch::splat(1000.0).select(dist, zero_grad.transmute::<f32>())
            }
        }
    }
}
