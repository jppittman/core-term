use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::backend::{Backend, BatchArithmetic, FloatBatchOps};
use pixelflow_core::curve::Mat3;

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
    pub projection: Mat3,
}

impl Quadratic {
    pub fn try_new(p0: Point, p1: Point, p2: Point) -> Option<Self> {
        // Compute projection matrix mapping (x, y) -> (u, v)
        // such that p0 -> (0,0), p1 -> (0.5, 0), p2 -> (1,1)
        let projection = Mat3::from_affine_points(p0, p1, p2)?;

        Some(Self {
            p0,
            p1,
            p2,
            projection,
        })
    }
}

#[derive(Clone, Copy)]
pub enum Segment {
    Line(Line),
    Quad(Quadratic),
}

impl Segment {
    /// Scalar winding number contribution of a ray cast from (x, y) to (+inf, y).
    #[inline]
    pub fn winding(&self, x: f32, y: f32) -> i32 {
        match self {
            Segment::Line(l) => {
                let p0y = l.p0[1];
                let p1y = l.p1[1];
                let p0x = l.p0[0];
                let p1x = l.p1[0];

                let c1 = p0y <= y && y < p1y;
                let c2 = p1y <= y && y < p0y;
                if !c1 && !c2 {
                    return 0;
                }

                let dy = p1y - p0y;
                let t = (y - p0y) / dy;
                let x_int = p0x + t * (p1x - p0x);

                if x < x_int {
                    if l.p0[1] < l.p1[1] { 1 } else { -1 }
                } else {
                    0
                }
            }
            Segment::Quad(q) => {
                let a_val = q.p0[1] - 2.0 * q.p1[1] + q.p2[1];
                let b_val = 2.0 * (q.p1[1] - q.p0[1]);
                let p0y = q.p0[1];
                let p0x = q.p0[0];
                let p1x = q.p1[0];
                let p2x = q.p2[0];

                let mut winding = 0i32;

                if a_val.abs() < 1e-6 {
                    if b_val.abs() > 1e-6 {
                        let t = (y - p0y) / b_val;
                        if t >= 0.0 && t < 1.0 {
                            let omt = 1.0 - t;
                            let xt = omt * omt * p0x + 2.0 * omt * t * p1x + t * t * p2x;
                            if x < xt {
                                winding += if b_val > 0.0 { 1 } else { -1 };
                            }
                        }
                    }
                } else {
                    let c = p0y - y;
                    let disc = b_val * b_val - 4.0 * a_val * c;
                    if disc >= 0.0 {
                        let sqrt_disc = disc.sqrt();
                        let two_a = 2.0 * a_val;
                        let t1 = (-b_val - sqrt_disc) / two_a;
                        let t2 = (-b_val + sqrt_disc) / two_a;

                        for t in [t1, t2] {
                            if t >= 0.0 && t < 1.0 {
                                let omt = 1.0 - t;
                                let xt = omt * omt * p0x + 2.0 * omt * t * p1x + t * t * p2x;
                                if x < xt {
                                    let dy_dt = 2.0 * a_val * t + b_val;
                                    winding += if dy_dt > 0.0 { 1 } else { -1 };
                                }
                            }
                        }
                    }
                }
                winding
            }
        }
    }

    /// Scalar signed pseudo-distance from (x, y) to the segment.
    #[inline]
    pub fn min_distance(&self, x: f32, y: f32) -> f32 {
        match self {
            Segment::Line(l) => {
                let dx = l.p1[0] - l.p0[0];
                let dy = l.p1[1] - l.p0[1];
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-6 {
                    return 1000.0;
                }
                ((x - l.p0[0]) * -dy + (y - l.p0[1]) * dx) / len
            }
            Segment::Quad(q) => {
                let m = q.projection.m;
                let u = x * m[0][0] + y * m[0][1] + m[0][2];
                let v = x * m[1][0] + y * m[1][1] + m[1][2];
                let f = u * u - v;

                let du_dx = m[0][0];
                let du_dy = m[0][1];
                let dv_dx = m[1][0];
                let dv_dy = m[1][1];

                let df_dx = 2.0 * u * du_dx - dv_dx;
                let df_dy = 2.0 * u * du_dy - dv_dy;
                let grad_len = (df_dx * df_dx + df_dy * df_dy).sqrt();

                if grad_len < 1e-6 { 1000.0 } else { f / grad_len }
            }
        }
    }

    /// Computes the winding number contribution of a ray cast from (x, y) to (+inf, y).
    #[inline(always)]
    pub fn winding_batch(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<u32> {
        match self {
            Segment::Line(l) => {
                let p0y = Batch::<f32>::splat(l.p0[1]);
                let p1y = Batch::<f32>::splat(l.p1[1]);
                let p0x = Batch::<f32>::splat(l.p0[0]);
                let p1x = Batch::<f32>::splat(l.p1[0]);

                let c1 = p0y.cmp_le(y) & y.cmp_lt(p1y);
                let c2 = p1y.cmp_le(y) & y.cmp_lt(p0y);
                let in_y = c1 | c2;

                let dy = p1y - p0y;
                let t = (y - p0y) / dy;
                let x_int = p0x + t * (p1x - p0x);

                let is_left = x.cmp_lt(x_int);
                let mask_f32 = in_y & is_left;
                let mask = NativeBackend::transmute_f32_to_u32(mask_f32);

                let dir = if l.p0[1] < l.p1[1] { 1i32 } else { -1i32 };
                let dir_batch = Batch::<u32>::splat(dir as u32);
                let zero = Batch::<u32>::splat(0);

                mask.select(dir_batch, zero)
            }
            Segment::Quad(q) => {
                let a_val = q.p0[1] - 2.0 * q.p1[1] + q.p2[1];
                let b_val = 2.0 * (q.p1[1] - q.p0[1]);

                let p0y = Batch::<f32>::splat(q.p0[1]);
                let p0x = Batch::<f32>::splat(q.p0[0]);
                let p1x = Batch::<f32>::splat(q.p1[0]);
                let p2x = Batch::<f32>::splat(q.p2[0]);

                let mut winding = Batch::<u32>::splat(0u32);

                if a_val.abs() < 1e-6 {
                    if b_val.abs() > 1e-6 {
                        let b = Batch::<f32>::splat(b_val);
                        let t = (y - p0y) / b;
                        let zero_f = Batch::<f32>::splat(0.0);
                        let one_f = Batch::<f32>::splat(1.0);
                        let valid_t = t.cmp_ge(zero_f) & t.cmp_lt(one_f);

                        let omt = one_f - t;
                        let two_f = Batch::<f32>::splat(2.0);
                        let xt = (omt * omt) * p0x + (two_f * omt * t) * p1x + (t * t) * p2x;

                        let is_left = x.cmp_lt(xt);
                        let mask_f32 = valid_t & is_left;
                        let mask = NativeBackend::transmute_f32_to_u32(mask_f32);

                        let dir = if b_val > 0.0 { 1i32 } else { -1i32 };
                        let dir_batch = Batch::<u32>::splat(dir as u32);
                        let zero = Batch::<u32>::splat(0);
                        winding = winding + mask.select(dir_batch, zero);
                    }
                } else {
                    let a = Batch::<f32>::splat(a_val);
                    let b = Batch::<f32>::splat(b_val);
                    let c = p0y - y;

                    let four = Batch::<f32>::splat(4.0);
                    let disc = b * b - four * a * c;
                    let zero_f = Batch::<f32>::splat(0.0);
                    let has_roots = disc.cmp_ge(zero_f);

                    let sqrt_disc = disc.sqrt();
                    let two = Batch::<f32>::splat(2.0);
                    let two_a = two * a;
                    let t1 = (zero_f - b - sqrt_disc) / two_a;
                    let t2 = (zero_f - b + sqrt_disc) / two_a;

                    let one_f = Batch::<f32>::splat(1.0);
                    for t in [t1, t2] {
                        let valid_t = t.cmp_ge(zero_f) & t.cmp_lt(one_f);

                        let omt = one_f - t;
                        let xt = (omt * omt) * p0x + (two * omt * t) * p1x + (t * t) * p2x;

                        let is_left = x.cmp_lt(xt);
                        let dy_dt = two * a * t + b;
                        let increasing = dy_dt.cmp_gt(zero_f);

                        let one_u = Batch::<u32>::splat(1u32);
                        let neg_one_u = Batch::<u32>::splat((-1i32) as u32);
                        let zero_u = Batch::<u32>::splat(0);

                        let increasing_mask = NativeBackend::transmute_f32_to_u32(increasing);
                        let val = increasing_mask.select(one_u, neg_one_u);

                        let mask_f32 = has_roots & valid_t & is_left;
                        let mask = NativeBackend::transmute_f32_to_u32(mask_f32);
                        winding = winding + mask.select(val, zero_u);
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

                if len < 1e-6 {
                    return Batch::<f32>::splat(1000.0);
                }

                let p0x = Batch::<f32>::splat(l.p0[0]);
                let p0y = Batch::<f32>::splat(l.p0[1]);
                let dx_batch = Batch::<f32>::splat(dx);
                let dy_batch = Batch::<f32>::splat(dy);
                let len_batch = Batch::<f32>::splat(len);
                let zero = Batch::<f32>::splat(0.0);

                ((x - p0x) * (zero - dy_batch) + (y - p0y) * dx_batch) / len_batch
            }
            Segment::Quad(q) => {
                let m = q.projection.m;

                let u = x * Batch::<f32>::splat(m[0][0])
                      + y * Batch::<f32>::splat(m[0][1])
                      + Batch::<f32>::splat(m[0][2]);

                let v = x * Batch::<f32>::splat(m[1][0])
                      + y * Batch::<f32>::splat(m[1][1])
                      + Batch::<f32>::splat(m[1][2]);

                let f = u * u - v;

                let du_dx = Batch::<f32>::splat(m[0][0]);
                let du_dy = Batch::<f32>::splat(m[0][1]);
                let dv_dx = Batch::<f32>::splat(m[1][0]);
                let dv_dy = Batch::<f32>::splat(m[1][1]);

                let two = Batch::<f32>::splat(2.0);
                let df_dx = two * u * du_dx - dv_dx;
                let df_dy = two * u * du_dy - dv_dy;

                let grad_sq = df_dx * df_dx + df_dy * df_dy;
                let grad_len = grad_sq.sqrt();

                let epsilon = Batch::<f32>::splat(1e-6);
                let zero_grad = grad_len.cmp_lt(epsilon);

                let dist = f / grad_len;
                let large_dist = Batch::<f32>::splat(1000.0);

                let mask = NativeBackend::transmute_f32_to_u32(zero_grad);
                let large_u32 = NativeBackend::transmute_f32_to_u32(large_dist);
                let dist_u32 = NativeBackend::transmute_f32_to_u32(dist);
                NativeBackend::transmute_u32_to_f32(mask.select(large_u32, dist_u32))
            }
        }
    }
}
