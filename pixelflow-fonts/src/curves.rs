use pixelflow_core::batch::{Batch, SimdFloatOps};

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
    pub projection: [[f32; 3]; 2],
}

impl Quadratic {
    pub fn try_new(p0: Point, p1: Point, p2: Point) -> Option<Self> {
        let det = p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]);
        if det.abs() < 1e-6 { return None; }
        let inv_det = 1.0 / det;
        let a = (0.0 * (p1[1] - p2[1]) + 0.5 * (p2[1] - p0[1]) + 1.0 * (p0[1] - p1[1])) * inv_det;
        let b = (0.0 * (p2[0] - p1[0]) + 0.5 * (p0[0] - p2[0]) + 1.0 * (p1[0] - p0[0])) * inv_det;
        let c = (0.0 * (p1[0] * p2[1] - p2[0] * p1[1]) + 0.5 * (p2[0] * p0[1] - p0[0] * p2[1]) + 1.0 * (p0[0] * p1[1] - p1[0] * p0[1])) * inv_det;
        let d = (0.0 * (p1[1] - p2[1]) + 0.0 * (p2[1] - p0[1]) + 1.0 * (p0[1] - p1[1])) * inv_det;
        let e = (0.0 * (p2[0] - p1[0]) + 0.0 * (p0[0] - p2[0]) + 1.0 * (p1[0] - p0[0])) * inv_det;
        let f = (0.0 * (p1[0] * p2[1] - p2[0] * p1[1]) + 0.0 * (p2[0] * p0[1] - p0[0] * p2[1]) + 1.0 * (p0[0] * p1[1] - p1[0] * p0[1])) * inv_det;
        Some(Self { p0, p1, p2, projection: [[a, b, c], [d, e, f]] })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Segment {
    Line(Line),
    Quad(Quadratic),
}

impl Segment {
    pub fn winding_batch(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<f32> {
        match self {
            Segment::Line(l) => {
                let p0y = Batch::splat(l.p0[1]);
                let p1y = Batch::splat(l.p1[1]);
                let p0x = Batch::splat(l.p0[0]);
                let p1x = Batch::splat(l.p1[0]);

                // Condition: (p0.y <= y < p1.y) || (p1.y <= y < p0.y)
                // Correct logic: (p0 <= y) AND (y < p1)
                // Using bitand/bitor for boolean logic on masks
                let cond1 = p0y.le(y) & y.lt(p1y);
                let cond2 = p1y.le(y) & y.lt(p0y);
                let active = cond1 | cond2;

                // t = (y - p0.y) / (p1.y - p0.y)
                let dy = p1y - p0y;
                let dy_safe = dy.select(Batch::splat(1.0), dy.abs().lt(Batch::splat(1e-6)));
                let t = (y - p0y) / dy_safe;

                let x_int = p0x + t * (p1x - p0x);

                // if x < x_int
                let x_cond = x.lt(x_int);

                // winding += 1 if p0.y < p1.y else -1
                let val = if l.p0[1] < l.p1[1] { 1.0 } else { -1.0 };
                let w = Batch::splat(val);

                // Result: active & x_cond ? w : 0
                let mask = active & x_cond;
                w.select(Batch::splat(0.0), mask)
            }
            Segment::Quad(q) => {
                let p0y = Batch::splat(q.p0[1]);
                let p1y = Batch::splat(q.p1[1]);
                let p2y = Batch::splat(q.p2[1]);
                let p0x = Batch::splat(q.p0[0]);
                let p1x = Batch::splat(q.p1[0]);
                let p2x = Batch::splat(q.p2[0]);

                let a = p0y - Batch::splat(2.0) * p1y + p2y;
                let b = Batch::splat(2.0) * (p1y - p0y);
                let c = p0y - y;

                // Two cases: a ~ 0 (linear) and quadratic
                let is_linear = a.abs().lt(Batch::splat(1e-6));

                // Linear part: -c / b
                let b_safe = b.select(Batch::splat(1.0), b.abs().lt(Batch::splat(1e-6)));
                let t_lin = -c / b_safe;
                let valid_lin = t_lin.ge(Batch::splat(0.0)) & t_lin.lt(Batch::splat(1.0));

                // xt_lin
                let one_t = Batch::splat(1.0) - t_lin;
                let xt_lin = one_t * one_t * p0x + Batch::splat(2.0) * one_t * t_lin * p1x + t_lin * t_lin * p2x;

                // Actually b determines direction: b = 2(p1-p0). b>0 -> p1>p0.
                let w_lin = Batch::splat(1.0).select(Batch::splat(0.0), b.gt(Batch::splat(0.0))) -
                            Batch::splat(1.0).select(Batch::splat(0.0), b.lt(Batch::splat(0.0)));

                let mask_lin = valid_lin & x.lt(xt_lin) & b.abs().gt(Batch::splat(1e-6));
                let res_lin = w_lin.select(Batch::splat(0.0), mask_lin);

                // Quadratic part
                let disc = b * b - Batch::splat(4.0) * a * c;
                let has_roots = disc.ge(Batch::splat(0.0));
                let sqrt_disc = disc.sqrt(); // NaN if disc < 0, but masked out later

                let two_a = Batch::splat(2.0) * a;
                let two_a_safe = two_a.select(Batch::splat(1.0), two_a.abs().lt(Batch::splat(1e-6)));

                let t1 = (-b - sqrt_disc) / two_a_safe;
                let t2 = (-b + sqrt_disc) / two_a_safe;

                // Process t1
                let valid_t1 = t1.ge(Batch::splat(0.0)) & t1.lt(Batch::splat(1.0));
                let one_t1 = Batch::splat(1.0) - t1;
                let xt1 = one_t1 * one_t1 * p0x + Batch::splat(2.0) * one_t1 * t1 * p1x + t1 * t1 * p2x;
                let dy_dt1 = two_a * t1 + b;
                let w1 = Batch::splat(1.0).select(Batch::splat(0.0), dy_dt1.gt(Batch::splat(0.0))) -
                         Batch::splat(1.0).select(Batch::splat(0.0), dy_dt1.lt(Batch::splat(0.0)));
                let mask_t1 = valid_t1 & x.lt(xt1);
                let res_t1 = w1.select(Batch::splat(0.0), mask_t1);

                // Process t2
                let valid_t2 = t2.ge(Batch::splat(0.0)) & t2.lt(Batch::splat(1.0));
                let one_t2 = Batch::splat(1.0) - t2;
                let xt2 = one_t2 * one_t2 * p0x + Batch::splat(2.0) * one_t2 * t2 * p1x + t2 * t2 * p2x;
                let dy_dt2 = two_a * t2 + b;
                let w2 = Batch::splat(1.0).select(Batch::splat(0.0), dy_dt2.gt(Batch::splat(0.0))) -
                         Batch::splat(1.0).select(Batch::splat(0.0), dy_dt2.lt(Batch::splat(0.0)));
                let mask_t2 = valid_t2 & x.lt(xt2);
                let res_t2 = w2.select(Batch::splat(0.0), mask_t2);

                let res_quad = res_t1 + res_t2;

                // Combine (res_quad if not linear, else res_lin) -> so select(res_lin, res_quad, is_linear)
                // Scalar: (self & m) | (other & !m). If m true, self.
                // So if is_linear is true, we want res_lin.
                let result = res_lin.select(res_quad, is_linear);
                // Also mask out if discriminant < 0 for quadratic case (implied by NaN or logic?)
                result.select(Batch::splat(0.0), is_linear | has_roots)
            }
        }
    }

    pub fn min_dist_batch(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<f32> {
         match self {
            Segment::Line(l) => {
                let dx = Batch::splat(l.p1[0] - l.p0[0]);
                let dy = Batch::splat(l.p1[1] - l.p0[1]);
                let len = (dx * dx + dy * dy).sqrt();

                let mask_small = len.lt(Batch::splat(1e-6));
                let len_safe = Batch::splat(1.0).select(len, mask_small);

                let num = (x - Batch::splat(l.p0[0])) * -dy + (y - Batch::splat(l.p0[1])) * dx;
                let dist = num / len_safe;

                Batch::splat(1000.0).select(dist, mask_small)
            }
            Segment::Quad(q) => {
                let u = x * Batch::splat(q.projection[0][0]) + y * Batch::splat(q.projection[0][1]) + Batch::splat(q.projection[0][2]);
                let v = x * Batch::splat(q.projection[1][0]) + y * Batch::splat(q.projection[1][1]) + Batch::splat(q.projection[1][2]);

                let f = u * u - v;

                let du_dx = Batch::splat(q.projection[0][0]);
                let du_dy = Batch::splat(q.projection[0][1]);
                let dv_dx = Batch::splat(q.projection[1][0]);
                let dv_dy = Batch::splat(q.projection[1][1]);

                let df_dx = Batch::splat(2.0) * u * du_dx - dv_dx;
                let df_dy = Batch::splat(2.0) * u * du_dy - dv_dy;

                let grad_len = (df_dx * df_dx + df_dy * df_dy).sqrt();
                let mask_small = grad_len.lt(Batch::splat(1e-6));
                let grad_len_safe = Batch::splat(1.0).select(grad_len, mask_small);

                let dist = f / grad_len_safe;
                Batch::splat(1000.0).select(dist, mask_small)
            }
         }
    }
}

// Scalar impls for testing (optional, can be removed if not used)
impl Segment {
    pub fn winding(&self, x: f32, y: f32) -> i32 {
        let xb = Batch::splat(x);
        let yb = Batch::splat(y);
        let res = self.winding_batch(xb, yb);
        // Extract lane 0, cast to int
        unsafe {
            // result is float 1.0 or -1.0 or 0.0.
            // transmute to u32 gives garbage.
            // need to store to f32 array
            let mut buf = [0.0f32; 4];
            res.store(buf.as_mut_ptr());
            buf[0] as i32
        }
    }

    pub fn signed_pseudo_distance(&self, x: f32, y: f32) -> f32 {
        let xb = Batch::splat(x);
        let yb = Batch::splat(y);
        let res = self.min_dist_batch(xb, yb);
        unsafe {
            let mut buf = [0.0f32; 4];
            res.store(buf.as_mut_ptr());
            buf[0]
        }
    }
}
