pub type Point = [f32; 2];

#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub p0: Point,
    pub p1: Point,
}

impl Line {
    // No translate
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
    pub fn winding(&self, x: f32, y: f32) -> i32 {
        match self {
            Segment::Line(l) => {
                if (l.p0[1] <= y && y < l.p1[1]) || (l.p1[1] <= y && y < l.p0[1]) {
                    let t = (y - l.p0[1]) / (l.p1[1] - l.p0[1]);
                    let x_int = l.p0[0] + t * (l.p1[0] - l.p0[0]);
                    if x < x_int {
                        if l.p0[1] < l.p1[1] { 1 } else { -1 }
                    } else { 0 }
                } else { 0 }
            }
            Segment::Quad(q) => {
                let a = q.p0[1] - 2.0 * q.p1[1] + q.p2[1];
                let b = 2.0 * (q.p1[1] - q.p0[1]);
                let c = q.p0[1] - y;
                let mut winding = 0;
                if a.abs() < 1e-6 {
                    if b.abs() > 1e-6 {
                        let t = -c / b;
                        if t >= 0.0 && t < 1.0 {
                            let xt = (1.0 - t).powi(2) * q.p0[0] + 2.0 * (1.0 - t) * t * q.p1[0] + t.powi(2) * q.p2[0];
                            if x < xt {
                                if b > 0.0 { winding += 1; } else { winding -= 1; }
                            }
                        }
                    }
                } else {
                    let disc = b * b - 4.0 * a * c;
                    if disc >= 0.0 {
                        let sqrt_disc = disc.sqrt();
                        let t1 = (-b - sqrt_disc) / (2.0 * a);
                        let t2 = (-b + sqrt_disc) / (2.0 * a);
                        for t in [t1, t2] {
                            if t >= 0.0 && t < 1.0 {
                                let xt = (1.0 - t).powi(2) * q.p0[0] + 2.0 * (1.0 - t) * t * q.p1[0] + t.powi(2) * q.p2[0];
                                if x < xt {
                                    let dy_dt = 2.0 * a * t + b;
                                    if dy_dt > 0.0 { winding += 1; } else { winding -= 1; }
                                }
                            }
                        }
                    }
                }
                winding
            }
        }
    }

    pub fn signed_pseudo_distance(&self, x: f32, y: f32) -> f32 {
        match self {
            Segment::Line(l) => {
                let dx = l.p1[0] - l.p0[0];
                let dy = l.p1[1] - l.p0[1];
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-6 { return 1000.0; }
                ((x - l.p0[0]) * -dy + (y - l.p0[1]) * dx) / len
            }
            Segment::Quad(q) => {
                let u = x * q.projection[0][0] + y * q.projection[0][1] + q.projection[0][2];
                let v = x * q.projection[1][0] + y * q.projection[1][1] + q.projection[1][2];
                let f = u * u - v;
                let du_dx = q.projection[0][0];
                let du_dy = q.projection[0][1];
                let dv_dx = q.projection[1][0];
                let dv_dy = q.projection[1][1];
                let df_dx = 2.0 * u * du_dx - dv_dx;
                let df_dy = 2.0 * u * du_dy - dv_dy;
                let grad_len = (df_dx * df_dx + df_dy * df_dy).sqrt();
                if grad_len < 1e-6 { 1000.0 } else { f / grad_len }
            }
        }
    }
}
