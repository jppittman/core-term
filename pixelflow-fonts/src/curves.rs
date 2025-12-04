use pixelflow_core::backend::{Backend, BatchArithmetic, FloatBatchOps, SimdBatch};
use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::geometry::Mat3;

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

#[derive(Clone, Copy, Debug)]
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
                    if l.p0[1] < l.p1[1] {
                        1
                    } else {
                        -1
                    }
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
                        if (0.0..1.0).contains(&t) {
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
                            if (0.0..1.0).contains(&t) {
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
                let len_sq = dx * dx + dy * dy;
                let len = len_sq.sqrt();
                if len < 1e-6 {
                    return 1000.0;
                }
                // Check if point projects onto the segment (t in [0, 1])
                let t = ((x - l.p0[0]) * dx + (y - l.p0[1]) * dy) / len_sq;
                if !(0.0..=1.0).contains(&t) {
                    return 1000.0;
                }
                ((x - l.p0[0]) * -dy + (y - l.p0[1]) * dx) / len
            }
            Segment::Quad(q) => {
                let m = q.projection.m;
                let u = x * m[0][0] + y * m[0][1] + m[0][2];
                let v = x * m[1][0] + y * m[1][1] + m[1][2];

                // Check if projection falls outside the curve segment [0, 1].
                // If so, use endpoint distance instead of implicit distance.
                if !(0.0..=1.0).contains(&u) {
                    let d0_sq = (x - q.p0[0]).powi(2) + (y - q.p0[1]).powi(2);
                    let d2_sq = (x - q.p2[0]).powi(2) + (y - q.p2[1]).powi(2);
                    return d0_sq.min(d2_sq).sqrt();
                }

                let f = u * u - v;

                let du_dx = m[0][0];
                let du_dy = m[0][1];
                let dv_dx = m[1][0];
                let dv_dy = m[1][1];

                let df_dx = 2.0 * u * du_dx - dv_dx;
                let df_dy = 2.0 * u * du_dy - dv_dy;
                let grad_len = (df_dx * df_dx + df_dy * df_dy).sqrt();

                if grad_len < 1e-6 {
                    1000.0
                } else {
                    f / grad_len
                }
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
                let len_sq_batch = Batch::<f32>::splat(len_sq);
                let len_batch = Batch::<f32>::splat(len);
                let zero = Batch::<f32>::splat(0.0);
                let one = Batch::<f32>::splat(1.0);

                // Compute parameter t for closest point on infinite line
                // t = dot(p - p0, p1 - p0) / |p1 - p0|^2
                let t = ((x - p0x) * dx_batch + (y - p0y) * dy_batch) / len_sq_batch;

                // Check if t is in [0, 1] (point projects onto segment)
                let in_segment = t.cmp_ge(zero) & t.cmp_le(one);

                // Perpendicular (signed) distance to infinite line
                let perp_dist = ((x - p0x) * (zero - dy_batch) + (y - p0y) * dx_batch) / len_batch;

                // Return perpendicular distance only if in segment, else large distance
                let large_dist = Batch::<f32>::splat(1000.0);
                let mask = NativeBackend::transmute_f32_to_u32(in_segment);
                let perp_u32 = NativeBackend::transmute_f32_to_u32(perp_dist);
                let large_u32 = NativeBackend::transmute_f32_to_u32(large_dist);
                NativeBackend::transmute_u32_to_f32(mask.select(perp_u32, large_u32))
            }
            Segment::Quad(q) => {
                let m = q.projection.m;

                let u = x * Batch::<f32>::splat(m[0][0])
                    + y * Batch::<f32>::splat(m[0][1])
                    + Batch::<f32>::splat(m[0][2]);

                let v = x * Batch::<f32>::splat(m[1][0])
                    + y * Batch::<f32>::splat(m[1][1])
                    + Batch::<f32>::splat(m[1][2]);

                let zero = Batch::<f32>::splat(0.0);
                let one = Batch::<f32>::splat(1.0);

                // Check if u is outside [0, 1] - use endpoint distance instead
                let u_in_range = u.cmp_ge(zero) & u.cmp_le(one);

                // Compute endpoint distances
                let p0x = Batch::<f32>::splat(q.p0[0]);
                let p0y = Batch::<f32>::splat(q.p0[1]);
                let p2x = Batch::<f32>::splat(q.p2[0]);
                let p2y = Batch::<f32>::splat(q.p2[1]);

                let dx0 = x - p0x;
                let dy0 = y - p0y;
                let d0_sq = dx0 * dx0 + dy0 * dy0;

                let dx2 = x - p2x;
                let dy2 = y - p2y;
                let d2_sq = dx2 * dx2 + dy2 * dy2;

                // Min of endpoint distances
                let d0_sq_u32 = NativeBackend::transmute_f32_to_u32(d0_sq);
                let d2_sq_u32 = NativeBackend::transmute_f32_to_u32(d2_sq);
                let d0_lt_d2 = d0_sq.cmp_lt(d2_sq);
                let d0_lt_d2_mask = NativeBackend::transmute_f32_to_u32(d0_lt_d2);
                let min_endpoint_sq =
                    NativeBackend::transmute_u32_to_f32(d0_lt_d2_mask.select(d0_sq_u32, d2_sq_u32));
                let endpoint_dist = min_endpoint_sq.sqrt();

                // Compute implicit distance
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

                let implicit_dist = f / grad_len;
                let large_dist = Batch::<f32>::splat(1000.0);

                // If gradient is zero, use large distance for implicit
                let zero_grad_mask = NativeBackend::transmute_f32_to_u32(zero_grad);
                let large_u32 = NativeBackend::transmute_f32_to_u32(large_dist);
                let implicit_u32 = NativeBackend::transmute_f32_to_u32(implicit_dist);
                let safe_implicit = NativeBackend::transmute_u32_to_f32(
                    zero_grad_mask.select(large_u32, implicit_u32),
                );

                // Select: if u in range use implicit, else use endpoint distance
                let in_range_mask = NativeBackend::transmute_f32_to_u32(u_in_range);
                let safe_implicit_u32 = NativeBackend::transmute_f32_to_u32(safe_implicit);
                let endpoint_u32 = NativeBackend::transmute_f32_to_u32(endpoint_dist);
                NativeBackend::transmute_u32_to_f32(
                    in_range_mask.select(safe_implicit_u32, endpoint_u32),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::backend::SimdBatch;

    /// A CCW unit square from (0,0) to (1,1).
    fn ccw_unit_square() -> [Segment; 4] {
        [
            Segment::Line(Line {
                p0: [0.0, 0.0],
                p1: [1.0, 0.0],
            }), // bottom
            Segment::Line(Line {
                p0: [1.0, 0.0],
                p1: [1.0, 1.0],
            }), // right
            Segment::Line(Line {
                p0: [1.0, 1.0],
                p1: [0.0, 1.0],
            }), // top
            Segment::Line(Line {
                p0: [0.0, 1.0],
                p1: [0.0, 0.0],
            }), // left
        ]
    }

    fn total_winding(segments: &[Segment], x: f32, y: f32) -> i32 {
        segments.iter().map(|s| s.winding(x, y)).sum()
    }

    fn total_winding_batch(segments: &[Segment], x: f32, y: f32) -> i32 {
        let x_batch = Batch::<f32>::splat(x);
        let y_batch = Batch::<f32>::splat(y);
        let mut winding = Batch::<u32>::splat(0);
        for s in segments {
            winding = winding + s.winding_batch(x_batch, y_batch);
        }
        winding.first() as i32
    }

    // ========================================================================
    // Winding Number Tests
    // ========================================================================

    #[test]
    fn returns_nonzero_winding_inside_ccw_square() {
        let square = ccw_unit_square();
        let w = total_winding(&square, 0.5, 0.5);
        assert_ne!(
            w, 0,
            "Point (0.5, 0.5) should be inside square, winding={}",
            w
        );
    }

    #[test]
    fn returns_zero_winding_outside_ccw_square() {
        let square = ccw_unit_square();

        let w_left = total_winding(&square, -1.0, 0.5);
        assert_eq!(
            w_left, 0,
            "Point (-1, 0.5) should be outside, winding={}",
            w_left
        );

        let w_right = total_winding(&square, 2.0, 0.5);
        assert_eq!(
            w_right, 0,
            "Point (2, 0.5) should be outside, winding={}",
            w_right
        );

        let w_above = total_winding(&square, 0.5, 2.0);
        assert_eq!(
            w_above, 0,
            "Point (0.5, 2) should be outside, winding={}",
            w_above
        );

        let w_below = total_winding(&square, 0.5, -1.0);
        assert_eq!(
            w_below, 0,
            "Point (0.5, -1) should be outside, winding={}",
            w_below
        );
    }

    #[test]
    fn returns_zero_winding_for_horizontal_line() {
        let horiz = Segment::Line(Line {
            p0: [0.0, 5.0],
            p1: [10.0, 5.0],
        });
        let w = horiz.winding(5.0, 5.0);
        assert_eq!(w, 0, "Horizontal line should never contribute winding");
    }

    #[test]
    fn matches_scalar_winding_for_ccw_square() {
        let square = ccw_unit_square();

        let test_points = [
            (0.5, 0.5),  // inside
            (-1.0, 0.5), // outside left
            (2.0, 0.5),  // outside right
            (0.5, 2.0),  // outside above
            (0.5, -1.0), // outside below
            (0.1, 0.1),  // inside near corner
            (0.9, 0.9),  // inside near corner
        ];

        for (x, y) in test_points {
            let scalar = total_winding(&square, x, y);
            let batch = total_winding_batch(&square, x, y);
            assert_eq!(
                scalar, batch,
                "Winding mismatch at ({}, {}): scalar={}, batch={}",
                x, y, scalar, batch
            );
        }
    }

    #[test]
    fn handles_horizontal_line_in_batch_without_nan() {
        let horiz = Segment::Line(Line {
            p0: [0.0, 5.0],
            p1: [10.0, 5.0],
        });
        let x = Batch::<f32>::splat(5.0);
        let y = Batch::<f32>::splat(5.0);
        let w = horiz.winding_batch(x, y);
        let result = w.first();
        assert!(
            !f32::from_bits(result).is_nan(),
            "Horizontal line winding should not produce NaN"
        );
        assert_eq!(result, 0, "Horizontal line should contribute 0 winding");
    }

    // ========================================================================
    // Distance Tests
    // ========================================================================

    #[test]
    fn returns_positive_distance_on_positive_side_of_line() {
        let line = Segment::Line(Line {
            p0: [0.0, 0.0],
            p1: [10.0, 0.0],
        });
        let d = line.min_distance(5.0, 1.0);
        assert!(
            d > 0.0,
            "Point above horizontal line should have positive distance, got {}",
            d
        );
    }

    #[test]
    fn returns_negative_distance_on_negative_side_of_line() {
        let line = Segment::Line(Line {
            p0: [0.0, 0.0],
            p1: [10.0, 0.0],
        });
        let d = line.min_distance(5.0, -1.0);
        assert!(
            d < 0.0,
            "Point below horizontal line should have negative distance, got {}",
            d
        );
    }

    #[test]
    fn returns_large_distance_for_degenerate_line() {
        let degen = Segment::Line(Line {
            p0: [5.0, 5.0],
            p1: [5.0, 5.0],
        });
        let d = degen.min_distance(0.0, 0.0);
        assert!(
            d > 100.0,
            "Degenerate line should return large distance, got {}",
            d
        );
    }

    #[test]
    fn matches_scalar_distance_for_lines() {
        // Line from (0,0) to (10,5)
        let line = Segment::Line(Line {
            p0: [0.0, 0.0],
            p1: [10.0, 5.0],
        });

        // Only test points that project onto the segment (t in [0, 1])
        // For this line: direction is (10, 5), length^2 = 125
        // t = dot(p - p0, dir) / 125
        let test_points = [
            (5.0, 2.5), // on line, t = 0.5
            (5.0, 3.5), // above line middle, t ~ 0.5
            (5.0, 1.5), // below line middle, t ~ 0.5
            (2.0, 1.0), // on line, t = 0.2
            (8.0, 4.0), // on line, t = 0.8
        ];

        for (x, y) in test_points {
            let scalar = line.min_distance(x, y);
            let x_batch = Batch::<f32>::splat(x);
            let y_batch = Batch::<f32>::splat(y);
            let batch = line.min_distance_batch(x_batch, y_batch).first();

            let diff = (scalar - batch).abs();
            assert!(
                diff < 1e-5,
                "Distance mismatch at ({}, {}): scalar={}, batch={}",
                x,
                y,
                scalar,
                batch
            );
        }
    }

    #[test]
    fn handles_horizontal_line_distance_without_nan() {
        let horiz = Segment::Line(Line {
            p0: [0.0, 5.0],
            p1: [10.0, 5.0],
        });
        let x = Batch::<f32>::splat(5.0);
        let y = Batch::<f32>::splat(6.0);
        let d = horiz.min_distance_batch(x, y).first();
        assert!(!d.is_nan(), "Distance should not be NaN");
        assert!((d - 1.0).abs() < 1e-5, "Distance should be ~1.0, got {}", d);
    }

    // ========================================================================
    // Quadratic Tests
    // ========================================================================

    #[test]
    fn creates_valid_quadratic_for_non_degenerate_curve() {
        let q = Quadratic::try_new([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]);
        assert!(q.is_some(), "Non-degenerate quadratic should be created");
    }

    #[test]
    fn rejects_degenerate_collinear_quadratic() {
        let q = Quadratic::try_new([0.0, 0.0], [0.5, 0.5], [1.0, 1.0]);
        assert!(q.is_none(), "Collinear points should be rejected");
    }

    #[test]
    fn matches_scalar_winding_for_quadratic() {
        let q = Quadratic::try_new([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]).unwrap();
        let seg = Segment::Quad(q);

        let test_points = [
            (0.5, 0.3),  // inside curve
            (0.5, 0.8),  // above curve
            (-1.0, 0.5), // left
            (2.0, 0.5),  // right
        ];

        for (x, y) in test_points {
            let scalar = seg.winding(x, y);
            let x_batch = Batch::<f32>::splat(x);
            let y_batch = Batch::<f32>::splat(y);
            let batch = seg.winding_batch(x_batch, y_batch).first() as i32;

            assert_eq!(
                scalar, batch,
                "Quad winding mismatch at ({}, {}): scalar={}, batch={}",
                x, y, scalar, batch
            );
        }
    }

    #[test]
    fn matches_scalar_distance_for_quadratic() {
        let q = Quadratic::try_new([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]).unwrap();
        let seg = Segment::Quad(q);

        let test_points = [(0.5, 0.5), (0.5, 0.0), (0.0, 0.0), (1.0, 0.0)];

        for (x, y) in test_points {
            let scalar = seg.min_distance(x, y);
            let x_batch = Batch::<f32>::splat(x);
            let y_batch = Batch::<f32>::splat(y);
            let batch = seg.min_distance_batch(x_batch, y_batch).first();

            let diff = (scalar - batch).abs();
            assert!(
                diff < 1e-5,
                "Quad distance mismatch at ({}, {}): scalar={}, batch={}",
                x,
                y,
                scalar,
                batch
            );
        }
    }

    // ========================================================================
    // Bug Regression Tests
    // ========================================================================

    #[test]
    fn quadratic_distance_far_outside_segment_returns_large() {
        // Bug: Loop-Blinn implicit distance returns small values for points
        // far outside the curve segment's parameter range [0, 1].
        //
        // This causes spurious glyph coverage - pixels far from glyphs
        // get non-zero alpha because the implicit equation u² - v doesn't
        // care about parameter bounds.
        let q = Quadratic::try_new([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]).unwrap();
        let seg = Segment::Quad(q);

        // Point 10 units above the curve apex (which is at y=1.0)
        // This point should NOT be considered close to the curve
        let dist = seg.min_distance(0.5, 10.0);

        // The actual distance to the curve is at least 9.0 (10 - 1 = 9)
        // With the bug, this returns a small value like ~0.4
        assert!(
            dist > 5.0,
            "Point 10 units above curve should have large distance, got {}",
            dist
        );
    }

    #[test]
    fn quadratic_distance_batch_far_outside_returns_large() {
        // Same test for the batch version
        let q = Quadratic::try_new([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]).unwrap();
        let seg = Segment::Quad(q);

        let x = Batch::<f32>::splat(0.5);
        let y = Batch::<f32>::splat(10.0);
        let dist = seg.min_distance_batch(x, y).first();

        assert!(
            dist > 5.0,
            "Batch: Point 10 units above curve should have large distance, got {}",
            dist
        );
    }
}
