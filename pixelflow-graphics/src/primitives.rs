use pixelflow_core::backend::{Backend, FloatBatchOps};
use pixelflow_core::batch::{Batch, BatchOps, Field, NativeBackend};
use pixelflow_core::traits::Surface;

/// A circle primitive defined by a center (cx, cy) and radius r.
#[derive(Copy, Clone, Debug)]
pub struct Circle {
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
}

impl Circle {
    pub fn new(cx: f32, cy: f32, r: f32) -> Self {
        Self { cx, cy, r }
    }
}

/// Signed Distance Field (SDF) implementation.
/// Returns negative values inside, positive outside.
impl Surface<f32> for Circle {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Field {
        let xf = NativeBackend::u32_to_f32(x);
        let yf = NativeBackend::u32_to_f32(y);

        // dist = sqrt((x-cx)^2 + (y-cy)^2) - r
        let dx = xf - self.cx;
        let dy = yf - self.cy;

        let dist_sq = dx * dx + dy * dy;
        let dist = dist_sq.sqrt();

        dist - self.r
    }
}

/// Mask implementation (returns 0 or !0).
/// Returns !0 (all 1s) if inside (dist <= 0), 0 otherwise.
impl Surface<u32> for Circle {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        let xf = NativeBackend::u32_to_f32(x);
        let yf = NativeBackend::u32_to_f32(y);

        let dx = xf - self.cx;
        let dy = yf - self.cy;

        // Optimization: avoid sqrt for mask
        let dist_sq = dx * dx + dy * dy;
        let r_sq = self.r * self.r;

        // inside if dist_sq <= r_sq
        // cmp_le returns a float-mask (NaN/bits for true), we transmute to u32
        let mask_f = dist_sq.cmp_le(Field::splat(r_sq));
        NativeBackend::transmute_f32_to_u32(mask_f)
    }
}

/// A floating-point rectangle primitive.
/// Defined by top-left (x, y) and dimensions (w, h).
#[derive(Copy, Clone, Debug)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
}

/// SDF implementation for Rect (2D signed distance).
/// Creates a Euclidean distance field.
impl Surface<f32> for Rect {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Field {
        let xf = NativeBackend::u32_to_f32(x);
        let yf = NativeBackend::u32_to_f32(y);

        // Center relative to the rect half-sizes
        let half_w = self.w * 0.5;
        let half_h = self.h * 0.5;
        let cx = self.x + half_w;
        let cy = self.y + half_h;

        let px = (xf - cx).abs();
        let py = (yf - cy).abs();

        // d = max(q.x, q.y) where q = abs(p) - size
        let qx = px - half_w;
        let qy = py - half_h;

        let zero = Field::splat(0.0);
        let qx_max_0 = qx.max(zero);
        let qy_max_0 = qy.max(zero);

        let len_max_d = (qx_max_0 * qx_max_0 + qy_max_0 * qy_max_0).sqrt();

        let max_d_x_y = qx.max(qy);
        let min_max_d_0 = max_d_x_y.min(zero);

        len_max_d + min_max_d_0
    }
}

/// Mask implementation for Rect.
impl Surface<u32> for Rect {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        let xf = NativeBackend::u32_to_f32(x);
        let yf = NativeBackend::u32_to_f32(y);

        let left = self.x;
        let right = self.x + self.w;
        let top = self.y;
        let bottom = self.y + self.h;

        // x >= left && x < right && y >= top && y < bottom
        // Note: Field comparison returns Field mask.
        let mask_f = xf.cmp_ge(Field::splat(left))
            & xf.cmp_lt(Field::splat(right))
            & yf.cmp_ge(Field::splat(top))
            & yf.cmp_lt(Field::splat(bottom));

        NativeBackend::transmute_f32_to_u32(mask_f)
    }
}
