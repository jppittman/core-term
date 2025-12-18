use pixelflow_core::dsl::SurfaceExt;
use pixelflow_core::surfaces::coords::{X, Y};
use pixelflow_core::traits::{Manifold, Surface};

/// A circle primitive defined by center (cx, cy) and radius r.
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

/// Signed Distance Field (SDF) implementation using pure composition.
impl<C> Surface<C, f32> for Circle
where
    C: Manifold<C, f32>,
{
    #[inline(always)]
    fn eval(&self, x: C, y: C) -> f32 {
        // Translate to origin: (X - cx, Y - cy)
        let dx = X.eval(x, y) - self.cx;
        let dy = Y.eval(x, y) - self.cy;

        // dist = sqrt(dx² + dy²) - r
        let dist_sq = dx.clone() * dx + dy.clone() * dy;
        dist_sq.sqrt() - self.r
    }
}

/// Mask implementation using composition with bounding box optimization.
impl<C> Surface<C, u32> for Circle
where
    C: Manifold<C, u32>,
{
    #[inline(always)]
    fn eval(&self, x: C, y: C) -> u32 {
        // Bounding box check: |x - cx| < r && |y - cy| < r
        let dx = X.eval(x, y) - self.cx;
        let dy = Y.eval(x, y) - self.cy;

        let in_box = dx.abs().lt(self.r) & dy.abs().lt(self.r);

        // Only compute expensive sqrt if inside bounding box
        let dist_sq = dx.clone() * dx + dy.clone() * dy;
        let r_sq = self.r * self.r;
        let in_circle = dist_sq.le(r_sq);

        // Combine: in_box && in_circle
        (in_box & in_circle).eval(x, y)
    }
}

/// A floating-point rectangle primitive.
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

/// SDF implementation for Rect using composition.
impl<C> Surface<C, f32> for Rect
where
    C: Manifold<C, f32>,
{
    #[inline(always)]
    fn eval(&self, x: C, y: C) -> f32 {
        let half_w = self.w * 0.5;
        let half_h = self.h * 0.5;
        let cx = self.x + half_w;
        let cy = self.y + half_h;

        // Translate to center and take absolute distance
        let px = (X.eval(x, y) - cx).abs();
        let py = (Y.eval(x, y) - cy).abs();

        // Box SDF: max(abs(p) - size, 0)
        let qx = px - half_w;
        let qy = py - half_h;

        let qx_max_0 = qx.max(0.0);
        let qy_max_0 = qy.max(0.0);

        // Euclidean distance from corner
        let len_max_d = (qx_max_0.clone() * qx_max_0 + qy_max_0.clone() * qy_max_0).sqrt();

        // Interior distance
        let max_d_x_y = qx.max(qy);
        let min_max_d_0 = max_d_x_y.min(0.0);

        len_max_d + min_max_d_0
    }
}

/// Mask implementation for Rect using composition.
impl<C> Surface<C, u32> for Rect
where
    C: Manifold<C, u32>,
{
    #[inline(always)]
    fn eval(&self, x: C, y: C) -> u32 {
        let xf = X.eval(x, y);
        let yf = Y.eval(x, y);

        let left = self.x;
        let right = self.x + self.w;
        let top = self.y;
        let bottom = self.y + self.h;

        // x >= left && x < right && y >= top && y < bottom
        let mask = xf.ge(left) & xf.lt(right) & yf.ge(top) & yf.lt(bottom);
        mask.eval(x, y)
    }
}
