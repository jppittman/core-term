use crate::batch::Batch;
use crate::traits::Manifold;
use crate::backend::{BatchArithmetic, SimdBatch};
use core::fmt::Debug;

/// Offsets the coordinate system by a fixed amount.
#[derive(Copy, Clone)]
pub struct Offset<S> {
    /// The source surface to offset.
    pub source: S,
    /// Horizontal offset.
    pub dx: i32,
    /// Vertical offset.
    pub dy: i32,
}

// u32 implementation (Discrete)
impl<T, S> Manifold<T, u32> for Offset<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Manifold<T, u32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<T> {
        let ox = Batch::<u32>::splat(self.dx as u32);
        let oy = Batch::<u32>::splat(self.dy as u32);
        self.source.eval(x + ox, y + oy, z, w)
    }
}

// f32 implementation (Continuous)
impl<T, S> Manifold<T, f32> for Offset<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Manifold<T, f32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, z: Batch<f32>, w: Batch<f32>) -> Batch<T> {
        let ox = Batch::<f32>::splat(self.dx as f32);
        let oy = Batch::<f32>::splat(self.dy as f32);
        self.source.eval(x + ox, y + oy, z, w)
    }
}

/// Scales the coordinate system by fixed factors.
#[derive(Copy, Clone)]
pub struct Scale<S> {
    /// The source surface to scale.
    pub source: S,
    /// Inverse X scale factor in fixed-point format (16.16).
    pub inv_scale_x_fp: u32,
    /// Inverse Y scale factor in fixed-point format (16.16).
    pub inv_scale_y_fp: u32,
}

impl<S> Scale<S> {
    /// Creates a new `Scale` wrapper with separate X and Y factors.
    ///
    /// # Arguments
    /// * `source` - The source surface.
    /// * `scale_x` - Horizontal scaling factor (e.g., 3.0 for 3x width).
    /// * `scale_y` - Vertical scaling factor (e.g., 1.0 for no change).
    #[inline]
    pub fn new(source: S, scale_x: f64, scale_y: f64) -> Self {
        Self {
            source,
            inv_scale_x_fp: ((1.0 / scale_x) * 65536.0) as u32,
            inv_scale_y_fp: ((1.0 / scale_y) * 65536.0) as u32,
        }
    }

    /// Creates a new `Scale` wrapper with uniform scaling.
    #[inline]
    pub fn uniform(source: S, scale: f64) -> Self {
        Self::new(source, scale, scale)
    }
}

// u32 implementation (Discrete)
impl<T, S> Manifold<T, u32> for Scale<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Manifold<T, u32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<T> {
        let inv_x = Batch::<u32>::splat(self.inv_scale_x_fp);
        let inv_y = Batch::<u32>::splat(self.inv_scale_y_fp);
        let lx = (x * inv_x) >> 16;
        let ly = (y * inv_y) >> 16;
        self.source.eval(lx, ly, z, w)
    }
}

// f32 implementation (Continuous)
impl<T, S> Manifold<T, f32> for Scale<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Manifold<T, f32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, z: Batch<f32>, w: Batch<f32>) -> Batch<T> {
        // Convert fixed-point (16.16) back to float
        let scale_factor = 1.0 / 65536.0;
        let inv_x_f = (self.inv_scale_x_fp as f32) * scale_factor;
        let inv_y_f = (self.inv_scale_y_fp as f32) * scale_factor;

        let inv_x = Batch::<f32>::splat(inv_x_f);
        let inv_y = Batch::<f32>::splat(inv_y_f);

        // Apply scaling (multiply coordinate by inverse scale)
        self.source.eval(x * inv_x, y * inv_y, z, w)
    }
}

/// Skews the coordinate system horizontally based on the Y coordinate.
#[derive(Copy, Clone)]
pub struct Skew<S> {
    /// The source surface to skew.
    pub source: S,
    /// Shear factor (horizontal displacement per vertical pixel) in fixed point (24.8).
    pub shear: i32,
}

impl<S: Manifold<u8, u32>> Manifold<u8, u32> for Skew<S> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<u8> {
        let offset = (y * Batch::<u32>::splat(self.shear as u32)) >> 8;
        self.source.eval(x.saturating_sub(offset), y, z, w)
    }
}

/// Applies a transformation function to surface output.
///
/// Use this for gamma correction, color adjustments, or any per-pixel transform.
#[derive(Copy, Clone)]
pub struct Map<S, F> {
    /// The source surface.
    pub source: S,
    /// The transformation function.
    pub transform: F,
}

impl<S, F> Map<S, F> {
    /// Creates a new `Map` combinator.
    #[inline]
    pub fn new(source: S, transform: F) -> Self {
        Self { source, transform }
    }
}

impl<T, S, F, C> Manifold<T, C> for Map<S, F>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Manifold<T, C>,
    F: Fn(Batch<T>) -> Batch<T> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        (self.transform)(self.source.eval(x, y, z, w))
    }
}
