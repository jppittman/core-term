use crate::batch::Batch;
use crate::traits::Surface;
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

impl<T, S> Surface<T> for Offset<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        let ox = Batch::<u32>::splat(self.dx as u32);
        let oy = Batch::<u32>::splat(self.dy as u32);
        self.source.eval(x + ox, y + oy)
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

impl<T, S> Surface<T> for Scale<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        let inv_x = Batch::<u32>::splat(self.inv_scale_x_fp);
        let inv_y = Batch::<u32>::splat(self.inv_scale_y_fp);
        let lx = (x * inv_x) >> 16;
        let ly = (y * inv_y) >> 16;
        self.source.eval(lx, ly)
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

impl<S: Surface<u8>> Surface<u8> for Skew<S> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let offset = (y * Batch::<u32>::splat(self.shear as u32)) >> 8;
        self.source.eval(x.saturating_sub(offset), y)
    }
}
