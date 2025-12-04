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

/// Scales the coordinate system by a fixed factor.
#[derive(Copy, Clone)]
pub struct Scale<S> {
    /// The source surface to scale.
    pub source: S,
    /// Inverse scale factor in fixed-point format (16.16).
    pub inv_scale_fp: u32,
}

impl<S> Scale<S> {
    /// Creates a new `Scale` wrapper.
    ///
    /// # Arguments
    /// * `source` - The source surface.
    /// * `scale_factor` - The scaling factor (e.g., 2.0 for 2x zoom).
    #[inline]
    pub fn new(source: S, scale_factor: f64) -> Self {
        let inv_scale_fp = ((1.0 / scale_factor) * 65536.0) as u32;
        Self {
            source,
            inv_scale_fp,
        }
    }
}

impl<T, S> Surface<T> for Scale<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        let inv = Batch::<u32>::splat(self.inv_scale_fp);
        let lx = (x * inv) >> 16;
        let ly = (y * inv) >> 16;
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
