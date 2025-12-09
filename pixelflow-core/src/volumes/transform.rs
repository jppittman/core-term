use crate::batch::Batch;
use crate::traits::Volume;
use crate::backend::{BatchArithmetic, SimdBatch};
use core::fmt::Debug;

/// Offsets the 3D coordinate system by a fixed amount.
///
/// Effectively moves the volume content by `(dx, dy, dz)`.
/// If the volume is defined at the origin, wrapping it in `Translate`
/// with `(dx, dy, dz)` will make it appear centered at `(dx, dy, dz)`.
///
/// Mathematically: `vol_new(p) = vol_orig(p - offset)`
#[derive(Copy, Clone)]
pub struct Translate<V, C> {
    /// The source volume.
    pub source: V,
    /// X offset.
    pub dx: C,
    /// Y offset.
    pub dy: C,
    /// Z offset.
    pub dz: C,
}

impl<V, C> Translate<V, C> {
    /// Creates a new `Translate` combinator.
    #[inline]
    pub fn new(source: V, dx: C, dy: C, dz: C) -> Self {
        Self { source, dx, dy, dz }
    }
}

// f32 implementation (Continuous)
impl<T, V> Volume<T, f32> for Translate<V, f32>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    V: Volume<T, f32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, z: Batch<f32>) -> Batch<T> {
        // To move the object TO (dx, dy, dz), we must subtract the offset
        // from the input coordinates before passing them to the source volume
        // (which expects coordinates relative to its local origin).
        let ox = Batch::<f32>::splat(self.dx);
        let oy = Batch::<f32>::splat(self.dy);
        let oz = Batch::<f32>::splat(self.dz);
        self.source.eval(x - ox, y - oy, z - oz)
    }
}

// u32 implementation (Discrete/Voxel)
impl<T, V> Volume<T, u32> for Translate<V, u32>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    V: Volume<T, u32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>) -> Batch<T> {
        let ox = Batch::<u32>::splat(self.dx);
        let oy = Batch::<u32>::splat(self.dy);
        let oz = Batch::<u32>::splat(self.dz);
        // Using wrapping math or saturating? Standard subtraction is typical for translation.
        // Assuming wrapping for u32 to match simple math ops.
        self.source.eval(x - ox, y - oy, z - oz)
    }
}
