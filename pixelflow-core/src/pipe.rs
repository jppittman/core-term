use crate::batch::Batch;
use crate::backend::SimdBatch;
use alloc::boxed::Box;
use core::fmt::Debug;

/// A pure functional surface: `(x, y) -> T`.
///
/// Surfaces are evaluated using the platform's native SIMD backend.
/// This trait is object-safe, so `Box<dyn Surface<T>>` works.
pub trait Surface<T>: Send + Sync
where T: Copy + Debug + Default + Send + Sync + 'static
{
    /// Evaluates the surface at the specified batch of coordinates.
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T>;

    /// Evaluates the surface at a single coordinate (point sample).
    ///
    /// Splats the coordinate to all lanes, evaluates, extracts first lane.
    /// For anti-aliased sampling, use different x/y offsets per lane and
    /// a different reduction (average, etc).
    #[inline]
    fn eval_one(&self, x: u32, y: u32) -> T {
        self.eval(Batch::splat(x), Batch::splat(y)).first()
    }
}

// Implement Surface for Box<dyn Surface<T>>
impl<T> Surface<T> for Box<dyn Surface<T> + Send + Sync>
where T: Copy + Debug + Default + Send + Sync + 'static
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        (**self).eval(x, y)
    }
}
