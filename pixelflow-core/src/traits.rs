use crate::backend::SimdBatch;
use crate::batch::Batch;
use alloc::boxed::Box;
use alloc::sync::Arc;
use core::fmt::Debug;

/// A pure functional surface: `(x, y) -> T`.
///
/// Surfaces are evaluated using the platform's native SIMD backend.
/// This trait is object-safe, so `Box<dyn Surface<T>>` works.
pub trait Surface<T>: Send + Sync
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Evaluate the surface at the given coordinates.
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T>;

    /// Evaluates the surface at a single coordinate (point sample).
    ///
    /// Splats the coordinate to all lanes, evaluates, extracts first lane.
    /// For anti-aliased sampling, use different x/y offsets per lane and
    /// a different reduction (average, etc).
    #[inline]
    fn eval_one(&self, x: u32, y: u32) -> T {
        self.eval(Batch::<u32>::splat(x), Batch::<u32>::splat(y))
            .first()
    }

    /// Wraps this surface in an Arc trait object, making it cheap to clone and share.
    ///
    /// This essentially "compiles" the surface into a reusable shader handle.
    fn shader(self) -> Arc<dyn Surface<T> + Send + Sync>
    where
        Self: Sized + 'static,
    {
        Arc::new(self)
    }
}

// Implement Surface for Box<dyn Surface<T>>
impl<T> Surface<T> for Box<dyn Surface<T> + Send + Sync>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        (**self).eval(x, y)
    }
}

// Implement Surface for Arc<dyn Surface<T>>
impl<T> Surface<T> for Arc<dyn Surface<T> + Send + Sync>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        (**self).eval(x, y)
    }
}

/// A pure functional volume: `(x, y, z) -> T`.
pub trait Volume<T>: Send + Sync
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Evaluate the volume at the given coordinates.
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>) -> Batch<T>;
}
