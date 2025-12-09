use crate::backend::SimdBatch;
use crate::batch::Batch;
use alloc::boxed::Box;
use alloc::sync::Arc;
use core::fmt::Debug;

/// A pure functional surface: `(x, y) -> T`.
///
/// Surfaces are evaluated using the platform's native SIMD backend.
/// This trait is object-safe, so `Box<dyn Surface<T>>` works.
pub trait Surface<T, C = u32>: Send + Sync
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Evaluate the surface at the given coordinates.
    fn eval(&self, x: Batch<C>, y: Batch<C>) -> Batch<T>;

    /// Evaluates the surface at a single coordinate (point sample).
    ///
    /// Splats the coordinate to all lanes, evaluates, extracts first lane.
    /// For anti-aliased sampling, use different x/y offsets per lane and
    /// a different reduction (average, etc).
    #[inline]
    fn eval_one(&self, x: C, y: C) -> T {
        self.eval(Batch::<C>::splat(x), Batch::<C>::splat(y))
            .first()
    }

    /// Wraps this surface in an Arc trait object, making it cheap to clone and share.
    ///
    /// This essentially "compiles" the surface into a reusable shader handle.
    fn shader(self) -> Arc<dyn Surface<T, C> + Send + Sync>
    where
        Self: Sized + 'static,
    {
        Arc::new(self)
    }
}

// Implement Surface for Box<dyn Surface> (trait objects only, to avoid conflicts)
impl<T, C> Surface<T, C> for Box<dyn Surface<T, C> + Send + Sync>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>) -> Batch<T> {
        (**self).eval(x, y)
    }
}

// Implement Surface for Arc<S> where S: Surface (covers Arc<ConcreteType> and Arc<dyn Surface>)
impl<T, C, S> Surface<T, C> for Arc<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T, C> + ?Sized,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>) -> Batch<T> {
        (**self).eval(x, y)
    }
}

/// A pure functional volume: `(x, y, z) -> T`.
pub trait Volume<T, C = u32>: Send + Sync
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Evaluate the volume at the given coordinates.
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>) -> Batch<T>;

    /// Create a Volume from a closure (convenience constructor).
    ///
    /// # Example
    /// ```
    /// use pixelflow_core::traits::Volume;
    /// use pixelflow_core::batch::Batch;
    ///
    /// let vol = Volume::from_fn(|x, y, z| x ^ y ^ z);
    /// ```
    fn from_fn<F>(func: F) -> crate::volumes::FnVolume<F, T, C>
    where
        F: Fn(Batch<C>, Batch<C>, Batch<C>) -> Batch<T> + Send + Sync,
    {
        crate::volumes::FnVolume::new(func)
    }
}
