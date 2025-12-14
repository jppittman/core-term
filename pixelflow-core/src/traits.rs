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

/// A pure functional volume: `(x, y, z) -> T`.
pub trait Volume<T, C = u32>: Send + Sync
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Evaluate the volume at the given coordinates.
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>) -> Batch<T>;
}

/// A pure functional manifold: `(x, y, z, w) -> T`.
pub trait Manifold<T, C = u32>: Send + Sync
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Evaluate the manifold at the given coordinates.
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T>;
}

// ============================================================================
// Trait Object Adapters (Dyn Manifold)
// ============================================================================

// Forward Box<dyn Manifold>
impl<T, C> Manifold<T, C> for Box<dyn Manifold<T, C> + Send + Sync>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        (**self).eval(x, y, z, w)
    }
}

// Forward Arc<dyn Manifold>
impl<T, C> Manifold<T, C> for Arc<dyn Manifold<T, C> + Send + Sync>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        (**self).eval(x, y, z, w)
    }
}

// Extrude Box<dyn Surface> (Implicitly Manifold)
impl<T, C> Manifold<T, C> for Box<dyn Surface<T, C> + Send + Sync>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, _z: Batch<C>, _w: Batch<C>) -> Batch<T> {
        (**self).eval(x, y)
    }
}

// Extrude Arc<dyn Surface> (Implicitly Manifold)
impl<T, C> Manifold<T, C> for Arc<dyn Surface<T, C> + Send + Sync>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, _z: Batch<C>, _w: Batch<C>) -> Batch<T> {
        (**self).eval(x, y)
    }
}

// ============================================================================
// Dimensional Collapse (Blanket Impls)
// ============================================================================

// Manifold -> Volume (w = 0)
impl<T, C, M> Volume<T, C> for M
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    M: Manifold<T, C>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>) -> Batch<T> {
        let w = Batch::<C>::splat(C::default());
        self.eval(x, y, z, w)
    }
}

// Volume -> Surface (z = 0)
impl<T, C, V> Surface<T, C> for V
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    V: Volume<T, C>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>) -> Batch<T> {
        let z = Batch::<C>::splat(C::default());
        self.eval(x, y, z)
    }
}
