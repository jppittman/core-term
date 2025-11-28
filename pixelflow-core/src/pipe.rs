use crate::batch::Batch;
use crate::simd::Simd;

/// A pure functional surface: `(x, y) -> T`.
///
/// This is a "Lazy Array". It doesn't own memory; it describes how to
/// compute a value at any coordinate.
pub trait Surface<V: Simd, T: Copy>: Copy {
    /// Evaluates the surface at the specified batch of coordinates.
    ///
    /// # Parameters
    /// * `x` - A batch of X coordinates (always u32).
    /// * `y` - A batch of Y coordinates (always u32).
    ///
    /// # Returns
    /// * A batch of computed values.
    fn eval(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<T>;
}

// Blanket impl allows closures to act as Surfaces
impl<V, F, T> Surface<V, T> for F
where
    V: Simd,
    T: Copy,
    F: Fn(V::Cast<u32>, V::Cast<u32>) -> V::Cast<T> + Copy + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<T> {
        self(x, y)
    }
}

// Constant Surface (Optimization for solid colors)
#[derive(Copy, Clone)]
pub struct Constant<T: Copy>(pub T);

impl<V: Simd, T: Copy> Surface<V, T> for Constant<T> {
    #[inline(always)]
    fn eval(&self, _x: V::Cast<u32>, _y: V::Cast<u32>) -> V::Cast<T> {
        <V::Cast<T> as Simd>::splat(self.0)
    }
}

// Legacy Compatibility: Batch is a Surface for itself (4 lanes).
impl<T: Copy> Surface<Batch<T>, T> for Batch<T>
where
    Batch<T>: Simd,
{
    #[inline(always)]
    fn eval(
        &self,
        _x: <Batch<T> as Simd>::Cast<u32>,
        _y: <Batch<T> as Simd>::Cast<u32>,
    ) -> <Batch<T> as Simd>::Cast<T> {
        *self
    }
}
