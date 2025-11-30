use crate::Batch;

/// A pure functional surface: `(x, y) -> T`.
///
/// This is a "Lazy Array". It describes how to compute a value at any coordinate.
/// The output type T must be Copy (for SIMD Batch operations), but the
/// Surface itself can own memory (like a grid buffer).
pub trait Surface<T: Copy> {
    /// Evaluates the surface at the specified batch of coordinates.
    ///
    /// # Parameters
    /// * `x` - A batch of X coordinates.
    /// * `y` - A batch of Y coordinates.
    ///
    /// # Returns
    /// * A batch of computed values.
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T>;
}

// Blanket impl allows closures to act as Surfaces
impl<F, T: Copy> Surface<T> for F
where
    F: Fn(Batch<u32>, Batch<u32>) -> Batch<T> + Copy + Send + Sync,
{
    /// Evaluates the closure as a surface.
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        self(x, y)
    }
}

// THE UNIFICATION: A Batch is a constant Surface.
// This allows using constant values (like Batch::splat(color)) directly in the graph.
impl<T: Copy> Surface<T> for Batch<T> {
    /// Evaluates the constant surface (returns the batch itself).
    #[inline(always)]
    fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<T> {
        *self
    }
}

// A raw u32 value acts as an infinite constant Surface.
// This allows passing colors directly: `fg_color.over(bg_color)`
impl Surface<u32> for u32 {
    #[inline(always)]
    fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u32> {
        Batch::splat(*self)
    }
}

// Box<dyn Surface> delegates to the vtable.
// This allows combinators to wrap boxed trait objects.
impl<T: Copy> Surface<T> for alloc::boxed::Box<dyn Surface<T> + Send + Sync> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        (**self).eval(x, y)
    }
}
