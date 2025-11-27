use crate::Batch;

/// A pure functional surface: `(x, y) -> T`.
///
/// This is a "Lazy Array". It doesn't own memory; it describes how to
/// compute a value at any coordinate.
pub trait Surface<T: Copy>: Copy {
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
