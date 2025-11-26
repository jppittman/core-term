use crate::Batch;

/// A pure functional surface: `(x, y) -> T`.
///
/// This is a "Lazy Array". It doesn't own memory; it describes how to
/// compute a value at any coordinate.
pub trait Surface<T: Copy>: Copy {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T>;
}

// Blanket impl allows closures to act as Surfaces
impl<F, T: Copy> Surface<T> for F
where
    F: Fn(Batch<u32>, Batch<u32>) -> Batch<T> + Copy + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        self(x, y)
    }
}

// THE UNIFICATION: A Batch is a constant Surface.
// This allows using constant values (like Batch::splat(color)) directly in the graph.
impl<T: Copy> Surface<T> for Batch<T> {
    #[inline(always)]
    fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<T> {
        *self
    }
}
