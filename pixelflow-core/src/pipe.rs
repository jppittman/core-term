use crate::backend::{Backend, BatchArithmetic};
use core::fmt::Debug;

/// A pure functional surface: `(x, y) -> T`.
pub trait Surface<T>: Send + Sync
where T: Copy + Debug + Default + Send + Sync + 'static
{
    /// Evaluates the surface at the specified batch of coordinates.
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<T>
    where B::Batch<u32>: BatchArithmetic<u32>;
}
