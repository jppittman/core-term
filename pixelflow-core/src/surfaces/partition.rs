use crate::batch::{Batch, LANES};
use crate::traits::Manifold;
use crate::backend::{SimdBatch};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::Debug;

/// Partitions evaluation across dynamically indexed surfaces.
///
/// Uses an indexer `Surface<u32, C>` that returns indices into a table of surfaces.
/// Enables O(1) lookup instead of O(log n) Select tree traversal.
///
/// Implements coherence optimization: if all lanes have the same index,
/// evaluate once instead of scalar evaluations.
#[derive(Clone)]
pub struct Partition<I, T, C = u32> {
    /// Indexer surface that returns indices for each pixel
    pub indexer: I,
    /// Table of surfaces indexed by the indexer
    pub surfaces: Vec<Arc<dyn Manifold<T, C>>>,
}

impl<I, T, C> Partition<I, T, C> {
    /// Creates a new Partition combinator.
    pub fn new(indexer: I, surfaces: Vec<Arc<dyn Manifold<T, C>>>) -> Self {
        Self { indexer, surfaces }
    }
}

impl<I, T, C> Manifold<T, C> for Partition<I, T, C>
where
    I: Manifold<u32, C>,
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        let indices: Batch<u32> = self.indexer.eval(x, y, z, w);

        // Extract indices for each lane
        let idx0 = indices.extract_lane(0) as usize;
        let idx1 = indices.extract_lane(1) as usize;

        // Coherence optimization: check if all lanes have the same index
        let all_same = idx0 == idx1
            && (LANES <= 2 || idx0 == indices.extract_lane(2) as usize)
            && (LANES <= 3 || idx0 == indices.extract_lane(3) as usize);

        if all_same && idx0 < self.surfaces.len() {
            // All lanes use the same surface - evaluate once
            return self.surfaces[idx0].eval(x, y, z, w);
        }

        // Mixed case - evaluate each lane through its surface
        let mut result_array = [T::default(); LANES];

        for i in 0..LANES {
            let idx = indices.extract_lane(i) as usize;
            if idx < self.surfaces.len() {
                let xi = Batch::<C>::splat(x.extract_lane(i));
                let yi = Batch::<C>::splat(y.extract_lane(i));
                let zi = Batch::<C>::splat(z.extract_lane(i));
                let wi = Batch::<C>::splat(w.extract_lane(i));
                let pixel_batch = self.surfaces[idx].eval(xi, yi, zi, wi);
                result_array[i] = pixel_batch.first();
            }
            // else: out of bounds, use default value (already set)
        }

        Batch::<T>::load(&result_array)
    }
}
