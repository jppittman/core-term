use crate::batch::{Batch, LANES};
use crate::traits::Surface;
use crate::pixel::Pixel;
use crate::backend::{BatchArithmetic, SimdBatch};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Partitions evaluation across dynamically indexed surfaces.
///
/// Uses an indexer `Surface<u32>` that returns indices into a table of surfaces.
/// Enables O(1) lookup instead of O(log n) Select tree traversal.
///
/// Implements coherence optimization: if all lanes have the same index,
/// evaluate once instead of scalar evaluations.
#[derive(Clone)]
pub struct Partition<I, P> {
    /// Indexer surface that returns indices for each pixel
    pub indexer: I,
    /// Table of surfaces indexed by the indexer
    pub surfaces: Vec<Arc<dyn Surface<P>>>,
}

impl<I, P> Partition<I, P> {
    /// Creates a new Partition combinator.
    pub fn new(indexer: I, surfaces: Vec<Arc<dyn Surface<P>>>) -> Self {
        Self { indexer, surfaces }
    }
}

impl<I, P> Surface<P> for Partition<I, P>
where
    I: Surface<u32>,
    P: Pixel,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let indices: Batch<u32> = self.indexer.eval(x, y);

        // Extract indices for each lane
        let idx0 = indices.extract_lane(0) as usize;
        let idx1 = indices.extract_lane(1) as usize;

        // Coherence optimization: check if all lanes have the same index
        let all_same = idx0 == idx1
            && (LANES <= 2 || idx0 == indices.extract_lane(2) as usize)
            && (LANES <= 3 || idx0 == indices.extract_lane(3) as usize);

        if all_same && idx0 < self.surfaces.len() {
            // All lanes use the same surface - evaluate once
            return self.surfaces[idx0].eval(x, y);
        }

        // Mixed case - evaluate each lane through its surface
        let mut result_array = [P::default(); 4];

        for i in 0..LANES {
            let idx = indices.extract_lane(i) as usize;
            if idx < self.surfaces.len() {
                let xi = Batch::<u32>::splat(x.extract_lane(i));
                let yi = Batch::<u32>::splat(y.extract_lane(i));
                let pixel_batch = self.surfaces[idx].eval(xi, yi);
                result_array[i] = pixel_batch.first();
            }
            // else: out of bounds, use default value (already set)
        }

        Batch::<P>::load(&result_array)
    }
}
