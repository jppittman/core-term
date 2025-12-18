// pixelflow-core/src/surfaces/max.rs

use crate::backend::BatchArithmetic;
use crate::batch::Batch;
use crate::traits::Manifold;
use core::fmt::Debug;

/// Computes the maximum value of two surfaces.
#[derive(Copy, Clone)]
pub struct Max<A, B>(pub A, pub B);

macro_rules! impl_max_manifold {
    ($($t:ty),*) => {
        $(
            impl<A, B, C> Manifold<$t, C> for Max<A, B>
            where
                A: Manifold<$t, C>,
                B: Manifold<$t, C>,
                C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            {
                #[inline(always)]
                fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<$t> {
                    let a = self.0.eval(x, y, z, w);
                    let b = self.1.eval(x, y, z, w);
                    <Batch<$t> as BatchArithmetic<$t>>::max(a, b)
                }
            }
        )*
    }
}

impl_max_manifold!(u32, u8, f32);
