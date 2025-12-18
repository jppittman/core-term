// pixelflow-core/src/surfaces/comparison.rs
//! Comparison surface combinators (returning u32 masks).

use crate::backend::BatchArithmetic;
use crate::batch::Batch;
use crate::bitwise::Bitwise;
use crate::traits::Manifold;
use core::fmt::Debug;
use core::marker::PhantomData;

macro_rules! impl_compare_op {
    ($name:ident, $method:ident) => {
        /// Comparison combinator.
        #[derive(Copy, Clone, Debug)]
        pub struct $name<A, B, T>(pub A, pub B, pub PhantomData<T>);

        impl<A, B, T, C> Manifold<u32, C> for $name<A, B, T>
        where
            A: Manifold<T, C>,
            B: Manifold<T, C>,
            T: Bitwise + Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            Batch<T>: BatchArithmetic<T>,
        {
            #[inline(always)]
            fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<u32> {
                let a = self.0.eval(x, y, z, w);
                let b = self.1.eval(x, y, z, w);
                let res = a.$method(b);
                T::batch_to_bits(res)
            }
        }
    };
}

impl_compare_op!(Eq, cmp_eq);
impl_compare_op!(Ne, cmp_ne);
impl_compare_op!(Lt, cmp_lt);
impl_compare_op!(Le, cmp_le);
impl_compare_op!(Gt, cmp_gt);
impl_compare_op!(Ge, cmp_ge);
