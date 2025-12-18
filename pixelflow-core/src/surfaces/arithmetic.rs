// pixelflow-core/src/surfaces/arithmetic.rs
//! Binary arithmetic surface combinators.

use crate::backend::BatchArithmetic;
use crate::batch::Batch;
use crate::traits::Manifold;
use core::fmt::Debug;

macro_rules! impl_binary_op {
    ($name:ident, $op:tt) => {
        /// Binary arithmetic combinator.
        #[derive(Copy, Clone, Debug)]
        pub struct $name<A, B>(pub A, pub B);

        impl<A, B, T, C> Manifold<T, C> for $name<A, B>
        where
            A: Manifold<T, C>,
            B: Manifold<T, C>,
            T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            Batch<T>: BatchArithmetic<T>,
        {
            #[inline(always)]
            fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
                let a = self.0.eval(x, y, z, w);
                let b = self.1.eval(x, y, z, w);
                a $op b
            }
        }
    };
}

impl_binary_op!(Add, +);
impl_binary_op!(Sub, -);
impl_binary_op!(Mul, *);
impl_binary_op!(Div, /);
