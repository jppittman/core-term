// pixelflow-core/src/surfaces/math.rs
//! Unary math surface combinators.

use crate::backend::{BatchArithmetic, FloatBatchOps};
use crate::batch::Batch;
use crate::traits::Manifold;
use core::fmt::Debug;
use core::ops::Neg;

/// Unary negation combinator.
#[derive(Copy, Clone, Debug)]
pub struct Negate<S>(pub S);

impl<S, T, C> Manifold<T, C> for Negate<S>
where
    S: Manifold<T, C>,
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    Batch<T>: BatchArithmetic<T> + Neg<Output = Batch<T>>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        let val = self.0.eval(x, y, z, w);
        -val
    }
}

/// Square root combinator.
#[derive(Copy, Clone, Debug)]
pub struct Sqrt<S>(pub S);

impl<S, C> Manifold<f32, C> for Sqrt<S>
where
    S: Manifold<f32, C>,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    Batch<f32>: FloatBatchOps,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<f32> {
        let val = self.0.eval(x, y, z, w);
        val.sqrt()
    }
}

/// Absolute value combinator.
#[derive(Copy, Clone, Debug)]
pub struct Abs<S>(pub S);

impl<S, C> Manifold<f32, C> for Abs<S>
where
    S: Manifold<f32, C>,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    Batch<f32>: FloatBatchOps,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<f32> {
        let val = self.0.eval(x, y, z, w);
        val.abs()
    }
}
