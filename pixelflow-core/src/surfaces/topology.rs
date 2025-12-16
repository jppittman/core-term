use crate::batch::Batch;
use crate::traits::{Manifold, Surface, Volume};
use core::fmt::Debug;

/// Explicitly promotes a Surface to a Manifold (Infinite Hyperprism).
///
/// `z` and `w` are ignored. Returns `source.eval(x, y)`.
#[derive(Copy, Clone)]
pub struct Extrude<S>(pub S);

impl<T, C, S> Manifold<T, C> for Extrude<S>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Surface<T, C>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, _z: Batch<C>, _w: Batch<C>) -> Batch<T> {
        self.0.eval(x, y)
    }
}

/// Explicitly promotes a Volume to a Manifold (Infinite Hyperprism).
///
/// `w` is ignored. Returns `source.eval(x, y, z)`.
#[derive(Copy, Clone)]
pub struct ExtrudeVolume<V>(pub V);

impl<T, C, V> Manifold<T, C> for ExtrudeVolume<V>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    V: Volume<T, C>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, _w: Batch<C>) -> Batch<T> {
        self.0.eval(x, y, z)
    }
}
