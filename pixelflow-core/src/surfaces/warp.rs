use crate::Surface;
use crate::{Batch, Manifold};
use core::fmt::Debug;
use core::marker::PhantomData;

/// Warps the coordinate space of a surface.
///
/// This is one of the Six Eigenshaders. It corresponds to `(S, ω) -> S`.
/// The coordinate mapping function `ω` transforms inputs `(x, y, z, w)`
/// into new coordinates `(x', y', z', w')`, which are then used to sample `S`.
#[derive(Copy, Clone)]
pub struct Warp<S, F, C> {
    /// The source surface to sample.
    pub source: S,
    /// The coordinate mapping function: `(x, y, z, w) -> (x', y', z', w')`.
    pub mapping: F,
    /// Phantom data for coordinate type.
    pub _marker: PhantomData<C>,
}

impl<S, F, C> Warp<S, F, C> {
    /// Creates a new `Warp` combinator.
    #[inline]
    pub fn new(source: S, mapping: F) -> Self {
        Self {
            source,
            mapping,
            _marker: PhantomData,
        }
    }
}

impl<T, S, F, C> Manifold<T, C> for Warp<S, F, C>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Manifold<T, C>,
    F: Fn(Batch<C>, Batch<C>, Batch<C>, Batch<C>) -> (Batch<C>, Batch<C>, Batch<C>, Batch<C>)
        + Send
        + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        let (nx, ny, nz, nw) = (self.mapping)(x, y, z, w);
        self.source.eval(nx, ny, nz, nw)
    }
}
