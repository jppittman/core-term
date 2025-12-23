//! Coordinate transformations for manifolds.

use pixelflow_core::{Manifold, Numeric};

/// Uniform scaling of the manifold domain.
///
/// Effectively scales the object size by `factor`.
/// Internally, coordinates are divided by `factor`.
#[derive(Clone, Debug)]
pub struct Scale<M> {
    pub manifold: M,
    pub factor: f32,
}

impl<M, I> Manifold<I> for Scale<M>
where
    M: Manifold<I>,
    I: Numeric,
{
    type Output = M::Output;

    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let s = I::from_f32(self.factor);
        self.manifold.eval_raw(x / s, y / s, z, w)
    }
}

/// Translation of the manifold domain.
///
/// Shifts the object by `offset` (dx, dy).
/// Internally, coordinates are subtracted by `offset`.
#[derive(Clone, Debug)]
pub struct Translate<M> {
    pub manifold: M,
    pub offset: [f32; 2],
}

impl<M, I> Manifold<I> for Translate<M>
where
    M: Manifold<I>,
    I: Numeric,
{
    type Output = M::Output;

    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let dx = I::from_f32(self.offset[0]);
        let dy = I::from_f32(self.offset[1]);
        self.manifold.eval_raw(x - dx, y - dy, z, w)
    }
}
