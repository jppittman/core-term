//! # Map Combinator
//!
//! Transforms the output of a manifold.
//! This is fmap for the Manifold functor.

use crate::{Field, Manifold};

/// Maps a function over a manifold's output.
///
/// This is the functor `fmap` operation for manifolds.
/// It transforms every output value while preserving the spatial structure.
///
/// # Example
///
/// ```ignore
/// // Double every output value
/// let doubled = Map::new(sdf, |v| v * 2.0);
///
/// // Clamp to [0, 1]
/// let clamped = Map::new(color_channel, |v| v.max(0.0).min(1.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Map<M, F> {
    inner: M,
    func: F,
}

impl<M, F> Map<M, F> {
    /// Create a new Map combinator.
    #[inline(always)]
    pub fn new(inner: M, func: F) -> Self {
        Self { inner, func }
    }
}

impl<M, F> Manifold for Map<M, F>
where
    M: Manifold<Output = Field>,
    F: Fn(Field) -> Field + Send + Sync,
{
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let val = self.inner.eval_raw(x, y, z, w);
        (self.func)(val)
    }
}
