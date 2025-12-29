//! Evaluate a manifold at fixed coordinates.
//!
//! The `At` combinator "pins" a manifold to specific coordinates.
//! When evaluated, it ignores input coordinates and instead evaluates
//! the inner manifold at the stored coordinates.
//!
//! This is useful for composing manifolds that should be evaluated at
//! different points, then combined via Select or other operators.

use crate::{Manifold, Computational};

/// Pins a manifold to fixed coordinates.
///
/// Evaluates the inner manifold at the stored coordinates,
/// regardless of what coordinates are passed to `eval_raw`.
///
/// # Example
///
/// ```ignore
/// // Evaluate material at warped hit point, background at ray origin
/// let mat_at_hit = At {
///     inner: &material,
///     x: hx, y: hy, z: hz, w,
/// };
/// let bg_at_ray = At {
///     inner: &background,
///     x: rx, y: ry, z: rz, w,
/// };
/// // Then use with Select to blend based on a mask
/// ```
#[derive(Clone, Copy, Debug)]
pub struct At<I, M> {
    /// The inner manifold to evaluate at fixed coordinates.
    pub inner: M,
    /// X coordinate to pin the manifold at.
    pub x: I,
    /// Y coordinate to pin the manifold at.
    pub y: I,
    /// Z coordinate to pin the manifold at.
    pub z: I,
    /// W coordinate to pin the manifold at.
    pub w: I,
}

impl<I, M> Manifold<I> for At<I, M>
where
    I: Computational,
    M: Manifold<I>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> Self::Output {
        self.inner.eval_raw(self.x, self.y, self.z, self.w)
    }
}
