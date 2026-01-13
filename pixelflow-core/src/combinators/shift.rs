//! Translate a manifold by constant offsets.
//!
//! The `Shift` combinator subtracts constant offsets from input coordinates
//! before evaluating the inner manifold. This is the inverse of translation:
//! shifting by (dx, dy, dz, dw) moves the manifold's origin to (dx, dy, dz, dw).
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::{ManifoldExt};
//!
//! // Move a unit sphere to center (0, 0, 4)
//! let sphere_at_z4 = unit_sphere.shift(0.0, 0.0, 4.0, 0.0);
//!
//! // Equivalent to: unit_sphere evaluated at (x - 0, y - 0, z - 4, w - 0)
//! ```
//!
//! # Why Shift Instead of At?
//!
//! - `At` evaluates at computed coordinate expressions (general, but heavier)
//! - `Shift` subtracts constants (simpler, more efficient for translation)
//!
//! For constant translations, Shift avoids creating manifold expressions.

use crate::{Computational, Manifold};

/// Translates a manifold by subtracting constant offsets from coordinates.
///
/// Given `inner.shift(dx, dy, dz, dw)`, evaluation at `(x, y, z, w)` returns
/// `inner.eval_raw(x - dx, y - dy, z - dz, w - dw)`.
#[derive(Clone, Debug)]
pub struct Shift<M> {
    /// The inner manifold to translate.
    pub inner: M,
    /// X offset (subtracted from input x).
    pub dx: f32,
    /// Y offset (subtracted from input y).
    pub dy: f32,
    /// Z offset (subtracted from input z).
    pub dz: f32,
    /// W offset (subtracted from input w).
    pub dw: f32,
}

impl<I, M> Manifold<I> for Shift<M>
where
    I: Computational,
    M: Manifold<I>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        self.inner.eval_raw(
            x - I::from_f32(self.dx),
            y - I::from_f32(self.dy),
            z - I::from_f32(self.dz),
            w - I::from_f32(self.dw),
        )
    }
}
