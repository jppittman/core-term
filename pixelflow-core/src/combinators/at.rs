//! Evaluate a manifold at computed coordinates.
//!
//! The `At` combinator takes manifold expressions for coordinates
//! and evaluates the inner manifold at those computed points.
//! This enables composing manifolds evaluated at different coordinate systems.
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::{X, Y, Z, Jet3};
//!
//! // Warp coordinates and evaluate material at warped point
//! let mat_warped = At {
//!     inner: &material,
//!     x: X * scale,        // Manifold expression
//!     y: Y * scale,
//!     z: Z * scale,
//!     w: W,
//! };
//!
//! // Or use constant manifolds (Jet3 implements Manifold)
//! let mat_at_hit = At {
//!     inner: &material,
//!     x: hx,               // Jet3 constant, treated as a manifold
//!     y: hy,
//!     z: hz,
//!     w,
//! };
//! ```

use crate::{Manifold, Computational};

/// Evaluates a manifold at manifold-computed coordinates.
///
/// The coordinate expressions (x, y, z, w) can be any manifolds,
/// including constants (Field, Jet3) or expressions (X + dx, etc).
#[derive(Clone, Copy, Debug)]
pub struct At<Cx, Cy, Cz, Cw, M> {
    /// The inner manifold to evaluate.
    pub inner: M,
    /// X coordinate expression (any Manifold<I> where I is the input type).
    pub x: Cx,
    /// Y coordinate expression.
    pub y: Cy,
    /// Z coordinate expression.
    pub z: Cz,
    /// W coordinate expression.
    pub w: Cw,
}

impl<I, Cx, Cy, Cz, Cw, M> Manifold<I> for At<Cx, Cy, Cz, Cw, M>
where
    I: Computational,
    Cx: Manifold<I, Output = I>,
    Cy: Manifold<I, Output = I>,
    Cz: Manifold<I, Output = I>,
    Cw: Manifold<I, Output = I>,
    M: Manifold<I>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        // Evaluate coordinate expressions
        let new_x = self.x.eval_raw(x, y, z, w);
        let new_y = self.y.eval_raw(x, y, z, w);
        let new_z = self.z.eval_raw(x, y, z, w);
        let new_w = self.w.eval_raw(x, y, z, w);
        // Evaluate inner manifold at computed coordinates
        self.inner.eval_raw(new_x, new_y, new_z, new_w)
    }
}
