//! The Universal Contramap: Domain Mapping via Coordinate Transformation.
//!
//! The `At` combinator is a **contramap** that maps a Manifold from one domain
//! to another by transforming coordinates.
//!
//! # Semantics
//!
//! Given a manifold `M: Manifold<B>` and coordinate expressions mapping A -> B,
//! then `At` produces a `Manifold<A>` by composing: domain A → domain B → output.
//!
//! The key insight: **all coordinate expressions receive the same input A**.
//! This allows x to depend on w (time), z to depend on y (swizzle), etc.
//!
//! # Example: Swizzle Coordinates
//!
//! ```ignore
//! use pixelflow_core::{X, Y, Z, W, Manifold};
//!
//! // Treat time (W) as depth (Z)
//! let swizzled = At {
//!     inner: animation,
//!     x: X,
//!     y: Y,
//!     z: W,
//!     w: Z,
//! };
//! ```
//!
//! # Example: Pin to a Point
//!
//! ```ignore
//! use pixelflow_core::{Jet3, At};
//!
//! // Pin to (3, 4, 0, 0) in Jet3 domain
//! let pinned = At {
//!     inner: material,
//!     x: Jet3::from(3.0),
//!     y: Jet3::from(4.0),
//!     z: Jet3::from(0.0),
//!     w: Jet3::from(0.0),
//! };
//! // Caller passes any A, mapped to (3, 4, 0, 0), material evaluates on Jet3
//! ```

use crate::{Computational, Manifold};

/// The universal contramap combinator (tuple-based).
///
/// Maps a Manifold from domain B to domain A via coordinate transformation.
/// - **Cx, Cy, Cz, Cw**: Coordinate expressions (A -> B)
/// - **M**: Inner manifold (B -> Output)
#[derive(Clone, Debug)]
pub struct At<Cx, Cy, Cz, Cw, M> {
    /// The inner manifold to evaluate.
    pub inner: M,
    /// X coordinate expression.
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
        // 1. Map Domain A -> Domain B using coordinate expressions.
        // Note: All coordinates receive the full input (x, y, z, w).
        // This allows 'x' to depend on 'w' (time) or 'z' (swizzle), etc.
        let new_x = self.x.eval_raw(x, y, z, w);
        let new_y = self.y.eval_raw(x, y, z, w);
        let new_z = self.z.eval_raw(x, y, z, w);
        let new_w = self.w.eval_raw(x, y, z, w);
        // 2. Evaluate inner manifold on domain B.
        self.inner.eval_raw(new_x, new_y, new_z, new_w)
    }
}

impl<Cx, Cy, Cz, Cw, M> At<Cx, Cy, Cz, Cw, M>
where
    Cx: Manifold<crate::Field, Output = crate::Field>,
    Cy: Manifold<crate::Field, Output = crate::Field>,
    Cz: Manifold<crate::Field, Output = crate::Field>,
    Cw: Manifold<crate::Field, Output = crate::Field>,
    M: Manifold<crate::Field>,
{
    /// Collapse the pinned manifold to a value.
    ///
    /// Since coordinates are already bound via At, evaluates at origin.
    /// `foo.at(x, y, z, w).eval()` is cleaner than `foo.eval_raw(x, y, z, w)`.
    #[inline(always)]
    pub fn eval(&self) -> M::Output {
        let zero = crate::Field::from(0.0);
        self.eval_raw(zero, zero, zero, zero)
    }
}

// ============================================================================
// Array-based At for cleaner syntax
// ============================================================================

/// The universal contramap combinator (array-based form).
///
/// Maps a Manifold from domain B to domain A via coordinate transformation,
/// using an array of homogeneous coordinate expressions.
///
/// This form is useful when all coordinates have the same type and you want
/// cleaner constructors. For heterogeneous coordinates, use `At` (tuple form).
///
/// - **M**: Inner manifold (B -> Output)
/// - **C**: Coordinate expression type (all coordinates have type C)
/// - **N**: Number of coordinates (4 for xyzw)
///
/// # Example
///
/// ```ignore
/// use pixelflow_core::{X, Y, Z, W};
///
/// let at_array = AtArray {
///     inner: material,
///     coords: [X, Y, Z, W + 1.0],
/// };
/// ```
#[derive(Clone, Debug)]
pub struct AtArray<M, const N: usize, C> {
    /// The inner manifold to evaluate on domain B.
    pub inner: M,
    /// Coordinate expressions: [cx, cy, cz, cw] each mapping A -> B.
    pub coords: [C; N],
}

impl<A, B, C, M> Manifold<A> for AtArray<M, 4, C>
where
    A: Computational,
    B: Computational,
    C: Manifold<A, Output = B> + Copy,
    M: Manifold<B>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: A, y: A, z: A, w: A) -> Self::Output {
        // 1. Map Domain A -> Domain B using coordinate expressions.
        // Note: All coordinates receive the full input (x, y, z, w).
        // This allows 'x' to depend on 'w' (time) or 'z' (swizzle), etc.
        let nx = self.coords[0].eval_raw(x, y, z, w);
        let ny = self.coords[1].eval_raw(x, y, z, w);
        let nz = self.coords[2].eval_raw(x, y, z, w);
        let nw = self.coords[3].eval_raw(x, y, z, w);

        // 2. Evaluate inner manifold on domain B.
        self.inner.eval_raw(nx, ny, nz, nw)
    }
}

impl<M, C> AtArray<M, 4, C>
where
    C: Manifold<crate::Field, Output = crate::Field> + Copy,
    M: Manifold<crate::Field>,
{
    /// Collapse the pinned manifold to a value.
    ///
    /// Since coordinates are already bound via At, evaluates at origin.
    #[inline(always)]
    pub fn eval(&self) -> M::Output {
        let zero = crate::Field::from(0.0);
        self.eval_raw(zero, zero, zero, zero)
    }
}
