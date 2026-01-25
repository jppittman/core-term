//! The Universal Contramap: Domain Mapping via Coordinate Transformation.
//!
//! The `At` combinator is a **contramap** that maps a Manifold from one domain
//! to another by transforming coordinates.
//!
//! # Semantics
//!
//! Given a manifold `M: Manifold<(I, I, I, I)>` and coordinate expressions mapping P -> (I, I, I, I),
//! then `At` produces a `Manifold<P>` by composing: domain P → domain (I, I, I, I) → output.
//!
//! The key insight: **all coordinate expressions receive the same input P**.
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
//! // Caller passes any P, mapped to (3, 4, 0, 0), material evaluates
//! ```

use crate::{Computational, Manifold};
use pixelflow_macros::Element;

/// The universal contramap combinator (tuple-based).
///
/// Maps a Manifold from domain B to domain A via coordinate transformation.
/// - **Cx, Cy, Cz, Cw**: Coordinate expressions (P -> I)
/// - **M**: Inner manifold ((I, I, I, I) -> Output)
#[derive(Clone, Debug, Element)]
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

impl<P, Cx, Cy, Cz, Cw, M> Manifold<P> for At<Cx, Cy, Cz, Cw, M>
where
    P: Copy + Send + Sync,
    Cx: Manifold<P>,
    Cx::Output: Computational,
    Cy: Manifold<P>,
    Cy::Output: Into<Cx::Output>,
    Cz: Manifold<P>,
    Cz::Output: Into<Cx::Output>,
    Cw: Manifold<P>,
    Cw::Output: Into<Cx::Output>,
    M: Manifold<(Cx::Output, Cx::Output, Cx::Output, Cx::Output)>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        // 1. Map Domain P -> Domain (I, I, I, I) using coordinate expressions.
        // Cx::Output is the canonical type; other coords convert into it.
        // This allows heterogeneous coord types that all convert to a common type.
        // Note: All coordinates receive the full input p.
        let new_x = self.x.eval(p);
        let new_y: Cx::Output = self.y.eval(p).into();
        let new_z: Cx::Output = self.z.eval(p).into();
        let new_w: Cx::Output = self.w.eval(p).into();
        // 2. Evaluate inner manifold on domain (Cx::Output, ...).
        self.inner.eval((new_x, new_y, new_z, new_w))
    }
}

impl<Cx, Cy, Cz, Cw, M> At<Cx, Cy, Cz, Cw, M>
where
    Cx: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field), Output = crate::Field>,
    Cy: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field), Output = crate::Field>,
    Cz: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field), Output = crate::Field>,
    Cw: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field), Output = crate::Field>,
    M: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field)>,
{
    /// Collapse the pinned manifold to a value.
    ///
    /// Since coordinates are already bound via At, evaluates at origin.
    /// `foo.at(x, y, z, w).eval()` is cleaner than `foo.eval((x, y, z, w))`.
    #[inline(always)]
    pub fn collapse(&self) -> M::Output {
        let zero = crate::Field::from(0.0);
        self.eval((zero, zero, zero, zero))
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
#[derive(Clone, Debug, Element)]
pub struct AtArray<M, const N: usize, C> {
    /// The inner manifold to evaluate on domain B.
    pub inner: M,
    /// Coordinate expressions: [cx, cy, cz, cw] each mapping P -> I.
    pub coords: [C; N],
}

impl<P, I, C, M> Manifold<P> for AtArray<M, 4, C>
where
    P: Copy + Send + Sync,
    I: Computational,
    C: Manifold<P, Output = I> + Copy,
    M: Manifold<(I, I, I, I)>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        // 1. Map Domain P -> Domain (I, I, I, I) using coordinate expressions.
        // Note: All coordinates receive the full input p.
        // This allows 'x' to depend on 'w' (time) or 'z' (swizzle), etc.
        let nx = self.coords[0].eval(p);
        let ny = self.coords[1].eval(p);
        let nz = self.coords[2].eval(p);
        let nw = self.coords[3].eval(p);

        // 2. Evaluate inner manifold on domain (I, I, I, I).
        self.inner.eval((nx, ny, nz, nw))
    }
}

impl<M, C> AtArray<M, 4, C>
where
    C: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field), Output = crate::Field>
        + Copy,
    M: Manifold<(crate::Field, crate::Field, crate::Field, crate::Field)>,
{
    /// Collapse the pinned manifold to a value.
    ///
    /// Since coordinates are already bound via At, evaluates at origin.
    #[inline(always)]
    pub fn collapse(&self) -> M::Output {
        let zero = crate::Field::from(0.0);
        self.eval((zero, zero, zero, zero))
    }
}
