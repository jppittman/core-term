//! # Map Combinator
//!
//! Transforms the output of a manifold using another manifold.
//! This is the algebraic version of fmap.

use crate::{Field, Manifold};

/// Maps a manifold transformation over a manifold's output.
///
/// This transforms the output value of the inner manifold using the transformation manifold.
/// The output of the inner manifold becomes the X coordinate for the transformation manifold.
/// The Y, Z, and W coordinates are passed through unchanged, allowing context-aware mapping.
///
/// # Semantics
///
/// Given `inner` and `transform`:
/// `result(x, y, z, w) = transform(inner(x, y, z, w), y, z, w)`
///
/// # Example
///
/// ```ignore
/// // Double every output value (Field -> Field)
/// let doubled = Map::new(sdf, X * 2.0);
///
/// // Add the Y coordinate to the output
/// let skewed = Map::new(sdf, X + Y);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Map<M, T> {
    inner: M,
    transform: T,
}

impl<M, T> Map<M, T> {
    /// Create a new Map combinator.
    #[inline(always)]
    pub fn new(inner: M, transform: T) -> Self {
        Self { inner, transform }
    }
}

impl<I, M, T> Manifold<I> for Map<M, T>
where
    I: crate::numeric::Computational,
    M: Manifold<I, Output = I>,
    T: Manifold<I, Output = I>,
{
    type Output = I;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let val = self.inner.eval_raw(x, y, z, w);
        // Map the output of inner to X of transform, pass others through
        self.transform.eval_raw(val, y, z, w)
    }
}

// ============================================================================
// ClosureMap (Legacy/Functional Map)
// ============================================================================

/// Maps a Rust closure over a manifold's output.
///
/// This is used for transformations that cannot be expressed as manifolds,
/// such as lifting to complex types like `PathJet` via factory functions.
#[derive(Clone, Copy, Debug)]
pub struct ClosureMap<M, F> {
    inner: M,
    func: F,
}

impl<M, F> ClosureMap<M, F> {
    /// Create a new ClosureMap combinator.
    #[inline(always)]
    pub fn new(inner: M, func: F) -> Self {
        Self { inner, func }
    }
}

impl<M, F> Manifold for ClosureMap<M, F>
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

/// Covariant map for Field -> PathJet<Field>
impl<M, F> Manifold<crate::jet::PathJet<Field>> for ClosureMap<M, F>
where
    M: Manifold<Field, Output = Field>,
    F: Fn(Field) -> crate::jet::PathJet<Field> + Send + Sync,
{
    type Output = crate::jet::PathJet<Field>;

    #[inline(always)]
    fn eval_raw(
        &self,
        x: crate::jet::PathJet<Field>,
        y: crate::jet::PathJet<Field>,
        z: crate::jet::PathJet<Field>,
        w: crate::jet::PathJet<Field>,
    ) -> crate::jet::PathJet<Field> {
        // Extract origin components and evaluate inner manifold
        let val = self.inner.eval_raw(x.val, y.val, z.val, w.val);
        (self.func)(val)
    }
}
