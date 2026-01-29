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
/// `result(p) = transform((inner(p), y, z, w))`
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
#[derive(Clone, Debug)]
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

// Map on 4D domains: output of inner becomes X for transform
impl<I, M, T> Manifold<(I, I, I, I)> for Map<M, T>
where
    I: crate::numeric::Computational,
    M: Manifold<(I, I, I, I), Output = I>,
    T: Manifold<(I, I, I, I), Output = I>,
{
    type Output = I;

    #[inline(always)]
    fn eval(&self, p: (I, I, I, I)) -> Self::Output {
        let val = self.inner.eval(p);
        // Map the output of inner to X of transform, pass others through
        self.transform.eval((val, p.1, p.2, p.3))
    }
}

// ============================================================================
// ClosureMap (Legacy/Functional Map)
// ============================================================================

/// Maps a Rust closure over a manifold's output.
///
/// This is used for transformations that cannot be expressed as manifolds,
/// such as lifting to complex types like `PathJet` via factory functions.
#[derive(Clone, Debug)]
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

type Field4 = (Field, Field, Field, Field);

impl<M, F> Manifold<Field4> for ClosureMap<M, F>
where
    M: Manifold<Field4, Output = Field>,
    F: Fn(Field) -> Field + Send + Sync,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let val = self.inner.eval(p);
        (self.func)(val)
    }
}

/// Covariant map for Field -> PathJet<Field>
type PathJet4 = (
    crate::jet::PathJet<Field>,
    crate::jet::PathJet<Field>,
    crate::jet::PathJet<Field>,
    crate::jet::PathJet<Field>,
);

impl<M, F> Manifold<PathJet4> for ClosureMap<M, F>
where
    M: Manifold<Field4, Output = Field>,
    F: Fn(Field) -> crate::jet::PathJet<Field> + Send + Sync,
{
    type Output = crate::jet::PathJet<Field>;

    #[inline(always)]
    fn eval(&self, p: PathJet4) -> crate::jet::PathJet<Field> {
        // Extract origin components and evaluate inner manifold
        let val = self.inner.eval((p.0.val, p.1.val, p.2.val, p.3.val));
        (self.func)(val)
    }
}
