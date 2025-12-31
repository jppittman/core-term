//! # Map Combinator
//!
//! Transforms the output of a manifold.
//! This is fmap for the Manifold functor.

use crate::{Field, Manifold};

/// Maps a function over a manifold's output (covariant functor).
///
/// This is the functor `fmap` operation for manifolds.
/// It transforms every output value while preserving the spatial structure.
///
/// ## Covariant Mapping
///
/// Map can change the output type, enabling conversions like:
/// - `Field → Field` (same type)
/// - `Field → PathJet<Field>` (lifting to ray space)
/// - `Field → Discrete` (color packing)
///
/// # Example
///
/// ```ignore
/// // Double every output value (Field → Field)
/// let doubled = Map::new(sdf, |v| v * 2.0);
///
/// // Convert screen coord to ray (Field → PathJet)
/// let ray_x = Map::new(X, PathJet::from_slope);
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

/// Field → Field mapping (backward compatible)
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

/// Covariant map for Field → PathJet<Field>
impl<M, F> Manifold<crate::jet::PathJet<Field>> for Map<M, F>
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
