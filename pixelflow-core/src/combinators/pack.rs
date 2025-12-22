//! # Pack Combinator
//!
//! Folds a vector manifold to a scalar using a binary operator.
//! This is the algebraic dual of `Project` - where Project extracts one axis,
//! Pack combines all axes.

use crate::{Field, Manifold, ops::Vector, variables::Axis};

/// Folds a vector manifold to a scalar using a binary operator.
///
/// Given a manifold that outputs a `Vector<Component = Field>` and a binary
/// operator, Pack evaluates the manifold and folds all components using the
/// operator, producing a single `Field`.
///
/// # Example
///
/// For RGBA → packed u32:
/// ```ignore
/// let packed = Pack::new(color_manifold, |a, b| {
///     // shift and OR to pack bytes
/// });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Pack<M, Op> {
    /// The inner vector manifold.
    pub inner: M,
    /// The binary fold operation.
    pub op: Op,
}

impl<M, Op> Pack<M, Op> {
    /// Create a new Pack combinator.
    pub fn new(inner: M, op: Op) -> Self {
        Self { inner, op }
    }
}

impl<M, Op, V> Manifold for Pack<M, Op>
where
    M: Manifold<Output = V>,
    V: Vector<Component = Field>,
    Op: Fn(Field, Field) -> Field + Send + Sync,
{
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let vec = self.inner.eval_raw(x, y, z, w);

        // Fold: x op y op z op w
        let acc = (self.op)(vec.get(Axis::X), vec.get(Axis::Y));
        let acc = (self.op)(acc, vec.get(Axis::Z));
        (self.op)(acc, vec.get(Axis::W))
    }
}
