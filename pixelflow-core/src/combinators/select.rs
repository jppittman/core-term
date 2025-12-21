//! # Select Combinator
//!
//! Branchless conditional with short-circuit evaluation.

use crate::backend::{BatchArithmetic, SimdBatch};
use crate::{Field, Manifold};

/// Branchless conditional with short-circuit.
///
/// Evaluates `if_true` only when some lanes are true,
/// `if_false` only when some are false.
#[derive(Clone, Copy, Debug)]
pub struct Select<C, T, F> {
    /// The condition (produces a mask).
    pub cond: C,
    /// Value when condition is true.
    pub if_true: T,
    /// Value when condition is false.
    pub if_false: F,
}

impl<C: Manifold, T: Manifold, F: Manifold> Manifold for Select<C, T, F> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let mask = self.cond.eval(x, y, z, w);

        // Short-circuit: skip evaluation if all lanes agree
        if mask.all() {
            return self.if_true.eval(x, y, z, w);
        }
        if !mask.any() {
            return self.if_false.eval(x, y, z, w);
        }

        // Mixed: evaluate both and blend
        BatchArithmetic::select(
            mask,
            self.if_true.eval(x, y, z, w),
            self.if_false.eval(x, y, z, w),
        )
    }
}
