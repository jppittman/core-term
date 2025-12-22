//! # Select Combinator
//!
//! Branchless conditional with short-circuit evaluation.

use crate::Manifold;
use crate::numeric::Numeric;

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

impl<C, T, F, I, O> Manifold<I> for Select<C, T, F>
where
    I: Numeric,
    O: Numeric,
    C: Manifold<I, Output = O>,
    T: Manifold<I, Output = O>,
    F: Manifold<I, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> O {
        let mask = self.cond.eval_raw(x, y, z, w);
        let true_val = self.if_true.eval_raw(x, y, z, w);
        let false_val = self.if_false.eval_raw(x, y, z, w);

        O::select(mask, true_val, false_val)
    }
}
