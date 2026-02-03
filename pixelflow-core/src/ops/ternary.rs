//! # Ternary Operations
//!
//! AST nodes for three-operand operations like clamp.

use crate::Manifold;
use crate::numeric::Numeric;
use pixelflow_compiler::Element;

/// Clamp: constrain value to range [lo, hi].
///
/// Returns lo if value < lo, hi if value > hi, otherwise value.
#[derive(Clone, Debug, Element)]
pub struct Clamp<V, Lo, Hi> {
    /// The value to clamp.
    pub value: V,
    /// Lower bound.
    pub lo: Lo,
    /// Upper bound.
    pub hi: Hi,
}

impl<P, V, Lo, Hi, O> Manifold<P> for Clamp<V, Lo, Hi>
where
    P: Copy + Send + Sync,
    O: Numeric,
    V: Manifold<P, Output = O>,
    Lo: Manifold<P, Output = O>,
    Hi: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        let v = self.value.eval(p);
        let lo = self.lo.eval(p);
        let hi = self.hi.eval(p);
        v.clamp(lo, hi)
    }
}
