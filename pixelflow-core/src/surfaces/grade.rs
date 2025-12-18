use crate::{Batch, Manifold};
use core::fmt::Debug;
use core::ops::{Add, Mul};

/// Applies a linear transform to the values of a surface.
///
/// This is one of the Six Eigenshaders. It corresponds to `(S, M, b) -> S`.
/// The result is `S(x) * M + b`.
#[derive(Copy, Clone)]
pub struct Grade<S, M, B> {
    /// The source surface.
    pub source: S,
    /// The multiplier.
    pub slope: M,
    /// The bias (y-intercept).
    pub bias: B,
}

impl<S, M, B> Grade<S, M, B> {
    /// Creates a new `Grade` combinator.
    #[inline]
    pub fn new(source: S, slope: M, bias: B) -> Self {
        Self {
            source,
            slope,
            bias,
        }
    }
}

// Support for scalar constants broadcasted to batches would be nice,
// but for now we follow the exact types. The slope/bias could be surfaces themselves
// (spatially varying grade) or constants. The simplest Grade is constant M and B.
// If M and B are Surfaces, this is even more powerful (modulation).

// Let's implement the general case where M and B are also Surfaces (conceptually constants are constant surfaces).
// Wait, the North Star says `(S, M, b) -> S`. M and b usually imply constants or fields?
// "Linear transform on values (matrix + bias)".
// If specific types are needed we can specialize, but generally T * T + T is what we want.

impl<T, S, M, B, C> Manifold<T, C> for Grade<S, M, B>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    S: Manifold<T, C>,
    M: Manifold<T, C>,
    B: Manifold<T, C>,
    Batch<T>: Mul<Output = Batch<T>> + Add<Output = Batch<T>>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        let val = self.source.eval(x, y, z, w);
        let m = self.slope.eval(x, y, z, w);
        let b = self.bias.eval(x, y, z, w);
        // val * m + b
        val * m + b
    }
}
