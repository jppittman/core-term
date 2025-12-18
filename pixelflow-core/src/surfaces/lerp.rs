use crate::{Batch, Manifold};
use core::fmt::Debug;
use core::ops::{Add, Mul, Sub};

/// Linearly interpolates between two surfaces.
///
/// This is one of the Six Eigenshaders. It corresponds to `(t, a, b) -> S`.
/// Result is `a + t * (b - a)`.
#[derive(Copy, Clone)]
pub struct Lerp<Param, A, B> {
    /// The interpolation parameter `t` (domain [0, 1]).
    pub t: Param,
    /// The 'start' surface (t=0).
    pub a: A,
    /// The 'end' surface (t=1).
    pub b: B,
}

impl<Param, A, B> Lerp<Param, A, B> {
    /// Creates a new `Lerp` combinator.
    #[inline]
    pub fn new(t: Param, a: A, b: B) -> Self {
        Self { t, a, b }
    }
}

impl<T, Param, A, B, C> Manifold<T, C> for Lerp<Param, A, B>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    Param: Manifold<T, C>,
    A: Manifold<T, C>,
    B: Manifold<T, C>,
    Batch<T>: Add<Output = Batch<T>> + Sub<Output = Batch<T>> + Mul<Output = Batch<T>>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        let t_val = self.t.eval(x, y, z, w);
        let a_val = self.a.eval(x, y, z, w);
        let b_val = self.b.eval(x, y, z, w);

        // a + t * (b - a)
        a_val + t_val * (b_val - a_val)
    }
}
