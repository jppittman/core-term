//! # Unary Operations
//!
//! AST nodes for unary ops: Sqrt, Abs, Floor, Sin, Min, Max.

use crate::Manifold;
use crate::numeric::{Computational, Numeric};

/// Square root.
#[derive(Clone, Debug)]
pub struct Sqrt<M>(pub M);

/// Negation: -M
#[derive(Clone, Debug)]
pub struct Neg<M>(pub M);

/// Absolute value.
#[derive(Clone, Debug)]
pub struct Abs<M>(pub M);

/// Floor (round toward negative infinity).
#[derive(Clone, Debug)]
pub struct Floor<M>(pub M);

/// Reciprocal square root: 1/sqrt(M).
/// Uses fast SIMD rsqrt with Newton-Raphson refinement.
#[derive(Clone, Debug)]
pub struct Rsqrt<M>(pub M);

/// Sine function.
#[derive(Clone, Debug)]
pub struct Sin<M>(pub M);

/// Cosine function.
#[derive(Clone, Debug)]
pub struct Cos<M>(pub M);

/// Base-2 logarithm.
#[derive(Clone, Debug)]
pub struct Log2<M>(pub M);

/// Base-2 exponential.
#[derive(Clone, Debug)]
pub struct Exp2<M>(pub M);

/// Element-wise maximum.
#[derive(Clone, Debug)]
pub struct Max<L, R>(pub L, pub R);

/// Element-wise minimum.
#[derive(Clone, Debug)]
pub struct Min<L, R>(pub L, pub R);

// ============================================================================
// Domain-Generic Manifold Implementations
// ============================================================================

impl<P, M, O> Manifold<P> for Sqrt<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).sqrt()
    }
}

impl<P, M, O> Manifold<P> for Neg<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        <O as Computational>::from_f32(0.0).raw_sub(self.0.eval(p))
    }
}

impl<P, M, O> Manifold<P> for Abs<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).abs()
    }
}

impl<P, M, O> Manifold<P> for Floor<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).floor()
    }
}

impl<P, M, O> Manifold<P> for Rsqrt<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).rsqrt()
    }
}

impl<P, M, O> Manifold<P> for Sin<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).sin()
    }
}

impl<P, M, O> Manifold<P> for Cos<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).cos()
    }
}

impl<P, M, O> Manifold<P> for Log2<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).log2()
    }
}

impl<P, M, O> Manifold<P> for Exp2<M>
where
    P: Copy + Send + Sync,
    O: Numeric,
    M: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).exp2()
    }
}

impl<P, L, R, O> Manifold<P> for Max<L, R>
where
    P: Copy + Send + Sync,
    O: Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).max(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Min<L, R>
where
    P: Copy + Send + Sync,
    O: Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).min(self.1.eval(p))
    }
}
