//! # Fix Combinators
//!
//! Combinators for fixed-point iteration and recursion.
//!
//! - [`Fix`]: Dynamic iteration until convergence (runtime `loop`).
//! - [`RecFix`]: Static recursion with compile-time unrolling (Peano numbers).

use crate::{Field, Manifold};
use crate::combinators::binding::{Pred, UTerm}; // Use existing Peano infrastructure
use core::marker::PhantomData;

type Field4 = (Field, Field, Field, Field);

// ============================================================================
// Dynamic Fix (Runtime Loop)
// ============================================================================

/// Iterate until convergence.
#[derive(Clone, Debug)]
pub struct Fix<Seed, Step, Done> {
    /// Initial state manifold.
    pub seed: Seed,
    /// State update manifold applied each iteration.
    pub step: Step,
    /// Termination predicate manifold (non-zero = done).
    pub done: Done,
}

impl<Seed, Step, Done> Manifold<Field4> for Fix<Seed, Step, Done>
where
    Seed: Manifold<Field4, Output = Field>,
    Step: Manifold<Field4, Output = Field>,
    Done: Manifold<Field4, Output = Field>,
{
    type Output = Field;
    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, _w) = p;
        let mut state = self.seed.eval(p);
        let zero = Field::from(0.0f32);
        let mut active = zero.ge(zero);

        loop {
            let done_mask = self.done.eval((x, y, z, state));
            active = Field::select_raw(done_mask, zero, active);
            if !active.any() {
                break;
            }
            let next = self.step.eval((x, y, z, state));
            state = Field::select_raw(active, next, state);
        }
        state
    }
}

impl<Seed, Step, Done> crate::ManifoldExpr for Fix<Seed, Step, Done> {}

// ============================================================================
// Static Recursion (Compile-Time Unrolling)
// ============================================================================

/// ZST marker: "Recurse here".
#[derive(Clone, Copy, Debug, Default)]
pub struct Recurse;

impl crate::ManifoldExpr for Recurse {}

/// Domain extension carrying the recursion target type.
#[derive(Clone, Copy, Debug)]
pub struct RecDomain<Target, P> {
    _target: PhantomData<Target>,
    /// The original coordinates before domain extension.
    pub coords: P,
}

impl<Target, P: crate::domain::Spatial> crate::domain::Spatial for RecDomain<Target, P> {
    type Coord = P::Coord;
    type Scalar = P::Scalar;
    #[inline(always)]
    fn x(&self) -> Self::Coord { self.coords.x() }
    #[inline(always)]
    fn y(&self) -> Self::Coord { self.coords.y() }
    #[inline(always)]
    fn z(&self) -> Self::Coord { self.coords.z() }
    #[inline(always)]
    fn w(&self) -> Self::Coord { self.coords.w() }
}

/// # Example: Sierpinski
/// ```ignore
/// type Step = Min<Recurse, ...>; // Simplified
/// type Sierpinski = RecFix<5, Step, Triangle>;
/// ```
#[derive(Debug)]
pub struct RecFix<N, Step, Base> {
    _phantom: PhantomData<(N, Step, Base)>,
}

impl<N, S, B> Default for RecFix<N, S, B> {
    fn default() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<N, S, B> Clone for RecFix<N, S, B> {
    fn clone(&self) -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<N, S, B> Copy for RecFix<N, S, B> {}

impl<N, S, B> crate::ManifoldExpr for RecFix<N, S, B> {}

// 1. Implement Manifold for Recurse
impl<Target, P> Manifold<RecDomain<Target, P>> for Recurse
where
    P: Copy + Send + Sync,
    Target: Manifold<P> + Default,
{
    type Output = Target::Output;

    #[inline(always)]
    fn eval(&self, p: RecDomain<Target, P>) -> Self::Output {
        Target::default().eval(p.coords)
    }
}

// 2. Implement Manifold for RecFix (Recursive Case: N > 0)
impl<N, Step, Base, P> Manifold<P> for RecFix<N, Step, Base>
where
    P: Copy + Send + Sync,
    N: Pred + Send + Sync,
    N::Output: Send + Sync,
    Step: Manifold<RecDomain<RecFix<N::Output, Step, Base>, P>> + Default,
    Base: Manifold<P> + Send + Sync,
    Step: Manifold<RecDomain<RecFix<N::Output, Step, Base>, P>, Output = Base::Output>,
{
    type Output = Base::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let extended = RecDomain {
            _target: PhantomData::<RecFix<N::Output, Step, Base>>,
            coords: p,
        };
        Step::default().eval(extended)
    }
}

// 3. Implement Manifold for RecFix (Base Case: N = UTerm)
impl<Step, Base, P> Manifold<P> for RecFix<UTerm, Step, Base>
where
    P: Copy + Send + Sync,
    Base: Manifold<P> + Default,
    Step: Send + Sync, // Required for RecFix to be Send+Sync
{
    type Output = Base::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        Base::default().eval(p)
    }
}
