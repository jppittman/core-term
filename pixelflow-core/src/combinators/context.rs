//! # Array-Based Context System
//!
//! Provides `WithContext` and `CtxVar` for binding parameters in kernel expressions.
//! Uses array-based indexing instead of flat tuples for scalable impl coverage.
//!
//! ## Problem Solved
//!
//! The previous tuple-based approach required one impl per tuple arity:
//! - `(V0, V1, ..., V15)` needed 16 separate impls
//! - Adding support for 17+ elements required more impls
//!
//! The array-based approach groups values by type:
//! - `([Field; 17], [f32; 1])` only needs impls for "2 arrays"
//! - Impl count is bounded by number of distinct types, not value count
//!
//! ## Architecture
//!
//! ```text
//! // Old: 18-element tuple - needs impl for size 18
//! (Field, Field, Field, ..., f32)
//!
//! // New: 2-element tuple of arrays - just need impl for 2-array combo
//! ([Field; 17], [f32; 1])
//! ```
//!
//! ## CtxVar Indexing
//!
//! `CtxVar<A0, 5>` reads from array A0 at index 5.
//! `CtxVar<A1, 0>` reads from array A1 at index 0.
//!
//! Array position markers: `A0`, `A1`, `A2`, `A3`

use crate::Manifold;
use crate::domain::Spatial;
use core::marker::PhantomData;

// ============================================================================
// Array Position Markers
// ============================================================================

/// Marker for the first array (index 0) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A0;

/// Marker for the second array (index 1) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A1;

/// Marker for the third array (index 2) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A2;

/// Marker for the fourth array (index 3) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A3;

/// Marker for the fifth array (index 4) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A4;

/// Marker for the sixth array (index 5) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A5;

/// Marker for the seventh array (index 6) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A6;

/// Marker for the eighth array (index 7) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A7;

/// Marker for the ninth array (index 8) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A8;

/// Marker for the tenth array (index 9) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A9;

/// Marker for the eleventh array (index 10) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A10;

/// Marker for the twelfth array (index 11) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A11;

/// Marker for the thirteenth array (index 12) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A12;

/// Marker for the fourteenth array (index 13) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A13;

/// Marker for the fifteenth array (index 14) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A14;

/// Marker for the sixteenth array (index 15) in a context tuple.
#[derive(Clone, Copy, Debug, Default)]
pub struct A15;

// ============================================================================
// Context Combinator
// ============================================================================

/// Context combinator: evaluates manifolds in `Ctx` arrays, passes results to `Body`.
///
/// ## Domain Structure
///
/// After evaluation, creates an extended domain:
/// - Single array: `(([T; N],), P)`
/// - Two arrays: `(([T0; N], [T1; M]), P)`
/// - etc.
#[derive(Clone, Debug)]
pub struct WithContext<Ctx, Body> {
    /// The context tuple of arrays to bind.
    pub ctx: Ctx,
    /// The body manifold that receives the evaluated context.
    pub body: Body,
}

impl<Ctx, Body> WithContext<Ctx, Body> {
    /// Create a new context combinator.
    pub const fn new(ctx: Ctx, body: Body) -> Self {
        Self { ctx, body }
    }
}

// ============================================================================
// CtxVar - Array-Indexed Variable Reference
// ============================================================================

/// Type-level index into a context array.
///
/// `ArrayPos` selects which array (A0, A1, A2, A3).
/// `INDEX` is the position within that array.
///
/// ZST, so expressions using it are Copy.
#[derive(Clone, Copy, Debug, Default)]
pub struct CtxVar<ArrayPos, const INDEX: usize>(PhantomData<ArrayPos>);

impl<ArrayPos, const INDEX: usize> CtxVar<ArrayPos, INDEX> {
    /// Create a new context variable reference.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<ArrayPos, const INDEX: usize> crate::ext::ManifoldExpr for CtxVar<ArrayPos, INDEX> {}

// ============================================================================
// Operator Implementations for CtxVar
// ============================================================================

impl<ArrayPos, const INDEX: usize, R> core::ops::Add<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Add<CtxVar<ArrayPos, INDEX>, R>;
    fn add(self, rhs: R) -> Self::Output {
        crate::ops::Add(self, rhs)
    }
}

impl<ArrayPos, const INDEX: usize, R> core::ops::Sub<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Sub<CtxVar<ArrayPos, INDEX>, R>;
    fn sub(self, rhs: R) -> Self::Output {
        crate::ops::Sub(self, rhs)
    }
}

impl<ArrayPos, const INDEX: usize, R> core::ops::Mul<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Mul<CtxVar<ArrayPos, INDEX>, R>;
    fn mul(self, rhs: R) -> Self::Output {
        crate::ops::Mul(self, rhs)
    }
}

impl<ArrayPos, const INDEX: usize, R> core::ops::Div<R> for CtxVar<ArrayPos, INDEX> {
    type Output = crate::ops::Div<CtxVar<ArrayPos, INDEX>, R>;
    fn div(self, rhs: R) -> Self::Output {
        crate::ops::Div(self, rhs)
    }
}

// ============================================================================
// 0-element context (special case - no arrays)
// ============================================================================

impl<P, B, Out> Manifold<P> for WithContext<(), B>
where
    P: Copy + Send + Sync,
    B: Manifold<P, Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        self.body.eval(p)
    }
}

impl<B: Copy> Copy for WithContext<(), B> {}

// ============================================================================
// Single Array Context: ([T; N],)
// ============================================================================

impl<T, const N: usize, P, B, Out> Manifold<P> for WithContext<([T; N],), B>
where
    P: Copy + Send + Sync,
    T: Copy + Send + Sync,
    B: Manifold<(([T; N],), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        self.body.eval((self.ctx, p))
    }
}

impl<T: Copy, const N: usize, B: Copy> Copy for WithContext<([T; N],), B> {}

// ============================================================================
// Two Array Context: ([T0; N], [T1; M])
// ============================================================================

impl<T0, T1, const N: usize, const M: usize, P, B, Out> Manifold<P>
    for WithContext<([T0; N], [T1; M]), B>
where
    P: Copy + Send + Sync,
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    B: Manifold<(([T0; N], [T1; M]), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        self.body.eval((self.ctx, p))
    }
}

impl<T0: Copy, T1: Copy, const N: usize, const M: usize, B: Copy> Copy
    for WithContext<([T0; N], [T1; M]), B>
{
}

// ============================================================================
// Three Array Context: ([T0; N], [T1; M], [T2; K])
// ============================================================================

impl<T0, T1, T2, const N: usize, const M: usize, const K: usize, P, B, Out> Manifold<P>
    for WithContext<([T0; N], [T1; M], [T2; K]), B>
where
    P: Copy + Send + Sync,
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    B: Manifold<(([T0; N], [T1; M], [T2; K]), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        self.body.eval((self.ctx, p))
    }
}

impl<T0: Copy, T1: Copy, T2: Copy, const N: usize, const M: usize, const K: usize, B: Copy> Copy
    for WithContext<([T0; N], [T1; M], [T2; K]), B>
{
}

// ============================================================================
// Four Array Context: ([T0; N], [T1; M], [T2; K], [T3; L])
// ============================================================================

impl<T0, T1, T2, T3, const N: usize, const M: usize, const K: usize, const L: usize, P, B, Out>
    Manifold<P> for WithContext<([T0; N], [T1; M], [T2; K], [T3; L]), B>
where
    P: Copy + Send + Sync,
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    B: Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        self.body.eval((self.ctx, p))
    }
}

impl<
        T0: Copy,
        T1: Copy,
        T2: Copy,
        T3: Copy,
        const N: usize,
        const M: usize,
        const K: usize,
        const L: usize,
        B: Copy,
    > Copy for WithContext<([T0; N], [T1; M], [T2; K], [T3; L]), B>
{
}

// ============================================================================
// CtxVar Manifold Implementations - Single Array Domain
// ============================================================================

impl<T, const N: usize, const INDEX: usize, P> Manifold<(([T; N],), P)> for CtxVar<A0, INDEX>
where
    T: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T;

    #[inline(always)]
    fn eval(&self, p: (([T; N],), P)) -> T {
        p.0 .0[INDEX]
    }
}

// ============================================================================
// CtxVar Manifold Implementations - Two Array Domain
// ============================================================================

impl<T0, T1, const N: usize, const M: usize, const INDEX: usize, P>
    Manifold<(([T0; N], [T1; M]), P)> for CtxVar<A0, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T0;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M]), P)) -> T0 {
        p.0 .0[INDEX]
    }
}

impl<T0, T1, const N: usize, const M: usize, const INDEX: usize, P>
    Manifold<(([T0; N], [T1; M]), P)> for CtxVar<A1, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T1;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M]), P)) -> T1 {
        p.0 .1[INDEX]
    }
}

// ============================================================================
// CtxVar Manifold Implementations - Three Array Domain
// ============================================================================

impl<T0, T1, T2, const N: usize, const M: usize, const K: usize, const INDEX: usize, P>
    Manifold<(([T0; N], [T1; M], [T2; K]), P)> for CtxVar<A0, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T0;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K]), P)) -> T0 {
        p.0 .0[INDEX]
    }
}

impl<T0, T1, T2, const N: usize, const M: usize, const K: usize, const INDEX: usize, P>
    Manifold<(([T0; N], [T1; M], [T2; K]), P)> for CtxVar<A1, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T1;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K]), P)) -> T1 {
        p.0 .1[INDEX]
    }
}

impl<T0, T1, T2, const N: usize, const M: usize, const K: usize, const INDEX: usize, P>
    Manifold<(([T0; N], [T1; M], [T2; K]), P)> for CtxVar<A2, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T2;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K]), P)) -> T2 {
        p.0 .2[INDEX]
    }
}

// ============================================================================
// CtxVar Manifold Implementations - Four Array Domain
// ============================================================================

impl<
        T0,
        T1,
        T2,
        T3,
        const N: usize,
        const M: usize,
        const K: usize,
        const L: usize,
        const INDEX: usize,
        P,
    > Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P)> for CtxVar<A0, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T0;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K], [T3; L]), P)) -> T0 {
        p.0 .0[INDEX]
    }
}

impl<
        T0,
        T1,
        T2,
        T3,
        const N: usize,
        const M: usize,
        const K: usize,
        const L: usize,
        const INDEX: usize,
        P,
    > Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P)> for CtxVar<A1, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T1;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K], [T3; L]), P)) -> T1 {
        p.0 .1[INDEX]
    }
}

impl<
        T0,
        T1,
        T2,
        T3,
        const N: usize,
        const M: usize,
        const K: usize,
        const L: usize,
        const INDEX: usize,
        P,
    > Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P)> for CtxVar<A2, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T2;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K], [T3; L]), P)) -> T2 {
        p.0 .2[INDEX]
    }
}

impl<
        T0,
        T1,
        T2,
        T3,
        const N: usize,
        const M: usize,
        const K: usize,
        const L: usize,
        const INDEX: usize,
        P,
    > Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P)> for CtxVar<A3, INDEX>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = T3;

    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K], [T3; L]), P)) -> T3 {
        p.0 .3[INDEX]
    }
}

// ============================================================================
// Spatial Implementations for Array-Based Context Domains
// ============================================================================

impl<T, const N: usize, P> Spatial for (([T; N],), P)
where
    P: Spatial,
{
    type Coord = P::Coord;

    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }

    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }

    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }

    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

impl<T0, T1, const N: usize, const M: usize, P> Spatial for (([T0; N], [T1; M]), P)
where
    P: Spatial,
{
    type Coord = P::Coord;

    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }

    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }

    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }

    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

impl<T0, T1, T2, const N: usize, const M: usize, const K: usize, P> Spatial
    for (([T0; N], [T1; M], [T2; K]), P)
where
    P: Spatial,
{
    type Coord = P::Coord;

    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }

    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }

    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }

    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

impl<T0, T1, T2, T3, const N: usize, const M: usize, const K: usize, const L: usize, P> Spatial
    for (([T0; N], [T1; M], [T2; K], [T3; L]), P)
where
    P: Spatial,
{
    type Coord = P::Coord;

    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }

    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }

    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }

    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}
