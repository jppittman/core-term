//! # Context Tuple Approach - Flat Alternative to Nested Let
//!
//! **Prototype**: Replace nested `Let<V0, Let<V1, Let<V2, Body>>>` with a single
//! combinator that holds all bindings in a flat tuple.
//!
//! ## Problem: Nested Let Trait Bound Explosion
//!
//! Current architecture creates nested Manifold impls that cause trait solver explosion.
//!
//! ## Solution: Flat Context Tuple
//!
//! `WithContext<(V0, V1, V2, V3, V4), Body>` - single Manifold impl, no recursion
//!
//! ## Benefits:
//! - **Flat trait bounds**: One impl per tuple size (not recursive)
//! - **Direct indexing**: CtxVar<0>, CtxVar<1>, etc use const generics  
//! - **Maintains CSE**: Each value computed once
//! - **Same runtime**: Monomorphizes identically

use crate::Manifold;

#[derive(Clone, Debug)]
pub struct WithContext<Ctx, Body> {
    pub ctx: Ctx,
    pub body: Body,
}

impl<Ctx, Body> WithContext<Ctx, Body> {
    pub const fn new(ctx: Ctx, body: Body) -> Self {
        Self { ctx, body }
    }
}

// 0-element context (empty tuple) - body evaluates directly with original domain
impl<P, B, Out> Manifold<P> for WithContext<(), B>
where
    P: Copy + Send + Sync,
    B: Manifold<P, Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        // No context to add - evaluate body directly
        self.body.eval(p)
    }
}

// Conditional Copy for WithContext<(), B> when B is Copy
impl<B: Copy> Copy for WithContext<(), B> {}

// 5-element context - KEY TEST (replaces 5 nested Lets)
impl<P, V0, V1, V2, V3, V4, B, O0, O1, O2, O3, O4, Out> Manifold<P>
    for WithContext<(V0, V1, V2, V3, V4), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    V4: Manifold<P, Output = O4>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    O4: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3, O4), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        let v4 = self.ctx.4.eval(p);
        self.body.eval(((v0, v1, v2, v3, v4), p))
    }
}

// CtxVar - type-level indexing into tuple (stable Rust compatible)
use core::marker::PhantomData;

#[derive(Clone, Copy, Debug, Default)]
pub struct CtxVar<N>(PhantomData<N>);

impl<N> CtxVar<N> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

// CtxVar is a Manifold expression (enables ManifoldExt methods)
impl<N> crate::ext::ManifoldExpr for CtxVar<N> {}

// Reuse our binary type-level numbers from binding module
use super::binding::{N0, N1, N2, N3, N4, N5, N6, N7, N8, N9};

// CtxVar<N0> - first element
impl<V0, V1, V2, V3, V4, P> Manifold<((V0, V1, V2, V3, V4), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4), P)) -> V0 {
        p.0.0
    }
}

// CtxVar<N1> - second element
impl<V0, V1, V2, V3, V4, P> Manifold<((V0, V1, V2, V3, V4), P)> for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4), P)) -> V1 {
        p.0.1
    }
}

// CtxVar<N2> - third element
impl<V0, V1, V2, V3, V4, P> Manifold<((V0, V1, V2, V3, V4), P)> for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4), P)) -> V2 {
        p.0.2
    }
}

// CtxVar<N3> - fourth element
impl<V0, V1, V2, V3, V4, P> Manifold<((V0, V1, V2, V3, V4), P)> for CtxVar<N3>
where
    V3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V3;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4), P)) -> V3 {
        p.0.3
    }
}

// CtxVar<N4> - fifth element
impl<V0, V1, V2, V3, V4, P> Manifold<((V0, V1, V2, V3, V4), P)> for CtxVar<N4>
where
    V4: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V4;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4), P)) -> V4 {
        p.0.4
    }
}

// 1-element context
impl<P, V0, B, O0, Out> Manifold<P> for WithContext<(V0,), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    O0: Copy + Send + Sync,
    B: Manifold<((O0,), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        self.body.eval(((v0,), p))
    }
}

// 2-element context
impl<P, V0, V1, B, O0, O1, Out> Manifold<P> for WithContext<(V0, V1), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    B: Manifold<((O0, O1), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        self.body.eval(((v0, v1), p))
    }
}

// 3-element context
impl<P, V0, V1, V2, B, O0, O1, O2, Out> Manifold<P> for WithContext<(V0, V1, V2), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        self.body.eval(((v0, v1, v2), p))
    }
}

// 4-element context
impl<P, V0, V1, V2, V3, B, O0, O1, O2, O3, Out> Manifold<P> for WithContext<(V0, V1, V2, V3), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        self.body.eval(((v0, v1, v2, v3), p))
    }
}

// 6-element context
impl<P, V0, V1, V2, V3, V4, V5, B, O0, O1, O2, O3, O4, O5, Out> Manifold<P>
    for WithContext<(V0, V1, V2, V3, V4, V5), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    V4: Manifold<P, Output = O4>,
    V5: Manifold<P, Output = O5>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    O4: Copy + Send + Sync,
    O5: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3, O4, O5), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        let v4 = self.ctx.4.eval(p);
        let v5 = self.ctx.5.eval(p);
        self.body.eval(((v0, v1, v2, v3, v4, v5), p))
    }
}

// 7-element context
impl<P, V0, V1, V2, V3, V4, V5, V6, B, O0, O1, O2, O3, O4, O5, O6, Out> Manifold<P>
    for WithContext<(V0, V1, V2, V3, V4, V5, V6), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    V4: Manifold<P, Output = O4>,
    V5: Manifold<P, Output = O5>,
    V6: Manifold<P, Output = O6>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    O4: Copy + Send + Sync,
    O5: Copy + Send + Sync,
    O6: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3, O4, O5, O6), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        let v4 = self.ctx.4.eval(p);
        let v5 = self.ctx.5.eval(p);
        let v6 = self.ctx.6.eval(p);
        self.body.eval(((v0, v1, v2, v3, v4, v5, v6), p))
    }
}

// 8-element context
impl<P, V0, V1, V2, V3, V4, V5, V6, V7, B, O0, O1, O2, O3, O4, O5, O6, O7, Out> Manifold<P>
    for WithContext<(V0, V1, V2, V3, V4, V5, V6, V7), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    V4: Manifold<P, Output = O4>,
    V5: Manifold<P, Output = O5>,
    V6: Manifold<P, Output = O6>,
    V7: Manifold<P, Output = O7>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    O4: Copy + Send + Sync,
    O5: Copy + Send + Sync,
    O6: Copy + Send + Sync,
    O7: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3, O4, O5, O6, O7), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        let v4 = self.ctx.4.eval(p);
        let v5 = self.ctx.5.eval(p);
        let v6 = self.ctx.6.eval(p);
        let v7 = self.ctx.7.eval(p);
        self.body.eval(((v0, v1, v2, v3, v4, v5, v6, v7), p))
    }
}

// 9-element context
impl<P, V0, V1, V2, V3, V4, V5, V6, V7, V8, B, O0, O1, O2, O3, O4, O5, O6, O7, O8, Out> Manifold<P>
    for WithContext<(V0, V1, V2, V3, V4, V5, V6, V7, V8), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    V4: Manifold<P, Output = O4>,
    V5: Manifold<P, Output = O5>,
    V6: Manifold<P, Output = O6>,
    V7: Manifold<P, Output = O7>,
    V8: Manifold<P, Output = O8>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    O4: Copy + Send + Sync,
    O5: Copy + Send + Sync,
    O6: Copy + Send + Sync,
    O7: Copy + Send + Sync,
    O8: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3, O4, O5, O6, O7, O8), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        let v4 = self.ctx.4.eval(p);
        let v5 = self.ctx.5.eval(p);
        let v6 = self.ctx.6.eval(p);
        let v7 = self.ctx.7.eval(p);
        let v8 = self.ctx.8.eval(p);
        self.body.eval(((v0, v1, v2, v3, v4, v5, v6, v7, v8), p))
    }
}

// 10-element context
impl<P, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, B, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, Out>
    Manifold<P> for WithContext<(V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    V4: Manifold<P, Output = O4>,
    V5: Manifold<P, Output = O5>,
    V6: Manifold<P, Output = O6>,
    V7: Manifold<P, Output = O7>,
    V8: Manifold<P, Output = O8>,
    V9: Manifold<P, Output = O9>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    O4: Copy + Send + Sync,
    O5: Copy + Send + Sync,
    O6: Copy + Send + Sync,
    O7: Copy + Send + Sync,
    O8: Copy + Send + Sync,
    O9: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3, O4, O5, O6, O7, O8, O9), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        let v4 = self.ctx.4.eval(p);
        let v5 = self.ctx.5.eval(p);
        let v6 = self.ctx.6.eval(p);
        let v7 = self.ctx.7.eval(p);
        let v8 = self.ctx.8.eval(p);
        let v9 = self.ctx.9.eval(p);
        self.body
            .eval(((v0, v1, v2, v3, v4, v5, v6, v7, v8, v9), p))
    }
}

// CtxVar impls for 1-element tuples
impl<V0, P> Manifold<((V0,), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0,), P)) -> V0 {
        p.0.0
    }
}

// CtxVar impls for 2-element tuples
impl<V0, V1, P> Manifold<((V0, V1), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, P> Manifold<((V0, V1), P)> for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1), P)) -> V1 {
        p.0.1
    }
}

// CtxVar impls for 3-element tuples
impl<V0, V1, V2, P> Manifold<((V0, V1, V2), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, V2, P> Manifold<((V0, V1, V2), P)> for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2), P)) -> V1 {
        p.0.1
    }
}

impl<V0, V1, V2, P> Manifold<((V0, V1, V2), P)> for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2), P)) -> V2 {
        p.0.2
    }
}

// CtxVar impls for 4-element tuples
impl<V0, V1, V2, V3, P> Manifold<((V0, V1, V2, V3), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, V2, V3, P> Manifold<((V0, V1, V2, V3), P)> for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3), P)) -> V1 {
        p.0.1
    }
}

impl<V0, V1, V2, V3, P> Manifold<((V0, V1, V2, V3), P)> for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3), P)) -> V2 {
        p.0.2
    }
}

impl<V0, V1, V2, V3, P> Manifold<((V0, V1, V2, V3), P)> for CtxVar<N3>
where
    V3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V3;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3), P)) -> V3 {
        p.0.3
    }
}

// CtxVar impls for 6-element tuples
impl<V0, V1, V2, V3, V4, V5, P> Manifold<((V0, V1, V2, V3, V4, V5), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, V2, V3, V4, V5, P> Manifold<((V0, V1, V2, V3, V4, V5), P)> for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5), P)) -> V1 {
        p.0.1
    }
}

impl<V0, V1, V2, V3, V4, V5, P> Manifold<((V0, V1, V2, V3, V4, V5), P)> for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5), P)) -> V2 {
        p.0.2
    }
}

impl<V0, V1, V2, V3, V4, V5, P> Manifold<((V0, V1, V2, V3, V4, V5), P)> for CtxVar<N3>
where
    V3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V3;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5), P)) -> V3 {
        p.0.3
    }
}

impl<V0, V1, V2, V3, V4, V5, P> Manifold<((V0, V1, V2, V3, V4, V5), P)> for CtxVar<N4>
where
    V4: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V4;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5), P)) -> V4 {
        p.0.4
    }
}

impl<V0, V1, V2, V3, V4, V5, P> Manifold<((V0, V1, V2, V3, V4, V5), P)> for CtxVar<N5>
where
    V5: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V5;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5), P)) -> V5 {
        p.0.5
    }
}

// CtxVar impls for 7-element tuples
impl<V0, V1, V2, V3, V4, V5, V6, P> Manifold<((V0, V1, V2, V3, V4, V5, V6), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, P> Manifold<((V0, V1, V2, V3, V4, V5, V6), P)> for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6), P)) -> V1 {
        p.0.1
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, P> Manifold<((V0, V1, V2, V3, V4, V5, V6), P)> for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6), P)) -> V2 {
        p.0.2
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, P> Manifold<((V0, V1, V2, V3, V4, V5, V6), P)> for CtxVar<N3>
where
    V3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V3;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6), P)) -> V3 {
        p.0.3
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, P> Manifold<((V0, V1, V2, V3, V4, V5, V6), P)> for CtxVar<N4>
where
    V4: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V4;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6), P)) -> V4 {
        p.0.4
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, P> Manifold<((V0, V1, V2, V3, V4, V5, V6), P)> for CtxVar<N5>
where
    V5: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V5;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6), P)) -> V5 {
        p.0.5
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, P> Manifold<((V0, V1, V2, V3, V4, V5, V6), P)> for CtxVar<N6>
where
    V6: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V6;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6), P)) -> V6 {
        p.0.6
    }
}

// CtxVar impls for 8-element tuples
impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V1 {
        p.0.1
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V2 {
        p.0.2
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N3>
where
    V3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V3;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V3 {
        p.0.3
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N4>
where
    V4: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V4;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V4 {
        p.0.4
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N5>
where
    V5: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V5;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V5 {
        p.0.5
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N6>
where
    V6: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V6;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V6 {
        p.0.6
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>
    for CtxVar<N7>
where
    V7: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V7;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> V7 {
        p.0.7
    }
}

// CtxVar impls for 9-element tuples
impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V1 {
        p.0.1
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V2 {
        p.0.2
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N3>
where
    V3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V3;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V3 {
        p.0.3
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N4>
where
    V4: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V4;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V4 {
        p.0.4
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N5>
where
    V5: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V5;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V5 {
        p.0.5
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N6>
where
    V6: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V6;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V6 {
        p.0.6
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N7>
where
    V7: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V7;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V7 {
        p.0.7
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>
    for CtxVar<N8>
where
    V8: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V8;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> V8 {
        p.0.8
    }
}

// CtxVar impls for 10-element tuples
impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N0>
where
    V0: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V0;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V0 {
        p.0.0
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N1>
where
    V1: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V1;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V1 {
        p.0.1
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N2>
where
    V2: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V2;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V2 {
        p.0.2
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N3>
where
    V3: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V3;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V3 {
        p.0.3
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N4>
where
    V4: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V4;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V4 {
        p.0.4
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N5>
where
    V5: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V5;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V5 {
        p.0.5
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N6>
where
    V6: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V6;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V6 {
        p.0.6
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N7>
where
    V7: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V7;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V7 {
        p.0.7
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N8>
where
    V8: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V8;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V8 {
        p.0.8
    }
}

impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P>
    Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)> for CtxVar<N9>
where
    V9: Copy + Send + Sync,
    P: Copy + Send + Sync,
{
    type Output = V9;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)) -> V9 {
        p.0.9
    }
}

// ============================================================================
// Arithmetic Operators for CtxVar
// ============================================================================
//
// These blanket impls allow CtxVar to be used in arithmetic expressions.
// They construct Manifold AST nodes (Add, Sub, Mul, Div) without requiring
// that the RHS be a Manifold - validation happens at evaluation time.

impl<N, Rhs> core::ops::Add<Rhs> for CtxVar<N> {
    type Output = crate::ops::Add<CtxVar<N>, Rhs>;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output {
        crate::ops::Add(self, rhs)
    }
}

impl<N, Rhs> core::ops::Sub<Rhs> for CtxVar<N> {
    type Output = crate::ops::Sub<CtxVar<N>, Rhs>;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output {
        crate::ops::Sub(self, rhs)
    }
}

impl<N, Rhs> core::ops::Mul<Rhs> for CtxVar<N> {
    type Output = crate::ops::Mul<CtxVar<N>, Rhs>;
    #[inline(always)]
    fn mul(self, rhs: Rhs) -> Self::Output {
        crate::ops::Mul(self, rhs)
    }
}

impl<N, Rhs> core::ops::Div<Rhs> for CtxVar<N> {
    type Output = crate::ops::Div<CtxVar<N>, Rhs>;
    #[inline(always)]
    fn div(self, rhs: Rhs) -> Self::Output {
        crate::ops::Div(self, rhs)
    }
}

impl<N> core::ops::Neg for CtxVar<N> {
    type Output = crate::ops::Neg<CtxVar<N>>;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        crate::ops::Neg(self)
    }
}

// ============================================================================
// Spatial Trait Implementations for Tuple Domains
// ============================================================================
//
// These implementations allow X, Y, Z, W to access coordinates from the
// extended domain ((V0, V1, ...), P) by delegating to the P component.

// 1-element tuple domain
impl<V0, P> crate::domain::Spatial for ((V0,), P)
where
    P: crate::domain::Spatial,
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

// 2-element tuple domain
impl<V0, V1, P> crate::domain::Spatial for ((V0, V1), P)
where
    P: crate::domain::Spatial,
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

// 3-element tuple domain
impl<V0, V1, V2, P> crate::domain::Spatial for ((V0, V1, V2), P)
where
    P: crate::domain::Spatial,
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

// 4-element tuple domain
impl<V0, V1, V2, V3, P> crate::domain::Spatial for ((V0, V1, V2, V3), P)
where
    P: crate::domain::Spatial,
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

// 5-element tuple domain
impl<V0, V1, V2, V3, V4, P> crate::domain::Spatial for ((V0, V1, V2, V3, V4), P)
where
    P: crate::domain::Spatial,
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

// 6-element tuple domain
impl<V0, V1, V2, V3, V4, V5, P> crate::domain::Spatial for ((V0, V1, V2, V3, V4, V5), P)
where
    P: crate::domain::Spatial,
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

// 7-element tuple domain
impl<V0, V1, V2, V3, V4, V5, V6, P> crate::domain::Spatial for ((V0, V1, V2, V3, V4, V5, V6), P)
where
    P: crate::domain::Spatial,
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

// 8-element tuple domain
impl<V0, V1, V2, V3, V4, V5, V6, V7, P> crate::domain::Spatial
    for ((V0, V1, V2, V3, V4, V5, V6, V7), P)
where
    P: crate::domain::Spatial,
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

// 9-element tuple domain
impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P> crate::domain::Spatial
    for ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)
where
    P: crate::domain::Spatial,
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

// 10-element tuple domain
impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, P> crate::domain::Spatial
    for ((V0, V1, V2, V3, V4, V5, V6, V7, V8, V9), P)
where
    P: crate::domain::Spatial,
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
