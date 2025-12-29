//! # Chained Operator Overloads
//!
//! Enables arithmetic chaining without boxing, e.g., `(X + Y) * Z`.
//!
//! ## FMA Fusion
//!
//! When `Mul<A, B> + C` is written, it returns `MulAdd<A, B, C>` instead of
//! `Add<Mul<A, B>, C>`. This enables automatic compile-time fusion into FMA
//! instructions.
//!
//! Note: Symmetric fusion (`C + Mul<A, B>`) would require specialization,
//! which is unstable. Write `A * B + C` (multiply first) to get FMA.

use super::{Abs, Add, AddMasked, Cos, Div, Floor, Max, Min, Mul, MulAdd, MulRecip, MulRsqrt, Rsqrt, Sin, Sqrt, Sub};
use crate::Manifold;
use crate::combinators::Select;

// ============================================================================
// Standard chained ops macro (for types that DON'T need special FMA handling)
// ============================================================================

macro_rules! impl_chained_ops {
    ($ty:ident <$($gen:ident),*>) => {
        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Add<Rhs> for $ty<$($gen),*> {
            type Output = Add<Self, Rhs>;
            #[inline(always)]
            fn add(self, rhs: Rhs) -> Self::Output { Add(self, rhs) }
        }

        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Sub<Rhs> for $ty<$($gen),*> {
            type Output = Sub<Self, Rhs>;
            #[inline(always)]
            fn sub(self, rhs: Rhs) -> Self::Output { Sub(self, rhs) }
        }

        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Mul<Rhs> for $ty<$($gen),*> {
            type Output = Mul<Self, Rhs>;
            #[inline(always)]
            fn mul(self, rhs: Rhs) -> Self::Output { Mul(self, rhs) }
        }

        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Div<Rhs> for $ty<$($gen),*> {
            type Output = Div<Self, Rhs>;
            #[inline(always)]
            fn div(self, rhs: Rhs) -> Self::Output { Div(self, rhs) }
        }
    };
}

// ============================================================================
// FMA Fusion: Mul + Rhs → MulAdd
// ============================================================================

// Mul gets special treatment: Mul + Rhs becomes MulAdd
impl<L: Manifold, R: Manifold, Rhs: Manifold> core::ops::Add<Rhs> for Mul<L, R> {
    type Output = MulAdd<L, R, Rhs>;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output {
        MulAdd(self.0, self.1, rhs)
    }
}

impl<L: Manifold, R: Manifold, Rhs: Manifold> core::ops::Sub<Rhs> for Mul<L, R> {
    type Output = Sub<Self, Rhs>;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output { Sub(self, rhs) }
}

impl<L: Manifold, R: Manifold, Rhs: Manifold> core::ops::Mul<Rhs> for Mul<L, R> {
    type Output = Mul<Self, Rhs>;
    #[inline(always)]
    fn mul(self, rhs: Rhs) -> Self::Output { Mul(self, rhs) }
}

impl<L: Manifold, R: Manifold, Rhs: Manifold> core::ops::Div<Rhs> for Mul<L, R> {
    type Output = Div<Self, Rhs>;
    #[inline(always)]
    fn div(self, rhs: Rhs) -> Self::Output { Div(self, rhs) }
}

// ============================================================================
// Standard chained ops for other types
// ============================================================================

// Binary nodes (without Mul, which has special handling above)
impl_chained_ops!(Add<L, R>);
impl_chained_ops!(Sub<L, R>);
impl_chained_ops!(Div<L, R>);
impl_chained_ops!(Max<L, R>);
impl_chained_ops!(Min<L, R>);

// Unary nodes
impl_chained_ops!(Sqrt<M>);
impl_chained_ops!(Abs<M>);
impl_chained_ops!(Floor<M>);
impl_chained_ops!(Rsqrt<M>);
impl_chained_ops!(Sin<M>);
impl_chained_ops!(Cos<M>);

// Combinators
impl_chained_ops!(Select<C, T, F>);

// MulAdd and other compound ops need chained ops for further composition
impl_chained_ops!(MulAdd<A, B, C>);
impl_chained_ops!(MulRecip<M>);
impl_chained_ops!(MulRsqrt<L, R>);
impl_chained_ops!(AddMasked<Acc, Val, Mask>);

// ============================================================================
// Thunk Operators
// ============================================================================

// Thunk needs manual impls because its Manifold bound is on F's return type
impl<F: Fn() -> M + Send + Sync, M: Manifold, Rhs: Manifold> core::ops::Add<Rhs> for crate::Thunk<F> {
    type Output = Add<Self, Rhs>;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output { Add(self, rhs) }
}

impl<F: Fn() -> M + Send + Sync, M: Manifold, Rhs: Manifold> core::ops::Sub<Rhs> for crate::Thunk<F> {
    type Output = Sub<Self, Rhs>;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output { Sub(self, rhs) }
}

impl<F: Fn() -> M + Send + Sync, M: Manifold, Rhs: Manifold> core::ops::Mul<Rhs> for crate::Thunk<F> {
    type Output = Mul<Self, Rhs>;
    #[inline(always)]
    fn mul(self, rhs: Rhs) -> Self::Output { Mul(self, rhs) }
}

impl<F: Fn() -> M + Send + Sync, M: Manifold, Rhs: Manifold> core::ops::Div<Rhs> for crate::Thunk<F> {
    type Output = Div<Self, Rhs>;
    #[inline(always)]
    fn div(self, rhs: Rhs) -> Self::Output { Div(self, rhs) }
}
