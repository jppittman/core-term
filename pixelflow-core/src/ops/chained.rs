//! # Chained Operator Overloads
//!
//! Enables arithmetic chaining without boxing, e.g., `(X + Y) * Z`.
//!
//! # Generalized Implementation
//!
//! Previously, this module contained manual operator fusion logic (creating `MulAdd`
//! from `Mul + Add` and `MulRsqrt` from `Div + Sqrt`).
//!
//! This has been removed in favor of a purely algebraic approach:
//! 1. Operators produce standard AST nodes (e.g., `Add(Mul(..), ..)`).
//! 2. The E-Graph optimizer discovers fusions like `MulAdd` and `MulRsqrt` automatically
//!    during the optimization pass.
//!
//! This simplifies the trait implementations significantly, as we no longer need
//! to explicitly enumerate divisor types to avoid conflict with specialized fusion traits.

use super::{
    Abs, Add, AddMasked, Cos, Div, Exp, Exp2, Floor, Ge, Gt, Le, Log2, Lt, Max, Min, Mul, MulAdd,
    MulRecip, MulRsqrt, Neg, Rsqrt, Sin, Sqrt, Sub,
};
use super::logic::{And, Or};
use super::derivative::{
    Antialias2D, Antialias3D, Curvature2D, GradientMag2D, GradientMag3D,
    // Simple accessor combinators
    DxOf, DxxOf, DxyOf, DyOf, DyyOf, DzOf, ValOf,
};
use crate::combinators::Select;
use crate::combinators::binding::{Let, Var};

// ============================================================================
// Generic Arithmetic Implementation
// ============================================================================

macro_rules! impl_arithmetic {
    ($ty:ident <$($gen:ident),*>) => {
        impl<$($gen,)* Rhs> core::ops::Add<Rhs> for $ty<$($gen),*> {
            type Output = Add<Self, Rhs>;
            #[inline(always)]
            fn add(self, rhs: Rhs) -> Self::Output { Add(self, rhs) }
        }

        impl<$($gen,)* Rhs> core::ops::Sub<Rhs> for $ty<$($gen),*> {
            type Output = Sub<Self, Rhs>;
            #[inline(always)]
            fn sub(self, rhs: Rhs) -> Self::Output { Sub(self, rhs) }
        }

        impl<$($gen,)* Rhs> core::ops::Mul<Rhs> for $ty<$($gen),*> {
            type Output = Mul<Self, Rhs>;
            #[inline(always)]
            fn mul(self, rhs: Rhs) -> Self::Output { Mul(self, rhs) }
        }

        impl<$($gen,)* Rhs> core::ops::Div<Rhs> for $ty<$($gen),*> {
            type Output = Div<Self, Rhs>;
            #[inline(always)]
            fn div(self, rhs: Rhs) -> Self::Output { Div(self, rhs) }
        }
    };
}

/// Helper to implement all arithmetic ops and the ManifoldExpr marker
macro_rules! impl_chained_ops {
    ($ty:ident <$($gen:ident),*>) => {
        impl_arithmetic!($ty<$($gen),*>);
        impl<$($gen),*> crate::ManifoldExpr for $ty<$($gen),*> {}
    };
}

// ============================================================================
// Standard AST Nodes
// ============================================================================

// Binary nodes
impl_chained_ops!(Add<L, R>);
impl_chained_ops!(Sub<L, R>);
impl_chained_ops!(Mul<L, R>); // Mul is now just a standard node!
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
impl_chained_ops!(Log2<M>);
impl_chained_ops!(Exp2<M>);
impl_chained_ops!(Exp<M>);

// Combinators
impl_chained_ops!(Select<C, T, F>);

// Compound ops (still supported as types, but created via optimization)
impl_chained_ops!(MulAdd<A, B, C>);
impl_chained_ops!(MulRecip<M>);
impl_chained_ops!(MulRsqrt<L, R>);
impl_chained_ops!(AddMasked<Acc, Val, Mask>);

// Fused derivative combinators
impl_chained_ops!(GradientMag2D<M>);
impl_chained_ops!(GradientMag3D<M>);
impl_chained_ops!(Antialias2D<M>);
impl_chained_ops!(Antialias3D<M>);
impl_chained_ops!(Curvature2D<M>);

// Simple accessor combinators
impl_chained_ops!(ValOf<M>);
impl_chained_ops!(DxOf<M>);
impl_chained_ops!(DyOf<M>);
impl_chained_ops!(DzOf<M>);
impl_chained_ops!(DxxOf<M>);
impl_chained_ops!(DxyOf<M>);
impl_chained_ops!(DyyOf<M>);

// ============================================================================
// ManifoldExpr Only (Nodes that don't support chaining or handled elsewhere)
// ============================================================================

// Additional unary ops (maybe these should support chaining too? Keeping as is for now)
impl<M> crate::ManifoldExpr for Neg<M> {}

// Comparison ops
impl<L, R> crate::ManifoldExpr for Lt<L, R> {}
impl<L, R> crate::ManifoldExpr for Gt<L, R> {}
impl<L, R> crate::ManifoldExpr for Le<L, R> {}
impl<L, R> crate::ManifoldExpr for Ge<L, R> {}

// Logic ops
impl<L, R> crate::ManifoldExpr for And<L, R> {}
impl<L, R> crate::ManifoldExpr for Or<L, R> {}

// At combinator
impl<Cx, Cy, Cz, Cw, M> crate::ManifoldExpr for crate::combinators::At<Cx, Cy, Cz, Cw, M> {}

// Binding combinators
impl<N> crate::ManifoldExpr for Var<N> {}
impl<V, B> crate::ManifoldExpr for Let<V, B> {}

// Thunk
impl<F: Fn() -> M + Send + Sync, M> crate::ManifoldExpr for crate::Thunk<F> {}

// ============================================================================
// Thunk Operators
// ============================================================================

// Thunk needs manual impls because its Manifold bound is on F's return type
// But now we can use generic Rhs!

impl<F: Fn() -> M + Send + Sync, M, Rhs> core::ops::Add<Rhs> for crate::Thunk<F> {
    type Output = Add<Self, Rhs>;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output { Add(self, rhs) }
}

impl<F: Fn() -> M + Send + Sync, M, Rhs> core::ops::Sub<Rhs> for crate::Thunk<F> {
    type Output = Sub<Self, Rhs>;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output { Sub(self, rhs) }
}

impl<F: Fn() -> M + Send + Sync, M, Rhs> core::ops::Mul<Rhs> for crate::Thunk<F> {
    type Output = Mul<Self, Rhs>;
    #[inline(always)]
    fn mul(self, rhs: Rhs) -> Self::Output { Mul(self, rhs) }
}

impl<F: Fn() -> M + Send + Sync, M, Rhs> core::ops::Div<Rhs> for crate::Thunk<F> {
    type Output = Div<Self, Rhs>;
    #[inline(always)]
    fn div(self, rhs: Rhs) -> Self::Output { Div(self, rhs) }
}