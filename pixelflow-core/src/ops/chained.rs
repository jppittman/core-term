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
//!
//! ## Rsqrt Fusion
//!
//! When `L / Sqrt<R>` is written, it returns `MulRsqrt<L, R>` instead of
//! `Div<L, Sqrt<R>>`. This uses fast rsqrt (~3 cycles) instead of
//! sqrt (~12 cycles) + div (~12 cycles).

use super::derivative::{
    Antialias2D,
    Antialias3D,
    Curvature2D,
    // Simple accessor combinators
    DxOf,
    DxxOf,
    DxyOf,
    DyOf,
    DyyOf,
    DzOf,
    GradientMag2D,
    GradientMag3D,
    ValOf,
};
use super::logic::{And, Or};
use super::{
    Abs, Add, AddMasked, Cos, Div, Exp2, Floor, Ge, Gt, Le, Log2, Lt, Max, Min, Mul, MulAdd,
    MulRecip, MulRsqrt, Neg, Rsqrt, Sin, Sqrt, Sub,
};
use crate::Field;
use crate::combinators::CtxVar;
use crate::combinators::Select;
use crate::combinators::binding::{Let, Var};
use crate::variables::{W, X, Y, Z};

// ============================================================================
// Add, Sub, Mul (no Div - that's handled separately for rsqrt fusion)
// ============================================================================
//
// Note: No `Manifold` bound on Rhs! Operators construct AST nodes without
// validating that operands are manifolds. Validation happens at evaluation time.
// This allows expressions with Var<N> (which only implements Manifold for
// domains with Head trait).

macro_rules! impl_add_sub_mul {
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
    };
}

// ============================================================================
// Rsqrt Fusion: L / Sqrt<R> → MulRsqrt<L, R>
// ============================================================================
//
// To avoid conflicting with a generic `Div<Rhs>`, we enumerate
// all divisor types explicitly. This is the same pattern as FMA fusion:
// Mul doesn't have a generic Add impl, so MulAdd doesn't conflict.

macro_rules! impl_rsqrt_fusion {
    ($ty:ident <$($gen:ident),*>) => {
        impl<$($gen,)* SqrtInner> core::ops::Div<Sqrt<SqrtInner>> for $ty<$($gen),*> {
            type Output = MulRsqrt<Self, SqrtInner>;
            #[inline(always)]
            fn div(self, rhs: Sqrt<SqrtInner>) -> Self::Output { MulRsqrt(self, rhs.0) }
        }
    };
}

/// Generate Div impl for a specific divisor type (non-Sqrt)
macro_rules! impl_div_for {
    // Self<...> / Divisor<...> (generic divisor)
    ($self_ty:ident <$($sg:ident),*> / $div_ty:ident <$($dg:ident),*>) => {
        impl<$($sg,)* $($dg),*> core::ops::Div<$div_ty<$($dg),*>> for $self_ty<$($sg),*> {
            type Output = Div<Self, $div_ty<$($dg),*>>;
            #[inline(always)]
            fn div(self, rhs: $div_ty<$($dg),*>) -> Self::Output { Div(self, rhs) }
        }
    };
    // Self<...> / ConcreteDivisor (no generics on divisor)
    ($self_ty:ident <$($sg:ident),*> / @ $div_ty:ty) => {
        impl<$($sg),*> core::ops::Div<$div_ty> for $self_ty<$($sg),*> {
            type Output = Div<Self, $div_ty>;
            #[inline(always)]
            fn div(self, rhs: $div_ty) -> Self::Output { Div(self, rhs) }
        }
    };
}

/// Generate all Div impls for a type (rsqrt fusion + all other divisors)
macro_rules! impl_all_divs {
    ($ty:ident <$($gen:ident),*>) => {
        // Rsqrt fusion: / Sqrt → MulRsqrt
        impl_rsqrt_fusion!($ty<$($gen),*>);

        // Generic divisor types → normal Div
        impl_div_for!($ty<$($gen),*> / Add<DL, DR>);
        impl_div_for!($ty<$($gen),*> / Sub<DL, DR>);
        impl_div_for!($ty<$($gen),*> / Mul<DL, DR>);
        impl_div_for!($ty<$($gen),*> / Div<DL, DR>);
        impl_div_for!($ty<$($gen),*> / Max<DL, DR>);
        impl_div_for!($ty<$($gen),*> / Min<DL, DR>);
        impl_div_for!($ty<$($gen),*> / Abs<DM>);
        impl_div_for!($ty<$($gen),*> / Floor<DM>);
        impl_div_for!($ty<$($gen),*> / Rsqrt<DM>);
        impl_div_for!($ty<$($gen),*> / Sin<DM>);
        impl_div_for!($ty<$($gen),*> / Cos<DM>);
        impl_div_for!($ty<$($gen),*> / Select<DC, DT, DF>);
        impl_div_for!($ty<$($gen),*> / MulAdd<DA, DB, DC2>);
        impl_div_for!($ty<$($gen),*> / MulRecip<DM2>);
        impl_div_for!($ty<$($gen),*> / MulRsqrt<DL2, DR2>);
        impl_div_for!($ty<$($gen),*> / AddMasked<DAcc, DVal, DMask>);
        impl_div_for!($ty<$($gen),*> / Var<DN>);
        impl_div_for!($ty<$($gen),*> / CtxVar<DCN>);

        // Fused derivative combinators
        impl_div_for!($ty<$($gen),*> / GradientMag2D<DDM>);
        impl_div_for!($ty<$($gen),*> / GradientMag3D<DDM2>);
        impl_div_for!($ty<$($gen),*> / Antialias2D<DDM3>);
        impl_div_for!($ty<$($gen),*> / Antialias3D<DDM4>);
        impl_div_for!($ty<$($gen),*> / Curvature2D<DDM5>);

        // Simple accessor combinators
        impl_div_for!($ty<$($gen),*> / ValOf<DDA1>);
        impl_div_for!($ty<$($gen),*> / DxOf<DDA2>);
        impl_div_for!($ty<$($gen),*> / DyOf<DDA3>);
        impl_div_for!($ty<$($gen),*> / DzOf<DDA4>);
        impl_div_for!($ty<$($gen),*> / DxxOf<DDA5>);
        impl_div_for!($ty<$($gen),*> / DxyOf<DDA6>);
        impl_div_for!($ty<$($gen),*> / DyyOf<DDA7>);

        // Concrete divisor types (base variables, Field, scalars)
        impl_div_for!($ty<$($gen),*> / @X);
        impl_div_for!($ty<$($gen),*> / @Y);
        impl_div_for!($ty<$($gen),*> / @Z);
        impl_div_for!($ty<$($gen),*> / @W);
        impl_div_for!($ty<$($gen),*> / @Field);
        impl_div_for!($ty<$($gen),*> / @f32);
        impl_div_for!($ty<$($gen),*> / @i32);
    };
}

/// Complete chained ops: Add, Sub, Mul + all Divs with rsqrt fusion
macro_rules! impl_chained_ops {
    ($ty:ident <$($gen:ident),*>) => {
        impl_add_sub_mul!($ty<$($gen),*>);
        impl_all_divs!($ty<$($gen),*>);
    };
}

// ============================================================================
// FMA Fusion: Mul + Rhs → MulAdd
// ============================================================================

// Mul gets special treatment: Mul + Rhs becomes MulAdd
// Note: No Manifold bounds - operators construct AST nodes without validation
impl<L, R, Rhs> core::ops::Add<Rhs> for Mul<L, R> {
    type Output = MulAdd<L, R, Rhs>;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output {
        MulAdd(self.0, self.1, rhs)
    }
}

impl<L, R, Rhs> core::ops::Sub<Rhs> for Mul<L, R> {
    type Output = Sub<Self, Rhs>;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output {
        Sub(self, rhs)
    }
}

impl<L, R, Rhs> core::ops::Mul<Rhs> for Mul<L, R> {
    type Output = Mul<Self, Rhs>;
    #[inline(always)]
    fn mul(self, rhs: Rhs) -> Self::Output {
        Mul(self, rhs)
    }
}

// Mul gets rsqrt fusion too (enumerate all divisor types to avoid conflict)
impl_all_divs!(Mul<L, R>);

// ============================================================================
// ManifoldExpr marker trait - gates access to ManifoldExt methods
// ============================================================================

macro_rules! impl_manifold_expr {
    ($ty:ident <$($gen:ident),*>) => {
        impl<$($gen),*> crate::ManifoldExpr for $ty<$($gen),*> {}
    };
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

// Fused derivative combinators (GradientMag, Antialias, Curvature)
impl_chained_ops!(GradientMag2D<M>);
impl_chained_ops!(GradientMag3D<M>);
impl_chained_ops!(Antialias2D<M>);
impl_chained_ops!(Antialias3D<M>);
impl_chained_ops!(Curvature2D<M>);

// Simple accessor combinators (V, DX, DY, DZ, DXX, DXY, DYY)
impl_chained_ops!(ValOf<M>);
impl_chained_ops!(DxOf<M>);
impl_chained_ops!(DyOf<M>);
impl_chained_ops!(DzOf<M>);
impl_chained_ops!(DxxOf<M>);
impl_chained_ops!(DxyOf<M>);
impl_chained_ops!(DyyOf<M>);

// ============================================================================
// ManifoldExpr implementations for all combinator types
// ============================================================================

// Mul gets ManifoldExpr too (even though it has special chained ops handling)
impl_manifold_expr!(Mul<L, R>);

// Binary nodes
impl_manifold_expr!(Add<L, R>);
impl_manifold_expr!(Sub<L, R>);
impl_manifold_expr!(Div<L, R>);
impl_manifold_expr!(Max<L, R>);
impl_manifold_expr!(Min<L, R>);

// Unary nodes
impl_manifold_expr!(Sqrt<M>);
impl_manifold_expr!(Abs<M>);
impl_manifold_expr!(Floor<M>);
impl_manifold_expr!(Rsqrt<M>);
impl_manifold_expr!(Sin<M>);
impl_manifold_expr!(Cos<M>);

// Select combinator
impl_manifold_expr!(Select<C, T, F>);

// Compound ops
impl_manifold_expr!(MulAdd<A, B, C>);
impl_manifold_expr!(MulRecip<M>);
impl_manifold_expr!(MulRsqrt<L, R>);
impl_manifold_expr!(AddMasked<Acc, Val, Mask>);

// Fused derivative combinators
impl_manifold_expr!(GradientMag2D<M>);
impl_manifold_expr!(GradientMag3D<M>);
impl_manifold_expr!(Antialias2D<M>);
impl_manifold_expr!(Antialias3D<M>);
impl_manifold_expr!(Curvature2D<M>);

// Simple accessor combinators
impl_manifold_expr!(ValOf<M>);
impl_manifold_expr!(DxOf<M>);
impl_manifold_expr!(DyOf<M>);
impl_manifold_expr!(DzOf<M>);
impl_manifold_expr!(DxxOf<M>);
impl_manifold_expr!(DxyOf<M>);
impl_manifold_expr!(DyyOf<M>);

// Additional unary ops
impl_manifold_expr!(Neg<M>);
impl_manifold_expr!(Log2<M>);
impl_manifold_expr!(Exp2<M>);

// Comparison ops
impl_manifold_expr!(Lt<L, R>);
impl_manifold_expr!(Gt<L, R>);
impl_manifold_expr!(Le<L, R>);
impl_manifold_expr!(Ge<L, R>);

// Logic ops
impl_manifold_expr!(And<L, R>);
impl_manifold_expr!(Or<L, R>);

// At combinator (coordinate remapping)
impl<Cx, Cy, Cz, Cw, M> crate::ManifoldExpr for crate::combinators::At<Cx, Cy, Cz, Cw, M> {}

// Binding combinators
impl_manifold_expr!(Var<N>);
impl_manifold_expr!(Let<V, B>);

// Thunk gets ManifoldExpr too
impl<F: Fn() -> M + Send + Sync, M> crate::ManifoldExpr for crate::Thunk<F> {}

// ============================================================================
// Thunk Operators
// ============================================================================

// Thunk needs manual impls because its Manifold bound is on F's return type
// Note: No Manifold bound on Rhs - operators construct AST nodes without validation
impl<F: Fn() -> M + Send + Sync, M, Rhs> core::ops::Add<Rhs> for crate::Thunk<F> {
    type Output = Add<Self, Rhs>;
    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output {
        Add(self, rhs)
    }
}

impl<F: Fn() -> M + Send + Sync, M, Rhs> core::ops::Sub<Rhs> for crate::Thunk<F> {
    type Output = Sub<Self, Rhs>;
    #[inline(always)]
    fn sub(self, rhs: Rhs) -> Self::Output {
        Sub(self, rhs)
    }
}

impl<F: Fn() -> M + Send + Sync, M, Rhs> core::ops::Mul<Rhs> for crate::Thunk<F> {
    type Output = Mul<Self, Rhs>;
    #[inline(always)]
    fn mul(self, rhs: Rhs) -> Self::Output {
        Mul(self, rhs)
    }
}

// Rsqrt fusion: Thunk / Sqrt<R> → MulRsqrt<Thunk, R>
impl<F: Fn() -> M + Send + Sync, M, R> core::ops::Div<Sqrt<R>> for crate::Thunk<F> {
    type Output = MulRsqrt<Self, R>;
    #[inline(always)]
    fn div(self, rhs: Sqrt<R>) -> Self::Output {
        MulRsqrt(self, rhs.0)
    }
}

// Enumerate all other divisor types for Thunk
impl<F: Fn() -> M + Send + Sync, M, DL, DR> core::ops::Div<Add<DL, DR>> for crate::Thunk<F> {
    type Output = Div<Self, Add<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: Add<DL, DR>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DL, DR> core::ops::Div<Sub<DL, DR>> for crate::Thunk<F> {
    type Output = Div<Self, Sub<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: Sub<DL, DR>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DL, DR> core::ops::Div<Mul<DL, DR>> for crate::Thunk<F> {
    type Output = Div<Self, Mul<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: Mul<DL, DR>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DL, DR> core::ops::Div<Div<DL, DR>> for crate::Thunk<F> {
    type Output = Div<Self, Div<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: Div<DL, DR>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DL, DR> core::ops::Div<Max<DL, DR>> for crate::Thunk<F> {
    type Output = Div<Self, Max<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: Max<DL, DR>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DL, DR> core::ops::Div<Min<DL, DR>> for crate::Thunk<F> {
    type Output = Div<Self, Min<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: Min<DL, DR>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Abs<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Abs<DM>>;
    #[inline(always)]
    fn div(self, rhs: Abs<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Floor<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Floor<DM>>;
    #[inline(always)]
    fn div(self, rhs: Floor<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Rsqrt<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Rsqrt<DM>>;
    #[inline(always)]
    fn div(self, rhs: Rsqrt<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Sin<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Sin<DM>>;
    #[inline(always)]
    fn div(self, rhs: Sin<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Cos<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Cos<DM>>;
    #[inline(always)]
    fn div(self, rhs: Cos<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DC, DT, DF> core::ops::Div<Select<DC, DT, DF>>
    for crate::Thunk<F>
{
    type Output = Div<Self, Select<DC, DT, DF>>;
    #[inline(always)]
    fn div(self, rhs: Select<DC, DT, DF>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DA, DB, DC2> core::ops::Div<MulAdd<DA, DB, DC2>>
    for crate::Thunk<F>
{
    type Output = Div<Self, MulAdd<DA, DB, DC2>>;
    #[inline(always)]
    fn div(self, rhs: MulAdd<DA, DB, DC2>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM2> core::ops::Div<MulRecip<DM2>> for crate::Thunk<F> {
    type Output = Div<Self, MulRecip<DM2>>;
    #[inline(always)]
    fn div(self, rhs: MulRecip<DM2>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DL2, DR2> core::ops::Div<MulRsqrt<DL2, DR2>>
    for crate::Thunk<F>
{
    type Output = Div<Self, MulRsqrt<DL2, DR2>>;
    #[inline(always)]
    fn div(self, rhs: MulRsqrt<DL2, DR2>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DAcc, DVal, DMask> core::ops::Div<AddMasked<DAcc, DVal, DMask>>
    for crate::Thunk<F>
{
    type Output = Div<Self, AddMasked<DAcc, DVal, DMask>>;
    #[inline(always)]
    fn div(self, rhs: AddMasked<DAcc, DVal, DMask>) -> Self::Output {
        Div(self, rhs)
    }
}
// Concrete divisor types for Thunk
impl<F: Fn() -> M + Send + Sync, M> core::ops::Div<X> for crate::Thunk<F> {
    type Output = Div<Self, X>;
    #[inline(always)]
    fn div(self, rhs: X) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M> core::ops::Div<Y> for crate::Thunk<F> {
    type Output = Div<Self, Y>;
    #[inline(always)]
    fn div(self, rhs: Y) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M> core::ops::Div<Z> for crate::Thunk<F> {
    type Output = Div<Self, Z>;
    #[inline(always)]
    fn div(self, rhs: Z) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M> core::ops::Div<W> for crate::Thunk<F> {
    type Output = Div<Self, W>;
    #[inline(always)]
    fn div(self, rhs: W) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M> core::ops::Div<Field> for crate::Thunk<F> {
    type Output = Div<Self, Field>;
    #[inline(always)]
    fn div(self, rhs: Field) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M> core::ops::Div<f32> for crate::Thunk<F> {
    type Output = Div<Self, f32>;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M> core::ops::Div<i32> for crate::Thunk<F> {
    type Output = Div<Self, i32>;
    #[inline(always)]
    fn div(self, rhs: i32) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, N> core::ops::Div<Var<N>> for crate::Thunk<F> {
    type Output = Div<Self, Var<N>>;
    #[inline(always)]
    fn div(self, rhs: Var<N>) -> Self::Output {
        Div(self, rhs)
    }
}
// Fused derivative combinator divisors for Thunk
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<GradientMag2D<DM>> for crate::Thunk<F> {
    type Output = Div<Self, GradientMag2D<DM>>;
    #[inline(always)]
    fn div(self, rhs: GradientMag2D<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<GradientMag3D<DM>> for crate::Thunk<F> {
    type Output = Div<Self, GradientMag3D<DM>>;
    #[inline(always)]
    fn div(self, rhs: GradientMag3D<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Antialias2D<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Antialias2D<DM>>;
    #[inline(always)]
    fn div(self, rhs: Antialias2D<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Antialias3D<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Antialias3D<DM>>;
    #[inline(always)]
    fn div(self, rhs: Antialias3D<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<Curvature2D<DM>> for crate::Thunk<F> {
    type Output = Div<Self, Curvature2D<DM>>;
    #[inline(always)]
    fn div(self, rhs: Curvature2D<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
// Simple accessor combinator divisors for Thunk
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<ValOf<DM>> for crate::Thunk<F> {
    type Output = Div<Self, ValOf<DM>>;
    #[inline(always)]
    fn div(self, rhs: ValOf<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<DxOf<DM>> for crate::Thunk<F> {
    type Output = Div<Self, DxOf<DM>>;
    #[inline(always)]
    fn div(self, rhs: DxOf<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<DyOf<DM>> for crate::Thunk<F> {
    type Output = Div<Self, DyOf<DM>>;
    #[inline(always)]
    fn div(self, rhs: DyOf<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<DzOf<DM>> for crate::Thunk<F> {
    type Output = Div<Self, DzOf<DM>>;
    #[inline(always)]
    fn div(self, rhs: DzOf<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<DxxOf<DM>> for crate::Thunk<F> {
    type Output = Div<Self, DxxOf<DM>>;
    #[inline(always)]
    fn div(self, rhs: DxxOf<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<DxyOf<DM>> for crate::Thunk<F> {
    type Output = Div<Self, DxyOf<DM>>;
    #[inline(always)]
    fn div(self, rhs: DxyOf<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
impl<F: Fn() -> M + Send + Sync, M, DM> core::ops::Div<DyyOf<DM>> for crate::Thunk<F> {
    type Output = Div<Self, DyyOf<DM>>;
    #[inline(always)]
    fn div(self, rhs: DyyOf<DM>) -> Self::Output {
        Div(self, rhs)
    }
}
