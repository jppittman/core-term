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

use super::{
    Abs, Add, AddMasked, Cos, Div, Floor, Max, Min, Mul, MulAdd, MulRecip, MulRsqrt, Rsqrt, Sin,
    Sqrt, Sub,
};
use crate::Field;
use crate::combinators::Select;
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
