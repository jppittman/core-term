//! # Chained Operator Overloads
//!
//! Enables arithmetic chaining without boxing, e.g., `(X + Y) * Z`.

use super::{Abs, Add, Div, Max, Min, Mul, Sqrt, Sub};
use crate::Manifold;
use crate::combinators::Select;

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

// Binary nodes
impl_chained_ops!(Add<L, R>);
impl_chained_ops!(Sub<L, R>);
impl_chained_ops!(Mul<L, R>);
impl_chained_ops!(Div<L, R>);
impl_chained_ops!(Max<L, R>);
impl_chained_ops!(Min<L, R>);

// Unary nodes
impl_chained_ops!(Sqrt<M>);
impl_chained_ops!(Abs<M>);

// Combinators
impl_chained_ops!(Select<C, T, F>);
