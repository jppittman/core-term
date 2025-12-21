//! # Chained Operator Overloads
//!
//! This module provides operator overloads for composite AST types,
//! enabling expressions like `(X + Y) * Z` to work.
//!
//! ## What It Does
//!
//! After evaluating `X + Y`, you get an `Add<X, Y>` type.
//! To multiply that result by `Z`, we need `Add<X, Y>` to implement `Mul<Z>`.
//! This module provides those implementations for all composite types.
//!
//! ## Generated Code Example
//!
//! For `Add<L, R>`, the macro generates:
//! ```rust,ignore
//! impl<L: Manifold, R: Manifold, Rhs: Manifold> core::ops::Add<Rhs> for Add<L, R> {
//!     type Output = Add<Add<L, R>, Rhs>;
//!     fn add(self, rhs: Rhs) -> Self::Output { Add(self, rhs) }
//! }
//! // ... and Sub, Mul, Div similarly
//! ```

use crate::Manifold;
use crate::combinators::Select;
use crate::ops::{Abs, Add, Div, Ge, Gt, Le, Lt, Max, Min, Mul, Sqrt, Sub};

// ============================================================================
// The Macro
// ============================================================================

/// Implements chained binary operators for a composite AST type.
macro_rules! impl_chained_ops_for {
    ($name:ident < $($gen:ident),* >) => {
        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Add<Rhs> for $name<$($gen),*> {
            type Output = Add<Self, Rhs>;
            fn add(self, rhs: Rhs) -> Self::Output {
                Add(self, rhs)
            }
        }

        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Sub<Rhs> for $name<$($gen),*> {
            type Output = Sub<Self, Rhs>;
            fn sub(self, rhs: Rhs) -> Self::Output {
                Sub(self, rhs)
            }
        }

        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Mul<Rhs> for $name<$($gen),*> {
            type Output = Mul<Self, Rhs>;
            fn mul(self, rhs: Rhs) -> Self::Output {
                Mul(self, rhs)
            }
        }

        impl<$($gen: Manifold,)* Rhs: Manifold> core::ops::Div<Rhs> for $name<$($gen),*> {
            type Output = Div<Self, Rhs>;
            fn div(self, rhs: Rhs) -> Self::Output {
                Div(self, rhs)
            }
        }
    };
}

// ============================================================================
// Apply to All Composite Types
// ============================================================================

impl_chained_ops_for!(Add<L, R>);
impl_chained_ops_for!(Sub<L, R>);
impl_chained_ops_for!(Mul<L, R>);
impl_chained_ops_for!(Div<L, R>);
impl_chained_ops_for!(Sqrt<T>);
impl_chained_ops_for!(Abs<T>);
impl_chained_ops_for!(Max<L, R>);
impl_chained_ops_for!(Min<L, R>);
impl_chained_ops_for!(Lt<L, R>);
impl_chained_ops_for!(Gt<L, R>);
impl_chained_ops_for!(Le<L, R>);
impl_chained_ops_for!(Ge<L, R>);
impl_chained_ops_for!(Select<C, T, F>);
