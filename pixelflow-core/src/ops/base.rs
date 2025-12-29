//! # Base Operator Overloads
//!
//! This module provides operator overloads for the base types (X, Y, Z, W, f32, i32),
//! enabling expressions like `X + Y` and `X * 2.0`.
//!
//! ## What It Does
//!
//! When you write `X + Y`, Rust needs `X` to implement `Add<Y>`.
//! This module provides those implementations.
//!
//! ## Generated Code Example
//!
//! For `X`, the macro generates:
//! ```rust,ignore
//! impl<Rhs: Manifold> core::ops::Add<Rhs> for X {
//!     type Output = Add<X, Rhs>;
//!     fn add(self, rhs: Rhs) -> Self::Output { Add(self, rhs) }
//! }
//! // ... and Sub, Mul, Div similarly
//! ```
//!
//! ## Rsqrt Fusion
//!
//! When dividing by `Sqrt<R>`, the result is `MulRsqrt<L, R>` instead of `Div<L, Sqrt<R>>`.
//! This uses fast rsqrt (~3 cycles) instead of sqrt (~12) + div (~12).

use super::{Abs, Add, AddMasked, Cos, Div, Floor, Max, Min, Mul, MulAdd, MulRecip, MulRsqrt, Rsqrt, Sin, Sqrt, Sub};
use crate::Manifold;
use crate::combinators::Select;
use crate::variables::{W, X, Y, Z};
use crate::Field;

// ============================================================================
// The Macro
// ============================================================================

/// Implements binary operators for a single base manifold type.
/// Includes rsqrt fusion: L / Sqrt<R> → MulRsqrt<L, R>
macro_rules! impl_binary_ops_for {
    ($ty:ty) => {
        impl<Rhs: Manifold> core::ops::Add<Rhs> for $ty {
            type Output = Add<$ty, Rhs>;
            fn add(self, rhs: Rhs) -> Self::Output {
                Add(self, rhs)
            }
        }

        impl<Rhs: Manifold> core::ops::Sub<Rhs> for $ty {
            type Output = Sub<$ty, Rhs>;
            fn sub(self, rhs: Rhs) -> Self::Output {
                Sub(self, rhs)
            }
        }

        impl<Rhs: Manifold> core::ops::Mul<Rhs> for $ty {
            type Output = Mul<$ty, Rhs>;
            fn mul(self, rhs: Rhs) -> Self::Output {
                Mul(self, rhs)
            }
        }

        // Rsqrt fusion: L / Sqrt<R> → MulRsqrt<L, R>
        impl<R: Manifold> core::ops::Div<Sqrt<R>> for $ty {
            type Output = MulRsqrt<$ty, R>;
            #[inline(always)]
            fn div(self, rhs: Sqrt<R>) -> Self::Output {
                MulRsqrt(self, rhs.0)
            }
        }

        // Enumerate all other divisor types to avoid conflict with Sqrt
        impl_base_div!($ty, Add<DL, DR>);
        impl_base_div!($ty, Sub<DL, DR>);
        impl_base_div!($ty, Mul<DL, DR>);
        impl_base_div!($ty, Div<DL, DR>);
        impl_base_div!($ty, Max<DL, DR>);
        impl_base_div!($ty, Min<DL, DR>);
        impl_base_div!($ty, Abs<DM>);
        impl_base_div!($ty, Floor<DM>);
        impl_base_div!($ty, Rsqrt<DM>);
        impl_base_div!($ty, Sin<DM>);
        impl_base_div!($ty, Cos<DM>);
        impl_base_div!($ty, Select<DC, DT, DF>);
        impl_base_div!($ty, MulAdd<DA, DB, DC2>);
        impl_base_div!($ty, MulRecip<DM2>);
        impl_base_div!($ty, MulRsqrt<DL2, DR2>);
        impl_base_div!($ty, AddMasked<DAcc, DVal, DMask>);

        // Concrete divisor types
        impl_base_div_concrete!($ty, X);
        impl_base_div_concrete!($ty, Y);
        impl_base_div_concrete!($ty, Z);
        impl_base_div_concrete!($ty, W);
        impl_base_div_concrete!($ty, Field);
        impl_base_div_concrete!($ty, f32);
        impl_base_div_concrete!($ty, i32);
    };
}

/// Generate Div impl for base type with generic divisor
macro_rules! impl_base_div {
    ($self_ty:ty, $div_ty:ident <$($dg:ident),*>) => {
        impl<$($dg: Manifold),*> core::ops::Div<$div_ty<$($dg),*>> for $self_ty {
            type Output = Div<$self_ty, $div_ty<$($dg),*>>;
            #[inline(always)]
            fn div(self, rhs: $div_ty<$($dg),*>) -> Self::Output { Div(self, rhs) }
        }
    };
}

/// Generate Div impl for base type with concrete divisor
macro_rules! impl_base_div_concrete {
    ($self_ty:ty, $div_ty:ty) => {
        impl core::ops::Div<$div_ty> for $self_ty {
            type Output = Div<$self_ty, $div_ty>;
            #[inline(always)]
            fn div(self, rhs: $div_ty) -> Self::Output { Div(self, rhs) }
        }
    };
}

// ============================================================================
// Apply to Base Types
// ============================================================================

// Note: We can only implement Add/Sub/Mul/Div for OUR types (X, Y, Z, W).
// For f32 and i32, the orphan rules prevent us from implementing foreign
// traits for foreign types. This means:
// - `X + 1.0` works (X is ours, 1.0 becomes Add<X, f32>)
// - `1.0 + X` does NOT work (would need impl Add<X> for f32, which is forbidden)
// If you need `1.0 + X`, write `X + 1.0` or use ManifoldExt methods.

impl_binary_ops_for!(X);
impl_binary_ops_for!(Y);
impl_binary_ops_for!(Z);
impl_binary_ops_for!(W);
