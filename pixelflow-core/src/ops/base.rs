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

use super::{Add, Div, Mul, Sub};
use crate::Manifold;
use crate::variables::{W, X, Y, Z};

// ============================================================================
// The Macro
// ============================================================================

/// Implements binary operators for a single base manifold type.
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

        impl<Rhs: Manifold> core::ops::Div<Rhs> for $ty {
            type Output = Div<$ty, Rhs>;
            fn div(self, rhs: Rhs) -> Self::Output {
                Div(self, rhs)
            }
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
