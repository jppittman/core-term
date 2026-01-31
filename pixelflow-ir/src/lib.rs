//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and Backend abstraction.
//!
//! - **Traits**: `Op` trait defines behavior.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **Kind**: `OpKind` enum unifies them for storage.
//! - **Backend**: SIMD execution traits.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod expr;
pub mod kind;
pub mod ops;
pub mod traits;

pub use kind::OpKind;
pub use traits::Op;

#[cfg(feature = "alloc")]
pub use expr::Expr;
