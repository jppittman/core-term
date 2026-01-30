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

pub mod kind;
pub mod traits;
pub mod ops;
pub mod expr;
pub mod backend;

pub use kind::OpKind;
pub use traits::Op;

#[cfg(feature = "alloc")]
pub use expr::Expr;
