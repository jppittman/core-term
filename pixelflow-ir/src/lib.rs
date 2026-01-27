//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and Backend abstraction.
//!
//! - **Algebra**: Algebraic structures (`Algebra`, `Transcendental`, `Dual<N>`)
//! - **Traits**: `Op` trait defines behavior.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **Kind**: `OpKind` enum unifies them for storage.
//! - **Backend**: SIMD execution traits.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod algebra;
pub mod dual;
pub mod kind;
pub mod traits;
pub mod ops;
pub mod expr;
pub mod backend;

pub use algebra::{Algebra, Transcendental};
pub use dual::Dual;
pub use kind::OpKind;
pub use traits::Op;
