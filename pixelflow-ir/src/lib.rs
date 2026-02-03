//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and Backend abstraction.
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **ALL_OPS**: The single source of truth for all operations.
//! - **Backend**: SIMD execution traits.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod kind;
pub mod traits;
pub mod ops;
pub mod expr;
pub mod backend;

// Primary API - Op traits and EmitStyle
pub use traits::{EmitStyle, Op, OpMeta};
pub use ops::{ALL_OPS, OP_COUNT, op_by_name, op_by_index, known_method_names};

// Legacy - OpKind enum (being phased out)
pub use kind::OpKind;

#[cfg(feature = "alloc")]
pub use expr::Expr;
