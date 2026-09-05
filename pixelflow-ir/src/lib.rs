//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and backend abstraction.
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **Backend**: SIMD execution traits.

// NOTE: `no_std` support (disabling the `std` feature) is currently
// incomplete: `cargo check -p pixelflow-ir --no-default-features` fails with
// over 200 errors (missing f32 methods, `ExprId` deref mismatches in
// `arena.rs`). No CI job builds this crate with `std` off -- the default
// feature set and `--all-features` both enable it -- so this has never been
// exercised. Tracked as a known, non-blocking gap by the
// `pixelflow-ir-nostd-status` job in `.github/workflows/rust.yaml`; treat
// `no_std` as aspirational until that job is green.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Exact dyadic rationals — the constant domain the e-graph folds in, so
/// that folding cannot contradict the algebraic rewrites. See the module docs.
pub mod dyadic;
pub mod kind;
pub mod traits;
pub mod variance;

pub use variance::{LatticeShape, Variance};

pub mod arena;

/// IR-to-IR transforms: each takes an expression graph and returns another.
/// Target-blind by construction — nothing here knows which ISA it is feeding.
pub mod passes;
pub use arena::{ExprArena, ExprId, ExprNode};

pub mod lower;
pub use lower::{Lower, LowerEnv};

pub mod binding;
pub use binding::{BindError, BindingTable};

// The differential-testing oracle, not an execution tier: PixelFlow is
// JIT-only, so nothing in a shipped build may reach a tree-walking evaluator.
// Gating it here is what enforces that — `cargo build` cannot name it.
#[cfg(any(test, feature = "oracle"))]
pub mod eval;
#[cfg(any(test, feature = "oracle"))]
pub use eval::{
    DifferentialCheck, MaskComparison, MaskVerdict, PointCheck, PointVerdict, Tolerance,
    compare_mask_root, equivalence_tolerance, eval_scalar, is_mask_valued, is_valid_mask,
    op_is_divergent_at, trunc_input_is_divergent,
};

pub mod kernel;
pub use kernel::{Kernel, Monoid};

pub use kind::OpKind;
pub use kind::known_method_names;
pub use traits::EmitStyle;
