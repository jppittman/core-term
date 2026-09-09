//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR).
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//!
//! A SIMD backend abstraction (`Backend`/`SimdOps`) lived here, then moved to
//! `pixelflow-core` on 2026-08-02 (it was not IR and not codegen — it lived
//! beside `Field`, which it backed). It backed a per-batch "combinator"
//! evaluation tier that the JIT superseded; both the tier and the abstraction
//! are gone (docs/plans/2026-09-06-kernel-with-a-lattice.md). `Field` now
//! reaches its two remaining constructors as inherent methods on
//! `pixelflow-core`'s own `pub(crate)` lane types, with no trait at all.

// `no_std` is built and blocking: `cargo check -p pixelflow-ir
// --no-default-features` with `-D warnings` runs presubmit (the "pixelflow-ir
// no_std + pixelflow-search std-feature-off builds" job). It is no longer the
// aspirational, unexercised gap an earlier revision of this note described —
// and a comment that said so is precisely what let `store.rs` land three
// unconditional `use std::` lines.
//
// What the `std` feature buys is exactly one module: [`store`], the
// process-global map a `Ref` names into. See its docs for why that is the
// whole of it.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Exact dyadic rationals — the constant domain the e-graph folds in, so
/// that folding cannot contradict the algebraic rewrites. See the module docs.
pub mod dyadic;

/// What a bounded reduction *is*: the algebra it folds under, the index it
/// binds, and the range that index runs over. See the module docs for why
/// those are a type rather than three `Const` children.
pub mod fold;
pub use fold::{Binder, Fold, Monoid};

pub mod kind;
pub mod traits;
pub mod variance;

pub use variance::{LatticeShape, Variance};

pub mod arena;

/// What it means for two kernels to be the same kernel: the canonical form of
/// a reachable subgraph, and the fixed-size [`KernelKey`] that digests it.
pub mod key;
// Only the key is re-exported at the root: it is a *name* other crates hold
// (a `Ref` leaf, an e-graph refusal), where `Canonical`/`canonical` are the
// machinery behind it and one consumer — the compile cache — spells the
// module out.
pub use key::KernelKey;

/// The process-global, content-addressed store a `Ref` names into.
///
/// `std` only: a global map needs a lock. Without it nothing can mint a
/// `Ref`, so nothing can need to resolve one — see the module's own docs.
#[cfg(feature = "std")]
pub mod store;
#[cfg(feature = "std")]
pub use store::KernelStore;

/// IR-to-IR transforms: each takes an expression graph and returns another.
/// Target-blind by construction — nothing here knows which ISA it is feeding.
pub mod passes;
pub use arena::{ExprArena, ExprId, ExprNode};

/// The term language the e-graph speaks: destructure a node, rebuild a node.
/// Naming it is what makes an optimizer expressible as an endomorphism on the
/// IR rather than as a hand-rolled conversion per tier.
pub mod term;
mod term_arena;
pub use term::{Children, Ir, Shape};

/// Optimization as an endomorphism on the IR — including the identity, which
/// is what `kernel_raw!` means and what a measurement's control arm needs.
pub mod optimize;
pub use optimize::{Identity, Optimize, Rewritten, Then};

pub mod binding;
pub use binding::{BindError, BindingTable};

// The differential-testing oracle, not an execution tier: PixelFlow is
// JIT-only, so nothing in a shipped build may reach a tree-walking evaluator.
// Gating it here is what enforces that — `cargo build` cannot name it.
#[cfg(any(test, feature = "oracle"))]
pub mod eval;
#[cfg(any(test, feature = "oracle"))]
pub use eval::{
    DifferentialCheck, Evaluator, MaskComparison, MaskVerdict, PointCheck, PointVerdict, Tolerance,
    compare_mask_root, equivalence_tolerance, eval_scalar, is_mask_valued, is_valid_mask,
    op_is_divergent_at, trunc_input_is_divergent,
};

pub mod kernel;
pub use kernel::{Bits, Kernel, Scalar, Uniform};

pub use kind::OpKind;
pub use kind::known_method_names;
pub use traits::EmitStyle;
