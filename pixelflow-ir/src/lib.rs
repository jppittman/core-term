//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and backend abstraction.
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **ALL_OPS**: The single source of truth for all operations.
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

pub mod emit;
pub mod kind;
pub mod ops;
pub mod traits;
pub mod variance;

pub use variance::Variance;
pub use variance::{compute_arena_variance, find_hoistable_arena_nodes};

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
pub use eval::eval_scalar;

pub mod kernel;
pub use kernel::{Kernel, Monoid};

pub mod jit_manifold;
pub use jit_manifold::JitManifold;

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod jit_cache;

pub use kind::OpKind;
pub use ops::{ALL_OPS, OP_COUNT, known_method_names, op_by_index, op_by_name};
pub use traits::{EmitStyle, Op, OpMeta};

/// Byte width of the SIMD vector this build's JIT emits and calls — i.e. the
/// size of one [`KernelFn`](backend::emit::executable::KernelFn) argument and
/// return value.
///
/// The JIT has no dependency on `pixelflow-core`, so it cannot name `Field`
/// directly. This const is the single source of truth for the width the emitter
/// and the `KernelFn` ABI agree on. Callers that bridge `Field` to a JIT kernel
/// (the `kernel_jit!` wrapper) assert `size_of::<Field>() == JIT_VECTOR_BYTES`
/// at compile time, turning any width disagreement into a clear build error
/// rather than a raw `transmute` size error (or, worse, a silent miscompile).
///
/// A genuine 3-way split, checked against `target_feature` — the flag that
/// actually governs what the compiler may emit. `pixelflow-core` gates
/// `NativeF32Storage` on exactly the same predicate, which is what keeps the
/// two crates' widths in step.
///
/// They once did not. `pixelflow-core` used to AND in a build-script cfg set by
/// probing the *build host's* CPU, on the theory that it was a safety net
/// against a host told to target a feature it lacks. It was not: a build script
/// is per-crate and the cfg it emits does not cross a crate boundary, so this
/// crate went on emitting 512-bit code while core's `Field` quietly narrowed to
/// 256 — the disagreement that net was supposed to prevent, caused by the net
/// itself. Building `+avx512f` on a non-AVX-512 host failed on the `transmute`
/// in core's `lattice`. Host CPU is the wrong question for a target decision;
/// running wide code on a narrow host is gated by `cargo xtask isa-matrix`,
/// which builds every level and runs only what `host_has_feature` allows.
///
/// 64 (512-bit, AVX-512) when compiled with `target_feature = "avx512f"`,
/// where `compile_arena_dag` routes to `Avx512Backend` and `KernelFn` is
/// `__m512`; 32 (256-bit, AVX2) when compiled with `target_feature = "avx2"`
/// and not `"avx512f"`, routing to `Avx2Backend` (`KernelFn` = `__m256`);
/// otherwise 16 (128-bit, SSE2/NEON). This matches `pixelflow-core`'s
/// `Field` width under the same build flags, so the `kernel_jit!` assert
/// holds.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub const JIT_VECTOR_BYTES: usize = 64;
/// See the AVX-512 variant above.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub const JIT_VECTOR_BYTES: usize = 32;
/// See the AVX-512 variant above.
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
pub const JIT_VECTOR_BYTES: usize = 16;
