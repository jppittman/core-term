//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and backend abstraction.
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **ALL_OPS**: The single source of truth for all operations.
//! - **Backend**: SIMD execution traits.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod kind;
pub mod ops;
pub mod traits;
pub mod variance;

pub use variance::Variance;
pub use variance::{compute_arena_variance, find_hoistable_arena_nodes};

pub mod arena;
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
pub use jit_manifold::{JitManifold, ScanlineJitManifold};

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
/// A genuine 3-way split, checked only against `target_feature` (the flag
/// that actually governs what the compiler may emit into this crate) — NOT
/// ANDed against a build.rs host-detected cfg the way `pixelflow-core`'s
/// `storage.rs` gates `NativeF32Storage`. That AND is a real safety net over
/// there (see its comment): it guards against a build host whose CPU lacks
/// the feature it was explicitly told to target, which would emit
/// AVX-512/AVX2 instructions the box running the build can't execute. This
/// crate has no build script of its own to reproduce that host probe — it is
/// `no_std`-first and carries no `is_x86_feature_detected!` machinery — and
/// duplicating pixelflow-core's `pixelflow_avx512f`/`pixelflow_avx2` cfg here
/// would only recreate the mismatch it's meant to catch (a build script is
/// per-crate; a cfg it emits does not cross a crate boundary). Every other
/// `target_feature = "avx512f"` gate already in this crate (`x86.rs`,
/// `emit/mod.rs`, `emit/executable.rs`, `jit_manifold.rs`) is bare
/// `target_feature`, so this stays consistent with the rest of the crate
/// rather than inventing a second rule for one const.
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
