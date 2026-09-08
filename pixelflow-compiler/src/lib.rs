//! # PixelFlow Kernel Compiler Frontend
//!
//! A compiler frontend for the PixelFlow DSL, implemented as Rust proc-macros.
//!
//! ## Architecture
//!
//! ```text
//! Source (macro input)
//!     │
//!     ▼ Parser (parser.rs)
//! AST (ast.rs)
//!     │
//!     ▼ Semantic Analysis (sema.rs)
//! Analyzed AST + Symbol Table
//!     │
//!     ▼ E-graph optimization (optimize.rs)   — `kernel!` only
//! Optimized AST
//!     │
//!     ▼ Arena lowering (ir_bridge.rs, jit_backend.rs)
//! Rust TokenStream that rebuilds a `Kernel` at load time
//! ```
//!
//! There is **one backend**, and it produces a [`Kernel`] — an arena fragment,
//! the language's own value. Nothing is compiled at macro-expansion time and
//! nothing is compiled at construction: a `Kernel` becomes machine code when a
//! consumer compiles it at a lattice's shape and collapses it
//! (`Lattice::bake`), which is the only way a kernel turns into numbers.
//!
//! So there are two macros, and the only difference between them is whether
//! the e-graph runs: [`kernel!`](macro@kernel) optimizes,
//! [`kernel_raw!`](macro@kernel_raw) does not.
//!
//! [`Kernel`]: pixelflow_core::Kernel

mod ast;
mod ir_bridge;
mod jit_backend;
mod optimize;
mod parser;
mod sema;
mod symbol;

use proc_macro::TokenStream;

/// The `kernel!` macro: closure syntax for a [`Kernel`](pixelflow_core::Kernel),
/// optimized by the e-graph at macro-expansion time.
///
/// - Zero params → a `Kernel` value.
/// - N params → a builder closure `move |p0: f32, ...| -> Kernel` that
///   constant-folds its arguments into the fragment.
///
/// Kernels compose as values — `Kernel::at`/`sum`/`select`/arithmetic — so
/// there is no manifold-typed parameter. Derivatives (`DX`/`DY`) become
/// symbolic `Dwrt` nodes, resolved by the e-graph here when it can and by
/// codegen otherwise.
///
/// # Syntax
///
/// ```ignore
/// kernel!(|param1: f32, param2: f32, ...| expression)
/// ```
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{Kernel, Lattice};
///
/// let circle = kernel!(|cx: f32, cy: f32, r: f32| {
///     let dx = X - cx;
///     let dy = Y - cy;
///     (dx * dx + dy * dy).sqrt() - r
/// });
///
/// let unit_circle: Kernel = circle(0.0, 0.0, 1.0);
/// let plane = Lattice::frame(64, 64).bake(&unit_circle);
/// ```
///
/// # Parameters
///
/// A builder's arguments are anything `Into<Scalar>`, and the type at the
/// call site decides what the parameter is. An `f32` is folded into the
/// fragment as a constant, so `circle(0.0, 0.0, 1.0)` is the same kernel it
/// always was. A [`Uniform`](pixelflow_core::Uniform) handle makes the
/// parameter an *argument* of the compiled kernel instead — invariant across
/// the lattice, bound per call from a `UniformBlock`, never folded — so a
/// scene transform or a cursor position moves without a recompile:
///
/// ```ignore
/// let cx = Uniform::new(0.0);
/// let moving = circle(cx, 0.0, 1.0);   // cx is an argument; cy and r are folded
/// ```
///
/// Each `let` binding of a builder is one signature: the same binding cannot
/// be called with an `f32` and a `Uniform` in the same position.
///
/// # Pipeline
///
/// 1. **Parser**: closure syntax → AST
/// 2. **Semantic analysis**: symbol resolution, method validation
/// 3. **Optimization**: e-graph saturation + latency-prior extraction
/// 4. **Arena lowering**: the optimized AST becomes an `ExprArena`
#[proc_macro]
pub fn kernel(input: TokenStream) -> TokenStream {
    let analyzed = match front_end(input) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    // E-graph saturation + latency-prior extraction at macro-expansion time:
    // FMA fusion, algebraic simplification, CSE and rsqrt all happen here,
    // before the arena is emitted.
    let analyzed = optimize::optimize(analyzed);
    emit(&analyzed)
}

/// The `kernel_raw!` macro: like [`kernel!`](macro@kernel) but **without**
/// e-graph optimization, so the emitted arena has the shape that was written.
///
/// # Use Cases
///
/// - Benchmarking an exact expression form: `X * Y + Z` against
///   `(X).mul_add(Y, Z)`, which `kernel!` would fuse into the same node.
/// - Corpus generation, where the input to the optimizer is the subject.
/// - Debugging: what the front end built, before anything rewrote it.
///
/// # Example
///
/// ```ignore
/// // Two different arenas — mul then add, against one MulAdd.
/// let unoptimized = kernel_raw!(|| X * Y + Z);
/// let explicit_fma = kernel_raw!(|| (X).mul_add(Y, Z));
/// ```
#[proc_macro]
pub fn kernel_raw(input: TokenStream) -> TokenStream {
    let analyzed = match front_end(input) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    emit(&analyzed)
}

/// Parse and analyze: the half both macros share, verbatim.
fn front_end(input: TokenStream) -> syn::Result<sema::AnalyzedKernel> {
    let tokens = proc_macro2::TokenStream::from(input);
    sema::analyze(parser::parse(tokens)?)
}

/// Lower to an arena and emit the code that rebuilds it.
fn emit(analyzed: &sema::AnalyzedKernel) -> TokenStream {
    match jit_backend::emit_kernel(analyzed) {
        Ok(tokens) => tokens.into(),
        Err(e) => syn::Error::new(proc_macro2::Span::call_site(), e)
            .to_compile_error()
            .into(),
    }
}
