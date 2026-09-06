//! # PixelFlow Kernel Compiler Frontend
//!
//! A compiler frontend for the PixelFlow DSL, implemented as Rust proc-macros.
//!
//! ## Architecture
//!
//! The live frontend pipeline is:
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
//!     ▼ E-graph optimization (optimize.rs)
//! Optimized AST
//!     │
//!     ▼ Code generation / arena lowering
//! Rust TokenStream (output)
//! ```
//!
//! The compiler still has some older utility modules, but the hot path is
//! parse -> analyze -> optimize -> emit.

mod annotate;
mod ast;
mod codegen;
mod element;
mod ir_bridge;
mod jit_backend;
mod manifold_expr;
mod optimize;
mod parser;
mod sema;
mod symbol;

use proc_macro::TokenStream;
use quote::format_ident;
use syn::parse::{Parse, ParseStream};
use syn::{LitInt, parse_macro_input};

/// Derive macro for the `Element` trait.
///
/// This macro generates the "Applicative" structure for a type, making it behave
/// like a first-class value in the DSL. It automatically implements:
///
/// - `ManifoldExpr` marker trait
/// - Arithmetic operators: `Add`, `Sub`, `Mul`, `Div`, `Neg`
/// - Logic operators: `BitAnd`, `BitOr`, `Not`
///
/// # Usage
///
/// ```ignore
/// #[derive(Element)]
/// pub struct MyCombinator<A, B>(pub A, pub B);
/// ```
#[proc_macro_derive(Element)]
pub fn derive_element(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    element::derive_element(input).into()
}

/// The `kernel!` macro: closure-like syntax for SIMD manifold kernels.
///
/// # Syntax
///
/// ```ignore
/// kernel!(|param1: Type1, param2: Type2, ...| expression)
/// ```
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::combinator::Manifold;
/// use pixelflow_core::{X, Y, ManifoldExt};
///
/// let circle = kernel!(|cx: f32, cy: f32, r: f32| {
///     let dx = X - cx;
///     let dy = Y - cy;
///     (dx * dx + dy * dy).sqrt() - r
/// });
///
/// let unit_circle = circle(0.0, 0.0, 1.0);
/// ```
///
/// # Compiler Pipeline
///
/// 1. **Parser**: Builds AST from closure syntax
/// 2. **Semantic Analysis**: Resolves symbols, validates types
/// 3. **Optimization**: E-graph saturation + learned extraction
/// 4. **Code Generation**: Emits struct + Manifold impl
#[proc_macro]
pub fn kernel(input: TokenStream) -> TokenStream {
    // Phase 1: Lex (syn handles this)
    let tokens = proc_macro2::TokenStream::from(input);

    // Phase 2: Parse
    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 3: Semantic analysis
    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 4: Optimization
    let optimized = optimize::optimize(analyzed);

    // Phase 5: Code generation (the monomorphizing combinator backend).
    //
    // `kernel!` is the LLVM tier: it emits combinator ZSTs that are
    // polymorphic over the *evaluation domain* — a kernel declared `-> Field`
    // is still routinely evaluated over `Jet2`/`Jet3` domains for
    // antialiasing and 3D surfaces. The arena tier (`kernel_jit!` /
    // `kernel_value!`) produces a `Kernel` instead, which is not a manifold at
    // all and so cannot be routed to transparently; a consumer picks a tier by
    // picking a macro.
    codegen::emit(optimized).into()
}

/// The `kernel_raw!` macro: like `kernel!` but **without optimization**.
///
/// This macro skips the AST optimization phase (constant folding, FMA fusion,
/// algebraic simplification). Use this when you need to benchmark the exact
/// expression form without the compiler transforming it.
///
/// # Use Cases
///
/// - Training data generation: benchmark `X * Y + Z` vs `mul_add(X, Y, Z)` separately
/// - Debugging: see what code is generated without optimization
/// - Testing: verify optimization actually improves things
///
/// # Example
///
/// ```ignore
/// // These will benchmark DIFFERENT code with kernel_raw!
/// let unoptimized = kernel_raw!(|| X * Y + Z);  // Stays as mul + add
/// let explicit_fma = kernel_raw!(|| (X).mul_add(Y, Z));  // Uses FMA
///
/// // With kernel!, both might compile to the same FMA instruction
/// ```
#[proc_macro]
pub fn kernel_raw(input: TokenStream) -> TokenStream {
    // Phase 1: Lex (syn handles this)
    let tokens = proc_macro2::TokenStream::from(input);

    // Phase 2: Parse
    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 3: Semantic analysis
    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 4: SKIP optimization - go directly to codegen
    // This preserves the exact expression structure for benchmarking

    // Phase 5: Code generation
    codegen::emit(analyzed).into()
}

/// The `kernel_value!` macro: build a [`Kernel`](pixelflow_core::Kernel) value.
///
/// Same front-end pipeline as `kernel!`/`kernel_jit!` (parse → sema → e-graph
/// optimize), but the result is the language's *runtime value* — an uncompiled
/// arena fragment — not a JIT-compiled or combinator manifold. This is the
/// JIT-first surface: produce `Kernel` values, compose them
/// (`Kernel::sum`/`at`/`select`/arithmetic), and bake once at a root
/// (`Lattice::bake`). Derivatives (`DX`/`DY`) become symbolic `Dwrt` nodes
/// resolved at bake — no jet domain.
///
/// - Zero params → a `Kernel` value directly.
/// - N scalar params → a builder closure `move |p0: f32, ...| -> Kernel` that
///   constant-folds the params into the fragment (no JIT — leaves are
///   bake-time-only, fused at the root).
///
/// Manifold params are unsupported: compose `Kernel` values instead of
/// splicing through macro slots.
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::kernel_value;
/// use pixelflow_core::Kernel;
///
/// let leaf = kernel_value!(|cx: f32, r: f32| (X - cx) * r);
/// let a = leaf(1.0, 2.0);          // a Kernel value, not compiled
/// let scene = Kernel::sum(&[a, leaf(3.0, 0.5)]);
/// // lattice.bake(&scene) compiles the fused arena once.
/// ```
#[proc_macro]
pub fn kernel_value(input: TokenStream) -> TokenStream {
    let tokens = proc_macro2::TokenStream::from(input);
    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };
    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    // Same e-graph optimization as the other macros; the only difference is the
    // backend that consumes the optimized AST.
    let analyzed = optimize::optimize(analyzed);

    match jit_backend::emit_kernel_value(&analyzed) {
        Ok(tokens) => tokens.into(),
        Err(e) => syn::Error::new(proc_macro2::Span::call_site(), e)
            .to_compile_error()
            .into(),
    }
}

/// The `kernel_jit!` macro: a [`Kernel`](pixelflow_core::Kernel) over the
/// arena backend, with macro-time e-graph optimization.
///
/// - Without parameters: a `Kernel` value.
/// - With parameters: a builder closure returning a `Kernel`. Scalar params
///   are folded in as constants; manifold-typed params (`name: kernel`) are
///   spliced through `Lower`, so the builder composes ONE fused arena.
///
/// The difference from [`kernel_value!`](macro@kernel_value) is composition:
/// `kernel_value!` has no manifold params, because `Kernel` values compose
/// through `Kernel::at`/`sum`/`select` directly.
///
/// Nothing is compiled at construction. A `Kernel` becomes machine code when a
/// consumer bakes it over a lattice (`Lattice::bake(&kernel)`), which is the
/// only way a kernel turns into numbers.
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::kernel_jit;
/// use pixelflow_core::Lattice;
///
/// let builder = kernel_jit!(|cx: f32, r: f32| (X - cx) * r);
/// let kernel = builder(1.0, 2.0);
/// let plane = Lattice::frame(64, 64, 0.0).bake(&kernel);
/// ```
#[proc_macro]
pub fn kernel_jit(input: TokenStream) -> TokenStream {
    // Phase 1: Parse
    let tokens = proc_macro2::TokenStream::from(input);
    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 2: Semantic analysis
    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 3: Optimization (e-graph saturation + latency-prior extraction at
    // macro-expansion time). Same pipeline as `kernel!` — the only difference
    // between the two is the backend that consumes the optimized AST, not the
    // optimization. FMA fusion, algebraic simplification, CSE and rsqrt all
    // happen here, before the arena is emitted.
    let analyzed = optimize::optimize(analyzed);

    // Phase 4: arena backend. Unlike `kernel!` there is no fallback — a
    // lowering failure is a hard error, since the caller explicitly asked for
    // the arena tier.
    match jit_backend::emit_jit(&analyzed) {
        Ok(tokens) => tokens.into(),
        Err(e) => syn::Error::new(proc_macro2::Span::call_site(), e)
            .to_compile_error()
            .into(),
    }
}

/// Derive macro for the `ManifoldExpr` marker trait.
///
/// This trait gates access to `ManifoldExt` methods, preventing them from
/// polluting the method namespace of non-manifold types like iterators.
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::ManifoldExpr;
///
/// #[derive(ManifoldExpr)]
/// pub struct MyCustomCombinator<M>(pub M);
/// ```
///
/// # Generated Code
///
/// ```ignore
/// impl<M> ::pixelflow_core::ManifoldExpr for MyCustomCombinator<M> {}
/// ```
#[proc_macro_derive(ManifoldExpr)]
pub fn derive_manifold_expr(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    manifold_expr::derive_manifold_expr(input).into()
}

/// Configuration for `generate_peano_types!` macro.
struct PeanoConfig {
    count: usize,
}

impl Parse for PeanoConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lit: LitInt = input.parse()?;
        let count = lit.base10_parse()?;
        Ok(PeanoConfig { count })
    }
}

/// Generate binary-encoded type aliases N0, N1, ..., N{count-1}.
///
/// Uses binary encoding with UTerm/UInt/B0/B1 for logarithmic depth:
/// - N0 = UTerm
/// - N1 = UInt<UTerm, B1>  (0b1)
/// - N2 = UInt<UInt<UTerm, B1>, B0>  (0b10)
/// - N3 = UInt<UInt<UTerm, B1>, B1>  (0b11)
/// - etc.
///
/// This reduces type nesting from O(n) to O(log n).
///
/// # Example
///
/// ```ignore
/// generate_binary_types!(256);
/// // N30 = UInt<UInt<UInt<UInt<UInt<UTerm, B1>, B1>, B1>, B1>, B0>  (0b11110)
/// // Instead of Succ<Succ<Succ<...30 times...>>>
/// ```
#[proc_macro]
pub fn generate_binary_types(input: TokenStream) -> TokenStream {
    let config = parse_macro_input!(input as PeanoConfig);
    let count = config.count;

    let mut types = Vec::new();

    for i in 0..count {
        let name = format_ident!("N{}", i);
        let doc = format!("Index {} (0b{:b})", i, i);
        let ty = to_binary_type(i);

        types.push(quote::quote! {
            #[doc = #doc]
            pub type #name = #ty;
        });
    }

    TokenStream::from(quote::quote! {
        #(#types)*
    })
}

/// Convert a number to its binary type representation.
fn to_binary_type(n: usize) -> proc_macro2::TokenStream {
    if n == 0 {
        return quote::quote! { UTerm };
    }

    let bit = if n % 2 == 0 {
        quote::quote! { B0 }
    } else {
        quote::quote! { B1 }
    };

    let rest = to_binary_type(n >> 1);

    quote::quote! { UInt<#rest, #bit> }
}
