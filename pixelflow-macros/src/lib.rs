//! # PixelFlow Kernel Compiler Frontend
//!
//! A compiler frontend for the PixelFlow DSL, implemented as Rust proc-macros.
//!
//! ## Architecture
//!
//! The compiler follows a traditional pipeline:
//!
//! ```text
//! Source (macro input)
//!     │
//!     ▼ Lexer (lexer.rs)
//! Token Stream
//!     │
//!     ▼ Parser (parser.rs)
//! AST (ast.rs)
//!     │
//!     ▼ Semantic Analysis (sema.rs)
//! Analyzed AST + Symbol Table
//!     │
//!     ▼ Annotation (annotate.rs)
//! Annotated AST (literals have Var indices)
//!     │
//!     ▼ Code Generation (codegen.rs)
//! Rust TokenStream (output)
//! ```
//!
//! ## Symbol Table
//!
//! The compiler maintains a symbol table with two classes of symbols:
//!
//! 1. **Intrinsic coordinates** (X, Y, Z, W) - bound at evaluation time
//! 2. **Captured parameters** - bound at kernel construction time
//!
//! This mirrors the layered contramap pattern: parameters are fixed when you
//! call the kernel constructor, coordinates are fixed when `eval_raw` is called.

mod annotate;
mod ast;
mod codegen;
mod element;
mod fold;
mod lexer;
mod manifold_expr;
mod optimize;
mod parser;
mod sema;
mod symbol;

use proc_macro::TokenStream;
use quote::format_ident;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, LitInt};

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
/// use pixelflow_macros::kernel;
/// use pixelflow_core::{X, Y, Manifold, ManifoldExt};
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
/// 1. **Lexer**: Tokenizes the input (delegated to `syn`)
/// 2. **Parser**: Builds AST from closure syntax
/// 3. **Semantic Analysis**: Resolves symbols, validates types
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

    // Phase 5: Code generation
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

/// Derive macro for the `ManifoldExpr` marker trait.
///
/// This trait gates access to `ManifoldExt` methods, preventing them from
/// polluting the method namespace of non-manifold types like iterators.
///
/// # Example
///
/// ```ignore
/// use pixelflow_macros::ManifoldExpr;
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
