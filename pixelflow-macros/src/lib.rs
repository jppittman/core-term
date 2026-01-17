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
//! Annotated AST + Symbol Table
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

mod ast;
mod codegen;
mod lexer;
mod parser;
mod sema;
mod symbol;

use proc_macro::TokenStream;

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

    // Phase 4: Code generation
    codegen::emit(analyzed).into()
}
