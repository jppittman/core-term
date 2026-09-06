//! Arena backend codegen: the one backend, for `kernel!` and `kernel_raw!`.
//!
//! Both macros run the identical front end (parse → sema); `kernel!` also runs
//! the e-graph, and that is the only difference between them. This module
//! lowers the resulting AST to an [`ExprArena`](pixelflow_ir::arena::ExprArena)
//! and emits code that rebuilds it at load time as a
//! [`Kernel`](pixelflow_core::Kernel).
//!
//! A `Kernel` is the whole surface: a consumer hands it to `Lattice::bake`
//! together with the lattice it wants tabulated, and the compiler owns the loop
//! nest. There is no per-batch entry — no `Manifold` impl calling into the JIT
//! once per SIMD batch — so nothing here emits one.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ir_bridge;
use crate::sema::AnalyzedKernel;

/// Emit arena-backend code for an analyzed kernel.
///
/// On success, returns a token stream evaluating to:
/// - zero params — a [`Kernel`](pixelflow_core::Kernel) value, built at load
///   time from the arena this expansion computed.
/// - N params — a builder `|p0, ..., pN| -> Kernel` whose arguments are
///   anything `Into<Scalar>`: an `f32` is constant-folded into the fragment
///   when the builder runs (no JIT — leaves are bake-time-only and fuse at a
///   root, which is what lets a font build thousands of leaf kernels and
///   compile one arena), and a `Uniform` handle declares a per-call slot
///   instead. The *type* at the call site chooses, so every site that passes
///   an `f32` keeps folding.
///
/// Kernels compose as *values* — `Kernel::at`/`sum`/`select`/arithmetic — not
/// by splicing a manifold through a macro slot, so there is no manifold-typed
/// parameter and nothing here to lower one with.
///
/// Returns `Err` if the body contains an operation the IR bridge cannot lower.
pub fn emit_kernel(analyzed: &AnalyzedKernel) -> Result<TokenStream, String> {
    let param_map = ir_bridge::param_indices(analyzed);
    let arena_code = ir_bridge::ast_to_runtime_arena(&analyzed.def.body, &param_map)?;

    if analyzed.def.params.is_empty() {
        return Ok(quote! {
            {
                let (__arena, __root) = #arena_code;
                ::pixelflow_core::Kernel::from_parts(__arena, __root)
            }
        });
    }

    // The builder. A closure cannot be generic over its argument types, so
    // the expansion is a generic `fn` returning `impl Fn`: the type parameters
    // are inferred at the call site — `f32` from a float literal or variable,
    // `Uniform` from a handle — and each `let` binding of a builder is one
    // signature. Arguments appear in declaration order.
    let param_names: Vec<proc_macro2::Ident> =
        analyzed.def.params.iter().map(|p| p.name.clone()).collect();
    let generics: Vec<proc_macro2::Ident> = (0..param_names.len())
        .map(|i| format_ident!("__A{i}"))
        .collect();
    let scalar = quote! { ::pixelflow_core::__macro::ir::Scalar };
    let arity = param_names.len();

    Ok(quote! {
        {
            fn __builder< #( #generics: ::core::convert::Into<#scalar> ),* >()
                -> impl Fn( #( #generics ),* ) -> ::pixelflow_core::Kernel
            {
                move | #( #param_names: #generics ),* | {
                    let (mut __arena, __root) = #arena_code;
                    let __params: [#scalar; #arity] = [ #( #param_names.into() ),* ];
                    let __root = __arena.substitute_params(__root, &__params);
                    ::pixelflow_core::Kernel::from_parts(__arena, __root)
                }
            }
            __builder()
        }
    })
}
