//! Arena backend codegen for `kernel_jit!` and `kernel_value!`.
//!
//! Every macro optimizes an [`AnalyzedKernel`] identically (e-graph saturation
//! then latency-prior extraction); the only difference is the backend that
//! consumes the optimized AST. This module is the arena backend: it lowers the
//! optimized body to an [`ExprArena`](pixelflow_ir::arena::ExprArena) and emits
//! code that reconstructs it at load time as a
//! [`Kernel`](pixelflow_core::Kernel).
//!
//! A `Kernel` is the whole surface: a consumer hands it to `Lattice::bake`
//! together with the lattice it wants tabulated, and the compiler owns the loop
//! nest. There is no per-batch entry — no `Manifold` impl calling into the JIT
//! once per SIMD batch — so nothing here emits one.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ast::ParamKind;
use crate::ir_bridge;
use crate::sema::AnalyzedKernel;

/// `None` (the default `Field` domain/return) or an explicit `Field` annotation.
/// Anything else (`Jet2`, `Jet3`, `Discrete`, tuples) is not the JIT's lane.
pub(crate) fn is_field_ty(ty: &Option<syn::Type>) -> bool {
    match ty {
        None => true,
        // Compare on the type's final path segment so both `Field` and
        // `pixelflow_core::Field` match, while `Jet3`/`Discrete` do not.
        Some(syn::Type::Path(p)) => p
            .path
            .segments
            .last()
            .is_some_and(|seg| seg.ident == "Field"),
        Some(_) => false,
    }
}

/// Emit arena-backend code for an (already optimized) kernel.
///
/// On success, returns a token stream evaluating to:
/// - zero params — a [`Kernel`](pixelflow_core::Kernel) value, built at load
///   time from the macro-time-optimized arena.
/// - N params — a builder closure `move |p0, ...| -> Kernel`. Scalar params
///   are substituted as constants; manifold params are spliced through
///   `Lower`, so a builder call composes ONE fused arena rather than N
///   kernels calling each other.
///
/// Nothing is compiled here. A `Kernel` is an arena fragment; machine code is
/// emitted when a consumer bakes it over a lattice, which is the only
/// evaluation entry there is.
///
/// Returns `Err` if the body contains an operation the IR bridge cannot lower.
pub fn emit_jit(analyzed: &AnalyzedKernel) -> Result<TokenStream, String> {
    // Scalar params (dense over scalars, declaration order) become `Param(i)`
    // arena nodes constant-folded at build time; manifold params become
    // reserved slot variables (`Var(8+k)`) substituted with the argument
    // kernels' spliced fragments at build time.
    let scalar_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Scalar(_)))
        .collect();
    let manifold_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Manifold))
        .collect();
    if manifold_params.len() > ir_bridge::MAX_MANIFOLD_PARAMS {
        return Err(format!(
            "kernel has {} manifold params; the arena backend supports at most {}",
            manifold_params.len(),
            ir_bridge::MAX_MANIFOLD_PARAMS
        ));
    }

    let param_map = ir_bridge::scalar_param_indices(analyzed);
    let manifold_map = ir_bridge::manifold_param_indices(analyzed);
    let (arena_code, plan) =
        ir_bridge::ast_to_runtime_arena(&analyzed.def.body, &param_map, &manifold_map)?;

    if scalar_params.is_empty() && manifold_params.is_empty() {
        return Ok(quote! {
            {
                let (__arena, __root) = #arena_code;
                ::pixelflow_core::Kernel::from_parts(__arena, __root)
            }
        });
    }

    // Builder closure that composes on call. Args appear in declaration order;
    // scalar params are typed `f32`, manifold params are untyped (closures
    // cannot be generic — the single call site's inference binds each to any
    // `Lower` kernel).
    let arg_tokens: Vec<TokenStream> = analyzed
        .def
        .params
        .iter()
        .map(|p| {
            let name = &p.name;
            match p.kind {
                ParamKind::Scalar(_) => quote! { #name: f32 },
                ParamKind::Manifold => quote! { #name },
            }
        })
        .collect();

    // Composition: splice bare fragments and per-`.at()`-site warped
    // fragments, then substitute every slot (shared logic with named
    // kernel structs — see `ir_bridge::composition_stmts`).
    let manifold_accessors: Vec<TokenStream> = manifold_params
        .iter()
        .map(|p| {
            let name = &p.name;
            quote! { #name }
        })
        .collect();
    let compose = ir_bridge::composition_stmts(&plan, &manifold_accessors);

    // Scalar values, dense in scalar declaration order (matches the
    // `Param(i)` numbering from `scalar_param_indices`).
    let scalar_names: Vec<proc_macro2::Ident> =
        scalar_params.iter().map(|p| p.name.clone()).collect();
    let param_slice = quote! { &[ #( #scalar_names as f32 ),* ] };

    Ok(quote! {
        move | #( #arg_tokens ),* | {
            let (mut __arena, mut __root) = #arena_code;
            #compose
            __root = __arena.substitute_params(__root, #param_slice);
            ::pixelflow_core::Kernel::from_parts(__arena, __root)
        }
    })
}

/// Emit code that builds a [`Kernel`](pixelflow_core::Kernel) *value* — an
/// uncompiled arena fragment — rather than a JIT-compiled manifold.
///
/// This is the "JIT-first" front-end surface: the result is the language value
/// that consumers *compose* (`Kernel::sum`/`at`/`select`/arithmetic) and bake
/// once at a root, never a per-instance JIT (the plan's "leaves are
/// bake-time-only, fuse at roots" constraint — a font glyph builds thousands of
/// leaf kernels but compiles one fused arena). Scalar params are constant-folded
/// into the fragment at build time; manifold params are unsupported here on
/// purpose — composition happens at the `Kernel` value level, not through macro
/// slots.
///
/// Zero scalar params → a `Kernel` value directly. N scalar params → a builder
/// closure `move |p0: f32, ...| -> Kernel`.
///
/// Returns `Err` if the body needs a construct outside this lane (named struct,
/// non-`Field` domain/return, manifold params, or an op the IR bridge rejects).
pub fn emit_kernel_value(analyzed: &AnalyzedKernel) -> Result<TokenStream, String> {
    if analyzed.def.struct_decl.is_some() {
        return Err("kernel_value! does not support named struct kernels; \
                    build the fragment anonymously and compose Kernel values"
            .into());
    }
    if !is_field_ty(&analyzed.def.domain_ty) || !is_field_ty(&analyzed.def.return_ty) {
        return Err("kernel_value! supports only the Field domain/return; \
                    derivatives are resolved symbolically at bake, not via a jet domain"
            .into());
    }
    if analyzed
        .def
        .params
        .iter()
        .any(|p| matches!(p.kind, ParamKind::Manifold))
    {
        return Err("kernel_value! does not support manifold params; \
                    compose Kernel values with Kernel::at/sum/select instead"
            .into());
    }

    let scalar_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Scalar(_)))
        .collect();

    let param_map = ir_bridge::scalar_param_indices(analyzed);
    let manifold_map = ir_bridge::manifold_param_indices(analyzed); // empty by the gate above
    let (arena_code, _plan) =
        ir_bridge::ast_to_runtime_arena(&analyzed.def.body, &param_map, &manifold_map)?;

    if scalar_params.is_empty() {
        Ok(quote! {
            {
                let (__arena, __root) = #arena_code;
                ::pixelflow_core::Kernel::from_parts(__arena, __root)
            }
        })
    } else {
        let arg_tokens: Vec<TokenStream> = scalar_params
            .iter()
            .map(|p| {
                let name = &p.name;
                quote! { #name: f32 }
            })
            .collect();
        // Scalar values, dense in scalar declaration order (matches the
        // `Param(i)` numbering from `scalar_param_indices`).
        let scalar_names: Vec<proc_macro2::Ident> =
            scalar_params.iter().map(|p| p.name.clone()).collect();
        let param_slice = quote! { &[ #( #scalar_names as f32 ),* ] };
        Ok(quote! {
            move | #( #arg_tokens ),* | {
                let (mut __arena, __root) = #arena_code;
                let __root = __arena.substitute_params(__root, #param_slice);
                ::pixelflow_core::Kernel::from_parts(__arena, __root)
            }
        })
    }
}
