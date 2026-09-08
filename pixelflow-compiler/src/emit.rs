//! `ExprArena` → the `TokenStream` that rebuilds it at load time.
//!
//! The back end, and there is one. It produces a [`Kernel`] — an arena
//! fragment, the language's own value. Nothing is compiled at
//! macro-expansion time and nothing is compiled at construction: a `Kernel`
//! becomes machine code when a consumer compiles it at a lattice's shape and
//! collapses it (`Lattice::bake`), which is the only way a kernel turns into
//! numbers. So there is no per-batch entry — no `Manifold` impl calling into
//! the JIT once per SIMD batch — and nothing here emits one.
//!
//! Two steps, because the interesting thing happens between them: an arena is
//! lowered from the AST ([`crate::lower`]), *then* optionally rewritten, then
//! emitted. Emission takes an arena rather than an AST so that the optimizer
//! has somewhere to stand.
//!
//! [`Kernel`]: pixelflow_core::Kernel

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::lower;
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
/// by inlining a manifold through a macro slot, so there is no
/// manifold-typed parameter and nothing here to lower one with.
///
/// Returns `Err` if the body contains an operation the IR bridge cannot lower.
pub fn emit_kernel(analyzed: &AnalyzedKernel) -> Result<TokenStream, String> {
    let param_map = lower::param_indices(analyzed);
    let mut arena = ExprArena::new();
    let root = lower::ast_to_arena(&analyzed.def.body, &param_map, &mut arena)?;
    let arena_code = arena_to_tokens(&arena, root);

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

/// Emit the arena, node for node, as code that rebuilds it at load time.
///
/// `Dwrt` nodes are emitted as they were built and resolved at bake time, by
/// the one `LowerDwrt` pass in the runtime pipeline. Resolving them at
/// expansion time — which this front end used to do — is not an optimization
/// but a miscompilation under composition: `Kernel::at` warps a kernel by
/// substituting into its `Var` leaves, so a surviving `Dwrt(f, x)` has the
/// warp reach its *operand* and differentiates the warped function, which is
/// the chain rule. A `Dwrt` already resolved to `f'` has no operand left for
/// the warp to reach, and the substitution silently lands inside `f'`.
///
/// See docs/plans/2026-09-08-macro-tier-is-arena-native.md.
pub fn arena_to_tokens(arena: &ExprArena, root: ExprId) -> TokenStream {
    let nodes = arena.nodes_raw();
    let nary_children = arena.nary_children_raw();

    let node_tokens: Vec<TokenStream> = nodes
        .iter()
        .map(|node| match node {
            pixelflow_ir::arena::ExprNode::Var(i) => {
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Var(#i) }
            }
            // By bit pattern, not as a decimal literal: `quote`'s `f32`
            // impl goes through `Literal::f32_suffixed`, which asserts
            // `is_finite()` — and non-finite constants are ordinary here. A
            // true comparison mask is all-ones (`OpKind::mask`), which is
            // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
            // the folder now produces those. Bits also roundtrip exactly, with
            // no decimal-formatting question to get wrong.
            pixelflow_ir::arena::ExprNode::Const(v) => {
                let bits = v.to_bits();
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Const(f32::from_bits(#bits)) }
            }
            pixelflow_ir::arena::ExprNode::Param(i) => {
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Param(#i) }
            }
            // The `kernel!` macro has no buffer surface yet, so this is
            // unreachable in practice; fail loud rather than emit a node that
            // references a buffer table `from_raw` does not reconstruct.
            pixelflow_ir::arena::ExprNode::Buffer(b) => {
                panic!(
                    "kernel! produced ExprNode::Buffer({}) — lattice parameters are not wired \
                     into the compiler yet (KERNELS_AND_LATTICES.md M4)",
                    b.0
                )
            }
            // Likewise unreachable: a uniform enters a kernel at the builder
            // call (`substitute_params`), never from the macro's own arena.
            pixelflow_ir::arena::ExprNode::Uniform(u) => {
                panic!(
                    "kernel! produced ExprNode::Uniform({}) — uniforms are chosen at the \
                     builder call site, not in the macro body",
                    u.0
                )
            }
            pixelflow_ir::arena::ExprNode::Unary(op, child) => {
                let op_code = opkind_to_tokens(*op);
                let child = child.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Unary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#child)) }
            }
            pixelflow_ir::arena::ExprNode::Binary(op, a, b) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Binary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#a), ::pixelflow_core::__macro::ir::arena::ExprId(#b)) }
            }
            pixelflow_ir::arena::ExprNode::Ternary(op, a, b, c) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                let c = c.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Ternary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#a), ::pixelflow_core::__macro::ir::arena::ExprId(#b), ::pixelflow_core::__macro::ir::arena::ExprId(#c)) }
            }
            pixelflow_ir::arena::ExprNode::Nary(op, start, len) => {
                let op_code = opkind_to_tokens(*op);
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Nary(#op_code, #start, #len) }
            }
        })
        .collect();

    let child_tokens: Vec<TokenStream> = nary_children
        .iter()
        .map(|id| {
            let id = id.0;
            quote! { ::pixelflow_core::__macro::ir::arena::ExprId(#id) }
        })
        .collect();

    let root = root.0;
    let tokens = quote! {{
        let __nodes = vec![#(#node_tokens),*];
        let __nary_children = vec![#(#child_tokens),*];
        let __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::from_raw(__nodes, __nary_children);
        (__arena, ::pixelflow_core::__macro::ir::arena::ExprId(#root))
    }};
    tokens
}

/// The path naming `kind` in generated code.
///
/// One line per op used to live here — 40 of the 50, closing with a
/// `_ => panic!("Unsupported OpKind for JIT")` that refused ops the arena
/// holds and codegen emits perfectly well (`Reduce`, the integer-domain ops,
/// `Gather`). It was the fourth independently-maintained copy of the op
/// table, and the third one found drifting from it this week.
///
/// [`OpKind::variant_name`] is generated by `op_table!`, so the identifier
/// cannot drift from the enum and a newly added op needs no edit here.
fn opkind_to_tokens(kind: OpKind) -> TokenStream {
    let variant = format_ident!("{}", kind.variant_name());
    quote! { ::pixelflow_core::__macro::ir::OpKind::#variant }
}
