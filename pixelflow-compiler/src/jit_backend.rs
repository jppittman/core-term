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
use quote::quote;

use crate::ir_bridge;
use crate::sema::AnalyzedKernel;

/// Emit arena-backend code for an analyzed kernel.
///
/// On success, returns a token stream evaluating to:
/// - zero params — a [`Kernel`](pixelflow_core::Kernel) value, built at load
///   time from the arena this expansion computed.
/// - N params — a builder closure `move |p0: f32, ...| -> Kernel`, whose
///   arguments are constant-folded into the fragment when it runs. No JIT:
///   leaves are bake-time-only and fuse at a root, which is what lets a font
///   build thousands of leaf kernels and compile one arena.
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

    // Builder closure. Arguments appear in declaration order and are typed
    // `f32`: `substitute_params` folds them into the fragment as constants,
    // which is the only thing a parameter has ever been on this backend.
    let arg_tokens: Vec<TokenStream> = analyzed
        .def
        .params
        .iter()
        .map(|p| {
            let name = &p.name;
            quote! { #name: f32 }
        })
        .collect();
    let param_names: Vec<proc_macro2::Ident> =
        analyzed.def.params.iter().map(|p| p.name.clone()).collect();
    let param_slice = quote! { &[ #( #param_names as f32 ),* ] };

    Ok(quote! {
        move | #( #arg_tokens ),* | {
            let (mut __arena, __root) = #arena_code;
            let __root = __arena.substitute_params(__root, #param_slice);
            ::pixelflow_core::Kernel::from_parts(__arena, __root)
        }
    })
}
