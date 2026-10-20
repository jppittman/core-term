//! `ExprArena` → the `TokenStream` that rebuilds it at load time.
//!
//! The back end, and there is one. It produces a `pixelflow_core::Kernel` — an
//! arena fragment, the language's own value. Nothing is compiled at
//! macro-expansion time and nothing is compiled at construction: a `Kernel`
//! becomes machine code when a consumer compiles it at a lattice's shape and
//! collapses it (`Lattice::bake`), which is the only way a kernel turns into
//! numbers.
//!
//! Two steps, because the interesting thing happens between them: an arena is
//! lowered from the AST ([`crate::lower`]), *then* optionally rewritten, then
//! emitted. Emission takes an arena rather than an AST so that the optimizer
//! has somewhere to stand.
//!
//! What is emitted depends on how the block was spelled
//! ([`KernelDef`]). The closure form is an expression:
//! a `Kernel`, or a closure over its parameters' `f32`s returning one. The
//! items form is items: a host `#[repr(C)]` struct per record, a host `const`
//! per `pub const`, and per entry a host `fn` returning a `Kernel` and, when
//! it has uniform parameters and no kernel-typed one, its `Args` record. A
//! helper is inlined and a private `const` is folded, so neither leaves a
//! trace.
//!
//! **An entry that takes a kernel is staged** (plan Phase D-a). Its
//! argument, `k: impl Fn(f32, f32) -> f32`, is a `&Kernel` the host passes
//! when it calls the entry's host function, so the program does not exist
//! at expansion: the host function *is* lowering, run then. [`Staged`] is
//! the [`Site`] that writes each of lowering's steps as the statement that
//! takes it — the same IR call [`Expansion`] makes, in the same order —
//! and `k(x, y)` is one more, [`ExprArena::apply`]. No optimizer runs at
//! expansion on such an entry, as on a template: the composed program is
//! optimized when it is baked (§1.8).
//!
//! [`ExprArena::apply`]: pixelflow_ir::arena::ExprArena::apply
//!
//! **Binding times** (docs/language/2026-09-25-the-language-is-kernel.md §1.4).
//! A host function takes its parameters by their declared types, and every
//! one is a uniform: the kernel it returns declares one uniform per scalar —
//! a record parameter, one per field — in declaration order, with the call's
//! value as its default. So the call's values are the kernel's arguments,
//! never its constants: every call of an entry is one program, keyed and
//! compiled once, and a compiled program is rebound per call from the
//! entry's `Args` record. An entry's structural parameters (`const N:
//! usize`) are the host function's const generics, and each value is its
//! own program.

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::optimize::Optimize;
use pixelflow_ir::{Fold, OpKind};
use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::Ident;

use crate::ast::{Callable, ClosureEntry, ConstItem, FnItem, KernelDef, Name, RecordItem, Role};
use crate::lower::{Expansion, FoldRange, Holes, Lowered, Site, lower_entry};
use crate::sema::{AnalyzedKernel, ConstValue, Scalar};
use pixelflow_ir::arena::Axis;
use pixelflow_ir::{Binder, Monoid};

/// Emit arena-backend code for an analyzed kernel.
///
/// For the closure form, a token stream evaluating to:
/// - zero params — a `pixelflow_core::Kernel` value, built at load
///   time from the arena this expansion computed.
/// - N params — a closure `move |p0: f32, ...| -> Kernel` whose kernel
///   declares one uniform per parameter, its default the call's argument.
///   The closure sugar has no `Args` record: a program compiled from it is
///   rebound by position, with `UniformBlock::set_declared`.
///
/// For the items form: one struct per record, one `const` per `pub const`,
/// and for each entry its host `fn`, generic over its structural parameters
/// and taking its parameters by their declared types — a kernel-typed one as
/// a `&Kernel` — and its `Args` record. An entry that takes a kernel is
/// [`Staged`]: lowered when its host function is called.
///
/// `optimizer` rewrites each lowered arena before it is emitted. It is a
/// parameter rather than a branch because "do not optimize" is a value:
/// `kernel_raw!` passes [`Identity`](pixelflow_ir::optimize::Identity), and
/// that is the entire difference between the two macros.
///
/// # Errors
///
/// When lowering refuses what `sema` accepted: a fold bound past 2^24, folds
/// nested past `Binder::COUNT`, or more than 256 structural parameters read
/// as values.
pub(crate) fn emit_kernel(
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> syn::Result<TokenStream> {
    match &analyzed.def {
        KernelDef::Closure(entry) => {
            // A closure has no `Args` record to supply the scalars to.
            let (arena_code, _) = entry_arena(entry.callable(), analyzed, optimizer)?;
            Ok(emit_closure(entry, &arena_code))
        }
        KernelDef::Items {
            records,
            consts,
            fns,
        } => {
            let mut items = TokenStream::new();
            for record in records {
                items.extend(emit_record(record));
            }
            for c in consts.iter().filter(|c| is_pub(&c.vis)) {
                items.extend(emit_const(c, analyzed.consts[&Name::of(&c.name)]));
            }
            for entry in fns.iter().filter(|f| f.role() == Role::Entry) {
                if entry.takes_a_kernel() {
                    let body = Staged::lower(entry.callable(), analyzed)?;
                    items.extend(emit_entry(entry, &body));
                    continue;
                }
                let (arena_code, scalars) = entry_arena(entry.callable(), analyzed, optimizer)?;
                items.extend(emit_entry(entry, &kernel_from(&arena_code)));
                items.extend(emit_args(entry, &scalars));
            }
            Ok(items)
        }
    }
}

/// `quote!` for the emission's own tokens: every identifier written in it is
/// at the macro's mixed-site span, so a local it binds (`__arena`, `__t3`,
/// `__lo`) is a different variable from a user's of the same spelling, which
/// keeps the call site's. A user's tokens interpolated into it keep theirs.
///
/// Mixed-site hygiene covers locals and labels only; an item or a path
/// resolves at the call site as ever. So a structural parameter or a `const`
/// spelled like an emitted local (`const __lo: usize`) still meets it, and
/// rustc refuses that loudly (E0530); the `__` prefix is what keeps an
/// unprefixed one from meeting a structural parameter `lo`.
macro_rules! emitted {
    ($($tokens:tt)*) => {
        quote_spanned!(Span::mixed_site()=> $($tokens)*)
    };
}

/// A local of the emitted code, `prefix` then `n`.
fn local(prefix: &str, n: u64) -> Ident {
    Ident::new(&format!("{prefix}{n}"), Span::mixed_site())
}

/// The statement declaring `slot`'s uniform in the arena being built, with
/// the call's `default`: an identity minted per call, so an entry expanded at
/// compile time and one staged at run time declare an argument alike.
fn declare_uniform(slot: &Ident, default: &TokenStream) -> TokenStream {
    emitted! {
        let #slot = __arena.declare_uniform(::pixelflow_core::__macro::ir::arena::UniformDecl {
            id: ::pixelflow_core::__macro::ir::arena::UniformIdentity::mint(),
            default: #default,
        });
    }
}

/// The expression decoding `fold`, which travels as bits because its fields
/// are private precisely so no caller can assemble one that means nothing.
fn fold_from_bits(fold: Fold) -> TokenStream {
    let bits = fold.to_bits();
    emitted! {
        ::pixelflow_core::__macro::ir::fold::Fold::from_bits(#bits)
            .expect("kernel! emitted a well-formed fold")
    }
}

/// The `Kernel` made of the `(arena, root)` that `arena_code` evaluates to.
fn kernel_from(arena_code: &TokenStream) -> TokenStream {
    emitted! {
        let (__arena, __root) = #arena_code;
        ::pixelflow_core::Kernel::from_parts(__arena, __root)
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    !matches!(vis, syn::Visibility::Inherited)
}

/// Lower an entry, run the optimizer over it, and emit the code that
/// rebuilds the arena: an expression evaluating to `(arena, root)`, and the
/// scalars the lowering declared its uniforms for, which the host function
/// and its `Args` record supply by position.
///
/// An entry with structural parameters is a template: its open folds hold
/// placeholder ranges a rewrite would read as ranges (an empty one, to
/// `EmptyFold`), so it is emitted as lowered and optimized when its
/// instantiation is baked — "no optimization runs at expansion unless the
/// instance is declared" (plan §1.8, Phase E).
fn entry_arena(
    entry: Callable<'_>,
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> syn::Result<(TokenStream, Vec<Scalar>)> {
    let Lowered {
        arena,
        root,
        holes,
        scalars,
    } = Expansion::lower(entry, analyzed)?;
    let context = EntryContext {
        arguments: scalars.iter().map(argument).collect(),
        structural: entry.structural,
        holes: &holes,
    };
    if !entry.structural.is_empty() {
        return Ok((arena_to_tokens(&arena, root, &context), scalars));
    }

    // Declining is ordinary and needs no arm: the lowered term stands, and a
    // kernel that reaches the runtime tier unoptimized is optimized there.
    let (arena, root) = match optimizer.optimize(&arena, root).into_changed() {
        // Extraction declares the uniforms it reaches in its own walk order
        // and drops the ones it does not; relinking puts back the entry's
        // declaration order, every scalar in it, which is the order a
        // positional binding supplies them in.
        Some((optimized, optimized_root)) if !arena.uniforms().is_empty() => {
            optimized.relink(optimized_root, arena.buffers(), arena.uniforms())
        }
        Some(changed) => changed,
        None => (arena, root),
    };
    Ok((arena_to_tokens(&arena, root, &context), scalars))
}

/// What a scalar argument is in the host function: `param`, or
/// `param.field` for a record parameter's field.
fn argument(scalar: &Scalar) -> TokenStream {
    match scalar {
        Scalar::Param(param) => quote!(#param),
        Scalar::Field(param, field) => quote!(#param.#field),
    }
}

/// What an entry's arena is emitted against: the argument each declared
/// uniform defaults to, in declaration order, and — for a template — the
/// structural parameters and open ranges its holes are filled from.
struct EntryContext<'a> {
    arguments: Vec<TokenStream>,
    structural: &'a [Ident],
    holes: &'a Holes,
}

/// The closure form's expansion: a `Kernel`, or a closure over `f32`s.
fn emit_closure(entry: &ClosureEntry, arena_code: &TokenStream) -> TokenStream {
    let body = kernel_from(arena_code);
    if entry.params.is_empty() {
        return quote! {{ #body }};
    }
    let params = entry.params.iter().map(|p| {
        let (name, ty) = (&p.name, &p.ty);
        quote!(#name: #ty)
    });
    quote! {
        move | #( #params ),* | -> ::pixelflow_core::Kernel { #body }
    }
}

/// An entry's expansion: a host function returning a `Kernel` — `body`,
/// statements that build it — generic over its structural parameters and
/// taking its parameters by their declared types, with the entry's
/// visibility and doc comments.
///
/// A kernel-typed parameter is taken as a `&Kernel`, not a `Kernel`: the
/// host function only reads it, splicing a copy of its term into the
/// program it builds, and a borrow passes one instance to two parameters,
/// `sum2(&a, &a)`, as `Kernel::at` and `Lattice::bake` take theirs. Owning it would buy the
/// argument's arena only by building on top of it, which would declare its
/// uniforms before the entry's own.
fn emit_entry(entry: &FnItem, body: &TokenStream) -> TokenStream {
    let attrs = &entry.attrs;
    let vis = &entry.vis;
    let name = &entry.name;
    let (generics, _) = structural_generics(&entry.structural);
    let params = entry.params.iter().map(|p| {
        let (name, ty) = (&p.name, &p.ty);
        match p.is_kernel() {
            true => quote!(#name: &::pixelflow_core::Kernel),
            false => quote!(#name: #ty),
        }
    });
    quote! {
        #(#attrs)*
        #[must_use]
        #vis fn #name #generics ( #(#params),* ) -> ::pixelflow_core::Kernel {
            #body
        }
    }
}

/// An entry's `Args` record (plan §1.4): its parameters as fields, by their
/// declared types, generic over its structural parameters; and
/// `write_into`, which writes them — a record's fields in field order —
/// into a block by the entry's declaration order
/// (`UniformBlock::set_declared`), as an array literal whose length rustc
/// counts. An entry with no parameters has none, and neither has one that
/// takes a kernel ([`FnItem::has_args_record`]).
fn emit_args(entry: &FnItem, scalars: &[Scalar]) -> TokenStream {
    if !entry.has_args_record() {
        return TokenStream::new();
    }
    let vis = &entry.vis;
    let args = entry.args_record();
    let (generics, arguments) = structural_generics(&entry.structural);
    let entry_name = entry.name.to_string();
    let record_doc = format!(
        "The arguments of `{entry_name}`: a program compiled from its kernel is rebound per \
         call from one of these, with [`{args}::write_into`]."
    );
    let fields = entry.params.iter().map(|p| {
        let (name, ty) = (&p.name, &p.ty);
        let doc = format!("The argument `{name}` of `{entry_name}`.");
        quote! {
            #[doc = #doc]
            #vis #name: #ty,
        }
    });
    let values = scalars.iter().map(|scalar| {
        let value = argument(scalar);
        quote!(self.#value)
    });
    quote! {
        #[doc = #record_doc]
        #[derive(Clone, Copy, Debug, PartialEq)]
        #vis struct #args #generics {
            #(#fields)*
        }

        impl #generics #args #arguments {
            /// This call's values, written into `block` — one made by a
            /// program compiled from this entry's kernel
            /// (`Manifold::block`), kept and rewritten call after call:
            /// every uniform scalar in declaration order, a record's fields
            /// in field order.
            ///
            /// # Errors
            ///
            /// When `block`'s program declares a different number of
            /// arguments — it was compiled from some other kernel, or from
            /// this one composed beside other arguments. Nothing is
            /// written. The count is all that is checked: a block of
            /// another program that declares as many arguments takes these
            /// values by position (plan §1.4).
            #vis fn write_into(
                &self,
                block: &mut ::pixelflow_core::UniformBlock,
            ) -> ::core::result::Result<(), ::pixelflow_core::ArityMismatch> {
                block.set_declared([ #(#values),* ])
            }
        }
    }
}

/// An entry's structural parameters as generics: their declaration,
/// `<const N: usize, …>`, and their use, `<N, …>` — or nothing, for an
/// entry with none.
fn structural_generics(structural: &[Ident]) -> (TokenStream, TokenStream) {
    if structural.is_empty() {
        return (TokenStream::new(), TokenStream::new());
    }
    (
        quote!(< #( const #structural: usize ),* >),
        quote!(< #( #structural ),* >),
    )
}

/// A record's host twin (§1.3): a `#[repr(C)]` struct of the same name,
/// fields and visibility, with its attributes and its fields' — the type an
/// entry's parameter and its `Args` record name.
fn emit_record(record: &RecordItem) -> TokenStream {
    let attrs = &record.attrs;
    let vis = &record.vis;
    let name = &record.name;
    let fields = record.fields.iter().map(|field| {
        let (attrs, vis, name, ty) = (&field.attrs, &field.vis, &field.name, &field.ty);
        quote! {
            #(#attrs)*
            #vis #name: #ty,
        }
    });
    quote! {
        #(#attrs)*
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #vis struct #name {
            #(#fields)*
        }
    }
}

/// A `pub const`'s host twin, holding the value `sema` evaluated. An `f32`
/// by bit pattern, for the reason [`arena_to_tokens`] gives: it is exact,
/// and a const may be non-finite (`1.0 / 0.0`), which a decimal literal
/// cannot spell. A `usize` as the integer it is.
fn emit_const(item: &ConstItem, value: ConstValue) -> TokenStream {
    let attrs = &item.attrs;
    let vis = &item.vis;
    let name = &item.name;
    match value {
        ConstValue::F32(value) => {
            let bits = value.to_bits();
            quote! {
                #(#attrs)*
                #vis const #name: f32 = f32::from_bits(#bits);
            }
        }
        ConstValue::Usize(count) => {
            let count = proc_macro2::Literal::u64_unsuffixed(count);
            quote! {
                #(#attrs)*
                #vis const #name: usize = #count;
            }
        }
    }
}

/// An open fold's range, evaluated when the host function is instantiated:
/// in a `const` block, so rustc checks each operation of the bounds as it
/// checks any `const`, and refuses — per instantiation — the ranges the IR
/// refuses ([`Fold::admits`]): one that runs backwards, and one past
/// [`Fold::EXACT_BOUND`], the last bound a fold's index names exactly.
/// The bound is the IR's, imported; what the `const` block adds is that the
/// refusal is rustc's rather than `Fold::new`'s panic.
///
/// The refusal is rustc's at monomorphization, which `cargo build` reaches
/// and `cargo check` (so clippy) does not: a bad instantiation checks clean
/// and fails to build, in the calling crate, naming the range. It is never
/// a run-time panic.
fn instantiated_range(range: &crate::sema::StructuralRange) -> TokenStream {
    let (lo, hi) = (&range.lo, &range.hi);
    let bound = proc_macro2::Literal::u32_unsuffixed(Fold::EXACT_BOUND);
    let backwards = format!(
        "kernel!: the range `{}` runs backwards at this instantiation",
        range.text()
    );
    let past = format!(
        "kernel!: the range `{}` reaches past 2^24 at this instantiation: a fold's index is an \
         `f32` lane, which names every integer only that far",
        range.text()
    );
    emitted! {
        const {
            let __lo: usize = #lo;
            let __hi: usize = #hi;
            ::core::assert!(__lo <= __hi, #backwards);
            ::core::assert!(__hi <= #bound, #past);
            (__lo as u32)..(__hi as u32)
        }
    }
}

/// Emit the arena as code that rebuilds it at load time: every node the
/// root reaches, in the arena's order, children before parents.
///
/// `Dwrt` nodes are emitted as built: the macro tier never resolves one
/// (`macro_tier`), so a warp applied to the emitted kernel reaches the
/// `Dwrt`'s operand and the chain rule holds
/// (`tests/derivative_under_warp.rs`).
///
/// The arena's uniform table is declared first, one per argument of
/// `context`, in order: each with an identity minted per call and the
/// call's value as its default, so the kernel's arguments are that call's
/// and its program is every call's — every one declared, read or not, so a
/// positional binding cannot shift. A template's holes are filled from the
/// host function's structural parameters: a `Param(k)` is the `k`th one as
/// an `f32`, and an open fold's range is evaluated per instantiation
/// ([`instantiated_range`]).
fn arena_to_tokens(arena: &ExprArena, root: ExprId, context: &EntryContext) -> TokenStream {
    debug_assert_eq!(
        arena.uniforms().len(),
        context.arguments.len(),
        "lowering declares one uniform per argument, and a relink restores the table"
    );
    let decls = context
        .arguments
        .iter()
        .enumerate()
        .map(|(slot, value)| declare_uniform(&uniform_var(slot as u64), value));
    let nodes = reached(arena, root).into_iter().map(|id| {
        let var = node_var(id);
        let expr = node(arena, id, context);
        emitted! {
            let #var = #expr;
        }
    });
    let root_ident = node_var(root);
    emitted! {{
        let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
        #(#decls)*
        #(#nodes)*
        (__arena, #root_ident)
    }}
}

/// The variable holding a uniform slot's `UniformId` in the arena being
/// built.
fn uniform_var(slot: u64) -> Ident {
    local("__u", slot)
}

/// The variable holding a node's id in the arena being built.
fn node_var(id: ExprId) -> Ident {
    local("__e", u64::from(id.0))
}

/// The nodes `root` reaches, in the arena's order, children before parents:
/// what the emitted code builds. A node nothing reads — an unread
/// parameter's leaf — is not rebuilt; its uniform is still declared.
fn reached(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
    let mut reached = vec![false; arena.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reached[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }
    arena
        .nodes()
        .map(|(id, _)| id)
        .filter(|id| reached[id.0 as usize])
        .collect()
}

/// One node, built in `__arena` from the nodes before it.
fn node(arena: &ExprArena, id: ExprId, context: &EntryContext) -> TokenStream {
    match arena.node(id) {
        ExprNode::Var(i) => emitted! { __arena.push_var(#i) },
        // By bit pattern, not as a decimal literal: `quote`'s `f32`
        // impl goes through `Literal::f32_suffixed`, which asserts
        // `is_finite()` — and non-finite constants are ordinary here. A
        // true comparison mask is all-ones (`OpKind::mask`), which is
        // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
        // the folder produces those. Bits also roundtrip exactly, with
        // no decimal-formatting question to get wrong.
        ExprNode::Const(v) => {
            let bits = v.to_bits();
            emitted! { __arena.push_const(f32::from_bits(#bits)) }
        }
        // A template's hole: the structural parameter, as its `f32`, as
        // Rust's `as` rounds it — which is `N as f32` in the body.
        ExprNode::Param(k) => {
            let count = context.structural.get(usize::from(k)).unwrap_or_else(|| {
                panic!(
                    "kernel! produced ExprNode::Param({k}) outside a template's \
                     structural parameters"
                )
            });
            emitted! { __arena.push_const(#count as f32) }
        }
        // No body can spell a buffer: the language has none
        // (docs/DENOTATION.md §11), so one here is a front-end bug. Fail
        // loud rather than emit a node that references a buffer table the
        // emitted code never declares.
        ExprNode::Buffer(b) => {
            panic!(
                "kernel! produced ExprNode::Buffer({}) — the language has no buffers \
                 (docs/DENOTATION.md §11)",
                b.0
            )
        }
        // An argument: the slot declared above for it.
        ExprNode::Uniform(u) => {
            let slot = uniform_var(u.0);
            emitted! { __arena.push_uniform(#slot) }
        }
        // And a reference is minted by `Kernel::by_ref` at composition
        // time — a runtime value, and the key it carries names a store
        // in the *build host's* process, which the compiled program is
        // not. Emitting one would name nothing.
        ExprNode::Ref(k) => {
            panic!(
                "kernel! produced ExprNode::Ref({k:?}) — a reference names a kernel \
                 interned in this process, which the emitted program does not share"
            )
        }
        ExprNode::Unary(op, child) => {
            let op_code = opkind_to_tokens(op);
            let child = node_var(child);
            emitted! { __arena.push_unary(#op_code, #child) }
        }
        ExprNode::Binary(op, a, b) => {
            let op_code = opkind_to_tokens(op);
            let (a, b) = (node_var(a), node_var(b));
            emitted! { __arena.push_binary(#op_code, #a, #b) }
        }
        ExprNode::Ternary(op, a, b, c) => {
            let op_code = opkind_to_tokens(op);
            let (a, b, c) = (node_var(a), node_var(b), node_var(c));
            emitted! { __arena.push_ternary(#op_code, #a, #b, #c) }
        }
        ExprNode::Nary(op, ..) => {
            let op_code = opkind_to_tokens(op);
            let children: Vec<Ident> = arena.children(id).map(node_var).collect();
            emitted! { __arena.push_nary(#op_code, &[#(#children),*]) }
        }
        // A fold's metadata is a `Fold`, whose fields are private
        // precisely so no caller can assemble one that means nothing —
        // so it travels the way the two cache keys carry it, as bits
        // with a total inverse on the far side.
        ExprNode::Reduce { fold, body } => {
            let body = node_var(body);
            let decoded = fold_from_bits(fold);
            // An open fold keeps its monoid and binder, which are
            // structure, and takes its range from this instantiation.
            let emitted = match context.holes.range_of(fold) {
                None => decoded,
                Some(range) => {
                    let range = instantiated_range(range);
                    emitted! {{
                        let __placeholder = #decoded;
                        ::pixelflow_core::__macro::ir::fold::Fold::new(
                            __placeholder.monoid(),
                            __placeholder.binder(),
                            #range,
                        )
                    }}
                }
            };
            emitted! { __arena.push_reduce(#emitted, #body) }
        }
        // A store is post-legalize vocabulary: the passes that wrap a
        // kernel in the lattice's folds build one, after extraction,
        // and no kernel! body can spell it.
        ExprNode::Write { .. } => {
            panic!(
                "kernel! produced ExprNode::Write — a store has no surface syntax; \
                 the legalize passes build one after extraction"
            )
        }
    }
}

/// The site an entry that takes a kernel is lowered at: the Rust statements
/// that build its arena when its host function is called (plan Phase D-a).
///
/// Each of lowering's steps becomes the statement that takes it, naming a
/// term by the variable that holds its node: the same IR call [`Expansion`]
/// makes at expansion — a push, a [`library`](pixelflow_ir::library)
/// definition, [`ExprArena::open_fold`] and [`ExprArena::close_fold`] — in
/// the same order, so a program built at load time is the one lowering
/// would have built at expansion. The steps only `Staged` has are an
/// argument's: [`ExprArena::admit`] when the host function is called, after
/// the entry's own uniforms, and [`ExprArena::apply`] at each `k(x, y)`.
///
/// What is left is compacted ([`ExprArena::compact`]): a node a
/// substitution replaced or nothing read is dropped, as emission drops one
/// from an arena built at expansion, so the kernel holds the program alone.
///
/// A fold staged here closes when the host function runs, and panics then
/// if its body — an argument's folds among them — binds every binder, as
/// `Kernel::over` panics.
///
/// [`ExprArena::open_fold`]: pixelflow_ir::arena::ExprArena::open_fold
/// [`ExprArena::close_fold`]: pixelflow_ir::arena::ExprArena::close_fold
/// [`ExprArena::admit`]: pixelflow_ir::arena::ExprArena::admit
/// [`ExprArena::apply`]: pixelflow_ir::arena::ExprArena::apply
/// [`ExprArena::compact`]: pixelflow_ir::arena::ExprArena::compact
#[derive(Default)]
struct Staged {
    statements: Vec<TokenStream>,
    /// How many variables have been named: every one is fresh.
    named: u64,
    /// The folds open, innermost last: the variable holding each, and the
    /// closure that makes its `Fold` once its binder is chosen.
    open: Vec<(Ident, TokenStream)>,
}

impl Staged {
    /// The host function's body for `entry`, an entry that takes a kernel:
    /// statements building its kernel when it is called.
    ///
    /// # Errors
    ///
    /// When the body has a construct lowering cannot express.
    fn lower(entry: Callable<'_>, analyzed: &AnalyzedKernel) -> syn::Result<TokenStream> {
        let mut site = Staged::default();
        let root = lower_entry(entry, analyzed, &mut site)?;
        let statements = site.statements;
        let kernel = kernel_from(&emitted!(__arena.compact(#root)));
        Ok(emitted! {
            let mut __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::new();
            #(#statements)*
            #kernel
        })
    }

    /// A fresh variable, `prefix` then a number.
    fn fresh(&mut self, prefix: &str) -> Ident {
        let name = local(prefix, self.named);
        self.named += 1;
        name
    }

    /// The statement binding `step`'s node to a fresh variable, which names
    /// it.
    fn step(&mut self, step: TokenStream) -> Ident {
        let term = self.fresh("__t");
        self.statements.push(emitted! { let #term = #step; });
        term
    }

    /// The closure an open fold closes into: its monoid, carried as a
    /// fold's bits (as [`node`] carries a `Fold`), its binder the one
    /// `close_fold` chooses, and its range — known, or this
    /// instantiation's ([`instantiated_range`]).
    fn fold_at(monoid: Monoid, range: FoldRange) -> TokenStream {
        let first = Binder::from_slot(0).expect("the IR has a binder");
        let monoid_of = fold_from_bits(pixelflow_ir::Fold::new(monoid, first, 0..0));
        let range = match range {
            FoldRange::Known(range) => {
                let (lo, hi) = (range.start, range.end);
                emitted!(#lo..#hi)
            }
            FoldRange::Structural(range) => instantiated_range(&range),
        };
        emitted! {
            |__binder| ::pixelflow_core::__macro::ir::fold::Fold::new(
                #monoid_of.monoid(),
                __binder,
                #range,
            )
        }
    }
}

impl Site for Staged {
    type Term = Ident;
    type Argument = Ident;

    fn constant(&mut self, value: f32) -> Ident {
        // By bit pattern, for `node`'s reason.
        let bits = value.to_bits();
        self.step(emitted!(__arena.push_const(f32::from_bits(#bits))))
    }

    fn coordinate(&mut self, axis: Axis) -> Ident {
        let var = axis.var();
        self.step(emitted!(__arena.push_var(#var)))
    }

    fn unary(&mut self, op: pixelflow_ir::OpKind, a: Ident) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(emitted!(__arena.push_unary(#op, #a)))
    }

    fn binary(&mut self, op: pixelflow_ir::OpKind, [a, b]: [Ident; 2]) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(emitted!(__arena.push_binary(#op, #a, #b)))
    }

    fn ternary(&mut self, op: pixelflow_ir::OpKind, [a, b, c]: [Ident; 3]) -> Ident {
        let op = opkind_to_tokens(op);
        self.step(emitted!(__arena.push_ternary(#op, #a, #b, #c)))
    }

    fn fract(&mut self, x: Ident) -> Ident {
        self.step(emitted!(::pixelflow_core::__macro::ir::library::fract(&mut __arena, #x)))
    }

    fn hypot(&mut self, [a, b]: [Ident; 2]) -> Ident {
        self.step(emitted!(::pixelflow_core::__macro::ir::library::hypot(&mut __arena, [#a, #b])))
    }

    fn clamp(&mut self, x: Ident, [lo, hi]: [Ident; 2]) -> Ident {
        self.step(emitted!(
            ::pixelflow_core::__macro::ir::library::clamp(&mut __arena, #x, [#lo, #hi])
        ))
    }

    fn derivative(&mut self, e: Ident, axis: Axis) -> Ident {
        let axis = match axis {
            Axis::X => emitted!(::pixelflow_core::__macro::ir::arena::Axis::X),
            Axis::Y => emitted!(::pixelflow_core::__macro::ir::arena::Axis::Y),
        };
        self.step(emitted!(
            ::pixelflow_core::__macro::ir::library::derivative(&mut __arena, #e, #axis)
        ))
    }

    /// Declared with the call's value as its default, as [`arena_to_tokens`]
    /// declares an expanded entry's, so baking the kernel draws the call.
    fn uniform(&mut self, scalar: Scalar) -> Ident {
        let slot = self.fresh("__u");
        self.statements
            .push(declare_uniform(&slot, &argument(&scalar)));
        self.step(emitted!(__arena.push_uniform(#slot)))
    }

    fn kernel_parameter(&mut self, name: &Ident) -> syn::Result<Ident> {
        let argument = self.fresh("__k");
        self.statements
            .push(emitted! { let #argument = __arena.admit(#name); });
        Ok(argument)
    }

    /// The host function's own `N as f32`: it is generic over `N`.
    fn count(&mut self, _position: usize, name: &Ident) -> syn::Result<Ident> {
        Ok(self.step(emitted!(__arena.push_const(#name as f32))))
    }

    fn open_fold(&mut self, depth: usize, monoid: Monoid, range: FoldRange) -> Ident {
        let fold = self.fresh("__f");
        let depth = proc_macro2::Literal::usize_unsuffixed(depth);
        self.statements.push(emitted! {
            let #fold = __arena
                .open_fold(#depth)
                .expect("kernel! refused a fold nested past the binders at expansion");
        });
        self.open.push((fold.clone(), Self::fold_at(monoid, range)));
        self.step(emitted!(#fold.index()))
    }

    fn close_fold(&mut self, body: Ident) -> Ident {
        let (fold, fold_at) = self
            .open
            .pop()
            .expect("lowering closes only the folds it opened");
        self.step(emitted! {
            __arena.close_fold(#fold, #body, #fold_at).expect(
                "kernel!: a fold around a kernel's application has no binder left: its body, \
                 the argument's folds among them, binds every one of the IR's indices \
                 (`Binder::COUNT`)"
            )
        })
    }

    fn apply(&mut self, kernel: Ident, [x, y]: [Ident; 2]) -> Ident {
        self.step(emitted!(__arena.apply(&#kernel, [#x, #y])))
    }
}

/// The path naming `kind` in generated code.
///
/// [`OpKind::variant_name`] is generated by `op_table!`, so the identifier
/// cannot drift from the enum and a newly added op needs no edit here.
fn opkind_to_tokens(kind: OpKind) -> TokenStream {
    let variant = format_ident!("{}", kind.variant_name());
    quote! { ::pixelflow_core::__macro::ir::OpKind::#variant }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::front_end;
    use pixelflow_ir::optimize::Identity;
    use quote::quote;

    /// The expansion of `input`, unoptimized, as text.
    fn expansion(input: proc_macro2::TokenStream) -> String {
        front_end(input, &mut Identity)
            .expect("expands")
            .to_string()
    }

    /// A record is a host `#[repr(C)]` struct, and no path the expansion
    /// names is `::std`: a `no_std` crate expands any block. One block
    /// exercises every emission path (a record, a `pub const`, a helper, a
    /// fold over `N`, an entry taking a kernel).
    #[test]
    fn a_record_is_repr_c_and_the_expansion_names_no_std_path() {
        let code = expansion(quote! {
            pub struct Bounds { pub x0: f32, pub x1: f32 }
            pub const TWO: f32 = 2.0;
            fn sq(x: f32) -> f32 { x * x }
            pub fn inside<const N: usize>(b: Bounds, r: f32) -> f32 {
                sq(r) + (0..N).map(|i| X * (i as f32)).sum::<f32>() + b.x0
            }
            pub fn glyph(ink: impl Fn(f32, f32) -> f32, b: Bounds) -> f32 {
                if (X > b.x0) & (X < b.x1) { ink(X + 0.5, Y) } else { 0.0 }
            }
        });
        assert!(code.contains("# [repr (C)]"), "{code}");
        // Spelled as `to_string` spaces a path: `"::std"` matches nothing.
        // `items_block.rs`'s `#[no_implicit_prelude]` module still resolves
        // `::std`, so this grep is the only check that a `no_std` crate can
        // expand any block.
        assert!(
            !code.contains(":: std ::"),
            "no `::std` path, so a `no_std` crate expands it: {code}"
        );
    }

    /// A template's open fold takes its range from the instantiation, in a
    /// `const` block that refuses at the instantiation what `sema` refuses
    /// of a known range.
    #[test]
    fn a_template_refuses_at_instantiation_what_sema_refuses_of_a_known_range() {
        let code = expansion(quote! {
            pub fn mean<const N: usize>() -> f32 {
                (0..N).map(|i| X * (i as f32)).sum::<f32>() / (N as f32)
            }
        });
        assert!(
            code.contains("runs backwards at this instantiation"),
            "{code}"
        );
        assert!(code.contains("__hi <= 16777216"), "{code}");
    }

    /// An optimizer that counts the arenas it is offered, and declines each.
    struct Offered(usize);

    impl Optimize for Offered {
        fn optimize(&mut self, _arena: &ExprArena, _root: ExprId) -> pixelflow_ir::Rewritten {
            self.0 += 1;
            pixelflow_ir::Rewritten::Declined
        }
    }

    /// No optimizer sees a template, whose open folds hold placeholder ranges
    /// a rewrite would read as ranges, nor an entry that takes a kernel,
    /// whose program exists only once its host function is called: both are
    /// emitted as lowered. An entry with neither is offered.
    #[test]
    fn an_optimizer_is_offered_only_entries_it_can_read() {
        let mut offered = Offered(0);
        front_end(
            quote! {
                pub struct Bounds { pub x0: f32, pub x1: f32 }
                pub fn counted<const N: usize>(r: f32) -> f32 {
                    r + (0..N).map(|i| (i as f32) * X).sum::<f32>()
                }
                pub fn glyph(ink: impl Fn(f32, f32) -> f32, b: Bounds) -> f32 {
                    if (X > b.x0) & (X < b.x1) { ink(X + 0.5, Y) } else { 0.0 }
                }
                pub fn sum2(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
                    a(X, Y) + b(X, Y)
                }
                pub fn plain(r: f32) -> f32 { r + X }
            },
            &mut offered,
        )
        .expect("expands");
        assert_eq!(offered.0, 1, "`plain` alone");
    }
}
