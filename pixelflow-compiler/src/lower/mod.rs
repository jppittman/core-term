//! Macro AST → a term, built at a [`Site`].
//!
//! The front end's one lowering step: the surface syntax a user wrote becomes
//! the IR everything downstream speaks. `let` bindings resolve to the term
//! they name, so the arena is a DAG and a shared subexpression is one node;
//! operators and DSL methods resolve through [`OpKind`], so the op table is
//! not restated here.
//!
//! A helper is inlined at each call — β-reduction. Its arguments are lowered
//! in the caller's scope, once each, and its body is lowered in a scope of
//! its own where its parameters name those nodes: the callee sees its
//! parameters, the block's `const`s and nothing of the caller's, which is
//! what lexical scoping means. A `const` lowers to the value `sema` gave it.
//! An `if` lowers to [`OpKind::If`], the same node `.select` does.
//!
//! **What the IR defines, lowering calls; it restates none of it** (plan
//! §1.1, B5). The library methods (`fract`, `hypot`, `clamp`) and the
//! derivative projections build through [`pixelflow_ir::library`], the definitions
//! `Kernel`'s methods build through; the coordinates are [`Axis`]'s; a
//! fold is opened and closed by [`pixelflow_ir::arena::ExprArena::open_fold`] and
//! [`pixelflow_ir::arena::ExprArena::close_fold`], whose binder is chosen and placeholder renamed
//! by [`pixelflow_ir::arena::ExprArena::close_over`], as `Kernel::over`'s are; a range is admitted by
//! [`pixelflow_ir::Fold::admits`], in `sema`; a kernel is applied by
//! [`pixelflow_ir::arena::ExprArena::apply`]. So each
//! construction written here and the same one built with the builder are
//! one program (`tests/the_library_is_the_builders.rs`,
//! `tests/fold_is_kernel_over.rs`, `tests/kernel_typed_parameters.rs`).
//!
//! A fold lowers to one `Reduce` node over its [`pixelflow_ir::Fold`], built as
//! `Kernel::over` builds it: the body against a placeholder, then closed
//! over it. Nothing here unrolls: that is the e-graph's (`HalveFold`,
//! `PeelFold`), when the kernel is baked.
//!
//! An entry's parameters are its uniforms
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.4): each scalar — an
//! `f32` parameter, or one field of a record parameter — is declared as a
//! uniform, in [`crate::sema::AnalyzedKernel::parameters`]' order, and read through its
//! `Uniform` leaf. Nothing a call passes is a constant of the program, so
//! every call of an entry is one program.
//!
//! **Where a term is built is a [`Site`]** (Phase D-a). Lowering is one walk,
//! and there are two places it can build:
//!
//! - [`Expansion`]: an [`ExprArena`], at macro expansion — every entry that
//!   takes no kernel. Its declarations hold a placeholder default, and
//!   emission declares each with the call's value; an entry with structural
//!   parameters lowers to a *template* whose folds over a range that names
//!   one, and its `N as f32`s, are left open (`site.rs`'s `Holes`, which
//!   [`Lowered::range_of`] reads) and filled when its host function is
//!   instantiated.
//! - `emit::Staged`: Rust statements that build the arena
//!   when the entry's host function is called — an entry that takes a
//!   kernel-typed parameter, whose argument exists only then. Each step is
//!   the step [`Expansion`] takes, run later: the same IR calls, in the same
//!   order.
//!
//! A kernel-typed parameter is admitted when the host function is called,
//! after the entry's own uniforms ([`crate::ast::Callable::kernel_parameters`]),
//! and `k(x, y)` is [`Site::apply`]. [`Expansion`] has no argument to apply
//! — its [`Site::Argument`] is uninhabited — so no entry that takes one can
//! be lowered there.
//!
//! # Contract
//!
//! One stage, lowering, whose output is the IR: the one representation
//! boundary where two stages can disagree about what the language is. The
//! contract is [`Site`], the one real trait here — where a term is built,
//! with two implementations that differ only in when a step runs — and the
//! shape below, implemented by [`lower_entry`], [`Expansion`] and
//! [`Lowered`], each `pub(crate)` where it is implemented and re-exported
//! here. What a template leaves open (`Holes`, its placeholder range and
//! stride) is private to `site.rs`: emission reads an open fold's range
//! through [`Lowered::range_of`] and nothing else. Every other item under
//! `lower/` is the implementation and is `pub(super)` or private. Every
//! fallible entry returns `syn::Result`, `Err` a `syn::Error` spanned at the
//! construct.
//!
//! ```ignore
//! /// Implemented by `lower::lower_entry` and the types below.
//! trait Lower {
//!     /// `entry`'s body as a term at `site`: uniforms declared first, every scalar read or not,
//!     /// in `analyzed.parameters(entry.params)` order; kernel parameters admitted next in
//!     /// `kernel_parameters()` order; then the body, children first, helpers inlined (β), consts
//!     /// folded, each fold one `Reduce`, a `let` one shared node. Err: a coordinate in a helper;
//!     /// a fold at depth `Binder::COUNT` counting inlined ones.
//!     fn lower_entry<S: Site>(entry: Callable<'_>, analyzed: &AnalyzedKernel, site: &mut S) -> syn::Result<S::Term>;
//!     /// `lower_entry` at a fresh arena. Err: `lower_entry`'s, and an entry that takes a kernel.
//!     fn Expansion::lower(entry: Callable<'_>, analyzed: &AnalyzedKernel) -> syn::Result<Lowered>;
//!     /// The range of one of this template's open folds; `None` for a known fold; panics on a
//!     /// placeholder no hole of this entry names.
//!     fn Lowered::range_of(&self, fold: Fold) -> Option<&StructuralRange>;
//! }
//! /// An entry lowered at expansion: its arena and root, its uniform scalars in declaration
//! /// order, and what the arena leaves open if it is a template.
//! struct Lowered { pub(crate) arena: ExprArena, pub(crate) root: ExprId, pub(crate) scalars: Vec<Scalar>, holes: Holes /* private */ }
//! /// The site an entry that takes no kernel is lowered at: an `ExprArena`, at macro expansion.
//! struct Expansion;   // Default; its fields are private
//! ```

mod site;
mod walk;

use crate::sema::{Bounds, Scalar};
use pixelflow_ir::arena::Axis;
use pixelflow_ir::{Monoid, OpKind};
use syn::Ident;

pub(crate) use site::{Expansion, Lowered};
pub(crate) use walk::lower_entry;

/// Somewhere lowering builds a term: each step of the walk, as a call.
///
/// Two of them, and they differ only in *when* a step runs ([`Expansion`]
/// at expansion, `Staged` when the entry's host function is called) — every step is
/// the same IR call either way, so the walk is written once, here, and a
/// program an entry builds at load time is the one it would have built at
/// expansion.
pub(crate) trait Site {
    /// How a term built here is named.
    type Term: Clone;
    /// What a kernel-typed parameter is bound to here.
    type Argument: Clone;

    /// The literal `value`.
    fn constant(&mut self, value: f32) -> Self::Term;
    /// The coordinate `axis`.
    fn coordinate(&mut self, axis: Axis) -> Self::Term;
    /// `op(operand)`.
    fn unary(&mut self, op: OpKind, operand: Self::Term) -> Self::Term;
    /// `op(a, b)`.
    fn binary(&mut self, op: OpKind, operands: [Self::Term; 2]) -> Self::Term;
    /// `op(a, b, c)`.
    fn ternary(&mut self, op: OpKind, operands: [Self::Term; 3]) -> Self::Term;
    /// [`pixelflow_ir::library::fract`].
    fn fract(&mut self, x: Self::Term) -> Self::Term;
    /// [`pixelflow_ir::library::hypot`].
    fn hypot(&mut self, operands: [Self::Term; 2]) -> Self::Term;
    /// [`pixelflow_ir::library::clamp`].
    fn clamp(&mut self, x: Self::Term, bounds: [Self::Term; 2]) -> Self::Term;
    /// [`pixelflow_ir::library::derivative`].
    fn derivative(&mut self, e: Self::Term, axis: Axis) -> Self::Term;
    /// One of the entry's uniform scalars, declared next, and its leaf.
    fn uniform(&mut self, scalar: Scalar) -> Self::Term;
    /// The entry's kernel-typed parameter `name`, admitted next
    /// ([`pixelflow_ir::arena::ExprArena::admit`]).
    ///
    /// # Errors
    ///
    /// Where there is no argument to admit.
    fn kernel_parameter(&mut self, name: &Ident) -> syn::Result<Self::Argument>;
    /// `name as f32`, the entry's `position`th structural parameter.
    ///
    /// # Errors
    ///
    /// Where the position has no room.
    fn count(&mut self, position: usize, name: &Ident) -> syn::Result<Self::Term>;
    /// Open a fold over `bounds` under `monoid`, `depth` folds being open
    /// already: its index ([`pixelflow_ir::arena::ExprArena::open_fold`]).
    ///
    /// # Panics
    ///
    /// At `depth` [`pixelflow_ir::Binder::COUNT`] or more: lowering refuses a fold nested
    /// that deep before it opens one.
    fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: Bounds) -> Self::Term;
    /// Close the innermost open fold over `body` ([`pixelflow_ir::arena::ExprArena::close_fold`]).
    fn close_fold(&mut self, body: Self::Term) -> Self::Term;
    /// `kernel` applied at `(x, y)` ([`pixelflow_ir::arena::ExprArena::apply`]).
    fn apply(&mut self, kernel: Self::Argument, at: [Self::Term; 2]) -> Self::Term;
}
