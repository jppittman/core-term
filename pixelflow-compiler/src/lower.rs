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
//! derivative projections build through [`library`], the definitions
//! `Kernel`'s methods build through; the coordinates are [`Axis`]'s; a
//! fold is opened and closed by [`ExprArena::open_fold`] and
//! [`ExprArena::close_fold`], whose binder is chosen and placeholder renamed
//! by [`ExprArena::close_over`], as `Kernel::over`'s are; a range is admitted by
//! [`Fold::admits`], in `sema`; a kernel is applied by [`ExprArena::apply`]. So each
//! construction written here and the same one built with the builder are
//! one program (`tests/the_library_is_the_builders.rs`,
//! `tests/fold_is_kernel_over.rs`, `tests/kernel_typed_parameters.rs`).
//!
//! A fold lowers to one `Reduce` node over its [`Fold`], built as
//! `Kernel::over` builds it: the body against a placeholder, then closed
//! over it. Nothing here unrolls: that is the e-graph's (`HalveFold`,
//! `PeelFold`), when the kernel is baked.
//!
//! An entry's parameters are its uniforms
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.4): each scalar — an
//! `f32` parameter, or one field of a record parameter — is declared as a
//! uniform, in [`AnalyzedKernel::parameters`]' order, and read through its
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
//!   one, and its `N as f32`s, are left open ([`Holes`]) and filled when its
//!   host function is instantiated.
//! - `emit::Staged`: Rust statements that build the arena
//!   when the entry's host function is called — an entry that takes a
//!   kernel-typed parameter, whose argument exists only then. Each step is
//!   the step [`Expansion`] takes, run later: the same IR calls, in the same
//!   order.
//!
//! A kernel-typed parameter is admitted when the host function is called,
//! after the entry's own uniforms ([`Callable::kernel_parameters`]),
//! and `k(x, y)` is [`Site::apply`]. [`Expansion`] has no argument to apply
//! — its [`Site::Argument`] is uninhabited — so no entry that takes one can
//! be lowered there.

use crate::ast::{
    BinaryOp, BlockExpr, Callable, CastExpr, Expr, FieldExpr, FnItem, FoldExpr, LetStmt,
    LibraryMethod, Method, MethodCallExpr, Name, PLAN, ProjectExpr, Projection, RecordId,
    Reduction, Role, Scopes, Stmt, UnaryOp, same_name,
};
use crate::sema::{
    AnalyzedKernel, Bounds, ConstValue, Parameter, RangeScope, Scalar, StructuralRange,
    folds_nested_too_deep, range_bounds,
};
use pixelflow_ir::arena::{Axis, ExprArena, ExprId, OpenFold, UniformDecl, UniformIdentity};
use pixelflow_ir::library;
use pixelflow_ir::{Binder, Fold, Monoid, OpKind};
use std::collections::HashMap;
use std::convert::Infallible;
use std::ops::Range;
use std::rc::Rc;
use syn::Ident;

/// Why a method's operands are all there: [`Method::resolve`] admits a name
/// only at the arity it takes.
const RESOLVED_AT_ITS_ARITY: &str = "`Method::resolve` admits a method only at its arity";

impl Projection {
    /// The axes the projection differentiates along, innermost first: `DXY`
    /// is `∂/∂Y` of `∂e/∂X`, as `Kernel`'s `e.dx().dy()` is.
    fn axes(self) -> &'static [Axis] {
        match self {
            Self::Value => &[],
            Self::Dx => &[Axis::X],
            Self::Dy => &[Axis::Y],
            Self::Dxx => &[Axis::X, Axis::X],
            Self::Dxy => &[Axis::X, Axis::Y],
            Self::Dyy => &[Axis::Y, Axis::Y],
        }
    }
}

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
    /// [`library::fract`].
    fn fract(&mut self, x: Self::Term) -> Self::Term;
    /// [`library::hypot`].
    fn hypot(&mut self, operands: [Self::Term; 2]) -> Self::Term;
    /// [`library::clamp`].
    fn clamp(&mut self, x: Self::Term, bounds: [Self::Term; 2]) -> Self::Term;
    /// [`library::derivative`].
    fn derivative(&mut self, e: Self::Term, axis: Axis) -> Self::Term;
    /// One of the entry's uniform scalars, declared next, and its leaf.
    fn uniform(&mut self, scalar: Scalar) -> Self::Term;
    /// The entry's kernel-typed parameter `name`, admitted next
    /// ([`ExprArena::admit`]).
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
    /// already: its index ([`ExprArena::open_fold`]).
    ///
    /// # Panics
    ///
    /// At `depth` [`Binder::COUNT`] or more: lowering refuses a fold nested
    /// that deep before it opens one.
    fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: Bounds) -> Self::Term;
    /// Close the innermost open fold over `body` ([`ExprArena::close_fold`]).
    fn close_fold(&mut self, body: Self::Term) -> Self::Term;
    /// `kernel` applied at `(x, y)` ([`ExprArena::apply`]).
    fn apply(&mut self, kernel: Self::Argument, at: [Self::Term; 2]) -> Self::Term;
}

/// A name in scope while a body is lowered.
#[derive(Debug, Clone)]
enum Binding<T, A> {
    /// A value: a `let`'s node, an entry's parameter's uniform, or a
    /// helper's parameter bound to its argument's node.
    Value(T),
    /// A fold's index, a `usize`: its placeholder `Var` while the fold's
    /// body is built. A body reads it only as `i as f32`.
    Index(T),
    /// A record: its fields' nodes, in field order — an entry's record
    /// parameter's uniforms, a helper's record argument, or a `let` alias of
    /// either. A record has no node of its own.
    Record(RecordId, Rc<[T]>),
    /// A kernel: an entry's kernel-typed parameter, admitted, or a helper's
    /// bound to the one passed to it. It has no node of its own either; each
    /// application builds one.
    Kernel(A),
}

/// A record as a binding holds it: which record, and its fields' terms in
/// field order.
type RecordFields<T> = (RecordId, Rc<[T]>);

/// The monoid a fold's spelling names.
fn monoid(reduction: Reduction) -> Monoid {
    match reduction {
        Reduction::Sum => Monoid::SUM,
        Reduction::Product => Monoid::PRODUCT,
        Reduction::Min => Monoid::MIN,
        Reduction::Max => Monoid::MAX,
        Reduction::Any => Monoid::ANY,
        Reduction::All => Monoid::ALL,
    }
}

/// The default a uniform is declared with here, where no call has supplied
/// one. Nothing at expansion reads it — a uniform is never folded, and
/// emission declares each with the call's value — so it is a NaN, which
/// would poison whatever read it by mistake rather than pass for a number.
const UNBOUND: f32 = f32::NAN;

/// The range every open fold of a template is built over ([`Holes`]): empty,
/// because the ends of a range [`Fold::admits`] have no room for a
/// placeholder and the stride ([`HOLE_STRIDE`]) carries it instead. Left
/// unfilled it would read as its monoid's identity, so [`Holes::range_of`]
/// names every open fold and emission fills each one.
const OPEN_RANGE: Range<u32> = 0..0;

/// The stride an open fold over a template's first distinct range text is
/// built with; the `k`th's is this plus `k`. No known fold has it: lowering
/// builds every known fold with `Fold::new`, whose stride is 1, so no known
/// fold interns with an open one, and none reads as one.
const HOLE_STRIDE: u32 = 2;

/// What a structural entry's template leaves open, for its host function to
/// fill per instantiation (plan §1.4): each fold whose range names a
/// structural parameter, and — as `Param(k)` leaves — each `N as f32`,
/// `k` the parameter's position.
///
/// An open fold is built as any fold is, over [`OPEN_RANGE`] strided by
/// [`HOLE_STRIDE`]` + k` for the `k`th distinct range text: the ends of a
/// range [`Fold::admits`] have no room for a placeholder, and its
/// stride does. Two folds over one range text, of one monoid and one body,
/// are one fold, and are interned as one; two over different ranges never
/// are. The range text lives in the fold's bits rather than beside a node
/// id because lowering splices a fold's arena into its enclosing one, which
/// renumbers every id.
#[derive(Debug, Default)]
pub(crate) struct Holes {
    ranges: Vec<StructuralRange>,
}

impl Holes {
    /// The stride an open fold over `range` is built with, over
    /// [`OPEN_RANGE`].
    fn stride(&mut self, range: StructuralRange) -> u32 {
        let text = range.text();
        let known = self.ranges.iter().position(|r| r.text() == text);
        let index = known.unwrap_or(self.ranges.len());
        if known.is_none() {
            self.ranges.push(range);
        }
        u32::try_from(index)
            .ok()
            .and_then(|index| HOLE_STRIDE.checked_add(index))
            .expect("an entry names fewer than 2^32 distinct structural ranges")
    }

    /// The range an open fold of this template is over, or `None` if `fold`
    /// is a known one.
    ///
    /// # Panics
    ///
    /// On a placeholder stride no hole names — one from another template.
    /// Emitted as it stands it would be an empty fold, its monoid's
    /// identity, with plausible pixels.
    pub(crate) fn range_of(&self, range: Fold) -> Option<&StructuralRange> {
        if range.range() != OPEN_RANGE {
            return None;
        }
        let index = range.stride().checked_sub(HOLE_STRIDE)?;
        let hole = usize::try_from(index)
            .ok()
            .and_then(|index| self.ranges.get(index));
        Some(hole.unwrap_or_else(|| {
            panic!(
                "kernel!: a fold over the placeholder stride {} that no structural range of \
                 this entry names",
                range.stride()
            )
        }))
    }
}

/// An entry lowered at expansion: its arena and root, and what the arena
/// leaves open if it is a template.
pub(crate) struct Lowered {
    pub arena: ExprArena,
    pub root: ExprId,
    pub holes: Holes,
    /// The entry's uniform scalars, in the order the arena's uniform table
    /// declares them: the host function's argument each defaults to.
    pub scalars: Vec<Scalar>,
}

/// The site an entry that takes no kernel is lowered at: an [`ExprArena`],
/// at macro expansion, with the folds open in it and the template's holes.
#[derive(Default)]
pub(crate) struct Expansion {
    arena: ExprArena,
    holes: Holes,
    /// The scalar behind each uniform declared, in declaration order.
    scalars: Vec<Scalar>,
    /// The folds whose bodies are being built, innermost last, each with
    /// the fold it closes into.
    open: Vec<(OpenFold, ExpansionFold)>,
}

/// The fold an open one closes into at expansion: a known range, or a
/// template's hole ([`Holes`]).
#[derive(Debug, Clone)]
enum ExpansionFold {
    Known(Monoid, Range<u32>),
    Hole(Monoid, u32),
}

impl Expansion {
    /// Lower `entry`, an entry that takes no kernel, at expansion: its
    /// parameters are declared first, as uniforms, in
    /// [`AnalyzedKernel::parameters`]' order, so the arena's uniform table is
    /// the entry's declaration order — every scalar, read or not, so that a
    /// positional binding cannot shift when a parameter goes unread.
    ///
    /// # Errors
    ///
    /// When the body has a construct lowering cannot express, or `entry`
    /// takes a kernel — an argument that exists only when its host function
    /// is called.
    pub(crate) fn lower(entry: Callable<'_>, analyzed: &AnalyzedKernel) -> syn::Result<Lowered> {
        let mut site = Expansion::default();
        let root = lower_entry(entry, analyzed, &mut site)?;
        Ok(Lowered {
            arena: site.arena,
            root,
            holes: site.holes,
            scalars: site.scalars,
        })
    }
}

impl Site for Expansion {
    type Term = ExprId;
    /// There is no argument at expansion.
    type Argument = Infallible;

    fn constant(&mut self, value: f32) -> ExprId {
        self.arena.push_const(value)
    }

    fn coordinate(&mut self, axis: Axis) -> ExprId {
        self.arena.push_var(axis.var())
    }

    fn unary(&mut self, op: OpKind, operand: ExprId) -> ExprId {
        self.arena.push_unary(op, operand)
    }

    fn binary(&mut self, op: OpKind, [a, b]: [ExprId; 2]) -> ExprId {
        self.arena.push_binary(op, a, b)
    }

    fn ternary(&mut self, op: OpKind, [a, b, c]: [ExprId; 3]) -> ExprId {
        self.arena.push_ternary(op, a, b, c)
    }

    fn fract(&mut self, x: ExprId) -> ExprId {
        library::fract(&mut self.arena, x)
    }

    fn hypot(&mut self, operands: [ExprId; 2]) -> ExprId {
        library::hypot(&mut self.arena, operands)
    }

    fn clamp(&mut self, x: ExprId, bounds: [ExprId; 2]) -> ExprId {
        library::clamp(&mut self.arena, x, bounds)
    }

    fn derivative(&mut self, e: ExprId, axis: Axis) -> ExprId {
        library::derivative(&mut self.arena, e, axis)
    }

    /// A declaration here holds a placeholder default; emission declares
    /// each with the call's value, which the scalar names.
    fn uniform(&mut self, scalar: Scalar) -> ExprId {
        self.scalars.push(scalar);
        let slot = self.arena.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: UNBOUND,
        });
        self.arena.push_uniform(slot)
    }

    fn kernel_parameter(&mut self, name: &Ident) -> syn::Result<Infallible> {
        Err(syn::Error::new(
            name.span(),
            format!(
                "`{name}` is a kernel, and an entry that takes one is lowered when its host \
                 function is called, not at expansion (Phase D-a of {PLAN})"
            ),
        ))
    }

    /// A template's `Param` hole, which its host function fills with the
    /// same `N as f32`.
    fn count(&mut self, position: usize, name: &Ident) -> syn::Result<ExprId> {
        // Borrowed: `Param(u8)` is a rewrite rule's metavariable too, and a
        // `u8` is narrower than the control plane allows. It is loud past
        // 256 rather than wrapping, and it goes with `Param` when D2 of the
        // plan deletes it — a structural hole wants a leaf of its own then.
        let hole = u8::try_from(position).map_err(|_| {
            syn::Error::new(
                name.span(),
                format!(
                    "`{name} as f32`: an entry reads at most {} structural parameters as values, \
                     the width of the IR's `Param` leaf",
                    usize::from(u8::MAX) + 1
                ),
            )
        })?;
        Ok(self.arena.push_param(hole))
    }

    fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: Bounds) -> ExprId {
        let fold = match bounds {
            Bounds::Known(range) => ExpansionFold::Known(monoid, range),
            Bounds::Structural(range) => ExpansionFold::Hole(monoid, self.holes.stride(range)),
        };
        let open = self
            .arena
            .open_fold(depth)
            .expect("lowering refuses a fold nested past the binders before it opens one");
        let index = open.index();
        self.open.push((open, fold));
        index
    }

    fn close_fold(&mut self, body: ExprId) -> ExprId {
        let (open, fold) = self
            .open
            .pop()
            .expect("lowering closes only the folds it opened");
        let fold_at = |binder: Binder| match fold {
            ExpansionFold::Known(monoid, range) => Fold::new(monoid, binder, range),
            ExpansionFold::Hole(monoid, stride) => {
                Fold::strided(monoid, binder, OPEN_RANGE, stride)
            }
        };
        self.arena
            .close_fold(open, body, fold_at)
            .expect("lowering refuses a fold nested past the binders, so one is always free")
    }

    fn apply(&mut self, kernel: Infallible, _at: [ExprId; 2]) -> ExprId {
        match kernel {}
    }
}

/// Lower an entry's body at `site`, inlining the block's helpers and folding
/// its `const`s. Its uniform parameters are declared first, in
/// [`AnalyzedKernel::parameters`]' order, then its kernel-typed parameters
/// are admitted, in [`Callable::kernel_parameters`]' — each
/// argument's uniforms after the entry's own, in the argument's order.
/// Children are lowered first so that a parent's operands always exist.
///
/// # Errors
///
/// When the body has a construct lowering cannot express there.
pub(crate) fn lower_entry<S: Site>(
    entry: Callable<'_>,
    analyzed: &AnalyzedKernel,
    site: &mut S,
) -> syn::Result<S::Term> {
    let helpers = analyzed
        .def
        .fns()
        .iter()
        .filter(|f| f.role() == Role::Helper)
        .map(|f| (Name::of(&f.name), f))
        .collect();
    let mut locals = Scopes::default();
    for parameter in analyzed.parameters(entry.params) {
        let (name, binding) = match parameter {
            Parameter::Scalar(name) => (
                name,
                Binding::Value(site.uniform(Scalar::Param(name.clone()))),
            ),
            Parameter::Record { name, id, fields } => {
                let uniforms = fields
                    .iter()
                    .map(|field| site.uniform(Scalar::Field(name.clone(), field.name.clone())))
                    .collect();
                (name, Binding::Record(id, uniforms))
            }
        };
        locals.bind(Name::of(name), binding);
    }
    for name in entry.kernel_parameters() {
        let argument = site.kernel_parameter(name)?;
        locals.bind(Name::of(name), Binding::Kernel(argument));
    }
    let mut lowering = Lowering {
        program: Program { analyzed, helpers },
        frame: Frame {
            role: entry.role,
            structural: entry.structural,
            locals,
        },
        site,
        open_folds: 0,
    };
    lowering.lower(entry.body)
}

/// The block's items: what every body can name besides its own scope.
struct Program<'a> {
    /// The block, analyzed: its records and its `const`s' values.
    analyzed: &'a AnalyzedKernel,
    helpers: HashMap<Name, &'a FnItem>,
}

/// The function being lowered: its structural parameters, and the bindings
/// in scope — its parameters in the function's own scope, then the
/// `let`-bound locals and fold indices of the blocks being walked (one scope
/// per block, and one per fold body, with Rust's lexical scoping). An
/// entry's parameters are bound to their uniforms and admitted arguments; an
/// inlined helper's, to its argument nodes and the kernels passed to it.
struct Frame<'a, S: Site> {
    role: Role,
    /// An entry's structural parameters; a helper has none.
    structural: &'a [Ident],
    locals: Scopes<Binding<S::Term, S::Argument>>,
}

/// State threaded through the AST → term walk.
struct Lowering<'a, S: Site> {
    program: Program<'a>,
    frame: Frame<'a, S>,
    site: &'a mut S,
    /// How many folds' bodies are being built, across inlined helpers too:
    /// the depth that picks the next one's placeholder.
    open_folds: usize,
}

impl<S: Site> Lowering<'_, S> {
    /// Translate an AST node into a term at the site, resolving `let`-bound
    /// locals via the frame's scopes. Each binding maps to a single term, so
    /// a local used twice is one node and the arena is a DAG rather than
    /// duplicated subtrees.
    fn lower(&mut self, expr: &Expr) -> syn::Result<S::Term> {
        match expr {
            Expr::Ident(name) => self.resolve(name),

            Expr::Literal(lit) => {
                let value = lit
                    .f32_value()
                    .expect("sema refuses a literal that is no `f32`");
                Ok(self.site.constant(value))
            }

            Expr::Binary(binary) => {
                let lhs = self.lower(&binary.lhs)?;
                let rhs = self.lower(&binary.rhs)?;

                let op = match binary.op {
                    BinaryOp::Add => OpKind::Add,
                    BinaryOp::Sub => OpKind::Sub,
                    BinaryOp::Mul => OpKind::Mul,
                    BinaryOp::Div => OpKind::Div,
                    BinaryOp::Lt => OpKind::Lt,
                    BinaryOp::Le => OpKind::Le,
                    BinaryOp::Gt => OpKind::Gt,
                    BinaryOp::Ge => OpKind::Ge,
                    BinaryOp::Eq => OpKind::Eq,
                    BinaryOp::Ne => OpKind::Ne,
                    // A `bool` is a canonical mask lane — all-ones or all-zero
                    // — so bitwise AND/OR is logical AND/OR exactly. `sema`
                    // types both operands as `bool`.
                    BinaryOp::BitAnd => OpKind::BitAnd,
                    BinaryOp::BitOr => OpKind::BitOr,
                };

                Ok(self.site.binary(op, [lhs, rhs]))
            }

            Expr::Unary(unary) => {
                let operand = self.lower(&unary.operand)?;
                let op = match unary.op {
                    UnaryOp::Neg => OpKind::Neg,
                };
                Ok(self.site.unary(op, operand))
            }

            Expr::MethodCall(call) => self.lower_method_call(call),

            Expr::Project(project) => self.lower_projection(project),

            // A name in scope is resolved first, as `sema` resolves it: a
            // kernel is applied, and any other binding is not a function.
            // Then a helper is inlined.
            Expr::Call(call) => {
                let func = Name::of(&call.func);
                if let Some(binding) = self.frame.locals.lookup(&func) {
                    let Binding::Kernel(kernel) = binding else {
                        unreachable!("sema refuses applying a name that is not a kernel");
                    };
                    let kernel = kernel.clone();
                    return self.apply(kernel, &call.args);
                }
                let helper = self.program.helpers.get(&func).copied().expect(
                    "`sema` refuses a call to anything but a kernel, a helper or a projection",
                );
                self.inline(helper, &call.args)
            }

            // The choice, the same node `.select` lowers to: `If(m, a, b)`
            // is `if m then a else b`.
            Expr::If(choice) => {
                let cond = self.lower(&choice.cond)?;
                let then = self.lower_block(&choice.then_branch)?;
                let otherwise = self.lower(&choice.else_branch)?;
                Ok(self.site.ternary(OpKind::If, [cond, then, otherwise]))
            }

            Expr::Fold(fold) => self.lower_fold(fold),

            Expr::Cast(cast) => self.lower_cast(cast),

            Expr::Field(field) => Ok(self.lower_field(field)),

            Expr::Paren(inner) => self.lower(inner),

            Expr::Block(block) => self.lower_block(block),
        }
    }

    /// `k(x, y)`: the kernel `k` applied at the coordinates, which are
    /// lowered first, in order — contramap (§1.2).
    fn apply(&mut self, kernel: S::Argument, args: &[Expr]) -> syn::Result<S::Term> {
        let [x, y] = args else {
            unreachable!("sema refuses a kernel applied at other than two coordinates");
        };
        let x = self.lower(x)?;
        let y = self.lower(y)?;
        Ok(self.site.apply(kernel, [x, y]))
    }

    /// `⊕_{i ∈ [lo, hi)} body` as one `Reduce` over its [`Fold`]: a
    /// known range, or one naming the entry's structural parameters.
    ///
    /// The body is built against the fold's index, a placeholder, and the
    /// fold is closed after it, by the IR's one definition of how a binder
    /// is chosen and a placeholder renamed ([`ExprArena::close_over`]),
    /// which `Kernel`'s folds are built through too.
    fn lower_fold(&mut self, fold: &FoldExpr) -> syn::Result<S::Term> {
        let scope = RangeScope {
            consts: &self.program.analyzed.consts,
            structural: self.frame.structural,
        };
        let bounds = range_bounds(&fold.range, scope)
            .expect("sema refuses a range it cannot evaluate, admit or order");
        // One placeholder per fold open at once, the `n`th for `n` open, so a
        // nested fold's rename never reaches its enclosing fold's index. No
        // more open than the IR has binders: the refusal names the depth,
        // where running out of placeholders would name nothing the author
        // wrote.
        if self.open_folds >= Binder::COUNT {
            return Err(folds_nested_too_deep(fold.span));
        }
        let index = self
            .site
            .open_fold(self.open_folds, monoid(fold.reduction), bounds);
        self.frame.locals.push_scope();
        self.frame
            .locals
            .bind(Name::of(&fold.binder), Binding::Index(index));
        self.open_folds += 1;
        let body = self.lower(&fold.body);
        self.open_folds -= 1;
        self.frame.locals.pop_scope();
        Ok(self.site.close_fold(body?))
    }

    /// `i as f32`. A fold's index is an `f32` lane already, so its
    /// conversion is its binder's `Var`; a `usize` const is its value's
    /// `f32`, rounded as Rust's `as` rounds it; a structural parameter is the
    /// site's [`Site::count`], the same `N as f32` its host function has.
    fn lower_cast(&mut self, cast: &CastExpr) -> syn::Result<S::Term> {
        let name = cast
            .named()
            .expect("sema refuses an `as f32` of anything but a name");
        if let Some(binding) = self.frame.locals.lookup(&Name::of(name)) {
            let Binding::Index(index) = binding else {
                unreachable!("sema refuses `{name} as f32` where `{name}` is no `usize`");
            };
            return Ok(index.clone());
        }
        if let Some(position) = self
            .frame
            .structural
            .iter()
            .position(|n| same_name(n, name))
        {
            return self.site.count(position, name);
        }
        let Some(&ConstValue::Usize(count)) = self.program.analyzed.consts.get(&Name::of(name))
        else {
            unreachable!("sema refuses `{name} as f32` where `{name}` is no `usize`");
        };
        Ok(self.site.constant(count as f32))
    }

    /// `p.x0`: the node of one field of a record binding.
    ///
    /// A record is only ever written by name, and read at a field it has
    /// (`sema` refuses anything else).
    fn lower_field(&self, field: &FieldExpr) -> S::Term {
        let (record, fields) = self
            .record_named(&field.base)
            .expect("sema refuses a field read of anything but a record by name");
        let index = self
            .program
            .analyzed
            .def
            .record(record)
            .fields
            .iter()
            .position(|f| same_name(&f.name, &field.member))
            .expect("sema refuses a field the record does not have");
        fields[index].clone()
    }

    /// The record binding `expr` names, if it names one.
    fn record_named(&self, expr: &Expr) -> Option<RecordFields<S::Term>> {
        match self.frame.locals.lookup(&Name::of(expr.named()?))? {
            Binding::Record(record, fields) => Some((*record, Rc::clone(fields))),
            Binding::Value(_) | Binding::Index(_) | Binding::Kernel(_) => None,
        }
    }

    /// The kernel binding `expr` names, if it names one: a kernel is only
    /// ever passed by name (`sema` refuses anything else).
    fn kernel_named(&self, expr: &Expr) -> Option<S::Argument> {
        match self.frame.locals.lookup(&Name::of(expr.named()?))? {
            Binding::Kernel(kernel) => Some(kernel.clone()),
            Binding::Value(_) | Binding::Index(_) | Binding::Record(..) => None,
        }
    }

    /// `receiver.method(args)`: the operands are lowered in order, receiver
    /// first, then the method builds its one node. The parser resolved the
    /// method at this arity, so a method always finds the operands it takes.
    fn lower_method_call(&mut self, call: &MethodCallExpr) -> syn::Result<S::Term> {
        let receiver = self.lower(&call.receiver)?;
        let mut args = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            args.push(self.lower(arg)?);
        }
        let mut args = args.into_iter();
        let mut operand = || args.next().expect(RESOLVED_AT_ITS_ARITY);
        Ok(match call.method {
            // Arena expressions are values, so `.clone()` is the identity.
            Method::Clone => receiver,
            // Primitive ops: one `OpKind` per (name, arity), read from the
            // single table `OpKind::from_method_call` resolves against.
            Method::Op(op) => match call.args.len() {
                0 => self.site.unary(op, receiver),
                1 => self.site.binary(op, [receiver, operand()]),
                2 => self.site.ternary(op, [receiver, operand(), operand()]),
                _ => unreachable!("OpKind::from_method_call only resolves ops of arity 1..=3"),
            },
            // Library, not primitives: the IR's one definition of each,
            // which `Kernel`'s method of the same name builds too.
            Method::Library(LibraryMethod::Fract) => self.site.fract(receiver),
            Method::Library(LibraryMethod::Hypot) => self.site.hypot([receiver, operand()]),
            Method::Library(LibraryMethod::Clamp) => {
                self.site.clamp(receiver, [operand(), operand()])
            }
        })
    }

    /// Derivative projections (V/DX/DY and the Hessian family) map to
    /// [`library::derivative`]s, one per axis, as `Kernel::dx`/`dy` build
    /// them: the runtime `lower_dwrt` pass (pixelflow-ir) rewrites them into
    /// chain-rule arithmetic before codegen.
    fn lower_projection(&mut self, project: &ProjectExpr) -> syn::Result<S::Term> {
        let mut e = self.lower(&project.arg)?;
        for &axis in project.projection.axes() {
            e = self.site.derivative(e, axis);
        }
        Ok(e)
    }

    /// β-reduction: `helper(args)` is the helper's body with each parameter
    /// bound to its argument's node, to a record argument's fields, or to
    /// the kernel passed to it.
    ///
    /// The arguments are lowered in the caller's frame, once each, so an
    /// argument used twice in the body is one node. The body is lowered in a
    /// frame of its own: the helper's parameters are its base scope, and
    /// nothing of the caller's — no local, no entry parameter, no structural
    /// parameter — is visible.
    fn inline(&mut self, helper: &FnItem, args: &[Expr]) -> syn::Result<S::Term> {
        assert_eq!(
            args.len(),
            helper.params.len(),
            "sema refuses a call to `{}` at another arity",
            helper.name
        );
        let mut locals = Scopes::default();
        for (param, arg) in helper.params.iter().zip(args) {
            let binding = match (self.record_named(arg), self.kernel_named(arg)) {
                (Some((record, fields)), _) => Binding::Record(record, fields),
                (None, Some(kernel)) => Binding::Kernel(kernel),
                (None, None) => Binding::Value(self.lower(arg)?),
            };
            locals.bind(Name::of(&param.name), binding);
        }
        let callee = Frame {
            role: Role::Helper,
            structural: &[],
            locals,
        };
        // The frame is swapped and the fold depth is not: a helper's fold
        // inlined inside a fold is nested in it, and its placeholder must
        // not be the caller's, or its rename would reach the caller's index
        // through the argument (pinned against rustc and the builder).
        let caller = std::mem::replace(&mut self.frame, callee);
        let body = self.lower(&helper.body);
        self.frame = caller;
        body
    }

    /// The node a name refers to: the innermost binding of it in scope — a
    /// `let`, a fold's index, a parameter — else a `const`, else a
    /// coordinate.
    ///
    /// Bindings come first because that is what lexical scoping means. `sema`
    /// refuses a `let` named X or Y, so today the order only decides a
    /// local against a parameter it shadows; but matching `"X"` before the
    /// locals would let `{ let X = Y; X }` read the coordinate, and a
    /// lowering that is right only because an earlier stage refused its
    /// input is one refactor away from that bug. For the same reason a
    /// coordinate in a helper is refused here too, not only in `sema`.
    fn resolve(&mut self, ident: &Ident) -> syn::Result<S::Term> {
        let name = &Name::of(ident);
        if let Some(binding) = self.frame.locals.lookup(name) {
            let Binding::Value(id) = binding else {
                unreachable!("sema refuses `{name}` read as a value: it is no `f32` or `bool`");
            };
            return Ok(id.clone());
        }
        // The same order sema documents: a binding, then a const, then a
        // structural parameter, then the coordinates. Sema refuses a
        // parameter named X or Y, so the two stages agree without one
        // relying on the other's refusal.
        match self.program.analyzed.consts.get(name) {
            Some(&ConstValue::F32(value)) => return Ok(self.site.constant(value)),
            Some(ConstValue::Usize(_)) => {
                unreachable!("sema refuses the `usize` const `{name}` read as a value")
            }
            None => {}
        }
        assert!(
            !self.frame.structural.iter().any(|n| Name::of(n) == *name),
            "sema refuses the structural parameter `{name}` read as a value"
        );
        let axis = name
            .coordinate()
            .unwrap_or_else(|| unreachable!("sema refuses the unknown name `{name}`"));
        match self.frame.role {
            Role::Entry => Ok(self.site.coordinate(axis)),
            Role::Helper => Err(syn::Error::new(
                ident.span(),
                format!("`{name}` in a helper: a helper takes its coordinates as arguments"),
            )),
        }
    }

    /// A block's `let`s live in a scope of their own, which ends with it.
    fn lower_block(&mut self, block: &BlockExpr) -> syn::Result<S::Term> {
        self.frame.locals.push_scope();
        let value = self.lower_block_contents(block);
        self.frame.locals.pop_scope();
        value
    }

    /// What a `let` binds its name to: its initializer's node, or — a record
    /// being aliased — the same fields.
    fn let_binding(&mut self, let_stmt: &LetStmt) -> syn::Result<Binding<S::Term, S::Argument>> {
        match self.record_named(&let_stmt.init) {
            Some((record, fields)) => Ok(Binding::Record(record, fields)),
            None => Ok(Binding::Value(self.lower(&let_stmt.init)?)),
        }
    }

    /// A block's statements in order, then its value, in the scope the
    /// caller opened for it.
    fn lower_block_contents(&mut self, block: &BlockExpr) -> syn::Result<S::Term> {
        for stmt in &block.stmts {
            match stmt {
                // Every initializer before any name: `let (a, b) = (b, a);`
                // swaps, as Rust's does, and `let a = a + 1.0;` sees whatever
                // the name meant before.
                Stmt::Let(lets) => {
                    let mut bindings = Vec::with_capacity(lets.len());
                    for let_stmt in lets {
                        bindings.push(self.let_binding(let_stmt)?);
                    }
                    for (let_stmt, binding) in lets.iter().zip(bindings) {
                        self.frame.locals.bind(Name::of(&let_stmt.name), binding);
                    }
                }
                // A non-binding statement has no value to thread; lower it so
                // any nested error surfaces, then discard the id.
                Stmt::Expr(e) => {
                    self.lower(e)?;
                }
            }
        }
        let value = block
            .expr
            .as_ref()
            .expect("sema refuses a block with no final expression");
        self.lower(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::KernelDef;
    use crate::parser::parse;
    use pixelflow_ir::arena::ExprNode;
    use quote::quote;

    /// Lower a block straight from the parser, with no `sema` in front: what
    /// lowering itself makes of a name, whatever an earlier stage would
    /// refuse. The `const`s are evaluated, since lowering reads their values.
    /// A refusal comes back as its text: the arena has no `Debug` to unwrap
    /// through.
    fn lower_unanalyzed(input: proc_macro2::TokenStream) -> Result<(ExprArena, ExprId), String> {
        let def = parse(input).expect("the body parses");
        let consts = def
            .consts()
            .iter()
            .map(|c| {
                let Expr::Literal(lit) = &c.init else {
                    panic!("this harness holds literal consts only");
                };
                let value = match matches!(&c.ty, syn::Type::Path(p) if p.path.is_ident("usize")) {
                    true => ConstValue::Usize(lit.usize_value().expect("a count")),
                    false => ConstValue::F32(lit.f32_value().expect("a value")),
                };
                (Name::of(&c.name), value)
            })
            .collect();
        let unanalyzed = AnalyzedKernel { def, consts };
        let entry = match &unanalyzed.def {
            KernelDef::Closure(entry) => entry.callable(),
            KernelDef::Items { fns, .. } => fns
                .iter()
                .find(|f| f.role() == Role::Entry)
                .expect("one entry")
                .callable(),
        };
        let Lowered { arena, root, .. } =
            Expansion::lower(entry, &unanalyzed).map_err(|refusal| refusal.to_string())?;
        Ok((arena, root))
    }

    fn lowered(input: proc_macro2::TokenStream) -> (ExprArena, ExprId) {
        lower_unanalyzed(input).expect("the body lowers")
    }

    /// `sema` refuses `let X`, but lowering resolves a name through its
    /// scopes before the coordinates regardless: `{ let X = Y; X }` is the
    /// local, which is `Y`, not `Var(0)`.
    #[test]
    fn a_binding_is_resolved_before_a_coordinate_of_the_same_name() {
        let (arena, root) = lowered(quote! { || { let X = Y; X } });
        assert!(
            matches!(arena.node(root), ExprNode::Var(1)),
            "`X` is the local bound to Y, got {:?}",
            arena.node(root)
        );
    }

    /// A coordinate in a helper is refused by lowering itself, not only by
    /// `sema`.
    #[test]
    fn a_coordinate_in_a_helper_is_refused_here_too() {
        let Err(err) = lower_unanalyzed(quote! {
            fn shifted() -> f32 { X }
            pub fn f() -> f32 { shifted() }
        }) else {
            panic!("a helper reads no coordinate");
        };
        assert!(err.contains("`X` in a helper"), "got: {err}");
    }

    // ───────────────────────────── folds ─────────────────────────────

    /// `sema` counts the folds one body nests; a helper's fold inlined into
    /// them nests deeper than any body shows, and lowering refuses that one,
    /// where the IR would run out of binders.
    #[test]
    fn a_helper_fold_nests_inside_the_callers_and_is_refused_past_the_binders() {
        use proc_macro2::TokenStream;
        let fold = |body: TokenStream| quote! { (0..1).map(|i| #body).sum::<f32>() };
        let nest =
            |depth: usize, innermost: TokenStream| (0..depth).fold(innermost, |body, _| fold(body));
        let program = |depth: usize| {
            let nested = nest(depth, quote! { deepest() });
            quote! {
                fn deepest() -> f32 { (0..1).map(|i| 1.0).sum::<f32>() }
                pub fn f() -> f32 { #nested }
            }
        };
        let through_sema = |input| {
            let analyzed = crate::sema::analyze(parse(input).expect("parses")).expect("analyzes");
            let entry = analyzed.def.fns().iter().find(|f| f.role() == Role::Entry);
            Expansion::lower(entry.expect("one entry").callable(), &analyzed)
        };
        let Ok(Lowered { arena, root, .. }) = through_sema(program(Binder::COUNT - 1)) else {
            panic!("as deep as the IR has binders lowers");
        };
        let ExprNode::Reduce { fold: outer, .. } = arena.node(root) else {
            panic!("expected a fold");
        };
        assert_eq!(usize::from(outer.binder().slot()), Binder::COUNT - 1);
        let Err(err) = through_sema(program(Binder::COUNT)) else {
            panic!("one binder deeper than the IR has");
        };
        let err = err.to_string();
        assert!(err.contains("folds nested more than"), "got: {err}");
    }
}
