//! The site an entry that takes no kernel is lowered at: an [`ExprArena`] at
//! macro expansion, and what a template's arena leaves open.

use super::Site;
use super::walk::lower_entry;
use crate::ast::{Callable, PLAN};
use crate::sema::{AnalyzedKernel, Bounds, Scalar, StructuralRange};
use pixelflow_ir::arena::{Axis, ExprArena, ExprId, OpenFold, UniformDecl, UniformIdentity};
use pixelflow_ir::library;
use pixelflow_ir::{Binder, Fold, Monoid, OpKind};
use std::convert::Infallible;
use std::ops::Range;
use syn::Ident;

/// The default a uniform is declared with here, where no call has supplied
/// one. Nothing at expansion reads it — a uniform is never folded, and
/// emission declares each with the call's value — so it is a NaN, which
/// would poison whatever read it by mistake rather than pass for a number.
const UNBOUND: f32 = f32::NAN;

/// The range every open fold of a template is built over ([`Holes`]): empty,
/// because the ends of a range [`Fold::admits`] have no room for a
/// placeholder and the stride ([`HOLE_STRIDE`]) carries it instead. Left
/// unfilled it would read as its monoid's identity, so [`Lowered::range_of`]
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
struct Holes {
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
}

/// An entry lowered at expansion: its arena and root, and what the arena
/// leaves open if it is a template.
pub(crate) struct Lowered {
    pub(crate) arena: ExprArena,
    pub(crate) root: ExprId,
    holes: Holes,
    /// The entry's uniform scalars, in the order the arena's uniform table
    /// declares them: the host function's argument each defaults to.
    pub(crate) scalars: Vec<Scalar>,
}

impl Lowered {
    /// The range an open fold of this template is over, or `None` if `fold`
    /// is a known one.
    ///
    /// # Panics
    ///
    /// On a placeholder stride no hole names — one from another template.
    /// Emitted as it stands it would be an empty fold, its monoid's
    /// identity, with plausible pixels.
    pub(crate) fn range_of(&self, fold: Fold) -> Option<&StructuralRange> {
        if fold.range() != OPEN_RANGE {
            return None;
        }
        let index = fold.stride().checked_sub(HOLE_STRIDE)?;
        let hole = usize::try_from(index)
            .ok()
            .and_then(|index| self.holes.ranges.get(index));
        Some(hole.unwrap_or_else(|| {
            panic!(
                "kernel!: a fold over the placeholder stride {} that no structural range of \
                 this entry names",
                fold.stride()
            )
        }))
    }
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
