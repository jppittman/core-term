//! Core e-graph data structures: EClassId and ENode.

use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use pixelflow_ir::arena::{BufferDecl, UniformDecl};
use pixelflow_ir::fold::Fold;
use pixelflow_ir::{KernelKey, OpKind, Variance};

/// Identifier for an equivalence class in the e-graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EClassId(pub(crate) u32);

impl EClassId {
    /// Get the index of this e-class ID.
    ///
    /// This is useful for using EClassId as a key in external data structures.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The most operands any operation takes: the widest [`OpKind::arity`]
/// (`MulAdd`, `If`, `Gather`).
pub(crate) const MAX_OPERANDS: usize = 3;

/// The e-classes an operation is applied to: exactly [`OpKind::arity`] of
/// them, held inline.
///
/// The live operands are a prefix of a fixed array whose tail is always the
/// zero class, so two operand lists are equal exactly when their live
/// prefixes are, and the derived `Eq` and `Hash` need no hand-written case
/// for the tail. The fields are private and the one constructor takes the
/// operation it is for, so the tail cannot be written and the length is
/// checked against that operation's arity. The slice view, and `Debug`, are
/// the live prefix only.
///
/// The operation is not stored here, so the pairing is checked when the
/// operands are built and not afterwards: a variant's fields are public, and
/// `ENode::Op` can be written with another node's operands under a different
/// `op`. [`ENode::apply`] is the constructor that keeps them together.
///
/// `len` counts slots of an inline array, not anything a program can say: its
/// range is `MAX_OPERANDS` whatever its width, so a wider count would let no
/// program express more.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Operands {
    len: u8,
    ids: [EClassId; MAX_OPERANDS],
}

impl core::fmt::Debug for Operands {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Operands {
    /// The operands of `op`, one class per operand.
    ///
    /// # Panics
    ///
    /// If `classes` is not exactly `op.arity()` long, or `op` is not an
    /// operation over one to `MAX_OPERANDS` operands (a leaf kind, or a
    /// `Tuple`, has none to hold). An operation applied to the wrong number
    /// of operands is not a node, and a rule that builds one has a bug to
    /// find, not a graph to carry it.
    #[must_use]
    pub(crate) fn new(op: OpKind, classes: &[EClassId]) -> Self {
        assert!(
            (1..=MAX_OPERANDS).contains(&op.arity()) && classes.len() == op.arity(),
            "{op:?} takes {} operands (an operation takes 1 to {MAX_OPERANDS}), got {}",
            op.arity(),
            classes.len(),
        );
        let mut ids = [EClassId(0); MAX_OPERANDS];
        ids[..classes.len()].copy_from_slice(classes);
        Self {
            len: classes.len() as u8,
            ids,
        }
    }
}

impl Deref for Operands {
    type Target = [EClassId];

    fn deref(&self) -> &[EClassId] {
        &self.ids[..usize::from(self.len)]
    }
}

impl DerefMut for Operands {
    fn deref_mut(&mut self) -> &mut [EClassId] {
        &mut self.ids[..usize::from(self.len)]
    }
}

/// An expression node in the e-graph: an operator with its static
/// parameters, plus the e-classes of its dynamic operands.
///
/// A plain value. Operands point to e-classes (not other nodes), enabling
/// compact representation of equivalent expressions, and equality and
/// hashing are structural, which is what hash-consing the graph's memo needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ENode {
    /// Variable with index (0=X, 1=Y, 2=Z, 3=W, etc.)
    Var(u8),
    /// Constant value (stored as f32 bits)
    Const(u32),
    /// Bound-memory leaf: a buffer declaration, read through a `Gather` node.
    ///
    /// Carries the full [`BufferDecl`] rather than an arena-local `BufferId`
    /// because e-classes outlive any one arena: `BufferIdentity` is
    /// process-unique, so two `Buffer` leaves are the same e-class iff they
    /// name the same memory with the same extents — which is exactly the
    /// hash-consing CSE this leaf exists for. Equality is on the whole decl,
    /// not the identity alone: two decls that disagree on extents must not
    /// hash-cons, or extraction would redeclare only one of them and corrupt
    /// the other's extent. Extraction redeclares the decl into the output
    /// arena.
    Buffer(BufferDecl),
    /// Per-call scalar leaf: a uniform declaration, hash-consed by identity
    /// for the same reason as [`ENode::Buffer`]. No rule matches it as a
    /// `Const`, so it is never folded; its gain in the e-graph is CSE of the
    /// arithmetic that depends on it alone.
    Uniform(UniformDecl),
    /// Unbound scalar slot of a `kernel!` builder, hash-consed by index.
    ///
    /// The same opaque leaf as [`ENode::Uniform`], one tier earlier: no rule
    /// matches it, nothing folds it, its derivative is zero, and its gain in
    /// the e-graph is CSE of the arithmetic depending on it alone. A builder
    /// substitutes it away before anything is compiled, so only the macro
    /// tier's [`Vocabulary::Templates`](super::ops::Vocabulary) may hold one.
    ///
    /// Before this leaf existed the e-graph refused `Param` outright, and
    /// each macro-side caller smuggled one past as something else — as
    /// `Var(16 + i)` in the compiler's arena bridge, as an opaque synthetic
    /// identifier in its AST optimizer. Two encodings of one idea, next to
    /// two worked examples of it.
    Param(u8),
    /// An operation applied to the classes of its operands.
    Op { op: OpKind, children: Operands },
    /// A bounded fold: `⊕_{k} body[fold.binder() := k]`, `k` ranging over
    /// `fold`'s own visited indices (see `Fold`'s doc, in `pixelflow-ir`).
    ///
    /// The one node in the graph that *binds*, and the reason it can be in
    /// the graph at all: its algebra, binder and range live in the node's
    /// identity rather than in three `Const` children. As children they were
    /// e-classes like any other — the `Const(4.0)` naming the `Add` combiner
    /// was the same class as any literal `4.0` in the kernel, and every
    /// arithmetic rule in the set could reach a trip count — so `Reduce` was
    /// simply lowered before insertion and the e-graph never saw a fold.
    ///
    /// Hash-consing therefore does what it should: two folds are one node iff
    /// they fold the same body, under the same algebra, over the same range.
    Reduce { fold: Fold, body: EClassId },
    /// A **unit**: a kernel named by content, optimized by itself and linked
    /// in after extraction (`crate::runtime`;
    /// docs/plans/2026-09-25-the-language-is-kernel.md §4, O1). A leaf here,
    /// opaque to every rule as a [`ENode::Uniform`] is: its body is not in
    /// this graph, so no rewrite can reach inside it, and extraction hands
    /// back the name.
    ///
    /// It carries `variance`, its referent's, because that is the one fact a
    /// rule reads off a leaf — `FactorFold`'s side condition, and the binder
    /// substitution `PeelFold`/`HalveFold` skip a binder-free class by — and
    /// the one fact extraction prices a leaf's evaluations by. A unit is
    /// closed (its variance names no binder), so every rewrite conditioned
    /// on binder-invariance is the one it would be with the body inlined.
    /// The key is the referent and the variance a function of it, so
    /// comparing both is comparing the key, checked.
    /// Only the runtime tier's unit walk admits one (`EGraph::admit_unit`);
    /// [`insert`](fn@super::insert) declines a reference the graph was not told
    /// is a unit.
    Ref { key: KernelKey, variance: Variance },
}

impl ENode {
    /// Create a constant node.
    pub fn constant(val: f32) -> Self {
        ENode::Const(val.to_bits())
    }

    /// Get the constant value if this is a Const node.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ENode::Const(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    /// Check if this node is a specific constant value.
    pub fn is_const(&self, val: f32) -> bool {
        self.as_f32() == Some(val)
    }

    /// `op` applied to `operands`, one class per operand.
    ///
    /// # Panics
    ///
    /// If `operands.len()` is not `op.arity()`, or `op` is not an operation
    /// over one to `MAX_OPERANDS` operands (a leaf kind, or a `Tuple`, has
    /// none to hold).
    #[must_use]
    pub fn apply(op: OpKind, operands: &[EClassId]) -> Self {
        ENode::Op {
            op,
            children: Operands::new(op, operands),
        }
    }

    /// Get the operation if this is an Op node.
    pub fn op(&self) -> Option<OpKind> {
        match self {
            ENode::Op { op, .. } => Some(*op),
            _ => None,
        }
    }

    /// The fold this node performs, if it is one.
    pub fn fold(&self) -> Option<Fold> {
        match self {
            ENode::Reduce { fold, .. } => Some(*fold),
            _ => None,
        }
    }

    /// Get children of this node.
    ///
    /// Allocates and clones — see [`Self::children_slice`] for the
    /// zero-allocation borrow, which every call site on a saturation/
    /// extraction hot path should prefer. This owned form stays for callers
    /// (training/eval tooling outside this crate) that want a `Vec` to hold
    /// past the node's borrow.
    pub fn children(&self) -> Vec<EClassId> {
        self.children_slice().to_vec()
    }

    /// Borrow this node's children with no allocation.
    pub fn children_slice(&self) -> &[EClassId] {
        match self {
            ENode::Var(_)
            | ENode::Const(_)
            | ENode::Buffer(_)
            | ENode::Uniform(_)
            | ENode::Param(_)
            | ENode::Ref { .. } => &[],
            ENode::Op { children, .. } => children,
            ENode::Reduce { body, .. } => core::slice::from_ref(body),
        }
    }

    /// Borrow this node's children mutably — what canonicalization needs, and
    /// the only reason a caller should want it.
    pub fn children_slice_mut(&mut self) -> &mut [EClassId] {
        match self {
            ENode::Var(_)
            | ENode::Const(_)
            | ENode::Buffer(_)
            | ENode::Uniform(_)
            | ENode::Param(_)
            | ENode::Ref { .. } => &mut [],
            ENode::Op { children, .. } => children,
            ENode::Reduce { body, .. } => core::slice::from_mut(body),
        }
    }

    /// Get binary operands if this is a binary operation.
    pub fn binary_operands(&self) -> Option<(EClassId, EClassId)> {
        match self {
            ENode::Op { children, .. } if children.len() == 2 => Some((children[0], children[1])),
            _ => None,
        }
    }

    /// Which variables this node depends on, given what each child class
    /// depends on — the forward transfer function of variance.
    ///
    /// A leaf names its own variable (or none); an operation unions its
    /// children; a fold removes the index it binds. The one definition both
    /// consumers read: the e-graph's per-class fact (`EGraph::variance`),
    /// which feeds it the children's facts, and the extractor's
    /// `node_variance`, which feeds it the variance of the forms it chose.
    ///
    /// Monotone in `child`: a larger answer for a child can only give a
    /// larger answer here, so an over-approximation in gives an
    /// over-approximation out.
    pub(crate) fn variance(&self, mut child: impl FnMut(EClassId) -> Variance) -> Variance {
        match self {
            ENode::Var(v) => var_variance(*v),
            // A buffer's contents are fixed for the kernel's lifetime — a
            // read of one varies with its index, which is the `Gather`'s
            // other children, unioned by the `Op` arm. A uniform or a
            // builder's parameter is one number for the whole lattice.
            ENode::Const(_) | ENode::Buffer(_) | ENode::Uniform(_) | ENode::Param(_) => {
                Variance::CONST
            }
            // A unit varies as its referent does, which the leaf carries
            // because its body is not here to ask.
            ENode::Ref { variance, .. } => *variance,
            ENode::Op { children, .. } => children
                .iter()
                .fold(Variance::CONST, |acc, &c| acc.union(child(c))),
            // The one node that *shrinks* the set: its index is bound, so it
            // is not free in the result.
            ENode::Reduce { fold, body } => {
                child(*body).without(Variance::from_var(fold.binder().var()))
            }
        }
    }
}

/// The variance of a `Var` leaf: its own bit.
///
/// 0→X, 1→Y, 2 and 3 the retired axes, then the reduction index slots up to
/// [`Variance::VARIABLES`]. An index past them is not a variable the bitset
/// can name, so it is conservatively [`Variance::ALL`] — an answer that can
/// only decline an invariance, never claim one.
fn var_variance(v: u8) -> Variance {
    if v < Variance::VARIABLES {
        Variance::from_var(v)
    } else {
        Variance::ALL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classes(n: u32) -> Vec<EClassId> {
        (1..=n).map(EClassId).collect()
    }

    /// An operand count that is not the op's arity is a bug in whoever built
    /// the node, named with the op and the count.
    #[test]
    #[should_panic(expected = "Add takes 2 operands")]
    fn an_operation_applied_to_too_few_operands_is_refused() {
        let _ = ENode::apply(OpKind::Add, &classes(1));
    }

    /// A leaf kind has no operands to apply, so it is not an operation node.
    #[test]
    #[should_panic(expected = "Const takes 0 operands")]
    fn a_leaf_kind_is_not_an_operation() {
        let _ = ENode::apply(OpKind::Const, &[]);
    }

    #[test]
    #[should_panic(expected = "got 4")]
    fn an_operation_applied_to_too_many_operands_is_refused() {
        let _ = ENode::apply(OpKind::MulAdd, &classes(4));
    }

    /// The operands are the live prefix and nothing else: the slice view
    /// has exactly `arity` classes, and writing through it reaches those.
    #[test]
    fn operands_are_exactly_the_live_prefix() {
        for op in [OpKind::Neg, OpKind::Add, OpKind::MulAdd] {
            let mut node = ENode::apply(op, &classes(op.arity() as u32));
            assert_eq!(node.children_slice(), classes(op.arity() as u32));
            for class in node.children_slice_mut() {
                *class = EClassId(9);
            }
            assert_eq!(node.children_slice(), vec![EClassId(9); op.arity()]);
        }
    }

    /// Equality and hashing are structural over the live operands: the same
    /// op on the same classes is one node, and a different class, a
    /// different op, or a different order is another.
    #[test]
    fn nodes_are_equal_exactly_when_op_and_operands_are() {
        use core::hash::{BuildHasher, BuildHasherDefault};
        let hash =
            |node: &ENode| BuildHasherDefault::<rustc_hash::FxHasher>::default().hash_one(node);
        let a = ENode::apply(OpKind::Sub, &classes(2));
        let same = ENode::apply(OpKind::Sub, &classes(2));
        assert_eq!(a, same);
        assert_eq!(hash(&a), hash(&same));
        assert_ne!(a, ENode::apply(OpKind::Add, &classes(2)));
        assert_ne!(a, ENode::apply(OpKind::Sub, &[EClassId(2), EClassId(1)]));
        assert_ne!(
            ENode::apply(OpKind::Neg, &classes(1)),
            ENode::apply(OpKind::Neg, &[EClassId(2)])
        );
    }

    /// The coordinates and every reduction slot get a bit each; past the
    /// word the analysis gives up and says everything.
    #[test]
    fn a_var_leaf_names_its_own_bit() {
        let leaf = |v: u8| ENode::Var(v).variance(|_| Variance::ALL);
        assert_eq!(leaf(0), Variance::X);
        assert_eq!(leaf(1), Variance::Y);
        // Above the axes: the retired Z/W indices and every binder slot, one
        // bit each.
        assert_eq!(leaf(3), Variance::from_var(3));
        assert_eq!(leaf(8), Variance::from_var(8));
        assert_eq!(
            leaf(Variance::VARIABLES - 1),
            Variance::from_var(Variance::VARIABLES - 1)
        );
        assert_eq!(leaf(Variance::VARIABLES), Variance::ALL);
    }
}
