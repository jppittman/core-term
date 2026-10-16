//! Core e-graph data structures: EClassId and ENode.

use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use pixelflow_ir::arena::{BufferDecl, UniformDecl};
use pixelflow_ir::fold::Fold;
use pixelflow_ir::{KernelKey, OpKind, ShiftCount, Variance};

use super::shift::ShiftDirection;

/// Identifier for an equivalence class in the e-graph.
///
/// 32 bits wide, against the control plane's 64-bit rule, for a measured
/// reason (2026-10-14). The field was widened to `u64` and measured on a
/// native release build: the cold compile of the 29 programs ASCII needs, and
/// of the n=40 program (a per-count program of n pieces,
/// docs/language/2026-09-25-the-language-is-kernel.md), in interleaved
/// before/after pairs, each in its own process, peak RSS read from the parent
/// (`ru_maxrss`, KiB, shown here divided by 1000). The harness is out of tree.
///
/// - Peak RSS, this field alone widened, 3 pairs: 132.8 to 156.7 MB (+18%) on
///   the 29 programs, 19.2 to 20.8 MB (+8%) on n=40. Also widening the sparse
///   reach members, as a first attempt did, gave 158.3 MB (+19%, 6 pairs) and
///   22.2 MB (+16%).
/// - Cold time on the 29 programs, field alone: 1000 to 1058 ms (+6%), inside
///   the unchanged binary's own spread (946 to 1083 ms over 9 runs). Time on
///   n=40 is not resolved: the unchanged binary alone ran 67 to 98 ms. RSS is
///   the deterministic signal.
///
/// The cause is the node: [`ENode`] grew from 24 to 40 bytes because
/// `Operands` grew from 16 to 32, and an e-graph is mostly e-nodes.
///
/// The width does not bind. Ids number the classes a graph holds: one per
/// distinct node the input inserts (`insert` has no cap, and the input
/// arena's `ExprId` is `u32`), plus what saturation adds, which stops near
/// [`HARD_MINTED_LIMIT`](super::HARD_MINTED_LIMIT) (100,000, approximately:
/// the runner holds it to its prediction's slack). That is saturation's
/// growth cap, not a cap on ids. The largest program seen inlined, Noto's
/// ASCII, holds 169,263 classes before any rule fires, about 25,000 times
/// below 2^32.
///
/// Every class id in the crate is an `EClassId`, and every container of them
/// holds one, so widening is this field plus `index` and `from_index`. The
/// sparse reach members in `extract.rs` are live positions, not ids, and stay
/// `u32` on purpose (a budget question, a separate change); they are the next
/// cap after this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EClassId(pub(crate) u32);

impl EClassId {
    /// Get the index of this e-class ID.
    ///
    /// This is useful for using EClassId as a key in external data structures.
    pub fn index(self) -> usize {
        // Lossless: `usize` is 64 bits on both supported targets.
        self.0 as usize
    }

    /// The class id at `index`: the inverse of [`EClassId::index`].
    ///
    /// # Panics
    ///
    /// If `index` does not fit 32 bits (2^32 classes), refused here and never
    /// wrapped to a small id.
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("a class id past u32::MAX: more than 2^32 e-classes"))
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
/// prefixes are, and the derived `Eq`, `Hash` and `Ord` need no hand-written
/// case for the tail. The derived order is by length, then by the ids; it
/// reads the tail too, which is sound because the tail is always zero: it
/// compares equal and decides nothing, so among lists of one length the order
/// is the live prefix's. The fields are private and the one constructor takes
/// the operation it is for, so the tail cannot be written and the length is
/// checked against that operation's arity. The slice view, and `Debug`, are
/// the live prefix only.
///
/// The operation is not stored here, so the pairing is checked when the
/// operands are built and not afterwards: a variant's fields are public, and
/// `ENode::Op` can be written with another node's operands under a different
/// `op`. [`ENode::apply`] is the constructor that keeps them together.
///
/// A shift has no operands in this sense: its count is a parameter, held by
/// [`ENode::Shift`], so the constructor refuses `Shl` and `Shr`.
///
/// `len` counts slots of an inline array, not anything a program can say: its
/// range is `MAX_OPERANDS` whatever its width, so a wider count would let no
/// program express more.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    /// `Tuple`, has none to hold), or is a shift, whose count is a parameter
    /// and not an operand. An operation applied to the wrong number of
    /// operands is not a node, and a rule that builds one has a bug to find,
    /// not a graph to carry it.
    #[must_use]
    pub(crate) fn new(op: OpKind, classes: &[EClassId]) -> Self {
        assert!(
            ShiftDirection::of(op).is_none(),
            "{op:?} is a shift: its count is a parameter of ENode::Shift, not an operand class",
        );
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
/// The order is structural too: variant in declaration order, then fields. It
/// is a total order to sort and key by, not a measure of the node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    /// A shift of one operand by a static count.
    ///
    /// The count is a parameter of the operator, as a [`ENode::Reduce`]'s
    /// fold is, and for the same reason: as a child class it would be an
    /// e-class like any other, and the `Const(8.0)` naming a count would be
    /// the same class as every literal `8.0` in the kernel. Every arithmetic
    /// rule could then rewrite it into something that is not a [`ShiftCount`],
    /// and extraction would have to repair the count afterwards.
    /// Here no rule reaches it, no class holds it, and a value of
    /// [`ShiftCount`] is already valid.
    ///
    /// The arena spells a shift `Shl(x, Const(count))`, with the count as a
    /// second operand; [`insert`](fn@super::insert) reads the count off that
    /// literal without visiting it, and extraction writes the literal back.
    Shift {
        /// Which way the bits move.
        direction: ShiftDirection,
        /// By how many.
        count: ShiftCount,
        /// The class being shifted.
        operand: EClassId,
    },
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
    /// docs/language/2026-09-25-the-language-is-kernel.md §4, O1). A leaf here,
    /// opaque to every rule as a [`ENode::Uniform`] is: its body is not in
    /// this graph, so no rewrite can reach inside it, and extraction hands
    /// back the name.
    ///
    /// The node is the key and nothing else. Its variance — its referent's,
    /// the one fact a rule reads off a leaf (`FactorFold`'s side condition,
    /// and the binder substitution `PeelFold`/`HalveFold` skip a binder-free
    /// class by) and the one fact extraction prices a leaf's evaluations by —
    /// is a function of the key that the graph holds
    /// (`EGraph::unit_variance`), not a second copy the node could disagree
    /// with. A unit is closed (its variance names no binder), so every
    /// rewrite conditioned on binder-invariance is the one it would be with
    /// the body inlined.
    /// Only the runtime tier's unit walk admits one (`EGraph::admit_unit`);
    /// [`insert`](fn@super::insert) declines a reference the graph was not told
    /// is a unit, and [`EGraph::add`](super::EGraph::add) refuses one.
    Ref { key: KernelKey },
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
    /// none to hold), or is a shift, which is an [`ENode::Shift`] and takes
    /// its count as a parameter.
    #[must_use]
    pub fn apply(op: OpKind, operands: &[EClassId]) -> Self {
        ENode::Op {
            op,
            children: Operands::new(op, operands),
        }
    }

    /// The operation this node performs, if it is one: an `Op`'s own, or the
    /// shift a [`ENode::Shift`] names.
    pub fn op(&self) -> Option<OpKind> {
        match self {
            ENode::Op { op, .. } => Some(*op),
            ENode::Shift { direction, .. } => Some(direction.op()),
            _ => None,
        }
    }

    /// The node's shape: the one `OpKind` that names what it is, leaves and
    /// folds included; `None` for a [`ENode::Ref`], which no `OpKind` names.
    ///
    /// The one kind function: a rule's root-shape check, the e-graph's
    /// per-class kind masks and its per-kind class sets all ask it, so they
    /// cannot disagree about what a node is.
    pub(crate) fn shape(&self) -> Option<OpKind> {
        match self {
            ENode::Var(_) => Some(OpKind::Var),
            ENode::Const(_) => Some(OpKind::Const),
            ENode::Buffer(_) => Some(OpKind::Buffer),
            ENode::Uniform(_) => Some(OpKind::Uniform),
            ENode::Param(_) => Some(OpKind::Param),
            ENode::Reduce { .. } => Some(OpKind::Reduce),
            ENode::Op { .. } | ENode::Shift { .. } => self.op(),
            ENode::Ref { .. } => None,
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
            ENode::Shift { operand, .. } => core::slice::from_ref(operand),
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
            ENode::Shift { operand, .. } => core::slice::from_mut(operand),
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
    ///
    /// A unit's body is not in the graph, so `unit` is where its variance
    /// comes from: the graph's record of what it was admitted as.
    pub(crate) fn variance(
        &self,
        mut unit: impl FnMut(KernelKey) -> Variance,
        mut child: impl FnMut(EClassId) -> Variance,
    ) -> Variance {
        match self {
            ENode::Var(v) => var_variance(*v),
            // A buffer's contents are fixed for the kernel's lifetime — a
            // read of one varies with its index, which is the `Gather`'s
            // other children, unioned by the `Op` arm. A uniform or a
            // builder's parameter is one number for the whole lattice.
            ENode::Const(_) | ENode::Buffer(_) | ENode::Uniform(_) | ENode::Param(_) => {
                Variance::CONST
            }
            // A unit varies as its referent does, which the graph recorded
            // because the body is not here to ask.
            ENode::Ref { key } => unit(*key),
            ENode::Op { children, .. } => children
                .iter()
                .fold(Variance::CONST, |acc, &c| acc.union(child(c))),
            ENode::Shift { operand, .. } => child(*operand),
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

    fn classes(n: usize) -> Vec<EClassId> {
        (1..=n).map(EClassId::from_index).collect()
    }

    /// The unit lookup for a node that reads none.
    fn no_unit(key: KernelKey) -> Variance {
        panic!("this node names no unit, but asked for {key:?}")
    }

    /// An index at `u32::MAX` round-trips through a class id.
    #[test]
    fn a_class_id_at_u32_max_round_trips() {
        let top = u32::MAX as usize;
        assert_eq!(EClassId::from_index(top).index(), top);
    }

    /// An index one past `u32::MAX` is refused, never wrapped to a small id.
    #[test]
    #[should_panic(expected = "a class id past u32::MAX")]
    fn a_class_id_past_u32_max_is_refused() {
        let _ = EClassId::from_index(u32::MAX as usize + 1);
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

    /// A shift's count is a parameter of [`ENode::Shift`], so no operand list
    /// is a shift's: a count class beside the operand cannot be built.
    #[test]
    #[should_panic(expected = "Shl is a shift")]
    fn a_shift_is_not_an_operation_over_operand_classes() {
        let _ = ENode::apply(OpKind::Shl, &classes(2));
    }

    /// A shift has its one operand as its only child, and its op is the one
    /// its direction names.
    #[test]
    fn a_shift_has_one_child_and_names_its_op() {
        let count = ShiftCount::from_literal(8.0).expect("in range");
        let node = ENode::Shift {
            direction: ShiftDirection::Right,
            count,
            operand: EClassId(3),
        };
        assert_eq!(node.children_slice(), [EClassId(3)]);
        assert_eq!(node.op(), Some(OpKind::Shr));
        assert_eq!(node.variance(no_unit, |_| Variance::X), Variance::X);
        assert_ne!(
            node,
            ENode::Shift {
                direction: ShiftDirection::Right,
                count: ShiftCount::from_literal(16.0).expect("in range"),
                operand: EClassId(3),
            },
            "the count is part of the node's identity"
        );
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
            let mut node = ENode::apply(op, &classes(op.arity()));
            assert_eq!(node.children_slice(), classes(op.arity()));
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
        let leaf = |v: u8| ENode::Var(v).variance(no_unit, |_| Variance::ALL);
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
