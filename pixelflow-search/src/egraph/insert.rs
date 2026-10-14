//! Inserting a term into an e-graph.
//!
//! One function, over any [`Ir`]. It replaces two hand-rolled arena→e-graph
//! conversions (`EGraph::add_arena` and `runtime::arena_to_egraph`) that
//! existed for the *same* IR and disagreed about panicking, about which ops
//! were representable, and about whether unreachable nodes were inserted —
//! see docs/plans/2026-09-04-ir-as-a-trait.md §2.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use pixelflow_ir::{Children, Ir, Shape};

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Vocabulary;
use super::shift::{self, ShiftDirection};

/// Why a term could not be inserted.
///
/// Declining is the only failure mode: a caller that meets one compiles the
/// term unoptimized, which is always available because optimization is never
/// required for correctness. The previous `add_arena` panicked on each of
/// these instead, so its callers had to pre-screen the term to avoid aborting
/// the build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Declined {
    /// An op this [`Vocabulary`] may not hold.
    Op(pixelflow_ir::OpKind),
    /// An op applied to a number of operands other than its arity.
    ///
    /// An arena's n-ary builder (`push_nary`) checks no arity, so a term can
    /// carry one; a node cannot, and building it would abort the build for a
    /// term the caller could have compiled unoptimized.
    Arity {
        /// The operation.
        op: pixelflow_ir::OpKind,
        /// How many operands the term gave it.
        operands: usize,
    },
    /// A shift whose count is not a `Const` holding an integer in `0..32`.
    ///
    /// A count outside that range has a different answer on each target, and
    /// a count that is not a literal is not a parameter of the shift but an
    /// operand, which a shift node cannot hold. The term compiles
    /// unoptimized, and the emitter refuses it there as it always has: a
    /// literal out of range as "not an integer in 0..32", a count that is not
    /// a literal as "must be a Const".
    ShiftCount(ShiftDirection),
    /// A macro-parameter slot, under a [`Vocabulary`] that may not hold one.
    ///
    /// Only [`Vocabulary::Runtime`] declines it: a `Param` at bake time means
    /// a builder was never called, and the term reached compilation without
    /// being specialized. The macro tier holds params natively as
    /// [`ENode::Param`], because an unbound slot
    /// is what a builder *is*.
    Param(u8),
    /// A kernel named by content, which the graph was not told is a unit.
    ///
    /// The runtime tier holds a reference as an opaque leaf when its unit
    /// walk has admitted it (`EGraph::admit_unit`): the unit is optimized by
    /// itself and linked after extraction
    /// (docs/plans/2026-09-25-the-language-is-kernel.md §4, O1). Anything
    /// else — the macro tier, a research tool, a reference no walk admitted —
    /// meets a name with no structure to rewrite and no variance to read,
    /// and declines it: inlining inside saturation is a rule that does not
    /// exist (docs/plans/2026-09-09-composition-is-linking.md §3).
    Ref(pixelflow_ir::KernelKey),
    /// A `Write` — the store the lattice's folds wrap a kernel in
    /// (docs/plans/2026-09-16-collapse-is-a-fold.md §2.4). Declined for a
    /// standing reason: an effect is not a value, so no rule may rewrite
    /// it, and it is built by the legalize passes *after* extraction, so a
    /// term carrying one into saturation skipped the pipeline.
    Write,
}

/// Insert the subgraph reachable from `root` into `egraph`, returning the
/// e-class the root lands in.
///
/// Memoized on `I::Ref` (on top of the e-graph's own hash-consing by node
/// shape) so a DAG-shared term is walked once per node, not once per
/// reference. Iterative: term depth is unbounded in principle (`Dwrt`
/// chain-rule expansion, deep composition), so this must not blow the Rust
/// stack.
///
/// `Buffer` and `Uniform` leaves insert as themselves, carrying their
/// declarations, and under [`Vocabulary::Templates`] so does `Param`. Under
/// [`Vocabulary::Runtime`] a `Param` declines: by bake time a builder should
/// have substituted it, so one surviving is a term that was never
/// specialized. Which vocabulary may hold which leaf is the job `Vocabulary`
/// exists for, and saying it there is what stopped the macro tier from
/// smuggling params past this gate disguised as `Var`s. A `Ref` inserts as
/// an opaque [`ENode::Ref`] leaf under [`Vocabulary::Runtime`] when the graph
/// has admitted it as a unit (`EGraph::admit_unit`, which carries the
/// variance its body is not here to give), and declines otherwise.
///
/// **Reachable-only.** A term representation may hold nodes no longer reached
/// from `root` — an arena accumulates construction garbage — and inserting
/// those would spend `max_minted`, a budget dimension, on nodes the budget's
/// own `node_count` never counted.
pub fn insert<I: Ir>(
    term: &I,
    root: I::Ref,
    egraph: &mut EGraph,
    vocab: Vocabulary,
) -> Result<EClassId, Declined> {
    enum Task<R> {
        /// Resolve children first, then build this node.
        Visit(R),
        /// Children are on the result stack; pop and build.
        Complete(R),
    }

    let mut memo: BTreeMap<I::Ref, EClassId> = BTreeMap::new();
    let mut tasks: Vec<Task<I::Ref>> = alloc::vec![Task::Visit(root)];
    let mut built: Vec<EClassId> = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(r) => {
                if let Some(&class) = memo.get(&r) {
                    built.push(class);
                    continue;
                }
                let class = match term.project(r) {
                    Shape::Var(i) => egraph.add(ENode::Var(i)),
                    Shape::Const(v) => egraph.add(ENode::constant(v)),
                    Shape::Param(i) => match vocab {
                        Vocabulary::Templates => egraph.add(ENode::Param(i)),
                        Vocabulary::Runtime => return Err(Declined::Param(i)),
                    },
                    Shape::Ref(key) => match (vocab, egraph.unit_variance(key)) {
                        (Vocabulary::Runtime, Some(variance)) => {
                            egraph.add(ENode::Ref { key, variance })
                        }
                        _ => return Err(Declined::Ref(key)),
                    },
                    Shape::Write { .. } => return Err(Declined::Write),
                    Shape::Buffer(decl) => egraph.add(ENode::Buffer(decl)),
                    Shape::Uniform(decl) => egraph.add(ENode::Uniform(decl)),
                    Shape::Op(kind, children) => {
                        // Check before descending so an unrepresentable op
                        // is reported at the node that carries it.
                        if !vocab.admits(kind) {
                            return Err(Declined::Op(kind));
                        }
                        if children.len() != kind.arity() {
                            return Err(Declined::Arity {
                                op: kind,
                                operands: children.len(),
                            });
                        }
                        tasks.push(Task::Complete(r));
                        // A shift's count is the shift's own parameter, not
                        // a term to visit: only its operand is.
                        if let Some(direction) = ShiftDirection::of(kind) {
                            let Some((operand, _)) = shift::spelled(term, children) else {
                                return Err(Declined::ShiftCount(direction));
                            };
                            tasks.push(Task::Visit(operand));
                            continue;
                        }
                        // Reverse, so children pop in operand order.
                        for i in (0..children.len()).rev() {
                            let child = children.get(i).expect("index < len");
                            tasks.push(Task::Visit(child));
                        }
                        continue;
                    }
                    // No vocabulary check: a fold is not an op, so there is
                    // nothing for a vocabulary to admit or refuse. It enters
                    // as itself, under either.
                    Shape::Reduce { body, .. } => {
                        tasks.push(Task::Complete(r));
                        tasks.push(Task::Visit(body));
                        continue;
                    }
                };
                memo.insert(r, class);
                built.push(class);
            }
            Task::Complete(r) => {
                if let Some(&class) = memo.get(&r) {
                    built.push(class);
                    continue;
                }
                let class = match term.project(r) {
                    Shape::Op(kind, children) => match ShiftDirection::of(kind) {
                        Some(direction) => {
                            let (_, count) = shift::spelled(term, children)
                                .expect("the count was checked when the shift was visited");
                            let operand = built.pop().expect("the operand was visited first");
                            egraph.add(ENode::Shift {
                                direction,
                                count,
                                operand,
                            })
                        }
                        None => {
                            let start = built.len() - children.len();
                            let node = ENode::apply(kind, &built[start..]);
                            built.truncate(start);
                            egraph.add(node)
                        }
                    },
                    Shape::Reduce { fold, .. } => {
                        let body = built.pop().expect("the body was visited first");
                        egraph.add(ENode::Reduce { fold, body })
                    }
                    _ => unreachable!("Complete is scheduled only for compound shapes"),
                };
                memo.insert(r, class);
                built.push(class);
            }
        }
    }

    Ok(built.pop().expect("insert: root produced no e-class"))
}

/// The classes [`insert`] makes from `root`, without making them — the size
/// measure saturation budgets key on.
///
/// Shares [`insert`]'s reachability so a budget cannot be picked from one node
/// set while the graph is built from another. That is not the count of nodes
/// in the term: a shift's count is read, not visited, so its literal is
/// counted only if something else reaches it. A measure of the term itself
/// is a walk of the term.
pub fn inserted_class_count<I: Ir>(term: &I, root: I::Ref) -> usize {
    let mut seen: BTreeSet<I::Ref> = BTreeSet::new();
    let mut stack = alloc::vec![root];
    while let Some(r) = stack.pop() {
        if !seen.insert(r) {
            continue;
        }
        match term.project(r) {
            // A shift's count is read, not visited, as in `insert`.
            Shape::Op(kind, children) if ShiftDirection::of(kind).is_some() => {
                match shift::spelled(term, children) {
                    Some((operand, _)) => stack.push(operand),
                    None => stack.extend((0..children.len()).filter_map(|i| children.get(i))),
                }
            }
            Shape::Op(_, children) => match children {
                Children::Many(s) => stack.extend_from_slice(s),
                other => {
                    for i in 0..other.len() {
                        stack.push(other.get(i).expect("index < len"));
                    }
                }
            },
            Shape::Reduce { body, .. } => stack.push(body),
            Shape::Write { value, .. } => stack.push(value),
            _ => {}
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::{ExprArena, ExprId, Kernel, KernelKey, KernelStore, OpKind, Variance};

    /// `Ref(named)·Y` over a fresh arena, the reference, and the key it names.
    fn named_times_y() -> (ExprArena, ExprId, ExprId, KernelKey) {
        let named = Kernel::x().mul(&Kernel::constant(5.5));
        let key = KernelStore::intern(&named);
        let mut arena = ExprArena::new();
        let reference = arena.push_ref(key);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Mul, reference, y);
        (arena, root, reference, key)
    }

    /// A unit the graph admitted inserts as one opaque leaf, carrying the
    /// variance it was admitted with — so the product over it varies with X
    /// as well as Y, which nothing could have learned from the name.
    #[test]
    fn an_admitted_unit_is_a_leaf_carrying_its_variance() {
        let (arena, root, reference, key) = named_times_y();
        let mut eg = EGraph::new();
        eg.admit_unit(key, Variance::X);
        let product = insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("admitted");
        let leaf = insert(&arena, reference, &mut eg, Vocabulary::Runtime).expect("admitted");
        assert!(
            eg.nodes(leaf).iter().any(|n| matches!(
                n,
                ENode::Ref { key: k, variance } if *k == key && *variance == Variance::X
            )),
            "the leaf names the unit and carries its variance"
        );
        assert_eq!(eg.variance(product), Variance::X.union(Variance::Y));
    }

    /// Admission is the runtime tier's: under the macro tier's vocabulary a
    /// reference still declines, admitted or not, as it does in a graph no
    /// one told about it.
    #[test]
    fn only_the_runtime_vocabulary_holds_a_unit() {
        let (arena, root, _, key) = named_times_y();
        let mut admitted = EGraph::new();
        admitted.admit_unit(key, Variance::X);
        assert_eq!(
            insert(&arena, root, &mut admitted, Vocabulary::Templates),
            Err(Declined::Ref(key))
        );
        let mut untold = EGraph::new();
        assert_eq!(
            insert(&arena, root, &mut untold, Vocabulary::Runtime),
            Err(Declined::Ref(key))
        );
    }

    /// `push_nary` takes any op and any count. A term that gives an op the
    /// wrong number of operands is declined, naming both, and leaves the
    /// graph untouched, where building the node would abort.
    #[test]
    fn an_op_with_the_wrong_operand_count_is_declined_not_built() {
        for (op, operands) in [
            (OpKind::Add, 1),
            (OpKind::Add, 3),
            (OpKind::Neg, 2),
            (OpKind::MulAdd, 4),
        ] {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let root = arena.push_nary(op, &alloc::vec![x; operands]);
            let mut eg = EGraph::new();
            assert_eq!(
                insert(&arena, root, &mut eg, Vocabulary::Runtime),
                Err(Declined::Arity { op, operands }),
            );
            assert_eq!(eg.num_classes(), 0, "{op:?}: nothing was built");
        }
    }

    /// `Shl(x, 3)` over the arena's literal count inserts as one shift node
    /// over `x`'s class, and no class is made for the count: two classes,
    /// not three. A literal that is also an ordinary operand elsewhere is
    /// made there, once, and the shift's count does not mention it.
    #[test]
    fn a_shift_inserts_its_count_as_a_parameter_not_a_class() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let three = arena.push_const(3.0);
        let shifted = arena.push_binary(OpKind::Shl, x, three);
        let mut eg = EGraph::new();
        let class = insert(&arena, shifted, &mut eg, Vocabulary::Runtime).expect("a valid shift");
        assert_eq!(eg.num_classes(), 2);
        assert_eq!(inserted_class_count(&arena, shifted), 2);
        assert!(matches!(
            eg.nodes(class),
            [ENode::Shift {
                direction: ShiftDirection::Left,
                count,
                ..
            }] if count.bits() == 3
        ));

        let scaled = arena.push_binary(OpKind::Mul, x, three);
        let both = arena.push_binary(OpKind::Add, shifted, scaled);
        let mut eg = EGraph::new();
        insert(&arena, both, &mut eg, Vocabulary::Runtime).expect("valid");
        assert_eq!(
            eg.num_classes(),
            5,
            "x, the shift, the literal (made by the Mul), the Mul, the Add"
        );
        assert_eq!(inserted_class_count(&arena, both), 5);
    }

    /// Two shifts of one operand by one count are one node; by two counts,
    /// two.
    #[test]
    fn a_shift_is_hash_consed_on_its_count() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let (two, three) = (arena.push_const(2.0), arena.push_const(3.0));
        let a = arena.push_binary(OpKind::Shr, x, two);
        let b = arena.push_binary(OpKind::Shr, x, three);
        let root = arena.push_binary(OpKind::Add, a, b);
        let mut eg = EGraph::new();
        insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("valid");
        assert_eq!(eg.num_classes(), 4, "x, two shifts, the sum");
    }

    /// A count that is not a `Const` holding an integer in `0..32` is
    /// declined, naming the direction, before anything is built: a graph cannot
    /// hold what the emitter would refuse, and the term compiles unoptimized.
    #[test]
    fn a_shift_by_anything_but_a_valid_literal_is_declined() {
        for op in [OpKind::Shl, OpKind::Shr] {
            let direction = ShiftDirection::of(op).expect("a shift");
            for count in [-1.0, -0.0, 0.5, 32.0, 33.0, 256.0, f32::NAN, f32::INFINITY] {
                let mut arena = ExprArena::new();
                let x = arena.push_var(0);
                let count_node = arena.push_const(count);
                let root = arena.push_binary(op, x, count_node);
                let mut eg = EGraph::new();
                assert_eq!(
                    insert(&arena, root, &mut eg, Vocabulary::Runtime),
                    Err(Declined::ShiftCount(direction)),
                    "{op:?} by {count}"
                );
                assert_eq!(eg.num_classes(), 0, "nothing was built");
            }

            // A count that is arithmetic, or a leaf that is not a literal, is
            // an operand and not a parameter.
            for count_node in [
                {
                    let mut arena = ExprArena::new();
                    let x = arena.push_var(0);
                    let (a, b) = (arena.push_const(1.0), arena.push_const(2.0));
                    let sum = arena.push_binary(OpKind::Add, a, b);
                    (arena, x, sum)
                },
                {
                    let mut arena = ExprArena::new();
                    let x = arena.push_var(0);
                    let y = arena.push_var(1);
                    (arena, x, y)
                },
            ] {
                let (mut arena, x, count) = count_node;
                let root = arena.push_binary(op, x, count);
                let mut eg = EGraph::new();
                assert_eq!(
                    insert(&arena, root, &mut eg, Vocabulary::Runtime),
                    Err(Declined::ShiftCount(direction)),
                );
                assert_eq!(eg.num_classes(), 0, "nothing was built");
            }
        }
    }

    /// A `Tuple` has no fixed arity, so no vocabulary holds one: it is
    /// declined as an op before its operand count is asked about.
    #[test]
    fn a_tuple_is_declined_by_every_vocabulary() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let root = arena.push_nary(OpKind::Tuple, &[x, x]);
        for vocabulary in [Vocabulary::Templates, Vocabulary::Runtime] {
            assert_eq!(
                insert(&arena, root, &mut EGraph::new(), vocabulary),
                Err(Declined::Op(OpKind::Tuple)),
            );
        }
    }

    /// A unit is optimized out of its context, which is sound only for a
    /// closed term: one reading a binder is refused at the door.
    #[test]
    #[should_panic(expected = "a unit must be")]
    fn an_open_unit_is_refused() {
        let key = KernelStore::intern(&Kernel::y().add(&Kernel::constant(0.75)));
        EGraph::new().admit_unit(key, Variance::Y.union(Variance::BINDERS));
    }
}
