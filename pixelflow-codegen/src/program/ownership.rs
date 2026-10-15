//! Who a value is computed for: the arm of an `If` that owns it, read off the
//! DAG.
//!
//! An `If` has two cases, and a value computed for exactly one of them is
//! dead weight whenever the other is taken. Which values those are is a fact
//! about the **DAG** — who reads whom — and no order of the schedule changes
//! it. So it is stated here once, as a function of the reads, instead of being
//! recovered from the order the values happen to sit in.
//!
//! The denotation. Every `If` opens two **regions**, one per arm, nested in
//! the region the `If` itself belongs to. The `If` reads its mask in its own
//! region and each arm's operand in that arm's region. A value belongs to the
//! **lowest common ancestor** of the regions that read it: if only one arm
//! reads it, it is that arm's; if both arms, or the mask, or anything outside
//! the `If`, read it, it belongs to the region around the `If`. A loop reads
//! what it reads at its `Reduce` ([`FoldReads`]), so a value only the loops in
//! one arm read is that arm's, and a batch that skips them skips it too.
//!
//! That is the whole relation. It is computed in one reverse pass over the
//! schedule: every reader of a value comes after it, so by the time the pass
//! reaches a value every region that reads it is known.
//!
//! One pass, O(reads × log depth): the lowest common ancestor is a climb on
//! skip pointers, so a ladder of `else if`s a thousand deep costs what a flat
//! scope does.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::CompileError;
use crate::pass::Pass;
use crate::program::tree::Tree;
use crate::program::{Def, IfArm, ScheduledOp, ScopedSchedule, ValueId, all_operands};

/// A region of one scope: the scope itself, or one arm of one `If` in it.
///
/// Regions are numbered in the order the reverse pass meets them, so a
/// parent's number is always smaller than its children's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Region(pub(super) usize);

impl Region {
    /// The scope itself: what no arm owns.
    const SCOPE: Self = Self(0);
}

/// One arm of one `If`, as a region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Arm {
    /// The position of the `If` in the schedule.
    pub(super) at: usize,
    /// The `If`'s mask.
    pub(super) mask: ValueId,
    /// Which arm.
    pub(super) arm: IfArm,
    /// The region the arm's values belong to.
    pub(super) region: Region,
}

/// The backward pass: every scope's [`Ownership`], read off the nest.
///
/// Its own stage, between [`Nest`](super::Nest) and [`Layout`](super::Layout),
/// because ownership is a fact about the DAG that layout uses and neither
/// computes the other.
///
/// # Panics
///
/// If a scope is not closed ([`ScopedSchedule::assert_closed`]): ownership
/// sees a loop's reads only at its `Reduce`, so a read the nest did not name
/// would let an arm own a value a loop outside the arm reads.
pub(crate) struct Own;

impl Pass for Own {
    type Input = ScopedSchedule<()>;
    type Output = ScopedSchedule<Ownership>;

    fn run(self, scoped: ScopedSchedule<()>) -> Result<ScopedSchedule<Ownership>, CompileError> {
        Ok(scoped.analyze(|nest, scope| {
            nest.assert_closed(scope);
            let (schedule, _, ()) = nest.at(scope);
            let opened = nest
                .folds
                .iter()
                .filter(|fold| fold.parent == scope)
                .map(|fold| (fold.reduce, fold.reads().collect()))
                .collect();
            Ownership::of(schedule, &opened)
        }))
    }
}

/// What each loop a scope opens reads, keyed by the loop's `Reduce` def: the
/// ids its body names, which the scope's own schedule holds or an enclosing
/// scope does (`ScopeFold::reads`).
///
/// To [`operands`](crate::program::operands) a `Reduce` def is a leaf: once
/// `scopes::extract_folds` has carved a fold's body into a scope of its own,
/// the def is only where the loop opens. But the loop *runs* there, and its
/// body reads this scope: a sibling fold's result from its accumulator slot, a
/// value invariant in the fold from the park this scope leaves it in. Without
/// these edges a guard could skip a fold whose result a sibling fold's body
/// still reads, or a value a fold reads, which the fold then finds unwritten
/// (`pixelflow-core/tests/guard_sibling_fold.rs` pins both).
///
/// They take no walk of a body: a fold's schedule keeps its parent's `ValueId`s,
/// so what it reads from its parent is among the ids it names, and a fold
/// nested in it reads nothing its parent does not name
/// (`ScopedSchedule::assert_closed`, run before this). Every named id is an
/// edge, a leaf the body rebuilds for itself included, which over-approximates
/// (such a leaf is kept out of an arm the fold does not run in); one this scope
/// does not define is not a position here and is ignored.
pub(super) type FoldReads = BTreeMap<ValueId, Vec<ValueId>>;

/// Everything the def `def` reads: its operands of both register classes, and,
/// for a `Reduce` that opens a fold, what the fold reads.
pub(super) fn reads<'a>(folds: &'a FoldReads, def: &'a Def) -> impl Iterator<Item = ValueId> + 'a {
    let fold = folds.get(&def.value).into_iter().flatten().copied();
    all_operands(&def.op).chain(fold)
}

/// Ownership of one scope's schedule.
pub(crate) struct Ownership {
    /// The region each schedule position belongs to.
    region_of: Vec<Region>,
    /// The regions, nested as the `If`s are.
    regions: Tree,
    /// Every arm of every `If`, in schedule order, true arm first.
    arms: Vec<Arm>,
}

/// Where a value sits in the scope's schedule, by `ValueId`: dense, because
/// ids are handed out sequentially. Absent for a value this scope does not
/// define (a live-in from an enclosing scope).
pub(super) struct Positions(Vec<Option<usize>>);

impl Positions {
    pub(super) fn of(schedule: &[Def]) -> Self {
        let len = schedule
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut at = alloc::vec![None; len];
        for (pos, def) in schedule.iter().enumerate() {
            at[def.value.0 as usize] = Some(pos);
        }
        Self(at)
    }

    pub(super) fn get(&self, value: ValueId) -> Option<usize> {
        self.0.get(value.0 as usize).copied().flatten()
    }
}

impl Ownership {
    /// The ownership of `schedule`.
    ///
    /// `folds` is what each loop the scope opens reads from it, which makes
    /// the loop's `Reduce` def a reader of each of those values.
    pub(super) fn of(schedule: &[Def], folds: &FoldReads) -> Self {
        let positions = Positions::of(schedule);
        let mut me = Self {
            region_of: alloc::vec![Region::SCOPE; schedule.len()],
            regions: Tree::rooted(),
            arms: Vec::new(),
        };
        // The region every reader of each position agrees on so far.
        let mut readers: Vec<Option<Region>> = alloc::vec![None; schedule.len()];

        for (pos, def) in schedule.iter().enumerate().rev() {
            // A value nothing reads is the scope's own: the scope's result, or
            // an effect.
            let here = readers[pos].unwrap_or(Region::SCOPE);
            me.region_of[pos] = here;

            let mut read = |me: &mut Self, value: ValueId, region: Region| {
                let Some(at) = positions.get(value) else {
                    return;
                };
                readers[at] = Some(match readers[at] {
                    None => region,
                    Some(known) => Region(me.regions.common_ancestor(known.0, region.0)),
                });
            };

            match &def.op {
                ScheduledOp::If(mask, if_true, if_false) => {
                    let on_true = Region(me.regions.grow(here.0));
                    let on_false = Region(me.regions.grow(here.0));
                    // Pushed false arm first: the sweep meets the `If`s
                    // last-to-first and the list is reversed once at the end.
                    me.arms.push(Arm {
                        at: pos,
                        mask: *mask,
                        arm: IfArm::False,
                        region: on_false,
                    });
                    me.arms.push(Arm {
                        at: pos,
                        mask: *mask,
                        arm: IfArm::True,
                        region: on_true,
                    });
                    read(&mut me, *mask, here);
                    read(&mut me, *if_true, on_true);
                    read(&mut me, *if_false, on_false);
                }
                _ => {
                    for value in reads(folds, def) {
                        read(&mut me, value, here);
                    }
                }
            }
        }

        me.arms.reverse();
        me
    }

    /// Every arm of every `If`, in schedule order, an `If`'s true arm first.
    pub(super) fn arms(&self) -> &[Arm] {
        &self.arms
    }

    /// The region the value at `pos` belongs to.
    pub(super) fn region_of(&self, pos: usize) -> Region {
        self.region_of[pos]
    }

    /// The regions, for a stage that nests something in them.
    pub(super) fn regions(&self) -> &Tree {
        &self.regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{CompareOp, Item, Layout, Nest, UnaryOp};
    use pixelflow_ir::LatticeShape;
    use pixelflow_ir::arena::{ExprArena, ExprId};
    use pixelflow_ir::fold::{Binder, Fold, Monoid};
    use pixelflow_ir::kind::OpKind;
    use pixelflow_ir::variance::Variance;

    /// A def whose variance is a placeholder, `CONST`: the analyses under
    /// test read operands and nesting, never a def's variance (scoping does,
    /// and these fixtures do not go through it). Only [`var`] is exact.
    fn def(value: u64, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
            variance: Variance::CONST,
        }
    }

    /// A binder's value: a leaf the scope reads and does not compute, varying
    /// with that binder alone.
    fn var(value: u64, slot: u8) -> Def {
        let binder = Binder::from_slot(slot).expect("a binder slot");
        Def {
            value: ValueId(value),
            op: ScheduledOp::Var(binder),
            variance: Variance::from_var(binder.var()),
        }
    }

    fn if_of(value: u64, mask: u64, if_true: u64, if_false: u64) -> Def {
        def(
            value,
            ScheduledOp::If(ValueId(mask), ValueId(if_true), ValueId(if_false)),
        )
    }

    /// The positions an arm owns, nested arms included.
    fn owned(own: &Ownership, arm: &Arm, len: usize) -> Vec<usize> {
        (0..len)
            .filter(|&pos| {
                own.regions()
                    .common_ancestor(own.region_of(pos).0, arm.region.0)
                    == arm.region.0
            })
            .collect()
    }

    /// Only the true arm reads `Rsqrt(v1)`, so the true arm owns it; the mask
    /// and the false arm (the mask again) own nothing.
    #[test]
    fn an_arm_owns_what_only_it_reads() {
        let schedule = alloc::vec![
            var(0, 0),
            var(1, 1),
            def(2, ScheduledOp::Unary(UnaryOp::Rsqrt, ValueId(1))),
            if_of(3, 0, 2, 0),
        ];
        let own = Ownership::of(&schedule, &FoldReads::new());
        let [on_true, on_false] = own.arms() else {
            panic!("one If is two arms");
        };
        assert_eq!((on_true.arm, on_false.arm), (IfArm::True, IfArm::False));
        assert_eq!(owned(&own, on_true, schedule.len()), alloc::vec![1, 2]);
        assert_eq!(owned(&own, on_false, schedule.len()), Vec::<usize>::new());
    }

    /// A value both arms read belongs to the region around the `If`, and so
    /// does the mask's cone.
    #[test]
    fn a_value_both_arms_read_belongs_to_neither() {
        let schedule = alloc::vec![
            var(0, 0),
            var(1, 1),
            def(2, ScheduledOp::Unary(UnaryOp::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(UnaryOp::Neg, ValueId(2))),
            def(4, ScheduledOp::Unary(UnaryOp::Sqrt, ValueId(2))),
            if_of(5, 0, 3, 4),
        ];
        let own = Ownership::of(&schedule, &FoldReads::new());
        let [on_true, on_false] = own.arms() else {
            panic!("one If is two arms");
        };
        assert_eq!(owned(&own, on_true, schedule.len()), alloc::vec![3]);
        assert_eq!(owned(&own, on_false, schedule.len()), alloc::vec![4]);
        assert_eq!(own.region_of(2), Region::SCOPE);
    }

    /// A value whose readers are all loops one arm runs is the arm's, through
    /// production scoping (`Nest`, `Own`, `Layout`): a loop reads what
    /// it reads at its `Reduce` ([`FoldReads`]), so a batch that skips the arm
    /// skips the loops and what only they read. A reader outside the arm keeps
    /// the value outside (`layout`'s
    /// `an_arm_does_not_own_what_a_sibling_fold_reads`).
    #[test]
    fn an_arm_owns_the_roots_only_its_loops_read() {
        type Placed = Vec<(Def, Option<IfArm>)>;
        // `Σ_{j<4} (v + j)`, over the binder in `slot`: a loop that reads `v`.
        let loop_over = |a: &mut ExprArena, v: ExprId, slot: u8| {
            let binder = Binder::from_slot(slot).expect("the slot exists");
            let j = a.push_var(binder.var());
            let body = a.push_binary(OpKind::Add, v, j);
            a.push_reduce(Fold::new(Monoid::SUM, binder, 0..4), body)
        };
        // The defs of the scope of `root` that holds its `If`, each with the arm
        // of the block it runs in, if any.
        let placed = |a: &ExprArena, root: ExprId| -> Placed {
            let lanes = (crate::jit_vector_bytes() / core::mem::size_of::<f32>()) as u32;
            let schedule =
                crate::pipeline::tests::schedule_for(a, root, LatticeShape::new([lanes, 1]), lanes);
            let scoped = Nest.then(Own).then(Layout).run(schedule).expect("scoped");
            let ScopedSchedule { body, folds } = scoped;
            let (schedule, flow) = core::iter::once((body.schedule, body.analysis))
                .chain(folds.into_iter().map(|fold| (fold.schedule, fold.analysis)))
                .find(|(schedule, _)| {
                    schedule
                        .iter()
                        .any(|def| matches!(def.op, ScheduledOp::If(..)))
                })
                .expect("a scope holds the If");
            let mut arm_of = alloc::vec![None; schedule.len()];
            for block in &flow.arms {
                for item in &block.items {
                    if let Item::Def(at) = *item {
                        arm_of[at] = Some(block.arm);
                    }
                }
            }
            schedule.into_iter().zip(arm_of).collect()
        };
        // The arms of the defs `which` picks, in schedule order.
        let arms = |placed: &Placed, which: fn(&ScheduledOp) -> bool| -> Vec<Option<IfArm>> {
            (placed.iter())
                .filter(|(def, _)| which(&def.op))
                .map(|&(_, arm)| arm)
                .collect()
        };
        let reduce = |op: &ScheduledOp| matches!(op, ScheduledOp::Reduce(..));
        let sqrt = |op: &ScheduledOp| matches!(op, ScheduledOp::Unary(UnaryOp::Sqrt, _));

        // `select(X < 8, R(F(X)), 0)`, `F` invariant in `R`'s binder: `F` is a
        // sibling of `R`, a root of the scope that opens both, and only `R`
        // reads it.
        let mut a = ExprArena::new();
        let (x, eight, zero) = (a.push_var(0), a.push_const(8.0), a.push_const(0.0));
        let f = loop_over(&mut a, x, 1);
        let r = loop_over(&mut a, f, 0);
        let mask = a.push_binary(OpKind::Lt, x, eight);
        let root = a.push_ternary(OpKind::If, mask, r, zero);
        assert_eq!(
            arms(&placed(&a, root), reduce),
            [Some(IfArm::True); 2],
            "the true arm runs R and the sibling F it reads"
        );

        // `select(X < 8, 0, F(√(X + Y)))`: the loop reads `√(X + Y)`, which the
        // mask does not, so the false arm owns it. The mask reads `X`, as does
        // `X + Y`, so `X` stays in the region around the `If`.
        let mut a = ExprArena::new();
        let (x, y) = (a.push_var(0), a.push_var(1));
        let (eight, zero) = (a.push_const(8.0), a.push_const(0.0));
        let sum = a.push_binary(OpKind::Add, x, y);
        let s = a.push_unary(OpKind::Sqrt, sum);
        let f = loop_over(&mut a, s, 0);
        let mask = a.push_binary(OpKind::Lt, x, eight);
        let arm = a.push_ternary(OpKind::If, mask, zero, f);
        let placed_arm = placed(&a, arm);
        assert_eq!(
            arms(&placed_arm, sqrt),
            [Some(IfArm::False)],
            "the loop's input"
        );
        // `X`, told by being what the one comparison, the mask, compares.
        let compared: Vec<ValueId> = (placed_arm.iter())
            .filter_map(|(def, _)| match def.op {
                ScheduledOp::Compare(CompareOp::Lt, x, _) => Some(x),
                _ => None,
            })
            .collect();
        let [x] = compared[..] else {
            panic!("one comparison is the mask: {compared:?}")
        };
        let (_, x_arm) = (placed_arm.iter())
            .find(|(def, _)| def.value == x)
            .expect("the mask's X is a def of this scope");
        assert_eq!(*x_arm, None, "the mask reads X too, so X stays outside");

        // The control: `select(X < 8, 0, F(√(X + Y))) + G(√(X + Y))`. `G` runs
        // whatever the mask, so `√(X + Y)` is not the arm's, while `F` still is.
        let g = loop_over(&mut a, s, 1);
        let root = a.push_binary(OpKind::Add, arm, g);
        let placed_both = placed(&a, root);
        assert_eq!(
            arms(&placed_both, sqrt),
            [None],
            "a loop outside the arm reads it"
        );
        let mut loops = arms(&placed_both, reduce);
        loops.sort();
        assert_eq!(
            loops,
            [None, Some(IfArm::False)],
            "F is the arm's, G is not"
        );
    }

    /// An arm that owns an `If` owns what that `If`'s arms own: the outer true
    /// arm holds everything nested in it.
    #[test]
    fn a_nested_arm_is_owned_by_the_arm_around_it() {
        let schedule = alloc::vec![
            var(0, 0),
            var(1, 1),
            def(2, ScheduledOp::Unary(UnaryOp::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(UnaryOp::Sqrt, ValueId(1))),
            if_of(4, 0, 2, 3),
            def(5, ScheduledOp::Unary(UnaryOp::Recip, ValueId(1))),
            if_of(6, 0, 4, 5),
        ];
        let own = Ownership::of(&schedule, &FoldReads::new());
        let [inner_true, inner_false, outer_true, outer_false] = own.arms() else {
            panic!("two Ifs are four arms, in schedule order");
        };
        assert_eq!(owned(&own, inner_true, schedule.len()), alloc::vec![2]);
        assert_eq!(owned(&own, inner_false, schedule.len()), alloc::vec![3]);
        assert_eq!(
            owned(&own, outer_true, schedule.len()),
            alloc::vec![2, 3, 4],
            "the inner If and everything under it belongs to the arm it is the root of"
        );
        assert_eq!(owned(&own, outer_false, schedule.len()), alloc::vec![5]);
    }

    /// The relation is a property of the reads, so reordering the schedule
    /// within what the reads allow changes no arm's membership.
    #[test]
    fn the_order_of_the_schedule_does_not_change_what_is_owned() {
        let ops = alloc::vec![
            var(0, 0),
            var(1, 1),
            def(2, ScheduledOp::Unary(UnaryOp::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(UnaryOp::Sqrt, ValueId(1))),
            def(4, ScheduledOp::Unary(UnaryOp::Neg, ValueId(3))),
            if_of(5, 0, 2, 4),
        ];
        // The same defs with the two arms' work interleaved the other way.
        let swapped = alloc::vec![
            ops[0].clone(),
            ops[1].clone(),
            ops[3].clone(),
            ops[2].clone(),
            ops[4].clone(),
            ops[5].clone(),
        ];
        let by_value = |schedule: &[Def]| {
            let own = Ownership::of(schedule, &FoldReads::new());
            own.arms()
                .iter()
                .map(|arm| {
                    let mut values: Vec<u64> = (0..schedule.len())
                        .filter(|&pos| {
                            own.regions()
                                .common_ancestor(own.region_of(pos).0, arm.region.0)
                                == arm.region.0
                        })
                        .map(|pos| schedule[pos].value.0)
                        .collect();
                    values.sort_unstable();
                    (arm.arm, values)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(by_value(&ops), by_value(&swapped));
    }
}
