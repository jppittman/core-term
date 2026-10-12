//! The blocks of a scope: which arms are skipped by a branch, and where each
//! sits among the scope's values.
//!
//! **What earns a block.** An arm region that owns a value (`ownership`; an
//! `If` nested in it is one). There is no price: an arm is skipped when no lane
//! selects it, however little it holds, and whatever scope computes its mask.
//! An arm with nothing of its own has no block.
//!
//! **Where a block goes.** Every value stays in the order it was constructed,
//! within the block that owns it. An `If`'s earning arms go immediately before
//! the `If`, the true arm first. That is always after everything an arm reads:
//! an arm-owned value is read only inside its arm, by a loop whose `Reduce` the
//! arm holds, or by its `If`, so what it reads from the region around it was
//! constructed before it, hence before the `If`; and the mask is read by the
//! `If`. The same holds one level up for
//! anything the arm's own region owns, since its block goes before its own
//! `If`.
//!
//! Selection walks the [`Flow`]; the schedule itself is never reordered.

use alloc::vec::Vec;

use crate::program::guards::FoldReads;
use crate::program::ownership::{Arm, Ownership};
use crate::program::{ArmBlock, Def, Flow, Item};

impl Flow {
    /// The flow of `schedule`, from the ownership of its values.
    ///
    /// `folds` is as for [`Ownership::of`].
    pub(super) fn of(schedule: &[Def], folds: &FoldReads) -> Self {
        let own = Ownership::of(schedule, folds);
        let mut owns = alloc::vec![false; own.regions().len()];
        for pos in 0..schedule.len() {
            owns[own.region_of(pos).0] = true;
        }
        let earning: Vec<&Arm> = own.arms().iter().filter(|arm| owns[arm.region.0]).collect();
        // The items of each region's block: the scope's is block 0, and an
        // earning arm's is one past its index in `earning`.
        let mut block = alloc::vec![0; owns.len()];
        for (i, arm) in earning.iter().enumerate() {
            block[arm.region.0] = i + 1;
        }
        let mut items: Vec<Vec<Item>> = alloc::vec![Vec::new(); earning.len() + 1];

        // The arms are in `If` order, true arm first, so they are met in the
        // order they are spliced.
        let mut pending = own.arms().iter().peekable();
        for pos in 0..schedule.len() {
            let here = block[own.region_of(pos).0];
            while let Some(arm) = pending.next_if(|arm| arm.at == pos) {
                if owns[arm.region.0] {
                    items[here].push(Item::Arm(block[arm.region.0] - 1));
                }
            }
            items[here].push(Item::Def(pos));
        }
        debug_assert!(pending.next().is_none(), "every arm is spliced at its If");

        let mut items = items.into_iter();
        let top = items.next().expect("the scope is block 0");
        let arms = earning
            .into_iter()
            .zip(items)
            .map(|(arm, items)| ArmBlock {
                arm: arm.arm,
                mask: arm.mask,
                items,
            })
            .collect();
        Self { items: top, arms }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::ownership::Positions;
    use crate::program::{IfArm, ScheduledOp, ValueId};
    use pixelflow_ir::kind::OpKind;

    /// The positions of the defs under `items`, nested arms included, in the
    /// order they run.
    fn run(flow: &Flow, items: &[Item], out: &mut Vec<usize>) {
        for item in items {
            match *item {
                Item::Def(at) => out.push(at),
                Item::Arm(inner) => run(flow, &flow.arms[inner].items, out),
            }
        }
    }

    /// The positions of the defs under `arm`, in the order they run.
    fn under(flow: &Flow, arm: usize) -> Vec<usize> {
        let mut out = Vec::new();
        run(flow, &flow.arms[arm].items, &mut out);
        out
    }

    /// Whether, taking `schedule`'s defs in `order`, every read follows the value
    /// it reads. A value this schedule does not define (a live-in) is not read
    /// from here.
    fn reads_follow(order: &[usize], schedule: &[Def], folds: &FoldReads) -> bool {
        let positions = Positions::of(schedule);
        let mut at = alloc::vec![0usize; schedule.len()];
        for (new, &old) in order.iter().enumerate() {
            at[old] = new;
        }
        order.iter().enumerate().all(|(new, &old)| {
            let def = &schedule[old];
            folds
                .reads(def.value, &def.op)
                .filter_map(|read| positions.get(read))
                .all(|source| at[source] < new)
        })
    }

    fn def(value: u64, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
        }
    }

    fn if_of(value: u64, mask: u64, if_true: u64, if_false: u64) -> Def {
        def(
            value,
            ScheduledOp::Ternary(
                OpKind::If,
                ValueId(mask),
                ValueId(if_true),
                ValueId(if_false),
            ),
        )
    }

    fn unary(value: u64, op: OpKind, of: u64) -> Def {
        def(value, ScheduledOp::Unary(op, ValueId(of)))
    }

    /// The flow of `schedule` and the values in the order it runs them, after
    /// every property a flow owes has been checked: every read follows its
    /// value; each arm it branches over is exactly the run of the values that
    /// arm owns — no stranger inside it, nothing the arm owns outside it; and
    /// an `If`'s earning arms run immediately before it, the true arm first.
    fn laid_out(schedule: &[Def]) -> (Flow, Vec<u64>) {
        laid_out_with(schedule, &FoldReads::default())
    }

    fn laid_out_with(schedule: &[Def], folds: &FoldReads) -> (Flow, Vec<u64>) {
        let flow = Flow::of(schedule, folds);
        let mut order = Vec::new();
        run(&flow, &flow.items, &mut order);
        assert_eq!(order.len(), schedule.len(), "a flow runs every def once");
        // A schedule that reads a value before it defines it (a hand-built
        // fixture) has no order to keep.
        let in_order = reads_follow(&(0..schedule.len()).collect::<Vec<_>>(), schedule, folds);
        assert!(
            !in_order || reads_follow(&order, schedule, folds),
            "a read runs before the value it reads: {order:?}"
        );
        let own = Ownership::of(schedule, folds);
        // What an arm owns, by position.
        let owned = |arm: &Arm| -> Vec<usize> {
            (0..schedule.len())
                .filter(|&pos| {
                    own.regions()
                        .common_ancestor(own.region_of(pos).0, arm.region.0)
                        == arm.region.0
                })
                .collect()
        };
        let earning = own.arms().iter().filter(|arm| !owned(arm).is_empty());
        assert_eq!(
            flow.arms.len(),
            earning.clone().count(),
            "an arm that owns a value is a block"
        );
        for (i, (arm, block)) in earning.zip(&flow.arms).enumerate() {
            let mut run = under(&flow, i);
            assert_eq!((block.arm, block.mask), (arm.arm, arm.mask));
            run.sort_unstable();
            assert_eq!(
                owned(arm),
                run,
                "the {:?} arm is not exactly what it owns",
                block.arm
            );
        }
        // Immediately before each `If`: the run of what its true arm owns, then
        // of what its false arm owns.
        for (at, def) in schedule.iter().enumerate() {
            if !matches!(def.op, ScheduledOp::Ternary(OpKind::If, ..)) {
                continue;
            }
            let arms: Vec<Vec<usize>> = own
                .arms()
                .iter()
                .filter(|arm| arm.at == at)
                .map(&owned)
                .collect();
            let end = order
                .iter()
                .position(|&pos| pos == at)
                .expect("every def runs");
            let mut start = end - arms.iter().map(Vec::len).sum::<usize>();
            for owned in arms {
                let mut ran = order[start..start + owned.len()].to_vec();
                ran.sort_unstable();
                assert_eq!(ran, owned, "an arm of the If at {at} is not just before it");
                start += owned.len();
            }
        }
        let values = order.iter().map(|&pos| schedule[pos].value.0).collect();
        (flow, values)
    }

    /// An arm's values run immediately before the `If`, the mask's cone staying
    /// where it was.
    #[test]
    fn an_arm_runs_immediately_before_its_if() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            if_of(3, 0, 2, 0),
        ];
        let (flow, values) = laid_out(&schedule);
        assert_eq!(values, [0, 1, 2, 3]);
        assert_eq!(flow.arms.len(), 1);
        assert_eq!(flow.arms[0].arm, IfArm::True);
        assert_eq!(under(&flow, 0), [1, 2]);
    }

    /// A stranger between the arm's values: the arm's values become one run at
    /// the `If`, and the stranger, which the arm does not read, stays where it
    /// was.
    #[test]
    fn a_stranger_between_an_arms_values_stays_where_it_was() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Sqrt, 1),  // the arm's, first value
            unary(3, OpKind::Neg, 0),   // a stranger: nothing in the arm reads it
            unary(4, OpKind::Rsqrt, 2), // the arm's, second value
            if_of(5, 0, 4, 0),
            def(6, ScheduledOp::Binary(OpKind::Add, ValueId(5), ValueId(3))),
        ];
        let (_, values) = laid_out(&schedule);
        assert_eq!(values, [0, 3, 1, 2, 4, 5, 6]);
    }

    /// Both arms of one `If` run just before it, the true arm first.
    #[test]
    fn both_arms_of_an_if_run_before_it_true_arm_first() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            unary(3, OpKind::Rsqrt, 0),
            unary(4, OpKind::Sqrt, 2),
            unary(5, OpKind::Sqrt, 3),
            if_of(6, 0, 4, 5),
        ];
        let (flow, values) = laid_out(&schedule);
        let arms = &flow.arms;
        assert_eq!((arms[0].arm, arms[1].arm), (IfArm::True, IfArm::False));
        assert_eq!(values, [0, 1, 2, 4, 3, 5, 6]);
    }

    /// An arm nested in an arm: the inner runs before its `If`, inside the outer.
    #[test]
    fn nested_arms_run_inside_the_arm_around_them() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            unary(3, OpKind::Rsqrt, 0),
            unary(4, OpKind::Sqrt, 2),
            if_of(5, 0, 4, 3),
            unary(6, OpKind::Recip, 1),
            if_of(7, 0, 5, 6),
        ];
        let (flow, values) = laid_out(&schedule);
        assert_eq!(values, [0, 1, 2, 4, 3, 5, 6, 7]);
        let holds = |block: &ArmBlock| {
            block
                .items
                .iter()
                .filter_map(|item| match *item {
                    Item::Arm(inner) => Some(inner),
                    Item::Def(_) => None,
                })
                .collect::<Vec<_>>()
        };
        let (outer, inner) = flow
            .arms
            .iter()
            .position(|block| !holds(block).is_empty())
            .map(|outer| (outer, holds(&flow.arms[outer])))
            .expect("an arm holds another");
        assert_eq!(inner.len(), 2, "both of the inner If's arms");
        for inner in inner {
            let (inner, outer) = (under(&flow, inner), under(&flow, outer));
            assert!(
                inner.iter().all(|at| outer.contains(at)),
                "the inner arm's run is inside the outer's: {inner:?} in {outer:?}"
            );
        }
    }

    /// An input the arm shares with the region around it was constructed ahead
    /// of the `If`, so it is ahead of the arm placed there.
    #[test]
    fn a_block_follows_an_input_it_shares_with_the_scope() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Sqrt, 1), // the arm's
            unary(3, OpKind::Neg, 0),  // shared: the arm reads it below
            def(4, ScheduledOp::Binary(OpKind::Add, ValueId(2), ValueId(3))),
            unary(5, OpKind::Rsqrt, 4), // the arm's root
            def(6, ScheduledOp::Binary(OpKind::Add, ValueId(3), ValueId(0))),
            if_of(7, 0, 5, 6),
        ];
        let (_, values) = laid_out(&schedule);
        assert_eq!(values, [0, 3, 1, 2, 4, 5, 6, 7]);
    }

    /// A deterministic stream, so a failure names its seed.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// A random DAG with `Ifs` in it: every value reads earlier ones, an `If`
    /// reads three.
    fn random_schedule(seed: u64, len: usize) -> Vec<Def> {
        let mut rng = Lcg(seed);
        let mut schedule: Vec<Def> =
            alloc::vec![def(0, ScheduledOp::Var(0)), def(1, ScheduledOp::Var(1))];
        while schedule.len() < len {
            let value = schedule.len() as u64;
            let earlier = |rng: &mut Lcg| rng.below(value as usize) as u64;
            let op = match rng.below(10) {
                0..=2 => ScheduledOp::Unary(OpKind::Rsqrt, ValueId(earlier(&mut rng))),
                3..=4 => ScheduledOp::Unary(OpKind::Neg, ValueId(earlier(&mut rng))),
                5..=6 => ScheduledOp::Binary(
                    OpKind::Add,
                    ValueId(earlier(&mut rng)),
                    ValueId(earlier(&mut rng)),
                ),
                _ => ScheduledOp::Ternary(
                    OpKind::If,
                    ValueId(earlier(&mut rng)),
                    ValueId(earlier(&mut rng)),
                    ValueId(earlier(&mut rng)),
                ),
            };
            schedule.push(def(value, op));
        }
        schedule
    }

    /// Whatever the DAG, every read follows its value, and every arm the flow
    /// branches over is exactly the run of the values that arm owns, just
    /// before its `If`.
    #[test]
    fn a_random_dag_is_laid_out_validly() {
        for seed in 0..300 {
            let len = 20 + (seed as usize % 40);
            let schedule = random_schedule(seed, len);
            laid_out_with(&schedule, &FoldReads::default());
        }
    }

    // -- What a loop reads ------------------------------------------------

    /// A four-trip fold; the scope its body was carved into is not what these
    /// tests look at, so the body names a hole.
    fn reduce() -> ScheduledOp {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};
        let fold = Fold::new(
            Monoid::SUM,
            Binder::from_slot(0).expect("slot 0 exists"),
            0..4,
        );
        ScheduledOp::Reduce(fold, ValueId(99))
    }

    fn at(values: &[u64], v: u64) -> usize {
        values
            .iter()
            .position(|&x| x == v)
            .expect("a flow runs every def")
    }

    /// `W` is read by the true arm and by a sibling fold's body, which runs
    /// whatever the mask: the arm may not own `W`. Its `Reduce` def names no
    /// operand, so without the sibling's reads the arm owned it, and a
    /// uniformly-false mask skipped the loop the sibling then read.
    #[test]
    fn an_arm_does_not_own_what_a_sibling_fold_reads() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, reduce()),
            unary(2, OpKind::Rsqrt, 1),
            def(3, ScheduledOp::Binary(OpKind::Mul, ValueId(1), ValueId(2))),
            if_of(4, 0, 3, 0),
            def(5, reduce()),
            def(6, ScheduledOp::Binary(OpKind::Add, ValueId(4), ValueId(5))),
        ];
        // The sibling's body names `W`'s id, which an enclosing scope computed.
        let folds = FoldReads::new(&schedule, [(ValueId(5), [ValueId(1)])]);

        let (blind, _) = laid_out(&schedule);
        let (seen, values) = laid_out_with(&schedule, &folds);
        assert_eq!(under(&blind, 0).len(), 3, "the arm owned `W` unseen");
        assert_eq!(under(&seen, 0).len(), 2, "`W` is not the arm's");
        assert!(
            at(&values, 1) < at(&values, 2),
            "and it stays ahead of the arm"
        );
    }

    /// `s` is read only by `W`'s body, and `W` is in the true arm: `s` is the
    /// arm's, and goes with it, ahead of the loop that reads it.
    #[test]
    fn what_a_fold_reads_stays_ahead_of_the_fold() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(1), ValueId(1))),
            def(4, reduce()),
            def(5, ScheduledOp::Binary(OpKind::Mul, ValueId(2), ValueId(4))),
            if_of(6, 0, 5, 0),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(6), ValueId(1))),
        ];
        let folds = FoldReads::new(&schedule, [(ValueId(4), [ValueId(3)])]);
        let (_, values) = laid_out_with(&schedule, &folds);
        assert!(at(&values, 3) < at(&values, 4), "{values:?}");
    }

    /// A gather's base is a pointer operand, not a vector one, and it is a
    /// read all the same: the `Context` the arm's `Broadcast` addresses
    /// through goes ahead of its reader, not behind it.
    #[test]
    fn a_pointer_stays_ahead_of_its_reader() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(1), ValueId(1))),
            def(4, ScheduledOp::Context(0)),
            def(5, ScheduledOp::Broadcast(ValueId(2), ValueId(4))),
            if_of(6, 0, 5, 0),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(6), ValueId(3))),
        ];
        let (_, values) = laid_out(&schedule);
        assert!(at(&values, 4) < at(&values, 5), "{values:?}");
    }

    /// A value the schedule never defines is not a position: it must not be
    /// mistaken for one, which would corrupt a run or overflow computing its
    /// end.
    #[test]
    fn a_read_of_a_value_the_schedule_never_defines_is_ignored() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            unary(1, OpKind::Rsqrt, 3), // ValueId(3) has no Def
            if_of(4, 0, 0, 1),
        ];
        let (flow, _) = laid_out(&schedule);
        assert_eq!(flow.arms.len(), 1);
        assert_eq!(flow.arms[0].arm, IfArm::False);
        assert_eq!(under(&flow, 0), [1]);
    }
}
