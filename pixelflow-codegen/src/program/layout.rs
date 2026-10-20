//! The blocks of a scope: which arms are skipped by a branch, and where each
//! sits among the scope's values.
//!
//! **What gets a block.** An arm region that owns a value of this scope
//! (`ownership`; an `If` nested in it is one). There is no price: an arm is
//! skipped when no lane selects it, however little it holds. Where its mask
//! was computed does not matter, since selection reads the mask from whichever
//! enclosing scope defined it. An arm with nothing of its own has no block.
//!
//! **Where a block goes.** Every value stays in the order it was constructed,
//! within the block that owns it. An `If`'s arms that own work go immediately
//! before the `If`, the true arm first. That is always after everything an arm reads:
//! an arm-owned value is read only inside its arm, by a loop whose `Reduce` the
//! arm holds, or by its `If`, so what it reads from the region around it was
//! constructed before it, hence before the `If`; and the mask is read by the
//! `If`. The same holds one level up for
//! anything the arm's own region owns, since its block goes before its own
//! `If`.
//!
//! Selection walks the [`Flow`]; the schedule itself is never reordered.

use alloc::vec::Vec;

use crate::error::CompileError;
use crate::pass::Pass;
use crate::program::ownership::{Arm, Ownership};
use crate::program::{ArmBlock, Def, Flow, Item, ScopedSchedule};

/// Every scope's [`Flow`], from its [`Ownership`]: the last stage before
/// selection, which walks what it makes.
pub(crate) struct Layout;

impl Pass for Layout {
    type Input = ScopedSchedule<Ownership>;
    type Output = ScopedSchedule<Flow>;

    fn run(self, owned: ScopedSchedule<Ownership>) -> Result<ScopedSchedule<Flow>, CompileError> {
        Ok(owned.analyze(|nest, scope| {
            let (schedule, _, own) = nest.at(scope);
            Flow::of(schedule, own)
        }))
    }
}

impl Flow {
    /// The flow of `schedule`, from the ownership of its values.
    fn of(schedule: &[Def], own: &Ownership) -> Self {
        let mut owns = alloc::vec![false; own.regions().len()];
        for pos in 0..schedule.len() {
            owns[own.region_of(pos).0] = true;
        }
        let earning: Vec<&Arm> = own.arms().iter().filter(|arm| owns[arm.region.0]).collect();
        // The items of each region's block: the scope's is block 0, and an arm
        // that owns work is one past its index in `earning`.
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
