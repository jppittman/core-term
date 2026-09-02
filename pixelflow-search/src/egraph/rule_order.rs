//! Harness arms for the rule-order measurement
//! (docs/results/2026-09-01-rule-order-real-kernels.md).
//!
//! `all_rules()` is a `Vec`, and `EGraph::with_rules` sweeps it in order —
//! every saturation round applies rule 0, then rule 1, ..., then rule 61.
//! Which rule fires *first* inside a round, when several match the same
//! e-class, changes what an early-terminated sweep (an iteration cap, a
//! class cap, a wall-clock deadline) has done by the time it stops. That
//! measurement is why `all_rules()` now ships numeric-first
//! (`crate::math::rule_order`); this module is the harness-scale piece that
//! lets `crate::runtime`'s `rule_order_bench` keep comparing the shipped
//! order against the legacy order it replaced and against seeded shuffles.
//!
//! **Harness-only.** Nothing here changes `all_rules()` or any production
//! call — [`RuleOrder::Production`] returns `all_rules()` verbatim, and every
//! other variant is reached only by a caller that explicitly asks for it.

use super::rewrite::Rewrite;
use crate::math::{NUMERIC_FIRST_ORDER, all_rules, production_order_2026_08, rule_order::permute};

/// Which order to sweep the base-62 rule set in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleOrder {
    /// `all_rules()` verbatim — what ships today (numeric-first).
    Production,
    /// The module-concatenation order `all_rules()` shipped in until
    /// 2026-09-01 (`crate::math::production_order_2026_08`) — the
    /// "production" arm of docs/results/2026-09-01-rule-order-real-kernels.md.
    Legacy2026_08,
    /// [`NUMERIC_FIRST_ORDER`] applied to [`RuleOrder::Legacy2026_08`]:
    /// descending TRAIN strict-positive rate, ties broken by ascending
    /// legacy index. Equal to [`RuleOrder::Production`] since the reorder
    /// shipped; kept as its own arm so the harness states that equality
    /// rather than assuming it.
    NumericFirst,
    /// A Fisher-Yates shuffle of the legacy order, seeded.
    Shuffled(u64),
}

impl core::fmt::Display for RuleOrder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Production => write!(f, "production"),
            Self::Legacy2026_08 => write!(f, "legacy-2026-08"),
            Self::NumericFirst => write!(f, "numeric-first"),
            Self::Shuffled(seed) => write!(f, "shuffled({seed})"),
        }
    }
}

/// Build the base-62 rule set in `order`. Always the same 62 rules
/// (`all_rules()`'s multiset) — only the sequence changes, so a caller
/// comparing arms never has to worry about a rule being present in one arm
/// and absent in another.
#[must_use]
pub fn build_rule_set(order: RuleOrder) -> Vec<Box<dyn Rewrite>> {
    match order {
        RuleOrder::Production => all_rules(),
        RuleOrder::Legacy2026_08 => production_order_2026_08(),
        RuleOrder::NumericFirst => permute(production_order_2026_08(), &NUMERIC_FIRST_ORDER),
        RuleOrder::Shuffled(seed) => {
            let mut rules = production_order_2026_08();
            fisher_yates(&mut rules, seed);
            rules
        }
    }
}

/// Deterministic Fisher-Yates shuffle, seeded by a small xorshift64 PRNG —
/// no external `rand` dependency, and no reliance on any particular crate
/// version's shuffle algorithm (a portability concern for a *pinned*
/// ordering: `docs/plans/2026-09-01-phase3-round2-registration-v3.md` uses
/// the same construction for its `Interleave`/`Shuffled` orders, reproduced
/// here rather than imported since that plan's `inflate.rs` never shipped
/// to `main`).
fn fisher_yates<T>(v: &mut [T], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_u64 = || {
        // xorshift64*
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..v.len()).rev() {
        let j = (next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(rules: &[Box<dyn Rewrite>]) -> Vec<String> {
        rules.iter().map(|r| r.name().to_string()).collect()
    }

    #[test]
    fn production_order_is_all_rules_verbatim() {
        assert_eq!(
            names(&build_rule_set(RuleOrder::Production)),
            names(&all_rules())
        );
    }

    /// The shipped order *is* numeric-first — the harness's two arms
    /// coincide, by construction, until someone re-derives the order.
    #[test]
    fn production_order_is_numeric_first() {
        assert_eq!(
            names(&build_rule_set(RuleOrder::Production)),
            names(&build_rule_set(RuleOrder::NumericFirst))
        );
    }

    #[test]
    fn legacy_order_is_the_module_concatenation() {
        let legacy = names(&build_rule_set(RuleOrder::Legacy2026_08));
        assert_eq!(legacy.len(), 62);
        assert_eq!(legacy[0], "canonicalize");
        assert_eq!(legacy[8], "constant-fold");
        assert_eq!(legacy[59], "fma-fusion");
        assert_eq!(legacy[61], "differentiate");
        assert_ne!(
            legacy,
            names(&all_rules()),
            "the reorder must actually reorder"
        );
    }

    #[test]
    fn every_arm_is_a_reordering_of_the_same_62_rules() {
        let mut reference = names(&all_rules());
        reference.sort_unstable();
        for order in [
            RuleOrder::Production,
            RuleOrder::Legacy2026_08,
            RuleOrder::NumericFirst,
            RuleOrder::Shuffled(1),
            RuleOrder::Shuffled(2),
            RuleOrder::Shuffled(3),
        ] {
            let mut arm = names(&build_rule_set(order));
            assert_eq!(arm.len(), 62, "{order}");
            arm.sort_unstable();
            assert_eq!(
                arm, reference,
                "{order} is not a permutation of all_rules()"
            );
        }
    }

    #[test]
    fn shuffled_orders_differ_by_seed() {
        let n1 = names(&build_rule_set(RuleOrder::Shuffled(1)));
        let n2 = names(&build_rule_set(RuleOrder::Shuffled(2)));
        let n3 = names(&build_rule_set(RuleOrder::Shuffled(3)));
        assert_ne!(
            n1, n2,
            "seeds 1 and 2 must not coincidentally produce the same order"
        );
        assert_ne!(
            n2, n3,
            "seeds 2 and 3 must not coincidentally produce the same order"
        );
    }
}
