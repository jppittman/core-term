//! The sweep order `all_rules()` ships in, and where it came from.
//!
//! `EGraph::with_rules` sweeps its rule `Vec` in order — every saturation
//! round applies rule 0, then rule 1, ... — so which rule fires *first* when
//! several match the same e-class decides what a budget-truncated sweep (an
//! iteration cap, the class cap, a wall-clock deadline) has done by the time
//! it stops. That order used to be an accident of module concatenation:
//! `algebra_rules()` then `parity_rules()` then ... then `derivative_rules()`.
//! Measured on 204 real kernels (12 `shader_bench` shaders, the psychedelic
//! kernel, the 623-node packed cell grid, 95 ASCII glyphs at two densities —
//! docs/results/2026-09-01-rule-order-real-kernels.md), that accidental
//! order carried 21.8% median anytime regret at 100 applications and still
//! 5.1% at 12,800, while sweeping the *same 62 rules* numeric-first was
//! best-or-tied at every checkpoint and reached 0% median regret by 12,800.
//! Round 2 v3 (docs/plans/2026-09-01-phase3-round2-registration-v3.md §6b)
//! found the same on synthetic classical expressions, ~86x larger.
//!
//! "Numeric-first" is a static priority: descending TRAIN strict-positive
//! rate from the Guide cold-start run
//! (docs/results/2026-09-01-train-guide-report.md, per-rule table), ties
//! broken by ascending position in the old concatenation order. The rate
//! table is frozen here as [`TRAIN_STRICT_POSITIVE_RATE`] — provenance, not
//! something recomputed at runtime — and the permutation it yields is pinned
//! as [`NUMERIC_FIRST_ORDER`]. Three tests keep the three honest: the
//! permutation re-derives from the table, the table re-derives from the
//! report's markdown, and the table's rule names match the rules actually
//! sitting at those legacy positions.
//!
//! # What changes for anything keyed by `rule_idx`
//!
//! A rule's index in `all_rules()` is its `rule_idx` everywhere provenance
//! records one (`ApplicationRecord::rule_idx`, `UnionEvent::rule_idx`) and in
//! every per-rule table a harness writes out. Those indices now mean
//! [`NUMERIC_FIRST_ORDER`]`[i]`, not `i`, relative to any artifact minted
//! before this reorder: the train-guide report's `idx` column, the
//! 2026-08-30 guide-headroom / scope-delta JSONs, and any Guide checkpoint
//! whose weights are a dense `rule_idx -> ...` array (the linear cold-start
//! checkpoint on `claude/phase3-guide` is one). Such a checkpoint must be
//! retrained or remapped through [`NUMERIC_FIRST_ORDER`] before it is
//! trusted, and its loader must fold [`rule_library_identity`] into whatever
//! identity it checks so the next reorder fails at load instead of silently
//! scoring the wrong rule.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::egraph::rewrite::Rewrite;

/// A rule's TRAIN strict-positive rate as the 2026-09-01 train-guide report
/// measured it — or the explicit fact that the report never mined a
/// candidate for the rule and so measured nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainRate {
    /// `train rate` column of the report's per-rule table.
    Measured(f64),
    /// Absent from the report: no candidate for this rule was ever mined
    /// during label generation. Sorts as 0.0, tied with every measured zero.
    NeverMined,
}

impl TrainRate {
    /// The sort key: a never-mined rule ties with a measured-zero rule.
    #[must_use]
    pub fn sort_key(self) -> f64 {
        match self {
            Self::Measured(rate) => rate,
            Self::NeverMined => 0.0,
        }
    }
}

/// One row of the provenance table: a rule, named and numbered as
/// [`super::production_order_2026_08`] numbers it, and its TRAIN rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepPriority {
    /// Position in [`super::production_order_2026_08`] — the `idx` column of
    /// docs/results/2026-09-01-train-guide-report.md.
    pub legacy_idx: usize,
    /// `Rewrite::name()` of the rule at `legacy_idx`. Not unique (four rules
    /// are named `commutative`), so `legacy_idx` is the key and the name is
    /// a checked annotation.
    pub name: &'static str,
    /// TRAIN strict-positive rate, the sort key.
    pub train_rate: TrainRate,
}

const fn row(legacy_idx: usize, name: &'static str, train_rate: TrainRate) -> SweepPriority {
    SweepPriority {
        legacy_idx,
        name,
        train_rate,
    }
}

use TrainRate::{Measured, NeverMined};

/// TRAIN strict-positive rate per rule, transcribed from the per-rule table
/// of docs/results/2026-09-01-train-guide-report.md (strict-v1 labels,
/// 660,404 TRAIN samples over 223 families), listed in legacy index order so
/// `TRAIN_STRICT_POSITIVE_RATE[i].legacy_idx == i`.
///
/// Frozen: this is the measurement the shipped order was derived from, not a
/// live statistic. Re-measuring produces a *new* table and a *new* pinned
/// order, reviewed as such.
#[rustfmt::skip]
pub const TRAIN_STRICT_POSITIVE_RATE: [SweepPriority; 62] = [
    row( 0, "canonicalize",           Measured(0.01636)),
    row( 1, "involution",             Measured(0.00000)),
    row( 2, "cancellation",           Measured(0.00000)),
    row( 3, "inverse-annihilation",   Measured(0.01477)),
    row( 4, "canonicalize",           Measured(0.00000)),
    row( 5, "involution",             NeverMined),
    row( 6, "cancellation",           Measured(0.00000)),
    row( 7, "inverse-annihilation",   Measured(0.00000)),
    row( 8, "constant-fold",          Measured(0.01860)),
    row( 9, "commutative",            Measured(0.00000)),
    row(10, "commutative",            Measured(0.00000)),
    row(11, "commutative",            Measured(0.00000)),
    row(12, "commutative",            Measured(0.00000)),
    row(13, "identity",               Measured(0.00000)),
    row(14, "identity",               Measured(0.00000)),
    row(15, "annihilator",            Measured(0.00000)),
    row(16, "idempotent",             Measured(0.00000)),
    row(17, "idempotent",             Measured(0.00000)),
    row(18, "distribute",             Measured(0.00000)),
    row(19, "factor",                 Measured(0.00231)),
    row(20, "doubling",               Measured(0.00499)),
    row(21, "halving",                Measured(0.00009)),
    row(22, "associative",            Measured(0.00033)),
    row(23, "associative",            Measured(0.00055)),
    row(24, "associative",            Measured(0.00000)),
    row(25, "associative",            Measured(0.00379)),
    row(26, "reverse-associative",    Measured(0.00014)),
    row(27, "reverse-associative",    Measured(0.00078)),
    row(28, "reverse-associative",    Measured(0.00000)),
    row(29, "reverse-associative",    Measured(0.00000)),
    row(30, "odd-negation",           Measured(0.01408)),
    row(31, "odd-negation",           Measured(0.00324)),
    row(32, "odd-negation",           NeverMined),
    row(33, "odd-negation",           NeverMined),
    row(34, "even-negation",          Measured(0.05533)),
    row(35, "even-negation",          Measured(0.05557)),
    row(36, "sin-angle-addition",     Measured(0.00262)),
    row(37, "cos-angle-addition",     Measured(0.00882)),
    row(38, "reverse-angle-addition", Measured(0.01066)),
    row(39, "half-angle-product",     Measured(0.00452)),
    row(40, "pythagorean",            NeverMined),
    row(41, "exp-ln-cancel",          Measured(0.00000)),
    row(42, "ln-exp-cancel",          Measured(0.00000)),
    row(43, "exp2-log2-cancel",       Measured(0.00000)),
    row(44, "log2-exp2-cancel",       Measured(0.00000)),
    row(45, "exp-homomorphism",       Measured(0.00000)),
    row(46, "ln-homomorphism",        NeverMined),
    row(47, "power-combine",          Measured(0.04278)),
    row(48, "power-zero",             Measured(0.12500)),
    row(49, "power-identity",         Measured(0.00000)),
    row(50, "power-expand-2",         NeverMined),
    row(51, "power-sqrt",             Measured(0.13546)),
    row(52, "power-recip",            Measured(0.16080)),
    row(53, "power-rsqrt",            Measured(0.18105)),
    row(54, "power-recurrence",       NeverMined),
    row(55, "log-power",              NeverMined),
    row(56, "log2-power",             Measured(0.00000)),
    row(57, "expand-square",          NeverMined),
    row(58, "diff-of-squares",        Measured(0.00000)),
    row(59, "fma-fusion",             Measured(0.00407)),
    row(60, "recip-sqrt",             Measured(0.16464)),
    row(61, "differentiate",          NeverMined),
];

/// The order `all_rules()` ships in, as a permutation of
/// [`super::production_order_2026_08`]: `all_rules()[i]` is legacy rule
/// `NUMERIC_FIRST_ORDER[i]`. Descending [`TRAIN_STRICT_POSITIVE_RATE`], ties
/// by ascending legacy index; pinned rather than sorted at runtime so the
/// shipped order is a reviewable literal, and re-derived by
/// `numeric_first_order_is_the_table_sorted` so the literal cannot drift
/// from the table it claims to come from.
#[rustfmt::skip]
pub const NUMERIC_FIRST_ORDER: [usize; 62] = [
    53, 60, 52, 51, 48, 35, 34, 47, 8, 0,
    3, 30, 38, 37, 20, 39, 59, 25, 31, 36,
    19, 27, 23, 22, 26, 21, 1, 2, 4, 5,
    6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 24, 28, 29, 32, 33, 40, 41, 42,
    43, 44, 45, 46, 49, 50, 54, 55, 56, 57,
    58, 61,
];

/// Sort the legacy indices by descending TRAIN rate, ties by ascending
/// index — the derivation [`NUMERIC_FIRST_ORDER`] pins.
#[must_use]
pub fn numeric_first_permutation() -> [usize; 62] {
    let mut order: [usize; 62] = core::array::from_fn(|i| i);
    order.sort_by(|&a, &b| {
        let ra = TRAIN_STRICT_POSITIVE_RATE[a].train_rate.sort_key();
        let rb = TRAIN_STRICT_POSITIVE_RATE[b].train_rate.sort_key();
        rb.partial_cmp(&ra)
            .expect("TRAIN rates are finite")
            .then(a.cmp(&b))
    });
    order
}

/// Reorder `rules` so position `i` holds `rules[order[i]]`. Panics if
/// `order` is not a permutation of `0..rules.len()` — a rule visited twice
/// or never is a broken table, never something to paper over.
#[must_use]
pub(crate) fn permute(rules: Vec<Box<dyn Rewrite>>, order: &[usize]) -> Vec<Box<dyn Rewrite>> {
    assert_eq!(
        rules.len(),
        order.len(),
        "rule order permutation covers {} positions but the rule set has {}",
        order.len(),
        rules.len()
    );
    let mut slots: Vec<Option<Box<dyn Rewrite>>> = rules.into_iter().map(Some).collect();
    order
        .iter()
        .map(|&legacy_idx| {
            slots
                .get_mut(legacy_idx)
                .unwrap_or_else(|| {
                    panic!("rule order names legacy index {legacy_idx}, out of range")
                })
                .take()
                .unwrap_or_else(|| panic!("rule order visits legacy index {legacy_idx} twice"))
        })
        .collect()
}

/// A content identity for the rule library *as ordered*: FNV-1a 64 over
/// `(position, legacy index, name)` for every rule of `all_rules()`, plus
/// the count. Two same-named rules (`commutative` x4) are told apart by
/// their legacy index, so swapping them changes the identity too.
///
/// Anything that persists a `rule_idx` — a Guide checkpoint with a dense
/// per-rule weight array, a per-rule results table meant to be re-read by
/// code — should store this and refuse to load under a different value:
/// a reorder is a renumbering, and a renumbering read through the old
/// numbering scores the wrong rule without a single error.
#[must_use]
pub fn rule_library_identity() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let rules = super::all_rules();
    assert_eq!(rules.len(), NUMERIC_FIRST_ORDER.len());
    let mut hash = FNV_OFFSET;
    let mut fold = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    fold(&(rules.len() as u32).to_le_bytes());
    for (position, rule) in rules.iter().enumerate() {
        fold(&(position as u32).to_le_bytes());
        fold(&(NUMERIC_FIRST_ORDER[position] as u32).to_le_bytes());
        let name = rule.name().as_bytes();
        fold(&(name.len() as u32).to_le_bytes());
        fold(name);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{all_rules, production_order_2026_08};
    use std::collections::HashSet;

    fn names(rules: &[Box<dyn Rewrite>]) -> Vec<String> {
        rules.iter().map(|r| r.name().to_string()).collect()
    }

    /// The table is keyed by legacy index and annotated with the rule's
    /// name; both must agree with the rules that actually sit at those
    /// positions, or the provenance is fiction.
    #[test]
    fn table_names_the_rules_at_their_legacy_positions() {
        let legacy = production_order_2026_08();
        assert_eq!(legacy.len(), TRAIN_STRICT_POSITIVE_RATE.len());
        for (i, (rule, entry)) in legacy
            .iter()
            .zip(TRAIN_STRICT_POSITIVE_RATE.iter())
            .enumerate()
        {
            assert_eq!(
                entry.legacy_idx, i,
                "table row {i} carries legacy_idx {}",
                entry.legacy_idx
            );
            assert_eq!(
                entry.name,
                rule.name(),
                "legacy rule {i} is `{}` but the table calls it `{}`",
                rule.name(),
                entry.name
            );
        }
    }

    /// The pinned permutation is exactly what sorting the table yields.
    #[test]
    fn numeric_first_order_is_the_table_sorted() {
        assert_eq!(numeric_first_permutation(), NUMERIC_FIRST_ORDER);
    }

    /// `all_rules()` is the legacy order permuted by the pinned constant —
    /// position `i` holds the legacy rule `NUMERIC_FIRST_ORDER[i]`.
    #[test]
    fn all_rules_ships_numeric_first() {
        let shipped = names(&all_rules());
        let legacy = names(&production_order_2026_08());
        for (position, &legacy_idx) in NUMERIC_FIRST_ORDER.iter().enumerate() {
            assert_eq!(
                shipped[position], legacy[legacy_idx],
                "all_rules()[{position}] should be legacy rule {legacy_idx}"
            );
        }
        assert_eq!(shipped[0], "power-rsqrt");
        assert_eq!(shipped[61], "differentiate");
    }

    /// The reorder is a permutation: same 62 rules, same names, nothing
    /// added, dropped, or duplicated.
    #[test]
    fn the_rule_set_is_unchanged() {
        let mut shipped = names(&all_rules());
        let mut legacy = names(&production_order_2026_08());
        assert_eq!(shipped.len(), 62);
        assert_eq!(legacy.len(), 62);
        shipped.sort_unstable();
        legacy.sort_unstable();
        assert_eq!(shipped, legacy);

        let visited: HashSet<usize> = NUMERIC_FIRST_ORDER.iter().copied().collect();
        assert_eq!(
            visited.len(),
            62,
            "NUMERIC_FIRST_ORDER must visit every legacy index once"
        );
        assert_eq!(visited.iter().max(), Some(&61));
    }

    /// `permute` refuses a non-permutation instead of silently dropping or
    /// duplicating a rule.
    #[test]
    #[should_panic(expected = "visits legacy index 0 twice")]
    fn permute_rejects_a_repeated_index() {
        let mut order = NUMERIC_FIRST_ORDER;
        order[1] = 0;
        let _ = permute(production_order_2026_08(), &order);
    }

    /// The identity is a pinned literal so a reorder, a rename, or a rule
    /// added anywhere in the library changes a value a checkpoint loader
    /// can compare against. Update this literal deliberately, in the same
    /// commit as the change that moved it, and retrain or remap every
    /// `rule_idx`-keyed artifact in that commit's description.
    #[test]
    fn rule_library_identity_is_pinned() {
        assert_eq!(
            rule_library_identity(),
            RULE_LIBRARY_IDENTITY_2026_09_01,
            "rule library identity changed: the rule set or its order moved — \
             every rule_idx-keyed checkpoint is now stale"
        );
    }

    const RULE_LIBRARY_IDENTITY_2026_09_01: u64 = 0x44d7_e351_168f_91cf;

    /// Re-derive [`TRAIN_STRICT_POSITIVE_RATE`] from the report's markdown
    /// per-rule table (descending "train rate", ties by ascending "idx"; a
    /// rule absent from the table is `NeverMined`) and assert the frozen
    /// table matches it row for row. A hand-transcription slip is a test
    /// failure here, not a silently different shipped order.
    #[test]
    fn table_is_the_train_guide_report() {
        let report = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../docs/results/2026-09-01-train-guide-report.md"
        ))
        .expect("read docs/results/2026-09-01-train-guide-report.md");

        let header = "| rule | idx | train fired | train rate | DEV fired | DEV measured rate | DEV mean predicted |";
        let body = report
            .split_once(header)
            .expect("train-guide report's per-rule table header changed — update this test")
            .1
            .trim_start();

        let mut from_report: [Option<(String, f64)>; 62] = core::array::from_fn(|_| None);
        for line in body.lines() {
            if line.trim().is_empty() {
                break;
            }
            if line.starts_with("|---") {
                continue;
            }
            let cols: Vec<&str> = line.split('|').map(str::trim).collect();
            assert!(cols.len() >= 5, "malformed per-rule row: {line}");
            let name = cols[1].to_string();
            let idx: usize = cols[2]
                .parse()
                .unwrap_or_else(|e| panic!("idx column: {e}: {line}"));
            let rate: f64 = cols[4]
                .parse()
                .unwrap_or_else(|e| panic!("train rate column: {e}: {line}"));
            assert!(idx < 62, "report idx {idx} out of the base-62 range");
            assert!(
                from_report[idx].replace((name, rate)).is_none(),
                "report lists idx {idx} twice"
            );
        }

        let mined = from_report.iter().flatten().count();
        assert_eq!(mined, 52, "the report mined a candidate for 52 of 62 rules");

        for (i, entry) in TRAIN_STRICT_POSITIVE_RATE.iter().enumerate() {
            match (&from_report[i], entry.train_rate) {
                (Some((name, rate)), Measured(table_rate)) => {
                    assert_eq!(
                        name, entry.name,
                        "idx {i}: report names `{name}`, table `{}`",
                        entry.name
                    );
                    assert!(
                        (rate - table_rate).abs() < 1e-9,
                        "idx {i} ({name}): report rate {rate}, table rate {table_rate}"
                    );
                }
                (None, NeverMined) => {}
                (Some((name, rate)), NeverMined) => {
                    panic!(
                        "idx {i} ({name}) is NeverMined in the table but the report measured {rate}"
                    )
                }
                (None, Measured(table_rate)) => {
                    panic!("idx {i} is absent from the report but the table claims {table_rate}")
                }
            }
        }
    }
}
