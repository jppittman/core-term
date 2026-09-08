//! Deterministic, learning-free ordering diversity for the R2G trajectory
//! mint (`docs/plans/2026-09-01-guide-return-to-go.md` §2: "without variation
//! in return across trajectories of the same expression there is no credit
//! signal"). [`UniformRandomGuide`] is the one new [`SaturationGuide`] this
//! module adds; it exists so a caller minting several independent orderings
//! of the same expression (the `random:<seed>` policies in that design's
//! per-expression trajectory set) can get one deterministically-reproducible
//! pseudo-random ordering per seed without any learned weights, any RNG
//! state, or any interior mutability — a `GuidedSaturation` episode scores
//! the SAME candidate batch identically on every call by construction
//! (`SaturationGuide::score_candidates(&self, ...)` takes `&self`), so this
//! type keeps that contract exactly like [`super::linear::LinearCandidateGuide`]
//! and [`super::linear::PerRuleRateGuide`] do.
//!
//! # Why a hash, not a PRNG
//!
//! A conventional PRNG (`rand::SmallRng` etc.) needs mutable state advanced
//! call-to-call, which would make two `score_candidates` calls over the same
//! candidates in the same round order-dependent on how many candidates a
//! PRIOR round consumed — a live guided loop's dedup/resume semantics
//! (`GuidedSaturation::until_applications`, checkpoint-resumable) depend on
//! re-scoring being stable. Hashing `seed` together with the candidate's own
//! scalar fields sidesteps that: the score is a pure function of `(seed,
//! candidate)`, so it is automatically stable across repeated/resumed calls
//! and trivially reproducible from the seed alone (no state to serialize or
//! replay).

use crate::egraph::REGISTERED_PRIMARY_BUDGET_APPLICATIONS;

use super::{CandidateSummary, SaturationGuide};

/// FNV-1a, 64-bit — a small, dependency-free, well-distributed non-crypto
/// hash; adequate here because the only property this module needs is
/// "looks uniform, is deterministic," not collision resistance.
pub(super) fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// SplitMix64's output mixer — cheap, well-studied avalanche for turning one
/// `u64` (here, `seed ^ fingerprint`) into a second `u64` that does not
/// share the first's low-order-bit structure (a raw `fnv1a64` output alone
/// is fine for hashing but was not designed as its own bit-mixer).
pub(super) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic fingerprint of everything [`CandidateSummary`] tells a
/// Guide about one candidate — deliberately every scalar field EXCEPT
/// `rule_embed` (an NNUE-tower-specific encoding this hash-based Guide has
/// no business reading, matching [`super::linear::LinearCandidateGuide`]'s
/// own choice to ignore it): the rule's stable [`RuleId`], `budget_fraction`'s bit pattern,
/// `match_class_node_count`, `expr_node_count`, and the neighborhood ops in
/// order (so two candidates differing only in which ops border the match
/// still fingerprint differently, matching the mask-architecture discipline
/// that a real feature must actually move the score).
fn candidate_fingerprint(c: &CandidateSummary) -> u64 {
    let mut bytes: Vec<u8> = Vec::with_capacity(24 + c.neighborhood_ops.len());
    bytes.extend_from_slice(&c.rule.get().to_le_bytes());
    bytes.extend_from_slice(&c.budget_fraction.to_bits().to_le_bytes());
    bytes.extend_from_slice(&(c.match_class_node_count as u64).to_le_bytes());
    bytes.extend_from_slice(&(c.expr_node_count as u64).to_le_bytes());
    for op in &c.neighborhood_ops {
        bytes.push(*op as u8);
    }
    fnv1a64(&bytes)
}

/// One candidate's deterministic pseudo-random score, `seed` mixed with the
/// candidate's own fingerprint (see [`candidate_fingerprint`]) via
/// [`splitmix64`], normalized to `[0.0, 1.0)`. Exposed at module scope (not
/// just via [`UniformRandomGuide`]) so `pixelflow-pipeline`'s trajectory
/// minter can reuse the identical uniform term for its `EpsilonMix` ordering
/// policy (mixing a NAMED base Guide's rank with this same hash-uniform)
/// without a second, drifting implementation of "what does a uniform
/// candidate score mean here" — the anti-drift discipline this crate's own
/// doctrine states plainly: a copy is a future divergence.
#[must_use]
pub fn uniform_unit_score(seed: u64, c: &CandidateSummary) -> f32 {
    let mixed = splitmix64(seed ^ candidate_fingerprint(c));
    // `u64::MAX` as the divisor keeps the range within `[0.0, 1.0]` inclusive
    // of neither bound exactly at the extremes mattering — `RankMixGuide`-
    // style callers only ever compare these scores to each other, never to
    // an exact 0.0/1.0 sentinel.
    (mixed as f64 / u64::MAX as f64) as f32
}

/// A deterministic, learning-free ordering: every candidate's score is a
/// hash of `seed` and the candidate's own scalar fields (see
/// [`uniform_unit_score`]) — no rule identity, no budget state actually
/// *used* for a value judgment, just a fixed pseudo-random preference over
/// candidate shapes. Two `UniformRandomGuide`s with different `seed`s over
/// the SAME e-graph produce two structurally different application orders;
/// the same `seed` reproduces the same order every time (needed by
/// `GuidedSaturation`'s checkpoint-resume semantics — see module doc).
///
/// This is the diversity half of the R2G trajectory mint's "ordering
/// diversity is the dataset" design (`docs/plans/2026-09-01-guide-return-to-go.md`
/// §2): without SOME arms whose ordering has nothing to do with rule
/// identity or the trained linear Guide, every trajectory of a given
/// expression samples nearby points in the SAME ordering family and the
/// per-expression return spread that carries the credit signal collapses.
#[derive(Clone, Copy, Debug)]
pub struct UniformRandomGuide {
    pub seed: u64,
}

impl SaturationGuide for UniformRandomGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        candidates
            .iter()
            .map(|c| uniform_unit_score(self.seed, c))
            .collect()
    }
}

/// The registered primary budget tier, re-exported here purely so a caller
/// that only imports this module (e.g. a unit test constructing a
/// `CandidateSummary` by hand) does not also need to reach into
/// `crate::egraph` for a constant this module's own doc already discusses —
/// **not** used by any scoring logic above, which is budget-agnostic by
/// design (a fixed pseudo-random preference over candidate SHAPES, not a
/// function of how far through the budget the episode is).
#[allow(dead_code)]
const _DOC_ONLY_BUDGET_REFERENCE: usize = REGISTERED_PRIMARY_BUDGET_APPLICATIONS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::factored::EMBED_DIM;
    use pixelflow_ir::OpKind;

    fn candidate(
        rule_slot: usize,
        budget_fraction: f32,
        match_class_node_count: usize,
        expr_node_count: usize,
        ops: Vec<OpKind>,
    ) -> CandidateSummary {
        CandidateSummary {
            rule_embed: [0.0; EMBED_DIM],
            neighborhood_ops: ops,
            budget_fraction,
            rule: crate::egraph::RuleSet::production()
                .id_of(rule_slot)
                .expect("test rule slot within the production set"),
            match_class_node_count,
            expr_node_count,
        }
    }

    #[test]
    fn same_seed_and_candidate_always_scores_identically() {
        let c = candidate(3, 0.4, 2, 10, vec![OpKind::Add, OpKind::Mul]);
        let guide = UniformRandomGuide { seed: 7 };
        let a = guide.score_candidates(&[c.clone_for_test()])[0];
        let b = guide.score_candidates(&[c.clone_for_test()])[0];
        assert_eq!(a, b, "scoring is a pure function of (seed, candidate)");
    }

    #[test]
    fn different_seeds_should_usually_disagree() {
        let c = candidate(3, 0.4, 2, 10, vec![OpKind::Add, OpKind::Mul]);
        let a = UniformRandomGuide { seed: 1 }.score_candidates(&[c.clone_for_test()])[0];
        let b = UniformRandomGuide { seed: 2 }.score_candidates(&[c.clone_for_test()])[0];
        assert!(
            (a - b).abs() > 1e-6,
            "two different seeds landing on the exact same score would be a one-in-4-billion \
             coincidence for a well-mixed hash — treat it as a mixing bug: a={a} b={b}"
        );
    }

    #[test]
    fn different_candidates_should_usually_disagree_under_the_same_seed() {
        let a = candidate(0, 0.1, 1, 5, vec![OpKind::Add]);
        let b = candidate(1, 0.9, 9, 50, vec![OpKind::Sqrt, OpKind::Sqrt]);
        let guide = UniformRandomGuide { seed: 42 };
        let scores = guide.score_candidates(&[a, b]);
        assert!((scores[0] - scores[1]).abs() > 1e-6);
    }

    #[test]
    fn scores_are_finite_and_within_unit_range() {
        let c = candidate(5, 1.7, 30, 300, vec![]);
        let guide = UniformRandomGuide { seed: u64::MAX };
        let s = guide.score_candidates(&[c])[0];
        assert!(s.is_finite());
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn empty_batch_scores_to_empty() {
        let guide = UniformRandomGuide { seed: 0 };
        assert!(guide.score_candidates(&[]).is_empty());
    }

    // `CandidateSummary` has no `Clone` derive (its owning module keeps it
    // minimal — see `mod.rs`'s doc on deliberately small public surface), so
    // this test-only helper builds an independent equal copy rather than
    // adding a `Clone` impl to production code for a test's convenience.
    trait CloneForTest {
        fn clone_for_test(&self) -> CandidateSummary;
    }
    impl CloneForTest for CandidateSummary {
        fn clone_for_test(&self) -> CandidateSummary {
            CandidateSummary {
                rule_embed: self.rule_embed,
                neighborhood_ops: self.neighborhood_ops.clone(),
                budget_fraction: self.budget_fraction,
                rule: self.rule,
                match_class_node_count: self.match_class_node_count,
                expr_node_count: self.expr_node_count,
            }
        }
    }
}
