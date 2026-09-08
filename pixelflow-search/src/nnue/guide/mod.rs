//! # Saturation Guide (Phase 3 — wiring in progress)
//!
//! Given a set of candidate rewrite-rule applications, produce per-candidate
//! move-ordering scores to guide equality saturation. That sentence is the
//! entire contract; see `docs/plans/2026-07-07-guided-saturation-redesign.md`
//! for the design this module exists to satisfy once training lands, purely
//! supervised on hindsight rule-provenance labels from `crate::egraph::labeler`
//! — no REINFORCE, no self-play.
//!
//! ## Contract revision, v1 -> v2 (2026-09-01, design doc §4)
//!
//! **Loud note, per that task's explicit instruction: this trait's shape
//! changed.** The original contract scored a candidate against a *whole-graph*
//! summary (`GraphSummary`, wrapping [`GraphAccumulator`]) plus a bare rule
//! embedding (`RuleCandidate`). `docs/plans/2026-08-31-guide-design-revision.md`
//! §4 argues from measurement, not taste, that this is the wrong granularity:
//! the dominant per-round cost is candidate-level deduplication (90.4% of
//! scored candidates commit no change — §2.2), which is a property of *one
//! rule instance at one e-class*, not of the graph as a whole; and the
//! label-semantics gap (§3) is per-rule, not per-graph. [`SaturationGuide`]
//! now scores [`CandidateSummary`] — rule identity + this match's own one-hop
//! e-class neighborhood + budget state, mirroring
//! [`crate::egraph::candidate::CandidateFeatures`] (the same object minted for
//! the strict-label training pipeline, `gen_strict_labels`) field for field —
//! rather than [`GraphSummary`]/[`RuleCandidate`].
//!
//! `GraphAccumulator` itself is **not deleted** (segregation rule,
//! `docs/plans/2026-08-31-guide-design-revision.md` §4 last paragraph: "the
//! accumulator's whole-graph summary remains the right shape for the
//! *extraction-cost* Judge, a genuinely different question from 'should this
//! match fire now'"). It stays exactly where it was (`accumulator.rs`), still
//! exercised by its own characterization tests and by
//! `scoring::SaturationHead::mask_score_all_rules_graph` (also kept, also
//! unused by [`SaturationGuide`] as of this revision) — a live roadmap seam
//! for a future whole-graph Judge, not a Guide-scoring code path. The
//! `GraphSummary`/`RuleCandidate` *wrapper types* that named that path as part
//! of the public [`SaturationGuide`] contract are removed, since nothing
//! outside this module ever constructed them (confirmed by grep before this
//! change) and keeping unused public wrapper types around a live private
//! implementation is exactly the "dozen public methods to pick from" this
//! module's original doc said it wanted to avoid.
//!
//! **This module is still inert in production.** Nothing calls
//! [`SaturationGuide::score_candidates`] outside this module's own tests and
//! `egraph::saturate`'s guided-mode ordering hook's own tests — production
//! saturation (`EGraph::saturate`, `saturate_with_limits`,
//! `saturate_until_applications`) is unguided and untouched by this revision.
//! The public surface below — the trait, [`CandidateSummary`], and [`Guide`]'s
//! constructor — is deliberately the entire contract: every mask/rule-
//! projection/candidate-tower weight and the bilinear scorer are private
//! machinery behind it (`scoring.rs`), so a future Phase-3 trainer has one
//! seam to fill in rather than a dozen public methods to pick from.
//!
//! Kept, not deleted: the 2026-08-17 disposition inventory confirmed zero
//! non-test callers for this whole surface but a live roadmap role
//! (ROADMAP-ADMITTED, `docs/plans/2026-08-17-cost-model-domain.md` §0 row J10).
//! The extraction head this module used to share a backbone with is gone
//! (its shape was deleted 2026-09-01; the denotation is kept, see
//! docs/plans/2026-09-01-schedule-cost-model-denotation.md); what it shares now is only
//! [`OpEmbeddings`], and the trunk that used to be "shared" is this head's
//! own (`scoring.rs`).

mod accumulator;
pub mod bilinear;
pub mod diversity;
pub mod filter;
pub mod linear;
mod scoring;

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use scoring::SaturationHead;

use crate::egraph::candidate::CandidateFeatures;
use crate::egraph::rules::RuleId;
use crate::nnue::factored::{EMBED_DIM, K, OpEmbeddings};
use pixelflow_ir::OpKind;

/// Cyclic rotation by `amount % K` positions (generalised VSA permutation).
///
/// Shared by [`accumulator::GraphAccumulator`]'s VSA bindings. `shift_by(emb,
/// 0)` is the identity; higher amounts encode hierarchical depth so that
/// `Add(Add(X,Y),Z)` and `Add(X,Add(Y,Z))` bind to different vectors.
#[inline]
fn shift_by(emb: &[f32; K], amount: usize) -> [f32; K] {
    let amount = amount % K;
    let mut out = [0.0f32; K];
    for i in 0..K {
        out[i] = emb[(i + amount) % K];
    }
    out
}

/// Cyclic shift by 1 position — the depth-1 case of [`shift_by`], used by
/// [`accumulator::GraphAccumulator`]'s 2-hop binding.
#[inline]
fn shift1(emb: &[f32; K]) -> [f32; K] {
    shift_by(emb, 1)
}

/// One candidate rewrite-rule application, as
/// [`SaturationGuide::score_candidates`] scores it (design doc §4 — see the
/// module doc's "Contract revision" section for what this replaced).
///
/// Deliberately field-for-field the same information as
/// [`crate::egraph::candidate::CandidateFeatures`] plus a pre-encoded rule
/// embedding, rather than that type itself: `CandidateFeatures` lives in
/// `egraph::candidate` because it doubles as the saturation loop's dedup key
/// (see that module's doc), and this crate's rule embeddings are an
/// NNUE-specific encoding (`scoring::SaturationHead::encode_rule`)
/// that `egraph::candidate` has no business knowing about — keeping them
/// separate keeps the dedup-key module free of NNUE machinery it doesn't own.
/// [`CandidateSummary::new`] is the single seam that combines the two, so
/// there is exactly one place a caller can get this wrong.
///
/// # Extension for the linear cold-start Guide (2026-09-01, design doc §5 task 2)
///
/// The linear cold-start Guide deploys the same
/// linear model `pixelflow-pipeline`'s `train_guide` trains — a per-rule
/// bias, a bag-of-neighborhood-ops term, and a handful of scalar features
/// keyed on the rule's identity, the matched class's node count, and the originating
/// episode's expression size, none of which the original (tower-only)
/// contract carried. Rather than have `LinearCandidateGuide` reconstruct
/// those from a second computation (the anti-drift rule this module's own
/// doc states: "there is exactly one place a caller can get this wrong"),
/// `CandidateSummary` carries them too — additively, alongside the fields
/// [`scoring::SaturationHead`]'s tower already consumed. `rule` and
/// `match_class_node_count` come straight off `features.key` (already
/// computed, nothing new to derive); `expr_node_count` has no home in
/// [`CandidateFeatures`] (it's an episode-level constant, not part of any
/// one candidate's dedup key) so [`CandidateSummary::new`] takes it as an
/// explicit parameter from the caller, which already has it (the offline
/// label-minting replay reads it off the arena; the live guided loop reads
/// `EGraph::node_count()` at the episode's start — see
/// `saturate_guided_until_applications`).
pub struct CandidateSummary {
    pub rule_embed: [f32; EMBED_DIM],
    pub neighborhood_ops: Vec<OpKind>,
    pub budget_fraction: f32,
    /// Which rule this candidate is, by stable identity
    /// ([`CandidateKey::rule`](crate::egraph::candidate::CandidateKey::rule))
    /// — consumed by the per-rule linear guides, not by
    /// [`scoring::SaturationHead`]'s tower (which reads only `rule_embed`).
    ///
    /// A [`RuleId`] rather than a position: a per-rule weight vector keyed
    /// by index is repointed wholesale by a same-length reorder of the rule
    /// table, and nothing anywhere is the wrong length. See
    /// [`crate::egraph::rules`].
    pub rule: RuleId,
    /// Node count of the matched e-class's canonical content
    /// ([`ClassContentKey::node_count`]) — `train_guide`'s
    /// `log_match_class` feature, before the `log1p` the linear model
    /// applies.
    pub match_class_node_count: usize,
    /// Node count of the episode's original root expression —
    /// `train_guide`'s `log_expr_size` feature, before `log1p`. Constant
    /// across every candidate scored in one episode; see the struct doc for
    /// why it is a caller-supplied parameter rather than a
    /// [`CandidateFeatures`] field.
    pub expr_node_count: usize,
}

impl CandidateSummary {
    /// Build a `CandidateSummary` from a candidate-local feature (see
    /// [`crate::egraph::candidate::CandidateFeatures::observe`]) plus a
    /// pre-encoded rule embedding and the originating episode's expression
    /// size (see the struct doc's "Extension for the linear cold-start
    /// Guide" section for why the latter is a parameter here rather than a
    /// `CandidateFeatures` field).
    ///
    /// The rule embedding is supplied separately rather than encoded here:
    /// its encoding (`SaturationHead::encode_rule`) is a pure
    /// function of the rule's LHS/RHS side embeddings, independent of any e-graph
    /// state, so a caller scoring many candidates for the same rule computes
    /// it once per rule per episode, not once per candidate.
    #[must_use]
    pub fn new(
        features: &CandidateFeatures,
        rule_embed: [f32; EMBED_DIM],
        expr_node_count: usize,
    ) -> Self {
        Self {
            rule_embed,
            neighborhood_ops: features.neighborhood_ops.clone(),
            budget_fraction: features.budget_fraction(),
            rule: features.key.rule,
            match_class_node_count: features.key.content.node_count(),
            expr_node_count,
        }
    }
}

/// Given a batch of candidate rewrite-rule applications, produce per-candidate
/// move-ordering scores.
///
/// The only public contract of the saturation head. See the module doc for
/// why an implementation of this trait is, today, necessarily untrained, and
/// for the v1->v2 contract revision this signature reflects.
///
/// Scoring is batched deliberately (design doc §5/binding rule: "batch per
/// iteration, no incrementality work") — a caller collects one saturation
/// round's surviving (post-dedup) candidates and scores them in one call,
/// never one candidate at a time.
pub trait SaturationGuide {
    /// Score `candidates`, one score per candidate, in order.
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32>;

    /// This guide's own encoding of the rule named by `rule`, for the
    /// [`CandidateSummary::rule_embed`] field the saturation loop fills in.
    ///
    /// Defaults to all-zero, which is the honest answer for every guide that
    /// does not encode rules as vectors at all — the per-rule linear models
    /// key on [`CandidateSummary::rule`] directly and would otherwise carry
    /// a field of boilerplate zeros they had to write themselves.
    ///
    /// The loop calls this **once per rule per episode**, not once per
    /// candidate: an encoding is a pure function of the rule, and the
    /// alternative (having the caller thread a positional
    /// `rule_embeds: &[[f32; EMBED_DIM]]` array through every entry point)
    /// is the positional-rule-key bug [`crate::egraph::rules`] exists to
    /// prevent, in the one place a Guide would have felt it.
    fn rule_embed(&self, rule: crate::egraph::rules::RuleId) -> [f32; EMBED_DIM] {
        let _ = rule;
        [0.0; EMBED_DIM]
    }
}

/// The (untrained, Phase-3-gated) saturation guide: the op embeddings a
/// candidate's neighborhood is pooled in, plus this head's own trunk and
/// mask/candidate/rule-projection weights.
///
/// Owns its embeddings by value because nothing today keeps a `Guide` alive
/// across embedding updates — Phase 3 training will need to decide how the
/// two stay in sync (the domain model doc's `EmbeddingsEpoch` seam is the
/// shape that answer will likely take; out of scope for this round).
pub struct Guide {
    embeddings: OpEmbeddings,
    head: SaturationHead,
    /// Per-rule encodings, keyed by stable identity, or empty.
    ///
    /// Built by whoever has an expression encoder to hand — the rule-pair
    /// encoder `SaturationHead::encode_rule` needs each side's *embedded*
    /// template, which lives in the training crate, not here. Empty means
    /// [`SaturationGuide::rule_embed`]'s all-zero default, which is what an
    /// untrained guide should say.
    rule_embeds: BTreeMap<RuleId, [f32; EMBED_DIM]>,
}

impl Guide {
    /// Build a guide with a randomly-initialized saturation head over the
    /// given (already-trained-or-not) op embeddings.
    #[must_use]
    pub fn new_random(embeddings: OpEmbeddings, seed: u64) -> Self {
        let mut head = SaturationHead::new();
        head.randomize(seed);
        Self {
            embeddings,
            head,
            rule_embeds: BTreeMap::new(),
        }
    }

    /// Attach per-rule encodings, keyed by stable identity.
    ///
    /// A `BTreeMap` rather than a vector: a positional table is repointed
    /// wholesale by a same-length reorder of the rule vocabulary and nothing
    /// anywhere is the wrong length. A rule absent from the table scores
    /// against the all-zero default rather than against some other rule's
    /// vector.
    #[must_use]
    pub fn with_rule_embeds(mut self, embeds: BTreeMap<RuleId, [f32; EMBED_DIM]>) -> Self {
        self.rule_embeds = embeds;
        self
    }

    /// The op embeddings every candidate scored by this guide is pooled in.
    #[must_use]
    pub fn embeddings(&self) -> &OpEmbeddings {
        &self.embeddings
    }
}

impl SaturationGuide for Guide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        candidates
            .iter()
            .map(|c| self.head.score_candidate(&self.embeddings, c))
            .collect()
    }

    fn rule_embed(&self, rule: RuleId) -> [f32; EMBED_DIM] {
        self.rule_embeds
            .get(&rule)
            .copied()
            .unwrap_or([0.0; EMBED_DIM])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // shift_by / shift1
    // ========================================================================

    /// A K-element ramp `[0.0, 1.0, ..., K-1 as f32]` — every element
    /// distinct, so a rotation, a constant fill, or a mis-indexed lookup all
    /// produce visibly different arrays from the correct answer.
    fn ramp() -> [f32; K] {
        let mut a = [0.0f32; K];
        for (i, v) in a.iter_mut().enumerate() {
            *v = i as f32;
        }
        a
    }

    /// Left-rotation computed independently of [`shift_by`] (slice-style,
    /// not the `(i + amount) % K` formula under test), used as the expected
    /// value in the tests below.
    fn rotate_left(a: &[f32; K], amount: usize) -> [f32; K] {
        let amount = amount % K;
        let mut out = [0.0f32; K];
        out[..K - amount].copy_from_slice(&a[amount..]);
        out[K - amount..].copy_from_slice(&a[..amount]);
        out
    }

    #[test]
    fn shift_by_should_return_the_identity_for_a_zero_amount() {
        let arr = ramp();
        assert_eq!(shift_by(&arr, 0), arr);
    }

    #[test]
    fn shift_by_should_rotate_every_element_left_by_one_for_an_amount_of_one() {
        let arr = ramp();
        assert_eq!(shift_by(&arr, 1), rotate_left(&arr, 1));
    }

    #[test]
    fn shift_by_should_rotate_almost_all_the_way_around_for_k_minus_one() {
        let arr = ramp();
        assert_eq!(shift_by(&arr, K - 1), rotate_left(&arr, K - 1));
    }

    #[test]
    fn shift_by_should_wrap_back_to_the_identity_at_k() {
        let arr = ramp();
        assert_eq!(shift_by(&arr, K), arr);
    }

    #[test]
    fn shift_by_should_match_shift_by_one_at_k_plus_one() {
        let arr = ramp();
        assert_eq!(shift_by(&arr, K + 1), shift_by(&arr, 1));
    }

    #[test]
    fn shift_by_should_not_overflow_near_usize_max_and_should_match_its_residue_mod_k() {
        // `amount % K` never overflows, but a `%` -> `+` mutation of the
        // internal `amount = amount % K` line is arithmetically equivalent
        // to the correct code for every amount that doesn't overflow
        // `amount + K` (both feed into an outer `% K` that discards any
        // added multiple of K). usize::MAX is chosen specifically to make
        // that mutated addition overflow (and panic, under this workspace's
        // debug overflow-checks) where the correct code would not.
        let arr = ramp();
        let want = rotate_left(&arr, usize::MAX % K);
        assert_eq!(shift_by(&arr, usize::MAX), want);
    }

    #[test]
    fn shift1_should_equal_shift_by_one() {
        let arr = ramp();
        assert_eq!(shift1(&arr), shift_by(&arr, 1));
        assert_eq!(shift1(&arr), rotate_left(&arr, 1));
    }

    // ========================================================================
    // Guide::score_candidates (candidate-local contract, v2)
    // ========================================================================

    fn sample_candidate(rule_embed_fill: f32, neighborhood: Vec<OpKind>) -> CandidateSummary {
        CandidateSummary {
            rule_embed: [rule_embed_fill; EMBED_DIM],
            neighborhood_ops: neighborhood,
            budget_fraction: 0.25,
            rule: RuleId::from_label("stub"),
            match_class_node_count: 1,
            expr_node_count: 1,
        }
    }

    #[test]
    fn guide_score_candidates_should_return_one_finite_score_per_candidate() {
        let guide = Guide::new_random(OpEmbeddings::new_random(1), 2);

        let candidates = alloc::vec![
            sample_candidate(0.1, alloc::vec![OpKind::Add, OpKind::Mul]),
            sample_candidate(0.9, alloc::vec![OpKind::Sqrt]),
        ];

        let scores = guide.score_candidates(&candidates);
        assert_eq!(scores.len(), candidates.len(), "one score per candidate");
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn guide_score_candidates_should_return_empty_for_empty_candidates() {
        let guide = Guide::new_random(OpEmbeddings::new_random(1), 2);
        assert!(guide.score_candidates(&[]).is_empty());
    }

    #[test]
    fn guide_score_candidates_should_distinguish_different_neighborhoods() {
        // Same rule embedding, different local neighborhoods: a Guide that
        // collapsed candidate-local structure into a constant (e.g. reading
        // only `rule_embed`) would score these identically.
        let guide = Guide::new_random(OpEmbeddings::new_random(5), 9);

        let candidates = alloc::vec![
            sample_candidate(0.5, alloc::vec![OpKind::Add]),
            sample_candidate(0.5, alloc::vec![OpKind::Sqrt, OpKind::Sqrt, OpKind::Sqrt]),
        ];
        let scores = guide.score_candidates(&candidates);
        assert!(
            (scores[0] - scores[1]).abs() > 1e-6,
            "distinct neighborhoods should score differently: {scores:?}"
        );
    }

    // ========================================================================
    // CandidateSummary::new — contract round-trip with egraph::candidate
    // ========================================================================

    #[test]
    fn candidate_summary_new_should_round_trip_features_and_rule_embed() {
        use crate::egraph::candidate::Firing;
        use crate::egraph::ops;
        use crate::egraph::{EGraph, ENode, Rewrite};
        use crate::math::algebra::Commutative;

        let rules: Vec<Box<dyn Rewrite>> = vec![Commutative::new(&ops::Add)];
        let mut eg = EGraph::with_rules(rules);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });

        let firing = Firing {
            rule: eg.rule_id(0).expect("the graph's only rule"),
            match_root: eg.find(sum),
            application_ordinal: 5,
            registered_budget: 20,
        };
        let features = CandidateFeatures::observe(&eg, &firing);
        let rule_embed = [0.42f32; EMBED_DIM];

        let summary = CandidateSummary::new(&features, rule_embed, 42);

        assert_eq!(summary.rule_embed, rule_embed);
        assert_eq!(summary.neighborhood_ops, features.neighborhood_ops);
        assert!((summary.budget_fraction - features.budget_fraction()).abs() < 1e-9);
        assert_eq!(summary.rule, features.key.rule);
        assert_eq!(
            summary.match_class_node_count,
            features.key.content.node_count()
        );
        assert_eq!(summary.expr_node_count, 42);

        // And the round-tripped summary scores identically to scoring the
        // same fields directly through the head — i.e. `new` doesn't silently
        // drop or reorder a field before it reaches the scorer. `rule` is the
        // one field varied here, because it is the only one the tower does
        // not read: it keys the per-rule linear guides
        // (`linear::LinearCandidateGuide`), while this head sees a rule only
        // through `rule_embed`. Every other field is held equal, since the
        // candidate tower now reads all four scalars (see
        // `scoring::CANDIDATE_SCALAR_COUNT` for why).
        let guide = Guide::new_random(OpEmbeddings::new_random(3), 4);
        let via_summary = guide.score_candidates(&[summary])[0];
        let direct = guide.score_candidates(&[CandidateSummary {
            rule_embed,
            neighborhood_ops: features.neighborhood_ops.clone(),
            budget_fraction: features.budget_fraction(),
            rule: RuleId::from_label("a-different-rule"),
            match_class_node_count: features.key.content.node_count(),
            expr_node_count: 42,
        }])[0];
        assert!((via_summary - direct).abs() < 1e-6);
    }
}
