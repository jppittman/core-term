//! # Saturation Guide (Phase 3 — not started)
//!
//! Given an e-graph's current state and a set of candidate rewrite-rule
//! applications, produce per-candidate move-ordering scores to guide equality
//! saturation. That sentence is the entire contract; see
//! `docs/plans/2026-07-07-guided-saturation-redesign.md` for the design this
//! module exists to satisfy once training lands, purely supervised on
//! hindsight rule-provenance labels from `crate::egraph::labeler` — no
//! REINFORCE, no self-play.
//!
//! **This module is inert today.** `GraphAccumulator::new()` appears in
//! production only as an unused placeholder value; nothing calls
//! [`SaturationGuide::score_candidates`] outside this module's own tests. The
//! public surface below — the trait and [`Guide`]'s constructor — is
//! deliberately the entire contract: every mask/graph/rule-projection weight,
//! the VSA accumulator, and the bilinear scorer are private machinery behind
//! it (`accumulator.rs`, `scoring.rs`), so a future Phase-3 trainer has one
//! seam to fill in rather than a dozen public methods to pick from.
//!
//! Kept, not deleted: the 2026-08-17 disposition inventory confirmed zero
//! non-test callers for this whole surface but a live roadmap role
//! (ROADMAP-ADMITTED, `docs/plans/2026-08-17-cost-model-domain.md` §0 row J10).
//! Encapsulating it here — rather than continuing to expose it as `pub` API
//! on [`crate::nnue::factored::ExprNnue`] — is also what closes a real bug:
//! before this change, `apply_unified_sgd` weight-decayed these
//! randomly-initialized, never-trained parameters toward zero on every
//! training run, and the shared "TRIE" checkpoint format serialized them as
//! if they were live (roughly half the file's bytes). Neither is possible
//! now: this module's weights are outside the trainer's gradient surface
//! (`pixelflow-pipeline`'s `unified_backward` no longer imports anything from
//! here) and outside the extraction head's checkpoint format entirely.

mod accumulator;
mod scoring;

extern crate alloc;

use alloc::vec::Vec;

use accumulator::GraphAccumulator;
use scoring::SaturationHead;

use crate::nnue::factored::{EMBED_DIM, ExprNnue, K};

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

/// Opaque view of e-graph state, as [`SaturationGuide::score_candidates`]
/// consumes it. Wraps a [`GraphAccumulator`] — the VSA encoding is private;
/// nothing outside this module names that type.
pub struct GraphSummary {
    pub(crate) gacc: GraphAccumulator,
}

/// One candidate rewrite-rule application, as
/// [`SaturationGuide::score_candidates`] scores it. Wraps a pre-encoded rule
/// embedding (see `scoring::SaturationHead::encode_rule_from_arena`, the
/// arena-template encoder that replaced the legacy hand-crafted
/// `RuleFeatures` path).
pub struct RuleCandidate {
    pub(crate) rule_embed: [f32; EMBED_DIM],
}

/// Given the current e-graph state and a set of candidate rewrite-rule
/// applications, produce per-candidate move-ordering scores.
///
/// The only public contract of the saturation head. See the module doc for
/// why an implementation of this trait is, today, necessarily untrained.
pub trait SaturationGuide {
    /// Score `candidates` against `graph`, one score per candidate, in order.
    fn score_candidates(&self, graph: &GraphSummary, candidates: &[RuleCandidate]) -> Vec<f32>;
}

/// The (untrained, Phase-3-gated) saturation guide: a snapshot of the shared
/// extraction backbone plus this head's own mask/graph/rule-projection
/// weights.
///
/// Owns its backbone by value rather than borrowing
/// [`crate::nnue::factored::ExprNnue`] because nothing today keeps a `Guide`
/// alive across backbone updates — Phase 3 training will need to decide how
/// the two stay in sync (the domain model doc's `EmbeddingsEpoch` seam,
/// `Extraction`, is the shape that answer will likely take; out of scope for
/// this round).
pub struct Guide {
    backbone: ExprNnue,
    head: SaturationHead,
}

impl Guide {
    /// Build a guide with a randomly-initialized saturation head over the
    /// given (already-trained-or-not) backbone.
    #[must_use]
    pub fn new_random(backbone: ExprNnue, seed: u64) -> Self {
        let mut head = SaturationHead::new();
        head.randomize(seed);
        Self { backbone, head }
    }
}

impl SaturationGuide for Guide {
    fn score_candidates(&self, graph: &GraphSummary, candidates: &[RuleCandidate]) -> Vec<f32> {
        let rule_embeds: Vec<[f32; EMBED_DIM]> = candidates.iter().map(|c| c.rule_embed).collect();
        self.head
            .mask_score_all_rules_graph(&self.backbone, &graph.gacc, &rule_embeds)
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
    // Guide::score_candidates
    // ========================================================================

    #[test]
    fn guide_score_candidates_should_return_one_finite_score_per_candidate() {
        let backbone = ExprNnue::new_random(1);
        let guide = Guide::new_random(backbone, 2);

        let mut gacc = GraphAccumulator::new();
        gacc.node_count = 1;
        let graph = GraphSummary { gacc };

        let candidates = alloc::vec![
            RuleCandidate {
                rule_embed: [0.1; EMBED_DIM],
            },
            RuleCandidate {
                rule_embed: [0.9; EMBED_DIM],
            },
        ];

        let scores = guide.score_candidates(&graph, &candidates);
        assert_eq!(scores.len(), candidates.len(), "one score per candidate");
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn guide_score_candidates_should_return_empty_for_empty_candidates() {
        let backbone = ExprNnue::new_random(1);
        let guide = Guide::new_random(backbone, 2);
        let graph = GraphSummary {
            gacc: GraphAccumulator::new(),
        };
        assert!(guide.score_candidates(&graph, &[]).is_empty());
    }
}
