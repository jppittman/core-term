//! Saturation-head scoring: the mask MLP + bilinear (mask, rule) scorer,
//! and the graph-state tower that feeds it from a [`GraphAccumulator`].
//!
//! Private implementation detail of the [`super::SaturationGuide`] contract.
//! See the module doc on [`super`] for why this is inert today.
//!
//! `#![allow(dead_code)]`: see the same note in `accumulator.rs` — every
//! method here is exercised only by this module's characterization tests
//! until Phase 3 (docs/plans/2026-07-07-guided-saturation-redesign.md) trains
//! it and a real caller reaches through [`super::Guide`].
#![allow(dead_code)]

extern crate alloc;

use alloc::vec::Vec;
use libm::{log2f, sqrtf};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};

use super::accumulator::{GRAPH_ACC_DIM, GraphAccumulator};
use crate::nnue::factored::{
    EMBED_DIM, EdgeAccumulator, ExprNnue, HIDDEN_DIM, K, MLP_HIDDEN, SCALAR_FEATURE_COUNT,
};

/// Mask MLP input dimension: expr_embed / graph_embed / candidate_embed
/// directly (`EMBED_DIM`). `value_pred` was removed — it is a deterministic
/// function of the embedding and adds zero information.
pub(crate) const MASK_INPUT_DIM: usize = EMBED_DIM;

/// Graph backbone input: `GRAPH_ACC_DIM` + 4 scalars (edge_count, node_count,
/// node_budget, epoch_budget).
pub(crate) const GRAPH_INPUT_DIM: usize = GRAPH_ACC_DIM + SCALAR_FEATURE_COUNT;

/// Candidate-local tower input (design doc §4): a bag-of-embeddings pooling
/// of the match's one-hop neighborhood ops (`K` dims, same op-embedding space
/// [`GraphAccumulator`]'s marginal sections use) plus one scalar —
/// `budget_fraction` (see
/// [`crate::egraph::candidate::CandidateFeatures::budget_fraction`]).
/// Deliberately much smaller than [`GRAPH_INPUT_DIM`]: the candidate tower
/// scores one match's own local structure, not the whole graph's aggregate
/// shape (§4's locality argument).
pub(crate) const CANDIDATE_INPUT_DIM: usize = K + 1;

/// Concatenated rule features: `[z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS]` (4 × `EMBED_DIM`).
pub(crate) const RULE_CONCAT_DIM: usize = 4 * EMBED_DIM;

/// The saturation head's own weights: mask MLP, bilinear interaction, rule
/// projection, and the graph-state tower that feeds them.
///
/// Deliberately **not** part of [`ExprNnue`] — that struct is the live
/// extraction-head checkpoint (its own "TRIF" format); these weights are
/// untrained noise until Phase 3 gives them a training loop, and keeping them
/// in a separate struct is what lets that checkpoint carry only live params
/// (see `nnue/factored.rs`'s save/load doc).
///
/// Shares the backbone (`OpEmbeddings`, the shared trunk, `expr_proj`) with
/// the extraction head — every method below that needs backbone state takes
/// `&ExprNnue` explicitly rather than owning a copy, so there is exactly one
/// trunk in memory.
#[derive(Clone)]
pub(crate) struct SaturationHead {
    /// Mask MLP layer 1 weights: embed (32) → hidden (16).
    mask_mlp_w1: [[f32; MLP_HIDDEN]; MASK_INPUT_DIM],
    /// Mask MLP layer 1 bias.
    mask_mlp_b1: [f32; MLP_HIDDEN],
    /// Mask MLP layer 2 weights: hidden (16) → mask_features (32).
    mask_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    /// Mask MLP layer 2 bias.
    mask_mlp_b2: [f32; EMBED_DIM],

    /// Rule projection weights: [`RULE_CONCAT_DIM`] x `EMBED_DIM`. Projects the
    /// 4-way LHS/RHS concatenation to a rule embedding.
    rule_proj_w: [[f32; EMBED_DIM]; RULE_CONCAT_DIM],
    /// Rule projection bias.
    rule_proj_b: [f32; EMBED_DIM],

    /// Bilinear interaction matrix: `mask_features @ interaction @ rule_embed`.
    interaction: [[f32; EMBED_DIM]; EMBED_DIM],
    /// Learned bias projection: per-rule bias via `dot(mask_bias_proj, rule_embed)`.
    mask_bias_proj: [f32; EMBED_DIM],

    /// Graph backbone weights: [`GRAPH_INPUT_DIM`] x `HIDDEN_DIM`. Kept —
    /// unused by [`super::SaturationGuide`] as of the v2 candidate-local
    /// contract (see `super`'s module doc, "Contract revision") but still
    /// live for `mask_score_all_rules_graph`, the segregated seam for a
    /// future whole-graph Judge.
    graph_w1: [[f32; HIDDEN_DIM]; GRAPH_INPUT_DIM],
    /// Graph backbone biases.
    graph_b1: [f32; HIDDEN_DIM],
    /// Graph → embed projection weights: `HIDDEN_DIM` x `EMBED_DIM`.
    graph_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Graph → embed projection bias.
    graph_proj_b: [f32; EMBED_DIM],

    /// Candidate-local tower weights (design doc §4, the v2 contract's actual
    /// scoring path): [`CANDIDATE_INPUT_DIM`] x `HIDDEN_DIM`.
    candidate_w1: [[f32; HIDDEN_DIM]; CANDIDATE_INPUT_DIM],
    /// Candidate tower biases.
    candidate_b1: [f32; HIDDEN_DIM],
    /// Candidate → embed projection weights: `HIDDEN_DIM` x `EMBED_DIM`.
    candidate_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Candidate → embed projection bias.
    candidate_proj_b: [f32; EMBED_DIM],
}

impl Default for SaturationHead {
    fn default() -> Self {
        Self::new()
    }
}

impl SaturationHead {
    /// Zero-initialized saturation head.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            mask_mlp_w1: [[0.0; MLP_HIDDEN]; MASK_INPUT_DIM],
            mask_mlp_b1: [0.0; MLP_HIDDEN],
            mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            mask_mlp_b2: [0.0; EMBED_DIM],
            rule_proj_w: [[0.0; EMBED_DIM]; RULE_CONCAT_DIM],
            rule_proj_b: [0.0; EMBED_DIM],
            interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            mask_bias_proj: [0.0; EMBED_DIM],
            graph_w1: [[0.0; HIDDEN_DIM]; GRAPH_INPUT_DIM],
            graph_b1: [0.0; HIDDEN_DIM],
            graph_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            graph_proj_b: [0.0; EMBED_DIM],
            candidate_w1: [[0.0; HIDDEN_DIM]; CANDIDATE_INPUT_DIM],
            candidate_b1: [0.0; HIDDEN_DIM],
            candidate_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            candidate_proj_b: [0.0; EMBED_DIM],
        }
    }

    /// Randomize all saturation-head weights (mask MLP, rule proj,
    /// interaction, graph backbone). Does not touch the shared backbone —
    /// that lives on [`ExprNnue`] and has its own `randomize`.
    pub(crate) fn randomize(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(54321);

        let mut next_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        // He initialization scales
        let scale_mask_input = sqrtf(2.0 / MASK_INPUT_DIM as f32);
        let scale_hidden = sqrtf(2.0 / MLP_HIDDEN as f32);
        let scale_concat = sqrtf(2.0 / RULE_CONCAT_DIM as f32);
        let scale_graph = sqrtf(2.0 / GRAPH_INPUT_DIM as f32);
        let scale_candidate = sqrtf(2.0 / CANDIDATE_INPUT_DIM as f32);
        let scale_proj = sqrtf(2.0 / HIDDEN_DIM as f32);

        // Mask MLP: MASK_INPUT_DIM → MLP_HIDDEN → EMBED_DIM
        for i in 0..MASK_INPUT_DIM {
            for j in 0..MLP_HIDDEN {
                self.mask_mlp_w1[i][j] = next_f32() * scale_mask_input;
            }
        }
        for b in &mut self.mask_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.mask_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.mask_mlp_b2 {
            *b = 0.0;
        }

        // Rule Projection: RULE_CONCAT_DIM → EMBED_DIM
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                self.rule_proj_w[i][k] = next_f32() * scale_concat;
            }
        }
        for b in &mut self.rule_proj_b {
            *b = 0.0;
        }

        // Interaction matrix: start near identity
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                self.interaction[i][j] = if i == j { 1.0 } else { next_f32() * 0.1 };
            }
        }

        // Bias projection: neutral
        for b in &mut self.mask_bias_proj {
            *b = 0.0;
        }

        // Graph backbone: GRAPH_INPUT_DIM → HIDDEN_DIM
        for row in 0..GRAPH_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.graph_w1[row][col] = next_f32() * scale_graph;
            }
        }
        for b in &mut self.graph_b1 {
            *b = next_f32().abs() * 0.1;
        }

        // Graph projection: HIDDEN_DIM → EMBED_DIM
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.graph_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        for b in &mut self.graph_proj_b {
            *b = next_f32().abs() * 0.1;
        }

        // Candidate tower: CANDIDATE_INPUT_DIM → HIDDEN_DIM
        for row in 0..CANDIDATE_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.candidate_w1[row][col] = next_f32() * scale_candidate;
            }
        }
        for b in &mut self.candidate_b1 {
            *b = next_f32().abs() * 0.1;
        }

        // Candidate projection: HIDDEN_DIM → EMBED_DIM
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.candidate_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        for b in &mut self.candidate_proj_b {
            *b = next_f32().abs() * 0.1;
        }
    }

    /// Graph state forward pass: `GraphAccumulator` → hidden (`HIDDEN_DIM`).
    ///
    /// Uses `1/sqrt(node_count)` scaling and log2 scalars, matching
    /// `EdgeAccumulator::extraction_input`'s conventions, then routes through
    /// `backbone`'s shared trunk — the same trunk the extraction head's
    /// `forward_expr_only` uses.
    #[inline]
    pub(crate) fn forward_graph(
        &self,
        backbone: &ExprNnue,
        gacc: &GraphAccumulator,
    ) -> [f32; HIDDEN_DIM] {
        let mut hidden = self.graph_b1;

        // Scale factor: 1/sqrt(N) prevents variance explosion from summing N embeddings.
        let scale = if gacc.node_count > 0 {
            1.0 / sqrtf(gacc.node_count as f32)
        } else {
            1.0
        };

        // Process graph accumulator (128 dims: 4K sections)
        for (i, &val) in gacc.values.iter().enumerate() {
            let scaled_val = val * scale;
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += scaled_val * self.graph_w1[i][j];
            }
        }

        // Process scalar features (4 dims: edge_count, node_count, node_budget, epoch_budget).
        // Use log2 to compress the range for large e-graphs.
        let base = GRAPH_ACC_DIM;
        let ec = log2f(1.0 + gacc.edge_count as f32);
        let nc = log2f(1.0 + gacc.node_count as f32);
        let nb = log2f(1.0 + gacc.node_budget as f32);
        let eb = log2f(1.0 + gacc.epoch_budget as f32);
        for (j, h) in hidden.iter_mut().enumerate() {
            *h += ec * self.graph_w1[base][j];
            *h += nc * self.graph_w1[base + 1][j];
            *h += nb * self.graph_w1[base + 2][j];
            *h += eb * self.graph_w1[base + 3][j];
        }

        // ReLU activation
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Shared trunk
        backbone.apply_trunk(&hidden)
    }

    /// Project graph hidden to graph embedding (`EMBED_DIM`).
    ///
    /// Same structure as `ExprNnue::compute_expr_embed` but with this head's
    /// own `graph_proj_w`/`graph_proj_b`.
    #[inline]
    pub(crate) fn compute_graph_embed(&self, hidden: &[f32; HIDDEN_DIM]) -> [f32; EMBED_DIM] {
        let mut embed = self.graph_proj_b;
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                embed[k] += hidden[j] * self.graph_proj_w[j][k];
            }
        }
        embed
    }

    /// Compute mask features from an embedding (expr or graph) for bilinear scoring.
    ///
    /// MLP: `MASK_INPUT_DIM` → `MLP_HIDDEN` (ReLU) → `EMBED_DIM`.
    #[inline]
    fn compute_mask_features(&self, embed: &[f32; EMBED_DIM]) -> [f32; EMBED_DIM] {
        let mut h = self.mask_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += embed[i] * self.mask_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        let mut out = self.mask_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.mask_mlp_w2[j][k];
            }
        }
        out
    }

    /// Score all rules using graph state (not expression state).
    ///
    /// `forward_graph → compute_graph_embed → compute_mask_features → bilinear_score`.
    #[must_use]
    pub(crate) fn mask_score_all_rules_graph(
        &self,
        backbone: &ExprNnue,
        gacc: &GraphAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let hidden = self.forward_graph(backbone, gacc);
        let graph_embed = self.compute_graph_embed(&hidden);
        let mask_features = self.compute_mask_features(&graph_embed);
        rule_embeds
            .iter()
            .map(|re| self.bilinear_score(&mask_features, re))
            .collect()
    }

    /// Score all rules with a pre-computed shared-backbone hidden state
    /// (e.g. the extraction head's own `forward_expr_only` output).
    #[must_use]
    pub(crate) fn mask_score_all_rules_with_hidden(
        &self,
        backbone: &ExprNnue,
        hidden: &[f32; HIDDEN_DIM],
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let expr_embed = backbone.compute_expr_embed(hidden);
        let mask_features = self.compute_mask_features(&expr_embed);
        rule_embeds
            .iter()
            .map(|rule_embed| self.bilinear_score(&mask_features, rule_embed))
            .collect()
    }

    /// Candidate-local forward pass (design doc §4 — this is the v2
    /// [`super::SaturationGuide`] contract's actual scoring path,
    /// [`Self::forward_graph`]'s replacement for that purpose): the match's
    /// one-hop neighborhood ops, bag-of-embeddings pooled the same way
    /// [`GraphAccumulator`]'s marginal child section pools (`1/sqrt(n)`
    /// scaling — same rationale, see [`Self::forward_graph`]'s doc), plus the
    /// scalar `budget_fraction`, then routed through `backbone`'s shared
    /// trunk.
    #[inline]
    pub(crate) fn forward_candidate(
        &self,
        backbone: &ExprNnue,
        neighborhood_ops: &[OpKind],
        budget_fraction: f32,
    ) -> [f32; HIDDEN_DIM] {
        let mut hidden = self.candidate_b1;

        let scale = if neighborhood_ops.is_empty() {
            1.0
        } else {
            1.0 / sqrtf(neighborhood_ops.len() as f32)
        };

        for &op in neighborhood_ops {
            let emb = backbone.embeddings.get(op);
            for i in 0..K {
                let v = emb[i] * scale;
                for (j, h) in hidden.iter_mut().enumerate() {
                    *h += v * self.candidate_w1[i][j];
                }
            }
        }

        // budget_fraction lives in the one scalar row past the K
        // neighborhood-op rows.
        let base = K;
        for (j, h) in hidden.iter_mut().enumerate() {
            *h += budget_fraction * self.candidate_w1[base][j];
        }

        for h in &mut hidden {
            *h = h.max(0.0);
        }

        backbone.apply_trunk(&hidden)
    }

    /// Project candidate hidden to candidate embedding (`EMBED_DIM`). Same
    /// structure as [`Self::compute_graph_embed`]/`ExprNnue::compute_expr_embed`
    /// with this tower's own `candidate_proj_w`/`candidate_proj_b`.
    #[inline]
    pub(crate) fn compute_candidate_embed(&self, hidden: &[f32; HIDDEN_DIM]) -> [f32; EMBED_DIM] {
        let mut embed = self.candidate_proj_b;
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                embed[k] += hidden[j] * self.candidate_proj_w[j][k];
            }
        }
        embed
    }

    /// Score one candidate (design doc §4's actual v2 scoring path):
    /// `forward_candidate -> compute_candidate_embed -> compute_mask_features
    /// -> bilinear_score`. [`super::Guide::score_candidates`] calls this once
    /// per candidate in its batch — see that impl for the batching contract.
    #[must_use]
    pub(crate) fn score_candidate(
        &self,
        backbone: &ExprNnue,
        neighborhood_ops: &[OpKind],
        budget_fraction: f32,
        rule_embed: &[f32; EMBED_DIM],
    ) -> f32 {
        let hidden = self.forward_candidate(backbone, neighborhood_ops, budget_fraction);
        let embed = self.compute_candidate_embed(&hidden);
        let mask_features = self.compute_mask_features(&embed);
        self.bilinear_score(&mask_features, rule_embed)
    }

    /// Encode a single rule's LHS/RHS arena subtrees into a rule embedding.
    ///
    /// 4-way concatenation `[z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS]`
    /// projected to `EMBED_DIM`, using the same shared backbone as the
    /// extraction head (`backbone.forward_expr_only` / `compute_expr_embed`)
    /// — the design that replaced the legacy hand-crafted `RuleFeatures` path.
    #[must_use]
    pub(crate) fn encode_rule_from_arena(
        &self,
        backbone: &ExprNnue,
        arena: &ExprArena,
        lhs: ExprId,
        rhs: ExprId,
    ) -> [f32; EMBED_DIM] {
        let lhs_acc = EdgeAccumulator::from_arena_dag(arena, lhs, &backbone.embeddings);
        let lhs_hidden = backbone.forward_expr_only(&lhs_acc);
        let z_lhs = backbone.compute_expr_embed(&lhs_hidden);

        let rhs_acc = EdgeAccumulator::from_arena_dag(arena, rhs, &backbone.embeddings);
        let rhs_hidden = backbone.forward_expr_only(&rhs_acc);
        let z_rhs = backbone.compute_expr_embed(&rhs_hidden);

        let mut concat = [0.0f32; RULE_CONCAT_DIM];
        for i in 0..EMBED_DIM {
            concat[i] = z_lhs[i];
            concat[EMBED_DIM + i] = z_rhs[i];
            concat[2 * EMBED_DIM + i] = z_lhs[i] - z_rhs[i];
            concat[3 * EMBED_DIM + i] = z_lhs[i] * z_rhs[i];
        }

        let mut out = self.rule_proj_b;
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                out[k] += concat[i] * self.rule_proj_w[i][k];
            }
        }
        out
    }

    /// Bilinear score: `mask_features @ interaction @ rule_embed + bias`.
    #[inline]
    #[must_use]
    pub(crate) fn bilinear_score(
        &self,
        mask_features: &[f32; EMBED_DIM],
        rule_embed: &[f32; EMBED_DIM],
    ) -> f32 {
        let mut transformed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                transformed[j] += mask_features[i] * self.interaction[i][j];
            }
        }

        let mut score = 0.0f32;
        for k in 0..EMBED_DIM {
            score += (transformed[k] + self.mask_bias_proj[k]) * rule_embed[k];
        }
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::factored::{OpEmbeddings, OpKind};

    fn test_backbone() -> ExprNnue {
        ExprNnue::new_random(42)
    }

    #[test]
    fn bilinear_score_should_match_manual_computation_for_all_ones_vectors() {
        let head = SaturationHead::new();
        let mut randomized = head.clone();
        randomized.randomize(42);

        let mask_features = [1.0f32; EMBED_DIM];
        let rule_embed = [1.0f32; EMBED_DIM];

        let score = randomized.bilinear_score(&mask_features, &rule_embed);
        assert!(score.is_finite(), "Bilinear score should be finite");

        // Manual verification: score = dot(mask @ interaction + bias_proj, rule)
        // With all-ones vectors: sum of interaction matrix + sum of bias_proj
        let mut expected = 0.0f32;
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                expected += randomized.interaction[i][j];
            }
        }
        for k in 0..EMBED_DIM {
            expected += randomized.mask_bias_proj[k];
        }
        assert!(
            (score - expected).abs() < 1e-4,
            "Bilinear computation mismatch: got {}, expected {}",
            score,
            expected
        );
    }

    #[test]
    fn randomize_should_be_deterministic_and_finite_with_a_near_identity_diagonal() {
        let mut a = SaturationHead::new();
        a.randomize(42);
        let mut b = SaturationHead::new();
        b.randomize(42);

        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                assert!(
                    (a.interaction[i][j] - b.interaction[i][j]).abs() < 1e-9,
                    "randomize(seed) must be deterministic"
                );
                assert!(a.interaction[i][j].is_finite());
            }
        }
        // Interaction matrix should be near identity diagonal.
        for i in 0..EMBED_DIM {
            assert!(
                (a.interaction[i][i] - 1.0).abs() < 0.5,
                "Diagonal of interaction should be near 1.0"
            );
        }
    }

    #[test]
    fn forward_graph_should_use_the_backbone_trunk() {
        let backbone = test_backbone();
        let mut head = SaturationHead::new();
        head.randomize(7);

        let mut gacc = GraphAccumulator::new();
        gacc.add_edge(&backbone.embeddings, OpKind::Add, OpKind::Mul);
        gacc.node_count = 2;
        gacc.edge_count = 1;

        let hidden = head.forward_graph(&backbone, &gacc);
        assert!(hidden.iter().all(|v| v.is_finite()));

        let embed = head.compute_graph_embed(&hidden);
        assert!(embed.iter().all(|v| v.is_finite()));

        let scores = head.mask_score_all_rules_graph(&backbone, &gacc, &[[0.5f32; EMBED_DIM]]);
        assert_eq!(scores.len(), 1);
        assert!(scores[0].is_finite());
    }

    #[test]
    fn encode_rule_from_arena_should_be_deterministic() {
        let backbone = test_backbone();
        let mut head = SaturationHead::new();
        head.randomize(3);

        let mut arena = ExprArena::with_capacity(4);
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let lhs = arena.push_binary(OpKind::Add, x, one);
        let rhs = arena.push_binary(OpKind::Mul, x, one);

        let e1 = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);
        let e2 = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);
        for i in 0..EMBED_DIM {
            assert!((e1[i] - e2[i]).abs() < 1e-6, "must be deterministic");
            assert!(e1[i].is_finite());
        }
    }

    #[test]
    fn graph_input_dim_should_equal_graph_acc_dim_plus_the_scalar_feature_count() {
        // GRAPH_ACC_DIM (128) + SCALAR_FEATURE_COUNT (4) = 132. Written as a
        // literal (rather than re-deriving via `+`) so a `+` -> `*` mutation
        // on the const declaration (128 * 4 = 512) is unambiguously wrong.
        assert_eq!(GRAPH_INPUT_DIM, 132);
        assert_eq!(GRAPH_ACC_DIM, 128);
        assert_eq!(SCALAR_FEATURE_COUNT, 4);
    }

    /// An `ExprNnue` whose shared trunk is the identity (zero bias, identity
    /// weight matrix). `apply_trunk` then returns its input unchanged for any
    /// non-negative vector, letting `forward_graph` tests assert an exact
    /// pre-trunk hidden value without also having to hand-verify the trunk.
    fn identity_trunk_backbone() -> ExprNnue {
        let mut backbone = ExprNnue::new();
        for i in 0..HIDDEN_DIM {
            backbone.trunk_w[i][i] = 1.0;
        }
        backbone
    }

    #[test]
    fn forward_graph_should_index_the_weight_matrix_by_both_lane_and_hidden_column() {
        // The hand-computed test below fills every accumulator lane and every
        // weight row uniformly, which pins the arithmetic but not the
        // indexing: with `values` all 2.0 and every row `[3.0; HIDDEN_DIM]`,
        // reading `graph_w1[0][j]` or `graph_w1[i][0]` instead of
        // `graph_w1[i][j]` produces the same answer. This one is sparse
        // instead, so exactly one (lane, column) pair is live.
        //
        // Neither index is zero, so both substitutions read a zero and
        // collapse the output.
        const LANE: usize = 7;
        const COLUMN: usize = 5;

        let backbone = identity_trunk_backbone();
        let mut head = SaturationHead::new();
        head.graph_w1[LANE][COLUMN] = 3.0;

        let mut gacc = GraphAccumulator::new();
        gacc.values[LANE] = 2.0;
        gacc.node_count = 1; // scale = 1/sqrt(1) = 1

        // Every scalar-feature row is left at zero, so `log2(1 + node_count)`
        // being nonzero here contributes nothing either way.
        let hidden = head.forward_graph(&backbone, &gacc);

        for (j, &h) in hidden.iter().enumerate() {
            let want = if j == COLUMN { 6.0 } else { 0.0 };
            assert!(
                (h - want).abs() < 1e-6,
                "hidden[{j}]: got {h}, expected {want}"
            );
        }
    }

    #[test]
    fn forward_graph_should_match_a_hand_computed_value_when_node_count_is_positive() {
        let backbone = identity_trunk_backbone();
        let mut head = SaturationHead::new();
        head.graph_b1 = [10.0; HIDDEN_DIM];
        for i in 0..GRAPH_ACC_DIM {
            head.graph_w1[i] = [3.0; HIDDEN_DIM];
        }
        head.graph_w1[GRAPH_ACC_DIM] = [100.0; HIDDEN_DIM]; // edge_count row
        head.graph_w1[GRAPH_ACC_DIM + 1] = [200.0; HIDDEN_DIM]; // node_count row
        head.graph_w1[GRAPH_ACC_DIM + 2] = [300.0; HIDDEN_DIM]; // node_budget row
        head.graph_w1[GRAPH_ACC_DIM + 3] = [400.0; HIDDEN_DIM]; // epoch_budget row

        let mut gacc = GraphAccumulator::new();
        gacc.values = [2.0; GRAPH_ACC_DIM];
        gacc.node_count = 4; // sqrtf(4) = 2 -> scale = 0.5
        gacc.edge_count = 2;
        gacc.node_budget = 5;
        gacc.epoch_budget = 6;

        let hidden = head.forward_graph(&backbone, &gacc);

        let scale = 1.0 / sqrtf(4.0f32);
        let main_sum = GRAPH_ACC_DIM as f32 * (2.0 * scale) * 3.0;
        let ec = log2f(1.0 + 2.0f32);
        let nc = log2f(1.0 + 4.0f32);
        let nb = log2f(1.0 + 5.0f32);
        let eb = log2f(1.0 + 6.0f32);
        let expected = 10.0 + main_sum + ec * 100.0 + nc * 200.0 + nb * 300.0 + eb * 400.0;
        assert!(
            expected > 0.0,
            "fixture must stay in the ReLU's positive branch"
        );

        for (j, &h) in hidden.iter().enumerate() {
            assert!(
                (h - expected).abs() < 1e-2,
                "hidden[{j}]: got {h}, expected {expected}"
            );
        }
    }

    #[test]
    fn forward_graph_should_use_a_scale_of_one_when_node_count_is_zero() {
        let backbone = identity_trunk_backbone();
        let mut head = SaturationHead::new();
        head.graph_b1 = [5.0; HIDDEN_DIM];
        head.graph_w1[0] = [1.0; HIDDEN_DIM];

        // One populated lane, so the output depends on the scale's *value* and
        // not merely on its finiteness: an all-zero accumulator multiplies
        // every candidate scale by 0.0 and lands on the bias either way, which
        // would leave 0.0, 1.0 and 2.0 indistinguishable here.
        let mut gacc = GraphAccumulator::new();
        gacc.values[0] = 8.0;

        // node_count stays 0, so `node_count > 0` is false and the scale is
        // 1.0. Every scalar feature is log2(1 + 0) = 0 and contributes nothing.
        // A `>` -> `>=`/`==` mutation takes the other branch and computes
        // `1.0 / sqrtf(0.0)` = inf, sending the lane to inf rather than 13.0.
        let expected = 5.0 + 8.0 * 1.0;

        let hidden = head.forward_graph(&backbone, &gacc);
        for (j, &h) in hidden.iter().enumerate() {
            assert!(
                (h - expected).abs() < 1e-6,
                "hidden[{j}]: got {h}, expected {expected}"
            );
        }
    }

    #[test]
    fn compute_graph_embed_should_match_a_hand_computed_value_for_constant_inputs() {
        let mut head = SaturationHead::new();
        head.graph_proj_b = [1.0; EMBED_DIM];
        head.graph_proj_w = [[4.0; EMBED_DIM]; HIDDEN_DIM];
        let hidden = [2.0f32; HIDDEN_DIM];

        let embed = head.compute_graph_embed(&hidden);

        // expected[k] = b[k] + sum_j hidden[j] * w[j][k] = 1.0 + 64 * (2.0 * 4.0)
        let expected = 1.0 + HIDDEN_DIM as f32 * (2.0 * 4.0);
        for (k, &e) in embed.iter().enumerate() {
            assert!(
                (e - expected).abs() < 1e-2,
                "embed[{k}]: got {e}, expected {expected}"
            );
        }
    }

    #[test]
    fn compute_mask_features_should_match_a_hand_computed_value_for_constant_inputs() {
        let mut head = SaturationHead::new();
        head.mask_mlp_b1 = [1.0; MLP_HIDDEN];
        head.mask_mlp_w1 = [[4.0; MLP_HIDDEN]; MASK_INPUT_DIM];
        head.mask_mlp_b2 = [5.0; EMBED_DIM];
        head.mask_mlp_w2 = [[3.0; EMBED_DIM]; MLP_HIDDEN];
        let embed = [2.0f32; EMBED_DIM];

        let out = head.compute_mask_features(&embed);

        // h[j] = 1.0 + 32 * (2.0 * 4.0) = 257.0 (positive, ReLU is a no-op)
        let hidden_val = 1.0 + MASK_INPUT_DIM as f32 * (2.0 * 4.0);
        assert!(
            hidden_val > 0.0,
            "fixture must stay in the ReLU's positive branch"
        );
        // out[k] = 5.0 + 16 * (257.0 * 3.0)
        let expected = 5.0 + MLP_HIDDEN as f32 * (hidden_val * 3.0);
        for (k, &o) in out.iter().enumerate() {
            assert!(
                (o - expected).abs() < 1e-1,
                "out[{k}]: got {o}, expected {expected}"
            );
        }
    }

    #[test]
    fn mask_score_all_rules_with_hidden_should_match_a_manual_composition_of_backbone_and_head() {
        let backbone = test_backbone();
        let mut head = SaturationHead::new();
        head.randomize(11);

        let hidden = [0.3f32; HIDDEN_DIM];
        let rule_embeds = [[0.2f32; EMBED_DIM], [0.7f32; EMBED_DIM]];

        // Reference: replay the same two steps `mask_score_all_rules_with_hidden`
        // is documented to perform, calling the same (private, same-module)
        // helpers it calls.
        let expr_embed = backbone.compute_expr_embed(&hidden);
        let mask_features = head.compute_mask_features(&expr_embed);
        let expected: Vec<f32> = rule_embeds
            .iter()
            .map(|re| head.bilinear_score(&mask_features, re))
            .collect();

        let actual = head.mask_score_all_rules_with_hidden(&backbone, &hidden, &rule_embeds);

        assert_eq!(actual.len(), expected.len(), "one score per rule embed");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-4, "got {a}, expected {e}");
        }
        // Distinct rule embeddings must not collapse to the same score — guards
        // against a constant-vector replacement of the whole function body.
        assert!(
            (actual[0] - actual[1]).abs() > 1e-6,
            "distinct rule embeds should score differently"
        );
    }

    #[test]
    fn bilinear_score_should_respond_to_bias_addition_and_rule_embed_scaling() {
        let mut head = SaturationHead::new();
        head.interaction = [[1.0; EMBED_DIM]; EMBED_DIM];
        head.mask_bias_proj = [3.0; EMBED_DIM];
        let mask_features = [1.0f32; EMBED_DIM];
        let rule_embed = [2.0f32; EMBED_DIM];

        let score = head.bilinear_score(&mask_features, &rule_embed);

        // transformed[k] = sum_i mask_features[i] * interaction[i][k] = 32.0
        // score = sum_k (transformed[k] + bias[k]) * rule_embed[k]
        //       = 32 * ((32.0 + 3.0) * 2.0)
        let transformed = EMBED_DIM as f32;
        let expected = EMBED_DIM as f32 * ((transformed + 3.0) * 2.0);
        assert!(
            (score - expected).abs() < 1e-2,
            "got {score}, expected {expected}"
        );
    }

    fn sample_rule_arena() -> (ExprArena, ExprId, ExprId) {
        let mut arena = ExprArena::with_capacity(4);
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let lhs = arena.push_binary(OpKind::Add, x, one);
        let rhs = arena.push_binary(OpKind::Mul, x, one);
        (arena, lhs, rhs)
    }

    fn embed_of(backbone: &ExprNnue, arena: &ExprArena, root: ExprId) -> [f32; EMBED_DIM] {
        let acc = EdgeAccumulator::from_arena_dag(arena, root, &backbone.embeddings);
        let hidden = backbone.forward_expr_only(&acc);
        backbone.compute_expr_embed(&hidden)
    }

    /// A `SaturationHead` whose `rule_proj_w` is a "selector": projecting
    /// picks out `concat[block_start + k]` (scaled by `weight`) as output `k`,
    /// with every other row zeroed. Lets a concat-block test read that block
    /// straight out of `encode_rule_from_arena`'s output without disturbing
    /// the other three blocks (a real `RULE_CONCAT_DIM x EMBED_DIM` matrix
    /// would mix them all together).
    fn concat_block_selector(block_start: usize, weight: f32) -> SaturationHead {
        let mut head = SaturationHead::new();
        for k in 0..EMBED_DIM {
            head.rule_proj_w[block_start + k][k] = weight;
        }
        head
    }

    #[test]
    fn encode_rule_from_arena_should_place_the_lhs_embedding_at_the_first_concat_block() {
        let backbone = test_backbone();
        let (arena, lhs, rhs) = sample_rule_arena();
        let z_lhs = embed_of(&backbone, &arena, lhs);

        let head = concat_block_selector(0, 2.0);
        let out = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);

        for k in 0..EMBED_DIM {
            let expected = 2.0 * z_lhs[k];
            assert!(
                (out[k] - expected).abs() < 1e-3,
                "block1[{k}]: got {}, expected {expected}",
                out[k]
            );
        }
    }

    #[test]
    fn encode_rule_from_arena_should_place_the_rhs_embedding_at_the_second_concat_block() {
        let backbone = test_backbone();
        let (arena, lhs, rhs) = sample_rule_arena();
        let z_rhs = embed_of(&backbone, &arena, rhs);

        let head = concat_block_selector(EMBED_DIM, 2.0);
        let out = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);

        for k in 0..EMBED_DIM {
            let expected = 2.0 * z_rhs[k];
            assert!(
                (out[k] - expected).abs() < 1e-3,
                "block2[{k}]: got {}, expected {expected}",
                out[k]
            );
        }
    }

    #[test]
    fn encode_rule_from_arena_should_place_the_elementwise_difference_at_the_third_concat_block() {
        let backbone = test_backbone();
        let (arena, lhs, rhs) = sample_rule_arena();
        let z_lhs = embed_of(&backbone, &arena, lhs);
        let z_rhs = embed_of(&backbone, &arena, rhs);

        let head = concat_block_selector(2 * EMBED_DIM, 2.0);
        let out = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);

        for k in 0..EMBED_DIM {
            let expected = 2.0 * (z_lhs[k] - z_rhs[k]);
            assert!(
                (out[k] - expected).abs() < 1e-3,
                "block3[{k}]: got {}, expected {expected}",
                out[k]
            );
        }
    }

    #[test]
    fn encode_rule_from_arena_should_place_the_elementwise_product_at_the_fourth_concat_block() {
        let backbone = test_backbone();
        let (arena, lhs, rhs) = sample_rule_arena();
        let z_lhs = embed_of(&backbone, &arena, lhs);
        let z_rhs = embed_of(&backbone, &arena, rhs);

        let head = concat_block_selector(3 * EMBED_DIM, 2.0);
        let out = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);

        for k in 0..EMBED_DIM {
            let expected = 2.0 * (z_lhs[k] * z_rhs[k]);
            assert!(
                (out[k] - expected).abs() < 1e-3,
                "block4[{k}]: got {}, expected {expected}",
                out[k]
            );
        }
    }

    #[test]
    fn randomize_should_match_a_hand_recomputed_he_initialization_reference() {
        let seed = 99u64;
        let mut head = SaturationHead::new();
        head.randomize(seed);

        // Independent reference re-implementation of the same LCG stepping and
        // He-init formulas, kept out of `SaturationHead::randomize` on
        // purpose: an operator mutation there (RNG update, a scale formula, or
        // a `next_f32() * scale` use site) shows up here as a numeric
        // mismatch instead of hiding behind an `is_finite()`/determinism-only
        // check.
        let mut rng_state = seed.wrapping_add(54321);
        let mut next_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        let scale_mask_input = sqrtf(2.0 / MASK_INPUT_DIM as f32);
        let scale_hidden = sqrtf(2.0 / MLP_HIDDEN as f32);
        let scale_concat = sqrtf(2.0 / RULE_CONCAT_DIM as f32);
        let scale_graph = sqrtf(2.0 / GRAPH_INPUT_DIM as f32);
        let scale_candidate = sqrtf(2.0 / CANDIDATE_INPUT_DIM as f32);
        let scale_proj = sqrtf(2.0 / HIDDEN_DIM as f32);

        let mut exp_mask_mlp_w1 = [[0.0f32; MLP_HIDDEN]; MASK_INPUT_DIM];
        for i in 0..MASK_INPUT_DIM {
            for j in 0..MLP_HIDDEN {
                exp_mask_mlp_w1[i][j] = next_f32() * scale_mask_input;
            }
        }
        let mut exp_mask_mlp_b1 = [0.0f32; MLP_HIDDEN];
        for b in &mut exp_mask_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        let mut exp_mask_mlp_w2 = [[0.0f32; EMBED_DIM]; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                exp_mask_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        // mask_mlp_b2 is set to a literal 0.0 in `randomize` — no RNG consumed.

        let mut exp_rule_proj_w = [[0.0f32; EMBED_DIM]; RULE_CONCAT_DIM];
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                exp_rule_proj_w[i][k] = next_f32() * scale_concat;
            }
        }
        // rule_proj_b: literal 0.0, no RNG consumed.

        let mut exp_interaction = [[0.0f32; EMBED_DIM]; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                exp_interaction[i][j] = if i == j { 1.0 } else { next_f32() * 0.1 };
            }
        }
        // mask_bias_proj: literal 0.0, no RNG consumed.

        let mut exp_graph_w1 = [[0.0f32; HIDDEN_DIM]; GRAPH_INPUT_DIM];
        for row in 0..GRAPH_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                exp_graph_w1[row][col] = next_f32() * scale_graph;
            }
        }
        let mut exp_graph_b1 = [0.0f32; HIDDEN_DIM];
        for b in &mut exp_graph_b1 {
            *b = next_f32().abs() * 0.1;
        }
        let mut exp_graph_proj_w = [[0.0f32; EMBED_DIM]; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                exp_graph_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        let mut exp_graph_proj_b = [0.0f32; EMBED_DIM];
        for b in &mut exp_graph_proj_b {
            *b = next_f32().abs() * 0.1;
        }

        let mut exp_candidate_w1 = [[0.0f32; HIDDEN_DIM]; CANDIDATE_INPUT_DIM];
        for row in 0..CANDIDATE_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                exp_candidate_w1[row][col] = next_f32() * scale_candidate;
            }
        }
        let mut exp_candidate_b1 = [0.0f32; HIDDEN_DIM];
        for b in &mut exp_candidate_b1 {
            *b = next_f32().abs() * 0.1;
        }
        let mut exp_candidate_proj_w = [[0.0f32; EMBED_DIM]; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                exp_candidate_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        let mut exp_candidate_proj_b = [0.0f32; EMBED_DIM];
        for b in &mut exp_candidate_proj_b {
            *b = next_f32().abs() * 0.1;
        }

        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;

        for i in 0..MASK_INPUT_DIM {
            for j in 0..MLP_HIDDEN {
                assert!(
                    close(head.mask_mlp_w1[i][j], exp_mask_mlp_w1[i][j]),
                    "mask_mlp_w1[{i}][{j}]: got {}, expected {}",
                    head.mask_mlp_w1[i][j],
                    exp_mask_mlp_w1[i][j]
                );
            }
        }
        for j in 0..MLP_HIDDEN {
            assert!(
                close(head.mask_mlp_b1[j], exp_mask_mlp_b1[j]),
                "mask_mlp_b1[{j}]"
            );
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                assert!(
                    close(head.mask_mlp_w2[j][k], exp_mask_mlp_w2[j][k]),
                    "mask_mlp_w2[{j}][{k}]"
                );
            }
        }
        // The three arrays `randomize` leaves at a literal 0.0. Asserted rather
        // than merely noted above: without these, `randomize` could start
        // writing a nonzero literal into any of them, or consume RNG while
        // populating them, and every other assertion here would still pass —
        // the reference stream would simply shift in lockstep.
        for k in 0..EMBED_DIM {
            assert!(
                close(head.mask_mlp_b2[k], 0.0),
                "mask_mlp_b2[{k}] must stay zero, got {}",
                head.mask_mlp_b2[k]
            );
        }

        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                assert!(
                    close(head.rule_proj_w[i][k], exp_rule_proj_w[i][k]),
                    "rule_proj_w[{i}][{k}]"
                );
            }
        }
        for k in 0..EMBED_DIM {
            assert!(
                close(head.rule_proj_b[k], 0.0),
                "rule_proj_b[{k}] must stay zero, got {}",
                head.rule_proj_b[k]
            );
        }
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                assert!(
                    close(head.interaction[i][j], exp_interaction[i][j]),
                    "interaction[{i}][{j}]"
                );
            }
        }
        for k in 0..EMBED_DIM {
            assert!(
                close(head.mask_bias_proj[k], 0.0),
                "mask_bias_proj[{k}] must stay zero, got {}",
                head.mask_bias_proj[k]
            );
        }
        for row in 0..GRAPH_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                assert!(
                    close(head.graph_w1[row][col], exp_graph_w1[row][col]),
                    "graph_w1[{row}][{col}]"
                );
            }
        }
        for j in 0..HIDDEN_DIM {
            assert!(close(head.graph_b1[j], exp_graph_b1[j]), "graph_b1[{j}]");
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                assert!(
                    close(head.graph_proj_w[j][k], exp_graph_proj_w[j][k]),
                    "graph_proj_w[{j}][{k}]"
                );
            }
        }
        for k in 0..EMBED_DIM {
            assert!(
                close(head.graph_proj_b[k], exp_graph_proj_b[k]),
                "graph_proj_b[{k}]"
            );
        }
        for row in 0..CANDIDATE_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                assert!(
                    close(head.candidate_w1[row][col], exp_candidate_w1[row][col]),
                    "candidate_w1[{row}][{col}]"
                );
            }
        }
        for j in 0..HIDDEN_DIM {
            assert!(
                close(head.candidate_b1[j], exp_candidate_b1[j]),
                "candidate_b1[{j}]"
            );
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                assert!(
                    close(head.candidate_proj_w[j][k], exp_candidate_proj_w[j][k]),
                    "candidate_proj_w[{j}][{k}]"
                );
            }
        }
        for k in 0..EMBED_DIM {
            assert!(
                close(head.candidate_proj_b[k], exp_candidate_proj_b[k]),
                "candidate_proj_b[{k}]"
            );
        }
    }

    // ── Index and orientation pinning ────────────────────────────────────
    //
    // The exact-value tests above use uniform fixtures, which pin arithmetic
    // but not indexing: when every input and weight is the same number,
    // reading row 0, column 0, or the transpose gives the same answer. Each
    // test below makes exactly one (index, index) pair live, with no index
    // equal to zero, so a substituted index reads a zero and collapses the
    // result.

    #[test]
    fn compute_graph_embed_should_index_the_projection_by_both_hidden_lane_and_column() {
        const LANE: usize = 3;
        const COLUMN: usize = 5;

        let mut head = SaturationHead::new();
        head.graph_proj_w[LANE][COLUMN] = 3.0;
        let mut hidden = [0.0f32; HIDDEN_DIM];
        hidden[LANE] = 2.0;

        let embed = head.compute_graph_embed(&hidden);

        for (k, &v) in embed.iter().enumerate() {
            let want = if k == COLUMN { 6.0 } else { 0.0 };
            assert!(
                (v - want).abs() < 1e-6,
                "embed[{k}]: got {v}, expected {want}"
            );
        }
    }

    #[test]
    fn compute_mask_features_should_index_both_mlp_layers_by_lane_and_column() {
        const IN: usize = 2;
        const MID: usize = 4;
        const OUT: usize = 7;

        let mut head = SaturationHead::new();
        head.mask_mlp_w1[IN][MID] = 3.0;
        head.mask_mlp_w2[MID][OUT] = 5.0;

        let mut embed = [0.0f32; EMBED_DIM];
        embed[IN] = 2.0;

        // Layer 1: h[MID] = 2*3 = 6, every other hidden lane 0 (and the ReLU
        // leaves both alone). Layer 2: out[OUT] = 6*5 = 30.
        let out = head.compute_mask_features(&embed);

        for (k, &v) in out.iter().enumerate() {
            let want = if k == OUT { 30.0 } else { 0.0 };
            assert!(
                (v - want).abs() < 1e-6,
                "out[{k}]: got {v}, expected {want}"
            );
        }
    }

    #[test]
    fn bilinear_score_should_respect_the_interaction_orientation_and_the_bias_lane() {
        // `transformed[j] += mask_features[i] * interaction[i][j]`, so with a
        // single off-diagonal entry live the transpose sends the mass to a
        // different lane than the rule embedding reads.
        const ROW: usize = 2;
        const COL: usize = 5;

        let mut head = SaturationHead::new();
        head.interaction[ROW][COL] = 1.0;
        head.mask_bias_proj[COL] = 7.0;

        let mut mask_features = [0.0f32; EMBED_DIM];
        mask_features[ROW] = 1.0;
        let mut rule_embed = [0.0f32; EMBED_DIM];
        rule_embed[COL] = 1.0;

        // transformed[COL] = 1, plus the bias lane 7, times rule_embed[COL].
        // Transposing the interaction puts the 1 on lane ROW instead, which
        // rule_embed does not read, leaving 7. Reading `mask_bias_proj[0]`
        // instead of `[k]` drops the 7, leaving 1.
        let score = head.bilinear_score(&mask_features, &rule_embed);
        assert!(
            (score - 8.0).abs() < 1e-6,
            "score: got {score}, expected 8.0"
        );
    }

    #[test]
    fn forward_graph_should_clamp_a_negative_preactivation_before_the_shared_trunk() {
        // `apply_trunk` ends in its own ReLU, so an identity trunk cannot
        // separate the two stages — a negative lane is clamped either way.
        // A negating trunk can: clamped-then-negated stays 0, while
        // negated-without-clamping becomes positive and survives the trunk's
        // own ReLU.
        let mut backbone = ExprNnue::new();
        for i in 0..HIDDEN_DIM {
            backbone.trunk_w[i][i] = -1.0;
        }

        let mut head = SaturationHead::new();
        head.graph_b1 = [-1.0; HIDDEN_DIM];

        let hidden = head.forward_graph(&backbone, &GraphAccumulator::new());

        for (j, &h) in hidden.iter().enumerate() {
            assert!(
                h.abs() < 1e-6,
                "hidden[{j}]: got {h}, expected 0.0 — a -1.0 preactivation must be \
                 clamped by the tower's ReLU before the negating trunk sees it"
            );
        }
    }

    #[test]
    fn encode_rule_from_arena_should_start_the_projection_from_the_rule_bias() {
        // Every selector fixture above leaves `rule_proj_b` at zero, and
        // `randomize` deliberately zeroes it too, so replacing
        // `let mut out = self.rule_proj_b` with an all-zero array is
        // invisible everywhere else. Zero the projection weights and give the
        // bias distinct lanes: the output is then exactly the bias.
        let backbone = test_backbone();
        let (arena, lhs, rhs) = sample_rule_arena();

        let mut head = SaturationHead::new();
        for k in 0..EMBED_DIM {
            head.rule_proj_b[k] = k as f32 + 1.0;
        }

        let out = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);

        for (k, &v) in out.iter().enumerate() {
            let want = k as f32 + 1.0;
            assert!(
                (v - want).abs() < 1e-4,
                "out[{k}]: got {v}, expected {want}"
            );
        }
    }

    #[test]
    fn mask_score_all_rules_graph_should_compose_the_graph_tower_projection_and_scorer() {
        // `Guide::score_candidates` routes through this path, but the only
        // exact composition test covers the `_with_hidden` variant; the graph
        // variant was checked for length and finiteness alone. Rewiring it to
        // `backbone.compute_expr_embed` instead of `self.compute_graph_embed`
        // would go unnoticed.
        let backbone = test_backbone();
        let mut head = SaturationHead::new();
        head.randomize(7);

        let mut gacc = GraphAccumulator::new();
        let emb = OpEmbeddings::new_random(3);
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_edge(&emb, OpKind::Sub, OpKind::Div);
        gacc.node_count = 3;
        gacc.edge_count = 2;

        let (arena, lhs, rhs) = sample_rule_arena();
        let rule = head.encode_rule_from_arena(&backbone, &arena, lhs, rhs);
        let rules = [rule];

        let got = head.mask_score_all_rules_graph(&backbone, &gacc, &rules);

        let hidden = head.forward_graph(&backbone, &gacc);
        let graph_embed = head.compute_graph_embed(&hidden);
        let mask_features = head.compute_mask_features(&graph_embed);
        let want = head.bilinear_score(&mask_features, &rules[0]);

        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - want).abs() < 1e-3,
            "graph scoring must be forward_graph -> compute_graph_embed -> \
             compute_mask_features -> bilinear_score; got {}, want {want}",
            got[0]
        );
    }

    // ========================================================================
    // Candidate-local tower (design doc §4, v2 SaturationGuide contract)
    // ========================================================================

    #[test]
    fn forward_candidate_should_be_finite_for_an_empty_neighborhood() {
        let backbone = test_backbone();
        let mut head = SaturationHead::new();
        head.randomize(1);

        let hidden = head.forward_candidate(&backbone, &[], 0.0);
        assert!(hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_candidate_should_use_a_scale_of_one_when_the_neighborhood_is_empty() {
        // Mirrors `forward_graph_should_use_a_scale_of_one_when_node_count_is_zero`:
        // an empty neighborhood must not divide by `sqrtf(0)` (= inf/NaN). With
        // no ops to sum, only the budget_fraction row and bias contribute.
        let backbone = identity_trunk_backbone();
        let mut head = SaturationHead::new();
        head.candidate_b1 = [5.0; HIDDEN_DIM];
        head.candidate_w1[K] = [2.0; HIDDEN_DIM]; // the budget_fraction row

        let hidden = head.forward_candidate(&backbone, &[], 3.0);

        let expected = 5.0 + 3.0 * 2.0;
        for (j, &h) in hidden.iter().enumerate() {
            assert!(
                (h - expected).abs() < 1e-6,
                "hidden[{j}]: got {h}, expected {expected}"
            );
        }
    }

    #[test]
    fn forward_candidate_should_index_the_op_weight_matrix_by_both_lane_and_column() {
        // Sparse fixture (see the analogous `forward_graph` index-pinning
        // test): exactly one (K-lane, hidden-column) pair is live, with
        // neither index zero, so a `candidate_w1[0][j]`/`candidate_w1[i][0]`
        // substitution collapses to zero instead of coincidentally matching.
        const LANE: usize = 5;
        const COLUMN: usize = 7;

        let mut backbone = identity_trunk_backbone();
        let mut head = SaturationHead::new();
        head.candidate_w1[LANE][COLUMN] = 3.0;

        // A neighborhood of exactly one op whose embedding is all-zero except
        // lane LANE keeps only `candidate_w1[LANE]` live; scale = 1/sqrt(1) = 1.
        const LANE_VAL: f32 = 2.0;
        let mut op_embed = [0.0f32; K];
        op_embed[LANE] = LANE_VAL;
        backbone.embeddings.e[OpKind::Add] = op_embed;

        let hidden = head.forward_candidate(&backbone, &[OpKind::Add], 0.0);

        for (j, &h) in hidden.iter().enumerate() {
            let want = if j == COLUMN { LANE_VAL * 3.0 } else { 0.0 };
            assert!(
                (h - want).abs() < 1e-6,
                "hidden[{j}]: got {h}, expected {want}"
            );
        }
    }

    #[test]
    fn forward_candidate_should_scale_by_one_over_sqrt_of_neighborhood_size() {
        // Identity trunk (so the returned hidden is directly comparable to a
        // hand-computed pre-trunk value) with each op's embedding manually
        // pinned to a known constant vector — `identity_trunk_backbone`'s
        // default `ExprNnue::new()` embeddings are all-zero (would make every
        // candidate op contribute nothing), and real random embeddings could
        // sum to either sign depending on the seed, which would make the
        // "stay positive" assumption below a coin flip instead of a fact
        // about the code under test.
        let mut backbone = identity_trunk_backbone();
        let ops = [OpKind::Add, OpKind::Mul, OpKind::Sub, OpKind::Div];
        for &op in &ops {
            backbone.embeddings.e[op] = [1.0; K];
        }
        let mut head = SaturationHead::new();
        for i in 0..K {
            head.candidate_w1[i] = [1.0; HIDDEN_DIM];
        }

        let hidden = head.forward_candidate(&backbone, &ops, 0.0);

        let scale = 1.0 / sqrtf(ops.len() as f32);
        // Each op's embedding sums to K (all-ones, K dims); candidate_w1 rows
        // are all-ones too, so each hidden column accumulates `K * scale`
        // once per op in the neighborhood.
        let expected = ops.len() as f32 * (K as f32) * scale;
        assert!(
            expected > 0.0,
            "fixture must stay in the ReLU's positive branch"
        );
        for (j, &h) in hidden.iter().enumerate() {
            assert!(
                (h - expected).abs() < 1e-2,
                "hidden[{j}]: got {h}, expected {expected}"
            );
        }
    }

    #[test]
    fn compute_candidate_embed_should_index_the_projection_by_both_hidden_lane_and_column() {
        const LANE: usize = 2;
        const COLUMN: usize = 6;

        let mut head = SaturationHead::new();
        head.candidate_proj_w[LANE][COLUMN] = 4.0;
        let mut hidden = [0.0f32; HIDDEN_DIM];
        hidden[LANE] = 3.0;

        let embed = head.compute_candidate_embed(&hidden);

        for (k, &v) in embed.iter().enumerate() {
            let want = if k == COLUMN { 12.0 } else { 0.0 };
            assert!(
                (v - want).abs() < 1e-6,
                "embed[{k}]: got {v}, expected {want}"
            );
        }
    }

    #[test]
    fn score_candidate_should_compose_forward_candidate_embed_and_bilinear_score() {
        let backbone = test_backbone();
        let mut head = SaturationHead::new();
        head.randomize(13);

        let neighborhood = [OpKind::Sqrt, OpKind::Neg];
        let rule_embed = [0.3f32; EMBED_DIM];

        let got = head.score_candidate(&backbone, &neighborhood, 0.4, &rule_embed);

        let hidden = head.forward_candidate(&backbone, &neighborhood, 0.4);
        let embed = head.compute_candidate_embed(&hidden);
        let mask_features = head.compute_mask_features(&embed);
        let want = head.bilinear_score(&mask_features, &rule_embed);

        assert!(
            (got - want).abs() < 1e-6,
            "score_candidate must be forward_candidate -> compute_candidate_embed \
             -> compute_mask_features -> bilinear_score; got {got}, want {want}"
        );
    }

    #[test]
    fn score_candidate_should_distinguish_different_budget_fractions() {
        let backbone = test_backbone();
        let mut head = SaturationHead::new();
        head.randomize(21);
        let neighborhood = [OpKind::Add];
        let rule_embed = [0.5f32; EMBED_DIM];

        let low = head.score_candidate(&backbone, &neighborhood, 0.0, &rule_embed);
        let high = head.score_candidate(&backbone, &neighborhood, 1.0, &rule_embed);
        assert!(
            (low - high).abs() > 1e-6,
            "distinct budget fractions should score differently: low={low}, high={high}"
        );
    }
}
