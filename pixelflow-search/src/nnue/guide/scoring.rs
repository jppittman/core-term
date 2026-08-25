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
use pixelflow_ir::arena::{ExprArena, ExprId};

use super::accumulator::{GRAPH_ACC_DIM, GraphAccumulator};
use crate::nnue::factored::{
    EMBED_DIM, EdgeAccumulator, ExprNnue, HIDDEN_DIM, MLP_HIDDEN, SCALAR_FEATURE_COUNT,
};

/// Mask MLP input dimension: expr_embed / graph_embed directly (`EMBED_DIM`).
/// `value_pred` was removed — it is a deterministic function of the
/// embedding and adds zero information.
pub(crate) const MASK_INPUT_DIM: usize = EMBED_DIM;

/// Graph backbone input: `GRAPH_ACC_DIM` + 4 scalars (edge_count, node_count,
/// node_budget, epoch_budget).
pub(crate) const GRAPH_INPUT_DIM: usize = GRAPH_ACC_DIM + SCALAR_FEATURE_COUNT;

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

    /// Graph backbone weights: [`GRAPH_INPUT_DIM`] x `HIDDEN_DIM`.
    graph_w1: [[f32; HIDDEN_DIM]; GRAPH_INPUT_DIM],
    /// Graph backbone biases.
    graph_b1: [f32; HIDDEN_DIM],
    /// Graph → embed projection weights: `HIDDEN_DIM` x `EMBED_DIM`.
    graph_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Graph → embed projection bias.
    graph_proj_b: [f32; EMBED_DIM],
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
    use crate::nnue::factored::OpKind;

    fn test_backbone() -> ExprNnue {
        ExprNnue::new_random(42)
    }

    #[test]
    fn bilinear_score_matches_manual_computation_for_all_ones_vectors() {
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
    fn randomize_is_deterministic_and_finite_with_near_identity_diagonal() {
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
    fn forward_graph_uses_backbone_trunk() {
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
    fn encode_rule_from_arena_is_deterministic() {
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
}
