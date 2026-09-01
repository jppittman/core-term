//! # Analytical Backward Pass Through ExprNnue's Value (Extraction) Head
//!
//! Hand-derived gradients through the ExprNnue forward path:
//!
//! ```text
//! Extraction: EdgeAccumulator → W1 → ReLU → trunk → ReLU → expr_proj → value_mlp → value_pred
//! ```
//!
//! ## Why This Exists
//!
//! `ExprNnue` used to also carry a saturation (mask/policy) head sharing this
//! trunk, and this module's `forward_cached` ran both towers in one pass —
//! its signature took a `GraphAccumulator`/`rule_embed` that the value loss
//! never used. The policy path's backward half — REINFORCE with an
//! advantage, entropy-bonus regularization, `backward_policy` — was deleted
//! per docs/plans/2026-07-07-guided-saturation-redesign.md: that estimator
//! was methodologically unsound (deterministic policy scored as if it were
//! sampled, advantage collapse, censored failures) and its trained policy was
//! never consumed by the compiler. The 2026-08-17 cost-model domain-model
//! reorganization finished that split: the saturation head's weights moved
//! out of `ExprNnue` entirely, into `pixelflow_search::nnue::guide` (private,
//! its own checkpoint format, inert until Phase 3), so this trainer's
//! gradient surface — [`UnifiedGradients`] — now mirrors exactly the
//! parameters it can train: the shared backbone (embeddings, `w1`, `b1`,
//! trunk) and the value head. It cannot see, and does not need a dummy value
//! for, anything behind `SaturationGuide`.
//!
//! What remains is the value loss: MSE against ground-truth JIT cost,
//! chain-ruled through `value_mlp` → `expr_proj` → trunk → `w1` (edge tower).
//! This is what `bootstrap_extraction_head` trains with. A second loss,
//! [`backward_pairwise`] (Round 2b, `train_contrastive`), shares the same
//! tail through `backward_from_d_value` — it differs from `backward_value`
//! only in supplying two `d_value`s (one per side of a ranked pair, opposite
//! sign) instead of one, so everything below applies to it identically.
//!
//! ## Op embeddings train through the recorded edge stream
//!
//! The chain no longer stops at `w1`: [`backward_value`] takes the sample's
//! [`EdgeTrace`] — the typed record of the accumulator walk — computes
//! `d_acc_input` through the edge tower, and hands it to
//! [`backward_through_accumulator`], which differentiates the same fold
//! [`EdgeTrace::realize`] runs forward. So `grads.d_embeddings` is real, and
//! [`apply_unified_sgd`] applies it (momentum + clip, deliberately no weight
//! decay — see the embeddings block there). [`backward_pairwise`] takes ONE
//! trace per side and calls the same accumulator backward for both, so a
//! contrastive run trains embeddings too — the "frozen reference
//! embeddings, shared across every lambda" framing `train_contrastive`'s own
//! module doc used to describe (written when this file's `backward_value`/
//! `backward_pairwise` had no accumulator-backward path at all) no longer
//! holds; see that binary's updated doc for what changed operationally.
//!
//! History (P1(a), docs/plans/2026-08-17-cost-model-domain.md): before the
//! trace existed, nothing produced `d_embeddings`, yet SGD decayed and
//! unit-sphere-projected the table every batch — drift with no signal, and a
//! train/DEV feature skew because `ValueSample` cached its accumulator
//! against the pre-drift table. Both halves are now structural non-issues:
//! embeddings move only under a real gradient, and the trainer keeps only
//! the trace, realizing features from the LIVE table on every forward
//! (`dev_and_train_path_features_match_when_realized_from_live_embeddings`).

#![expect(
    clippy::needless_range_loop,
    reason = "the hand-derived tensor equations use indices shared across several fixed-size arrays"
)]

use pixelflow_ir::OpKind;
use pixelflow_ir::kind::OpMap;
use pixelflow_search::nnue::factored::{
    EMBED_DIM, EdgeAccumulator, EdgeTrace, ExprNnue, HIDDEN_DIM, INPUT_DIM, K, MLP_HIDDEN,
};

// ============================================================================
// Forward Cache
// ============================================================================

/// All intermediate activations from a forward pass, cached for backprop.
///
/// Every tensor that participates in the chain rule is stored here.
/// This avoids recomputing activations during the backward pass.
pub struct UnifiedForwardCache {
    /// Backbone input: [`EdgeAccumulator::extraction_input`] — scaled
    /// acc.values[0..128] + the 4 variance-histogram fractions = 132 floats.
    /// Identical by construction to what the deployed extractor feeds
    /// `forward_expr_only` (train/deploy skew guard, 2026-08 round 0).
    pub acc_input: [f32; INPUT_DIM],
    /// Pre-ReLU edge tower hidden: b1 + W1^T @ acc_input.
    pub edge_tower_pre_relu: [f32; HIDDEN_DIM],
    /// Post-ReLU edge tower output (pre-trunk): max(0, edge_tower_pre_relu).
    pub edge_tower_out: [f32; HIDDEN_DIM],
    /// Pre-ReLU shared trunk output (edge path): trunk_b + trunk_w^T @ edge_tower_out.
    pub edge_trunk_pre_relu: [f32; HIDDEN_DIM],
    /// Post-trunk ReLU output for edge path: max(0, edge_trunk_pre_relu).
    pub hidden: [f32; HIDDEN_DIM],
    /// Expression embedding: expr_proj_b + expr_proj_w^T @ hidden.
    pub expr_embed: [f32; EMBED_DIM],
    /// Value MLP pre-ReLU: value_mlp_b1 + value_mlp_w1^T @ expr_embed.
    pub value_h_pre: [f32; MLP_HIDDEN],
    /// Value MLP post-ReLU.
    pub value_h: [f32; MLP_HIDDEN],
    /// Scalar value prediction.
    pub value_pred: f32,
}

// ============================================================================
// Forward Cached
// ============================================================================

/// Replicate the ExprNnue forward pass, caching every intermediate activation.
///
/// This mirrors the exact computation in:
/// - `ExprNnue::forward_expr_only` (layer 1 — consuming the SAME
///   `EdgeAccumulator::extraction_input` vector, so the trained function and
///   the deployed function cannot diverge in feature semantics)
/// - `ExprNnue::compute_expr_embed` (layer 2)
/// - `ExprNnue::value_mlp_forward` (layer 3, private on `ExprNnue`; inlined
///   here since the backward pass needs the intermediate activations)
#[must_use]
pub fn forward_cached(net: &ExprNnue, acc: &EdgeAccumulator) -> UnifiedForwardCache {
    // ---- Extraction-head input: the ONE shared feature constructor ----
    let acc_input = acc.extraction_input();

    // ---- Edge Tower ----
    let mut edge_tower_pre_relu = net.b1;
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            edge_tower_pre_relu[j] += acc_input[i] * net.w1[i][j];
        }
    }

    let mut edge_tower_out = edge_tower_pre_relu;
    for h in &mut edge_tower_out {
        *h = h.max(0.0);
    }

    // ---- Shared trunk ----
    let mut edge_trunk_pre_relu = net.trunk_b;
    for i in 0..HIDDEN_DIM {
        for j in 0..HIDDEN_DIM {
            edge_trunk_pre_relu[j] += edge_tower_out[i] * net.trunk_w[i][j];
        }
    }
    let mut hidden = edge_trunk_pre_relu;
    for h in &mut hidden {
        *h = h.max(0.0);
    }

    // ---- Layer 2: Expr Projection ----
    let mut expr_embed = net.expr_proj_b;
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            expr_embed[k] += hidden[j] * net.expr_proj_w[j][k];
        }
    }

    // ---- Layer 3: Value MLP ----
    let mut value_h_pre = net.value_mlp_b1;
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            value_h_pre[j] += expr_embed[i] * net.value_mlp_w1[i][j];
        }
    }

    let mut value_h = value_h_pre;
    for h in &mut value_h {
        *h = h.max(0.0);
    }

    let mut value_pred = net.value_mlp_b2;
    for j in 0..MLP_HIDDEN {
        value_pred += value_h[j] * net.value_mlp_w2[j];
    }

    UnifiedForwardCache {
        acc_input,
        edge_tower_pre_relu,
        edge_tower_out,
        edge_trunk_pre_relu,
        hidden,
        expr_embed,
        value_h_pre,
        value_h,
        value_pred,
    }
}

// ============================================================================
// Gradient Buffer
// ============================================================================

/// Gradient accumulator mirroring every trainable parameter in `ExprNnue`.
///
/// Exactly the parameters this trainer can train: the shared backbone
/// (embeddings, `w1`/`b1`, trunk) and the value head. Nothing here mirrors
/// `nnue::guide`'s saturation-head weights — that module is a separate
/// checkpoint with its own (as-yet-nonexistent) trainer, and mirroring its
/// fields here without a gradient producer was the confirmed defect this
/// split closes: `apply_unified_sgd` used to weight-decay those
/// randomly-initialized, never-trained fields toward zero on every run.
pub struct UnifiedGradients {
    /// Backbone weight gradients: INPUT_DIM x HIDDEN_DIM.
    pub d_w1: [[f32; HIDDEN_DIM]; INPUT_DIM],
    /// Backbone bias gradients: HIDDEN_DIM.
    pub d_b1: [f32; HIDDEN_DIM],
    /// Expr projection weight gradients: HIDDEN_DIM x EMBED_DIM.
    pub d_expr_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Expr projection bias gradients: EMBED_DIM.
    pub d_expr_proj_b: [f32; EMBED_DIM],
    /// Value MLP layer 1 weight gradients: EMBED_DIM x MLP_HIDDEN.
    pub d_value_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    /// Value MLP layer 1 bias gradients: MLP_HIDDEN.
    pub d_value_mlp_b1: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 weight gradients: MLP_HIDDEN.
    pub d_value_mlp_w2: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 bias gradients: scalar.
    pub d_value_mlp_b2: f32,
    /// OpEmbedding gradients: one K-vector per op.
    pub d_embeddings: OpMap<[f32; K]>,
    /// Shared trunk weight gradients: HIDDEN_DIM x HIDDEN_DIM.
    pub d_trunk_w: [[f32; HIDDEN_DIM]; HIDDEN_DIM],
    /// Shared trunk bias gradients: HIDDEN_DIM.
    pub d_trunk_b: [f32; HIDDEN_DIM],
}

/// Per-group gradient clipping diagnostics.
///
/// `raw_norm` is the full pre-clip gradient norm. `clipped_norm` is the norm of
/// the gradient actually fed to momentum before weight decay is added.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientClipStats {
    pub raw_norm: f32,
    pub clipped_norm: f32,
    pub backbone_norm: f32,
    pub value_norm: f32,
    pub embeddings_norm: f32,
    pub trunk_norm: f32,
    pub backbone_scale: f32,
    pub value_scale: f32,
    pub embeddings_scale: f32,
    pub trunk_scale: f32,
}

impl UnifiedGradients {
    /// Create a zero-initialized gradient buffer.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            d_w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            d_b1: [0.0; HIDDEN_DIM],
            d_expr_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            d_expr_proj_b: [0.0; EMBED_DIM],
            d_value_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            d_value_mlp_b1: [0.0; MLP_HIDDEN],
            d_value_mlp_w2: [0.0; MLP_HIDDEN],
            d_value_mlp_b2: 0.0,
            d_embeddings: OpMap::splat([0.0; K]),
            d_trunk_w: [[0.0; HIDDEN_DIM]; HIDDEN_DIM],
            d_trunk_b: [0.0; HIDDEN_DIM],
        }
    }

    /// Scale all gradients by a constant factor.
    pub fn scale(&mut self, s: f32) {
        for row in &mut self.d_w1 {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_b1 {
            *v *= s;
        }
        for row in &mut self.d_expr_proj_w {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_expr_proj_b {
            *v *= s;
        }
        for row in &mut self.d_value_mlp_w1 {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_value_mlp_b1 {
            *v *= s;
        }
        for v in &mut self.d_value_mlp_w2 {
            *v *= s;
        }
        self.d_value_mlp_b2 *= s;
        for row in self.d_embeddings.as_mut_slice() {
            for v in row {
                *v *= s;
            }
        }
        for row in &mut self.d_trunk_w {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_trunk_b {
            *v *= s;
        }
    }

    /// L2 norm of the entire gradient vector.
    #[must_use]
    pub fn norm(&self) -> f32 {
        let mut sum = 0.0f64;
        for row in &self.d_w1 {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_b1 {
            sum += (v as f64) * (v as f64);
        }
        for row in &self.d_expr_proj_w {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_expr_proj_b {
            sum += (v as f64) * (v as f64);
        }
        for row in &self.d_value_mlp_w1 {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_value_mlp_b1 {
            sum += (v as f64) * (v as f64);
        }
        for &v in &self.d_value_mlp_w2 {
            sum += (v as f64) * (v as f64);
        }
        sum += (self.d_value_mlp_b2 as f64) * (self.d_value_mlp_b2 as f64);
        for row in self.d_embeddings.as_slice() {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for row in &self.d_trunk_w {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_trunk_b {
            sum += (v as f64) * (v as f64);
        }
        libm::sqrt(sum) as f32
    }

    /// L2 norm of the shared expr backbone (w1, b1, expr_proj_w, expr_proj_b).
    pub fn norm_backbone(&self) -> f32 {
        let mut sum = 0.0f64;
        for row in &self.d_w1 {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_b1 {
            sum += (v as f64) * (v as f64);
        }
        for row in &self.d_expr_proj_w {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_expr_proj_b {
            sum += (v as f64) * (v as f64);
        }
        libm::sqrt(sum) as f32
    }

    /// L2 norm of the extraction head (value_mlp_w1, b1, w2, b2).
    pub fn norm_value_head(&self) -> f32 {
        let mut sum = 0.0f64;
        for row in &self.d_value_mlp_w1 {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_value_mlp_b1 {
            sum += (v as f64) * (v as f64);
        }
        for &v in &self.d_value_mlp_w2 {
            sum += (v as f64) * (v as f64);
        }
        sum += (self.d_value_mlp_b2 as f64) * (self.d_value_mlp_b2 as f64);
        libm::sqrt(sum) as f32
    }

    /// L2 norm of the op embeddings table.
    pub fn norm_embeddings(&self) -> f32 {
        let mut sum = 0.0f64;
        for row in self.d_embeddings.as_slice() {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        libm::sqrt(sum) as f32
    }

    /// L2 norm of the shared trunk (trunk_w, trunk_b).
    pub fn norm_trunk(&self) -> f32 {
        let mut sum = 0.0f64;
        for row in &self.d_trunk_w {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_trunk_b {
            sum += (v as f64) * (v as f64);
        }
        libm::sqrt(sum) as f32
    }

    /// Compute clipping scales and the norm of the clipped gradient update.
    #[must_use]
    pub fn clip_stats(&self, max_norm: f32) -> GradientClipStats {
        let backbone_norm = self.norm_backbone();
        let value_norm = self.norm_value_head();
        let embeddings_norm = self.norm_embeddings();
        let trunk_norm = self.norm_trunk();

        let backbone_scale = group_clip_scale(backbone_norm, max_norm);
        let value_scale = group_clip_scale(value_norm, max_norm);
        let embeddings_scale = group_clip_scale(embeddings_norm, max_norm);
        let trunk_scale = group_clip_scale(trunk_norm, max_norm);

        let clipped_sq = [
            backbone_norm * backbone_scale,
            value_norm * value_scale,
            embeddings_norm * embeddings_scale,
            trunk_norm * trunk_scale,
        ]
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>();

        GradientClipStats {
            raw_norm: self.norm(),
            clipped_norm: libm::sqrt(clipped_sq) as f32,
            backbone_norm,
            value_norm,
            embeddings_norm,
            trunk_norm,
            backbone_scale,
            value_scale,
            embeddings_scale,
            trunk_scale,
        }
    }

    /// Accumulate another gradient buffer into this one (element-wise add).
    pub fn accumulate(&mut self, other: &Self) {
        for i in 0..INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                self.d_w1[i][j] += other.d_w1[i][j];
            }
        }
        for j in 0..HIDDEN_DIM {
            self.d_b1[j] += other.d_b1[j];
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.d_expr_proj_w[j][k] += other.d_expr_proj_w[j][k];
            }
        }
        for k in 0..EMBED_DIM {
            self.d_expr_proj_b[k] += other.d_expr_proj_b[k];
        }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                self.d_value_mlp_w1[i][j] += other.d_value_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            self.d_value_mlp_b1[j] += other.d_value_mlp_b1[j];
        }
        for j in 0..MLP_HIDDEN {
            self.d_value_mlp_w2[j] += other.d_value_mlp_w2[j];
        }
        self.d_value_mlp_b2 += other.d_value_mlp_b2;
        for op in OpKind::all() {
            for i in 0..K {
                self.d_embeddings[op][i] += other.d_embeddings[op][i];
            }
        }
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                self.d_trunk_w[i][j] += other.d_trunk_w[i][j];
            }
        }
        for j in 0..HIDDEN_DIM {
            self.d_trunk_b[j] += other.d_trunk_b[j];
        }
    }
}

// ============================================================================
// Backward Pass: Value Loss (MSE)
// ============================================================================

/// The value-loss objective for one sample: the label and its loss weight.
#[derive(Clone, Copy, Debug)]
pub struct ValueObjective {
    /// Ground-truth log-nanosecond cost.
    pub target_log_ns: f32,
    /// Loss weight (`value_coeff`).
    pub coeff: f32,
}

/// Backprop value loss through value_mlp → expr_proj → backbone → embeddings.
///
/// Loss = (value_pred - target_log_ns)^2 * coeff
/// d_value = 2.0 * (value_pred - target_log_ns) * coeff
///
/// `trace` is the typed record of the walk that produced `cache.acc_input`
/// ([`EdgeAccumulator::from_arena_dag_traced`]) — requiring it here is what
/// forces every caller to carry feature provenance, so the embedding
/// gradient is differentiated through exactly the fold the forward ran.
pub fn backward_value(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    trace: &EdgeTrace,
    objective: ValueObjective,
    grads: &mut UnifiedGradients,
) {
    let ValueObjective {
        target_log_ns,
        coeff,
    } = objective;
    let d_value = (2.0 * (cache.value_pred - target_log_ns) * coeff).clamp(-10.0, 10.0);
    backward_from_d_value(net, cache, trace, d_value, grads);
}

/// Backprop an upstream scalar gradient `d_value` (= dL/d(value_pred))
/// through value_mlp → expr_proj → backbone → embeddings.
///
/// This is the tail every scalar-output loss on `value_pred` shares —
/// [`backward_value`] (regression MSE) and [`backward_pairwise`] (ranking
/// hinge, below) differ only in what `d_value` (and which sample's `trace`)
/// they hand it, not in how it propagates from there. Both flow into
/// `grads.d_embeddings` through the same [`backward_through_accumulator`]
/// call — there is no cached-accumulator/frozen-embeddings tier left for
/// either loss to opt out of.
fn backward_from_d_value(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    trace: &EdgeTrace,
    d_value: f32,
    grads: &mut UnifiedGradients,
) {
    // ---- Value MLP backward ----
    grads.d_value_mlp_b2 += d_value;

    let mut d_value_h = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        d_value_h[m] = d_value * net.value_mlp_w2[m];
        grads.d_value_mlp_w2[m] += d_value * cache.value_h[m];
    }

    // ReLU gate
    let mut d_value_h_pre = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        d_value_h_pre[m] = if cache.value_h_pre[m] > 0.0 {
            d_value_h[m]
        } else {
            0.0
        };
    }

    // value_h_pre = value_mlp_b1 + value_mlp_w1^T @ expr_embed
    let mut d_expr_embed = [0.0f32; EMBED_DIM];
    for m in 0..MLP_HIDDEN {
        grads.d_value_mlp_b1[m] += d_value_h_pre[m];
        for k in 0..EMBED_DIM {
            d_expr_embed[k] += d_value_h_pre[m] * net.value_mlp_w1[k][m];
            grads.d_value_mlp_w1[k][m] += d_value_h_pre[m] * cache.expr_embed[k];
        }
    }

    // ---- Expr proj + backbone backward ----
    let d_acc_input = backward_expr_proj_and_backbone(net, cache, &d_expr_embed, grads);

    // ---- Through the accumulator fold into the embedding table ----
    backward_through_accumulator(&d_acc_input, trace, grads);
}

// ============================================================================
// Backward Pass: Pairwise Ranking (Round 2b, contrastive objective)
// ============================================================================

/// Margin-zero pairwise hinge ("contrastive") loss over one ordered pair of
/// forward caches from the SAME e-graph/base expression: `cheaper` is the
/// member whose measured `target_log_ns` is lower than `pricier`'s.
///
/// # Why hinge, and why margin zero
///
/// The task allows margin or logistic (RankNet-style); hinge is chosen
/// because it saturates: once the model ranks the pair correctly by ANY
/// amount, `violation <= 0` and the term contributes exactly zero loss and
/// zero gradient. A logistic pairwise loss never reaches zero gradient, so
/// two loss terms would keep pulling `value_pred` in different directions
/// (the MSE term toward calibrated absolute log-ns, the ranking term toward
/// ever-larger separation) even on pairs the model already orders correctly
/// — an argument for a *second* hyperparameter (its temperature) on top of
/// `lambda`. A zero margin (rather than a positive one sized to the noise
/// floor) keeps this a one-hyperparameter addition to the existing MSE loss,
/// which is what the task's lambda sweep is actually measuring; a nonzero
/// margin is deliberately not introduced as a second axis to sweep.
///
/// `cheaper`/`pricier` must already be pairs the minting pipeline judged
/// orderable (measured `|delta|` at/above the run's noise floor,
/// docs/plans/2026-08-05-egraph-nnue-research-workflow.md §4) — a pair
/// UNDER the floor is not training signal and the caller must not construct
/// one here (see `training/variant_set.rs` and the noise-floor filtering in
/// `bin/train_contrastive.rs`).
///
/// Loss = max(0, pred(cheaper) − pred(pricier)) · lambda
///
/// Each side's [`PairwiseSide::trace`] is the same per-sample provenance
/// [`backward_value`] requires, for the same reason: each side's forward
/// pass realized its features from the LIVE [`ExprNnue::embeddings`] table
/// (`PairwiseSide::cache` is the resulting cache), so the pairwise gradient
/// flows into `grads.d_embeddings` through both sides' recorded edge
/// streams, not just the shared trunk. There is no longer a
/// frozen-embeddings tier for this loss to skip that path through — see the
/// module docs' `EdgeTrace` discussion.
///
/// Returns the UNWEIGHTED hinge violation (`max(0, pred(cheaper) -
/// pred(pricier))`, before `lambda` is applied) for the caller's loss
/// reporting; `0.0` exactly when the pair was already correctly ordered
/// (which is also when this function contributes no gradient).
pub fn backward_pairwise(
    net: &ExprNnue,
    cheaper: PairwiseSide<'_>,
    pricier: PairwiseSide<'_>,
    lambda: f32,
    grads: &mut UnifiedGradients,
) -> f32 {
    let violation = cheaper.cache.value_pred - pricier.cache.value_pred;
    if violation <= 0.0 {
        // Already correctly ordered: hinge is flat here, zero gradient by
        // design (see the function doc for why zero margin, not a "close
        // enough" epsilon — this is an exact comparison of two f32
        // predictions from the same forward pass shape, not a measurement).
        return 0.0;
    }
    // d(violation * lambda)/d(pred(cheaper)) = +lambda
    // d(violation * lambda)/d(pred(pricier)) = -lambda
    backward_from_d_value(net, cheaper.cache, cheaper.trace, lambda, grads);
    backward_from_d_value(net, pricier.cache, pricier.trace, -lambda, grads);
    violation
}

/// One side of a [`backward_pairwise`] comparison: a forward cache paired
/// with the [`EdgeTrace`] that produced it. Grouped into one type rather
/// than passed as two more parameters — `backward_pairwise` needs both
/// halves for BOTH sides of the pair, and a bare `(&UnifiedForwardCache,
/// &EdgeTrace)` tuple would let a caller silently swap a `cheaper` cache
/// with a `pricier` trace, which `PairwiseSide` field names rule out.
#[derive(Clone, Copy)]
pub struct PairwiseSide<'a> {
    pub cache: &'a UnifiedForwardCache,
    pub trace: &'a EdgeTrace,
}

// ============================================================================
// Shared: Expr Projection + Backbone Backward
// ============================================================================

/// Backprop from d_expr_embed through expr_proj, shared trunk, and edge tower.
///
/// Chain: d_expr_embed -> expr_proj backward -> d_hidden -> trunk backward
///        -> d_tower_out -> edge tower backward (d_w1, d_acc_input)
///
/// Returns `d_acc_input`, the gradient w.r.t. the network's input vector,
/// for [`backward_through_accumulator`] to push into the embedding table.
fn backward_expr_proj_and_backbone(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    d_expr_embed: &[f32; EMBED_DIM],
    grads: &mut UnifiedGradients,
) -> [f32; INPUT_DIM] {
    // ---- expr_proj backward ----
    // expr_embed = expr_proj_b + expr_proj_w^T @ hidden
    let mut d_hidden = [0.0f32; HIDDEN_DIM];
    for k in 0..EMBED_DIM {
        grads.d_expr_proj_b[k] += d_expr_embed[k];
    }
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            d_hidden[j] += d_expr_embed[k] * net.expr_proj_w[j][k];
            grads.d_expr_proj_w[j][k] += d_expr_embed[k] * cache.hidden[j];
        }
    }

    // ---- Trunk backward (edge path) ----
    // hidden = ReLU(edge_trunk_pre_relu), edge_trunk_pre_relu = trunk_b + trunk_w^T @ edge_tower_out
    let mut d_trunk_pre = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        d_trunk_pre[j] = if cache.edge_trunk_pre_relu[j] > 0.0 {
            d_hidden[j]
        } else {
            0.0
        };
    }
    for j in 0..HIDDEN_DIM {
        grads.d_trunk_b[j] += d_trunk_pre[j];
    }
    let mut d_tower_out = [0.0f32; HIDDEN_DIM];
    for i in 0..HIDDEN_DIM {
        for j in 0..HIDDEN_DIM {
            grads.d_trunk_w[i][j] += d_trunk_pre[j] * cache.edge_tower_out[i];
            d_tower_out[i] += d_trunk_pre[j] * net.trunk_w[i][j];
        }
    }

    // ---- Edge tower backward ----
    // edge_tower_out = ReLU(edge_tower_pre_relu), edge_tower_pre_relu = b1 + w1^T @ acc_input
    backward_edge_tower_from_hidden(net, cache, &d_tower_out, grads)
}

/// Backprop through edge tower only, starting from d_tower_out.
///
/// Returns `d_acc_input = w1 @ d_pre_relu` — the gradient w.r.t. the input
/// vector itself, which the embedding backward consumes.
fn backward_edge_tower_from_hidden(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    d_tower_out: &[f32; HIDDEN_DIM],
    grads: &mut UnifiedGradients,
) -> [f32; INPUT_DIM] {
    // ReLU gate
    let mut d_pre_relu = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        d_pre_relu[j] = if cache.edge_tower_pre_relu[j] > 0.0 {
            d_tower_out[j]
        } else {
            0.0
        };
    }
    for j in 0..HIDDEN_DIM {
        grads.d_b1[j] += d_pre_relu[j];
    }
    let mut d_acc_input = [0.0f32; INPUT_DIM];
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            grads.d_w1[i][j] += d_pre_relu[j] * cache.acc_input[i];
            d_acc_input[i] += d_pre_relu[j] * net.w1[i][j];
        }
    }
    d_acc_input
}

// ============================================================================
// Embedding Backward: d_acc_input → d_embeddings
// ============================================================================

/// Flow gradients from d_acc_input through the accumulator fold to OpEmbeddings.
///
/// Given the gradient w.r.t. the scaled accumulator input (d_acc_input), this
/// differentiates the fold [`EdgeTrace::realize`] runs forward — the same
/// typed [`CostEdge`](pixelflow_search::nnue::factored::CostEdge) stream, so
/// there is nothing to decode and nothing to re-validate: a `CostEdge` is
/// ops and a PE row by construction.
///
/// The forward path is:
/// ```text
/// for each CostEdge { parent, child, pe }:
///   values[0..K]     += E[parent]
///   values[K..2K]    += E[child]
///   values[2K..3K]   += complex_mul(E[parent], pe)
///   values[3K..4K]   += complex_mul(E[child], pe)
/// acc_input[i] = values[i] * scale   (scale = 1/sqrt(node_count))
/// ```
pub fn backward_through_accumulator(
    d_acc_input: &[f32; INPUT_DIM],
    trace: &EdgeTrace,
    grads: &mut UnifiedGradients,
) {
    // Undo the sqrt(node_count) scaling: d_values[i] = d_acc_input[i] * scale
    let scale = if trace.node_count() > 0 {
        1.0 / libm::sqrtf(trace.node_count() as f32)
    } else {
        1.0
    };

    let mut d_values = [0.0f32; 4 * K];
    for i in 0..4 * K {
        d_values[i] = d_acc_input[i] * scale;
    }
    // d_acc_input[4*K..4*K+4] are the variance-histogram fractions — they
    // don't depend on embeddings, so we skip them.

    for edge in trace.edges() {
        let pi = edge.parent;
        let ci = edge.child;
        let pe = edge.pe.pe();

        // Flat parent half: values[i] += parent_emb[i]
        // d_parent_emb[i] += d_values[i]
        for i in 0..K {
            grads.d_embeddings[pi][i] += d_values[i];
        }

        // Flat child half: values[K+i] += child_emb[i]
        for i in 0..K {
            grads.d_embeddings[ci][i] += d_values[K + i];
        }

        // Depth-encoded parent half (complex multiply backward):
        // Forward: values[2K+2f]   += p_re * cos_d - p_im * sin_d
        //          values[2K+2f+1] += p_re * sin_d + p_im * cos_d
        // Backward: d_p_re += dv_re * cos_d + dv_im * sin_d
        //           d_p_im += -dv_re * sin_d + dv_im * cos_d
        for f in 0..K / 2 {
            let sin_d = pe[2 * f];
            let cos_d = pe[2 * f + 1];
            let dv_re = d_values[2 * K + 2 * f];
            let dv_im = d_values[2 * K + 2 * f + 1];
            grads.d_embeddings[pi][2 * f] += dv_re * cos_d + dv_im * sin_d;
            grads.d_embeddings[pi][2 * f + 1] += -dv_re * sin_d + dv_im * cos_d;
        }

        // Depth-encoded child half:
        for f in 0..K / 2 {
            let sin_d = pe[2 * f];
            let cos_d = pe[2 * f + 1];
            let dv_re = d_values[3 * K + 2 * f];
            let dv_im = d_values[3 * K + 2 * f + 1];
            grads.d_embeddings[ci][2 * f] += dv_re * cos_d + dv_im * sin_d;
            grads.d_embeddings[ci][2 * f + 1] += -dv_re * sin_d + dv_im * cos_d;
        }
    }
}

// ============================================================================
// SGD with Momentum + Weight Decay + Gradient Clipping
// ============================================================================

/// Compute the L2-norm clip scale for a group.
///
/// Returns `max_norm / norm` when the group norm exceeds `max_norm`, else `1.0`.
/// Clipping each semantic group independently prevents large gradients in one
/// path (e.g. the graph backbone) from crowding out updates in other paths.
#[inline]
fn group_clip_scale(norm: f32, max_norm: f32) -> f32 {
    assert!(
        max_norm.is_finite() && max_norm >= 0.0,
        "grad_clip must be finite and non-negative, got {max_norm}"
    );
    assert!(
        norm.is_finite(),
        "gradient norm must be finite before clipping, got {norm}"
    );
    if norm > max_norm {
        max_norm / norm
    } else {
        1.0
    }
}

/// Apply unified SGD update to all trainable parameters.
///
/// Uses **per-group L2 norm clipping**: each semantic group (expr backbone,
/// extraction head, saturation head, graph backbone, embeddings, trunk) is
/// clipped to `grad_clip` independently, so a gradient explosion in one path
/// cannot suppress updates in others.  Direction is preserved within each group.
///
/// For each parameter p (after per-group clipping):
/// ```text
/// momentum_buf = momentum * momentum_buf + grad + weight_decay * param
/// param -= lr * momentum_buf
/// ```
/// Hyperparameters for one unified SGD update.
#[derive(Clone, Copy, Debug)]
pub struct SgdConfig {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
}

pub fn apply_unified_sgd(
    net: &mut ExprNnue,
    grads: &UnifiedGradients,
    momentum_buf: &mut UnifiedGradients,
    config: SgdConfig,
) {
    let SgdConfig {
        lr,
        momentum,
        weight_decay,
        grad_clip,
    } = config;
    // Per-group L2 norm clipping.  Each semantic pathway is clipped
    // independently so an explosion in one group cannot suppress others.
    let clip_stats = grads.clip_stats(grad_clip);
    let scale_backbone = clip_stats.backbone_scale;
    let scale_value = clip_stats.value_scale;
    let scale_embeddings = clip_stats.embeddings_scale;
    let scale_trunk = clip_stats.trunk_scale;

    // Macro to apply SGD update to a single scalar parameter.
    // $scale is the per-group clip scale for this parameter.
    macro_rules! sgd_scalar {
        ($param:expr, $grad:expr, $mbuf:expr, $scale:expr) => {{
            let clipped = $grad * $scale;
            $mbuf = momentum * $mbuf + clipped + weight_decay * $param;
            $param -= lr * $mbuf;
        }};
    }

    // ── Expr backbone (scale_backbone) ───────────────────────────────────────

    // w1: [INPUT_DIM][HIDDEN_DIM]
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            sgd_scalar!(
                net.w1[i][j],
                grads.d_w1[i][j],
                momentum_buf.d_w1[i][j],
                scale_backbone
            );
        }
    }

    // b1: [HIDDEN_DIM]
    for j in 0..HIDDEN_DIM {
        sgd_scalar!(
            net.b1[j],
            grads.d_b1[j],
            momentum_buf.d_b1[j],
            scale_backbone
        );
    }

    // expr_proj_w: [HIDDEN_DIM][EMBED_DIM]
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            sgd_scalar!(
                net.expr_proj_w[j][k],
                grads.d_expr_proj_w[j][k],
                momentum_buf.d_expr_proj_w[j][k],
                scale_backbone
            );
        }
    }

    // expr_proj_b: [EMBED_DIM]
    for k in 0..EMBED_DIM {
        sgd_scalar!(
            net.expr_proj_b[k],
            grads.d_expr_proj_b[k],
            momentum_buf.d_expr_proj_b[k],
            scale_backbone
        );
    }

    // ── Extraction head (scale_value) ────────────────────────────────────────

    // value_mlp_w1: [EMBED_DIM][MLP_HIDDEN]
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            sgd_scalar!(
                net.value_mlp_w1[i][j],
                grads.d_value_mlp_w1[i][j],
                momentum_buf.d_value_mlp_w1[i][j],
                scale_value
            );
        }
    }

    // value_mlp_b1: [MLP_HIDDEN]
    for j in 0..MLP_HIDDEN {
        sgd_scalar!(
            net.value_mlp_b1[j],
            grads.d_value_mlp_b1[j],
            momentum_buf.d_value_mlp_b1[j],
            scale_value
        );
    }

    // value_mlp_w2: [MLP_HIDDEN]
    for j in 0..MLP_HIDDEN {
        sgd_scalar!(
            net.value_mlp_w2[j],
            grads.d_value_mlp_w2[j],
            momentum_buf.d_value_mlp_w2[j],
            scale_value
        );
    }

    // value_mlp_b2: scalar
    sgd_scalar!(
        net.value_mlp_b2,
        grads.d_value_mlp_b2,
        momentum_buf.d_value_mlp_b2,
        scale_value
    );

    // ── Embeddings (scale_embeddings; NO weight decay) ───────────────────────
    //
    // `backward_value` now differentiates through the accumulator fold
    // (`backward_through_accumulator` over the sample's `EdgeTrace`), so
    // `grads.d_embeddings` carries a real signal and the P1(a) freeze is
    // lifted: momentum + per-group clipping, same as every other group.
    //
    // Weight decay is deliberately EXCLUDED for this group. Dimension 0 of
    // each op's embedding is initialized to its measured-latency prior — a
    // full-scale feature, not a near-zero weight — and L2 decay pulls
    // exactly that dimension toward zero hardest, eroding the one
    // initialization signal the table carries (the same erosion P1(a)
    // documented, now with a gradient that would mask it). Standard practice
    // for embedding tables (no decay on lookups) agrees. Revisit only with
    // evidence of embedding overfit, and prefer decay toward the PRIOR, not
    // toward zero.
    //
    // The P1(a) invariant survives in its sharpened form: with a zero
    // gradient this block moves nothing (no decay term exists to drift on),
    // pinned by `embeddings_hold_still_under_zero_gradient` below.
    for op in OpKind::all() {
        for i in 0..K {
            let clipped = grads.d_embeddings[op][i] * scale_embeddings;
            momentum_buf.d_embeddings[op][i] =
                momentum * momentum_buf.d_embeddings[op][i] + clipped;
            net.embeddings.e[op][i] -= lr * momentum_buf.d_embeddings[op][i];
        }
    }

    // ── Shared trunk (scale_trunk) ───────────────────────────────────────────

    // trunk_w: [HIDDEN_DIM][HIDDEN_DIM]
    for i in 0..HIDDEN_DIM {
        for j in 0..HIDDEN_DIM {
            sgd_scalar!(
                net.trunk_w[i][j],
                grads.d_trunk_w[i][j],
                momentum_buf.d_trunk_w[i][j],
                scale_trunk
            );
        }
    }

    // trunk_b: [HIDDEN_DIM]
    for j in 0..HIDDEN_DIM {
        sgd_scalar!(
            net.trunk_b[j],
            grads.d_trunk_b[j],
            momentum_buf.d_trunk_b[j],
            scale_trunk
        );
    }

    // Still no post-SGD embedding normalization, even now that a real
    // gradient producer exists: a unit-sphere projection rescales every
    // dimension uniformly, which flattens the measured-latency prior in
    // dimension 0 relative to the noise dimensions — the same erosion the
    // no-decay comment above declines. If embedding norms ever need
    // controlling, control them in the update (projected gradient descent),
    // never as a blanket post-pass over dimensions the gradient did not touch.
}

// ============================================================================
// Tests: Numerical Gradient Checking
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LCG for deterministic random initialization.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_add(12345))
        }

        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        }
    }

    /// Initialize a network with small random weights everywhere.
    fn make_test_net() -> ExprNnue {
        let mut net = ExprNnue::new();

        let mut rng = Lcg::new(9999);
        let scale_input = libm::sqrtf(2.0 / INPUT_DIM as f32);
        let scale_hidden = libm::sqrtf(2.0 / HIDDEN_DIM as f32);
        let scale_embed = libm::sqrtf(2.0 / EMBED_DIM as f32);

        for i in 0..INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                net.w1[i][j] = rng.next_f32() * scale_input;
            }
        }
        for j in 0..HIDDEN_DIM {
            net.b1[j] = rng.next_f32() * 0.1;
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                net.expr_proj_w[j][k] = rng.next_f32() * scale_hidden;
            }
        }
        for k in 0..EMBED_DIM {
            net.expr_proj_b[k] = rng.next_f32() * 0.1;
        }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                net.value_mlp_w1[i][j] = rng.next_f32() * scale_embed;
            }
        }
        for j in 0..MLP_HIDDEN {
            net.value_mlp_b1[j] = rng.next_f32() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            net.value_mlp_w2[j] = rng.next_f32() * libm::sqrtf(2.0 / MLP_HIDDEN as f32);
        }
        net.value_mlp_b2 = rng.next_f32() * 0.5;

        // Randomize shared trunk: identity + small noise for smooth transition
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                net.trunk_w[i][j] = if i == j { 1.0 } else { 0.0 } + rng.next_f32() * 0.01;
            }
        }
        // trunk_b: zeros (already zero from new(), but be explicit)
        for j in 0..HIDDEN_DIM {
            net.trunk_b[j] = 0.0;
        }

        net
    }

    /// Create a test accumulator with nonzero values.
    ///
    /// The variance fractions are nonzero so the scalar-slot rows of `w1`
    /// (4K..4K+4) actually participate in the gradient checks.
    fn make_test_acc() -> EdgeAccumulator {
        let mut acc = EdgeAccumulator::new();
        let mut rng = Lcg::new(7777);
        for v in &mut acc.values {
            *v = rng.next_f32() * 0.5;
        }
        acc.edge_count = 5;
        acc.node_count = 4;
        acc.variance_frac_const = 0.25;
        acc.variance_frac_frame = 0.25;
        acc.variance_frac_scanline = 0.25;
        acc.variance_frac_pixel = 0.25;
        acc
    }

    /// Compute value loss: (value_pred - target)^2 * value_coeff.
    fn value_loss(net: &ExprNnue, acc: &EdgeAccumulator, target: (f32, f32)) -> f64 {
        let (target_cost, value_coeff) = target;
        let cache = forward_cached(net, acc);
        let diff = cache.value_pred as f64 - target_cost as f64;
        diff * diff * value_coeff as f64
    }

    /// Value loss with features realized from the net's LIVE embeddings —
    /// the trainer's actual forward. Required wherever the perturbed
    /// parameter is an embedding, since perturbing one changes the features.
    fn value_loss_traced(net: &ExprNnue, trace: &EdgeTrace, target: (f32, f32)) -> f64 {
        let acc = trace.realize(&net.embeddings);
        value_loss(net, &acc, target)
    }

    /// An arena covering every feature the walker can emit: all four
    /// variance classes (const / frame / scanline / pixel) so the histogram
    /// slots are nonzero, plus a shared node so a register-reload edge
    /// (child = Var) is on record.
    ///
    /// Returns its own random embedding table (make_test_net leaves
    /// embeddings zero, and its LCG stream must not be perturbed — the
    /// weight-path FD test's step sizes were verified against that exact
    /// stream). Callers install it with `net.embeddings = emb`.
    fn make_test_trace() -> (
        pixelflow_search::nnue::OpEmbeddings,
        EdgeAccumulator,
        EdgeTrace,
    ) {
        use pixelflow_ir::ExprArena;
        let mut arena = ExprArena::new();
        let x = arena.push_var(0); // pixel
        let y = arena.push_var(1); // scanline
        let z = arena.push_var(2); // frame
        let w = arena.push_var(3); // frame
        let zw = arena.push_binary(OpKind::Add, z, w); // frame
        let yy = arena.push_binary(OpKind::Mul, y, y); // scanline
        let s = arena.push_unary(OpKind::Sqrt, yy); // scanline
        let c = arena.push_const(2.0); // const
        let p = arena.push_binary(OpKind::Add, x, zw); // pixel
        let q = arena.push_binary(OpKind::Mul, p, s); // pixel
        let shared = arena.push_binary(OpKind::Add, q, c); // pixel
        let root = arena.push_binary(OpKind::Mul, shared, shared); // pixel + reload

        let emb = pixelflow_search::nnue::OpEmbeddings::new_random(4242);
        let (acc, trace) = EdgeAccumulator::from_arena_dag_traced(&arena, root, &emb);
        (emb, acc, trace)
    }

    /// Check analytical gradient against numerical gradient for a single parameter.
    ///
    /// Returns (analytical, numerical, relative_error).
    fn check_gradient(analytical: f32, numerical: f64) -> (f32, f64, f64) {
        let a = analytical as f64;
        let n = numerical;
        let denom = a.abs() + n.abs() + 1e-8;
        let rel_err = (a - n).abs() / denom;
        (analytical, numerical, rel_err)
    }

    // ========================================================================
    // Test 4: Value path gradients
    // ========================================================================

    #[test]
    fn numerical_gradient_check_value() {
        // A dozen `net.clone()` sites × ~136KB of inline arrays: debug builds
        // give each site its own stack slot (no slot reuse below opt-level 1),
        // which overflows the 2MB default test-thread stack as a SIGSEGV, not
        // a failed assertion. Run the body on a thread with room to spare.
        std::thread::Builder::new()
            .name("numerical_gradient_check_value".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(numerical_gradient_check_value_body)
            .expect("failed to spawn gradient-check thread")
            .join()
            .expect("gradient-check thread panicked");
    }

    fn numerical_gradient_check_value_body() {
        let net = make_test_net();
        let acc = make_test_acc();
        let target_cost = 3.5f32;
        let value_coeff = 0.5f32;

        let cache = forward_cached(&net, &acc);
        let mut grads = Box::new(UnifiedGradients::zero());
        backward_value(
            &net,
            &cache,
            &EdgeTrace::default(),
            ValueObjective {
                target_log_ns: target_cost,
                coeff: value_coeff,
            },
            &mut grads,
        );

        let eps = 1e-3f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;

        // value_mlp_w2
        for j in [0, 8, 15] {
            let mut net_p = net.clone();
            net_p.value_mlp_w2[j] += eps;
            let loss_plus = value_loss(&net_p, &acc, (target_cost, value_coeff));

            let mut net_m = net.clone();
            net_m.value_mlp_w2[j] -= eps;
            let loss_minus = value_loss(&net_m, &acc, (target_cost, value_coeff));

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
            let (a, n, err) = check_gradient(grads.d_value_mlp_w2[j], num_grad);
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 0.05,
                "value_mlp_w2[{j}]: analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
            );
            checked += 1;
        }

        // value_mlp_b2
        {
            let mut net_p = net.clone();
            net_p.value_mlp_b2 += eps;
            let loss_plus = value_loss(&net_p, &acc, (target_cost, value_coeff));

            let mut net_m = net.clone();
            net_m.value_mlp_b2 -= eps;
            let loss_minus = value_loss(&net_m, &acc, (target_cost, value_coeff));

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
            let (a, n, err) = check_gradient(grads.d_value_mlp_b2, num_grad);
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 0.05,
                "value_mlp_b2: analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
            );
            checked += 1;
        }

        // value_mlp_w1
        for i in [0, 12, 23] {
            for j in [0, 8, 15] {
                let mut net_p = net.clone();
                net_p.value_mlp_w1[i][j] += eps;
                let loss_plus = value_loss(&net_p, &acc, (target_cost, value_coeff));

                let mut net_m = net.clone();
                net_m.value_mlp_w1[i][j] -= eps;
                let loss_minus = value_loss(&net_m, &acc, (target_cost, value_coeff));

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_value_mlp_w1[i][j], num_grad);
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "value_mlp_w1[{i}][{j}]: analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        // expr_proj_w (value path)
        //
        // This block uses a larger step than the rest of the test. The forward
        // pass is f32, so the central-difference numerator carries a constant
        // absolute roundoff noise of ~ulp(loss)/(2*eps) (~1.5e-4 at eps=1e-3),
        // which swamps small true gradients (~1.7e-4 for expr_proj_w[32][12])
        // and made this assertion flaky (rel_err 0.086 in June 2026, 0.101 in
        // July 2026; the June "fix" loosened the threshold instead of fixing
        // the step size). With a frozen ReLU pattern the loss is exactly
        // quadratic in any single weight, so the central difference has zero
        // truncation error and a larger eps strictly improves the
        // signal-to-roundoff ratio, provided no ReLU flips sign. An
        // expr_proj_w perturbation only reaches the value-MLP ReLUs, and none
        // flip at eps=1e-2 for this net/input (verified; unlike trunk_b,
        // where eps=1e-2 does cross a kink, so the global eps stays 1e-3).
        // Measured rel_err on the once-failing parameter:
        // 0.38 @ eps=1e-4, 0.10 @ 1e-3, 0.005 @ 3e-3, 0.001 @ 1e-2.
        let proj_eps = 1e-2f32;
        for j in [0, 32, 63] {
            for k in [0, 12, 23] {
                let mut net_p = net.clone();
                net_p.expr_proj_w[j][k] += proj_eps;
                let loss_plus = value_loss(&net_p, &acc, (target_cost, value_coeff));

                let mut net_m = net.clone();
                net_m.expr_proj_w[j][k] -= proj_eps;
                let loss_minus = value_loss(&net_m, &acc, (target_cost, value_coeff));

                let num_grad = (loss_plus - loss_minus) / (2.0 * proj_eps as f64);
                let (a, n, err) = check_gradient(grads.d_expr_proj_w[j][k], num_grad);
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "expr_proj_w[{j}][{k}] (value): analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        // trunk_w (value path through shared trunk)
        for i in [0, 32, 63] {
            for j in [0, 32, 63] {
                let mut net_p = net.clone();
                net_p.trunk_w[i][j] += eps;
                let loss_plus = value_loss(&net_p, &acc, (target_cost, value_coeff));

                let mut net_m = net.clone();
                net_m.trunk_w[i][j] -= eps;
                let loss_minus = value_loss(&net_m, &acc, (target_cost, value_coeff));

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_trunk_w[i][j], num_grad);
                if err > max_err {
                    max_err = err;
                }
                let abs_diff = (a as f64 - n).abs();
                assert!(
                    err < 0.05 || abs_diff < 1e-5,
                    "trunk_w[{i}][{j}] (value): analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}, abs_diff={abs_diff:.6e}"
                );
                checked += 1;
            }
        }

        // trunk_b (value path through shared trunk)
        for j in [0, 32, 63] {
            let mut net_p = net.clone();
            net_p.trunk_b[j] += eps;
            let loss_plus = value_loss(&net_p, &acc, (target_cost, value_coeff));

            let mut net_m = net.clone();
            net_m.trunk_b[j] -= eps;
            let loss_minus = value_loss(&net_m, &acc, (target_cost, value_coeff));

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
            let (a, n, err) = check_gradient(grads.d_trunk_b[j], num_grad);
            if err > max_err {
                max_err = err;
            }
            let abs_diff = (a as f64 - n).abs();
            assert!(
                err < 0.05 || abs_diff < 1e-5,
                "trunk_b[{j}] (value): analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}, abs_diff={abs_diff:.6e}"
            );
            checked += 1;
        }

        // w1 (value path through edge tower)
        for i in [0, 64, 129] {
            for j in [0, 32, 63] {
                let mut net_p = net.clone();
                net_p.w1[i][j] += eps;
                let loss_plus = value_loss(&net_p, &acc, (target_cost, value_coeff));

                let mut net_m = net.clone();
                net_m.w1[i][j] -= eps;
                let loss_minus = value_loss(&net_m, &acc, (target_cost, value_coeff));

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_w1[i][j], num_grad);
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "w1[{i}][{j}] (value): analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        assert!(checked >= 38, "checked {checked} value path elements");
        eprintln!("  value path max rel error: {max_err:.6e}  ({checked} elements)");
    }

    // ========================================================================
    // Test 6: forward_cached's acc_input IS EdgeAccumulator::extraction_input
    // ========================================================================

    #[test]
    fn forward_cached_acc_input_matches_extraction_input() {
        let net = make_test_net();
        let acc = make_test_acc();

        let cache = forward_cached(&net, &acc);

        // Verify acc_input IS the shared extraction_input vector — the whole
        // train/deploy skew guard is that this cannot be built any other way.
        let expected = acc.extraction_input();
        for i in 0..INPUT_DIM {
            assert!(
                (cache.acc_input[i] - expected[i]).abs() < 1e-6,
                "acc_input[{i}] must equal EdgeAccumulator::extraction_input"
            );
        }
        // And spot-check the documented layout: scaled values, then the
        // variance histogram (NOT log2 counts — that was the round-0 skew).
        let scale = if acc.node_count > 0 {
            1.0 / libm::sqrtf(acc.node_count as f32)
        } else {
            1.0
        };
        for i in 0..4 * K {
            assert!(
                (cache.acc_input[i] - acc.values[i] * scale).abs() < 1e-6,
                "acc_input[{i}] should match scaled acc.values"
            );
        }
        assert!(
            (cache.acc_input[4 * K] - acc.variance_frac_const).abs() < 1e-6,
            "acc_input[4K] must be variance_frac_const"
        );
        assert!(
            (cache.acc_input[4 * K + 1] - acc.variance_frac_frame).abs() < 1e-6,
            "acc_input[4K+1] must be variance_frac_frame"
        );
        assert!(
            (cache.acc_input[4 * K + 2] - acc.variance_frac_scanline).abs() < 1e-6,
            "acc_input[4K+2] must be variance_frac_scanline"
        );
        assert!(
            (cache.acc_input[4 * K + 3] - acc.variance_frac_pixel).abs() < 1e-6,
            "acc_input[4K+3] must be variance_frac_pixel"
        );
    }

    // ========================================================================
    // Test 6b: the trained forward IS the deployed forward
    // ========================================================================

    /// `forward_cached` (what the trainer optimizes) and
    /// `predict_log_cost_with_features` (what `IncrementalExtractor` calls at
    /// extraction time) must produce the same prediction for the same
    /// accumulator. This is the forward-pass half of the train/deploy skew
    /// guard: if either side grows a feature the other lacks, this fails.
    #[test]
    fn forward_cached_value_matches_deployed_prediction() {
        let net = make_test_net();
        let acc = make_test_acc();

        let cache = forward_cached(&net, &acc);
        let deployed = net.predict_log_cost_with_features(&acc);

        assert!(
            (cache.value_pred - deployed).abs() < 1e-6,
            "trained-path prediction {} != deployed-path prediction {} — \
             the trainer is optimizing a function the extractor does not call",
            cache.value_pred,
            deployed
        );
    }

    // ========================================================================
    // Test 7: Gradient norm is nonzero
    // ========================================================================

    #[test]
    fn gradient_norm_nonzero() {
        let net = make_test_net();
        let acc = make_test_acc();

        let cache = forward_cached(&net, &acc);

        let mut grads = UnifiedGradients::zero();
        backward_value(
            &net,
            &cache,
            &EdgeTrace::default(),
            ValueObjective {
                target_log_ns: 3.0,
                coeff: 1.0,
            },
            &mut grads,
        );
        let value_norm = grads.norm();
        assert!(
            value_norm > 1e-8,
            "value gradient norm should be nonzero, got {value_norm}"
        );

        eprintln!("  value grad norm:  {value_norm:.6}");
    }

    // ========================================================================
    // Test 8: SGD actually moves parameters
    // ========================================================================

    #[test]
    fn sgd_moves_parameters() {
        let mut net = make_test_net();
        let acc = make_test_acc();

        let w1_before = net.w1[0][0];
        let trunk_w_before = net.trunk_w[0][0];

        let cache = forward_cached(&net, &acc);
        let mut grads = UnifiedGradients::zero();
        backward_value(
            &net,
            &cache,
            &EdgeTrace::default(),
            ValueObjective {
                target_log_ns: 3.0,
                coeff: 0.5,
            },
            &mut grads,
        );

        let mut momentum_buf = UnifiedGradients::zero();
        apply_unified_sgd(
            &mut net,
            &grads,
            &mut momentum_buf,
            SgdConfig {
                lr: 0.01,
                momentum: 0.9,
                weight_decay: 1e-4,
                grad_clip: 1.0,
            },
        );

        let w1_after = net.w1[0][0];
        let trunk_w_after = net.trunk_w[0][0];

        // w1 (edge tower) and trunk_w (shared) are both on the value loss's
        // chain and should move.
        assert!(
            (w1_after - w1_before).abs() > 1e-10,
            "w1[0][0] should have moved: before={w1_before}, after={w1_after}"
        );
        assert!(
            (trunk_w_after - trunk_w_before).abs() > 1e-10,
            "trunk_w[0][0] should have moved: before={trunk_w_before}, after={trunk_w_after}"
        );
    }

    #[test]
    fn clip_stats_include_shared_trunk() {
        let mut grads = UnifiedGradients::zero();
        grads.d_w1[0][0] = 2.0;
        grads.d_trunk_w[0][0] = 10.0;

        let stats = grads.clip_stats(1.0);

        assert!((stats.raw_norm - libm::sqrt(104.0) as f32).abs() < 1e-6);
        assert!((stats.backbone_norm - 2.0).abs() < 1e-6);
        assert!((stats.trunk_norm - 10.0).abs() < 1e-6);
        assert!((stats.backbone_scale - 0.5).abs() < 1e-6);
        assert!((stats.trunk_scale - 0.1).abs() < 1e-6);
        assert!((stats.clipped_norm - libm::sqrt(2.0) as f32).abs() < 1e-6);
    }

    #[test]
    fn sgd_clips_shared_trunk_update() {
        let mut net = make_test_net();
        let mut grads = UnifiedGradients::zero();
        let mut momentum_buf = UnifiedGradients::zero();

        let before = net.trunk_w[0][0];
        grads.d_trunk_w[0][0] = 10.0;

        apply_unified_sgd(
            &mut net,
            &grads,
            &mut momentum_buf,
            SgdConfig {
                lr: 1.0,
                momentum: 0.0,
                weight_decay: 0.0,
                grad_clip: 1.0,
            },
        );

        let delta = net.trunk_w[0][0] - before;
        assert!(
            (delta + 1.0).abs() < 1e-6,
            "trunk update should be clipped to -1.0, got {delta}"
        );
    }

    // ========================================================================
    // Test 8b: embeddings move under their gradient, and ONLY under it
    // ========================================================================

    #[test]
    fn embeddings_hold_still_under_zero_gradient() {
        // The sharpened P1(a) invariant: with `d_embeddings == 0`, SGD must
        // leave the table bit-identical — no weight decay, no projection, no
        // drift — even while every other group trains with nonzero decay.
        let mut net = make_test_net();
        net.embeddings = pixelflow_search::nnue::OpEmbeddings::new_random(7);
        let init = net.embeddings.clone();

        let mut momentum_buf = UnifiedGradients::zero();
        let config = SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 1e-2,
            grad_clip: 5.0,
        };

        for _ in 0..50u32 {
            let mut grads = UnifiedGradients::zero();
            // Nonzero everywhere else, so this exercises the same
            // multi-group clipping/momentum path a real batch does.
            grads.d_w1[0][0] = 1.0;
            grads.d_trunk_w[3][4] = -2.0;
            grads.d_value_mlp_b2 = 0.7;
            apply_unified_sgd(&mut net, &grads, &mut momentum_buf, config);
        }

        assert_eq!(
            net.embeddings.e.as_slice(),
            init.e.as_slice(),
            "embeddings must be bit-identical after zero-gradient batches: no decay term \
             may exist for this group (it would erode the latency prior in dim 0)"
        );
    }

    #[test]
    fn embeddings_follow_gradient_and_ignore_weight_decay() {
        // With momentum 0 and a gradient under the clip norm, the update must
        // be exactly -lr * grad — in particular UNAFFECTED by weight_decay,
        // which is applied to every other group but excluded here.
        let mut net = make_test_net();
        net.embeddings = pixelflow_search::nnue::OpEmbeddings::new_random(7);
        let op = OpKind::Mul;
        let before = net.embeddings.e[op][3];

        let mut grads = UnifiedGradients::zero();
        grads.d_embeddings[op][3] = 2.0;

        let mut momentum_buf = UnifiedGradients::zero();
        apply_unified_sgd(
            &mut net,
            &grads,
            &mut momentum_buf,
            SgdConfig {
                lr: 0.5,
                momentum: 0.0,
                weight_decay: 1e-2, // must NOT reach the embeddings
                grad_clip: 100.0,
            },
        );

        let delta = net.embeddings.e[op][3] - before;
        assert!(
            (delta + 1.0).abs() < 1e-6,
            "embedding update must be exactly -lr*grad = -1.0 (no decay term), got {delta}"
        );
    }

    #[test]
    fn dev_and_train_path_features_match_when_realized_from_live_embeddings() {
        // P1(a)'s successor invariant. The trainer keeps only the
        // `EdgeTrace` and realizes features from the LIVE embeddings on
        // every forward — so after training moves the table, the train-path
        // features and a fresh DEV-path walk of the same arena must still
        // agree exactly. (Under the old cached-accumulator scheme this was
        // only true because embeddings were frozen; now it is true because
        // staleness is unrepresentable.)
        use pixelflow_ir::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let sum = arena.push_binary(OpKind::Add, x, one);
        let two = arena.push_const(2.0);
        let root = arena.push_binary(OpKind::Mul, sum, two);

        let mut net = make_test_net();
        net.embeddings = pixelflow_search::nnue::OpEmbeddings::new_random(7);
        let init_embeddings = net.embeddings.clone();

        // TRAIN-path: the trace is captured once (as `bootstrap_extraction_head`
        // does at label minting)…
        let (_, trace) = EdgeAccumulator::from_arena_dag_traced(&arena, root, &net.embeddings);

        let mut momentum_buf = UnifiedGradients::zero();
        let config = SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 1e-2,
            grad_clip: 5.0,
        };
        for _ in 0..20 {
            // …but features are REALIZED fresh each step.
            let acc = trace.realize(&net.embeddings);
            let cache = forward_cached(&net, &acc);
            let mut grads = UnifiedGradients::zero();
            backward_value(
                &net,
                &cache,
                &trace,
                ValueObjective {
                    target_log_ns: 4.0,
                    coeff: 1.0,
                },
                &mut grads,
            );
            apply_unified_sgd(&mut net, &grads, &mut momentum_buf, config);
        }

        // Training must actually have moved the table — otherwise this test
        // would pass vacuously, proving frozen-ness rather than liveness.
        assert_ne!(
            net.embeddings.e.as_slice(),
            init_embeddings.e.as_slice(),
            "embedding gradient is wired: 20 steps on a nonzero loss must move the table"
        );

        // DEV-path: a fresh walk of the SAME arena with the trained table.
        let dev_acc = EdgeAccumulator::from_arena_dag(&arena, root, &net.embeddings);

        assert_eq!(
            trace.realize(&net.embeddings).extraction_input(),
            dev_acc.extraction_input(),
            "train-path (trace realized from live embeddings) and DEV-path (fresh walk) \
             features diverged for the same expression"
        );
    }

    // ========================================================================
    // Test 8c: numerical gradient check for the embedding path
    // ========================================================================

    #[test]
    fn numerical_gradient_check_embeddings() {
        // Same stack-size note as `numerical_gradient_check_value`.
        std::thread::Builder::new()
            .name("numerical_gradient_check_embeddings".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(numerical_gradient_check_embeddings_body)
            .expect("failed to spawn gradient-check thread")
            .join()
            .expect("gradient-check thread panicked");
    }

    fn numerical_gradient_check_embeddings_body() {
        let (emb, _, trace) = make_test_trace();
        let mut net = make_test_net();
        net.embeddings = emb;
        let target = (3.5f32, 0.5f32);

        let acc = trace.realize(&net.embeddings);
        let cache = forward_cached(&net, &acc);
        let mut grads = Box::new(UnifiedGradients::zero());
        backward_value(
            &net,
            &cache,
            &trace,
            ValueObjective {
                target_log_ns: target.0,
                coeff: target.1,
            },
            &mut grads,
        );

        // Ops actually present in the trace's arena. Ops absent from it must
        // have exactly zero gradient — checked below.
        let present = [
            OpKind::Add,
            OpKind::Mul,
            OpKind::Sqrt,
            OpKind::Var,
            OpKind::Const,
        ];
        // Dim 0 (flat, latency-prior slot), an even/odd rotation pair, and a
        // high dim — the complex backward differs between even and odd.
        let dims = [0usize, 7, 8, 31];

        let eps = 1e-3f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;

        for &op in &present {
            for &d in &dims {
                let mut net_p = net.clone();
                net_p.embeddings.e[op][d] += eps;
                let loss_plus = value_loss_traced(&net_p, &trace, target);

                let mut net_m = net.clone();
                net_m.embeddings.e[op][d] -= eps;
                let loss_minus = value_loss_traced(&net_m, &trace, target);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_embeddings[op][d], num_grad);
                if err > max_err {
                    max_err = err;
                }
                let abs_diff = (a as f64 - n).abs();
                assert!(
                    err < 0.05 || abs_diff < 1e-5,
                    "d_embeddings[{op:?}][{d}]: analytical={a:.8}, numerical={n:.8}, \
                     rel_err={err:.6}, abs_diff={abs_diff:.6e}"
                );
                checked += 1;
            }
        }

        // An op that never appears in the trace gets no gradient.
        assert_eq!(
            grads.d_embeddings[OpKind::Tan],
            [0.0f32; K],
            "ops absent from the trace must receive exactly zero gradient"
        );

        assert!(checked >= 20, "checked {checked} embedding entries");
        eprintln!("  embedding path max rel error: {max_err:.6e}  ({checked} entries)");
    }

    // ========================================================================
    // Test 8d: Pairwise ranking hinge (Round 2b) — numerical gradient check
    // ========================================================================

    /// Combined pairwise hinge loss for two forward passes sharing one
    /// `net` (matches how `backward_pairwise` is called: two forward
    /// caches, one shared `net`), with BOTH sides' features realized from
    /// the net's LIVE embeddings — the perturbation target for finite
    /// differences, including embedding perturbations (mirrors
    /// `value_loss_traced` above, doubled for the pair).
    fn pairwise_loss_traced(
        net: &ExprNnue,
        trace_cheaper: &EdgeTrace,
        trace_pricier: &EdgeTrace,
        lambda: f32,
    ) -> f64 {
        let acc_cheaper = trace_cheaper.realize(&net.embeddings);
        let acc_pricier = trace_pricier.realize(&net.embeddings);
        let cache_cheaper = forward_cached(net, &acc_cheaper);
        let cache_pricier = forward_cached(net, &acc_pricier);
        (f64::from(cache_cheaper.value_pred - cache_pricier.value_pred)).max(0.0)
            * f64::from(lambda)
    }

    /// A second arena, structurally distinct from [`make_test_trace`]'s —
    /// the pairwise check needs two genuinely different expressions (not
    /// the same trace compared against itself), realized against the SAME
    /// embeddings table `emb` so both sides' `d_embeddings` contributions
    /// land in the finite-difference perturbation together.
    fn make_test_trace_b(emb: &pixelflow_search::nnue::OpEmbeddings) -> EdgeTrace {
        use pixelflow_ir::ExprArena;
        let mut arena = ExprArena::new();
        let x = arena.push_var(0); // pixel
        let y = arena.push_var(1); // scanline
        let c = arena.push_const(3.0); // const
        let xy = arena.push_binary(OpKind::Mul, x, y); // pixel
        let root = arena.push_binary(OpKind::Add, xy, c); // pixel
        let (_, trace) = EdgeAccumulator::from_arena_dag_traced(&arena, root, emb);
        trace
    }

    #[test]
    fn numerical_gradient_check_pairwise() {
        // Same stack-size note as `numerical_gradient_check_value`.
        std::thread::Builder::new()
            .name("numerical_gradient_check_pairwise".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(numerical_gradient_check_pairwise_body)
            .expect("failed to spawn gradient-check thread")
            .join()
            .expect("gradient-check thread panicked");
    }

    fn numerical_gradient_check_pairwise_body() {
        let (emb, _, trace_a) = make_test_trace();
        let trace_b = make_test_trace_b(&emb);
        let mut net = make_test_net();
        net.embeddings = emb;
        let lambda = 0.5f32;

        // Force the hinge into its gradient-carrying regime: whichever side
        // starts with the higher value_pred is "cheaper" in the call, so
        // pred(cheaper) - pred(pricier) > 0 and the early-return does not
        // fire, exercising the actual backward path rather than its zero
        // branch.
        let cache_a0 = forward_cached(&net, &trace_a.realize(&net.embeddings));
        let cache_b0 = forward_cached(&net, &trace_b.realize(&net.embeddings));
        let (trace_cheaper, trace_pricier) = if cache_a0.value_pred > cache_b0.value_pred {
            (&trace_a, &trace_b)
        } else {
            (&trace_b, &trace_a)
        };

        let acc_cheaper = trace_cheaper.realize(&net.embeddings);
        let acc_pricier = trace_pricier.realize(&net.embeddings);
        let cache_cheaper = forward_cached(&net, &acc_cheaper);
        let cache_pricier = forward_cached(&net, &acc_pricier);
        let mut grads = Box::new(UnifiedGradients::zero());
        let violation = backward_pairwise(
            &net,
            PairwiseSide {
                cache: &cache_cheaper,
                trace: trace_cheaper,
            },
            PairwiseSide {
                cache: &cache_pricier,
                trace: trace_pricier,
            },
            lambda,
            &mut grads,
        );
        assert!(
            violation > 0.0,
            "test setup should exercise the gradient-carrying branch, got violation={violation}"
        );

        let eps = 1e-3f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;

        // w1 perturbation affects BOTH forward passes (shared net), so the
        // finite difference must perturb once and re-run both caches — this
        // is exactly what makes the pairwise case a distinct check from the
        // single-cache value-loss test above.
        for i in [0, 64, 129] {
            for j in [0, 32, 63] {
                let mut net_p = net.clone();
                net_p.w1[i][j] += eps;
                let loss_plus = pairwise_loss_traced(&net_p, trace_cheaper, trace_pricier, lambda);

                let mut net_m = net.clone();
                net_m.w1[i][j] -= eps;
                let loss_minus = pairwise_loss_traced(&net_m, trace_cheaper, trace_pricier, lambda);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_w1[i][j], num_grad);
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "w1[{i}][{j}] (pairwise): analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        // value_mlp_w2: direct participant in d_value on both sides.
        for j in [0, 8, 15] {
            let mut net_p = net.clone();
            net_p.value_mlp_w2[j] += eps;
            let loss_plus = pairwise_loss_traced(&net_p, trace_cheaper, trace_pricier, lambda);

            let mut net_m = net.clone();
            net_m.value_mlp_w2[j] -= eps;
            let loss_minus = pairwise_loss_traced(&net_m, trace_cheaper, trace_pricier, lambda);

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
            let (a, n, err) = check_gradient(grads.d_value_mlp_w2[j], num_grad);
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 0.05,
                "value_mlp_w2[{j}] (pairwise): analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
            );
            checked += 1;
        }

        // d_embeddings: the path that did not exist when this test was
        // first written (P1(a) — see the module docs and
        // `backward_pairwise`'s doc). Both traces reference `OpKind::Mul`,
        // so perturbing it moves both forward passes at once, and the
        // analytical gradient must already carry both sides' contributions
        // in this one accumulated entry — exactly what
        // `backward_from_d_value`'s two calls (one per side, opposite
        // sign) are supposed to produce.
        for &d in &[0usize, 7, 31] {
            let mut net_p = net.clone();
            net_p.embeddings.e[OpKind::Mul][d] += eps;
            let loss_plus = pairwise_loss_traced(&net_p, trace_cheaper, trace_pricier, lambda);

            let mut net_m = net.clone();
            net_m.embeddings.e[OpKind::Mul][d] -= eps;
            let loss_minus = pairwise_loss_traced(&net_m, trace_cheaper, trace_pricier, lambda);

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
            let (a, n, err) = check_gradient(grads.d_embeddings[OpKind::Mul][d], num_grad);
            if err > max_err {
                max_err = err;
            }
            let abs_diff = (a as f64 - n).abs();
            assert!(
                err < 0.05 || abs_diff < 1e-5,
                "d_embeddings[Mul][{d}] (pairwise): analytical={a:.8}, numerical={n:.8}, \
                 rel_err={err:.6}, abs_diff={abs_diff:.6e}"
            );
            checked += 1;
        }

        assert!(checked >= 15, "checked {checked} pairwise elements");
        eprintln!("  pairwise path max rel error: {max_err:.6e}  ({checked} elements)");
    }

    #[test]
    fn pairwise_already_ordered_pair_has_zero_gradient() {
        // The other side of the hinge: when `cheaper` already predicts
        // lower than `pricier`, the term must contribute exactly nothing —
        // this is the "coexist with the regression term without a margin
        // hyperparameter" property the function doc claims. Zero gradient
        // must hold for `d_embeddings` too, not just the backbone/head.
        let (emb, _, trace_a) = make_test_trace();
        let trace_b = make_test_trace_b(&emb);
        let mut net = make_test_net();
        net.embeddings = emb;

        let cache_a = forward_cached(&net, &trace_a.realize(&net.embeddings));
        let cache_b = forward_cached(&net, &trace_b.realize(&net.embeddings));
        let (cheaper, cheaper_trace, pricier, pricier_trace) =
            if cache_a.value_pred <= cache_b.value_pred {
                (&cache_a, &trace_a, &cache_b, &trace_b)
            } else {
                (&cache_b, &trace_b, &cache_a, &trace_a)
            };

        let mut grads = UnifiedGradients::zero();
        let violation = backward_pairwise(
            &net,
            PairwiseSide {
                cache: cheaper,
                trace: cheaper_trace,
            },
            PairwiseSide {
                cache: pricier,
                trace: pricier_trace,
            },
            0.5,
            &mut grads,
        );
        assert_eq!(
            violation, 0.0,
            "already-ordered pair must report zero violation"
        );
        assert_eq!(
            grads.norm(),
            0.0,
            "already-ordered pair must contribute exactly zero gradient"
        );
    }

    // ========================================================================
    // Test 9: Scale and accumulate
    // ========================================================================

    #[test]
    fn scale_and_accumulate() {
        let net = make_test_net();
        let acc = make_test_acc();
        let cache = forward_cached(&net, &acc);

        let mut g1 = UnifiedGradients::zero();
        backward_value(
            &net,
            &cache,
            &EdgeTrace::default(),
            ValueObjective {
                target_log_ns: 3.0,
                coeff: 1.0,
            },
            &mut g1,
        );

        let mut g2 = UnifiedGradients::zero();
        backward_value(
            &net,
            &cache,
            &EdgeTrace::default(),
            ValueObjective {
                target_log_ns: 3.0,
                coeff: 1.0,
            },
            &mut g2,
        );

        // g2 should equal g1
        for i in 0..INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                assert!(
                    (g1.d_w1[i][j] - g2.d_w1[i][j]).abs() < 1e-10,
                    "duplicate gradients should match"
                );
            }
        }

        // Scale by 0.5
        g1.scale(0.5);
        let norm_half = g1.norm();

        // Accumulate g2 (still unscaled) into g1
        g1.accumulate(&g2);

        // Now g1 = 0.5*g + g = 1.5*g, so norm should be 1.5 * original
        let norm_orig = g2.norm();
        let expected = 1.5 * norm_orig;
        let actual = g1.norm();
        let rel = (actual - expected).abs() / (expected + 1e-8);
        assert!(
            rel < 0.01,
            "accumulate: expected norm {expected:.6}, got {actual:.6}, rel_err={rel:.6}"
        );
        eprintln!(
            "  scale/accumulate: half_norm={norm_half:.6}, orig_norm={norm_orig:.6}, 1.5x_norm={actual:.6}"
        );
    }
}
