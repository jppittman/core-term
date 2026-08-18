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
//! This is what `bootstrap_extraction_head` trains with.
//!
//! ## Op embeddings are frozen (P1(a))
//!
//! The chain above stops at `w1` — [`backward_value`] never reaches
//! [`backward_through_accumulator`], which is the only function that turns a
//! `d_acc_input` gradient into a `d_embeddings` one, and it has no live
//! caller (grep it). So `grads.d_embeddings` is always exactly zero on the
//! live path. [`apply_unified_sgd`] does not update `net.embeddings` from it
//! — see that function's doc for why running weight decay and unit-sphere
//! projection against an always-zero gradient was the confirmed defect
//! (docs/plans/2026-08-17-cost-model-domain.md, P1(a)) this freeze closes.

#![expect(
    clippy::needless_range_loop,
    reason = "the hand-derived tensor equations use indices shared across several fixed-size arrays"
)]

use pixelflow_ir::OpKind;
use pixelflow_ir::kind::{OpCode, OpMap};
use pixelflow_search::nnue::factored::{
    EMBED_DIM, EdgeAccumulator, ExprNnue, HIDDEN_DIM, INPUT_DIM, K, MLP_HIDDEN, depth_pe,
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

/// Backprop value loss through value_mlp → expr_proj → backbone.
///
/// Loss = (value_pred - target_cost)^2 * value_coeff
/// d_value = 2.0 * (value_pred - target_cost) * value_coeff
pub fn backward_value(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    target_cost: f32,
    value_coeff: f32,
    grads: &mut UnifiedGradients,
) {
    let d_value = (2.0 * (cache.value_pred - target_cost) * value_coeff).clamp(-10.0, 10.0);

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
    backward_expr_proj_and_backbone(net, cache, &d_expr_embed, grads);
}

// ============================================================================
// Shared: Expr Projection + Backbone Backward
// ============================================================================

/// Backprop from d_expr_embed through expr_proj, shared trunk, and edge tower.
///
/// Chain: d_expr_embed -> expr_proj backward -> d_hidden -> trunk backward
///        -> d_tower_out -> edge tower backward (d_w1)
fn backward_expr_proj_and_backbone(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    d_expr_embed: &[f32; EMBED_DIM],
    grads: &mut UnifiedGradients,
) {
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
    backward_edge_tower_from_hidden(cache, &d_tower_out, grads);
}

/// Backprop through edge tower only, starting from d_tower_out.
fn backward_edge_tower_from_hidden(
    cache: &UnifiedForwardCache,
    d_tower_out: &[f32; HIDDEN_DIM],
    grads: &mut UnifiedGradients,
) {
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
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            grads.d_w1[i][j] += d_pre_relu[j] * cache.acc_input[i];
        }
    }
}

// ============================================================================
// Embedding Backward: d_acc_input → d_embeddings
// ============================================================================

/// Flow gradients from d_acc_input through EdgeAccumulator construction to OpEmbeddings.
///
/// Given the gradient w.r.t. the scaled accumulator input (d_acc_input), this function
/// reverses the accumulator construction to compute per-op embedding gradients.
///
/// The forward path is:
/// ```text
/// for each edge (parent, child, depth):
///   values[0..K]     += parent_emb
///   values[K..2K]    += child_emb
///   values[2K..3K]   += complex_mul(parent_emb, PE(depth))
///   values[3K..4K]   += complex_mul(child_emb, PE(depth))
/// acc_input[i] = values[i] * scale   (scale = 1/sqrt(node_count))
/// ```
///
/// # Panics
///
/// Panics if any op byte in `edges` encodes no op.
pub fn backward_through_accumulator(
    d_acc_input: &[f32; INPUT_DIM],
    edges: &[(u8, u8, u16)],
    node_count: u32,
    grads: &mut UnifiedGradients,
) {
    // Undo the sqrt(node_count) scaling: d_values[i] = d_acc_input[i] * scale
    let scale = if node_count > 0 {
        1.0 / libm::sqrtf(node_count as f32)
    } else {
        1.0
    };

    let mut d_values = [0.0f32; 4 * K];
    for i in 0..4 * K {
        d_values[i] = d_acc_input[i] * scale;
    }
    // d_acc_input[4*K..4*K+4] are the variance-histogram fractions — they
    // don't depend on embeddings, so we skip them.

    for &(parent_op_u8, child_op_u8, depth_u16) in edges {
        // Decoded once, loudly. These arrive as raw bytes from a serialized
        // edge record, so "this byte names an op" is a claim about the data,
        // not something the type system already knows.
        let pi = OpKind::unmarshal(OpCode::from_bytes([parent_op_u8]))
            .unwrap_or_else(|| panic!("parent op byte {parent_op_u8} names no OpKind"));
        let ci = OpKind::unmarshal(OpCode::from_bytes([child_op_u8]))
            .unwrap_or_else(|| panic!("child op byte {child_op_u8} names no OpKind"));
        let pe = depth_pe(depth_u16 as u32);

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
    // (Embeddings are not among these groups — see the frozen-embeddings
    // block below.)
    let clip_stats = grads.clip_stats(grad_clip);
    let scale_backbone = clip_stats.backbone_scale;
    let scale_value = clip_stats.value_scale;
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

    // ── Embeddings: FROZEN (P1(a)) ───────────────────────────────────────────
    //
    // `net.embeddings` is deliberately left untouched here — no SGD update,
    // no weight decay, no momentum accumulation — because nothing in this
    // trainer's live call graph computes a gradient for it: `backward_value`
    // never calls `backward_through_accumulator` (the only function that
    // turns a `d_acc_input` gradient into `d_embeddings`), so
    // `grads.d_embeddings` is always exactly zero. Running weight decay
    // against an always-zero gradient is not "no update", it is a slow decay
    // toward zero with no opposing signal — pure drift, not training.
    //
    // That drift was P1(a): `ValueSample::acc` is cached ONCE (from
    // `model.embeddings` before any SGD step), but this function used to
    // mutate `net.embeddings` every batch regardless, so the cached
    // accumulator's baked-in embeddings silently diverged from the live
    // `model.embeddings` used to build DEV-time features
    // (`bootstrap_extraction_head`'s Phase 3) — a train/deploy skew with no
    // training benefit behind it. Freezing embeddings makes both paths read
    // the same (constant) embeddings by construction; see
    // `embeddings_are_frozen_without_a_gradient_producer` and
    // `dev_and_train_path_features_match_after_training_with_frozen_embeddings`
    // below.
    //
    // Unfreeze this block only once a real gradient producer exists (wiring
    // `backward_through_accumulator` into `backward_value`) — mutation
    // without a gradient is the defect, not the freeze.

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

    // No post-SGD embedding normalization: the unit-sphere projection that
    // used to run here unconditionally is itself a mutation, and P1(a)
    // applies to it exactly as it does to the SGD update above — projecting
    // a parameter every batch with no gradient behind it is drift, not
    // training. Re-introduce it (as projected gradient descent on the
    // embeddings' own update, not a blanket post-pass) only alongside a real
    // gradient producer.
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
        backward_value(&net, &cache, target_cost, value_coeff, &mut grads);

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
        backward_value(&net, &cache, 3.0, 1.0, &mut grads);
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
        backward_value(&net, &cache, 3.0, 0.5, &mut grads);

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
    // Test 8b: P1(a) — embeddings are frozen without a gradient producer
    // ========================================================================

    #[test]
    fn embeddings_are_frozen_without_a_gradient_producer() {
        // `backward_value` never populates `d_embeddings` on the live path
        // (only `backward_through_accumulator` would, and it has no live
        // caller — see this module's doc). `apply_unified_sgd` must
        // therefore never decay, project, or otherwise move
        // `net.embeddings`: with no opposing gradient, doing so anyway is
        // pure drift, not training (P1(a)).
        let mut net = make_test_net();
        let init = net.embeddings.clone();

        let mut momentum_buf = UnifiedGradients::zero();
        let config = SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 1e-2,
            grad_clip: 5.0,
        };

        for step in 0..50u32 {
            let mut grads = UnifiedGradients::zero();
            // Nonzero everywhere else, so this exercises the same
            // multi-group clipping/momentum path a real batch does.
            grads.d_w1[0][0] = 1.0;
            grads.d_trunk_w[3][4] = -2.0;
            grads.d_value_mlp_b2 = 0.7;
            // Deliberately nonzero (unlike reality) to prove the freeze is
            // unconditional — a decay-of-an-always-zero-gradient coincidence
            // would not catch a regression that starts reading this field.
            for op in OpKind::all() {
                grads.d_embeddings[op][0] = 3.0 + step as f32;
            }
            apply_unified_sgd(&mut net, &grads, &mut momentum_buf, config);
        }

        assert_eq!(
            net.embeddings.e.as_slice(),
            init.e.as_slice(),
            "embeddings must be bit-identical to init after training batches: \
             apply_unified_sgd must not decay, project, or otherwise update embeddings while \
             nothing computes a gradient for them"
        );
    }

    #[test]
    fn dev_and_train_path_features_match_after_training_with_frozen_embeddings() {
        // P1(a)'s DEV-path/train-path skew: `ValueSample::acc` is built once
        // (train-path snapshot), embeddings decayed/projected every batch
        // with no gradient behind it, and `EdgeAccumulator::from_arena_dag`
        // was rebuilt for DEV against the now-drifted `model.embeddings` —
        // two different embeddings for the same expression. With embeddings
        // frozen, the two paths must read identical features.
        use pixelflow_ir::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let sum = arena.push_binary(OpKind::Add, x, one);
        let two = arena.push_const(2.0);
        let root = arena.push_binary(OpKind::Mul, sum, two);

        let mut net = make_test_net();
        // TRAIN-path snapshot: built once, as `bootstrap_extraction_head`'s
        // Phase 1d does, before any SGD step.
        let train_acc = EdgeAccumulator::from_arena_dag(&arena, root, &net.embeddings);

        let mut momentum_buf = UnifiedGradients::zero();
        let config = SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 1e-2,
            grad_clip: 5.0,
        };
        for _ in 0..20 {
            let cache = forward_cached(&net, &train_acc);
            let mut grads = UnifiedGradients::zero();
            backward_value(&net, &cache, 4.0, 1.0, &mut grads);
            apply_unified_sgd(&mut net, &grads, &mut momentum_buf, config);
        }

        // DEV-path: rebuilt from the SAME arena, AFTER training, against
        // whatever `net.embeddings` training left behind — the deployed
        // prediction path (`bootstrap_extraction_head`'s Phase 3).
        let dev_acc = EdgeAccumulator::from_arena_dag(&arena, root, &net.embeddings);

        assert_eq!(
            train_acc.extraction_input(),
            dev_acc.extraction_input(),
            "train-path and DEV-path features diverged for the same expression — embeddings \
             drifted between accumulator construction and DEV evaluation (P1(a))"
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
        backward_value(&net, &cache, 3.0, 1.0, &mut g1);

        let mut g2 = UnifiedGradients::zero();
        backward_value(&net, &cache, 3.0, 1.0, &mut g2);

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
