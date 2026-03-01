//! # Full Analytical Backward Pass Through ExprNnue
//!
//! Hand-derived gradients through the ENTIRE ExprNnue forward path with
//! separate backbones for value and policy heads:
//!
//! ```text
//! Value:  EdgeAccumulator  → W1        → expr_proj  → value_mlp → value_pred
//! Policy: GraphAccumulator → graph_w1  → graph_proj → mask_mlp  → bilinear → score
//! ```
//!
//! ## Why This Exists
//!
//! The existing REINFORCE code (`train_mask_reinforce_with_embed` in factored.rs)
//! only backprops through `interaction + mask_bias_proj`. This module extends
//! gradients through the FULL path for both heads.
//!
//! This enables joint training of both the value head (judge) and policy head
//! (mask/guide) through independent backbones, like AlphaZero.
//!
//! ## Two Losses
//!
//! - **Policy loss**: REINFORCE with advantage, chain-ruled through
//!   bilinear → mask_mlp → graph_proj → graph_w1 (graph backbone)
//! - **Value loss**: MSE against ground-truth cost, chain-ruled through
//!   value_mlp → expr_proj → W1 (expr backbone)
//!
//! The two heads have independent backbones: value uses EdgeAccumulator → W1,
//! policy uses GraphAccumulator → graph_w1. No shared backbone gradients.

use pixelflow_ir::OpKind;
use pixelflow_search::nnue::factored::{
    depth_pe, EdgeAccumulator, ExprNnue, GraphAccumulator, EMBED_DIM, GRAPH_ACC_DIM,
    GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K, MLP_HIDDEN,
};

// ============================================================================
// Forward Cache
// ============================================================================

/// All intermediate activations from a forward pass, cached for backprop.
///
/// Every tensor that participates in the chain rule is stored here.
/// This avoids recomputing activations during the backward pass.
pub struct UnifiedForwardCache {
    /// Backbone input: acc.values[0..128] + edge_count + node_count = 130 floats.
    pub acc_input: [f32; INPUT_DIM],
    /// Pre-ReLU backbone hidden: b1 + W1^T @ acc_input.
    pub pre_relu: [f32; HIDDEN_DIM],
    /// Post-ReLU backbone hidden: max(0, pre_relu).
    pub hidden: [f32; HIDDEN_DIM],
    /// Expression embedding: expr_proj_b + expr_proj_w^T @ hidden.
    pub expr_embed: [f32; EMBED_DIM],
    /// Value MLP pre-ReLU: value_mlp_b1 + value_mlp_w1^T @ expr_embed.
    pub value_h_pre: [f32; MLP_HIDDEN],
    /// Value MLP post-ReLU.
    pub value_h: [f32; MLP_HIDDEN],
    /// Scalar value prediction.
    pub value_pred: f32,
    /// Graph backbone input: gacc.values[0..96] + edge_count + node_count = 98 floats.
    pub graph_input: [f32; GRAPH_INPUT_DIM],
    /// Pre-ReLU graph backbone hidden.
    pub graph_pre_relu: [f32; HIDDEN_DIM],
    /// Post-ReLU graph backbone hidden.
    pub graph_hidden: [f32; HIDDEN_DIM],
    /// Graph embedding: graph_proj_b + graph_proj_w^T @ graph_hidden.
    pub graph_embed: [f32; EMBED_DIM],
    /// Mask MLP input: graph_embed (32 dims) — comes from graph backbone, not expr backbone.
    pub mask_input: [f32; EMBED_DIM],
    /// Mask MLP pre-ReLU.
    pub mask_h_pre: [f32; MLP_HIDDEN],
    /// Mask MLP post-ReLU.
    pub mask_h: [f32; MLP_HIDDEN],
    /// Mask features: mask_mlp_b2 + mask_mlp_w2^T @ mask_h.
    pub mask_features: [f32; EMBED_DIM],
    /// Transformed vector: mask_features @ interaction.
    pub transformed: [f32; EMBED_DIM],
    /// Raw bilinear score (pre-sigmoid).
    pub score: f32,
    /// sigmoid(score).
    pub prob: f32,
}

// ============================================================================
// Forward Cached
// ============================================================================

/// Replicate the ExprNnue forward pass, caching every intermediate activation.
///
/// This mirrors the exact computation in:
/// - `ExprNnue::forward_shared` (layer 1)
/// - `ExprNnue::compute_expr_embed` (layer 2)
/// - `ExprNnue::value_mlp_forward` (layer 3a)
/// - `ExprNnue::compute_mask_features` (layer 3b)
/// - `ExprNnue::bilinear_score` (layer 4)
#[must_use]
pub fn forward_cached(
    net: &ExprNnue,
    acc: &EdgeAccumulator,
    gacc: &GraphAccumulator,
    rule_embed: &[f32; EMBED_DIM],
) -> UnifiedForwardCache {
    // ---- Build acc_input from EdgeAccumulator (for value head) ----
    let mut acc_input = [0.0f32; INPUT_DIM];

    let scale = if acc.node_count > 0 {
        1.0 / libm::sqrtf(acc.node_count as f32)
    } else {
        1.0
    };

    for i in 0..4 * K {
        acc_input[i] = acc.values[i] * scale;
    }

    acc_input[4 * K] = libm::log2f(1.0 + acc.edge_count as f32);
    acc_input[4 * K + 1] = libm::log2f(1.0 + acc.node_count as f32);

    // ---- Layer 1: Shared Backbone (for value head) ----
    let mut pre_relu = net.b1;
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            pre_relu[j] += acc_input[i] * net.w1[i][j];
        }
    }

    let mut hidden = pre_relu;
    for h in &mut hidden {
        *h = h.max(0.0);
    }

    // ---- Layer 2: Expr Projection (for value head) ----
    let mut expr_embed = net.expr_proj_b;
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            expr_embed[k] += hidden[j] * net.expr_proj_w[j][k];
        }
    }

    // ---- Layer 3a: Value MLP ----
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

    // ---- Graph Backbone (for mask head) ----
    // Build graph_input from GraphAccumulator: scale by 1/sqrt(node_count),
    // use log2(1+edge_count) and log2(1+node_count) as scalars.
    let mut graph_input = [0.0f32; GRAPH_INPUT_DIM];

    let graph_scale = if gacc.node_count > 0 {
        1.0 / libm::sqrtf(gacc.node_count as f32)
    } else {
        1.0
    };

    for i in 0..GRAPH_ACC_DIM {
        graph_input[i] = gacc.values[i] * graph_scale;
    }

    graph_input[GRAPH_ACC_DIM] = libm::log2f(1.0 + gacc.edge_count as f32);
    graph_input[GRAPH_ACC_DIM + 1] = libm::log2f(1.0 + gacc.node_count as f32);

    // graph_pre_relu = graph_b1 + graph_w1^T @ graph_input
    let mut graph_pre_relu = net.graph_b1;
    for i in 0..GRAPH_INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            graph_pre_relu[j] += graph_input[i] * net.graph_w1[i][j];
        }
    }

    // graph_hidden = max(0, graph_pre_relu) (ReLU)
    let mut graph_hidden = graph_pre_relu;
    for h in &mut graph_hidden {
        *h = h.max(0.0);
    }

    // graph_embed = graph_proj_b + graph_proj_w^T @ graph_hidden
    let mut graph_embed = net.graph_proj_b;
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            graph_embed[k] += graph_hidden[j] * net.graph_proj_w[j][k];
        }
    }

    // ---- Layer 3b: Mask MLP (fed by graph_embed, not expr_embed) ----
    let mask_input = graph_embed;

    let mut mask_h_pre = net.mask_mlp_b1;
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            mask_h_pre[j] += mask_input[i] * net.mask_mlp_w1[i][j];
        }
    }

    let mut mask_h = mask_h_pre;
    for h in &mut mask_h {
        *h = h.max(0.0);
    }

    let mut mask_features = net.mask_mlp_b2;
    for j in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            mask_features[k] += mask_h[j] * net.mask_mlp_w2[j][k];
        }
    }

    // ---- Layer 4: Bilinear Score ----
    let mut transformed = [0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        for j in 0..EMBED_DIM {
            transformed[j] += mask_features[i] * net.interaction[i][j];
        }
    }

    let mut score = 0.0f32;
    for k in 0..EMBED_DIM {
        score += (transformed[k] + net.mask_bias_proj[k]) * rule_embed[k];
    }

    let prob = sigmoid(score);

    UnifiedForwardCache {
        acc_input,
        pre_relu,
        hidden,
        expr_embed,
        value_h_pre,
        value_h,
        value_pred,
        graph_input,
        graph_pre_relu,
        graph_hidden,
        graph_embed,
        mask_input,
        mask_h_pre,
        mask_h,
        mask_features,
        transformed,
        score,
        prob,
    }
}

// ============================================================================
// Gradient Buffer
// ============================================================================

/// Gradient accumulator mirroring every trainable parameter in ExprNnue.
///
/// Both policy and value losses accumulate into the same buffer, so the
/// shared backbone (w1, b1) receives gradients from both heads.
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
    /// Mask MLP layer 1 weight gradients: EMBED_DIM x MLP_HIDDEN.
    pub d_mask_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    /// Mask MLP layer 1 bias gradients: MLP_HIDDEN.
    pub d_mask_mlp_b1: [f32; MLP_HIDDEN],
    /// Mask MLP layer 2 weight gradients: MLP_HIDDEN x EMBED_DIM.
    pub d_mask_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    /// Mask MLP layer 2 bias gradients: EMBED_DIM.
    pub d_mask_mlp_b2: [f32; EMBED_DIM],
    /// Interaction matrix gradients: EMBED_DIM x EMBED_DIM.
    pub d_interaction: [[f32; EMBED_DIM]; EMBED_DIM],
    /// Bias projection gradients: EMBED_DIM.
    pub d_mask_bias_proj: [f32; EMBED_DIM],
    /// OpEmbedding gradients: [OpKind::COUNT][K].
    pub d_embeddings: [[f32; K]; OpKind::COUNT],
    /// Graph backbone weight gradients: GRAPH_INPUT_DIM x HIDDEN_DIM.
    pub d_graph_w1: [[f32; HIDDEN_DIM]; GRAPH_INPUT_DIM],
    /// Graph backbone bias gradients: HIDDEN_DIM.
    pub d_graph_b1: [f32; HIDDEN_DIM],
    /// Graph projection weight gradients: HIDDEN_DIM x EMBED_DIM.
    pub d_graph_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Graph projection bias gradients: EMBED_DIM.
    pub d_graph_proj_b: [f32; EMBED_DIM],
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
            d_mask_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            d_mask_mlp_b1: [0.0; MLP_HIDDEN],
            d_mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            d_mask_mlp_b2: [0.0; EMBED_DIM],
            d_interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            d_mask_bias_proj: [0.0; EMBED_DIM],
            d_embeddings: [[0.0; K]; OpKind::COUNT],
            d_graph_w1: [[0.0; HIDDEN_DIM]; GRAPH_INPUT_DIM],
            d_graph_b1: [0.0; HIDDEN_DIM],
            d_graph_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            d_graph_proj_b: [0.0; EMBED_DIM],
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
        for row in &mut self.d_mask_mlp_w1 {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_mask_mlp_b1 {
            *v *= s;
        }
        for row in &mut self.d_mask_mlp_w2 {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_mask_mlp_b2 {
            *v *= s;
        }
        for row in &mut self.d_interaction {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_mask_bias_proj {
            *v *= s;
        }
        for row in &mut self.d_embeddings {
            for v in row {
                *v *= s;
            }
        }
        for row in &mut self.d_graph_w1 {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_graph_b1 {
            *v *= s;
        }
        for row in &mut self.d_graph_proj_w {
            for v in row {
                *v *= s;
            }
        }
        for v in &mut self.d_graph_proj_b {
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
        for row in &self.d_mask_mlp_w1 {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_mask_mlp_b1 {
            sum += (v as f64) * (v as f64);
        }
        for row in &self.d_mask_mlp_w2 {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_mask_mlp_b2 {
            sum += (v as f64) * (v as f64);
        }
        for row in &self.d_interaction {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_mask_bias_proj {
            sum += (v as f64) * (v as f64);
        }
        for row in &self.d_embeddings {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for row in &self.d_graph_w1 {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_graph_b1 {
            sum += (v as f64) * (v as f64);
        }
        for row in &self.d_graph_proj_w {
            for &v in row {
                sum += (v as f64) * (v as f64);
            }
        }
        for &v in &self.d_graph_proj_b {
            sum += (v as f64) * (v as f64);
        }
        libm::sqrt(sum) as f32
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
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                self.d_mask_mlp_w1[i][j] += other.d_mask_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            self.d_mask_mlp_b1[j] += other.d_mask_mlp_b1[j];
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.d_mask_mlp_w2[j][k] += other.d_mask_mlp_w2[j][k];
            }
        }
        for k in 0..EMBED_DIM {
            self.d_mask_mlp_b2[k] += other.d_mask_mlp_b2[k];
        }
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                self.d_interaction[i][j] += other.d_interaction[i][j];
            }
        }
        for k in 0..EMBED_DIM {
            self.d_mask_bias_proj[k] += other.d_mask_bias_proj[k];
        }
        for op in 0..OpKind::COUNT {
            for i in 0..K {
                self.d_embeddings[op][i] += other.d_embeddings[op][i];
            }
        }
        for i in 0..GRAPH_INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                self.d_graph_w1[i][j] += other.d_graph_w1[i][j];
            }
        }
        for j in 0..HIDDEN_DIM {
            self.d_graph_b1[j] += other.d_graph_b1[j];
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.d_graph_proj_w[j][k] += other.d_graph_proj_w[j][k];
            }
        }
        for k in 0..EMBED_DIM {
            self.d_graph_proj_b[k] += other.d_graph_proj_b[k];
        }
    }
}

// ============================================================================
// Backward Pass: Policy Loss (REINFORCE)
// ============================================================================

/// Backprop policy loss through the FULL path.
///
/// Chain rule: bilinear → mask_mlp → graph_proj → graph backbone W1.
/// (value_mlp and expr backbone are decoupled — only trained via value loss.)
///
/// The policy loss gradient w.r.t. score:
/// ```text
/// Loss = -log(sigmoid(score)) * advantage         if matched
///      = -log(1 - sigmoid(score)) * advantage      if not matched
///
/// d_loss/d_score = -(1-prob) * advantage           if matched
///                = prob * advantage                 if not matched
/// ```
///
/// When `entropy_coeff > 0`, an entropy bonus gradient is added:
/// ```text
/// d_entropy/d_score = entropy_coeff * prob * (1 - prob) * score
/// ```
/// This prevents the policy from collapsing to deterministic 0/1 outputs.
/// Setting `entropy_coeff = 0.0` recovers exact REINFORCE behavior.
///
/// Then standard chain rule through each layer, accumulating weight gradients.
pub fn backward_policy(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    rule_embed: &[f32; EMBED_DIM],
    matched: bool,
    advantage: f32,
    entropy_coeff: f32,
    miss_penalty: f32,
    grads: &mut UnifiedGradients,
) {
    // ---- d_score from REINFORCE ----
    // In our self-play loop, a step is only recorded if `prob > threshold`
    // (i.e. the policy chose Action = 1).
    // The environmental outcome (`matched`) determines the `advantage` value,
    // but the policy gradient must always reflect the action taken (a=1).
    //
    // Loss = -log(prob) * advantage
    // d_loss/d_score = -(1 - prob) * advantage
    //
    // If advantage < 0 (penalty), d_score is positive, meaning gradient descent
    // will subtract a positive value, driving the score (and prob) DOWN.
    //
    // miss_penalty: when the rule didn't match (matched=false), we scale the
    // advantage down to avoid punishing exploration too harshly. A low
    // miss_penalty (e.g. 0.1) means the policy barely gets penalized for
    // trying rules that don't match, encouraging it to explore more rules.
    let effective_advantage = if matched { advantage } else { advantage * miss_penalty };
    let mut d_score = -(1.0 - cache.prob) * effective_advantage;

    // ---- Entropy bonus: prevent policy from collapsing to 0/1 ----
    // H(p) = -[p*log(p) + (1-p)*log(1-p)]  (Bernoulli entropy)
    // dH/d_score = p*(1-p)*log((1-p)/p) = p*(1-p)*(-score)
    // We MAXIMIZE entropy (minimize -H), so gradient contribution is:
    //   d(-H)/d_score = -dH/d_score = p*(1-p)*score
    // This gets ADDED to d_score (gradient descent on L_policy - beta*H).
    if entropy_coeff != 0.0 {
        let entropy_grad = entropy_coeff * cache.prob * (1.0 - cache.prob) * cache.score;
        d_score += entropy_grad;
    }

    // ---- Layer 4: Bilinear backward ----
    // score = sum_k((transformed[k] + mask_bias_proj[k]) * rule_embed[k])
    // where transformed[j] = sum_i(mask_features[i] * interaction[i][j])
    // ∂score/∂bias_proj_k = rule_embed[k]
    for k in 0..EMBED_DIM {
        grads.d_mask_bias_proj[k] += d_score * rule_embed[k];
    }

    let mut d_transformed = [0.0f32; EMBED_DIM];
    for j in 0..EMBED_DIM {
        d_transformed[j] = d_score * rule_embed[j];
    }

    let mut d_mask_features = [0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        for j in 0..EMBED_DIM {
            d_mask_features[i] += d_transformed[j] * net.interaction[i][j];
        }
    }

    // d_interaction[i][j] = d_score * mask_features[i] * rule_embed[j]
    for i in 0..EMBED_DIM {
        for j in 0..EMBED_DIM {
            grads.d_interaction[i][j] +=
                d_score * cache.mask_features[i] * rule_embed[j];
        }
    }

    // ---- Layer 3b: Mask MLP backward ----
    // mask_features = mask_mlp_b2 + mask_mlp_w2^T @ mask_h
    for k in 0..EMBED_DIM {
        grads.d_mask_mlp_b2[k] += d_mask_features[k];
    }

    let mut d_mask_h = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            d_mask_h[m] += d_mask_features[k] * net.mask_mlp_w2[m][k];
            grads.d_mask_mlp_w2[m][k] += d_mask_features[k] * cache.mask_h[m];
        }
    }

    // ReLU gate
    let mut d_mask_h_pre = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        d_mask_h_pre[m] = if cache.mask_h_pre[m] > 0.0 {
            d_mask_h[m]
        } else {
            0.0
        };
    }

    // mask_h_pre = mask_mlp_b1 + mask_mlp_w1^T @ mask_input
    // mask_input is graph_embed (32 dims) — from graph backbone, not expr backbone.
    let mut d_graph_embed = [0.0f32; EMBED_DIM];
    for m in 0..MLP_HIDDEN {
        grads.d_mask_mlp_b1[m] += d_mask_h_pre[m];
        for i in 0..EMBED_DIM {
            d_graph_embed[i] += d_mask_h_pre[m] * net.mask_mlp_w1[i][m];
            grads.d_mask_mlp_w1[i][m] += d_mask_h_pre[m] * cache.mask_input[i];
        }
    }

    // ---- Graph projection backward ----
    // graph_embed = graph_proj_b + graph_proj_w^T @ graph_hidden
    backward_graph_proj_and_backbone(net, cache, &d_graph_embed, grads);
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
    let d_value = 2.0 * (cache.value_pred - target_cost) * value_coeff;

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

/// Backprop from d_expr_embed through expr_proj and backbone.
///
/// Shared by both policy and value backward passes.
fn backward_expr_proj_and_backbone(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    d_expr_embed: &[f32; EMBED_DIM],
    grads: &mut UnifiedGradients,
) {
    // ---- Layer 2: expr_proj backward ----
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

    // ---- Layer 1: Backbone backward ----
    backward_backbone_from_hidden(cache, &d_hidden, grads);
}

/// Backprop through backbone only, starting from d_hidden.
fn backward_backbone_from_hidden(
    cache: &UnifiedForwardCache,
    d_hidden: &[f32; HIDDEN_DIM],
    grads: &mut UnifiedGradients,
) {
    // ReLU gate
    let mut d_pre_relu = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        d_pre_relu[j] = if cache.pre_relu[j] > 0.0 { d_hidden[j] } else { 0.0 };
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
// Graph Backbone Backward (for policy/mask head)
// ============================================================================

/// Backprop from d_graph_embed through graph_proj and graph backbone.
///
/// This is the graph-backbone analog of `backward_expr_proj_and_backbone`.
/// The policy gradient flows: mask_mlp -> graph_proj -> graph_w1.
fn backward_graph_proj_and_backbone(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    d_graph_embed: &[f32; EMBED_DIM],
    grads: &mut UnifiedGradients,
) {
    // ---- Graph projection backward ----
    // graph_embed = graph_proj_b + graph_proj_w^T @ graph_hidden
    let mut d_graph_hidden = [0.0f32; HIDDEN_DIM];
    for k in 0..EMBED_DIM {
        grads.d_graph_proj_b[k] += d_graph_embed[k];
    }
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            d_graph_hidden[j] += d_graph_embed[k] * net.graph_proj_w[j][k];
            grads.d_graph_proj_w[j][k] += d_graph_embed[k] * cache.graph_hidden[j];
        }
    }

    // ---- Graph backbone backward ----
    // ReLU gate
    let mut d_graph_pre_relu = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        d_graph_pre_relu[j] = if cache.graph_pre_relu[j] > 0.0 {
            d_graph_hidden[j]
        } else {
            0.0
        };
    }
    for j in 0..HIDDEN_DIM {
        grads.d_graph_b1[j] += d_graph_pre_relu[j];
    }
    for i in 0..GRAPH_INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            grads.d_graph_w1[i][j] += d_graph_pre_relu[j] * cache.graph_input[i];
        }
    }
}

// ============================================================================
// Embedding Backward: d_acc_input → d_embeddings
// ============================================================================

/// Compute gradient w.r.t. EdgeAccumulator input from the policy loss path.
///
/// With the graph backbone architecture, the policy path flows through
/// `graph_w1 → graph_input` (from GraphAccumulator), NOT through
/// `w1 → acc_input` (from EdgeAccumulator). Therefore this returns all zeros.
///
/// Policy embedding gradients should instead use [`compute_d_graph_input_policy`]
/// to flow through the graph accumulator construction.
#[must_use]
pub fn compute_d_acc_input_policy(
    _net: &ExprNnue,
    _cache: &UnifiedForwardCache,
    _rule_embed: &[f32; EMBED_DIM],
    _advantage: f32,
    _entropy_coeff: f32,
) -> [f32; INPUT_DIM] {
    // Policy path now flows through graph backbone, not expr backbone.
    // No gradient flows from the policy loss into the EdgeAccumulator input.
    [0.0f32; INPUT_DIM]
}

/// Compute gradient w.r.t. graph accumulator input from the policy loss path.
///
/// Re-derives the full chain from d_score through bilinear → mask_mlp → graph_proj → graph_w1 → graph_input.
/// This duplicates some computation from `backward_policy` but avoids changing its return type.
/// The duplicated work is cheap (~microseconds per step).
#[must_use]
pub fn compute_d_graph_input_policy(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    rule_embed: &[f32; EMBED_DIM],
    advantage: f32,
    entropy_coeff: f32,
) -> [f32; GRAPH_INPUT_DIM] {
    // d_score from REINFORCE (action=1 always, since steps are only recorded when approved)
    let mut d_score = -(1.0 - cache.prob) * advantage;
    if entropy_coeff != 0.0 {
        d_score += entropy_coeff * cache.prob * (1.0 - cache.prob) * cache.score;
    }

    // Bilinear backward → d_mask_features
    let mut d_transformed = [0.0f32; EMBED_DIM];
    for j in 0..EMBED_DIM {
        d_transformed[j] = d_score * rule_embed[j];
    }
    let mut d_mask_features = [0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        for j in 0..EMBED_DIM {
            d_mask_features[i] += d_transformed[j] * net.interaction[i][j];
        }
    }

    // Mask MLP backward → d_graph_embed
    let mut d_mask_h = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            d_mask_h[m] += d_mask_features[k] * net.mask_mlp_w2[m][k];
        }
    }
    let mut d_mask_h_pre = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        d_mask_h_pre[m] = if cache.mask_h_pre[m] > 0.0 { d_mask_h[m] } else { 0.0 };
    }
    let mut d_graph_embed = [0.0f32; EMBED_DIM];
    for m in 0..MLP_HIDDEN {
        for i in 0..EMBED_DIM {
            d_graph_embed[i] += d_mask_h_pre[m] * net.mask_mlp_w1[i][m];
        }
    }

    // Graph proj backward → d_graph_hidden
    let mut d_graph_hidden = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            d_graph_hidden[j] += d_graph_embed[k] * net.graph_proj_w[j][k];
        }
    }

    // Graph backbone backward → d_graph_input
    let mut d_graph_pre_relu = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        d_graph_pre_relu[j] = if cache.graph_pre_relu[j] > 0.0 {
            d_graph_hidden[j]
        } else {
            0.0
        };
    }
    let mut d_graph_input = [0.0f32; GRAPH_INPUT_DIM];
    for i in 0..GRAPH_INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            d_graph_input[i] += d_graph_pre_relu[j] * net.graph_w1[i][j];
        }
    }
    d_graph_input
}

/// Compute gradient w.r.t. accumulator input from the value loss path.
///
/// Re-derives the full chain from d_value through value_mlp → expr_proj → backbone → acc_input.
#[must_use]
pub fn compute_d_acc_input_value(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    target_cost: f32,
    value_coeff: f32,
) -> [f32; INPUT_DIM] {
    let d_value = 2.0 * (cache.value_pred - target_cost) * value_coeff;

    // Value MLP backward → d_expr_embed
    let mut d_value_h = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        d_value_h[m] = d_value * net.value_mlp_w2[m];
    }
    let mut d_value_h_pre = [0.0f32; MLP_HIDDEN];
    for m in 0..MLP_HIDDEN {
        d_value_h_pre[m] = if cache.value_h_pre[m] > 0.0 { d_value_h[m] } else { 0.0 };
    }
    let mut d_expr_embed = [0.0f32; EMBED_DIM];
    for m in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            d_expr_embed[k] += d_value_h_pre[m] * net.value_mlp_w1[k][m];
        }
    }

    // Expr proj backward → d_hidden
    let mut d_hidden = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            d_hidden[j] += d_expr_embed[k] * net.expr_proj_w[j][k];
        }
    }

    // Backbone backward → d_acc_input
    let mut d_pre_relu = [0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        d_pre_relu[j] = if cache.pre_relu[j] > 0.0 { d_hidden[j] } else { 0.0 };
    }
    let mut d_acc_input = [0.0f32; INPUT_DIM];
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            d_acc_input[i] += d_pre_relu[j] * net.w1[i][j];
        }
    }
    d_acc_input
}

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
/// Panics if any op index in `edges` is out of range for `OpKind::COUNT`.
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
    // d_acc_input[4*K] and d_acc_input[4*K+1] are log2-scaled edge/node counts —
    // these don't depend on embeddings, so we skip them.

    for &(parent_op_u8, child_op_u8, depth_u16) in edges {
        let pi = parent_op_u8 as usize;
        let ci = child_op_u8 as usize;
        assert!(
            pi < OpKind::COUNT,
            "parent op index {pi} out of range for OpKind::COUNT={}",
            OpKind::COUNT
        );
        assert!(
            ci < OpKind::COUNT,
            "child op index {ci} out of range for OpKind::COUNT={}",
            OpKind::COUNT
        );
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

/// Apply unified SGD update to all trainable parameters.
///
/// Uses **global L2 norm clipping**: if `||grads||_2 > grad_clip`, the entire
/// gradient vector is scaled by `grad_clip / ||grads||_2`, preserving direction.
/// Per-parameter scalar clipping would rotate the gradient direction, breaking
/// the hand-derived analytical gradients.
///
/// For each parameter p (after global clipping):
/// ```text
/// momentum_buf = momentum * momentum_buf + grad + weight_decay * param
/// param -= lr * momentum_buf
/// ```
pub fn apply_unified_sgd(
    net: &mut ExprNnue,
    grads: &UnifiedGradients,
    momentum_buf: &mut UnifiedGradients,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    grad_clip: f32,
) {
    // Global L2 norm clipping: scale entire gradient to preserve direction.
    let grad_norm = grads.norm();
    let clip_scale = if grad_norm > grad_clip {
        grad_clip / grad_norm
    } else {
        1.0
    };

    // Macro to apply SGD update to a single scalar parameter.
    macro_rules! sgd_scalar {
        ($param:expr, $grad:expr, $mbuf:expr) => {{
            let clipped = $grad * clip_scale;
            $mbuf = momentum * $mbuf + clipped + weight_decay * $param;
            $param -= lr * $mbuf;
        }};
    }

    // w1: [INPUT_DIM][HIDDEN_DIM]
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            sgd_scalar!(
                net.w1[i][j],
                grads.d_w1[i][j],
                momentum_buf.d_w1[i][j]
            );
        }
    }

    // b1: [HIDDEN_DIM]
    for j in 0..HIDDEN_DIM {
        sgd_scalar!(net.b1[j], grads.d_b1[j], momentum_buf.d_b1[j]);
    }

    // expr_proj_w: [HIDDEN_DIM][EMBED_DIM]
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            sgd_scalar!(
                net.expr_proj_w[j][k],
                grads.d_expr_proj_w[j][k],
                momentum_buf.d_expr_proj_w[j][k]
            );
        }
    }

    // expr_proj_b: [EMBED_DIM]
    for k in 0..EMBED_DIM {
        sgd_scalar!(
            net.expr_proj_b[k],
            grads.d_expr_proj_b[k],
            momentum_buf.d_expr_proj_b[k]
        );
    }

    // value_mlp_w1: [EMBED_DIM][MLP_HIDDEN]
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            sgd_scalar!(
                net.value_mlp_w1[i][j],
                grads.d_value_mlp_w1[i][j],
                momentum_buf.d_value_mlp_w1[i][j]
            );
        }
    }

    // value_mlp_b1: [MLP_HIDDEN]
    for j in 0..MLP_HIDDEN {
        sgd_scalar!(
            net.value_mlp_b1[j],
            grads.d_value_mlp_b1[j],
            momentum_buf.d_value_mlp_b1[j]
        );
    }

    // value_mlp_w2: [MLP_HIDDEN]
    for j in 0..MLP_HIDDEN {
        sgd_scalar!(
            net.value_mlp_w2[j],
            grads.d_value_mlp_w2[j],
            momentum_buf.d_value_mlp_w2[j]
        );
    }

    // value_mlp_b2: scalar
    sgd_scalar!(
        net.value_mlp_b2,
        grads.d_value_mlp_b2,
        momentum_buf.d_value_mlp_b2
    );

    // mask_mlp_w1: [EMBED_DIM][MLP_HIDDEN]
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            sgd_scalar!(
                net.mask_mlp_w1[i][j],
                grads.d_mask_mlp_w1[i][j],
                momentum_buf.d_mask_mlp_w1[i][j]
            );
        }
    }

    // mask_mlp_b1: [MLP_HIDDEN]
    for j in 0..MLP_HIDDEN {
        sgd_scalar!(
            net.mask_mlp_b1[j],
            grads.d_mask_mlp_b1[j],
            momentum_buf.d_mask_mlp_b1[j]
        );
    }

    // mask_mlp_w2: [MLP_HIDDEN][EMBED_DIM]
    for j in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            sgd_scalar!(
                net.mask_mlp_w2[j][k],
                grads.d_mask_mlp_w2[j][k],
                momentum_buf.d_mask_mlp_w2[j][k]
            );
        }
    }

    // mask_mlp_b2: [EMBED_DIM]
    for k in 0..EMBED_DIM {
        sgd_scalar!(
            net.mask_mlp_b2[k],
            grads.d_mask_mlp_b2[k],
            momentum_buf.d_mask_mlp_b2[k]
        );
    }

    // interaction: [EMBED_DIM][EMBED_DIM]
    for i in 0..EMBED_DIM {
        for j in 0..EMBED_DIM {
            sgd_scalar!(
                net.interaction[i][j],
                grads.d_interaction[i][j],
                momentum_buf.d_interaction[i][j]
            );
        }
    }

    // mask_bias_proj: [EMBED_DIM]
    for k in 0..EMBED_DIM {
        sgd_scalar!(
            net.mask_bias_proj[k],
            grads.d_mask_bias_proj[k],
            momentum_buf.d_mask_bias_proj[k]
        );
    }

    // embeddings: [OpKind::COUNT][K]
    for op in 0..OpKind::COUNT {
        for i in 0..K {
            sgd_scalar!(
                net.embeddings.e[op][i],
                grads.d_embeddings[op][i],
                momentum_buf.d_embeddings[op][i]
            );
        }
    }

    // graph_w1: [GRAPH_INPUT_DIM][HIDDEN_DIM]
    for i in 0..GRAPH_INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            sgd_scalar!(
                net.graph_w1[i][j],
                grads.d_graph_w1[i][j],
                momentum_buf.d_graph_w1[i][j]
            );
        }
    }

    // graph_b1: [HIDDEN_DIM]
    for j in 0..HIDDEN_DIM {
        sgd_scalar!(
            net.graph_b1[j],
            grads.d_graph_b1[j],
            momentum_buf.d_graph_b1[j]
        );
    }

    // graph_proj_w: [HIDDEN_DIM][EMBED_DIM]
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            sgd_scalar!(
                net.graph_proj_w[j][k],
                grads.d_graph_proj_w[j][k],
                momentum_buf.d_graph_proj_w[j][k]
            );
        }
    }

    // graph_proj_b: [EMBED_DIM]
    for k in 0..EMBED_DIM {
        sgd_scalar!(
            net.graph_proj_b[k],
            grads.d_graph_proj_b[k],
            momentum_buf.d_graph_proj_b[k]
        );
    }
}

// ============================================================================
// Helpers
// ============================================================================

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[inline]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
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
            self.0 = self.0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (self.0 >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        }
    }

    /// Initialize a network with small random weights everywhere (not just mask).
    fn make_test_net() -> ExprNnue {
        let mut net = ExprNnue::new();
        net.randomize_mask_only(42);

        // Also randomize backbone + expr_proj + value_mlp so gradients are nonzero
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

        // Randomize graph backbone (for mask/policy head)
        let scale_graph = libm::sqrtf(2.0 / GRAPH_INPUT_DIM as f32);
        for i in 0..GRAPH_INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                net.graph_w1[i][j] = rng.next_f32() * scale_graph;
            }
        }
        for j in 0..HIDDEN_DIM {
            net.graph_b1[j] = rng.next_f32() * 0.1;
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                net.graph_proj_w[j][k] = rng.next_f32() * scale_hidden;
            }
        }
        for k in 0..EMBED_DIM {
            net.graph_proj_b[k] = rng.next_f32() * 0.1;
        }

        net
    }

    /// Create a test accumulator with nonzero values.
    fn make_test_acc() -> EdgeAccumulator {
        let mut acc = EdgeAccumulator::new();
        let mut rng = Lcg::new(7777);
        for v in &mut acc.values {
            *v = rng.next_f32() * 0.5;
        }
        acc.edge_count = 5;
        acc.node_count = 4;
        acc
    }

    /// Create a test graph accumulator with nonzero values.
    fn make_test_gacc() -> GraphAccumulator {
        let mut gacc = GraphAccumulator::new();
        let mut rng = Lcg::new(3333);
        for v in &mut gacc.values {
            *v = rng.next_f32() * 0.5;
        }
        gacc.edge_count = 7;
        gacc.node_count = 5;
        gacc
    }

    /// Create a test rule embedding.
    fn make_test_rule_embed() -> [f32; EMBED_DIM] {
        let mut embed = [0.0f32; EMBED_DIM];
        let mut rng = Lcg::new(5555);
        for v in &mut embed {
            *v = rng.next_f32() * 0.3;
        }
        embed
    }

    /// Compute policy loss using numerically stable log-sigmoid.
    ///
    /// Loss = -log(sigmoid(score)) * advantage  if matched
    /// Loss = -log(1-sigmoid(score)) * advantage  if not matched
    ///
    /// We use the identity: -log(sigmoid(s)) = log(1 + exp(-s)) = softplus(-s)
    /// and: -log(1-sigmoid(s)) = log(1 + exp(s)) = softplus(s)
    ///
    /// The f64 computation avoids the f32 precision bottleneck where
    /// sigmoid(score) → log() loses significant digits.
    fn policy_loss(net: &ExprNnue, acc: &EdgeAccumulator, gacc: &GraphAccumulator, rule_embed: &[f32; EMBED_DIM], _matched: bool, advantage: f32) -> f64 {
        let cache = forward_cached(net, acc, gacc, rule_embed);
        let s = cache.score as f64;
        // In self-play, steps are only recorded when the policy chose Action=1 (approve).
        // Loss = -log(sigma(s)) * advantage = softplus(-s) * advantage
        let neg_log_p = softplus_f64(-s);
        neg_log_p * advantage as f64
    }

    /// Compute value loss: (value_pred - target)^2 * value_coeff.
    fn value_loss(net: &ExprNnue, acc: &EdgeAccumulator, gacc: &GraphAccumulator, rule_embed: &[f32; EMBED_DIM], target_cost: f32, value_coeff: f32) -> f64 {
        let cache = forward_cached(net, acc, gacc, rule_embed);
        let diff = cache.value_pred as f64 - target_cost as f64;
        diff * diff * value_coeff as f64
    }

    /// Numerically stable softplus in f64: log(1 + exp(x)).
    fn softplus_f64(x: f64) -> f64 {
        if x > 20.0 {
            x  // exp(x) >> 1, so log(1+exp(x)) ≈ x
        } else if x < -20.0 {
            libm::exp(x) // exp(x) << 1, so log(1+exp(x)) ≈ exp(x)
        } else {
            libm::log(1.0 + libm::exp(x))
        }
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
    // Test 1: Interaction matrix gradient (policy)
    // ========================================================================

    #[test]
    fn numerical_gradient_check_policy_interaction() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();
        let matched = true;
        let advantage = 1.5;

        // Analytical gradient
        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let mut grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, matched, advantage, 0.0, 1.0, &mut grads);

        // Check a sample of interaction matrix elements
        // eps=5e-4 balances truncation error (O(eps^2)) and rounding error (O(1/eps))
        // for the log-sigmoid policy loss in f32 arithmetic.
        let eps = 5e-4f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;

        for i in [0, 5, 12, 23] {
            for j in [0, 7, 15, 23] {
                let mut net_p = net.clone();
                net_p.interaction[i][j] += eps;
                let loss_plus = policy_loss(&net_p, &acc, &gacc, &rule_embed, matched, advantage);

                let mut net_m = net.clone();
                net_m.interaction[i][j] -= eps;
                let loss_minus = policy_loss(&net_m, &acc, &gacc, &rule_embed, matched, advantage);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_interaction[i][j], num_grad);
                if err > max_err {
                    max_err = err;
                }
                // Relative error can blow up when both values are near zero.
                // Accept if absolute difference < 1e-5 OR relative error < 5%.
                let abs_diff = (a as f64 - n).abs();
                assert!(
                    err < 0.05 || abs_diff < 1e-5,
                    "interaction[{i}][{j}]: analytical={a:.6}, numerical={n:.6}, rel_err={err:.6}, abs_diff={abs_diff:.6e}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 16, "checked {checked} interaction elements");
        eprintln!("  interaction max rel error: {max_err:.6e}  ({checked} elements)");
    }

    // ========================================================================
    // Test 2: Mask MLP w1 gradient (policy)
    // ========================================================================

    #[test]
    fn numerical_gradient_check_policy_mask_mlp() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();
        let matched = false;
        let advantage = 2.0;

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let mut grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, matched, advantage, 0.0, 1.0, &mut grads);

        // eps=5e-4 balances truncation error (O(eps^2)) and rounding error (O(1/eps))
        // for the log-sigmoid policy loss in f32 arithmetic.
        let eps = 5e-4f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;

        // Check mask_mlp_w1 (24 x 16) — sample a few elements
        for i in [0, 12, 23] {
            for j in [0, 8, 15] {
                let mut net_p = net.clone();
                net_p.mask_mlp_w1[i][j] += eps;
                let loss_plus = policy_loss(&net_p, &acc, &gacc, &rule_embed, matched, advantage);

                let mut net_m = net.clone();
                net_m.mask_mlp_w1[i][j] -= eps;
                let loss_minus = policy_loss(&net_m, &acc, &gacc, &rule_embed, matched, advantage);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_mask_mlp_w1[i][j], num_grad);
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "mask_mlp_w1[{i}][{j}]: analytical={a:.6}, numerical={n:.6}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        // Also check mask_mlp_w2, mask_mlp_b1, mask_mlp_b2
        for j in [0, 8, 15] {
            for k in [0, 12, 23] {
                let mut net_p = net.clone();
                net_p.mask_mlp_w2[j][k] += eps;
                let loss_plus = policy_loss(&net_p, &acc, &gacc, &rule_embed, matched, advantage);

                let mut net_m = net.clone();
                net_m.mask_mlp_w2[j][k] -= eps;
                let loss_minus = policy_loss(&net_m, &acc, &gacc, &rule_embed, matched, advantage);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_mask_mlp_w2[j][k], num_grad);
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "mask_mlp_w2[{j}][{k}]: analytical={a:.6}, numerical={n:.6}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        assert!(checked >= 18, "checked {checked} mask_mlp elements");
        eprintln!("  mask_mlp max rel error: {max_err:.6e}  ({checked} elements)");
    }

    // ========================================================================
    // Test 3: Graph backbone gradient (policy — deepest chain rule)
    // ========================================================================

    #[test]
    fn numerical_gradient_check_policy_graph_backbone() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();
        let matched = true;
        let advantage = 0.8;

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let mut grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, matched, advantage, 0.0, 1.0, &mut grads);

        let eps = 1e-3f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;
        let mut nonzero_grads = 0;

        // Check graph_w1 (98 x 64) — sample some elements
        for i in [0, 30, 64, 96, 97] {
            for j in [0, 16, 32, 63] {
                let mut net_p = net.clone();
                net_p.graph_w1[i][j] += eps;
                let loss_plus = policy_loss(&net_p, &acc, &gacc, &rule_embed, matched, advantage);

                let mut net_m = net.clone();
                net_m.graph_w1[i][j] -= eps;
                let loss_minus = policy_loss(&net_m, &acc, &gacc, &rule_embed, matched, advantage);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_graph_w1[i][j], num_grad);
                if err > max_err {
                    max_err = err;
                }
                if grads.d_graph_w1[i][j].abs() > 1e-10 {
                    nonzero_grads += 1;
                }
                assert!(
                    err < 0.05,
                    "graph_w1[{i}][{j}]: analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        // Also check graph_b1
        for j in [0, 16, 32, 63] {
            let mut net_p = net.clone();
            net_p.graph_b1[j] += eps;
            let loss_plus = policy_loss(&net_p, &acc, &gacc, &rule_embed, matched, advantage);

            let mut net_m = net.clone();
            net_m.graph_b1[j] -= eps;
            let loss_minus = policy_loss(&net_m, &acc, &gacc, &rule_embed, matched, advantage);

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
            let (a, n, err) = check_gradient(grads.d_graph_b1[j], num_grad);
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 0.05,
                "graph_b1[{j}]: analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
            );
            checked += 1;
        }

        // Also check graph_proj_w
        for j in [0, 32, 63] {
            for k in [0, 12, 23] {
                let mut net_p = net.clone();
                net_p.graph_proj_w[j][k] += eps;
                let loss_plus = policy_loss(&net_p, &acc, &gacc, &rule_embed, matched, advantage);

                let mut net_m = net.clone();
                net_m.graph_proj_w[j][k] -= eps;
                let loss_minus = policy_loss(&net_m, &acc, &gacc, &rule_embed, matched, advantage);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_graph_proj_w[j][k], num_grad);
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 0.05,
                    "graph_proj_w[{j}][{k}]: analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}"
                );
                checked += 1;
            }
        }

        // Verify that w1/b1/expr_proj_w have ZERO policy gradient (policy no longer flows through expr backbone)
        for i in [0, 64, 129] {
            for j in [0, 32, 63] {
                assert!(
                    grads.d_w1[i][j].abs() < 1e-10,
                    "w1[{i}][{j}] should have zero policy gradient, got {}",
                    grads.d_w1[i][j]
                );
            }
        }
        for j in [0, 32, 63] {
            assert!(
                grads.d_b1[j].abs() < 1e-10,
                "b1[{j}] should have zero policy gradient, got {}",
                grads.d_b1[j]
            );
        }

        assert!(checked >= 29, "checked {checked} graph backbone elements");
        assert!(nonzero_grads > 0, "at least some graph backbone gradients must be nonzero");
        eprintln!("  graph backbone max rel error: {max_err:.6e}  ({checked} elements, {nonzero_grads} nonzero)");
    }

    // ========================================================================
    // Test 4: Value path gradients
    // ========================================================================

    #[test]
    fn numerical_gradient_check_value() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();
        let target_cost = 3.5f32;
        let value_coeff = 0.5f32;

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let mut grads = UnifiedGradients::zero();
        backward_value(&net, &cache, target_cost, value_coeff, &mut grads);

        let eps = 1e-3f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;

        // value_mlp_w2
        for j in [0, 8, 15] {
            let mut net_p = net.clone();
            net_p.value_mlp_w2[j] += eps;
            let loss_plus = value_loss(&net_p, &acc, &gacc, &rule_embed, target_cost, value_coeff);

            let mut net_m = net.clone();
            net_m.value_mlp_w2[j] -= eps;
            let loss_minus = value_loss(&net_m, &acc, &gacc, &rule_embed, target_cost, value_coeff);

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
            let loss_plus = value_loss(&net_p, &acc, &gacc, &rule_embed, target_cost, value_coeff);

            let mut net_m = net.clone();
            net_m.value_mlp_b2 -= eps;
            let loss_minus = value_loss(&net_m, &acc, &gacc, &rule_embed, target_cost, value_coeff);

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
                let loss_plus = value_loss(&net_p, &acc, &gacc, &rule_embed, target_cost, value_coeff);

                let mut net_m = net.clone();
                net_m.value_mlp_w1[i][j] -= eps;
                let loss_minus = value_loss(&net_m, &acc, &gacc, &rule_embed, target_cost, value_coeff);

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
        for j in [0, 32, 63] {
            for k in [0, 12, 23] {
                let mut net_p = net.clone();
                net_p.expr_proj_w[j][k] += eps;
                let loss_plus = value_loss(&net_p, &acc, &gacc, &rule_embed, target_cost, value_coeff);

                let mut net_m = net.clone();
                net_m.expr_proj_w[j][k] -= eps;
                let loss_minus = value_loss(&net_m, &acc, &gacc, &rule_embed, target_cost, value_coeff);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
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

        // w1 (value path through backbone)
        for i in [0, 64, 129] {
            for j in [0, 32, 63] {
                let mut net_p = net.clone();
                net_p.w1[i][j] += eps;
                let loss_plus = value_loss(&net_p, &acc, &gacc, &rule_embed, target_cost, value_coeff);

                let mut net_m = net.clone();
                net_m.w1[i][j] -= eps;
                let loss_minus = value_loss(&net_m, &acc, &gacc, &rule_embed, target_cost, value_coeff);

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

        assert!(checked >= 26, "checked {checked} value path elements");
        eprintln!("  value path max rel error: {max_err:.6e}  ({checked} elements)");
    }

    // ========================================================================
    // Test 5: Joint gradient accumulation (independent backbones)
    // ========================================================================

    #[test]
    fn joint_gradient_accumulates() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();
        let matched = true;
        let advantage = 1.0;
        let target_cost = 2.0f32;
        let value_coeff = 0.5f32;

        // Compute joint gradient (both losses into same buffer)
        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let mut joint_grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, matched, advantage, 0.0, 1.0, &mut joint_grads);
        backward_value(&net, &cache, target_cost, value_coeff, &mut joint_grads);

        // Compute separate gradients
        let mut policy_grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, matched, advantage, 0.0, 1.0, &mut policy_grads);

        let mut value_grads = UnifiedGradients::zero();
        backward_value(&net, &cache, target_cost, value_coeff, &mut value_grads);

        // Joint loss for numerical check
        let joint_loss = |net: &ExprNnue| -> f64 {
            policy_loss(net, &acc, &gacc, &rule_embed, matched, advantage)
                + value_loss(net, &acc, &gacc, &rule_embed, target_cost, value_coeff)
        };

        let eps = 2e-4f32;
        let mut max_err = 0.0f64;
        let mut checked = 0;

        // w1 is value-only now (policy flows through graph_w1 instead).
        // Verify joint w1 gradient matches value-only gradient.
        for i in [0, 64, 129] {
            for j in [0, 32, 63] {
                // Policy gradient for w1 should be zero
                assert!(
                    policy_grads.d_w1[i][j].abs() < 1e-10,
                    "w1[{i}][{j}] should have zero policy gradient, got {}",
                    policy_grads.d_w1[i][j]
                );

                // Joint = value-only for w1
                let joint = joint_grads.d_w1[i][j];
                let value_only = value_grads.d_w1[i][j];
                let diff = (joint - value_only).abs();
                assert!(
                    diff < 1e-6,
                    "w1[{i}][{j}]: joint={joint:.8} != value={value_only:.8}, diff={diff:.8e}"
                );

                // Verify vs numerical gradient of joint loss
                let mut net_p = net.clone();
                net_p.w1[i][j] += eps;
                let loss_plus = joint_loss(&net_p);

                let mut net_m = net.clone();
                net_m.w1[i][j] -= eps;
                let loss_minus = joint_loss(&net_m);

                let num_grad = (loss_plus - loss_minus) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(joint, num_grad);
                if err > max_err {
                    max_err = err;
                }
                let abs_diff = (a as f64 - n).abs();
                assert!(
                    err < 0.10 || abs_diff < 1e-4,
                    "joint w1[{i}][{j}]: analytical={a:.8}, numerical={n:.8}, rel_err={err:.6}, abs_diff={abs_diff:.6e}"
                );
                checked += 1;
            }
        }

        // graph_w1 is policy-only (value flows through w1 instead).
        // Verify joint graph_w1 gradient matches policy-only gradient.
        for i in [0, 48, 97] {
            for j in [0, 32, 63] {
                // Value gradient for graph_w1 should be zero
                assert!(
                    value_grads.d_graph_w1[i][j].abs() < 1e-10,
                    "graph_w1[{i}][{j}] should have zero value gradient, got {}",
                    value_grads.d_graph_w1[i][j]
                );

                // Joint = policy-only for graph_w1
                let joint = joint_grads.d_graph_w1[i][j];
                let policy_only = policy_grads.d_graph_w1[i][j];
                let diff = (joint - policy_only).abs();
                assert!(
                    diff < 1e-6,
                    "graph_w1[{i}][{j}]: joint={joint:.8} != policy={policy_only:.8}, diff={diff:.8e}"
                );
                checked += 1;
            }
        }

        // Verify policy-only params (interaction) are NOT affected by value loss
        for i in [0, 12, 23] {
            for j in [0, 12, 23] {
                assert!(
                    value_grads.d_interaction[i][j].abs() < 1e-10,
                    "interaction[{i}][{j}] should have zero value gradient, got {}",
                    value_grads.d_interaction[i][j]
                );
            }
        }

        // Verify value-only params are NOT affected by policy loss
        // mask_bias_proj only appears in the bilinear score, so value loss doesn't touch it
        for k in 0..EMBED_DIM {
            assert!(
                value_grads.d_mask_bias_proj[k].abs() < 1e-10,
                "mask_bias_proj[{k}] should have zero value gradient, got {}",
                value_grads.d_mask_bias_proj[k]
            );
        }

        assert!(checked >= 18, "checked {checked} joint elements");
        eprintln!("  joint accumulation max rel error: {max_err:.6e}  ({checked} elements)");
    }

    // ========================================================================
    // Test 6: Forward cache score matches public ExprNnue::bilinear_score
    // ========================================================================

    #[test]
    fn forward_cached_matches_bilinear_score() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);

        // bilinear_score is public — verify our cached mask_features + score agree
        let score_from_public = net.bilinear_score(&cache.mask_features, &rule_embed);
        assert!(
            (cache.score - score_from_public).abs() < 1e-5,
            "score mismatch: cached={}, bilinear_score={}",
            cache.score,
            score_from_public
        );

        // Verify prob = sigmoid(score)
        let expected_prob = sigmoid(cache.score);
        assert!(
            (cache.prob - expected_prob).abs() < 1e-6,
            "prob mismatch: cached={}, sigmoid(score)={}",
            cache.prob,
            expected_prob
        );

        // Verify acc_input was built correctly from the accumulator (now with scaling)
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
            (cache.acc_input[4 * K] - libm::log2f(1.0 + acc.edge_count as f32)).abs() < 1e-6,
            "acc_input edge_count mismatch"
        );
        assert!(
            (cache.acc_input[4 * K + 1] - libm::log2f(1.0 + acc.node_count as f32)).abs() < 1e-6,
            "acc_input node_count mismatch"
        );

        // Verify graph_input was built correctly from the graph accumulator
        let graph_scale = if gacc.node_count > 0 {
            1.0 / libm::sqrtf(gacc.node_count as f32)
        } else {
            1.0
        };
        for i in 0..GRAPH_ACC_DIM {
            assert!(
                (cache.graph_input[i] - gacc.values[i] * graph_scale).abs() < 1e-6,
                "graph_input[{i}] should match scaled gacc.values"
            );
        }
        assert!(
            (cache.graph_input[GRAPH_ACC_DIM] - libm::log2f(1.0 + gacc.edge_count as f32)).abs() < 1e-6,
            "graph_input edge_count mismatch"
        );
        assert!(
            (cache.graph_input[GRAPH_ACC_DIM + 1] - libm::log2f(1.0 + gacc.node_count as f32)).abs() < 1e-6,
            "graph_input node_count mismatch"
        );
    }

    // ========================================================================
    // Test 7: Gradient norm is nonzero
    // ========================================================================

    #[test]
    fn gradient_norm_nonzero() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);

        let mut grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, true, 1.0, 0.0, 1.0, &mut grads);
        let policy_norm = grads.norm();
        assert!(
            policy_norm > 1e-8,
            "policy gradient norm should be nonzero, got {policy_norm}"
        );

        let mut grads = UnifiedGradients::zero();
        backward_value(&net, &cache, 3.0, 1.0, &mut grads);
        let value_norm = grads.norm();
        assert!(
            value_norm > 1e-8,
            "value gradient norm should be nonzero, got {value_norm}"
        );

        eprintln!("  policy grad norm: {policy_norm:.6}");
        eprintln!("  value grad norm:  {value_norm:.6}");
    }

    // ========================================================================
    // Test 8: SGD actually moves parameters
    // ========================================================================

    #[test]
    fn sgd_moves_parameters() {
        let mut net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();

        let w1_before = net.w1[0][0];
        let interaction_before = net.interaction[0][0];
        let graph_w1_before = net.graph_w1[0][0];

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let mut grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, true, 1.0, 0.0, 1.0, &mut grads);
        backward_value(&net, &cache, 3.0, 0.5, &mut grads);

        let mut momentum_buf = UnifiedGradients::zero();
        apply_unified_sgd(
            &mut net,
            &grads,
            &mut momentum_buf,
            0.01,  // lr
            0.9,   // momentum
            1e-4,  // weight_decay
            1.0,   // grad_clip
        );

        let w1_after = net.w1[0][0];
        let interaction_after = net.interaction[0][0];
        let graph_w1_after = net.graph_w1[0][0];

        assert!(
            (w1_after - w1_before).abs() > 1e-10,
            "w1[0][0] should have moved: before={w1_before}, after={w1_after}"
        );
        assert!(
            (interaction_after - interaction_before).abs() > 1e-10,
            "interaction[0][0] should have moved: before={interaction_before}, after={interaction_after}"
        );
        assert!(
            (graph_w1_after - graph_w1_before).abs() > 1e-10,
            "graph_w1[0][0] should have moved: before={graph_w1_before}, after={graph_w1_after}"
        );
    }

    // ========================================================================
    // Test 9: Scale and accumulate
    // ========================================================================

    #[test]
    fn scale_and_accumulate() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();
        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);

        let mut g1 = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, true, 1.0, 0.0, 1.0, &mut g1);

        let mut g2 = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, true, 1.0, 0.0, 1.0, &mut g2);

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
        eprintln!("  scale/accumulate: half_norm={norm_half:.6}, orig_norm={norm_orig:.6}, 1.5x_norm={actual:.6}");
    }

    // ========================================================================
    // Test 11: Numerical gradient check for entropy bonus
    // ========================================================================

    #[test]
    fn numerical_gradient_check_entropy() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let entropy_coeff = 0.1f32;

        // Compute analytical gradient with entropy only (advantage=0 isolates entropy)
        let mut grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, true, 0.0, entropy_coeff, 1.0, &mut grads);

        // Compute entropy numerically
        let entropy_at = |net: &ExprNnue| -> f64 {
            let c = forward_cached(net, &acc, &gacc, &rule_embed);
            let p = c.prob as f64;
            let p = p.clamp(1e-7, 1.0 - 1e-7);
            // We minimize -entropy, so loss = -(- p*ln(p) - (1-p)*ln(1-p)) * coeff
            //                                = (p*ln(p) + (1-p)*ln(1-p)) * coeff
            (p * libm::log(p) + (1.0 - p) * libm::log(1.0 - p)) * entropy_coeff as f64
        };

        // eps=5e-4 balances truncation error (O(eps^2)) and rounding error (O(1/eps))
        // for the sigmoid-based entropy loss in f32 arithmetic.
        let eps = 5e-4f32;
        let mut max_err = 0.0f64;

        // Check interaction matrix (directly affects score).
        // For near-zero gradients (both analytical and numerical < abs_threshold),
        // the relative error is meaningless — check absolute error instead.
        let abs_threshold = 1e-5;
        let mut checked = 0;
        for i in [0, 12, 23] {
            for j in [0, 12, 23] {
                let mut net_p = net.clone();
                net_p.interaction[i][j] += eps;
                let mut net_m = net.clone();
                net_m.interaction[i][j] -= eps;
                let num_grad = (entropy_at(&net_p) - entropy_at(&net_m)) / (2.0 * eps as f64);
                let (a, n, err) = check_gradient(grads.d_interaction[i][j], num_grad);

                // Skip relative error check for near-zero gradients where noise dominates
                if (a as f64).abs() < abs_threshold as f64 && n.abs() < abs_threshold as f64 {
                    let abs_err = (a as f64 - n).abs();
                    assert!(
                        abs_err < abs_threshold as f64,
                        "interaction[{i}][{j}]: near-zero gradient abs_err={abs_err:.8e} exceeds threshold"
                    );
                    eprintln!("  interaction[{i}][{j}]: analytical={a:.8e}, numerical={n:.8e}, abs_err={abs_err:.8e} (near-zero, skip rel check)");
                } else {
                    if err > max_err { max_err = err; }
                    eprintln!("  interaction[{i}][{j}]: analytical={a:.8e}, numerical={n:.8e}, rel_err={err:.6}");
                }
                checked += 1;
            }
        }

        assert!(checked >= 9, "checked {checked} entropy gradient elements");
        assert!(
            max_err < 0.05,
            "Entropy gradient max relative error: {max_err:.6} (threshold: 0.05)"
        );
        eprintln!("  entropy gradient max rel error: {max_err:.6e}  ({checked} elements)");
    }

    // ========================================================================
    // Test 12: Entropy pushes probability toward 0.5
    // ========================================================================

    #[test]
    fn entropy_pushes_toward_half() {
        let net = make_test_net();
        let acc = make_test_acc();
        let gacc = make_test_gacc();
        let rule_embed = make_test_rule_embed();

        let cache = forward_cached(&net, &acc, &gacc, &rule_embed);
        let initial_prob = cache.prob;

        // Apply entropy gradient only (advantage=0)
        let mut grads = UnifiedGradients::zero();
        backward_policy(&net, &cache, &rule_embed, true, 0.0, 0.1, 1.0, &mut grads);

        // Apply one SGD step
        let mut net2 = net.clone();
        let mut momentum = UnifiedGradients::zero();
        apply_unified_sgd(&mut net2, &grads, &mut momentum, 0.01, 0.0, 0.0, 10.0);

        let cache2 = forward_cached(&net2, &acc, &gacc, &rule_embed);
        let updated_prob = cache2.prob;

        // Entropy gradient should push probability toward 0.5
        if initial_prob > 0.5 {
            assert!(updated_prob < initial_prob,
                "Entropy should decrease prob from {initial_prob:.4} toward 0.5, got {updated_prob:.4}");
        } else {
            assert!(updated_prob > initial_prob,
                "Entropy should increase prob from {initial_prob:.4} toward 0.5, got {updated_prob:.4}");
        }
        eprintln!("  entropy push: {initial_prob:.4} -> {updated_prob:.4} (toward 0.5)");
    }

}
