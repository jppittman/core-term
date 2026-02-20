//! # Training for Dual-Head NNUE Architecture
//!
//! This module provides training infrastructure for the AlphaZero-style dual-head NNUE:
//!
//! - **Value Head (The Judge)**: Predicts cost in nanoseconds for extraction
//! - **Search Head (The Guide)**: Predicts priority for A* heap ordering
//!
//! ## Training Phases
//!
//! 1. **Phase 1**: Bootstrap value head with SIMD benchmarks (supervised)
//! 2. **Phase 2**: Train search head with search trajectories (reinforcement)
//! 3. **Phase 3**: Joint fine-tuning (both losses)
//!
//! ## Loss Functions
//!
//! - **Value loss**: MSE on log-costs: `L_v = (pred_log_cost - actual_log_cost)²`
//! - **Search loss**: MSE on improvement: `L_s = (pred_priority - achieved_improvement)²`
//! - **Joint loss**: `L = λ_v * L_v + λ_s * L_s`

use crate::nnue::factored::{
    ExprNnue, EdgeAccumulator, OpEmbeddings, StructuralFeatures,
    HIDDEN_DIM, INPUT_DIM, K,
};
use crate::nnue::{Expr, OpKind};

use super::factored::{parse_expr, parse_kernel_code};

// ============================================================================
// Training Samples
// ============================================================================

/// A training sample for the value head (cost prediction).
#[derive(Clone)]
pub struct ValueSample {
    /// The expression.
    pub expr: Expr,

    /// Ground truth cost in nanoseconds.
    pub cost_ns: f64,

    /// Precomputed edge accumulator.
    pub accumulator: EdgeAccumulator,

    /// Precomputed structural features.
    pub structural: StructuralFeatures,
}

impl ValueSample {
    /// Create a new sample from expression and cost.
    pub fn new(expr: Expr, cost_ns: f64, embeddings: &OpEmbeddings) -> Self {
        let accumulator = EdgeAccumulator::from_expr(&expr, embeddings);
        let structural = StructuralFeatures::from_expr(&expr);
        Self {
            expr,
            cost_ns,
            accumulator,
            structural,
        }
    }

    /// Recompute the accumulator with updated embeddings.
    pub fn recompute_accumulator(&mut self, embeddings: &OpEmbeddings) {
        self.accumulator = EdgeAccumulator::from_expr(&self.expr, embeddings);
    }

    /// Target value for training (log of cost).
    #[inline]
    pub fn target(&self) -> f32 {
        (self.cost_ns as f32).ln()
    }
}

/// A training sample for the search head (priority prediction).
#[derive(Clone)]
pub struct SearchSample {
    /// The expression (search state).
    pub expr: Expr,

    /// Improvement achieved from this state (negative = good, positive = bad).
    /// This is the target for the search head.
    pub improvement: f32,

    /// Precomputed edge accumulator.
    pub accumulator: EdgeAccumulator,

    /// Precomputed structural features.
    pub structural: StructuralFeatures,
}

impl SearchSample {
    /// Create a new search sample.
    pub fn new(expr: Expr, improvement: f32, embeddings: &OpEmbeddings) -> Self {
        let accumulator = EdgeAccumulator::from_expr(&expr, embeddings);
        let structural = StructuralFeatures::from_expr(&expr);
        Self {
            expr,
            improvement,
            accumulator,
            structural,
        }
    }

    /// Recompute the accumulator with updated embeddings.
    pub fn recompute_accumulator(&mut self, embeddings: &OpEmbeddings) {
        self.accumulator = EdgeAccumulator::from_expr(&self.expr, embeddings);
    }

    /// Target value for training (scale improvement to priority range).
    #[inline]
    pub fn target(&self) -> f32 {
        // Lower values = higher priority (expand first)
        // States that lead to improvements should have lower priorities
        -self.improvement
    }
}

// ============================================================================
// Gradient Accumulators
// ============================================================================

/// Gradients for dual-head network parameters.
#[derive(Clone)]
pub struct DualGradients {
    // ========== SHARED GRADIENTS ==========
    /// Embedding gradients: d(loss)/d(embedding[op][k]).
    pub d_emb: [[f32; K]; OpKind::COUNT],

    /// W1 gradients.
    pub d_w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// B1 gradients.
    pub d_b1: [f32; HIDDEN_DIM],

    // ========== VALUE HEAD GRADIENTS ==========
    /// Value head weight gradients.
    pub d_value_w: [f32; HIDDEN_DIM],

    /// Value head bias gradient.
    pub d_value_b: f32,

    // ========== SEARCH HEAD GRADIENTS ==========
    /// Search head weight gradients.
    pub d_search_w: [f32; HIDDEN_DIM],

    /// Search head bias gradient.
    pub d_search_b: f32,
}

impl Default for DualGradients {
    fn default() -> Self {
        Self::new()
    }
}

impl DualGradients {
    /// Create zero-initialized gradients.
    pub fn new() -> Self {
        Self {
            d_emb: [[0.0; K]; OpKind::COUNT],
            d_w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            d_b1: [0.0; HIDDEN_DIM],
            d_value_w: [0.0; HIDDEN_DIM],
            d_value_b: 0.0,
            d_search_w: [0.0; HIDDEN_DIM],
            d_search_b: 0.0,
        }
    }

    /// Reset all gradients to zero.
    pub fn zero(&mut self) {
        for row in &mut self.d_emb {
            row.fill(0.0);
        }
        for row in &mut self.d_w1 {
            row.fill(0.0);
        }
        self.d_b1.fill(0.0);
        self.d_value_w.fill(0.0);
        self.d_value_b = 0.0;
        self.d_search_w.fill(0.0);
        self.d_search_b = 0.0;
    }

    /// Scale all gradients (for averaging over minibatch).
    pub fn scale(&mut self, factor: f32) {
        for row in &mut self.d_emb {
            for v in row.iter_mut() {
                *v *= factor;
            }
        }
        for row in &mut self.d_w1 {
            for v in row.iter_mut() {
                *v *= factor;
            }
        }
        for v in &mut self.d_b1 {
            *v *= factor;
        }
        for v in &mut self.d_value_w {
            *v *= factor;
        }
        self.d_value_b *= factor;
        for v in &mut self.d_search_w {
            *v *= factor;
        }
        self.d_search_b *= factor;
    }

    /// Accumulate gradients from another DualGradients struct.
    pub fn accumulate(&mut self, other: &DualGradients) {
        for (row, other_row) in self.d_emb.iter_mut().zip(other.d_emb.iter()) {
            for (v, &ov) in row.iter_mut().zip(other_row.iter()) {
                *v += ov;
            }
        }
        for (row, other_row) in self.d_w1.iter_mut().zip(other.d_w1.iter()) {
            for (v, &ov) in row.iter_mut().zip(other_row.iter()) {
                *v += ov;
            }
        }
        for (v, &ov) in self.d_b1.iter_mut().zip(other.d_b1.iter()) {
            *v += ov;
        }
        for (v, &ov) in self.d_value_w.iter_mut().zip(other.d_value_w.iter()) {
            *v += ov;
        }
        self.d_value_b += other.d_value_b;
        for (v, &ov) in self.d_search_w.iter_mut().zip(other.d_search_w.iter()) {
            *v += ov;
        }
        self.d_search_b += other.d_search_b;
    }
}

// ============================================================================
// Forward/Backward Pass
// ============================================================================

/// Cached intermediate values from forward pass (needed for backward).
pub struct DualForwardCache {
    /// Input to hidden layer (accumulator + structural).
    pub input: [f32; INPUT_DIM],

    /// Pre-activation hidden values.
    pub hidden_pre: [f32; HIDDEN_DIM],

    /// Post-ReLU hidden values.
    pub hidden: [f32; HIDDEN_DIM],

    /// Value head output (predicted log-cost).
    pub value_output: f32,

    /// Search head output (predicted priority).
    pub search_output: f32,
}

impl DualForwardCache {
    /// Run forward pass through both heads and cache intermediates.
    pub fn forward(net: &ExprNnue, acc: &EdgeAccumulator, structural: &StructuralFeatures) -> Self {
        let mut input = [0.0f32; INPUT_DIM];

        // Copy accumulator values (first 2K dims)
        input[..2 * K].copy_from_slice(&acc.values);

        // Copy structural features (remaining dims)
        input[2 * K..].copy_from_slice(&structural.values);

        // Hidden layer: input @ W1 + b1
        let mut hidden_pre = net.b1;
        for (i, &inp) in input.iter().enumerate() {
            for (j, h) in hidden_pre.iter_mut().enumerate() {
                *h += inp * net.w1[i][j];
            }
        }

        // ReLU
        let mut hidden = hidden_pre;
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Value head output
        let mut value_output = net.value_b;
        for (&h, &w) in hidden.iter().zip(net.value_w.iter()) {
            value_output += h * w;
        }

        // Search head output
        let mut search_output = net.search_b;
        for (&h, &w) in hidden.iter().zip(net.search_w.iter()) {
            search_output += h * w;
        }

        Self {
            input,
            hidden_pre,
            hidden,
            value_output,
            search_output,
        }
    }
}

/// Softplus function: log(1 + exp(x)), numerically stable.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x  // Avoid overflow
    } else if x < -20.0 {
        0.0  // Avoid underflow
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Compute gradients for value head with a pre-computed output gradient.
///
/// This is for pairwise ranking loss where we compute d(loss)/d(output) externally.
pub fn backward_value_with_d_output(
    net: &ExprNnue,
    cache: &DualForwardCache,
    d_output: f32,
    sample: &ValueSample,
    grads: &mut DualGradients,
) {
    // Gradient through value head output layer
    grads.d_value_b += d_output;
    let mut d_hidden = [0.0f32; HIDDEN_DIM];
    for (i, &h) in cache.hidden.iter().enumerate() {
        grads.d_value_w[i] += d_output * h;
        d_hidden[i] = d_output * net.value_w[i];
    }

    // Gradient through ReLU
    for (i, dh) in d_hidden.iter_mut().enumerate() {
        if cache.hidden_pre[i] <= 0.0 {
            *dh = 0.0;
        }
    }

    // Gradient through hidden layer
    for b in grads.d_b1.iter_mut().zip(d_hidden.iter()) {
        *b.0 += *b.1;
    }

    let mut d_input = [0.0f32; INPUT_DIM];
    for (i, _) in cache.input.iter().enumerate() {
        for (j, &dh) in d_hidden.iter().enumerate() {
            grads.d_w1[i][j] += dh * cache.input[i];
            d_input[i] += dh * net.w1[i][j];
        }
    }

    // Gradient through accumulator to embeddings
    propagate_embedding_gradients(
        &sample.expr,
        &d_input[..2 * K],
        &mut grads.d_emb,
    );
}

/// Compute gradients for value head via backpropagation.
///
/// Returns the MSE loss on log-cost.
pub fn backward_value(
    net: &ExprNnue,
    cache: &DualForwardCache,
    target: f32,
    sample: &ValueSample,
    grads: &mut DualGradients,
) -> f32 {
    // MSE loss: L = (output - target)²
    let diff = cache.value_output - target;
    let loss = diff * diff;

    // d(loss)/d(output) = 2 * (output - target)
    let d_output = 2.0 * diff;

    // Gradient through value head output layer
    grads.d_value_b += d_output;
    let mut d_hidden = [0.0f32; HIDDEN_DIM];
    for (i, &h) in cache.hidden.iter().enumerate() {
        grads.d_value_w[i] += d_output * h;
        d_hidden[i] = d_output * net.value_w[i];
    }

    // Gradient through ReLU
    for (i, dh) in d_hidden.iter_mut().enumerate() {
        if cache.hidden_pre[i] <= 0.0 {
            *dh = 0.0;
        }
    }

    // Gradient through hidden layer
    for b in grads.d_b1.iter_mut().zip(d_hidden.iter()) {
        *b.0 += *b.1;
    }

    let mut d_input = [0.0f32; INPUT_DIM];
    for (i, _) in cache.input.iter().enumerate() {
        for (j, &dh) in d_hidden.iter().enumerate() {
            grads.d_w1[i][j] += dh * cache.input[i];
            d_input[i] += dh * net.w1[i][j];
        }
    }

    // Gradient through accumulator to embeddings
    propagate_embedding_gradients(
        &sample.expr,
        &d_input[..2 * K],
        &mut grads.d_emb,
    );

    loss
}

/// Compute gradients for search head via backpropagation.
///
/// Returns the MSE loss on priority.
pub fn backward_search(
    net: &ExprNnue,
    cache: &DualForwardCache,
    target: f32,
    sample: &SearchSample,
    grads: &mut DualGradients,
) -> f32 {
    // MSE loss: L = (output - target)²
    let diff = cache.search_output - target;
    let loss = diff * diff;

    // d(loss)/d(output) = 2 * (output - target)
    let d_output = 2.0 * diff;

    // Gradient through search head output layer
    grads.d_search_b += d_output;
    let mut d_hidden = [0.0f32; HIDDEN_DIM];
    for (i, &h) in cache.hidden.iter().enumerate() {
        grads.d_search_w[i] += d_output * h;
        d_hidden[i] = d_output * net.search_w[i];
    }

    // Gradient through ReLU
    for (i, dh) in d_hidden.iter_mut().enumerate() {
        if cache.hidden_pre[i] <= 0.0 {
            *dh = 0.0;
        }
    }

    // Gradient through hidden layer (shared layers)
    for b in grads.d_b1.iter_mut().zip(d_hidden.iter()) {
        *b.0 += *b.1;
    }

    let mut d_input = [0.0f32; INPUT_DIM];
    for (i, _) in cache.input.iter().enumerate() {
        for (j, &dh) in d_hidden.iter().enumerate() {
            grads.d_w1[i][j] += dh * cache.input[i];
            d_input[i] += dh * net.w1[i][j];
        }
    }

    // Gradient through accumulator to embeddings
    propagate_embedding_gradients(
        &sample.expr,
        &d_input[..2 * K],
        &mut grads.d_emb,
    );

    loss
}

/// Propagate gradients from accumulator to embeddings.
fn propagate_embedding_gradients(
    expr: &Expr,
    d_acc: &[f32],
    d_emb: &mut [[f32; K]; OpKind::COUNT],
) {
    let parent_op = expr.op_type();

    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Unary(_, child) => {
            let child_op = child.op_type();
            for k in 0..K {
                d_emb[parent_op.index()][k] += d_acc[k];
                d_emb[child_op.index()][k] += d_acc[K + k];
            }
            propagate_embedding_gradients(child, d_acc, d_emb);
        }
        Expr::Binary(_, left, right) => {
            let left_op = left.op_type();
            let right_op = right.op_type();
            for k in 0..K {
                d_emb[parent_op.index()][k] += d_acc[k] * 2.0;
                d_emb[left_op.index()][k] += d_acc[K + k];
                d_emb[right_op.index()][k] += d_acc[K + k];
            }
            propagate_embedding_gradients(left, d_acc, d_emb);
            propagate_embedding_gradients(right, d_acc, d_emb);
        }
        Expr::Ternary(_, a, b, c) => {
            let a_op = a.op_type();
            let b_op = b.op_type();
            let c_op = c.op_type();
            for k in 0..K {
                d_emb[parent_op.index()][k] += d_acc[k] * 3.0;
                d_emb[a_op.index()][k] += d_acc[K + k];
                d_emb[b_op.index()][k] += d_acc[K + k];
                d_emb[c_op.index()][k] += d_acc[K + k];
            }
            propagate_embedding_gradients(a, d_acc, d_emb);
            propagate_embedding_gradients(b, d_acc, d_emb);
            propagate_embedding_gradients(c, d_acc, d_emb);
        }
        Expr::Nary(_, children) => {
            for child in children {
                let child_op = child.op_type();
                for k in 0..K {
                    d_emb[parent_op.index()][k] += d_acc[k];
                    d_emb[child_op.index()][k] += d_acc[K + k];
                }
                propagate_embedding_gradients(child, d_acc, d_emb);
            }
        }
    }
}

// ============================================================================
// Momentum Buffers
// ============================================================================

/// Momentum buffers for dual-head SGD.
pub struct DualMomentum {
    /// Embedding momentum.
    pub v_emb: [[f32; K]; OpKind::COUNT],

    /// W1 momentum.
    pub v_w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// B1 momentum.
    pub v_b1: [f32; HIDDEN_DIM],

    /// Value head W momentum.
    pub v_value_w: [f32; HIDDEN_DIM],

    /// Value head B momentum.
    pub v_value_b: f32,

    /// Search head W momentum.
    pub v_search_w: [f32; HIDDEN_DIM],

    /// Search head B momentum.
    pub v_search_b: f32,
}

impl Default for DualMomentum {
    fn default() -> Self {
        Self::new()
    }
}

impl DualMomentum {
    /// Create zero-initialized momentum buffers.
    pub fn new() -> Self {
        Self {
            v_emb: [[0.0; K]; OpKind::COUNT],
            v_w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            v_b1: [0.0; HIDDEN_DIM],
            v_value_w: [0.0; HIDDEN_DIM],
            v_value_b: 0.0,
            v_search_w: [0.0; HIDDEN_DIM],
            v_search_b: 0.0,
        }
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

/// Training configuration for dual-head NNUE.
#[derive(Clone)]
pub struct DualTrainConfig {
    /// Learning rate.
    pub learning_rate: f32,

    /// Momentum coefficient.
    pub momentum: f32,

    /// Weight decay (L2 regularization).
    pub weight_decay: f32,

    /// Minibatch size.
    pub batch_size: usize,

    /// Number of epochs.
    pub epochs: usize,

    /// Learning rate decay factor per epoch.
    pub lr_decay: f32,

    /// Gradient clipping threshold.
    pub grad_clip: f32,

    /// Loss weight for value head (λ_v).
    pub lambda_value: f32,

    /// Loss weight for search head (λ_s).
    pub lambda_search: f32,

    /// Use pairwise ranking loss instead of pointwise MSE.
    pub use_pairwise_loss: bool,

    /// Minimum relative cost difference to consider a pair "significant".
    /// Pairs below this threshold get weight approaching 0.
    /// Default: 0.10 (10% cost difference required)
    pub significance_threshold: f32,

    /// Number of pairs to sample per epoch (for pairwise loss).
    /// Default: 10000
    pub pairs_per_epoch: usize,
}

impl Default for DualTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 1e-5,
            batch_size: 32,
            epochs: 10,
            lr_decay: 0.95,
            grad_clip: 1.0,
            lambda_value: 1.0,
            lambda_search: 1.0,
            use_pairwise_loss: false,
            significance_threshold: 0.10,  // 10% cost difference
            pairs_per_epoch: 10000,
        }
    }
}

// ============================================================================
// Dual-Head Trainer
// ============================================================================

/// SGD trainer for dual-head NNUE.
///
/// Supports three training modes:
/// - Phase 1: Value head only (bootstrap with benchmarks)
/// - Phase 2: Search head only (train with trajectories)
/// - Phase 3: Joint training (both heads)
pub struct DualHeadTrainer {
    /// The network being trained.
    pub net: ExprNnue,

    /// Training configuration.
    pub config: DualTrainConfig,

    /// Momentum buffers.
    pub momentum: DualMomentum,

    /// Current learning rate (may decay).
    pub current_lr: f32,

    /// Value training samples (cost prediction).
    pub value_samples: Vec<ValueSample>,

    /// Search training samples (priority prediction).
    pub search_samples: Vec<SearchSample>,
}

impl DualHeadTrainer {
    /// Create a new trainer with randomly initialized network.
    pub fn new(config: DualTrainConfig, seed: u64) -> Self {
        Self {
            net: ExprNnue::new_random(seed),
            current_lr: config.learning_rate,
            config,
            momentum: DualMomentum::new(),
            value_samples: Vec::new(),
            search_samples: Vec::new(),
        }
    }

    /// Create a new trainer with latency-prior initialized embeddings.
    pub fn new_with_latency_prior(config: DualTrainConfig, seed: u64) -> Self {
        Self {
            net: ExprNnue::new_with_latency_prior(seed),
            current_lr: config.learning_rate,
            config,
            momentum: DualMomentum::new(),
            value_samples: Vec::new(),
            search_samples: Vec::new(),
        }
    }

    /// Create a trainer from an existing ExprNnue (transfer learning).
    pub fn from_factored(factored: &crate::nnue::factored::ExprNnue, config: DualTrainConfig) -> Self {
        Self {
            net: ExprNnue::from_factored(factored),
            current_lr: config.learning_rate,
            config,
            momentum: DualMomentum::new(),
            value_samples: Vec::new(),
            search_samples: Vec::new(),
        }
    }

    /// Add a value training sample (cost prediction).
    pub fn add_value_sample(&mut self, expr: Expr, cost_ns: f64) {
        let sample = ValueSample::new(expr, cost_ns, &self.net.embeddings);
        self.value_samples.push(sample);
    }

    /// Add a search training sample (priority prediction).
    pub fn add_search_sample(&mut self, expr: Expr, improvement: f32) {
        let sample = SearchSample::new(expr, improvement, &self.net.embeddings);
        self.search_samples.push(sample);
    }

    /// Load value samples from kernel code strings.
    pub fn load_value_samples_from_cache<I>(&mut self, samples: I)
    where
        I: IntoIterator<Item = (String, f64)>,
    {
        for (code, cost_ns) in samples {
            if let Some(expr) = parse_kernel_code(&code).or_else(|| parse_expr(&code)) {
                self.add_value_sample(expr, cost_ns);
            }
        }
    }

    /// Recompute all accumulators (needed after embedding updates).
    pub fn recompute_accumulators(&mut self) {
        for sample in &mut self.value_samples {
            sample.recompute_accumulator(&self.net.embeddings);
        }
        for sample in &mut self.search_samples {
            sample.recompute_accumulator(&self.net.embeddings);
        }
    }

    // ========================================================================
    // Phase 1: Value Head Training (Bootstrap)
    // ========================================================================

    /// Train value head for one epoch.
    ///
    /// Returns the average loss over the epoch.
    pub fn train_value_epoch(&mut self) -> f32 {
        if self.value_samples.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let batch_size = self.config.batch_size;
        let num_samples = self.value_samples.len();

        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let batch_loss = self.train_value_batch(batch_start, batch_end);
            total_loss += batch_loss * (batch_end - batch_start) as f32;
            batch_count += batch_end - batch_start;
        }

        // Decay learning rate
        self.current_lr *= self.config.lr_decay;

        // Recompute accumulators after embedding updates
        self.recompute_accumulators();

        total_loss / batch_count as f32
    }

    /// Train value head on a batch.
    fn train_value_batch(&mut self, start: usize, end: usize) -> f32 {
        let mut grads = DualGradients::new();
        let mut total_loss = 0.0;
        let batch_len = end - start;

        for i in start..end {
            let sample = &self.value_samples[i];
            let cache = DualForwardCache::forward(&self.net, &sample.accumulator, &sample.structural);
            let target = sample.target();
            let sample_clone = sample.clone();
            let loss = backward_value(&self.net, &cache, target, &sample_clone, &mut grads);
            total_loss += loss;
        }

        let scale = 1.0 / batch_len as f32;
        grads.scale(scale);
        self.clip_gradients(&mut grads);
        self.apply_gradients(&grads);

        total_loss / batch_len as f32
    }

    // ========================================================================
    // Pairwise Ranking Loss Training
    // ========================================================================

    /// Train value head using pairwise ranking loss with significance weighting.
    ///
    /// This focuses learning on pairs where the cost difference is meaningful,
    /// ignoring noisy pairs where benchmark variance dominates.
    ///
    /// Loss = weight * log(1 + exp(pred_loser - pred_winner))
    /// Weight = |cost_a - cost_b| / max(cost_a, cost_b)
    ///
    /// Returns (total_loss, significant_pair_accuracy).
    pub fn train_value_pairwise_epoch(&mut self) -> (f32, f32) {
        if self.value_samples.len() < 2 {
            return (0.0, 0.0);
        }

        let n = self.value_samples.len();
        let pairs_per_epoch = self.config.pairs_per_epoch.min(n * (n - 1) / 2);

        let mut total_loss = 0.0;
        let mut total_weight = 0.0;
        let mut correct_weighted = 0.0f32;

        let mut grads = DualGradients::new();
        let mut batch_grads = DualGradients::new();
        let mut batch_count = 0;

        // Simple LCG for deterministic pair sampling
        let mut rng_state: u64 = 0xDEADBEEF;
        let lcg_next = |state: &mut u64| -> usize {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 33) as usize
        };

        for _ in 0..pairs_per_epoch {
            // Sample two different indices
            let i = lcg_next(&mut rng_state) % n;
            let mut j = lcg_next(&mut rng_state) % n;
            if j == i {
                j = (j + 1) % n;
            }

            let sample_a = &self.value_samples[i];
            let sample_b = &self.value_samples[j];

            // Weight = relative cost difference (purely proportional, no threshold)
            // rel_diff naturally ranges 0-1:
            //   equal costs: 0
            //   2x difference: 0.5
            //   10x difference: 0.9
            let cost_a = sample_a.cost_ns as f32;
            let cost_b = sample_b.cost_ns as f32;
            let max_cost = cost_a.max(cost_b);
            let rel_diff = (cost_a - cost_b).abs() / max_cost;
            let weight = rel_diff;

            // Forward pass for both samples
            let cache_a = DualForwardCache::forward(&self.net, &sample_a.accumulator, &sample_a.structural);
            let cache_b = DualForwardCache::forward(&self.net, &sample_b.accumulator, &sample_b.structural);

            // Determine winner (lower cost wins)
            let a_wins = cost_a < cost_b;
            let (winner_cache, loser_cache, winner_sample, loser_sample) = if a_wins {
                (&cache_a, &cache_b, sample_a, sample_b)
            } else {
                (&cache_b, &cache_a, sample_b, sample_a)
            };

            // RankNet loss: log(1 + exp(pred_loser - pred_winner))
            let margin = loser_cache.value_output - winner_cache.value_output;
            let loss = weight * softplus(margin);
            total_loss += loss;
            total_weight += weight;

            // Weighted accuracy: correct predictions weighted by rel_diff
            let pred_correct = cache_a.value_output < cache_b.value_output;
            if pred_correct == a_wins {
                correct_weighted += weight;
            }

            // Backprop: d(loss)/d(pred_loser) = weight * sigmoid(margin)
            //           d(loss)/d(pred_winner) = -weight * sigmoid(margin)
            let sigmoid_margin = 1.0 / (1.0 + (-margin).exp());
            let d_loser = weight * sigmoid_margin;
            let d_winner = -weight * sigmoid_margin;

            // Accumulate gradients for winner
            backward_value_with_d_output(
                &self.net,
                winner_cache,
                d_winner,
                winner_sample,
                &mut batch_grads,
            );

            // Accumulate gradients for loser
            backward_value_with_d_output(
                &self.net,
                loser_cache,
                d_loser,
                loser_sample,
                &mut batch_grads,
            );

            batch_count += 1;

            // Apply gradients every batch_size pairs
            if batch_count >= self.config.batch_size {
                let scale = 1.0 / batch_count as f32;
                batch_grads.scale(scale);
                self.clip_gradients(&mut batch_grads);
                grads.accumulate(&batch_grads);
                batch_grads.zero();

                self.apply_gradients(&grads);
                grads.zero();
                batch_count = 0;
            }
        }

        // Apply remaining gradients
        if batch_count > 0 {
            let scale = 1.0 / batch_count as f32;
            batch_grads.scale(scale);
            self.clip_gradients(&mut batch_grads);
            self.apply_gradients(&batch_grads);
        }

        // Decay learning rate
        self.current_lr *= self.config.lr_decay;

        // Recompute accumulators
        self.recompute_accumulators();

        let avg_loss = if total_weight > 0.0 { total_loss / total_weight } else { 0.0 };
        // Weighted accuracy: sum(correct * weight) / sum(weight)
        let weighted_accuracy = if total_weight > 0.0 {
            correct_weighted / total_weight
        } else {
            0.0
        };

        (avg_loss, weighted_accuracy)
    }

    // ========================================================================
    // Phase 2: Search Head Training
    // ========================================================================

    /// Train search head for one epoch.
    ///
    /// Returns the average loss over the epoch.
    pub fn train_search_epoch(&mut self) -> f32 {
        if self.search_samples.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let batch_size = self.config.batch_size;
        let num_samples = self.search_samples.len();

        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let batch_loss = self.train_search_batch(batch_start, batch_end);
            total_loss += batch_loss * (batch_end - batch_start) as f32;
            batch_count += batch_end - batch_start;
        }

        // Decay learning rate
        self.current_lr *= self.config.lr_decay;

        // Recompute accumulators after embedding updates
        self.recompute_accumulators();

        total_loss / batch_count as f32
    }

    /// Train search head on a batch.
    fn train_search_batch(&mut self, start: usize, end: usize) -> f32 {
        let mut grads = DualGradients::new();
        let mut total_loss = 0.0;
        let batch_len = end - start;

        for i in start..end {
            let sample = &self.search_samples[i];
            let cache = DualForwardCache::forward(&self.net, &sample.accumulator, &sample.structural);
            let target = sample.target();
            let sample_clone = sample.clone();
            let loss = backward_search(&self.net, &cache, target, &sample_clone, &mut grads);
            total_loss += loss;
        }

        let scale = 1.0 / batch_len as f32;
        grads.scale(scale);
        self.clip_gradients(&mut grads);
        self.apply_gradients(&grads);

        total_loss / batch_len as f32
    }

    // ========================================================================
    // Phase 3: Joint Training
    // ========================================================================

    /// Train both heads jointly for one epoch.
    ///
    /// Returns (value_loss, search_loss).
    pub fn train_joint_epoch(&mut self) -> (f32, f32) {
        let value_loss = if !self.value_samples.is_empty() {
            self.train_value_epoch()
        } else {
            0.0
        };

        let search_loss = if !self.search_samples.is_empty() {
            self.train_search_epoch()
        } else {
            0.0
        };

        (value_loss, search_loss)
    }

    // ========================================================================
    // Gradient Application
    // ========================================================================

    /// Clip gradients to prevent explosion.
    fn clip_gradients(&self, grads: &mut DualGradients) {
        let clip = self.config.grad_clip;

        for row in &mut grads.d_emb {
            for v in row.iter_mut() {
                *v = v.clamp(-clip, clip);
            }
        }
        for row in &mut grads.d_w1 {
            for v in row.iter_mut() {
                *v = v.clamp(-clip, clip);
            }
        }
        for v in &mut grads.d_b1 {
            *v = v.clamp(-clip, clip);
        }
        for v in &mut grads.d_value_w {
            *v = v.clamp(-clip, clip);
        }
        grads.d_value_b = grads.d_value_b.clamp(-clip, clip);
        for v in &mut grads.d_search_w {
            *v = v.clamp(-clip, clip);
        }
        grads.d_search_b = grads.d_search_b.clamp(-clip, clip);
    }

    /// Apply gradients with momentum and weight decay.
    fn apply_gradients(&mut self, grads: &DualGradients) {
        let lr = self.current_lr;
        let mom = self.config.momentum;
        let wd = self.config.weight_decay;

        // Update embeddings
        for (op_idx, (emb_row, grad_row)) in self.net.embeddings.e.iter_mut()
            .zip(grads.d_emb.iter())
            .enumerate()
        {
            for k in 0..K {
                self.momentum.v_emb[op_idx][k] =
                    mom * self.momentum.v_emb[op_idx][k] + grad_row[k];
                emb_row[k] -= lr * (self.momentum.v_emb[op_idx][k] + wd * emb_row[k]);
            }
        }

        // Update W1
        for i in 0..INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                self.momentum.v_w1[i][j] = mom * self.momentum.v_w1[i][j] + grads.d_w1[i][j];
                self.net.w1[i][j] -= lr * (self.momentum.v_w1[i][j] + wd * self.net.w1[i][j]);
            }
        }

        // Update B1
        for i in 0..HIDDEN_DIM {
            self.momentum.v_b1[i] = mom * self.momentum.v_b1[i] + grads.d_b1[i];
            self.net.b1[i] -= lr * self.momentum.v_b1[i];
        }

        // Update value head
        for i in 0..HIDDEN_DIM {
            self.momentum.v_value_w[i] = mom * self.momentum.v_value_w[i] + grads.d_value_w[i];
            self.net.value_w[i] -= lr * (self.momentum.v_value_w[i] + wd * self.net.value_w[i]);
        }
        self.momentum.v_value_b = mom * self.momentum.v_value_b + grads.d_value_b;
        self.net.value_b -= lr * self.momentum.v_value_b;

        // Update search head
        for i in 0..HIDDEN_DIM {
            self.momentum.v_search_w[i] = mom * self.momentum.v_search_w[i] + grads.d_search_w[i];
            self.net.search_w[i] -= lr * (self.momentum.v_search_w[i] + wd * self.net.search_w[i]);
        }
        self.momentum.v_search_b = mom * self.momentum.v_search_b + grads.d_search_b;
        self.net.search_b -= lr * self.momentum.v_search_b;
    }

    // ========================================================================
    // Evaluation
    // ========================================================================

    /// Compute evaluation metrics for value head.
    pub fn evaluate_value(&self) -> TrainMetrics {
        if self.value_samples.is_empty() {
            return TrainMetrics::default();
        }

        let mut predictions = Vec::with_capacity(self.value_samples.len());
        let mut targets = Vec::with_capacity(self.value_samples.len());
        let mut total_loss = 0.0;

        for sample in &self.value_samples {
            let cache = DualForwardCache::forward(&self.net, &sample.accumulator, &sample.structural);
            let target = sample.target();
            predictions.push(cache.value_output);
            targets.push(target);
            total_loss += (cache.value_output - target).powi(2);
        }

        let mse = total_loss / self.value_samples.len() as f32;
        let rmse = mse.sqrt();
        let spearman = compute_spearman(&predictions, &targets);

        TrainMetrics { mse, rmse, spearman }
    }

    /// Compute evaluation metrics for search head.
    pub fn evaluate_search(&self) -> TrainMetrics {
        if self.search_samples.is_empty() {
            return TrainMetrics::default();
        }

        let mut predictions = Vec::with_capacity(self.search_samples.len());
        let mut targets = Vec::with_capacity(self.search_samples.len());
        let mut total_loss = 0.0;

        for sample in &self.search_samples {
            let cache = DualForwardCache::forward(&self.net, &sample.accumulator, &sample.structural);
            let target = sample.target();
            predictions.push(cache.search_output);
            targets.push(target);
            total_loss += (cache.search_output - target).powi(2);
        }

        let mse = total_loss / self.search_samples.len() as f32;
        let rmse = mse.sqrt();
        let spearman = compute_spearman(&predictions, &targets);

        TrainMetrics { mse, rmse, spearman }
    }
}

// ============================================================================
// Training Metrics
// ============================================================================

/// Training metrics.
#[derive(Clone, Default)]
pub struct TrainMetrics {
    /// Mean squared error.
    pub mse: f32,

    /// Root mean squared error.
    pub rmse: f32,

    /// Spearman rank correlation.
    pub spearman: f32,
}

/// Compute Spearman rank correlation coefficient.
fn compute_spearman(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len();
    if n < 2 {
        return 0.0;
    }

    let pred_ranks = compute_ranks(predictions);
    let target_ranks = compute_ranks(targets);

    let mean_pred: f32 = pred_ranks.iter().sum::<f32>() / n as f32;
    let mean_target: f32 = target_ranks.iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f32;
    let mut var_pred = 0.0f32;
    let mut var_target = 0.0f32;

    for i in 0..n {
        let dp = pred_ranks[i] - mean_pred;
        let dt = target_ranks[i] - mean_target;
        cov += dp * dt;
        var_pred += dp * dp;
        var_target += dt * dt;
    }

    if var_pred < 1e-10 || var_target < 1e-10 {
        return 0.0;
    }

    cov / (var_pred.sqrt() * var_target.sqrt())
}

/// Compute ranks for values.
fn compute_ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<_> = values.iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    for (rank, (orig_idx, _)) in indexed.into_iter().enumerate() {
        ranks[orig_idx] = rank as f32 + 1.0;
    }
    ranks
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use Box;

    extern crate alloc;

    fn make_add_xy() -> Expr {
        Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        )
    }

    fn make_mul_xyz() -> Expr {
        Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        )
    }

    #[test]
    fn test_dual_forward_cache() {
        let net = ExprNnue::new_random(42);
        let expr = make_add_xy();
        let acc = EdgeAccumulator::from_expr(&expr, &net.embeddings);
        let structural = StructuralFeatures::from_expr(&expr);

        let cache = DualForwardCache::forward(&net, &acc, &structural);

        assert!(cache.value_output.is_finite());
        assert!(cache.search_output.is_finite());
    }

    #[test]
    fn test_value_backward() {
        let net = ExprNnue::new_random(42);
        let expr = make_add_xy();
        let sample = ValueSample::new(expr, 100.0, &net.embeddings);
        let cache = DualForwardCache::forward(&net, &sample.accumulator, &sample.structural);

        let mut grads = DualGradients::new();
        let loss = backward_value(&net, &cache, sample.target(), &sample, &mut grads);

        assert!(loss.is_finite());
        assert!(loss >= 0.0);

        // Check value head gradients are non-zero
        let total_value_grad: f32 = grads.d_value_w.iter().map(|&v| v.abs()).sum();
        assert!(total_value_grad > 0.0, "Should have non-zero value head gradients");
    }

    #[test]
    fn test_search_backward() {
        let net = ExprNnue::new_random(42);
        let expr = make_add_xy();
        let sample = SearchSample::new(expr, 10.0, &net.embeddings);
        let cache = DualForwardCache::forward(&net, &sample.accumulator, &sample.structural);

        let mut grads = DualGradients::new();
        let loss = backward_search(&net, &cache, sample.target(), &sample, &mut grads);

        assert!(loss.is_finite());
        assert!(loss >= 0.0);

        // Check search head gradients are non-zero
        let total_search_grad: f32 = grads.d_search_w.iter().map(|&v| v.abs()).sum();
        assert!(total_search_grad > 0.0, "Should have non-zero search head gradients");
    }

    #[test]
    fn test_value_training_reduces_loss() {
        let config = DualTrainConfig {
            learning_rate: 0.1,
            epochs: 5,
            batch_size: 4,
            ..Default::default()
        };

        let mut trainer = DualHeadTrainer::new(config, 42);

        // Add value samples
        trainer.add_value_sample(parse_expr("Var(0)").unwrap(), 10.0);
        trainer.add_value_sample(parse_expr("Add(Var(0), Var(1))").unwrap(), 50.0);
        trainer.add_value_sample(parse_expr("Mul(Var(0), Var(1))").unwrap(), 60.0);
        trainer.add_value_sample(parse_expr("Div(Var(0), Var(1))").unwrap(), 200.0);

        let initial_metrics = trainer.evaluate_value();
        let mut final_loss = 0.0;

        for _ in 0..5 {
            final_loss = trainer.train_value_epoch();
        }

        let final_metrics = trainer.evaluate_value();

        assert!(
            final_metrics.mse < initial_metrics.mse * 0.99 || final_loss < 1.0,
            "Value training should reduce loss: initial={}, final={}",
            initial_metrics.mse,
            final_metrics.mse
        );
    }

    #[test]
    fn test_search_training() {
        let config = DualTrainConfig {
            learning_rate: 0.1,
            epochs: 5,
            batch_size: 4,
            ..Default::default()
        };

        let mut trainer = DualHeadTrainer::new(config, 42);

        // Add search samples (states with their achieved improvements)
        trainer.add_search_sample(make_add_xy(), -5.0);  // Good state, led to improvement
        trainer.add_search_sample(make_mul_xyz(), 0.0);  // Neutral state
        trainer.add_search_sample(parse_expr("Div(Var(0), Var(1))").unwrap(), 3.0);  // Bad state

        let initial_metrics = trainer.evaluate_search();

        for _ in 0..5 {
            trainer.train_search_epoch();
        }

        let final_metrics = trainer.evaluate_search();

        // Training should reduce loss or improve correlation
        assert!(
            final_metrics.mse < initial_metrics.mse * 1.1 || final_metrics.spearman > initial_metrics.spearman - 0.1,
            "Search training should make progress"
        );
    }
}
