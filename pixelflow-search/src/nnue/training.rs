//! Training infrastructure for DualMaskGuide.
//!
//! Uses the existing `PairRecord` and `DualMaskEpochRecord` structures from
//! `guided_search` directly for training data.
//!
//! # Architecture: Lite Transformer Style
//!
//! ```text
//! Expression Features ──► ExprHead ──► expr_vec [D]
//!                                           │
//!                                           ├─── dot product ───► score
//!                                           │
//! Rule Features ────────► RuleHead ──► rule_vec [D]
//!                                    + interaction_bias
//! ```
//!
//! The current implementation uses a simple product of scalar scores.
//! A future iteration could add a learned interaction term (covariance vector)
//! that makes this more like cross-attention.

use alloc::vec::Vec;
use core::f32;

use super::dual_mask::DualMaskGuide;
use crate::egraph::{DualMaskEpochRecord, PairRecord};

// ============================================================================
// Training Metrics
// ============================================================================

/// Metrics for evaluating DualMaskGuide performance.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    /// Total samples evaluated
    pub total: usize,
    /// True positives (predicted fire, did fire)
    pub true_positives: usize,
    /// False positives (predicted fire, didn't fire)
    pub false_positives: usize,
    /// True negatives (predicted no fire, didn't fire)
    pub true_negatives: usize,
    /// False negatives (predicted no fire, did fire)
    pub false_negatives: usize,
    /// Sum of losses
    pub loss_sum: f32,
}

impl Metrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with a prediction result.
    pub fn update(&mut self, predicted_prob: f32, actual_fired: bool, loss: f32) {
        self.total += 1;
        self.loss_sum += loss;

        let predicted_fire = predicted_prob > 0.5;
        match (predicted_fire, actual_fired) {
            (true, true) => self.true_positives += 1,
            (true, false) => self.false_positives += 1,
            (false, true) => self.false_negatives += 1,
            (false, false) => self.true_negatives += 1,
        }
    }

    /// Average loss.
    pub fn avg_loss(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.loss_sum / self.total as f32 }
    }

    /// Accuracy: (TP + TN) / total.
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 { 0.0 } else {
            (self.true_positives + self.true_negatives) as f32 / self.total as f32
        }
    }

    /// Precision: TP / (TP + FP).
    pub fn precision(&self) -> f32 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 { 0.0 } else { self.true_positives as f32 / denom as f32 }
    }

    /// Recall: TP / (TP + FN).
    pub fn recall(&self) -> f32 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 { 0.0 } else { self.true_positives as f32 / denom as f32 }
    }

    /// F1 score.
    pub fn f1(&self) -> f32 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

/// Configuration for training.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Initial learning rate
    pub lr: f32,
    /// Learning rate decay per epoch (multiply by this)
    pub lr_decay: f32,
    /// Minimum learning rate
    pub min_lr: f32,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size (number of samples per step)
    pub batch_size: usize,
    /// Weight for positive samples (to handle imbalance)
    pub positive_weight: f32,
    /// Fraction of negatives to keep (downsample if < 1.0)
    pub negative_keep_ratio: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            lr_decay: 0.95,
            min_lr: 0.0001,
            epochs: 50,
            batch_size: 32,
            positive_weight: 2.0,
            negative_keep_ratio: 0.1, // Keep 10% of negatives
            seed: 42,
        }
    }
}

// ============================================================================
// Trainer
// ============================================================================

/// Train the DualMaskGuide from search trajectory data.
///
/// This is the main entry point for training.
pub struct Trainer {
    /// The model being trained
    pub model: DualMaskGuide,
    /// Configuration
    pub config: TrainConfig,
    /// RNG state
    rng: u64,
    /// Loss history per epoch
    pub loss_history: Vec<f32>,
    /// Accuracy history per epoch
    pub acc_history: Vec<f32>,
}

impl Trainer {
    /// Create a new trainer with a randomly initialized model.
    pub fn new(config: TrainConfig) -> Self {
        let model = DualMaskGuide::new_random(config.seed);
        Self {
            model,
            rng: config.seed,
            config,
            loss_history: Vec::new(),
            acc_history: Vec::new(),
        }
    }

    /// Create with an existing model.
    pub fn with_model(model: DualMaskGuide, config: TrainConfig) -> Self {
        Self {
            model,
            rng: config.seed,
            config,
            loss_history: Vec::new(),
            acc_history: Vec::new(),
        }
    }

    /// Train from trajectory data (collected via `run_dual_mask`).
    ///
    /// Returns (final_loss, final_accuracy).
    pub fn train_from_trajectory(&mut self, trajectory: &[DualMaskEpochRecord]) -> (f32, f32) {
        // Flatten trajectory into samples, downsampling negatives
        let (positives, negatives) = self.split_by_label(trajectory);

        if positives.is_empty() {
            return (0.0, 0.0);
        }

        let mut lr = self.config.lr;

        for epoch in 0..self.config.epochs {
            let metrics = self.train_epoch(&positives, &negatives, lr);
            self.loss_history.push(metrics.avg_loss());
            self.acc_history.push(metrics.accuracy());

            // Decay learning rate
            lr = (lr * self.config.lr_decay).max(self.config.min_lr);
        }

        let final_loss = self.loss_history.last().copied().unwrap_or(0.0);
        let final_acc = self.acc_history.last().copied().unwrap_or(0.0);
        (final_loss, final_acc)
    }

    /// Split trajectory into positive and negative samples.
    fn split_by_label<'a>(
        &mut self,
        trajectory: &'a [DualMaskEpochRecord],
    ) -> (Vec<&'a PairRecord>, Vec<&'a PairRecord>) {
        let mut positives = Vec::new();
        let mut negatives = Vec::new();

        for epoch in trajectory {
            for pair in &epoch.pairs {
                if pair.fired {
                    positives.push(pair);
                } else {
                    // Downsample negatives
                    if random_f32(&mut self.rng) < self.config.negative_keep_ratio {
                        negatives.push(pair);
                    }
                }
            }
        }

        (positives, negatives)
    }

    /// Train one epoch with balanced sampling.
    fn train_epoch(
        &mut self,
        positives: &[&PairRecord],
        negatives: &[&PairRecord],
        lr: f32,
    ) -> Metrics {
        let mut metrics = Metrics::new();

        if positives.is_empty() || negatives.is_empty() {
            return metrics;
        }

        // Number of batches to cover all positives at least once
        let num_batches = (positives.len() * 2 / self.config.batch_size).max(1);
        let half_batch = self.config.batch_size / 2;

        for _ in 0..num_batches {
            // Sample balanced batch
            for _ in 0..half_batch.min(positives.len()) {
                let idx = random_usize(&mut self.rng, positives.len());
                let pair = positives[idx];
                let loss = self.train_pair(pair, lr * self.config.positive_weight);
                metrics.update(self.predict_prob(pair), pair.fired, loss);
            }

            for _ in 0..half_batch.min(negatives.len()) {
                let idx = random_usize(&mut self.rng, negatives.len());
                let pair = negatives[idx];
                let loss = self.train_pair(pair, lr);
                metrics.update(self.predict_prob(pair), pair.fired, loss);
            }
        }

        metrics
    }

    /// Train on a single pair.
    fn train_pair(&mut self, pair: &PairRecord, lr: f32) -> f32 {
        self.model.train_step(
            &pair.expr_features,
            &pair.rule_features,
            pair.rule_idx,
            pair.fired,
            lr,
        )
    }

    /// Predict probability for a pair.
    fn predict_prob(&self, pair: &PairRecord) -> f32 {
        let expr_score = self.model.score_expr(&pair.expr_features);
        let rule_score = self.model.score_rule(&pair.rule_features, pair.rule_idx);
        sigmoid(expr_score * rule_score)
    }

    /// Evaluate on trajectory without training.
    pub fn evaluate(&self, trajectory: &[DualMaskEpochRecord]) -> Metrics {
        let mut metrics = Metrics::new();

        for epoch in trajectory {
            for pair in &epoch.pairs {
                let predicted = self.predict_prob(pair);
                let target = if pair.fired { 1.0 } else { 0.0 };
                let p = predicted.clamp(1e-7, 1.0 - 1e-7);
                let loss = -(target * libm::logf(p) + (1.0 - target) * libm::logf(1.0 - p));
                metrics.update(predicted, pair.fired, loss);
            }
        }

        metrics
    }
}

// ============================================================================
// Simple Training Run (for testing)
// ============================================================================

/// Run a simple training loop on synthetic data.
///
/// This is useful for testing that training converges without needing
/// a full e-graph search.
pub fn train_synthetic(config: TrainConfig, num_samples: usize) -> (DualMaskGuide, Metrics) {
    let mut model = DualMaskGuide::new_random(config.seed);
    let mut rng = config.seed;
    let mut lr = config.lr;

    // Generate synthetic training data
    // Pattern: fired when expr_features[0] > 0.5 AND rule_features[0] > 0.5
    let mut positives = Vec::new();
    let mut negatives = Vec::new();

    for i in 0..num_samples {
        let mut expr_f = [0.0f32; super::dual_mask::EXPR_FEATURE_DIM];
        let mut rule_f = [0.0f32; super::dual_mask::RULE_FEATURE_DIM];

        // Fill with noise
        for f in &mut expr_f {
            *f = random_f32(&mut rng) * 0.5;
        }
        for f in &mut rule_f {
            *f = random_f32(&mut rng) * 0.5;
        }

        // Set signal
        let expr_high = i % 2 == 0;
        let rule_high = i % 4 < 2;
        let fired = expr_high && rule_high;

        expr_f[0] = if expr_high { 1.0 } else { -1.0 };
        rule_f[0] = if rule_high { 1.0 } else { -1.0 };

        let pair = SyntheticPair {
            expr_features: expr_f,
            rule_features: rule_f,
            rule_idx: i % 10,
            fired,
        };

        if fired {
            positives.push(pair);
        } else {
            negatives.push(pair);
        }
    }

    // Train
    for _epoch in 0..config.epochs {
        let half_batch = config.batch_size / 2;
        let num_batches = (positives.len() * 2 / config.batch_size).max(1);

        for _ in 0..num_batches {
            // Positive samples
            for _ in 0..half_batch.min(positives.len()) {
                let idx = random_usize(&mut rng, positives.len());
                let p = &positives[idx];
                model.train_step(&p.expr_features, &p.rule_features, p.rule_idx, true, lr * config.positive_weight);
            }

            // Negative samples
            for _ in 0..half_batch.min(negatives.len()) {
                let idx = random_usize(&mut rng, negatives.len());
                let p = &negatives[idx];
                model.train_step(&p.expr_features, &p.rule_features, p.rule_idx, false, lr);
            }
        }

        lr = (lr * config.lr_decay).max(config.min_lr);
    }

    // Evaluate
    let mut metrics = Metrics::new();
    for p in &positives {
        let score = model.score_expr(&p.expr_features) * model.score_rule(&p.rule_features, p.rule_idx);
        let prob = sigmoid(score);
        let loss = -libm::logf(prob.clamp(1e-7, 1.0 - 1e-7));
        metrics.update(prob, true, loss);
    }
    for p in &negatives {
        let score = model.score_expr(&p.expr_features) * model.score_rule(&p.rule_features, p.rule_idx);
        let prob = sigmoid(score);
        let loss = -libm::logf((1.0 - prob).clamp(1e-7, 1.0 - 1e-7));
        metrics.update(prob, false, loss);
    }

    (model, metrics)
}

/// Synthetic pair for testing.
struct SyntheticPair {
    expr_features: [f32; super::dual_mask::EXPR_FEATURE_DIM],
    rule_features: [f32; super::dual_mask::RULE_FEATURE_DIM],
    rule_idx: usize,
    fired: bool,
}

// ============================================================================
// Utilities
// ============================================================================

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[inline]
fn random_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

#[inline]
fn random_usize(state: &mut u64, max: usize) -> usize {
    if max == 0 { return 0; }
    (random_f32(state) * max as f32) as usize
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics() {
        let mut m = Metrics::new();
        m.update(0.9, true, 0.1);  // TP
        m.update(0.8, false, 1.0); // FP
        m.update(0.2, false, 0.2); // TN
        m.update(0.1, true, 2.0);  // FN

        assert_eq!(m.total, 4);
        assert_eq!(m.true_positives, 1);
        assert_eq!(m.false_positives, 1);
        assert_eq!(m.true_negatives, 1);
        assert_eq!(m.false_negatives, 1);
        assert!((m.accuracy() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_synthetic_training() {
        let config = TrainConfig {
            epochs: 30,
            batch_size: 16,
            lr: 0.05,
            lr_decay: 0.9,
            ..Default::default()
        };

        let (model, metrics) = train_synthetic(config, 200);

        // Model should learn the pattern
        assert!(
            metrics.accuracy() > 0.7,
            "Expected accuracy > 0.7, got {}",
            metrics.accuracy()
        );

        // Verify model scores reflect the pattern
        let mut high_high = [0.0f32; super::super::dual_mask::EXPR_FEATURE_DIM];
        let mut low_low = [0.0f32; super::super::dual_mask::EXPR_FEATURE_DIM];
        high_high[0] = 1.0;
        low_low[0] = -1.0;

        let mut rule_high = [0.0f32; super::super::dual_mask::RULE_FEATURE_DIM];
        let mut rule_low = [0.0f32; super::super::dual_mask::RULE_FEATURE_DIM];
        rule_high[0] = 1.0;
        rule_low[0] = -1.0;

        let score_high = model.score_expr(&high_high) * model.score_rule(&rule_high, 0);
        let score_low = model.score_expr(&low_low) * model.score_rule(&rule_low, 0);

        // High features should score higher
        assert!(
            score_high > score_low,
            "Expected score_high ({}) > score_low ({})",
            score_high,
            score_low
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = TrainConfig::default();
        assert!(config.lr > 0.0);
        assert!(config.epochs > 0);
        assert!(config.batch_size > 0);
    }
}
