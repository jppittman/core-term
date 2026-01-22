//! # NNUE Trainer for Expression Cost Prediction
//!
//! Trains NNUE weights from benchmark data using gradient descent.
//!
//! ## Training Pipeline
//!
//! 1. Collect (Expr features, runtime_ns) pairs from benchmarks
//! 2. Convert to training samples with sparse HalfEP features
//! 3. Train via gradient descent to minimize MSE
//! 4. Export weights for use in e-graph cost function
//!
//! ## Loss Function
//!
//! We optimize for ranking accuracy (Spearman correlation) indirectly by
//! minimizing MSE on log-transformed costs. This helps because:
//! - Log transform reduces impact of outliers
//! - Relative cost differences matter more than absolute
//! - Ranking is what matters for instruction selection

extern crate alloc;

use alloc::vec::Vec;
use crate::nnue::{
    Expr, Nnue, NnueConfig, Accumulator, HalfEPFeature,
    extract_features,
};
use libm::logf;

// ============================================================================
// Training Sample
// ============================================================================

/// A training sample with features and target cost.
#[derive(Clone, Debug)]
pub struct NnueSample {
    /// Sparse feature indices (sorted, deduplicated).
    pub feature_indices: Vec<usize>,
    /// Target cost (in arbitrary units, usually nanoseconds).
    pub target_cost: f32,
    /// Log-transformed target for training.
    pub log_target: f32,
}

impl NnueSample {
    /// Create from an expression and measured cost.
    pub fn from_expr(expr: &Expr, cost_ns: f32) -> Self {
        let features = extract_features(expr);
        let mut indices: Vec<usize> = features.iter()
            .map(|f| f.to_index())
            .collect();
        indices.sort_unstable();
        indices.dedup();

        let log_target = logf(cost_ns.max(1.0));

        Self {
            feature_indices: indices,
            target_cost: cost_ns,
            log_target,
        }
    }

    /// Create from pre-computed feature indices.
    pub fn from_features(feature_indices: Vec<usize>, cost_ns: f32) -> Self {
        let mut indices = feature_indices;
        indices.sort_unstable();
        indices.dedup();

        let log_target = logf(cost_ns.max(1.0));

        Self {
            feature_indices: indices,
            target_cost: cost_ns,
            log_target,
        }
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

/// Configuration for NNUE training.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Learning rate for gradient descent.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// L2 regularization strength.
    pub l2_lambda: f32,
    /// Whether to use log-transformed targets.
    pub use_log_transform: bool,
    /// Print progress every N epochs.
    pub print_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            l2_lambda: 0.0001,
            use_log_transform: true,
            print_every: 10,
        }
    }
}

// ============================================================================
// NNUE Trainer
// ============================================================================

/// Training state and methods for NNUE.
pub struct NnueTrainer {
    /// The NNUE network being trained.
    pub nnue: Nnue,
    /// Training configuration.
    pub config: TrainConfig,
    /// Training samples.
    samples: Vec<NnueSample>,
    /// Random state for shuffling.
    rng_state: u64,
}

impl NnueTrainer {
    /// Create a new trainer with default config.
    pub fn new() -> Self {
        Self {
            nnue: Nnue::with_defaults(),
            config: TrainConfig::default(),
            samples: Vec::new(),
            rng_state: 42,
        }
    }

    /// Create a new trainer with custom config.
    pub fn with_config(nnue_config: NnueConfig, train_config: TrainConfig) -> Self {
        Self {
            nnue: Nnue::new(nnue_config),
            config: train_config,
            samples: Vec::new(),
            rng_state: 42,
        }
    }

    /// Add a training sample.
    pub fn add_sample(&mut self, sample: NnueSample) {
        self.samples.push(sample);
    }

    /// Add multiple samples from (Expr, cost) pairs.
    pub fn add_expr_samples(&mut self, pairs: &[(&Expr, f32)]) {
        for (expr, cost) in pairs {
            self.samples.push(NnueSample::from_expr(expr, *cost));
        }
    }

    /// Simple LCG random number generator.
    fn rand_f32(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Shuffle samples for mini-batch training.
    fn shuffle_samples(&mut self) {
        let n = self.samples.len();
        for i in (1..n).rev() {
            let j = (self.rand_f32() * (i + 1) as f32) as usize;
            self.samples.swap(i, j);
        }
    }

    /// Forward pass for a single sample.
    fn forward(&self, sample: &NnueSample) -> (i32, Accumulator) {
        let mut acc = Accumulator::new(&self.nnue);

        // Add all features
        for &idx in &sample.feature_indices {
            acc.add_feature(&self.nnue, idx);
        }

        let output = acc.forward(&self.nnue);
        (output, acc)
    }

    /// Compute loss for a batch of samples.
    fn compute_batch_loss(&self, samples: &[NnueSample]) -> f32 {
        let mut total_loss = 0.0f32;

        for sample in samples {
            let (output, _) = self.forward(sample);
            let prediction = output as f32 / 64.0; // Scale factor

            let target = if self.config.use_log_transform {
                sample.log_target
            } else {
                sample.target_cost
            };

            let diff = prediction - target;
            total_loss += diff * diff;
        }

        total_loss / samples.len() as f32
    }

    /// Train the network.
    ///
    /// Returns training history (loss per epoch).
    #[cfg(feature = "std")]
    pub fn train(&mut self) -> Vec<f32> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let mut history = Vec::with_capacity(self.config.epochs);

        // Initialize weights with small random values
        self.initialize_weights();

        for epoch in 0..self.config.epochs {
            self.shuffle_samples();

            let mut epoch_loss = 0.0f32;
            let num_batches = (self.samples.len() + self.config.batch_size - 1)
                / self.config.batch_size;

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(self.samples.len());
                // Clone batch to avoid borrow conflict
                let batch: Vec<_> = self.samples[start..end].to_vec();

                let batch_loss = self.train_batch(&batch);
                epoch_loss += batch_loss;
            }

            epoch_loss /= num_batches as f32;
            history.push(epoch_loss);

            if self.config.print_every > 0 && (epoch + 1) % self.config.print_every == 0 {
                eprintln!("Epoch {}/{}: loss = {:.6}", epoch + 1, self.config.epochs, epoch_loss);
            }
        }

        history
    }

    /// Initialize weights with small random values.
    fn initialize_weights(&mut self) {
        // Pre-generate random values to avoid borrow conflicts
        let w1_len = self.nnue.w1.len();
        let w2_len = self.nnue.w2.len();
        let w3_len = self.nnue.w3.len();
        let w_out_len = self.nnue.w_out.len();

        let scale_w1 = 1.0 / (HalfEPFeature::COUNT as f32).sqrt();
        let scale_w2 = 1.0 / (self.nnue.config.l1_size as f32).sqrt();
        let scale_w3 = 1.0 / (self.nnue.config.l2_size as f32).sqrt();
        let scale_out = 1.0 / (self.nnue.config.l3_size as f32).sqrt();

        // Generate random values
        let mut w1_vals: Vec<i16> = Vec::with_capacity(w1_len);
        for _ in 0..w1_len {
            w1_vals.push(((self.rand_f32() - 0.5) * 2.0 * scale_w1 * 32767.0) as i16);
        }

        let mut w2_vals: Vec<i8> = Vec::with_capacity(w2_len);
        for _ in 0..w2_len {
            w2_vals.push(((self.rand_f32() - 0.5) * 2.0 * scale_w2 * 127.0) as i8);
        }

        let mut w3_vals: Vec<i8> = Vec::with_capacity(w3_len);
        for _ in 0..w3_len {
            w3_vals.push(((self.rand_f32() - 0.5) * 2.0 * scale_w3 * 127.0) as i8);
        }

        let mut w_out_vals: Vec<i8> = Vec::with_capacity(w_out_len);
        for _ in 0..w_out_len {
            w_out_vals.push(((self.rand_f32() - 0.5) * 2.0 * scale_out * 127.0) as i8);
        }

        // Apply to network
        self.nnue.w1.copy_from_slice(&w1_vals);
        self.nnue.w2.copy_from_slice(&w2_vals);
        self.nnue.w3.copy_from_slice(&w3_vals);
        self.nnue.w_out.copy_from_slice(&w_out_vals);

        // Zero biases
        for b in &mut self.nnue.b1 { *b = 0; }
        for b in &mut self.nnue.b2 { *b = 0; }
        for b in &mut self.nnue.b3 { *b = 0; }
        self.nnue.b_out = 0;
    }

    /// Train on a single batch using simplified gradient descent.
    ///
    /// This is a simplified training loop that updates weights based on
    /// the gradient of MSE loss. For production use, consider using
    /// proper autodiff or a more sophisticated optimizer.
    fn train_batch(&mut self, batch: &[NnueSample]) -> f32 {
        let lr = self.config.learning_rate;
        let batch_size = batch.len() as f32;

        // Accumulate gradients for output layer only (simplified)
        // Full backprop would require tracking all intermediate activations
        let _grad_w_out = vec![0.0f32; self.nnue.config.l3_size];
        let mut grad_b_out = 0.0f32;
        let mut total_loss = 0.0f32;

        for sample in batch {
            let (output, acc) = self.forward(sample);
            let prediction = output as f32 / 64.0;

            let target = if self.config.use_log_transform {
                sample.log_target
            } else {
                sample.target_cost
            };

            let error = prediction - target;
            total_loss += error * error;

            // Gradient for output layer
            // d(loss)/d(w_out[i]) = 2 * error * l3_activation[i]
            // We need L3 activations, which requires forward pass state

            // Simplified: just update output bias based on error direction
            grad_b_out += error;

            // For sparse first layer, we can update directly
            // d(loss)/d(w1[feature_idx, j]) = 2 * error * (chain rule through layers)
            // This is simplified - real training would track all gradients

            // Update first layer weights for active features
            for &idx in &sample.feature_indices {
                let offset = idx * self.nnue.config.l1_size;
                for j in 0..self.nnue.config.l1_size {
                    // Simplified gradient: error propagated through
                    let grad = error * (if acc.values[j] > 0 { 1.0 } else { 0.0 });
                    let current = self.nnue.w1[offset + j] as f32;
                    let updated = current - lr * grad / batch_size;
                    self.nnue.w1[offset + j] = updated.clamp(-32767.0, 32767.0) as i16;
                }
            }
        }

        // Update output bias
        let new_b_out = self.nnue.b_out as f32 - lr * grad_b_out / batch_size;
        self.nnue.b_out = new_b_out.clamp(-2147483648.0, 2147483647.0) as i32;

        total_loss / batch_size
    }

    /// Evaluate the network on a set of samples.
    ///
    /// Returns (predictions, targets) for correlation analysis.
    pub fn evaluate(&self, samples: &[NnueSample]) -> (Vec<f32>, Vec<f32>) {
        let mut predictions = Vec::with_capacity(samples.len());
        let mut targets = Vec::with_capacity(samples.len());

        for sample in samples {
            let (output, _) = self.forward(sample);
            predictions.push(output as f32 / 64.0);
            targets.push(sample.target_cost);
        }

        (predictions, targets)
    }

    /// Compute Spearman rank correlation between predictions and targets.
    pub fn spearman_correlation(&self, samples: &[NnueSample]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let (predictions, targets) = self.evaluate(samples);

        // Compute ranks
        let pred_ranks = compute_ranks(&predictions);
        let target_ranks = compute_ranks(&targets);

        // Pearson correlation of ranks
        let n = samples.len() as f32;
        let mean_pred = pred_ranks.iter().sum::<f32>() / n;
        let mean_target = target_ranks.iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_pred = 0.0f32;
        let mut var_target = 0.0f32;

        for i in 0..samples.len() {
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
}

impl Default for NnueTrainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute ranks for a vector of values.
fn compute_ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f32; n];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as f32 + 1.0;
    }

    ranks
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::OpType;

    #[test]
    fn test_sample_creation() {
        let expr = Expr::Binary(
            OpType::Add,
            alloc::boxed::Box::new(Expr::Var(0)),
            alloc::boxed::Box::new(Expr::Var(1)),
        );
        let sample = NnueSample::from_expr(&expr, 100.0);
        assert!(!sample.feature_indices.is_empty());
        assert!((sample.target_cost - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = NnueTrainer::new();
        assert_eq!(trainer.samples.len(), 0);
    }

    #[test]
    fn test_compute_ranks() {
        let values = vec![3.0, 1.0, 4.0, 1.5, 2.0];
        let ranks = compute_ranks(&values);
        // Expected: 1.0→1, 1.5→2, 2.0→3, 3.0→4, 4.0→5
        // So: [4, 1, 5, 2, 3]
        assert!((ranks[0] - 4.0).abs() < 0.001); // 3.0 is 4th smallest
        assert!((ranks[1] - 1.0).abs() < 0.001); // 1.0 is 1st smallest
        assert!((ranks[2] - 5.0).abs() < 0.001); // 4.0 is 5th smallest
        assert!((ranks[3] - 2.0).abs() < 0.001); // 1.5 is 2nd smallest
        assert!((ranks[4] - 3.0).abs() < 0.001); // 2.0 is 3rd smallest
    }
}
