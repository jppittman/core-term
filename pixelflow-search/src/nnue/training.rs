//! Training utilities for DualMaskGuide.
//!
//! Self-Imitation Learning (SIL) with resource-asymmetric training:
//! - Oracle runs with abundant resources (high max_classes, many epochs)
//! - Guide runs with limited resources (low max_classes, few epochs)
//! - Guide learns to match oracle quality with fewer resources
//!
//! The key insight: saturation is the limit case. An oracle with unlimited
//! resources can explore freely. A resource-constrained guide must be selective.

use super::dual_mask::{DualMaskGuide, EXPR_FEATURE_DIM, RULE_FEATURE_DIM};

/// Resource configuration for search.
#[derive(Clone, Debug)]
pub struct ResourceConfig {
    /// Maximum e-graph classes before stopping.
    pub max_classes: usize,
    /// Maximum epochs to run.
    pub max_epochs: usize,
    /// Filtering threshold (0.5 = balanced).
    pub threshold: f32,
    /// Exploration rate for epsilon-greedy.
    pub epsilon: f32,
}

impl ResourceConfig {
    /// Oracle config: abundant resources for near-saturation.
    pub fn oracle() -> Self {
        Self {
            max_classes: 500,
            max_epochs: 20,
            threshold: 0.3,  // permissive
            epsilon: 0.0,    // no exploration - oracle is the teacher
        }
    }

    /// Constrained config: limited resources, must be selective.
    pub fn constrained() -> Self {
        Self {
            max_classes: 50,
            max_epochs: 5,
            threshold: 0.5,  // balanced
            epsilon: 0.3,    // exploration during training
        }
    }

    /// Evaluation config: like constrained but no exploration.
    pub fn evaluation() -> Self {
        Self {
            max_classes: 50,
            max_epochs: 5,
            threshold: 0.5,
            epsilon: 0.0,  // no exploration for fair eval
        }
    }
}

/// A training sample for DualMaskGuide.
pub type Sample = ([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, bool);

/// Simple metrics for monitoring training.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub total: usize,
    pub correct: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub loss_sum: f32,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn accuracy(&self) -> f32 {
        if self.total == 0 { 1.0 } else { self.correct as f32 / self.total as f32 }
    }

    pub fn avg_loss(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.loss_sum / self.total as f32 }
    }
}

/// Training result from resource-asymmetric training.
#[derive(Clone, Debug)]
pub struct TrainingResult {
    /// Oracle's final cost (target quality).
    pub oracle_cost: i64,
    /// Guided search's initial cost (before training).
    pub initial_guided_cost: i64,
    /// Guided search's final cost (after training).
    pub final_guided_cost: i64,
    /// Oracle's pairs tried.
    pub oracle_pairs: usize,
    /// Guided search's initial pairs tried.
    pub initial_guided_pairs: usize,
    /// Guided search's final pairs tried.
    pub final_guided_pairs: usize,
}

impl TrainingResult {
    /// Did the guide learn to match oracle quality?
    pub fn quality_achieved(&self) -> bool {
        self.final_guided_cost <= self.oracle_cost
    }

    /// Resource efficiency: oracle_pairs / guided_pairs.
    /// Higher = guide is more efficient.
    pub fn efficiency_ratio(&self) -> f32 {
        if self.final_guided_pairs == 0 {
            0.0
        } else {
            self.oracle_pairs as f32 / self.final_guided_pairs as f32
        }
    }
}

/// Run a simple training test to verify the dual-mask architecture learns.
///
/// Returns (model, final_accuracy).
pub fn sanity_check(epochs: usize, lr: f32, seed: u64) -> (DualMaskGuide, f32) {
    let mut model = DualMaskGuide::new_random(seed);
    let mut rng = seed;

    // Generate synthetic data where:
    // improved = (expr_features[0] > 0) AND (rule_features[0] > 0)
    let mut samples: Vec<Sample> = Vec::new();
    for i in 0..200 {
        let mut expr_f = [0.0f32; EXPR_FEATURE_DIM];
        let mut rule_f = [0.0f32; RULE_FEATURE_DIM];

        // Add noise
        for f in &mut expr_f { *f = lcg_f32(&mut rng) * 0.3; }
        for f in &mut rule_f { *f = lcg_f32(&mut rng) * 0.3; }

        // Set signal
        let expr_high = i % 2 == 0;
        let rule_high = i % 4 < 2;
        let improved = expr_high && rule_high;

        expr_f[0] = if expr_high { 1.0 } else { -1.0 };
        rule_f[0] = if rule_high { 1.0 } else { -1.0 };

        samples.push((expr_f, rule_f, i % 10, improved));
    }

    // Train
    for _ in 0..epochs {
        model.train_batch(&samples, lr);
    }

    // Evaluate
    let (accuracy, _, _) = model.evaluate(&samples);
    (model, accuracy)
}

#[inline]
fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanity_check() {
        let (_, accuracy) = sanity_check(50, 0.05, 42);
        assert!(
            accuracy > 0.7,
            "DualMaskGuide should learn simple AND pattern, got accuracy={}",
            accuracy
        );
    }
}
