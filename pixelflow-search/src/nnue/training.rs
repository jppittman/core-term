//! Training utilities for DualMaskGuide.
//!
//! Simple helpers for Self-Imitation Learning (SIL) with the dual-mask architecture.
//! The actual training loop lives in the search code - this just provides utilities.

use super::dual_mask::{DualMaskGuide, EXPR_FEATURE_DIM, RULE_FEATURE_DIM};

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
