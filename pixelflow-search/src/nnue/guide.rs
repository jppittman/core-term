//! GuideNnue: Match probability predictor for guided search.
//!
//! This module implements a simple neural network that predicts whether
//! a rewrite rule will match given the current e-graph state.
//!
//! # Architecture
//!
//! Unlike the tree-based ExprNnue, the Guide uses flat rule features:
//!
//! ```text
//! RuleFeatures (8 floats)
//!        │
//!        ▼
//! ┌─────────────────────┐
//! │  Hidden Layer       │
//! │  (8 → 32, ReLU)     │
//! └──────────┬──────────┘
//!            │
//!            ▼
//! ┌─────────────────────┐
//! │  Output Layer       │
//! │  (32 → 1, sigmoid)  │
//! └─────────────────────┘
//!            │
//!            ▼
//!    P(rule matches)
//! ```
//!
//! # Training
//!
//! Binary classification with asymmetric loss:
//! - False positive (predict match, doesn't): Heavy penalty (wastes compute)
//! - False negative (predict no match, does): Light penalty (conservative)

use libm::{expf, sqrtf};

// ============================================================================
// Constants
// ============================================================================

/// Input dimension (RuleFeatures array size).
pub const RULE_FEATURE_COUNT: usize = 8;

/// Hidden layer dimension.
pub const GUIDE_HIDDEN_DIM: usize = 32;

/// Penalty weight for false positives (predicting match when it doesn't).
/// Higher = more conservative Guide (fewer false alarms).
pub const FALSE_POSITIVE_WEIGHT: f32 = 2.0;

/// Penalty weight for false negatives (predicting no match when it does).
/// Lower than FP = Guide is okay missing some matches.
pub const FALSE_NEGATIVE_WEIGHT: f32 = 0.5;

/// Gradient clipping threshold.
pub const GRAD_CLIP: f32 = 1.0;

// ============================================================================
// GuideNnue
// ============================================================================

/// Simple feedforward network for match probability prediction.
///
/// Takes rule features and outputs P(rule will match) in [0, 1].
#[derive(Clone)]
pub struct GuideNnue {
    /// Hidden layer weights: [RULE_FEATURE_COUNT][GUIDE_HIDDEN_DIM]
    pub w1: [[f32; GUIDE_HIDDEN_DIM]; RULE_FEATURE_COUNT],

    /// Hidden layer biases: [GUIDE_HIDDEN_DIM]
    pub b1: [f32; GUIDE_HIDDEN_DIM],

    /// Output layer weights: [GUIDE_HIDDEN_DIM]
    pub w2: [f32; GUIDE_HIDDEN_DIM],

    /// Output layer bias
    pub b2: f32,
}

impl Default for GuideNnue {
    fn default() -> Self {
        Self::new()
    }
}

impl GuideNnue {
    /// Create a zero-initialized network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            w1: [[0.0; GUIDE_HIDDEN_DIM]; RULE_FEATURE_COUNT],
            b1: [0.0; GUIDE_HIDDEN_DIM],
            w2: [0.0; GUIDE_HIDDEN_DIM],
            b2: 0.0,
        }
    }

    /// Create a randomly initialized network using He initialization.
    #[must_use]
    pub fn new_random(seed: u64) -> Self {
        let mut net = Self::new();
        net.randomize(seed);
        net
    }

    /// Randomize weights using He initialization.
    pub fn randomize(&mut self, seed: u64) {
        let mut state = seed;
        let mut next_random = || -> f32 {
            // Simple LCG
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to float in [-1, 1]
            let u = (state as f64) / (u64::MAX as f64);
            ((u * 2.0) - 1.0) as f32
        };

        // He initialization scale for ReLU: sqrt(2/fan_in)
        let scale1 = sqrtf(2.0 / RULE_FEATURE_COUNT as f32);
        let scale2 = sqrtf(2.0 / GUIDE_HIDDEN_DIM as f32);

        for i in 0..RULE_FEATURE_COUNT {
            for j in 0..GUIDE_HIDDEN_DIM {
                self.w1[i][j] = next_random() * scale1;
            }
        }

        for j in 0..GUIDE_HIDDEN_DIM {
            self.b1[j] = 0.0; // Biases start at zero
            self.w2[j] = next_random() * scale2;
        }

        self.b2 = 0.0;
    }

    /// Predict match probability for given rule features.
    ///
    /// Returns P(match) in [0, 1].
    #[must_use]
    pub fn predict(&self, features: &[f32; RULE_FEATURE_COUNT]) -> f32 {
        // Hidden layer: h = ReLU(W1 * x + b1)
        let mut hidden = [0.0f32; GUIDE_HIDDEN_DIM];
        for j in 0..GUIDE_HIDDEN_DIM {
            let mut sum = self.b1[j];
            for i in 0..RULE_FEATURE_COUNT {
                sum += self.w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // Output layer: y = sigmoid(W2 * h + b2)
        let mut logit = self.b2;
        for j in 0..GUIDE_HIDDEN_DIM {
            logit += self.w2[j] * hidden[j];
        }

        sigmoid(logit)
    }

    /// Predict and return hidden activations (for gradient computation).
    fn forward_with_hidden(&self, features: &[f32; RULE_FEATURE_COUNT]) -> (f32, [f32; GUIDE_HIDDEN_DIM]) {
        // Hidden layer
        let mut hidden = [0.0f32; GUIDE_HIDDEN_DIM];
        for j in 0..GUIDE_HIDDEN_DIM {
            let mut sum = self.b1[j];
            for i in 0..RULE_FEATURE_COUNT {
                sum += self.w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // Output layer
        let mut logit = self.b2;
        for j in 0..GUIDE_HIDDEN_DIM {
            logit += self.w2[j] * hidden[j];
        }

        (sigmoid(logit), hidden)
    }

    /// Compute binary cross-entropy loss with asymmetric weights.
    pub fn loss(&self, predicted: f32, matched: bool) -> f32 {
        let label = if matched { 1.0 } else { 0.0 };

        // Clamp prediction to avoid log(0)
        let p = predicted.clamp(1e-7, 1.0 - 1e-7);

        // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        let bce = -(label * p.ln() + (1.0 - label) * (1.0 - p).ln());

        // Asymmetric weighting
        if predicted > 0.5 && !matched {
            bce * FALSE_POSITIVE_WEIGHT // Said yes, was wrong
        } else if predicted <= 0.5 && matched {
            bce * FALSE_NEGATIVE_WEIGHT // Said no, was wrong
        } else {
            bce // Correct prediction
        }
    }

    /// Update weights with a single training sample.
    ///
    /// Uses stochastic gradient descent with the asymmetric BCE loss.
    pub fn train_step(
        &mut self,
        features: &[f32; RULE_FEATURE_COUNT],
        matched: bool,
        lr: f32,
    ) -> f32 {
        // Forward pass
        let (predicted, hidden) = self.forward_with_hidden(features);

        // Compute loss for monitoring
        let loss = self.loss(predicted, matched);

        // Compute gradients via backprop
        let label = if matched { 1.0 } else { 0.0 };

        // Asymmetric weight based on error type
        let weight = if predicted > 0.5 && !matched {
            FALSE_POSITIVE_WEIGHT
        } else if predicted <= 0.5 && matched {
            FALSE_NEGATIVE_WEIGHT
        } else {
            1.0
        };

        // d(BCE)/d(logit) = predicted - label (for sigmoid output)
        // With asymmetric weighting
        let d_logit = weight * (predicted - label);

        // Gradient clipping
        let d_logit = d_logit.clamp(-GRAD_CLIP, GRAD_CLIP);

        // Output layer gradients
        self.b2 -= lr * d_logit;
        for j in 0..GUIDE_HIDDEN_DIM {
            self.w2[j] -= lr * d_logit * hidden[j];
        }

        // Hidden layer gradients
        for j in 0..GUIDE_HIDDEN_DIM {
            // Gradient through ReLU
            let d_hidden = if hidden[j] > 0.0 { d_logit * self.w2[j] } else { 0.0 };
            let d_hidden = d_hidden.clamp(-GRAD_CLIP, GRAD_CLIP);

            self.b1[j] -= lr * d_hidden;
            for i in 0..RULE_FEATURE_COUNT {
                self.w1[i][j] -= lr * d_hidden * features[i];
            }
        }

        loss
    }

    /// Train on a batch of samples.
    ///
    /// Returns average loss.
    pub fn train_batch(
        &mut self,
        samples: &[([f32; RULE_FEATURE_COUNT], bool)],
        lr: f32,
    ) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for (features, matched) in samples {
            total_loss += self.train_step(features, *matched, lr);
        }

        total_loss / samples.len() as f32
    }

    /// Evaluate accuracy on a set of samples.
    ///
    /// Returns (accuracy, false_positive_rate, false_negative_rate).
    pub fn evaluate(&self, samples: &[([f32; RULE_FEATURE_COUNT], bool)]) -> (f32, f32, f32) {
        if samples.is_empty() {
            return (1.0, 0.0, 0.0);
        }

        let mut correct = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        let mut total_positives = 0;
        let mut total_negatives = 0;

        for (features, matched) in samples {
            let predicted = self.predict(features);
            let pred_positive = predicted > 0.5;

            if *matched {
                total_positives += 1;
                if pred_positive {
                    correct += 1;
                } else {
                    false_negatives += 1;
                }
            } else {
                total_negatives += 1;
                if pred_positive {
                    false_positives += 1;
                } else {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f32 / samples.len() as f32;
        let fp_rate = if total_negatives > 0 {
            false_positives as f32 / total_negatives as f32
        } else {
            0.0
        };
        let fn_rate = if total_positives > 0 {
            false_negatives as f32 / total_positives as f32
        } else {
            0.0
        };

        (accuracy, fp_rate, fn_rate)
    }

    /// Save network weights to a byte vector.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // w1
        for row in &self.w1 {
            for val in row {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }

        // b1
        for val in &self.b1 {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // w2
        for val in &self.w2 {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // b2
        bytes.extend_from_slice(&self.b2.to_le_bytes());

        bytes
    }

    /// Load network weights from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let expected_size = (RULE_FEATURE_COUNT * GUIDE_HIDDEN_DIM + GUIDE_HIDDEN_DIM + GUIDE_HIDDEN_DIM + 1) * 4;
        if bytes.len() != expected_size {
            return None;
        }

        let mut net = Self::new();
        let mut offset = 0;

        // w1
        for row in &mut net.w1 {
            for val in row.iter_mut() {
                *val = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                offset += 4;
            }
        }

        // b1
        for val in &mut net.b1 {
            *val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
        }

        // w2
        for val in &mut net.w2 {
            *val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
        }

        // b2
        net.b2 = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);

        Some(net)
    }

    /// Save to a file.
    #[cfg(feature = "std")]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    /// Load from a file.
    #[cfg(feature = "std")]
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid guide model file")
        })
    }

    /// Total number of parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        RULE_FEATURE_COUNT * GUIDE_HIDDEN_DIM + // w1
        GUIDE_HIDDEN_DIM + // b1
        GUIDE_HIDDEN_DIM + // w2
        1 // b2
    }
}

// ============================================================================
// Utilities
// ============================================================================

/// Sigmoid activation function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf(-x))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guide_basic() {
        let guide = GuideNnue::new_random(42);

        // Random features
        let features = [0.5, 10.0, 100.0, 0.3, 5.0, 3.0, 50.0, 0.7];

        let p = guide.predict(&features);
        assert!(p >= 0.0 && p <= 1.0, "Prediction should be in [0, 1]");
    }

    #[test]
    fn test_guide_training() {
        let mut guide = GuideNnue::new_random(42);

        // Create simple training data
        // Features: [rule_idx, egraph_classes, egraph_nodes, match_rate, ...]
        let samples: Vec<([f32; RULE_FEATURE_COUNT], bool)> = vec![
            ([0.0, 10.0, 50.0, 0.8, 0.0, 0.0, 10.0, 1.0], true),   // High match rate → match
            ([1.0, 10.0, 50.0, 0.1, 10.0, 5.0, 10.0, 0.5], false), // Low match rate → no match
            ([2.0, 10.0, 50.0, 0.9, 0.0, 0.0, 10.0, 0.9], true),
            ([3.0, 10.0, 50.0, 0.05, 20.0, 8.0, 10.0, 0.3], false),
        ];

        // Train for a few epochs
        for _ in 0..100 {
            guide.train_batch(&samples, 0.1);
        }

        // Evaluate
        let (accuracy, _fp_rate, _fn_rate) = guide.evaluate(&samples);
        assert!(accuracy > 0.5, "Should learn simple patterns");
    }

    #[test]
    fn test_guide_serialization() {
        let guide = GuideNnue::new_random(123);
        let bytes = guide.to_bytes();
        let loaded = GuideNnue::from_bytes(&bytes).expect("Should load");

        // Compare weights
        for i in 0..RULE_FEATURE_COUNT {
            for j in 0..GUIDE_HIDDEN_DIM {
                assert!((guide.w1[i][j] - loaded.w1[i][j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
