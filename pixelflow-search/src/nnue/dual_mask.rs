//! Dual-Mask Guide: Factorized action selection for e-graph search.
//!
//! # Architecture
//!
//! The action space for e-graph rewriting is `|expressions| × |rules|`, but most
//! (expr, rule) pairs are illegal (rule doesn't match). Instead of learning a full
//! matrix, we factorize into two independent masks:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     DualMaskGuide                           │
//! │                                                             │
//! │  Expression Features ──► ExprHead ──► ExprMask [E floats]   │
//! │                              │                              │
//! │                              ├─────────────────┐            │
//! │                              │                 │            │
//! │  Rule Features ──────► RuleHead ──► RuleMask [R floats]     │
//! │                                                             │
//! │                                                             │
//! │  Actions = (ExprMask ⊗ RuleMask) ∩ Legal                    │
//! │                                                             │
//! │  score(e, r) = expr_mask[e] * rule_mask[r]                  │
//! │              + interaction[e, r]  (optional sparse term)    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Benefits
//!
//! 1. **Parameter efficient**: O(E + R) instead of O(E × R)
//! 2. **Generalizes**: Expression features transfer across rules
//! 3. **Sparse evaluation**: Only score legal (matching) pairs
//! 4. **Interpretable**: Can inspect "promising locations" vs "promising rules" separately
//!
//! # Training
//!
//! Each head is trained independently:
//! - ExprHead: "Did applying ANY rule here lead to improvement?"
//! - RuleHead: "Did this rule lead to improvement when it matched?"
//!
//! The combined score is trained end-to-end on actual (expr, rule, outcome) tuples.

use alloc::vec::Vec;
use libm::sqrtf;

// ============================================================================
// Constants
// ============================================================================

/// Expression feature dimension (from e-class structural analysis).
pub const EXPR_FEATURE_DIM: usize = 16;

/// Rule feature dimension (rule characteristics + context).
pub const RULE_FEATURE_DIM: usize = 12;

/// Hidden layer dimension for both heads.
pub const HIDDEN_DIM: usize = 32;

/// Maximum number of rules we support.
pub const MAX_RULES: usize = 64;

// ============================================================================
// DualMaskGuide
// ============================================================================

/// Dual-mask guide for factorized action selection.
///
/// Produces two masks:
/// - `expr_mask`: Score for each e-class (where to focus)
/// - `rule_mask`: Score for each rule (what to try)
///
/// Final action scores are the outer product, filtered by legality.
#[derive(Clone)]
pub struct DualMaskGuide {
    // ===== Expression Head =====
    /// First layer: [EXPR_FEATURE_DIM][HIDDEN_DIM]
    pub expr_w1: [[f32; HIDDEN_DIM]; EXPR_FEATURE_DIM],
    /// First layer bias
    pub expr_b1: [f32; HIDDEN_DIM],
    /// Output layer: [HIDDEN_DIM] → scalar
    pub expr_w2: [f32; HIDDEN_DIM],
    /// Output bias
    pub expr_b2: f32,

    // ===== Rule Head =====
    /// First layer: [RULE_FEATURE_DIM][HIDDEN_DIM]
    pub rule_w1: [[f32; HIDDEN_DIM]; RULE_FEATURE_DIM],
    /// First layer bias
    pub rule_b1: [f32; HIDDEN_DIM],
    /// Output layer: [HIDDEN_DIM] → scalar
    pub rule_w2: [f32; HIDDEN_DIM],
    /// Output bias
    pub rule_b2: f32,

    // ===== Rule Embeddings (learned rule priors) =====
    /// Per-rule learned bias (how promising is this rule in general?)
    pub rule_prior: [f32; MAX_RULES],
}

impl Default for DualMaskGuide {
    fn default() -> Self {
        Self::new()
    }
}

impl DualMaskGuide {
    /// Create a zero-initialized network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expr_w1: [[0.0; HIDDEN_DIM]; EXPR_FEATURE_DIM],
            expr_b1: [0.0; HIDDEN_DIM],
            expr_w2: [0.0; HIDDEN_DIM],
            expr_b2: 0.0,

            rule_w1: [[0.0; HIDDEN_DIM]; RULE_FEATURE_DIM],
            rule_b1: [0.0; HIDDEN_DIM],
            rule_w2: [0.0; HIDDEN_DIM],
            rule_b2: 0.0,

            rule_prior: [0.0; MAX_RULES],
        }
    }

    /// Create with random initialization (He init for ReLU).
    #[must_use]
    pub fn new_random(seed: u64) -> Self {
        let mut guide = Self::new();
        guide.randomize(seed);
        guide
    }

    /// Randomize weights using He initialization.
    pub fn randomize(&mut self, seed: u64) {
        let mut state = seed;
        let mut next_random = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state as f64) / (u64::MAX as f64);
            ((u * 2.0) - 1.0) as f32
        };

        // Expression head
        let scale_expr1 = sqrtf(2.0 / EXPR_FEATURE_DIM as f32);
        let scale_expr2 = sqrtf(2.0 / HIDDEN_DIM as f32);

        for i in 0..EXPR_FEATURE_DIM {
            for j in 0..HIDDEN_DIM {
                self.expr_w1[i][j] = next_random() * scale_expr1;
            }
        }
        for j in 0..HIDDEN_DIM {
            self.expr_b1[j] = 0.0;
            self.expr_w2[j] = next_random() * scale_expr2;
        }
        self.expr_b2 = 0.0;

        // Rule head
        let scale_rule1 = sqrtf(2.0 / RULE_FEATURE_DIM as f32);
        let scale_rule2 = sqrtf(2.0 / HIDDEN_DIM as f32);

        for i in 0..RULE_FEATURE_DIM {
            for j in 0..HIDDEN_DIM {
                self.rule_w1[i][j] = next_random() * scale_rule1;
            }
        }
        for j in 0..HIDDEN_DIM {
            self.rule_b1[j] = 0.0;
            self.rule_w2[j] = next_random() * scale_rule2;
        }
        self.rule_b2 = 0.0;

        // Rule priors start at zero (no bias)
        for r in 0..MAX_RULES {
            self.rule_prior[r] = 0.0;
        }
    }

    // ========================================================================
    // Forward Pass
    // ========================================================================

    /// Compute expression score for a single e-class.
    ///
    /// Higher score = more promising location to apply rewrites.
    #[must_use]
    pub fn score_expr(&self, features: &[f32; EXPR_FEATURE_DIM]) -> f32 {
        // Hidden layer with ReLU
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.expr_b1[j];
            for i in 0..EXPR_FEATURE_DIM {
                sum += self.expr_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Output (no activation - raw score)
        let mut score = self.expr_b2;
        for j in 0..HIDDEN_DIM {
            score += self.expr_w2[j] * hidden[j];
        }
        score
    }

    /// Compute rule score given rule features and rule index.
    ///
    /// Higher score = more promising rule to try.
    #[must_use]
    pub fn score_rule(&self, features: &[f32; RULE_FEATURE_DIM], rule_idx: usize) -> f32 {
        // Hidden layer with ReLU
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.rule_b1[j];
            for i in 0..RULE_FEATURE_DIM {
                sum += self.rule_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Output + rule prior
        let mut score = self.rule_b2;
        for j in 0..HIDDEN_DIM {
            score += self.rule_w2[j] * hidden[j];
        }

        // Add learned rule prior
        if rule_idx < MAX_RULES {
            score += self.rule_prior[rule_idx];
        }

        score
    }

    /// Compute expression mask for multiple e-classes.
    ///
    /// Returns scores for each e-class (higher = more promising).
    #[must_use]
    pub fn expr_mask(&self, expr_features: &[[f32; EXPR_FEATURE_DIM]]) -> Vec<f32> {
        expr_features.iter().map(|f| self.score_expr(f)).collect()
    }

    /// Compute rule mask for all rules given current context.
    ///
    /// Returns scores for each rule (higher = more promising).
    #[must_use]
    pub fn rule_mask(&self, rule_features: &[[f32; RULE_FEATURE_DIM]]) -> Vec<f32> {
        rule_features
            .iter()
            .enumerate()
            .map(|(idx, f)| self.score_rule(f, idx))
            .collect()
    }

    /// Score all legal (expr, rule) actions.
    ///
    /// Takes:
    /// - `expr_features`: Features for each candidate expression
    /// - `rule_features`: Features for each rule
    /// - `legal_actions`: List of (expr_idx, rule_idx) pairs that are legal
    ///
    /// Returns: Scores for each legal action (same order as `legal_actions`).
    #[must_use]
    pub fn score_actions(
        &self,
        expr_features: &[[f32; EXPR_FEATURE_DIM]],
        rule_features: &[[f32; RULE_FEATURE_DIM]],
        legal_actions: &[(usize, usize)],
    ) -> Vec<f32> {
        // Compute masks once
        let expr_mask = self.expr_mask(expr_features);
        let rule_mask = self.rule_mask(rule_features);

        // Score each legal action
        legal_actions
            .iter()
            .map(|&(e, r)| {
                let expr_score = expr_mask.get(e).copied().unwrap_or(0.0);
                let rule_score = rule_mask.get(r).copied().unwrap_or(0.0);
                expr_score * rule_score
            })
            .collect()
    }

    /// Select the best action from legal options.
    ///
    /// Returns (expr_idx, rule_idx, score) for the highest-scoring action.
    #[must_use]
    pub fn best_action(
        &self,
        expr_features: &[[f32; EXPR_FEATURE_DIM]],
        rule_features: &[[f32; RULE_FEATURE_DIM]],
        legal_actions: &[(usize, usize)],
    ) -> Option<(usize, usize, f32)> {
        if legal_actions.is_empty() {
            return None;
        }

        let scores = self.score_actions(expr_features, rule_features, legal_actions);

        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(idx, &score)| {
                let (e, r) = legal_actions[idx];
                (e, r, score)
            })
    }

    /// Select top-k actions by score.
    ///
    /// Returns Vec of (expr_idx, rule_idx, score) sorted by descending score.
    #[must_use]
    pub fn top_k_actions(
        &self,
        expr_features: &[[f32; EXPR_FEATURE_DIM]],
        rule_features: &[[f32; RULE_FEATURE_DIM]],
        legal_actions: &[(usize, usize)],
        k: usize,
    ) -> Vec<(usize, usize, f32)> {
        if legal_actions.is_empty() {
            return Vec::new();
        }

        let scores = self.score_actions(expr_features, rule_features, legal_actions);

        let mut scored: Vec<_> = legal_actions
            .iter()
            .zip(scores.iter())
            .map(|(&(e, r), &s)| (e, r, s))
            .collect();

        // Sort by descending score
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(core::cmp::Ordering::Equal));

        scored.truncate(k);
        scored
    }

    // ========================================================================
    // Training
    // ========================================================================

    /// Train on a single (expr, rule, outcome) sample.
    ///
    /// - `expr_features`: Features for the expression where rule was applied
    /// - `rule_features`: Features for the rule
    /// - `rule_idx`: Index of the rule
    /// - `improved`: Did applying this rule lead to cost improvement?
    /// - `lr`: Learning rate
    ///
    /// Returns the loss for this sample.
    pub fn train_step(
        &mut self,
        expr_features: &[f32; EXPR_FEATURE_DIM],
        rule_features: &[f32; RULE_FEATURE_DIM],
        rule_idx: usize,
        improved: bool,
        lr: f32,
    ) -> f32 {
        // Forward pass with hidden activations
        let (expr_score, expr_hidden) = self.forward_expr_with_hidden(expr_features);
        let (rule_score, rule_hidden) = self.forward_rule_with_hidden(rule_features, rule_idx);

        // Combined score (product)
        let combined_score = expr_score * rule_score;

        // Target: 1.0 if improved, 0.0 otherwise
        // Use sigmoid + BCE for training stability
        let predicted = sigmoid(combined_score);
        let target = if improved { 1.0 } else { 0.0 };

        // BCE loss
        let p = predicted.clamp(1e-7, 1.0 - 1e-7);
        let loss = -(target * p.ln() + (1.0 - target) * (1.0 - p).ln());

        // Gradient of BCE w.r.t. combined_score (through sigmoid)
        let d_combined = predicted - target;

        // Gradient flows to both heads via product rule:
        // d_loss/d_expr_score = d_loss/d_combined * d_combined/d_expr_score
        //                     = d_combined * rule_score
        let d_expr_score = d_combined * rule_score;
        let d_rule_score = d_combined * expr_score;

        // Clip gradients
        let d_expr_score = d_expr_score.clamp(-1.0, 1.0);
        let d_rule_score = d_rule_score.clamp(-1.0, 1.0);

        // Backprop through expression head
        self.backprop_expr(expr_features, &expr_hidden, d_expr_score, lr);

        // Backprop through rule head
        self.backprop_rule(rule_features, &rule_hidden, rule_idx, d_rule_score, lr);

        loss
    }

    /// Forward pass for expression head, returning hidden activations.
    fn forward_expr_with_hidden(&self, features: &[f32; EXPR_FEATURE_DIM]) -> (f32, [f32; HIDDEN_DIM]) {
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.expr_b1[j];
            for i in 0..EXPR_FEATURE_DIM {
                sum += self.expr_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        let mut score = self.expr_b2;
        for j in 0..HIDDEN_DIM {
            score += self.expr_w2[j] * hidden[j];
        }

        (score, hidden)
    }

    /// Forward pass for rule head, returning hidden activations.
    fn forward_rule_with_hidden(&self, features: &[f32; RULE_FEATURE_DIM], rule_idx: usize) -> (f32, [f32; HIDDEN_DIM]) {
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.rule_b1[j];
            for i in 0..RULE_FEATURE_DIM {
                sum += self.rule_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        let mut score = self.rule_b2;
        for j in 0..HIDDEN_DIM {
            score += self.rule_w2[j] * hidden[j];
        }

        if rule_idx < MAX_RULES {
            score += self.rule_prior[rule_idx];
        }

        (score, hidden)
    }

    /// Backprop through expression head.
    fn backprop_expr(
        &mut self,
        features: &[f32; EXPR_FEATURE_DIM],
        hidden: &[f32; HIDDEN_DIM],
        d_output: f32,
        lr: f32,
    ) {
        // Output layer
        self.expr_b2 -= lr * d_output;
        for j in 0..HIDDEN_DIM {
            self.expr_w2[j] -= lr * d_output * hidden[j];
        }

        // Hidden layer (through ReLU)
        for j in 0..HIDDEN_DIM {
            let d_hidden = if hidden[j] > 0.0 { d_output * self.expr_w2[j] } else { 0.0 };
            let d_hidden = d_hidden.clamp(-1.0, 1.0);

            self.expr_b1[j] -= lr * d_hidden;
            for i in 0..EXPR_FEATURE_DIM {
                self.expr_w1[i][j] -= lr * d_hidden * features[i];
            }
        }
    }

    /// Backprop through rule head.
    fn backprop_rule(
        &mut self,
        features: &[f32; RULE_FEATURE_DIM],
        hidden: &[f32; HIDDEN_DIM],
        rule_idx: usize,
        d_output: f32,
        lr: f32,
    ) {
        // Output layer
        self.rule_b2 -= lr * d_output;
        for j in 0..HIDDEN_DIM {
            self.rule_w2[j] -= lr * d_output * hidden[j];
        }

        // Rule prior
        if rule_idx < MAX_RULES {
            self.rule_prior[rule_idx] -= lr * d_output;
        }

        // Hidden layer (through ReLU)
        for j in 0..HIDDEN_DIM {
            let d_hidden = if hidden[j] > 0.0 { d_output * self.rule_w2[j] } else { 0.0 };
            let d_hidden = d_hidden.clamp(-1.0, 1.0);

            self.rule_b1[j] -= lr * d_hidden;
            for i in 0..RULE_FEATURE_DIM {
                self.rule_w1[i][j] -= lr * d_hidden * features[i];
            }
        }
    }

    /// Train on a batch of samples.
    ///
    /// Each sample is (expr_features, rule_features, rule_idx, improved).
    /// Returns average loss.
    pub fn train_batch(
        &mut self,
        samples: &[([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, bool)],
        lr: f32,
    ) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for (expr_f, rule_f, rule_idx, improved) in samples {
            total_loss += self.train_step(expr_f, rule_f, *rule_idx, *improved, lr);
        }

        total_loss / samples.len() as f32
    }

    /// Evaluate accuracy on a set of samples.
    ///
    /// Returns (accuracy, false_positive_rate, false_negative_rate).
    pub fn evaluate(
        &self,
        samples: &[([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, bool)],
    ) -> (f32, f32, f32) {
        if samples.is_empty() {
            return (1.0, 0.0, 0.0);
        }

        let mut correct = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        let mut total_positives = 0;
        let mut total_negatives = 0;

        for (expr_f, rule_f, rule_idx, improved) in samples {
            let score = self.score_expr(expr_f) * self.score_rule(rule_f, *rule_idx);
            let predicted = sigmoid(score) > 0.5;

            if *improved {
                total_positives += 1;
                if predicted {
                    correct += 1;
                } else {
                    false_negatives += 1;
                }
            } else {
                total_negatives += 1;
                if predicted {
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

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Total number of parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        // Expression head
        let expr_params = EXPR_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1;
        // Rule head
        let rule_params = RULE_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1;
        // Rule priors
        let prior_params = MAX_RULES;

        expr_params + rule_params + prior_params
    }

    /// Serialize to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.param_count() * 4);

        // Expression head
        for row in &self.expr_w1 {
            for &v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &v in &self.expr_b1 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.expr_w2 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&self.expr_b2.to_le_bytes());

        // Rule head
        for row in &self.rule_w1 {
            for &v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &v in &self.rule_b1 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.rule_w2 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&self.rule_b2.to_le_bytes());

        // Rule priors
        for &v in &self.rule_prior {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        bytes
    }

    /// Deserialize from bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let expected = (EXPR_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1
            + RULE_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1
            + MAX_RULES) * 4;

        if bytes.len() != expected {
            return None;
        }

        let mut guide = Self::new();
        let mut offset = 0;

        let mut read_f32 = || {
            let v = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            v
        };

        // Expression head
        for row in &mut guide.expr_w1 {
            for v in row.iter_mut() {
                *v = read_f32();
            }
        }
        for v in &mut guide.expr_b1 {
            *v = read_f32();
        }
        for v in &mut guide.expr_w2 {
            *v = read_f32();
        }
        guide.expr_b2 = read_f32();

        // Rule head
        for row in &mut guide.rule_w1 {
            for v in row.iter_mut() {
                *v = read_f32();
            }
        }
        for v in &mut guide.rule_b1 {
            *v = read_f32();
        }
        for v in &mut guide.rule_w2 {
            *v = read_f32();
        }
        guide.rule_b2 = read_f32();

        // Rule priors
        for v in &mut guide.rule_prior {
            *v = read_f32();
        }

        Some(guide)
    }

    /// Save to file.
    #[cfg(feature = "std")]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    /// Load from file.
    #[cfg(feature = "std")]
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid dual-mask guide file")
        })
    }
}

// ============================================================================
// Utilities
// ============================================================================

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

// ============================================================================
// Feature Extraction Helpers
// ============================================================================

/// Expression features for an e-class.
///
/// These capture "is this location promising for rewriting?"
#[derive(Clone, Copy, Debug, Default)]
pub struct ExprFeatures {
    /// Number of nodes in this e-class
    pub class_size: f32,
    /// Depth of the expression rooted here
    pub depth: f32,
    /// Number of children (fan-out)
    pub num_children: f32,
    /// Whether this is a leaf (var/const)
    pub is_leaf: f32,
    /// Whether this contains expensive ops (div, sqrt, etc.)
    pub has_expensive_op: f32,
    /// Whether this is on the critical path
    pub on_critical_path: f32,
    /// Number of uses (fan-in from other e-classes)
    pub num_uses: f32,
    /// Normalized depth in tree (0=root, 1=deepest)
    pub relative_depth: f32,
    /// Op type one-hot encoded (8 categories)
    pub op_category: [f32; 8],
}

impl ExprFeatures {
    /// Convert to array for neural network input.
    #[must_use]
    pub fn to_array(&self) -> [f32; EXPR_FEATURE_DIM] {
        [
            self.class_size,
            self.depth,
            self.num_children,
            self.is_leaf,
            self.has_expensive_op,
            self.on_critical_path,
            self.num_uses,
            self.relative_depth,
            self.op_category[0],
            self.op_category[1],
            self.op_category[2],
            self.op_category[3],
            self.op_category[4],
            self.op_category[5],
            self.op_category[6],
            self.op_category[7],
        ]
    }
}

/// Rule features for a rewrite rule.
///
/// These capture "is this rule likely to help?"
#[derive(Clone, Copy, Debug, Default)]
pub struct RuleFeatures {
    /// Historical match rate (how often does this rule apply?)
    pub match_rate: f32,
    /// Historical improvement rate (when it matches, how often does it help?)
    pub improvement_rate: f32,
    /// Estimated size change (negative = smaller)
    pub size_delta: f32,
    /// Estimated depth change
    pub depth_delta: f32,
    /// Does this rule introduce expensive ops?
    pub adds_expensive: f32,
    /// Does this rule remove expensive ops?
    pub removes_expensive: f32,
    /// Is this a simplification rule (algebraic identity)?
    pub is_simplification: f32,
    /// Is this a canonicalization rule?
    pub is_canonicalization: f32,
    /// Is this a strength reduction rule?
    pub is_strength_reduction: f32,
    /// Number of pattern nodes in LHS
    pub lhs_size: f32,
    /// Number of pattern nodes in RHS
    pub rhs_size: f32,
    /// Bidirectional? (can apply in reverse)
    pub is_bidirectional: f32,
}

impl RuleFeatures {
    /// Convert to array for neural network input.
    #[must_use]
    pub fn to_array(&self) -> [f32; RULE_FEATURE_DIM] {
        [
            self.match_rate,
            self.improvement_rate,
            self.size_delta,
            self.depth_delta,
            self.adds_expensive,
            self.removes_expensive,
            self.is_simplification,
            self.is_canonicalization,
            self.is_strength_reduction,
            self.lhs_size,
            self.rhs_size,
            self.is_bidirectional,
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_mask_basic() {
        let guide = DualMaskGuide::new_random(42);

        let expr_features = [0.5f32; EXPR_FEATURE_DIM];
        let rule_features = [0.5f32; RULE_FEATURE_DIM];

        let expr_score = guide.score_expr(&expr_features);
        let rule_score = guide.score_rule(&rule_features, 0);

        assert!(expr_score.is_finite());
        assert!(rule_score.is_finite());
    }

    #[test]
    fn test_action_scoring() {
        let guide = DualMaskGuide::new_random(42);

        let expr_features = vec![
            [0.5f32; EXPR_FEATURE_DIM],
            [0.8f32; EXPR_FEATURE_DIM],
            [0.2f32; EXPR_FEATURE_DIM],
        ];
        let rule_features = vec![
            [0.5f32; RULE_FEATURE_DIM],
            [0.9f32; RULE_FEATURE_DIM],
        ];
        let legal_actions = vec![(0, 0), (0, 1), (1, 0), (2, 1)];

        let scores = guide.score_actions(&expr_features, &rule_features, &legal_actions);
        assert_eq!(scores.len(), legal_actions.len());

        let best = guide.best_action(&expr_features, &rule_features, &legal_actions);
        assert!(best.is_some());
    }

    #[test]
    fn test_training() {
        let mut guide = DualMaskGuide::new_random(42);

        let expr_features = [0.5f32; EXPR_FEATURE_DIM];
        let rule_features = [0.5f32; RULE_FEATURE_DIM];

        // Train a few steps
        for _ in 0..10 {
            let loss = guide.train_step(&expr_features, &rule_features, 0, true, 0.01);
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn test_serialization() {
        let guide = DualMaskGuide::new_random(123);
        let bytes = guide.to_bytes();
        let loaded = DualMaskGuide::from_bytes(&bytes).expect("should load");

        // Check some weights match
        assert!((guide.expr_w1[0][0] - loaded.expr_w1[0][0]).abs() < 1e-6);
        assert!((guide.rule_b2 - loaded.rule_b2).abs() < 1e-6);
    }

    #[test]
    fn test_top_k() {
        let guide = DualMaskGuide::new_random(42);

        let expr_features = vec![[0.5f32; EXPR_FEATURE_DIM]; 10];
        let rule_features = vec![[0.5f32; RULE_FEATURE_DIM]; 5];

        // All combinations are legal
        let legal_actions: Vec<_> = (0..10).flat_map(|e| (0..5).map(move |r| (e, r))).collect();

        let top_5 = guide.top_k_actions(&expr_features, &rule_features, &legal_actions, 5);
        assert_eq!(top_5.len(), 5);

        // Should be sorted descending
        for i in 1..top_5.len() {
            assert!(top_5[i - 1].2 >= top_5[i].2);
        }
    }
}
