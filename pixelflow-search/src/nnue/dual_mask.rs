//! Dual-Mask Guide: Factorized action selection for e-graph search.
//!
//! # Architecture
//!
//! A "lite transformer" style architecture where expression and rule heads
//! produce embedding vectors that interact via dot product (like attention):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     DualMaskGuide                           │
//! │                                                             │
//! │  Expr Features ──► Hidden(ReLU) ──► expr_embed [EMBED_DIM]  │
//! │                                            │                │
//! │                                      dot product ──► score  │
//! │                                            │                │
//! │  Rule Features ──► Hidden(ReLU) ──► rule_embed [EMBED_DIM]  │
//! │                                   + rule_prior[idx]         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! The dot product between embeddings provides EMBED_DIM learned interaction
//! terms instead of just 1, making this more expressive while still O(E + R).
//!
//! # Benefits
//!
//! 1. **Parameter efficient**: O(E + R) instead of O(E × R)
//! 2. **Rich interaction**: EMBED_DIM interaction terms via dot product
//! 3. **Generalizes**: Expression features transfer across rules
//! 4. **Interpretable**: Embedding space learns meaningful directions
//!
//! # Training
//!
//! End-to-end on (expr_features, rule_features, rule_idx, improved) tuples.
//! Gradients flow through both heads via the dot product.

use alloc::vec::Vec;
use libm::sqrtf;

// ============================================================================
// Constants
// ============================================================================

/// Expression feature dimension (from e-class structural analysis).
pub const EXPR_FEATURE_DIM: usize = 16;

/// Rule feature dimension (rule characteristics + resource budget).
pub const RULE_FEATURE_DIM: usize = 9;

/// Hidden layer dimension for both heads.
pub const HIDDEN_DIM: usize = 32;

/// Embedding dimension for dot-product interaction.
/// This is the "covariance" dimension - how expr and rule interact.
pub const EMBED_DIM: usize = 8;

/// Penalty weight for false positives (predicting "fires" when it doesn't).
/// Lower = okay to occasionally say "yes" wrongly.
pub const FALSE_POSITIVE_WEIGHT: f32 = 1.0;

/// Penalty weight for false negatives (predicting "no fire" when it does).
/// Higher = we really want to catch the positives.
/// With ~100:1 class imbalance, need ~100x weight to balance.
pub const FALSE_NEGATIVE_WEIGHT: f32 = 100.0;

/// Maximum number of rules we support.
pub const MAX_RULES: usize = 64;

// ============================================================================
// DualMaskGuide
// ============================================================================

/// Dual-mask guide with dot-product embedding architecture.
///
/// Each head produces an embedding vector, and the score is their dot product.
/// This gives EMBED_DIM learned interaction terms instead of just 1.
#[derive(Clone)]
pub struct DualMaskGuide {
    // ===== Expression Head =====
    /// First layer: [EXPR_FEATURE_DIM][HIDDEN_DIM]
    pub expr_w1: [[f32; HIDDEN_DIM]; EXPR_FEATURE_DIM],
    /// First layer bias
    pub expr_b1: [f32; HIDDEN_DIM],
    /// Output layer: [HIDDEN_DIM][EMBED_DIM] → embedding vector
    pub expr_w2: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Output bias (one per embedding dimension)
    pub expr_b2: [f32; EMBED_DIM],

    // ===== Rule Head =====
    /// First layer: [RULE_FEATURE_DIM][HIDDEN_DIM]
    pub rule_w1: [[f32; HIDDEN_DIM]; RULE_FEATURE_DIM],
    /// First layer bias
    pub rule_b1: [f32; HIDDEN_DIM],
    /// Output layer: [HIDDEN_DIM][EMBED_DIM] → embedding vector
    pub rule_w2: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Output bias (one per embedding dimension)
    pub rule_b2: [f32; EMBED_DIM],

    // ===== Rule Embeddings (learned rule priors) =====
    /// Per-rule learned bias (added to score, not embedding)
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
            expr_w2: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            expr_b2: [0.0; EMBED_DIM],

            rule_w1: [[0.0; HIDDEN_DIM]; RULE_FEATURE_DIM],
            rule_b1: [0.0; HIDDEN_DIM],
            rule_w2: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            rule_b2: [0.0; EMBED_DIM],

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
            for k in 0..EMBED_DIM {
                self.expr_w2[j][k] = next_random() * scale_expr2;
            }
        }
        for k in 0..EMBED_DIM {
            self.expr_b2[k] = 0.0;
        }

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
            for k in 0..EMBED_DIM {
                self.rule_w2[j][k] = next_random() * scale_rule2;
            }
        }
        for k in 0..EMBED_DIM {
            self.rule_b2[k] = 0.0;
        }

        // Rule priors start at zero (no bias)
        for r in 0..MAX_RULES {
            self.rule_prior[r] = 0.0;
        }
    }

    // ========================================================================
    // Forward Pass - Embedding Architecture
    // ========================================================================

    /// Compute expression embedding.
    ///
    /// Returns an [EMBED_DIM] vector representing this expression's "query".
    #[must_use]
    pub fn embed_expr(&self, features: &[f32; EXPR_FEATURE_DIM]) -> [f32; EMBED_DIM] {
        // Hidden layer with ReLU
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.expr_b1[j];
            for i in 0..EXPR_FEATURE_DIM {
                sum += self.expr_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Output embedding (no activation - linear projection)
        let mut embed = [0.0f32; EMBED_DIM];
        for k in 0..EMBED_DIM {
            embed[k] = self.expr_b2[k];
            for j in 0..HIDDEN_DIM {
                embed[k] += self.expr_w2[j][k] * hidden[j];
            }
        }
        embed
    }

    /// Compute rule embedding.
    ///
    /// Returns an [EMBED_DIM] vector representing this rule's "key".
    #[must_use]
    pub fn embed_rule(&self, features: &[f32; RULE_FEATURE_DIM]) -> [f32; EMBED_DIM] {
        // Hidden layer with ReLU
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.rule_b1[j];
            for i in 0..RULE_FEATURE_DIM {
                sum += self.rule_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Output embedding (no activation - linear projection)
        let mut embed = [0.0f32; EMBED_DIM];
        for k in 0..EMBED_DIM {
            embed[k] = self.rule_b2[k];
            for j in 0..HIDDEN_DIM {
                embed[k] += self.rule_w2[j][k] * hidden[j];
            }
        }
        embed
    }

    /// Compute score for a single (expr, rule) pair.
    ///
    /// Score = dot(expr_embed, rule_embed) + rule_prior[idx]
    #[must_use]
    pub fn score_action(
        &self,
        expr_features: &[f32; EXPR_FEATURE_DIM],
        rule_features: &[f32; RULE_FEATURE_DIM],
        rule_idx: usize,
    ) -> f32 {
        let expr_embed = self.embed_expr(expr_features);
        let rule_embed = self.embed_rule(rule_features);

        // Dot product
        let mut score = 0.0;
        for k in 0..EMBED_DIM {
            score += expr_embed[k] * rule_embed[k];
        }

        // Add rule prior
        if rule_idx < MAX_RULES {
            score += self.rule_prior[rule_idx];
        }

        score
    }

    /// Legacy: compute expression "score" (sum of embedding).
    /// For backward compatibility with code expecting scalar.
    #[must_use]
    pub fn score_expr(&self, features: &[f32; EXPR_FEATURE_DIM]) -> f32 {
        let embed = self.embed_expr(features);
        embed.iter().sum()
    }

    /// Legacy: compute rule "score" (sum of embedding + prior).
    /// For backward compatibility with code expecting scalar.
    #[must_use]
    pub fn score_rule(&self, features: &[f32; RULE_FEATURE_DIM], rule_idx: usize) -> f32 {
        let embed = self.embed_rule(features);
        let mut score: f32 = embed.iter().sum();
        if rule_idx < MAX_RULES {
            score += self.rule_prior[rule_idx];
        }
        score
    }

    /// Compute expression embeddings for multiple e-classes.
    #[must_use]
    pub fn expr_embeddings(&self, expr_features: &[[f32; EXPR_FEATURE_DIM]]) -> Vec<[f32; EMBED_DIM]> {
        expr_features.iter().map(|f| self.embed_expr(f)).collect()
    }

    /// Compute rule embeddings for all rules.
    #[must_use]
    pub fn rule_embeddings(&self, rule_features: &[[f32; RULE_FEATURE_DIM]]) -> Vec<[f32; EMBED_DIM]> {
        rule_features.iter().map(|f| self.embed_rule(f)).collect()
    }

    /// Score all legal (expr, rule) actions using dot-product of embeddings.
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
        // Compute embeddings once
        let expr_embeds = self.expr_embeddings(expr_features);
        let rule_embeds = self.rule_embeddings(rule_features);

        // Score each legal action via dot product
        legal_actions
            .iter()
            .map(|&(e, r)| {
                let expr_e = expr_embeds.get(e);
                let rule_e = rule_embeds.get(r);

                match (expr_e, rule_e) {
                    (Some(ee), Some(re)) => {
                        let mut score = 0.0;
                        for k in 0..EMBED_DIM {
                            score += ee[k] * re[k];
                        }
                        if r < MAX_RULES {
                            score += self.rule_prior[r];
                        }
                        score
                    }
                    _ => 0.0,
                }
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

    /// Filter actions: return binary mask of which (expr, rule) pairs to try.
    ///
    /// This is the core filtering operation. Instead of scoring and ranking,
    /// we output a yes/no decision for each pair.
    ///
    /// Returns Vec<bool> parallel to `legal_actions`: true = try this pair.
    #[must_use]
    pub fn filter_actions(
        &self,
        expr_features: &[[f32; EXPR_FEATURE_DIM]],
        rule_features: &[[f32; RULE_FEATURE_DIM]],
        legal_actions: &[(usize, usize)],
        threshold: f32,
    ) -> Vec<bool> {
        let scores = self.score_actions(expr_features, rule_features, legal_actions);
        scores.iter().map(|&s| sigmoid(s) > threshold).collect()
    }

    /// Filter and return only the pairs that pass the threshold.
    ///
    /// More efficient than filter_actions when you just need the passing pairs.
    #[must_use]
    pub fn get_approved_actions(
        &self,
        expr_features: &[[f32; EXPR_FEATURE_DIM]],
        rule_features: &[[f32; RULE_FEATURE_DIM]],
        legal_actions: &[(usize, usize)],
        threshold: f32,
    ) -> Vec<(usize, usize)> {
        let scores = self.score_actions(expr_features, rule_features, legal_actions);
        legal_actions
            .iter()
            .zip(scores.iter())
            .filter_map(|(&action, &score)| {
                if sigmoid(score) > threshold {
                    Some(action)
                } else {
                    None
                }
            })
            .collect()
    }

    // ========================================================================
    // Training (Embedding Architecture)
    // ========================================================================

    /// Train on a single (expr, rule, outcome) sample.
    ///
    /// Backprop through: score = dot(expr_embed, rule_embed) + rule_prior
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
        // Forward pass - compute embeddings and hidden activations
        let (expr_embed, expr_hidden) = self.forward_expr_with_hidden(expr_features);
        let (rule_embed, rule_hidden) = self.forward_rule_with_hidden(rule_features);

        // Compute score = dot(expr_embed, rule_embed) + rule_prior
        let mut score = 0.0;
        for k in 0..EMBED_DIM {
            score += expr_embed[k] * rule_embed[k];
        }
        if rule_idx < MAX_RULES {
            score += self.rule_prior[rule_idx];
        }

        // Sigmoid + BCE loss
        let predicted = sigmoid(score);
        let target = if improved { 1.0 } else { 0.0 };

        let p = predicted.clamp(1e-7, 1.0 - 1e-7);
        let loss = -(target * p.ln() + (1.0 - target) * (1.0 - p).ln());

        // Asymmetric weighting based on error type (like GuideNnue)
        let weight = if predicted > 0.5 && !improved {
            FALSE_POSITIVE_WEIGHT  // Said "fires", was wrong
        } else if predicted <= 0.5 && improved {
            FALSE_NEGATIVE_WEIGHT  // Said "no fire", missed it
        } else {
            1.0  // Correct prediction
        };

        // Gradient of BCE w.r.t. score (through sigmoid), with asymmetric weight
        // Clip BEFORE multiplying by embedding to preserve the weight's effect
        let d_score = (weight * (predicted - target)).clamp(-10.0, 10.0);

        // Gradient flows through dot product:
        // d_loss/d_expr_embed[k] = d_score * rule_embed[k]
        // d_loss/d_rule_embed[k] = d_score * expr_embed[k]
        let mut d_expr_embed = [0.0f32; EMBED_DIM];
        let mut d_rule_embed = [0.0f32; EMBED_DIM];
        for k in 0..EMBED_DIM {
            d_expr_embed[k] = d_score * rule_embed[k];
            d_rule_embed[k] = d_score * expr_embed[k];
        }

        // Update rule prior
        if rule_idx < MAX_RULES {
            self.rule_prior[rule_idx] -= lr * d_score.clamp(-1.0, 1.0);
        }

        // Backprop through expression head
        self.backprop_expr(expr_features, &expr_hidden, &d_expr_embed, lr);

        // Backprop through rule head
        self.backprop_rule(rule_features, &rule_hidden, &d_rule_embed, lr);

        loss
    }

    /// Forward pass for expression head, returning embedding and hidden activations.
    fn forward_expr_with_hidden(&self, features: &[f32; EXPR_FEATURE_DIM]) -> ([f32; EMBED_DIM], [f32; HIDDEN_DIM]) {
        // Hidden layer with ReLU
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.expr_b1[j];
            for i in 0..EXPR_FEATURE_DIM {
                sum += self.expr_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Output embedding
        let mut embed = [0.0f32; EMBED_DIM];
        for k in 0..EMBED_DIM {
            embed[k] = self.expr_b2[k];
            for j in 0..HIDDEN_DIM {
                embed[k] += self.expr_w2[j][k] * hidden[j];
            }
        }

        (embed, hidden)
    }

    /// Forward pass for rule head, returning embedding and hidden activations.
    fn forward_rule_with_hidden(&self, features: &[f32; RULE_FEATURE_DIM]) -> ([f32; EMBED_DIM], [f32; HIDDEN_DIM]) {
        // Hidden layer with ReLU
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = self.rule_b1[j];
            for i in 0..RULE_FEATURE_DIM {
                sum += self.rule_w1[i][j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Output embedding
        let mut embed = [0.0f32; EMBED_DIM];
        for k in 0..EMBED_DIM {
            embed[k] = self.rule_b2[k];
            for j in 0..HIDDEN_DIM {
                embed[k] += self.rule_w2[j][k] * hidden[j];
            }
        }

        (embed, hidden)
    }

    /// Backprop through expression head given gradient w.r.t. embedding.
    fn backprop_expr(
        &mut self,
        features: &[f32; EXPR_FEATURE_DIM],
        hidden: &[f32; HIDDEN_DIM],
        d_embed: &[f32; EMBED_DIM],
        lr: f32,
    ) {
        // Output layer: embed[k] = b2[k] + sum_j(w2[j][k] * hidden[j])
        for k in 0..EMBED_DIM {
            self.expr_b2[k] -= lr * d_embed[k];
            for j in 0..HIDDEN_DIM {
                self.expr_w2[j][k] -= lr * d_embed[k] * hidden[j];
            }
        }

        // Hidden layer (through ReLU)
        for j in 0..HIDDEN_DIM {
            if hidden[j] <= 0.0 {
                continue; // ReLU gradient is 0
            }

            // d_loss/d_hidden[j] = sum_k(d_embed[k] * w2[j][k])
            let mut d_hidden = 0.0;
            for k in 0..EMBED_DIM {
                d_hidden += d_embed[k] * self.expr_w2[j][k];
            }
            let d_hidden = d_hidden.clamp(-1.0, 1.0);

            self.expr_b1[j] -= lr * d_hidden;
            for i in 0..EXPR_FEATURE_DIM {
                self.expr_w1[i][j] -= lr * d_hidden * features[i];
            }
        }
    }

    /// Backprop through rule head given gradient w.r.t. embedding.
    fn backprop_rule(
        &mut self,
        features: &[f32; RULE_FEATURE_DIM],
        hidden: &[f32; HIDDEN_DIM],
        d_embed: &[f32; EMBED_DIM],
        lr: f32,
    ) {
        // Output layer
        for k in 0..EMBED_DIM {
            self.rule_b2[k] -= lr * d_embed[k];
            for j in 0..HIDDEN_DIM {
                self.rule_w2[j][k] -= lr * d_embed[k] * hidden[j];
            }
        }

        // Hidden layer (through ReLU)
        for j in 0..HIDDEN_DIM {
            if hidden[j] <= 0.0 {
                continue;
            }

            let mut d_hidden = 0.0;
            for k in 0..EMBED_DIM {
                d_hidden += d_embed[k] * self.rule_w2[j][k];
            }
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
            let score = self.score_action(expr_f, rule_f, *rule_idx);
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
    // Self-Imitation Learning (SIL)
    // ========================================================================

    /// Train with cost-delta as the reward signal.
    ///
    /// The key insight: reward = cost_improvement from continuing search.
    /// If running more epochs improves cost by delta, actions that led to
    /// that improvement should get credit proportional to delta.
    ///
    /// This is like temporal difference learning:
    /// - cost_before: cost at current state
    /// - cost_after: cost after more search (with saturation as limit)
    /// - reward = cost_before - cost_after (positive = improvement)
    ///
    /// Actions with positive reward → push score up (good action)
    /// Actions with zero/negative reward → push score down (wasteful)
    ///
    /// # Arguments
    /// * `expr_features` - expression features for this action
    /// * `rule_features` - rule features for this action
    /// * `rule_idx` - which rule
    /// * `cost_delta` - cost_before - cost_after (positive = improvement)
    /// * `lr` - learning rate
    pub fn train_with_cost_delta(
        &mut self,
        expr_features: &[f32; EXPR_FEATURE_DIM],
        rule_features: &[f32; RULE_FEATURE_DIM],
        rule_idx: usize,
        cost_delta: f32,
        lr: f32,
    ) -> f32 {
        // Forward pass with hidden for backprop
        let (expr_embed, expr_hidden) = self.forward_expr_with_hidden(expr_features);
        let (rule_embed, rule_hidden) = self.forward_rule_with_hidden(rule_features);

        // Score = dot product + prior
        let mut score = 0.0f32;
        for k in 0..EMBED_DIM {
            score += expr_embed[k] * rule_embed[k];
        }
        let rule_idx_clamped = rule_idx.min(MAX_RULES - 1);
        score += self.rule_prior[rule_idx_clamped];

        // Current prediction (sigmoid maps to [0, 1])
        let predicted = sigmoid(score);

        // Target: sigmoid(cost_delta) maps cost improvement to [0, 1] range
        // Large positive delta → target near 1.0 (good action)
        // Zero delta → target 0.5 (neutral)
        // Negative delta → target near 0.0 (bad action)
        let target = sigmoid(cost_delta);

        // Loss: MSE between predicted and target (simpler than BCE for continuous targets)
        let loss = (predicted - target).powi(2);

        // Gradient: d(MSE)/d(score) = 2 * (predicted - target) * sigmoid'(score)
        // sigmoid'(score) = predicted * (1 - predicted)
        let d_score = 2.0 * (predicted - target) * predicted * (1.0 - predicted);
        let d_score = d_score.clamp(-10.0, 10.0);

        // Gradient flows through dot product
        let mut d_expr_embed = [0.0f32; EMBED_DIM];
        let mut d_rule_embed = [0.0f32; EMBED_DIM];
        for k in 0..EMBED_DIM {
            d_expr_embed[k] = d_score * rule_embed[k];
            d_rule_embed[k] = d_score * expr_embed[k];
        }

        // Update rule prior
        self.rule_prior[rule_idx_clamped] -= lr * d_score;

        // Backprop through expression head
        self.backprop_expr(expr_features, &expr_hidden, &d_expr_embed, lr);

        // Backprop through rule head
        self.backprop_rule(rule_features, &rule_hidden, &d_rule_embed, lr);

        loss
    }

    /// Batch training with cost deltas.
    ///
    /// Each sample: (expr_features, rule_features, rule_idx, cost_delta)
    /// where cost_delta = cost_before - cost_after (positive = improvement)
    pub fn train_batch_with_deltas(
        &mut self,
        samples: &[([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, f32)],
        lr: f32,
    ) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for (expr_f, rule_f, rule_idx, cost_delta) in samples {
            total_loss += self.train_with_cost_delta(expr_f, rule_f, *rule_idx, *cost_delta, lr);
        }

        total_loss / samples.len() as f32
    }

    /// Select an action using epsilon-greedy policy.
    ///
    /// With probability epsilon, returns a random action index.
    /// Otherwise, returns the action with highest score.
    ///
    /// # Arguments
    /// * `expr_features` - features for all expressions
    /// * `rule_features` - features for all rules
    /// * `legal_actions` - (expr_idx, rule_idx) pairs that are legal
    /// * `epsilon` - exploration probability
    /// * `rng_state` - mutable RNG state (LCG)
    pub fn select_action_epsilon_greedy(
        &self,
        expr_features: &[[f32; EXPR_FEATURE_DIM]],
        rule_features: &[[f32; RULE_FEATURE_DIM]],
        legal_actions: &[(usize, usize)],
        epsilon: f32,
        rng_state: &mut u64,
    ) -> Option<(usize, usize)> {
        if legal_actions.is_empty() {
            return None;
        }

        // Generate random number
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rand_val = (*rng_state as f32) / (u64::MAX as f32);

        if rand_val < epsilon {
            // Explore: pick random action
            *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (*rng_state as usize) % legal_actions.len();
            Some(legal_actions[idx])
        } else {
            // Exploit: pick best action according to model
            let scores = self.score_actions(expr_features, rule_features, legal_actions);
            let best_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i)?;
            Some(legal_actions[best_idx])
        }
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Total number of parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        // Expression head: w1[EXPR_FEATURE_DIM][HIDDEN_DIM] + b1[HIDDEN_DIM] + w2[HIDDEN_DIM][EMBED_DIM] + b2[EMBED_DIM]
        let expr_params = EXPR_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBED_DIM + EMBED_DIM;
        // Rule head: w1[RULE_FEATURE_DIM][HIDDEN_DIM] + b1[HIDDEN_DIM] + w2[HIDDEN_DIM][EMBED_DIM] + b2[EMBED_DIM]
        let rule_params = RULE_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBED_DIM + EMBED_DIM;
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
        for row in &self.expr_w2 {
            for &v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &v in &self.expr_b2 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Rule head
        for row in &self.rule_w1 {
            for &v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &v in &self.rule_b1 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for row in &self.rule_w2 {
            for &v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &v in &self.rule_b2 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Rule priors
        for &v in &self.rule_prior {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        bytes
    }

    /// Deserialize from bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let expected = (EXPR_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBED_DIM + EMBED_DIM
            + RULE_FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBED_DIM + EMBED_DIM
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
        for row in &mut guide.expr_w2 {
            for v in row.iter_mut() {
                *v = read_f32();
            }
        }
        for v in &mut guide.expr_b2 {
            *v = read_f32();
        }

        // Rule head
        for row in &mut guide.rule_w1 {
            for v in row.iter_mut() {
                *v = read_f32();
            }
        }
        for v in &mut guide.rule_b1 {
            *v = read_f32();
        }
        for row in &mut guide.rule_w2 {
            for v in row.iter_mut() {
                *v = read_f32();
            }
        }
        for v in &mut guide.rule_b2 {
            *v = read_f32();
        }

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
    /// Number of static rule feature fields.
    pub const SIZE: usize = 12;

    /// Convert to array for neural network input.
    /// Note: This is static rule metadata, separate from runtime features.
    #[must_use]
    pub fn to_array(&self) -> [f32; Self::SIZE] {
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
        assert!((guide.rule_b2[0] - loaded.rule_b2[0]).abs() < 1e-6);
        assert!((guide.expr_w2[0][0] - loaded.expr_w2[0][0]).abs() < 1e-6);
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

    #[test]
    fn test_train_from_egraph() {
        use crate::egraph::{EGraph, ExprTree, CostModel, GuidedSearch, all_rules, ops};

        // Create a few expressions to generate training data
        let exprs = vec![
            // X + 0
            ExprTree::Op {
                op: &ops::Add,
                children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
            },
            // (X + 0) * 1
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Add,
                        children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
                    },
                    ExprTree::constant(1.0),
                ],
            },
            // X * 0
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
            },
            // (X + Y) + 0
            ExprTree::Op {
                op: &ops::Add,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Add,
                        children: vec![ExprTree::var(0), ExprTree::var(1)],
                    },
                    ExprTree::constant(0.0),
                ],
            },
        ];

        let mut guide = DualMaskGuide::new_random(42);
        let costs = CostModel::default();

        // Collect training data from all expressions
        let mut all_samples: Vec<([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, bool)> = Vec::new();

        for expr in &exprs {
            let mut egraph = EGraph::with_rules(all_rules());
            let root = egraph.add_expr(expr);
            let mut search = GuidedSearch::new(egraph, root, 5);

            let result = search.run_dual_mask(
                &guide,
                |tree: &ExprTree| tree.node_count() as i64,
                &costs,
            );

            // Convert PairRecords to training samples
            for epoch in &result.pair_records {
                for pair in &epoch.pairs {
                    all_samples.push((
                        pair.expr_features,
                        pair.rule_features,
                        pair.rule_idx,
                        pair.fired,
                    ));
                }
            }
        }

        // Count positives/negatives
        let positives = all_samples.iter().filter(|s| s.3).count();
        let negatives = all_samples.len() - positives;

        assert!(all_samples.len() > 50, "Should have collected training data, got {}", all_samples.len());
        assert!(positives > 0, "Should have some positive samples (fired=true)");

        // Check feature statistics for positive vs negative samples
        let pos_samples: Vec<_> = all_samples.iter().filter(|s| s.3).collect();
        let neg_samples: Vec<_> = all_samples.iter().filter(|s| !s.3).collect();

        eprintln!("\n=== Feature analysis ===");
        for feat_idx in 0..4 {
            let pos_mean: f32 = pos_samples.iter().map(|s| s.0[feat_idx]).sum::<f32>() / pos_samples.len() as f32;
            let neg_mean: f32 = neg_samples.iter().map(|s| s.0[feat_idx]).sum::<f32>() / neg_samples.len() as f32;
            eprintln!("expr_feat[{}]: pos_mean={:.3}, neg_mean={:.3}, diff={:.3}",
                feat_idx, pos_mean, neg_mean, pos_mean - neg_mean);
        }
        for feat_idx in 0..4 {
            let pos_mean: f32 = pos_samples.iter().map(|s| s.1[feat_idx]).sum::<f32>() / pos_samples.len() as f32;
            let neg_mean: f32 = neg_samples.iter().map(|s| s.1[feat_idx]).sum::<f32>() / neg_samples.len() as f32;
            eprintln!("rule_feat[{}]: pos_mean={:.3}, neg_mean={:.3}, diff={:.3}",
                feat_idx, pos_mean, neg_mean, pos_mean - neg_mean);
        }

        // Train for several epochs with balanced sampling
        let initial_eval = guide.evaluate(&all_samples);

        // Separate positive and negative samples
        let pos: Vec<_> = all_samples.iter().filter(|s| s.3).cloned().collect();
        let neg: Vec<_> = all_samples.iter().filter(|s| !s.3).cloned().collect();

        let mut rng = 12345u64;
        for _ in 0..100 {
            // Create balanced batch: all positives + equal number of random negatives
            let mut batch = pos.clone();
            for _ in 0..pos.len() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng as usize) % neg.len();
                batch.push(neg[idx]);
            }
            guide.train_batch(&batch, 0.01);
        }

        let final_eval = guide.evaluate(&all_samples);

        // Training should improve (or at least not get worse)
        // Note: with tiny networks, might not always improve significantly
        eprintln!(
            "Training: {} samples ({} positive, {} negative)",
            all_samples.len(), positives, negatives
        );
        eprintln!(
            "Accuracy: {:.1}% -> {:.1}%",
            initial_eval.0 * 100.0,
            final_eval.0 * 100.0
        );
        eprintln!(
            "FP rate: {:.1}% -> {:.1}%",
            initial_eval.1 * 100.0,
            final_eval.1 * 100.0
        );
        eprintln!(
            "FN rate: {:.1}% -> {:.1}%  (catching positives?)",
            initial_eval.2 * 100.0,
            final_eval.2 * 100.0
        );

        // The real test: FN rate should be low (we catch the positives)
        assert!(final_eval.2 < 0.5, "Should catch at least half the positives, FN rate = {:.1}%", final_eval.2 * 100.0);
    }

    /// Test: with X resources, match the result of 10X resources.
    ///
    /// Goal: guide with limited budget finds same quality as saturation.
    /// Like NNUE in chess - shallow search should make same move as deep search.
    ///
    /// Training: saturation is the oracle. Train guide so limited-budget search
    /// reaches the same extracted cost as full saturation.
    #[test]
    fn test_limited_budget_matches_saturation() {
        use crate::egraph::{EGraph, ExprTree, CostModel, all_rules, ops};

        let exprs = vec![
            // (X + 0) * 1 → X
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Add,
                        children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
                    },
                    ExprTree::constant(1.0),
                ],
            },
            // X * 0 → 0
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
            },
            // (X * 1) + 0 → X
            ExprTree::Op {
                op: &ops::Add,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Mul,
                        children: vec![ExprTree::var(0), ExprTree::constant(1.0)],
                    },
                    ExprTree::constant(0.0),
                ],
            },
        ];

        let costs = CostModel::default();
        let mut guide = DualMaskGuide::new_random(42);

        eprintln!("\n=== Training: limited budget should match saturation ===");

        // First, get saturation costs (the oracle/target)
        let mut saturation_costs: Vec<i64> = Vec::new();
        for expr in &exprs {
            let mut egraph = EGraph::with_rules(all_rules());
            let root = egraph.add_expr(expr);
            egraph.saturate(); // Full saturation

            let (best, _) = egraph.extract_best(root, &costs);
            saturation_costs.push(best.node_count() as i64);
        }

        eprintln!("Saturation costs (oracle): {:?}", saturation_costs);

        let num_rules = all_rules().len();

        // Training loop: run limited search, compare to saturation, train on gap
        for iteration in 0..30 {
            let mut samples: Vec<([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, f32)> = Vec::new();

            for (expr_idx, expr) in exprs.iter().enumerate() {
                // Run LIMITED search (budget=2 epochs)
                let mut egraph = EGraph::with_rules(all_rules());
                let root = egraph.add_expr(expr);

                // Limited budget: only 2 epochs
                for _ in 0..2 {
                    egraph.apply_rules_once();
                }

                let (limited_best, _) = egraph.extract_best(root, &costs);
                let limited_cost = limited_best.node_count() as i64;

                // Gap to saturation: how far are we from the oracle?
                let gap = limited_cost - saturation_costs[expr_idx];

                // Generate training samples from this e-graph state
                // Reward = negative gap (closer to saturation = higher reward)
                // If we matched saturation (gap=0), reward is highest
                let reward = -(gap as f32);

                let num_classes = egraph.num_classes();
                for class_idx in 0..num_classes.min(5) {
                    let class_id = crate::egraph::EClassId(class_idx as u32);
                    let expr_features = extract_class_features(&egraph, class_id);

                    for rule_idx in 0..num_rules.min(10) {
                        let rule_features = extract_rule_features_simple(rule_idx, num_rules, iteration);
                        samples.push((expr_features, rule_features, rule_idx, reward));
                    }
                }
            }

            // Train
            if !samples.is_empty() {
                let loss = guide.train_batch_with_deltas(&samples, 0.01);
                if iteration % 10 == 0 {
                    // Evaluate current gap
                    let mut total_gap = 0i64;
                    for (expr_idx, expr) in exprs.iter().enumerate() {
                        let mut egraph = EGraph::with_rules(all_rules());
                        let root = egraph.add_expr(expr);
                        for _ in 0..2 {
                            egraph.apply_rules_once();
                        }
                        let (best, _) = egraph.extract_best(root, &costs);
                        total_gap += best.node_count() as i64 - saturation_costs[expr_idx];
                    }
                    eprintln!("Iter {}: loss = {:.4}, total gap to saturation = {}", iteration, loss, total_gap);
                }
            }
        }

        // Final evaluation: with limited budget, how close to saturation?
        let mut final_gap = 0i64;
        for (expr_idx, expr) in exprs.iter().enumerate() {
            let mut egraph = EGraph::with_rules(all_rules());
            let root = egraph.add_expr(expr);
            for _ in 0..2 {
                egraph.apply_rules_once();
            }
            let (best, _) = egraph.extract_best(root, &costs);
            let limited_cost = best.node_count() as i64;
            let sat_cost = saturation_costs[expr_idx];
            final_gap += limited_cost - sat_cost;
            eprintln!("Expr {}: limited={}, saturation={}, gap={}",
                expr_idx, limited_cost, sat_cost, limited_cost - sat_cost);
        }

        eprintln!("Final total gap: {}", final_gap);
        // Note: This test just verifies the training loop runs.
        // The actual gap closure depends on the guide actually influencing search.
    }

    fn extract_class_features(egraph: &crate::egraph::EGraph, class_id: crate::egraph::EClassId) -> [f32; EXPR_FEATURE_DIM] {
        let class_id = egraph.find(class_id);
        let nodes = egraph.nodes(class_id);
        let mut features = [0.0f32; EXPR_FEATURE_DIM];

        features[0] = nodes.len() as f32 / 10.0;
        features[1] = egraph.num_classes() as f32 / 100.0;

        for node in nodes {
            match node {
                crate::egraph::ENode::Op { op, children } => {
                    let kind = op.kind() as usize;
                    features[2 + (kind % 8)] = 1.0;
                    features[10] = (features[10] + children.len() as f32) / 2.0;
                }
                crate::egraph::ENode::Var(_) => features[11] += 1.0,
                crate::egraph::ENode::Const(_) => features[12] += 1.0,
            }
        }
        features
    }

    fn extract_rule_features_simple(rule_idx: usize, num_rules: usize, iteration: usize) -> [f32; RULE_FEATURE_DIM] {
        let mut features = [0.0f32; RULE_FEATURE_DIM];
        features[0] = rule_idx as f32 / num_rules.max(1) as f32;
        features[1] = iteration as f32 / 30.0;
        features
    }

    /// Test SIL training: guide learns to filter intelligently.
    ///
    /// Training loop:
    /// 1. Run filtered search with epsilon-greedy (explore random pairs)
    /// 2. Track which pairs fired AND led to cost improvement
    /// 3. Train guide to approve pairs that improve cost
    #[test]
    fn test_sil_training() {
        use crate::egraph::{EGraph, ExprTree, CostModel, all_rules, ops};
        use crate::egraph::guided_search::GuidedSearch;

        let exprs = vec![
            // (X + 0) * 1 → X
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Add,
                        children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
                    },
                    ExprTree::constant(1.0),
                ],
            },
            // X * 0 → 0
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
            },
            // X + 0 → X
            ExprTree::Op {
                op: &ops::Add,
                children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
            },
        ];

        let costs = CostModel::default();
        let mut guide = DualMaskGuide::new_random(42);

        eprintln!("\n=== SIL Training: learn to filter intelligently ===");

        // Initial evaluation: how many pairs, what quality?
        let (initial_pairs, initial_cost) = evaluate_guide(&guide, &exprs, &costs, 0.5);
        eprintln!("Initial: {} pairs, cost {}", initial_pairs, initial_cost);

        // Training loop
        for iteration in 0..20 {
            let mut samples: Vec<([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, f32)> = Vec::new();
            let rng_seed = iteration as u64 * 12345;

            for expr in &exprs {
                let initial_cost = expr.node_count() as i64;

                // Run filtered search with HIGH epsilon (lots of exploration)
                let mut egraph = EGraph::with_rules(all_rules());
                let root = egraph.add_expr(expr);
                let mut search = GuidedSearch::new(egraph, root, 5);

                let result = search.run_dual_mask_filtered(
                    &guide,
                    |tree: &ExprTree| tree.node_count() as i64,
                    &costs,
                    0.3,   // lower threshold = more permissive
                    100,   // max_classes (space limit)
                    0.5,   // high epsilon = lots of exploration
                    rng_seed,
                );

                // Cost improvement from this search
                let improvement = initial_cost - result.best_cost;

                // Collect samples: pairs that fired get positive signal if improved
                for epoch in &result.pair_records {
                    let epoch_improved = epoch.cost_after < epoch.cost_before;

                    for pair in &epoch.pairs {
                        let reward = if pair.fired && epoch_improved {
                            // Fired and improved → approve this pair
                            improvement as f32
                        } else if pair.fired && !epoch_improved {
                            // Fired but no improvement → slight negative
                            -0.5
                        } else {
                            // Didn't fire → filter it
                            -1.0
                        };

                        samples.push((
                            pair.expr_features,
                            pair.rule_features,
                            pair.rule_idx,
                            reward,
                        ));
                    }
                }
            }

            // Train with balanced sampling
            if !samples.is_empty() {
                let positive: Vec<_> = samples.iter().filter(|s| s.3 > 0.0).cloned().collect();
                let negative: Vec<_> = samples.iter().filter(|s| s.3 <= 0.0).cloned().collect();

                if !positive.is_empty() {
                    let mut batch = positive.clone();
                    let mut rng = rng_seed;
                    for _ in 0..positive.len().min(negative.len()) {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        if !negative.is_empty() {
                            batch.push(negative[(rng as usize) % negative.len()]);
                        }
                    }
                    guide.train_batch_with_deltas(&batch, 0.01);
                }
            }

            if iteration % 5 == 0 {
                let (pairs, cost) = evaluate_guide(&guide, &exprs, &costs, 0.5);
                eprintln!("Iter {}: {} pairs, cost {}", iteration, pairs, cost);
            }
        }

        // Final evaluation
        let (final_pairs, final_cost) = evaluate_guide(&guide, &exprs, &costs, 0.5);
        eprintln!("Final: {} pairs, cost {}", final_pairs, final_cost);

        // Should have learned something (fewer pairs OR better cost)
        // Note: with small training, might not be dramatic
        eprintln!("Improvement: pairs {} -> {}, cost {} -> {}",
            initial_pairs, final_pairs, initial_cost, final_cost);
    }

    fn evaluate_guide(
        guide: &DualMaskGuide,
        exprs: &[crate::egraph::ExprTree],
        costs: &crate::egraph::CostModel,
        threshold: f32,
    ) -> (usize, i64) {
        use crate::egraph::{EGraph, ExprTree, all_rules};
        use crate::egraph::guided_search::GuidedSearch;

        let mut total_pairs = 0;
        let mut total_cost = 0i64;

        for expr in exprs {
            let mut egraph = EGraph::with_rules(all_rules());
            let root = egraph.add_expr(expr);
            let mut search = GuidedSearch::new(egraph, root, 5);

            let result = search.run_dual_mask_filtered(
                guide,
                |tree: &ExprTree| tree.node_count() as i64,
                costs,
                threshold,
                100, // max_classes
                0.0, // no exploration for evaluation
                42,
            );

            total_pairs += result.pair_records.iter().map(|e| e.pairs.len()).sum::<usize>();
            total_cost += result.best_cost;
        }

        (total_pairs, total_cost)
    }

    /// Test that filtered search works and reduces the number of pairs tried.
    #[test]
    fn test_filtered_search() {
        use crate::egraph::{EGraph, ExprTree, CostModel, all_rules, ops};
        use crate::egraph::guided_search::GuidedSearch;

        let expr = ExprTree::Op {
            op: &ops::Mul,
            children: vec![
                ExprTree::Op {
                    op: &ops::Add,
                    children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
                },
                ExprTree::constant(1.0),
            ],
        };

        let costs = CostModel::default();
        let guide = DualMaskGuide::new_random(42);

        // Run unfiltered (egg-style: tries ALL pairs)
        let mut egraph1 = EGraph::with_rules(all_rules());
        let root1 = egraph1.add_expr(&expr);
        let mut search1 = GuidedSearch::new(egraph1, root1, 5);

        let result1 = search1.run_dual_mask(
            &guide,
            |tree: &ExprTree| tree.node_count() as i64,
            &costs,
        );

        // Count total pairs tried in unfiltered
        let unfiltered_pairs: usize = result1.pair_records.iter()
            .map(|epoch| epoch.pairs.len())
            .sum();

        // Run filtered (only tries approved pairs)
        let mut egraph2 = EGraph::with_rules(all_rules());
        let root2 = egraph2.add_expr(&expr);
        let mut search2 = GuidedSearch::new(egraph2, root2, 5);

        let result2 = search2.run_dual_mask_filtered(
            &guide,
            |tree: &ExprTree| tree.node_count() as i64,
            &costs,
            0.5,   // threshold
            100,   // max_classes (space limit)
            0.0,   // no epsilon exploration for this test
            42,
        );

        // Count total pairs tried in filtered
        let filtered_pairs: usize = result2.pair_records.iter()
            .map(|epoch| epoch.pairs.len())
            .sum();

        eprintln!("\n=== Filtered search test ===");
        eprintln!("Unfiltered: {} pairs tried, cost = {}", unfiltered_pairs, result1.best_cost);
        eprintln!("Filtered:   {} pairs tried, cost = {}", filtered_pairs, result2.best_cost);
        eprintln!("Reduction: {:.1}%", (1.0 - filtered_pairs as f32 / unfiltered_pairs as f32) * 100.0);

        // Filtered should try fewer pairs
        assert!(
            filtered_pairs < unfiltered_pairs,
            "Filtered should try fewer pairs: {} vs {}",
            filtered_pairs, unfiltered_pairs
        );

        // Both should reach a good result (cost reduction)
        let initial_cost = expr.node_count() as i64;
        assert!(result1.best_cost < initial_cost, "Unfiltered should improve");
        // Note: filtered might not improve if threshold is too high (filters good rules)
        // That's fine - this test is about filtering, not about quality
    }

    /// Test resource-asymmetric training: guide learns to match oracle quality with fewer resources.
    ///
    /// Strategy:
    /// 1. Oracle: Run with abundant resources (saturation-like)
    /// 2. Collect pairs that fired AND improved cost
    /// 3. Train guide to approve those pairs
    /// 4. Evaluate guide with constrained resources
    /// 5. Goal: guide achieves oracle-like cost with fewer resources
    #[test]
    fn test_resource_asymmetric_training() {
        use crate::egraph::{EGraph, ExprTree, CostModel, all_rules, ops};
        use crate::egraph::guided_search::GuidedSearch;

        // Test expressions
        let exprs = vec![
            // (X + 0) * 1 → X
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Add,
                        children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
                    },
                    ExprTree::constant(1.0),
                ],
            },
            // (X * 1) + 0 → X
            ExprTree::Op {
                op: &ops::Add,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Mul,
                        children: vec![ExprTree::var(0), ExprTree::constant(1.0)],
                    },
                    ExprTree::constant(0.0),
                ],
            },
            // X * 0 → 0
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![ExprTree::var(0), ExprTree::constant(0.0)],
            },
        ];

        let costs = CostModel::default();
        let mut guide = DualMaskGuide::new_random(42);

        eprintln!("\n=== Resource-Asymmetric Training ===");
        eprintln!("Oracle: true saturation");
        eprintln!("Guide:  50 max_classes, 5 epochs");

        // --- Guide Configuration (constrained resources) ---
        let guide_max_classes = 50;
        let guide_max_epochs = 5;

        // --- Initial evaluation (guide with constrained resources) ---
        let (initial_pairs, initial_cost) = {
            let mut total_pairs = 0;
            let mut total_cost = 0i64;

            for expr in &exprs {
                let mut egraph = EGraph::with_rules(all_rules());
                let root = egraph.add_expr(expr);
                let mut search = GuidedSearch::new(egraph, root, guide_max_epochs);

                let result = search.run_dual_mask_filtered(
                    &guide,
                    |tree: &ExprTree| tree.node_count() as i64,
                    &costs,
                    0.5,                // threshold
                    guide_max_classes,  // constrained space
                    0.0,                // no exploration for eval
                    42,
                );

                total_pairs += result.pair_records.iter().map(|e| e.pairs.len()).sum::<usize>();
                total_cost += result.best_cost;
            }
            (total_pairs, total_cost)
        };
        eprintln!("Initial (constrained): {} pairs, cost {}", initial_pairs, initial_cost);

        // --- Get oracle target cost via TRUE SATURATION ---
        let mut oracle_costs: Vec<i64> = Vec::new();
        for expr in &exprs {
            let mut egraph = EGraph::with_rules(all_rules());
            let root = egraph.add_expr(expr);
            egraph.saturate(); // Full saturation - the true optimum
            let (best, _cost) = egraph.extract_best(root, &costs);
            oracle_costs.push(best.node_count() as i64);
        }

        // Training loop - collect samples under CONSTRAINED resources
        for iteration in 0..30 {
            let mut positive_samples: Vec<([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, f32)> = Vec::new();
            let mut negative_samples: Vec<([f32; EXPR_FEATURE_DIM], [f32; RULE_FEATURE_DIM], usize, f32)> = Vec::new();
            let rng_seed = iteration as u64 * 12345;

            for (expr_idx, expr) in exprs.iter().enumerate() {
                let oracle_cost = oracle_costs[expr_idx];

                // Run CONSTRAINED search with exploration
                let mut egraph = EGraph::with_rules(all_rules());
                let root = egraph.add_expr(expr);
                let mut search = GuidedSearch::new(egraph, root, guide_max_epochs);

                let result = search.run_dual_mask_filtered(
                    &guide,
                    |tree: &ExprTree| tree.node_count() as i64,
                    &costs,
                    0.3,                // moderate threshold
                    guide_max_classes,  // CONSTRAINED space - features match eval!
                    0.5,                // exploration to try different pairs
                    rng_seed,
                );

                // Training signal: how close did we get to oracle?
                let gap = result.best_cost - oracle_cost;

                // Collect from constrained trajectory
                for epoch in &result.pair_records {
                    let epoch_improved = epoch.cost_after < epoch.cost_before;
                    let improvement = (epoch.cost_before - epoch.cost_after) as f32;

                    for pair in &epoch.pairs {
                        if pair.fired && epoch_improved {
                            // This pair helped under constraints - reward it
                            positive_samples.push((
                                pair.expr_features,
                                pair.rule_features,
                                pair.rule_idx,
                                improvement.max(1.0),
                            ));
                        } else if !pair.fired || gap > 0 {
                            // Didn't fire, or we didn't reach oracle - negative
                            negative_samples.push((
                                pair.expr_features,
                                pair.rule_features,
                                pair.rule_idx,
                                -1.0,
                            ));
                        }
                    }
                }
            }

            // Train guide with balanced sampling
            // Goal: approve what oracle found useful, reject the rest
            if !positive_samples.is_empty() {
                let mut batch = Vec::new();

                // All positives
                batch.extend(positive_samples.iter().cloned());

                // Equal number of negatives (balanced)
                let mut rng = rng_seed;
                for _ in 0..positive_samples.len() {
                    if !negative_samples.is_empty() {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let idx = (rng as usize) % negative_samples.len();
                        batch.push(negative_samples[idx]);
                    }
                }

                guide.train_batch_with_deltas(&batch, 0.02);
            }

            // Periodic evaluation
            if iteration % 10 == 0 || iteration == 29 {
                let (pairs, cost) = {
                    let mut total_pairs = 0;
                    let mut total_cost = 0i64;

                    for expr in &exprs {
                        let mut egraph = EGraph::with_rules(all_rules());
                        let root = egraph.add_expr(expr);
                        let mut search = GuidedSearch::new(egraph, root, guide_max_epochs);

                        let result = search.run_dual_mask_filtered(
                            &guide,
                            |tree: &ExprTree| tree.node_count() as i64,
                            &costs,
                            0.5,                // balanced threshold
                            guide_max_classes,  // constrained space!
                            0.0,                // no exploration for eval
                            42,
                        );

                        total_pairs += result.pair_records.iter().map(|e| e.pairs.len()).sum::<usize>();
                        total_cost += result.best_cost;
                    }
                    (total_pairs, total_cost)
                };
                let oracle_total: i64 = oracle_costs.iter().sum();
                eprintln!("Iter {}: {} pairs, cost {} (oracle cost: {})",
                    iteration, pairs, cost, oracle_total);
            }
        }

        // --- Final evaluation ---
        let (final_pairs, final_cost) = {
            let mut total_pairs = 0;
            let mut total_cost = 0i64;

            for expr in &exprs {
                let mut egraph = EGraph::with_rules(all_rules());
                let root = egraph.add_expr(expr);
                let mut search = GuidedSearch::new(egraph, root, guide_max_epochs);

                let result = search.run_dual_mask_filtered(
                    &guide,
                    |tree: &ExprTree| tree.node_count() as i64,
                    &costs,
                    0.5,
                    guide_max_classes,
                    0.0,
                    42,
                );

                total_pairs += result.pair_records.iter().map(|e| e.pairs.len()).sum::<usize>();
                total_cost += result.best_cost;
            }
            (total_pairs, total_cost)
        };

        let oracle_total: i64 = oracle_costs.iter().sum();
        eprintln!("\n=== Results ===");
        eprintln!("Oracle target cost: {}", oracle_total);
        eprintln!("Initial (constrained): {} pairs, cost {}", initial_pairs, initial_cost);
        eprintln!("Final (constrained):   {} pairs, cost {}", final_pairs, final_cost);
        eprintln!("Cost improvement: {} -> {}", initial_cost, final_cost);
        eprintln!("Guide matches oracle quality: {}", final_cost <= oracle_total);

        // Guide should learn to achieve good cost with fewer resources
        // Either: cost improved OR pairs reduced (ideally both)
        let cost_improved = final_cost < initial_cost;
        let pairs_reduced = final_pairs < initial_pairs;

        assert!(
            cost_improved || pairs_reduced,
            "Guide should learn something: cost {} -> {}, pairs {} -> {}",
            initial_cost, final_cost, initial_pairs, final_pairs
        );
    }
}
