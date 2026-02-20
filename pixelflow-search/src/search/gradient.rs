//! GradientSignal implementations.

extern crate alloc;

use alloc::vec::Vec;

use crate::nnue::EMBED_DIM;
use super::{GradientSignal, Gradients, SearchResult, MaskMlpGrad, ValueMlpGrad};

/// Supervised gradient signal.
///
/// Used when we have ground truth costs (e.g., from saturation).
/// Loss = MSE(predicted, actual)
pub struct SupervisedSignal;

impl GradientSignal for SupervisedSignal {
    fn compute(
        &mut self,
        results: &[SearchResult],
        _oracle_costs: Option<&[f32]>,
        _rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Gradients {
        // TODO: Implement actual supervised gradient
        // For value head: MSE loss between predicted and actual cost
        let loss: f32 = results
            .iter()
            .map(|r| (r.initial_cost - r.best_cost).powi(2))
            .sum::<f32>() / results.len() as f32;

        Gradients {
            mask_mlp_grad: None,  // Supervised doesn't train mask
            interaction_grad: None,
            rule_bias_grad: None,
            value_mlp_grad: None,  // TODO: compute actual gradient
            loss,
        }
    }

    fn name(&self) -> &'static str {
        "Supervised"
    }
}

/// Synthetic oracle gradient signal.
///
/// Uses value head to generate policy targets:
/// 1. Sample actions, score with value head
/// 2. Target = softmax(-cost_delta / temperature)
/// 3. Loss = cross_entropy(mask_scores, target)
pub struct SyntheticOracleSignal {
    /// Number of actions to sample per state.
    pub actions_per_state: usize,
    /// Temperature for target distribution.
    pub temperature: f32,
}

impl SyntheticOracleSignal {
    pub fn new(actions_per_state: usize, temperature: f32) -> Self {
        Self { actions_per_state, temperature }
    }
}

impl GradientSignal for SyntheticOracleSignal {
    fn compute(
        &mut self,
        results: &[SearchResult],
        _oracle_costs: Option<&[f32]>,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Gradients {
        // TODO: Implement actual synthetic oracle gradient
        // For each state in trajectory:
        //   1. Sample random actions
        //   2. Compute cost_delta for each action using value head
        //   3. Target = softmax(-cost_delta / temperature)
        //   4. Loss = cross_entropy(mask_scores, target)
        //   5. Backprop through mask MLP

        let loss = 0.0; // Placeholder

        Gradients {
            mask_mlp_grad: None,  // TODO: compute actual gradient
            interaction_grad: None,
            rule_bias_grad: None,
            value_mlp_grad: None,
            loss,
        }
    }

    fn name(&self) -> &'static str {
        "SyntheticOracle"
    }
}

/// REINFORCE gradient signal.
///
/// Uses sparse reward from oracle comparison:
/// - reward = 1 if student <= oracle, else 0
/// - gradient = advantage * grad_log_prob
pub struct ReinforceSignal {
    /// Baseline for variance reduction (exponential moving average).
    baseline: f32,
    /// Decay rate for baseline update.
    pub decay: f32,
}

impl ReinforceSignal {
    pub fn new(decay: f32) -> Self {
        Self { baseline: 0.0, decay }
    }
}

impl GradientSignal for ReinforceSignal {
    fn compute(
        &mut self,
        results: &[SearchResult],
        oracle_costs: Option<&[f32]>,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Gradients {
        // Compute rewards and advantages
        let rewards: Vec<f32> = match oracle_costs {
            Some(oracles) => {
                results.iter().zip(oracles.iter())
                    .map(|(r, &oracle)| {
                        // reward = 1 if student <= oracle
                        if r.best_cost <= oracle { 1.0 } else { 0.0 }
                    })
                    .collect()
            }
            None => {
                // No oracle, use improvement as reward
                results.iter()
                    .map(|r| (r.initial_cost - r.best_cost).max(0.0))
                    .collect()
            }
        };

        // Update baseline
        let mean_reward: f32 = rewards.iter().sum::<f32>() / rewards.len() as f32;
        self.baseline = self.decay * self.baseline + (1.0 - self.decay) * mean_reward;

        // Compute advantages
        let advantages: Vec<f32> = rewards.iter()
            .map(|&r| r - self.baseline)
            .collect();

        // TODO: Implement actual REINFORCE gradient
        // For each decision in trajectory:
        //   if approved: d_log_prob = 1 - sigmoid(score)
        //   else: d_log_prob = -sigmoid(score)
        //   gradient += advantage * d_log_prob

        let loss = -mean_reward; // Negative reward as "loss"

        Gradients {
            mask_mlp_grad: None,  // TODO: compute actual gradient
            interaction_grad: None,
            rule_bias_grad: None,
            value_mlp_grad: None,
            loss,
        }
    }

    fn name(&self) -> &'static str {
        "REINFORCE"
    }
}

/// No gradient (inference only).
pub struct NoGradient;

impl GradientSignal for NoGradient {
    fn compute(
        &mut self,
        _results: &[SearchResult],
        _oracle_costs: Option<&[f32]>,
        _rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Gradients {
        Gradients {
            mask_mlp_grad: None,
            interaction_grad: None,
            rule_bias_grad: None,
            value_mlp_grad: None,
            loss: 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "None"
    }
}
