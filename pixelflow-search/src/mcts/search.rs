//! MCTS search algorithm with UCB selection.
//!
//! Implements the core AlphaZero-style MCTS loop:
//! 1. SELECT: Traverse tree using UCB until reaching a leaf
//! 2. EXPAND: Add children using policy prior
//! 3. EVALUATE: Use value head to estimate state value
//! 4. BACKPROP: Update Q-values along the path

extern crate alloc;

use alloc::vec::Vec;
use super::node::{MctsArena, NodeRef};
use super::state::{AccumulatorDelta, MctsAction, MctsState};
use crate::nnue::factored::{ExprNnue, EMBED_DIM};

/// MCTS configuration parameters.
#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Number of simulations per search.
    pub num_simulations: usize,

    /// Exploration constant for UCB (c_puct).
    /// Higher values encourage more exploration.
    pub c_puct: f32,

    /// Temperature for action selection.
    /// 1.0 = proportional to visits, 0.0 = greedy.
    pub temperature: f32,

    /// Maximum tree depth.
    pub max_depth: usize,

    /// Dirichlet noise alpha for root exploration.
    /// Set to 0.0 to disable noise.
    pub dirichlet_alpha: f32,

    /// Weight of Dirichlet noise vs prior.
    pub dirichlet_weight: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 32,
            c_puct: 1.5,
            temperature: 1.0,
            max_depth: 50,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
        }
    }
}

impl MctsConfig {
    /// Config for training (with exploration).
    #[must_use]
    pub fn training() -> Self {
        Self {
            num_simulations: 64,
            c_puct: 1.5,
            temperature: 1.0,
            max_depth: 50,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
        }
    }

    /// Config for inference (greedy).
    #[must_use]
    pub fn inference() -> Self {
        Self {
            num_simulations: 32,
            c_puct: 1.0,
            temperature: 0.0,
            max_depth: 50,
            dirichlet_alpha: 0.0,
            dirichlet_weight: 0.0,
        }
    }
}

/// Result of MCTS search.
#[derive(Clone)]
pub struct MctsResult {
    /// Best action to take from root.
    pub best_action: Option<MctsAction>,

    /// Visit distribution over actions (policy target for training).
    pub action_probs: Vec<(MctsAction, f32)>,

    /// Estimated value of root state.
    pub root_value: f32,

    /// Number of simulations performed.
    pub simulations: usize,

    /// Tree statistics.
    pub tree_size: usize,
    pub max_depth_reached: usize,
}

/// MCTS search state.
pub struct MctsSearch<'a> {
    /// Configuration.
    pub config: MctsConfig,

    /// Neural network for evaluation.
    model: &'a ExprNnue,

    /// Pre-computed rule embeddings.
    rule_embeds: &'a [[f32; EMBED_DIM]],

    /// Tree arena.
    arena: MctsArena,

    /// Available actions (class, rule) pairs.
    /// In a real implementation, this would be computed dynamically
    /// based on which rules can fire at which e-classes.
    available_actions: Vec<MctsAction>,

    /// Pre-computed deltas for each action.
    /// Maps action index to accumulator delta.
    action_deltas: Vec<AccumulatorDelta>,
}

impl<'a> MctsSearch<'a> {
    /// Create a new MCTS search.
    #[must_use]
    pub fn new(
        config: MctsConfig,
        model: &'a ExprNnue,
        rule_embeds: &'a [[f32; EMBED_DIM]],
    ) -> Self {
        Self {
            config,
            model,
            rule_embeds,
            arena: MctsArena::with_capacity(1024),
            available_actions: Vec::new(),
            action_deltas: Vec::new(),
        }
    }

    /// Set available actions for search.
    ///
    /// Call this before `search()` to specify which (class, rule) pairs
    /// are legal moves from the current state.
    pub fn set_actions(&mut self, actions: Vec<MctsAction>, deltas: Vec<AccumulatorDelta>) {
        assert_eq!(
            actions.len(),
            deltas.len(),
            "Actions and deltas must have same length"
        );
        self.available_actions = actions;
        self.action_deltas = deltas;
    }

    /// Run MCTS search from the given state.
    pub fn search(&mut self, root_state: &MctsState) -> MctsResult {
        // Reset arena
        self.arena.reset();
        let root = self.arena.root();

        // Expand root with policy prior
        self.expand(root, root_state);

        // Add Dirichlet noise to root priors if configured
        if self.config.dirichlet_alpha > 0.0 {
            self.add_root_noise(root);
        }

        let mut max_depth = 0;

        // Run simulations
        for _ in 0..self.config.num_simulations {
            let mut current = root;
            let mut state = root_state.clone();
            let mut path = Vec::new();

            // SELECT: Descend tree using UCB
            while self.arena.get(current).is_expanded && !self.arena.get(current).is_terminal {
                if let Some(child) = self.arena.select_child_ucb(current, self.config.c_puct) {
                    path.push(current);
                    current = child;

                    // Apply action to state
                    if let Some(action) = self.arena.get(current).action {
                        if let Some(idx) = self.action_index(action) {
                            state.apply_in_place(action, &self.action_deltas[idx]);
                        }
                    }
                } else {
                    break;
                }
            }

            max_depth = max_depth.max(state.depth);

            // EXPAND: Add children if not terminal
            let node = self.arena.get(current);
            if !node.is_terminal && !node.is_expanded && node.visit_count > 0 {
                self.expand(current, &state);
            }

            // EVALUATE: Get value estimate
            let value = self.evaluate(&state);

            // BACKPROP: Update path
            self.arena.backpropagate(current, value);
        }

        // Extract results
        self.extract_results(root, max_depth)
    }

    /// Expand a node by adding children for all available actions.
    fn expand(&mut self, node_ref: NodeRef, state: &MctsState) {
        if self.available_actions.is_empty() {
            self.arena.get_mut(node_ref).is_terminal = true;
            return;
        }

        // Get policy priors from network
        let policy = self.model.policy_from_accumulator(
            &state.accumulator,
            self.rule_embeds,
        );

        // Convert to action priors with softmax
        let priors = self.compute_priors(&policy);

        // Add child for each action
        for (action, prior) in priors {
            self.arena.add_child(node_ref, action, prior);
        }

        self.arena.get_mut(node_ref).is_expanded = true;
    }

    /// Compute action priors from policy logits using softmax.
    fn compute_priors(&self, policy: &[f32]) -> Vec<(MctsAction, f32)> {
        if self.available_actions.is_empty() {
            return Vec::new();
        }

        // Get logits for available actions
        let mut logits: Vec<f32> = self.available_actions
            .iter()
            .map(|action| {
                // Policy is indexed by rule_idx
                if action.rule_idx < policy.len() {
                    policy[action.rule_idx]
                } else {
                    0.0
                }
            })
            .collect();

        // Softmax for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| libm::expf(x - max_logit)).sum();

        self.available_actions
            .iter()
            .zip(logits.iter())
            .map(|(&action, &logit)| {
                let prob = libm::expf(logit - max_logit) / exp_sum;
                (action, prob)
            })
            .collect()
    }

    /// Evaluate state using value head.
    fn evaluate(&self, state: &MctsState) -> f32 {
        // Use value head to estimate cost
        // Lower cost = better, so we negate for value
        let cost = self.model.predict_cost_from_accumulator(&state.accumulator);
        // Convert cost to value (higher is better)
        // Use negative log cost so lower costs give higher values
        -cost
    }

    /// Add Dirichlet noise to root priors for exploration.
    fn add_root_noise(&mut self, root: NodeRef) {
        let root_node = self.arena.get(root);
        let num_children = root_node.children.len();
        if num_children == 0 {
            return;
        }

        // Generate Dirichlet noise (simplified: use exponential samples)
        let mut noise: Vec<f32> = Vec::with_capacity(num_children);
        let mut rng_state = 12345u64; // Simple LCG for no_std
        for _ in 0..num_children {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
            // Approximate Gamma(alpha, 1) using exponential
            let sample = -libm::logf(1.0 - u) * self.config.dirichlet_alpha;
            noise.push(sample);
        }

        // Normalize noise
        let noise_sum: f32 = noise.iter().sum();
        if noise_sum > 0.0 {
            for n in &mut noise {
                *n /= noise_sum;
            }
        }

        // Mix noise with priors
        let children: Vec<NodeRef> = root_node.children.clone();
        for (child_ref, noise_val) in children.into_iter().zip(noise) {
            let child = self.arena.get_mut(child_ref);
            child.prior = (1.0 - self.config.dirichlet_weight) * child.prior
                + self.config.dirichlet_weight * noise_val;
        }
    }

    /// Find index of action in available_actions.
    fn action_index(&self, action: MctsAction) -> Option<usize> {
        self.available_actions
            .iter()
            .position(|&a| a == action)
    }

    /// Extract search results from tree.
    fn extract_results(&self, root: NodeRef, max_depth: usize) -> MctsResult {
        let root_node = self.arena.get(root);

        // Get action probabilities from visit counts
        let action_probs = if self.config.temperature > 0.0 {
            self.temperature_action_probs(root)
        } else {
            root_node.visit_distribution(&self.arena)
        };

        // Select best action
        let best_action = if self.config.temperature == 0.0 {
            // Greedy: pick most visited
            root_node
                .best_child_by_visits(&self.arena)
                .and_then(|node| node.action)
        } else {
            // Sample proportional to visits
            action_probs.first().map(|(action, _)| *action)
        };

        MctsResult {
            best_action,
            action_probs,
            root_value: root_node.q_value(),
            simulations: self.config.num_simulations,
            tree_size: self.arena.len(),
            max_depth_reached: max_depth,
        }
    }

    /// Compute action probabilities with temperature.
    fn temperature_action_probs(&self, root: NodeRef) -> Vec<(MctsAction, f32)> {
        let root_node = self.arena.get(root);

        let visits: Vec<(MctsAction, f32)> = root_node
            .children
            .iter()
            .filter_map(|&child_ref| {
                let child = self.arena.get(child_ref);
                child.action.map(|action| (action, child.visit_count as f32))
            })
            .collect();

        if visits.is_empty() {
            return Vec::new();
        }

        // Apply temperature
        let temp_inv = 1.0 / self.config.temperature.max(0.01);
        let powered: Vec<f32> = visits
            .iter()
            .map(|(_, v)| libm::powf(*v, temp_inv))
            .collect();

        let sum: f32 = powered.iter().sum();
        if sum == 0.0 {
            return Vec::new();
        }

        visits
            .into_iter()
            .zip(powered)
            .map(|((action, _), p)| (action, p / sum))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::factored::{ExprNnue, EdgeAccumulator, StructuralFeatures};

    #[test]
    fn test_mcts_config_default() {
        let config = MctsConfig::default();
        assert_eq!(config.num_simulations, 32);
        assert!((config.c_puct - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_mcts_search_empty_actions() {
        let model = ExprNnue::new_random(42);
        let rule_embeds: Vec<[f32; EMBED_DIM]> = Vec::new();

        let mut search = MctsSearch::new(MctsConfig::default(), &model, &rule_embeds);
        search.set_actions(Vec::new(), Vec::new());

        let state = MctsState::new(EdgeAccumulator::new(), StructuralFeatures::default());
        let result = search.search(&state);

        assert!(result.best_action.is_none());
        assert!(result.action_probs.is_empty());
    }

    #[test]
    fn test_mcts_search_single_action() {
        let model = ExprNnue::new_random(42);

        // Create simple rule embeddings
        let rule_embeds: Vec<[f32; EMBED_DIM]> = vec![[0.1; EMBED_DIM]];

        let mut search = MctsSearch::new(
            MctsConfig {
                num_simulations: 10,
                ..MctsConfig::default()
            },
            &model,
            &rule_embeds,
        );

        let action = MctsAction {
            class_idx: 0,
            rule_idx: 0,
        };
        search.set_actions(vec![action], vec![AccumulatorDelta::zero()]);

        let state = MctsState::new(EdgeAccumulator::new(), StructuralFeatures::default());
        let result = search.search(&state);

        // With single action, it should be selected
        assert_eq!(result.best_action, Some(action));
        assert_eq!(result.action_probs.len(), 1);
    }
}
