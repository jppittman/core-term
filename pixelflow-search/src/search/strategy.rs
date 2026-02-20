//! SearchStrategy implementations.

extern crate alloc;

use alloc::vec::Vec;

use crate::nnue::{Expr, EMBED_DIM};
use super::{
    SearchStrategy, SearchResult, Trajectory, TrajectoryStep,
    ActionScore, StateEmbedding, StopReason, Evaluator,
};

/// Exhaustive search (saturation).
///
/// Applies all rules until no more changes. Only feasible for small kernels.
/// Used in curriculum learning Phase 1 to get ground truth.
pub struct ExhaustiveSearch {
    /// Maximum e-graph classes.
    pub max_classes: usize,
    /// Maximum epochs.
    pub max_epochs: usize,
}

impl ExhaustiveSearch {
    pub fn new(max_classes: usize, max_epochs: usize) -> Self {
        Self { max_classes, max_epochs }
    }
}

impl SearchStrategy for ExhaustiveSearch {
    fn search(
        &mut self,
        expr: &Expr,
        evaluator: &dyn Evaluator,
        _rule_embeds: &[[f32; EMBED_DIM]],
    ) -> SearchResult {
        // TODO: Implement actual saturation
        // For now, just return the input expression evaluated
        let cost = evaluator.evaluate(expr);
        SearchResult {
            best_expr: expr.clone(),
            best_cost: cost,
            initial_cost: cost,
            trajectory: Trajectory::default(),
            stop_reason: StopReason::Saturated,
        }
    }

    fn name(&self) -> &'static str {
        "Exhaustive"
    }
}

/// Best-first search with ε-greedy exploration.
///
/// Priority queue ordered by value prediction, with random exploration.
pub struct BestFirstSearch {
    /// Maximum expansions.
    pub max_expansions: usize,
    /// Exploration probability.
    pub epsilon: f32,
    /// Random seed.
    seed: u64,
}

impl BestFirstSearch {
    pub fn new(max_expansions: usize, epsilon: f32, seed: u64) -> Self {
        Self { max_expansions, epsilon, seed }
    }

    fn rand(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f32 / (1u64 << 31) as f32
    }
}

impl SearchStrategy for BestFirstSearch {
    fn search(
        &mut self,
        expr: &Expr,
        evaluator: &dyn Evaluator,
        _rule_embeds: &[[f32; EMBED_DIM]],
    ) -> SearchResult {
        // TODO: Implement actual best-first search
        // For now, placeholder
        let cost = evaluator.evaluate(expr);
        SearchResult {
            best_expr: expr.clone(),
            best_cost: cost,
            initial_cost: cost,
            trajectory: Trajectory::default(),
            stop_reason: StopReason::BudgetExhausted,
        }
    }

    fn name(&self) -> &'static str {
        "BestFirst"
    }
}

/// Filtered search using mask predictions.
///
/// Only applies rules that the mask predicts will help.
pub struct FilteredSearch {
    /// Threshold for rule approval (sigmoid(score) > threshold).
    pub threshold: f32,
    /// Exploration probability (apply random rule).
    pub epsilon: f32,
    /// Maximum epochs.
    pub max_epochs: usize,
    /// Maximum e-graph classes.
    pub max_classes: usize,
    /// Random seed.
    seed: u64,
}

impl FilteredSearch {
    pub fn new(threshold: f32, epsilon: f32, max_epochs: usize, max_classes: usize, seed: u64) -> Self {
        Self { threshold, epsilon, max_epochs, max_classes, seed }
    }

    fn rand(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f32 / (1u64 << 31) as f32
    }
}

impl SearchStrategy for FilteredSearch {
    fn search(
        &mut self,
        expr: &Expr,
        evaluator: &dyn Evaluator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> SearchResult {
        // TODO: Implement actual filtered search with mask
        // For now, placeholder
        let cost = evaluator.evaluate(expr);
        SearchResult {
            best_expr: expr.clone(),
            best_cost: cost,
            initial_cost: cost,
            trajectory: Trajectory::default(),
            stop_reason: StopReason::BudgetExhausted,
        }
    }

    fn name(&self) -> &'static str {
        "Filtered"
    }
}

/// MCTS search with neural policy/value guidance.
pub struct MctsSearch {
    /// Number of simulations per move.
    pub num_simulations: usize,
    /// Exploration constant (c_puct).
    pub c_puct: f32,
    /// Temperature for action selection.
    pub temperature: f32,
    /// Maximum depth.
    pub max_depth: usize,
}

impl MctsSearch {
    pub fn new(num_simulations: usize, c_puct: f32, temperature: f32, max_depth: usize) -> Self {
        Self { num_simulations, c_puct, temperature, max_depth }
    }
}

impl SearchStrategy for MctsSearch {
    fn search(
        &mut self,
        expr: &Expr,
        evaluator: &dyn Evaluator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> SearchResult {
        // TODO: Implement actual MCTS
        // For now, placeholder
        let cost = evaluator.evaluate(expr);
        SearchResult {
            best_expr: expr.clone(),
            best_cost: cost,
            initial_cost: cost,
            trajectory: Trajectory::default(),
            stop_reason: StopReason::BudgetExhausted,
        }
    }

    fn name(&self) -> &'static str {
        "MCTS"
    }
}

/// Identity search - returns input unchanged (baseline).
pub struct IdentitySearch;

impl SearchStrategy for IdentitySearch {
    fn search(
        &mut self,
        expr: &Expr,
        evaluator: &dyn Evaluator,
        _rule_embeds: &[[f32; EMBED_DIM]],
    ) -> SearchResult {
        let cost = evaluator.evaluate(expr);
        SearchResult {
            best_expr: expr.clone(),
            best_cost: cost,
            initial_cost: cost,
            trajectory: Trajectory::default(),
            stop_reason: StopReason::Saturated,
        }
    }

    fn name(&self) -> &'static str {
        "Identity"
    }
}
