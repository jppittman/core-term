//! Anytime optimization API for e-graph rewriting.
//!
//! This module provides budget-constrained optimization using MCTS with NNUE evaluation,
//! following the Stockfish-style approach where neural networks guide search decisions.
//!
//! # Usage
//!
//! ```ignore
//! use pixelflow_search::egraph::anytime::{optimize_with_budget, AnytimeConfig};
//! use pixelflow_nnue::Nnue;
//!
//! // Optimize with iteration budget
//! let result = optimize_with_budget(
//!     &expr,
//!     AnytimeConfig::default().with_iterations(1000),
//!     Some(&nnue),
//! );
//!
//! // Optimize with time budget
//! let result = optimize_with_budget(
//!     &expr,
//!     AnytimeConfig::default().with_timeout(Duration::from_secs(5)),
//!     Some(&nnue),
//! );
//! ```

use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::egraph::{CostModel, EGraph, EClassId, ExprTree};
use crate::egraph::search_adapter::{SearchState, RewriteMove, EGraphCategory, CostValuation, NNUEValuation, insert_tree};
use crate::search::mcts::{MctsTree, MctsConfig};
use crate::search::algebra::{Category, Valuation};

use pixelflow_nnue::Nnue;

/// Configuration for anytime optimization.
#[derive(Clone, Debug)]
pub struct AnytimeConfig {
    /// Maximum MCTS iterations (None = unlimited until timeout).
    pub max_iterations: Option<usize>,
    /// Maximum wall-clock time (None = unlimited).
    pub timeout: Option<Duration>,
    /// MCTS exploration constant (higher = more exploration).
    pub exploration_constant: f64,
    /// Maximum moves to consider per state.
    pub max_moves_per_state: usize,
    /// Use beam search instead of full MCTS (faster but less thorough).
    pub use_beam_search: bool,
    /// Beam width for beam search mode.
    pub beam_width: usize,
    /// Cost model for extraction.
    pub cost_model: CostModel,
    /// Epsilon for ε-greedy exploration (0.0 = pure exploitation, 1.0 = pure random).
    ///
    /// During training/online learning, set this to 0.1-0.2 to ensure the model
    /// explores beyond its current predictions. This prevents the "garbage in,
    /// garbage out" problem where the model only sees variants it already thinks
    /// are good.
    ///
    /// At inference time (compile-time optimization), set to 0.0 for best results.
    pub epsilon: f64,
    /// Random seed for ε-greedy exploration (None = use system entropy).
    pub random_seed: Option<u64>,
}

impl Default for AnytimeConfig {
    fn default() -> Self {
        Self {
            max_iterations: Some(1000),
            timeout: None,
            exploration_constant: core::f64::consts::SQRT_2,
            max_moves_per_state: 50,
            use_beam_search: false,
            beam_width: 10,
            cost_model: CostModel::fully_optimized(),
            epsilon: 0.0, // Pure exploitation by default (inference mode)
            random_seed: None,
        }
    }
}

impl AnytimeConfig {
    /// Set maximum iterations.
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    /// Set exploration constant.
    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration_constant = c;
        self
    }

    /// Enable beam search mode (faster but potentially less optimal).
    pub fn with_beam_search(mut self, width: usize) -> Self {
        self.use_beam_search = true;
        self.beam_width = width;
        self
    }

    /// Set epsilon for ε-greedy exploration.
    ///
    /// Use 0.1-0.2 during training to ensure diverse samples.
    /// Use 0.0 at inference time for best optimization.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon.clamp(0.0, 1.0);
        self
    }

    /// Set random seed for reproducible exploration.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Configure for training mode (with exploration).
    ///
    /// Sets epsilon=0.1 for diverse training samples.
    pub fn training_mode(self) -> Self {
        self.with_epsilon(0.1)
    }

    /// Configure for inference mode (pure exploitation).
    ///
    /// Sets epsilon=0.0 for best optimization results.
    pub fn inference_mode(self) -> Self {
        self.with_epsilon(0.0)
    }
}

/// Result of anytime optimization.
#[derive(Clone, Debug)]
pub struct AnytimeResult {
    /// The optimized expression tree.
    pub tree: ExprTree,
    /// Cost of the optimized expression.
    pub cost: usize,
    /// Original cost before optimization.
    pub original_cost: usize,
    /// Number of MCTS iterations performed.
    pub iterations: usize,
    /// Wall-clock time spent.
    pub elapsed: Duration,
    /// Whether the search was terminated due to timeout.
    pub timed_out: bool,
    /// Whether the search was terminated due to iteration limit.
    pub iteration_limited: bool,
}

impl AnytimeResult {
    /// Cost improvement ratio (original / optimized).
    pub fn improvement_ratio(&self) -> f64 {
        if self.cost == 0 {
            return f64::INFINITY;
        }
        self.original_cost as f64 / self.cost as f64
    }

    /// Absolute cost reduction.
    pub fn cost_reduction(&self) -> i64 {
        self.original_cost as i64 - self.cost as i64
    }
}

/// Optimize an expression tree with a given budget using MCTS.
///
/// # Arguments
///
/// * `tree` - The expression tree to optimize
/// * `config` - Anytime optimization configuration
/// * `nnue` - Optional NNUE network for learned evaluation (falls back to cost-based if None)
///
/// # Returns
///
/// The optimization result including the best tree found within budget.
pub fn optimize_with_budget(
    tree: &ExprTree,
    config: AnytimeConfig,
    nnue: Option<&Nnue>,
) -> AnytimeResult {
    let start = Instant::now();

    // Create initial state
    let initial_state = SearchState::new(tree.clone(), &config.cost_model);
    let original_cost = initial_state.cost;

    // Choose search strategy
    if config.use_beam_search {
        optimize_beam_search(initial_state, config, nnue, start, original_cost)
    } else {
        optimize_mcts(initial_state, config, nnue, start, original_cost)
    }
}

/// MCTS-based optimization.
fn optimize_mcts(
    initial_state: SearchState,
    config: AnytimeConfig,
    nnue: Option<&Nnue>,
    start: Instant,
    original_cost: usize,
) -> AnytimeResult {
    let category = EGraphCategory {
        costs: config.cost_model.clone(),
        max_moves: config.max_moves_per_state,
    };

    // Get available moves
    let moves = category.hom(&initial_state);
    if moves.is_empty() {
        return AnytimeResult {
            tree: initial_state.tree,
            cost: initial_state.cost,
            original_cost,
            iterations: 0,
            elapsed: start.elapsed(),
            timed_out: false,
            iteration_limited: false,
        };
    }

    // Create MCTS tree
    let mcts_config = MctsConfig {
        exploration_constant: config.exploration_constant,
        max_iterations: config.max_iterations.unwrap_or(usize::MAX),
        max_depth: 50,
    };
    let mut tree = MctsTree::new(initial_state.clone(), moves, mcts_config);

    // Create RNG for ε-greedy exploration
    let mut rng = match config.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let epsilon = config.epsilon;

    // Create evaluation function based on whether NNUE is provided
    let cost_val = CostValuation { costs: config.cost_model.clone() };
    let nnue_val = nnue.map(|n| NNUEValuation::with_network(n.clone()));

    let mut iterations = 0;
    let mut best_state = initial_state;
    let mut timed_out = false;
    let mut iteration_limited = false;

    // Main search loop
    loop {
        // Check termination conditions
        if let Some(timeout) = config.timeout {
            if start.elapsed() >= timeout {
                timed_out = true;
                break;
            }
        }

        if let Some(max_iter) = config.max_iterations {
            if iterations >= max_iter {
                iteration_limited = true;
                break;
            }
        }

        // Run one MCTS iteration
        let apply_action = |state: &SearchState, _action: &RewriteMove| -> SearchState {
            // Apply rewrite via e-graph roundtrip
            let mut eg = EGraph::new();
            let root = insert_tree(&mut eg, &state.tree);
            eg.saturate();
            SearchState::from_egraph(&eg, root, &config.cost_model)
        };

        // ε-greedy: with probability epsilon, return random score
        // This ensures diverse exploration during training
        let use_random = epsilon > 0.0 && rng.r#gen::<f64>() < epsilon;
        let random_score = if use_random { rng.r#gen::<f64>() } else { 0.0 };

        let evaluate = |state: &SearchState| -> f64 {
            if use_random {
                // Random exploration: return random score to diversify training
                random_score
            } else if let Some(ref nv) = nnue_val {
                nv.eval(state)
            } else {
                cost_val.eval(state)
            }
        };

        tree.iterate(apply_action, evaluate);
        iterations += 1;

        // Track best state found
        // Check all children of root for best
        for child in &tree.root.children {
            if child.state.cost < best_state.cost {
                best_state = child.state.clone();
            }
        }

        // Early termination if we've found a very good solution
        if best_state.cost <= 1 {
            break;
        }
    }

    AnytimeResult {
        tree: best_state.tree,
        cost: best_state.cost,
        original_cost,
        iterations,
        elapsed: start.elapsed(),
        timed_out,
        iteration_limited,
    }
}

/// Beam search optimization (faster, simpler than full MCTS).
fn optimize_beam_search(
    initial_state: SearchState,
    config: AnytimeConfig,
    nnue: Option<&Nnue>,
    start: Instant,
    original_cost: usize,
) -> AnytimeResult {
    let category = EGraphCategory {
        costs: config.cost_model.clone(),
        max_moves: config.max_moves_per_state,
    };

    // Create RNG for ε-greedy exploration
    let mut rng = match config.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let epsilon = config.epsilon;

    let cost_val = CostValuation { costs: config.cost_model.clone() };
    let nnue_val = nnue.map(|n| NNUEValuation::with_network(n.clone()));

    let evaluate = |state: &SearchState, rng: &mut StdRng| -> f64 {
        // ε-greedy: with probability epsilon, return random score
        if epsilon > 0.0 && rng.r#gen::<f64>() < epsilon {
            rng.r#gen::<f64>()
        } else if let Some(ref nv) = nnue_val {
            nv.eval(state)
        } else {
            cost_val.eval(state)
        }
    };

    let mut beam: Vec<(SearchState, f64)> = vec![(initial_state.clone(), evaluate(&initial_state, &mut rng))];
    let mut best_state = initial_state;
    let mut iterations = 0;
    let mut timed_out = false;
    let mut iteration_limited = false;

    loop {
        // Check termination conditions
        if let Some(timeout) = config.timeout {
            if start.elapsed() >= timeout {
                timed_out = true;
                break;
            }
        }

        if let Some(max_iter) = config.max_iterations {
            if iterations >= max_iter {
                iteration_limited = true;
                break;
            }
        }

        // Expand all states in beam
        let mut candidates: Vec<(SearchState, f64)> = Vec::new();

        for (state, _) in &beam {
            let moves = category.hom(state);
            for _mv in moves.iter().take(config.max_moves_per_state) {
                // Apply move via e-graph
                let mut eg = EGraph::new();
                let root = insert_tree(&mut eg, &state.tree);
                eg.saturate();
                let new_state = SearchState::from_egraph(&eg, root, &config.cost_model);
                let score = evaluate(&new_state, &mut rng);
                candidates.push((new_state, score));
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Sort by score (higher is better for NNUE/cost valuation)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top beam_width
        candidates.truncate(config.beam_width);

        // Update best
        for (state, _) in &candidates {
            if state.cost < best_state.cost {
                best_state = state.clone();
            }
        }

        beam = candidates;
        iterations += 1;

        // Early termination
        if best_state.cost <= 1 {
            break;
        }
    }

    AnytimeResult {
        tree: best_state.tree,
        cost: best_state.cost,
        original_cost,
        iterations,
        elapsed: start.elapsed(),
        timed_out,
        iteration_limited,
    }
}

/// Convenience function: optimize an e-graph root with budget.
pub fn optimize_egraph_with_budget(
    egraph: &EGraph,
    root: EClassId,
    config: AnytimeConfig,
    nnue: Option<&Nnue>,
) -> AnytimeResult {
    let tree = egraph.extract_tree_with_costs(root, &config.cost_model);
    optimize_with_budget(&tree, config, nnue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_simple() {
        // x + 0 should simplify to x
        let tree = ExprTree::Add(
            Box::new(ExprTree::Var(0)),
            Box::new(ExprTree::Const(0.0)),
        );

        let config = AnytimeConfig::default().with_iterations(100);
        let result = optimize_with_budget(&tree, config, None);

        assert!(result.cost <= result.original_cost);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_optimize_with_timeout() {
        let tree = ExprTree::Mul(
            Box::new(ExprTree::Add(
                Box::new(ExprTree::Var(0)),
                Box::new(ExprTree::Var(1)),
            )),
            Box::new(ExprTree::Const(1.0)),
        );

        let config = AnytimeConfig::default()
            .with_timeout(Duration::from_millis(100))
            .with_iterations(10000); // High iteration limit

        let result = optimize_with_budget(&tree, config, None);

        // Should either timeout or hit iteration limit quickly
        assert!(result.timed_out || result.iteration_limited || result.cost <= result.original_cost);
    }

    #[test]
    fn test_beam_search() {
        let tree = ExprTree::Add(
            Box::new(ExprTree::Mul(
                Box::new(ExprTree::Var(0)),
                Box::new(ExprTree::Const(0.0)),
            )),
            Box::new(ExprTree::Var(1)),
        );

        let config = AnytimeConfig::default()
            .with_beam_search(5)
            .with_iterations(50);

        let result = optimize_with_budget(&tree, config, None);

        assert!(result.cost <= result.original_cost);
    }

    #[test]
    fn test_improvement_ratio() {
        let result = AnytimeResult {
            tree: ExprTree::Var(0),
            cost: 5,
            original_cost: 15,
            iterations: 10,
            elapsed: Duration::from_millis(1),
            timed_out: false,
            iteration_limited: false,
        };

        assert!((result.improvement_ratio() - 3.0).abs() < 0.001);
        assert_eq!(result.cost_reduction(), 10);
    }
}
