//! Composable search pipeline traits.
//!
//! This module defines the trait boundaries for mix-and-match experimentation
//! with different search strategies, evaluation methods, and training signals.
//!
//! # Architecture
//!
//! ```text
//! DataSource → SearchStrategy → Evaluator → GradientSignal → WeightUpdate
//!     │              │              │              │
//!     │              │              │              └─ REINFORCE, SyntheticOracle, Supervised
//!     │              │              └─ NeuralValue (fast), Oracle (slow, for training oracle)
//!     │              └─ BestFirst, MCTS, Filtered, Exhaustive
//!     └─ Generated, FromFile
//! ```
//!
//! # Key Insight
//!
//! The Oracle (value head) is trained via benchmarks (slow, offline).
//! At runtime, we use the trained neural value head (fast).
//! So `Evaluator::Oracle` and `Evaluator::NeuralValue` are the same network -
//! the distinction is whether we're *training* the oracle or *using* it.

extern crate alloc;

use alloc::vec::Vec;
use alloc::boxed::Box;

use crate::nnue::factored::{EdgeAccumulator, StructuralFeatures, EMBED_DIM};
use crate::nnue::Expr;

// ============================================================================
// Phase 1: Data Source
// ============================================================================

/// Source of expressions to optimize.
pub trait DataSource {
    /// Generate or load the next batch of expressions.
    fn next_batch(&mut self, count: usize) -> Vec<Expr>;

    /// Reset to beginning (for multiple epochs).
    fn reset(&mut self);

    /// Total expressions available (None if infinite/unknown).
    fn len(&self) -> Option<usize>;
}

// ============================================================================
// Phase 2: Search Strategy
// ============================================================================

/// Outcome of a single search run.
#[derive(Clone)]
pub struct SearchResult {
    /// Best expression found.
    pub best_expr: Expr,
    /// Cost of best expression (from evaluator).
    pub best_cost: f32,
    /// Initial cost before optimization.
    pub initial_cost: f32,
    /// Search trajectory for training.
    pub trajectory: Trajectory,
    /// Why search stopped.
    pub stop_reason: StopReason,
}

/// Why search terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Found optimal (saturation).
    Saturated,
    /// Hit expansion/epoch budget.
    BudgetExhausted,
    /// E-graph size limit.
    SizeLimit,
    /// No rules predicted to help.
    NoPredictedProgress,
}

/// Trajectory of decisions for training.
#[derive(Clone, Default)]
pub struct Trajectory {
    /// Sequence of decision points.
    pub steps: Vec<TrajectoryStep>,
}

/// A single decision point in search.
#[derive(Clone)]
pub struct TrajectoryStep {
    /// State embedding at this point.
    pub state: StateEmbedding,
    /// Actions considered with their scores.
    pub actions: Vec<ActionScore>,
    /// Which action was taken (index into actions).
    pub taken: usize,
    /// Value estimate at this state.
    pub value_estimate: f32,
}

/// An action and its score.
#[derive(Clone)]
pub struct ActionScore {
    /// Rule index.
    pub rule_idx: usize,
    /// E-class to apply at (if applicable).
    pub class_idx: Option<usize>,
    /// Score from policy/mask.
    pub score: f32,
    /// Was this action taken?
    pub taken: bool,
}

/// Embedding of search state for neural network.
#[derive(Clone)]
pub struct StateEmbedding {
    /// Accumulator values [64 dims].
    pub accumulator: [f32; 64],
    /// Expression embedding [24 dims] (if computed).
    pub expr_embed: Option<[f32; EMBED_DIM]>,
    /// Value prediction (if computed).
    pub value_pred: Option<f32>,
    /// Mask features [24 dims] (if computed).
    pub mask_features: Option<[f32; EMBED_DIM]>,
}

impl StateEmbedding {
    /// Create from EdgeAccumulator.
    pub fn from_accumulator(acc: &EdgeAccumulator) -> Self {
        Self {
            accumulator: acc.values,
            expr_embed: None,
            value_pred: None,
            mask_features: None,
        }
    }
}

/// Search strategy that explores the rewrite space.
pub trait SearchStrategy {
    /// Run search on an expression, using the evaluator for guidance.
    fn search(
        &mut self,
        expr: &Expr,
        evaluator: &dyn Evaluator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> SearchResult;

    /// Name for logging.
    fn name(&self) -> &'static str;
}

// ============================================================================
// Phase 3: Evaluator
// ============================================================================

/// Evaluates expression quality.
///
/// Two modes:
/// - **NeuralValue**: Fast, uses trained value head
/// - **Oracle**: Expensive search to get "ground truth" (still uses value head,
///   but with more compute budget). Used for REINFORCE reward signal.
///
/// Note: The actual ground truth (benchmark) is only used to train the value head
/// in a separate offline loop. See `train_judge.rs`.
pub trait Evaluator {
    /// Estimate cost of an expression (lower is better).
    fn evaluate(&self, expr: &Expr) -> f32;

    /// Evaluate from pre-computed state embedding (faster if available).
    fn evaluate_from_state(&self, state: &StateEmbedding) -> f32 {
        // Default: can't use state, subclasses override
        let _ = state;
        unimplemented!("evaluate_from_state not available for this evaluator")
    }

    /// Batch evaluation.
    fn evaluate_batch(&self, exprs: &[&Expr]) -> Vec<f32> {
        exprs.iter().map(|e| self.evaluate(e)).collect()
    }

    /// Name for logging.
    fn name(&self) -> &'static str;
}

// ============================================================================
// Phase 4: Gradient Signal
// ============================================================================

/// Computed gradients ready for weight update.
pub struct Gradients {
    /// Gradient for mask MLP weights.
    pub mask_mlp_grad: Option<MaskMlpGrad>,
    /// Gradient for interaction matrix.
    pub interaction_grad: Option<[[f32; EMBED_DIM]; EMBED_DIM]>,
    /// Gradient for rule biases.
    pub rule_bias_grad: Option<Vec<f32>>,
    /// Gradient for value MLP (if training value head).
    pub value_mlp_grad: Option<ValueMlpGrad>,
    /// Loss value for logging.
    pub loss: f32,
}

/// Gradients for mask MLP.
pub struct MaskMlpGrad {
    pub w1: [[f32; 16]; 25],  // MASK_INPUT_DIM x MLP_HIDDEN
    pub b1: [f32; 16],
    pub w2: [[f32; EMBED_DIM]; 16],
    pub b2: [f32; EMBED_DIM],
}

/// Gradients for value MLP.
pub struct ValueMlpGrad {
    pub w1: [[f32; 16]; EMBED_DIM],
    pub b1: [f32; 16],
    pub w2: [f32; 16],
    pub b2: f32,
}

/// Computes gradients from search trajectories.
pub trait GradientSignal {
    /// Compute gradients from a batch of search results.
    ///
    /// # Arguments
    /// * `results` - Search results with trajectories
    /// * `oracle_costs` - Optional oracle costs for REINFORCE (if different from result.best_cost)
    /// * `rule_embeds` - Rule embeddings (frozen, not trained)
    fn compute(
        &mut self,
        results: &[SearchResult],
        oracle_costs: Option<&[f32]>,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Gradients;

    /// Name for logging.
    fn name(&self) -> &'static str;
}

// ============================================================================
// Phase 5: Weight Update (trivial - just SGD for now)
// ============================================================================

/// Applies gradients to model weights.
pub trait WeightUpdate {
    /// Apply gradients with learning rate.
    fn apply(&mut self, grads: &Gradients, lr: f32);
}

// ============================================================================
// Pipeline Composition
// ============================================================================

/// A complete training pipeline.
pub struct Pipeline<D, S, E, G, W> {
    pub data: D,
    pub search: S,
    pub evaluator: E,
    pub gradient: G,
    pub updater: W,
}

impl<D, S, E, G, W> Pipeline<D, S, E, G, W>
where
    D: DataSource,
    S: SearchStrategy,
    E: Evaluator,
    G: GradientSignal,
    W: WeightUpdate,
{
    /// Run one training step.
    ///
    /// Returns (mean_loss, mean_improvement).
    pub fn step(
        &mut self,
        batch_size: usize,
        rule_embeds: &[[f32; EMBED_DIM]],
        lr: f32,
    ) -> (f32, f32) {
        // Phase 1: Get data
        let exprs = self.data.next_batch(batch_size);

        // Phase 2 & 3: Search (uses evaluator internally)
        let results: Vec<SearchResult> = exprs
            .iter()
            .map(|e| self.search.search(e, &self.evaluator, rule_embeds))
            .collect();

        // Compute improvement
        let mean_improvement: f32 = results
            .iter()
            .map(|r| r.initial_cost - r.best_cost)
            .sum::<f32>() / results.len() as f32;

        // Phase 4: Compute gradients
        let grads = self.gradient.compute(&results, None, rule_embeds);
        let loss = grads.loss;

        // Phase 5: Update weights
        self.updater.apply(&grads, lr);

        (loss, mean_improvement)
    }

    /// Run multiple training steps.
    pub fn train(
        &mut self,
        steps: usize,
        batch_size: usize,
        rule_embeds: &[[f32; EMBED_DIM]],
        lr: f32,
    ) -> Vec<(f32, f32)> {
        (0..steps)
            .map(|_| self.step(batch_size, rule_embeds, lr))
            .collect()
    }
}

// ============================================================================
// Module structure
// ============================================================================

pub mod data;      // DataSource implementations
pub mod strategy;  // SearchStrategy implementations
pub mod eval;      // Evaluator implementations
pub mod gradient;  // GradientSignal implementations
pub mod update;    // WeightUpdate implementations
