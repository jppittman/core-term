//! Evaluator implementations.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::nnue::{ExprNnue, Expr, EMBED_DIM};
use crate::nnue::factored::EdgeAccumulator;
use super::{Evaluator, StateEmbedding};

/// Neural network value head evaluator.
///
/// This is the fast path - uses the trained value head to predict cost.
/// The value head was trained via benchmarks in a separate offline loop.
pub struct NeuralEvaluator {
    model: Arc<ExprNnue>,
}

impl NeuralEvaluator {
    pub fn new(model: Arc<ExprNnue>) -> Self {
        Self { model }
    }
}

impl Evaluator for NeuralEvaluator {
    fn evaluate(&self, expr: &Expr) -> f32 {
        let acc = EdgeAccumulator::from_expr(expr, &self.model.embeddings);
        self.model.predict_cost_from_accumulator(&acc)
    }

    fn evaluate_from_state(&self, state: &StateEmbedding) -> f32 {
        // If we have value_pred cached, use it
        if let Some(v) = state.value_pred {
            return v;
        }
        // Otherwise compute from accumulator
        let mut acc = EdgeAccumulator::new();
        acc.values = state.accumulator;
        self.model.predict_cost_from_accumulator(&acc)
    }

    fn name(&self) -> &'static str {
        "NeuralValue"
    }
}

/// Oracle evaluator - runs expensive search to get "ground truth".
///
/// Note: This still uses the neural value head, just with more compute budget.
/// The actual ground truth (benchmark) is only used to train the value head.
///
/// Used for:
/// - REINFORCE reward signal (compare student vs oracle)
/// - Curriculum learning ground truth (saturate small kernels)
pub struct OracleEvaluator {
    model: Arc<ExprNnue>,
    /// Maximum e-graph classes before stopping.
    pub max_classes: usize,
    /// Maximum search epochs.
    pub max_epochs: usize,
}

impl OracleEvaluator {
    pub fn new(model: Arc<ExprNnue>, max_classes: usize, max_epochs: usize) -> Self {
        Self { model, max_classes, max_epochs }
    }
}

impl Evaluator for OracleEvaluator {
    fn evaluate(&self, expr: &Expr) -> f32 {
        // TODO: Run actual e-graph search with saturation
        // For now, just use value head (same as NeuralEvaluator)
        // The difference is this would run GuidedSearch with all rules enabled
        let acc = EdgeAccumulator::from_expr(expr, &self.model.embeddings);
        self.model.predict_cost_from_accumulator(&acc)
    }

    fn name(&self) -> &'static str {
        "Oracle"
    }
}

/// Evaluator that returns a fixed cost (for testing).
pub struct FixedEvaluator {
    cost: f32,
}

impl FixedEvaluator {
    pub fn new(cost: f32) -> Self {
        Self { cost }
    }
}

impl Evaluator for FixedEvaluator {
    fn evaluate(&self, _expr: &Expr) -> f32 {
        self.cost
    }

    fn evaluate_from_state(&self, _state: &StateEmbedding) -> f32 {
        self.cost
    }

    fn name(&self) -> &'static str {
        "Fixed"
    }
}
