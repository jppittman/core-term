//! MCTS state representation using accumulator.
//!
//! The key insight: we don't need to clone the e-graph for MCTS simulation.
//! Instead, we track the accumulator state and the sequence of applied rules.
//! The accumulator can be incrementally updated and used for value estimation.

extern crate alloc;

use alloc::vec::Vec;
use crate::nnue::factored::{EdgeAccumulator, OpEmbeddings, StructuralFeatures, EMBED_DIM};

/// An action in the MCTS tree: apply a rule to an e-class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MctsAction {
    /// E-class index to apply the rule to.
    pub class_idx: usize,
    /// Rule index to apply.
    pub rule_idx: usize,
}

/// MCTS state: tracks accumulator and applied actions.
///
/// This is a lightweight representation that enables cheap MCTS simulation
/// without cloning the e-graph. The accumulator is incrementally updated
/// as rules are applied.
#[derive(Clone)]
pub struct MctsState {
    /// Current accumulator state (incrementally updated).
    pub accumulator: EdgeAccumulator,

    /// Structural features (may be stale for simulation; updated lazily).
    pub structural: StructuralFeatures,

    /// Sequence of actions taken from root state.
    pub actions: Vec<MctsAction>,

    /// Cached rule embeddings (shared across all states).
    /// Not cloned - use Arc or pass by reference in practice.
    /// TODO: Use for rule embedding lookups during expansion.
    #[allow(dead_code)]
    rule_embed_cache: Option<Vec<[f32; EMBED_DIM]>>,

    /// Current depth in the search tree.
    pub depth: usize,

    /// Whether this state is terminal (no more actions available).
    pub is_terminal: bool,
}

impl MctsState {
    /// Create a new root state from an initial accumulator.
    #[must_use]
    pub fn new(accumulator: EdgeAccumulator, structural: StructuralFeatures) -> Self {
        Self {
            accumulator,
            structural,
            actions: Vec::new(),
            rule_embed_cache: None,
            depth: 0,
            is_terminal: false,
        }
    }

    /// Create a new root state from an expression.
    #[must_use]
    pub fn from_expr(expr: &pixelflow_ir::Expr, embeddings: &OpEmbeddings) -> Self {
        let accumulator = EdgeAccumulator::from_expr(expr, embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        Self::new(accumulator, structural)
    }

    /// Apply an action to the state.
    ///
    /// Updates the accumulator incrementally based on the rule delta.
    /// Returns a new state (self is not modified).
    #[must_use]
    pub fn apply(&self, action: MctsAction, delta: &AccumulatorDelta) -> Self {
        let mut new_state = self.clone();
        new_state.apply_in_place(action, delta);
        new_state
    }

    /// Apply an action in place.
    pub fn apply_in_place(&mut self, action: MctsAction, delta: &AccumulatorDelta) {
        // Update accumulator with rule delta
        delta.apply(&mut self.accumulator);

        // Track action
        self.actions.push(action);
        self.depth += 1;
    }

    /// Undo the last action.
    ///
    /// Restores the accumulator to its previous state.
    pub fn undo(&mut self, delta: &AccumulatorDelta) {
        if self.actions.pop().is_some() {
            delta.undo(&mut self.accumulator);
            self.depth = self.depth.saturating_sub(1);
        }
    }

    /// Get the action sequence from root to this state.
    #[must_use]
    pub fn action_sequence(&self) -> &[MctsAction] {
        &self.actions
    }

    /// Check if this state is at the root.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.actions.is_empty()
    }

    /// Set terminal flag (no more actions available).
    pub fn set_terminal(&mut self) {
        self.is_terminal = true;
    }
}

/// Delta to apply to an accumulator when a rule fires.
///
/// Captures the change in edge embeddings when a rule transforms an e-class.
/// Pre-computed for each rule so we can quickly update accumulators during MCTS.
#[derive(Clone)]
pub struct AccumulatorDelta {
    /// Changes to accumulator values.
    /// `values[i]` is added to `acc.values[i]` on apply, subtracted on undo.
    pub values: [f32; 64], // 2 * K where K = 32
}

impl AccumulatorDelta {
    /// Create a zero delta (no change).
    #[must_use]
    pub fn zero() -> Self {
        Self { values: [0.0; 64] }
    }

    /// Create a delta from removed and added edges.
    ///
    /// When a rule fires, some edges are removed (old expression structure)
    /// and others are added (new expression structure).
    #[must_use]
    pub fn from_edges(
        removed: &EdgeAccumulator,
        added: &EdgeAccumulator,
    ) -> Self {
        let mut values = [0.0; 64];
        for i in 0..64 {
            values[i] = added.values[i] - removed.values[i];
        }
        Self { values }
    }

    /// Apply delta to accumulator.
    pub fn apply(&self, acc: &mut EdgeAccumulator) {
        for i in 0..64 {
            acc.values[i] += self.values[i];
        }
    }

    /// Undo delta from accumulator.
    pub fn undo(&self, acc: &mut EdgeAccumulator) {
        for i in 0..64 {
            acc.values[i] -= self.values[i];
        }
    }

    /// Check if this is a zero delta.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.values.iter().all(|&v| v.abs() < 1e-9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_delta_apply_undo() {
        let mut acc = EdgeAccumulator::new();
        acc.values[0] = 1.0;
        acc.values[32] = 2.0;

        let mut delta = AccumulatorDelta::zero();
        delta.values[0] = 0.5;
        delta.values[32] = -0.5;

        let original = acc.values;

        delta.apply(&mut acc);
        assert!((acc.values[0] - 1.5).abs() < 1e-6);
        assert!((acc.values[32] - 1.5).abs() < 1e-6);

        delta.undo(&mut acc);
        for i in 0..64 {
            assert!(
                (acc.values[i] - original[i]).abs() < 1e-6,
                "Value at {} should be restored",
                i
            );
        }
    }

    #[test]
    fn test_mcts_state_apply_undo() {
        let acc = EdgeAccumulator::new();
        let structural = StructuralFeatures::default();
        let mut state = MctsState::new(acc, structural);

        let action = MctsAction {
            class_idx: 0,
            rule_idx: 5,
        };

        let mut delta = AccumulatorDelta::zero();
        delta.values[0] = 1.0;

        state.apply_in_place(action, &delta);
        assert_eq!(state.depth, 1);
        assert_eq!(state.actions.len(), 1);
        assert!((state.accumulator.values[0] - 1.0).abs() < 1e-6);

        state.undo(&delta);
        assert_eq!(state.depth, 0);
        assert_eq!(state.actions.len(), 0);
        assert!((state.accumulator.values[0] - 0.0).abs() < 1e-6);
    }
}
