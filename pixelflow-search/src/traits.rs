//! # Core Search Traits
//!
//! Abstractions for generic graph search problems.

use alloc::vec::Vec;

/// A state in the search space.
pub trait State: Clone {
    /// The type of move that transitions between states.
    type Move: Clone;

    /// Apply a move to this state, returning the new state.
    /// Returns None if the move is invalid.
    fn apply_move(&self, mov: &Self::Move) -> Option<Self>;

    /// Generate all legal moves from this state.
    fn legal_moves(&self) -> Vec<Self::Move>;
    
    /// Check if this is a terminal state.
    fn is_terminal(&self) -> bool;
}

/// A heuristic or learned evaluator for states.
pub trait Evaluator<S: State> {
    /// Score type (e.g., f32 cost, i32 centipawns).
    type Score: PartialOrd + Copy;

    /// Evaluate the state.
    fn evaluate(&self, state: &S) -> Self::Score;
}

/// A feature extractor for neural evaluation.
///
/// Maps a state to a sparse feature vector.
pub trait FeatureExtractor<S: State> {
    /// The maximum index of a feature.
    fn feature_space_size(&self) -> usize;

    /// Extract sparse active feature indices from the state.
    fn extract(&self, state: &S) -> Vec<usize>;
}
