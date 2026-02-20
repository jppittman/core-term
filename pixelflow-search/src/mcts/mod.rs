//! # MCTS for E-graph Optimization
//!
//! AlphaZero-style Monte Carlo Tree Search for guiding e-graph saturation.
//!
//! ## Key Insight
//!
//! The accumulator enables cheap MCTS simulation without e-graph cloning.
//! We can incrementally update the accumulator as rules are applied,
//! then use the value head to evaluate the resulting state.
//!
//! ## AlphaZero Mapping
//!
//! | Chess/Go | E-graph |
//! |----------|---------|
//! | Board position | Accumulator state |
//! | Legal moves | (class, rule) pairs |
//! | Policy head | Mask scores |
//! | Value head | Cost prediction |
//! | Game outcome | Final extraction cost |
//! | Move | Apply rule at class |
//!
//! ## Architecture
//!
//! - **MctsNode**: Tree node with visit counts, Q-values, and action priors
//! - **MctsState**: Incrementally updated accumulator state
//! - **MctsSearch**: UCB selection, expansion, and backpropagation

mod node;
mod search;
mod state;

pub use node::{MctsNode, NodeRef};
pub use search::{MctsConfig, MctsSearch, MctsResult};
pub use state::{MctsState, MctsAction, AccumulatorDelta};
