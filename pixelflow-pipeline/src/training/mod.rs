//! # Training Infrastructure for NNUE
//!
//! This module provides training infrastructure for NNUE-based cost prediction:
//!
//! ## Submodules
//!
//! - [`data_gen`]: Training data generation (expression generation, benchmarking)
//! - [`features`]: Feature extraction from ExprTree (HalfEP, dense ILP features)
//! - [`backprop`]: Forward/backward passes for NNUE training
//! - [`egraph`]: E-graph specific training (The Judge, The Guide)
//! - [`factored`]: **NEW** O(ops) factored embedding training
//!
//! ## Feature Flags
//!
//! - `training`: Basic data generation (requires `std`)
//! - `egraph-training`: Full e-graph training pipeline (requires `pixelflow-search`, `pixelflow-nnue`)

mod data_gen;

pub use data_gen::*;

// E-graph training modules (require pixelflow-search and pixelflow-nnue)
#[cfg(feature = "training")]
pub mod features;

#[cfg(feature = "training")]
pub mod backprop;

#[cfg(feature = "training")]
pub mod egraph;

// Expression parsing/serialization utilities (used by self-play, bench corpus, etc.)
#[cfg(feature = "training")]
pub mod factored;

// Unified self-play trajectory payload structs (Rust↔Python IPC)
#[cfg(feature = "training")]
pub mod unified;

// Full analytical backward pass through entire ExprNnue forward path
#[cfg(feature = "training")]
pub mod unified_backward;

// Self-play trajectory generator (GENERATE phase of unified training loop)
#[cfg(feature = "training")]
pub mod self_play;

// ES-guided adaptive expression generation
#[cfg(feature = "training")]
pub mod gen_es;

// Persistent file-backed replay buffer
#[cfg(feature = "training")]
pub mod replay;
