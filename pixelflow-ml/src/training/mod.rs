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
//!
//! ## Feature Flags
//!
//! - `training`: Basic data generation (requires `std`)
//! - `egraph-training`: Full e-graph training pipeline (requires `pixelflow-search`, `pixelflow-nnue`)

mod data_gen;

pub use data_gen::*;

// E-graph training modules (require pixelflow-search and pixelflow-nnue)
#[cfg(feature = "egraph-training")]
pub mod features;

#[cfg(feature = "egraph-training")]
pub mod backprop;

#[cfg(feature = "egraph-training")]
pub mod egraph;
