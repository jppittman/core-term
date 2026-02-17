//! E-graph based expression optimizer.
//!
//! An e-graph (equality graph) compactly represents many equivalent expressions.
//! We use equality saturation to find the cheapest form of mathematical expressions.
//!
//! # Module Structure
//!
//! - [`node`]: Core data structures (EClassId, Op, ENode)
//! - [`cost`]: Cost model for extraction
//! - [`rewrite`]: Rewrite rule infrastructure
//! - [`algebra`]: Type-based algebraic relationships (InversePair)
//! - [`rules`]: Concrete rewrite rules
//! - [`extract`]: Expression tree extraction, including DAG-aware extraction
//! - [`graph`]: The EGraph itself
//! - [`deps`]: Dependency analysis for uniform hoisting
//! - [`codegen`]: Code generation from extracted expressions (tree & DAG)

mod algebra;
pub mod best_first;
pub mod codegen;
mod cost;
mod deps;
mod extract;
mod graph;
pub mod guided;
pub mod nnue_adapter;
mod node;
pub mod ops;
pub mod rewrite; // Made public for domain-specific rules in pixelflow-macros
mod rules;
pub mod saturate;

// Re-export public API
pub use best_first::{BestFirstConfig, BestFirstContext, BestFirstPlanner, BestFirstResult};
pub use cost::{CostFunction, CostModel};
pub use deps::{Deps, DepsAnalysis};
pub use extract::{ExprTree, ExtractedDAG, Leaf, extract_dag};
pub use graph::{EGraph, RewriteTarget};
pub use guided::{
    ActionRecord, GuidedAction, GuidedConfig, GuidedMcts, GuidedResult, GuidedState, GuidedStats,
    guided_optimize,
};
pub use node::{EClassId, ENode};
pub use ops::Op;
pub use rewrite::{Rewrite, RewriteAction}; // Export for domain-specific rules
pub use saturate::{SaturationResult, achievable_cost_within_budget, saturate_with_budget};
