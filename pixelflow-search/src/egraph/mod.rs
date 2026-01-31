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
//! - [`extract`]: Expression tree for extracted results
//! - [`graph`]: The EGraph itself
//! - [`deps`]: Dependency analysis for uniform hoisting

mod algebra;
pub mod best_first;
pub mod codegen;
mod cost;
mod deps;
mod extract;
mod graph;
pub mod guided;
mod node;
pub mod nnue_adapter;
pub mod ops;
pub mod rewrite;  // Made public for domain-specific rules in pixelflow-macros
mod rules;
pub mod saturate;

// Re-export public API
pub use cost::CostModel;
pub use deps::{Deps, DepsAnalysis};
pub use extract::{ExprTree, Leaf};
pub use graph::{EGraph, RewriteTarget};
pub use guided::{GuidedAction, GuidedResult, GuidedState, GuidedStats, ActionRecord, GuidedConfig, GuidedMcts, guided_optimize};
pub use node::{EClassId, ENode};
pub use ops::Op;
pub use rewrite::{Rewrite, RewriteAction};  // Export for domain-specific rules
pub use saturate::{SaturationResult, saturate_with_budget, achievable_cost_within_budget};
pub use best_first::{BestFirstPlanner, BestFirstConfig, BestFirstResult, BestFirstContext};
