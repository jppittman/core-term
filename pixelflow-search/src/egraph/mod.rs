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
//! - [`extract`]: Expression tree extraction, including DAG-aware extraction
//! - [`graph`]: The EGraph itself
//! - [`deps`]: Dependency analysis for uniform hoisting
//! - [`codegen`]: Code generation from extracted expressions (tree & DAG)
//! - [`best_first`]: Chess-style time-controlled best-first search
//! - [`nnue_adapter`]: NNUE integration for learned cost/priority prediction
//!
//! Mathematical rewrite rules are now in the [`crate::math`] module.

pub mod best_first;
pub mod codegen;
mod cost;
mod deps;
mod extract;
mod graph;
pub mod guided_search;  // Guided search with learned rule filtering
mod node;
pub mod nnue_adapter;
pub mod nnue_cache;     // NNUE metadata cache for e-graph nodes
pub mod nnue_optimize;  // NNUE-guided expression optimization
pub mod ops;
pub mod rewrite;
pub mod saturate;

// Re-export public API
pub use cost::{CostFunction, CostModel};
pub use deps::{Deps, DepsAnalysis};
pub use extract::{ExprTree, ExtractedDAG, Leaf, extract_dag};
pub use graph::{EGraph, RewriteTarget};
pub use node::{EClassId, ENode};
pub use ops::Op;
pub use rewrite::{Rewrite, RewriteAction};
pub use saturate::{SaturationResult, saturate_with_budget, achievable_cost_within_budget};
pub use best_first::{BestFirstPlanner, BestFirstConfig, BestFirstResult, BestFirstContext, StopReason, TrajectoryStep, SearchTrajectory};
pub use guided_search::{
    GuidedSearch, GuidedSearchResult, RuleFeatures, RuleStats, EpochRecord, RuleRecord,
    UnifiedMaskSearchResult, UnifiedMaskEpochRecord, UnifiedPairRecord,
};
pub use nnue_optimize::{NnueOptimizer, OptimizeConfig, OptimizeResult};
pub use nnue_cache::{NnueCache, ENodeMetadata};

// Re-export NNUE adapter types
pub use nnue_adapter::{NnueCostAdapter, predict_tree_cost, predict_tree_priority};
pub use nnue_adapter::{expr_tree_to_nnue, expr_to_egraph, eclass_to_expr};
pub use nnue_adapter::{extract_neural, extract_beam};

// Re-export rule types from math module for backward compatibility
pub use crate::math::{
    // Algebra
    InversePair, AddNeg, MulRecip,
    Commutative, Identity, Annihilator, Associative,
    algebra_rules, inverse_pair_rules, basic_algebra_rules,
    // Parity
    Parity, ParityKind,
    parity_rules,
    // Trig
    AngleAddition, Sign,
    trig_rules,
    // Exp
    FunctionInverse, Homomorphism,
    exp_rules,
    // All rules
    all_math_rules, core_rules, transcendental_rules,
};

/// All mathematical rewrite rules (no CPU-specific fusion).
///
/// For full optimization including FMA/rsqrt fusion, use
/// `pixelflow_compiler::optimize::standard_rules()` which combines
/// math rules with compiler-specific instruction selection.
pub fn all_rules() -> Vec<Box<dyn Rewrite>> {
    crate::math::all_math_rules()
}
