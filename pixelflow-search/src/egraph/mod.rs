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
//! - [`nnue_adapter`]: NNUE integration for learned cost prediction
//! - [`guided_search`]: Guided search with learned rule filtering
//!
//! Mathematical rewrite rules are now in the [`crate::math`] module.

pub mod codegen;
mod cost;
mod deps;
mod extract;
mod graph;
pub mod guided_search;  // Guided search with learned rule filtering
mod node;
pub mod nnue_adapter;
pub mod nnue_optimize;  // NNUE-guided expression optimization
pub mod ops;
pub mod rewrite;
pub mod saturate;

// Re-export public API
pub use cost::{CostFunction, CostModel};
pub use deps::{Deps, DepsAnalysis};
pub use extract::{ExprTree, ExtractedDAG, IncrementalExtractor, Leaf, extract_dag};
pub use graph::{ApplyResult, EGraph, RewriteTarget};
pub use node::{EClassId, ENode};
pub use ops::Op;
pub use rewrite::{Rewrite, RewriteAction};
pub use saturate::{SaturationResult, saturate_with_budget, achievable_cost_within_budget};
pub use guided_search::{
    GuidedSearch, GuidedSearchResult, RuleStats, EpochRecord, RuleRecord,
    UnifiedMaskSearchResult, UnifiedMaskEpochRecord, UnifiedPairRecord,
};
pub use nnue_optimize::{NnueOptimizer, OptimizeConfig, OptimizeResult};

// Re-export NNUE adapter types
pub use nnue_adapter::predict_tree_cost;
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
    FunctionInverse, Homomorphism, ExpLn, Exp2Log2,
    exp_rules,
    // Fusion
    FmaFusion, RecipSqrt, fusion_rules,
    // Rule collections
    all_math_rules, core_rules, transcendental_rules,
};

/// All rewrite rules: 40 math + 2 fusion = 42 total.
///
/// This is the complete rule set for optimization, training, and production.
pub fn all_rules() -> Vec<Box<dyn Rewrite>> {
    crate::math::all_rules()
}

/// Build [`RuleTemplates`] from all registered rules.
///
/// Collects LHS/RHS expression templates from every rule that provides them.
/// Rules without templates (returning `None`) get empty slots.
///
/// # Panics
///
/// Panics if `all_rules()` returns an empty list (should never happen).
#[must_use]
pub fn collect_rule_templates() -> crate::nnue::RuleTemplates {
    let rules = all_rules();
    assert!(!rules.is_empty(), "collect_rule_templates: all_rules() returned 0 rules");

    let mut templates = crate::nnue::RuleTemplates::with_capacity(rules.len());

    for (idx, rule) in rules.iter().enumerate() {
        if let (Some(lhs), Some(rhs)) = (rule.lhs_template(), rule.rhs_template()) {
            templates.set(idx, lhs, rhs);
        }
        // Rules without templates get None slots (handled by RuleTemplates)
    }

    templates
}
