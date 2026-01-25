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
mod cost;
mod deps;
mod extract;
mod graph;
mod node;
mod rewrite;
pub mod rules;

// Re-export public API
pub use cost::CostModel;
pub use deps::{Deps, DepsAnalysis};
pub use extract::ExprTree;
pub use graph::EGraph;
pub use node::{EClassId, ENode, Op};
