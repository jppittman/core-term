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

mod algebra;
mod cost;
mod extract;
mod graph;
mod node;
mod rewrite;
mod rules;

// Re-export public API
pub use cost::CostModel;
pub use extract::ExprTree;
pub use graph::EGraph;
pub use node::{EClassId, ENode, Op};
