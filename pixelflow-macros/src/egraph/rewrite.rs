//! Rewrite rule infrastructure.

use super::graph::EGraph;
use super::node::{EClassId, ENode, Op};

/// Actions that a rewrite rule can produce.
#[derive(Debug, Clone)]
pub enum RewriteAction {
    /// Union this e-class with another
    Union(EClassId),
    /// Create a new e-node and union with it
    Create(ENode),
    /// Distribute: A * (B + C) -> A*B + A*C
    Distribute {
        outer: Op,
        inner: Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// Factor: A*B + A*C -> A * (B + C)
    Factor {
        outer: Op,
        inner: Op,
        common: EClassId,
        unique_l: EClassId,
        unique_r: EClassId,
    },
    /// Canonicalize: Sub(a,b) -> Add(a, Neg(b))
    Canonicalize {
        target: Op,
        inverse: Op,
        a: EClassId,
        b: EClassId,
    },
    /// Associate: (a op b) op c -> a op (b op c)
    Associate {
        op: Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
}

/// A rewrite rule that can be applied to e-graph nodes.
pub trait Rewrite {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Try to apply this rule to a node in an e-class.
    /// Returns `Some(action)` if the rule matches.
    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<RewriteAction>;
}
