//! Rewrite rule infrastructure.

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Op;

/// Actions that a rewrite rule can produce.
#[derive(Debug, Clone)]
pub enum RewriteAction {
    /// Union this e-class with another
    Union(EClassId),
    /// Create a new e-node and union with it
    Create(ENode),
    /// Distribute: A * (B + C) -> A*B + A*C
    Distribute {
        outer: &'static dyn Op,
        inner: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// Factor: A*B + A*C -> A * (B + C)
    Factor {
        outer: &'static dyn Op,
        inner: &'static dyn Op,
        common: EClassId,
        unique_l: EClassId,
        unique_r: EClassId,
    },
    /// Canonicalize: Sub(a,b) -> Add(a, Neg(b))
    Canonicalize {
        target: &'static dyn Op,
        inverse: &'static dyn Op,
        a: EClassId,
        b: EClassId,
    },
    /// Associate: (a op b) op c -> a op (b op c)
    Associate {
        op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },

    // ========================================================================
    // Calculus Actions
    // ========================================================================

    /// Binary op on two new nodes: result = lhs_node op rhs_node
    BinaryOp {
        op: &'static dyn Op,
        lhs_node: ENode,
        rhs_node: ENode,
    },

    /// Unary op on a new node: result = op(inner_node)
    UnaryOp {
        op: &'static dyn Op,
        inner_node: ENode,
    },

    /// Product rule: D[f * g] → D[f] * g + f * D[g]
    ProductRule {
        diff_op: &'static dyn Op,
        f: EClassId,
        g: EClassId,
    },

    /// Quotient rule: D[f / g] → (D[f] * g - f * D[g]) / (g * g)
    QuotientRule {
        diff_op: &'static dyn Op,
        f: EClassId,
        g: EClassId,
    },

    /// Chain rule for sqrt: D[sqrt(f)] → D[f] / (2 * sqrt(f))
    ChainSqrt {
        diff_op: &'static dyn Op,
        f: EClassId,
    },

    /// Chain rule for sin: D[sin(f)] → cos(f) * D[f]
    ChainSin {
        diff_op: &'static dyn Op,
        f: EClassId,
    },

    /// Chain rule for cos: D[cos(f)] → -sin(f) * D[f]
    ChainCos {
        diff_op: &'static dyn Op,
        f: EClassId,
    },

    /// Chain rule for exp: D[exp(f)] → exp(f) * D[f]
    ChainExp {
        diff_op: &'static dyn Op,
        f: EClassId,
    },

    /// Chain rule for ln: D[ln(f)] → D[f] / f
    ChainLn {
        diff_op: &'static dyn Op,
        f: EClassId,
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
