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
    /// OddParity: Op(neg(x)) -> neg(Op(x))
    /// Creates Op(inner), then wraps in Neg.
    OddParity {
        func: &'static dyn Op,
        inner: EClassId,
    },
    /// AngleAddition: sin(a+b) -> sin(a)cos(b) + cos(a)sin(b)
    /// or cos(a+b) -> cos(a)cos(b) - sin(a)sin(b)
    AngleAddition {
        term1_op1: &'static dyn Op,
        term1_op2: &'static dyn Op,
        term2_op1: &'static dyn Op,
        term2_op2: &'static dyn Op,
        term2_sign: crate::math::trig::Sign,
        a: EClassId,
        b: EClassId,
    },
    /// Homomorphism: f(a ⊕ b) -> f(a) ⊗ f(b)
    /// e.g., exp(a + b) -> exp(a) * exp(b)
    Homomorphism {
        func: &'static dyn Op,
        target_op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
    },
    /// PowerCombine: x^a * x^b -> x^(a+b)
    PowerCombine {
        base: EClassId,
        exp_a: EClassId,
        exp_b: EClassId,
    },
    /// ReverseAngleAddition: sin(a)cos(b) + cos(a)sin(b) -> sin(a + b)
    /// (The inverse of angle addition, enables double angle discovery)
    ReverseAngleAddition {
        trig_op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
    },
    /// HalfAngleProduct: sin(x) * cos(x) -> sin(x + x) / 2
    /// Derived from sin(2x) = 2*sin(x)*cos(x)
    HalfAngleProduct {
        x: EClassId,
    },
    /// Doubling: a + a -> 2 * a
    Doubling {
        a: EClassId,
    },
    /// Halving: 2 * a -> a + a (reverse of doubling)
    Halving {
        a: EClassId,
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
