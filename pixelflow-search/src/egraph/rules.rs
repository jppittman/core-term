//! Non-InversePair rewrite rules.
//!
//! These rules use Op directly rather than being derived from InversePair.

use super::graph::EGraph;
use super::node::{EClassId, ENode, Op};
use super::rewrite::{Rewrite, RewriteAction};

/// Associativity: (a op b) op c → a op (b op c)
#[allow(dead_code)]
pub struct Associative {
    op: Op,
}

impl Associative {
    #[allow(dead_code)]
    pub fn new(op: Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Associative {
    fn name(&self) -> &str { "associative" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(self.op) { return None; }
        let (left, right) = node.binary_operands()?;

        for child in egraph.nodes(left) {
            if child.op() == Some(self.op) {
                if let Some((a, b)) = child.binary_operands() {
                    return Some(RewriteAction::Associate {
                        op: self.op,
                        a,
                        b,
                        c: right,
                    });
                }
            }
        }
        None
    }
}

/// Commutativity: a op b → b op a
pub struct Commutative {
    op: Op,
}

impl Commutative {
    pub fn new(op: Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Commutative {
    fn name(&self) -> &str { "commutative" }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(self.op) { return None; }
        let (a, b) = node.binary_operands()?;
        if a == b { return None; }

        let swapped = self.op.make_binary(b, a)?;
        Some(RewriteAction::Create(swapped))
    }
}

/// Distributivity: A * (B + C) → A*B + A*C
pub struct Distributive {
    outer: Op,
    inner: Op,
}

impl Distributive {
    pub fn new(outer: Op, inner: Op) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Distributive {
    fn name(&self) -> &str { "distribute" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(self.outer) { return None; }
        let (a, other) = node.binary_operands()?;

        for child_node in egraph.nodes(other) {
            if child_node.op() == Some(self.inner) {
                if let Some((b, c)) = child_node.binary_operands() {
                    return Some(RewriteAction::Distribute {
                        outer: self.outer,
                        inner: self.inner,
                        a,
                        b,
                        c,
                    });
                }
            }
        }
        None
    }
}

/// Factoring: A*B + A*C → A * (B + C)
pub struct Factor {
    outer: Op,
    inner: Op,
}

impl Factor {
    pub fn new(outer: Op, inner: Op) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Factor {
    fn name(&self) -> &str { "factor" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(self.outer) { return None; }
        let (left, right) = node.binary_operands()?;

        for l_node in egraph.nodes(left) {
            if l_node.op() != Some(self.inner) { continue; }
            let (la, lb) = l_node.binary_operands()?;

            for r_node in egraph.nodes(right) {
                if r_node.op() != Some(self.inner) { continue; }
                let (ra, rb) = r_node.binary_operands()?;

                let (common, unique_l, unique_r) = if egraph.find(la) == egraph.find(ra) {
                    (la, lb, rb)
                } else if egraph.find(la) == egraph.find(rb) {
                    (la, lb, ra)
                } else if egraph.find(lb) == egraph.find(ra) {
                    (lb, la, rb)
                } else if egraph.find(lb) == egraph.find(rb) {
                    (lb, la, ra)
                } else {
                    continue;
                };

                return Some(RewriteAction::Factor {
                    outer: self.outer,
                    inner: self.inner,
                    common,
                    unique_l,
                    unique_r,
                });
            }
        }
        None
    }
}

/// Identity: x op identity → x
pub struct Identity {
    op: Op,
}

impl Identity {
    pub fn new(op: Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Identity {
    fn name(&self) -> &str { "identity" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(self.op) { return None; }
        let id_val = self.op.identity()?;
        let (a, b) = node.binary_operands()?;

        if egraph.contains_const(b, id_val) { return Some(RewriteAction::Union(a)); }
        if egraph.contains_const(a, id_val) { return Some(RewriteAction::Union(b)); }
        None
    }
}

/// Annihilator: x op annihilator → annihilator
pub struct Annihilator {
    op: Op,
}

impl Annihilator {
    pub fn new(op: Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Annihilator {
    fn name(&self) -> &str { "annihilator" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(self.op) { return None; }
        let zero_val = self.op.annihilator()?;
        let (a, b) = node.binary_operands()?;

        if egraph.contains_const(a, zero_val) || egraph.contains_const(b, zero_val) {
            return Some(RewriteAction::Create(ENode::constant(zero_val)));
        }
        None
    }
}

/// Idempotence: x op x → x
pub struct Idempotent {
    op: Op,
}

impl Idempotent {
    pub fn new(op: Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Idempotent {
    fn name(&self) -> &str { "idempotent" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(self.op) { return None; }
        if !self.op.is_idempotent() { return None; }
        let (a, b) = node.binary_operands()?;

        if egraph.find(a) == egraph.find(b) {
            return Some(RewriteAction::Union(a));
        }
        None
    }
}

/// RecipSqrt: 1/sqrt(x) → rsqrt(x)
pub struct RecipSqrt;

impl Rewrite for RecipSqrt {
    fn name(&self) -> &str { "recip-sqrt" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if let ENode::Recip(a) = node {
            for child in egraph.nodes(*a) {
                if let ENode::Sqrt(inner) = child {
                    return Some(RewriteAction::Create(ENode::Rsqrt(*inner)));
                }
            }
        }
        None
    }
}

/// FmaFusion: a * b + c → MulAdd(a, b, c)
///
/// Fused multiply-add is typically a single instruction on modern CPUs.
/// This rule adds the fused form to the e-graph; extraction cost model
/// determines whether it's actually used.
pub struct FmaFusion;

impl Rewrite for FmaFusion {
    fn name(&self) -> &str { "fma-fusion" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match Add(Mul(a, b), c) → MulAdd(a, b, c)
        if let ENode::Add(left, right) = node {
            // Check if left is a Mul
            for left_node in egraph.nodes(*left) {
                if let ENode::Mul(a, b) = left_node {
                    return Some(RewriteAction::Create(ENode::MulAdd(*a, *b, *right)));
                }
            }
            // Check if right is a Mul (commutativity)
            for right_node in egraph.nodes(*right) {
                if let ENode::Mul(a, b) = right_node {
                    return Some(RewriteAction::Create(ENode::MulAdd(*a, *b, *left)));
                }
            }
        }
        None
    }
}

