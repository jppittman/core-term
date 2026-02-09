//! Non-InversePair rewrite rules.
//!
//! These rules use Op directly rather than being derived from InversePair.

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::{self, Op};
use super::rewrite::{Rewrite, RewriteAction};

/// Associativity: (a op b) op c → a op (b op c)
#[allow(dead_code)]
pub struct Associative {
    op: &'static dyn Op,
}

impl Associative {
    #[allow(dead_code)]
    #[allow(dead_code)]
pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Associative {
    fn name(&self) -> &str {
        "associative"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.op.name() {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        for child in egraph.nodes(left) {
            if let Some(child_op) = child.op()
                && child_op.name() == self.op.name()
                && let Some((a, b)) = child.binary_operands() {
                    return Some(RewriteAction::Associate {
                        op: self.op,
                        a,
                        b,
                        c: right,
                    });
                }
        }
        None
    }
}

/// Commutativity: a op b → b op a
pub struct Commutative {
    op: &'static dyn Op,
}

impl Commutative {
    #[allow(dead_code)]
pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Commutative {
    fn name(&self) -> &str {
        "commutative"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.op.name() {
            return None;
        }

        let (a, b) = node.binary_operands()?;
        if a == b {
            return None;
        }

        let swapped = ENode::Op {
            op: self.op,
            children: vec![b, a],
        };
        Some(RewriteAction::Create(swapped))
    }
}

/// Distributivity: A * (B + C) → A*B + A*C
pub struct Distributive {
    outer: &'static dyn Op,
    inner: &'static dyn Op,
}

impl Distributive {
    #[allow(dead_code)]
pub fn new(outer: &'static dyn Op, inner: &'static dyn Op) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Distributive {
    fn name(&self) -> &str {
        "distribute"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.outer.name() {
            return None;
        }

        let (a, other) = node.binary_operands()?;

        for child_node in egraph.nodes(other) {
            if let Some(child_op) = child_node.op()
                && child_op.name() == self.inner.name()
                && let Some((b, c)) = child_node.binary_operands() {
                    return Some(RewriteAction::Distribute {
                        outer: self.outer,
                        inner: self.inner,
                        a,
                        b,
                        c,
                    });
                }
        }
        None
    }
}

/// Factoring: A*B + A*C → A * (B + C)
#[allow(dead_code)]
pub struct Factor {
    outer: &'static dyn Op,
    inner: &'static dyn Op,
}

impl Factor {
    #[allow(dead_code)]
pub fn new(outer: &'static dyn Op, inner: &'static dyn Op) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Factor {
    fn name(&self) -> &str {
        "factor"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.outer.name() {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        for l_node in egraph.nodes(left) {
            let l_op = l_node.op()?;
            if l_op.name() != self.inner.name() {
                continue;
            }
            let (la, lb) = l_node.binary_operands()?;

            for r_node in egraph.nodes(right) {
                let r_op = r_node.op()?;
                if r_op.name() != self.inner.name() {
                    continue;
                }
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
    op: &'static dyn Op,
}

impl Identity {
    #[allow(dead_code)]
pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Identity {
    fn name(&self) -> &str {
        "identity"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.op.name() {
            return None;
        }

        let id_val = self.op.identity()?;
        let (a, b) = node.binary_operands()?;

        if egraph.contains_const(b, id_val) {
            return Some(RewriteAction::Union(a));
        }
        if egraph.contains_const(a, id_val) {
            return Some(RewriteAction::Union(b));
        }
        None
    }
}

/// Annihilator: x op annihilator → annihilator
pub struct Annihilator {
    op: &'static dyn Op,
}

impl Annihilator {
    #[allow(dead_code)]
pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Annihilator {
    fn name(&self) -> &str {
        "annihilator"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.op.name() {
            return None;
        }

        let zero_val = self.op.annihilator()?;
        let (a, b) = node.binary_operands()?;

        if egraph.contains_const(a, zero_val) || egraph.contains_const(b, zero_val) {
            return Some(RewriteAction::Create(ENode::constant(zero_val)));
        }
        None
    }
}

/// Idempotence: x op x → x
#[allow(dead_code)]
pub struct Idempotent {
    op: &'static dyn Op,
}

impl Idempotent {
    #[allow(dead_code)]
pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Idempotent {
    fn name(&self) -> &str {
        "idempotent"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.op.name() {
            return None;
        }
        if !self.op.is_idempotent() {
            return None;
        }

        let (a, b) = node.binary_operands()?;

        if egraph.find(a) == egraph.find(b) {
            return Some(RewriteAction::Union(a));
        }
        None
    }
}

/// RecipSqrt: 1/sqrt(x) → rsqrt(x)
#[allow(dead_code)]
pub struct RecipSqrt;

impl Rewrite for RecipSqrt {
    fn name(&self) -> &str {
        "recip-sqrt"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if let ENode::Op { op, children } = node
            && op.name() == "recip" && children.len() == 1
        {
            let a = children[0];
            for child in egraph.nodes(a) {
                if let ENode::Op {
                    op: child_op,
                    children: child_children,
                } = child
                    && child_op.name() == "sqrt" && child_children.len() == 1
                {
                    let inner = child_children[0];
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Rsqrt,
                        children: vec![inner],
                    }));
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
#[allow(dead_code)]
pub struct FmaFusion;

impl Rewrite for FmaFusion {
    fn name(&self) -> &str {
        "fma-fusion"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match Add(Mul(a, b), c) → MulAdd(a, b, c)
        if let ENode::Op { op, children } = node
            && op.name() == "add" && children.len() == 2
        {
            let left = children[0];
            let right = children[1];

            // Check if left is a Mul
            for left_node in egraph.nodes(left) {
                if let ENode::Op {
                    op: left_op,
                    children: left_children,
                } = left_node
                    && left_op.name() == "mul" && left_children.len() == 2
                {
                    let a = left_children[0];
                    let b = left_children[1];
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::MulAdd,
                        children: vec![a, b, right],
                    }));
                }
            }

            // Check if right is a Mul (commutativity)
            for right_node in egraph.nodes(right) {
                if let ENode::Op {
                    op: right_op,
                    children: right_children,
                } = right_node
                    && right_op.name() == "mul" && right_children.len() == 2
                {
                    let a = right_children[0];
                    let b = right_children[1];
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::MulAdd,
                        children: vec![a, b, left],
                    }));
                }
            }
        }
        None
    }
}
