//! Non-InversePair rewrite rules.
//!
//! These rules use Op directly rather than being derived from InversePair.

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::{self, Op};
use super::rewrite::{Rewrite, RewriteAction};

/// Commutativity: a op b → b op a
pub struct Commutative {
    op: &'static dyn Op,
}

impl Commutative {
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
            if let Some(child_op) = child_node.op() {
                if child_op.name() == self.inner.name() {
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
        }
        None
    }
}

/// Identity: x op identity → x
pub struct Identity {
    op: &'static dyn Op,
}

impl Identity {
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

