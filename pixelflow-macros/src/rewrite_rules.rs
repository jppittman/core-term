//! Domain-specific rewrite rules for PixelFlow compiler optimization.
//!
//! These rules encode knowledge about CPU instruction sets and performance.
//! They are NOT generic algebraic rules - those belong in pixelflow-search.

use pixelflow_search::egraph::rewrite::{Rewrite, RewriteAction};
use pixelflow_search::egraph::{EClassId, EGraph, ENode, ops};

/// FmaFusion: a * b + c → MulAdd(a, b, c)
///
/// Fused multiply-add is typically a single instruction on modern CPUs.
/// This rule adds the fused form to the e-graph; extraction cost model
/// determines whether it's actually used.
///
/// This is a DOMAIN-SPECIFIC rule (knows about CPU instructions) and belongs
/// in the compiler (pixelflow-macros), NOT the generic e-graph library.
pub struct FmaFusion;

impl Rewrite for FmaFusion {
    fn name(&self) -> &str {
        "fma-fusion"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match Add(Mul(a, b), c) → MulAdd(a, b, c)
        if let ENode::Op { op, children } = node {
            if op.name() == "add" && children.len() == 2 {
                let left = children[0];
                let right = children[1];

                // Check if left is a Mul
                for left_node in egraph.nodes(left) {
                    if let ENode::Op {
                        op: left_op,
                        children: left_children,
                    } = left_node
                    {
                        if left_op.name() == "mul" && left_children.len() == 2 {
                            let a = left_children[0];
                            let b = left_children[1];
                            return Some(RewriteAction::Create(ENode::Op {
                                op: &ops::MulAdd,
                                children: vec![a, b, right],
                            }));
                        }
                    }
                }

                // Check if right is a Mul (commutativity)
                for right_node in egraph.nodes(right) {
                    if let ENode::Op {
                        op: right_op,
                        children: right_children,
                    } = right_node
                    {
                        if right_op.name() == "mul" && right_children.len() == 2 {
                            let a = right_children[0];
                            let b = right_children[1];
                            return Some(RewriteAction::Create(ENode::Op {
                                op: &ops::MulAdd,
                                children: vec![a, b, left],
                            }));
                        }
                    }
                }
            }
        }
        None
    }
}

/// RecipSqrt: 1/sqrt(x) → rsqrt(x)
///
/// Reciprocal square root is a single instruction on many CPUs (rsqrtss/rsqrtps).
/// This is a DOMAIN-SPECIFIC optimization rule and belongs in the compiler.
pub struct RecipSqrt;

impl Rewrite for RecipSqrt {
    fn name(&self) -> &str {
        "recip-sqrt"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if let ENode::Op { op, children } = node {
            if op.name() == "recip" && children.len() == 1 {
                let a = children[0];
                for child in egraph.nodes(a) {
                    if let ENode::Op {
                        op: child_op,
                        children: child_children,
                    } = child
                    {
                        if child_op.name() == "sqrt" && child_children.len() == 1 {
                            let inner = child_children[0];
                            return Some(RewriteAction::Create(ENode::Op {
                                op: &ops::Rsqrt,
                                children: vec![inner],
                            }));
                        }
                    }
                }
            }
        }
        None
    }
}
