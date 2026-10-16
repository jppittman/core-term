//! Symbolic differentiation as e-graph rewrites.
//!
//! Autodiff is a single operator, `Dwrt(expr, var)` ([`OpKind::Dwrt`]). The
//! author writes `D(expr, var)` and never learns the mechanism: the chain rule
//! lives here, as one rewrite rule that expands a `Dwrt` node one step toward
//! the leaves. Equality saturation runs it to fixpoint, and the residual
//! arithmetic is then optimised by the ordinary algebra/fusion rules in the
//! same e-graph — there is no ordering problem, symbolic differentiation and
//! FMA fusion saturate together.
//!
//! A `Dwrt` that survives saturation (an operator with no differentiation
//! rule, or a budget miss) is left in the graph with a prohibitive cost so the
//! extractor never prefers it. The fallback tier is the runtime `lower_dwrt`
//! pass in pixelflow-ir — the same algebra applied directly to the arena —
//! which errors loudly on genuinely non-differentiable ops.
//!
//! The actual derivative construction lives in `EGraph::build_derivative`,
//! reached through [`RewriteAction::Differentiate`]; this rule only recognises
//! a `Dwrt` node, reads the variable index from its constant operand, and picks
//! a representative of the differentiand to hand off.

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::rewrite::{Rewrite, RewriteAction, RootShapes};
use alloc::boxed::Box;
use alloc::vec::Vec;
use pixelflow_ir::kind::OpKind;

/// The chain rule: expand `Dwrt(expr, var)` one differentiation step.
pub struct ChainRule;

impl ChainRule {
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for ChainRule {
    fn name(&self) -> &str {
        "differentiate"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Dwrt)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match `Dwrt(expr, var)`.
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != OpKind::Dwrt || children.len() != 2 {
            return None;
        }
        let expr_class = children[0];
        let var_class = children[1];

        // The differentiation variable is encoded as a constant operand.
        let var = egraph.nodes(var_class).find_map(ENode::as_f32)? as u8;

        // Differentiate a representative of the differentiand. All nodes in the
        // class are equal, so any non-`Dwrt` representative gives the same
        // derivative; skipping `Dwrt` nodes avoids differentiating a pending
        // derivative back into itself.
        let inner = egraph.nodes(expr_class).find(|n| !is_dwrt(n)).cloned()?;

        Some(RewriteAction::Differentiate { inner, var })
    }
}

fn is_dwrt(node: &ENode) -> bool {
    matches!(node, ENode::Op { op, .. } if *op == OpKind::Dwrt)
}

/// The differentiation rule set: just the chain rule. It only matches `Dwrt`
/// nodes, so it is inert for kernels that contain no derivatives.
#[must_use]
pub fn derivative_rules() -> Vec<Box<dyn Rewrite>> {
    vec![ChainRule::new()]
}
