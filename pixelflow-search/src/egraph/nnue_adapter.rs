//! Adapter to convert between e-graph expressions and NNUE expression trees.

use crate::egraph::{EGraph, EClassId, ENode};
use pixelflow_ml::nnue::{Expr, OpType};
use alloc::boxed::Box;

// Helper to map `OpType` to a function that creates an `ENode`.
// This avoids a direct `OpType` -> `Op` mapping if they diverge.
fn nnue_to_op_constructor(op_type: OpType) -> Option<fn(EClassId, EClassId) -> ENode> {
    match op_type {
        OpType::Add => Some(ENode::Add),
        OpType::Sub => Some(ENode::Sub),
        OpType::Mul => Some(ENode::Mul),
        OpType::Div => Some(ENode::Div),
        OpType::Min => Some(ENode::Min),
        OpType::Max => Some(ENode::Max),
        _ => None,
    }
}

fn nnue_to_unary_op_constructor(op_type: OpType) -> Option<fn(EClassId) -> ENode> {
    match op_type {
        OpType::Neg => Some(ENode::Neg),
        OpType::Sqrt => Some(ENode::Sqrt),
        OpType::Rsqrt => Some(ENode::Rsqrt),
        OpType::Abs => Some(ENode::Abs),
        _ => None,
    }
}


/// Extract a concrete `Expr` from an e-class.
/// Picks the lowest-cost representative using greedy extraction.
pub fn eclass_to_expr(egraph: &EGraph, class: EClassId) -> Expr {
    // For now, let's just pick the first node. A real implementation
    // would use a cost model to extract the best one.
    let node = &egraph.nodes(class)[0];

    match node {
        ENode::Var(idx) => Expr::Var(*idx as u8),
        ENode::Const(val) => Expr::Const(f32::from_bits(*val)),
        ENode::Add(a, b) => Expr::Binary(OpType::Add, Box::new(eclass_to_expr(egraph, *a)), Box::new(eclass_to_expr(egraph, *b))),
        ENode::Sub(a, b) => Expr::Binary(OpType::Sub, Box::new(eclass_to_expr(egraph, *a)), Box::new(eclass_to_expr(egraph, *b))),
        ENode::Mul(a, b) => Expr::Binary(OpType::Mul, Box::new(eclass_to_expr(egraph, *a)), Box::new(eclass_to_expr(egraph, *b))),
        ENode::Div(a, b) => Expr::Binary(OpType::Div, Box::new(eclass_to_expr(egraph, *a)), Box::new(eclass_to_expr(egraph, *b))),
        ENode::Min(a, b) => Expr::Binary(OpType::Min, Box::new(eclass_to_expr(egraph, *a)), Box::new(eclass_to_expr(egraph, *b))),
        ENode::Max(a, b) => Expr::Binary(OpType::Max, Box::new(eclass_to_expr(egraph, *a)), Box::new(eclass_to_expr(egraph, *b))),
        ENode::Neg(a) => Expr::Unary(OpType::Neg, Box::new(eclass_to_expr(egraph, *a))),
        ENode::Sqrt(a) => Expr::Unary(OpType::Sqrt, Box::new(eclass_to_expr(egraph, *a))),
        ENode::Rsqrt(a) => Expr::Unary(OpType::Rsqrt, Box::new(eclass_to_expr(egraph, *a))),
        ENode::Abs(a) => Expr::Unary(OpType::Abs, Box::new(eclass_to_expr(egraph, *a))),
        ENode::MulAdd(a, b, c) => Expr::Ternary(
            OpType::MulAdd,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
            Box::new(eclass_to_expr(egraph, *c)),
        ),
        // All other variants would be handled here.
        // For now, panic if we see something we don't support.
        _ => panic!("Unsupported ENode variant in eclass_to_expr"),
    }
}

/// Insert an `Expr` into the e-graph, returning the root e-class.
pub fn expr_to_egraph(expr: &Expr, egraph: &mut EGraph) -> EClassId {
    match expr {
        Expr::Var(idx) => egraph.add(ENode::Var(*idx as u8)),
        Expr::Const(val) => egraph.add(ENode::Const(val.to_bits())),
        Expr::Binary(op, a, b) => {
            let a_class = expr_to_egraph(a, egraph);
            let b_class = expr_to_egraph(b, egraph);
            let constructor = nnue_to_op_constructor(*op)
                .unwrap_or_else(|| panic!("Unsupported binary op type: {:?}", op));
            let node = constructor(a_class, b_class);
            egraph.add(node)
        }
        Expr::Unary(op, a) => {
            let a_class = expr_to_egraph(a, egraph);
            let constructor = nnue_to_unary_op_constructor(*op)
                .unwrap_or_else(|| panic!("Unsupported unary op type: {:?}", op));
            let node = constructor(a_class);
            egraph.add(node)
        }
        Expr::Ternary(op, a, b, c) => {
            let a_class = expr_to_egraph(a, egraph);
            let b_class = expr_to_egraph(b, egraph);
            let c_class = expr_to_egraph(c, egraph);
            let node = match op {
                OpType::MulAdd => ENode::MulAdd(a_class, b_class, c_class),
                _ => panic!("Unsupported ternary op type: {:?}", op),
            };
            egraph.add(node)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{EGraph, ENode};
    use pixelflow_ml::nnue::{Expr, OpType};

    #[test]
    fn test_eclass_to_expr_leaf() {
        let mut egraph = EGraph::new();
        let var_class = egraph.add(ENode::Var(0));
        let expr = eclass_to_expr(&egraph, var_class);
        assert!(matches!(expr, Expr::Var(0)));
    }

    // Custom equality check for Expr
    fn expr_equals(a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Var(i), Expr::Var(j)) => i == j,
            (Expr::Const(x), Expr::Const(y)) => (x - y).abs() < 1e-6,
            (Expr::Unary(op1, a1), Expr::Unary(op2, a2)) => {
                op1 == op2 && expr_equals(a1, a2)
            }
            (Expr::Binary(op1, a1, b1), Expr::Binary(op2, a2, b2)) => {
                op1 == op2 && expr_equals(a1, a2) && expr_equals(b1, b2)
            }
            (Expr::Ternary(op1, a1, b1, c1), Expr::Ternary(op2, a2, b2, c2)) => {
                op1 == op2 && expr_equals(a1, a2) && expr_equals(b1, b2) && expr_equals(c1, c2)
            }
            _ => false,
        }
    }

    #[test]
    fn test_roundtrip_simple() {
        let expr = Expr::Binary(OpType::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let mut egraph = EGraph::new();
        let class = expr_to_egraph(&expr, &mut egraph);
        let recovered = eclass_to_expr(&egraph, class);
        assert!(expr_equals(&expr, &recovered), "Roundtrip failed for simple binary expression");
    }

    #[test]
    fn test_roundtrip_nested() {
        // (x * 2.0) + y
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Var(1)),
        );
        let mut egraph = EGraph::new();
        let class = expr_to_egraph(&expr, &mut egraph);
        let recovered = eclass_to_expr(&egraph, class);
        assert!(expr_equals(&expr, &recovered), "Roundtrip failed for nested expression");
    }
}
