//! Adapter to convert between e-graph expressions and NNUE expression trees.
//!
//! Bridges the gap between `pixelflow_search::egraph` types and `pixelflow_nnue` types
//! for feature extraction and training data generation.

use crate::egraph::{EClassId, EGraph, ENode};
use alloc::boxed::Box;
use pixelflow_nnue::{Expr, OpType};

/// Convert e-graph `Op` to NNUE `OpType`.
pub fn op_to_nnue(op: &crate::egraph::Op) -> OpType {
    match op {
        crate::egraph::Op::Add => OpType::Add,
        crate::egraph::Op::Sub => OpType::Sub,
        crate::egraph::Op::Mul => OpType::Mul,
        crate::egraph::Op::Div => OpType::Div,
        crate::egraph::Op::Neg => OpType::Neg,
        crate::egraph::Op::Recip => OpType::Div, // Recip = 1/x, map to Div
        crate::egraph::Op::Min => OpType::Min,
        crate::egraph::Op::Max => OpType::Max,
        crate::egraph::Op::Sqrt => OpType::Sqrt,
        crate::egraph::Op::Rsqrt => OpType::Rsqrt,
        crate::egraph::Op::Abs => OpType::Abs,
        crate::egraph::Op::MulAdd => OpType::MulAdd,
    }
}

/// Convert NNUE `OpType` to e-graph `Op` (where possible).
pub fn nnue_to_op(op_type: OpType) -> Option<crate::egraph::Op> {
    match op_type {
        OpType::Add => Some(crate::egraph::Op::Add),
        OpType::Sub => Some(crate::egraph::Op::Sub),
        OpType::Mul => Some(crate::egraph::Op::Mul),
        OpType::Div => Some(crate::egraph::Op::Div),
        OpType::Neg => Some(crate::egraph::Op::Neg),
        OpType::Min => Some(crate::egraph::Op::Min),
        OpType::Max => Some(crate::egraph::Op::Max),
        OpType::Sqrt => Some(crate::egraph::Op::Sqrt),
        OpType::Rsqrt => Some(crate::egraph::Op::Rsqrt),
        OpType::Abs => Some(crate::egraph::Op::Abs),
        OpType::MulAdd => Some(crate::egraph::Op::MulAdd),
        OpType::MulRsqrt => None, // No direct equivalent in e-graph
        OpType::Var | OpType::Const => None, // Not operations
    }
}

/// Extract a concrete `Expr` from an e-class.
/// Picks the first representative (a real implementation would use cost-based extraction).
pub fn eclass_to_expr(egraph: &EGraph, class: EClassId) -> Expr {
    let node = &egraph.nodes(class)[0];

    match node {
        ENode::Var(idx) => Expr::Var(*idx),
        ENode::Const(bits) => Expr::Const(f32::from_bits(*bits)),

        // Binary operations
        ENode::Add(a, b) => Expr::Binary(
            OpType::Add,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
        ),
        ENode::Sub(a, b) => Expr::Binary(
            OpType::Sub,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
        ),
        ENode::Mul(a, b) => Expr::Binary(
            OpType::Mul,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
        ),
        ENode::Div(a, b) => Expr::Binary(
            OpType::Div,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
        ),
        ENode::Min(a, b) => Expr::Binary(
            OpType::Min,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
        ),
        ENode::Max(a, b) => Expr::Binary(
            OpType::Max,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
        ),

        // Unary operations
        ENode::Neg(a) => Expr::Unary(OpType::Neg, Box::new(eclass_to_expr(egraph, *a))),
        ENode::Sqrt(a) => Expr::Unary(OpType::Sqrt, Box::new(eclass_to_expr(egraph, *a))),
        ENode::Rsqrt(a) => Expr::Unary(OpType::Rsqrt, Box::new(eclass_to_expr(egraph, *a))),
        ENode::Abs(a) => Expr::Unary(OpType::Abs, Box::new(eclass_to_expr(egraph, *a))),
        ENode::Recip(a) => {
            // Recip(x) = 1/x = Div(1, x)
            Expr::Binary(
                OpType::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(eclass_to_expr(egraph, *a)),
            )
        }

        // Ternary
        ENode::MulAdd(a, b, c) => Expr::Ternary(
            OpType::MulAdd,
            Box::new(eclass_to_expr(egraph, *a)),
            Box::new(eclass_to_expr(egraph, *b)),
            Box::new(eclass_to_expr(egraph, *c)),
        ),

        // Pass-through operations - map to closest NNUE equivalent or panic
        _ => panic!("Unsupported ENode variant in eclass_to_expr: {:?}", node),
    }
}

/// Convert an `ExprTree` to a NNUE `Expr` for feature extraction.
pub fn expr_tree_to_nnue(tree: &crate::egraph::ExprTree) -> Expr {
    use crate::egraph::ExprTree;

    match tree {
        ExprTree::Var(i) => Expr::Var(*i),
        ExprTree::Const(c) => Expr::Const(*c),

        // Unary operations
        ExprTree::Neg(a) => Expr::Unary(OpType::Neg, Box::new(expr_tree_to_nnue(a))),
        ExprTree::Sqrt(a) => Expr::Unary(OpType::Sqrt, Box::new(expr_tree_to_nnue(a))),
        ExprTree::Rsqrt(a) => Expr::Unary(OpType::Rsqrt, Box::new(expr_tree_to_nnue(a))),
        ExprTree::Abs(a) => Expr::Unary(OpType::Abs, Box::new(expr_tree_to_nnue(a))),
        ExprTree::Recip(a) => {
            // Recip(x) = 1/x
            Expr::Binary(
                OpType::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(expr_tree_to_nnue(a)),
            )
        }
        // Map unsupported unary ops to Neg (placeholder)
        ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
        | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
        | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a)
        | ExprTree::Exp(a) | ExprTree::Exp2(a) | ExprTree::Ln(a)
        | ExprTree::Log2(a) | ExprTree::Log10(a) => {
            // Map to Neg as a placeholder - these aren't in NNUE's OpType
            Expr::Unary(OpType::Neg, Box::new(expr_tree_to_nnue(a)))
        }

        // Binary operations
        ExprTree::Add(a, b) => Expr::Binary(OpType::Add, Box::new(expr_tree_to_nnue(a)), Box::new(expr_tree_to_nnue(b))),
        ExprTree::Sub(a, b) => Expr::Binary(OpType::Sub, Box::new(expr_tree_to_nnue(a)), Box::new(expr_tree_to_nnue(b))),
        ExprTree::Mul(a, b) => Expr::Binary(OpType::Mul, Box::new(expr_tree_to_nnue(a)), Box::new(expr_tree_to_nnue(b))),
        ExprTree::Div(a, b) => Expr::Binary(OpType::Div, Box::new(expr_tree_to_nnue(a)), Box::new(expr_tree_to_nnue(b))),
        ExprTree::Min(a, b) => Expr::Binary(OpType::Min, Box::new(expr_tree_to_nnue(a)), Box::new(expr_tree_to_nnue(b))),
        ExprTree::Max(a, b) => Expr::Binary(OpType::Max, Box::new(expr_tree_to_nnue(a)), Box::new(expr_tree_to_nnue(b))),
        // Map unsupported binary ops to Add (placeholder)
        ExprTree::Atan2(a, b) | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b)
        | ExprTree::Lt(a, b) | ExprTree::Le(a, b) | ExprTree::Gt(a, b)
        | ExprTree::Ge(a, b) | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
            Expr::Binary(OpType::Add, Box::new(expr_tree_to_nnue(a)), Box::new(expr_tree_to_nnue(b)))
        }

        // Ternary operations
        ExprTree::MulAdd(a, b, c) => Expr::Ternary(
            OpType::MulAdd,
            Box::new(expr_tree_to_nnue(a)),
            Box::new(expr_tree_to_nnue(b)),
            Box::new(expr_tree_to_nnue(c)),
        ),
        // Map unsupported ternary ops to MulAdd (placeholder)
        ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
            Expr::Ternary(
                OpType::MulAdd,
                Box::new(expr_tree_to_nnue(a)),
                Box::new(expr_tree_to_nnue(b)),
                Box::new(expr_tree_to_nnue(c)),
            )
        }

        ExprTree::Tuple(elems) => {
            // Tuples don't have a direct mapping - just use first element
            if let Some(first) = elems.first() {
                expr_tree_to_nnue(first)
            } else {
                Expr::Const(0.0)
            }
        }
    }
}

/// Insert an `Expr` into the e-graph, returning the root e-class.
pub fn expr_to_egraph(expr: &Expr, egraph: &mut EGraph) -> EClassId {
    match expr {
        Expr::Var(idx) => egraph.add(ENode::Var(*idx)),
        Expr::Const(val) => egraph.add(ENode::Const(val.to_bits())),
        Expr::Binary(op, a, b) => {
            let a_class = expr_to_egraph(a, egraph);
            let b_class = expr_to_egraph(b, egraph);
            let node = match op {
                OpType::Add => ENode::Add(a_class, b_class),
                OpType::Sub => ENode::Sub(a_class, b_class),
                OpType::Mul => ENode::Mul(a_class, b_class),
                OpType::Div => ENode::Div(a_class, b_class),
                OpType::Min => ENode::Min(a_class, b_class),
                OpType::Max => ENode::Max(a_class, b_class),
                _ => panic!("Unsupported binary op type: {:?}", op),
            };
            egraph.add(node)
        }
        Expr::Unary(op, a) => {
            let a_class = expr_to_egraph(a, egraph);
            let node = match op {
                OpType::Neg => ENode::Neg(a_class),
                OpType::Sqrt => ENode::Sqrt(a_class),
                OpType::Rsqrt => ENode::Rsqrt(a_class),
                OpType::Abs => ENode::Abs(a_class),
                _ => panic!("Unsupported unary op type: {:?}", op),
            };
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

    fn expr_equals(a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Var(i), Expr::Var(j)) => i == j,
            (Expr::Const(x), Expr::Const(y)) => (x - y).abs() < 1e-6,
            (Expr::Unary(op1, a1), Expr::Unary(op2, a2)) => op1 == op2 && expr_equals(a1, a2),
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
    fn test_op_to_nnue_roundtrip() {
        use crate::egraph::Op;
        let ops = [
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Neg,
            Op::Min,
            Op::Max,
            Op::Sqrt,
            Op::Rsqrt,
            Op::Abs,
            Op::MulAdd,
        ];
        for op in ops {
            let nnue_op = op_to_nnue(&op);
            let back = nnue_to_op(nnue_op);
            // Recip maps to Div, so skip that case
            if !matches!(op, Op::Recip) {
                assert_eq!(back, Some(op), "Roundtrip failed for {:?}", op);
            }
        }
    }

    #[test]
    fn test_eclass_to_expr_leaf() {
        let mut egraph = EGraph::new();
        let var_class = egraph.add(ENode::Var(0));
        let expr = eclass_to_expr(&egraph, var_class);
        assert!(matches!(expr, Expr::Var(0)));
    }

    #[test]
    fn test_roundtrip_simple() {
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let mut egraph = EGraph::new();
        let class = expr_to_egraph(&expr, &mut egraph);
        let recovered = eclass_to_expr(&egraph, class);
        assert!(
            expr_equals(&expr, &recovered),
            "Roundtrip failed for simple binary expression"
        );
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
        assert!(
            expr_equals(&expr, &recovered),
            "Roundtrip failed for nested expression"
        );
    }
}
