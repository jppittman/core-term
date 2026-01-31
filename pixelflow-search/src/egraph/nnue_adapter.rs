//! Adapter to convert between e-graph expressions and NNUE expression trees.
//!
//! Bridges the gap between `pixelflow_search::egraph` types and `pixelflow_nnue` types
//! for feature extraction and training data generation.

use crate::egraph::{EClassId, EGraph, ENode, ops};
use crate::egraph::extract::{ExprTree, Leaf};
use alloc::boxed::Box;
use alloc::vec;
use pixelflow_nnue::{Expr, OpKind};

/// Convert e-graph `Op` to NNUE `OpKind` by name.
pub fn op_to_nnue(op: &dyn crate::egraph::ops::Op) -> OpKind {
    match op.name() {
        "add" => OpKind::Add,
        "sub" => OpKind::Sub,
        "mul" => OpKind::Mul,
        "div" => OpKind::Div,
        "neg" => OpKind::Neg,
        "recip" => OpKind::Div, // Recip = 1/x, map to Div
        "min" => OpKind::Min,
        "max" => OpKind::Max,
        "sqrt" => OpKind::Sqrt,
        "rsqrt" => OpKind::Rsqrt,
        "abs" => OpKind::Abs,
        "mul_add" => OpKind::MulAdd,
        _ => OpKind::Add, // Fallback for unsupported ops
    }
}

/// Convert NNUE `OpKind` to e-graph `Op` reference (where possible).
pub fn nnue_to_op(op_type: OpKind) -> Option<&'static dyn crate::egraph::ops::Op> {
    match op_type {
        OpKind::Add => Some(&ops::Add),
        OpKind::Sub => Some(&ops::Sub),
        OpKind::Mul => Some(&ops::Mul),
        OpKind::Div => Some(&ops::Div),
        OpKind::Neg => Some(&ops::Neg),
        OpKind::Min => Some(&ops::Min),
        OpKind::Max => Some(&ops::Max),
        OpKind::Sqrt => Some(&ops::Sqrt),
        OpKind::Rsqrt => Some(&ops::Rsqrt),
        OpKind::Abs => Some(&ops::Abs),
        OpKind::MulAdd => Some(&ops::MulAdd),
        // Extended operations not yet in e-graph
        OpKind::MulRsqrt | OpKind::Var | OpKind::Const |
        OpKind::Recip | OpKind::Floor | OpKind::Ceil | OpKind::Round | OpKind::Fract |
        OpKind::Sin | OpKind::Cos | OpKind::Tan | OpKind::Asin | OpKind::Acos | OpKind::Atan | OpKind::Atan2 |
        OpKind::Exp | OpKind::Exp2 | OpKind::Ln | OpKind::Log2 | OpKind::Log10 | OpKind::Pow | OpKind::Hypot |
        OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne |
        OpKind::Select | OpKind::Clamp | OpKind::Tuple => None,
    }
}

/// Extract a concrete `Expr` from an e-class.
/// Picks the first representative (a real implementation would use cost-based extraction).
pub fn eclass_to_expr(egraph: &EGraph, class: EClassId) -> Expr {
    let node = &egraph.nodes(class)[0];

    match node {
        ENode::Var(idx) => Expr::Var(*idx),
        ENode::Const(bits) => Expr::Const(f32::from_bits(*bits)),
        ENode::Op { op, children } => {
            let op_type = op_to_nnue(*op);
            match children.len() {
                1 => Expr::Unary(op_type, Box::new(eclass_to_expr(egraph, children[0]))),
                2 => Expr::Binary(
                    op_type,
                    Box::new(eclass_to_expr(egraph, children[0])),
                    Box::new(eclass_to_expr(egraph, children[1])),
                ),
                3 => Expr::Ternary(
                    op_type,
                    Box::new(eclass_to_expr(egraph, children[0])),
                    Box::new(eclass_to_expr(egraph, children[1])),
                    Box::new(eclass_to_expr(egraph, children[2])),
                ),
                _ => panic!("Unsupported arity {} for op {}", children.len(), op.name()),
            }
        }
    }
}

/// Convert an `ExprTree` to a NNUE `Expr` for feature extraction.
pub fn expr_tree_to_nnue(tree: &ExprTree) -> Expr {
    match tree {
        ExprTree::Leaf(Leaf::Var(i)) => Expr::Var(*i),
        ExprTree::Leaf(Leaf::Const(c)) => Expr::Const(*c),
        ExprTree::Op { op, children } => {
            let op_type = op_to_nnue(*op);
            match children.len() {
                1 => Expr::Unary(op_type, Box::new(expr_tree_to_nnue(&children[0]))),
                2 => Expr::Binary(
                    op_type,
                    Box::new(expr_tree_to_nnue(&children[0])),
                    Box::new(expr_tree_to_nnue(&children[1])),
                ),
                3 => Expr::Ternary(
                    op_type,
                    Box::new(expr_tree_to_nnue(&children[0])),
                    Box::new(expr_tree_to_nnue(&children[1])),
                    Box::new(expr_tree_to_nnue(&children[2])),
                ),
                0 => Expr::Const(0.0), // Empty children = constant
                _ => {
                    // For tuple or other variadic ops, just use first element
                    expr_tree_to_nnue(&children[0])
                }
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
            let op_ref: &'static dyn crate::egraph::ops::Op = match op {
                OpKind::Add => &ops::Add,
                OpKind::Sub => &ops::Sub,
                OpKind::Mul => &ops::Mul,
                OpKind::Div => &ops::Div,
                OpKind::Min => &ops::Min,
                OpKind::Max => &ops::Max,
                OpKind::MulRsqrt => {
                    // Decompose: mul_rsqrt(a, b) = a * rsqrt(b)
                    let rsqrt_b = egraph.add(ENode::Op {
                        op: &ops::Rsqrt,
                        children: vec![b_class],
                    });
                    return egraph.add(ENode::Op {
                        op: &ops::Mul,
                        children: vec![a_class, rsqrt_b],
                    });
                }
                _ => panic!("Unsupported binary op type: {:?}", op),
            };
            egraph.add(ENode::Op {
                op: op_ref,
                children: vec![a_class, b_class],
            })
        }
        Expr::Unary(op, a) => {
            let a_class = expr_to_egraph(a, egraph);
            let op_ref: &'static dyn crate::egraph::ops::Op = match op {
                OpKind::Neg => &ops::Neg,
                OpKind::Sqrt => &ops::Sqrt,
                OpKind::Rsqrt => &ops::Rsqrt,
                OpKind::Abs => &ops::Abs,
                _ => panic!("Unsupported unary op type: {:?}", op),
            };
            egraph.add(ENode::Op {
                op: op_ref,
                children: vec![a_class],
            })
        }
        Expr::Ternary(op, a, b, c) => {
            let a_class = expr_to_egraph(a, egraph);
            let b_class = expr_to_egraph(b, egraph);
            let c_class = expr_to_egraph(c, egraph);
            let op_ref: &'static dyn crate::egraph::ops::Op = match op {
                OpKind::MulAdd => &ops::MulAdd,
                _ => panic!("Unsupported ternary op type: {:?}", op),
            };
            egraph.add(ENode::Op {
                op: op_ref,
                children: vec![a_class, b_class, c_class],
            })
        }
        Expr::Nary(op, children) => {
            let child_classes: Vec<_> = children.iter()
                .map(|c| expr_to_egraph(c, egraph))
                .collect();
            // Currently only Tuple is supported as Nary
            match op {
                OpKind::Tuple => {
                    // Tuple has no e-graph representation - flatten to first element or panic
                    if child_classes.is_empty() {
                        panic!("expr_to_egraph: empty Tuple not supported");
                    }
                    // For now, just return the first child (tuples aren't fully supported in e-graph yet)
                    child_classes[0]
                }
                _ => panic!("Unsupported n-ary op type: {:?}", op),
            }
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
            (Expr::Nary(op1, c1), Expr::Nary(op2, c2)) => {
                op1 == op2 && c1.len() == c2.len() &&
                c1.iter().zip(c2.iter()).all(|(x, y)| expr_equals(x, y))
            }
            _ => false,
        }
    }

    #[test]
    fn test_op_to_nnue_roundtrip() {
        let ops_to_test: &[&dyn crate::egraph::ops::Op] = &[
            &ops::Add,
            &ops::Sub,
            &ops::Mul,
            &ops::Div,
            &ops::Neg,
            &ops::Min,
            &ops::Max,
            &ops::Sqrt,
            &ops::Rsqrt,
            &ops::Abs,
            &ops::MulAdd,
        ];
        for op in ops_to_test {
            let nnue_op = op_to_nnue(*op);
            let back = nnue_to_op(nnue_op);
            // Recip maps to Div, so skip that case
            if op.name() != "recip" {
                assert!(back.is_some(), "Roundtrip failed for {}", op.name());
                assert_eq!(back.unwrap().name(), op.name(), "Roundtrip failed for {}", op.name());
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
            OpKind::Add,
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
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
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
