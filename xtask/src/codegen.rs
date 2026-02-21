//! Kernel code generation task.

use pixelflow_search::egraph::{ops, CostModel, EGraph, ENode, ExprTree, Leaf};
use std::fs;
use std::path::Path;

pub fn generate_kernels(workspace_root: &Path) {
    println!("Generating kernels...");

    // Example: Optimize a discriminant calculation: d^2 - (c^2 - r^2)
    // This tests the algebraic simplification and FMA fusion.
    let mut eg = EGraph::new();

    // Vars: 0=d, 1=c, 2=r
    let d = eg.add(ENode::Var(0));
    let c = eg.add(ENode::Var(1));
    let r = eg.add(ENode::Var(2));

    let d_sq = eg.add(ENode::Op {
        op: &ops::Mul,
        children: vec![d, d],
    });
    let c_sq = eg.add(ENode::Op {
        op: &ops::Mul,
        children: vec![c, c],
    });
    let r_sq = eg.add(ENode::Op {
        op: &ops::Mul,
        children: vec![r, r],
    });

    // inner = c^2 - r^2
    let inner = eg.add(ENode::Op {
        op: &ops::Sub,
        children: vec![c_sq, r_sq],
    });

    // res = d^2 - inner
    let res = eg.add(ENode::Op {
        op: &ops::Sub,
        children: vec![d_sq, inner],
    });

    eg.saturate();

    let costs = CostModel::fully_optimized();
    let best_tree = eg.extract_tree_with_costs(res, &costs);

    println!("Optimized Tree: {:?}", best_tree);

    // Generate Rust code from the tree (simple emitter)
    let code = emit_rust_expr(&best_tree);
    println!("Generated Code: {}", code);

    // Write to a file in pixelflow-core/src/generated_kernels.rs (just for demo)
    let output_path = workspace_root.join("pixelflow-core/src/generated_kernels.rs");

    let file_content = format!(
        r#"//! Auto-generated kernels by xtask codegen.

use crate::{{Field, Manifold}};

/// Discriminant: d^2 - (c^2 - r^2)
/// Optimized form: {}
pub fn discriminant(d: Field, c: Field, r: Field) -> Field {{
    {}
}}
"#,
        code, code
    );

    fs::write(&output_path, file_content).expect("Failed to write generated kernels");
    println!("Wrote kernels to {}", output_path.display());
}

fn emit_rust_expr(tree: &ExprTree) -> String {
    match tree {
        ExprTree::Leaf(Leaf::Var(0)) => "d".to_string(),
        ExprTree::Leaf(Leaf::Var(1)) => "c".to_string(),
        ExprTree::Leaf(Leaf::Var(2)) => "r".to_string(),
        ExprTree::Leaf(Leaf::Var(n)) => format!("var_{}", n),
        ExprTree::Leaf(Leaf::Const(c)) => format!("{:.1}", c),
        ExprTree::Op { op, children } => match (op.name(), children.as_slice()) {
            ("add", [a, b]) => format!("({} + {})", emit_rust_expr(a), emit_rust_expr(b)),
            ("sub", [a, b]) => format!("({} - {})", emit_rust_expr(a), emit_rust_expr(b)),
            ("mul", [a, b]) => format!("({} * {})", emit_rust_expr(a), emit_rust_expr(b)),
            ("mul_add", [a, b, c]) => format!(
                "{}.mul_add({}, {})",
                emit_rust_expr(a),
                emit_rust_expr(b),
                emit_rust_expr(c)
            ),
            ("neg", [a]) => format!("-{}", emit_rust_expr(a)),
            _ => format!("/* todo: {} */", op.name()),
        },
    }
}
