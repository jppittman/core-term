//! Code generation from ExprTree to kernel! macro code.
//!
//! This module converts extracted expression trees into Rust code strings
//! that can be compiled and benchmarked using the kernel! macro.
//!
//! # Usage
//!
//! ```ignore
//! use pixelflow_search::egraph::{ExprTree, codegen};
//!
//! let tree = ExprTree::Add(
//!     Box::new(ExprTree::Var(0)),
//!     Box::new(ExprTree::Mul(
//!         Box::new(ExprTree::Var(1)),
//!         Box::new(ExprTree::Const(2.0)),
//!     )),
//! );
//!
//! let body = codegen::expr_tree_to_kernel_body(&tree);
//! // Returns: "(X + (Y * 2.0))"
//!
//! let code = codegen::expr_tree_to_kernel_code(&tree, "my_kernel");
//! // Returns: "let my_kernel = kernel!(|| (X + (Y * 2.0)));"
//! ```

use super::ExprTree;

/// Convert an ExprTree to a kernel! macro code string.
///
/// This generates a complete `let` statement with the kernel! macro.
///
/// # Arguments
///
/// * `tree` - The expression tree to convert
/// * `name` - The variable name for the kernel
///
/// # Returns
///
/// A Rust code string like: `let my_kernel = kernel!(|| X + Y);`
pub fn expr_tree_to_kernel_code(tree: &ExprTree, name: &str) -> String {
    let body = expr_tree_to_kernel_body(tree);
    format!("let {} = kernel!(|| {});", name, body)
}

/// Convert an ExprTree to the body of a kernel! macro.
///
/// This generates just the expression part, suitable for use inside kernel!.
///
/// # Arguments
///
/// * `tree` - The expression tree to convert
///
/// # Returns
///
/// A Rust expression string like: `(X + (Y * 2.0))`
pub fn expr_tree_to_kernel_body(tree: &ExprTree) -> String {
    match tree {
        ExprTree::Var(0) => "X".to_string(),
        ExprTree::Var(1) => "Y".to_string(),
        ExprTree::Var(2) => "Z".to_string(),
        ExprTree::Var(3) => "W".to_string(),
        ExprTree::Var(i) => format!("V{}", i),

        ExprTree::Const(v) => format_const(*v),

        // Unary operations
        ExprTree::Neg(a) => format!("(-{})", expr_tree_to_kernel_body(a)),
        ExprTree::Recip(a) => format!("(1.0 / {})", expr_tree_to_kernel_body(a)),
        ExprTree::Sqrt(a) => format!("({}).sqrt()", expr_tree_to_kernel_body(a)),
        ExprTree::Rsqrt(a) => format!("({}).rsqrt()", expr_tree_to_kernel_body(a)),
        ExprTree::Abs(a) => format!("({}).abs()", expr_tree_to_kernel_body(a)),
        ExprTree::Floor(a) => format!("({}).floor()", expr_tree_to_kernel_body(a)),
        ExprTree::Ceil(a) => format!("({}).ceil()", expr_tree_to_kernel_body(a)),
        ExprTree::Round(a) => format!("({}).round()", expr_tree_to_kernel_body(a)),
        ExprTree::Fract(a) => format!("({}).fract()", expr_tree_to_kernel_body(a)),
        ExprTree::Sin(a) => format!("({}).sin()", expr_tree_to_kernel_body(a)),
        ExprTree::Cos(a) => format!("({}).cos()", expr_tree_to_kernel_body(a)),
        ExprTree::Tan(a) => format!("({}).tan()", expr_tree_to_kernel_body(a)),
        ExprTree::Asin(a) => format!("({}).asin()", expr_tree_to_kernel_body(a)),
        ExprTree::Acos(a) => format!("({}).acos()", expr_tree_to_kernel_body(a)),
        ExprTree::Atan(a) => format!("({}).atan()", expr_tree_to_kernel_body(a)),
        ExprTree::Exp(a) => format!("({}).exp()", expr_tree_to_kernel_body(a)),
        ExprTree::Exp2(a) => format!("({}).exp2()", expr_tree_to_kernel_body(a)),
        ExprTree::Ln(a) => format!("({}).ln()", expr_tree_to_kernel_body(a)),
        ExprTree::Log2(a) => format!("({}).log2()", expr_tree_to_kernel_body(a)),
        ExprTree::Log10(a) => format!("({}).log10()", expr_tree_to_kernel_body(a)),

        // Binary operations
        ExprTree::Add(a, b) => format!(
            "({} + {})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Sub(a, b) => format!(
            "({} - {})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Mul(a, b) => format!(
            "({} * {})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Div(a, b) => format!(
            "({} / {})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Min(a, b) => format!(
            "({}).min({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Max(a, b) => format!(
            "({}).max({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Atan2(a, b) => format!(
            "({}).atan2({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Pow(a, b) => format!(
            "({}).powf({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Hypot(a, b) => format!(
            "({}).hypot({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Lt(a, b) => format!(
            "({}).lt({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Le(a, b) => format!(
            "({}).le({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Gt(a, b) => format!(
            "({}).gt({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Ge(a, b) => format!(
            "({}).ge({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Eq(a, b) => format!(
            "({}).eq({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),
        ExprTree::Ne(a, b) => format!(
            "({}).ne({})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b)
        ),

        // Ternary operations
        ExprTree::MulAdd(a, b, c) => format!(
            "({}).mul_add({}, {})",
            expr_tree_to_kernel_body(a),
            expr_tree_to_kernel_body(b),
            expr_tree_to_kernel_body(c)
        ),
        ExprTree::Select(cond, then_val, else_val) => format!(
            "({}).select({}, {})",
            expr_tree_to_kernel_body(cond),
            expr_tree_to_kernel_body(then_val),
            expr_tree_to_kernel_body(else_val)
        ),
        ExprTree::Clamp(val, min, max) => format!(
            "({}).clamp({}, {})",
            expr_tree_to_kernel_body(val),
            expr_tree_to_kernel_body(min),
            expr_tree_to_kernel_body(max)
        ),

        ExprTree::Tuple(elems) => {
            let parts: Vec<_> = elems.iter().map(expr_tree_to_kernel_body).collect();
            format!("({})", parts.join(", "))
        }
    }
}

/// Format a constant value as Rust code.
fn format_const(v: f32) -> String {
    if v == 0.0 {
        "0.0".to_string()
    } else if v == 1.0 {
        "1.0".to_string()
    } else if v == -1.0 {
        "(-1.0)".to_string()
    } else if v.fract() == 0.0 && v.abs() < 1000.0 {
        // Integer-valued floats
        format!("{:.1}", v)
    } else {
        format!("({:.6})", v)
    }
}

/// Generate a complete benchmark file for a set of expression trees.
///
/// This creates a Criterion benchmark file that can measure the actual
/// SIMD execution cost of each expression.
///
/// # Arguments
///
/// * `variants` - List of (name, tree) pairs to benchmark
///
/// # Returns
///
/// Complete Rust source code for a Criterion benchmark file.
pub fn generate_benchmark_file(variants: &[(String, ExprTree)]) -> String {
    let mut code = String::new();

    // Header
    code.push_str(
        r#"//! Auto-generated kernels for NNUE cost model training.
//!
//! Generated by: cargo run -p pixelflow-ml --example gen_egraph_variants --features training
//!
//! DO NOT EDIT MANUALLY - regenerate with the command above.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_core::{Field, Manifold};
use pixelflow_macros::kernel;

fn bench_generated_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("generated_kernels");
    group.sample_size(100);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

"#,
    );

    // Generate each kernel benchmark
    for (name, tree) in variants {
        let kernel_body = expr_tree_to_kernel_body(tree);
        let node_count = tree.node_count();
        let depth = tree.depth();

        code.push_str(&format!(
            r#"    // {name} - {node_count} nodes, depth {depth}
    {{
        let k = kernel!(|| {kernel_body});
        let m = k();
        group.bench_function("{name}", |b| {{
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        }});
    }}

"#
        ));
    }

    // Footer
    code.push_str(
        r#"    group.finish();
}

criterion_group!(
    name = generated;
    config = Criterion::default().sample_size(100);
    targets = bench_generated_kernels,
);

criterion_main!(generated);
"#,
    );

    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_tree_to_kernel_body_var() {
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Var(0)), "X");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Var(1)), "Y");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Var(2)), "Z");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Var(3)), "W");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Var(4)), "V4");
    }

    #[test]
    fn test_expr_tree_to_kernel_body_const() {
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Const(0.0)), "0.0");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Const(1.0)), "1.0");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Const(-1.0)), "(-1.0)");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::Const(2.0)), "2.0");
    }

    #[test]
    fn test_expr_tree_to_kernel_body_unary() {
        let x = Box::new(ExprTree::Var(0));
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::Neg(x.clone())),
            "(-X)"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::Sqrt(x.clone())),
            "(X).sqrt()"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::Abs(x.clone())),
            "(X).abs()"
        );
    }

    #[test]
    fn test_expr_tree_to_kernel_body_binary() {
        let x = Box::new(ExprTree::Var(0));
        let y = Box::new(ExprTree::Var(1));

        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::Add(x.clone(), y.clone())),
            "(X + Y)"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::Mul(x.clone(), y.clone())),
            "(X * Y)"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::Min(x.clone(), y.clone())),
            "(X).min(Y)"
        );
    }

    #[test]
    fn test_expr_tree_to_kernel_body_nested() {
        // (X + Y) * Z
        let tree = ExprTree::Mul(
            Box::new(ExprTree::Add(
                Box::new(ExprTree::Var(0)),
                Box::new(ExprTree::Var(1)),
            )),
            Box::new(ExprTree::Var(2)),
        );
        assert_eq!(expr_tree_to_kernel_body(&tree), "((X + Y) * Z)");
    }

    #[test]
    fn test_expr_tree_to_kernel_body_mul_add() {
        let tree = ExprTree::MulAdd(
            Box::new(ExprTree::Var(0)),
            Box::new(ExprTree::Var(1)),
            Box::new(ExprTree::Var(2)),
        );
        assert_eq!(expr_tree_to_kernel_body(&tree), "(X).mul_add(Y, Z)");
    }

    #[test]
    fn test_expr_tree_to_kernel_code() {
        let tree = ExprTree::Add(
            Box::new(ExprTree::Var(0)),
            Box::new(ExprTree::Const(1.0)),
        );
        let code = expr_tree_to_kernel_code(&tree, "my_kernel");
        assert_eq!(code, "let my_kernel = kernel!(|| (X + 1.0));");
    }

    #[test]
    fn test_generate_benchmark_file() {
        let variants = vec![
            ("k0".to_string(), ExprTree::Var(0)),
            (
                "k1".to_string(),
                ExprTree::Add(
                    Box::new(ExprTree::Var(0)),
                    Box::new(ExprTree::Var(1)),
                ),
            ),
        ];

        let code = generate_benchmark_file(&variants);

        // Check key parts are present
        assert!(code.contains("criterion_group!"));
        assert!(code.contains("criterion_main!"));
        assert!(code.contains("kernel!(|| X)"));
        assert!(code.contains("kernel!(|| (X + Y))"));
        assert!(code.contains(r#"group.bench_function("k0""#));
        assert!(code.contains(r#"group.bench_function("k1""#));
    }
}
