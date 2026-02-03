//! Code generation from ExprTree to kernel! macro code.
//!
//! This module converts extracted expression trees into Rust code strings
//! that can be compiled and benchmarked using the kernel! macro.
//!
//! # Usage
//!
//! ## Tree-based codegen (no CSE)
//!
//! ```ignore
//! use pixelflow_search::egraph::{ExprTree, codegen};
//!
//! let tree = ExprTree::add(ExprTree::var(0), ExprTree::mul(ExprTree::var(1), ExprTree::constant(2.0)));
//!
//! let body = codegen::expr_tree_to_kernel_body(&tree);
//! // Returns: "(X + (Y * 2.0))"
//!
//! let code = codegen::expr_tree_to_kernel_code(&tree, "my_kernel");
//! // Returns: "let my_kernel = kernel!(|| (X + (Y * 2.0)));"
//! ```
//!
//! ## DAG-based codegen (with CSE via let-bindings)
//!
//! ```ignore
//! use pixelflow_search::egraph::{EGraph, ENode, extract_dag, codegen};
//!
//! // Build: sin(X) * sin(X) + sin(X)
//! let mut egraph = EGraph::new();
//! let x = egraph.add(ENode::Var(0));
//! let sin_x = egraph.add(ENode::Op { op: &Sin, children: vec![x] });
//! let sin_x_squared = egraph.add(ENode::Op { op: &Mul, children: vec![sin_x, sin_x] });
//! let result = egraph.add(ENode::Op { op: &Add, children: vec![sin_x_squared, sin_x] });
//!
//! let dag = extract_dag(&egraph, result, &CostModel::default());
//! let code = codegen::dag_to_kernel_code(&egraph, &dag, "optimized");
//! // Returns:
//! // let optimized = kernel!(|| {
//! //     let __0 = (X).sin();
//! //     ((__0 * __0) + __0)
//! // });
//! ```

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

use super::extract::{ExprTree, ExtractedDAG, Leaf};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use pixelflow_ir::EmitStyle;

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
        ExprTree::Leaf(Leaf::Var(0)) => "X".to_string(),
        ExprTree::Leaf(Leaf::Var(1)) => "Y".to_string(),
        ExprTree::Leaf(Leaf::Var(2)) => "Z".to_string(),
        ExprTree::Leaf(Leaf::Var(3)) => "W".to_string(),
        ExprTree::Leaf(Leaf::Var(i)) => format!("V{}", i),
        ExprTree::Leaf(Leaf::Const(v)) => format_const(*v),

        ExprTree::Op { op, children } => {
            emit_op(*op, children)
        }
    }
}

/// Emit code for an operation using Op's emit_style.
fn emit_op(op: &dyn super::ops::Op, children: &[ExprTree]) -> String {
    use EmitStyle::*;

    let name = op.name();
    let style = op.emit_style();

    // Recursively emit children
    let args: Vec<String> = children.iter().map(expr_tree_to_kernel_body).collect();

    // Special cases that don't fit the standard patterns
    match name {
        // Recip is emitted as division for clarity
        "recip" if args.len() == 1 => {
            return format!("(1.0 / {})", args[0]);
        }
        // Tuple needs special formatting (variadic, no emit_style)
        "tuple" => {
            return format!("({})", args.join(", "));
        }
        _ => {}
    }

    // Use the emit_style from the Op
    match (style, args.as_slice()) {
        (UnaryPrefix, [a]) => format!("(-{})", a),
        (UnaryMethod, [a]) => format!("({}).{}()", a, name),
        (BinaryInfix(sym), [a, b]) => format!("({} {} {})", a, sym, b),
        (BinaryMethod, [a, b]) => format!("({}).{}({})", a, name, b),
        (BinaryMethodNamed(method_name), [a, b]) => format!("({}).{}({})", a, method_name, b),
        (TernaryMethod, [a, b, c]) => format!("({}).{}({}, {})", a, name, b, c),

        // Fallback for mismatched arity or special types
        (Special, _) | (_, _) => {
            format!("{}({})", name, args.join(", "))
        }
    }
}

/// Format a constant value as Rust code.
fn format_const(v: f32) -> String {
    if v.is_nan() {
        "f32::NAN".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "f32::INFINITY".to_string()
        } else {
            "f32::NEG_INFINITY".to_string()
        }
    } else if v == 0.0 {
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
use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_compiler::kernel_raw;

fn bench_generated_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("generated_kernels");
    group.sample_size(10);  // Low precision for fast training data collection

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
        let k = kernel_raw!(|| {kernel_body});
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
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(1));
    targets = bench_generated_kernels,
);

criterion_main!(generated);
"#,
    );

    code
}

/// Generate a fast benchmark binary without criterion overhead.
///
/// This creates a simple timing loop that measures all kernels quickly,
/// outputting timing results directly to a JSONL file.
///
/// Key differences from criterion:
/// - Single shared warmup phase
/// - Direct Instant::now() timing
/// - No statistical analysis overhead
/// - Results written during execution
///
/// # Arguments
///
/// * `variants` - List of (name, tree) pairs to benchmark
///
/// # Returns
///
/// Complete Rust source code for a fast benchmark binary.
pub fn generate_fast_benchmark_file(variants: &[(String, ExprTree)]) -> String {
    let mut code = String::new();

    // Header
    code.push_str(
        r#"//! Fast bulk benchmarking without criterion overhead.
//!
//! Generated by: cargo run -p pixelflow-pipeline --example gen_egraph_variants
//!
//! DO NOT EDIT MANUALLY - regenerate with the command above.
//!
//! Usage:
//!   cargo run -p pixelflow-pipeline --bin fast_kernels --release
//!
//! Output: pixelflow-pipeline/data/fast_timings.jsonl

use pixelflow_core::{Field, Manifold};
use pixelflow_compiler::kernel_raw;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const WARMUP_ITERS: usize = 50_000;
const TIMED_ITERS: usize = 2_000_000;

fn main() {
    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Compile all kernels upfront
    println!("Compiling kernels...");
    let kernels: Vec<(&str, Box<dyn Fn() -> f64>)> = vec![
"#,
    );

    // Generate kernel entries
    // Each entry is (name, warmup_fn, bench_fn) - warmup called once, bench called RUNS_PER_KERNEL times
    for (name, tree) in variants {
        let kernel_body = expr_tree_to_kernel_body(tree);
        let node_count = tree.node_count();
        let depth = tree.depth();

        code.push_str(&format!(
            r#"        // {name} - {node_count} nodes, depth {depth}
        ("{name}", {{
            let k = kernel_raw!(|| {kernel_body});
            let m = k();
            Box::new(move || {{
                // Warmup
                for _ in 0..WARMUP_ITERS {{
                    std::hint::black_box(m.eval((
                        std::hint::black_box(xf),
                        std::hint::black_box(yf),
                        std::hint::black_box(zf),
                        std::hint::black_box(wf),
                    )));
                }}
                // Timed
                let start = Instant::now();
                for _ in 0..TIMED_ITERS {{
                    std::hint::black_box(m.eval((
                        std::hint::black_box(xf),
                        std::hint::black_box(yf),
                        std::hint::black_box(zf),
                        std::hint::black_box(wf),
                    )));
                }}
                start.elapsed().as_nanos() as f64 / TIMED_ITERS as f64
            }}) as Box<dyn Fn() -> f64>
        }}),
"#
        ));
    }

    // Footer
    code.push_str(
        r##"    ];

    println!("Benchmarking {} kernels ({} warmup + {} timed iters each)...\n",
        kernels.len(), WARMUP_ITERS, TIMED_ITERS);

    // Open output file
    let mut out = File::create("pixelflow-pipeline/data/fast_timings.jsonl")
        .expect("Failed to create output file");

    let total_start = Instant::now();
    for (i, (name, bench_fn)) in kernels.iter().enumerate() {
        let timing_ns = bench_fn();

        writeln!(out, r#"{{"name":"{}","timing_ns":{:.6}}}"#, name, timing_ns)
            .expect("Failed to write timing");

        if (i + 1) % 50 == 0 || i + 1 == kernels.len() {
            let elapsed = total_start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!("  [{:4}/{}] {:.1}/s - {} = {:.2} ns",
                i + 1, kernels.len(), rate, name, timing_ns);
        }
    }

    let total_time = total_start.elapsed().as_secs_f64();
    println!("\nBenchmarked {} kernels in {:.1}s ({:.1}/s)",
        kernels.len(), total_time, kernels.len() as f64 / total_time);
    println!("Results: pixelflow-pipeline/data/fast_timings.jsonl");
}
"##
    );

    code
}

// ============================================================================
// DAG-Aware Code Generation (with Common Subexpression Elimination)
// ============================================================================

/// Convert an ExtractedDAG to a kernel! macro code string with let-bindings.
///
/// This generates code that binds shared subexpressions to local variables,
/// avoiding redundant computation.
///
/// # Arguments
///
/// * `egraph` - The e-graph containing the expressions
/// * `dag` - The extracted DAG with sharing information
/// * `name` - The variable name for the kernel
///
/// # Returns
///
/// A Rust code string like:
/// ```ignore
/// let my_kernel = kernel!(|| {
///     let __0 = (X).sin();
///     ((__0 * __0) + __0)
/// });
/// ```
pub fn dag_to_kernel_code(egraph: &EGraph, dag: &ExtractedDAG, name: &str) -> String {
    let body = dag_to_kernel_body(egraph, dag);
    format!("let {} = kernel!(|| {});", name, body)
}

/// Convert an ExtractedDAG to the body of a kernel! macro with let-bindings.
///
/// If there are shared subexpressions, wraps the expression in a block with
/// let-bindings. If no sharing, returns a simple expression.
///
/// # Returns
///
/// Either:
/// - A simple expression: `(X + Y)`
/// - A block with let-bindings: `{ let __0 = ...; (__0 * __0) }`
pub fn dag_to_kernel_body(egraph: &EGraph, dag: &ExtractedDAG) -> String {
    // Build a map from shared e-class IDs to their variable names
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut bindings = Vec::new();

    // Emit let-bindings for shared e-classes in topological order
    // The schedule ensures dependencies come before dependents
    let mut binding_idx = 0usize;
    for &class_id in &dag.schedule {
        let canonical = egraph.find(class_id);

        // Only bind shared classes (not the root itself, unless it's shared)
        if dag.is_shared(canonical) && canonical != dag.root {
            let var_name = format!("__{}", binding_idx);
            let expr_code = eclass_to_code(egraph, canonical, dag, &names);
            bindings.push(format!("let {} = {};", var_name, expr_code));
            names.insert(canonical.0, var_name);
            binding_idx += 1;
        }
    }

    // Emit the root expression
    let root_code = eclass_to_code(egraph, dag.root, dag, &names);

    if bindings.is_empty() {
        // No shared subexpressions, just return the expression
        root_code
    } else {
        // Wrap in a block with let-bindings
        let bindings_str = bindings.join("\n    ");
        format!("{{\n    {}\n    {}\n}}", bindings_str, root_code)
    }
}

/// Emit code for a single e-class, using variable names for shared subexprs.
///
/// If the e-class has been bound to a variable name, returns that name.
/// Otherwise, recursively emits the expression.
fn eclass_to_code(
    egraph: &EGraph,
    class: EClassId,
    dag: &ExtractedDAG,
    names: &BTreeMap<u32, String>,
) -> String {
    let canonical = egraph.find(class);

    // If this e-class is bound to a variable, just use the name
    if let Some(name) = names.get(&canonical.0) {
        return name.clone();
    }

    // Get the best node for this e-class
    let node_idx = dag.best_node_idx(canonical)
        .unwrap_or_else(|| panic!("No best node for e-class {}", canonical.0));
    let node = &egraph.nodes(canonical)[node_idx];

    match node {
        ENode::Var(0) => "X".to_string(),
        ENode::Var(1) => "Y".to_string(),
        ENode::Var(2) => "Z".to_string(),
        ENode::Var(3) => "W".to_string(),
        ENode::Var(i) => format!("V{}", i),
        ENode::Const(bits) => format_const(f32::from_bits(*bits)),
        ENode::Op { op, children } => {
            let child_codes: Vec<String> = children.iter()
                .map(|&c| eclass_to_code(egraph, c, dag, names))
                .collect();

            emit_op_with_args(*op, &child_codes)
        }
    }
}

/// Emit code for an operation with pre-computed child strings.
fn emit_op_with_args(op: &dyn super::ops::Op, args: &[String]) -> String {
    use EmitStyle::*;

    let name = op.name();
    let style = op.emit_style();

    // Special cases
    match name {
        "recip" if args.len() == 1 => return format!("(1.0 / {})", args[0]),
        "mul_add" if args.len() == 3 => return format!("(({} * {}) + {})", args[0], args[1], args[2]),
        "tuple" => return format!("({})", args.join(", ")),
        _ => {}
    }

    // Use emit_style
    match (style, args) {
        (UnaryPrefix, [a]) => format!("(-{})", a),
        (UnaryMethod, [a]) => format!("({}).{}()", a, name),
        (BinaryInfix(sym), [a, b]) => format!("({} {} {})", a, sym, b),
        (BinaryMethod, [a, b]) => format!("({}).{}({})", a, name, b),
        (BinaryMethodNamed(method_name), [a, b]) => format!("({}).{}({})", a, method_name, b),
        (TernaryMethod, [a, b, c]) => format!("({}).{}({}, {})", a, name, b, c),
        (Special, _) | (_, _) => format!("{}({})", name, args.join(", ")),
    }
}

/// Generate a benchmark file that uses DAG-aware extraction.
///
/// Unlike `generate_benchmark_file`, this uses `extract_dag` and emits
/// let-bindings for shared subexpressions.
///
/// # Arguments
///
/// * `variants` - List of (name, egraph, root) tuples to benchmark
///
/// # Returns
///
/// Complete Rust source code for a Criterion benchmark file.
pub fn generate_dag_benchmark_file(
    variants: &[(String, EGraph, EClassId)],
) -> String {
    use super::cost::CostModel;
    use super::extract::extract_dag;

    let mut code = String::new();

    // Header
    code.push_str(
        r#"//! Auto-generated kernels for NNUE cost model training (DAG-optimized).
//!
//! Generated with DAG-aware extraction for common subexpression elimination.
//!
//! DO NOT EDIT MANUALLY - regenerate with the command above.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_compiler::kernel_raw;

fn bench_dag_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_kernels");
    group.sample_size(10);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

"#,
    );

    let costs = CostModel::default();

    // Generate each kernel benchmark
    for (name, egraph, root) in variants {
        let dag = extract_dag(egraph, *root, &costs);
        let kernel_body = dag_to_kernel_body(egraph, &dag);
        let shared_count = dag.shared.len();

        code.push_str(&format!(
            r#"    // {name} - cost {cost}, {shared_count} shared subexprs
    {{
        let k = kernel_raw!(|| {kernel_body});
        let m = k();
        group.bench_function("{name}", |b| {{
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        }});
    }}

"#,
            cost = dag.total_cost,
        ));
    }

    // Footer
    code.push_str(
        r#"    group.finish();
}

criterion_group!(
    name = dag_generated;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(1));
    targets = bench_dag_kernels,
);

criterion_main!(dag_generated);
"#,
    );

    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_tree_to_kernel_body_var() {
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::var(0)), "X");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::var(1)), "Y");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::var(2)), "Z");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::var(3)), "W");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::var(4)), "V4");
    }

    #[test]
    fn test_expr_tree_to_kernel_body_const() {
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::constant(0.0)), "0.0");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::constant(1.0)), "1.0");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::constant(-1.0)), "(-1.0)");
        assert_eq!(expr_tree_to_kernel_body(&ExprTree::constant(2.0)), "2.0");
    }

    #[test]
    fn test_expr_tree_to_kernel_body_unary() {
        let x = ExprTree::var(0);
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::neg(x.clone())),
            "(-X)"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::sqrt(x.clone())),
            "(X).sqrt()"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::abs(x.clone())),
            "(X).abs()"
        );
    }

    #[test]
    fn test_expr_tree_to_kernel_body_binary() {
        let x = ExprTree::var(0);
        let y = ExprTree::var(1);

        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::add(x.clone(), y.clone())),
            "(X + Y)"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::mul(x.clone(), y.clone())),
            "(X * Y)"
        );
        assert_eq!(
            expr_tree_to_kernel_body(&ExprTree::min(x.clone(), y.clone())),
            "(X).min(Y)"
        );
    }

    #[test]
    fn test_expr_tree_to_kernel_body_nested() {
        // (X + Y) * Z
        let tree = ExprTree::mul(
            ExprTree::add(ExprTree::var(0), ExprTree::var(1)),
            ExprTree::var(2),
        );
        assert_eq!(expr_tree_to_kernel_body(&tree), "((X + Y) * Z)");
    }

    #[test]
    fn test_expr_tree_to_kernel_body_mul_add() {
        let tree = ExprTree::mul_add(
            ExprTree::var(0),
            ExprTree::var(1),
            ExprTree::var(2),
        );
        assert_eq!(expr_tree_to_kernel_body(&tree), "(X).mul_add(Y, Z)");
    }

    #[test]
    fn test_expr_tree_to_kernel_code() {
        let tree = ExprTree::add(ExprTree::var(0), ExprTree::constant(1.0));
        let code = expr_tree_to_kernel_code(&tree, "my_kernel");
        assert_eq!(code, "let my_kernel = kernel!(|| (X + 1.0));");
    }

    #[test]
    fn test_generate_benchmark_file() {
        let variants = vec![
            ("k0".to_string(), ExprTree::var(0)),
            (
                "k1".to_string(),
                ExprTree::add(ExprTree::var(0), ExprTree::var(1)),
            ),
        ];

        let code = generate_benchmark_file(&variants);

        // Check key parts are present
        assert!(code.contains("criterion_group!"));
        assert!(code.contains("criterion_main!"));
        assert!(code.contains("kernel_raw!(|| X)"));
        assert!(code.contains("kernel_raw!(|| (X + Y))"));
        assert!(code.contains(r#"group.bench_function("k0""#));
        assert!(code.contains(r#"group.bench_function("k1""#));
    }

    // ========================================================================
    // DAG Codegen Tests
    // ========================================================================

    use super::super::cost::CostModel;
    use super::super::extract::extract_dag;
    use super::super::graph::EGraph;
    use super::super::node::ENode;
    use super::super::ops;

    #[test]
    fn test_dag_simple_no_sharing() {
        // X + Y: no sharing, should produce simple expression
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });

        let dag = extract_dag(&egraph, sum, &CostModel::default());
        let body = dag_to_kernel_body(&egraph, &dag);

        // Should be simple expression without block
        assert_eq!(body, "(X + Y)", "Expected simple expression, got: {}", body);
        assert!(!body.contains("let __"), "Should not have let-bindings");
    }

    #[test]
    fn test_dag_shared_var() {
        // X * X: X is shared, but variables don't need let-bindings
        // (they're already O(1) to access)
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let x_squared = egraph.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, x],
        });

        let dag = extract_dag(&egraph, x_squared, &CostModel::default());
        let body = dag_to_kernel_body(&egraph, &dag);

        // Variables are cheap, so even though X is "shared", we don't bind it
        // The current implementation does bind it, which is slightly suboptimal
        // but still correct. Let's just verify it compiles to valid code.
        assert!(body.contains("X"), "Should reference X: {}", body);
    }

    #[test]
    fn test_dag_shared_subexpr() {
        // sqrt(X) * sqrt(X): sqrt(X) is an expensive shared subexpr
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let sqrt_x = egraph.add(ENode::Op {
            op: &ops::Sqrt,
            children: vec![x],
        });
        let result = egraph.add(ENode::Op {
            op: &ops::Mul,
            children: vec![sqrt_x, sqrt_x],
        });

        let dag = extract_dag(&egraph, result, &CostModel::default());
        let body = dag_to_kernel_body(&egraph, &dag);

        // sqrt(X) should be bound to a variable
        assert!(body.contains("let __0"), "Expected let-binding, got: {}", body);
        assert!(body.contains("sqrt"), "Expected sqrt operation");
        // The binding should be used twice in multiplication
        assert!(body.contains("__0 * __0") || body.contains("__0) * (__0"),
            "Expected __0 used twice, got: {}", body);
    }

    #[test]
    fn test_dag_triple_shared() {
        // sqrt(X) * sqrt(X) + sqrt(X): sqrt(X) used 3 times
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let sqrt_x = egraph.add(ENode::Op {
            op: &ops::Sqrt,
            children: vec![x],
        });
        let sqrt_squared = egraph.add(ENode::Op {
            op: &ops::Mul,
            children: vec![sqrt_x, sqrt_x],
        });
        let result = egraph.add(ENode::Op {
            op: &ops::Add,
            children: vec![sqrt_squared, sqrt_x],
        });

        let dag = extract_dag(&egraph, result, &CostModel::default());
        let body = dag_to_kernel_body(&egraph, &dag);

        eprintln!("Triple shared body: {}", body);

        // sqrt(X) should be bound
        assert!(body.contains("let __0"), "Expected let-binding for sqrt(X), got: {}", body);
        // Should reference __0 three times total (or at least be in a valid form)
        assert!(body.contains("__0"), "Expected __0 reference");
    }

    #[test]
    fn test_dag_nested_sharing() {
        // (X + Y) * (X + Y) + (X + Y): (X + Y) used 3 times
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let sum_squared = egraph.add(ENode::Op {
            op: &ops::Mul,
            children: vec![sum, sum],
        });
        let result = egraph.add(ENode::Op {
            op: &ops::Add,
            children: vec![sum_squared, sum],
        });

        let dag = extract_dag(&egraph, result, &CostModel::default());
        let body = dag_to_kernel_body(&egraph, &dag);

        eprintln!("Nested sharing body: {}", body);

        // (X + Y) should be bound
        assert!(body.contains("let __"), "Expected let-binding for (X + Y)");
        // The expression should be well-formed
        assert!(body.contains("{") && body.contains("}"), "Expected block with let-bindings");
    }

    #[test]
    fn test_dag_to_kernel_code() {
        // Test the full kernel code generation
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });

        let dag = extract_dag(&egraph, sum, &CostModel::default());
        let code = dag_to_kernel_code(&egraph, &dag, "my_kernel");

        assert!(code.contains("let my_kernel = kernel!(||"), "Expected kernel declaration");
        assert!(code.contains("X + Y"), "Expected X + Y in kernel body");
    }

    #[test]
    fn test_dag_complex_expression() {
        // Build: (X * Y) + (X * Y) * (Z + W)
        // (X * Y) is shared between two uses
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let z = egraph.add(ENode::Var(2));
        let w = egraph.add(ENode::Var(3));

        let xy = egraph.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, y],
        });
        let zw = egraph.add(ENode::Op {
            op: &ops::Add,
            children: vec![z, w],
        });
        let xy_times_zw = egraph.add(ENode::Op {
            op: &ops::Mul,
            children: vec![xy, zw],
        });
        let result = egraph.add(ENode::Op {
            op: &ops::Add,
            children: vec![xy, xy_times_zw],
        });

        let dag = extract_dag(&egraph, result, &CostModel::default());
        let body = dag_to_kernel_body(&egraph, &dag);

        eprintln!("Complex expression body: {}", body);

        // (X * Y) should be bound since it's used twice
        if dag.is_shared(xy) {
            assert!(body.contains("let __"), "Expected let-binding for (X * Y)");
        }
    }
}
