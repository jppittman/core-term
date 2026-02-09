//! Large-scale expression benchmarks for HCE/NNUE validation
//!
//! Uses the Expr interpreter which LLVM cannot fully optimize away.
//! This validates whether cost models predict actual ranking correctly.
//!
//! Run with: cargo bench -p pixelflow-ml --bench expr_perf

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use pixelflow_ml::evaluator::{default_expr_weights, extract_expr_features};
use pixelflow_ml::nnue::{Expr, ExprGenConfig, ExprGenerator, OpType};

/// Generate expressions with specific structural properties
fn generate_wide_tree(depth: usize, width: usize) -> Expr {
    if depth == 0 {
        Expr::Var(0)
    } else {
        // Create a balanced tree of adds
        let mut exprs: Vec<Expr> = (0..width)
            .map(|i| {
                let child = generate_wide_tree(depth - 1, width);
                if i % 2 == 0 {
                    Expr::Binary(OpType::Mul, Box::new(child), Box::new(Expr::Const(1.001)))
                } else {
                    child
                }
            })
            .collect();

        // Combine them pairwise with adds
        while exprs.len() > 1 {
            let mut new_exprs = Vec::new();
            for chunk in exprs.chunks(2) {
                if chunk.len() == 2 {
                    new_exprs.push(Expr::Binary(
                        OpType::Add,
                        Box::new(chunk[0].clone()),
                        Box::new(chunk[1].clone()),
                    ));
                } else {
                    new_exprs.push(chunk[0].clone());
                }
            }
            exprs = new_exprs;
        }
        exprs.pop().unwrap_or(Expr::Var(0))
    }
}

fn generate_deep_chain(length: usize) -> Expr {
    let mut expr = Expr::Var(0);
    for i in 0..length {
        let op = match i % 4 {
            0 => OpType::Add,
            1 => OpType::Mul,
            2 => OpType::Sub,
            _ => OpType::Add,
        };
        let rhs = Expr::Const((i as f32) * 0.001 + 1.0);
        expr = Expr::Binary(op, Box::new(expr), Box::new(rhs));
    }
    expr
}

fn generate_with_expensive_ops(size: usize, expensive_fraction: f32) -> Expr {
    let config = ExprGenConfig {
        max_depth: 8,
        leaf_prob: 0.2,
        num_vars: 4,
        include_fused: false,
    };
    let mut expr_gen = ExprGenerator::new(42, config);

    // Generate base expression
    let mut expr = expr_gen.generate();

    // Wrap with expensive ops based on fraction
    for i in 0..size {
        let should_be_expensive = (i as f32 / size as f32) < expensive_fraction;
        if should_be_expensive {
            expr = Expr::Unary(OpType::Sqrt, Box::new(expr));
            expr = Expr::Binary(OpType::Add, Box::new(expr), Box::new(Expr::Const(1.0)));
        } else {
            expr = Expr::Binary(OpType::Add, Box::new(expr), Box::new(Expr::Const(0.001)));
        }
    }
    expr
}

/// Benchmark expression evaluation throughput
fn bench_expr_eval_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_eval_throughput");

    let vars = [1.5f32, 2.5, 3.5, 0.5];

    for (i, &target_size) in [20, 50, 100, 200].iter().enumerate() {
        let config = ExprGenConfig {
            max_depth: 12,
            leaf_prob: 0.1 + (i as f32) * 0.05, // Vary leaf_prob to get different sizes
            num_vars: 4,
            include_fused: false,
        };
        let mut expr_gen = ExprGenerator::new((i * 1000) as u64, config);

        // Generate until we hit target size
        let mut expr = expr_gen.generate();
        let mut attempts = 0;
        while expr.node_count() < target_size && attempts < 50 {
            expr = Expr::Binary(OpType::Add, Box::new(expr), Box::new(expr_gen.generate()));
            attempts += 1;
        }

        let node_count = expr.node_count();
        group.throughput(Throughput::Elements(node_count as u64));

        group.bench_with_input(
            BenchmarkId::new("nodes", format!("~{}", target_size)),
            &expr,
            |b, expr| b.iter(|| black_box(expr.eval(black_box(&vars)))),
        );
    }

    group.finish();
}

/// Compare wide vs deep expressions with same node count
fn bench_wide_vs_deep(c: &mut Criterion) {
    let mut group = c.benchmark_group("wide_vs_deep");

    let vars = [1.5f32, 2.5, 3.5, 0.5];
    let hce = default_expr_weights();

    // Test different sizes with matched node counts
    for (i, target_nodes) in [30, 60, 120].iter().enumerate() {
        // For fair comparison, generate both wide and deep with same approx node count
        // Wide: balanced binary tree of adds
        // Deep: linear chain of ops

        // Deep chain is easy - just target_nodes operations
        let deep = generate_deep_chain(*target_nodes);
        let deep_size = deep.node_count();

        // Wide tree: approximate with depth = log2(target_nodes)
        let depth = ((deep_size as f64).log2().ceil() as usize).max(2);
        let wide = generate_wide_tree(depth, 2);
        let wide_size = wide.node_count();

        // Get HCE predictions
        let wide_features = extract_expr_features(&wide);
        let deep_features = extract_expr_features(&deep);
        let wide_hce = hce.evaluate_linear(&wide_features);
        let deep_hce = hce.evaluate_linear(&deep_features);

        println!("\n=== Target ~{} nodes ===", target_nodes);
        println!(
            "Wide: {} nodes, HCE={}, critical_path={}, depth={}",
            wide_size, wide_hce, wide_features.critical_path, wide_features.depth
        );
        println!(
            "Deep: {} nodes, HCE={}, critical_path={}, depth={}",
            deep_size, deep_hce, deep_features.critical_path, deep_features.depth
        );

        group.bench_with_input(
            BenchmarkId::new(format!("wide_n{}", i), wide_size),
            &wide,
            |b, expr| b.iter(|| black_box(expr.eval(black_box(&vars)))),
        );

        group.bench_with_input(
            BenchmarkId::new(format!("deep_n{}", i), deep_size),
            &deep,
            |b, expr| b.iter(|| black_box(expr.eval(black_box(&vars)))),
        );
    }

    group.finish();
}

/// Validate HCE ranking predictions at scale
fn bench_hce_ranking_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hce_ranking");

    let vars = [1.5f32, 2.5, 3.5, 0.5];
    let hce = default_expr_weights();

    // Generate many expressions and check if HCE ordering matches runtime ordering
    let config = ExprGenConfig {
        max_depth: 6,
        leaf_prob: 0.25,
        num_vars: 4,
        include_fused: false,
    };

    let mut expressions: Vec<Expr> = Vec::new();
    for seed in 0..20 {
        let mut expr_gen = ExprGenerator::new(seed, config.clone());
        expressions.push(expr_gen.generate());
    }

    // Sort by HCE cost
    let mut with_hce: Vec<(usize, i32, &Expr)> = expressions
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let f = extract_expr_features(e);
            (i, hce.evaluate_linear(&f), e)
        })
        .collect();
    with_hce.sort_by_key(|(_, cost, _)| *cost);

    // Benchmark each expression
    println!("\n=== HCE Ranking Validation ===");
    println!(
        "Benchmarking {} expressions sorted by HCE cost...",
        expressions.len()
    );

    for (rank, (idx, hce_cost, expr)) in with_hce.iter().enumerate().take(10) {
        let features = extract_expr_features(expr);
        group.bench_with_input(
            BenchmarkId::new(
                format!("rank{:02}_hce{}", rank, hce_cost),
                expr.node_count(),
            ),
            expr,
            |b, expr| b.iter(|| black_box(expr.eval(black_box(&vars)))),
        );

        println!(
            "  Rank {}: HCE={}, nodes={}, depth={}, critical_path={}",
            rank, hce_cost, features.node_count, features.depth, features.critical_path
        );
    }

    group.finish();
}

/// Benchmark effect of expensive operations
fn bench_expensive_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("expensive_ops");

    let vars = [2.5f32, 3.5, 4.5, 1.5]; // Positive values for sqrt

    for expensive_frac in [0.0, 0.1, 0.3, 0.5] {
        let expr = generate_with_expensive_ops(50, expensive_frac);
        let features = extract_expr_features(&expr);

        println!(
            "\n{}% expensive ops: sqrt_count={}, div_count={}, critical_path={}",
            (expensive_frac * 100.0) as i32,
            features.sqrt_count,
            features.div_count,
            features.critical_path
        );

        group.bench_with_input(
            BenchmarkId::new("sqrt_fraction", format!("{:.0}%", expensive_frac * 100.0)),
            &expr,
            |b, expr| b.iter(|| black_box(expr.eval(black_box(&vars)))),
        );
    }

    group.finish();
}

criterion_group!(
    name = expr_perf_benches;
    config = Criterion::default().sample_size(100);
    targets =
        bench_expr_eval_throughput,
        bench_wide_vs_deep,
        bench_hce_ranking_validation,
        bench_expensive_ops,
);

criterion_main!(expr_perf_benches);
