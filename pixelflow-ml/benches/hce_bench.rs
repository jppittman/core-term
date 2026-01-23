//! Benchmarks comparing HCE-based extraction vs simple cost models.
//!
//! This benchmark suite evaluates:
//! 1. Feature extraction speed
//! 2. HCE evaluation speed
//! 3. Greedy optimization with different iteration limits
//! 4. Quality of optimization (cost reduction achieved)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pixelflow_ml::evaluator::{
    default_expr_weights, fma_optimized_weights, extract_expr_features,
};
use pixelflow_ml::hce_extractor::{HceExtractor, FeatureAccumulator, SpsaTuner, SpsaConfig};
use pixelflow_ml::nnue::{Expr, OpType, ExprGenerator, ExprGenConfig};

/// Generate a test expression of given depth.
fn generate_expr(seed: u64, max_depth: usize) -> Expr {
    let config = ExprGenConfig {
        max_depth,
        leaf_prob: 0.3,
        num_vars: 4,
        include_fused: false, // Start without fused ops
    };
    let mut generator = ExprGenerator::new(seed, config);
    generator.generate()
}

/// Benchmark feature extraction speed.
fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    for depth in [3, 5, 7] {
        let expr = generate_expr(42, depth);
        let node_count = expr.node_count();

        group.bench_with_input(
            BenchmarkId::new("extract", format!("depth{}_nodes{}", depth, node_count)),
            &expr,
            |b, expr| {
                b.iter(|| extract_expr_features(black_box(expr)))
            },
        );
    }

    group.finish();
}

/// Benchmark HCE evaluation speed.
fn bench_hce_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hce_evaluation");

    let hce_default = default_expr_weights();
    let hce_fma = fma_optimized_weights();

    for depth in [3, 5, 7] {
        let expr = generate_expr(42, depth);
        let features = extract_expr_features(&expr);

        group.bench_with_input(
            BenchmarkId::new("default_weights", format!("depth{}", depth)),
            &features,
            |b, features| {
                b.iter(|| hce_default.evaluate_linear(black_box(features)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fma_weights", format!("depth{}", depth)),
            &features,
            |b, features| {
                b.iter(|| hce_fma.evaluate_linear(black_box(features)))
            },
        );
    }

    group.finish();
}

/// Benchmark accumulator operations (incremental updates).
fn bench_accumulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulator");

    for depth in [3, 5, 7] {
        let expr = generate_expr(42, depth);
        let old_subtree = generate_expr(123, 2);
        let new_subtree = Expr::Var(0);

        group.bench_with_input(
            BenchmarkId::new("from_expr", format!("depth{}", depth)),
            &expr,
            |b, expr| {
                b.iter(|| FeatureAccumulator::from_expr(black_box(expr)))
            },
        );

        let mut acc = FeatureAccumulator::from_expr(&expr);
        group.bench_function(
            BenchmarkId::new("apply_delta", format!("depth{}", depth)),
            |b| {
                b.iter(|| {
                    acc.apply_delta(black_box(&old_subtree), black_box(&new_subtree));
                    // Reset to avoid accumulating errors
                    acc.features = extract_expr_features(&expr);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark greedy optimization at different iteration limits.
fn bench_greedy_optimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_optimize");
    group.sample_size(50); // Fewer samples because this is slow

    let extractor = HceExtractor::with_fma();

    for max_iters in [1, 5, 10, 20] {
        // Generate expressions with obvious optimization opportunities
        for seed in [42u64, 123, 456] {
            // Create expression with identity patterns: ((x * 1) + 0)
            let expr = Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Mul,
                    Box::new(generate_expr(seed, 3)),
                    Box::new(Expr::Const(1.0)),
                )),
                Box::new(Expr::Const(0.0)),
            );

            group.bench_with_input(
                BenchmarkId::new(format!("iters{}", max_iters), format!("seed{}", seed)),
                &(expr.clone(), max_iters),
                |b, (expr, iters)| {
                    b.iter(|| extractor.greedy_optimize(black_box(expr), *iters))
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SPSA tuner iterations.
fn bench_spsa_tuner(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsa_tuner");
    group.sample_size(20); // Fewer samples because this involves many evaluations

    let config = SpsaConfig {
        max_iters: 10,
        samples_per_eval: 10,
        ..Default::default()
    };

    group.bench_function("single_step", |b| {
        let mut tuner = SpsaTuner::from_defaults(config.clone());

        // Simple loss function for benchmarking
        let loss_fn = |w: &[i32]| -> f64 {
            w.iter().map(|&x| (x as f64).powi(2)).sum()
        };

        b.iter(|| {
            tuner.step(black_box(&loss_fn))
        })
    });

    group.finish();
}

/// Compare optimization quality: measure cost reduction achieved.
fn bench_optimization_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_quality");
    group.sample_size(30);

    let extractor_default = HceExtractor::new();
    let extractor_fma = HceExtractor::with_fma();

    // Test on various expression types
    let test_cases: Vec<(&str, Expr)> = vec![
        // FMA opportunity: a * b + c
        ("fma_pattern", Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        )),
        // Identity pattern: x * 1
        ("identity_mul", Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        )),
        // Complex nested: ((x * 1) + 0) * (y - y)
        ("complex_nested", Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Mul,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Const(1.0)),
                )),
                Box::new(Expr::Const(0.0)),
            )),
            Box::new(Expr::Binary(
                OpType::Sub,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Var(1)),
            )),
        )),
    ];

    for (name, expr) in test_cases {
        // Benchmark default weights
        group.bench_with_input(
            BenchmarkId::new("default", name),
            &expr,
            |b, expr| {
                b.iter(|| {
                    let (optimized, reduction, _) = extractor_default.greedy_optimize(
                        black_box(expr),
                        10,
                    );
                    (optimized, reduction)
                })
            },
        );

        // Benchmark FMA weights
        group.bench_with_input(
            BenchmarkId::new("fma", name),
            &expr,
            |b, expr| {
                b.iter(|| {
                    let (optimized, reduction, _) = extractor_fma.greedy_optimize(
                        black_box(expr),
                        10,
                    );
                    (optimized, reduction)
                })
            },
        );
    }

    group.finish();
}

/// Measure actual cost reductions (not benchmark, just measurement).
/// Call this to print statistics about optimization quality.
#[allow(dead_code)]
pub fn print_optimization_stats() {
    println!("\n=== Optimization Quality Statistics ===\n");

    let extractor = HceExtractor::with_fma();

    // Generate 100 random expressions and optimize them
    let mut total_original_cost = 0i32;
    let mut total_optimized_cost = 0i32;
    let mut total_rewrites = 0usize;
    let mut num_improved = 0usize;

    for seed in 0..100 {
        let expr = generate_expr(seed, 5);
        let original_cost = extractor.cost(&expr);

        let (optimized, _reduction, rewrites) = extractor.greedy_optimize(&expr, 20);
        let optimized_cost = extractor.cost(&optimized);

        total_original_cost += original_cost;
        total_optimized_cost += optimized_cost;
        total_rewrites += rewrites;

        if optimized_cost < original_cost {
            num_improved += 1;
        }
    }

    let avg_reduction = (total_original_cost - total_optimized_cost) as f64 / 100.0;
    let avg_rewrites = total_rewrites as f64 / 100.0;

    println!("Expressions tested:    100");
    println!("Expressions improved:  {}", num_improved);
    println!("Average cost before:   {:.1}", total_original_cost as f64 / 100.0);
    println!("Average cost after:    {:.1}", total_optimized_cost as f64 / 100.0);
    println!("Average cost reduction: {:.1} cycles", avg_reduction);
    println!("Average rewrites applied: {:.1}", avg_rewrites);
    println!();
}

criterion_group!(
    benches,
    bench_feature_extraction,
    bench_hce_evaluation,
    bench_accumulator,
    bench_greedy_optimize,
    bench_spsa_tuner,
    bench_optimization_quality,
);

criterion_main!(benches);
