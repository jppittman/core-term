//! Fair benchmark: Judge predictions vs actual kernel execution
//!
//! This benchmark uses REAL kernels through the full pipeline:
//! kernel_raw! → Manifold → Field (SIMD) → LLVM
//!
//! NOT raw libm calls. This is what the Judge was trained on.

use pixelflow_core::{Field, Manifold};
use pixelflow_compiler::kernel_raw;
use pixelflow_search::egraph::{ExprTree, Leaf, ops, expr_tree_to_nnue};
use pixelflow_search::nnue::ExprNnue;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

const JUDGE_WEIGHTS: &str = "pixelflow-pipeline/data/judge.bin";
const JUDGE_META: &str = "pixelflow-pipeline/data/judge.meta.json";
const ITERATIONS: usize = 10_000_000;

/// Model metadata containing normalization parameters
#[derive(Debug, Deserialize)]
struct ModelMeta {
    target_mean: f32,
    target_std: f32,
}

/// Predict cost with proper denormalization
fn predict_tree_cost_denorm(tree: &ExprTree, nnue: &ExprNnue, meta: &ModelMeta) -> f32 {
    let expr = expr_tree_to_nnue(tree);
    // The raw output is normalized log cost
    let normalized = nnue.predict_log_cost(&expr);
    // Denormalize: actual_log = normalized * std + mean
    let log_cost = normalized * meta.target_std + meta.target_mean;
    // Convert to ns
    log_cost.exp()
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FAIR BENCHMARK: Judge vs Reality (Full Kernel Pipeline)");
    println!("  kernel_raw! → Manifold → Field (SIMD) → LLVM");
    println!("═══════════════════════════════════════════════════════════════\n");

    let judge = ExprNnue::load(Path::new(JUDGE_WEIGHTS))
        .unwrap_or_else(|e| panic!("Failed to load Judge: {}", e));

    // Load model metadata for denormalization
    let meta: ModelMeta = {
        let meta_content = fs::read_to_string(JUDGE_META)
            .unwrap_or_else(|e| panic!("Failed to load Judge metadata from {}: {}", JUDGE_META, e));
        serde_json::from_str(&meta_content)
            .unwrap_or_else(|e| panic!("Failed to parse Judge metadata: {}", e))
    };
    println!("Loaded model: mean={:.3}, std={:.3}", meta.target_mean, meta.target_std);

    println!("Iterations: {} million\n", ITERATIONS / 1_000_000);

    let mut correct = 0;
    let mut total = 0;

    // === DIVISION FAMILY ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    DIVISION FAMILY                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    correct += bench_simple_div(&judge, &meta); total += 1;
    correct += bench_div_sum_denom(&judge, &meta); total += 1;
    correct += bench_div_by_const(&judge, &meta); total += 1;

    // === NEGATION FAMILY ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    NEGATION FAMILY                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    correct += bench_simple_sub(&judge, &meta); total += 1;
    correct += bench_double_neg(&judge, &meta); total += 1;

    // === SQRT FAMILY ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      SQRT FAMILY                              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    correct += bench_rsqrt(&judge, &meta); total += 1;

    // === COMPOUND ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  COMPOUND EXPRESSIONS                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    correct += bench_soft_clamp(&judge, &meta); total += 1;
    correct += bench_distance_sq(&judge, &meta); total += 1;

    // === SUMMARY ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY: Judge got {}/{} correct ({:.0}%)",
             correct, total, 100.0 * correct as f64 / total as f64);
    println!("═══════════════════════════════════════════════════════════════");
}

// ============================================================================
// DIVISION FAMILY
// ============================================================================

fn bench_simple_div(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Simple Division: X/Y vs X*recip(Y) ━━━");

    // ExprTree for Judge prediction
    let div_tree = ExprTree::Op {
        op: &ops::Div,
        children: vec![ExprTree::var(0), ExprTree::var(1)],
    };
    let recip_tree = ExprTree::mul(
        ExprTree::var(0),
        ExprTree::Op { op: &ops::Recip, children: vec![ExprTree::var(1)] },
    );

    let div_pred = predict_tree_cost_denorm(&div_tree, judge, meta);
    let recip_pred = predict_tree_cost_denorm(&recip_tree, judge, meta);

    // Actual kernel benchmarks
    let k_div = kernel_raw!(|| X / Y);
    let k_recip = kernel_raw!(|| X * (Y).recip());

    let div_actual = bench_kernel(&k_div());
    let recip_actual = bench_kernel(&k_recip());

    print_result("X / Y", div_pred, div_actual);
    print_result("X * recip(Y)", recip_pred, recip_actual);
    check_winner(div_pred, recip_pred, div_actual, recip_actual)
}

fn bench_div_sum_denom(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Division with Sum Denominator: X/(Y+1) vs X*recip(Y+1) ━━━");

    let denom = ExprTree::add(ExprTree::var(1), ExprTree::Leaf(Leaf::Const(1.0)));
    let div_tree = ExprTree::Op {
        op: &ops::Div,
        children: vec![ExprTree::var(0), denom.clone()],
    };
    let recip_tree = ExprTree::mul(
        ExprTree::var(0),
        ExprTree::Op { op: &ops::Recip, children: vec![denom] },
    );

    let div_pred = predict_tree_cost_denorm(&div_tree, judge, meta);
    let recip_pred = predict_tree_cost_denorm(&recip_tree, judge, meta);

    let k_div = kernel_raw!(|| X / (Y + 1.0));
    let k_recip = kernel_raw!(|| X * (Y + 1.0).recip());

    let div_actual = bench_kernel(&k_div());
    let recip_actual = bench_kernel(&k_recip());

    print_result("X / (Y + 1)", div_pred, div_actual);
    print_result("X * recip(Y + 1)", recip_pred, recip_actual);
    check_winner(div_pred, recip_pred, div_actual, recip_actual)
}

fn bench_div_by_const(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Division by Constant: X/2 vs X*0.5 ━━━");

    let div_tree = ExprTree::Op {
        op: &ops::Div,
        children: vec![ExprTree::var(0), ExprTree::Leaf(Leaf::Const(2.0))],
    };
    let mul_tree = ExprTree::mul(ExprTree::var(0), ExprTree::Leaf(Leaf::Const(0.5)));

    let div_pred = predict_tree_cost_denorm(&div_tree, judge, meta);
    let mul_pred = predict_tree_cost_denorm(&mul_tree, judge, meta);

    let k_div = kernel_raw!(|| X / 2.0);
    let k_mul = kernel_raw!(|| X * 0.5);

    let div_actual = bench_kernel(&k_div());
    let mul_actual = bench_kernel(&k_mul());

    print_result("X / 2", div_pred, div_actual);
    print_result("X * 0.5", mul_pred, mul_actual);
    check_winner(div_pred, mul_pred, div_actual, mul_actual)
}

// ============================================================================
// NEGATION FAMILY
// ============================================================================

fn bench_simple_sub(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Simple Subtraction: X-Y vs X+neg(Y) ━━━");

    let sub_tree = ExprTree::Op {
        op: &ops::Sub,
        children: vec![ExprTree::var(0), ExprTree::var(1)],
    };
    let add_neg_tree = ExprTree::add(
        ExprTree::var(0),
        ExprTree::Op { op: &ops::Neg, children: vec![ExprTree::var(1)] },
    );

    let sub_pred = predict_tree_cost_denorm(&sub_tree, judge, meta);
    let add_neg_pred = predict_tree_cost_denorm(&add_neg_tree, judge, meta);

    let k_sub = kernel_raw!(|| X - Y);
    let k_add_neg = kernel_raw!(|| X + (-Y));

    let sub_actual = bench_kernel(&k_sub());
    let add_neg_actual = bench_kernel(&k_add_neg());

    print_result("X - Y", sub_pred, sub_actual);
    print_result("X + neg(Y)", add_neg_pred, add_neg_actual);
    check_winner(sub_pred, add_neg_pred, sub_actual, add_neg_actual)
}

fn bench_double_neg(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Double Negation: neg(neg(X)) vs X ━━━");

    let double_neg_tree = ExprTree::Op {
        op: &ops::Neg,
        children: vec![ExprTree::Op {
            op: &ops::Neg,
            children: vec![ExprTree::var(0)],
        }],
    };
    let identity_tree = ExprTree::var(0);

    let double_neg_pred = predict_tree_cost_denorm(&double_neg_tree, judge, meta);
    let identity_pred = predict_tree_cost_denorm(&identity_tree, judge, meta);

    let k_double_neg = kernel_raw!(|| -(-X));
    let k_identity = kernel_raw!(|| X);

    let double_neg_actual = bench_kernel(&k_double_neg());
    let identity_actual = bench_kernel(&k_identity());

    print_result("neg(neg(X))", double_neg_pred, double_neg_actual);
    print_result("X", identity_pred, identity_actual);
    check_winner(double_neg_pred, identity_pred, double_neg_actual, identity_actual)
}

// ============================================================================
// SQRT FAMILY
// ============================================================================

fn bench_rsqrt(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Reciprocal Sqrt: 1/sqrt(X) vs rsqrt(X) ━━━");

    let recip_sqrt_tree = ExprTree::Op {
        op: &ops::Div,
        children: vec![
            ExprTree::Leaf(Leaf::Const(1.0)),
            ExprTree::Op { op: &ops::Sqrt, children: vec![ExprTree::var(0)] },
        ],
    };
    let rsqrt_tree = ExprTree::Op {
        op: &ops::Rsqrt,
        children: vec![ExprTree::var(0)],
    };

    let recip_sqrt_pred = predict_tree_cost_denorm(&recip_sqrt_tree, judge, meta);
    let rsqrt_pred = predict_tree_cost_denorm(&rsqrt_tree, judge, meta);

    let k_recip_sqrt = kernel_raw!(|| 1.0 / (X).sqrt());
    let k_rsqrt = kernel_raw!(|| (X).rsqrt());

    let recip_sqrt_actual = bench_kernel(&k_recip_sqrt());
    let rsqrt_actual = bench_kernel(&k_rsqrt());

    print_result("1 / sqrt(X)", recip_sqrt_pred, recip_sqrt_actual);
    print_result("rsqrt(X)", rsqrt_pred, rsqrt_actual);
    check_winner(recip_sqrt_pred, rsqrt_pred, recip_sqrt_actual, rsqrt_actual)
}

// ============================================================================
// COMPOUND EXPRESSIONS
// ============================================================================

fn bench_soft_clamp(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Soft Clamp: X/(|X|+1) vs X*recip(|X|+1) ━━━");

    let abs_x = ExprTree::Op { op: &ops::Abs, children: vec![ExprTree::var(0)] };
    let denom = ExprTree::add(abs_x, ExprTree::Leaf(Leaf::Const(1.0)));

    let div_tree = ExprTree::Op {
        op: &ops::Div,
        children: vec![ExprTree::var(0), denom.clone()],
    };
    let recip_tree = ExprTree::mul(
        ExprTree::var(0),
        ExprTree::Op { op: &ops::Recip, children: vec![denom] },
    );

    let div_pred = predict_tree_cost_denorm(&div_tree, judge, meta);
    let recip_pred = predict_tree_cost_denorm(&recip_tree, judge, meta);

    let k_div = kernel_raw!(|| X / ((X).abs() + 1.0));
    let k_recip = kernel_raw!(|| X * ((X).abs() + 1.0).recip());

    let div_actual = bench_kernel(&k_div());
    let recip_actual = bench_kernel(&k_recip());

    print_result("X / (|X| + 1)", div_pred, div_actual);
    print_result("X * recip(|X| + 1)", recip_pred, recip_actual);
    check_winner(div_pred, recip_pred, div_actual, recip_actual)
}

fn bench_distance_sq(judge: &ExprNnue, meta: &ModelMeta) -> usize {
    println!("━━━ Distance Squared: (X-0.5)²+(Y-0.5)² forms ━━━");

    // Form A: (X-0.5)*(X-0.5) + (Y-0.5)*(Y-0.5)
    let dx = ExprTree::Op { op: &ops::Sub, children: vec![ExprTree::var(0), ExprTree::Leaf(Leaf::Const(0.5))] };
    let dy = ExprTree::Op { op: &ops::Sub, children: vec![ExprTree::var(1), ExprTree::Leaf(Leaf::Const(0.5))] };
    let sub_form = ExprTree::add(
        ExprTree::mul(dx.clone(), dx),
        ExprTree::mul(dy.clone(), dy),
    );

    // Form B: (X+neg(0.5))*(X+neg(0.5)) + (Y+neg(0.5))*(Y+neg(0.5))
    let dx_neg = ExprTree::add(ExprTree::var(0), ExprTree::Op { op: &ops::Neg, children: vec![ExprTree::Leaf(Leaf::Const(0.5))] });
    let dy_neg = ExprTree::add(ExprTree::var(1), ExprTree::Op { op: &ops::Neg, children: vec![ExprTree::Leaf(Leaf::Const(0.5))] });
    let add_neg_form = ExprTree::add(
        ExprTree::mul(dx_neg.clone(), dx_neg),
        ExprTree::mul(dy_neg.clone(), dy_neg),
    );

    let sub_pred = predict_tree_cost_denorm(&sub_form, judge, meta);
    let add_neg_pred = predict_tree_cost_denorm(&add_neg_form, judge, meta);

    let k_sub = kernel_raw!(|| (X - 0.5) * (X - 0.5) + (Y - 0.5) * (Y - 0.5));
    let k_add_neg = kernel_raw!(|| (X + (-0.5)) * (X + (-0.5)) + (Y + (-0.5)) * (Y + (-0.5)));

    let sub_actual = bench_kernel(&k_sub());
    let add_neg_actual = bench_kernel(&k_add_neg());

    print_result("(X-0.5)² + (Y-0.5)²", sub_pred, sub_actual);
    print_result("(X+neg(0.5))² + ...", add_neg_pred, add_neg_actual);
    check_winner(sub_pred, add_neg_pred, sub_actual, add_neg_actual)
}

// ============================================================================
// HELPERS
// ============================================================================

fn bench_kernel<M: Manifold<(Field, Field, Field, Field), Output = Field>>(m: &M) -> f64 {
    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Warmup
    for _ in 0..100_000 {
        std::hint::black_box(m.eval((
            std::hint::black_box(xf),
            std::hint::black_box(yf),
            std::hint::black_box(zf),
            std::hint::black_box(wf),
        )));
    }

    // Timed
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(m.eval((
            std::hint::black_box(xf),
            std::hint::black_box(yf),
            std::hint::black_box(zf),
            std::hint::black_box(wf),
        )));
    }

    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn print_result(name: &str, predicted: f32, actual: f64) {
    let ratio = actual / predicted as f64;
    println!("  {:<25} pred: {:>5.2} ns  actual: {:>5.2} ns  ratio: {:.2}x",
             name, predicted, actual, ratio);
}

fn check_winner(pred_a: f32, pred_b: f32, actual_a: f64, actual_b: f64) -> usize {
    let pred_winner = if pred_a < pred_b { "A" } else { "B" };
    let actual_winner = if actual_a < actual_b { "A" } else { "B" };
    let correct = pred_winner == actual_winner;

    // Check if within noise (5% difference)
    let diff_pct = ((actual_a - actual_b).abs() / actual_a.min(actual_b) * 100.0) as i32;
    let tie = diff_pct < 5;

    if tie {
        println!("  → TIE (within 5%): {:.2} vs {:.2} ns\n", actual_a, actual_b);
        1 // Count ties as correct
    } else {
        println!("  → Judge: {}  Reality: {}  {}\n",
                 pred_winner, actual_winner,
                 if correct { "✓ CORRECT" } else { "✗ WRONG" });
        if correct { 1 } else { 0 }
    }
}
