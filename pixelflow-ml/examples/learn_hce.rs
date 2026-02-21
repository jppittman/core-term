//! # Learn the HCE with Factored NNUE
//!
//! Generate 10K synthetic samples using the Hand-Crafted Evaluator (HCE) as ground truth,
//! then train the factored NNUE to match it.
//!
//! This proves whether the architecture can learn a known cost model.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example learn_hce --features egraph-training --release
//! ```

use std::time::Instant;

use pixelflow_ml::evaluator::{
    HandCraftedEvaluator, LinearFeatures, default_expr_weights, extract_expr_features,
};
use pixelflow_ml::training::factored::{FactoredTrainer, TrainConfig};
// Use pixelflow_ml's Expr for HCE, then convert for NNUE trainer
use pixelflow_ml::nnue::{Expr, ExprGenConfig, ExprGenerator, OpType};
use pixelflow_nnue::OpKind;

/// Convert pixelflow_ml::Expr to pixelflow_nnue::Expr
fn to_nnue_expr(expr: &Expr) -> pixelflow_nnue::Expr {
    match expr {
        Expr::Var(i) => pixelflow_nnue::Expr::Var(*i),
        Expr::Const(c) => pixelflow_nnue::Expr::Const(*c),
        Expr::Unary(op, a) => pixelflow_nnue::Expr::Unary(*op, Box::new(to_nnue_expr(a))),
        Expr::Binary(op, a, b) => {
            pixelflow_nnue::Expr::Binary(*op, Box::new(to_nnue_expr(a)), Box::new(to_nnue_expr(b)))
        }
        Expr::Ternary(op, a, b, c) => pixelflow_nnue::Expr::Ternary(
            *op,
            Box::new(to_nnue_expr(a)),
            Box::new(to_nnue_expr(b)),
            Box::new(to_nnue_expr(c)),
        ),
    }
}

fn main() {
    println!("=== Learning the HCE with Factored NNUE ===\n");

    // Get the HCE we're trying to learn
    let hce = default_expr_weights();
    println!("Target: Hand-Crafted Evaluator with weights:");
    print_hce_weights(&hce);

    // Generate training data
    let num_samples = 10_000;
    println!("\nGenerating {} random expressions...", num_samples);

    let gen_config = ExprGenConfig {
        max_depth: 5,
        leaf_prob: 0.35,
        num_vars: 4,
        include_fused: true,
    };

    let mut generator = ExprGenerator::new(42, gen_config);
    // Store both ML expr (for HCE) and NNUE expr (for training)
    let mut samples: Vec<(Expr, pixelflow_nnue::Expr, f64)> = Vec::with_capacity(num_samples);

    let gen_start = Instant::now();
    for _ in 0..num_samples {
        let ml_expr = generator.generate();
        let features = extract_expr_features(&ml_expr);
        let cost = hce.evaluate_linear(&features) as f64;
        // Clamp to reasonable range
        let cost = cost.max(1.0);
        let nnue_expr = to_nnue_expr(&ml_expr);
        samples.push((ml_expr, nnue_expr, cost));
    }
    println!("Generated in {:.2}s", gen_start.elapsed().as_secs_f64());

    // Analyze the data
    let costs: Vec<f64> = samples.iter().map(|(_, _, c)| *c).collect();
    let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_cost = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_cost: f64 = costs.iter().sum::<f64>() / costs.len() as f64;
    println!("\nCost statistics:");
    println!("  Min:  {:.1}", min_cost);
    println!("  Max:  {:.1}", max_cost);
    println!("  Mean: {:.1}", mean_cost);

    // Split into train/test (80/20)
    let split_idx = (num_samples * 8) / 10;
    let (train_samples, test_samples) = samples.split_at(split_idx);

    println!("\nTraining set: {} samples", train_samples.len());
    println!("Test set:     {} samples", test_samples.len());

    // Create trainer with latency priors
    let config = TrainConfig {
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 1e-5,
        batch_size: 64,
        epochs: 50,
        lr_decay: 0.98,
        grad_clip: 1.0,
    };

    let mut trainer = FactoredTrainer::new_with_latency_prior(config.clone(), 42);

    // Add training samples (use NNUE expr)
    for (_, nnue_expr, cost) in train_samples {
        trainer.add_sample(nnue_expr.clone(), *cost);
    }

    println!(
        "\nNetwork parameters: {}",
        pixelflow_nnue::factored::FactoredNnue::param_count()
    );

    // Initial metrics
    let initial = trainer.evaluate();
    println!("\nInitial train metrics:");
    println!("  MSE:      {:.4}", initial.mse);
    println!("  RMSE:     {:.4}", initial.rmse);
    println!("  Spearman: {:.4}", initial.spearman);

    // Training loop
    println!("\nTraining for {} epochs...\n", config.epochs);
    let train_start = Instant::now();

    for epoch in 0..config.epochs {
        let loss = trainer.train_epoch();
        if epoch % 10 == 9 || epoch == 0 {
            let metrics = trainer.evaluate();
            println!(
                "Epoch {:2}: loss={:.4}, rmse={:.4}, spearman={:.4}, lr={:.6}",
                epoch + 1,
                loss,
                metrics.rmse,
                metrics.spearman,
                trainer.current_lr
            );
        }
    }

    let train_time = train_start.elapsed();
    println!("\nTraining completed in {:.2}s", train_time.as_secs_f64());

    // Final training metrics
    let final_train = trainer.evaluate();
    println!("\nFinal train metrics:");
    println!("  MSE:      {:.4}", final_train.mse);
    println!("  RMSE:     {:.4}", final_train.rmse);
    println!("  Spearman: {:.4}", final_train.spearman);

    // Evaluate on test set
    println!("\n=== Test Set Evaluation ===\n");
    let mut test_predictions = Vec::with_capacity(test_samples.len());
    let mut test_targets = Vec::with_capacity(test_samples.len());

    for (_, nnue_expr, actual_cost) in test_samples {
        let pred_log = trainer.net.evaluate(nnue_expr);
        let pred_cost = pred_log.exp() as f64;
        test_predictions.push(pred_cost);
        test_targets.push(*actual_cost);
    }

    // Compute test metrics
    let test_mse: f64 = test_predictions
        .iter()
        .zip(test_targets.iter())
        .map(|(p, t)| {
            let log_p = (*p).ln();
            let log_t = (*t).ln();
            (log_p - log_t).powi(2)
        })
        .sum::<f64>()
        / test_samples.len() as f64;
    let test_rmse = test_mse.sqrt();
    let test_spearman = compute_spearman_f64(&test_predictions, &test_targets);

    println!("Test metrics:");
    println!("  MSE:      {:.4}", test_mse);
    println!("  RMSE:     {:.4}", test_rmse);
    println!("  Spearman: {:.4}", test_spearman);

    // Show some example predictions
    println!("\n=== Sample Predictions (Test Set) ===\n");
    println!(
        "{:50} | {:>10} | {:>10} | {:>8}",
        "Expression", "HCE", "NNUE", "Error%"
    );
    println!("{:-<50}-+-{:-<10}-+-{:-<10}-+-{:-<8}", "", "", "", "");

    for (i, (ml_expr, _, actual_cost)) in test_samples.iter().take(15).enumerate() {
        let pred_cost = test_predictions[i];
        let error_pct = ((pred_cost - actual_cost) / actual_cost * 100.0).abs();
        let expr_str = format_expr(ml_expr);
        let expr_short = if expr_str.len() > 48 {
            format!("{}...", &expr_str[..45])
        } else {
            expr_str
        };
        println!(
            "{:50} | {:10.1} | {:10.1} | {:7.1}%",
            expr_short, actual_cost, pred_cost, error_pct
        );
    }

    // Check rank ordering on simple cases
    println!("\n=== Rank Ordering Test ===\n");

    // Build test cases manually
    let test_cases: Vec<(&str, Expr, pixelflow_nnue::Expr)> = vec![
        ("Var(0)", Expr::Var(0), pixelflow_nnue::Expr::Var(0)),
        (
            "Neg(Var(0))",
            Expr::Unary(OpKind::Neg, Box::new(Expr::Var(0))),
            pixelflow_nnue::Expr::Unary(OpKind::Neg, Box::new(pixelflow_nnue::Expr::Var(0))),
        ),
        (
            "Add(Var(0), Var(1))",
            Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            pixelflow_nnue::Expr::Binary(
                OpKind::Add,
                Box::new(pixelflow_nnue::Expr::Var(0)),
                Box::new(pixelflow_nnue::Expr::Var(1)),
            ),
        ),
        (
            "Mul(Var(0), Var(1))",
            Expr::Binary(OpKind::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            pixelflow_nnue::Expr::Binary(
                OpKind::Mul,
                Box::new(pixelflow_nnue::Expr::Var(0)),
                Box::new(pixelflow_nnue::Expr::Var(1)),
            ),
        ),
        (
            "Div(Var(0), Var(1))",
            Expr::Binary(OpKind::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            pixelflow_nnue::Expr::Binary(
                OpKind::Div,
                Box::new(pixelflow_nnue::Expr::Var(0)),
                Box::new(pixelflow_nnue::Expr::Var(1)),
            ),
        ),
        (
            "Sqrt(Var(0))",
            Expr::Unary(OpKind::Sqrt, Box::new(Expr::Var(0))),
            pixelflow_nnue::Expr::Unary(OpKind::Sqrt, Box::new(pixelflow_nnue::Expr::Var(0))),
        ),
    ];

    let mut prev_hce = 0;
    let mut prev_nnue = 0.0f64;
    let mut rank_correct = 0;
    let mut rank_total = 0;

    for (name, ml_expr, nnue_expr) in &test_cases {
        let features = extract_expr_features(ml_expr);
        let hce_cost = hce.evaluate_linear(&features);
        let nnue_cost = trainer.net.predict_ns(nnue_expr) as f64;

        let hce_order_ok = hce_cost >= prev_hce;
        let nnue_order_ok = nnue_cost >= prev_nnue;

        if rank_total > 0 {
            if hce_order_ok == nnue_order_ok {
                rank_correct += 1;
            }
        }
        rank_total += 1;

        println!(
            "  {:30} | HCE: {:5} | NNUE: {:8.1}",
            name, hce_cost, nnue_cost
        );

        prev_hce = hce_cost;
        prev_nnue = nnue_cost;
    }

    let rank_pct = if rank_total > 1 {
        rank_correct as f64 / (rank_total - 1) as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "\nRank agreement: {}/{} ({:.0}%)",
        rank_correct,
        rank_total - 1,
        rank_pct
    );

    // Final verdict
    println!("\n=== Verdict ===\n");
    if test_spearman > 0.95 {
        println!(
            "✓ EXCELLENT: Factored NNUE can learn the HCE (ρ = {:.3})",
            test_spearman
        );
    } else if test_spearman > 0.85 {
        println!(
            "✓ GOOD: Factored NNUE approximates the HCE reasonably (ρ = {:.3})",
            test_spearman
        );
    } else if test_spearman > 0.70 {
        println!(
            "△ MARGINAL: Architecture works but needs tuning (ρ = {:.3})",
            test_spearman
        );
    } else {
        println!(
            "✗ POOR: Architecture may have fundamental issues (ρ = {:.3})",
            test_spearman
        );
    }
}

fn print_hce_weights(hce: &HandCraftedEvaluator) {
    let names = [
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "sqrt",
        "rsqrt",
        "abs",
        "min",
        "max",
        "fma",
        "mul_rsqrt",
        "nodes",
        "depth",
        "vars",
        "consts",
        "identity",
        "self_cancel",
        "fusable",
        "crit_path",
        "max_width",
    ];
    for (i, name) in names.iter().enumerate() {
        if hce.get_weight(i) != 0 {
            println!("  {:12}: {:3}", name, hce.get_weight(i));
        }
    }
}

fn compute_spearman_f64(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len();
    if n < 2 {
        return 0.0;
    }

    let pred_ranks = compute_ranks_f64(predictions);
    let target_ranks = compute_ranks_f64(targets);

    let mean_pred: f64 = pred_ranks.iter().sum::<f64>() / n as f64;
    let mean_target: f64 = target_ranks.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_pred = 0.0;
    let mut var_target = 0.0;

    for i in 0..n {
        let dp = pred_ranks[i] - mean_pred;
        let dt = target_ranks[i] - mean_target;
        cov += dp * dt;
        var_pred += dp * dp;
        var_target += dt * dt;
    }

    if var_pred < 1e-10 || var_target < 1e-10 {
        return 0.0;
    }

    cov / (var_pred.sqrt() * var_target.sqrt())
}

fn compute_ranks_f64(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<_> = values.iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    for (rank, (orig_idx, _)) in indexed.into_iter().enumerate() {
        ranks[orig_idx] = rank as f64 + 1.0;
    }
    ranks
}

fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Var(i) => format!("Var({})", i),
        Expr::Const(c) => format!("Const({:.2})", c),
        Expr::Unary(op, a) => format!("{}({})", op_name(*op), format_expr(a)),
        Expr::Binary(op, a, b) => {
            format!("{}({}, {})", op_name(*op), format_expr(a), format_expr(b))
        }
        Expr::Ternary(op, a, b, c) => format!(
            "{}({}, {}, {})",
            op_name(*op),
            format_expr(a),
            format_expr(b),
            format_expr(c)
        ),
    }
}

fn format_nnue_expr(expr: &pixelflow_nnue::Expr) -> String {
    match expr {
        pixelflow_nnue::Expr::Var(i) => format!("Var({})", i),
        pixelflow_nnue::Expr::Const(c) => format!("Const({:.2})", c),
        pixelflow_nnue::Expr::Unary(op, a) => format!("{}({})", op_name(*op), format_nnue_expr(a)),
        pixelflow_nnue::Expr::Binary(op, a, b) => format!(
            "{}({}, {})",
            op_name(*op),
            format_nnue_expr(a),
            format_nnue_expr(b)
        ),
        pixelflow_nnue::Expr::Ternary(op, a, b, c) => format!(
            "{}({}, {}, {})",
            op_name(*op),
            format_nnue_expr(a),
            format_nnue_expr(b),
            format_nnue_expr(c)
        ),
        pixelflow_nnue::Expr::Nary(op, children) => {
            let args = children
                .iter()
                .map(|c| format_nnue_expr(c))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", op_name(*op), args)
        }
    }
}

fn op_name(op: OpKind) -> &'static str {
    match op {
        OpKind::Add => "Add",
        OpKind::Sub => "Sub",
        OpKind::Mul => "Mul",
        OpKind::Div => "Div",
        OpKind::Neg => "Neg",
        OpKind::Sqrt => "Sqrt",
        OpKind::Rsqrt => "Rsqrt",
        OpKind::Abs => "Abs",
        OpKind::Min => "Min",
        OpKind::Max => "Max",
        OpKind::MulAdd => "MulAdd",
        OpKind::MulRsqrt => "MulRsqrt",
        _ => "Op",
    }
}
