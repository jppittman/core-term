//! Test if NNUE actually learns from training data.
//!
//! Run with: cargo run -p pixelflow-ml --example nnue_train_test --features training

use pixelflow_ml::nnue::{Expr, OpType};
use pixelflow_ml::nnue_trainer::{NnueSample, NnueTrainer, TrainConfig};

// Helper functions for building Expr trees
fn var(i: u8) -> Expr {
    Expr::Var(i)
}
fn cnst(v: f32) -> Expr {
    Expr::Const(v)
}
fn add(a: Expr, b: Expr) -> Expr {
    Expr::Binary(OpType::Add, Box::new(a), Box::new(b))
}
fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Binary(OpType::Mul, Box::new(a), Box::new(b))
}
fn div(a: Expr, b: Expr) -> Expr {
    Expr::Binary(OpType::Div, Box::new(a), Box::new(b))
}
fn sqrt(a: Expr) -> Expr {
    Expr::Unary(OpType::Sqrt, Box::new(a))
}
fn x() -> Expr {
    var(0)
}
fn y() -> Expr {
    var(1)
}
fn z() -> Expr {
    var(2)
}
fn w() -> Expr {
    var(3)
}

fn main() {
    println!("=== NNUE Training Test ===\n");

    // Create training data from benchmark results (actual measured ns from earlier run)
    let training_data: Vec<(Expr, f32)> = vec![
        // (expression, actual runtime in ns)
        (sqrt(x()), 1.37),
        (sqrt(add(x(), y())), 1.66),
        (sqrt(add(mul(x(), x()), mul(y(), y()))), 1.88), // dist2d
        (add(sqrt(x()), sqrt(y())), 2.50),               // sqrt2_wide
        (sqrt(add(sqrt(x()), cnst(1.0))), 2.61),         // sqrt2_deep
        (add(sqrt(x()), div(y(), z())), 3.69),           // sqrt_div_wide
        (div(sqrt(x()), sqrt(y())), 2.16),               // sqrt_div_deep - FASTER than wide!
        (add(add(sqrt(x()), sqrt(y())), sqrt(z())), 3.84), // sqrt3_wide
        (sqrt(add(sqrt(add(sqrt(x()), cnst(1.0))), cnst(1.0))), 4.08), // sqrt3_deep
        (
            add(add(sqrt(x()), sqrt(y())), add(sqrt(z()), sqrt(w()))),
            5.16,
        ), // sqrt4_wide
        // Add some arithmetic kernels
        (add(x(), y()), 0.8),
        (mul(x(), y()), 0.8),
        (add(mul(x(), y()), z()), 1.0),
        (mul(add(x(), y()), z()), 1.0),
        (add(add(x(), y()), add(z(), w())), 1.2),
        (div(x(), y()), 2.5),
        (div(div(x(), y()), z()), 3.0),
    ];

    // Create samples
    let samples: Vec<NnueSample> = training_data
        .iter()
        .map(|(expr, cost)| NnueSample::from_expr(expr, *cost))
        .collect();

    println!("Training samples: {}", samples.len());

    // Create trainer
    let mut trainer = NnueTrainer::new();
    trainer.config = TrainConfig {
        learning_rate: 0.001,
        epochs: 500,
        batch_size: 8,
        l2_lambda: 0.0001,
        use_log_transform: true,
        print_every: 100,
    };

    // Add samples
    for sample in &samples {
        trainer.add_sample(sample.clone());
    }

    // Evaluate before training
    let corr_before = trainer.spearman_correlation(&samples);

    println!("\n--- Before Training ---");
    println!("Spearman correlation: {:.4}", corr_before);

    // Train
    println!("\n--- Training ---");
    let _history = trainer.train();

    // Evaluate after training
    let (preds_after, _) = trainer.evaluate(&samples);
    let corr_after = trainer.spearman_correlation(&samples);

    println!("\n--- After Training ---");
    println!(
        "Spearman correlation: {:.4} (was {:.4})",
        corr_after, corr_before
    );
    println!("Improvement: {:+.4}", corr_after - corr_before);

    println!("\nPredictions vs Targets:");
    for (i, ((expr, cost), pred)) in training_data.iter().zip(preds_after.iter()).enumerate() {
        println!(
            "  {:2}. pred={:6.3}, actual={:5.2}ns, nodes={}",
            i,
            pred,
            cost,
            expr.node_count()
        );
    }

    // Check key ranking: sqrt_div_deep should rank BETTER than sqrt_div_wide
    // (since it's actually faster: 2.16ns vs 3.69ns)
    let sqrt_div_wide_idx = 5;
    let sqrt_div_deep_idx = 6;

    println!("\n--- Key Test: sqrt_div_deep vs sqrt_div_wide ---");
    println!(
        "Actual: deep={:.2}ns, wide={:.2}ns (deep is FASTER)",
        training_data[sqrt_div_deep_idx].1, training_data[sqrt_div_wide_idx].1
    );
    println!(
        "After:  deep={:.3}, wide={:.3} ({})",
        preds_after[sqrt_div_deep_idx],
        preds_after[sqrt_div_wide_idx],
        if preds_after[sqrt_div_deep_idx] < preds_after[sqrt_div_wide_idx] {
            "CORRECT ✓"
        } else {
            "WRONG ✗"
        }
    );

    // Summary
    println!("\n=== Summary ===");
    if corr_after > 0.7 {
        println!(
            "SUCCESS: NNUE learned to predict costs (Spearman = {:.3})",
            corr_after
        );
    } else if corr_after > corr_before {
        println!(
            "PARTIAL: NNUE improved but not great (Spearman = {:.3})",
            corr_after
        );
    } else {
        println!("FAILED: NNUE did not learn (Spearman = {:.3})", corr_after);
    }
}
