//! # Factored NNUE Training Example
//!
//! Trains a factored embedding NNUE on benchmark data.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example train_factored --features egraph-training
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use pixelflow_ml::training::factored::{parse_expr, parse_kernel_code, FactoredTrainer, TrainConfig};

fn main() {
    println!("=== Factored NNUE Training ===\n");

    // Load training data
    let data_path = "pixelflow-ml/data/benchmark_cache.jsonl";
    let samples = load_samples(data_path);

    if samples.is_empty() {
        println!("No samples found. Creating synthetic data for demo...\n");
        run_synthetic_demo();
        return;
    }

    println!("Loaded {} samples from {}\n", samples.len(), data_path);

    // Create trainer
    let config = TrainConfig {
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 1e-5,
        batch_size: 32,
        epochs: 20,
        lr_decay: 0.95,
        grad_clip: 1.0,
    };

    let mut trainer = FactoredTrainer::new(config.clone(), 42);

    // Add samples
    for (expr, cost) in samples {
        trainer.add_sample(expr, cost);
    }

    println!("Network parameters: {}", pixelflow_nnue::factored::FactoredNnue::param_count());
    println!("Memory usage: {} bytes\n", pixelflow_nnue::factored::FactoredNnue::memory_bytes());

    // Initial metrics
    let initial = trainer.evaluate();
    println!("Initial metrics:");
    println!("  MSE:      {:.4}", initial.mse);
    println!("  RMSE:     {:.4}", initial.rmse);
    println!("  Spearman: {:.4}\n", initial.spearman);

    // Training loop
    println!("Training for {} epochs...\n", config.epochs);
    let start = Instant::now();

    for epoch in 0..config.epochs {
        let loss = trainer.train_epoch();
        let metrics = trainer.evaluate();

        println!(
            "Epoch {:2}: loss={:.4}, mse={:.4}, spearman={:.4}, lr={:.6}",
            epoch + 1,
            loss,
            metrics.mse,
            metrics.spearman,
            trainer.current_lr
        );
    }

    let elapsed = start.elapsed();
    println!("\nTraining completed in {:.2}s\n", elapsed.as_secs_f64());

    // Final metrics
    let final_metrics = trainer.evaluate();
    println!("Final metrics:");
    println!("  MSE:      {:.4}", final_metrics.mse);
    println!("  RMSE:     {:.4}", final_metrics.rmse);
    println!("  Spearman: {:.4}", final_metrics.spearman);

    // Compare to initial
    let improvement = (1.0 - final_metrics.mse / initial.mse) * 100.0;
    println!("\nMSE improvement: {:.1}%", improvement);

    // Test some predictions
    println!("\n=== Sample Predictions ===\n");
    let test_exprs = [
        "Var(0)",
        "Add(Var(0), Var(1))",
        "Mul(Var(0), Var(1))",
        "Div(Var(0), Var(1))",
        "MulAdd(Var(0), Var(1), Var(2))",
        "Add(Mul(Var(0), Var(1)), Var(2))",
        "Sqrt(Var(0))",
    ];

    for expr_str in &test_exprs {
        if let Some(expr) = parse_expr(expr_str) {
            let pred_ns = trainer.net.predict_ns(&expr);
            println!("  {:40} -> {:.1} ns (log: {:.2})", expr_str, pred_ns, pred_ns.ln());
        }
    }
}

/// Load samples from JSONL file.
fn load_samples(path: &str) -> Vec<(pixelflow_nnue::Expr, f64)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Could not open {}: {}", path, e);
            return Vec::new();
        }
    };

    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    let mut parse_failures = 0;
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        // Skip comment lines and empty lines
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        // Parse JSON sample with kernel code syntax
        if let Some(sample) = parse_json_sample(&line) {
            samples.push(sample);
        } else {
            parse_failures += 1;
            if parse_failures <= 5 {
                // Debug: show what we're trying to parse
                if let Some(expr_str) = extract_expression(&line) {
                    eprintln!("Failed expr: {} (from line)", expr_str);
                } else {
                    eprintln!("Failed JSON: {}", &line[..line.len().min(80)]);
                }
            }
        }
    }

    if parse_failures > 0 {
        eprintln!("Total parse failures: {}", parse_failures);
    }

    samples
}

/// Extract expression string from JSON (for debugging)
fn extract_expression(line: &str) -> Option<&str> {
    let expr_key = "\"expression\":\"";
    let expr_start = line.find(expr_key)? + expr_key.len();
    let expr_end = line[expr_start..].find('"')? + expr_start;
    Some(&line[expr_start..expr_end])
}

/// Parse a JSON sample line.
///
/// Format: {"expression":"(X + Y)","cost_ns":1.23,...}
fn parse_json_sample(line: &str) -> Option<(pixelflow_nnue::Expr, f64)> {
    // Find expression field - look for the pattern: "expression":"value"
    let expr_key = "\"expression\":\"";
    let expr_start = line.find(expr_key)? + expr_key.len();
    let expr_end = line[expr_start..].find('"')? + expr_start;
    let expr_str = &line[expr_start..expr_end];

    // Find cost_ns field
    let cost_start = line.find("\"cost_ns\":")?;
    let cost_value_start = cost_start + "\"cost_ns\":".len();
    let rest = line[cost_value_start..].trim();
    let cost_end = rest.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(rest.len());
    let cost: f64 = rest[..cost_end].parse().ok()?;

    // Use kernel code parser for DSL syntax like "(X + Y)"
    let expr = parse_kernel_code(expr_str)?;
    Some((expr, cost))
}

/// Run a demo with synthetic data.
fn run_synthetic_demo() {
    println!("=== Synthetic Data Demo ===\n");

    let config = TrainConfig {
        learning_rate: 0.05,
        momentum: 0.9,
        batch_size: 8,
        epochs: 50,
        lr_decay: 0.98,
        ..Default::default()
    };

    // Use latency-prior initialization instead of random
    let mut trainer = FactoredTrainer::new_with_latency_prior(config.clone(), 42);

    // Add synthetic samples with realistic cost relationships
    let synthetic_data = [
        ("Var(0)", 5.0),
        ("Const(1.0)", 5.0),
        ("Add(Var(0), Var(1))", 20.0),
        ("Sub(Var(0), Var(1))", 20.0),
        ("Mul(Var(0), Var(1))", 25.0),
        ("Div(Var(0), Var(1))", 80.0),
        ("Neg(Var(0))", 10.0),
        ("Abs(Var(0))", 10.0),
        ("Sqrt(Var(0))", 75.0),
        ("Rsqrt(Var(0))", 30.0),
        ("Min(Var(0), Var(1))", 20.0),
        ("Max(Var(0), Var(1))", 20.0),
        ("Add(Mul(Var(0), Var(1)), Var(2))", 50.0),  // FMA pattern
        ("MulAdd(Var(0), Var(1), Var(2))", 30.0),    // Fused FMA
        ("Mul(Var(0), Rsqrt(Var(1)))", 55.0),
        ("Add(Add(Var(0), Var(1)), Var(2))", 45.0),
        ("Mul(Mul(Var(0), Var(1)), Var(2))", 55.0),
        ("Div(Var(0), Sqrt(Var(1)))", 150.0),
        ("Add(Mul(Var(0), Var(1)), Mul(Var(2), Var(3)))", 80.0),
    ];

    for (expr_str, cost) in &synthetic_data {
        if let Some(expr) = parse_expr(expr_str) {
            trainer.add_sample(expr, *cost);
        }
    }

    println!("Training on {} synthetic samples...\n", trainer.samples.len());

    let initial = trainer.evaluate();
    println!("Initial: MSE={:.2}, Spearman={:.3}\n", initial.mse, initial.spearman);

    for epoch in 0..config.epochs {
        let loss = trainer.train_epoch();
        if epoch % 10 == 9 {
            let m = trainer.evaluate();
            println!("Epoch {:2}: loss={:.3}, spearman={:.3}", epoch + 1, loss, m.spearman);
        }
    }

    let final_m = trainer.evaluate();
    println!("\nFinal: MSE={:.2}, Spearman={:.3}", final_m.mse, final_m.spearman);

    println!("\n=== Predictions vs Ground Truth ===\n");
    for (expr_str, actual) in &synthetic_data {
        if let Some(expr) = parse_expr(expr_str) {
            let pred = trainer.net.predict_ns(&expr);
            let error = ((pred as f64 - actual) / actual * 100.0).abs();
            println!(
                "  {:45} | actual: {:6.1} | pred: {:6.1} | error: {:5.1}%",
                expr_str, actual, pred, error
            );
        }
    }

    // Check if rank ordering is learned
    println!("\n=== Rank Ordering Check ===");
    println!("(Should have: Var < Add < Mul < FMA_unfused < Div < Sqrt)\n");

    let order_check = [
        ("Var(0)", "cheapest"),
        ("Add(Var(0), Var(1))", "cheap"),
        ("Mul(Var(0), Var(1))", "medium"),
        ("Add(Mul(Var(0), Var(1)), Var(2))", "unfused FMA"),
        ("Div(Var(0), Var(1))", "expensive"),
        ("Sqrt(Var(0))", "most expensive"),
    ];

    let mut prev_cost = 0.0f32;
    let mut ordering_correct = true;
    for (expr_str, label) in &order_check {
        if let Some(expr) = parse_expr(expr_str) {
            let cost = trainer.net.predict_ns(&expr);
            let correct = cost >= prev_cost;
            ordering_correct &= correct;
            println!(
                "  {:6.1} ns ({:15}) {}",
                cost,
                label,
                if correct { "✓" } else { "✗ (ordering wrong)" }
            );
            prev_cost = cost;
        }
    }

    println!("\nRank ordering {}!", if ordering_correct { "learned correctly" } else { "needs more training" });
}
