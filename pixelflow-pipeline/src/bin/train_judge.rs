//! Train the Judge (Value Head) on benchmark data.
//!
//! This binary trains the NNUE value head to predict expression cost
//! from SIMD benchmark timing data using the proper training infrastructure.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin train_judge --release --features training
//! ```
//!
//! # Input
//!
//! - `pixelflow-pipeline/data/judge_training.jsonl` - Training data from collect_judge_data
//!
//! # Output
//!
//! - `pixelflow-pipeline/data/judge.bin` - Trained value head weights

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use clap::Parser;
use serde::Deserialize;

use pixelflow_search::nnue::{DualHeadNnue, Expr};
use pixelflow_pipeline::training::factored::parse_kernel_code;
use pixelflow_pipeline::training::dual_head::{
    ValueSample, DualGradients, DualForwardCache, backward_value,
};
use pixelflow_pipeline::nnue::factored::{EdgeAccumulator, StructuralFeatures};

/// Train the Judge (Value Head).
#[derive(Parser, Debug)]
#[command(name = "train_judge")]
#[command(about = "Train the Judge (Value Head) on benchmark data")]
struct Args {
    /// Path to training data
    #[arg(long, default_value = "pixelflow-pipeline/data/judge_training.jsonl")]
    input: String,

    /// Output path for trained model
    #[arg(short, long, default_value = "pixelflow-pipeline/data/judge.bin")]
    output: String,

    /// Number of training epochs
    #[arg(long, default_value_t = 100)]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f32,

    /// Momentum for SGD
    #[arg(long, default_value_t = 0.9)]
    momentum: f32,

    /// L2 regularization weight
    #[arg(long, default_value_t = 0.0001)]
    weight_decay: f32,

    /// Batch size
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Print loss every N epochs
    #[arg(long, default_value_t = 10)]
    print_every: usize,

    /// Validation split ratio (0.0 to 1.0)
    #[arg(long, default_value_t = 0.1)]
    val_split: f32,
}

/// Training sample from collect_judge_data
#[derive(Debug, Deserialize)]
struct TrainingSample {
    #[allow(dead_code)]
    name: String,
    /// Expression in kernel code syntax
    expression: String,
    /// Benchmark timing - MUST be a valid non-null float
    timing_ns: f64,
}

impl TrainingSample {
    /// Validate that the sample has valid timing data.
    fn validate(&self) -> Result<(), String> {
        if self.timing_ns.is_nan() {
            return Err(format!("{}: timing_ns is NaN", self.name));
        }
        if self.timing_ns.is_infinite() {
            return Err(format!("{}: timing_ns is infinite", self.name));
        }
        if self.timing_ns <= 0.0 {
            return Err(format!("{}: timing_ns is non-positive ({})", self.name, self.timing_ns));
        }
        if self.expression.is_empty() {
            return Err(format!("{}: expression is empty", self.name));
        }
        Ok(())
    }
}

fn main() {
    let args = Args::parse();

    // Find workspace root
    let workspace_root = find_workspace_root();

    // Load training data
    let input_path = workspace_root.join(&args.input);
    println!("Loading training data from: {}", input_path.display());

    let input_file = fs::File::open(&input_path)
        .expect("Failed to open training data. Run collect_judge_data first.");
    let reader = BufReader::new(input_file);

    let mut samples: Vec<TrainingSample> = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let sample: TrainingSample = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("Failed to parse training data at line {}: {}\nLine: {}", line_num + 1, e, line));

        // FAIL FAST: No silent failures - validate timing data immediately
        sample.validate().unwrap_or_else(|e| panic!(
            "Invalid training sample at line {}: {}\nThis likely means benchmark data wasn't collected properly.\nRun: cargo bench -p pixelflow-pipeline --bench generated_kernels",
            line_num + 1, e
        ));

        samples.push(sample);
    }
    println!("Loaded {} training samples", samples.len());

    if samples.is_empty() {
        eprintln!("No training samples found!");
        std::process::exit(1);
    }

    // Split into train/val
    let val_count = ((samples.len() as f32) * args.val_split) as usize;
    let train_count = samples.len() - val_count;

    // Shuffle with deterministic seed
    let mut rng = SimpleRng::new(args.seed);
    for i in (1..samples.len()).rev() {
        let j = rng.next() as usize % (i + 1);
        samples.swap(i, j);
    }

    let (train_samples, val_samples) = samples.split_at(train_count);
    println!("Training samples: {}", train_samples.len());
    println!("Validation samples: {}", val_samples.len());

    // Initialize NNUE
    let mut nnue = DualHeadNnue::new_with_latency_prior(args.seed);
    println!("\nInitialized DualHeadNnue ({} parameters)", DualHeadNnue::param_count());

    // Parse expressions and create ValueSamples
    let mut parse_failures = 0usize;
    let train_data: Vec<ValueSample> = train_samples.iter()
        .filter_map(|s| {
            match parse_kernel_code(&s.expression) {
                Some(expr) => Some(ValueSample::new(expr, s.timing_ns, &nnue.embeddings)),
                None => {
                    parse_failures += 1;
                    None
                }
            }
        })
        .collect();
    if parse_failures > 0 {
        println!("  Warning: {} train samples failed to parse", parse_failures);
    }

    parse_failures = 0;
    let val_data: Vec<ValueSample> = val_samples.iter()
        .filter_map(|s| {
            match parse_kernel_code(&s.expression) {
                Some(expr) => Some(ValueSample::new(expr, s.timing_ns, &nnue.embeddings)),
                None => {
                    parse_failures += 1;
                    None
                }
            }
        })
        .collect();
    if parse_failures > 0 {
        println!("  Warning: {} val samples failed to parse", parse_failures);
    }

    println!("Successfully parsed {} train, {} val samples",
        train_data.len(), val_data.len());

    if train_data.is_empty() {
        eprintln!("No training data after parsing!");
        std::process::exit(1);
    }

    // Compute target statistics for normalization
    let targets: Vec<f32> = train_data.iter().map(|s| s.target()).collect();
    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_std = (targets.iter()
        .map(|t| (t - target_mean).powi(2))
        .sum::<f32>() / targets.len() as f32)
        .sqrt();

    println!("\nTarget statistics (log ns):");
    println!("  Mean: {:.3}", target_mean);
    println!("  Std:  {:.3}", target_std);

    // Initialize momentum buffers
    let mut momentum_grads = DualGradients::new();

    // Training loop
    println!("\nTraining for {} epochs with lr={}, momentum={}, batch_size={}",
        args.epochs, args.learning_rate, args.momentum, args.batch_size);

    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = 0;

    for epoch in 0..args.epochs {
        // Shuffle training data each epoch
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.next() as usize % (i + 1);
            indices.swap(i, j);
        }

        // Training pass with mini-batches
        let mut train_loss = 0.0f32;
        let mut batch_grads = DualGradients::new();

        for (step, &idx) in indices.iter().enumerate() {
            let sample = &train_data[idx];

            // Forward pass
            let acc = EdgeAccumulator::from_expr(&sample.expr, &nnue.embeddings);
            let structural = StructuralFeatures::from_expr(&sample.expr);
            let cache = DualForwardCache::forward(&nnue, &acc, &structural);

            // Normalize target
            let normalized_target = (sample.target() - target_mean) / target_std.max(1e-6);

            // Backward pass (value head only)
            let loss = backward_value(&nnue, &cache, normalized_target, sample, &mut batch_grads);
            train_loss += loss;

            // Update weights at end of batch
            if (step + 1) % args.batch_size == 0 || step == indices.len() - 1 {
                let batch_count = if (step + 1) % args.batch_size == 0 {
                    args.batch_size as f32
                } else {
                    ((step + 1) % args.batch_size) as f32
                };

                // Average gradients over batch
                batch_grads.scale(1.0 / batch_count);

                // SGD with momentum
                apply_momentum_update(&mut nnue, &batch_grads, &mut momentum_grads,
                    args.learning_rate, args.momentum, args.weight_decay);

                batch_grads.zero();
            }
        }

        let avg_train_loss = train_loss / train_data.len() as f32;

        // Validation pass
        let mut val_loss = 0.0f32;
        for sample in &val_data {
            let acc = EdgeAccumulator::from_expr(&sample.expr, &nnue.embeddings);
            let structural = StructuralFeatures::from_expr(&sample.expr);
            let normalized_target = (sample.target() - target_mean) / target_std.max(1e-6);
            let prediction = nnue.predict_log_cost_with_features(&acc, &structural);
            let error = prediction - normalized_target;
            val_loss += error * error;
        }
        let avg_val_loss = if !val_data.is_empty() {
            val_loss / val_data.len() as f32
        } else {
            avg_train_loss
        };

        // Track best
        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            best_epoch = epoch + 1;
        }

        if (epoch + 1) % args.print_every == 0 || epoch == 0 {
            println!("  Epoch {:4}: train_loss = {:.6}, val_loss = {:.6}",
                epoch + 1, avg_train_loss, avg_val_loss);
        }
    }

    println!("\nBest validation loss: {:.6} at epoch {}", best_val_loss, best_epoch);

    // Save model
    let output_path = workspace_root.join(&args.output);
    fs::create_dir_all(output_path.parent().unwrap()).expect("Failed to create output directory");

    nnue.save(Path::new(&output_path)).expect("Failed to save model");

    // Save metadata
    let meta = format!(
        r#"{{"target_mean":{},"target_std":{},"epochs":{},"samples":{},"best_val_loss":{}}}"#,
        target_mean, target_std, args.epochs, train_data.len(), best_val_loss
    );
    let meta_path = output_path.with_extension("meta.json");
    fs::write(&meta_path, &meta).expect("Failed to write model metadata");

    println!("\nSaved model to: {}", output_path.display());
    println!("Saved metadata to: {}", meta_path.display());

    // Final evaluation
    println!("\nFinal evaluation:");
    evaluate_model(&nnue, &train_data, target_mean, target_std, "Training");
    if !val_data.is_empty() {
        evaluate_model(&nnue, &val_data, target_mean, target_std, "Validation");
    }
}

/// Clip gradient value to prevent explosion.
#[inline]
fn clip_grad(g: f32, max_norm: f32) -> f32 {
    g.clamp(-max_norm, max_norm)
}

/// Apply SGD update with momentum, weight decay, and gradient clipping.
fn apply_momentum_update(
    net: &mut DualHeadNnue,
    grads: &DualGradients,
    momentum_buf: &mut DualGradients,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
) {
    const GRAD_CLIP: f32 = 1.0; // Clip gradients to [-1, 1]

    // Update embeddings
    for (op_idx, (emb_row, grad_row)) in net.embeddings.e.iter_mut().zip(grads.d_emb.iter()).enumerate() {
        for (emb, (grad, mom)) in emb_row.iter_mut()
            .zip(grad_row.iter().zip(momentum_buf.d_emb[op_idx].iter_mut()))
        {
            let clipped = clip_grad(*grad, GRAD_CLIP);
            *mom = momentum * *mom + clipped + weight_decay * *emb;
            *emb -= lr * *mom;
        }
    }

    // Update w1
    for (i, (w1_row, grad_row)) in net.w1.iter_mut().zip(grads.d_w1.iter()).enumerate() {
        for (w, (g, m)) in w1_row.iter_mut()
            .zip(grad_row.iter().zip(momentum_buf.d_w1[i].iter_mut()))
        {
            let clipped = clip_grad(*g, GRAD_CLIP);
            *m = momentum * *m + clipped + weight_decay * *w;
            *w -= lr * *m;
        }
    }

    // Update b1
    for (b, (g, m)) in net.b1.iter_mut().zip(grads.d_b1.iter().zip(momentum_buf.d_b1.iter_mut())) {
        let clipped = clip_grad(*g, GRAD_CLIP);
        *m = momentum * *m + clipped;
        *b -= lr * *m;
    }

    // Update value head weights
    for (w, (g, m)) in net.value_w.iter_mut()
        .zip(grads.d_value_w.iter().zip(momentum_buf.d_value_w.iter_mut()))
    {
        let clipped = clip_grad(*g, GRAD_CLIP);
        *m = momentum * *m + clipped + weight_decay * *w;
        *w -= lr * *m;
    }

    // Update value head bias
    let clipped_vb = clip_grad(grads.d_value_b, GRAD_CLIP);
    momentum_buf.d_value_b = momentum * momentum_buf.d_value_b + clipped_vb;
    net.value_b -= lr * momentum_buf.d_value_b;
}

/// Evaluate model and print metrics.
fn evaluate_model(
    nnue: &DualHeadNnue,
    data: &[ValueSample],
    target_mean: f32,
    target_std: f32,
    name: &str,
) {
    let mut predictions: Vec<(f32, f32)> = Vec::new();
    for sample in data {
        let acc = EdgeAccumulator::from_expr(&sample.expr, &nnue.embeddings);
        let structural = StructuralFeatures::from_expr(&sample.expr);
        let pred_normalized = nnue.predict_log_cost_with_features(&acc, &structural);
        // Denormalize
        let pred_log = pred_normalized * target_std + target_mean;
        let pred_ns = pred_log.exp();
        let actual_ns = sample.cost_ns as f32;
        predictions.push((pred_ns, actual_ns));
    }

    // Compute metrics
    let n = predictions.len() as f32;
    let sum_x: f32 = predictions.iter().map(|(p, _)| p).sum();
    let sum_y: f32 = predictions.iter().map(|(_, a)| a).sum();
    let sum_xy: f32 = predictions.iter().map(|(p, a)| p * a).sum();
    let sum_x2: f32 = predictions.iter().map(|(p, _)| p * p).sum();
    let sum_y2: f32 = predictions.iter().map(|(_, a)| a * a).sum();

    let denom = (n * sum_x2 - sum_x * sum_x).sqrt() * (n * sum_y2 - sum_y * sum_y).sqrt();
    let correlation = if denom > 0.0 {
        (n * sum_xy - sum_x * sum_y) / denom
    } else {
        0.0
    };

    // MAE
    let mae: f32 = predictions.iter()
        .map(|(p, a)| (p - a).abs())
        .sum::<f32>() / predictions.len() as f32;

    // MAPE (mean absolute percentage error)
    let mape: f32 = predictions.iter()
        .filter(|(_, a)| *a > 0.0)
        .map(|(p, a)| ((p - a).abs() / a).min(10.0)) // Cap at 1000%
        .sum::<f32>() / predictions.len() as f32 * 100.0;

    println!("\n{} set metrics:", name);
    println!("  Correlation (r): {:.4}", correlation);
    println!("  MAE: {:.2} ns", mae);
    println!("  MAPE: {:.1}%", mape);

    // Show sample predictions
    if !predictions.is_empty() {
        println!("\n  Sample predictions (ns):");
        for (pred, actual) in predictions.iter().take(5) {
            let error_pct = if *actual > 0.0 {
                100.0 * (pred - actual).abs() / actual
            } else {
                0.0
            };
            println!("    pred={:.2}, actual={:.2}, error={:.1}%", pred, actual, error_pct);
        }
    }
}

/// Simple LCG random number generator.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }
}

/// Find workspace root by looking for Cargo.toml with [workspace]
fn find_workspace_root() -> PathBuf {
    let mut current = std::env::current_dir().expect("Failed to get current directory");
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let contents = fs::read_to_string(&cargo_toml).unwrap_or_default();
            if contents.contains("[workspace]") {
                return current;
            }
        }
        if !current.pop() {
            return std::env::current_dir().expect("Failed to get current directory");
        }
    }
}
