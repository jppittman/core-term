//! Train the Guide on match prediction data.
//!
//! This binary trains the GuideNnue network to predict whether rewrite rules
//! will match given the current e-graph state. This is binary classification
//! with asymmetric loss (false positives cost more than false negatives).
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin train_guide --release
//! ```
//!
//! # Input
//!
//! - `pixelflow-pipeline/data/guide_training.jsonl` - From collect_guide_data
//!
//! # Output
//!
//! - `pixelflow-pipeline/data/guide.bin` - Trained GuideNnue model

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;
use pixelflow_search::nnue::guide::{GuideNnue, RULE_FEATURE_COUNT};

/// Train the Guide (Match Predictor).
#[derive(Parser, Debug)]
#[command(name = "train_guide")]
#[command(about = "Train the Guide (Match Predictor) on e-graph data")]
struct Args {
    /// Path to training data
    #[arg(long, default_value = "pixelflow-pipeline/data/guide_training.jsonl")]
    input: String,

    /// Output path for trained model
    #[arg(short, long, default_value = "pixelflow-pipeline/data/guide.bin")]
    output: String,

    /// Number of training epochs
    #[arg(long, default_value_t = 100)]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Print loss every N epochs
    #[arg(long, default_value_t = 10)]
    print_every: usize,

    /// Validation split ratio
    #[arg(long, default_value_t = 0.1)]
    val_split: f32,
}

/// Training sample from collect_guide_data.
#[derive(Debug, Deserialize)]
struct MatchSample {
    #[allow(dead_code)]
    expression_id: String,
    #[allow(dead_code)]
    epoch: usize,
    #[allow(dead_code)]
    rule_idx: usize,
    features: [f32; RULE_FEATURE_COUNT],
    matched: bool,
}

fn main() {
    let args = Args::parse();

    // Find workspace root
    let workspace_root = find_workspace_root();

    // Load training data
    let input_path = workspace_root.join(&args.input);
    println!("Loading training data from: {}", input_path.display());

    let input_file = fs::File::open(&input_path)
        .expect("Failed to open training data. Run collect_guide_data first.");
    let reader = BufReader::new(input_file);

    let mut samples: Vec<MatchSample> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let sample: MatchSample = serde_json::from_str(&line)
            .expect("Failed to parse training data");
        samples.push(sample);
    }
    println!("Loaded {} training samples", samples.len());

    if samples.is_empty() {
        eprintln!("No training samples found!");
        std::process::exit(1);
    }

    // Compute label distribution
    let num_positive = samples.iter().filter(|s| s.matched).count();
    let num_negative = samples.len() - num_positive;
    let match_rate = num_positive as f32 / samples.len() as f32;
    println!("\nLabel distribution:");
    println!("  Positive (matched): {} ({:.1}%)", num_positive, match_rate * 100.0);
    println!("  Negative (no match): {} ({:.1}%)", num_negative, (1.0 - match_rate) * 100.0);

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
    println!("\nTraining samples: {}", train_samples.len());
    println!("Validation samples: {}", val_samples.len());

    // Convert to (features, matched) pairs
    let train_data: Vec<([f32; RULE_FEATURE_COUNT], bool)> = train_samples
        .iter()
        .map(|s| (s.features, s.matched))
        .collect();

    let val_data: Vec<([f32; RULE_FEATURE_COUNT], bool)> = val_samples
        .iter()
        .map(|s| (s.features, s.matched))
        .collect();

    // Create model
    let mut guide = GuideNnue::new_random(args.seed);
    println!("\nModel: {} parameters", guide.param_count());

    // Training loop
    println!("\nTraining for {} epochs with lr={}", args.epochs, args.learning_rate);
    println!("(Using asymmetric BCE: FP weight=2.0, FN weight=0.5)\n");

    let mut best_val_accuracy = 0.0f32;
    let mut best_epoch = 0;
    let mut best_model = guide.clone();

    for epoch in 0..args.epochs {
        // Shuffle training data each epoch
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.next() as usize % (i + 1);
            indices.swap(i, j);
        }

        // Training pass
        let mut train_loss = 0.0f32;
        for &idx in &indices {
            let (features, matched) = &train_data[idx];
            train_loss += guide.train_step(features, *matched, args.learning_rate);
        }
        let avg_train_loss = train_loss / train_data.len() as f32;

        // Validation pass
        let (val_accuracy, val_fp_rate, val_fn_rate) = guide.evaluate(&val_data);

        // Compute validation loss
        let mut val_loss = 0.0f32;
        for (features, matched) in &val_data {
            let predicted = guide.predict(features);
            val_loss += guide.loss(predicted, *matched);
        }
        let avg_val_loss = if !val_data.is_empty() {
            val_loss / val_data.len() as f32
        } else {
            avg_train_loss
        };

        // Track best model based on accuracy
        if val_accuracy > best_val_accuracy {
            best_val_accuracy = val_accuracy;
            best_epoch = epoch + 1;
            best_model = guide.clone();
        }

        if (epoch + 1) % args.print_every == 0 || epoch == 0 {
            println!(
                "  Epoch {:4}: loss={:.4}/{:.4}, acc={:.1}%, FP={:.1}%, FN={:.1}%",
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                val_accuracy * 100.0,
                val_fp_rate * 100.0,
                val_fn_rate * 100.0
            );
        }
    }

    println!("\nBest validation accuracy: {:.1}% at epoch {}", best_val_accuracy * 100.0, best_epoch);

    // Use best model
    let guide = best_model;

    // Save model
    let output_path = workspace_root.join(&args.output);
    fs::create_dir_all(output_path.parent().unwrap()).expect("Failed to create output directory");
    guide.save(&output_path).expect("Failed to save model");

    // Save metadata
    let meta = format!(
        r#"{{"epochs":{},"samples":{},"train_samples":{},"val_samples":{},"best_val_accuracy":{},"match_rate":{}}}"#,
        args.epochs, samples.len(), train_samples.len(), val_samples.len(), best_val_accuracy, match_rate
    );
    let meta_path = output_path.with_extension("meta.json");
    fs::write(&meta_path, &meta).expect("Failed to write model metadata");

    println!("\nSaved model to: {}", output_path.display());
    println!("Saved metadata to: {}", meta_path.display());

    // Final evaluation
    println!("\n=== Final Evaluation ===\n");
    final_evaluation(&guide, &val_data, &train_data);
}

/// Detailed final evaluation with metrics.
fn final_evaluation(
    guide: &GuideNnue,
    val_data: &[([f32; RULE_FEATURE_COUNT], bool)],
    train_data: &[([f32; RULE_FEATURE_COUNT], bool)],
) {
    // Train metrics
    let (train_acc, train_fp, train_fn) = guide.evaluate(train_data);
    println!("Training set:");
    println!("  Accuracy:            {:.1}%", train_acc * 100.0);
    println!("  False positive rate: {:.1}%", train_fp * 100.0);
    println!("  False negative rate: {:.1}%", train_fn * 100.0);

    // Validation metrics
    let (val_acc, val_fp, val_fn) = guide.evaluate(val_data);
    println!("\nValidation set:");
    println!("  Accuracy:            {:.1}%", val_acc * 100.0);
    println!("  False positive rate: {:.1}%", val_fp * 100.0);
    println!("  False negative rate: {:.1}%", val_fn * 100.0);

    // Prediction distribution
    let mut predictions: Vec<f32> = val_data
        .iter()
        .map(|(f, _)| guide.predict(f))
        .collect();
    predictions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p_min = predictions.first().copied().unwrap_or(0.0);
    let p_max = predictions.last().copied().unwrap_or(0.0);
    let p_median = if !predictions.is_empty() {
        predictions[predictions.len() / 2]
    } else {
        0.0
    };

    println!("\nPrediction distribution:");
    println!("  Min:    {:.3}", p_min);
    println!("  Median: {:.3}", p_median);
    println!("  Max:    {:.3}", p_max);

    // Sample predictions
    println!("\nSample predictions (first 10 validation samples):");
    for (i, (features, matched)) in val_data.iter().take(10).enumerate() {
        let pred = guide.predict(features);
        let pred_label = if pred > 0.5 { "MATCH" } else { "NO MATCH" };
        let actual_label = if *matched { "MATCH" } else { "NO MATCH" };
        let correct = (pred > 0.5) == *matched;
        let marker = if correct { "✓" } else { "✗" };

        println!(
            "  {:2}: pred={:.3} ({:8}) actual={:8} {}",
            i, pred, pred_label, actual_label, marker
        );
    }

    // Check if guide learned anything useful
    println!("\n=== Guide Quality Check ===");

    // A good Guide should have different predictions for different rule contexts
    let pred_std = if predictions.len() > 1 {
        let mean = predictions.iter().sum::<f32>() / predictions.len() as f32;
        (predictions.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / predictions.len() as f32).sqrt()
    } else {
        0.0
    };

    println!("  Prediction std dev: {:.4}", pred_std);

    if pred_std < 0.1 {
        println!("  WARNING: Low prediction variance - Guide may be predicting constant values");
    }

    if val_acc < 0.55 {
        println!("  WARNING: Accuracy near baseline - Guide may not be learning useful patterns");
    } else if val_acc > 0.75 {
        println!("  GOOD: Guide is learning discriminative patterns");
    }

    // Check if false positive rate is reasonable (should be low due to asymmetric loss)
    if val_fp < 0.15 {
        println!("  GOOD: Low false positive rate (conservative Guide)");
    } else if val_fp > 0.30 {
        println!("  WARNING: High false positive rate - consider increasing FP weight");
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
