//! Collect Judge (Value Head) training data from benchmark results.
//!
//! This binary parses criterion benchmark output and merges it with
//! the features from the benchmark cache to create training data.
//!
//! # Usage
//!
//! ```bash
//! # First, run the benchmarks:
//! cargo bench -p pixelflow-pipeline --bench generated_kernels
//!
//! # Then collect the data:
//! cargo run -p pixelflow-pipeline --bin collect_judge_data --release
//! ```
//!
//! # Input
//!
//! - `target/criterion/generated_kernels/*/new/estimates.json` - Benchmark timings
//! - `pixelflow-pipeline/data/benchmark_cache.jsonl` - Cache with features from gen_egraph_variants
//!
//! # Output
//!
//! - `pixelflow-pipeline/data/judge_training.jsonl` - Training data (features + timing)

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;

/// Collect training data for the Judge (Value Head).
#[derive(Parser, Debug)]
#[command(name = "collect_judge_data")]
#[command(about = "Collect Judge training data from benchmark results")]
struct Args {
    /// Path to criterion output directory
    #[arg(long, default_value = "target/criterion")]
    criterion_dir: String,

    /// Path to benchmark cache JSONL file (from gen_egraph_variants)
    #[arg(long, default_value = "pixelflow-pipeline/data/benchmark_cache.jsonl")]
    cache_file: String,

    /// Output path for training data
    #[arg(short, long, default_value = "pixelflow-pipeline/data/judge_training.jsonl")]
    output: String,

    /// Benchmark group name
    #[arg(long, default_value = "generated_kernels")]
    group: String,
}

/// Criterion estimates.json structure
#[derive(Debug, Deserialize)]
struct Estimates {
    mean: EstimateValue,
}

#[derive(Debug, Deserialize)]
struct EstimateValue {
    point_estimate: f64,
}

/// Cache entry from gen_egraph_variants
#[derive(Debug, Deserialize)]
struct CacheEntry {
    name: String,
    /// The expression string in kernel syntax
    #[serde(default)]
    expression: String,
    #[serde(default)]
    features: Vec<u16>,
    #[serde(default)]
    node_count: usize,
    #[serde(default)]
    egraph_cost: usize,
    #[serde(default)]
    depth: usize,
}

fn main() {
    let args = Args::parse();

    // Find workspace root
    let workspace_root = find_workspace_root();

    // Load cache entries
    let cache_path = workspace_root.join(&args.cache_file);
    println!("Loading cache from: {}", cache_path.display());

    let cache_file = fs::File::open(&cache_path)
        .expect("Failed to open cache file. Run gen_egraph_variants first.");
    let reader = BufReader::new(cache_file);

    let mut cache_map: HashMap<String, CacheEntry> = HashMap::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<CacheEntry>(&line) {
            // Only keep entries with expression AND features
            if !entry.expression.is_empty() && !entry.features.is_empty() {
                cache_map.insert(entry.name.clone(), entry);
            }
        }
    }
    println!("Loaded {} cache entries with expressions and features", cache_map.len());

    // Find and parse criterion results
    let criterion_dir = workspace_root.join(&args.criterion_dir).join(&args.group);
    println!("Scanning criterion results in: {}", criterion_dir.display());

    let mut training_data: Vec<(CacheEntry, f64)> = Vec::new();
    let mut missing_features = 0usize;
    let mut missing_estimates = 0usize;

    if criterion_dir.exists() {
        for entry in fs::read_dir(&criterion_dir).expect("Failed to read criterion directory") {
            let entry = entry.expect("Failed to read directory entry");
            let benchmark_name = entry.file_name().to_string_lossy().to_string();

            // Skip report directory
            if benchmark_name == "report" {
                continue;
            }

            // Look for estimates.json
            let estimates_path = entry.path().join("new").join("estimates.json");
            if !estimates_path.exists() {
                missing_estimates += 1;
                continue;
            }

            // Parse estimates
            let estimates_content = fs::read_to_string(&estimates_path)
                .expect("Failed to read estimates.json");
            let estimates: Estimates = serde_json::from_str(&estimates_content)
                .expect("Failed to parse estimates.json");

            let timing_ns = estimates.mean.point_estimate;

            // FAIL FAST: Validate timing data immediately
            if timing_ns.is_nan() {
                panic!("Benchmark '{}' has NaN timing - criterion data is corrupted", benchmark_name);
            }
            if timing_ns.is_infinite() {
                panic!("Benchmark '{}' has infinite timing - criterion data is corrupted", benchmark_name);
            }
            if timing_ns <= 0.0 {
                panic!("Benchmark '{}' has non-positive timing ({}) - criterion data is corrupted", benchmark_name, timing_ns);
            }

            // Match with cache entry
            if let Some(cache_entry) = cache_map.remove(&benchmark_name) {
                training_data.push((cache_entry, timing_ns));
            } else {
                missing_features += 1;
            }
        }
    } else {
        eprintln!("Criterion directory not found: {}", criterion_dir.display());
        eprintln!("Run: cargo bench -p pixelflow-pipeline --bench generated_kernels");
        std::process::exit(1);
    }

    println!("Collected {} training samples", training_data.len());
    if missing_features > 0 {
        println!("  Warning: {} benchmarks had no matching cache entry", missing_features);
    }
    if missing_estimates > 0 {
        println!("  Warning: {} benchmarks had no estimates.json", missing_estimates);
    }

    // Write training data
    let output_path = workspace_root.join(&args.output);
    fs::create_dir_all(output_path.parent().unwrap()).expect("Failed to create output directory");

    let mut output_file = fs::File::create(&output_path).expect("Failed to create output file");
    for (cache_entry, timing_ns) in &training_data {
        // Escape the expression string for JSON
        let escaped_expr = cache_entry.expression.replace('\\', "\\\\").replace('"', "\\\"");
        let line = format!(
            r#"{{"name":"{}","expression":"{}","features":{:?},"node_count":{},"egraph_cost":{},"depth":{},"timing_ns":{:.6}}}"#,
            cache_entry.name,
            escaped_expr,
            cache_entry.features,
            cache_entry.node_count,
            cache_entry.egraph_cost,
            cache_entry.depth,
            timing_ns
        );
        writeln!(output_file, "{}", line).expect("Failed to write training data");
    }

    println!("Wrote training data to: {}", output_path.display());

    // Print statistics
    if !training_data.is_empty() {
        let timings: Vec<f64> = training_data.iter().map(|(_, t)| *t).collect();
        let min_timing = timings.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_timing = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_timing = timings.iter().sum::<f64>() / timings.len() as f64;

        println!("\nTiming statistics:");
        println!("  Min: {:.2} ns", min_timing);
        println!("  Max: {:.2} ns", max_timing);
        println!("  Avg: {:.2} ns", avg_timing);
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
            // Fallback to current directory
            return std::env::current_dir().expect("Failed to get current directory");
        }
    }
}
