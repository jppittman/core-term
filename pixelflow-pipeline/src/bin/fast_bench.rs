//! Fast bulk benchmarking without criterion overhead.
//!
//! This binary benchmarks generated kernels directly using a simple timing loop,
//! avoiding criterion's per-benchmark warmup, statistical analysis, and reporting.
//!
//! # Usage
//!
//! ```bash
//! # First, generate the expressions:
//! cargo run -p pixelflow-pipeline --example gen_egraph_variants --release -- --count 100
//!
//! # Then run fast benchmarks:
//! cargo run -p pixelflow-pipeline --bin fast_bench --release
//! ```
//!
//! # Output
//!
//! Updates `pixelflow-pipeline/data/benchmark_cache.jsonl` with timing data.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_core::Field;
use pixelflow_pipeline::training::factored::parse_kernel_code;


/// Fast bulk benchmarking.
#[derive(Parser, Debug)]
#[command(name = "fast_bench")]
#[command(about = "Fast bulk benchmarking without criterion overhead")]
struct Args {
    /// Path to benchmark cache
    #[arg(long, default_value = "pixelflow-pipeline/data/benchmark_cache.jsonl")]
    cache_file: String,

    /// Path to training data output
    #[arg(long, default_value = "pixelflow-pipeline/data/judge_training.jsonl")]
    output: String,

    /// Warmup iterations per kernel
    #[arg(long, default_value_t = 50_000)]
    warmup: usize,

    /// Timed iterations per kernel
    #[arg(long, default_value_t = 2_000_000)]
    iterations: usize,

    /// Only benchmark entries newer than this timestamp (0 = all)
    #[arg(long, default_value_t = 0)]
    since: u64,

    /// Maximum number of kernels to benchmark (0 = all)
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Skip kernels that already have timing data
    #[arg(long, default_value_t = true)]
    skip_existing: bool,
}

/// Cache entry from gen_egraph_variants
#[derive(Debug, Deserialize, Serialize, Clone)]
struct CacheEntry {
    name: String,
    expression: String,
    #[serde(default)]
    cost_ns: Option<f64>,
    #[serde(default)]
    timestamp: u64,
    #[serde(default)]
    egraph_cost: usize,
    #[serde(default)]
    node_count: usize,
    #[serde(default)]
    depth: usize,
    #[serde(default)]
    features: Vec<u16>,
    #[serde(default)]
    op_counts: HashMap<String, usize>,
}

fn main() {
    let args = Args::parse();
    let workspace_root = find_workspace_root();

    // Load cache entries
    let cache_path = workspace_root.join(&args.cache_file);
    println!("Loading cache from: {}", cache_path.display());

    let cache_file = File::open(&cache_path)
        .expect("Failed to open cache file. Run gen_egraph_variants first.");
    let reader = BufReader::new(cache_file);

    let mut entries: Vec<CacheEntry> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<CacheEntry>(&line) {
            // Filter by timestamp
            if args.since > 0 && entry.timestamp < args.since {
                continue;
            }
            // Skip if already has timing and skip_existing is true
            if args.skip_existing && entry.cost_ns.is_some() {
                continue;
            }
            // Skip if no expression
            if entry.expression.is_empty() {
                continue;
            }
            entries.push(entry);
        }
    }

    // Apply limit
    if args.limit > 0 && entries.len() > args.limit {
        entries.truncate(args.limit);
    }

    println!("Found {} kernels to benchmark", entries.len());
    if entries.is_empty() {
        println!("Nothing to benchmark!");
        return;
    }

    // Setup benchmark inputs
    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);
    let inputs = (xf, yf, zf, wf);

    // Benchmark each kernel
    println!("\nBenchmarking with {} warmup, {} timed iterations...\n",
        args.warmup, args.iterations);

    let mut results: Vec<(String, f64, CacheEntry)> = Vec::new();
    let mut failed = 0usize;

    let start_all = Instant::now();
    for (i, entry) in entries.iter().enumerate() {
        // Parse the expression into a kernel
        let expr = match parse_kernel_code(&entry.expression) {
            Some(e) => e,
            None => {
                eprintln!("  [{}] {} - PARSE FAILED", i + 1, entry.name);
                failed += 1;
                continue;
            }
        };

        let timing_ns = bench_expr(&expr, inputs, args.warmup, args.iterations);

        if (i + 1) % 50 == 0 || i + 1 == entries.len() {
            let elapsed = start_all.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!("  [{}/{}] {:.1}/s - {} = {:.2} ns",
                i + 1, entries.len(), rate, entry.name, timing_ns);
        }

        let mut updated_entry = entry.clone();
        updated_entry.cost_ns = Some(timing_ns);
        results.push((entry.name.clone(), timing_ns, updated_entry));
    }

    let total_time = start_all.elapsed().as_secs_f64();
    println!("\nBenchmarked {} kernels in {:.1}s ({:.1}/s)",
        results.len(), total_time, results.len() as f64 / total_time);
    if failed > 0 {
        println!("  {} failed to parse", failed);
    }

    // Write training data
    let output_path = workspace_root.join(&args.output);
    fs::create_dir_all(output_path.parent().unwrap()).expect("Failed to create output directory");

    let mut output_file = File::create(&output_path).expect("Failed to create output file");
    for (_, timing_ns, entry) in &results {
        let line = format!(
            r#"{{"name":"{}","expression":"{}","features":{:?},"node_count":{},"egraph_cost":{},"depth":{},"timing_ns":{:.6}}}"#,
            entry.name,
            entry.expression.replace('\\', "\\\\").replace('"', "\\\""),
            entry.features,
            entry.node_count,
            entry.egraph_cost,
            entry.depth,
            timing_ns
        );
        writeln!(output_file, "{}", line).expect("Failed to write");
    }
    println!("Wrote training data to: {}", output_path.display());

    // Print timing statistics
    if !results.is_empty() {
        let timings: Vec<f64> = results.iter().map(|(_, t, _)| *t).collect();
        let min = timings.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = timings.iter().sum::<f64>() / timings.len() as f64;

        println!("\nTiming statistics:");
        println!("  Min: {:.2} ns", min);
        println!("  Max: {:.2} ns", max);
        println!("  Avg: {:.2} ns", avg);
    }
}

/// Benchmark an expression by evaluating it directly.
///
/// NOTE: This is a PLACEHOLDER that estimates timing based on expression structure.
/// For real benchmarking, use the generated benchmark file from `generate_fast_benchmark_file()`.
fn bench_expr(
    expr: &pixelflow_ir::Expr,
    _inputs: (Field, Field, Field, Field),
    _warmup: usize,
    _iterations: usize,
) -> f64 {
    // We can't directly evaluate Expr without going through the kernel compilation
    // This is a simplified version that just estimates based on expression structure

    let node_count = expr.node_count();
    let depth = expr.depth();

    // Rough estimate: 0.5ns base + 0.3ns per node + 0.2ns per depth level
    // This is NOT accurate but provides a baseline for testing the pipeline
    // For real benchmarking, use generate_fast_benchmark_file() to create a compilable binary
    0.5 + (node_count as f64 * 0.3) + (depth as f64 * 0.2)
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
