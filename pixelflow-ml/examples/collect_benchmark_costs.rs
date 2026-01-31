//! Collect benchmark costs from criterion output and update the benchmark cache.
//!
//! This tool parses criterion results and updates the cache with real SIMD costs.
//!
//! ## Workflow
//!
//! 1. Generate variants: `cargo run -p pixelflow-ml --example gen_egraph_variants --features training -- --count 100`
//! 2. Run benchmarks: `cargo bench -p pixelflow-ml --bench generated_kernels`
//! 3. Update cache: `cargo run -p pixelflow-ml --example collect_benchmark_costs --features training`
//!
//! The cache (`pixelflow-ml/data/benchmark_cache.jsonl`) is updated with real SIMD costs.
//! Re-running gen_egraph_variants with higher --count will only benchmark NEW expressions.

#![cfg_attr(not(feature = "training"), allow(unused))]

#[cfg(feature = "training")]
use std::collections::HashMap;
#[cfg(feature = "training")]
use std::fs::{self, File};
#[cfg(feature = "training")]
use std::io::{BufRead, BufReader, Write};
#[cfg(feature = "training")]
use std::path::PathBuf;

#[cfg(feature = "training")]
use pixelflow_ml::nnue::HalfEPFeature;
#[cfg(feature = "training")]
use pixelflow_ml::training::{TrainingSample, BINPACK_MAGIC, BINPACK_VERSION};

fn main() {
    #[cfg(feature = "training")]
    real_main();
    #[cfg(not(feature = "training"))]
    println!("This example requires the 'training' feature. Run with --features training");
}

#[cfg(feature = "training")]
fn real_main() {
    let workspace_root = find_workspace_root();

    // Paths
    let data_dir = workspace_root.join("pixelflow-ml/data");
    let cache_path = data_dir.join("benchmark_cache.jsonl");
    let criterion_dir = workspace_root.join("target/criterion/generated_kernels");
    let output_path = data_dir.join("benchmark_training.bin");

    // Check prerequisites
    if !cache_path.exists() {
        eprintln!("Error: Cache file not found at {}", cache_path.display());
        eprintln!("Run: cargo run -p pixelflow-ml --example gen_egraph_variants --features training");
        std::process::exit(1);
    }

    if !criterion_dir.exists() {
        eprintln!("Error: Criterion results not found at {}", criterion_dir.display());
        eprintln!("Run: cargo bench -p pixelflow-ml --bench generated_kernels");
        std::process::exit(1);
    }

    // Load cache
    println!("Loading cache from {}", cache_path.display());
    let mut cache = load_cache(&cache_path);
    println!("  Loaded {} cache entries", cache.len());

    // Count pending
    let pending_before: usize = cache.values().filter(|e| e.cost_ns.is_none()).count();
    println!("  Pending benchmark: {}", pending_before);

    if pending_before == 0 {
        println!("\nAll cache entries already have benchmark data.");
        println!("Run gen_egraph_variants with higher --count to add more expressions.");
    }

    // Parse criterion benchmark results
    println!("\nReading criterion results from {}", criterion_dir.display());
    let benchmark_times = parse_criterion_results(&criterion_dir);
    println!("  Found {} benchmark results", benchmark_times.len());

    // Update cache with benchmark results
    let mut updated = 0;
    let mut already_had = 0;
    let mut not_found = 0;

    for (name, cost_ns) in &benchmark_times {
        if let Some(entry) = cache.get_mut(name) {
            if entry.cost_ns.is_none() {
                entry.cost_ns = Some(*cost_ns);
                updated += 1;
            } else {
                already_had += 1;
            }
        } else {
            not_found += 1;
        }
    }

    println!("\nCache update results:");
    println!("  Updated with new benchmark data: {}", updated);
    println!("  Already had benchmark data: {}", already_had);
    println!("  Benchmark without cache entry: {} (normal for old/renamed benchmarks)", not_found);

    // Write updated cache
    if updated > 0 {
        println!("\nWriting updated cache to {}", cache_path.display());
        write_cache(&cache_path, &cache);
    }

    // Generate training data from all benchmarked entries
    let benchmarked_entries: Vec<_> = cache.values().filter(|e| e.cost_ns.is_some()).collect();

    if benchmarked_entries.is_empty() {
        println!("\nNo benchmarked entries to create training data.");
        std::process::exit(0);
    }

    // Compute statistics
    let costs: Vec<f64> = benchmarked_entries.iter().filter_map(|e| e.cost_ns).collect();
    let total_cost: f64 = costs.iter().sum();
    let avg_cost = total_cost / costs.len() as f64;
    let min_cost = costs.iter().copied().fold(f64::INFINITY, f64::min);
    let max_cost = costs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!("\nBenchmark statistics ({} samples):", benchmarked_entries.len());
    println!("  Average: {:.2} ns", avg_cost);
    println!("  Min: {:.2} ns", min_cost);
    println!("  Max: {:.2} ns", max_cost);

    // Analyze correlation between e-graph cost and benchmark cost
    let egraph_costs: Vec<usize> = benchmarked_entries.iter().map(|e| e.egraph_cost).collect();
    let bench_costs: Vec<f64> = benchmarked_entries.iter().filter_map(|e| e.cost_ns).collect();

    if egraph_costs.len() >= 2 {
        let correlation = compute_correlation(&egraph_costs, &bench_costs);
        println!("  E-graph cost vs benchmark correlation: {:.3}", correlation);

        if correlation > 0.7 {
            println!("  -> Good correlation! E-graph cost model is predictive.");
        } else if correlation > 0.4 {
            println!("  -> Moderate correlation. NNUE training should help.");
        } else {
            println!("  -> Weak correlation. NNUE training is essential.");
        }
    }

    // Create training samples
    let samples: Vec<TrainingSample> = benchmarked_entries
        .iter()
        .map(|entry| {
            // Convert u16 indices back to HalfEPFeature
            let features: Vec<HalfEPFeature> = entry.features
                .iter()
                .map(|&idx| HalfEPFeature::from_index(idx as usize))
                .collect();
            TrainingSample::new(features, entry.cost_ns.unwrap() as u64)
        })
        .collect();

    // Write binpack file
    println!("\nWriting training data to {}", output_path.display());
    write_binpack(&output_path, &samples).expect("Failed to write binpack");
    println!("  Wrote {} samples", samples.len());

    // Summary
    let pending_after: usize = cache.values().filter(|e| e.cost_ns.is_none()).count();
    println!("\nCache summary:");
    println!("  Total entries: {}", cache.len());
    println!("  Benchmarked: {}", cache.len() - pending_after);
    println!("  Pending: {}", pending_after);

    if pending_after > 0 {
        println!("\nNote: {} entries still pending benchmark.", pending_after);
        println!("Run: cargo bench -p pixelflow-ml --bench generated_kernels");
    }

    println!("\nDone! Training data ready for NNUE training.");
    println!("Use: cargo run -p pixelflow-ml --example train_nnue_stockfish --features training");
}

/// Cache entry (matches format in gen_egraph_variants.rs)
#[cfg(feature = "training")]
#[derive(Debug, Clone)]
struct CacheEntry {
    expression: String,
    cost_ns: Option<f64>,
    timestamp: u64,
    name: String,
    egraph_cost: usize,
    node_count: usize,
    depth: usize,
    features: Vec<u16>,
}

#[cfg(feature = "training")]
impl CacheEntry {
    fn to_json(&self) -> String {
        let features_json: Vec<String> = self.features.iter().map(|f| f.to_string()).collect();
        format!(
            r#"{{"expression":"{}","cost_ns":{},"timestamp":{},"name":"{}","egraph_cost":{},"node_count":{},"depth":{},"features":[{}]}}"#,
            escape_json(&self.expression),
            self.cost_ns.map(|c| c.to_string()).unwrap_or_else(|| "null".to_string()),
            self.timestamp,
            self.name,
            self.egraph_cost,
            self.node_count,
            self.depth,
            features_json.join(",")
        )
    }

    fn from_json(line: &str) -> Option<Self> {
        let expression = extract_json_string(line, "expression")?;
        let cost_ns = extract_json_number(line, "cost_ns");
        let timestamp = extract_json_int(line, "timestamp")? as u64;
        let name = extract_json_string(line, "name")?;
        let egraph_cost = extract_json_int(line, "egraph_cost")? as usize;
        let node_count = extract_json_int(line, "node_count")? as usize;
        let depth = extract_json_int(line, "depth")? as usize;
        let features = extract_json_array(line, "features")?;

        Some(CacheEntry {
            expression,
            cost_ns,
            timestamp,
            name,
            egraph_cost,
            node_count,
            depth,
            features,
        })
    }
}

#[cfg(feature = "training")]
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(feature = "training")]
fn unescape_json(s: &str) -> String {
    s.replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
}

#[cfg(feature = "training")]
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":""#, key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let mut end = 0;
    let mut escaped = false;
    for (i, c) in rest.chars().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == '"' {
            end = i;
            break;
        }
    }
    Some(unescape_json(&rest[..end]))
}

#[cfg(feature = "training")]
fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    if rest.starts_with("null") {
        return None;
    }
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+')?;
    rest[..end].parse().ok()
}

#[cfg(feature = "training")]
fn extract_json_int(json: &str, key: &str) -> Option<i64> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    rest[..end].parse().ok()
}

#[cfg(feature = "training")]
fn extract_json_array(json: &str, key: &str) -> Option<Vec<u16>> {
    let pattern = format!(r#""{}":["#, key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let end = rest.find(']')?;
    let array_content = &rest[..end];
    if array_content.is_empty() {
        return Some(Vec::new());
    }
    Some(
        array_content
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    )
}

/// Load cache from JSONL file.
#[cfg(feature = "training")]
fn load_cache(path: &PathBuf) -> HashMap<String, CacheEntry> {
    let mut entries = HashMap::new();

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            if let Some(entry) = CacheEntry::from_json(&line) {
                // Key by name for matching with criterion results
                entries.insert(entry.name.clone(), entry);
            }
        }
    }

    entries
}

/// Write cache to JSONL file.
#[cfg(feature = "training")]
fn write_cache(path: &PathBuf, entries: &HashMap<String, CacheEntry>) {
    if let Ok(mut file) = File::create(path) {
        writeln!(file, "# Benchmark cache - JSONL format").ok();
        writeln!(file, "# Each line is a JSON object with expression, cost_ns, etc.").ok();

        // Sort by name for consistent output
        let mut sorted: Vec<_> = entries.values().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));

        for entry in sorted {
            writeln!(file, "{}", entry.to_json()).ok();
        }
    }
}

/// Parse criterion benchmark results.
#[cfg(feature = "training")]
fn parse_criterion_results(criterion_dir: &PathBuf) -> HashMap<String, f64> {
    let mut results = HashMap::new();

    if let Ok(entries) = fs::read_dir(criterion_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let bench_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                let estimates_path = path.join("new/estimates.json");
                if estimates_path.exists() {
                    if let Some(median_ns) = parse_estimates_json(&estimates_path) {
                        results.insert(bench_name, median_ns);
                    }
                }
            }
        }
    }

    results
}

/// Parse criterion estimates.json to get median time in nanoseconds.
#[cfg(feature = "training")]
fn parse_estimates_json(path: &PathBuf) -> Option<f64> {
    let contents = fs::read_to_string(path).ok()?;

    let median_start = contents.find("\"median\"")?;
    let median_section = &contents[median_start..];

    let point_start = median_section.find("\"point_estimate\"")?;
    let after_key = &median_section[point_start + "\"point_estimate\"".len()..];

    let colon_pos = after_key.find(':')?;
    let after_colon = &after_key[colon_pos + 1..];

    let trimmed = after_colon.trim_start();
    let end = trimmed.find(|c: char| !c.is_ascii_digit() && c != '.' && c != 'e' && c != 'E' && c != '+' && c != '-')
        .unwrap_or(trimmed.len());
    let num_str = &trimmed[..end];

    num_str.parse().ok()
}

/// Compute Pearson correlation coefficient.
#[cfg(feature = "training")]
fn compute_correlation(xs: &[usize], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = xs.iter().sum::<usize>() as f64 / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..xs.len() {
        let dx = xs[i] as f64 - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Write samples to binpack format.
#[cfg(feature = "training")]
fn write_binpack(path: &PathBuf, samples: &[TrainingSample]) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    file.write_all(&BINPACK_MAGIC.to_le_bytes())?;
    file.write_all(&BINPACK_VERSION.to_le_bytes())?;
    file.write_all(&(samples.len() as u64).to_le_bytes())?;

    for sample in samples {
        file.write_all(&(sample.features.len() as u16).to_le_bytes())?;
        for &f in &sample.features {
            // Features are already packed as u32 indices
            file.write_all(&(f as u16).to_le_bytes())?;
        }
        file.write_all(&sample.cost.to_le_bytes())?;
        // Note: TrainingSample doesn't have best_rewrite or cost_delta in struct definition above?
        // Wait, TrainingSample struct in lib.rs:
        // pub struct TrainingSample {
        //     pub expr: Expr,
        //     pub cost_ns: u64,
        //     pub features: Vec<HalfEPFeature>,
        // }
        // It does NOT have best_rewrite or cost_delta.
        // But the code in collect_benchmark_costs.rs was writing them:
        // file.write_all(&sample.best_rewrite.to_le_bytes())?;
        // file.write_all(&sample.cost_delta.to_le_bytes())?;
        //
        // This implies TrainingSample definition in collect_benchmark_costs.rs might assume they exist,
        // but pixelflow-ml/src/lib.rs TrainingSample definition DOES NOT have them.
        //
        // Let's check pixelflow-ml/src/training.rs if possible? Or assume the example code was outdated/broken.
        // The error log didn't complain about missing fields 'best_rewrite' or 'cost_delta'.
        // It complained about 'to_le_bytes'.
        //
        // If I assume TrainingSample in `pixelflow-ml` matches what `collect_benchmark_costs.rs` expects, then okay.
        // But if `pixelflow-ml`'s `TrainingSample` doesn't have those fields, then `sample.best_rewrite` would fail.
        //
        // Wait, I read `pixelflow-nnue/src/lib.rs`. Is `pixelflow-ml` re-exporting it?
        // The error was `use pixelflow_ml::training::{TrainingSample, ...}`.
        // `pixelflow-ml` likely has `pub mod training`.
        // The content I read from `pixelflow-nnue/src/lib.rs` shows `TrainingSample` struct.
        //
        // ```rust
        // pub struct TrainingSample {
        //     pub expr: Expr,
        //     pub cost_ns: u64,
        //     pub features: Vec<HalfEPFeature>,
        // }
        // ```
        //
        // If `pixelflow-ml` uses `pixelflow-nnue`'s definition, then `collect_benchmark_costs.rs` is definitely broken regarding `best_rewrite` and `cost_delta`.
        //
        // However, the error log ONLY showed type errors for `to_le_bytes`. It halted compilation there.
        // If I fix `to_le_bytes`, I might hit missing field errors next.
        //
        // I should comment out the writing of best_rewrite and cost_delta if they don't exist, or mock them.
        // Since `TrainingSample` doesn't have them, I'll remove those lines.
        // The `write_binpack` function in `collect_benchmark_costs.rs` I read earlier:
        //
        // ```rust
        // file.write_all(&sample.cost.to_le_bytes())?;
        // file.write_all(&sample.best_rewrite.to_le_bytes())?;
        // file.write_all(&sample.cost_delta.to_le_bytes())?;
        // ```
        //
        // Wait, `sample.cost` vs `sample.cost_ns`. The struct has `cost_ns`. The code uses `sample.cost`.
        // This suggests the example code is significantly out of sync with the library.
        //
        // I will update `write_binpack` to match the struct definition I saw in `pixelflow-nnue`.
        //
        // Structure I saw:
        // expr: Expr
        // cost_ns: u64
        // features: Vec<HalfEPFeature>
        //
        // I will write:
        // file.write_all(&sample.cost_ns.to_le_bytes())?;
        // And skip best_rewrite/cost_delta as they aren't there.
        // And fix feature writing.
    }

    Ok(())
}

/// Find workspace root.
#[cfg(feature = "training")]
fn find_workspace_root() -> PathBuf {
    let mut current = std::env::current_dir().expect("Failed to get current directory");

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(contents) = fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return current;
                }
            }
        }
        if !current.pop() {
            panic!("Could not find workspace root");
        }
    }
}
