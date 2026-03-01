//! Generate training data using e-graph variant extraction with lazy benchmarking.
//!
//! This example demonstrates the clean training pipeline with caching:
//! 1. Generate seed expressions (random + targeted families)
//! 2. Insert into e-graph and saturate to find equivalents
//! 3. Extract N variants from the same e-class
//! 4. Check cache - skip expressions already benchmarked
//! 5. Write benchmark code for NEW expressions only
//!
//! # Targeted Families
//!
//! In addition to random expressions, we generate targeted seeds that specifically
//! exercise micro-optimization rewrites:
//! - Division family: x/y, x/(y+1), a/b/c - triggers div↔mul+recip
//! - Negation family: a-b, -(a+b), -(-x) - triggers sub↔add+neg
//! - Sqrt family: 1/sqrt(x), sqrt(x*x) - triggers rsqrt, |x| identities
//! - Compound: lerp, distance, normalize patterns
//!
//! # Usage
//!
//! ```bash
//! # Generate variants (incrementally adds to cache)
//! cargo run -p pixelflow-pipeline --example gen_egraph_variants -- --count 100 --variants 5
//!
//! # Include targeted micro-optimization families
//! cargo run -p pixelflow-pipeline --example gen_egraph_variants -- --count 100 --targeted
//!
//! # JIT-benchmark new expressions
//! cargo run --release -p pixelflow-pipeline --bin bench_jit_corpus --features training
//!
//! # Collect costs and update cache
//! cargo run -p pixelflow-pipeline --bin collect_judge_data
//! ```
//!
//! The cache (`pixelflow-pipeline/data/benchmark_cache.jsonl`) persists across runs,
//! building a growing dataset. Re-running with higher --count adds more samples
//! without re-benchmarking existing expressions.
//!
//! The key insight: we use **actual algebraic equivalences** from the e-graph
//! rather than heuristic "unfusing" to generate training pairs.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use pixelflow_pipeline::fusion::fusion_rules;
use pixelflow_pipeline::nnue::{Expr, ExprGenConfig, ExprGenerator, HalfEPFeature, OpType, extract_features};
use pixelflow_search::egraph::{EGraph, ExprTree, Leaf, CostModel, codegen, ops};
use pixelflow_search::egraph::ops::op_from_kind;
use pixelflow_search::egraph::rewrite::Rewrite;
use pixelflow_search::math::all_math_rules;

/// All rules for training: math rules + CPU instruction selection rules.
///
/// This combines:
/// - `all_math_rules()` from pixelflow-search (~28 mathematical rules)
/// - `fusion_rules()` from pixelflow-pipeline (FMA, rsqrt)
///
/// The Judge neural network needs to learn about BOTH categories:
/// - Math rules: algebraic equivalences (a-b = a+neg(b), etc.)
/// - Fusion rules: CPU instruction selection (1/sqrt(x) → rsqrt(x), a*b+c → fma)
fn all_training_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_math_rules();
    rules.extend(fusion_rules());
    rules
}

/// Cached benchmark result for an expression.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The kernel body code (used as cache key)
    expression: String,
    /// Measured cost in nanoseconds (None if not yet benchmarked)
    cost_ns: Option<f64>,
    /// Unix timestamp when this entry was created
    timestamp: u64,
    /// Variant name (eg0001v0, etc.)
    name: String,
    /// E-graph cost model estimate (for comparison)
    egraph_cost: usize,
    /// Node count
    node_count: usize,
    /// Expression depth
    depth: usize,
    /// NNUE features (stored as indices)
    features: Vec<u16>,
    /// Operation counts keyed by op name (computed from ExprTree)
    op_counts: HashMap<String, usize>,
}

/// Count operations in an ExprTree. Returns map of op_name -> count.
fn count_ops(tree: &ExprTree) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    count_ops_recursive(tree, &mut counts);
    counts
}

fn count_ops_recursive(tree: &ExprTree, counts: &mut HashMap<String, usize>) {
    match tree {
        ExprTree::Leaf(_) => {}
        ExprTree::Op { op, children } => {
            *counts.entry(op.name().to_string()).or_insert(0) += 1;
            for child in children {
                count_ops_recursive(child, counts);
            }
        }
    }
}

fn op_counts_to_json(counts: &HashMap<String, usize>) -> String {
    let pairs: Vec<String> = counts
        .iter()
        .map(|(k, v)| format!(r#""{}":{}"#, k, v))
        .collect();
    format!("{{{}}}", pairs.join(","))
}

fn op_counts_from_json(json: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    // Parse {"add":5,"mul":3,...}
    let trimmed = json.trim().trim_start_matches('{').trim_end_matches('}');
    for pair in trimmed.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        // Parse "key":value
        if let Some(colon_pos) = pair.find(':') {
            let key = pair[..colon_pos].trim().trim_matches('"');
            let value = pair[colon_pos + 1..].trim();
            if let Ok(v) = value.parse::<usize>() {
                counts.insert(key.to_string(), v);
            }
        }
    }
    counts
}

impl CacheEntry {
    fn to_json(&self) -> String {
        let features_json: Vec<String> = self.features.iter().map(|f| f.to_string()).collect();
        format!(
            r#"{{"expression":"{}","cost_ns":{},"timestamp":{},"name":"{}","egraph_cost":{},"node_count":{},"depth":{},"features":[{}],"op_counts":{}}}"#,
            escape_json(&self.expression),
            self.cost_ns.map(|c| c.to_string()).unwrap_or_else(|| "null".to_string()),
            self.timestamp,
            self.name,
            self.egraph_cost,
            self.node_count,
            self.depth,
            features_json.join(","),
            op_counts_to_json(&self.op_counts)
        )
    }

    fn from_json(line: &str) -> Option<Self> {
        // Simple JSON parsing - extract fields by pattern matching
        let expression = extract_json_string(line, "expression")?;
        let cost_ns = extract_json_number(line, "cost_ns");
        let timestamp = extract_json_int(line, "timestamp")? as u64;
        let name = extract_json_string(line, "name")?;
        let egraph_cost = extract_json_int(line, "egraph_cost")? as usize;
        let node_count = extract_json_int(line, "node_count")? as usize;
        let depth = extract_json_int(line, "depth")? as usize;
        let features = extract_json_array(line, "features")?;
        let op_counts = extract_json_object(line, "op_counts")
            .map(|s| op_counts_from_json(&s))
            .unwrap_or_default();

        Some(CacheEntry {
            expression,
            cost_ns,
            timestamp,
            name,
            egraph_cost,
            node_count,
            depth,
            features,
            op_counts,
        })
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn unescape_json(s: &str) -> String {
    s.replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":""#, key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    // Find closing quote, handling escapes
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

fn extract_json_int(json: &str, key: &str) -> Option<i64> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    rest[..end].parse().ok()
}

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

fn extract_json_object(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":{{"#, key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len() - 1..]; // Include opening brace
    // Find matching closing brace
    let mut depth = 0;
    for (i, c) in rest.chars().enumerate() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Benchmark cache that persists across runs.
struct BenchmarkCache {
    /// Map from expression body → cache entry
    entries: HashMap<String, CacheEntry>,
    /// Path to the cache file
    path: PathBuf,
    /// Next available unique ID (always incrementing)
    next_id: usize,
}

impl BenchmarkCache {
    fn load(path: PathBuf) -> Self {
        let mut entries = HashMap::new();
        let mut max_id: usize = 0;

        if path.exists() {
            if let Ok(file) = File::open(&path) {
                let reader = BufReader::new(file);
                for line in reader.lines().map_while(Result::ok) {
                    if line.starts_with('#') || line.trim().is_empty() {
                        continue;
                    }
                    if let Some(entry) = CacheEntry::from_json(&line) {
                        // Track max ID for unique name generation
                        if let Some((seed_idx, _)) = parse_variant_name(&entry.name) {
                            max_id = max_id.max(seed_idx);
                        }
                        entries.insert(entry.expression.clone(), entry);
                    }
                }
            }
        }

        // Next ID starts after max existing
        let next_id = max_id + 1;
        println!("Loaded {} cached benchmark entries (next_id: {})", entries.len(), next_id);
        BenchmarkCache { entries, path, next_id }
    }

    /// Allocate a unique ID for a new entry
    fn allocate_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn contains(&self, expression: &str) -> bool {
        self.entries.contains_key(expression)
    }

    fn get(&self, expression: &str) -> Option<&CacheEntry> {
        self.entries.get(expression)
    }

    /// Append a new entry to the cache file immediately.
    fn append(&mut self, entry: CacheEntry) {
        let json = entry.to_json();
        let expression = entry.expression.clone();

        // Append to file
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&self.path) {
            if let Err(e) = writeln!(file, "{}", json) {
                eprintln!("Warning: failed to write cache entry: {}", e);
            }
        }

        self.entries.insert(expression, entry);
    }

    /// Get entries that are pending benchmark (have cost_ns = None).
    fn pending_entries(&self) -> Vec<&CacheEntry> {
        self.entries.values().filter(|e| e.cost_ns.is_none()).collect()
    }

    /// Update an entry with benchmark results.
    fn update_cost(&mut self, expression: &str, cost_ns: f64) {
        if let Some(entry) = self.entries.get_mut(expression) {
            entry.cost_ns = Some(cost_ns);
        }
    }

    /// Rewrite the entire cache file (after updates).
    fn flush(&self) {
        if let Ok(mut file) = File::create(&self.path) {
            writeln!(file, "# Benchmark cache - JSONL format").ok();
            writeln!(file, "# Each line is a JSON object with expression, cost_ns, etc.").ok();
            for entry in self.entries.values() {
                writeln!(file, "{}", entry.to_json()).ok();
            }
        }
    }

    /// Count entries with actual benchmark results.
    fn benchmarked_count(&self) -> usize {
        self.entries.values().filter(|e| e.cost_ns.is_some()).count()
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut count = 100;
    let mut variants_per_expr = 5;
    let mut seed = 42u64;
    let mut include_targeted = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--count" => {
                i += 1;
                if i < args.len() {
                    count = args[i].parse().expect("Invalid count");
                }
            }
            "--variants" => {
                i += 1;
                if i < args.len() {
                    variants_per_expr = args[i].parse().expect("Invalid variants");
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().expect("Invalid seed");
                }
            }
            "--targeted" => {
                include_targeted = true;
            }
            "--help" | "-h" => {
                println!("Usage: gen_egraph_variants [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --count N      Number of random seed expressions (default: 100)");
                println!("  --variants N   Variants to extract per expression (default: 5)");
                println!("  --seed N       Random seed (default: 42)");
                println!("  --targeted     Include targeted micro-optimization families");
                println!();
                println!("This generates training data by:");
                println!("  1. Creating seed expressions (random + targeted if enabled)");
                println!("  2. Inserting each into an e-graph and saturating");
                println!("  3. Extracting N different equivalent forms");
                println!("  4. Writing benchmark code for all variants");
                println!();
                println!("Targeted families include:");
                println!("  - Division: x/y, x/(y+1), x/y/z, x/2");
                println!("  - Negation: x-y, -(-x), -(x+y)");
                println!("  - Sqrt: 1/sqrt(x), sqrt(x*x), normalize");
                println!("  - Compound: lerp, distance, smoothstep, trig identities");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!(
        "Generating {} random seed expressions with {} variants each (seed={})",
        count, variants_per_expr, seed
    );
    if include_targeted {
        println!("Including {} targeted micro-optimization seeds", generate_targeted_seeds().len());
    }

    // Find workspace root
    let workspace_root = find_workspace_root();
    let data_dir = workspace_root.join("pixelflow-pipeline/data");
    let cache_path = data_dir.join("benchmark_cache.jsonl");
    let variants_path = data_dir.join("egraph_variants.txt");

    // Create data directory
    fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    // Load existing cache
    let mut cache = BenchmarkCache::load(cache_path);

    // Expression generation with varied configurations
    // We'll create generators with different num_vars for diversity
    let base_seed = seed;
    let costs = CostModel::new();

    // Simple LCG for deterministic "random" config selection
    let mut rng_state = base_seed;

    // Helper: Box-Muller transform for Gaussian sampling
    let gaussian = |state: &mut u64, mean: f32, std_dev: f32| -> f32 {
        // Generate two uniform random numbers
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (*state >> 33) as f32 / (1u64 << 31) as f32;
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (*state >> 33) as f32 / (1u64 << 31) as f32;

        // Box-Muller transform
        let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + z * std_dev
    };

    // Track what we're generating
    let mut new_variants = 0;
    let mut cached_variants = 0;
    let mut expressions_with_variants = 0;
    let mut seen_trees: HashSet<String> = HashSet::new();

    // Collect new (uncached) variants for benchmarking
    let mut pending_benchmarks: Vec<(String, ExprTree)> = Vec::new();

    for seed_idx in 0..count {
        // Gaussian-sampled configuration for diverse expressions
        // num_vars: Gaussian centered at 2.5, std=1.0, clamped to [1, 4]
        let num_vars_f = gaussian(&mut rng_state, 2.5, 1.0);
        let num_vars = (num_vars_f.round() as usize).clamp(1, 4);

        // max_depth: Gaussian centered at 10, std=3.0, clamped to [4, 18]
        // Gradient clipping allows training on deep expressions without NaN
        let max_depth_f = gaussian(&mut rng_state, 10.0, 3.0);
        let max_depth = (max_depth_f.round() as usize).clamp(4, 18);

        // leaf_prob: Gaussian centered at 0.25, std=0.08, clamped to [0.1, 0.5]
        let leaf_prob = gaussian(&mut rng_state, 0.25, 0.08).clamp(0.1, 0.5);

        let config = ExprGenConfig {
            max_depth,
            leaf_prob,
            num_vars,
            include_fused: false, // Don't include fused ops in seeds - let e-graph find them
        };

        // Each expression gets its own generator with unique seed
        let expr_seed = base_seed.wrapping_add(seed_idx as u64);
        let mut generator = ExprGenerator::new(expr_seed, config);

        // Generate a seed expression
        let expr = generator.generate();

        // Skip trivial expressions
        if expr.node_count() < 3 {
            continue;
        }

        // Convert to ExprTree and insert into e-graph
        let seed_tree = nnue_expr_to_tree(&expr);
        let mut eg = EGraph::with_rules(all_training_rules());
        let root = eg.add_expr(&seed_tree);

        // Saturate to find equivalences
        eg.saturate_with_limit(50);

        // Extract variants
        let variants = eg.extract_variants(root, variants_per_expr, &costs);

        // Filter out type-invalid expressions (e.g., comparison results used in arithmetic)
        let valid_variants: Vec<_> = variants
            .into_iter()
            .filter(|tree| tree.is_type_valid())
            .collect();

        if valid_variants.len() > 1 {
            expressions_with_variants += 1;
        }

        // Process each variant
        for (var_idx, tree) in valid_variants.iter().enumerate() {
            let body = codegen::expr_tree_to_kernel_body(tree);

            // Deduplicate by generated code (within this run)
            if seen_trees.contains(&body) {
                continue;
            }
            seen_trees.insert(body.clone());

            // Check cache - only skip if we have ACTUAL benchmark data
            if let Some(entry) = cache.get(&body) {
                if entry.cost_ns.is_some() {
                    cached_variants += 1;
                    continue; // Already have benchmark data
                }
                // Entry exists but no benchmark yet - add to pending but don't re-cache
                pending_benchmarks.push((entry.name.clone(), tree.clone()));
                new_variants += 1;
                continue;
            }

            // Truly new expression - add to cache AND pending
            // Use globally unique ID (not seed_idx which can repeat across runs)
            let unique_id = cache.allocate_id();
            let name = format!("eg{:05}", unique_id);
            let features = extract_features_from_tree(tree);
            let feature_indices: Vec<u16> = features.iter().map(|f| f.to_index() as u16).collect();

            let entry = CacheEntry {
                expression: body.clone(),
                cost_ns: None, // Will be filled after benchmarking
                timestamp: current_timestamp(),
                name: name.clone(),
                egraph_cost: tree.cost(&costs),
                node_count: tree.node_count(),
                depth: tree.depth(),
                features: feature_indices,
                op_counts: count_ops(tree),
            };

            cache.append(entry);
            pending_benchmarks.push((name, tree.clone()));
            new_variants += 1;

            // Progress indicator every 10 new variants
            if new_variants % 10 == 0 {
                print!("\rGenerated {} new variants...", new_variants);
                std::io::stdout().flush().ok();
            }
        }
    }

    // Process targeted seeds if enabled
    if include_targeted {
        println!("\nProcessing targeted micro-optimization seeds...");
        let targeted = generate_targeted_seeds();

        for (family_name, seed_tree) in targeted {
            let mut eg = EGraph::with_rules(all_training_rules());
            let root = eg.add_expr(&seed_tree);

            // Saturate to find equivalences
            eg.saturate_with_limit(50);

            // Extract more variants for targeted seeds (they're small, we want diversity)
            let variants = eg.extract_variants(root, variants_per_expr.max(8), &costs);

            let valid_variants: Vec<_> = variants
                .into_iter()
                .filter(|tree| tree.is_type_valid())
                .collect();

            if valid_variants.len() > 1 {
                expressions_with_variants += 1;
            }

            for tree in valid_variants.iter() {
                let body = codegen::expr_tree_to_kernel_body(tree);

                if seen_trees.contains(&body) {
                    continue;
                }
                seen_trees.insert(body.clone());

                if let Some(entry) = cache.get(&body) {
                    if entry.cost_ns.is_some() {
                        cached_variants += 1;
                        continue;
                    }
                    pending_benchmarks.push((entry.name.clone(), tree.clone()));
                    new_variants += 1;
                    continue;
                }

                // Use family name as prefix for targeted expressions
                let unique_id = cache.allocate_id();
                let name = format!("tg_{}_{:04}", family_name, unique_id);
                let features = extract_features_from_tree(tree);
                let feature_indices: Vec<u16> = features.iter().map(|f| f.to_index() as u16).collect();

                let entry = CacheEntry {
                    expression: body.clone(),
                    cost_ns: None,
                    timestamp: current_timestamp(),
                    name: name.clone(),
                    egraph_cost: tree.cost(&costs),
                    node_count: tree.node_count(),
                    depth: tree.depth(),
                    features: feature_indices,
                    op_counts: count_ops(tree),
                };

                cache.append(entry);
                pending_benchmarks.push((name, tree.clone()));
                new_variants += 1;
            }
        }
        println!("Processed {} targeted families", generate_targeted_seeds().len());
    }

    println!();
    println!(
        "Found {} cached variants, generated {} new variants ({} seeds with multiple variants)",
        cached_variants, new_variants, expressions_with_variants
    );

    if pending_benchmarks.is_empty() {
        println!("\nAll variants already cached. No benchmarking needed.");
        println!("Cache contains {} entries ({} with benchmark data)",
            cache.entries.len(), cache.benchmarked_count());
        return;
    }

    // Append new corpus entries to bench_corpus.jsonl (no more generated .rs files).
    let corpus_path = workspace_root.join("pixelflow-pipeline/data/bench_corpus.jsonl");
    let corpus_jsonl = codegen::generate_corpus_jsonl(&pending_benchmarks);
    let mut corpus_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&corpus_path)
        .expect("Failed to open bench_corpus.jsonl");
    corpus_file
        .write_all(corpus_jsonl.as_bytes())
        .expect("Failed to write corpus");
    println!("Appended {} entries to {}", pending_benchmarks.len(), corpus_path.display());

    // Also write variant data file (for all entries including cached)
    write_variants_file(&cache, &variants_path);

    // Print statistics
    print_cache_stats(&cache, &costs);

    println!("\nNext steps:");
    println!("  1. JIT benchmark: cargo run --release -p pixelflow-pipeline --bin bench_jit_corpus --features training");
    println!("  2. Retrain Judge: cargo run -p pixelflow-pipeline --bin train_judge --release --features training");
    println!("\nThe cache will grow with each run. Current: {} entries ({} benchmarked)",
        cache.entries.len(), cache.benchmarked_count());
}

fn write_variants_file(cache: &BenchmarkCache, path: &PathBuf) {
    let mut data_file = File::create(path).expect("Failed to create data file");
    writeln!(
        data_file,
        "# E-Graph Variants - Generated Training Data"
    )
    .unwrap();
    writeln!(
        data_file,
        "# Format: name|seed_idx|variant_idx|node_count|depth|cost|features..."
    )
    .unwrap();
    writeln!(data_file, "#").unwrap();

    // Sort entries by name for consistent output
    let mut entries: Vec<_> = cache.entries.values().collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    for entry in entries {
        // Parse seed_idx and variant_idx from name (eg0001v2 → 1, 2)
        let (seed_idx, variant_idx) = parse_variant_name(&entry.name).unwrap_or((0, 0));
        let features_str: Vec<String> = entry.features.iter().map(|f| f.to_string()).collect();
        writeln!(
            data_file,
            "{}|{}|{}|{}|{}|{}|{}",
            entry.name,
            seed_idx,
            variant_idx,
            entry.node_count,
            entry.depth,
            entry.egraph_cost,
            features_str.join(",")
        )
        .expect("Failed to write data");
    }
    println!("Wrote variant data to {}", path.display());
}

fn parse_variant_name(name: &str) -> Option<(usize, usize)> {
    // Parse both formats:
    // - New: "eg00001" → (1, 0)
    // - Old: "eg0001v2" → (1, 2)
    let rest = name.strip_prefix("eg")?;
    if let Some(v_pos) = rest.find('v') {
        // Old format with variant suffix
        let seed_idx: usize = rest[..v_pos].parse().ok()?;
        let variant_idx: usize = rest[v_pos + 1..].parse().ok()?;
        Some((seed_idx, variant_idx))
    } else {
        // New format - just a unique ID
        let unique_id: usize = rest.parse().ok()?;
        Some((unique_id, 0))
    }
}

/// Convert NNUE Expr to ExprTree.
/// Uses op_from_kind to handle ALL operations dynamically.
/// Note: OpType is just an alias for OpKind, so no conversion needed.
fn nnue_expr_to_tree(expr: &Expr) -> ExprTree {
    match expr {
        Expr::Var(i) => ExprTree::var(*i),
        Expr::Const(v) => ExprTree::constant(*v),
        Expr::Param(i) => panic!("Expr::Param({i}) in nnue_expr_to_tree — call substitute_params first"),
        Expr::Unary(op, a) => {
            let a_tree = nnue_expr_to_tree(a);
            match op_from_kind(*op) {
                Some(op_ref) => ExprTree::Op {
                    op: op_ref,
                    children: vec![a_tree],
                },
                None => {
                    eprintln!("WARNING: Unknown unary op {:?}, falling back to var(0)", op);
                    ExprTree::var(0)
                }
            }
        }
        Expr::Binary(op, a, b) => {
            let a_tree = nnue_expr_to_tree(a);
            let b_tree = nnue_expr_to_tree(b);
            match op_from_kind(*op) {
                Some(op_ref) => ExprTree::Op {
                    op: op_ref,
                    children: vec![a_tree, b_tree],
                },
                None => {
                    eprintln!("WARNING: Unknown binary op {:?}, falling back to var(0)", op);
                    ExprTree::var(0)
                }
            }
        }
        Expr::Ternary(op, a, b, c) => {
            let a_tree = nnue_expr_to_tree(a);
            let b_tree = nnue_expr_to_tree(b);
            let c_tree = nnue_expr_to_tree(c);
            match op_from_kind(*op) {
                Some(op_ref) => ExprTree::Op {
                    op: op_ref,
                    children: vec![a_tree, b_tree, c_tree],
                },
                None => {
                    eprintln!("WARNING: Unknown ternary op {:?}, falling back to var(0)", op);
                    ExprTree::var(0)
                }
            }
        }
        Expr::Nary(op, children) => {
            let child_trees: Vec<_> = children.iter().map(nnue_expr_to_tree).collect();
            match op_from_kind(*op) {
                Some(op_ref) => ExprTree::Op {
                    op: op_ref,
                    children: child_trees,
                },
                None => {
                    eprintln!("WARNING: Unknown nary op {:?}, falling back to var(0)", op);
                    ExprTree::var(0)
                }
            }
        }
    }
}

/// Extract NNUE features from an ExprTree.
fn extract_features_from_tree(tree: &ExprTree) -> Vec<HalfEPFeature> {
    let expr = tree_to_nnue_expr(tree);
    extract_features(&expr)
}

/// Convert ExprTree back to NNUE Expr for feature extraction.
fn tree_to_nnue_expr(tree: &ExprTree) -> Expr {
    match tree {
        ExprTree::Leaf(Leaf::Var(i)) => Expr::Var(*i),
        ExprTree::Leaf(Leaf::Const(v)) => Expr::Const(*v),
        ExprTree::Op { op, children } => {
            let name = op.name();
            match (name, children.len()) {
                // Unary ops
                ("neg", 1) => Expr::Unary(OpType::Neg, Box::new(tree_to_nnue_expr(&children[0]))),
                ("sqrt", 1) => Expr::Unary(OpType::Sqrt, Box::new(tree_to_nnue_expr(&children[0]))),
                ("rsqrt", 1) => Expr::Unary(OpType::Rsqrt, Box::new(tree_to_nnue_expr(&children[0]))),
                ("abs", 1) => Expr::Unary(OpType::Abs, Box::new(tree_to_nnue_expr(&children[0]))),
                ("recip", 1) => Expr::Binary(
                    OpType::Div,
                    Box::new(Expr::Const(1.0)),
                    Box::new(tree_to_nnue_expr(&children[0])),
                ),
                // Binary ops
                ("add", 2) => Expr::Binary(
                    OpType::Add,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                ),
                ("sub", 2) => Expr::Binary(
                    OpType::Sub,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                ),
                ("mul", 2) => Expr::Binary(
                    OpType::Mul,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                ),
                ("div", 2) => Expr::Binary(
                    OpType::Div,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                ),
                ("min", 2) => Expr::Binary(
                    OpType::Min,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                ),
                ("max", 2) => Expr::Binary(
                    OpType::Max,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                ),
                // Ternary ops
                ("mul_add", 3) => Expr::Ternary(
                    OpType::MulAdd,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                    Box::new(tree_to_nnue_expr(&children[2])),
                ),
                // Unknown op - approximate based on arity
                (_, 1) => Expr::Unary(OpType::Sqrt, Box::new(tree_to_nnue_expr(&children[0]))),
                (_, 2) => Expr::Binary(
                    OpType::Mul,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                ),
                (_, 3) => Expr::Ternary(
                    OpType::MulAdd,
                    Box::new(tree_to_nnue_expr(&children[0])),
                    Box::new(tree_to_nnue_expr(&children[1])),
                    Box::new(tree_to_nnue_expr(&children[2])),
                ),
                _ => Expr::Const(0.0),
            }
        }
    }
}

fn print_cache_stats(cache: &BenchmarkCache, costs: &CostModel) {
    if cache.entries.is_empty() {
        println!("\nNo variants in cache.");
        return;
    }

    let entries: Vec<_> = cache.entries.values().collect();

    let total_nodes: usize = entries.iter().map(|r| r.node_count).sum();
    let total_depth: usize = entries.iter().map(|r| r.depth).sum();
    let total_cost: usize = entries.iter().map(|r| r.egraph_cost).sum();

    let avg_nodes = total_nodes as f64 / entries.len() as f64;
    let avg_depth = total_depth as f64 / entries.len() as f64;
    let avg_cost = total_cost as f64 / entries.len() as f64;

    let max_depth = entries.iter().map(|r| r.depth).max().unwrap_or(0);
    let max_nodes = entries.iter().map(|r| r.node_count).max().unwrap_or(0);

    // Count benchmarked vs pending
    let benchmarked: Vec<_> = entries.iter().filter(|e| e.cost_ns.is_some()).collect();
    let pending = entries.len() - benchmarked.len();

    // Benchmark cost statistics (only for entries with real costs)
    let (avg_bench_ns, min_bench_ns, max_bench_ns) = if !benchmarked.is_empty() {
        let costs_ns: Vec<f64> = benchmarked.iter().filter_map(|e| e.cost_ns).collect();
        let avg = costs_ns.iter().sum::<f64>() / costs_ns.len() as f64;
        let min = costs_ns.iter().copied().fold(f64::INFINITY, f64::min);
        let max = costs_ns.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (Some(avg), Some(min), Some(max))
    } else {
        (None, None, None)
    };

    println!("\nCache Statistics:");
    println!("  Total entries: {}", entries.len());
    println!("  Benchmarked: {} ({:.1}%)", benchmarked.len(), 100.0 * benchmarked.len() as f64 / entries.len() as f64);
    println!("  Pending benchmark: {}", pending);
    println!("  Average node count: {:.1}", avg_nodes);
    println!("  Average depth: {:.1}", avg_depth);
    println!("  Average e-graph cost: {:.1}", avg_cost);
    println!("  Max depth: {}", max_depth);
    println!("  Max nodes: {}", max_nodes);

    if let (Some(avg), Some(min), Some(max)) = (avg_bench_ns, min_bench_ns, max_bench_ns) {
        println!("\nBenchmark Results (ns):");
        println!("  Average: {:.2}", avg);
        println!("  Min: {:.2}", min);
        println!("  Max: {:.2}", max);
    }

    // Show cost model info
    let cost_map = costs.to_map();
    println!("\nCost model weights:");
    println!("  add={}, mul={}, div={}",
        cost_map.get("add").unwrap_or(&0),
        cost_map.get("mul").unwrap_or(&0),
        cost_map.get("div").unwrap_or(&0));
    println!("  mul_add={}, sqrt={}",
        cost_map.get("mul_add").unwrap_or(&0),
        cost_map.get("sqrt").unwrap_or(&0));
}

// ============================================================================
// TARGETED SEED GENERATION
// ============================================================================

/// Generate targeted seed expressions that exercise specific rewrite families.
///
/// These are small kernels designed to produce equivalent variants that differ
/// only in micro-optimizations (div vs recip, sub vs add+neg, etc.).
fn generate_targeted_seeds() -> Vec<(&'static str, ExprTree)> {
    let x = || ExprTree::var(0);
    let y = || ExprTree::var(1);
    let z = || ExprTree::var(2);
    let c = |v: f32| ExprTree::Leaf(Leaf::Const(v));

    vec![
        // === DIVISION FAMILY ===
        // Simple div: x/y → x*recip(y)
        ("div_simple", ExprTree::Op {
            op: &ops::Div,
            children: vec![x(), y()],
        }),
        // Div with sum denominator: x/(y+1)
        ("div_sum_denom", ExprTree::Op {
            op: &ops::Div,
            children: vec![x(), ExprTree::add(y(), c(1.0))],
        }),
        // Chained div: x/y/z → x*recip(y)*recip(z)
        ("div_chained", ExprTree::Op {
            op: &ops::Div,
            children: vec![
                ExprTree::Op { op: &ops::Div, children: vec![x(), y()] },
                z(),
            ],
        }),
        // Div by constant: x/2 → x*0.5
        ("div_const", ExprTree::Op {
            op: &ops::Div,
            children: vec![x(), c(2.0)],
        }),
        // Product over sum: (x*y)/(x+y)
        ("div_product_sum", ExprTree::Op {
            op: &ops::Div,
            children: vec![
                ExprTree::mul(x(), y()),
                ExprTree::add(x(), y()),
            ],
        }),
        // Soft clamp: x/(|x|+1)
        ("div_soft_clamp", ExprTree::Op {
            op: &ops::Div,
            children: vec![
                x(),
                ExprTree::add(
                    ExprTree::Op { op: &ops::Abs, children: vec![x()] },
                    c(1.0),
                ),
            ],
        }),

        // === NEGATION FAMILY ===
        // Simple sub: x-y → x+neg(y)
        ("sub_simple", ExprTree::Op {
            op: &ops::Sub,
            children: vec![x(), y()],
        }),
        // Double neg: -(-x) → x
        ("neg_double", ExprTree::Op {
            op: &ops::Neg,
            children: vec![ExprTree::Op { op: &ops::Neg, children: vec![x()] }],
        }),
        // Neg of sum: -(x+y) → neg(x)+neg(y)
        ("neg_sum", ExprTree::Op {
            op: &ops::Neg,
            children: vec![ExprTree::add(x(), y())],
        }),
        // Sub from zero: 0-x → neg(x)
        ("sub_zero", ExprTree::Op {
            op: &ops::Sub,
            children: vec![c(0.0), x()],
        }),
        // Difference of products: x*y - x*z → x*(y-z)
        ("sub_factor", ExprTree::Op {
            op: &ops::Sub,
            children: vec![
                ExprTree::mul(x(), y()),
                ExprTree::mul(x(), z()),
            ],
        }),

        // === SQRT FAMILY ===
        // Recip sqrt: 1/sqrt(x) → rsqrt(x)
        ("rsqrt_form", ExprTree::Op {
            op: &ops::Div,
            children: vec![
                c(1.0),
                ExprTree::Op { op: &ops::Sqrt, children: vec![x()] },
            ],
        }),
        // Sqrt of square: sqrt(x*x) → |x|
        ("sqrt_sq", ExprTree::Op {
            op: &ops::Sqrt,
            children: vec![ExprTree::mul(x(), x())],
        }),
        // Normalize pattern: x/sqrt(x*x+y*y)
        ("normalize_2d", ExprTree::Op {
            op: &ops::Div,
            children: vec![
                x(),
                ExprTree::Op {
                    op: &ops::Sqrt,
                    children: vec![ExprTree::add(
                        ExprTree::mul(x(), x()),
                        ExprTree::mul(y(), y()),
                    )],
                },
            ],
        }),
        // rsqrt chain: x * rsqrt(y)
        ("scaled_rsqrt", ExprTree::mul(
            x(),
            ExprTree::Op { op: &ops::Rsqrt, children: vec![y()] },
        )),

        // === COMPOUND PATTERNS ===
        // Lerp: x + t*(y-x) → (1-t)*x + t*y
        ("lerp", ExprTree::add(
            x(),
            ExprTree::mul(
                z(), // t
                ExprTree::Op { op: &ops::Sub, children: vec![y(), x()] },
            ),
        )),
        // Distance squared: (x-a)² + (y-b)²
        ("dist_sq", ExprTree::add(
            ExprTree::mul(
                ExprTree::Op { op: &ops::Sub, children: vec![x(), c(0.5)] },
                ExprTree::Op { op: &ops::Sub, children: vec![x(), c(0.5)] },
            ),
            ExprTree::mul(
                ExprTree::Op { op: &ops::Sub, children: vec![y(), c(0.5)] },
                ExprTree::Op { op: &ops::Sub, children: vec![y(), c(0.5)] },
            ),
        )),
        // Smoothstep inner: 3*t² - 2*t³
        ("smoothstep_inner", ExprTree::Op {
            op: &ops::Sub,
            children: vec![
                ExprTree::mul(c(3.0), ExprTree::mul(x(), x())),
                ExprTree::mul(c(2.0), ExprTree::mul(x(), ExprTree::mul(x(), x()))),
            ],
        }),
        // Exponential decay: exp(-x*x)
        ("exp_decay", ExprTree::Op {
            op: &ops::Exp,
            children: vec![ExprTree::Op {
                op: &ops::Neg,
                children: vec![ExprTree::mul(x(), x())],
            }],
        }),
        // Double angle prep: 2*sin(x)*cos(x)
        ("double_angle", ExprTree::mul(
            c(2.0),
            ExprTree::mul(
                ExprTree::Op { op: &ops::Sin, children: vec![x()] },
                ExprTree::Op { op: &ops::Cos, children: vec![x()] },
            ),
        )),
        // Pythagorean: sin²(x) + cos²(x)
        ("pythagorean", ExprTree::add(
            ExprTree::mul(
                ExprTree::Op { op: &ops::Sin, children: vec![x()] },
                ExprTree::Op { op: &ops::Sin, children: vec![x()] },
            ),
            ExprTree::mul(
                ExprTree::Op { op: &ops::Cos, children: vec![x()] },
                ExprTree::Op { op: &ops::Cos, children: vec![x()] },
            ),
        )),
        // Radial falloff: exp(-4*(x²+y²))
        ("radial_falloff", ExprTree::Op {
            op: &ops::Exp,
            children: vec![ExprTree::mul(
                c(-4.0),
                ExprTree::add(
                    ExprTree::mul(x(), x()),
                    ExprTree::mul(y(), y()),
                ),
            )],
        }),
    ]
}

/// Find workspace root by looking for Cargo.toml with [workspace]
fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

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
