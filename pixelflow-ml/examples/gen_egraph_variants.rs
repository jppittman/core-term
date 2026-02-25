//! Generate training data using e-graph variant extraction with lazy benchmarking.
//!
//! This example demonstrates the clean training pipeline with caching:
//! 1. Generate seed expressions
//! 2. Insert into e-graph and saturate to find equivalents
//! 3. Extract N variants from the same e-class
//! 4. Check cache - skip expressions already benchmarked
//! 5. Write benchmark code for NEW expressions only
//!
//! # Usage
//!
//! ```bash
//! # Generate variants (incrementally adds to cache)
//! cargo run -p pixelflow-ml --example gen_egraph_variants --features training -- --count 100 --variants 5
//!
//! # Run benchmarks for new expressions
//! cargo bench -p pixelflow-ml --bench generated_kernels
//!
//! # Collect costs and update cache
//! cargo run -p pixelflow-ml --example collect_benchmark_costs --features training
//! ```
//!
//! The cache (`pixelflow-ml/data/benchmark_cache.jsonl`) persists across runs,
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

use pixelflow_ml::nnue::{
    Expr, ExprGenConfig, ExprGenerator, HalfEPFeature, OpType, extract_features,
};
use pixelflow_search::egraph::{CostModel, EGraph, ExprTree, Leaf, codegen, ops};

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
            self.cost_ns
                .map(|c| c.to_string())
                .unwrap_or_else(|| "null".to_string()),
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
    let end = rest.find(|c: char| {
        !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+'
    })?;
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
            .collect(),
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
}

impl BenchmarkCache {
    fn load(path: PathBuf) -> Self {
        let mut entries = HashMap::new();

        if path.exists()
            && let Ok(file) = File::open(&path) {
                let reader = BufReader::new(file);
                for line in reader.lines().map_while(Result::ok) {
                    if line.starts_with('#') || line.trim().is_empty() {
                        continue;
                    }
                    if let Some(entry) = CacheEntry::from_json(&line) {
                        entries.insert(entry.expression.clone(), entry);
                    }
                }
            }

        println!("Loaded {} cached benchmark entries", entries.len());
        BenchmarkCache { entries, path }
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
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            && let Err(e) = writeln!(file, "{}", json) {
                eprintln!("Warning: failed to write cache entry: {}", e);
            }

        self.entries.insert(expression, entry);
    }

    /// Get entries that are pending benchmark (have cost_ns = None).
    fn pending_entries(&self) -> Vec<&CacheEntry> {
        self.entries
            .values()
            .filter(|e| e.cost_ns.is_none())
            .collect()
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
            writeln!(
                file,
                "# Each line is a JSON object with expression, cost_ns, etc."
            )
            .ok();
            for entry in self.entries.values() {
                writeln!(file, "{}", entry.to_json()).ok();
            }
        }
    }

    /// Count entries with actual benchmark results.
    fn benchmarked_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.cost_ns.is_some())
            .count()
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
            "--help" | "-h" => {
                println!("Usage: gen_egraph_variants [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --count N      Number of seed expressions (default: 100)");
                println!("  --variants N   Variants to extract per expression (default: 5)");
                println!("  --seed N       Random seed (default: 42)");
                println!();
                println!("This generates training data by:");
                println!("  1. Creating random seed expressions");
                println!("  2. Inserting each into an e-graph and saturating");
                println!("  3. Extracting N different equivalent forms");
                println!("  4. Writing benchmark code for all variants");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!(
        "Generating {} seed expressions with {} variants each (seed={})",
        count, variants_per_expr, seed
    );

    // Find workspace root
    let workspace_root = find_workspace_root();
    let bench_path = workspace_root.join("pixelflow-ml/benches/generated_kernels.rs");
    let data_dir = workspace_root.join("pixelflow-ml/data");
    let cache_path = data_dir.join("benchmark_cache.jsonl");
    let variants_path = data_dir.join("egraph_variants.txt");

    // Create data directory
    fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    // Load existing cache
    let mut cache = BenchmarkCache::load(cache_path);

    // Configure expression generator
    let config = ExprGenConfig {
        max_depth: 5,
        leaf_prob: 0.3,
        num_vars: 4,
        include_fused: false, // Don't include fused ops in seeds - let e-graph find them
    };

    let mut generator = ExprGenerator::new(seed, config);
    let costs = CostModel::fully_optimized();

    // Track what we're generating
    let mut new_variants = 0;
    let mut cached_variants = 0;
    let mut expressions_with_variants = 0;
    let mut seen_trees: HashSet<String> = HashSet::new();

    // Collect new (uncached) variants for benchmarking
    let mut pending_benchmarks: Vec<(String, ExprTree)> = Vec::new();

    for seed_idx in 0..count {
        // Generate a seed expression
        let expr = generator.generate();

        // Skip trivial expressions
        if expr.node_count() < 3 {
            continue;
        }

        // Convert to ExprTree and insert into e-graph
        let seed_tree = nnue_expr_to_tree(&expr);
        let mut eg = EGraph::new();
        let root = eg.add_expr(&seed_tree);

        // Saturate to find equivalences
        eg.saturate_with_limit(50);

        // Extract variants
        let variants = eg.extract_variants(root, variants_per_expr, &costs);

        if variants.len() > 1 {
            expressions_with_variants += 1;
        }

        // Process each variant
        for (var_idx, tree) in variants.iter().enumerate() {
            let body = codegen::expr_tree_to_kernel_body(tree);

            // Deduplicate by generated code (within this run)
            if seen_trees.contains(&body) {
                continue;
            }
            seen_trees.insert(body.clone());

            // Check cache
            if cache.contains(&body) {
                cached_variants += 1;
                continue; // Already have benchmark data for this expression
            }

            // New expression - add to cache (pending benchmark)
            let name = format!("eg{:04}v{}", seed_idx, var_idx);
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

            // Append to cache immediately (incremental write)
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

    println!();
    println!(
        "Found {} cached variants, generated {} new variants ({} seeds with multiple variants)",
        cached_variants, new_variants, expressions_with_variants
    );

    if pending_benchmarks.is_empty() {
        println!("\nAll variants already cached. No benchmarking needed.");
        println!(
            "Cache contains {} entries ({} with benchmark data)",
            cache.entries.len(),
            cache.benchmarked_count()
        );
        return;
    }

    // Generate benchmark file for NEW variants only
    let bench_code = codegen::generate_benchmark_file(&pending_benchmarks);
    let mut bench_file = File::create(&bench_path).expect("Failed to create benchmark file");
    bench_file
        .write_all(bench_code.as_bytes())
        .expect("Failed to write benchmark");
    println!(
        "Wrote {} benchmarks to {}",
        pending_benchmarks.len(),
        bench_path.display()
    );

    // Also write variant data file (for all entries including cached)
    write_variants_file(&cache, &variants_path);

    // Print statistics
    print_cache_stats(&cache, &costs);

    println!("\nNext steps:");
    println!("  1. Run benchmarks: cargo bench -p pixelflow-ml --bench generated_kernels");
    println!(
        "  2. Update cache:   cargo run -p pixelflow-ml --example collect_benchmark_costs --features training"
    );
    println!(
        "\nThe cache will grow with each run. Current: {} entries ({} benchmarked)",
        cache.entries.len(),
        cache.benchmarked_count()
    );
}

fn write_variants_file(cache: &BenchmarkCache, path: &PathBuf) {
    let mut data_file = File::create(path).expect("Failed to create data file");
    writeln!(data_file, "# E-Graph Variants - Generated Training Data").unwrap();
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
    // Parse "eg0001v2" → (1, 2)
    let name = name.strip_prefix("eg")?;
    let v_pos = name.find('v')?;
    let seed_idx: usize = name[..v_pos].parse().ok()?;
    let variant_idx: usize = name[v_pos + 1..].parse().ok()?;
    Some((seed_idx, variant_idx))
}

/// Convert NNUE Expr to ExprTree.
fn nnue_expr_to_tree(expr: &Expr) -> ExprTree {
    match expr {
        Expr::Var(i) => ExprTree::var(*i),
        Expr::Const(v) => ExprTree::constant(*v),
        Expr::Unary(op, a) => {
            let a_tree = nnue_expr_to_tree(a);
            match op {
                OpType::Neg => ExprTree::op_neg(a_tree),
                OpType::Sqrt => ExprTree::sqrt(a_tree),
                OpType::Rsqrt => ExprTree::Op {
                    op: &ops::Rsqrt,
                    children: vec![a_tree],
                },
                OpType::Abs => ExprTree::abs(a_tree),
                _ => ExprTree::var(0), // Fallback for unexpected ops
            }
        }
        Expr::Binary(op, a, b) => {
            let a_tree = nnue_expr_to_tree(a);
            let b_tree = nnue_expr_to_tree(b);
            match op {
                OpType::Add => ExprTree::op_add(a_tree, b_tree),
                OpType::Sub => ExprTree::op_sub(a_tree, b_tree),
                OpType::Mul => ExprTree::op_mul(a_tree, b_tree),
                OpType::Div => ExprTree::op_div(a_tree, b_tree),
                OpType::Min => ExprTree::min(a_tree, b_tree),
                OpType::Max => ExprTree::max(a_tree, b_tree),
                _ => ExprTree::var(0), // Fallback
            }
        }
        Expr::Ternary(op, a, b, c) => {
            let a_tree = nnue_expr_to_tree(a);
            let b_tree = nnue_expr_to_tree(b);
            let c_tree = nnue_expr_to_tree(c);
            match op {
                OpType::MulAdd => ExprTree::mul_add(a_tree, b_tree, c_tree),
                _ => ExprTree::var(0), // Fallback
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
                ("rsqrt", 1) => {
                    Expr::Unary(OpType::Rsqrt, Box::new(tree_to_nnue_expr(&children[0])))
                }
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
    println!(
        "  Benchmarked: {} ({:.1}%)",
        benchmarked.len(),
        100.0 * benchmarked.len() as f64 / entries.len() as f64
    );
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
    println!(
        "  add={}, mul={}, div={}",
        cost_map.get("add").unwrap_or(&0),
        cost_map.get("mul").unwrap_or(&0),
        cost_map.get("div").unwrap_or(&0)
    );
    println!(
        "  mul_add={}, sqrt={}",
        cost_map.get("mul_add").unwrap_or(&0),
        cost_map.get("sqrt").unwrap_or(&0)
    );
}

/// Find workspace root by looking for Cargo.toml with [workspace]
fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists()
            && let Ok(contents) = fs::read_to_string(&cargo_toml)
                && contents.contains("[workspace]") {
                    return current;
                }
        if !current.pop() {
            panic!("Could not find workspace root");
        }
    }
}
