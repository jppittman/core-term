//! Train "The Judge": Cost Model with Learned Weights
//!
//! This is the first component in the two-loop training architecture:
//!
//! ## The Two-Loop Architecture
//!
//! **The Judge** (this file): Classical cost model with learned weights
//! - Predicts cost(expr) → nanoseconds
//! - Trained on real SIMD benchmark data (slow, run weekly)
//! - Once trained, evaluates expressions in nanoseconds
//!
//! **The Guide** (NNUE): Neural network predicting which rewrites are profitable
//! - Trained against The Judge (fast, no compilation needed)
//! - Learns to navigate the e-graph search space
//!
//! ## Training Loop
//!
//! ```text
//! for each pair (expr_a, expr_b) from benchmark_cache.jsonl:
//!   1. Get real SIMD costs from benchmarks
//!   2. Predict costs with current weights
//!   3. Compute ranking loss: does model ranking match benchmark ranking?
//!   4. Update weights via gradient descent
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # First, run benchmarks to populate the cache:
//! cargo run -p pixelflow-ml --example gen_egraph_variants --features training -- --count 100
//! cargo bench -p pixelflow-ml --bench generated_kernels
//! cargo run -p pixelflow-ml --example collect_benchmark_costs --features training
//!
//! # Then train The Judge:
//! cargo run -p pixelflow-ml --example train_cost_model --features training --release
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ============================================================================
// Cost Model (trainable weights)
// ============================================================================

/// Trainable cost model weights.
///
/// These are the parameters we're learning: how much does each operation
/// actually cost in practice?
#[derive(Clone, Debug)]
pub struct TrainableCostModel {
    // Operation costs (in abstract units, will be scaled)
    pub add: f32,
    pub sub: f32,
    pub mul: f32,
    pub div: f32,
    pub neg: f32,
    pub sqrt: f32,
    pub rsqrt: f32,
    pub abs: f32,
    pub min: f32,
    pub max: f32,
    pub mul_add: f32,
    pub mul_rsqrt: f32,
}

impl Default for TrainableCostModel {
    fn default() -> Self {
        // Start with reasonable guesses (will be refined by training)
        Self {
            add: 4.0,
            sub: 4.0,
            mul: 5.0,
            div: 15.0,
            neg: 1.0,
            sqrt: 15.0,
            rsqrt: 5.0,
            abs: 1.0,
            min: 4.0,
            max: 4.0,
            mul_add: 10.0,  // Key question: is FMA worth it?
            mul_rsqrt: 5.0,
        }
    }
}

impl TrainableCostModel {
    /// Predict cost for an expression.
    pub fn predict(&self, expr: &Expr) -> f32 {
        match expr {
            Expr::Var(_) | Expr::Const(_) => 0.0,
            Expr::Unary(op, a) => {
                let child_cost = self.predict(a);
                let op_cost = match op {
                    OpType::Neg => self.neg,
                    OpType::Sqrt => self.sqrt,
                    OpType::Rsqrt => self.rsqrt,
                    OpType::Abs => self.abs,
                    _ => 5.0, // Default for sin, cos, etc.
                };
                child_cost + op_cost
            }
            Expr::Binary(op, a, b) => {
                let child_cost = self.predict(a) + self.predict(b);
                let op_cost = match op {
                    OpType::Add => self.add,
                    OpType::Sub => self.sub,
                    OpType::Mul => self.mul,
                    OpType::Div => self.div,
                    OpType::Min => self.min,
                    OpType::Max => self.max,
                    _ => 5.0,
                };
                child_cost + op_cost
            }
            Expr::Ternary(op, a, b, c) => {
                let child_cost = self.predict(a) + self.predict(b) + self.predict(c);
                let op_cost = match op {
                    OpType::MulAdd => self.mul_add,
                    OpType::MulRsqrt => self.mul_rsqrt,
                    _ => 10.0,
                };
                child_cost + op_cost
            }
        }
    }

    /// Get weights as a vector for gradient descent.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.add, self.sub, self.mul, self.div, self.neg,
            self.sqrt, self.rsqrt, self.abs, self.min, self.max,
            self.mul_add, self.mul_rsqrt,
        ]
    }

    /// Set weights from a vector.
    pub fn from_vec(&mut self, v: &[f32]) {
        self.add = v[0].max(0.1);
        self.sub = v[1].max(0.1);
        self.mul = v[2].max(0.1);
        self.div = v[3].max(0.1);
        self.neg = v[4].max(0.1);
        self.sqrt = v[5].max(0.1);
        self.rsqrt = v[6].max(0.1);
        self.abs = v[7].max(0.1);
        self.min = v[8].max(0.1);
        self.max = v[9].max(0.1);
        self.mul_add = v[10].max(0.1);
        self.mul_rsqrt = v[11].max(0.1);
    }

    /// Predict cost from pre-computed op counts (for benchmark data).
    pub fn predict_from_counts(&self, counts: &OpCounts) -> f32 {
        counts.add as f32 * self.add
            + counts.sub as f32 * self.sub
            + counts.mul as f32 * self.mul
            + counts.div as f32 * self.div
            + counts.neg as f32 * self.neg
            + counts.sqrt as f32 * self.sqrt
            + counts.rsqrt as f32 * self.rsqrt
            + counts.abs as f32 * self.abs
            + counts.min as f32 * self.min
            + counts.max as f32 * self.max
            + counts.mul_add as f32 * self.mul_add
            + counts.mul_rsqrt as f32 * self.mul_rsqrt
    }

    /// Count operations in an expression (for gradient computation).
    pub fn count_ops(&self, expr: &Expr) -> OpCounts {
        let mut counts = OpCounts::default();
        self.count_ops_inner(expr, &mut counts);
        counts
    }

    fn count_ops_inner(&self, expr: &Expr, counts: &mut OpCounts) {
        match expr {
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::Unary(op, a) => {
                self.count_ops_inner(a, counts);
                match op {
                    OpType::Neg => counts.neg += 1,
                    OpType::Sqrt => counts.sqrt += 1,
                    OpType::Rsqrt => counts.rsqrt += 1,
                    OpType::Abs => counts.abs += 1,
                    _ => {}
                }
            }
            Expr::Binary(op, a, b) => {
                self.count_ops_inner(a, counts);
                self.count_ops_inner(b, counts);
                match op {
                    OpType::Add => counts.add += 1,
                    OpType::Sub => counts.sub += 1,
                    OpType::Mul => counts.mul += 1,
                    OpType::Div => counts.div += 1,
                    OpType::Min => counts.min += 1,
                    OpType::Max => counts.max += 1,
                    _ => {}
                }
            }
            Expr::Ternary(op, a, b, c) => {
                self.count_ops_inner(a, counts);
                self.count_ops_inner(b, counts);
                self.count_ops_inner(c, counts);
                match op {
                    OpType::MulAdd => counts.mul_add += 1,
                    OpType::MulRsqrt => counts.mul_rsqrt += 1,
                    _ => {}
                }
            }
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct OpCounts {
    pub add: usize,
    pub sub: usize,
    pub mul: usize,
    pub div: usize,
    pub neg: usize,
    pub sqrt: usize,
    pub rsqrt: usize,
    pub abs: usize,
    pub min: usize,
    pub max: usize,
    pub mul_add: usize,
    pub mul_rsqrt: usize,
}

impl OpCounts {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.add as f32, self.sub as f32, self.mul as f32, self.div as f32,
            self.neg as f32, self.sqrt as f32, self.rsqrt as f32, self.abs as f32,
            self.min as f32, self.max as f32, self.mul_add as f32, self.mul_rsqrt as f32,
        ]
    }
}

// ============================================================================
// Expression Types (from pixelflow_pipeline::nnue)
// ============================================================================

use pixelflow_pipeline::nnue::{Expr, OpType, BwdGenerator, BwdGenConfig};

// ============================================================================
// Cost Cache (memoization)
// ============================================================================

/// Memoized benchmark costs.
///
/// Running actual benchmarks is slow. We cache results keyed by expression
/// structure so repeated evaluations are instant.
pub struct CostCache {
    /// expr_hash -> measured_cost_ns
    cache: HashMap<u64, f32>,
    /// Cache hits
    pub hits: usize,
    /// Cache misses (actual benchmark runs)
    pub misses: usize,
}

impl CostCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Get cached cost or compute it.
    pub fn get_or_compute<F>(&mut self, expr: &Expr, compute: F) -> f32
    where
        F: FnOnce(&Expr) -> f32,
    {
        let hash = expr_hash(expr);
        if let Some(&cost) = self.cache.get(&hash) {
            self.hits += 1;
            cost
        } else {
            self.misses += 1;
            let cost = compute(expr);
            self.cache.insert(hash, cost);
            cost
        }
    }

    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f32 / total as f32 }
    }
}

/// Simple hash of expression structure.
fn expr_hash(expr: &Expr) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    expr_hash_inner(expr, &mut hasher);
    hasher.finish()
}

fn expr_hash_inner<H: std::hash::Hasher>(expr: &Expr, hasher: &mut H) {
    use std::hash::Hash as _;
    match expr {
        Expr::Var(i) => { 0u8.hash(hasher); i.hash(hasher); }
        Expr::Const(c) => { 1u8.hash(hasher); c.to_bits().hash(hasher); }
        Expr::Unary(op, a) => {
            2u8.hash(hasher);
            (*op as u8).hash(hasher);
            expr_hash_inner(a, hasher);
        }
        Expr::Binary(op, a, b) => {
            3u8.hash(hasher);
            (*op as u8).hash(hasher);
            expr_hash_inner(a, hasher);
            expr_hash_inner(b, hasher);
        }
        Expr::Ternary(op, a, b, c) => {
            4u8.hash(hasher);
            (*op as u8).hash(hasher);
            expr_hash_inner(a, hasher);
            expr_hash_inner(b, hasher);
            expr_hash_inner(c, hasher);
        }
    }
}

// ============================================================================
// Benchmark Data Loading
// ============================================================================

/// A benchmark sample with expression and measured cost.
#[derive(Debug, Clone)]
pub struct BenchmarkSample {
    pub name: String,
    pub expression: String,
    pub cost_ns: f64,
    pub egraph_cost: usize,
    pub node_count: usize,
    pub depth: usize,
    pub op_counts: OpCounts,
}

impl BenchmarkSample {
    /// Count operations from the expression string.
    fn count_ops_from_expr(expr: &str) -> OpCounts {
        let mut counts = OpCounts::default();

        // Simple pattern matching to count operations
        counts.add = expr.matches("Add(").count();
        counts.sub = expr.matches("Sub(").count();
        counts.mul = expr.matches("Mul(").count().saturating_sub(counts.add); // Exclude MulAdd
        counts.div = expr.matches("Div(").count();
        counts.neg = expr.matches("Neg(").count();
        counts.sqrt = expr.matches("Sqrt(").count();
        counts.rsqrt = expr.matches("Rsqrt(").count();
        counts.abs = expr.matches("Abs(").count();
        counts.min = expr.matches("Min(").count();
        counts.max = expr.matches("Max(").count();
        counts.mul_add = expr.matches("MulAdd(").count();

        counts
    }
}

/// Load benchmarked samples from the cache.
pub fn load_benchmark_cache(path: &Path) -> Vec<BenchmarkSample> {
    let mut samples = Vec::new();

    let Ok(file) = File::open(path) else {
        return samples;
    };

    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok) {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        // Parse JSON line
        let cost_ns = extract_json_number(&line, "cost_ns");
        if cost_ns.is_none() {
            continue; // Skip samples without benchmark data
        }

        let Some(expression) = extract_json_string(&line, "expression") else {
            continue;
        };
        let Some(name) = extract_json_string(&line, "name") else {
            continue;
        };
        let egraph_cost = extract_json_int(&line, "egraph_cost").unwrap_or(0) as usize;
        let node_count = extract_json_int(&line, "node_count").unwrap_or(0) as usize;
        let depth = extract_json_int(&line, "depth").unwrap_or(0) as usize;

        let op_counts = BenchmarkSample::count_ops_from_expr(&expression);

        samples.push(BenchmarkSample {
            name,
            expression,
            cost_ns: cost_ns.unwrap(),
            egraph_cost,
            node_count,
            depth,
            op_counts,
        });
    }

    samples
}

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

fn unescape_json(s: &str) -> String {
    s.replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
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

/// Find workspace root.
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

// ============================================================================
// Synthetic Cost Model (fallback for testing without real benchmarks)
// ============================================================================

/// Ground truth cost model (simulates what benchmarks would return).
///
/// Used when real benchmark data is not available.
fn ground_truth_cost(expr: &Expr) -> f32 {
    // Hidden ground truth: FMA is actually cheap, rsqrt is fast
    let model = TrainableCostModel {
        add: 3.0,
        sub: 3.0,
        mul: 4.0,
        div: 14.0,
        neg: 0.5,
        sqrt: 12.0,
        rsqrt: 4.0,  // Fast rsqrt!
        abs: 0.5,
        min: 3.0,
        max: 3.0,
        mul_add: 4.5,  // FMA is almost free!
        mul_rsqrt: 4.0,
    };

    // Add some noise to simulate measurement variance
    let base = model.predict(expr);
    base * (1.0 + 0.05 * pseudo_random(expr_hash(expr)))
}

fn pseudo_random(seed: u64) -> f32 {
    // Simple LCG
    let x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

// ============================================================================
// Training Loop
// ============================================================================

/// Training configuration.
pub struct TrainConfig {
    /// Learning rate for gradient descent.
    pub learning_rate: f32,
    /// Number of expression pairs per iteration.
    pub batch_size: usize,
    /// Number of training iterations.
    pub iterations: usize,
    /// Print progress every N iterations.
    pub print_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            iterations: 500,
            print_every: 50,
        }
    }
}

/// Train the cost model using pairwise ranking loss with REAL BENCHMARK DATA.
///
/// This is "The Judge" - trained on actual SIMD costs, not synthetic data.
///
/// For each pair of samples (A, B):
/// - If benchmark says A is faster, cost model should predict cost(A) < cost(B)
/// - Loss = max(0, cost(A) - cost(B) + margin) when A should be faster
pub fn train_from_benchmarks(samples: &[BenchmarkSample], config: &TrainConfig) -> TrainableCostModel {
    let mut model = TrainableCostModel::default();

    println!("=== Training The Judge (Cost Model) ===\n");
    println!("Training on {} real benchmark samples", samples.len());
    println!("Initial weights: {:?}\n", model.to_vec());

    let mut rng_state: u64 = 42;

    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_pairs = 0;

    for iter in 0..config.iterations {
        let mut batch_loss = 0.0;
        let mut batch_correct = 0;

        for _ in 0..config.batch_size {
            // Pick random pair
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx_a = (rng_state >> 33) as usize % samples.len();
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx_b = (rng_state >> 33) as usize % samples.len();

            if idx_a == idx_b {
                continue; // Skip same-sample comparison
            }

            let sample_a = &samples[idx_a];
            let sample_b = &samples[idx_b];

            // Get real SIMD costs from benchmarks
            let true_cost_a = sample_a.cost_ns as f32;
            let true_cost_b = sample_b.cost_ns as f32;

            // Get model predictions based on op counts
            let pred_cost_a = model.predict_from_counts(&sample_a.op_counts);
            let pred_cost_b = model.predict_from_counts(&sample_b.op_counts);

            // Pairwise ranking: does the model agree with benchmark?
            let true_a_better = true_cost_a < true_cost_b;
            let pred_a_better = pred_cost_a < pred_cost_b;

            if true_a_better == pred_a_better {
                batch_correct += 1;
            }

            // Compute ranking loss with margin (scaled to nanoseconds)
            let margin = 0.5;
            let loss = if true_a_better {
                (pred_cost_a - pred_cost_b + margin).max(0.0)
            } else {
                (pred_cost_b - pred_cost_a + margin).max(0.0)
            };
            batch_loss += loss;

            // Gradient update
            if loss > 0.0 {
                let mut weights = model.to_vec();
                let counts_a_vec = sample_a.op_counts.to_vec();
                let counts_b_vec = sample_b.op_counts.to_vec();

                let sign = if true_a_better { 1.0 } else { -1.0 };

                for i in 0..weights.len() {
                    let grad = (counts_a_vec[i] - counts_b_vec[i]) * sign;
                    weights[i] -= config.learning_rate * grad;
                }

                model.from_vec(&weights);
            }

            total_pairs += 1;
        }

        total_loss += batch_loss;
        total_correct += batch_correct;

        if (iter + 1) % config.print_every == 0 {
            let accuracy = total_correct as f32 / total_pairs as f32;
            let avg_loss = total_loss / total_pairs as f32;
            println!(
                "Iter {:4}: loss={:.4}, accuracy={:.1}%",
                iter + 1, avg_loss, accuracy * 100.0
            );
        }
    }

    println!("\n=== Training Complete ===\n");
    print_weights(&model);

    model
}

fn print_weights(model: &TrainableCostModel) {
    println!("Learned weights (The Judge):");
    println!("  add={:.2}, sub={:.2}, mul={:.2}, div={:.2}",
             model.add, model.sub, model.mul, model.div);
    println!("  neg={:.2}, sqrt={:.2}, rsqrt={:.2}, abs={:.2}",
             model.neg, model.sqrt, model.rsqrt, model.abs);
    println!("  min={:.2}, max={:.2}, mul_add={:.2}, mul_rsqrt={:.2}",
             model.min, model.max, model.mul_add, model.mul_rsqrt);
}

/// Train the cost model using pairwise ranking loss (synthetic fallback).
///
/// Used when benchmark data is not available.
pub fn train_cost_model(config: &TrainConfig) -> TrainableCostModel {
    let mut model = TrainableCostModel::default();
    let mut cache = CostCache::new();
    let mut generator = BwdGenerator::new(42, BwdGenConfig::default());

    println!("=== Cost Model Training Loop (SYNTHETIC) ===\n");
    println!("No benchmark data found, using synthetic ground truth.");
    println!("Initial weights: {:?}\n", model.to_vec());

    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_pairs = 0;

    for iter in 0..config.iterations {
        let mut batch_loss = 0.0;
        let mut batch_correct = 0;

        for _ in 0..config.batch_size {
            // Generate equivalent expression pair
            let pair = generator.generate();
            let expr_a = &pair.unoptimized;
            let expr_b = &pair.optimized;

            // Get ground truth costs (benchmark or cached)
            let true_cost_a = cache.get_or_compute(expr_a, ground_truth_cost);
            let true_cost_b = cache.get_or_compute(expr_b, ground_truth_cost);

            // Get model predictions
            let pred_cost_a = model.predict(expr_a);
            let pred_cost_b = model.predict(expr_b);

            // Pairwise ranking: does the model agree with ground truth?
            let true_a_better = true_cost_a < true_cost_b;
            let pred_a_better = pred_cost_a < pred_cost_b;

            if true_a_better == pred_a_better {
                batch_correct += 1;
            }

            // Compute ranking loss with margin
            let margin = 0.5;
            let loss = if true_a_better {
                // A should be predicted cheaper
                (pred_cost_a - pred_cost_b + margin).max(0.0)
            } else {
                // B should be predicted cheaper
                (pred_cost_b - pred_cost_a + margin).max(0.0)
            };
            batch_loss += loss;

            // Gradient update (simplified: adjust weights based on op counts)
            if loss > 0.0 {
                let counts_a = model.count_ops(expr_a);
                let counts_b = model.count_ops(expr_b);

                let mut weights = model.to_vec();
                let counts_a_vec = counts_a.to_vec();
                let counts_b_vec = counts_b.to_vec();

                let sign = if true_a_better { 1.0 } else { -1.0 };

                for i in 0..weights.len() {
                    // Gradient: d(cost_a - cost_b)/d(weight_i) = counts_a[i] - counts_b[i]
                    let grad = (counts_a_vec[i] - counts_b_vec[i]) * sign;
                    weights[i] -= config.learning_rate * grad;
                }

                model.from_vec(&weights);
            }

            total_pairs += 1;
        }

        total_loss += batch_loss;
        total_correct += batch_correct;

        if (iter + 1) % config.print_every == 0 {
            let accuracy = total_correct as f32 / total_pairs as f32;
            let avg_loss = total_loss / total_pairs as f32;
            println!(
                "Iter {:4}: loss={:.4}, accuracy={:.1}%, cache_hit={:.1}%",
                iter + 1, avg_loss, accuracy * 100.0, cache.hit_rate() * 100.0
            );
        }
    }

    println!("\n=== Training Complete ===\n");
    print_weights(&model);
    println!("\nCache stats: {} hits, {} misses ({:.1}% hit rate)",
             cache.hits, cache.misses, cache.hit_rate() * 100.0);

    model
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate the trained model on a test set.
pub fn evaluate_model(model: &TrainableCostModel, test_size: usize) -> f32 {
    let mut generator = BwdGenerator::new(9999, BwdGenConfig::default());
    let mut correct = 0;

    for _ in 0..test_size {
        let pair = generator.generate();
        let expr_a = &pair.unoptimized;
        let expr_b = &pair.optimized;

        let true_cost_a = ground_truth_cost(expr_a);
        let true_cost_b = ground_truth_cost(expr_b);

        let pred_cost_a = model.predict(expr_a);
        let pred_cost_b = model.predict(expr_b);

        let true_a_better = true_cost_a < true_cost_b;
        let pred_a_better = pred_cost_a < pred_cost_b;

        if true_a_better == pred_a_better {
            correct += 1;
        }
    }

    correct as f32 / test_size as f32
}

fn main() {
    let config = TrainConfig {
        learning_rate: 0.002,
        batch_size: 64,
        iterations: 300,
        print_every: 30,
    };

    // Try to load real benchmark data
    let workspace_root = find_workspace_root();
    let cache_path = workspace_root.join("pixelflow-ml/data/benchmark_cache.jsonl");

    let samples = load_benchmark_cache(&cache_path);

    if samples.len() >= 20 {
        // Train on REAL benchmark data
        println!("Found {} benchmark samples at {}", samples.len(), cache_path.display());
        println!("\n=== Training The Judge on REAL SIMD benchmarks ===\n");

        let start = Instant::now();
        let trained = train_from_benchmarks(&samples, &config);
        let elapsed = start.elapsed();

        println!("Training time: {:.2}s", elapsed.as_secs_f64());

        // Save the learned model
        let output_path = workspace_root.join("pixelflow-ml/data/learned_cost_model.toml");
        save_cost_model(&trained, &output_path);
        println!("\nSaved learned cost model to: {}", output_path.display());

        // Key insight: what did we learn?
        println!("\n=== Key Insights ===");
        println!("Default FMA cost: 10.0 (mul=5 + add=4 + 1)");
        println!("Learned:         mul_add={:.1}, mul+add={:.1}",
                 trained.mul_add, trained.mul + trained.add);

        if trained.mul_add < trained.mul + trained.add {
            println!("SUCCESS: Model learned that FMA is efficient!");
        } else {
            println!("Interesting: FMA not cheaper on this hardware?");
        }

    } else {
        // Fall back to synthetic training
        println!("No benchmark data found (need >= 20 samples)");
        println!("Run the benchmark pipeline first:");
        println!("  cargo run -p pixelflow-ml --example gen_egraph_variants --features training -- --count 100");
        println!("  cargo bench -p pixelflow-ml --bench generated_kernels");
        println!("  cargo run -p pixelflow-ml --example collect_benchmark_costs --features training");
        println!("");

        // Baseline: untrained model
        let baseline = TrainableCostModel::default();
        let baseline_acc = evaluate_model(&baseline, 500);
        println!("Baseline accuracy (default weights): {:.1}%\n", baseline_acc * 100.0);

        // Train on synthetic data
        let start = Instant::now();
        let trained = train_cost_model(&config);
        let elapsed = start.elapsed();

        // Evaluate
        let trained_acc = evaluate_model(&trained, 500);
        println!("\nTrained accuracy: {:.1}%", trained_acc * 100.0);
        println!("Improvement: {:+.1}%", (trained_acc - baseline_acc) * 100.0);
        println!("Training time: {:.2}s", elapsed.as_secs_f64());

        // Key insight: did we learn that FMA is cheap?
        println!("\n=== Key Insights (Synthetic Ground Truth) ===");
        println!("Ground truth: mul_add=4.5, mul+add=7.0 → FMA saves 2.5");
        println!("Learned:      mul_add={:.1}, mul+add={:.1} → FMA saves {:.1}",
                 trained.mul_add, trained.mul + trained.add,
                 (trained.mul + trained.add) - trained.mul_add);

        if trained.mul_add < trained.mul + trained.add - 1.0 {
            println!("\nSUCCESS: Model learned that FMA is efficient!");
        } else {
            println!("\nNEEDS MORE TRAINING: Model hasn't fully learned FMA advantage");
        }
    }
}

/// Save trained cost model to TOML file.
fn save_cost_model(model: &TrainableCostModel, path: &Path) {
    let contents = format!(
        r#"# Learned cost model weights (The Judge)
# Trained from real SIMD benchmark measurements
# Generated: {:?}

# Operation costs (relative units learned from benchmarks)
add = {:.0}
sub = {:.0}
mul = {:.0}
div = {:.0}
neg = {:.0}
sqrt = {:.0}
recip = {:.0}
rsqrt = {:.0}
abs = {:.0}
min = {:.0}
max = {:.0}
mul_add = {:.0}

# Depth penalty (compile-time optimization)
depth_threshold = 32
depth_penalty = 100
"#,
        std::time::SystemTime::now(),
        model.add, model.sub, model.mul, model.div, model.neg,
        model.sqrt, model.rsqrt, model.rsqrt, model.abs,
        model.min, model.max, model.mul_add,
    );

    if let Err(e) = fs::write(path, contents) {
        eprintln!("Warning: Failed to save cost model: {}", e);
    }
}
