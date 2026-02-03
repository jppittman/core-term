//! E-graph specific training: The Judge and The Guide.
//!
//! - **The Judge**: Trained on SIMD benchmark data, predicts actual runtime cost
//! - **The Guide**: Curriculum-trained, predicts optimization potential

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;

use pixelflow_search::egraph::{
    ExprTree, Leaf,
    BestFirstPlanner, BestFirstConfig, BestFirstContext, CostModel,
};
use crate::nnue::{
    Nnue, NnueConfig, HalfEPFeature, OpKind, DenseFeatures,
};

use super::features::{extract_tree_features, op_counts_to_dense};
use super::backprop::{forward_with_state, forward_with_state_hybrid, backward, backward_hybrid, ForwardState, HybridForwardState};

// ============================================================================
// Common Utilities
// ============================================================================

/// Simple LCG random number generator.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }

    pub fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        let len = range.end - range.start;
        range.start + (self.next_u64() as usize % len)
    }

    pub fn gen_bool(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }

    pub fn gen_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// Generate random expression trees for training.
pub struct ExprGenerator {
    rng: Rng,
    max_depth: usize,
    num_vars: usize,
}

impl ExprGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            max_depth: 5,
            num_vars: 4,
        }
    }

    pub fn generate_small(&mut self) -> ExprTree {
        self.max_depth = 3 + self.rng.gen_range(0..2);
        self.generate_inner(0)
    }

    pub fn generate_large(&mut self) -> ExprTree {
        self.max_depth = 6 + self.rng.gen_range(0..3);
        self.generate_inner(0)
    }

    fn generate_inner(&mut self, depth: usize) -> ExprTree {
        let leaf_prob = 0.2 + (depth as f64 * 0.15);

        if depth >= self.max_depth || self.rng.gen_f64() < leaf_prob {
            if self.rng.gen_bool() {
                ExprTree::Leaf(Leaf::Var(self.rng.gen_range(0..self.num_vars) as u8))
            } else {
                let constants = [0.0, 1.0, -1.0, 2.0, 0.5];
                ExprTree::Leaf(Leaf::Const(constants[self.rng.gen_range(0..constants.len())]))
            }
        } else {
            // Generate ops that ExprTree supports (10 ops available)
            // Missing from ExprTree: rsqrt helper, mul_rsqrt
            // These can still appear via e-graph rewriting
            let op_type = self.rng.gen_range(0..10);
            match op_type {
                // Binary ops (6)
                0 => ExprTree::add(self.generate_inner(depth + 1), self.generate_inner(depth + 1)),
                1 => ExprTree::sub(self.generate_inner(depth + 1), self.generate_inner(depth + 1)),
                2 => ExprTree::mul(self.generate_inner(depth + 1), self.generate_inner(depth + 1)),
                3 => ExprTree::div(self.generate_inner(depth + 1), self.generate_inner(depth + 1)),
                4 => ExprTree::min(self.generate_inner(depth + 1), self.generate_inner(depth + 1)),
                5 => ExprTree::max(self.generate_inner(depth + 1), self.generate_inner(depth + 1)),
                // Unary ops (3)
                6 => ExprTree::neg(self.generate_inner(depth + 1)),
                7 => ExprTree::sqrt(self.generate_inner(depth + 1)),
                8 => ExprTree::abs(self.generate_inner(depth + 1)),
                // Fused op (1)
                _ => ExprTree::mul_add(
                    self.generate_inner(depth + 1),
                    self.generate_inner(depth + 1),
                    self.generate_inner(depth + 1),
                ),
            }
        }
    }
}

// ============================================================================
// The Judge: Benchmark-Based Training
// ============================================================================

/// A benchmark sample with expression and measured cost.
#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkSample {
    pub name: String,
    pub expression: String,
    pub cost_ns: Option<f64>,
    pub egraph_cost: Option<usize>,
    pub node_count: Option<usize>,
    pub depth: Option<usize>,
    /// Precomputed HalfEP feature indices (packed as u32)
    #[serde(default)]
    pub features: Vec<u32>,
    #[serde(default)]
    pub op_counts: HashMap<String, usize>,
}

/// Load benchmarked samples from JSONL cache.
pub fn load_benchmark_cache(path: &Path) -> Vec<BenchmarkSample> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };

    BufReader::new(file)
        .lines()
        .filter_map(Result::ok)
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<BenchmarkSample>(&line).ok())
        .filter(|s| s.cost_ns.is_some())
        .collect()
}

/// Configuration for benchmark-based training.
#[derive(Clone)]
pub struct BenchmarkConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            seed: 42,
        }
    }
}

/// A prepared training sample for The Judge.
#[derive(Clone)]
pub struct JudgeTrainingSample {
    pub features: Vec<HalfEPFeature>,
    pub dense: DenseFeatures,
    pub cost_ns: f64,
}

/// Prepare benchmark samples for training.
pub fn prepare_judge_samples(samples: &[BenchmarkSample]) -> Vec<JudgeTrainingSample> {
    samples.iter()
        .filter_map(|s| {
            let cost_ns = s.cost_ns?;

            // Use precomputed features - they have proper tree structure
            assert!(
                !s.features.is_empty(),
                "Sample '{}' missing precomputed features - regenerate with collect_benchmark_costs",
                s.name
            );

            let features: Vec<HalfEPFeature> = s.features.iter()
                .map(|&idx| HalfEPFeature::from_index(idx as usize))
                .collect();

            // Build dense features from available metadata
            let mut dense = DenseFeatures::default();
            dense.values[DenseFeatures::NODE_COUNT] = s.node_count.unwrap_or(0) as i32;
            dense.values[DenseFeatures::DEPTH] = s.depth.unwrap_or(0) as i32;
            // Use egraph_cost as a rough proxy for critical_path
            dense.values[DenseFeatures::CRITICAL_PATH] = s.egraph_cost.unwrap_or(0) as i32;

            Some(JudgeTrainingSample { features, dense, cost_ns })
        })
        .collect()
}

/// Train NNUE on benchmark samples (The Judge).
pub fn train_judge_batch(
    nnue: &mut Nnue,
    batch: &[JudgeTrainingSample],
    lr: f32,
    cost_scale: f64,
) -> f32 {
    let mut total_loss = 0.0;

    for sample in batch {
        let (pred, state) = forward_with_state_hybrid(nnue, &sample.features, &sample.dense);
        let target = (sample.cost_ns / cost_scale) as f32;

        let error = pred - target;
        total_loss += error * error;

        let d_loss = 2.0 * error;
        backward_hybrid(nnue, &state, d_loss, lr);
    }

    total_loss / batch.len().max(1) as f32
}

/// Run The Judge training pipeline.
pub fn run_judge_training(config: BenchmarkConfig, cache_path: &Path) {
    println!("=== The Judge: Benchmark-Based NNUE Training ===");
    println!();

    let samples = load_benchmark_cache(cache_path);
    if samples.is_empty() {
        println!("ERROR: No benchmarked samples found!");
        println!("Run benchmarks first to populate the cache with cost_ns values.");
        return;
    }

    println!("Loaded {} samples with benchmark data", samples.len());

    let all_samples = prepare_judge_samples(&samples);

    // Group samples by expression family (eg0000, eg0001, etc.)
    // Variants of the same expression MUST be in the same split to avoid leakage
    let mut families: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (i, sample) in samples.iter().enumerate() {
        // Extract family from name like "eg0000v0" -> "eg0000"
        let family = sample.name.chars()
            .take_while(|c| !c.is_ascii_digit() || sample.name.matches(|x: char| x.is_ascii_digit()).count() > 4)
            .take_while(|&c| c != 'v')
            .collect::<String>();
        let family = if family.is_empty() { sample.name.clone() } else { family };
        families.entry(family).or_default().push(i);
    }

    // Shuffle families, then split 80/20
    let mut family_keys: Vec<_> = families.keys().cloned().collect();
    let mut rng = Rng::new(config.seed + 999);
    for i in (1..family_keys.len()).rev() {
        let j = rng.gen_range(0..i + 1);
        family_keys.swap(i, j);
    }

    let split_idx = (family_keys.len() * 80) / 100;
    let train_families: Vec<_> = family_keys[..split_idx].to_vec();
    let test_families: Vec<_> = family_keys[split_idx..].to_vec();

    // Collect samples by family assignment
    let mut train_indices: Vec<usize> = train_families.iter()
        .flat_map(|f| families.get(f).unwrap().iter().copied())
        .collect();
    let mut test_indices: Vec<usize> = test_families.iter()
        .flat_map(|f| families.get(f).unwrap().iter().copied())
        .collect();

    // Shuffle within each set
    for i in (1..train_indices.len()).rev() {
        let j = rng.gen_range(0..i + 1);
        train_indices.swap(i, j);
    }

    let train_samples: Vec<_> = train_indices.iter()
        .filter_map(|&i| all_samples.get(i).cloned())
        .collect();
    let test_samples: Vec<_> = test_indices.iter()
        .filter_map(|&i| all_samples.get(i).cloned())
        .collect();

    println!("Split {} families: {} train ({} samples) + {} test ({} samples)",
             families.len(), train_families.len(), train_samples.len(),
             test_families.len(), test_samples.len());

    let mut costs: Vec<f64> = train_samples.iter().map(|s| s.cost_ns).collect();
    costs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cost_scale = costs.get(costs.len() / 2).copied().unwrap_or(1000.0);
    println!("Cost scale (median): {:.2} ns", cost_scale);
    println!();

    // Use random initialization - critical for gradient flow!
    let mut nnue = Nnue::new_random(NnueConfig::default(), config.seed);
    let mut rng = Rng::new(config.seed);

    // Debug: check initial state BEFORE any training
    {
        let b1_mean: f64 = nnue.b1.iter().map(|&b| b as f64).sum::<f64>() / nnue.b1.len() as f64;
        let b1_min = nnue.b1.iter().min().unwrap();
        let b1_max = nnue.b1.iter().max().unwrap();

        // Check initial prediction on first sample
        let (pred, state) = super::backprop::forward_with_state_hybrid(
            &nnue, &train_samples[0].features, &train_samples[0].dense);
        let target = (train_samples[0].cost_ns / cost_scale) as f32;
        let l1_active = state.l1_post.iter().filter(|&&x| x > 0.0).count();

        println!("BEFORE TRAINING:");
        println!("  B1: mean={:.0} min={} max={}", b1_mean, b1_min, b1_max);
        println!("  First sample: pred={:.4} target={:.4} error={:.4}", pred, target, pred - target);
        println!("  L1 active neurons: {}/256", l1_active);
        println!();
    }

    let start = Instant::now();

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        for chunk in indices.chunks(config.batch_size) {
            let batch: Vec<_> = chunk.iter().map(|&i| train_samples[i].clone()).collect();
            epoch_loss += train_judge_batch(&mut nnue, &batch, config.learning_rate, cost_scale);
            batches += 1;
        }

        let avg_loss = epoch_loss / batches.max(1) as f32;

        // Debug: track neuron health
        if epoch == 0 || epoch == config.epochs - 1 {
            let b1_mean: f64 = nnue.b1.iter().map(|&b| b as f64).sum::<f64>() / nnue.b1.len() as f64;
            let b1_min = nnue.b1.iter().min().unwrap();
            let b1_max = nnue.b1.iter().max().unwrap();
            println!("Epoch {:3}: loss = {:.6}  B1 mean={:.0} min={} max={}",
                     epoch, avg_loss, b1_mean, b1_min, b1_max);
        } else if epoch % 10 == 0 {
            println!("Epoch {:3}: loss = {:.6}", epoch, avg_loss);
        }
    }

    println!();
    println!("Training completed in {:.2}s", start.elapsed().as_secs_f64());

    // Save The Judge weights
    let judge_path = cache_path.parent()
        .unwrap_or(Path::new("."))
        .join("nnue_judge_weights.bin");
    println!("Saving Judge weights to: {}", judge_path.display());
    if let Err(e) = save_nnue_weights(&nnue, &judge_path) {
        println!("Warning: Failed to save weights: {}", e);
    }
}

// ============================================================================
// The Guide: Curriculum Training
// ============================================================================

/// A training sample for curriculum learning.
#[derive(Clone)]
pub struct GuideTrainingSample {
    pub features: Vec<HalfEPFeature>,
    pub final_cost: usize,
    pub initial_cost: usize,
}

/// Experience replay buffer.
pub struct ReplayBuffer {
    samples: Vec<GuideTrainingSample>,
    max_size: usize,
    rng: Rng,
}

impl ReplayBuffer {
    pub fn new(max_size: usize, seed: u64) -> Self {
        Self {
            samples: Vec::new(),
            max_size,
            rng: Rng::new(seed),
        }
    }

    pub fn add(&mut self, sample: GuideTrainingSample) {
        if self.samples.len() >= self.max_size {
            let idx = self.rng.gen_range(0..self.samples.len());
            self.samples.swap_remove(idx);
        }
        self.samples.push(sample);
    }

    pub fn sample_batch(&mut self, batch_size: usize) -> Vec<GuideTrainingSample> {
        let n = self.samples.len().min(batch_size);
        let mut indices: Vec<usize> = (0..self.samples.len()).collect();

        for i in 0..n {
            let j = self.rng.gen_range(i..self.samples.len());
            indices.swap(i, j);
        }

        indices[..n].iter().map(|&i| self.samples[i].clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Configuration for curriculum training.
#[derive(Clone)]
pub struct CurriculumConfig {
    pub kindergarten_samples: usize,
    pub kindergarten_epochs: usize,
    pub university_samples: usize,
    pub university_epochs: usize,
    pub learning_rate: f32,
    pub epsilon: f32,
    pub max_expansions: usize,
    pub batch_size: usize,
    pub replay_buffer_size: usize,
    pub seed: u64,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            kindergarten_samples: 200,
            kindergarten_epochs: 10,
            university_samples: 100,
            university_epochs: 20,
            learning_rate: 0.001,
            epsilon: 0.1,
            max_expansions: 500,
            batch_size: 64,
            replay_buffer_size: 5000,
            seed: 42,
        }
    }
}

/// Train NNUE on curriculum samples.
pub fn train_guide_batch(nnue: &mut Nnue, batch: &[GuideTrainingSample], lr: f32) -> f32 {
    let mut total_loss = 0.0;

    for sample in batch {
        let (pred, state) = forward_with_state(nnue, &sample.features);

        let target = if sample.initial_cost > 0 {
            (sample.initial_cost as f32 - sample.final_cost as f32) / sample.initial_cost as f32
        } else {
            0.0
        };

        let error = pred - target;
        total_loss += error * error;

        let d_loss = 2.0 * error;
        backward(nnue, &state, d_loss, lr);
    }

    total_loss / batch.len() as f32
}

// ============================================================================
// Weight Persistence
// ============================================================================

/// Save NNUE weights to binary file.
pub fn save_nnue_weights(nnue: &Nnue, path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    file.write_all(b"NNUE")?;

    let config = &nnue.config;
    file.write_all(&(config.l1_size as u32).to_le_bytes())?;
    file.write_all(&(config.l2_size as u32).to_le_bytes())?;
    file.write_all(&(config.l3_size as u32).to_le_bytes())?;
    file.write_all(&(config.dense_size as u32).to_le_bytes())?;

    for w in &nnue.w1 {
        file.write_all(&w.to_le_bytes())?;
    }

    for b in &nnue.b1 {
        file.write_all(&b.to_le_bytes())?;
    }

    for w in &nnue.w2 {
        file.write_all(&[*w as u8])?;
    }

    for b in &nnue.b2 {
        file.write_all(&b.to_le_bytes())?;
    }

    for w in &nnue.w3 {
        file.write_all(&[*w as u8])?;
    }

    for b in &nnue.b3 {
        file.write_all(&b.to_le_bytes())?;
    }

    for w in &nnue.w_out {
        file.write_all(&[*w as u8])?;
    }

    file.write_all(&nnue.b_out.to_le_bytes())?;

    for w in &nnue.w_dense {
        file.write_all(&w.to_le_bytes())?;
    }

    for b in &nnue.b_dense {
        file.write_all(&b.to_le_bytes())?;
    }

    Ok(())
}

/// Load NNUE weights from binary file.
pub fn load_nnue_weights(path: &Path) -> std::io::Result<Nnue> {
    use std::io::Read;

    let mut file = File::open(path)?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"NNUE" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid NNUE file magic",
        ));
    }

    let mut buf4 = [0u8; 4];

    file.read_exact(&mut buf4)?;
    let l1_size = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf4)?;
    let l2_size = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf4)?;
    let l3_size = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf4)?;
    let dense_size = u32::from_le_bytes(buf4) as usize;

    let config = NnueConfig {
        l1_size,
        l2_size,
        l3_size,
        dense_size,
    };

    let mut nnue = Nnue::new(config);

    let mut buf2 = [0u8; 2];
    for w in &mut nnue.w1 {
        file.read_exact(&mut buf2)?;
        *w = i16::from_le_bytes(buf2);
    }

    for b in &mut nnue.b1 {
        file.read_exact(&mut buf4)?;
        *b = i32::from_le_bytes(buf4);
    }

    let mut buf1 = [0u8; 1];
    for w in &mut nnue.w2 {
        file.read_exact(&mut buf1)?;
        *w = buf1[0] as i8;
    }

    for b in &mut nnue.b2 {
        file.read_exact(&mut buf4)?;
        *b = i32::from_le_bytes(buf4);
    }

    for w in &mut nnue.w3 {
        file.read_exact(&mut buf1)?;
        *w = buf1[0] as i8;
    }

    for b in &mut nnue.b3 {
        file.read_exact(&mut buf4)?;
        *b = i32::from_le_bytes(buf4);
    }

    for w in &mut nnue.w_out {
        file.read_exact(&mut buf1)?;
        *w = buf1[0] as i8;
    }

    file.read_exact(&mut buf4)?;
    nnue.b_out = i32::from_le_bytes(buf4);

    for w in &mut nnue.w_dense {
        file.read_exact(&mut buf2)?;
        *w = i16::from_le_bytes(buf2);
    }

    for b in &mut nnue.b_dense {
        file.read_exact(&mut buf4)?;
        *b = i32::from_le_bytes(buf4);
    }

    Ok(nnue)
}
