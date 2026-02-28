//! # Unified NNUE Training CLI
//!
//! Entry point for NNUE training. Delegates to `pixelflow_ml::training::egraph`.
//!
//! ## Usage
//!
//! ```bash
//! # Train on SIMD benchmarks (The Judge)
//! cargo run -p pixelflow-ml --example guided_training --features egraph-training -- --benchmark
//!
//! # Curriculum training (The Guide) - quick test
//! cargo run -p pixelflow-ml --example guided_training --features egraph-training -- --quick
//!
//! # Full curriculum training
//! cargo run -p pixelflow-ml --example guided_training --features egraph-training
//!
//! # Evaluate trained models
//! cargo run -p pixelflow-ml --example guided_training --features egraph-training -- --eval
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use pixelflow_ml::training::egraph::{
    BenchmarkConfig, CurriculumConfig, ExprGenerator, GuideTrainingSample, ReplayBuffer,
    load_benchmark_cache, load_nnue_weights, prepare_judge_samples, run_judge_training,
    save_nnue_weights, train_guide_batch,
};

use pixelflow_ml::training::backprop::{forward_with_state, forward_with_state_hybrid};
use pixelflow_ml::training::features::extract_search_features;
use pixelflow_nnue::{Nnue, NnueConfig};
use pixelflow_search::egraph::{BestFirstConfig, BestFirstContext, BestFirstPlanner, CostModel};

// ============================================================================
// Evaluation Metrics
// ============================================================================

/// Compute Spearman rank correlation coefficient.
fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let rank = |vals: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<_> = vals.iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let mut ranks = vec![0.0; n];
        for (rank, (idx, _)) in indexed.into_iter().enumerate() {
            ranks[idx] = rank as f64 + 1.0;
        }
        ranks
    };

    let rx = rank(x);
    let ry = rank(y);

    // Pearson correlation of ranks
    let mean_x: f64 = rx.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = ry.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = rx[i] - mean_x;
        let dy = ry[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute R² (coefficient of determination).
fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len());
    let n = actual.len();
    if n == 0 {
        return 0.0;
    }

    let mean: f64 = actual.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = actual.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(y, p)| (y - p).powi(2))
        .sum();

    if ss_tot == 0.0 {
        return 1.0;
    }
    1.0 - (ss_res / ss_tot)
}

/// Compute mean absolute error.
fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len());
    if actual.is_empty() {
        return 0.0;
    }
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len() as f64
}

/// Compute mean absolute percentage error.
fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len());
    let valid: Vec<_> = actual
        .iter()
        .zip(predicted)
        .filter(|(a, _)| **a != 0.0)
        .map(|(a, p)| ((a - p) / a).abs())
        .collect();
    if valid.is_empty() {
        return 0.0;
    }
    valid.iter().sum::<f64>() / valid.len() as f64 * 100.0
}

/// Search result for comparison.
struct SearchResult {
    final_cost: usize,
    initial_cost: usize,
    expansions: usize,
}

impl SearchResult {
    fn improvement(&self) -> f64 {
        if self.initial_cost == 0 {
            0.0
        } else {
            1.0 - (self.final_cost as f64 / self.initial_cost as f64)
        }
    }

    fn regret(&self, optimal: usize) -> f64 {
        if optimal == 0 {
            0.0
        } else {
            (self.final_cost as f64 - optimal as f64) / optimal as f64
        }
    }
}

fn find_workspace_root() -> PathBuf {
    let mut current = std::env::current_dir().expect("Failed to get current directory");

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let benchmark_mode = args.iter().any(|a| a == "--benchmark");
    let quick_mode = args.iter().any(|a| a == "--quick");
    let kindergarten_only = args.iter().any(|a| a == "--kindergarten-only");
    let eval_mode = args.iter().any(|a| a == "--eval");

    let workspace = find_workspace_root();

    if eval_mode {
        run_evaluation(&workspace);
        return;
    }

    if benchmark_mode {
        let config = BenchmarkConfig::default();
        let cache_path = workspace.join("pixelflow-ml/data/benchmark_cache.jsonl");
        run_judge_training(config, &cache_path);
        return;
    }

    // Curriculum training (The Guide)
    let config = if quick_mode {
        CurriculumConfig {
            kindergarten_samples: 50,
            kindergarten_epochs: 3,
            university_samples: 20,
            university_epochs: 5,
            max_expansions: 100,
            ..CurriculumConfig::default()
        }
    } else {
        CurriculumConfig::default()
    };

    println!("=== The Guide: Curriculum Training ===");
    if quick_mode {
        println!("(quick mode)");
    }
    if kindergarten_only {
        println!("(kindergarten only)");
    }
    println!();

    // Load The Judge for cost evaluation (trained on benchmarks)
    let judge_path = workspace.join("pixelflow-ml/data/nnue_judge_weights.bin");
    let judge = match load_nnue_weights(&judge_path) {
        Ok(j) => {
            println!("Loaded The Judge from: {}", judge_path.display());
            Some(j)
        }
        Err(e) => {
            println!("Warning: Could not load Judge weights: {}", e);
            println!("Falling back to e-graph cost model for evaluation");
            None
        }
    };

    let cost_model = CostModel::load_or_default();
    println!(
        "Cost model: add={}, mul={}, div={}, sqrt={}",
        cost_model.cost_by_name("add"),
        cost_model.cost_by_name("mul"),
        cost_model.cost_by_name("div"),
        cost_model.cost_by_name("sqrt")
    );
    println!();

    // The Guide (being trained) - predicts optimization potential
    // Use random initialization - zero weights have zero gradients and never learn!
    let mut guide = Nnue::new_random(NnueConfig::default(), config.seed + 100);
    let mut expr_gen = ExprGenerator::new(config.seed);
    let mut replay = ReplayBuffer::new(config.replay_buffer_size, config.seed + 1);

    let start = Instant::now();

    // Phase 1: Kindergarten
    println!("=== Phase 1: Kindergarten (Saturation) ===");
    for epoch in 0..config.kindergarten_epochs {
        let mut epoch_samples = 0;
        let mut total_improvement = 0.0;

        for _ in 0..config.kindergarten_samples {
            let tree = expr_gen.generate_small();

            let bf_config = BestFirstConfig::default().with_max_expansions(100);
            let mut planner = BestFirstPlanner::from_tree(&tree, bf_config);

            // Use The Judge for cost evaluation if available, else use Guide for exploration
            let result = planner.run(|ctx: BestFirstContext<'_>| {
                let features = extract_search_features(&ctx);
                let evaluator = judge.as_ref().unwrap_or(&guide);
                let (pred, _) =
                    pixelflow_ml::training::backprop::forward_with_state(evaluator, &features);
                (pred * 1000.0) as i64
            });

            // result.used_saturation tells us if we got ground truth
            if result.initial_cost > 0 {
                let improvement = 1.0 - (result.best_cost as f64 / result.initial_cost as f64);
                total_improvement += improvement;
            }

            let features =
                pixelflow_ml::training::features::extract_tree_features(&result.best_tree);
            replay.add(GuideTrainingSample {
                features,
                final_cost: result.best_cost,
                initial_cost: result.initial_cost,
            });

            epoch_samples += 1;
        }

        // Train on replay buffer
        let batch = replay.sample_batch(config.batch_size);
        if !batch.is_empty() {
            let loss = train_guide_batch(&mut guide, &batch, config.learning_rate);
            let avg_imp = if epoch_samples > 0 {
                total_improvement / epoch_samples as f64
            } else {
                0.0
            };
            println!(
                "Epoch {}: samples={}, avg_improvement={:.2}%, loss={:.6}",
                epoch,
                epoch_samples,
                avg_imp * 100.0,
                loss
            );
        }
    }

    if kindergarten_only {
        println!();
        println!(
            "Kindergarten complete in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        return;
    }

    // Phase 2: University - same as Kindergarten but larger kernels + more expansions
    println!();
    println!("=== Phase 2: University (Guided Search) ===");
    for epoch in 0..config.university_epochs {
        let mut epoch_samples = 0;
        let mut total_improvement = 0.0;

        for _ in 0..config.university_samples {
            let tree = expr_gen.generate_large();

            let bf_config = BestFirstConfig::default()
                .with_max_expansions(config.max_expansions)
                .with_epsilon(config.epsilon as f64); // ε-greedy exploration

            let mut planner = BestFirstPlanner::from_tree(&tree, bf_config);

            // Same evaluator as Kindergarten - uses search context features
            let result = planner.run(|ctx: BestFirstContext<'_>| {
                let features = extract_search_features(&ctx);
                let evaluator = judge.as_ref().unwrap_or(&guide);
                let (pred, _) =
                    pixelflow_ml::training::backprop::forward_with_state(evaluator, &features);
                (pred * 1000.0) as i64
            });

            if result.initial_cost > 0 {
                let improvement = 1.0 - (result.best_cost as f64 / result.initial_cost as f64);
                total_improvement += improvement;
            }

            let features =
                pixelflow_ml::training::features::extract_tree_features(&result.best_tree);
            replay.add(GuideTrainingSample {
                features,
                final_cost: result.best_cost,
                initial_cost: result.initial_cost,
            });

            epoch_samples += 1;
        }

        // Train on replay buffer
        let batch = replay.sample_batch(config.batch_size);
        if !batch.is_empty() {
            let loss = train_guide_batch(&mut guide, &batch, config.learning_rate);
            let avg_imp = if epoch_samples > 0 {
                total_improvement / epoch_samples as f64
            } else {
                0.0
            };
            println!(
                "Epoch {}: samples={}, avg_improvement={:.2}%, loss={:.6}",
                epoch,
                epoch_samples,
                avg_imp * 100.0,
                loss
            );
        }
    }

    println!();
    println!("Training complete in {:.2}s", start.elapsed().as_secs_f64());

    // Save weights
    let weights_path = workspace.join("pixelflow-ml/data/nnue_guide_weights.bin");
    println!("Saving weights to: {}", weights_path.display());
    if let Err(e) = save_nnue_weights(&guide, &weights_path) {
        println!("Warning: Failed to save weights: {}", e);
    }
}

// ============================================================================
// Evaluation Mode
// ============================================================================

fn run_evaluation(workspace: &PathBuf) {
    println!("=== Model Evaluation ===");
    println!();

    // Load models
    let judge_path = workspace.join("pixelflow-ml/data/nnue_judge_weights.bin");
    let guide_path = workspace.join("pixelflow-ml/data/nnue_guide_weights.bin");

    let judge = load_nnue_weights(&judge_path).ok();
    let guide = load_nnue_weights(&guide_path).ok();

    if judge.is_none() && guide.is_none() {
        println!("ERROR: No trained models found. Run training first:");
        println!("  --benchmark  Train The Judge on SIMD benchmarks");
        println!("  --quick      Train The Guide (curriculum)");
        return;
    }

    // Evaluate The Judge (if available)
    if let Some(ref j) = judge {
        evaluate_judge(j, workspace);
    } else {
        println!("The Judge: NOT TRAINED (run --benchmark first)");
    }

    println!();

    // Evaluate The Guide (if available)
    if let Some(ref g) = guide {
        evaluate_guide(g, judge.as_ref());
    } else {
        println!("The Guide: NOT TRAINED (run --quick first)");
    }
}

/// Trace through forward pass to diagnose constant predictions.
fn trace_forward_pass(
    judge: &Nnue,
    sample_idx: usize,
    sample: &pixelflow_ml::training::egraph::JudgeTrainingSample,
) {
    let l1_size = judge.config.l1_size;
    let dense_size = judge.config.dense_size;

    println!("  --- Sample {} trace ---", sample_idx);
    println!("    Features: {} active", sample.features.len());

    // Sparse L1: accumulate bias + feature contributions
    let mut l1_pre: Vec<i32> = judge.b1.clone();
    for feature in &sample.features {
        let idx = feature.to_index();
        let offset = idx * l1_size;
        for i in 0..l1_size {
            l1_pre[i] += judge.w1[offset + i] as i32;
        }
    }

    // Count active L1 neurons (post clipped-ReLU)
    let l1_post: Vec<f32> = l1_pre
        .iter()
        .map(|&x| (x >> 6).clamp(0, 127) as f32)
        .collect();
    let l1_active = l1_post.iter().filter(|&&x| x > 0.0).count();
    let l1_sum: f32 = l1_post.iter().sum();
    let l1_mean = l1_sum / l1_size as f32;

    println!(
        "    L1 pre-act: min={}, max={}, mean={:.1}",
        l1_pre.iter().min().unwrap(),
        l1_pre.iter().max().unwrap(),
        l1_pre.iter().map(|&x| x as f64).sum::<f64>() / l1_size as f64
    );
    println!(
        "    L1 post-act: {}/{} active, mean={:.2}",
        l1_active, l1_size, l1_mean
    );

    // Dense branch
    let mut dense_pre: Vec<i32> = judge.b_dense.clone();
    for i in 0..pixelflow_nnue::DenseFeatures::COUNT {
        let val = sample.dense.values[i];
        if val != 0 {
            for j in 0..dense_size {
                dense_pre[j] += val * (judge.w_dense[i * dense_size + j] as i32);
            }
        }
    }
    let dense_post: Vec<f32> = dense_pre
        .iter()
        .map(|&x| (x >> 6).clamp(0, 127) as f32)
        .collect();
    let dense_active = dense_post.iter().filter(|&&x| x > 0.0).count();
    let dense_mean: f32 = dense_post.iter().sum::<f32>() / dense_size as f32;

    println!(
        "    Dense pre-act: min={}, max={}",
        dense_pre.iter().min().unwrap(),
        dense_pre.iter().max().unwrap()
    );
    println!(
        "    Dense post-act: {}/{} active, mean={:.2}",
        dense_active, dense_size, dense_mean
    );

    // Final output
    let (pred, _) = forward_with_state_hybrid(judge, &sample.features, &sample.dense);
    println!("    Final prediction: {:.4}", pred);
    println!();
}

/// Compute baseline: linear sum of op costs (no learning, just heuristics)
fn compute_linear_baseline(samples: &[pixelflow_ml::training::egraph::JudgeTrainingSample]) -> f64 {


    // Simple cycle costs per operation (typical modern CPU)
    const COSTS: [i32; 12] = [
        4,  // ADD
        4,  // SUB
        5,  // MUL
        15, // DIV
        1,  // NEG
        15, // SQRT
        5,  // RSQRT (fast approximation)
        1,  // ABS
        4,  // MIN
        4,  // MAX
        5,  // FMA
        6,  // MUL_RSQRT
    ];

    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    for sample in samples {
        // Sum up operation costs from dense features (indices 0-11 are op counts)
        let mut cost_sum = 0i32;
        for (i, &cost) in COSTS.iter().enumerate() {
            cost_sum += sample.dense.values[i] * cost;
        }
        predictions.push(cost_sum as f64);
        actuals.push(sample.cost_ns);
    }

    spearman_correlation(&actuals, &predictions)
}

/// Evaluate The Judge on benchmark data.
fn evaluate_judge(judge: &Nnue, workspace: &PathBuf) {
    println!("=== The Judge: Cost Prediction Quality ===");

    let cache_path = workspace.join("pixelflow-ml/data/benchmark_cache.jsonl");
    let samples = load_benchmark_cache(&cache_path);

    if samples.is_empty() {
        println!("  No benchmark data found at {}", cache_path.display());
        return;
    }

    let all_samples = prepare_judge_samples(&samples);

    // Family-aware split: variants of same expression stay together
    // Uses same logic as training to ensure consistency
    use std::collections::HashMap;
    let mut families: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, s) in samples.iter().enumerate() {
        let family = s.name.split('v').next().unwrap_or(&s.name).to_string();
        families.entry(family).or_default().push(i);
    }

    // Deterministic shuffle with same seed as training
    let mut family_keys: Vec<_> = families.keys().cloned().collect();
    family_keys.sort(); // Deterministic order first
    let mut rng_state = 42u64 + 999; // Same seed as training
    for i in (1..family_keys.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        family_keys.swap(i, j);
    }

    let split_idx = (family_keys.len() * 80) / 100;
    let train_samples: Vec<_> = family_keys[..split_idx]
        .iter()
        .flat_map(|f| families.get(f).unwrap().iter())
        .filter_map(|&i| all_samples.get(i).cloned())
        .collect();
    let test_samples: Vec<_> = family_keys[split_idx..]
        .iter()
        .flat_map(|f| families.get(f).unwrap().iter())
        .filter_map(|&i| all_samples.get(i).cloned())
        .collect();

    println!(
        "  {} families: {} train ({} samples), {} test ({} samples)",
        families.len(),
        split_idx,
        train_samples.len(),
        family_keys.len() - split_idx,
        test_samples.len()
    );

    // Compute linear baseline on BOTH sets
    let baseline_train = compute_linear_baseline(&train_samples);
    let baseline_test = compute_linear_baseline(&test_samples);
    println!(
        "  Linear baseline: train ρ={:.4}, test ρ={:.4}",
        baseline_train, baseline_test
    );

    // Skip trace for now - focus on metrics
    println!();

    // Helper to compute metrics on a sample set
    let compute_metrics = |samples: &[pixelflow_ml::training::egraph::JudgeTrainingSample],
                           label: &str| {
        let mut actuals = Vec::new();
        let mut predictions = Vec::new();

        for sample in samples {
            let (pred, _) = forward_with_state_hybrid(judge, &sample.features, &sample.dense);
            predictions.push(pred as f64);
            actuals.push(sample.cost_ns);
        }

        let spearman = spearman_correlation(&actuals, &predictions);
        println!("  {} Spearman ρ = {:.4}", label, spearman);
        (spearman, actuals, predictions)
    };

    // CRITICAL: Evaluate on BOTH train and test to detect overfitting
    let (train_spearman, _train_actuals, _train_preds) = compute_metrics(&train_samples, "TRAIN");
    let (test_spearman, test_actuals, test_preds) = compute_metrics(&test_samples, "TEST ");

    // Overfitting detection
    let overfit_gap = train_spearman - test_spearman;
    println!();
    if overfit_gap > 0.1 {
        println!(
            "  [WARNING] Overfitting detected! Train-Test gap = {:.4}",
            overfit_gap
        );
    } else {
        println!(
            "  [OK] Train-Test gap = {:.4} (no significant overfitting)",
            overfit_gap
        );
    }

    // Error analysis: find worst predictions on test set
    println!();
    println!("  === Error Analysis (TEST set) ===");
    let mut errors: Vec<(usize, f64, f64, f64)> = test_actuals
        .iter()
        .zip(test_preds.iter())
        .enumerate()
        .map(|(i, (&actual, &pred))| {
            let error = (pred - actual).abs() / actual.max(0.01);
            (i, actual, pred, error)
        })
        .collect();
    errors.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("  Top 5 worst predictions (by relative error):");
    for (i, actual, pred, error) in errors.iter().take(5) {
        println!(
            "    Sample {}: actual={:.3}ns pred={:.3} error={:.1}%",
            split_idx + i,
            actual,
            pred,
            error * 100.0
        );
    }
    println!();

    // Interpretation (based on TEST performance, not train)
    if test_spearman > 0.8 {
        println!("  [GOOD] Strong TEST rank correlation - Judge generalizes well");
    } else if test_spearman > 0.5 {
        println!("  [OK] Moderate TEST correlation - more data may help");
    } else {
        println!("  [WEAK] Low TEST correlation - likely overfitting or need more data");
    }
}

/// Evaluate The Guide by comparing search strategies.
fn evaluate_guide(guide: &Nnue, judge: Option<&Nnue>) {
    println!("=== The Guide: Search Heuristic Quality ===");
    println!();

    let mut expr_gen = ExprGenerator::new(12345); // Fixed seed for reproducibility
    let n_small = 50; // Small kernels (can compute optimal via saturation)
    let n_large = 30; // Large kernels (compare strategies)

    // Evaluate on small kernels (where we know optimal)
    println!(
        "  Small Kernels (n={}): Comparing to saturation optimal",
        n_small
    );

    let mut nnue_regrets = Vec::new();
    let mut random_regrets = Vec::new();
    let mut greedy_regrets = Vec::new();

    for _ in 0..n_small {
        let tree = expr_gen.generate_small();

        // Get optimal via saturation
        let optimal = {
            let config = BestFirstConfig::default().with_saturation_threshold(1000);
            let mut planner = BestFirstPlanner::from_tree(&tree, config);
            planner.run_default().best_cost
        };

        // NNUE-guided search (with limited expansions, no saturation)
        let nnue_result = run_search(&tree, guide, judge, SearchStrategy::Nnue, 50);
        nnue_regrets.push(nnue_result.regret(optimal));

        // Random priority search
        let random_result = run_search(&tree, guide, judge, SearchStrategy::Random, 50);
        random_regrets.push(random_result.regret(optimal));

        // Greedy (cost-only) search
        let greedy_result = run_search(&tree, guide, judge, SearchStrategy::Greedy, 50);
        greedy_regrets.push(greedy_result.regret(optimal));
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;

    println!(
        "    NNUE-guided regret:  {:.2}%",
        avg(&nnue_regrets) * 100.0
    );
    println!(
        "    Random regret:       {:.2}%",
        avg(&random_regrets) * 100.0
    );
    println!(
        "    Greedy regret:       {:.2}%",
        avg(&greedy_regrets) * 100.0
    );

    let nnue_vs_random = if avg(&random_regrets) > 0.0 {
        (avg(&random_regrets) - avg(&nnue_regrets)) / avg(&random_regrets) * 100.0
    } else {
        0.0
    };

    let nnue_vs_greedy = if avg(&greedy_regrets) > 0.0 {
        (avg(&greedy_regrets) - avg(&nnue_regrets)) / avg(&greedy_regrets) * 100.0
    } else {
        0.0
    };

    println!();
    println!("    NNUE vs Random: {:.1}% better", nnue_vs_random);
    println!("    NNUE vs Greedy: {:.1}% better", nnue_vs_greedy);

    // Evaluate on large kernels (compare improvement achieved)
    println!();
    println!(
        "  Large Kernels (n={}): Comparing improvement achieved",
        n_large
    );

    let mut nnue_improvements = Vec::new();
    let mut random_improvements = Vec::new();
    let mut greedy_improvements = Vec::new();

    for _ in 0..n_large {
        let tree = expr_gen.generate_large();

        let nnue_result = run_search(&tree, guide, judge, SearchStrategy::Nnue, 100);
        nnue_improvements.push(nnue_result.improvement());

        let random_result = run_search(&tree, guide, judge, SearchStrategy::Random, 100);
        random_improvements.push(random_result.improvement());

        let greedy_result = run_search(&tree, guide, judge, SearchStrategy::Greedy, 100);
        greedy_improvements.push(greedy_result.improvement());
    }

    println!(
        "    NNUE-guided improvement:  {:.2}%",
        avg(&nnue_improvements) * 100.0
    );
    println!(
        "    Random improvement:       {:.2}%",
        avg(&random_improvements) * 100.0
    );
    println!(
        "    Greedy improvement:       {:.2}%",
        avg(&greedy_improvements) * 100.0
    );

    println!();

    // Summary
    if nnue_vs_random > 10.0 && nnue_vs_greedy > 5.0 {
        println!("  [GOOD] Guide significantly outperforms baselines");
    } else if nnue_vs_random > 0.0 || nnue_vs_greedy > 0.0 {
        println!("  [OK] Guide shows some improvement over baselines");
    } else {
        println!("  [WEAK] Guide not clearly better than baselines - needs more training");
    }
}

#[derive(Clone, Copy)]
enum SearchStrategy {
    Nnue,
    Random,
    Greedy,
}

fn run_search(
    tree: &pixelflow_search::egraph::ExprTree,
    guide: &Nnue,
    judge: Option<&Nnue>,
    strategy: SearchStrategy,
    max_expansions: usize,
) -> SearchResult {
    let config = BestFirstConfig::default()
        .with_max_expansions(max_expansions)
        .with_saturation_threshold(0) // Disable saturation for fair comparison
        .with_epsilon(0.0); // Pure exploitation for evaluation

    let mut planner = BestFirstPlanner::from_tree(tree, config);
    let mut rng_state = 42u64;

    let result = planner.run(|ctx: BestFirstContext<'_>| {
        match strategy {
            SearchStrategy::Nnue => {
                let features = extract_search_features(&ctx);
                let evaluator = judge.unwrap_or(guide);
                let (pred, _) = forward_with_state(evaluator, &features);
                (pred * 1000.0) as i64
            }
            SearchStrategy::Random => {
                // LCG random
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng_state >> 33) as i64
            }
            SearchStrategy::Greedy => {
                // Pure cost-based (lower cost = higher priority, so negate)
                -(ctx.tree_cost as i64)
            }
        }
    });

    SearchResult {
        final_cost: result.best_cost,
        initial_cost: result.initial_cost,
        expansions: result.expansions,
    }
}
