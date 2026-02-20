//! AlphaZero-style MCTS Training for E-graph Optimization.
//!
//! This binary implements self-play training using MCTS with neural guidance:
//! - Policy head (mask) provides action priors
//! - Value head (judge) evaluates states
//! - MCTS improves policy via search
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin train_mcts --release --features training
//! ```
//!
//! # Training Strategy: Synthetic Learning with Value Head
//!
//! The value head (judge) was trained on real SIMD benchmarks. We use it as
//! ground truth to train the policy:
//!
//! 1. For each expression, sample random (class, rule) actions
//! 2. For each action, compute: `value_delta = value_head(after) - value_head(before)`
//! 3. Policy target = softmax(-value_delta / temperature) (lower cost = higher prob)
//! 4. Train policy to match this target via cross-entropy
//!
//! Key insight: No real benchmarking needed during training. Value head is the
//! oracle (already grounded in SIMD timing). Policy learns to select actions
//! that the judge says are good.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use clap::Parser;

use pixelflow_search::egraph::{Rewrite, all_rules};
use pixelflow_search::mcts::{MctsConfig, MctsState, AccumulatorDelta};
use pixelflow_search::nnue::{ExprNnue, Expr, RuleTemplates, EMBED_DIM};
use pixelflow_search::nnue::factored::{EdgeAccumulator, StructuralFeatures, K};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator, ExprGenConfig, ExprGenerator};
use pixelflow_pipeline::training::factored::parse_kernel_code;

/// AlphaZero-style MCTS training.
#[derive(Parser, Debug)]
#[command(name = "train_mcts")]
#[command(about = "Train mask/value heads using MCTS self-play")]
struct Args {
    /// Path to training expressions (one per line, kernel syntax)
    #[arg(long, default_value = "pixelflow-pipeline/data/search_training_superguide.jsonl")]
    input: String,

    /// Path to pre-trained judge model (value head)
    #[arg(long, default_value = "pixelflow-pipeline/data/judge.bin")]
    judge: String,

    /// Output path for trained model
    #[arg(short, long, default_value = "pixelflow-pipeline/data/mcts_trained.bin")]
    output: String,

    /// Number of self-play iterations
    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// Number of MCTS simulations per move
    #[arg(long, default_value_t = 32)]
    simulations: usize,

    /// Exploration constant (c_puct)
    #[arg(long, default_value_t = 1.5)]
    c_puct: f32,

    /// Temperature for action selection (training)
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Maximum samples per iteration
    #[arg(long, default_value_t = 100)]
    max_samples: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Value loss weight (vs policy loss)
    #[arg(long, default_value_t = 1.0)]
    value_weight: f32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Print progress every N samples
    #[arg(long, default_value_t = 10)]
    print_every: usize,

    /// Number of random actions to sample per expression
    #[arg(long, default_value_t = 16)]
    actions_per_expr: usize,

    /// Temperature for policy target softmax
    #[arg(long, default_value_t = 0.5)]
    policy_temperature: f32,

    /// Number of expressions to generate with BwdGenerator (large expressions)
    #[arg(long, default_value_t = 50)]
    gen_bwd_count: usize,

    /// Maximum depth for BwdGenerator expressions
    #[arg(long, default_value_t = 8)]
    gen_bwd_depth: usize,

    /// Number of expressions to generate with ExprGenerator (random expressions)
    #[arg(long, default_value_t = 50)]
    gen_expr_count: usize,

    /// Maximum depth for ExprGenerator expressions
    #[arg(long, default_value_t = 7)]
    gen_expr_depth: usize,

    /// Use only generated expressions (skip loading from file)
    #[arg(long, default_value_t = false)]
    gen_only: bool,
}

/// Training sample from synthetic learning with value head.
struct MctsSample {
    /// Accumulator state.
    accumulator: EdgeAccumulator,
    /// Policy target from value head deltas (softmax of negative improvements).
    policy_target: Vec<f32>,
    /// Value target (cost from value head).
    value_target: f32,
    /// Per-rule gradients for mask MLP (d_loss/d_logit * logit_gradient).
    #[allow(dead_code)]
    policy_grad: Vec<f32>,
}

/// Simple LCG random number generator for no_std compatibility.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== AlphaZero-style MCTS Training ===");
    println!();

    // Load pre-trained judge
    let judge_path = Path::new(&args.judge);
    let mut model = if judge_path.exists() {
        println!("Loading pre-trained judge from: {}", args.judge);
        ExprNnue::load(judge_path)?
    } else {
        println!("No judge found, starting with random initialization");
        ExprNnue::new_with_latency_prior(args.seed)
    };

    // Randomize only mask-specific weights (keep backbone + value head from judge)
    model.randomize_mask_only(args.seed);
    println!("Randomized mask-specific weights for training");

    // Load and generate training expressions
    let mut expressions = Vec::new();

    // Load from file unless gen_only is set
    if !args.gen_only {
        let loaded = load_expressions(&args.input, args.max_samples)?;
        println!("Loaded {} expressions from file", loaded.len());
        expressions.extend(loaded);
    }

    // Generate large expressions with BwdGenerator (optimized → unoptimized pairs)
    if args.gen_bwd_count > 0 {
        let bwd_exprs = generate_bwd_expressions(args.gen_bwd_count, args.gen_bwd_depth, args.seed);
        println!("Generated {} large expressions with BwdGenerator (depth={})", bwd_exprs.len(), args.gen_bwd_depth);
        expressions.extend(bwd_exprs);
    }

    // Generate random expressions with ExprGenerator
    if args.gen_expr_count > 0 {
        let expr_exprs = generate_random_expressions(args.gen_expr_count, args.gen_expr_depth, args.seed + 1000);
        println!("Generated {} random expressions with ExprGenerator (depth={})", expr_exprs.len(), args.gen_expr_depth);
        expressions.extend(expr_exprs);
    }

    println!("Total training expressions: {}", expressions.len());

    // Get available rules and create templates
    let rules = all_rules();
    let rule_templates = create_rule_templates(&rules);
    println!("Created templates for {} rules", rules.len());

    // Pre-encode rules
    let rule_embeds = model.encode_all_rules_from_templates(&rule_templates);
    println!("Pre-encoded {} rule embeddings", rule_embeds.len());
    println!();

    // MCTS config (may be used for full MCTS in the future)
    let _mcts_config = MctsConfig {
        num_simulations: args.simulations,
        c_puct: args.c_puct,
        temperature: args.temperature,
        max_depth: 50,
        dirichlet_alpha: 0.3,
        dirichlet_weight: 0.25,
    };

    // Random number generator
    let mut rng = SimpleRng::new(args.seed);
    let num_rules = rules.len();

    // Training loop
    for iteration in 0..args.iterations {
        println!("=== Iteration {}/{} ===", iteration + 1, args.iterations);
        let iter_start = Instant::now();

        let mut samples: Vec<MctsSample> = Vec::new();
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_avg_delta = 0.0;

        // Synthetic learning: generate policy targets from value head
        for (i, expr) in expressions.iter().enumerate() {
            let sample_start = Instant::now();

            // Get initial state and cost
            let state = MctsState::from_expr(expr, &model.embeddings);
            let initial_cost = model.predict_cost_from_accumulator(&state.accumulator);

            // Sample random actions and compute value deltas
            // For synthetic learning, we create fake "deltas" to simulate rule applications
            let mut deltas = Vec::with_capacity(args.actions_per_expr);
            let mut cost_deltas = Vec::with_capacity(args.actions_per_expr);

            for _ in 0..args.actions_per_expr {
                // Sample a random action (rule index)
                let rule_idx = rng.next_usize(num_rules);

                // Create a synthetic delta for this action
                // In a real implementation, this would come from actually applying the rule
                let delta = create_synthetic_delta(&mut rng);

                // Compute cost after applying this delta
                let mut acc_after = state.accumulator.clone();
                delta.apply(&mut acc_after);
                let cost_after = model.predict_cost_from_accumulator(&acc_after);

                // Cost delta: positive = worse, negative = better
                let cost_delta = cost_after - initial_cost;

                deltas.push((rule_idx, delta));
                cost_deltas.push(cost_delta);
            }

            // Convert cost deltas to policy targets via softmax(-delta / temperature)
            // Lower cost delta (improvement) = higher probability
            let policy_target = softmax_policy(&cost_deltas, args.policy_temperature);

            // Compute average improvement for logging
            let avg_delta: f32 = cost_deltas.iter().sum::<f32>() / cost_deltas.len() as f32;
            total_avg_delta += avg_delta;

            // Expand policy target to full rule space (most rules get 0 probability)
            let mut full_policy = vec![0.0f32; num_rules];
            for (idx, (rule_idx, _)) in deltas.iter().enumerate() {
                full_policy[*rule_idx] = policy_target[idx];
            }

            samples.push(MctsSample {
                accumulator: state.accumulator.clone(),
                policy_target: full_policy,
                value_target: initial_cost,
                policy_grad: vec![0.0; num_rules], // Computed during backprop
            });

            if (i + 1) % args.print_every == 0 {
                let elapsed = sample_start.elapsed();
                println!(
                    "  Sample {}/{}: cost={:.2}, avg_delta={:.3}, time={:.1}ms",
                    i + 1,
                    expressions.len(),
                    initial_cost,
                    avg_delta,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }

        println!("Generated {} training samples", samples.len());
        let avg_improvement = total_avg_delta / samples.len() as f32;
        println!("  Average cost delta (negative=improvement): {:.4}", avg_improvement);

        // Train on collected samples with gradient descent
        let train_start = Instant::now();
        for sample in &samples {
            let (policy_loss, value_loss) = train_step_with_backprop(
                &mut model,
                sample,
                &rule_embeds,
                args.lr,
                args.value_weight,
            );
            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
        }

        let avg_policy_loss = total_policy_loss / samples.len() as f32;
        let avg_value_loss = total_value_loss / samples.len() as f32;
        let train_time = train_start.elapsed();
        let iter_time = iter_start.elapsed();

        println!(
            "  policy_loss={:.4}, value_loss={:.4}",
            avg_policy_loss, avg_value_loss
        );
        println!(
            "  train_time={:.1}ms, total_time={:.1}s",
            train_time.as_secs_f64() * 1000.0,
            iter_time.as_secs_f64()
        );
        println!();
    }

    // Save trained model
    let output_path = Path::new(&args.output);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    model.save(output_path)?;
    println!("Saved trained model to: {}", args.output);

    Ok(())
}

/// Load expressions from training file.
fn load_expressions(path: &str, max_samples: usize) -> Result<Vec<Expr>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut expressions = Vec::new();
    for line in reader.lines().take(max_samples) {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as JSON first
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(expr_str) = json.get("expression").and_then(|v| v.as_str()) {
                if let Some(expr) = parse_kernel_code(expr_str) {
                    expressions.push(expr);
                }
            }
        } else {
            // Try direct parse
            if let Some(expr) = parse_kernel_code(&line) {
                expressions.push(expr);
            }
        }
    }

    Ok(expressions)
}

/// Generate large expressions using BwdGenerator.
///
/// BwdGenerator creates optimized expressions (with fused ops) then applies
/// unfusing rewrites to create equivalent unoptimized expressions.
/// We use the unoptimized expressions for training (harder to optimize).
fn generate_bwd_expressions(count: usize, max_depth: usize, seed: u64) -> Vec<Expr> {
    let config = BwdGenConfig {
        max_depth,
        leaf_prob: 0.2, // Lower leaf prob = larger expressions
        num_vars: 4,
        fused_op_prob: 0.5, // Higher = more fused ops to unfuse
        max_unfuse_passes: 4,
        unfuse_prob: 0.8,
    };

    let mut generator = BwdGenerator::new(seed, config);
    let mut expressions = Vec::with_capacity(count);

    for _ in 0..count {
        let pair = generator.generate();
        // Use the unoptimized expression (this is what the model needs to optimize)
        expressions.push(pair.unoptimized);
    }

    expressions
}

/// Generate random expressions using ExprGenerator.
///
/// These are randomly structured expressions with various operations.
fn generate_random_expressions(count: usize, max_depth: usize, seed: u64) -> Vec<Expr> {
    let config = ExprGenConfig {
        max_depth,
        leaf_prob: 0.2, // Lower = larger expressions
        num_vars: 4,
        include_fused: true,
    };

    let mut generator = ExprGenerator::new(seed, config);
    let mut expressions = Vec::with_capacity(count);

    for _ in 0..count {
        expressions.push(generator.generate());
    }

    expressions
}

/// Create rule templates for rule embedding.
fn create_rule_templates(rules: &[Box<dyn Rewrite>]) -> RuleTemplates {
    let mut templates = RuleTemplates::with_capacity(rules.len());

    // For now, create simple placeholder templates
    // In a full implementation, each rule would provide its LHS/RHS patterns
    for (i, _rule) in rules.iter().enumerate() {
        // Create placeholder expressions
        // Real rules would have proper LHS/RHS templates
        let placeholder = Expr::Var(0);
        templates.set(i, placeholder.clone(), placeholder);
    }

    templates
}

/// Single training step on an MCTS sample.
#[allow(dead_code)]
fn train_step(
    model: &mut ExprNnue,
    sample: &MctsSample,
    rule_embeds: &[[f32; EMBED_DIM]],
    lr: f32,
    value_weight: f32,
) -> (f32, f32) {
    // Forward pass
    let structural = StructuralFeatures::default();
    let hidden = forward_shared(model, &sample.accumulator, &structural);
    let expr_embed = compute_expr_embed(model, &hidden);

    // Compute value prediction first (used as input to mask)
    let pred_value = value_mlp_forward(model, &expr_embed);

    // Policy loss: cross-entropy with MCTS distribution
    // Mask sees value as context: "Given this costs X, should I try rule R?"
    let mask_features = compute_mask_features(model, &expr_embed, pred_value);
    let policy_logits: Vec<f32> = rule_embeds
        .iter()
        .enumerate()
        .map(|(r, re)| bilinear_score(model, &mask_features, re, r))
        .collect();

    let policy_loss = cross_entropy_loss(&policy_logits, &sample.policy_target);

    // Value loss: MSE
    let value_loss = (pred_value - sample.value_target).powi(2);

    // Combined loss gradient
    let total_loss = policy_loss + value_weight * value_loss;

    // Note: In a full implementation, we would compute gradients and update weights
    // For now, just return the losses
    let _ = (lr, total_loss);

    (policy_loss, value_loss)
}

// Forward pass helpers (copied from ExprNnue internals for training)

fn forward_shared(
    model: &ExprNnue,
    acc: &EdgeAccumulator,
    structural: &StructuralFeatures,
) -> [f32; 64] {
    let mut hidden = model.b1;

    // Edge accumulator
    for (i, &val) in acc.values.iter().enumerate() {
        for (j, h) in hidden.iter_mut().enumerate() {
            *h += val * model.w1[i][j];
        }
    }

    // Structural features
    for (i, &val) in structural.values.iter().enumerate() {
        let input_idx = 2 * K + i;
        for (j, h) in hidden.iter_mut().enumerate() {
            *h += val * model.w1[input_idx][j];
        }
    }

    // ReLU
    for h in &mut hidden {
        *h = h.max(0.0);
    }

    hidden
}

fn compute_expr_embed(model: &ExprNnue, hidden: &[f32; 64]) -> [f32; EMBED_DIM] {
    use pixelflow_search::nnue::factored::HIDDEN_DIM;

    let mut embed = model.expr_proj_b;
    for j in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            embed[k] += hidden[j] * model.expr_proj_w[j][k];
        }
    }
    embed
}

/// Compute mask features with value prediction as context (value-aware mask).
///
/// Input: [expr_embed (24), value_pred (1)] = 25 dims
/// The mask learns: "Given this costs X, should I try rule R?"
fn compute_mask_features(
    model: &ExprNnue,
    expr_embed: &[f32; EMBED_DIM],
    value_pred: f32,
) -> [f32; EMBED_DIM] {
    use pixelflow_search::nnue::factored::MLP_HIDDEN;

    let mut h = model.mask_mlp_b1;

    // Process expr_embed (first 24 dims)
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            h[j] += expr_embed[i] * model.mask_mlp_w1[i][j];
        }
    }

    // Process value_pred (25th dim) - the value-aware component
    for j in 0..MLP_HIDDEN {
        h[j] += value_pred * model.mask_mlp_w1[EMBED_DIM][j];
    }

    // ReLU
    for j in 0..MLP_HIDDEN {
        h[j] = h[j].max(0.0);
    }

    let mut out = model.mask_mlp_b2;
    for j in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            out[k] += h[j] * model.mask_mlp_w2[j][k];
        }
    }
    out
}

fn bilinear_score(
    model: &ExprNnue,
    mask_features: &[f32; EMBED_DIM],
    rule_embed: &[f32; EMBED_DIM],
    rule_idx: usize,
) -> f32 {
    let mut transformed = [0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        for j in 0..EMBED_DIM {
            transformed[j] += mask_features[i] * model.interaction[i][j];
        }
    }

    let mut score = model.mask_rule_bias[rule_idx];
    for k in 0..EMBED_DIM {
        score += transformed[k] * rule_embed[k];
    }
    score
}

fn value_mlp_forward(model: &ExprNnue, expr_embed: &[f32; EMBED_DIM]) -> f32 {
    use pixelflow_search::nnue::factored::MLP_HIDDEN;

    let mut h = model.value_mlp_b1;
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            h[j] += expr_embed[i] * model.value_mlp_w1[i][j];
        }
    }
    for j in 0..MLP_HIDDEN {
        h[j] = h[j].max(0.0);
    }

    let mut cost = model.value_mlp_b2;
    for j in 0..MLP_HIDDEN {
        cost += h[j] * model.value_mlp_w2[j];
    }
    cost
}

fn cross_entropy_loss(logits: &[f32], targets: &[f32]) -> f32 {
    if logits.is_empty() || targets.is_empty() {
        return 0.0;
    }

    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| libm::expf(x - max_logit)).sum();

    // Cross-entropy: -sum(target * log(prob))
    let mut loss = 0.0;
    for (logit, &target) in logits.iter().zip(targets) {
        if target > 0.0 {
            let prob = libm::expf(logit - max_logit) / exp_sum;
            loss -= target * libm::logf(prob.max(1e-7));
        }
    }
    loss
}

// ============================================================================
// Synthetic Learning Helpers
// ============================================================================

/// Create a synthetic accumulator delta (simulates rule application effect).
///
/// In a full implementation, this would come from actually applying a rule to
/// an e-graph and observing the change in edge structure. For synthetic learning,
/// we create small random perturbations that the value head can score.
fn create_synthetic_delta(rng: &mut SimpleRng) -> AccumulatorDelta {
    let mut delta = AccumulatorDelta::zero();

    // Create small random perturbations to the accumulator
    // This simulates adding/removing edges when a rule fires
    let num_changes = 2 + (rng.next_usize(4)); // 2-5 changes

    for _ in 0..num_changes {
        let idx = rng.next_usize(2 * K);
        // Small perturbation: [-0.5, 0.5]
        let change = (rng.next_f32() - 0.5) * 1.0;
        delta.values[idx] += change;
    }

    delta
}

/// Convert cost deltas to policy targets via temperature-scaled softmax.
///
/// Lower cost delta (more improvement) = higher probability.
/// We negate deltas so improvements have positive values before softmax.
fn softmax_policy(cost_deltas: &[f32], temperature: f32) -> Vec<f32> {
    if cost_deltas.is_empty() {
        return Vec::new();
    }

    // Negate so improvements are positive, scale by temperature
    let scaled: Vec<f32> = cost_deltas
        .iter()
        .map(|&d| -d / temperature.max(0.01))
        .collect();

    // Softmax with numerical stability
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| libm::expf(x - max_val)).collect();
    let sum: f32 = exps.iter().sum();

    if sum < 1e-10 {
        // Uniform fallback
        vec![1.0 / cost_deltas.len() as f32; cost_deltas.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Training step with actual backpropagation through mask MLP.
///
/// Computes gradients and updates weights for:
/// - Mask MLP (policy head, value-aware)
/// - Interaction matrix
/// - Rule biases
fn train_step_with_backprop(
    model: &mut ExprNnue,
    sample: &MctsSample,
    rule_embeds: &[[f32; EMBED_DIM]],
    lr: f32,
    value_weight: f32,
) -> (f32, f32) {
    use pixelflow_search::nnue::factored::{HIDDEN_DIM, MLP_HIDDEN};

    // ========== FORWARD PASS ==========
    let structural = StructuralFeatures::default();
    let hidden = forward_shared(model, &sample.accumulator, &structural);
    let expr_embed = compute_expr_embed(model, &hidden);

    // Value forward first (needed as input to mask)
    let pred_value = value_mlp_forward(model, &expr_embed);

    // Mask MLP forward with caching for backprop (value-aware)
    let (mask_hidden, mask_features) = mask_mlp_forward_cached(model, &expr_embed, pred_value);

    // Policy logits
    let policy_logits: Vec<f32> = rule_embeds
        .iter()
        .enumerate()
        .map(|(r, re)| bilinear_score(model, &mask_features, re, r))
        .collect();

    // Softmax for policy loss
    let max_logit = policy_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = policy_logits.iter().map(|&x| libm::expf(x - max_logit)).collect();
    let exp_sum: f32 = exps.iter().sum();
    let policy_probs: Vec<f32> = exps.iter().map(|&e| e / exp_sum.max(1e-10)).collect();

    // Policy loss: cross-entropy
    let mut policy_loss = 0.0;
    for (prob, &target) in policy_probs.iter().zip(sample.policy_target.iter()) {
        if target > 0.0 {
            policy_loss -= target * libm::logf(prob.max(1e-7));
        }
    }

    // Value loss: MSE
    let value_loss = (pred_value - sample.value_target).powi(2);

    // ========== BACKWARD PASS (Policy only) ==========
    // d_loss/d_logit = prob - target (softmax cross-entropy gradient)
    let policy_grad: Vec<f32> = policy_probs
        .iter()
        .zip(sample.policy_target.iter())
        .map(|(&prob, &target)| prob - target)
        .collect();

    // Gradient w.r.t. mask features
    let mut d_mask_features = [0.0f32; EMBED_DIM];

    for (r, d_logit) in policy_grad.iter().enumerate() {
        if d_logit.abs() < 1e-10 {
            continue;
        }

        // d_score/d_mask_features = interaction @ rule_embed
        // d_loss/d_mask_features += d_logit * d_score/d_mask_features
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                d_mask_features[i] += d_logit * model.interaction[i][j] * rule_embeds[r][j];
            }
        }

        // Update interaction matrix: d_loss/d_interaction = d_logit * mask_features ⊗ rule_embed
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                model.interaction[i][j] -= lr * d_logit * mask_features[i] * rule_embeds[r][j];
            }
        }

        // Update rule bias
        model.mask_rule_bias[r] -= lr * d_logit;
    }

    // Backprop through mask MLP layer 2: d_features/d_hidden
    let mut d_mask_hidden = [0.0f32; MLP_HIDDEN];
    for j in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            d_mask_hidden[j] += d_mask_features[k] * model.mask_mlp_w2[j][k];

            // Update w2
            model.mask_mlp_w2[j][k] -= lr * d_mask_features[k] * mask_hidden[j];
        }
    }

    // Update b2
    for k in 0..EMBED_DIM {
        model.mask_mlp_b2[k] -= lr * d_mask_features[k];
    }

    // ReLU backward
    for j in 0..MLP_HIDDEN {
        if mask_hidden[j] <= 0.0 {
            d_mask_hidden[j] = 0.0;
        }
    }

    // Backprop through mask MLP layer 1 (value-aware: 25 inputs)
    // Update weights for expr_embed (first 24 dims)
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            model.mask_mlp_w1[i][j] -= lr * d_mask_hidden[j] * expr_embed[i];
        }
    }

    // Update weights for value_pred (25th dim)
    for j in 0..MLP_HIDDEN {
        model.mask_mlp_w1[EMBED_DIM][j] -= lr * d_mask_hidden[j] * pred_value;
    }

    // Update b1
    for j in 0..MLP_HIDDEN {
        model.mask_mlp_b1[j] -= lr * d_mask_hidden[j];
    }

    // Note: We don't backprop further into expr_proj or backbone
    // The value head (judge) is frozen during policy training
    let _ = (value_weight, hidden, HIDDEN_DIM);

    (policy_loss, value_loss)
}

/// Mask MLP forward pass with cached activations for backprop (value-aware).
///
/// Input: [expr_embed (24), value_pred (1)] = 25 dims
fn mask_mlp_forward_cached(
    model: &ExprNnue,
    expr_embed: &[f32; EMBED_DIM],
    value_pred: f32,
) -> ([f32; 16], [f32; EMBED_DIM]) {
    use pixelflow_search::nnue::factored::MLP_HIDDEN;

    // Layer 1: MASK_INPUT_DIM (25) -> MLP_HIDDEN
    let mut h = model.mask_mlp_b1;

    // Process expr_embed (first 24 dims)
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            h[j] += expr_embed[i] * model.mask_mlp_w1[i][j];
        }
    }

    // Process value_pred (25th dim) - the value-aware component
    for j in 0..MLP_HIDDEN {
        h[j] += value_pred * model.mask_mlp_w1[EMBED_DIM][j];
    }

    // ReLU (keep pre-activation for gradient)
    let h_pre_relu = h;
    for j in 0..MLP_HIDDEN {
        h[j] = h[j].max(0.0);
    }

    // Layer 2: MLP_HIDDEN -> EMBED_DIM
    let mut out = model.mask_mlp_b2;
    for j in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM {
            out[k] += h[j] * model.mask_mlp_w2[j][k];
        }
    }

    // Return post-ReLU hidden and output (we need post-ReLU for gradient computation)
    let _ = h_pre_relu; // We actually use the post-ReLU hidden state
    (h, out)
}
