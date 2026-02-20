//! Resource-asymmetric training for the unified mask.
//!
//! The mask is a compute budget allocator, not a rule recognizer.
//! The only signal that matters: did the search find a good extraction?
//!
//! Training loop:
//! 1. Oracle: run with abundant resources → oracle_cost
//! 2. Student: run with mask + exploration → student_cost
//! 3. Reward: 1 if student_cost <= oracle_cost, else 0
//! 4. REINFORCE: update mask towards decisions that led to good outcomes
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin train_mask_reinforce --release --features training
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;

use pixelflow_ir::Expr;
use pixelflow_pipeline::training::factored::parse_kernel_code;
use pixelflow_search::egraph::{all_rules, EGraph, ExprTree, GuidedSearch, NnueCostAdapter, Rewrite};
use pixelflow_search::nnue::{ExprNnue, RuleTemplates, EMBED_DIM};
use pixelflow_search::nnue::training::RewardBaseline;

/// Resource-asymmetric mask training.
#[derive(Parser, Debug)]
#[command(name = "train_mask_reinforce")]
#[command(about = "Train mask with REINFORCE on search outcomes")]
struct Args {
    /// Path to training expressions (judge_training.jsonl)
    #[arg(long, default_value = "pixelflow-pipeline/data/judge_training.jsonl")]
    data: String,

    /// Path to load pre-trained judge model (REQUIRED for value head)
    #[arg(long, default_value = "pixelflow-pipeline/data/judge.bin")]
    load_model: String,

    /// Path to save trained model
    #[arg(short, long, default_value = "pixelflow-pipeline/data/mask_reinforce.bin")]
    output: String,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Number of training epochs over the dataset
    #[arg(long, default_value_t = 5)]
    epochs: usize,

    /// Oracle max e-graph classes
    #[arg(long, default_value_t = 200)]
    oracle_classes: usize,

    /// Oracle max search epochs
    #[arg(long, default_value_t = 15)]
    oracle_epochs: usize,

    /// Student max e-graph classes
    #[arg(long, default_value_t = 50)]
    student_classes: usize,

    /// Student max search epochs
    #[arg(long, default_value_t = 5)]
    student_epochs: usize,

    /// Student mask threshold
    #[arg(long, default_value_t = 0.4)]
    threshold: f32,

    /// Exploration epsilon (probability of random rule approval)
    #[arg(long, default_value_t = 0.2)]
    epsilon: f32,

    /// Baseline decay for REINFORCE
    #[arg(long, default_value_t = 0.95)]
    baseline_decay: f32,

    /// Oracle noise: probability of applying a random rule (breaks stable zero)
    #[arg(long, default_value_t = 0.1)]
    oracle_noise: f32,

    /// Maximum samples to process (0 = all)
    #[arg(long, default_value_t = 0)]
    max_samples: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Training sample from judge_training.jsonl
#[derive(Debug, Deserialize)]
struct JudgeSample {
    #[allow(dead_code)]
    name: String,
    expression: String,
    #[allow(dead_code)]
    timing_ns: f64,
}

fn main() {
    let args = Args::parse();
    let workspace_root = find_workspace_root();

    // Load rules and build templates
    let rules = all_rules();
    let num_rules = rules.len();
    println!("Loaded {} rewrite rules", num_rules);

    let templates = build_rule_templates(&rules);
    println!("Built LHS/RHS templates for {} rules", templates.len());

    // Load the judge model (REQUIRED - contains trained value head)
    // We train the mask on top of a pre-trained judge for cost estimation
    let model_path = workspace_root.join(&args.load_model);
    println!("Loading pre-trained judge model from: {}", model_path.display());
    let mut model = ExprNnue::load(&model_path)
        .expect("Failed to load judge model. Train it first with: cargo run --bin train_judge --release --features training");

    // Randomize ONLY the mask-specific weights - preserve backbone + value head
    model.randomize_mask_only(args.seed);
    println!("Randomized mask-specific weights (backbone + value head preserved)");

    // Load training data
    let data_path = workspace_root.join(&args.data);
    println!("Loading training data from: {}", data_path.display());

    let file = File::open(&data_path).expect("Failed to open training data");
    let reader = BufReader::new(file);

    let mut expressions: Vec<(String, ExprTree)> = Vec::new();
    let mut parse_failures = 0usize;

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }

        let sample: JudgeSample = serde_json::from_str(&line).expect("Failed to parse sample");

        match parse_kernel_code(&sample.expression) {
            Some(expr) => {
                // Convert Expr to ExprTree for e-graph
                let tree = expr_to_tree(&expr);
                expressions.push((sample.expression, tree));
            }
            None => {
                parse_failures += 1;
            }
        }

        if args.max_samples > 0 && expressions.len() >= args.max_samples {
            break;
        }
    }

    println!(
        "Loaded {} expressions ({} parse failures)",
        expressions.len(),
        parse_failures
    );

    // Setup RNG for exploration
    let mut rng_state = args.seed;
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f32 / (1u64 << 31) as f32
    };

    // Baseline for REINFORCE
    let mut baseline = RewardBaseline::new(args.baseline_decay);

    // NOTE: We use NnueCostAdapter inside the loop (created fresh each iteration)
    // because we need to borrow model immutably for cost estimation,
    // then mutably for training updates.

    // Training loop
    println!("\n=== Resource-Asymmetric Training ===");
    println!(
        "Oracle: max_classes={}, max_epochs={}",
        args.oracle_classes, args.oracle_epochs
    );
    println!(
        "Student: max_classes={}, max_epochs={}, threshold={}, epsilon={}",
        args.student_classes, args.student_epochs, args.threshold, args.epsilon
    );
    println!("lr={}, baseline_decay={}", args.lr, args.baseline_decay);

    for epoch in 0..args.epochs {
        let mut total = 0usize;
        let mut total_cost_ratio = 0.0f64;
        let mut best_ratio = 0.0f64;
        let mut worst_ratio = f64::MAX;
        let mut total_grad = 0.0f32;

        // Shuffle expressions each epoch
        for i in (1..expressions.len()).rev() {
            let j = (next_rand() * (i + 1) as f32) as usize;
            expressions.swap(i, j);
        }

        for (idx, (expr_str, tree)) in expressions.iter().enumerate() {
            if idx % 100 == 0 && idx > 0 {
                println!(
                    "  Epoch {} - {}/{} (avg ratio: {:.3}, best: {:.3})",
                    epoch + 1,
                    idx,
                    expressions.len(),
                    total_cost_ratio / total as f64,
                    best_ratio,
                );
            }

            // Create cost adapter from NNUE value head (NO hardcoded costs!)
            let costs = NnueCostAdapter::new(&model);

            // === ORACLE: Run with abundant resources ===
            let mut oracle_egraph = EGraph::with_rules(all_rules());
            let oracle_root = oracle_egraph.add_expr(tree);
            let mut oracle_search = GuidedSearch::new(oracle_egraph, oracle_root, args.oracle_epochs);

            // Oracle uses epsilon-greedy to break "stable zero" equilibrium.
            // With probability (1-epsilon): uniform (accept all rules)
            // With probability epsilon: random (explore)
            // This creates variance while still mostly finding good optimizations.
            let oracle_seed = (args.seed + idx as u64).wrapping_mul(2654435761);
            let oracle_result = if args.oracle_noise > 0.0 {
                oracle_search.run_epsilon_greedy(
                    |t| tree_cost(t),
                    &costs,
                    args.oracle_noise,
                    oracle_seed,
                )
            } else {
                oracle_search.run_uniform(|t| tree_cost(t), &costs)
            };
            // Get costs for comparison:
            // - original_cost: cost of the INPUT expression (before any optimization)
            // - oracle_cost: cost after oracle optimization (unlimited resources)
            // - student_cost: cost after student optimization (limited resources + mask)
            let original_expr_ir = tree_to_expr(tree);
            let original_cost = model.predict_log_cost(&original_expr_ir);

            let oracle_tree = &oracle_result.best_tree;
            let oracle_expr = tree_to_expr(oracle_tree);
            let oracle_cost = model.predict_log_cost(&oracle_expr);

            // === STUDENT: Run with mask + exploration ===
            let mut student_egraph = EGraph::with_rules(all_rules());
            let student_root = student_egraph.add_expr(tree);
            let mut student_search = GuidedSearch::new(student_egraph, student_root, args.student_epochs);

            // Run with unified mask using LHS/RHS templates
            let student_result = student_search.run_dual_mask_with_templates(
                &model,
                &templates,
                |t| tree_cost(t),
                &costs,
                args.threshold,
                args.student_classes,
            );
            let student_tree = &student_result.best_tree;
            let student_expr = tree_to_expr(student_tree);
            let student_cost = model.predict_log_cost(&student_expr);

            // Drop the cost adapter before we mutate the model
            drop(costs);

            // === REWARD ===
            // The key insight: we want to reward IMPROVEMENT over original, not just matching oracle.
            //
            // oracle_improvement = original_cost - oracle_cost (how much oracle improved)
            // student_improvement = original_cost - student_cost (how much student improved)
            //
            // If oracle_improvement > 0, there was room to optimize.
            // Reward = student_improvement / oracle_improvement (ratio of achieved improvement)
            //
            // Special cases:
            // - oracle_improvement <= 0: expression can't be improved, skip training on this
            // - student_improvement == oracle_improvement: student matched oracle -> reward = 1.0
            // - student_improvement < oracle_improvement: student missed some optimizations -> reward < 1.0
            // - student_improvement == 0: stable zero! -> reward = 0 (penalty)

            let oracle_improvement = original_cost - oracle_cost;
            let student_improvement = original_cost - student_cost;

            // Skip expressions that oracle couldn't improve (nothing to learn)
            // Threshold is in log space: 0.01 means oracle found at least 1% improvement
            if oracle_improvement < 0.01 {
                continue;
            }

            // Reward: fraction of possible improvement achieved (0 to 1+)
            // Can exceed 1.0 if student beats oracle (lucky exploration)
            let improvement_ratio = (student_improvement / oracle_improvement).clamp(0.0, 2.0);

            // Center reward around 0 for REINFORCE: subtract 0.5 so matching oracle = +0.5, zero = -0.5
            let reward = improvement_ratio - 0.5;
            let advantage = baseline.update(reward);

            // Track cost ratio for logging (in ns space, clamped)
            let cost_diff = (oracle_cost - student_cost).clamp(-5.0, 5.0);
            let cost_ratio = cost_diff.exp() as f64;

            // Debug: show first few samples each epoch
            let pairs_approved = student_result.pairs_tried;
            let pairs_total = student_result.pairs_tried + student_result.pairs_skipped;
            let approval_rate = if pairs_total > 0 {
                pairs_approved as f32 / pairs_total as f32 * 100.0
            } else {
                0.0
            };

            if total < 3 {
                eprintln!(
                    "  [{}] orig={:.2} oracle={:.2} student={:.2} impr={:.0}% appr={:.0}% reward={:.2}",
                    idx, original_cost, oracle_cost, student_cost,
                    improvement_ratio * 100.0, approval_rate, reward
                );
            }

            total += 1;
            total_cost_ratio += cost_ratio;
            best_ratio = best_ratio.max(cost_ratio);
            worst_ratio = worst_ratio.min(cost_ratio);

            // === REINFORCE UPDATE ===
            // Pre-encode rule embeddings (cached for this sample)
            let rule_embeds = model.encode_all_rules_from_templates(&templates);

            // Collect decisions from trajectory (both approved and rejected)
            let mut decisions: Vec<(Expr, [f32; EMBED_DIM], usize, bool)> = Vec::new();

            for epoch_record in &student_result.trajectory {
                for pair in &epoch_record.pairs {
                    // Parse the expression for this class (simplified: use root expr)
                    // In a full implementation, we'd track per-class expressions
                    if let Some(expr) = parse_kernel_code(expr_str) {
                        // Look up cached rule embedding
                        let rule_embed = if pair.rule_idx < rule_embeds.len() {
                            rule_embeds[pair.rule_idx]
                        } else {
                            [0.0f32; EMBED_DIM]
                        };
                        decisions.push((
                            expr,
                            rule_embed,
                            pair.rule_idx,
                            pair.approved,
                        ));
                    }
                }
            }

            // Apply REINFORCE updates using pre-computed rule embeddings
            if !decisions.is_empty() {
                let grad = model.train_mask_reinforce_batch_with_embeds(&decisions, advantage, args.lr);
                total_grad += grad;
            }
        }

        let avg_ratio = total_cost_ratio / total as f64;
        let avg_grad = total_grad / total as f32;

        println!(
            "Epoch {}/{}: avg_ratio={:.3}, best={:.3}, worst={:.3}, baseline={:.3}, grad={:.4}",
            epoch + 1,
            args.epochs,
            avg_ratio,
            best_ratio,
            worst_ratio,
            baseline.mean,
            avg_grad,
        );
    }

    // Save model
    let output_path = workspace_root.join(&args.output);
    model.save(&output_path).expect("Failed to save model");
    println!("\nSaved model to: {}", output_path.display());
}

/// Extract cost from NNUE (used by GuidedSearch as tie-breaker).
/// The actual cost comparison uses NnueCostAdapter directly.
fn tree_cost(tree: &ExprTree) -> i64 {
    // Simple node count as tie-breaker when NNUE costs are equal
    tree.node_count() as i64
}

/// Convert ExprTree to Expr (for NNUE prediction).
fn tree_to_expr(tree: &ExprTree) -> Expr {
    use pixelflow_search::egraph::{Leaf, nnue_adapter::op_to_nnue};

    match tree {
        ExprTree::Leaf(Leaf::Var(idx)) => Expr::Var(*idx),
        ExprTree::Leaf(Leaf::Const(val)) => Expr::Const(*val),
        ExprTree::Op { op, children } => {
            let kind = op_to_nnue(*op);
            match children.len() {
                1 => Expr::Unary(kind, Box::new(tree_to_expr(&children[0]))),
                2 => Expr::Binary(
                    kind,
                    Box::new(tree_to_expr(&children[0])),
                    Box::new(tree_to_expr(&children[1])),
                ),
                3 => Expr::Ternary(
                    kind,
                    Box::new(tree_to_expr(&children[0])),
                    Box::new(tree_to_expr(&children[1])),
                    Box::new(tree_to_expr(&children[2])),
                ),
                _ => Expr::Nary(
                    kind,
                    children.iter().map(tree_to_expr).collect(),
                ),
            }
        }
    }
}

/// Convert pixelflow_ir::Expr to ExprTree.
fn expr_to_tree(expr: &Expr) -> ExprTree {
    use pixelflow_search::egraph::{Leaf, ops};

    match expr {
        Expr::Var(idx) => ExprTree::Leaf(Leaf::Var(*idx)),
        Expr::Const(val) => ExprTree::Leaf(Leaf::Const(*val)),
        Expr::Unary(op, child) => {
            let child_tree = expr_to_tree(child);
            ExprTree::Op {
                op: ops::op_from_kind(*op).expect("Invalid unary op"),
                children: vec![child_tree],
            }
        }
        Expr::Binary(op, left, right) => {
            let left_tree = expr_to_tree(left);
            let right_tree = expr_to_tree(right);
            ExprTree::Op {
                op: ops::op_from_kind(*op).expect("Invalid binary op"),
                children: vec![left_tree, right_tree],
            }
        }
        Expr::Ternary(op, a, b, c) => {
            let a_tree = expr_to_tree(a);
            let b_tree = expr_to_tree(b);
            let c_tree = expr_to_tree(c);
            ExprTree::Op {
                op: ops::op_from_kind(*op).expect("Invalid ternary op"),
                children: vec![a_tree, b_tree, c_tree],
            }
        }
        Expr::Nary(op, children) => {
            let child_trees: Vec<_> = children.iter().map(expr_to_tree).collect();
            ExprTree::Op {
                op: ops::op_from_kind(*op).expect("Invalid nary op"),
                children: child_trees,
            }
        }
    }
}

/// Build rule templates from rule definitions.
///
/// Each rule provides LHS/RHS expression templates via the Rewrite trait.
fn build_rule_templates(rules: &[Box<dyn Rewrite>]) -> RuleTemplates {
    let mut templates = RuleTemplates::with_capacity(rules.len());

    for (idx, rule) in rules.iter().enumerate() {
        // Get LHS/RHS templates from the rule (if available)
        if let (Some(lhs), Some(rhs)) = (rule.lhs_template(), rule.rhs_template()) {
            templates.set(idx, lhs, rhs);
        }
        // Rules without templates get zero embedding (handled by RuleTemplates)
    }

    templates
}

/// Find workspace root.
fn find_workspace_root() -> PathBuf {
    let mut current = std::env::current_dir().expect("Failed to get current directory");
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let contents = std::fs::read_to_string(&cargo_toml).unwrap_or_default();
            if contents.contains("[workspace]") {
                return current;
            }
        }
        if !current.pop() {
            return std::env::current_dir().expect("Failed to get current directory");
        }
    }
}
