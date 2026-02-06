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
use pixelflow_search::egraph::{all_rules, CostModel, EGraph, ExprTree, GuidedSearch, Rewrite};
use pixelflow_search::nnue::{DualHeadNnue, RuleTemplates, EMBED_DIM};
use pixelflow_search::nnue::training::RewardBaseline;

/// Resource-asymmetric mask training.
#[derive(Parser, Debug)]
#[command(name = "train_mask_reinforce")]
#[command(about = "Train mask with REINFORCE on search outcomes")]
struct Args {
    /// Path to training expressions (judge_training.jsonl)
    #[arg(long, default_value = "pixelflow-pipeline/data/judge_training.jsonl")]
    data: String,

    /// Path to load pre-trained model (optional)
    #[arg(long)]
    load_model: Option<String>,

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

    // Load or create model
    let mut model = if let Some(model_path) = &args.load_model {
        let path = workspace_root.join(model_path);
        println!("Loading pre-trained model from: {}", path.display());
        DualHeadNnue::load(&path).expect("Failed to load model")
    } else {
        println!("Creating new model with random initialization");
        let factored = pixelflow_search::nnue::FactoredNnue::new();
        let mut model = DualHeadNnue::from_factored(&factored);
        model.randomize_mask_weights(args.seed);
        model
    };

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

    // Costs model
    let costs = CostModel::default();

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

            // === ORACLE: Run with abundant resources ===
            let mut oracle_egraph = EGraph::with_rules(all_rules());
            let oracle_root = oracle_egraph.add_expr(tree);
            let mut oracle_search = GuidedSearch::new(oracle_egraph, oracle_root, args.oracle_epochs);

            let oracle_result = oracle_search.run_uniform(
                |t| tree_cost(t),
                &costs,
            );
            let oracle_cost = oracle_result.best_cost;

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
            let student_cost = student_result.best_cost;

            // === REWARD ===
            // Cost ratio: 1.0 = matched oracle, >1.0 = beat oracle, <1.0 = worse
            let cost_ratio = if student_cost > 0 {
                oracle_cost as f64 / student_cost as f64
            } else {
                1.0
            };

            // Use cost ratio as reward (centered around 1.0)
            // Advantage = how much better/worse than baseline
            let reward = cost_ratio as f32;
            let advantage = baseline.update(reward);

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

/// Simple tree cost (node count).
fn tree_cost(tree: &ExprTree) -> i64 {
    tree.node_count() as i64
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
