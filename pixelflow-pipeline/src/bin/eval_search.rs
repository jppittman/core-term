//! Evaluate GuidedSearch vs egg baseline.
//!
//! This binary compares the Guide-filtered search against egg-style saturation
//! to measure:
//! 1. Whether Guide achieves similar final costs
//! 2. Whether Guide uses fewer epochs (efficiency)
//! 3. Whether Guide correctly prunes non-matching rules
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin eval_search --release
//! ```
//!
//! # Output
//!
//! Comparison metrics between guided and baseline search.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use pixelflow_search::egraph::{
    CostModel, EGraph, ExprTree, all_rules,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator, Expr, OpKind};
use pixelflow_search::nnue::guide::{GuideNnue, RULE_FEATURE_COUNT};

/// Evaluate GuidedSearch vs baseline.
#[derive(Parser, Debug)]
#[command(name = "eval_search")]
#[command(about = "Evaluate GuidedSearch vs egg baseline")]
struct Args {
    /// Number of expressions to evaluate
    #[arg(short, long, default_value_t = 50)]
    count: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 123)]
    seed: u64,

    /// Maximum epochs for search
    #[arg(long, default_value_t = 50)]
    max_epochs: usize,

    /// Path to trained Guide model
    #[arg(long, default_value = "pixelflow-pipeline/data/guide.bin")]
    guide_model: String,

    /// Match probability threshold for Guide
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
}

/// Results from one search run.
#[derive(Debug)]
struct SearchResult {
    final_cost: usize,
    epochs_used: usize,
    rules_applied: usize,
    duration_us: u128,
}

fn main() {
    let args = Args::parse();

    let workspace_root = find_workspace_root();

    // Load Guide model
    let guide_path = workspace_root.join(&args.guide_model);
    let guide = if guide_path.exists() {
        match GuideNnue::load(&guide_path) {
            Ok(g) => {
                println!("Loaded Guide model from: {}", guide_path.display());
                Some(g)
            }
            Err(e) => {
                eprintln!("Warning: Failed to load Guide model: {}", e);
                eprintln!("Running without Guide (baseline only)");
                None
            }
        }
    } else {
        eprintln!("Warning: Guide model not found at {}", guide_path.display());
        eprintln!("Running without Guide (baseline only)");
        None
    };

    println!("\nEvaluation Configuration:");
    println!("  Expressions: {}", args.count);
    println!("  Max epochs: {}", args.max_epochs);
    println!("  Guide threshold: {}", args.threshold);
    println!("  Seed: {}", args.seed);

    // Configure expression generator
    let config = BwdGenConfig {
        max_depth: 8,
        leaf_prob: 0.2,
        num_vars: 4,
        fused_op_prob: 0.3,
        max_unfuse_passes: 2,
        unfuse_prob: 0.5,
    };
    let mut expr_gen = BwdGenerator::new(args.seed, config);

    let costs = CostModel::default();

    let mut baseline_results = Vec::new();
    let mut guided_results = Vec::new();

    println!("\n=== Running Evaluation ===\n");

    for i in 0..args.count {
        // Generate expression
        let pair = expr_gen.generate();
        let expr_tree = expr_to_tree(&pair.unoptimized);

        // Run baseline (egg-style: apply all rules every epoch)
        let baseline = run_baseline(&expr_tree, &costs, args.max_epochs);
        baseline_results.push(baseline);

        // Run guided search (if Guide is available)
        if let Some(ref guide) = guide {
            let guided = run_guided(&expr_tree, &costs, guide, args.max_epochs, args.threshold);
            guided_results.push(guided);
        }

        if (i + 1) % 10 == 0 {
            println!("  Processed {}/{} expressions", i + 1, args.count);
        }
    }

    // Compute and display statistics
    println!("\n=== Results ===\n");

    print_stats("Baseline (egg-style)", &baseline_results);

    if !guided_results.is_empty() {
        println!();
        print_stats("Guided Search", &guided_results);

        // Compare
        println!("\n=== Comparison ===\n");
        compare_results(&baseline_results, &guided_results);
    }
}

/// Run baseline egg-style search (apply all rules every epoch).
fn run_baseline(
    expr_tree: &ExprTree,
    costs: &CostModel,
    max_epochs: usize,
) -> SearchResult {
    let start = Instant::now();

    let rules = all_rules();
    let num_rules = rules.len();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(expr_tree);

    let mut epochs_used = 0;
    let mut rules_applied = 0;

    for _ in 0..max_epochs {
        let changes = egraph.apply_rules_once();
        epochs_used += 1;
        rules_applied += num_rules;

        if changes == 0 {
            break; // Saturated
        }
    }

    let (_, final_cost) = egraph.extract_best(root, costs);
    let duration_us = start.elapsed().as_micros();

    SearchResult {
        final_cost,
        epochs_used,
        rules_applied,
        duration_us,
    }
}

/// Run guided search with learned rule filtering.
fn run_guided(
    expr_tree: &ExprTree,
    costs: &CostModel,
    guide: &GuideNnue,
    max_epochs: usize,
    threshold: f32,
) -> SearchResult {
    let start = Instant::now();

    let rules = all_rules();
    let num_rules = rules.len();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(expr_tree);

    let mut epochs_used = 0;
    let mut rules_applied = 0;
    let mut best_cost = egraph.extract_best(root, costs).1;

    for epoch in 0..max_epochs {
        // Collect rules that Guide predicts will match
        let mut approved_rules = Vec::new();

        for rule_idx in 0..num_rules {
            // Extract features for this rule
            let features = extract_features_for_eval(&egraph, rule_idx, epoch, max_epochs);
            let p_match = guide.predict(&features);

            if p_match > threshold {
                approved_rules.push(rule_idx);
            }
        }

        if approved_rules.is_empty() {
            // Guide predicts nothing will match - stop early
            break;
        }

        // Apply only approved rules
        let mut total_changes = 0;
        for &rule_idx in &approved_rules {
            let changes = egraph.apply_rule_at_index(rule_idx);
            total_changes += changes;
        }

        rules_applied += approved_rules.len();
        epochs_used += 1;

        if total_changes == 0 {
            break; // Saturated
        }

        // Track best cost
        let (_, cost) = egraph.extract_best(root, costs);
        if cost < best_cost {
            best_cost = cost;
        }
    }

    let (_, final_cost) = egraph.extract_best(root, costs);
    let duration_us = start.elapsed().as_micros();

    SearchResult {
        final_cost,
        epochs_used,
        rules_applied,
        duration_us,
    }
}

/// Extract features for evaluation (simplified version).
fn extract_features_for_eval(
    egraph: &EGraph,
    rule_idx: usize,
    epoch: usize,
    max_epochs: usize,
) -> [f32; RULE_FEATURE_COUNT] {
    let num_classes = egraph.num_classes();
    let num_nodes = egraph.node_count();

    // Use default match rate since we can't easily look up by index
    let historical_match_rate = 0.5;

    [
        rule_idx as f32,                           // rule_idx (normalized later)
        num_classes as f32,                        // egraph_classes
        num_nodes as f32,                          // egraph_nodes
        historical_match_rate,                     // historical_match_rate
        0.0,                                       // epochs_since_match (not tracked here)
        epoch as f32,                              // current_epoch
        max_epochs as f32,                         // max_epochs
        (max_epochs - epoch) as f32 / max_epochs as f32, // budget_fraction
    ]
}

/// Print statistics for a set of results.
fn print_stats(name: &str, results: &[SearchResult]) {
    if results.is_empty() {
        println!("{}: No results", name);
        return;
    }

    let n = results.len() as f64;

    let avg_cost: f64 = results.iter().map(|r| r.final_cost as f64).sum::<f64>() / n;
    let avg_epochs: f64 = results.iter().map(|r| r.epochs_used as f64).sum::<f64>() / n;
    let avg_rules: f64 = results.iter().map(|r| r.rules_applied as f64).sum::<f64>() / n;
    let avg_time: f64 = results.iter().map(|r| r.duration_us as f64).sum::<f64>() / n;

    println!("{}:", name);
    println!("  Avg final cost:    {:.1}", avg_cost);
    println!("  Avg epochs used:   {:.1}", avg_epochs);
    println!("  Avg rules applied: {:.1}", avg_rules);
    println!("  Avg time (µs):     {:.1}", avg_time);
}

/// Compare baseline and guided results.
fn compare_results(baseline: &[SearchResult], guided: &[SearchResult]) {
    if baseline.len() != guided.len() || baseline.is_empty() {
        println!("Cannot compare: different number of results");
        return;
    }

    let n = baseline.len() as f64;

    // Cost comparison
    let mut same_cost = 0;
    let mut guided_better = 0;
    let mut baseline_better = 0;
    let mut cost_diff_sum: f64 = 0.0;

    for (b, g) in baseline.iter().zip(guided.iter()) {
        if g.final_cost == b.final_cost {
            same_cost += 1;
        } else if g.final_cost < b.final_cost {
            guided_better += 1;
        } else {
            baseline_better += 1;
        }
        cost_diff_sum += (g.final_cost as i64 - b.final_cost as i64) as f64;
    }

    println!("Cost comparison:");
    println!("  Same cost:       {} ({:.1}%)", same_cost, same_cost as f64 / n * 100.0);
    println!("  Guided better:   {} ({:.1}%)", guided_better, guided_better as f64 / n * 100.0);
    println!("  Baseline better: {} ({:.1}%)", baseline_better, baseline_better as f64 / n * 100.0);
    println!("  Avg cost diff:   {:.2} (positive = guided worse)", cost_diff_sum / n);

    // Efficiency comparison
    let baseline_epochs: f64 = baseline.iter().map(|r| r.epochs_used as f64).sum::<f64>();
    let guided_epochs: f64 = guided.iter().map(|r| r.epochs_used as f64).sum::<f64>();
    let baseline_rules: f64 = baseline.iter().map(|r| r.rules_applied as f64).sum::<f64>();
    let guided_rules: f64 = guided.iter().map(|r| r.rules_applied as f64).sum::<f64>();
    let baseline_time: f64 = baseline.iter().map(|r| r.duration_us as f64).sum::<f64>();
    let guided_time: f64 = guided.iter().map(|r| r.duration_us as f64).sum::<f64>();

    println!("\nEfficiency comparison:");
    println!("  Epochs ratio:    {:.2}x (guided/baseline)", guided_epochs / baseline_epochs);
    println!("  Rules ratio:     {:.2}x (guided/baseline)", guided_rules / baseline_rules);
    println!("  Time ratio:      {:.2}x (guided/baseline)", guided_time / baseline_time);

    // Summary
    println!("\n=== Summary ===");

    let no_regression = baseline_better as f64 / n < 0.10; // < 10% regression
    let efficiency_gain = guided_rules / baseline_rules < 0.90; // > 10% fewer rules

    if no_regression && efficiency_gain {
        println!("✓ PASS: Guide achieves similar costs with fewer rule applications");
    } else if no_regression {
        println!("~ PARTIAL: Guide achieves similar costs but no efficiency gain");
    } else {
        println!("✗ FAIL: Guide causes cost regression in > 10% of cases");
    }
}

/// Convert pixelflow Expr to ExprTree.
fn expr_to_tree(expr: &Expr) -> ExprTree {
    match expr {
        Expr::Var(v) => ExprTree::var(*v),
        Expr::Const(c) => ExprTree::constant(*c),
        Expr::Unary(op, a) => {
            let child = expr_to_tree(a);
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: vec![child],
            }
        }
        Expr::Binary(op, a, b) => {
            let left = expr_to_tree(a);
            let right = expr_to_tree(b);
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: vec![left, right],
            }
        }
        Expr::Ternary(op, a, b, c) => {
            let c1 = expr_to_tree(a);
            let c2 = expr_to_tree(b);
            let c3 = expr_to_tree(c);
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: vec![c1, c2, c3],
            }
        }
        Expr::Nary(op, children) => {
            let child_trees: Vec<_> = children.iter().map(|c| expr_to_tree(c)).collect();
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: child_trees,
            }
        }
    }
}

/// Convert OpKind to static Op reference.
fn op_kind_to_static(kind: OpKind) -> &'static dyn pixelflow_search::egraph::ops::Op {
    use pixelflow_search::egraph::ops;

    match kind {
        OpKind::Add => &ops::Add,
        OpKind::Sub => &ops::Sub,
        OpKind::Mul => &ops::Mul,
        OpKind::Div => &ops::Div,
        OpKind::Neg => &ops::Neg,
        OpKind::Recip => &ops::Recip,
        OpKind::Sqrt => &ops::Sqrt,
        OpKind::Rsqrt => &ops::Rsqrt,
        OpKind::Abs => &ops::Abs,
        OpKind::Min => &ops::Min,
        OpKind::Max => &ops::Max,
        OpKind::MulAdd => &ops::MulAdd,
        OpKind::MulRsqrt => &ops::MulRsqrt,
        OpKind::Floor => &ops::Floor,
        OpKind::Ceil => &ops::Ceil,
        OpKind::Round => &ops::Round,
        OpKind::Fract => &ops::Fract,
        OpKind::Sin => &ops::Sin,
        OpKind::Cos => &ops::Cos,
        OpKind::Tan => &ops::Tan,
        OpKind::Asin => &ops::Asin,
        OpKind::Acos => &ops::Acos,
        OpKind::Atan => &ops::Atan,
        OpKind::Atan2 => &ops::Atan2,
        OpKind::Exp => &ops::Exp,
        OpKind::Exp2 => &ops::Exp2,
        OpKind::Ln => &ops::Ln,
        OpKind::Log2 => &ops::Log2,
        OpKind::Log10 => &ops::Log10,
        OpKind::Pow => &ops::Pow,
        OpKind::Hypot => &ops::Hypot,
        OpKind::Lt => &ops::Lt,
        OpKind::Le => &ops::Le,
        OpKind::Gt => &ops::Gt,
        OpKind::Ge => &ops::Ge,
        OpKind::Eq => &ops::Eq,
        OpKind::Ne => &ops::Ne,
        OpKind::Select => &ops::Select,
        OpKind::Clamp => &ops::Clamp,
        OpKind::Tuple => &ops::Tuple,
        OpKind::Var | OpKind::Const => panic!("Var/Const should not need op conversion"),
    }
}

/// Find workspace root by looking for Cargo.toml with [workspace].
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
