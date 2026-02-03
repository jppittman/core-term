//! Collect search trajectories for Guide (Search Head) training.
//!
//! This binary runs BestFirstPlanner searches and records trajectories
//! for AlphaZero-style training of the search guidance head.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin collect_search_data --release
//! ```
//!
//! # Output
//!
//! - `pixelflow-pipeline/data/search_training.jsonl` - Search trajectory data

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use pixelflow_search::egraph::{
    BestFirstPlanner, BestFirstConfig, CostModel, EGraph, all_rules,
    SearchTrajectory, TrajectoryStep,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator, DualHeadNnue, Expr, OpKind};

/// Collect search trajectories for Guide training.
#[derive(Parser, Debug)]
#[command(name = "collect_search_data")]
#[command(about = "Collect search trajectories for Guide training")]
struct Args {
    /// Number of expressions to generate
    #[arg(short, long, default_value_t = 50)]
    count: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Output path
    #[arg(short, long, default_value = "pixelflow-pipeline/data/search_training.jsonl")]
    output: String,

    /// Max expansions for short search
    #[arg(long, default_value_t = 50)]
    short_expansions: usize,

    /// Max expansions for long search (for hindsight labeling)
    #[arg(long, default_value_t = 200)]
    long_expansions: usize,

    /// Path to Judge model (optional, uses heuristic if not provided)
    #[arg(long)]
    judge_model: Option<String>,

    /// Superguide interval (every N expansions, do a deep exploration burst)
    #[arg(long)]
    superguide_interval: Option<usize>,

    /// Superguide multiplier (how many extra expansions during burst)
    #[arg(long, default_value_t = 10)]
    superguide_multiplier: usize,
}

fn main() {
    let args = Args::parse();

    // Find workspace root
    let workspace_root = find_workspace_root();

    println!("Collecting search trajectories...");
    println!("  Count: {}", args.count);
    println!("  Short search: {} expansions", args.short_expansions);
    println!("  Long search: {} expansions", args.long_expansions);
    if let Some(interval) = args.superguide_interval {
        println!("  Superguide: every {} expansions, {}x burst", interval, args.superguide_multiplier);
    }

    // Load Judge model if provided
    let judge = if let Some(judge_path) = &args.judge_model {
        let path = workspace_root.join(judge_path);
        match DualHeadNnue::load(&path) {
            Ok(model) => {
                println!("  Loaded Judge model from: {}", path.display());
                Some(model)
            }
            Err(e) => {
                eprintln!("Warning: Failed to load Judge model: {}", e);
                eprintln!("  Falling back to heuristic cost model");
                None
            }
        }
    } else {
        println!("  Using heuristic cost model (no Judge model provided)");
        None
    };

    // Configure expression generator
    // Balance complexity with search speed
    let config = BwdGenConfig {
        max_depth: 8,
        leaf_prob: 0.2,
        num_vars: 4,
        fused_op_prob: 0.3,
        max_unfuse_passes: 2,
        unfuse_prob: 0.5,
    };
    let mut expr_gen = BwdGenerator::new(args.seed, config);

    // Output file
    let output_path = workspace_root.join(&args.output);
    fs::create_dir_all(output_path.parent().unwrap()).expect("Failed to create output directory");
    let mut output_file = fs::File::create(&output_path).expect("Failed to create output file");

    let mut total_samples = 0usize;
    let mut successful = 0usize;

    for i in 0..args.count {
        let pair = expr_gen.generate();
        let expr = pair.unoptimized; // Start from unoptimized form

        // Convert to ExprTree for search
        let Some(tree) = expr_to_tree(&expr) else {
            continue;
        };

        // Short search (low saturation threshold to use best-first more often)
        let short_config = BestFirstConfig {
            max_expansions: args.short_expansions,
            hard_timeout: Duration::from_secs(5),
            max_classes: 1000,
            saturation_threshold: 20, // Only saturate tiny expressions
            ..BestFirstConfig::default()
        };

        let mut short_planner = BestFirstPlanner::from_tree_with_rules(&tree, short_config, all_rules());

        let (short_result, short_trajectory) = if let Some(ref judge_model) = judge {
            short_planner.run_recording_with_nnue(judge_model)
        } else {
            short_planner.run_recording(|ctx| ctx.tree_cost as i64)
        };

        // Long search for hindsight labeling (with optional superguide)
        let mut long_config = BestFirstConfig {
            max_expansions: args.long_expansions,
            hard_timeout: Duration::from_secs(10),
            max_classes: 2000,
            saturation_threshold: 20, // Only saturate tiny expressions
            ..BestFirstConfig::default()
        };
        if let Some(interval) = args.superguide_interval {
            long_config = long_config.with_superguide(interval, args.superguide_multiplier);
        }

        let mut long_planner = BestFirstPlanner::from_tree_with_rules(&tree, long_config, all_rules());

        let (long_result, _) = if let Some(ref judge_model) = judge {
            long_planner.run_recording_with_nnue(judge_model)
        } else {
            long_planner.run_recording(|ctx| ctx.tree_cost as i64)
        };

        // Extract training samples from trajectory
        // Label each step with the hindsight target (long search result)
        let target_cost = long_result.best_cost;

        for step in &short_trajectory.steps {
            // Improvement gap: how much better can we do from this state?
            let gap = step.best_cost_so_far as i64 - target_cost as i64;
            let improvement = if short_trajectory.initial_cost > 0 {
                gap as f64 / short_trajectory.initial_cost as f64
            } else {
                0.0
            };

            // Write sample as JSONL
            let sample = TrainingSampleJson {
                expression_idx: i,
                step_idx: step.expansion,
                tree_cost: step.tree_cost,
                best_cost_so_far: step.best_cost_so_far,
                target_cost,
                improvement_gap: gap,
                improvement_ratio: improvement,
                num_classes: step.num_classes,
            };

            let line = serde_json::to_string(&sample).expect("Failed to serialize sample");
            writeln!(output_file, "{}", line).expect("Failed to write sample");
            total_samples += 1;
        }

        successful += 1;

        if (i + 1) % 10 == 0 {
            println!("  Processed {}/{} expressions, {} samples collected",
                i + 1, args.count, total_samples);
        }
    }

    println!("\nCollected {} samples from {} expressions", total_samples, successful);
    println!("Wrote to: {}", output_path.display());

    // Print statistics
    if successful > 0 {
        println!("\nNext step: Train the Guide with:");
        println!("  cargo run -p pixelflow-pipeline --bin train_guide --release");
    }
}

#[derive(serde::Serialize)]
struct TrainingSampleJson {
    expression_idx: usize,
    step_idx: usize,
    tree_cost: usize,
    best_cost_so_far: usize,
    target_cost: usize,
    improvement_gap: i64,
    improvement_ratio: f64,
    num_classes: usize,
}

/// Convert Expr to ExprTree for search.
fn expr_to_tree(expr: &Expr) -> Option<pixelflow_search::egraph::ExprTree> {
    use pixelflow_search::egraph::{ExprTree, Leaf, ops};

    match expr {
        Expr::Var(idx) => Some(ExprTree::Leaf(Leaf::Var(*idx))),
        Expr::Const(val) => Some(ExprTree::Leaf(Leaf::Const(*val))),

        Expr::Unary(kind, a) => {
            let a_tree = expr_to_tree(a)?;
            let op = op_kind_to_static(*kind)?;
            Some(ExprTree::Op { op, children: vec![a_tree] })
        }

        Expr::Binary(kind, a, b) => {
            let a_tree = expr_to_tree(a)?;
            let b_tree = expr_to_tree(b)?;
            let op = op_kind_to_static(*kind)?;
            Some(ExprTree::Op { op, children: vec![a_tree, b_tree] })
        }

        Expr::Ternary(kind, a, b, c) => {
            let a_tree = expr_to_tree(a)?;
            let b_tree = expr_to_tree(b)?;
            let c_tree = expr_to_tree(c)?;
            let op = op_kind_to_static(*kind)?;
            Some(ExprTree::Op { op, children: vec![a_tree, b_tree, c_tree] })
        }

        Expr::Nary(kind, children) => {
            let child_trees: Option<Vec<_>> = children.iter().map(expr_to_tree).collect();
            let op = op_kind_to_static(*kind)?;
            Some(ExprTree::Op { op, children: child_trees? })
        }
    }
}

/// Convert OpKind to static Op reference.
fn op_kind_to_static(kind: OpKind) -> Option<&'static dyn pixelflow_search::egraph::Op> {
    use pixelflow_search::egraph::ops;

    match kind {
        OpKind::Add => Some(&ops::Add),
        OpKind::Sub => Some(&ops::Sub),
        OpKind::Mul => Some(&ops::Mul),
        OpKind::Div => Some(&ops::Div),
        OpKind::Neg => Some(&ops::Neg),
        OpKind::Sqrt => Some(&ops::Sqrt),
        OpKind::Rsqrt => Some(&ops::Rsqrt),
        OpKind::Abs => Some(&ops::Abs),
        OpKind::Min => Some(&ops::Min),
        OpKind::Max => Some(&ops::Max),
        OpKind::MulAdd => Some(&ops::MulAdd),
        OpKind::Recip => Some(&ops::Recip),
        OpKind::Floor => Some(&ops::Floor),
        OpKind::Ceil => Some(&ops::Ceil),
        OpKind::Round => Some(&ops::Round),
        OpKind::Sin => Some(&ops::Sin),
        OpKind::Cos => Some(&ops::Cos),
        OpKind::Var | OpKind::Const => None,
        _ => None, // Skip unsupported ops
    }
}

/// Find workspace root by looking for Cargo.toml with [workspace]
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
