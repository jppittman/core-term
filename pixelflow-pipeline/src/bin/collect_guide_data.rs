//! Collect match prediction training data for Guide.
//!
//! This binary runs egg-style saturation and records ground truth for
//! whether each rule matched at each epoch.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin collect_guide_data --release --features training
//! ```
//!
//! # Output
//!
//! - `pixelflow-pipeline/data/guide_training.jsonl` - Match prediction data

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use pixelflow_search::egraph::{
    CostModel, EGraph, ExprTree, all_rules, guided_search::collect_match_training_data,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator, Expr, OpKind, guide::RULE_FEATURE_COUNT};
use serde::Serialize;

/// Collect match prediction data for Guide training.
#[derive(Parser, Debug)]
#[command(name = "collect_guide_data")]
#[command(about = "Collect match prediction data for Guide training")]
struct Args {
    /// Number of expressions to generate
    #[arg(short, long, default_value_t = 100)]
    count: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Output path
    #[arg(short, long, default_value = "pixelflow-pipeline/data/guide_training.jsonl")]
    output: String,

    /// Maximum epochs per expression
    #[arg(long, default_value_t = 50)]
    max_epochs: usize,
}

/// Training sample for Guide.
#[derive(Serialize)]
struct MatchSample {
    /// Expression identifier
    expression_id: String,
    /// Epoch when this sample was recorded
    epoch: usize,
    /// Rule index
    rule_idx: usize,
    /// Feature vector (normalized)
    features: [f32; RULE_FEATURE_COUNT],
    /// Ground truth: did the rule match?
    matched: bool,
}

fn main() {
    let args = Args::parse();

    // Find workspace root
    let workspace_root = find_workspace_root();

    println!("Collecting Guide training data...");
    println!("  Expressions: {}", args.count);
    println!("  Max epochs: {}", args.max_epochs);
    println!("  Seed: {}", args.seed);

    // Configure expression generator with moderate complexity
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
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create output directory");
    }
    let file = File::create(&output_path).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let costs = CostModel::default();

    let mut total_samples = 0;
    let mut total_matches = 0;

    for i in 0..args.count {
        // Generate expression (use unoptimized form for more interesting e-graph exploration)
        let pair = expr_gen.generate();
        let expr_tree = expr_to_tree(&pair.unoptimized);

        // Create e-graph with rules
        let mut egraph = EGraph::with_rules(all_rules());
        let root = egraph.add_expr(&expr_tree);

        // Collect match data
        let trajectory = collect_match_training_data(&mut egraph, root, &costs, args.max_epochs);

        // Write samples
        for epoch_record in &trajectory {
            for rule_record in &epoch_record.rule_records {
                let features = rule_record.features.to_array();
                let sample = MatchSample {
                    expression_id: format!("expr_{:04}", i),
                    epoch: epoch_record.epoch,
                    rule_idx: rule_record.rule_idx,
                    features,
                    matched: rule_record.matched,
                };

                let line = serde_json::to_string(&sample).expect("Failed to serialize");
                writeln!(writer, "{}", line).expect("Failed to write");

                total_samples += 1;
                if rule_record.matched {
                    total_matches += 1;
                }
            }
        }

        if (i + 1) % 10 == 0 {
            println!("  Processed {}/{} expressions", i + 1, args.count);
        }
    }

    writer.flush().expect("Failed to flush");

    let match_rate = total_matches as f32 / total_samples as f32;
    println!("\nCollection complete!");
    println!("  Total samples: {}", total_samples);
    println!("  Total matches: {}", total_matches);
    println!("  Match rate: {:.1}%", match_rate * 100.0);
    println!("  Output: {}", output_path.display());
}

/// Convert pixelflow_ir::Expr to ExprTree for e-graph insertion.
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
        // Leaves don't have ops
        OpKind::Var | OpKind::Const => panic!("Var/Const should not need op conversion"),
    }
}

/// Find the workspace root by looking for Cargo.toml.
fn find_workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().expect("Failed to get current directory");
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            // Check if this is the workspace root (contains [workspace])
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return dir;
                }
            }
        }
        if !dir.pop() {
            // Fall back to current directory
            return std::env::current_dir().expect("Failed to get current directory");
        }
    }
}
