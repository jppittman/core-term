//! Generate mask training data by running rule matching on expressions.
//!
//! This takes expressions from judge_training.jsonl and tests which rules
//! actually fire on each expression, creating training data for the mask head.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin gen_mask_data --release --features training
//! ```

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_search::egraph::{all_rules, EClassId, EGraph, ExprTree, Leaf, Rewrite, ops};
use pixelflow_search::nnue::{RuleFeatures, RULE_FEATURE_DIM};
use pixelflow_ir::Expr;
use pixelflow_pipeline::training::factored::parse_kernel_code;

/// Generate mask training data.
#[derive(Parser, Debug)]
#[command(name = "gen_mask_data")]
#[command(about = "Generate mask training data from expressions")]
struct Args {
    /// Path to judge training data (has expressions)
    #[arg(long, default_value = "pixelflow-pipeline/data/judge_training.jsonl")]
    input: String,

    /// Output path for mask training data
    #[arg(short, long, default_value = "pixelflow-pipeline/data/mask_training.jsonl")]
    output: String,

    /// Maximum expressions to process (0 = all)
    #[arg(long, default_value_t = 0)]
    max_expr: usize,

    /// Sample rate for negative examples (0.0-1.0)
    /// Since most rules don't fire, we downsample negatives
    #[arg(long, default_value_t = 0.1)]
    neg_sample_rate: f32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Input sample from judge_training.jsonl
#[derive(Debug, Deserialize)]
struct JudgeSample {
    name: String,
    expression: String,
    #[allow(dead_code)]
    timing_ns: f64,
}

/// Output sample for mask training
#[derive(Debug, Serialize)]
struct MaskSample {
    /// Expression in kernel code syntax
    expression: String,
    /// Rule index
    rule_idx: usize,
    /// Rule name (for debugging)
    rule_name: String,
    /// Hand-crafted rule features
    rule_features: [f32; RULE_FEATURE_DIM],
    /// Whether the rule fired (ground truth)
    fired: bool,
}

fn main() {
    let args = Args::parse();
    let workspace_root = find_workspace_root();

    // Load rules
    let rules = all_rules();
    let num_rules = rules.len();
    println!("Loaded {} rewrite rules", num_rules);

    // Build rule features
    let rule_features = build_rule_features(&rules);

    // Load expressions
    let input_path = workspace_root.join(&args.input);
    println!("Loading expressions from: {}", input_path.display());

    let input_file = File::open(&input_path)
        .expect("Failed to open judge_training.jsonl");
    let reader = BufReader::new(input_file);

    let mut expressions: Vec<JudgeSample> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let sample: JudgeSample = serde_json::from_str(&line)
            .expect("Failed to parse judge sample");
        expressions.push(sample);

        if args.max_expr > 0 && expressions.len() >= args.max_expr {
            break;
        }
    }
    println!("Loaded {} expressions", expressions.len());

    // Setup output
    let output_path = workspace_root.join(&args.output);
    fs::create_dir_all(output_path.parent().unwrap()).ok();
    let mut output_file = File::create(&output_path)
        .expect("Failed to create output file");

    // RNG for negative sampling
    let mut rng_state = args.seed.wrapping_add(1);
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f32 / (1u64 << 31) as f32
    };

    // Stats
    let mut total_positive = 0usize;
    let mut total_negative = 0usize;
    let mut total_sampled_negative = 0usize;

    // Process each expression
    for (idx, sample) in expressions.iter().enumerate() {
        if idx % 500 == 0 {
            println!("Processing expression {}/{}", idx, expressions.len());
        }

        // Parse expression
        let expr = match parse_kernel_code(&sample.expression) {
            Some(e) => e,
            None => continue, // Skip unparseable expressions
        };

        // Convert to ExprTree for e-graph
        let tree = expr_to_tree(&expr);

        // Create single-node e-graph with this expression
        let mut egraph = EGraph::new();
        let root = egraph.add_expr(&tree);

        // Test each rule
        for (rule_idx, rule) in rules.iter().enumerate() {
            // Check if rule fires on this expression
            let fired = rule_fires_on_egraph(&egraph, root, rule.as_ref());

            if fired {
                total_positive += 1;
                // Always include positives
                let mask_sample = MaskSample {
                    expression: sample.expression.clone(),
                    rule_idx,
                    rule_name: rule.as_ref().name().to_string(),
                    rule_features: rule_features.features[rule_idx],
                    fired: true,
                };
                let json = serde_json::to_string(&mask_sample).unwrap();
                writeln!(output_file, "{}", json).unwrap();
            } else {
                total_negative += 1;
                // Downsample negatives
                if next_rand() < args.neg_sample_rate {
                    total_sampled_negative += 1;
                    let mask_sample = MaskSample {
                        expression: sample.expression.clone(),
                        rule_idx,
                        rule_name: rule.as_ref().name().to_string(),
                        rule_features: rule_features.features[rule_idx],
                        fired: false,
                    };
                    let json = serde_json::to_string(&mask_sample).unwrap();
                    writeln!(output_file, "{}", json).unwrap();
                }
            }
        }
    }

    println!("\n=== Statistics ===");
    println!("Total positives (rules fired): {}", total_positive);
    println!("Total negatives (rules didn't fire): {}", total_negative);
    println!("Sampled negatives ({}%): {}", args.neg_sample_rate * 100.0, total_sampled_negative);
    println!("Total samples written: {}", total_positive + total_sampled_negative);
    println!("Positive ratio: {:.1}%",
        total_positive as f32 / (total_positive + total_sampled_negative) as f32 * 100.0);
    println!("\nSaved to: {}", output_path.display());
}

/// Check if a rule fires on an e-graph starting from the root.
/// For a single expression, we just need to check if the rule matches the root's nodes.
fn rule_fires_on_egraph(
    egraph: &EGraph,
    root: EClassId,
    rule: &dyn Rewrite,
) -> bool {
    // Check if rule matches any node in the root e-class
    let nodes = egraph.nodes(root);
    for node in nodes {
        if rule.apply(egraph, root, node).is_some() {
            return true;
        }
    }
    false
}

/// Build hand-crafted rule features for all rules.
fn build_rule_features(rules: &[Box<dyn Rewrite>]) -> RuleFeatures {
    let mut features = RuleFeatures::new();

    for (idx, rule) in rules.iter().enumerate() {
        let name = rule.as_ref().name();

        // Category (algebraic=0.0, peephole=0.25, domain=0.5, cross-cutting=0.75)
        let category = if name.contains("assoc") || name.contains("commute") || name.contains("distrib") {
            0.0 // algebraic
        } else if name.contains("fma") || name.contains("rsqrt") || name.contains("recip") {
            0.25 // peephole
        } else if name.contains("trig") || name.contains("log") || name.contains("exp") {
            0.5 // domain-specific
        } else {
            0.75 // cross-cutting
        };

        // LHS complexity (estimate from name length as proxy)
        let lhs_nodes = (name.len() / 5).min(10) as usize;

        // Depth delta (most rules are depth-neutral)
        let depth_delta = if name.contains("flatten") { -1i8 } else { 0i8 };

        // Properties from name
        let commutative = name.contains("commute");
        let associative = name.contains("assoc");
        let creates_sharing = name.contains("cse") || name.contains("share");
        let expensive_op = name.contains("div") || name.contains("sqrt") || name.contains("trig");

        features.set_rule(
            idx,
            category,
            lhs_nodes,
            depth_delta,
            commutative,
            associative,
            creates_sharing,
            0.5, // Default match rate (will be updated during training)
            expensive_op,
        );
    }

    features
}

/// Convert pixelflow_ir::Expr to egraph::extract::ExprTree.
fn expr_to_tree(expr: &Expr) -> ExprTree {
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

/// Find workspace root.
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
