//! Analysis of ILP feature value for instruction selection
//!
//! Run with: cargo run -p pixelflow-ml --example ilp_analysis

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use pixelflow_ml::evaluator::{LinearFeatures, default_expr_weights, extract_expr_features};
use pixelflow_ml::nnue::{Expr, ExprGenConfig, ExprGenerator, OpType};
use pixelflow_ml::nonlinear_eval::{
    critical_path_cost, interaction_cost, linear_cost, ranking_correlation, total_cost,
};

fn main() {
    let config = ExprGenConfig {
        max_depth: 5,
        leaf_prob: 0.3,
        num_vars: 4,
        include_fused: false,
    };
    let mut generator = ExprGenerator::new(42, config);

    let mut exprs: Vec<Expr> = Vec::with_capacity(100);
    for _ in 0..100 {
        exprs.push(generator.generate());
    }

    // Compare linear vs interaction
    let linear_vs_interaction = ranking_correlation(
        &exprs,
        |e| {
            let f = extract_expr_features(e);
            linear_cost(&f)
        },
        |e| {
            let f = extract_expr_features(e);
            interaction_cost(&f)
        },
    );

    // Compare total vs critical path
    let total_vs_critical = ranking_correlation(&exprs, total_cost, critical_path_cost);

    // Compare HCE with vs without ILP features
    let hce = default_expr_weights();

    // HCE with ILP features (full model)
    let hce_full = |e: &Expr| -> i32 {
        let f = extract_expr_features(e);
        hce.evaluate_linear(&f)
    };

    // HCE without ILP (simulate by zeroing out critical_path and max_width)
    let hce_no_ilp = |e: &Expr| -> i32 {
        let f = extract_expr_features(e);
        // Manual calculation without ILP features (indices 19 and 20)
        let mut score = 0i32;
        for i in 0..19 {
            // Only first 19 features
            score = score.saturating_add(hce.get_weight(i).saturating_mul(f.get(i)));
        }
        score
    };

    let hce_with_vs_without_ilp = ranking_correlation(&exprs, hce_full, hce_no_ilp);

    println!("=== ILP Feature Value Analysis ===\n");
    println!("Expressions analyzed: {}", exprs.len());
    println!();
    println!("Ranking Correlations (1.0 = identical rankings):");
    println!("  Linear vs Interaction:     {:.3}", linear_vs_interaction);
    println!("  Total vs Critical Path:    {:.3}", total_vs_critical);
    println!(
        "  HCE+ILP vs HCE (no ILP):   {:.3}",
        hce_with_vs_without_ilp
    );
    println!();

    // Count how many pairs are ranked differently
    let mut different_rankings = 0;
    let mut total_pairs = 0;
    for i in 0..exprs.len() {
        for j in (i + 1)..exprs.len() {
            let total_i = total_cost(&exprs[i]);
            let total_j = total_cost(&exprs[j]);
            let crit_i = critical_path_cost(&exprs[i]);
            let crit_j = critical_path_cost(&exprs[j]);

            let total_prefers_i = total_i < total_j;
            let crit_prefers_i = crit_i < crit_j;

            if total_i != total_j && crit_i != crit_j && total_prefers_i != crit_prefers_i {
                different_rankings += 1;
            }
            total_pairs += 1;
        }
    }

    println!(
        "Pairs where critical_path disagrees with total: {}/{} ({:.1}%)",
        different_rankings,
        total_pairs,
        100.0 * different_rankings as f64 / total_pairs as f64
    );

    // Show a concrete example
    println!("\n=== Concrete Example ===");
    // Wide: (a+b)+(c+d)
    let wide = Expr::Binary(
        OpType::Add,
        Box::new(Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        )),
        Box::new(Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(2)),
            Box::new(Expr::Var(3)),
        )),
    );
    // Deep: ((a+b)+c)+d
    let deep = Expr::Binary(
        OpType::Add,
        Box::new(Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        )),
        Box::new(Expr::Var(3)),
    );

    let wide_f = extract_expr_features(&wide);
    let deep_f = extract_expr_features(&deep);

    println!("Wide (a+b)+(c+d):");
    println!("  Total ops cost: {}", total_cost(&wide));
    println!("  Critical path:  {}", critical_path_cost(&wide));
    println!("  max_width:      {}", wide_f.max_width);
    println!("  HCE cost:       {}", hce.evaluate_linear(&wide_f));

    println!("Deep ((a+b)+c)+d:");
    println!("  Total ops cost: {}", total_cost(&deep));
    println!("  Critical path:  {}", critical_path_cost(&deep));
    println!("  max_width:      {}", deep_f.max_width);
    println!("  HCE cost:       {}", hce.evaluate_linear(&deep_f));

    println!(
        "\nConclusion: Wide is {} by HCE (shorter critical path wins)",
        if hce.evaluate_linear(&wide_f) < hce.evaluate_linear(&deep_f) {
            "PREFERRED"
        } else {
            "NOT preferred"
        }
    );
}
