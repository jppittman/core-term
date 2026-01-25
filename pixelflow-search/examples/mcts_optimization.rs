//! MCTS-Guided Expression Optimization
//!
//! This example demonstrates using Monte Carlo Tree Search (MCTS) to find
//! optimal expression rewrites via neural-guided exploration.
//!
//! The algorithm:
//! 1. Start with an unoptimized expression
//! 2. Use MCTS to explore the space of possible rewrites
//! 3. Each "move" re-inserts into e-graph, saturates, and extracts
//! 4. MCTS balances exploration vs exploitation using UCB1
//!
//! Run with: cargo run -p pixelflow-search --example mcts_optimization

use pixelflow_search::egraph::{CostModel, EGraph, ExprTree};
use pixelflow_search::egraph::search_adapter::{
    SearchState, EGraphCategory, CostValuation, insert_tree
};
use pixelflow_search::search::algebra::{Category, Valuation};
use pixelflow_search::search::mcts::{MctsTree, MctsConfig};

fn main() {
    println!("=== MCTS-Guided Expression Optimization ===\n");

    // Create a complex unoptimized expression:
    // ((x * y) + (x * z)) + (-(-(w)))
    // Should simplify to: x * (y + z) + w (via factoring and double negation)
    let unoptimized = ExprTree::Add(
        Box::new(ExprTree::Add(
            Box::new(ExprTree::Mul(
                Box::new(ExprTree::Var(0)), // x
                Box::new(ExprTree::Var(1)), // y
            )),
            Box::new(ExprTree::Mul(
                Box::new(ExprTree::Var(0)), // x
                Box::new(ExprTree::Var(2)), // z
            )),
        )),
        Box::new(ExprTree::Neg(Box::new(ExprTree::Neg(
            Box::new(ExprTree::Var(3)), // w
        )))),
    );

    let costs = CostModel::fully_optimized();
    let initial_state = SearchState::new(unoptimized.clone(), &costs);

    println!("Initial expression: {:?}", unoptimized);
    println!("Initial cost: {}\n", initial_state.cost);

    // Set up MCTS
    let category = EGraphCategory::default();
    let valuation = CostValuation::default();

    // Get available moves
    let moves = category.hom(&initial_state);
    println!("Available rewrite points: {}", moves.len());

    // Configure MCTS
    let mut config = MctsConfig::default();
    config.max_iterations = 50;
    config.exploration_constant = 1.0; // Balance exploration vs exploitation

    // Create MCTS tree
    let mut tree = MctsTree::new(initial_state.clone(), moves.clone(), config);

    // Run MCTS search
    println!("\nRunning MCTS for {} iterations...", tree.config.max_iterations);

    tree.search(
        |state, _mv| {
            // Apply rewrite: re-insert into e-graph, saturate, extract
            let mut eg = EGraph::new();
            let root = insert_tree(&mut eg, &state.tree);
            eg.saturate();
            SearchState::from_egraph(&eg, root, &costs)
        },
        |state| valuation.eval(state),
    );

    println!("Root visits: {}", tree.root.visits);
    println!("Children explored: {}", tree.root.children.len());

    // Get the best result
    if let Some(best_action) = tree.best_action() {
        println!("\nBest action path: {:?}", best_action.path);
    }

    // The actual optimized result comes from the e-graph saturation
    let mut eg = EGraph::new();
    let root = insert_tree(&mut eg, &initial_state.tree);
    eg.saturate();
    let optimized = eg.extract_tree_with_costs(root, &costs);
    let final_state = SearchState::new(optimized.clone(), &costs);

    println!("\n=== Results ===\n");
    println!("Initial cost:   {}", initial_state.cost);
    println!("Optimized cost: {}", final_state.cost);
    println!("Cost reduction: {} ({:.1}%)",
        initial_state.cost as i64 - final_state.cost as i64,
        (1.0 - final_state.cost as f64 / initial_state.cost as f64) * 100.0
    );

    println!("\nOptimized expression: {:?}", optimized);

    // Show MCTS tree statistics
    println!("\n=== MCTS Statistics ===\n");
    println!("Total iterations: {}", tree.root.visits);

    let mut best_children: Vec<_> = tree.root.children.iter()
        .filter(|c| c.visits > 0)
        .collect();
    best_children.sort_by(|a, b| b.visits.cmp(&a.visits));

    println!("Top explored branches:");
    for (i, child) in best_children.iter().take(5).enumerate() {
        let avg_value = if child.visits > 0 {
            child.total_value / child.visits as f64
        } else {
            0.0
        };
        println!(
            "  {}. path={:?}, visits={}, avg_value={:.4}",
            i + 1,
            child.action.as_ref().map(|a| &a.path),
            child.visits,
            avg_value
        );
    }

    // Demonstrate that e-graph finds the optimal regardless of MCTS path
    println!("\n=== E-Graph Saturation ===\n");
    println!("The e-graph finds the optimal form through algebraic saturation,");
    println!("discovering equivalent expressions like:");
    println!("  (x*y + x*z) → x*(y+z)  [factoring]");
    println!("  --w → w               [double negation]");

    // Show rule application stats
    let mut counts: Vec<_> = eg.match_counts.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    println!("\nRule applications:");
    for (rule, count) in counts.iter().take(5) {
        println!("  {}: {} matches", rule, count);
    }
}
