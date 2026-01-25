//! E-Graph Based Training Data Generation
//!
//! This example demonstrates using the full e-graph algebraic system to:
//! 1. Generate expressions
//! 2. Saturate the e-graph with all rewrite rules
//! 3. Extract optimal and suboptimal variants
//! 4. Output training pairs for cost model calibration
//!
//! Run with: cargo run -p pixelflow-search --example egraph_training

use pixelflow_search::egraph::{CostModel, EGraph, ENode};

/// Training pair: (unoptimized expression, optimized expression, cost savings)
#[derive(Debug, Clone)]
struct TrainingPair {
    name: String,
    unopt_nodes: usize,
    opt_nodes: usize,
    unopt_depth: usize,
    opt_depth: usize,
    node_savings: i64,
}

fn main() {
    println!("=== E-Graph Training Data Generator ===\n");

    // Generate training pairs using the e-graph
    let pairs = generate_training_pairs();

    println!("Generated {} training pairs:\n", pairs.len());
    for pair in &pairs {
        println!(
            "{}: {} nodes → {} nodes (saved {} nodes, depth {} → {})",
            pair.name,
            pair.unopt_nodes,
            pair.opt_nodes,
            pair.node_savings,
            pair.unopt_depth,
            pair.opt_depth
        );
    }

    // Print the rule match counts to see which rules are most useful
    println!("\n=== Rule Application Statistics ===\n");

    // Run a test to get rule stats
    let mut eg = EGraph::new();
    let x = eg.add(ENode::Var(0));
    let y = eg.add(ENode::Var(1));
    let z = eg.add(ENode::Var(2));

    // Build: (x * y) + (x * z)  which should factor to x * (y + z)
    let xy = eg.add(ENode::Mul(x, y));
    let xz = eg.add(ENode::Mul(x, z));
    let _sum = eg.add(ENode::Add(xy, xz));

    eg.saturate();

    let mut counts: Vec<_> = eg.match_counts.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    for (rule, count) in counts {
        println!("  {}: {} matches", rule, count);
    }

    // Test different cost models
    println!("\n=== Cost Model Sensitivity ===\n");
    test_cost_model_sensitivity();
}

fn generate_training_pairs() -> Vec<TrainingPair> {
    let mut pairs = Vec::new();

    // 1. FMA Fusion: a*b + c → MulAdd(a, b, c)
    {
        let mut eg = EGraph::new();
        let a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let mul = eg.add(ENode::Mul(a, b));
        let root = eg.add(ENode::Add(mul, c));

        eg.saturate();

        let default_cost = CostModel::default();
        let fma_cost = CostModel::fully_optimized();

        let unopt = eg.extract_tree_with_costs(root, &default_cost);
        let opt = eg.extract_tree_with_costs(root, &fma_cost);

        pairs.push(TrainingPair {
            name: "fma_fusion".to_string(),
            unopt_nodes: unopt.node_count(),
            opt_nodes: opt.node_count(),
            unopt_depth: unopt.depth(),
            opt_depth: opt.depth(),
            node_savings: unopt.node_count() as i64 - opt.node_count() as i64,
        });
    }

    // 2. Distributive: x*(y+z) = x*y + x*z (or factor the reverse)
    {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let xy = eg.add(ENode::Mul(x, y));
        let xz = eg.add(ENode::Mul(x, z));
        let root = eg.add(ENode::Add(xy, xz));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(root, &costs);

        pairs.push(TrainingPair {
            name: "distributive".to_string(),
            unopt_nodes: 5, // x*y + x*z has 5 nodes
            opt_nodes: tree.node_count(),
            unopt_depth: 2,
            opt_depth: tree.depth(),
            node_savings: 5 - tree.node_count() as i64,
        });
    }

    // 3. Rsqrt fusion: 1/sqrt(x) → rsqrt(x)
    {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let sqrt_x = eg.add(ENode::Sqrt(x));
        let root = eg.add(ENode::Recip(sqrt_x));

        eg.saturate();

        let costs = CostModel::fully_optimized();
        let tree = eg.extract_tree_with_costs(root, &costs);

        pairs.push(TrainingPair {
            name: "rsqrt_fusion".to_string(),
            unopt_nodes: 3, // recip(sqrt(x)) has 3 nodes
            opt_nodes: tree.node_count(),
            unopt_depth: 2,
            opt_depth: tree.depth(),
            node_savings: 3 - tree.node_count() as i64,
        });
    }

    // 4. Double negation: --x → x
    {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::Neg(x));
        let root = eg.add(ENode::Neg(neg_x));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(root, &costs);

        pairs.push(TrainingPair {
            name: "double_neg".to_string(),
            unopt_nodes: 3,
            opt_nodes: tree.node_count(),
            unopt_depth: 2,
            opt_depth: tree.depth(),
            node_savings: 3 - tree.node_count() as i64,
        });
    }

    // 5. Identity elimination: x + 0 → x
    {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let root = eg.add(ENode::Add(x, zero));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(root, &costs);

        pairs.push(TrainingPair {
            name: "add_identity".to_string(),
            unopt_nodes: 3,
            opt_nodes: tree.node_count(),
            unopt_depth: 1,
            opt_depth: tree.depth(),
            node_savings: 3 - tree.node_count() as i64,
        });
    }

    // 6. Annihilator: x * 0 → 0
    {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let root = eg.add(ENode::Mul(x, zero));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(root, &costs);

        pairs.push(TrainingPair {
            name: "mul_annihilator".to_string(),
            unopt_nodes: 3,
            opt_nodes: tree.node_count(),
            unopt_depth: 1,
            opt_depth: tree.depth(),
            node_savings: 3 - tree.node_count() as i64,
        });
    }

    // 7. Complex: (x + y) * z + (x + y) * w → (x + y) * (z + w)
    {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let w = eg.add(ENode::Var(3));

        let x_plus_y = eg.add(ENode::Add(x, y));
        let term1 = eg.add(ENode::Mul(x_plus_y, z));
        let term2 = eg.add(ENode::Mul(x_plus_y, w));
        let root = eg.add(ENode::Add(term1, term2));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(root, &costs);

        pairs.push(TrainingPair {
            name: "complex_factor".to_string(),
            unopt_nodes: 8, // Full expanded form
            opt_nodes: tree.node_count(),
            unopt_depth: 3,
            opt_depth: tree.depth(),
            node_savings: 8 - tree.node_count() as i64,
        });
    }

    // 8. Inverse cancellation: (x * a) / a → x
    {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let a = eg.add(ENode::Var(1));
        let xa = eg.add(ENode::Mul(x, a));
        let root = eg.add(ENode::Div(xa, a));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(root, &costs);

        pairs.push(TrainingPair {
            name: "mul_div_cancel".to_string(),
            unopt_nodes: 4,
            opt_nodes: tree.node_count(),
            unopt_depth: 2,
            opt_depth: tree.depth(),
            node_savings: 4 - tree.node_count() as i64,
        });
    }

    pairs
}

fn test_cost_model_sensitivity() {
    // Build a complex expression and see how different cost weights affect extraction
    let mut eg = EGraph::new();
    let x = eg.add(ENode::Var(0));
    let y = eg.add(ENode::Var(1));
    let z = eg.add(ENode::Var(2));

    // Build: x*y + x*z + z (can fuse some parts)
    let xy = eg.add(ENode::Mul(x, y));
    let xz = eg.add(ENode::Mul(x, z));
    let sum1 = eg.add(ENode::Add(xy, xz));
    let root = eg.add(ENode::Add(sum1, z));

    eg.saturate();

    // Test different cost configurations
    let configs: Vec<(&str, CostModel)> = vec![
        ("default", CostModel::default()),
        ("fma_optimized", CostModel::fully_optimized()),
        (
            "expensive_mul",
            CostModel {
                mul: 20,
                mul_add: 25,
                ..CostModel::default()
            },
        ),
        (
            "cheap_fma",
            CostModel {
                mul_add: 3,
                ..CostModel::default()
            },
        ),
    ];

    for (name, costs) in configs {
        let tree = eg.extract_tree_with_costs(root, &costs);
        println!(
            "{:15}: {} nodes, depth {}",
            name,
            tree.node_count(),
            tree.depth()
        );
    }
}
