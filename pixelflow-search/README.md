# pixelflow-search

A graph search framework for algebraic optimization, combining e-graphs with MCTS and neural network evaluation.

## Overview

This crate provides infrastructure for optimizing expressions through:

1. **E-graph equality saturation** - Find all algebraically equivalent forms
2. **MCTS-guided search** - Explore the space of rewrites intelligently
3. **NNUE evaluation** - Predict costs without benchmarking

The result: compile-time optimization that finds efficient implementations without runtime measurement.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training (offline)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate seed expressions                                   │
│  2. E-graph saturation → find all equivalents                   │
│  3. Benchmark sampled variants (real SIMD costs)                │
│  4. Train NNUE: features(expr) → actual_cost_ns                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Inference (compile-time)                    │
├─────────────────────────────────────────────────────────────────┤
│  For new kernel K:                                              │
│  1. Insert K into e-graph, saturate                             │
│  2. MCTS explores equivalent forms                              │
│  3. NNUE predicts cost (no benchmark needed!)                   │
│  4. Extract best predicted form                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic E-Graph Optimization

```rust
use pixelflow_search::egraph::{EGraph, ExprTree, CostModel};
use pixelflow_search::egraph::search_adapter::insert_tree;

// Create expression: (x + 0) * 1
let tree = ExprTree::op_mul(
    ExprTree::op_add(
        ExprTree::var(0),
        ExprTree::constant(0.0),
    ),
    ExprTree::constant(1.0),
);

// Insert into e-graph and saturate
let mut eg = EGraph::new();
let root = insert_tree(&mut eg, &tree);
eg.saturate();

// Extract optimal form
let costs = CostModel::fully_optimized();
let optimized = eg.extract_tree_with_costs(root, &costs);
// Result: Var(0) - the simplification x + 0 * 1 → x
```

### Anytime Optimization with MCTS

```rust
use pixelflow_search::egraph::anytime::{optimize_with_budget, AnytimeConfig};
use pixelflow_nnue::Nnue;
use std::time::Duration;

// Load trained NNUE (or use None for cost-based search)
let nnue = Nnue::load("trained_weights.bin").ok();

// Configure search
let config = AnytimeConfig::default()
    .with_iterations(1000)           // Max MCTS iterations
    .with_timeout(Duration::from_secs(5))  // Time limit
    .with_exploration(1.414);        // UCB1 exploration constant

// Optimize
let result = optimize_with_budget(&tree, config, nnue.as_ref());

println!("Original cost: {}", result.original_cost);
println!("Optimized cost: {}", result.cost);
println!("Improvement: {:.1}x", result.improvement_ratio());
```

### Training Mode with ε-Greedy Exploration

When collecting training data, use exploration to ensure diverse samples:

```rust
use pixelflow_search::egraph::anytime::AnytimeConfig;

// Training mode: 10% random exploration
let training_config = AnytimeConfig::default()
    .training_mode()           // Sets epsilon=0.1
    .with_seed(42)             // Reproducible
    .with_iterations(500);

// Inference mode: pure exploitation (default)
let inference_config = AnytimeConfig::default()
    .inference_mode()          // Sets epsilon=0.0
    .with_iterations(1000);
```

### Custom ε-Greedy

```rust
let config = AnytimeConfig::default()
    .with_epsilon(0.15)        // 15% random exploration
    .with_seed(12345);         // Reproducible randomness
```

### Beam Search (Faster, Less Optimal)

```rust
let config = AnytimeConfig::default()
    .with_beam_search(10)      // Keep top 10 candidates
    .with_iterations(100);

let result = optimize_with_budget(&tree, config, None);
```

### Extract Multiple Variants

For training data generation, extract diverse equivalent forms:

```rust
use pixelflow_search::egraph::{EGraph, CostModel};

let mut eg = EGraph::new();
let root = insert_tree(&mut eg, &tree);
eg.saturate_with_limit(50);  // Budget-limited saturation

let costs = CostModel::fully_optimized();
let variants = eg.extract_variants(root, 5, &costs);  // Get 5 different forms

for (i, variant) in variants.iter().enumerate() {
    println!("Variant {}: {} nodes, cost {}",
             i, variant.node_count(), variant.cost(&costs));
}
```

## Exploration Strategies

### UCB1 (Built into MCTS)

The MCTS implementation uses UCB1 for node selection:

```
UCB1 = Q(node)/N(node) + C × √(ln(N(parent)) / N(node))
       ─────────────────   ─────────────────────────────
          exploitation              exploration
```

- **exploration_constant** (C): Higher = more exploration (default: √2)
- Unvisited nodes always selected first (score = ∞)

### ε-Greedy (Training Data Collection)

Prevents "garbage in, garbage out" when training on search results:

| Mode | Epsilon | Use Case |
|------|---------|----------|
| `inference_mode()` | 0.0 | Compile-time optimization |
| `training_mode()` | 0.1 | Collecting diverse training samples |
| `with_epsilon(0.2)` | 0.2 | Heavy exploration for initial training |

## Modules

| Module | Purpose |
|--------|---------|
| `egraph` | E-graph implementation with equality saturation |
| `egraph::anytime` | Budget-constrained optimization API |
| `egraph::codegen` | ExprTree → kernel! macro code generation |
| `egraph::search_adapter` | MCTS/NNUE integration |
| `egraph::saturate` | Budget-limited saturation |
| `search::mcts` | Monte Carlo Tree Search implementation |
| `search::algebra` | Category theory abstractions (Morphism, Valuation) |
| `eval` | Training data generation |
| `features` | HalfEP feature extraction |

## Training Pipeline

Generate training data using the companion `pixelflow-ml` crate:

```bash
# 1. Generate variants and benchmark code
cargo run -p pixelflow-ml --example gen_egraph_variants \
    --features training -- --count 100 --variants 5

# 2. Run benchmarks (measures real SIMD costs)
cargo bench -p pixelflow-ml --bench generated_kernels

# 3. Collect results into training data
cargo run -p pixelflow-ml --example collect_benchmark_costs \
    --features training

# 4. Train NNUE
cargo run -p pixelflow-ml --example train_nnue_stockfish \
    --features training
```

## Features

- `std` (default): Enable standard library features
- `training`: Enable training utilities

## Integration

This crate works with:

- **pixelflow-nnue**: Neural network for cost prediction
- **pixelflow-ml**: Training data generation and model training
- **pixelflow-macros**: Compile-time kernel optimization

## References

- [egg: E-Graphs Good](https://egraphs-good.github.io/) - E-graph background
- [Stockfish MCTS](https://github.com/official-stockfish/Stockfish) - Search inspiration
- [AlphaZero](https://arxiv.org/abs/1712.01815) - MCTS + neural evaluation
