# pixelflow-search

A graph search framework for algebraic optimization, combining e-graphs with best-first search and neural network evaluation.

## Overview

This crate provides infrastructure for optimizing expressions through:

1. **E-graph equality saturation** - Find all algebraically equivalent forms
2. **Best-first search** - Explore the space of rewrites intelligently (Dijkstra/A*)
3. **Dual-head NNUE** - Predict costs without benchmarking (AlphaZero-style architecture)

The result: compile-time optimization that finds efficient implementations without runtime measurement.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training (offline)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate seed expressions                                   │
│  2. E-graph saturation → find all equivalents                   │
│  3. Benchmark sampled variants (real SIMD costs)                │
│  4. Train DualHeadNnue:                                         │
│     - Value Head (Judge): features(expr) → cost_ns              │
│     - Search Head (Guide): features(expr) → search_priority     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Inference (compile-time)                    │
├─────────────────────────────────────────────────────────────────┤
│  For new kernel K:                                              │
│  1. Insert K into e-graph                                       │
│  2. BestFirstPlanner explores with NNUE-guided search           │
│     - Search head: heap ordering (which states to explore)      │
│     - Value head: cost evaluation (which result is best)        │
│  3. Extract best predicted form                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic E-Graph Optimization

```rust
use pixelflow_search::egraph::{EGraph, ExprTree, Leaf, CostModel, standard_rules};

// Create expression: (x + 0) * 1
let tree = ExprTree::Op {
    op: &pixelflow_search::egraph::ops::Mul,
    children: vec![
        ExprTree::Op {
            op: &pixelflow_search::egraph::ops::Add,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Const(0.0)),
            ],
        },
        ExprTree::Leaf(Leaf::Const(1.0)),
    ],
};

// Insert into e-graph and saturate
let mut eg = EGraph::with_rules(standard_rules());
let root = eg.add_expr(&tree);
eg.saturate();

// Extract optimal form
let costs = CostModel::default();
let (optimized, cost) = eg.extract_best(root, &costs);
// Result: Var(0) - the simplification x + 0 * 1 → x
```

### Best-First Search with NNUE

```rust
use pixelflow_search::egraph::{BestFirstPlanner, BestFirstConfig, standard_rules};
use pixelflow_search::nnue::DualHeadNnue;
use std::path::Path;

// Load trained NNUE (or create with latency prior for initial use)
let nnue = DualHeadNnue::load(Path::new("model.bin"))
    .unwrap_or_else(|_| DualHeadNnue::new_with_latency_prior(42));

// Configure search with chess-style time control
let config = BestFirstConfig::rapid();  // 100-200ms budget

// Create planner with expression tree and rules
let mut planner = BestFirstPlanner::from_tree_with_rules(
    &tree,
    config,
    standard_rules(),
);

// Run with custom evaluator using NNUE
let result = planner.run(|ctx| {
    let expr = pixelflow_search::egraph::expr_tree_to_nnue(ctx.tree);
    (nnue.predict_log_cost(&expr) * 1000.0) as i64
});

println!("Initial cost: {}", result.initial_cost);
println!("Optimized cost: {}", result.best_cost);
```

### Chess-Style Time Control Presets

```rust
use pixelflow_search::egraph::BestFirstConfig;

// Blitz: fast compilation (50-100ms)
let blitz = BestFirstConfig::blitz();

// Rapid: balanced speed/quality (100-200ms, default)
let rapid = BestFirstConfig::rapid();

// Classical: thorough optimization (500ms-1s)
let classical = BestFirstConfig::classical();

// Custom time control
let custom = BestFirstConfig::rapid()
    .with_soft_timeout(Duration::from_millis(150))
    .with_hard_timeout(Duration::from_millis(300))
    .with_max_classes(10_000);
```

### Training Mode with ε-Greedy Exploration

When collecting training data, use exploration to ensure diverse samples:

```rust
use pixelflow_search::egraph::BestFirstConfig;

// Training mode: 20% random exploration
let training_config = BestFirstConfig::rapid()
    .training_mode()           // Sets epsilon=0.2
    .with_max_expansions(500);

// Inference mode: pure exploitation
let inference_config = BestFirstConfig::rapid()
    .inference_mode();         // Sets epsilon=0.0
```

### Collecting Training Data

```rust
use pixelflow_search::egraph::{BestFirstPlanner, BestFirstConfig, standard_rules};
use pixelflow_search::nnue::DualHeadNnue;

let nnue = DualHeadNnue::load(Path::new("judge.bin"))?;

let mut planner = BestFirstPlanner::from_tree_with_rules(
    &tree,
    BestFirstConfig::rapid(),
    standard_rules(),
);

// Run search while recording trajectory for training
let (result, trajectory) = planner.run_recording(|ctx| {
    let expr = pixelflow_search::egraph::expr_tree_to_nnue(ctx.tree);
    (nnue.predict_log_cost(&expr) * 1000.0) as i64
});

// Use trajectory to train the search head
for step in &trajectory.steps {
    let gap = step.tree_cost as i64 - result.best_cost as i64;
    // gap > 0 means room for improvement from this state
}
```

## Modules

| Module | Purpose |
|--------|---------|
| `egraph` | E-graph implementation with equality saturation |
| `egraph::best_first` | Chess-style time-controlled best-first search |
| `egraph::nnue_adapter` | DualHeadNnue integration for cost/priority prediction |
| `egraph::codegen` | ExprTree → kernel! macro code generation |
| `egraph::rules` | Rewrite rules (Identity, FMA fusion, etc.) |
| `egraph::saturate` | Budget-limited saturation |
| `nnue` | Dual-head NNUE (value head + search head) |
| `nnue::factored` | O(ops) factored embedding architecture |
| `training` | Training infrastructure (feature-gated) |

## NNUE Architecture

The dual-head NNUE is inspired by AlphaZero:

```
Input: Expression features (O(ops) factored embeddings)
       │
       ▼
┌──────────────────────┐
│   Shared Layers      │  OpEmbeddings + EdgeAccumulator
│   (5,504 params)     │
└──────────────────────┘
       │
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌──────┐
│Value │ │Search│
│ Head │ │ Head │
│(65p) │ │(65p) │
└──────┘ └──────┘
   │       │
   ▼       ▼
 cost    priority
 (ns)    (heap order)
```

**Value Head (The Judge)**: Predicts runtime cost in nanoseconds
**Search Head (The Guide)**: Predicts which states are worth exploring

## Features

- `std` (default): Enable standard library features
- `training`: Enable training utilities (data generation, backprop)
- `nnue`: Enable NNUE integration for best-first search

## Training Pipeline

```bash
# 1. Generate variants and benchmark code
cargo run -p pixelflow-compiler --example gen_egraph_variants --features training

# 2. Collect benchmark costs
cargo run -p pixelflow-compiler --example collect_benchmark_costs --features training

# 3. Train the Judge (value head)
cargo run -p pixelflow-compiler --example train_with_validation --features training

# 4. Collect search training data using the Judge
cargo run -p pixelflow-compiler --example collect_search_training --features training

# 5. Train the Guide (search head)
cargo run -p pixelflow-compiler --example train_search_head --features training
```

## Integration

This crate works with:

- **pixelflow-ir**: Shared IR types (Expr, OpKind)
- **pixelflow-compiler**: Compile-time kernel optimization

## References

- [egg: E-Graphs Good](https://egraphs-good.github.io/) - E-graph background
- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish) - NNUE architecture inspiration
- [AlphaZero](https://arxiv.org/abs/1712.01815) - Dual-head value/policy network
