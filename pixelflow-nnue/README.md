# pixelflow-nnue

Efficiently Updatable Neural Network (NNUE) for expression cost prediction, inspired by Stockfish's approach to chess position evaluation.

## Overview

This crate provides the neural network infrastructure for predicting expression costs without benchmarking. The key insight: just as Stockfish uses NNUE to evaluate chess positions quickly, we can use NNUE to evaluate expression costs for compiler optimization.

| Chess (Stockfish)        | Compiler (PixelFlow)           |
|--------------------------|--------------------------------|
| Position                 | Expression AST / e-class       |
| Legal move               | Valid rewrite rule             |
| Evaluation (centipawns)  | Cost (cycles / nanoseconds)    |
| HalfKP features          | HalfEP features                |
| King position            | Root operation type            |
| Piece placement          | Subexpression structure        |

## HalfEP Feature Set

Inspired by HalfKP (Half-King-Piece), we define HalfEP (Half-Expression-Position):

```
Feature = (perspective_op, descendant_op, relative_depth, child_index)
```

- **perspective_op**: The operation we're evaluating from (like king position)
- **descendant_op**: An operation in the subtree
- **relative_depth**: How deep the descendant is (0-7)
- **child_index**: Left (0) or right (1) branch

This creates a sparse feature vector (~40K possible features) that captures expression structure.

## Usage

### Feature Extraction

```rust
use pixelflow_nnue::{Expr, OpType, extract_features, HalfEPFeature};

// Build an expression: (x + y) * z
let expr = Expr::Binary(
    OpType::Mul,
    Box::new(Expr::Binary(
        OpType::Add,
        Box::new(Expr::Var(0)),  // x
        Box::new(Expr::Var(1)),  // y
    )),
    Box::new(Expr::Var(2)),      // z
);

// Extract sparse features
let features: Vec<HalfEPFeature> = extract_features(&expr);
println!("Feature count: {}", features.len());
```

### Expression Generation (for training)

```rust
use pixelflow_nnue::{ExprGenerator, ExprGenConfig};

let config = ExprGenConfig {
    max_depth: 5,
    leaf_prob: 0.3,
    num_vars: 4,
    include_fused: false,
};

let mut generator = ExprGenerator::new(42, config);
let expr = generator.generate();
```

### NNUE Network

```rust
use pixelflow_nnue::{Nnue, Accumulator};

// Create network with default weights
let nnue = Nnue::new();

// Create accumulator for incremental updates
let mut acc = Accumulator::new();

// Add features for an expression
for feature in &features {
    acc.add_feature(feature.to_index(), &nnue);
}

// Evaluate
let score = nnue.evaluate(&acc);
```

### Incremental Updates

The key efficiency: when applying a rewrite, only features in the affected subtree change:

```rust
// Before rewrite: features_old
// After rewrite:  features_new

// Remove old features
for f in features_old.iter().filter(|f| !features_new.contains(f)) {
    acc.remove_feature(f.to_index(), &nnue);
}

// Add new features
for f in features_new.iter().filter(|f| !features_old.contains(f)) {
    acc.add_feature(f.to_index(), &nnue);
}

// Re-evaluate in O(rewrite_size) instead of O(expression_size)
let new_score = nnue.evaluate(&acc);
```

## Architecture

```
Input: Sparse HalfEP features (~40K possible)
         │
         ▼
┌─────────────────────────┐
│   Feature Transformer   │  768 hidden units
│   (accumulator-based)   │  Incremental updates
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│      Hidden Layer       │  32 units, ClippedReLU
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│        Output           │  1 unit (predicted cost)
└─────────────────────────┘
```

## Features

- `std` (default): Enable standard library features
- `training`: Enable training utilities

## Integration with pixelflow-search

This crate provides the evaluation function; `pixelflow-search` provides the search:

```rust
use pixelflow_nnue::Nnue;
use pixelflow_search::egraph::anytime::{optimize_with_budget, AnytimeConfig};

let nnue = Nnue::load("trained_weights.bin")?;
let result = optimize_with_budget(
    &expr_tree,
    AnytimeConfig::default().with_iterations(1000),
    Some(&nnue),
);
```

## References

- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish) - Original NNUE implementation
- [NNUE Paper](https://arxiv.org/abs/2007.14527) - Efficiently Updatable Neural Networks
