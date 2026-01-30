# NNUE Integration Status Analysis

## Executive Summary

**Question:** Is the elaborate NNUE-based cost model with "tons of features" integrated into the compiler pipeline?

**Answer:** **NO.** The NNUE infrastructure is comprehensively implemented but not wired into the actual cost evaluation during E-graph extraction.

## What EXISTS (Comprehensive Implementation)

### 1. Full NNUE Network (`pixelflow-ml/src/nnue.rs`)

**Architecture:** 256-32-32-1 neural network inspired by Stockfish
- **Feature Set:** HalfEP (Half-Expression-Position) with 401,408 possible features
  - Analogous to chess HalfKP (Half-King-Piece)
  - Encodes: `(perspective_op, descendant_op, depth, path)`
- **Incremental Updates:** Accumulator-based for O(move_size) feature updates
- **Training Data Generation:**
  - Forward generation: random expressions + enumerate rewrites
  - Backward generation (BWD): following Lample & Charton 2019
  - Unfusing rewrites to create training pairs

### 2. Evaluator Framework (`pixelflow-ml/src/evaluator.rs`)

**Hand-Crafted Evaluator (HCE):** Linear feature-based evaluation
- **21 Features** including:
  - Operation counts (add, mul, div, sqrt, etc.)
  - Fused operation counts (FMA, mul_rsqrt)
  - Structural features (node_count, depth)
  - **ILP-aware features:**
    - `critical_path`: Longest dependency chain (not sum of ops!)
    - `max_width`: Approximates register pressure
  - Pattern detection (identity, self-cancel, fusable)

**Domain Abstraction:** Clean separation of logic vs. physics
- `Domain` trait: Position, Move, Features
- `Evaluator` trait: Features → Score

### 3. NNUE Adapter (`pixelflow-search/src/egraph/nnue_adapter.rs`)

Converts between E-graph types and NNUE types:
- `expr_tree_to_nnue()`: ExprTree → NNUE Expr
- `eclass_to_expr()`: EClassId → NNUE Expr
- `expr_to_egraph()`: NNUE Expr → E-graph

### 4. References Throughout Codebase

Multiple files reference NNUE integration:
- `guided.rs`: "NNUE-driven optimization"
- `best_first.rs`: "NNUE as the heuristic"
- `saturate.rs`: "NNUE to predict extraction costs"
- README: "Train NNUE: features(expr) → actual_cost_ns"

## What is MISSING (The Integration Gap)

### Current Extraction Uses Simple Weights

**File:** `pixelflow-search/src/egraph/extract.rs:137-146`

```rust
pub fn cost(&self, costs: &CostModel) -> usize {
    match self {
        Self::Leaf(_) => 0,
        Self::Op { op, children } => {
            let op_cost = costs.cost_by_name(op.name());  // ❌ Simple lookup
            let children_cost: usize = children.iter().map(|c| c.cost(costs)).sum();
            op_cost + children_cost
        }
    }
}
```

**Current CostModel:** `pixelflow-search/src/egraph/cost.rs`

```rust
pub struct CostModel {
    pub add: usize,
    pub sub: usize,
    pub mul: usize,
    // ... just integer weights
}
```

### What SHOULD Happen

```rust
// IDEAL: NNUE-based cost evaluation
pub fn cost_with_nnue(&self, nnue: &Nnue) -> usize {
    // 1. Extract features from expression tree
    let features = extract_halfep_features(self);

    // 2. Create accumulator and add features
    let mut acc = Accumulator::new(nnue);
    for feature in features {
        acc.add_feature(nnue, feature.to_index());
    }

    // 3. Forward pass through network
    let predicted_cost = acc.forward(nnue);

    predicted_cost as usize
}
```

## Architecture Mismatch

### The Problem

The extraction algorithm expects a simple `cost: usize` per node. But NNUE evaluation requires:
1. **Full expression context** (not just node-local information)
2. **Feature extraction** from the subtree
3. **Neural network forward pass**

### Current Architecture

```
ExprTree::cost(&CostModel)
    └─> Simple sum: op_cost + sum(children_costs)
```

### Desired Architecture

```
ExprTree::cost_nnue(&Nnue)
    ├─> Extract HalfEP features from subtree
    ├─> Accumulator incremental update
    └─> Neural network forward pass → predicted cost
```

## Integration Options

### Option 1: Replace CostModel with NNUE

**Pros:**
- Uses full NNUE capability
- Learns from actual benchmarks
- Captures non-linear interactions

**Cons:**
- Requires trained model (where is the training data?)
- Slower evaluation (neural network vs. simple lookup)
- Needs feature extraction for every subtree during extraction

### Option 2: Use HCE Linear Features as Bridge

**Pros:**
- 21 features already include ILP-aware metrics (critical_path, max_width)
- Fast evaluation (just dot product)
- Debuggable (can see feature contributions)
- Can be tuned from benchmark data

**Cons:**
- Linear model can't capture complex interactions
- Still simpler than full NNUE

### Option 3: Hybrid Approach

**Phase 1 (Cheap):** Use HCE during E-graph saturation
**Phase 2 (Expensive):** Use NNUE for final extraction

This is similar to chess engines:
- Fast heuristic for search
- Slow NNUE for evaluation at leaves

## Missing Pieces

### 1. Trained NNUE Model

- **Training data generation:** Code exists (`nnue_trainer.rs`, `BwdGenerator`)
- **Actual trained model:** WHERE IS IT?
- **Model serialization:** How to load/save weights?

### 2. Integration Points

**Where to wire NNUE:**
- `extract()` function in `extract.rs`
- Replace or augment `CostModel` struct
- Add `CostStrategy` enum: `Simple | HCE | NNUE`

### 3. Feature Extraction from E-graph

Current adapter converts between types, but doesn't extract features during extraction:
- Need `extract_features_from_tree(tree: &ExprTree) -> Vec<HalfEPFeature>`
- Or use HCE: `extract_expr_features(expr: &Expr) -> ExprFeatures`

## Recommendation

**Immediate Action:** Wire up HCE (Hand-Crafted Evaluator) first
1. Already has 21 features including ILP-aware metrics
2. Fast evaluation (linear combination)
3. No training data required
4. Proves the integration works

**Future Work:** Full NNUE integration
1. Generate training data (random expressions + benchmarks)
2. Train NNUE model (backward generation for diverse data)
3. Serialize/deserialize model
4. Replace HCE with NNUE in extraction

## Code Changes Required

### Step 1: Add HCE to CostModel

```rust
// In pixelflow-search/src/egraph/cost.rs
use pixelflow_ml::evaluator::HandCraftedEvaluator;

pub enum CostStrategy {
    Simple(SimpleCosts),
    HCE(HandCraftedEvaluator),
}

pub struct CostModel {
    strategy: CostStrategy,
}

impl CostModel {
    pub fn evaluate_tree(&self, tree: &ExprTree) -> usize {
        match &self.strategy {
            CostStrategy::Simple(costs) => tree.cost_simple(costs),
            CostStrategy::HCE(hce) => {
                let nnue_expr = expr_tree_to_nnue(tree);
                let features = extract_expr_features(&nnue_expr);
                hce.evaluate(&features) as usize
            }
        }
    }
}
```

### Step 2: Update Extraction

```rust
// In pixelflow-search/src/egraph/extract.rs
pub fn extract_with_strategy(
    egraph: &EGraph,
    root: EClassId,
    costs: &CostModel
) -> (ExprTree, usize) {
    // ... existing extraction logic ...
    // Replace simple cost calculation with:
    let node_cost = costs.evaluate_tree(&candidate_tree);
}
```

## Conclusion

The NNUE infrastructure is **remarkably complete** but **completely disconnected** from the actual cost evaluation during E-graph extraction.

**Current state:** Simple weight lookup (add=4, mul=5, etc.)
**Available but unused:** 401k-feature NNUE network + 21-feature HCE
**Next step:** Wire up HCE first, then NNUE with trained model

The system is like having a Formula 1 car (NNUE) sitting in the garage while driving a bicycle (simple weights) to work.
