# NNUE Training Pipeline Design

## Metadata
- **Author**: jppittman + claude
- **Status**: REVISED - Curriculum Bootstrapping (Simplified)
- **Created**: 2026-01-24
- **Updated**: 2026-01-25
- **Reviewers**: -
- **Key Decision**: Replace MCTS with Curriculum Learning + Best-First Search

---

## 1. Overview

### 1.1 Problem Statement
Train NNUE to predict expression costs via **Curriculum Learning**, not complex RL.

### 1.2 The Pragmatic Pivot: Why Not MCTS?

MCTS adds complexity (UCB1, visit counts, tree backpropagation, Policy+Value heads) that's unnecessary when we have:
1. **Ground truth from saturation** (for small kernels)
2. **A simple priority queue with ε-greedy exploration**

**Delete the RL magic. Replace with standard engineering.**

### 1.3 Architecture: Curriculum Bootstrapping

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: KINDERGARTEN (Saturation = Ground Truth)              │
├─────────────────────────────────────────────────────────────────┤
│  • Generate TINY kernels (10-50 nodes)                          │
│  • SATURATE completely (brute force - affordable for small)     │
│  • Get PERFECT optimal cost                                     │
│  • Train NNUE: "This graph has potential cost X"                │
│  • Result: Network learns basic algebra (x*0=0, x+0=x, etc.)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: UNIVERSITY (Guided Search + Noise)                    │
├─────────────────────────────────────────────────────────────────┤
│  • Generate HUGE kernels (1000+ nodes)                          │
│  • Saturation IMPOSSIBLE                                        │
│  • Use Best-First Search (A*) with trained NNUE                 │
│  • ε-greedy exploration (10% random) to escape local minima     │
│  • Network already knows basics, now learns deeper tricks       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Key Simplifications

| Old (MCTS Complexity) | New (Curriculum + A*) |
|-----------------------|-----------------------|
| UCB1 formula | Priority queue |
| Tree backpropagation | Direct supervised learning |
| Visit counts | ε-greedy exploration |
| Policy + Value heads | Single value network |
| Online RL from scratch | Curriculum: small→large |

### 1.5 Goals
- [x] Convert e-graph nodes to NNUE expressions for feature extraction
- [x] CSE-aware feature extraction (BackRef tokens)
- [x] Full backpropagation through NNUE
- [x] Train NNUE on saturated small kernels (ground truth)
- [x] Use NNUE-guided Best-First search for large kernels
- [ ] Scale linearly with search budget, NOT exponentially with rule count

### 1.6 The Two-Loop Architecture

The system has two components trained in separate loops:

**The Judge** (CostModel with learned weights):
- Predicts `cost(expr) → nanoseconds` using learned operation weights
- Trained on real SIMD benchmark data (Loop A: slow, run weekly)
- Once trained, evaluates expressions instantly (nanoseconds)
- File: `pixelflow-ml/examples/train_cost_model.rs`

**The Guide** (NNUE):
- Predicts which rewrites lead to lower-cost expressions
- Trained against The Judge (Loop B: fast, no compilation)
- Learns to navigate the e-graph search space efficiently
- File: `pixelflow-ml/examples/guided_training.rs`

```
┌─────────────────────────────────────────────────────────────────┐
│  LOOP A: CALIBRATION (Weekly, Slow)                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate expression variants via gen_egraph_variants        │
│  2. Compile to SIMD kernels                                     │
│  3. Benchmark with criterion                                    │
│  4. Train The Judge: learns Op → Cost mapping                   │
│  Result: learned_cost_model.toml                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LOOP B: TRAINING (Continuous, Fast)                            │
├─────────────────────────────────────────────────────────────────┤
│  • Generate expressions                                         │
│  • E-graph saturation + extraction                              │
│  • Score with The Judge (instant, no compilation)               │
│  • Train NNUE to predict The Judge's scores                     │
│  Result: NNUE that guides search without physical benchmarks    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.7 Why This Works

1. **Cold Start Solved**: By saturating small kernels, network starts smart. No 10,000 rollouts to learn `x * 0 = 0`.

2. **Valley Crossing**: ε-greedy (10% random) forces exploration of "bad" intermediate states, finding brilliant optimizations.

3. **Simple Code**: Priority queue + ε-greedy. No UCB1, no tree backprop.

4. **Ground Truth**: Small kernel saturation gives mathematically perfect costs - no noise in training signal.

5. **Model-Based Training**: The Judge lets us train NNUE without running physical benchmarks every iteration.

### 1.8 Non-Goals
- GPU acceleration (CPU SIMD only)
- Distributed training
- Full equality saturation for large kernels (that's the point!)

---

## 2. CRITICAL: CSE-Aware Features

### 2.1 The Aliasing Problem

**E-Graph = DAG, ExprTree = Tree**

When the same subexpression appears twice, the e-graph has ONE e-class referenced multiple times. But `ExprTree` uses `Box<ExprTree>` - no sharing!

```
E-Graph: Add(e-class_5, e-class_5)  // Same e-class, computed once
ExprTree: Add(subtree, subtree)     // Two separate copies
Codegen: expensive() + expensive()  // Computed TWICE!
```

### 2.2 The Solution: DAG-Walk Feature Extraction ✓ DONE

The feature extractor now uses structural hashing to detect shared subexpressions:

```rust
// Before (Tree Walk - Double Vision):
x + x  →  [Feature(x), Feature(Add), Feature(x)]  // Counts x twice!

// After (DAG Walk - CSE Aware):
x + x  →  [Feature(x), Feature(Add), BackRef]     // Recognizes sharing!
```

**Implementation** (`guided_training.rs`):
- `structural_hash()` computes unique hash per subtree
- `extract_tree_features_dag()` uses visited HashSet
- `BACKREF_FEATURE` emitted for duplicate subtrees

This lets NNUE learn that shared computation is cheap.

---

## 3. Design

### 3.1 Best-First Search (`best_first.rs`) ✓ DONE

A simple priority queue planner replaces MCTS:

```rust
pub struct BestFirstPlanner {
    queue: BinaryHeap<Candidate>,
    best_found: Option<Candidate>,
    config: BestFirstConfig,
}

impl BestFirstPlanner {
    /// Run search with NNUE evaluator
    pub fn run<F>(&mut self, evaluator: F) -> BestFirstResult
    where
        F: FnMut(&ExprTree) -> i64,
    {
        // Small kernel? Just saturate.
        if self.should_saturate() {
            return self.run_saturation();
        }

        // Best-first with ε-greedy exploration
        while self.expansions < self.config.max_expansions {
            // ε-greedy: occasionally pick random instead of best
            let candidate = if rng.gen() < self.config.epsilon {
                self.pick_random()
            } else {
                self.queue.pop()
            };

            // Expand: apply one round of rewrites
            for child in self.expand(&candidate) {
                child.priority = evaluator(&child.best_tree);
                self.queue.push(child);
            }
        }
    }
}
```

### 3.2 Full Backpropagation (`guided_training.rs`) ✓ DONE

Complete gradient flow through all NNUE layers:

```rust
struct ForwardState {
    l1_pre: Vec<i32>,   // Before clipped ReLU
    l1_post: Vec<f32>,  // After clipped ReLU
    l2_pre: Vec<i32>,
    l2_post: Vec<f32>,
    l3_pre: Vec<i32>,
    l3_post: Vec<f32>,
    active_features: Vec<usize>,
}

fn backward(nnue: &mut Nnue, state: &ForwardState, error: f32, lr: f32) {
    // Output layer → L3 → L2 → L1 → sparse W1
    // Updates all weights: w1, b1, w2, b2, w3, b3, w_out, b_out
}
```

### 3.3 Curriculum Training Loop (TODO)

```rust
fn main() {
    let mut nnue = Nnue::new(NnueConfig::default());

    // ============================================
    // PHASE 1: KINDERGARTEN (Ground Truth)
    // ============================================
    println!("=== Phase 1: Kindergarten (Saturation) ===");

    for epoch in 0..KINDERGARTEN_EPOCHS {
        for _ in 0..SAMPLES_PER_EPOCH {
            // Generate SMALL kernel (10-50 nodes)
            let tree = generate_small_kernel();

            // SATURATE completely - get perfect cost
            let config = BestFirstConfig::default()
                .with_saturation_threshold(1000);  // Force saturation
            let mut planner = BestFirstPlanner::from_tree(&tree, config);
            let result = planner.run_default();

            // Train: features(tree) → optimal_cost
            train_on_ground_truth(&mut nnue, &tree, result.best_cost);
        }
    }

    // ============================================
    // PHASE 2: UNIVERSITY (Guided Search)
    // ============================================
    println!("=== Phase 2: University (Guided Search) ===");

    for epoch in 0..UNIVERSITY_EPOCHS {
        for _ in 0..SAMPLES_PER_EPOCH {
            // Generate LARGE kernel (100+ nodes)
            let tree = generate_large_kernel();

            // Best-First with trained NNUE + exploration
            let config = BestFirstConfig::default()
                .with_saturation_threshold(0)   // Never saturate
                .with_epsilon(0.1)              // 10% exploration
                .with_max_expansions(1000);

            let mut planner = BestFirstPlanner::from_tree(&tree, config);
            let result = planner.run(|tree| nnue_predict(&nnue, tree));

            // Train on achieved cost
            train_on_achieved(&mut nnue, &tree, result.best_cost);
        }
    }
}
```

---

## 4. Implementation Status

### ✓ COMPLETED

| Component | File | Description |
|-----------|------|-------------|
| Full Backpropagation | `guided_training.rs` | Updates all NNUE weights through 4 layers |
| CSE-Aware Features | `guided_training.rs` | `structural_hash()` + `BACKREF_FEATURE` |
| Best-First Search | `best_first.rs` | Priority queue with ε-greedy exploration |
| Auto-saturation | `best_first.rs` | Small kernels (<100 nodes) saturate automatically |

### ✓ COMPLETED (Session 3)

| Component | File | Description |
|-----------|------|-------------|
| Curriculum Training | `guided_training.rs` | Phase 1 (small/saturate) → Phase 2 (large/guided) |

### ✓ COMPLETED (Session 2)

| Component | File | Description |
|-----------|------|-------------|
| EGraph Cloneable | `graph.rs` | Arc-wrapped rules for cheap clone during search |
| Best-First Search | `best_first.rs` | A* with ε-greedy, auto-saturates small kernels |
| Export BestFirstPlanner | `egraph/mod.rs` | Exported via `pub mod best_first` |
| The Judge Training | `train_cost_model.rs` | Learns cost model weights from real SIMD benchmarks |

### DEFERRED (Not Needed)

| Component | Reason |
|-----------|--------|
| `rewrite_mcts.rs` | Best-First search is simpler and sufficient |
| UCB1 formula | ε-greedy exploration is sufficient |
| Visit counts | Not needed without MCTS tree |

---

## 5. Files Reference

| File | Status | Purpose |
|------|--------|---------|
| `pixelflow-search/src/egraph/best_first.rs` | **NEW** ✓ | Simple A* search with ε-greedy |
| `pixelflow-search/src/egraph/mod.rs` | MODIFY | Export `BestFirstPlanner` |
| `pixelflow-ml/examples/guided_training.rs` | MODIFY | Curriculum training loop |
| `pixelflow-search/src/egraph/guided.rs` | KEEP | Existing guided state (may simplify) |
| `pixelflow-nnue/src/lib.rs` | DONE | NNUE network implementation |

---

## 6. Verification

```bash
# Test Best-First search compiles and works
cargo test -p pixelflow-search -- best_first

# Test small kernels saturate correctly
cargo test -p pixelflow-search -- best_first::test_small_kernel_saturates

# Run curriculum training (quick mode)
cargo run -p pixelflow-ml --example guided_training --features training --release -- --quick

# Full training
cargo run -p pixelflow-ml --example guided_training --features training --release

# Compare guided vs random search
cargo test -p pixelflow-search -- guided_vs_random
```

---

## 7. Historical Notes

### 7.1 Previous Approach (MCTS-based)

The original design used MCTS over rewrite actions:
- UCB1 formula for action selection
- Tree backpropagation for value updates
- Policy + Value network heads
- Complex online RL from scratch

**Why abandoned**: Too complex for the problem. Curriculum learning with simple best-first search achieves the same goals with far less code.

### 7.2 CSE Fix Options Considered

| Option | Decision | Reason |
|--------|----------|--------|
| A: Hook into optimize.rs | Deferred | Requires proc-macro changes |
| B: CSE-aware extraction | Deferred | Complex |
| **C: DAG-walk features** | **CHOSEN** | Simple, works in feature extractor |

The DAG-walk approach using `structural_hash()` solves the CSE problem at the feature extraction level without requiring changes to the compilation pipeline.

---

## 8. Open Questions

- [x] How to handle Op variants not in OpType? → Map to closest equivalent (14 buckets)
- [x] How to detect shared subexpressions? → `structural_hash()` + visited HashSet
- [x] How to simplify MCTS? → Replace with Best-First + curriculum learning
- [x] Curriculum transition point? → Fixed epochs (kindergarten → university)
- [ ] Learning rate schedule? → Start with constant, tune later
- [ ] ε-greedy annealing? → Start with 0.1, may reduce over training
