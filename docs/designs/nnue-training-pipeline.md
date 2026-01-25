# NNUE Training Pipeline Design

## Metadata
- **Author**: jppittman + claude
- **Status**: Approved
- **Created**: 2026-01-24
- **Reviewers**: -

---

## 1. Overview

### 1.1 Problem Statement
We have NNUE infrastructure (`pixelflow-nnue`) and MCTS search (`pixelflow-search/mcts.rs`), but they're not connected. We need:
1. A bridge between e-graph `Op` types and NNUE `OpType`
2. Feature extraction from e-graph states
3. End-to-end training that uses MCTS to find optimal rewrites

### 1.2 Goals
- [ ] Convert e-graph nodes to NNUE expressions for feature extraction
- [ ] Train NNUE to predict rewrite costs with Spearman ρ > 0.7
- [ ] Use NNUE-guided MCTS to find better rewrites than random search

### 1.3 Non-Goals
- GPU acceleration (CPU SIMD only)
- Production deployment (research/prototype quality)
- Distributed training

---

## 2. Background

### 2.1 Current State

```
pixelflow-nnue/           # NNUE network, features, expressions
├── OpType                # 14 operation types
├── Expr                  # Expression AST
├── HalfEPFeature         # Sparse features
├── Nnue                  # Network weights
└── Accumulator           # Incremental evaluation

pixelflow-search/
├── egraph/               # E-graph implementation
│   ├── Op                # E-graph operation enum (DIFFERENT from OpType!)
│   └── EGraph            # Equality saturation
├── search/
│   └── mcts.rs           # MCTS (just implemented)
└── eval.rs               # Training data generation (uses pixelflow_nnue)
```

**The Gap**: `egraph::Op` and `nnue::OpType` are separate enums. No conversion exists.

### 2.2 Prior Art
- Stockfish NNUE: HalfKP features, incremental updates
- egg (e-graphs): Rust equality saturation library
- AlphaGo: MCTS + neural network evaluation

---

## 3. Design

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │ Generate │───▶│ Extract  │───▶│ Search   │───▶│ Train │ │
│  │ Expr     │    │ Features │    │ (MCTS)   │    │ NNUE  │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│       │               │               │               │     │
│       ▼               ▼               ▼               ▼     │
│  ExprGenerator   EGraph+Adapter   MctsTree      Gradient   │
│                                   +Evaluator    Descent    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Interfaces

**T1: Op ↔ OpType Adapter** (`pixelflow-search/src/egraph/nnue_adapter.rs`)
```rust
use crate::egraph::Op;
use pixelflow_nnue::OpType;

/// Convert e-graph Op to NNUE OpType.
pub fn op_to_nnue(op: &Op) -> OpType;

/// Convert NNUE OpType to e-graph Op (for generating test expressions).
pub fn nnue_to_op(op_type: OpType) -> Option<Op>;
```

**T2: EGraph → Expr Conversion** (`pixelflow-search/src/egraph/nnue_adapter.rs`)
```rust
use crate::egraph::{EGraph, EClassId};
use pixelflow_nnue::Expr;

/// Extract a concrete Expr from an e-class (picks canonical representative).
pub fn eclass_to_expr(egraph: &EGraph, class: EClassId) -> Expr;

/// Convert an Expr back to e-graph nodes (for testing).
pub fn expr_to_egraph(expr: &Expr, egraph: &mut EGraph) -> EClassId;
```

**T3: MCTS Evaluator using NNUE** (`pixelflow-search/src/search/nnue_eval.rs`)
```rust
use crate::search::mcts::MctsTree;
use pixelflow_nnue::{Nnue, Accumulator};

/// State for MCTS: an e-graph with a focus e-class.
pub struct SearchState {
    pub egraph: EGraph,
    pub focus: EClassId,
}

/// Action for MCTS: a rewrite rule application.
pub struct RewriteAction {
    pub rule_id: usize,
    pub target: EClassId,
}

/// Create an NNUE-based evaluator for MCTS.
pub fn nnue_evaluator(nnue: &Nnue) -> impl Fn(&SearchState) -> f64;
```

**T4: Training Loop** (`pixelflow-search/src/training.rs`)
```rust
use pixelflow_nnue::{Nnue, TrainingSample};

/// Training configuration.
pub struct TrainConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

/// Train NNUE on samples, return final loss.
pub fn train(nnue: &mut Nnue, samples: &[TrainingSample], config: &TrainConfig) -> f32;

/// Evaluate Spearman correlation on held-out data.
pub fn evaluate_correlation(nnue: &Nnue, samples: &[TrainingSample]) -> f64;
```

### 3.3 Data Flow

```
1. Generate random Expr (pixelflow_nnue::ExprGenerator)
         │
         ▼
2. Insert into EGraph (expr_to_egraph)
         │
         ▼
3. Run equality saturation (egraph.saturate())
         │
         ▼
4. Extract features from e-classes (eclass_to_expr → extract_features)
         │
         ▼
5. MCTS search with NNUE evaluation (nnue_evaluator)
         │
         ▼
6. Record (features, cost) pairs as TrainingSamples
         │
         ▼
7. Train NNUE (gradient descent on cost prediction)
         │
         ▼
8. Repeat with improved NNUE
```

---

## 4. Implementation Plan

### 4.1 Task Breakdown

| ID | Task | File | Deps | Size | Assignee |
|----|------|------|------|------|----------|
| T1 | Op ↔ OpType mapping | `egraph/nnue_adapter.rs` | None | S | Jules |
| T2 | EClass ↔ Expr conversion | `egraph/nnue_adapter.rs` | T1 | M | Jules |
| T3 | NNUE MCTS evaluator | `search/nnue_eval.rs` | T1 | M | Jules |
| T4 | Training loop | `training.rs` | T2, T3 | M | Claude |
| T5 | Integration test | `tests/nnue_e2e.rs` | T4 | S | Claude |

**Size**: S = <50 lines, M = 50-200 lines, L = >200 lines

### 4.2 Parallelization

```
T1 (Op mapping) ─────┬──▶ T2 (EClass conversion) ──┐
                     │                              ├──▶ T4 (Training) ──▶ T5 (E2E)
                     └──▶ T3 (NNUE evaluator) ─────┘
```

**Parallel batch 1**: T1 (single file, no deps)
**Parallel batch 2**: T2, T3 (both depend on T1, different files)
**Sequential**: T4, T5 (integration work)

### 4.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Op/OpType mismatch | Medium | High | Explicit test for all variants |
| MCTS too slow | Low | Medium | Limit iterations, profile |
| Training diverges | Medium | Medium | Gradient clipping, validation set |

---

## 5. Testing Strategy

### 5.1 Unit Tests

**T1 Tests** (in `nnue_adapter.rs`):
- `test_op_to_nnue_roundtrip`: All Op variants convert and back
- `test_op_arity_matches`: Op and OpType arities are consistent

**T2 Tests**:
- `test_eclass_to_expr_simple`: Leaf nodes convert correctly
- `test_eclass_to_expr_nested`: Binary ops convert correctly
- `test_expr_to_egraph_roundtrip`: Convert and extract matches

**T3 Tests**:
- `test_nnue_evaluator_deterministic`: Same state → same score
- `test_nnue_evaluator_prefers_simpler`: Simpler expr gets better score

### 5.2 Integration Tests

**T5** (`tests/nnue_e2e.rs`):
```rust
#[test]
fn test_training_improves_correlation() {
    // Generate 1000 samples
    // Train for 10 epochs
    // Assert Spearman ρ > 0.5
}
```

---

## 6. Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|-------------|------|------|---------|
| Unify Op/OpType into one enum | No conversion needed | Breaks crate separation | Tight coupling |
| Skip MCTS, use random search | Simpler | Less effective | Defeats purpose |
| Train on real compiler IR | More realistic | Complex setup | Research scope |

---

## 7. Open Questions

- [x] How to handle Op variants not in OpType? → Map to closest equivalent
- [x] Which e-class representative to extract? → Lowest cost (greedy)
- [ ] Learning rate schedule? → Start with constant, tune later
