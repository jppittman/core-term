# E-Graph Search Strategy Integration Status

## Executive Summary

**Question:** How does pixelflow-search integrate E-graph search? Are we using A* with learned weights for cases where saturation is impossible?

**Answer:** **NO**. Advanced search strategies (A*, MCTS + NNUE) exist but are **not integrated** into the compiler. Currently using only basic saturation.

## What EXISTS (Comprehensive Implementation)

### 1. Basic Saturation (`saturate.rs`)
- **Budget-limited saturation** for training data generation
- Tracks stats: iterations, unions, rule matches
- Function: `saturate_with_budget(egraph, max_iterations)`
- Used for generating ground truth labels (what's achievable within budget)

### 2. Best-First A* Search (`best_first.rs`)
- **A*/Dijkstra with NNUE heuristic**
- Much simpler than MCTS (just priority queue)
- Epsilon-greedy exploration (ε=0 for inference, ε>0 for training)
- **Auto-fallback:** Small kernels (<100 nodes) use saturation instead
- **Evaluator API:** Passes `BestFirstContext` with search metrics to NNUE

**Key Features:**
```rust
pub struct BestFirstContext<'a> {
    pub tree: &'a ExprTree,          // Expression being evaluated
    pub tree_cost: usize,             // Ground truth from CostModel
    pub depth: usize,                 // Depth in search tree
    pub best_cost: usize,             // Global best found
    pub expansions: usize,            // States expanded
    pub frontier_size: usize,         // Open set size
    pub initial_cost: usize,          // Before optimization
}
```

**NNUE Integration Point:**
```rust
planner.run(|ctx: BestFirstContext| -> i64 {
    // NNUE predicts: priority score (lower = more promising)
    nnue_evaluate_state(ctx.tree, ctx.depth, ctx.frontier_pressure())
});
```

### 3. Guided MCTS (`guided.rs`)
- **MCTS with UCB1 + NNUE priors**
- Incremental E-graph construction (applies only promising rewrites)
- Scales with budget, not rule count (unlike full saturation)
- **Two integration points:**
  1. UCB1 action selection (exploration/exploitation)
  2. NNUE prior Q-values (guide which rewrites to try)

**Action Selection with NNUE:**
```rust
mcts.iterate_with_evaluator(|state, action| -> f64 {
    // NNUE predicts: prior Q-value for (state, action)
    nnue_evaluate_action(state.egraph(), action.rule_idx, action.target_class)
});
```

**Blended Heuristic:**
```text
UCB1(a) = blended_Q(a) + c * sqrt(ln(N) / n(a))

where:
  blended_Q = (1-w) * nnue_prior(a) + w * empirical_avg(a)
  w = min(visits, 10) / 10  (gradually trust empirical more)
```

## What is MISSING (The Integration Gap)

### Current Compiler Uses Only Saturation

**File:** `pixelflow-compiler/src/optimize.rs:388-415`

```rust
fn optimize_via_egraph(expr: &Expr, costs: &CostModel) -> Expr {
    let mut ctx = EGraphContext::new();
    let root = ctx.expr_to_egraph(expr);
    ctx.egraph.saturate();  // ❌ Always saturates (no A*, no MCTS)
    let tree = ctx.egraph.extract_tree_with_costs(root, costs);
    ctx.tree_to_expr(&tree)
}

fn optimize_via_ir(expr: &Expr, costs: &CostModel) -> Expr {
    let mut ctx = IRToEGraphContext::new();
    let root = ctx.ir_to_egraph(&ir);
    ctx.egraph.saturate();  // ❌ Always saturates
    let tree = ctx.egraph.extract_tree_with_costs(root, costs);
    // ...
}
```

**Problem:** No budget limits, no fallback strategy, no NNUE guidance!

## Architecture Comparison

### Current (Saturation Only)

```
AST/IR → E-graph → saturate() → extract_best() → optimized AST
                    ^^^^^^^^
                    Always runs to completion
                    No budget, no heuristic
```

### Available But Unused (Hybrid Strategy)

```
AST/IR → E-graph → adaptive_optimize() → optimized AST
                    │
                    ├─ Small (<100 nodes)  → saturate()
                    ├─ Medium (100-1000)   → best_first(NNUE)
                    └─ Large (>1000)       → guided_mcts(NNUE)
```

## Recommended Integration

### Phase 1: Add Budget-Limited Saturation

Replace unlimited saturation with budget-limited version:

```rust
// Instead of:
ctx.egraph.saturate();

// Use:
use pixelflow_search::egraph::saturate_with_budget;
let result = saturate_with_budget(&mut ctx.egraph, 1000);

if !result.saturated {
    eprintln!("Warning: Saturation incomplete ({} iterations, {} classes)",
        result.iterations, result.classes_after);
}
```

### Phase 2: Add Hybrid Strategy

Automatically choose based on kernel size:

```rust
fn optimize_adaptive(expr: &Expr, costs: &CostModel) -> Expr {
    let mut ctx = EGraphContext::new();
    let root = ctx.expr_to_egraph(expr);

    let node_count = ctx.egraph.node_count();

    let tree = if node_count < 100 {
        // Small kernel: saturate fully (fast, optimal)
        ctx.egraph.saturate();
        ctx.egraph.extract_tree_with_costs(root, costs)

    } else if node_count < 1000 {
        // Medium kernel: best-first A* with NNUE
        use pixelflow_search::egraph::BestFirstPlanner;
        let mut planner = BestFirstPlanner::from_tree(&initial_tree, config);
        let result = planner.run(|ctx| {
            // Simple heuristic (or NNUE when available)
            (ctx.tree_cost + ctx.depth * 10) as i64
        });
        result.best_tree

    } else {
        // Large kernel: guided MCTS (incremental)
        use pixelflow_search::egraph::guided_optimize;
        let config = GuidedConfig::default()
            .with_iterations(5000)
            .inference_mode();
        let result = guided_optimize(&initial_tree, config);
        result.tree
    };

    ctx.tree_to_expr(&tree)
}
```

### Phase 3: Wire NNUE When Available

Add evaluator trait for pluggable heuristics:

```rust
pub trait SearchEvaluator {
    fn evaluate_state(&self, ctx: &BestFirstContext) -> i64;
    fn evaluate_action(&self, state: &GuidedState, action: &GuidedAction) -> f64;
}

// Simple HCE evaluator (default, no ML)
pub struct HceEvaluator;

impl SearchEvaluator for HceEvaluator {
    fn evaluate_state(&self, ctx: &BestFirstContext) -> i64 {
        // Use tree depth + cost as heuristic
        (ctx.tree_cost + ctx.depth * 10) as i64
    }

    fn evaluate_action(&self, state: &GuidedState, action: &GuidedAction) -> f64 {
        // Uniform prior (no preference)
        0.0
    }
}

// NNUE evaluator (when trained model available)
#[cfg(feature = "nnue-eval")]
pub struct NnueEvaluator {
    model: Nnue,
}

impl SearchEvaluator for NnueEvaluator {
    fn evaluate_state(&self, ctx: &BestFirstContext) -> i64 {
        // Extract HalfEP features, forward pass through network
        let features = extract_features(ctx.tree);
        self.model.evaluate(&features)
    }

    fn evaluate_action(&self, state: &GuidedState, action: &GuidedAction) -> f64 {
        // Predict Q-value for this (state, action) pair
        let features = extract_action_features(state, action);
        self.model.evaluate_action(&features)
    }
}
```

## Timeline

**Immediate (Now):**
- Document gap in `docs/EGRAPH_SEARCH_INTEGRATION.md` ✓ (this file)
- Add budget limits to existing saturation calls

**Short-term (Next):**
- Wire best-first search for medium kernels
- Use simple heuristic (tree depth + cost, no NNUE yet)
- Measure performance vs full saturation

**Long-term (Future):**
- Train NNUE model on search trajectories
- Integrate NNUE evaluator via trait
- Benchmark guided MCTS on large real-world kernels

## Key Files

| File | What It Does | Status |
|------|--------------|--------|
| `pixelflow-search/src/egraph/saturate.rs` | Budget-limited saturation | ✅ Exists, not used |
| `pixelflow-search/src/egraph/best_first.rs` | A* with NNUE heuristic | ✅ Exists, not used |
| `pixelflow-search/src/egraph/guided.rs` | MCTS with NNUE priors | ✅ Exists, not used |
| `pixelflow-compiler/src/optimize.rs` | Compiler optimizer | ❌ Uses only saturate() |
| `pixelflow-ml/src/nnue.rs` | NNUE network (401k features) | ✅ Exists, not integrated |

## Conclusion

**The infrastructure is remarkably complete:**
- Best-first A* ✅
- Guided MCTS ✅
- Budget-limited saturation ✅
- NNUE network ✅

**But none of it is wired up!**

Current state: Driving a Formula 1 car in first gear only (basic saturation).

Recommended: Add adaptive strategy that automatically uses the right tool for the kernel size.
