# GNN-Guided E-Graph Rewrite Selection: Vision & Architecture

## The Vision

**Goal:** Learn which rewrites are most productive in which E-graph contexts using Graph Neural Networks (GNNs).

**Why GNNs?**
- E-graphs ARE graphs → natural fit
- Learn structural patterns, not just hand-crafted features
- Capture long-range dependencies (multi-hop relationships)
- Generalize to new operation types
- Learn complex rewrite interactions

## Current State: Feedforward NNUE

### What Exists

**Architecture:** 256-32-32-1 feedforward network (Stockfish-style)
- **Input:** 401,408 HalfEP features (operation type, depth, path encoding)
- **Layers:** Linear → ReLU → Linear → ReLU → Linear → output
- **Output:** Single scalar cost estimate

**File:** `pixelflow-ml/src/nnue.rs`, `layer.rs`

### Limitations

1. **Fixed feature vocabulary**: HalfEP features hardcoded to specific ops
2. **No graph structure**: Treats expression as bag of (op_type, descendant, depth) tuples
3. **No message passing**: Can't learn multi-hop patterns
4. **Shallow context**: Only sees local neighborhoods, not global structure
5. **Can't generalize**: New operation types → need to retrain entire feature set

## The GNN Architecture We Need

### Message Passing on E-Graph Structure

**Nodes:** E-classes (equivalence classes of expressions)
**Edges:** Dependency relationships (op uses these children)

```rust
struct EGraphGNN {
    // Node embeddings: one per e-class
    node_embeddings: HashMap<EClassId, Tensor>,

    // Message passing layers
    layers: Vec<MessagePassingLayer>,

    // Readout for final prediction
    readout: GlobalPooling,
}

struct MessagePassingLayer {
    // Node update: combine neighbors' messages
    message_fn: Linear,        // h_neighbor → message
    aggregate_fn: Aggregation, // sum/mean/max over messages
    update_fn: GRU,            // h_old + messages → h_new
}
```

### Forward Pass

```rust
impl EGraphGNN {
    fn forward(&self, egraph: &EGraph, target_eclass: EClassId) -> f32 {
        // 1. Initialize node features from e-class contents
        let mut h = self.init_node_features(egraph);

        // 2. Message passing (k layers)
        for layer in &self.layers {
            h = layer.propagate(egraph, h);
        }

        // 3. Readout: predict value of target e-class
        let embedding = h[target_eclass];
        self.readout.predict(embedding)
    }

    fn init_node_features(&self, egraph: &EGraph) -> NodeFeatures {
        // Extract initial features for each e-class:
        // - Operations it contains (one-hot or learned embedding)
        // - Node count, depth in tree
        // - Cost of best extraction so far
        // - Whether it's been modified recently
    }
}

struct MessagePassingLayer {
    fn propagate(&self, egraph: &EGraph, h: NodeFeatures) -> NodeFeatures {
        let mut h_new = h.clone();

        for (eclass, neighbors) in egraph.adjacency() {
            // Aggregate messages from neighbors
            let messages: Vec<Tensor> = neighbors.iter()
                .map(|n| self.message_fn.forward(&h[n]))
                .collect();

            let aggregated = self.aggregate_fn.apply(messages);

            // Update this node's embedding
            h_new[eclass] = self.update_fn.forward(&h[eclass], &aggregated);
        }

        h_new
    }
}
```

## Use Cases for GNN Guidance

### 1. Rewrite Prioritization

**Question:** Which (rule, eclass, node) triple should we apply next?

```rust
fn score_rewrite_target(
    gnn: &EGraphGNN,
    egraph: &EGraph,
    target: &RewriteTarget
) -> f32 {
    // Run message passing to get e-class embeddings
    let embeddings = gnn.forward(egraph);

    // Predict: "How much will this rewrite improve extraction cost?"
    gnn.predict_improvement(
        embeddings[target.eclass],
        target.rule.id(),
        target.node_idx
    )
}

fn optimize_with_gnn(egraph: &mut EGraph, gnn: &EGraphGNN, budget: usize) {
    for _ in 0..budget {
        let targets = egraph.enumerate_rewrite_targets();
        if targets.is_empty() { break; } // Saturated

        // Pick highest-scoring rewrite
        let best = targets.into_iter()
            .max_by_key(|t| score_rewrite_target(gnn, egraph, t))
            .unwrap();

        egraph.apply_single_rewrite(best);
    }
}
```

### 2. Saturation Termination

**Question:** Should we keep saturating, or is current E-graph "good enough"?

```rust
fn should_stop_early(gnn: &EGraphGNN, egraph: &EGraph, budget_left: usize) -> bool {
    // Global graph embedding (pooled over all e-classes)
    let graph_embedding = gnn.global_readout(egraph);

    // Predict: "Expected improvement from `budget_left` more iterations"
    let expected_gain = gnn.predict_saturation_value(graph_embedding, budget_left);

    // Stop if expected gain is tiny
    expected_gain < THRESHOLD
}
```

### 3. Rule Set Selection

**Question:** Which rewrite rules should we even enable for this kernel?

Some rewrites are only useful in specific contexts (e.g., trig identities for graphics, FMA for dense math).

```rust
fn select_active_rules(gnn: &EGraphGNN, egraph: &EGraph) -> Vec<RuleId> {
    let graph_embedding = gnn.global_readout(egraph);

    // Predict: "Which rules are likely to be productive?"
    ALL_RULES.into_iter()
        .filter(|rule| gnn.predict_rule_usefulness(graph_embedding, rule) > 0.5)
        .collect()
}
```

## Training Data Generation

### Supervised: Oracle Labels

**Teacher:** Full saturation (run to completion, record all rewrites)

**Student (GNN):** Predict which rewrites were productive

```rust
struct TrainingExample {
    // Input: E-graph state before rewrite
    egraph_snapshot: EGraph,
    target: RewriteTarget,

    // Label: Did this rewrite improve best extraction?
    improvement: f32,  // cost_before - cost_after

    // Context
    budget_left: usize,
    best_cost_so_far: usize,
}

fn generate_training_data(expr: &ExprTree, costs: &CostModel) -> Vec<TrainingExample> {
    let mut egraph = EGraph::new();
    let root = egraph.add_expr(expr);
    let mut examples = Vec::new();

    loop {
        let targets = egraph.enumerate_rewrite_targets();
        if targets.is_empty() { break; }

        // Try each rewrite, measure improvement
        for target in targets {
            let snapshot = egraph.clone();
            let cost_before = egraph.extract_best(root, costs).1;

            egraph.apply_single_rewrite(target);
            let cost_after = egraph.extract_best(root, costs).1;

            examples.push(TrainingExample {
                egraph_snapshot: snapshot,
                target,
                improvement: (cost_before - cost_after) as f32,
                budget_left: usize::MAX, // Full saturation
                best_cost_so_far: cost_before,
            });

            // Restore and continue exploring
            egraph = snapshot;
        }

        // Actually apply the BEST rewrite (greedy oracle)
        let best = targets.into_iter()
            .max_by_key(|t| measure_improvement(egraph, t))
            .unwrap();
        egraph.apply_single_rewrite(best);
    }

    examples
}
```

### Reinforcement Learning: Budget-Limited Search

**Environment:** E-graph optimization
**State:** Current E-graph structure
**Action:** Apply specific rewrite
**Reward:** Improvement in extraction cost (or final cost at episode end)

```rust
fn train_with_rl(gnn: &mut EGraphGNN, budget: usize) {
    for episode in 0..NUM_EPISODES {
        let expr = sample_random_expr();
        let mut egraph = EGraph::new();
        let root = egraph.add_expr(&expr);

        let mut trajectory = Vec::new();

        for step in 0..budget {
            let state = egraph.clone();
            let targets = egraph.enumerate_rewrite_targets();
            if targets.is_empty() { break; }

            // ε-greedy: sometimes explore
            let action = if rand() < EPSILON {
                targets.sample_random()
            } else {
                targets.max_by_key(|t| gnn.score(egraph, t))
            };

            let cost_before = egraph.extract_best(root, costs).1;
            egraph.apply_single_rewrite(action);
            let cost_after = egraph.extract_best(root, costs).1;

            trajectory.push((state, action, cost_before - cost_after));
        }

        // Backpropagate rewards
        gnn.train_on_trajectory(trajectory);
    }
}
```

## Architecture Phases

### Phase 1: Budget-Limited Saturation (NOW)
**Status:** Simple, works today
```rust
fn optimize(egraph: &mut EGraph, budget: usize) {
    for _ in 0..budget {
        if egraph.apply_rules_once() == 0 { break; }
    }
}
```

### Phase 2: Heuristic Priority (NEXT)
**Status:** Use hand-crafted heuristics (no GNN yet)
```rust
fn optimize_with_heuristic(egraph: &mut EGraph, budget: usize) {
    for _ in 0..budget {
        let targets = egraph.enumerate_rewrite_targets();
        if targets.is_empty() { break; }

        // Simple heuristic: prioritize Identity rules, then Fusion
        let best = targets.max_by_key(|t| heuristic_score(t));
        egraph.apply_single_rewrite(best);
    }
}
```

### Phase 3: GNN-Guided Priority (FUTURE)
**Status:** Requires GNN infrastructure
```rust
fn optimize_with_gnn(egraph: &mut EGraph, gnn: &EGraphGNN, budget: usize) {
    for _ in 0..budget {
        let targets = egraph.enumerate_rewrite_targets();
        if targets.is_empty() { break; }

        // GNN predicts best rewrite
        let best = targets.max_by_key(|t| gnn.score(egraph, t));
        egraph.apply_single_rewrite(best);
    }
}
```

## What We Need to Build

### 1. GNN Primitives
- [ ] Message passing layers (sum, mean, max aggregation)
- [ ] Node embedding initialization from e-class contents
- [ ] Graph pooling (global readout)
- [ ] E-graph → adjacency list conversion

### 2. Training Infrastructure
- [ ] Supervised training: oracle labels from full saturation
- [ ] RL training: policy gradient on budget-limited episodes
- [ ] Feature extraction: e-class → initial embedding
- [ ] Batching: train on multiple graphs simultaneously

### 3. Integration
- [ ] `enumerate_rewrite_targets()` in `EGraph`
- [ ] `apply_single_rewrite()` in `EGraph`
- [ ] Priority-based saturation loop in `optimize.rs`
- [ ] Fallback: use heuristic when GNN unavailable

## Why This Matters for Complex E-Graphs

**Current E-graph:** Algebraic rewrites (add, mul, sqrt, etc.)
- Simple rules (~15 rules total)
- Saturates fast (<100 iterations typical)
- Hand-crafted rule ordering works fine

**Future E-graph:** Compiler IR (scheduling, fusion, layout, etc.)
- Hundreds of rewrite rules
- Complex interactions (applying X enables Y)
- Saturation may be infeasible (exponential blowup)
- **GNN is essential** for tractability

## Example: Why NNUE Isn't Enough

**NNUE limitation:** Fixed feature vocabulary

```rust
// NNUE sees: HalfEP features (op_type, descendant, depth)
// Example: (Mul, Add, depth=2) → feature index 12345
```

**Problem:** New operation type (e.g., `TensorContract`) → entire feature space breaks

**GNN advantage:** Learned operation embeddings
```rust
// GNN learns: embedding for each operation type
op_embeddings = {
    "Add": [0.2, -0.5, 0.1, ...],
    "Mul": [0.8, 0.3, -0.2, ...],
    "TensorContract": [???]  // Learned, not hardcoded
}
```

Can generalize to new ops via:
1. Initialize from similar ops (e.g., TensorContract ≈ MatMul)
2. Fine-tune on new examples
3. Meta-learning: learn to learn new operations

## Conclusion

**Short-term (now):** Budget-limited saturation with simple heuristics
- Sufficient for current algebraic E-graphs
- No GNN overhead
- Easy to implement and debug

**Long-term (when E-graphs get complex):** GNN-guided rewrite selection
- Essential for scaling to hundreds of rules
- Learns deep structural patterns
- Generalizes to new operation types
- Worth the complexity investment

**Architecture today should:**
1. Support single-rewrite application (not just batch iterations)
2. Expose rewrite enumeration API
3. Use pluggable evaluator trait (heuristic now, GNN later)
4. Keep saturation as fallback for small graphs

The priority-guided approach IS worth the complexity **when E-graphs justify it**.
