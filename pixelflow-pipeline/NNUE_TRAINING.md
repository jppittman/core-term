# NNUE Training Pipeline v2

## The Core Problem: Rule Explosion

This system exists because **saturation doesn't scale with rule count**.

Traditional e-graph optimizers (egg, etc.) work by:
1. Add all rewrite rules
2. Saturate (apply rules until fixpoint)
3. Extract cheapest

This works beautifully with ~20 algebraic rules. It breaks completely when you want:

| Domain | Example Rules | Combinatorial Explosion |
|--------|---------------|------------------------|
| Symbolic differentiation | Chain rule, product rule, quotient rule | Every derivative enables more |
| Trigonometry | sin²+cos²=1, angle addition, half-angle | Identities connect to everything |
| Exponentiation | exp/log laws, power rules | Interacts with trig via Euler |
| Change of basis | Coordinate transforms, rotations | Multiplies with all of the above |
| Operation reordering | Associativity, commutativity | Factorial blowup |

With 200+ rules, saturation **never terminates**. The equivalence class for `sin(x)²` contains hundreds of forms, each enabling dozens more rewrites. You're not optimizing expressions — you're navigating an infinite equivalence space.

**This is why we need learned search guidance.**

## The Research Bet

The deeper hypothesis: **most compiler optimization can be framed this way**.

LLVM's approach:
- Hand-written rewrite rules (peephole, instcombine, etc.)
- Hand-written heuristics for when to apply them
- Hand-written pass ordering
- Decades of expert tuning

Our approach:
- Write all the rules you can think of (no need to be conservative)
- Learn when to apply them from benchmark data
- Learn pass ordering implicitly via search trajectories

If the learned heuristics generalize, this could be competitive with LLVM — not by being smarter about individual optimizations, but by discovering optimization sequences humans never considered.

**The key insight**: LLVM engineers must be conservative about adding rules because bad interactions cause phase-ordering problems. We can be aggressive because The Guide learns to avoid bad interactions.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │     Shared Embedding Layer          │
                    │     401,408 × 256 (HalfEP)          │
                    │                                     │
                    │  Features: (perspective_op,         │
                    │             descendant_op,          │
                    │             depth, path)            │
                    └──────────────┬──────────────────────┘
                                   │
                                   │ Accumulator (256 floats)
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │    The Judge        │         │    The Guide        │
        │    (Extract Head)   │         │    (Search Head)    │
        │    256 → 32 → 1     │         │    256 → 32 → 1     │
        └─────────────────────┘         └─────────────────────┘
                   │                               │
                   ▼                               ▼
           cost_ns prediction            rewrite priority score
```

### The Judge (Extract Head)

**Purpose:** Predict actual runtime cost of a concrete expression tree.

**Trained on:** Real SIMD benchmark data (Loop A).

**Runs when:** End of search, to pick the final extraction. Optionally at checkpoints during long searches.

**Output:** Predicted nanoseconds for the compiled kernel.

### The Guide (Search Head)

**Purpose:** Score rewrite candidates in the search heap.

**Trained on:** The Judge's scores on search outcomes (Loop B).

**Runs when:** Every heap pop during guided search.

**Output:** Priority score (lower = more promising rewrite).

**Critical role:** With 200+ rules, most rewrites are useless or harmful at any given moment. The Guide learns context-dependent value: "chain rule is great here, trig identity is a dead end."

### Shared Embeddings

Both heads share the 401k × 256 embedding matrix. Benefits:

1. Features learned for cost prediction help search guidance
2. Half the memory footprint
3. Transfer learning built-in
4. Single accumulator state serves both evaluations

## The Two Training Loops

```
┌─────────────────────────────────────────────────────────────────┐
│  LOOP A: CALIBRATION (Weekly, Slow)                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate expression variants via gen_egraph_variants        │
│  2. Compile to SIMD kernels                                     │
│  3. Benchmark with criterion                                    │
│  4. Train The Judge: features(expr) → actual_cost_ns            │
│  5. Embeddings update during Judge training                     │
│                                                                 │
│  Artifacts:                                                     │
│    - benchmark_cache.jsonl (raw data)                           │
│    - nnue_shared_embeddings.bin                                 │
│    - nnue_judge_head.bin                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LOOP B: GUIDE TRAINING (Continuous, Fast)                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate random expressions                                 │
│  2. Run guided search with current Guide                        │
│  3. At search end, Judge scores the final extraction            │
│  4. Backprop: Guide should have predicted this outcome          │
│  5. Optionally fine-tune embeddings (or freeze them)            │
│                                                                 │
│  Artifacts:                                                     │
│    - nnue_guide_head.bin                                        │
│    - (optionally updated embeddings)                            │
└─────────────────────────────────────────────────────────────────┘
```

## Incremental Evaluation (Accumulator State)

The key performance optimization: **state is the accumulated hidden activations**, not the input features.

### Why This Matters for Rule-Heavy Systems

With 200+ rules, you might evaluate thousands of candidate rewrites per optimization. Each evaluation must be fast. NNUE accumulators give you O(1) evaluation after O(delta) updates, instead of O(tree_size) for naive approaches.

### The Accumulator

```rust
struct Accumulator {
    /// Hidden layer activations (THE STATE)
    values: [f32; 256],
}

impl Accumulator {
    fn apply_delta(&mut self, removed: &[FeatureIdx], added: &[FeatureIdx], weights: &Embeddings) {
        for &feat in removed {
            for i in 0..256 {
                self.values[i] -= weights[(feat, i)];
            }
        }
        for &feat in added {
            for i in 0..256 {
                self.values[i] += weights[(feat, i)];
            }
        }
    }
}
```

### Search State

```rust
struct SearchState {
    /// The e-graph being optimized
    egraph: EGraph,

    /// Accumulated hidden state for NNUE evaluation
    accumulator: Accumulator,

    /// Active features (for computing deltas on rewrite)
    active_features: HashSet<FeatureIdx>,
}
```

### Rewrite Application Flow

```rust
impl SearchState {
    fn apply_rewrite(&mut self, site: NodeId, rewrite: &Rewrite) {
        // 1. Extract features from OLD subtree
        let old_features = extract_local_features(&self.egraph, site);

        // 2. Apply rewrite (mutates e-graph)
        self.egraph.apply_at(site, rewrite);

        // 3. Extract features from NEW subtree
        let new_features = extract_local_features(&self.egraph, site);

        // 4. Compute delta
        let removed: Vec<_> = old_features.difference(&new_features).copied().collect();
        let added: Vec<_> = new_features.difference(&old_features).copied().collect();

        // 5. Incremental accumulator update: O(delta_size), not O(tree_size)
        self.accumulator.apply_delta(&removed, &added, &self.embeddings);
        self.active_features = new_features;

        // 6. Scan locally for newly enabled rewrites
        let new_candidates = scan_for_patterns(&self.egraph, site, radius: 2);
        self.heap.extend_scored(new_candidates, |r| self.score_rewrite(r));
    }

    fn score_rewrite(&self, rewrite: &RewriteCandidate) -> f32 {
        // Guide head: accumulator → priority score
        self.guide_head.forward(&self.accumulator)
    }
}
```

## The Search Loop

```rust
fn guided_optimize(expr: &Expr, budget: usize) -> Expr {
    let mut state = SearchState::from_expr(expr);

    // Initialize heap with all applicable rewrites, scored by Guide
    state.heap = find_all_rewrites(&state.egraph)
        .map(|r| (state.score_rewrite(&r), r))
        .collect();

    for _ in 0..budget {
        // Pop best rewrite (lowest score = most promising)
        let Some(rewrite) = state.heap.pop() else { break };

        // ε-greedy: occasionally pick random rewrite (exploration)
        let rewrite = if rng.gen::<f32>() < EPSILON {
            state.heap.random()
        } else {
            rewrite
        };

        // Apply with incremental accumulator update
        state.apply_rewrite(rewrite);
    }

    // Final extraction, scored by Judge
    let candidates = state.egraph.extract_k_best(root, k: 10);
    candidates
        .into_iter()
        .min_by_key(|tree| state.judge_head.forward(&Accumulator::from_tree(tree)))
        .unwrap()
}
```

## Comparison with Existing Approaches

### vs Egg (Equality Saturation)

| Aspect | Egg | PixelFlow NNUE |
|--------|-----|----------------|
| Rule count | Must be conservative | Add everything |
| Saturation | To fixpoint (explodes) | Never saturate |
| Search | Exhaustive enumeration | Learned guidance |
| Cost model | Static weights | Learned from benchmarks |
| Phase ordering | N/A (simultaneous) | Learned implicitly |

**Egg's fundamental limit:** Saturation complexity is O(rules × e-classes × iterations). With 200 rules, this is intractable.

### vs LLVM

| Aspect | LLVM | PixelFlow NNUE |
|--------|------|----------------|
| Rules | Hand-written, conservative | Aggressive, exhaustive |
| Heuristics | Hand-tuned over decades | Learned from data |
| Pass ordering | Fixed pipeline, tuned | Learned via search |
| New rules | Risk phase-ordering bugs | Just add and retrain |
| Target-specific | Separate tuning per target | Learns from target benchmarks |

**LLVM's fundamental limit:** Adding rules is dangerous. Interactions cause regressions. Engineers must be conservative. We can be aggressive because The Guide learns to avoid bad interactions.

### vs Superoptimizers (STOKE, etc.)

| Aspect | Superoptimizers | PixelFlow NNUE |
|--------|-----------------|----------------|
| Search space | Instruction sequences | Rewrite sequences |
| Guidance | Random/MCMC | Learned heuristic |
| Verification | SMT solver | Rewrite rules guarantee equivalence |
| Scalability | Small sequences only | Scales with training |

**Superoptimizer advantage:** Can find optimizations not expressible as rewrites.

**Our advantage:** Equivalence guaranteed by construction. No verification needed.

## Rule Domains (Planned)

### Implemented
- [x] Basic algebra (distribute, factor, identity)
- [x] FMA fusion
- [x] Strength reduction (mul → shift, etc.)

### In Progress
- [ ] Symbolic differentiation (chain, product, quotient rules)
- [ ] Trigonometric identities
- [ ] Exponential/logarithm laws

### Planned
- [ ] Change of basis / coordinate transforms
- [ ] Matrix operation fusion
- [ ] SIMD-specific rewrites (horizontal ops, shuffles)
- [ ] Memory access patterns (gather → sequential)

### Research
- [ ] Auto-discovery of rules from benchmark data
- [ ] Domain-specific rule learning (graphics, ML, physics)

## Phased Rollout

**Phase 1: The Judge only**
- Train cost model on benchmarks
- Use in extraction (replace static weights)
- Measure: better final costs?

**Phase 2: The Guide (basic)**
- Train search heuristic against Judge
- Enable guided search
- Measure: handles 50+ rules without exploding?

**Phase 3: Rule expansion**
- Add differentiation rules
- Add trig rules
- Measure: Guide generalizes to new rule domains?

**Phase 4: Incremental accumulator**
- Implement delta-based updates
- Measure: speedup for heavy search?

**Phase 5: LLVM comparison**
- Benchmark against clang -O3 on numeric kernels
- Identify gaps
- Add targeted rules

## Files

| File | Purpose |
|------|---------|
| `pixelflow-ml/src/nnue.rs` | NNUE network, HalfEP features |
| `pixelflow-ml/src/training/egraph.rs` | Training loops |
| `pixelflow-ml/examples/guided_training.rs` | Guide training entry point |
| `pixelflow-ml/examples/train_cost_model.rs` | Judge training entry point |
| `pixelflow-search/src/egraph/best_first.rs` | Guided search implementation |
| `pixelflow-ml/data/benchmark_cache.jsonl` | Benchmark results |
| `pixelflow-ml/data/nnue_*_weights.bin` | Trained weights |

## Open Questions

1. **Embedding dimensionality**: 256 is borrowed from Stockfish. Optimal size for expression optimization unknown.

2. **Feature encoding for CAS**: HalfEP designed for algebraic ops. May need domain-specific features for trig, calculus.

3. **Rule interaction learning**: Can The Guide learn non-obvious interactions (e.g., "trig identity X enables differentiation shortcut Y")?

4. **Transfer across domains**: Does a Guide trained on graphics kernels transfer to ML kernels?

5. **Curriculum for rules**: Should we train on basic algebra first, then add differentiation, then trig? Or all at once?

## Success Criteria

1. **Minimum viable**: Handles 100+ rules without saturation explosion
2. **Useful**: Beats egg on large kernels with many rules
3. **Competitive**: Matches LLVM -O2 on numeric kernels
4. **Research win**: Beats LLVM -O3 on specific domains (graphics, autodiff)
