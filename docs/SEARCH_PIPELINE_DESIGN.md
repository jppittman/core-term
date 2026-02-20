# Search Pipeline Design

## Overview

This document defines clean interfaces for composable search strategies in e-graph optimization.
The goal is to enable A/B testing different approaches.

## The Conceptual Mess (What We're Cleaning Up)

We have multiple overlapping ideas that need to be organized:

### Search Strategies
| Name | Status | Core Idea |
|------|--------|-----------|
| **Best-First + ε-greedy** | In plan doc, implemented | Priority queue, random exploration |
| **MCTS (AlphaZero)** | "Abandoned" in doc, but implemented | Tree search with UCB, backprop |
| **Guided Search** | Implemented | Match prediction filter |

### Training Signals
| Name | Status | Core Idea |
|------|--------|-----------|
| **Curriculum (Supervised)** | In plan doc | Saturate small kernels → ground truth |
| **REINFORCE** | Implemented | Sparse reward, policy gradient |
| **Synthetic Oracle** | Implemented in train_mcts | Use judge to generate targets |

### Neural Components
| Name | Current Usage | Note |
|------|---------------|------|
| **Value Head** | Cost prediction | Used everywhere |
| **Mask/Policy Head** | Rule scoring | Plan doc says "not needed" |
| **Guide Net** | Match probability | Separate small network |

## The Three Orthogonal Axes

```
        SEARCH STRATEGY
            │
            ├── BestFirst (priority queue + ε-greedy)
            ├── MCTS (tree search + UCB)
            └── Exhaustive (saturation for small kernels)

        TRAINING SIGNAL
            │
            ├── Supervised (ground truth from saturation)
            ├── SyntheticOracle (value head as teacher)
            └── REINFORCE (sparse reward, policy gradient)

        NEURAL ARCHITECTURE
            │
            ├── ValueOnly (single head, plan doc preference)
            ├── DualHead (value + mask/policy)
            └── Guide (separate match predictor)
```

Any combination should be testable!

## Core Abstraction: The Search Problem

All approaches solve the same problem:
- **State**: E-graph with accumulated expressions
- **Actions**: Apply rewrite rule R at e-class C
- **Reward**: Extraction cost (lower is better)

The differences are:
- How actions are **proposed** (policy/search)
- How states are **evaluated** (value)
- How gradients **flow** (training signal)

---

## Implemented Architecture

See `pixelflow-search/src/search/mod.rs` for the actual trait definitions.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SEARCH PIPELINE                               │
│                                                                      │
│  DataSource → SearchStrategy → Evaluator → GradientSignal → Update  │
│      │              │              │              │            │     │
│      │              │              │              │            │     │
│   Generated     Exhaustive    NeuralValue   Supervised       SGD    │
│   BwdGenerated  BestFirst     Oracle        SyntheticOracle  NoOp   │
│   FromVec       Filtered                    REINFORCE               │
│   Chained       MCTS                        None                    │
│                 Identity                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Usage (like sklearn Pipeline)

```rust
use pixelflow_search::search::*;

// Compose a pipeline
let pipeline = Pipeline {
    data: GeneratedData::new(seed, config, Some(1000)),
    search: FilteredSearch::new(0.4, 0.1, 5, 100, seed),
    evaluator: NeuralEvaluator::new(model.clone()),
    gradient: ReinforceSignal::new(0.95),
    updater: SgdUpdate::new(&mut model),
};

// Train
for _ in 0..epochs {
    let (loss, improvement) = pipeline.step(batch_size, &rule_embeds, lr);
    println!("loss={:.4}, improvement={:.2}", loss, improvement);
}
```

---

## Trait Definitions

### 1. State Representation

```rust
/// Incrementally-updateable search state.
///
/// This is the "accumulator" abstraction - enables cheap MCTS simulation
/// without cloning the full e-graph.
pub trait SearchState: Clone {
    /// Action that can be applied to this state.
    type Action: Clone;

    /// Delta for incremental update/undo.
    type Delta;

    /// Apply an action, returning the delta for undo.
    fn apply(&mut self, action: &Self::Action) -> Self::Delta;

    /// Undo an action using its delta.
    fn undo(&mut self, delta: Self::Delta);

    /// Extract a feature embedding for neural network input.
    fn embed(&self) -> StateEmbedding;
}

/// Fixed-size embedding of search state.
pub struct StateEmbedding {
    /// Accumulator values [2*K = 64 dims].
    pub accumulator: [f32; 64],
    /// Structural features [21 dims].
    pub structural: [f32; 21],
}
```

### 2. Action Representation

```rust
/// An action in the search space.
#[derive(Clone, Copy, Debug)]
pub struct Action {
    /// E-class to apply rule at.
    pub class_id: EClassId,
    /// Rule index to apply.
    pub rule_idx: usize,
    /// (Optional) specific e-node within class.
    pub node_idx: Option<usize>,
}

/// Embedding of a rewrite rule for scoring.
pub struct RuleEmbedding {
    /// Rule embedding [EMBED_DIM = 24 dims].
    pub embed: [f32; 24],
}
```

### 3. Proposer (Policy)

```rust
/// Proposes actions given current state.
///
/// Different implementations:
/// - `MaskedProposer`: Scores (state, rule) pairs with bilinear interaction
/// - `MctsProposer`: MCTS tree search with neural priors
/// - `UniformProposer`: Proposes all rules (baseline)
pub trait Proposer {
    /// Score all possible actions from this state.
    /// Returns (action, logit) pairs.
    fn propose(&self, state: &dyn SearchState, available_rules: &[usize]) -> Vec<(Action, f32)>;

    /// Sample an action using scores (with temperature).
    fn sample(&self, scores: &[(Action, f32)], temperature: f32, rng: &mut impl Rng) -> Action;

    /// Select action deterministically (argmax).
    fn select(&self, scores: &[(Action, f32)]) -> Action;
}

/// Mask-based proposer using bilinear expr-rule scoring.
pub struct MaskedProposer {
    /// Expression embedding dimension.
    expr_embed: [f32; 24],
    /// Mask features (computed from expr_embed + value_pred).
    mask_features: [f32; 24],
    /// Pre-computed rule embeddings.
    rule_embeds: Vec<[f32; 24]>,
    /// Bilinear interaction matrix.
    interaction: [[f32; 24]; 24],
    /// Per-rule bias.
    rule_bias: [f32; 1024],
    /// Threshold for approval.
    threshold: f32,
}

/// MCTS-based proposer with tree search.
pub struct MctsProposer {
    /// Number of simulations per move.
    num_simulations: usize,
    /// Exploration constant (c_puct).
    c_puct: f32,
    /// Temperature for action selection.
    temperature: f32,
    /// Neural network for policy priors.
    policy_net: Arc<dyn PolicyNet>,
    /// Neural network for value estimation.
    value_net: Arc<dyn ValueNet>,
}
```

### 4. Evaluator (Value)

```rust
/// Evaluates state quality.
///
/// Different implementations:
/// - `NeuralEvaluator`: Learned value head
/// - `OracleEvaluator`: Run full search as ground truth
/// - `CostEvaluator`: Direct extraction cost
pub trait Evaluator {
    /// Estimate the value (negative cost) of a state.
    fn evaluate(&self, state: &dyn SearchState) -> f32;

    /// Batch evaluation for efficiency.
    fn evaluate_batch(&self, states: &[&dyn SearchState]) -> Vec<f32>;
}

/// Neural network value head.
pub struct NeuralEvaluator {
    /// Shared backbone (computes expr_embed from state).
    backbone: Arc<Backbone>,
    /// Value MLP: 24 -> 16 -> 1.
    value_mlp: ValueMLP,
}

/// Oracle evaluator: runs full search to get ground truth.
pub struct OracleEvaluator {
    /// Maximum e-graph classes.
    max_classes: usize,
    /// Maximum search epochs.
    max_epochs: usize,
    /// Cost function for extraction.
    cost_fn: Arc<dyn CostFunction>,
}
```

### 5. Trainer (Gradient Computation)

```rust
/// Computes gradients from search trajectories.
///
/// Different implementations:
/// - `SyntheticTrainer`: Uses oracle value as ground truth (MCTS style)
/// - `ReinforceTrainer`: Policy gradient with sparse rewards
/// - `SupervisedTrainer`: Direct labels from data collection
pub trait Trainer {
    /// Update model from a batch of trajectories.
    fn train_step(
        &mut self,
        model: &mut ExprNnue,
        trajectories: &[Trajectory],
        learning_rate: f32,
    ) -> TrainMetrics;
}

/// A search trajectory for training.
pub struct Trajectory {
    /// Sequence of (state, action, logit) tuples.
    pub steps: Vec<TrajectoryStep>,
    /// Final cost (reward signal).
    pub final_cost: f32,
    /// Oracle cost (if available).
    pub oracle_cost: Option<f32>,
}

pub struct TrajectoryStep {
    pub state: StateEmbedding,
    pub action: Action,
    pub logit: f32,
    pub value_pred: f32,
}

/// REINFORCE trainer with baseline subtraction.
pub struct ReinforceTrainer {
    /// Baseline for variance reduction.
    baseline: RewardBaseline,
    /// Baseline decay rate.
    decay: f32,
}

/// Synthetic oracle trainer (MCTS style).
pub struct SyntheticTrainer {
    /// Policy temperature for target distribution.
    policy_temperature: f32,
    /// Number of actions to sample per state.
    actions_per_state: usize,
}
```

---

## Composed Pipelines

### Pipeline 1: AlphaZero MCTS

```rust
let pipeline = SearchPipeline {
    proposer: MctsProposer::new(
        num_simulations: 64,
        c_puct: 1.5,
        policy_net: model.clone(),
        value_net: model.clone(),
    ),
    evaluator: NeuralEvaluator::new(model.clone()),
    trainer: SyntheticTrainer::new(
        policy_temperature: 0.5,
        actions_per_state: 16,
    ),
};
```

### Pipeline 2: REINFORCE Filtering

```rust
let pipeline = SearchPipeline {
    proposer: MaskedProposer::new(
        model.clone(),
        threshold: 0.4,
    ),
    evaluator: OracleEvaluator::new(
        max_classes: 200,
        max_epochs: 15,
    ),
    trainer: ReinforceTrainer::new(
        baseline_decay: 0.95,
    ),
};
```

### Pipeline 3: Guided Search (Match Prediction)

```rust
let pipeline = SearchPipeline {
    proposer: GuideProposer::new(
        guide_net: GuideNnue::load("guide.bin"),
        threshold: 0.3,
    ),
    evaluator: CostEvaluator::new(costs),
    trainer: SupervisedTrainer::new(
        // Binary classification: did rule match?
        loss_fn: AsymmetricBCE { fp_weight: 2.0, fn_weight: 0.5 },
    ),
};
```

---

## Key Design Decisions

### 1. State is Incremental

All approaches use `EdgeAccumulator` + `AccumulatorDelta` for cheap simulation.
This is critical for MCTS where we simulate thousands of trajectories.

### 2. Proposer Owns Policy Logic

The `Proposer` trait encapsulates:
- How actions are scored (bilinear, softmax, tree search)
- How actions are selected (threshold, sampling, argmax)
- What neural network components are used

### 3. Evaluator is Swappable

Oracle vs learned value head is a key experimental variable.
- Oracle: slower but provides ground truth
- Learned: faster but may be inaccurate

### 4. Trainer Handles Gradient Flow

The critical difference between approaches:
- **MCTS**: Full backprop through oracle-generated targets
- **REINFORCE**: Policy gradient with baseline subtraction
- **Supervised**: Direct backprop through labels

### 5. Composition over Inheritance

Each component can be mixed and matched:
```rust
// MCTS proposer + REINFORCE training
let hybrid = SearchPipeline {
    proposer: MctsProposer::new(...),
    trainer: ReinforceTrainer::new(...),  // Use MCTS decisions as samples
};
```

---

## Migration Path

### Phase 1: Extract Traits
1. Define trait files in `pixelflow-search/src/search/`
2. Implement for existing code without changing behavior

### Phase 2: Refactor Binaries
1. `train_mcts.rs` -> uses `SyntheticTrainer`
2. `train_mask_reinforce.rs` -> uses `ReinforceTrainer`
3. `train_guide.rs` -> uses `SupervisedTrainer`

### Phase 3: A/B Testing
1. Create unified `train_search.rs` that accepts config
2. Config specifies which proposer/evaluator/trainer to use
3. Easy comparison of approaches on same data

---

---

## Pipeline Phases

The search/training pipeline has clear phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA GENERATION                                            │
│  ───────────────────────                                             │
│  Input: Expression generator (Expr, BwdGenerator, from file)         │
│  Output: Vec<Expr>                                                   │
│                                                                      │
│  Configurable: max_depth, count, seed                                │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: SEARCH (one of)                                            │
│  ─────────────────────────                                           │
│                                                                      │
│  A) Exhaustive (Saturation)                                          │
│     - Apply all rules until fixed point                              │
│     - Use for small kernels to get ground truth                      │
│     - Output: optimal ExprTree + cost                                │
│                                                                      │
│  B) Best-First (A* with ε-greedy)                                    │
│     - Priority queue ordered by value prediction                     │
│     - ε fraction random for exploration                              │
│     - Output: best found ExprTree + trajectory                       │
│                                                                      │
│  C) MCTS (AlphaZero style)                                           │
│     - Tree search with UCB action selection                          │
│     - Neural policy priors + value estimates                         │
│     - Output: improved policy + best ExprTree                        │
│                                                                      │
│  D) Filtered (Mask/Guide)                                            │
│     - Predict which rules will match/help                            │
│     - Only apply predicted-good rules                                │
│     - Output: ExprTree + filter decisions                            │
│                                                                      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: EVALUATION                                                 │
│  ──────────────────────                                              │
│                                                                      │
│  A) Oracle (Ground Truth)                                            │
│     - Run saturation or expensive search                             │
│     - Slow but accurate                                              │
│                                                                      │
│  B) Neural Value Head                                                │
│     - Instant cost prediction from embedding                         │
│     - Fast but may be wrong                                          │
│                                                                      │
│  C) Benchmark                                                        │
│     - Compile and run actual SIMD code                               │
│     - Slowest but ground truth                                       │
│                                                                      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: GRADIENT COMPUTATION                                       │
│  ─────────────────────────────                                       │
│                                                                      │
│  A) Supervised                                                       │
│     - Target: ground truth cost from saturation                      │
│     - Loss: MSE(predicted, actual)                                   │
│     - Full backprop through value head                               │
│                                                                      │
│  B) Synthetic Oracle (MCTS-style)                                    │
│     - Sample actions, score with value head                          │
│     - Target: softmax(-deltas) as policy distribution                │
│     - Loss: cross-entropy(policy, target)                            │
│                                                                      │
│  C) REINFORCE                                                        │
│     - Reward: 1 if student <= oracle, else 0                         │
│     - Gradient: reward * grad_log_prob                               │
│     - Baseline subtraction for variance reduction                    │
│                                                                      │
│  D) None (inference only)                                            │
│                                                                      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: WEIGHT UPDATE                                              │
│  ──────────────────────                                              │
│                                                                      │
│  Apply gradients to model with learning rate, momentum, etc.         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Concrete Experiment Configurations

### Experiment 1: Curriculum Learning (Plan Doc Preference)

```rust
let config = PipelineConfig {
    // Phase 1: Generate small kernels
    data: DataConfig::Generated {
        generator: ExprGenerator::new(seed, ExprGenConfig { max_depth: 4 }),
        count: 1000,
    },

    // Phase 2: Exhaust (saturate) for ground truth
    search: SearchConfig::Exhaustive { max_classes: 1000 },

    // Phase 3: No separate eval (saturation IS the eval)
    eval: EvalConfig::FromSearch,

    // Phase 4: Supervised learning
    gradient: GradientConfig::Supervised { loss: Mse },

    // No policy head needed
    arch: ArchConfig::ValueOnly,
};
```

### Experiment 2: AlphaZero MCTS

```rust
let config = PipelineConfig {
    // Phase 1: Generate medium kernels
    data: DataConfig::Generated {
        generator: BwdGenerator::new(seed, BwdGenConfig { max_depth: 8 }),
        count: 100,
    },

    // Phase 2: MCTS with neural guidance
    search: SearchConfig::Mcts {
        simulations: 64,
        c_puct: 1.5,
        temperature: 1.0,
    },

    // Phase 3: Value head predicts cost
    eval: EvalConfig::NeuralValue,

    // Phase 4: Synthetic oracle targets
    gradient: GradientConfig::SyntheticOracle {
        actions_per_state: 16,
        policy_temperature: 0.5,
    },

    // Dual-head architecture
    arch: ArchConfig::DualHead { mask_input_dim: 25 },
};
```

### Experiment 3: REINFORCE Filtering

```rust
let config = PipelineConfig {
    // Phase 1: Load from file
    data: DataConfig::FromFile { path: "judge_training.jsonl" },

    // Phase 2: Filtered search with mask
    search: SearchConfig::Filtered {
        threshold: 0.4,
        epsilon: 0.2, // Exploration
        max_epochs: 5,
    },

    // Phase 3: Oracle with abundant resources
    eval: EvalConfig::Oracle {
        max_classes: 200,
        max_epochs: 15,
    },

    // Phase 4: REINFORCE
    gradient: GradientConfig::Reinforce {
        baseline_decay: 0.95,
    },

    // Dual-head with mask
    arch: ArchConfig::DualHead { mask_input_dim: 25 },
};
```

---

## Open Questions

1. **NnueCache Integration**: Should cache be part of Search phase or separate?
   - Pro: Search owns scoring, cache accelerates scoring
   - Con: Cache is mutable, complicates phase boundaries

2. **E-Graph Mutation**: How does Search interact with actual e-graph?
   - Option A: Search returns actions, separate executor applies them
   - Option B: Search holds mutable e-graph reference (current approach)

3. **Batch vs Single**: Should phases be batch-first or single-first?
   - Batch: More efficient for GPU
   - Single: Simpler, more composable

4. **Lifetime Management**: How to share model across phases?
   - `Arc<ExprNnue>`: Simple, some overhead
   - Generics: Complex trait bounds
