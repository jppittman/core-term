//! Curriculum Training for NNUE E-Graph Cost Prediction.
//!
//! This implements **Curriculum Bootstrapping**: train NNUE via two phases
//! instead of complex MCTS/RL.
//!
//! ## Architecture: Curriculum Bootstrapping
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  PHASE 1: KINDERGARTEN (Saturation = Ground Truth)              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  • Generate TINY kernels (10-50 nodes)                          │
//! │  • SATURATE completely (brute force - affordable for small)     │
//! │  • Get PERFECT optimal cost                                     │
//! │  • Train NNUE: "This graph has potential cost X"                │
//! │  • Result: Network learns basic algebra (x*0=0, x+0=x, etc.)    │
//! └─────────────────────────────────────────────────────────────────┘
//!                               ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  PHASE 2: UNIVERSITY (Guided Search + Noise)                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  • Generate LARGE kernels (100+ nodes)                          │
//! │  • Saturation IMPOSSIBLE                                        │
//! │  • Use Best-First Search (A*) with trained NNUE                 │
//! │  • ε-greedy exploration (10% random) to escape local minima     │
//! │  • Network already knows basics, now learns deeper tricks       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Simplifications over MCTS
//!
//! | Old (MCTS Complexity) | New (Curriculum + A*) |
//! |-----------------------|-----------------------|
//! | UCB1 formula          | Priority queue        |
//! | Tree backpropagation  | Direct supervised learning |
//! | Visit counts          | ε-greedy exploration  |
//! | Policy + Value heads  | Single value network  |
//! | Online RL from scratch| Curriculum: small→large |
//!
//! ## Usage
//!
//! ```bash
//! # Quick test (5 epochs)
//! cargo run -p pixelflow-ml --example guided_training --features training --release -- --quick
//!
//! # Full training
//! cargo run -p pixelflow-ml --example guided_training --features training --release
//!
//! # Kindergarten only (ground truth training)
//! cargo run -p pixelflow-ml --example guided_training --features training --release -- --kindergarten-only
//! ```

use std::time::Instant;

use pixelflow_search::egraph::{
    ExprTree, GuidedConfig, GuidedMcts, GuidedAction, GuidedState,
    BestFirstPlanner, BestFirstConfig, CostModel,
};
use pixelflow_nnue::{Nnue, NnueConfig, HalfEPFeature, Accumulator, OpType};

// ============================================================================
// Simple LCG Random Number Generator
// ============================================================================

/// Simple LCG random number generator (avoids rand dependency).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }

    fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        let len = range.end - range.start;
        range.start + (self.next_u64() as usize % len)
    }

    fn gen_bool(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }

    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

// ============================================================================
// Training Data Structures
// ============================================================================

/// A training sample for NNUE cost prediction.
///
/// Used for both:
/// - MCTS episodes (legacy mode): action field populated
/// - Curriculum training: action field is None
///
/// The target is the FINAL best cost achieved, used to train Q(s) predictions.
#[derive(Debug, Clone)]
struct TrainingSample {
    /// Features representing the e-graph state
    features: Vec<HalfEPFeature>,
    /// The action that was taken (for MCTS debugging, None for curriculum)
    #[allow(dead_code)]
    action: Option<GuidedAction>,
    /// The FINAL best cost achieved at the end of optimization.
    final_cost: usize,
    /// Initial cost before optimization (for normalization)
    initial_cost: usize,
}

/// Experience replay buffer for training.
struct ReplayBuffer {
    samples: Vec<TrainingSample>,
    max_size: usize,
    rng: Rng,
}

impl ReplayBuffer {
    fn new(max_size: usize, seed: u64) -> Self {
        Self {
            samples: Vec::new(),
            max_size,
            rng: Rng::new(seed),
        }
    }

    fn add(&mut self, sample: TrainingSample) {
        if self.samples.len() >= self.max_size {
            // Remove random old sample (reservoir sampling variant)
            let idx = self.rng.gen_range(0..self.samples.len());
            self.samples.swap_remove(idx);
        }
        self.samples.push(sample);
    }

    fn sample_batch(&mut self, batch_size: usize) -> Vec<TrainingSample> {
        let n = self.samples.len().min(batch_size);
        let mut indices: Vec<usize> = (0..self.samples.len()).collect();

        // Fisher-Yates partial shuffle
        for i in 0..n {
            let j = self.rng.gen_range(i..self.samples.len());
            indices.swap(i, j);
        }

        indices[..n].iter().map(|&i| self.samples[i].clone()).collect()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

// ============================================================================
// Expression Generation
// ============================================================================

/// Generate random expression trees for training.
///
/// Supports two modes:
/// - **Small kernels** (10-50 nodes): For Phase 1 ground truth via saturation
/// - **Large kernels** (100+ nodes): For Phase 2 guided search
struct ExprGenerator {
    rng: Rng,
    max_depth: usize,
    num_vars: usize,
}

impl ExprGenerator {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            max_depth: 5,
            num_vars: 4,
        }
    }

    /// Generate a small kernel (10-50 nodes) for Phase 1 saturation.
    ///
    /// These are small enough to saturate completely, giving perfect ground truth.
    fn generate_small(&mut self) -> ExprTree {
        self.max_depth = 3 + self.rng.gen_range(0..2); // depth 3-4
        self.generate_inner(0)
    }

    /// Generate a large kernel (100+ nodes) for Phase 2 guided search.
    ///
    /// These are too large to saturate, requiring NNUE-guided search.
    fn generate_large(&mut self) -> ExprTree {
        self.max_depth = 6 + self.rng.gen_range(0..3); // depth 6-8
        self.generate_inner(0)
    }

    fn generate(&mut self) -> ExprTree {
        self.generate_inner(0)
    }

    fn generate_inner(&mut self, depth: usize) -> ExprTree {
        // Leaf probability increases with depth
        let leaf_prob = 0.2 + (depth as f64 * 0.15);

        if depth >= self.max_depth || self.rng.gen_f64() < leaf_prob {
            // Generate leaf
            if self.rng.gen_bool() {
                ExprTree::Var(self.rng.gen_range(0..self.num_vars) as u8)
            } else {
                let constants = [0.0, 1.0, -1.0, 2.0, 0.5];
                ExprTree::Const(constants[self.rng.gen_range(0..constants.len())])
            }
        } else {
            // Generate operation
            let op_type = self.rng.gen_range(0..6);
            match op_type {
                0 => ExprTree::Add(
                    Box::new(self.generate_inner(depth + 1)),
                    Box::new(self.generate_inner(depth + 1)),
                ),
                1 => ExprTree::Sub(
                    Box::new(self.generate_inner(depth + 1)),
                    Box::new(self.generate_inner(depth + 1)),
                ),
                2 => ExprTree::Mul(
                    Box::new(self.generate_inner(depth + 1)),
                    Box::new(self.generate_inner(depth + 1)),
                ),
                3 => ExprTree::Neg(Box::new(self.generate_inner(depth + 1))),
                4 => ExprTree::Sqrt(Box::new(self.generate_inner(depth + 1))),
                _ => ExprTree::Abs(Box::new(self.generate_inner(depth + 1))),
            }
        }
    }
}

// ============================================================================
// Feature Extraction from E-Graph State
// ============================================================================

/// Map ExprTree variant to OpType for feature encoding.
///
/// OpType has 14 variants, so we bucket the many ExprTree variants into
/// semantically similar groups.
fn expr_tree_to_op_type(tree: &ExprTree) -> OpType {
    match tree {
        ExprTree::Var(_) => OpType::Var,
        ExprTree::Const(_) => OpType::Const,
        ExprTree::Add(_, _) => OpType::Add,
        ExprTree::Sub(_, _) => OpType::Sub,
        ExprTree::Mul(_, _) => OpType::Mul,
        ExprTree::Div(_, _) => OpType::Div,
        ExprTree::Neg(_) => OpType::Neg,
        ExprTree::Sqrt(_) => OpType::Sqrt,
        ExprTree::Rsqrt(_) => OpType::Rsqrt,
        ExprTree::Abs(_) => OpType::Abs,
        ExprTree::Min(_, _) => OpType::Min,
        ExprTree::Max(_, _) => OpType::Max,
        ExprTree::MulAdd(_, _, _) => OpType::MulAdd,
        // Map other ops to closest OpType
        ExprTree::Recip(_) => OpType::Div,  // 1/x is like division
        // Rounding ops grouped with Abs (value manipulation)
        ExprTree::Floor(_) | ExprTree::Ceil(_) | ExprTree::Round(_) | ExprTree::Fract(_) => OpType::Abs,
        // Transcendentals grouped with Sqrt (expensive unary)
        ExprTree::Sin(_) | ExprTree::Cos(_) | ExprTree::Tan(_)
        | ExprTree::Asin(_) | ExprTree::Acos(_) | ExprTree::Atan(_)
        | ExprTree::Exp(_) | ExprTree::Exp2(_)
        | ExprTree::Ln(_) | ExprTree::Log2(_) | ExprTree::Log10(_) => OpType::Sqrt,
        // Binary transcendentals grouped with MulRsqrt (expensive binary)
        ExprTree::Atan2(_, _) | ExprTree::Pow(_, _) | ExprTree::Hypot(_, _) => OpType::MulRsqrt,
        // Comparisons grouped with Sub (produces difference-like values)
        ExprTree::Lt(_, _) | ExprTree::Le(_, _) | ExprTree::Gt(_, _)
        | ExprTree::Ge(_, _) | ExprTree::Eq(_, _) | ExprTree::Ne(_, _) => OpType::Sub,
        // Ternary ops grouped with MulAdd
        ExprTree::Select(_, _, _) | ExprTree::Clamp(_, _, _) => OpType::MulAdd,
        // Tuple is special - treat as const (structural)
        ExprTree::Tuple(_) => OpType::Const,
    }
}

/// Compute a structural hash for an ExprTree subtree.
/// Used to detect shared/duplicate subexpressions.
fn structural_hash(tree: &ExprTree) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Hash the discriminant (which variant)
    std::mem::discriminant(tree).hash(&mut hasher);

    match tree {
        ExprTree::Var(v) => v.hash(&mut hasher),
        ExprTree::Const(c) => c.to_bits().hash(&mut hasher),
        ExprTree::Neg(a) | ExprTree::Recip(a) | ExprTree::Sqrt(a) | ExprTree::Rsqrt(a)
        | ExprTree::Abs(a) | ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
        | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
        | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a)
        | ExprTree::Exp(a) | ExprTree::Exp2(a) | ExprTree::Ln(a)
        | ExprTree::Log2(a) | ExprTree::Log10(a) => {
            structural_hash(a).hash(&mut hasher);
        }
        ExprTree::Add(a, b) | ExprTree::Sub(a, b) | ExprTree::Mul(a, b) | ExprTree::Div(a, b)
        | ExprTree::Min(a, b) | ExprTree::Max(a, b) | ExprTree::Atan2(a, b)
        | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b)
        | ExprTree::Lt(a, b) | ExprTree::Le(a, b) | ExprTree::Gt(a, b)
        | ExprTree::Ge(a, b) | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
            structural_hash(a).hash(&mut hasher);
            structural_hash(b).hash(&mut hasher);
        }
        ExprTree::MulAdd(a, b, c) | ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
            structural_hash(a).hash(&mut hasher);
            structural_hash(b).hash(&mut hasher);
            structural_hash(c).hash(&mut hasher);
        }
        ExprTree::Tuple(elems) => {
            for elem in elems {
                structural_hash(elem).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Special feature indicating a back-reference to an already-seen subexpression.
/// This is CRITICAL for proper CSE awareness:
/// - `let x = expensive(); x + x` → [expensive, Add, BackRef] (cheap!)
/// - `expensive() + expensive()` → [expensive, Add, expensive] (expensive!)
const BACKREF_FEATURE: HalfEPFeature = HalfEPFeature {
    perspective_op: 15,  // Invalid OpType - signals special feature
    descendant_op: 15,
    depth: 0,
    path: 255,
};

/// Extract HalfEP features from an ExprTree with CSE awareness.
///
/// This extracts features using a DAG-walk instead of tree-walk:
/// - First visit to a subtree: extract full features
/// - Subsequent visits to structurally-identical subtrees: emit BackRef token
///
/// This lets NNUE learn that shared subexpressions are cheap.
fn extract_tree_features(tree: &ExprTree) -> Vec<HalfEPFeature> {
    let mut features = Vec::new();
    let mut visited = std::collections::HashSet::new();
    extract_tree_features_dag(tree, &mut features, &mut visited);
    features
}

fn extract_tree_features_dag(
    tree: &ExprTree,
    features: &mut Vec<HalfEPFeature>,
    visited: &mut std::collections::HashSet<u64>,
) {
    let hash = structural_hash(tree);

    // CSE check: have we seen this exact subtree before?
    if !visited.insert(hash) {
        // Already visited! Emit BackRef instead of recursing.
        // This is how NNUE learns that `x + x` (shared) is cheaper than
        // `expensive() + expensive()` (duplicated work).
        features.push(BACKREF_FEATURE);
        return;
    }

    let root_op = expr_tree_to_op_type(tree);

    // Add features for all descendants from this node's perspective
    add_tree_descendant_features(tree, features, root_op.index() as u8, 0, 0);

    // Recurse into children (DAG-aware)
    match tree {
        ExprTree::Var(_) | ExprTree::Const(_) => {}
        // Unary operations
        ExprTree::Neg(a) | ExprTree::Recip(a) | ExprTree::Sqrt(a) | ExprTree::Rsqrt(a)
        | ExprTree::Abs(a) | ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
        | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
        | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a)
        | ExprTree::Exp(a) | ExprTree::Exp2(a) | ExprTree::Ln(a)
        | ExprTree::Log2(a) | ExprTree::Log10(a) => {
            extract_tree_features_dag(a, features, visited);
        }
        // Binary operations
        ExprTree::Add(a, b) | ExprTree::Sub(a, b) | ExprTree::Mul(a, b) | ExprTree::Div(a, b)
        | ExprTree::Min(a, b) | ExprTree::Max(a, b) | ExprTree::Atan2(a, b)
        | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b)
        | ExprTree::Lt(a, b) | ExprTree::Le(a, b) | ExprTree::Gt(a, b)
        | ExprTree::Ge(a, b) | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
            extract_tree_features_dag(a, features, visited);
            extract_tree_features_dag(b, features, visited);
        }
        // Ternary operations
        ExprTree::MulAdd(a, b, c) | ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
            extract_tree_features_dag(a, features, visited);
            extract_tree_features_dag(b, features, visited);
            extract_tree_features_dag(c, features, visited);
        }
        // Tuple: recurse into all elements
        ExprTree::Tuple(elems) => {
            for elem in elems {
                extract_tree_features_dag(elem, features, visited);
            }
        }
    }
}

fn add_tree_descendant_features(
    tree: &ExprTree,
    features: &mut Vec<HalfEPFeature>,
    perspective_op: u8,
    depth: u8,
    path: u8,
) {
    const MAX_DEPTH: u8 = 8;
    if depth >= MAX_DEPTH {
        return;
    }

    // Add feature for this node
    let descendant_op = expr_tree_to_op_type(tree).index() as u8;
    features.push(HalfEPFeature {
        perspective_op,
        descendant_op,
        depth,
        path,
    });

    // Recurse into children with updated path
    match tree {
        ExprTree::Var(_) | ExprTree::Const(_) => {}
        // Unary operations
        ExprTree::Neg(a) | ExprTree::Recip(a) | ExprTree::Sqrt(a) | ExprTree::Rsqrt(a)
        | ExprTree::Abs(a) | ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
        | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
        | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a)
        | ExprTree::Exp(a) | ExprTree::Exp2(a) | ExprTree::Ln(a)
        | ExprTree::Log2(a) | ExprTree::Log10(a) => {
            add_tree_descendant_features(a, features, perspective_op, depth + 1, path << 1);
        }
        // Binary operations
        ExprTree::Add(a, b) | ExprTree::Sub(a, b) | ExprTree::Mul(a, b) | ExprTree::Div(a, b)
        | ExprTree::Min(a, b) | ExprTree::Max(a, b) | ExprTree::Atan2(a, b)
        | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b)
        | ExprTree::Lt(a, b) | ExprTree::Le(a, b) | ExprTree::Gt(a, b)
        | ExprTree::Ge(a, b) | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
            add_tree_descendant_features(a, features, perspective_op, depth + 1, path << 1);
            add_tree_descendant_features(b, features, perspective_op, depth + 1, (path << 1) | 1);
        }
        // Ternary operations: use bits 0, 1, 2 for the three children
        ExprTree::MulAdd(a, b, c) | ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
            add_tree_descendant_features(a, features, perspective_op, depth + 1, path << 2);
            add_tree_descendant_features(b, features, perspective_op, depth + 1, (path << 2) | 1);
            add_tree_descendant_features(c, features, perspective_op, depth + 1, (path << 2) | 2);
        }
        // Tuple: index-based path encoding
        ExprTree::Tuple(elems) => {
            for (i, elem) in elems.iter().enumerate() {
                let child_path = path.wrapping_shl(4) | (i as u8 & 0x0F);
                add_tree_descendant_features(elem, features, perspective_op, depth + 1, child_path);
            }
        }
    }
}

/// Extract features from the current guided state for NNUE evaluation.
///
/// Combines structural features from the best expression tree with
/// contextual features about the action being considered.
fn extract_state_features(
    state: &GuidedState,
    action: &GuidedAction,
    iterations: usize,
) -> Vec<HalfEPFeature> {
    let mut features = Vec::new();

    // Primary features: HalfEP features from the current best expression
    features.extend(extract_tree_features(state.best_tree()));

    // Contextual features about the search state and action
    // Feature: E-graph size bucket (0-13 to fit in perspective_op range)
    let size_bucket = (state.num_classes() / 4).min(13) as u8;
    features.push(HalfEPFeature {
        perspective_op: size_bucket,
        descendant_op: 0,
        depth: 0,
        path: 128,  // High bit set to distinguish from tree features
    });

    // Feature: Current cost bucket
    let cost_bucket = (state.best_cost() / 10).min(13) as u8;
    features.push(HalfEPFeature {
        perspective_op: 0,
        descendant_op: cost_bucket,
        depth: 1,
        path: 128,
    });

    // Feature: Rule type being considered
    let rule_bucket = (action.rule_idx % 14) as u8;
    features.push(HalfEPFeature {
        perspective_op: rule_bucket,
        descendant_op: 0,
        depth: 2,
        path: 128,
    });

    // Feature: Target class (combined hash for uniqueness)
    let class_bucket = ((action.node_idx + action.rule_idx) % 14) as u8;
    features.push(HalfEPFeature {
        perspective_op: 0,
        descendant_op: class_bucket,
        depth: 3,
        path: 128,
    });

    // Feature: Iteration count bucket (search progress)
    let iter_bucket = (iterations / 10).min(7) as u8;
    features.push(HalfEPFeature {
        perspective_op: 0,
        descendant_op: 0,
        depth: iter_bucket,
        path: 129,
    });

    features
}

/// Evaluate a state-action pair using NNUE.
///
/// Returns a Q-value estimate in the range [-1, 1].
fn evaluate_action_with_nnue(
    nnue: &Nnue,
    state: &GuidedState,
    action: &GuidedAction,
    iterations: usize,
) -> f64 {
    let features = extract_state_features(state, action, iterations);

    let mut acc = Accumulator::new(nnue);
    for f in &features {
        let idx = f.to_index();
        if idx < HalfEPFeature::COUNT {
            acc.add_feature(nnue, idx);
        }
    }

    // Forward pass and scale to [-1, 1] range
    let raw_score = acc.forward(nnue);
    // Tanh-like squashing to [-1, 1]
    (raw_score as f64 / 1000.0).tanh()
}

// ============================================================================
// Training Loop
// ============================================================================

/// Training configuration.
struct TrainConfig {
    /// Number of training epochs.
    epochs: usize,
    /// Episodes per epoch (expressions to optimize).
    episodes_per_epoch: usize,
    /// MCTS iterations per episode.
    mcts_iterations: usize,
    /// Epsilon for ε-greedy exploration.
    epsilon: f32,
    /// Learning rate.
    learning_rate: f32,
    /// Batch size for gradient updates.
    batch_size: usize,
    /// Replay buffer size.
    replay_buffer_size: usize,
    /// Random seed.
    seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            episodes_per_epoch: 100,
            mcts_iterations: 200,
            epsilon: 0.2,  // 20% random exploration
            learning_rate: 0.001,
            batch_size: 64,
            replay_buffer_size: 10000,
            seed: 42,
        }
    }
}

/// Intermediate data collected during an episode (before we know final cost).
struct TrajectoryStep {
    features: Vec<HalfEPFeature>,
    action: GuidedAction,
}

/// Run one MCTS episode and collect training samples.
///
/// CRITICAL: We collect (state, action) pairs during the trajectory, then
/// assign the FINAL best cost as the target for ALL samples. This is
/// Monte Carlo return - Q(s,a) learns to predict the eventual outcome.
///
/// Example: sin(v)*cos(v) may require "expensive" intermediate rewrites
/// before reaching sin(2v)/2. Those intermediate actions should have
/// HIGH Q-values because they lead to the optimal final result.
fn run_episode(
    tree: &ExprTree,
    config: &TrainConfig,
    nnue: &Nnue,
) -> Vec<TrainingSample> {
    let guided_config = GuidedConfig::default()
        .with_iterations(config.mcts_iterations)
        .with_epsilon(config.epsilon as f64)
        .training_mode();

    let mut mcts = GuidedMcts::from_tree(tree, guided_config);
    let initial_cost = mcts.best_cost();

    // Phase 1: Collect trajectory (state, action) pairs WITHOUT rewards
    let mut trajectory: Vec<TrajectoryStep> = Vec::new();

    while mcts.iterations() < config.mcts_iterations {
        let actions = mcts.state().available_actions();
        if actions.is_empty() {
            break;
        }

        let iter_before = mcts.iterations();

        // Extract features BEFORE taking action (this is the state we're in)
        // We need to pick a representative action for feature extraction
        // Use the first available action as proxy for state features
        let state_features = extract_state_features(
            mcts.state(),
            &actions[0],
            iter_before,
        );

        let action_result = mcts.iterate_with_evaluator(|state, action| {
            evaluate_action_with_nnue(nnue, state, action, iter_before)
        });

        if let Some((action, _was_improvement)) = action_result {
            trajectory.push(TrajectoryStep {
                features: state_features,
                action,
            });
        }

        // Early exit if we've reached minimal cost
        if mcts.best_cost() <= 1 {
            break;
        }
    }

    // Phase 2: Get the FINAL best cost achieved
    let final_cost = mcts.best_cost();

    // Phase 3: Convert trajectory to training samples with final cost as target
    // Every (state, action) pair in this trajectory gets the SAME target:
    // the final best cost we eventually achieved.
    trajectory
        .into_iter()
        .map(|step| TrainingSample {
            features: step.features,
            action: Some(step.action),
            final_cost,
            initial_cost,
        })
        .collect()
}

// ============================================================================
// Full Backpropagation Implementation
// ============================================================================

/// Intermediate activations stored during forward pass for backprop.
struct ForwardState {
    /// L1 activations (after accumulator, before clipped ReLU)
    l1_pre: Vec<i32>,
    /// L1 activations (after clipped ReLU, as f32 for gradient computation)
    l1_post: Vec<f32>,
    /// L2 activations (before clipped ReLU)
    l2_pre: Vec<i32>,
    /// L2 activations (after clipped ReLU)
    l2_post: Vec<f32>,
    /// L3 activations (before clipped ReLU)
    l3_pre: Vec<i32>,
    /// L3 activations (after clipped ReLU)
    l3_post: Vec<f32>,
    /// Active feature indices (for sparse gradient update)
    active_features: Vec<usize>,
}

/// Clipped ReLU: (x >> 6).clamp(0, 127)
/// Returns both the output and the derivative mask
#[inline]
fn clipped_relu(x: i32) -> (f32, f32) {
    let shifted = x >> 6;
    let clamped = shifted.clamp(0, 127);
    // Derivative is 1.0 if in active region (0 < shifted < 127), else 0.0
    let deriv = if shifted > 0 && shifted < 127 { 1.0 } else { 0.0 };
    (clamped as f32, deriv)
}

/// Forward pass that stores intermediate activations for backprop.
fn forward_with_state(
    nnue: &Nnue,
    features: &[HalfEPFeature],
) -> (f32, ForwardState) {
    let l1_size = nnue.config.l1_size;
    let l2_size = nnue.config.l2_size;
    let l3_size = nnue.config.l3_size;

    // Collect active features
    let active_features: Vec<usize> = features
        .iter()
        .map(|f| f.to_index())
        .filter(|&idx| idx < HalfEPFeature::COUNT)
        .collect();

    // L1: Sparse input via accumulator
    let mut l1_pre = nnue.b1.clone();
    for &feature_idx in &active_features {
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            l1_pre[i] += nnue.w1[offset + i] as i32;
        }
    }

    // L1 clipped ReLU
    let mut l1_post = Vec::with_capacity(l1_size);
    let mut l1_deriv = Vec::with_capacity(l1_size);
    for &x in &l1_pre {
        let (out, deriv) = clipped_relu(x);
        l1_post.push(out);
        l1_deriv.push(deriv);
    }

    // L2: Dense layer
    let mut l2_pre = nnue.b2.clone();
    for i in 0..l1_size {
        let a = l1_post[i] as i8 as i32;
        for j in 0..l2_size {
            l2_pre[j] += a * (nnue.w2[i * l2_size + j] as i32);
        }
    }

    // L2 clipped ReLU
    let mut l2_post = Vec::with_capacity(l2_size);
    for &x in &l2_pre {
        let (out, _) = clipped_relu(x);
        l2_post.push(out);
    }

    // L3: Dense layer
    let mut l3_pre = nnue.b3.clone();
    for i in 0..l2_size {
        let a = l2_post[i] as i8 as i32;
        for j in 0..l3_size {
            l3_pre[j] += a * (nnue.w3[i * l3_size + j] as i32);
        }
    }

    // L3 clipped ReLU
    let mut l3_post = Vec::with_capacity(l3_size);
    for &x in &l3_pre {
        let (out, _) = clipped_relu(x);
        l3_post.push(out);
    }

    // Output layer
    let mut output = nnue.b_out;
    for i in 0..l3_size {
        let a = l3_post[i] as i8 as i32;
        output += a * (nnue.w_out[i] as i32);
    }

    let state = ForwardState {
        l1_pre,
        l1_post,
        l2_pre,
        l2_post,
        l3_pre,
        l3_post,
        active_features,
    };

    (output as f32 / 1000.0, state)
}

/// Backward pass: compute gradients and update weights.
///
/// Uses the chain rule to propagate gradients from output to input.
/// Weight updates use gradient descent with the given learning rate.
fn backward(
    nnue: &mut Nnue,
    state: &ForwardState,
    error: f32,  // d(Loss)/d(output) = 2 * (pred - target)
    lr: f32,
) {
    let l1_size = nnue.config.l1_size;
    let l2_size = nnue.config.l2_size;
    let l3_size = nnue.config.l3_size;

    // Scale error for quantized weights (output was divided by 1000)
    let d_output = error;

    // ========================================================================
    // Output layer gradients
    // ========================================================================
    // output = b_out + sum(l3_post[i] * w_out[i])
    // d(output)/d(b_out) = 1
    // d(output)/d(w_out[i]) = l3_post[i]

    // Update b_out
    nnue.b_out -= (d_output * lr * 1000.0) as i32;

    // Gradient flowing into L3
    let mut d_l3_post = vec![0.0f32; l3_size];
    for i in 0..l3_size {
        // d(output)/d(l3_post[i]) = w_out[i]
        d_l3_post[i] = d_output * (nnue.w_out[i] as f32);

        // Update w_out[i]
        let grad_w_out = d_output * state.l3_post[i];
        let update = (grad_w_out * lr).clamp(-127.0, 127.0) as i8;
        nnue.w_out[i] = nnue.w_out[i].saturating_sub(update);
    }

    // ========================================================================
    // L3 layer gradients (through clipped ReLU)
    // ========================================================================
    // l3_post = clipped_relu(l3_pre)
    // d(l3_post)/d(l3_pre) = 1 if in active region, else 0

    let mut d_l3_pre = vec![0.0f32; l3_size];
    for i in 0..l3_size {
        let (_, deriv) = clipped_relu(state.l3_pre[i]);
        d_l3_pre[i] = d_l3_post[i] * deriv;
    }

    // l3_pre = b3 + sum(l2_post[i] * w3[i,j])
    // Update b3
    for j in 0..l3_size {
        nnue.b3[j] -= (d_l3_pre[j] * lr * 64.0) as i32;
    }

    // Gradient flowing into L2
    let mut d_l2_post = vec![0.0f32; l2_size];
    for i in 0..l2_size {
        for j in 0..l3_size {
            // d(l3_pre[j])/d(l2_post[i]) = w3[i,j]
            d_l2_post[i] += d_l3_pre[j] * (nnue.w3[i * l3_size + j] as f32);

            // Update w3[i,j]
            let grad_w3 = d_l3_pre[j] * state.l2_post[i];
            let update = (grad_w3 * lr * 0.01).clamp(-127.0, 127.0) as i8;
            nnue.w3[i * l3_size + j] = nnue.w3[i * l3_size + j].saturating_sub(update);
        }
    }

    // ========================================================================
    // L2 layer gradients (through clipped ReLU)
    // ========================================================================
    let mut d_l2_pre = vec![0.0f32; l2_size];
    for i in 0..l2_size {
        let (_, deriv) = clipped_relu(state.l2_pre[i]);
        d_l2_pre[i] = d_l2_post[i] * deriv;
    }

    // Update b2
    for j in 0..l2_size {
        nnue.b2[j] -= (d_l2_pre[j] * lr * 64.0) as i32;
    }

    // Gradient flowing into L1
    let mut d_l1_post = vec![0.0f32; l1_size];
    for i in 0..l1_size {
        for j in 0..l2_size {
            d_l1_post[i] += d_l2_pre[j] * (nnue.w2[i * l2_size + j] as f32);

            // Update w2[i,j]
            let grad_w2 = d_l2_pre[j] * state.l1_post[i];
            let update = (grad_w2 * lr * 0.01).clamp(-127.0, 127.0) as i8;
            nnue.w2[i * l2_size + j] = nnue.w2[i * l2_size + j].saturating_sub(update);
        }
    }

    // ========================================================================
    // L1 layer gradients (through clipped ReLU)
    // ========================================================================
    let mut d_l1_pre = vec![0.0f32; l1_size];
    for i in 0..l1_size {
        let (_, deriv) = clipped_relu(state.l1_pre[i]);
        d_l1_pre[i] = d_l1_post[i] * deriv;
    }

    // Update b1
    for i in 0..l1_size {
        nnue.b1[i] -= (d_l1_pre[i] * lr * 64.0) as i32;
    }

    // Update w1 (sparse - only for active features)
    for &feature_idx in &state.active_features {
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            // d(l1_pre[i])/d(w1[feature_idx, i]) = 1 (feature is active)
            let grad_w1 = d_l1_pre[i];
            let update = (grad_w1 * lr * 0.1).clamp(-32767.0, 32767.0) as i16;
            nnue.w1[offset + i] = nnue.w1[offset + i].saturating_sub(update);
        }
    }
}

/// Train NNUE on a batch of samples with full backpropagation.
///
/// Loss function: MSE between predicted Q-value and normalized final cost.
///
/// Target: (initial_cost - final_cost) / initial_cost
///   - Positive when we improved (final < initial)
///   - Zero when no improvement
///   - Approaches 1.0 for maximum improvement (final → 0)
///
/// The NNUE learns to predict: "How much improvement will this trajectory achieve?"
fn train_batch(nnue: &mut Nnue, batch: &[TrainingSample], lr: f32) -> f32 {
    let mut total_loss = 0.0;

    for sample in batch {
        // Forward pass with state for backprop
        let (pred, state) = forward_with_state(nnue, &sample.features);

        // Target: normalized improvement achieved by the end of the trajectory
        // Q(s,a) should predict "what fraction of the initial cost will I save?"
        let target = if sample.initial_cost > 0 {
            (sample.initial_cost as f32 - sample.final_cost as f32) / sample.initial_cost as f32
        } else {
            0.0
        };

        // MSE loss
        let error = pred - target;
        total_loss += error * error;

        // Full backpropagation
        let d_loss = 2.0 * error;  // d(MSE)/d(pred)
        backward(nnue, &state, d_loss, lr);
    }

    total_loss / batch.len() as f32
}

/// Evaluate the trained NNUE on test expressions.
///
/// Compares NNUE-guided search (inference mode, ε=0) against random baseline.
fn evaluate(
    nnue: &Nnue,
    test_exprs: &[ExprTree],
    config: &TrainConfig,
) -> (f64, f64) {
    let mut guided_improvements = 0.0;
    let mut random_improvements = 0.0;

    for tree in test_exprs {
        // Guided search (with trained NNUE, no exploration)
        let guided_config = GuidedConfig::default()
            .with_iterations(config.mcts_iterations / 2)
            .inference_mode();  // ε = 0

        let mut guided_mcts = GuidedMcts::from_tree(tree, guided_config);
        let initial_cost = guided_mcts.best_cost();

        // Run with NNUE-guided action selection
        let max_iters = config.mcts_iterations / 2;
        while guided_mcts.iterations() < max_iters {
            let iter_before = guided_mcts.iterations();
            let result = guided_mcts.iterate_with_evaluator(|state, action| {
                evaluate_action_with_nnue(nnue, state, action, iter_before)
            });
            if result.is_none() {
                break;
            }
            if guided_mcts.best_cost() <= 1 {
                break;
            }
        }

        let guided_cost = guided_mcts.best_cost();

        if initial_cost > 0 {
            guided_improvements += 1.0 - (guided_cost as f64 / initial_cost as f64);
        }

        // Random search (baseline, no NNUE)
        let random_config = GuidedConfig::default()
            .with_iterations(config.mcts_iterations / 2)
            .with_epsilon(1.0);  // Pure random

        let mut random_mcts = GuidedMcts::from_tree(tree, random_config);
        let _result = random_mcts.run();  // Uses iterate() internally which has no prior
        let random_cost = random_mcts.best_cost();

        if initial_cost > 0 {
            random_improvements += 1.0 - (random_cost as f64 / initial_cost as f64);
        }
    }

    let n = test_exprs.len() as f64;
    (guided_improvements / n, random_improvements / n)
}

// ============================================================================
// Curriculum Training: Phase 1 (Kindergarten) + Phase 2 (University)
// ============================================================================

/// Train NNUE on a single ground truth sample (Phase 1).
///
/// Uses saturation to get perfect optimal cost, then trains NNUE to predict it.
fn train_on_ground_truth(
    nnue: &mut Nnue,
    tree: &ExprTree,
    optimal_cost: usize,
    initial_cost: usize,
    lr: f32,
) -> f32 {
    let features = extract_tree_features(tree);
    let (pred, state) = forward_with_state(nnue, &features);

    // Target: normalized improvement (same as MCTS version)
    let target = if initial_cost > 0 {
        (initial_cost as f32 - optimal_cost as f32) / initial_cost as f32
    } else {
        0.0
    };

    let error = pred - target;
    let loss = error * error;

    // Backprop
    backward(nnue, &state, 2.0 * error, lr);

    loss
}

/// Predict cost reduction potential using NNUE.
///
/// Returns a priority score (lower = more promising = likely more reducible).
fn nnue_predict(nnue: &Nnue, tree: &ExprTree) -> i64 {
    let features = extract_tree_features(tree);

    let mut acc = Accumulator::new(nnue);
    for f in &features {
        let idx = f.to_index();
        if idx < HalfEPFeature::COUNT {
            acc.add_feature(nnue, idx);
        }
    }

    // Higher NNUE score = more improvement expected = lower priority (we want to explore less)
    // Lower NNUE score = less improvement = higher priority (already optimized)
    let score = acc.forward(nnue);

    // Invert: high NNUE prediction (good improvement) → low priority
    // This makes Best-First prioritize states NNUE thinks are already good
    -(score as i64)
}

/// Run Phase 1: Kindergarten training on small kernels with saturation.
///
/// Returns training statistics: (total_loss, samples_trained, avg_improvement)
fn run_kindergarten(
    nnue: &mut Nnue,
    config: &CurriculumConfig,
    expr_gen: &mut ExprGenerator,
) -> (f32, usize, f64) {
    let mut total_loss = 0.0;
    let mut samples = 0;
    let mut total_improvement = 0.0;

    for _ in 0..config.kindergarten_samples {
        let tree = expr_gen.generate_small();

        // Use BestFirstPlanner with high saturation threshold to force saturation
        let bf_config = BestFirstConfig::default()
            .with_saturation_threshold(1000)  // Always saturate
            .with_max_expansions(0);  // No search needed

        let mut planner = BestFirstPlanner::from_tree(&tree, bf_config);
        let result = planner.run_default();

        if !result.used_saturation {
            // Fallback: kernel was somehow too big, skip
            continue;
        }

        let initial_cost = result.initial_cost;
        let optimal_cost = result.best_cost;

        // Train on this ground truth
        let loss = train_on_ground_truth(
            nnue,
            &result.best_tree,
            optimal_cost,
            initial_cost,
            config.learning_rate,
        );

        total_loss += loss;
        samples += 1;

        if initial_cost > 0 {
            total_improvement += (initial_cost as f64 - optimal_cost as f64) / initial_cost as f64;
        }
    }

    let avg_improvement = if samples > 0 { total_improvement / samples as f64 } else { 0.0 };
    (total_loss, samples, avg_improvement)
}

/// Run Phase 2: University training on large kernels with NNUE-guided search.
///
/// Returns training statistics: (total_loss, samples_trained, avg_improvement)
fn run_university(
    nnue: &mut Nnue,
    config: &CurriculumConfig,
    expr_gen: &mut ExprGenerator,
    replay: &mut ReplayBuffer,
) -> (f32, usize, f64) {
    let mut total_loss = 0.0;
    let mut samples = 0;
    let mut total_improvement = 0.0;

    for _ in 0..config.university_samples {
        let tree = expr_gen.generate_large();

        // Best-First with trained NNUE + exploration
        let bf_config = BestFirstConfig::default()
            .with_saturation_threshold(0)  // Never saturate
            .with_epsilon(config.epsilon as f64)
            .with_max_expansions(config.max_expansions);

        let mut planner = BestFirstPlanner::from_tree(&tree, bf_config);

        // Clone nnue reference for closure (read-only)
        let result = planner.run(|t| nnue_predict(nnue, t));

        let initial_cost = result.initial_cost;
        let achieved_cost = result.best_cost;

        // Train on achieved cost (not perfect, but good)
        let loss = train_on_ground_truth(
            nnue,
            &result.best_tree,
            achieved_cost,
            initial_cost,
            config.learning_rate,
        );

        total_loss += loss;
        samples += 1;

        if initial_cost > 0 {
            total_improvement += (initial_cost as f64 - achieved_cost as f64) / initial_cost as f64;
        }

        // Add to replay buffer for mixed training
        let features = extract_tree_features(&result.best_tree);
        replay.add(TrainingSample {
            features,
            action: None,  // Curriculum training doesn't use MCTS actions
            final_cost: achieved_cost,
            initial_cost,
        });
    }

    // Train on replay buffer (experience replay)
    let num_batches = (replay.len() / config.batch_size).max(1).min(5);
    for _ in 0..num_batches {
        let batch = replay.sample_batch(config.batch_size);
        if !batch.is_empty() {
            total_loss += train_batch(nnue, &batch, config.learning_rate);
        }
    }

    let avg_improvement = if samples > 0 { total_improvement / samples as f64 } else { 0.0 };
    (total_loss, samples, avg_improvement)
}

/// Curriculum training configuration.
struct CurriculumConfig {
    /// Number of small kernels to train on in Phase 1
    kindergarten_samples: usize,
    /// Number of kindergarten epochs
    kindergarten_epochs: usize,
    /// Number of large kernels to train on in Phase 2
    university_samples: usize,
    /// Number of university epochs
    university_epochs: usize,
    /// Learning rate
    learning_rate: f32,
    /// Epsilon for exploration in Phase 2
    epsilon: f32,
    /// Max search expansions per kernel in Phase 2
    max_expansions: usize,
    /// Batch size for replay training
    batch_size: usize,
    /// Replay buffer size
    replay_buffer_size: usize,
    /// Random seed
    seed: u64,
    /// MCTS iterations for legacy evaluation
    #[allow(dead_code)]
    mcts_iterations: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            kindergarten_samples: 200,
            kindergarten_epochs: 10,
            university_samples: 100,
            university_epochs: 20,
            learning_rate: 0.001,
            epsilon: 0.1,
            max_expansions: 500,
            batch_size: 64,
            replay_buffer_size: 5000,
            seed: 42,
            mcts_iterations: 200,
        }
    }
}

fn main() {
    // Parse command-line args
    let args: Vec<String> = std::env::args().collect();
    let quick_mode = args.iter().any(|a| a == "--quick");
    let kindergarten_only = args.iter().any(|a| a == "--kindergarten-only");
    let legacy_mode = args.iter().any(|a| a == "--legacy");

    // Legacy mode: use the old MCTS-based training
    if legacy_mode {
        run_legacy_training(quick_mode);
        return;
    }

    let config = if quick_mode {
        CurriculumConfig {
            kindergarten_samples: 50,
            kindergarten_epochs: 3,
            university_samples: 20,
            university_epochs: 5,
            max_expansions: 100,
            ..CurriculumConfig::default()
        }
    } else {
        CurriculumConfig::default()
    };

    println!("=== Curriculum Training: NNUE for E-Graph Cost Prediction ===");
    if quick_mode { println!("(quick mode)"); }
    if kindergarten_only { println!("(kindergarten only)"); }
    println!();

    // Try to load learned cost model (The Judge)
    let cost_model = CostModel::load_or_default();
    println!("Cost model loaded (The Judge):");
    println!("  add={}, mul={}, div={}, sqrt={}",
             cost_model.add, cost_model.mul, cost_model.div, cost_model.sqrt);
    println!();

    // Initialize
    let mut nnue = Nnue::new(NnueConfig::default());
    let mut expr_gen = ExprGenerator::new(config.seed);
    let mut replay = ReplayBuffer::new(config.replay_buffer_size, config.seed + 1);

    let start = Instant::now();

    // =========================================================================
    // PHASE 1: KINDERGARTEN (Ground Truth from Saturation)
    // =========================================================================
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  PHASE 1: KINDERGARTEN (Saturation = Ground Truth)            ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  • Training on small kernels (10-50 nodes)                    ║");
    println!("║  • Saturating completely for PERFECT optimal costs            ║");
    println!("║  • NNUE learns: x*0=0, x+0=x, x*1=x, etc.                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    println!("{:>6} {:>10} {:>12} {:>14}",
             "Epoch", "Samples", "Avg Loss", "Avg Improve %");
    println!("{:-<50}", "");

    for epoch in 1..=config.kindergarten_epochs {
        let (loss, samples, improvement) = run_kindergarten(&mut nnue, &config, &mut expr_gen);
        let avg_loss = if samples > 0 { loss / samples as f32 } else { 0.0 };

        println!("{:>6} {:>10} {:>12.4} {:>13.1}%",
                 epoch, samples, avg_loss, improvement * 100.0);
    }

    println!("{:-<50}", "");
    println!();

    if kindergarten_only {
        let elapsed = start.elapsed();
        println!("Kindergarten training completed in {:.2}s", elapsed.as_secs_f64());
        println!("(Use without --kindergarten-only for full curriculum)");
        return;
    }

    // =========================================================================
    // PHASE 2: UNIVERSITY (Guided Search + Exploration)
    // =========================================================================
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  PHASE 2: UNIVERSITY (Guided Search + Exploration)            ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  • Training on large kernels (100+ nodes)                     ║");
    println!("║  • Saturation IMPOSSIBLE - using NNUE-guided A* search        ║");
    println!("║  • ε-greedy exploration ({:.0}% random) to escape local minima  ║", config.epsilon * 100.0);
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    println!("{:>6} {:>10} {:>12} {:>14}",
             "Epoch", "Samples", "Avg Loss", "Avg Improve %");
    println!("{:-<50}", "");

    for epoch in 1..=config.university_epochs {
        let (loss, samples, improvement) = run_university(
            &mut nnue, &config, &mut expr_gen, &mut replay
        );
        let avg_loss = if samples > 0 { loss / samples as f32 } else { 0.0 };

        println!("{:>6} {:>10} {:>12.4} {:>13.1}%",
                 epoch, samples, avg_loss, improvement * 100.0);
    }

    println!("{:-<50}", "");

    let elapsed = start.elapsed();
    println!("\nCurriculum training completed in {:.2}s", elapsed.as_secs_f64());

    // =========================================================================
    // Final Evaluation
    // =========================================================================
    println!("\n=== Final Evaluation ===");

    // Generate test expressions (mix of small and large)
    let mut test_gen = ExprGenerator::new(config.seed + 999);
    let test_small: Vec<_> = (0..25).map(|_| test_gen.generate_small()).collect();
    let test_large: Vec<_> = (0..25).map(|_| test_gen.generate_large()).collect();

    // Evaluate on small kernels
    let mut small_guided_improvement = 0.0;

    for tree in &test_small {
        let bf_config = BestFirstConfig::default()
            .with_saturation_threshold(1000);

        let mut planner = BestFirstPlanner::from_tree(tree, bf_config);
        let result = planner.run_default();

        let initial = result.initial_cost;
        let optimal = result.best_cost;

        if initial > 0 {
            small_guided_improvement += (initial as f64 - optimal as f64) / initial as f64;
        }
    }

    small_guided_improvement /= test_small.len() as f64;

    // Evaluate on large kernels (guided vs random)
    let mut large_guided_improvement = 0.0;
    let mut large_random_improvement = 0.0;

    for tree in &test_large {
        // Guided (with NNUE)
        let guided_config = BestFirstConfig::default()
            .with_saturation_threshold(0)
            .with_epsilon(0.0)  // Inference mode
            .with_max_expansions(200);

        let mut guided_planner = BestFirstPlanner::from_tree(tree, guided_config);
        let guided_result = guided_planner.run(|t| nnue_predict(&nnue, t));

        let initial = guided_result.initial_cost;
        if initial > 0 {
            large_guided_improvement +=
                (initial as f64 - guided_result.best_cost as f64) / initial as f64;
        }

        // Random baseline
        let random_config = BestFirstConfig::default()
            .with_saturation_threshold(0)
            .with_epsilon(1.0)  // Pure random
            .with_max_expansions(200);

        let mut random_planner = BestFirstPlanner::from_tree(tree, random_config);
        let random_result = random_planner.run_default();

        if initial > 0 {
            large_random_improvement +=
                (initial as f64 - random_result.best_cost as f64) / initial as f64;
        }
    }

    large_guided_improvement /= test_large.len() as f64;
    large_random_improvement /= test_large.len() as f64;

    println!("\nSmall kernels (saturation possible):");
    println!("  Improvement: {:.1}%", small_guided_improvement * 100.0);

    println!("\nLarge kernels (guided search required):");
    println!("  Guided improvement: {:.1}%", large_guided_improvement * 100.0);
    println!("  Random improvement: {:.1}%", large_random_improvement * 100.0);
    println!("  Advantage over random: {:+.1}%",
             (large_guided_improvement - large_random_improvement) * 100.0);

    // Summary
    println!("\n=== Summary ===");
    println!("Replay buffer: {} samples", replay.len());

    if large_guided_improvement > large_random_improvement + 0.02 {
        println!("\nSUCCESS: NNUE-guided search outperforms random!");
    } else if large_guided_improvement > large_random_improvement {
        println!("\nMARGINAL: Guided search slightly better than random.");
    } else {
        println!("\nNEEDS WORK: More training or better features needed.");
    }
}

/// Legacy MCTS-based training (for comparison).
fn run_legacy_training(quick_mode: bool) {
    let config = if quick_mode {
        TrainConfig {
            epochs: 5,
            episodes_per_epoch: 20,
            mcts_iterations: 50,
            ..TrainConfig::default()
        }
    } else {
        TrainConfig::default()
    };

    println!("=== Legacy MCTS-Based Training ==={}\n",
             if quick_mode { " (quick mode)" } else { "" });
    println!("Configuration:");
    println!("  Epochs: {}", config.epochs);
    println!("  Episodes/epoch: {}", config.episodes_per_epoch);
    println!("  MCTS iterations: {}", config.mcts_iterations);
    println!("  Epsilon: {}", config.epsilon);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Replay buffer: {}", config.replay_buffer_size);
    println!();

    // Initialize
    let mut nnue = Nnue::new(NnueConfig::default());
    let mut replay = ReplayBuffer::new(config.replay_buffer_size, config.seed);
    let mut expr_gen = ExprGenerator::new(config.seed + 1);

    // Generate test set
    let test_exprs: Vec<_> = (0..50).map(|_| expr_gen.generate()).collect();

    // Evaluate baseline
    let (baseline_guided, baseline_random) = evaluate(&nnue, &test_exprs, &config);
    println!("Baseline (untrained):");
    println!("  Guided improvement:  {:.1}%", baseline_guided * 100.0);
    println!("  Random improvement:  {:.1}%", baseline_random * 100.0);
    println!();

    println!("{:>6} {:>12} {:>12} {:>10} {:>12} {:>12}",
             "Epoch", "Episodes", "Samples", "Loss", "Guided %", "Random %");
    println!("{:-<70}", "");

    let start = Instant::now();

    for epoch in 1..=config.epochs {
        let mut epoch_samples = 0;
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        // Generate episodes
        for _ in 0..config.episodes_per_epoch {
            let tree = expr_gen.generate();
            let samples = run_episode(&tree, &config, &nnue);

            epoch_samples += samples.len();
            for s in samples {
                replay.add(s);
            }
        }

        // Train on replay buffer
        let num_batches = (replay.len() / config.batch_size).max(1);
        for _ in 0..num_batches.min(10) {
            let batch = replay.sample_batch(config.batch_size);
            if !batch.is_empty() {
                epoch_loss += train_batch(&mut nnue, &batch, config.learning_rate);
                batches += 1;
            }
        }

        // Evaluate periodically
        if epoch % 10 == 0 || epoch == 1 || epoch == config.epochs {
            let (guided_imp, random_imp) = evaluate(&nnue, &test_exprs, &config);
            let avg_loss = if batches > 0 { epoch_loss / batches as f32 } else { 0.0 };

            println!("{:>6} {:>12} {:>12} {:>10.4} {:>11.1}% {:>11.1}%",
                     epoch,
                     config.episodes_per_epoch,
                     epoch_samples,
                     avg_loss,
                     guided_imp * 100.0,
                     random_imp * 100.0);
        }
    }

    println!("{:-<70}", "");

    let elapsed = start.elapsed();
    println!("\nTraining completed in {:.2}s", elapsed.as_secs_f64());

    // Final evaluation
    println!("\n=== Final Results ===");
    let (final_guided, final_random) = evaluate(&nnue, &test_exprs, &config);
    println!("Final guided improvement:  {:.1}%", final_guided * 100.0);
    println!("Final random improvement:  {:.1}%", final_random * 100.0);
    println!("Advantage over random:     {:+.1}%", (final_guided - final_random) * 100.0);
}
