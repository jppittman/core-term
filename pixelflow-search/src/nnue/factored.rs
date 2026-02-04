//! # Factored Embedding NNUE Architecture
//!
//! An O(ops) alternative to the O(ops²) HalfEP feature encoding.
//!
//! ## The Problem
//!
//! HalfEP features encode all (perspective_op, descendant_op, depth, path) tuples:
//! - 42 ops → 42² × 8 × 256 = 3.6M possible features
//! - Feature space grows quadratically with operation count
//! - Training requires O(GB) of memory for weight matrices
//!
//! ## The Solution: Edge-based Factored Embeddings
//!
//! Instead of one-hot encoding each (parent, child) pair, we learn dense
//! embeddings for each operation and accumulate them edge-by-edge:
//!
//! ```text
//! For each parent→child edge in the expression tree:
//!     accumulator[0..K]  += E[parent_op]   // "what's above"
//!     accumulator[K..2K] += E[child_op]    // "what's below"
//! ```
//!
//! Key insight: **Position encodes role**. Parent ops contribute to the first
//! half of the accumulator, child ops to the second half. This ensures that
//! `Mul→Add` (FMA-eligible) produces a different vector than `Add→Mul` (not FMA).
//!
//! ## Complexity
//!
//! | Metric | HalfEP | Factored | Improvement |
//! |--------|--------|----------|-------------|
//! | Feature space | O(ops²) | O(ops) | O(ops) |
//! | Weight memory | ~1GB | ~10KB | 100,000× |
//! | Accumulator build | O(nodes²) | O(edges) | O(nodes) |
//! | Incremental update | O(subtree²) | O(Δedges × K) | O(subtree) |

#![allow(dead_code)] // Prototype code

extern crate alloc;

use alloc::vec::Vec;
use libm::sqrtf;

pub use pixelflow_ir::OpKind;
use pixelflow_ir::Expr;

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension per operation.
///
/// Each operation gets a K-dimensional learned embedding. The accumulator
/// stores 2K values: K for parent roles, K for child roles.
pub const K: usize = 32;

/// Number of structural features (operation-independent).
pub const STRUCTURAL_FEATURE_COUNT: usize = 21;

/// Total input dimension to the hidden layer: 2K (edge accumulator) + structural.
pub const INPUT_DIM: usize = 2 * K + STRUCTURAL_FEATURE_COUNT;

/// Hidden layer size.
pub const HIDDEN_DIM: usize = 64;

/// Rule feature input dimension (for policy head).
/// Features: rule_idx, graph_size, node_count, time_used, match_rate,
///           staleness, time_remaining, space_remaining, pairs_remaining
pub const RULE_INPUT_DIM: usize = 9;

// ============================================================================
// Operation Embeddings
// ============================================================================

/// Learned dense embeddings for each operation type.
///
/// Each of the 42 operations gets a K-dimensional embedding vector.
/// These are the primary learned parameters that capture semantic
/// similarity between operations.
#[derive(Clone)]
pub struct OpEmbeddings {
    /// E[op][i] = i-th dimension of op's embedding.
    /// Stored as [OpKind::COUNT][K] = 42 × 32 = 1,344 floats.
    pub e: [[f32; K]; OpKind::COUNT],
}

impl Default for OpEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl OpEmbeddings {
    /// Create zero-initialized embeddings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            e: [[0.0; K]; OpKind::COUNT],
        }
    }

    /// Initialize embeddings with random values using He initialization.
    ///
    /// Scale: sqrt(2/K) for ReLU networks.
    #[must_use]
    pub fn new_random(seed: u64) -> Self {
        let mut embeddings = Self::new();
        embeddings.randomize(seed);
        embeddings
    }

    /// Initialize with latency priors.
    ///
    /// This encodes known operation latencies into dimension 0 of each embedding,
    /// giving the model a strong starting point. Remaining dimensions are small
    /// random values that can learn subtle interactions.
    ///
    /// Scales to any number of ops - just provide latencies for new ops.
    #[must_use]
    pub fn new_with_latency_prior(seed: u64) -> Self {
        let mut embeddings = Self::new();
        embeddings.init_with_latency_prior(seed);
        embeddings
    }

    /// Initialize with latency priors in place.
    pub fn init_with_latency_prior(&mut self, seed: u64) {
        // Known latencies (cycles) - these are approximate and can be refined
        // Dimension 0 = latency, normalized to [0, 1] range (divide by max ~20)
        let latencies: [f32; OpKind::COUNT] = [
            0.0,   // Var - free
            0.0,   // Const - free
            0.2,   // Add - 4 cycles
            0.2,   // Sub - 4 cycles
            0.25,  // Mul - 5 cycles
            0.75,  // Div - 15 cycles
            0.05,  // Neg - 1 cycle
            0.75,  // Sqrt - 15 cycles
            0.25,  // Rsqrt - 5 cycles (fast approximation)
            0.05,  // Abs - 1 cycle
            0.2,   // Min - 4 cycles
            0.2,   // Max - 4 cycles
            0.25,  // MulAdd - 5 cycles (fused)
            0.3,   // MulRsqrt - 6 cycles
            0.5,   // Recip - 10 cycles
            0.2,   // Floor - 4 cycles
            0.2,   // Ceil - 4 cycles
            0.2,   // Round - 4 cycles
            0.2,   // Fract - 4 cycles
            0.5,   // Sin - 10 cycles
            0.5,   // Cos - 10 cycles
            0.5,   // Tan - 10 cycles
            0.5,   // Asin - 10 cycles
            0.5,   // Acos - 10 cycles
            0.5,   // Atan - 10 cycles
            0.5,   // Exp - 10 cycles
            0.5,   // Exp2 - 10 cycles
            0.5,   // Ln - 10 cycles
            0.5,   // Log2 - 10 cycles
            0.5,   // Log10 - 10 cycles
            0.5,   // Atan2 - 10 cycles
            0.6,   // Pow - 12 cycles
            0.4,   // Hypot - 8 cycles
            0.15,  // Lt - 3 cycles
            0.15,  // Le - 3 cycles
            0.15,  // Gt - 3 cycles
            0.15,  // Ge - 3 cycles
            0.15,  // Eq - 3 cycles
            0.15,  // Ne - 3 cycles
            0.2,   // Select - 4 cycles
            0.3,   // Clamp - 6 cycles (2x compare + select)
            0.0,   // Tuple - free (structural)
        ];

        let mut rng_state = seed.wrapping_add(1);
        let small_scale = 0.1; // Small noise for other dimensions

        for op_idx in 0..OpKind::COUNT {
            // Dimension 0: latency prior
            self.e[op_idx][0] = latencies[op_idx];

            // Dimensions 1..K: small random for learning interactions
            for dim in 1..K {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                self.e[op_idx][dim] = (uniform * 2.0 - 1.0) * small_scale;
            }
        }
    }

    /// Randomize embeddings in place (fully random, no priors).
    pub fn randomize(&mut self, seed: u64) {
        let scale = sqrtf(2.0 / K as f32);
        let mut rng_state = seed.wrapping_add(1);

        for op_idx in 0..OpKind::COUNT {
            for dim in 0..K {
                // LCG for no_std compatibility
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);

                // Convert to [-1, 1] and scale
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                let centered = uniform * 2.0 - 1.0;
                self.e[op_idx][dim] = centered * scale;
            }
        }
    }

    /// Get embedding for an operation.
    #[inline]
    #[must_use]
    pub fn get(&self, op: OpKind) -> &[f32; K] {
        &self.e[op.index()]
    }

    /// Total parameter count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpKind::COUNT * K
    }
}

// ============================================================================
// Edge Accumulator
// ============================================================================

/// Accumulator state for edge-based feature extraction.
///
/// For each parent→child edge in the expression tree, we add:
/// - `values[0..K]  += E[parent_op]`
/// - `values[K..2K] += E[child_op]`
///
/// This creates an asymmetric representation where position encodes role:
/// parent contributions go to the first half, child contributions to the second.
#[derive(Clone)]
pub struct EdgeAccumulator {
    /// Accumulated embedding values.
    /// - `[0..K]`: sum of all parent embeddings
    /// - `[K..2K]`: sum of all child embeddings
    pub values: [f32; 2 * K],

    /// Edge count for normalization (optional).
    pub edge_count: u32,
}

impl Default for EdgeAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeAccumulator {
    /// Create a zero-initialized accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: [0.0; 2 * K],
            edge_count: 0,
        }
    }

    /// Reset to zero state.
    pub fn reset(&mut self) {
        self.values = [0.0; 2 * K];
        self.edge_count = 0;
    }

    /// Add a single edge contribution.
    ///
    /// For a parent→child edge, adds the parent embedding to the first half
    /// and the child embedding to the second half.
    #[inline]
    pub fn add_edge(&mut self, emb: &OpEmbeddings, parent_op: OpKind, child_op: OpKind) {
        let parent_emb = emb.get(parent_op);
        let child_emb = emb.get(child_op);

        for i in 0..K {
            self.values[i] += parent_emb[i];
            self.values[K + i] += child_emb[i];
        }
        self.edge_count += 1;
    }

    /// Remove a single edge contribution (for incremental updates).
    #[inline]
    pub fn remove_edge(&mut self, emb: &OpEmbeddings, parent_op: OpKind, child_op: OpKind) {
        let parent_emb = emb.get(parent_op);
        let child_emb = emb.get(child_op);

        for i in 0..K {
            self.values[i] -= parent_emb[i];
            self.values[K + i] -= child_emb[i];
        }
        self.edge_count = self.edge_count.saturating_sub(1);
    }

    /// Build accumulator from an expression tree.
    ///
    /// Traverses the tree and accumulates edge contributions.
    pub fn from_expr(expr: &Expr, emb: &OpEmbeddings) -> Self {
        let mut acc = Self::new();
        acc.add_expr_edges(expr, emb);
        acc
    }

    /// Add all edges from an expression subtree.
    pub fn add_expr_edges(&mut self, expr: &Expr, emb: &OpEmbeddings) {
        let parent_op = expr.op_type();

        match expr {
            Expr::Var(_) | Expr::Const(_) => {
                // Leaves have no children, so no edges
            }
            Expr::Unary(_, child) => {
                let child_op = child.op_type();
                self.add_edge(emb, parent_op, child_op);
                self.add_expr_edges(child, emb);
            }
            Expr::Binary(_, left, right) => {
                let left_op = left.op_type();
                let right_op = right.op_type();
                self.add_edge(emb, parent_op, left_op);
                self.add_edge(emb, parent_op, right_op);
                self.add_expr_edges(left, emb);
                self.add_expr_edges(right, emb);
            }
            Expr::Ternary(_, a, b, c) => {
                let a_op = a.op_type();
                let b_op = b.op_type();
                let c_op = c.op_type();
                self.add_edge(emb, parent_op, a_op);
                self.add_edge(emb, parent_op, b_op);
                self.add_edge(emb, parent_op, c_op);
                self.add_expr_edges(a, emb);
                self.add_expr_edges(b, emb);
                self.add_expr_edges(c, emb);
            }
            Expr::Nary(_, children) => {
                for child in children {
                    let child_op = child.op_type();
                    self.add_edge(emb, parent_op, child_op);
                    self.add_expr_edges(child, emb);
                }
            }
        }
    }

    /// Remove all edges from an expression subtree.
    pub fn remove_expr_edges(&mut self, expr: &Expr, emb: &OpEmbeddings) {
        let parent_op = expr.op_type();

        match expr {
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::Unary(_, child) => {
                let child_op = child.op_type();
                self.remove_edge(emb, parent_op, child_op);
                self.remove_expr_edges(child, emb);
            }
            Expr::Binary(_, left, right) => {
                let left_op = left.op_type();
                let right_op = right.op_type();
                self.remove_edge(emb, parent_op, left_op);
                self.remove_edge(emb, parent_op, right_op);
                self.remove_expr_edges(left, emb);
                self.remove_expr_edges(right, emb);
            }
            Expr::Ternary(_, a, b, c) => {
                let a_op = a.op_type();
                let b_op = b.op_type();
                let c_op = c.op_type();
                self.remove_edge(emb, parent_op, a_op);
                self.remove_edge(emb, parent_op, b_op);
                self.remove_edge(emb, parent_op, c_op);
                self.remove_expr_edges(a, emb);
                self.remove_expr_edges(b, emb);
                self.remove_expr_edges(c, emb);
            }
            Expr::Nary(_, children) => {
                for child in children {
                    let child_op = child.op_type();
                    self.remove_edge(emb, parent_op, child_op);
                    self.remove_expr_edges(child, emb);
                }
            }
        }
    }
}

// ============================================================================
// Structural Features
// ============================================================================

/// Structural features that are operation-independent.
///
/// These capture global properties of the expression tree that the
/// edge accumulator might miss, such as critical path length and
/// tree shape metrics.
#[derive(Clone, Default)]
pub struct StructuralFeatures {
    /// Values for each structural feature.
    pub values: [f32; STRUCTURAL_FEATURE_COUNT],
}

impl StructuralFeatures {
    // Feature indices (grouped by category)

    // --- ILP Metrics ---
    /// Index: Critical path length (longest dependency chain).
    pub const CRITICAL_PATH: usize = 0;
    /// Index: Total node count.
    pub const TOTAL_NODES: usize = 1;
    /// Index: Maximum width at any depth level.
    pub const MAX_WIDTH: usize = 2;
    /// Index: Tree depth.
    pub const DEPTH: usize = 3;
    /// Index: Ratio of leaves to total nodes.
    pub const LEAF_RATIO: usize = 4;
    /// Index: Average branching factor.
    pub const BRANCHING_FACTOR: usize = 5;

    // --- Node Type Counts ---
    /// Index: Variable reference count.
    pub const VAR_COUNT: usize = 6;
    /// Index: Constant count.
    pub const CONST_COUNT: usize = 7;
    /// Index: Unary operation count.
    pub const UNARY_COUNT: usize = 8;
    /// Index: Binary operation count.
    pub const BINARY_COUNT: usize = 9;
    /// Index: Ternary operation count.
    pub const TERNARY_COUNT: usize = 10;

    // --- Expensive Operations ---
    /// Index: Division count (high latency).
    pub const DIV_COUNT: usize = 11;
    /// Index: Square root count (high latency).
    pub const SQRT_COUNT: usize = 12;
    /// Index: Ratio of expensive ops to total nodes.
    pub const EXPENSIVE_OP_RATIO: usize = 13;

    // --- Structure Metrics ---
    /// Index: Edge count (for accumulator normalization).
    pub const EDGE_COUNT: usize = 14;
    /// Index: Variance in leaf depths (tree balance measure).
    pub const LEAF_DEPTH_VAR: usize = 15;
    /// Index: Has FMA-eligible pattern (Mul→Add).
    pub const HAS_FMA_PATTERN: usize = 16;
    /// Index: Has rsqrt pattern.
    pub const HAS_RSQRT_PATTERN: usize = 17;

    // --- Log-Scale Features ---
    /// Index: log(total_nodes) for gradient stability.
    pub const LOG_NODES: usize = 18;
    /// Index: log(critical_path) for gradient stability.
    pub const LOG_CRITICAL_PATH: usize = 19;

    // --- Register Pressure ---
    /// Index: Unique variable count (distinct var indices used).
    /// This models register pressure - more unique vars = more registers needed.
    pub const UNIQUE_VAR_COUNT: usize = 20;

    /// Create zero-initialized features.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract structural features from an expression.
    #[must_use]
    pub fn from_expr(expr: &Expr) -> Self {
        use alloc::collections::BTreeSet;

        let mut features = Self::new();
        let mut width_at_depth: Vec<u32> = Vec::new();
        let mut leaf_depths: Vec<u32> = Vec::new();
        let mut unique_vars: BTreeSet<u32> = BTreeSet::new();

        // Collect stats with recursive traversal
        let critical_path = Self::collect_stats(
            expr,
            &mut features,
            0,
            &mut width_at_depth,
            &mut leaf_depths,
            &mut unique_vars,
        );

        features.values[Self::CRITICAL_PATH] = critical_path as f32;
        features.values[Self::MAX_WIDTH] = width_at_depth.iter().copied().max().unwrap_or(0) as f32;

        // Compute derived features
        let total_nodes = features.values[Self::TOTAL_NODES];
        let leaf_count = features.values[Self::VAR_COUNT] + features.values[Self::CONST_COUNT];

        if total_nodes > 0.0 {
            features.values[Self::LEAF_RATIO] = leaf_count / total_nodes;
            let non_leaf = total_nodes - leaf_count;
            if non_leaf > 0.0 {
                let edge_count = features.values[Self::EDGE_COUNT];
                features.values[Self::BRANCHING_FACTOR] = edge_count / non_leaf;
            }
        }

        // Expensive operation ratio
        let expensive_ops = features.values[Self::DIV_COUNT] + features.values[Self::SQRT_COUNT];
        if total_nodes > 0.0 {
            features.values[Self::EXPENSIVE_OP_RATIO] = expensive_ops / total_nodes;
        }

        // Leaf depth variance (measure of tree balance)
        if !leaf_depths.is_empty() {
            let mean_depth: f32 =
                leaf_depths.iter().sum::<u32>() as f32 / leaf_depths.len() as f32;
            let variance: f32 = leaf_depths
                .iter()
                .map(|&d| {
                    let diff = d as f32 - mean_depth;
                    diff * diff
                })
                .sum::<f32>()
                / leaf_depths.len() as f32;
            features.values[Self::LEAF_DEPTH_VAR] = sqrtf(variance);
        }

        // Log-scale features for better gradient flow
        features.values[Self::LOG_NODES] = libm::logf(total_nodes.max(1.0));
        features.values[Self::LOG_CRITICAL_PATH] = libm::logf(critical_path.max(1) as f32);

        // Unique variable count (register pressure)
        features.values[Self::UNIQUE_VAR_COUNT] = unique_vars.len() as f32;

        features
    }

    /// Recursive stat collection. Returns critical path cost.
    fn collect_stats(
        expr: &Expr,
        features: &mut Self,
        depth: u32,
        width_at_depth: &mut Vec<u32>,
        leaf_depths: &mut Vec<u32>,
        unique_vars: &mut alloc::collections::BTreeSet<u32>,
    ) -> u32 {
        // Track width at this depth
        if depth as usize >= width_at_depth.len() {
            width_at_depth.resize(depth as usize + 1, 0);
        }
        width_at_depth[depth as usize] += 1;

        features.values[Self::TOTAL_NODES] += 1.0;
        features.values[Self::DEPTH] = features.values[Self::DEPTH].max(depth as f32 + 1.0);

        match expr {
            Expr::Var(idx) => {
                features.values[Self::VAR_COUNT] += 1.0;
                leaf_depths.push(depth);
                unique_vars.insert(u32::from(*idx));
                0 // Zero latency for variable access
            }
            Expr::Const(_) => {
                features.values[Self::CONST_COUNT] += 1.0;
                leaf_depths.push(depth);
                0
            }
            Expr::Unary(op, child) => {
                features.values[Self::UNARY_COUNT] += 1.0;
                features.values[Self::EDGE_COUNT] += 1.0;

                // Track expensive ops
                let op_cost = match op {
                    OpKind::Sqrt => {
                        features.values[Self::SQRT_COUNT] += 1.0;
                        15
                    }
                    OpKind::Rsqrt => {
                        features.values[Self::HAS_RSQRT_PATTERN] = 1.0;
                        5
                    }
                    OpKind::Neg | OpKind::Abs => 1,
                    _ => 5, // Default for extended math ops
                };


                let child_cost = Self::collect_stats(child, features, depth + 1, width_at_depth, leaf_depths, unique_vars);
                op_cost + child_cost
            }
            Expr::Binary(op, left, right) => {
                features.values[Self::BINARY_COUNT] += 1.0;
                features.values[Self::EDGE_COUNT] += 2.0;

                let op_cost = match op {
                    OpKind::Div => {
                        features.values[Self::DIV_COUNT] += 1.0;
                        15
                    }
                    OpKind::Add | OpKind::Sub | OpKind::Min | OpKind::Max => 4,
                    OpKind::Mul => {
                        // Check for FMA pattern: this Mul is child of Add
                        5
                    }
                    OpKind::MulRsqrt => {
                        features.values[Self::HAS_RSQRT_PATTERN] = 1.0;
                        6
                    }
                    _ => 5,
                };

                // Check for FMA pattern: Mul as left child and current is in Add context
                if *op == OpKind::Add {
                    if matches!(left.as_ref(), Expr::Binary(OpKind::Mul, _, _)) {
                        features.values[Self::HAS_FMA_PATTERN] = 1.0;
                    }
                }


                let left_cost = Self::collect_stats(left, features, depth + 1, width_at_depth, leaf_depths, unique_vars);
                let right_cost = Self::collect_stats(right, features, depth + 1, width_at_depth, leaf_depths, unique_vars);

                // Critical path: max of parallel children + this op
                op_cost + left_cost.max(right_cost)
            }
            Expr::Ternary(op, a, b, c) => {
                features.values[Self::TERNARY_COUNT] += 1.0;
                features.values[Self::EDGE_COUNT] += 3.0;

                let op_cost = match op {
                    OpKind::MulAdd => {
                        features.values[Self::HAS_FMA_PATTERN] = 1.0;
                        5
                    }
                    _ => 10,
                };


                let a_cost = Self::collect_stats(a, features, depth + 1, width_at_depth, leaf_depths, unique_vars);
                let b_cost = Self::collect_stats(b, features, depth + 1, width_at_depth, leaf_depths, unique_vars);
                let c_cost = Self::collect_stats(c, features, depth + 1, width_at_depth, leaf_depths, unique_vars);

                op_cost + a_cost.max(b_cost).max(c_cost)
            }
            Expr::Nary(op, children) => {
                // N-ary ops like Tuple
                features.values[Self::EDGE_COUNT] += children.len() as f32;

                let op_cost = match op {
                    OpKind::Tuple => 0, // Tuple is structural, no cost
                    _ => 5,
                };

                let max_child_cost = children.iter()
                    .map(|c| Self::collect_stats(c, features, depth + 1, width_at_depth, leaf_depths, unique_vars))
                    .max()
                    .unwrap_or(0);

                op_cost + max_child_cost
            }
        }
    }

    /// Convert to array for neural network input.
    #[inline]
    #[must_use]
    pub fn as_array(&self) -> &[f32; STRUCTURAL_FEATURE_COUNT] {
        &self.values
    }
}

// ============================================================================
// Factored NNUE Network
// ============================================================================

/// Factored embedding NNUE network.
///
/// Architecture:
/// ```text
/// Edge Accumulator (64) + Structural Features (20)
///                     │
///                     ▼ 84-dim input
///               ┌──────────┐
///               │  Hidden  │ 84 → 64 (ReLU)
///               └────┬─────┘
///                    │
///                    ▼
///               ┌──────────┐
///               │  Output  │ 64 → 1
///               └──────────┘
/// ```
///
/// Total parameters: ~7K (vs ~1GB for HalfEP)
#[derive(Clone)]
pub struct FactoredNnue {
    /// Learned embeddings for each operation.
    pub embeddings: OpEmbeddings,

    /// Hidden layer weights: [INPUT_DIM][HIDDEN_DIM]
    pub w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// Hidden layer biases: [HIDDEN_DIM]
    pub b1: [f32; HIDDEN_DIM],

    /// Output layer weights: [HIDDEN_DIM]
    pub w2: [f32; HIDDEN_DIM],

    /// Output layer bias
    pub b2: f32,
}

impl Default for FactoredNnue {
    fn default() -> Self {
        Self::new()
    }
}

impl FactoredNnue {
    /// Create a zero-initialized network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            embeddings: OpEmbeddings::new(),
            w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            b1: [0.0; HIDDEN_DIM],
            w2: [0.0; HIDDEN_DIM],
            b2: 0.0,
        }
    }

    /// Create a randomly initialized network.
    #[must_use]
    pub fn new_random(seed: u64) -> Self {
        let mut net = Self::new();
        net.randomize(seed);
        net
    }

    /// Create a network with latency-prior initialized embeddings.
    ///
    /// This is the recommended initialization for cost prediction:
    /// - Embeddings encode known op latencies in dimension 0
    /// - Network weights are randomly initialized
    /// - Scales to any number of ops (just add latency for new ops)
    #[must_use]
    pub fn new_with_latency_prior(seed: u64) -> Self {
        let mut net = Self::new();
        net.embeddings.init_with_latency_prior(seed);
        net.randomize_weights_only(seed);
        net
    }

    /// Randomize only network weights, not embeddings.
    pub fn randomize_weights_only(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(12345);

        let mut next_f32 = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        let scale_w1 = sqrtf(2.0 / INPUT_DIM as f32);
        for row in 0..INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.w1[row][col] = next_f32() * scale_w1;
            }
        }

        for b in &mut self.b1 {
            *b = next_f32().abs() * 0.1;
        }

        let scale_w2 = sqrtf(2.0 / HIDDEN_DIM as f32);
        for w in &mut self.w2 {
            *w = next_f32() * scale_w2;
        }

        self.b2 = 5.0;
    }

    /// Randomize weights using He initialization.
    pub fn randomize(&mut self, seed: u64) {
        self.embeddings.randomize(seed);

        let mut rng_state = seed.wrapping_add(12345);

        // Helper for random generation
        let mut next_f32 = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        // W1: He initialization with fan_in = INPUT_DIM
        let scale_w1 = sqrtf(2.0 / INPUT_DIM as f32);
        for row in 0..INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.w1[row][col] = next_f32() * scale_w1;
            }
        }

        // B1: Small positive bias for ReLU
        for b in &mut self.b1 {
            *b = next_f32().abs() * 0.1;
        }

        // W2: He initialization with fan_in = HIDDEN_DIM
        let scale_w2 = sqrtf(2.0 / HIDDEN_DIM as f32);
        for w in &mut self.w2 {
            *w = next_f32() * scale_w2;
        }

        // B2: Start near typical cost values (log scale)
        self.b2 = 5.0; // ln(~150) as reasonable starting point
    }

    /// Forward pass: predict log-cost from accumulator and structural features.
    ///
    /// Returns predicted cost in log-scale (apply exp() to get nanoseconds).
    #[must_use]
    pub fn forward(&self, acc: &EdgeAccumulator, structural: &StructuralFeatures) -> f32 {
        // Combine inputs: [acc.values (64) | structural.values (20)]
        let mut hidden = self.b1;

        // Process edge accumulator (first 64 dims)
        for (i, &val) in acc.values.iter().enumerate() {
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += val * self.w1[i][j];
            }
        }

        // Process structural features (next 20 dims)
        for (i, &val) in structural.values.iter().enumerate() {
            let input_idx = 2 * K + i;
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += val * self.w1[input_idx][j];
            }
        }

        // ReLU activation
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Output layer
        let mut output = self.b2;
        for (h, &w) in hidden.iter().zip(self.w2.iter()) {
            output += h * w;
        }

        output
    }

    /// Convenience method: evaluate an expression directly.
    ///
    /// Builds the accumulator and structural features, then runs forward pass.
    #[must_use]
    pub fn evaluate(&self, expr: &Expr) -> f32 {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        self.forward(&acc, &structural)
    }

    /// Predict actual cost in nanoseconds (exp of log-cost).
    #[must_use]
    pub fn predict_ns(&self, expr: &Expr) -> f32 {
        libm::expf(self.evaluate(expr))
    }

    /// Total parameter count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpEmbeddings::param_count()           // embeddings: 42 * 32 = 1,344
            + INPUT_DIM * HIDDEN_DIM          // w1: 85 * 64 = 5,440
            + HIDDEN_DIM                      // b1: 64
            + HIDDEN_DIM                      // w2: 64
            + 1                               // b2: 1
        // Total: 6,913 parameters
    }

    /// Memory size in bytes (f32 weights).
    #[must_use]
    pub const fn memory_bytes() -> usize {
        Self::param_count() * 4
    }
}

// ============================================================================
// Edge Extraction Utilities
// ============================================================================

/// An edge in the expression tree: (parent_op, child_op).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    /// The parent operation type.
    pub parent: OpKind,
    /// The child operation type.
    pub child: OpKind,
}

/// Extract all parent→child edges from an expression tree.
#[must_use]
pub fn extract_edges(expr: &Expr) -> Vec<Edge> {
    let mut edges = Vec::new();
    extract_edges_recursive(expr, &mut edges);
    edges
}

fn extract_edges_recursive(expr: &Expr, edges: &mut Vec<Edge>) {
    let parent = expr.op_type();

    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Unary(_, child) => {
            edges.push(Edge {
                parent,
                child: child.op_type(),
            });
            extract_edges_recursive(child, edges);
        }
        Expr::Binary(_, left, right) => {
            edges.push(Edge {
                parent,
                child: left.op_type(),
            });
            edges.push(Edge {
                parent,
                child: right.op_type(),
            });
            extract_edges_recursive(left, edges);
            extract_edges_recursive(right, edges);
        }
        Expr::Ternary(_, a, b, c) => {
            edges.push(Edge {
                parent,
                child: a.op_type(),
            });
            edges.push(Edge {
                parent,
                child: b.op_type(),
            });
            edges.push(Edge {
                parent,
                child: c.op_type(),
            });
            extract_edges_recursive(a, edges);
            extract_edges_recursive(b, edges);
            extract_edges_recursive(c, edges);
        }
        Expr::Nary(_, children) => {
            for child in children {
                edges.push(Edge {
                    parent,
                    child: child.op_type(),
                });
                extract_edges_recursive(child, edges);
            }
        }
    }
}

// ============================================================================
// Dual-Head NNUE (AlphaZero-style)
// ============================================================================

/// Dual-head NNUE with shared embeddings and two output heads.
///
/// ## AlphaZero-Inspired Architecture
///
/// | AlphaZero | PixelFlow |
/// |-----------|-----------|
/// | Shared CNN backbone | Shared `OpEmbeddings` + hidden layer |
/// | Policy head: which move? | Search head: which state to expand? |
/// | Value head: who wins? | Value head: how many ns? |
///
/// ## Architecture
///
/// ```text
///                     ┌─────────────────────────────┐
///                     │   Shared OpEmbeddings       │
///                     │   (42 ops × 32 dims)        │
///                     │         +                   │
///                     │   Shared Hidden Layer       │
///                     │   (84 → 64, ReLU)           │
///                     └──────────┬──────────────────┘
///                                │
///               ┌────────────────┴────────────────┐
///               ▼                                 ▼
///     ┌─────────────────┐               ┌─────────────────┐
///     │   Value Head    │               │   Search Head   │
///     │   (64 → 1)      │               │   (64 → 1)      │
///     │                 │               │                 │
///     │ "Cost in ns"    │               │ "Heap priority" │
///     └─────────────────┘               └─────────────────┘
///           │                                   │
///           ▼                                   ▼
///     Used for EXTRACTION              Used for SEARCH
///     (pick lowest cost expr)          (A* queue ordering)
/// ```
///
/// ## Multi-Task Learning Benefits
///
/// - Shared representations learn richer features
/// - Regularization effect (harder to overfit)
/// - Transfer learning between tasks
/// - Both tasks need the same knowledge: "what ops cost"
///
/// ## Training Strategy
///
/// 1. **Phase 1**: Bootstrap value head with SIMD benchmarks
/// 2. **Phase 2**: Train search head with search trajectories
/// 3. **Phase 3**: Joint fine-tuning
///
/// Total parameters: ~6,914 (~28KB)
#[derive(Clone)]
pub struct DualHeadNnue {
    // ========== SHARED (Expression Backbone) ==========
    /// Learned embeddings for each operation (42 × 32 = 1,344 params)
    pub embeddings: OpEmbeddings,

    /// Hidden layer weights: [INPUT_DIM][HIDDEN_DIM] (84 × 64 = 5,376 params)
    pub w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// Hidden layer biases: [HIDDEN_DIM] (64 params)
    pub b1: [f32; HIDDEN_DIM],

    // ========== VALUE HEAD (The Judge) ==========
    /// Value head weights: [HIDDEN_DIM] (64 params)
    pub value_w: [f32; HIDDEN_DIM],

    /// Value head bias (1 param)
    pub value_b: f32,

    // ========== RULE HEAD (For Policy) ==========
    /// Rule input dimension (rule features)
    /// Rule first layer: [RULE_INPUT_DIM][HIDDEN_DIM]
    pub rule_w1: [[f32; HIDDEN_DIM]; RULE_INPUT_DIM],

    /// Rule first layer bias
    pub rule_b1: [f32; HIDDEN_DIM],

    // ========== SEARCH HEAD (Legacy - scalar priority) ==========
    /// Search head weights: [HIDDEN_DIM] (64 params)
    pub search_w: [f32; HIDDEN_DIM],

    /// Search head bias (1 param)
    pub search_b: f32,
}

impl Default for DualHeadNnue {
    fn default() -> Self {
        Self::new()
    }
}

impl DualHeadNnue {
    /// Create a zero-initialized dual-head network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            embeddings: OpEmbeddings::new(),
            w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            b1: [0.0; HIDDEN_DIM],
            value_w: [0.0; HIDDEN_DIM],
            value_b: 0.0,
            rule_w1: [[0.0; HIDDEN_DIM]; RULE_INPUT_DIM],
            rule_b1: [0.0; HIDDEN_DIM],
            search_w: [0.0; HIDDEN_DIM],
            search_b: 0.0,
        }
    }

    /// Create a randomly initialized dual-head network.
    #[must_use]
    pub fn new_random(seed: u64) -> Self {
        let mut net = Self::new();
        net.randomize(seed);
        net
    }

    /// Create a network with latency-prior initialized embeddings.
    ///
    /// Recommended initialization for cost prediction:
    /// - Embeddings encode known op latencies in dimension 0
    /// - Network weights are randomly initialized
    #[must_use]
    pub fn new_with_latency_prior(seed: u64) -> Self {
        let mut net = Self::new();
        net.embeddings.init_with_latency_prior(seed);
        net.randomize_weights_only(seed);
        net
    }

    /// Convert from single-head FactoredNnue (reuse trained embeddings).
    ///
    /// The value head inherits the original output weights.
    /// The rule head and search head start at zero (need training).
    #[must_use]
    pub fn from_factored(factored: &FactoredNnue) -> Self {
        Self {
            embeddings: factored.embeddings.clone(),
            w1: factored.w1,
            b1: factored.b1,
            // Value head inherits the original output
            value_w: factored.w2,
            value_b: factored.b2,
            // Rule head starts fresh (needs training)
            rule_w1: [[0.0; HIDDEN_DIM]; RULE_INPUT_DIM],
            rule_b1: [0.0; HIDDEN_DIM],
            // Search head starts fresh (needs training)
            search_w: [0.0; HIDDEN_DIM],
            search_b: 0.0,
        }
    }

    /// Randomize only network weights, not embeddings.
    pub fn randomize_weights_only(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(12345);

        let mut next_f32 = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        // Hidden layer
        let scale_w1 = sqrtf(2.0 / INPUT_DIM as f32);
        for row in 0..INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.w1[row][col] = next_f32() * scale_w1;
            }
        }

        for b in &mut self.b1 {
            *b = next_f32().abs() * 0.1;
        }

        // Value head
        let scale_head = sqrtf(2.0 / HIDDEN_DIM as f32);
        for w in &mut self.value_w {
            *w = next_f32() * scale_head;
        }
        self.value_b = 5.0; // Start near typical log-cost

        // Rule head (for policy)
        let scale_rule = sqrtf(2.0 / RULE_INPUT_DIM as f32);
        for row in 0..RULE_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.rule_w1[row][col] = next_f32() * scale_rule;
            }
        }
        for b in &mut self.rule_b1 {
            *b = next_f32().abs() * 0.1;
        }

        // Search head (legacy)
        for w in &mut self.search_w {
            *w = next_f32() * scale_head;
        }
        self.search_b = 0.0; // Neutral starting priority
    }

    /// Randomize all weights including embeddings.
    pub fn randomize(&mut self, seed: u64) {
        self.embeddings.randomize(seed);
        self.randomize_weights_only(seed);
    }

    /// Shared forward pass through embeddings + hidden layer.
    ///
    /// Returns the hidden layer activations after ReLU.
    #[inline]
    fn forward_shared(&self, acc: &EdgeAccumulator, structural: &StructuralFeatures) -> [f32; HIDDEN_DIM] {
        let mut hidden = self.b1;

        // Process edge accumulator (first 64 dims)
        for (i, &val) in acc.values.iter().enumerate() {
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += val * self.w1[i][j];
            }
        }

        // Process structural features (next 20 dims)
        for (i, &val) in structural.values.iter().enumerate() {
            let input_idx = 2 * K + i;
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += val * self.w1[input_idx][j];
            }
        }

        // ReLU activation
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        hidden
    }

    /// Value head: predict cost in log-nanoseconds.
    ///
    /// Used for **extraction**: pick the lowest-cost expression from an e-class.
    /// Apply exp() to get actual nanoseconds.
    #[must_use]
    pub fn predict_log_cost(&self, expr: &Expr) -> f32 {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        dot(&hidden, &self.value_w) + self.value_b
    }

    /// Value head: predict cost in nanoseconds (exp of log-cost).
    ///
    /// Convenience method that applies exp() to log-cost.
    #[must_use]
    pub fn predict_cost(&self, expr: &Expr) -> f32 {
        libm::expf(self.predict_log_cost(expr))
    }

    /// Value head with pre-computed features.
    ///
    /// More efficient when you already have the accumulator and features.
    #[must_use]
    pub fn predict_log_cost_with_features(
        &self,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
    ) -> f32 {
        let hidden = self.forward_shared(acc, structural);
        dot(&hidden, &self.value_w) + self.value_b
    }

    /// Search head: predict priority for A* heap.
    ///
    /// Used for **search**: order states in the priority queue.
    /// Lower values = higher priority (expand first).
    #[must_use]
    pub fn predict_priority(&self, expr: &Expr) -> i64 {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        self.predict_priority_with_features(&acc, &structural)
    }

    /// Search head with pre-computed features.
    ///
    /// More efficient when you already have the accumulator and features.
    #[must_use]
    pub fn predict_priority_with_features(
        &self,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
    ) -> i64 {
        let hidden = self.forward_shared(acc, structural);
        let score = dot(&hidden, &self.search_w) + self.search_b;
        // Convert to i64 for heap ordering (multiply by scale factor for precision)
        (score * 1000.0) as i64
    }

    // ========================================================================
    // Policy Head: (expr, rule) → approve/reject
    // ========================================================================

    /// Compute rule embedding from rule features.
    ///
    /// Returns hidden activations [HIDDEN_DIM] that can be dotted with expr hidden.
    #[must_use]
    pub fn embed_rule(&self, rule_features: &[f32; RULE_INPUT_DIM]) -> [f32; HIDDEN_DIM] {
        let mut hidden = self.rule_b1;

        for i in 0..RULE_INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                hidden[j] += rule_features[i] * self.rule_w1[i][j];
            }
        }

        // ReLU
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        hidden
    }

    /// Policy: score an (expr, rule) pair.
    ///
    /// Score = dot(expr_hidden, rule_hidden)
    /// High score → likely to fire and improve
    #[must_use]
    pub fn score_policy(&self, expr: &Expr, rule_features: &[f32; RULE_INPUT_DIM]) -> f32 {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let expr_hidden = self.forward_shared(&acc, &structural);
        let rule_hidden = self.embed_rule(rule_features);
        dot(&expr_hidden, &rule_hidden)
    }

    /// Policy with pre-computed expression features.
    #[must_use]
    pub fn score_policy_with_features(
        &self,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
        rule_features: &[f32; RULE_INPUT_DIM],
    ) -> f32 {
        let expr_hidden = self.forward_shared(acc, structural);
        let rule_hidden = self.embed_rule(rule_features);
        dot(&expr_hidden, &rule_hidden)
    }

    /// Batch score: evaluate multiple (expr, rule) pairs efficiently.
    ///
    /// Returns scores for each pair in `rule_features`.
    pub fn score_policy_batch(
        &self,
        expr: &Expr,
        rule_features_batch: &[[f32; RULE_INPUT_DIM]],
    ) -> Vec<f32> {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let expr_hidden = self.forward_shared(&acc, &structural);

        rule_features_batch
            .iter()
            .map(|rule_feat| {
                let rule_hidden = self.embed_rule(rule_feat);
                dot(&expr_hidden, &rule_hidden)
            })
            .collect()
    }

    /// Total parameter count.
    ///
    /// Parameter count including rule head.
    #[must_use]
    pub const fn param_count() -> usize {
        OpEmbeddings::param_count()           // embeddings: 42 * 32 = 1,344
            + INPUT_DIM * HIDDEN_DIM          // w1: 85 * 64 = 5,440
            + HIDDEN_DIM                      // b1: 64
            + HIDDEN_DIM                      // value_w: 64
            + 1                               // value_b: 1
            + RULE_INPUT_DIM * HIDDEN_DIM     // rule_w1: 9 * 64 = 576
            + HIDDEN_DIM                      // rule_b1: 64
            + HIDDEN_DIM                      // search_w: 64
            + 1                               // search_b: 1
        // Total: ~7,618 parameters (~30KB)
    }

    /// Memory size in bytes (f32 weights).
    #[must_use]
    pub const fn memory_bytes() -> usize {
        Self::param_count() * 4
    }

    /// Save weights to a binary file.
    ///
    /// Format: magic "DUAL" + all weights as little-endian f32.
    #[cfg(feature = "std")]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Magic header
        file.write_all(b"DUAL")?;

        // Embeddings
        for row in &self.embeddings.e {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // Hidden layer
        for row in &self.w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.b1 {
            file.write_all(&val.to_le_bytes())?;
        }

        // Value head
        for &val in &self.value_w {
            file.write_all(&val.to_le_bytes())?;
        }
        file.write_all(&self.value_b.to_le_bytes())?;

        // Search head
        for &val in &self.search_w {
            file.write_all(&val.to_le_bytes())?;
        }
        file.write_all(&self.search_b.to_le_bytes())?;

        Ok(())
    }

    /// Load weights from a binary file.
    #[cfg(feature = "std")]
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Check magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"DUAL" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid DualHeadNnue file magic (expected 'DUAL')",
            ));
        }

        let mut net = Self::new();

        // Read embeddings
        for row in &mut net.embeddings.e {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }

        // Read hidden layer
        for row in &mut net.w1 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.b1 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Read value head
        for val in &mut net.value_w {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            net.value_b = f32::from_le_bytes(buf);
        }

        // Read search head
        for val in &mut net.search_w {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            net.search_b = f32::from_le_bytes(buf);
        }

        Ok(net)
    }

    // =========================================================================
    // Training Methods
    // =========================================================================

    /// Forward pass storing intermediate activations for backprop.
    ///
    /// Returns (prediction, hidden_activations).
    #[must_use]
    pub fn forward_value_with_hidden(&self, expr: &Expr) -> (f32, [f32; HIDDEN_DIM]) {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        let prediction = dot(&hidden, &self.value_w) + self.value_b;
        (prediction, hidden)
    }

    /// SGD update for value head only.
    ///
    /// Given prediction, target, and hidden activations, updates value_w and value_b.
    /// Returns the squared error for this sample.
    ///
    /// # Arguments
    /// * `hidden` - Hidden layer activations from forward pass
    /// * `prediction` - Model prediction (from forward pass)
    /// * `target` - Ground truth target value
    /// * `learning_rate` - Learning rate for SGD
    ///
    /// # Returns
    /// The squared error (prediction - target)^2
    pub fn sgd_update_value_head(
        &mut self,
        hidden: &[f32; HIDDEN_DIM],
        prediction: f32,
        target: f32,
        learning_rate: f32,
    ) -> f32 {
        let error = prediction - target;
        let squared_error = error * error;

        // Gradient: d_loss/d_value_w = 2 * error * hidden
        // Gradient: d_loss/d_value_b = 2 * error
        let grad_scale = 2.0 * error * learning_rate;

        for (i, &h) in hidden.iter().enumerate() {
            self.value_w[i] -= grad_scale * h;
        }
        self.value_b -= grad_scale;

        squared_error
    }

    /// Train value head on a single sample.
    ///
    /// Convenience method that combines forward pass and gradient update.
    /// Returns the squared error for this sample.
    pub fn train_value_step(
        &mut self,
        expr: &Expr,
        target: f32,
        learning_rate: f32,
    ) -> f32 {
        let (prediction, hidden) = self.forward_value_with_hidden(expr);
        self.sgd_update_value_head(&hidden, prediction, target, learning_rate)
    }

    /// SGD update for search head only.
    ///
    /// Used for Guide training (AlphaZero-style).
    pub fn sgd_update_search_head(
        &mut self,
        hidden: &[f32; HIDDEN_DIM],
        prediction: f32,
        target: f32,
        learning_rate: f32,
    ) -> f32 {
        let error = prediction - target;
        let squared_error = error * error;

        let grad_scale = 2.0 * error * learning_rate;

        for (i, &h) in hidden.iter().enumerate() {
            self.search_w[i] -= grad_scale * h;
        }
        self.search_b -= grad_scale;

        squared_error
    }

    /// Forward pass for search head storing intermediate activations.
    #[must_use]
    pub fn forward_search_with_hidden(&self, expr: &Expr) -> (f32, [f32; HIDDEN_DIM]) {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        let prediction = dot(&hidden, &self.search_w) + self.search_b;
        (prediction, hidden)
    }
}

/// Dot product of two arrays.
#[inline]
fn dot(a: &[f32; HIDDEN_DIM], b: &[f32; HIDDEN_DIM]) -> f32 {
    let mut sum = 0.0;
    for i in 0..HIDDEN_DIM {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

    /// Create a simple expression: x + y
    fn make_add_xy() -> Expr {
        Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        )
    }

    /// Create FMA-eligible expression: a*b + c
    fn make_fma_pattern() -> Expr {
        Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        )
    }

    /// Create non-FMA pattern: a + b*c (Mul under Add, but on right side)
    fn make_add_mul_pattern() -> Expr {
        Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Var(2)),
            )),
        )
    }

    #[test]
    fn test_edge_extraction() {
        let expr = make_add_xy();
        let edges = extract_edges(&expr);

        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].parent, OpKind::Add);
        assert_eq!(edges[0].child, OpKind::Var);
        assert_eq!(edges[1].parent, OpKind::Add);
        assert_eq!(edges[1].child, OpKind::Var);
    }

    #[test]
    fn test_fma_edges() {
        let expr = make_fma_pattern();
        let edges = extract_edges(&expr);

        // Should have: Add→Mul, Add→Var, Mul→Var, Mul→Var
        assert_eq!(edges.len(), 4);

        // Check that Add→Mul exists (the FMA-critical edge)
        let has_add_mul = edges.iter().any(|e| e.parent == OpKind::Add && e.child == OpKind::Mul);
        assert!(has_add_mul, "Should have Add→Mul edge for FMA pattern");
    }

    #[test]
    fn test_asymmetric_accumulator() {
        let emb = OpEmbeddings::new_random(42);

        // Mul→Add (under Add parent)
        let fma = make_fma_pattern();
        let acc_fma = EdgeAccumulator::from_expr(&fma, &emb);

        // Add→Mul (Mul under Add, same ops but different structure)
        let add_mul = make_add_mul_pattern();
        let _acc_add_mul = EdgeAccumulator::from_expr(&add_mul, &emb);

        // The accumulators should be different because:
        // - FMA has Add→Mul, Add→Var edges
        // - ADD_MUL has Add→Var, Add→Mul edges
        // Wait, these are actually the same edges just in different order!
        // The key difference is in the CHILD subexpressions.

        // Actually, let's compare with a truly different pattern:
        // x * (y + z) vs (x * y) + z

        let mul_add = Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Var(2)),
            )),
        );

        let acc_mul_add = EdgeAccumulator::from_expr(&mul_add, &emb);

        // These should definitely differ:
        // - FMA (a*b + c): edges are Add→Mul, Add→Var, Mul→Var, Mul→Var
        // - mul_add (a * (b+c)): edges are Mul→Var, Mul→Add, Add→Var, Add→Var

        // They have different edge sets, so accumulators should differ
        let diff: f32 = acc_fma
            .values
            .iter()
            .zip(acc_mul_add.values.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 0.1, "Asymmetric patterns should produce different accumulators");
    }

    #[test]
    fn test_incremental_update() {
        let emb = OpEmbeddings::new_random(42);
        let expr = make_fma_pattern();

        // Build accumulator from scratch
        let acc_full = EdgeAccumulator::from_expr(&expr, &emb);

        // Build incrementally: start empty, add edges
        let mut acc_inc = EdgeAccumulator::new();
        acc_inc.add_expr_edges(&expr, &emb);

        // Should match
        for i in 0..acc_full.values.len() {
            assert!(
                (acc_full.values[i] - acc_inc.values[i]).abs() < 1e-6,
                "Incremental build should match full build"
            );
        }

        // Remove and verify we get back to zero
        acc_inc.remove_expr_edges(&expr, &emb);
        for &v in &acc_inc.values {
            assert!(v.abs() < 1e-6, "After removing all edges, accumulator should be zero");
        }
    }

    #[test]
    fn test_structural_features() {
        let expr = make_fma_pattern();
        let features = StructuralFeatures::from_expr(&expr);

        // Should have 5 nodes: Add, Mul, Var, Var, Var
        assert_eq!(features.values[StructuralFeatures::TOTAL_NODES] as u32, 5);

        // 3 variables
        assert_eq!(features.values[StructuralFeatures::VAR_COUNT] as u32, 3);

        // Should detect FMA pattern
        assert!(
            features.values[StructuralFeatures::HAS_FMA_PATTERN] > 0.0,
            "Should detect FMA pattern"
        );

        // 4 edges: Add→Mul, Add→Var, Mul→Var, Mul→Var
        assert_eq!(features.values[StructuralFeatures::EDGE_COUNT] as u32, 4);
    }

    #[test]
    fn test_forward_pass() {
        let net = FactoredNnue::new_random(42);
        let expr = make_fma_pattern();

        let cost = net.evaluate(&expr);

        // Should be a reasonable value (not NaN, not infinity)
        assert!(cost.is_finite(), "Cost should be finite");
    }

    #[test]
    fn test_param_count() {
        // Verify our parameter count calculation
        // INPUT_DIM = 2*K + STRUCTURAL_FEATURE_COUNT = 2*32 + 21 = 85
        assert_eq!(
            FactoredNnue::param_count(),
            42 * 32          // embeddings: OpKind::COUNT * K
            + 85 * 64        // w1: INPUT_DIM * HIDDEN_DIM
            + 64             // b1: HIDDEN_DIM
            + 64             // w2: HIDDEN_DIM
            + 1              // b2
        );

        // Should be much smaller than HalfEP
        assert!(
            FactoredNnue::memory_bytes() < 100_000,
            "Factored NNUE should use < 100KB"
        );
    }

    #[test]
    fn test_different_expressions_different_costs() {
        let net = FactoredNnue::new_random(42);

        let simple = Expr::Var(0);
        let complex = Expr::Binary(
            OpKind::Div,
            Box::new(Expr::Unary(OpKind::Sqrt, Box::new(Expr::Var(0)))),
            Box::new(Expr::Unary(OpKind::Sqrt, Box::new(Expr::Var(1)))),
        );

        let cost_simple = net.evaluate(&simple);
        let cost_complex = net.evaluate(&complex);

        // Complex expression should generally have higher predicted cost
        // (though with random weights this isn't guaranteed, just check they differ)
        assert!(
            (cost_simple - cost_complex).abs() > 1e-6,
            "Different expressions should produce different costs"
        );
    }

    // ========================================================================
    // DualHeadNnue Tests
    // ========================================================================

    #[test]
    fn test_dual_head_param_count() {
        // Verify our parameter count calculation
        // INPUT_DIM = 2*K + STRUCTURAL_FEATURE_COUNT = 2*32 + 21 = 85
        assert_eq!(
            DualHeadNnue::param_count(),
            42 * 32          // embeddings: OpKind::COUNT * K
            + 85 * 64        // w1: INPUT_DIM * HIDDEN_DIM
            + 64             // b1: HIDDEN_DIM
            + 64             // value_w: HIDDEN_DIM
            + 1              // value_b
            + 9 * 64         // rule_w1: RULE_INPUT_DIM * HIDDEN_DIM
            + 64             // rule_b1: HIDDEN_DIM
            + 64             // search_w: HIDDEN_DIM
            + 1              // search_b
        );

        // Should be slightly larger than single-head
        assert!(
            DualHeadNnue::param_count() > FactoredNnue::param_count(),
            "Dual-head should have more params than single-head"
        );

        // But still under 100KB
        assert!(
            DualHeadNnue::memory_bytes() < 100_000,
            "Dual-head NNUE should use < 100KB"
        );
    }

    #[test]
    fn test_dual_head_value_prediction() {
        let net = DualHeadNnue::new_random(42);
        let expr = make_fma_pattern();

        let log_cost = net.predict_log_cost(&expr);
        let cost = net.predict_cost(&expr);

        // Log cost should be finite
        assert!(log_cost.is_finite(), "Log cost should be finite");

        // Cost should be exp(log_cost)
        let expected = libm::expf(log_cost);
        assert!(
            (cost - expected).abs() < 1e-4,
            "predict_cost should be exp(predict_log_cost)"
        );
    }

    #[test]
    fn test_dual_head_search_prediction() {
        let net = DualHeadNnue::new_random(42);
        let expr = make_fma_pattern();

        let priority = net.predict_priority(&expr);

        // Priority should be a reasonable integer (not extremes)
        assert!(
            priority.abs() < i64::MAX / 2,
            "Priority should be a reasonable value"
        );
    }

    #[test]
    fn test_dual_head_two_heads_differ() {
        let net = DualHeadNnue::new_random(42);
        let expr = make_fma_pattern();

        let log_cost = net.predict_log_cost(&expr);
        let priority = net.predict_priority(&expr) as f32 / 1000.0; // Undo scaling

        // With random initialization, the two heads should produce different values
        // (they have different weight matrices)
        assert!(
            (log_cost - priority).abs() > 1e-6,
            "Value and search heads should produce different outputs"
        );
    }

    #[test]
    fn test_dual_head_from_factored() {
        let factored = FactoredNnue::new_random(42);
        let expr = make_fma_pattern();

        let factored_cost = factored.evaluate(&expr);

        // Convert to dual-head
        let dual = DualHeadNnue::from_factored(&factored);
        let dual_cost = dual.predict_log_cost(&expr);

        // Value head should match original output
        assert!(
            (factored_cost - dual_cost).abs() < 1e-6,
            "Value head should inherit from factored: factored={}, dual={}",
            factored_cost,
            dual_cost
        );

        // Search head should be zero (untrained)
        let priority = dual.predict_priority(&expr);
        // With zero weights, priority = search_b (scaled by 1000)
        assert_eq!(
            priority, 0,
            "Search head should be zero when freshly converted"
        );
    }

    #[test]
    fn test_dual_head_latency_prior() {
        // Test that latency priors are correctly set in embeddings.
        // Note: Random network weights can overwhelm these priors - this test
        // verifies initialization, not that untrained predictions are correct.
        let net = DualHeadNnue::new_with_latency_prior(42);

        // Check that expensive ops have higher latency values in dim 0
        let var_latency = net.embeddings.get(OpKind::Var)[0];
        let div_latency = net.embeddings.get(OpKind::Div)[0];
        let sqrt_latency = net.embeddings.get(OpKind::Sqrt)[0];

        // Var should be cheap (0.0 latency)
        assert!(
            var_latency < 0.1,
            "Var latency should be near zero: {}",
            var_latency
        );

        // Div and Sqrt should be expensive (0.75 latency)
        assert!(
            div_latency > 0.5,
            "Div latency should be high: {}",
            div_latency
        );
        assert!(
            sqrt_latency > 0.5,
            "Sqrt latency should be high: {}",
            sqrt_latency
        );

        // Verify the network can make predictions (no NaN/infinity)
        let expr = Expr::Var(0);
        let cost = net.predict_cost(&expr);
        assert!(cost.is_finite(), "Prediction should be finite");
    }

    #[test]
    fn test_policy_scoring() {
        let net = DualHeadNnue::new_random(42);
        let expr = make_fma_pattern();

        // Create some rule features
        let rule_features: [f32; RULE_INPUT_DIM] = [
            0.5,  // rule_idx normalized
            0.3,  // graph_size
            0.2,  // node_count
            0.1,  // time_used
            0.8,  // match_rate
            0.0,  // staleness
            0.9,  // time_remaining
            0.7,  // space_remaining
            0.6,  // pairs_remaining
        ];

        // Score should be finite
        let score = net.score_policy(&expr, &rule_features);
        assert!(score.is_finite(), "Policy score should be finite: {}", score);

        // Different rules should get different scores
        let mut other_rule = rule_features;
        other_rule[0] = 0.9; // different rule
        other_rule[4] = 0.1; // different match rate
        let other_score = net.score_policy(&expr, &other_rule);
        assert!(
            (score - other_score).abs() > 1e-6,
            "Different rules should get different scores: {} vs {}",
            score, other_score
        );

        // Batch scoring should match individual
        let batch_scores = net.score_policy_batch(&expr, &[rule_features, other_rule]);
        assert!((batch_scores[0] - score).abs() < 1e-6);
        assert!((batch_scores[1] - other_score).abs() < 1e-6);
    }
}
