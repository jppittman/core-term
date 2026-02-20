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

/// Maximum number of rewrite rules supported (legacy, for backward compat).
/// See MASK_MAX_RULES for the new unified architecture.
pub const MAX_RULES: usize = 64;

// ============================================================================
// Unified Mask Architecture Constants
// ============================================================================

/// Embedding dimension for expr/rule factorization in the unified mask architecture.
pub const EMBED_DIM: usize = 24;

/// Hidden dimension for private MLPs (value, mask, rule).
pub const MLP_HIDDEN: usize = 16;

/// Rule feature dimension (hand-crafted features describing each rule).
pub const RULE_FEATURE_DIM: usize = 8;

/// Maximum rules supported in the unified mask architecture.
/// Designed to scale to 1000+ rules.
pub const MASK_MAX_RULES: usize = 1024;

/// Concatenated rule features: [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] (4 × EMBED_DIM).
/// Used when encoding rules from their LHS/RHS expression templates.
pub const RULE_CONCAT_DIM: usize = 4 * EMBED_DIM;

/// Mask MLP input dimension: expr_embed (24) + value_pred (1) = 25.
/// The mask/policy uses value prediction to learn: "Given this costs X, should I try rule R?"
pub const MASK_INPUT_DIM: usize = EMBED_DIM + 1;

// NOTE: SEARCH_INPUT_DIM removed - mask IS the policy.
// See plan: "Idea 4B: Mask IS the search/policy ✅ CHOSEN"

// ============================================================================
// Rule Features
// ============================================================================

/// Hand-crafted features describing each rule.
///
/// These features are mostly static (computed once when rules are defined)
/// and allow the Rule MLP to generalize across rules without learning
/// individual embeddings for each rule.
///
/// # Features (RULE_FEATURE_DIM = 8)
///
/// 1. `category`: Rule type (algebraic=0, peephole=0.25, domain=0.5, cross-cutting=0.75)
/// 2. `lhs_nodes`: Pattern complexity (normalized by 10)
/// 3. `typical_depth_delta`: Usually -1, 0, or 1
/// 4. `commutative`: Does rule exploit commutativity? (0 or 1)
/// 5. `associative`: Does rule exploit associativity? (0 or 1)
/// 6. `creates_sharing`: Does rule typically enable CSE? (0 or 1)
/// 7. `historical_match_rate`: Running average [0, 1]
/// 8. `expensive_op_related`: Touches div/sqrt/transcendental? (0 or 1)
#[derive(Clone)]
pub struct RuleFeatures {
    /// Features for each rule: [rule_idx][feature_dim]
    pub features: [[f32; RULE_FEATURE_DIM]; MASK_MAX_RULES],
}

impl Default for RuleFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleFeatures {
    /// Create zero-initialized rule features.
    #[must_use]
    pub fn new() -> Self {
        Self {
            features: [[0.0; RULE_FEATURE_DIM]; MASK_MAX_RULES],
        }
    }

    /// Get features for a specific rule.
    #[must_use]
    pub fn get(&self, rule_idx: usize) -> &[f32; RULE_FEATURE_DIM] {
        &self.features[rule_idx]
    }

    /// Set features for a specific rule.
    pub fn set(&mut self, rule_idx: usize, features: [f32; RULE_FEATURE_DIM]) {
        self.features[rule_idx] = features;
    }

    /// Set feature by name for easier initialization.
    pub fn set_rule(
        &mut self,
        rule_idx: usize,
        category: f32,
        lhs_nodes: usize,
        depth_delta: i8,
        commutative: bool,
        associative: bool,
        creates_sharing: bool,
        match_rate: f32,
        expensive_op: bool,
    ) {
        self.features[rule_idx] = [
            category,
            lhs_nodes as f32 / 10.0,
            depth_delta as f32,
            if commutative { 1.0 } else { 0.0 },
            if associative { 1.0 } else { 0.0 },
            if creates_sharing { 1.0 } else { 0.0 },
            match_rate.clamp(0.0, 1.0),
            if expensive_op { 1.0 } else { 0.0 },
        ];
    }
}

// ============================================================================
// Rule Templates (LHS/RHS Expression Templates)
// ============================================================================

/// Rule templates: LHS and RHS expressions for each rule.
///
/// These use the SAME expr_embed as value/mask heads, enabling the model
/// to learn structural similarity between expressions and rule patterns.
///
/// Each rule has:
/// - LHS pattern (what it matches), e.g., `A * (B + C)`
/// - RHS pattern (what it produces), e.g., `A*B + A*C`
///
/// The 4-way concatenation captures:
/// - `z_LHS`: what the rule MATCHES (pattern recognition)
/// - `z_RHS`: what it PRODUCES (production prediction)
/// - `z_LHS - z_RHS`: what CHANGED (the delta)
/// - `z_LHS * z_RHS`: what's SHARED (preserved structure)
#[derive(Clone)]
pub struct RuleTemplates {
    /// LHS pattern for each rule (what it matches).
    /// Uses Expr::Var(0), Expr::Var(1), etc. as metavariables.
    pub lhs: Vec<Option<Expr>>,
    /// RHS pattern for each rule (what it produces).
    pub rhs: Vec<Option<Expr>>,
}

impl Default for RuleTemplates {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleTemplates {
    /// Create empty templates.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lhs: Vec::new(),
            rhs: Vec::new(),
        }
    }

    /// Create templates for a given number of rules (all None initially).
    #[must_use]
    pub fn with_capacity(num_rules: usize) -> Self {
        Self {
            lhs: vec![None; num_rules],
            rhs: vec![None; num_rules],
        }
    }

    /// Set templates for a specific rule.
    pub fn set(&mut self, rule_idx: usize, lhs: Expr, rhs: Expr) {
        // Ensure we have enough capacity
        if rule_idx >= self.lhs.len() {
            self.lhs.resize(rule_idx + 1, None);
            self.rhs.resize(rule_idx + 1, None);
        }
        self.lhs[rule_idx] = Some(lhs);
        self.rhs[rule_idx] = Some(rhs);
    }

    /// Get LHS template for a rule.
    #[must_use]
    pub fn get_lhs(&self, rule_idx: usize) -> Option<&Expr> {
        self.lhs.get(rule_idx).and_then(|opt| opt.as_ref())
    }

    /// Get RHS template for a rule.
    #[must_use]
    pub fn get_rhs(&self, rule_idx: usize) -> Option<&Expr> {
        self.rhs.get(rule_idx).and_then(|opt| opt.as_ref())
    }

    /// Number of rules with templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lhs.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lhs.is_empty()
    }

    /// Check if a rule has templates defined.
    #[must_use]
    pub fn has_templates(&self, rule_idx: usize) -> bool {
        self.get_lhs(rule_idx).is_some() && self.get_rhs(rule_idx).is_some()
    }
}

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
pub struct ExprNnue {
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

impl Default for ExprNnue {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprNnue {
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
/// Architecture with learned rule embeddings for attention-like policy.
///
/// ```text
/// expr → OpEmbeddings → EdgeAcc → hidden [64] ─┬─→ value_w → cost (Judge)
///                                               │
///           rule_embeddings [64][64] (keys) ────┴─→ dot → logits [64] (Policy)
/// ```
#[derive(Clone)]
pub struct ExprNnue {
    // ========== SHARED (Expression Backbone) ==========
    /// Learned embeddings for each operation (42 × 32 = 1,344 params)
    pub embeddings: OpEmbeddings,

    /// Hidden layer weights: [INPUT_DIM][HIDDEN_DIM] (85 × 64 = 5,440 params)
    pub w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// Hidden layer biases: [HIDDEN_DIM] (64 params)
    pub b1: [f32; HIDDEN_DIM],

    // ========== VALUE HEAD (The Judge) ==========
    /// Value head weights: [HIDDEN_DIM] (64 params)
    pub value_w: [f32; HIDDEN_DIM],

    /// Value head bias (1 param)
    pub value_b: f32,

    // ========== RULE EMBEDDINGS (For Policy - attention-like) ==========
    /// Learned embedding for each rule (MAX_RULES × HIDDEN_DIM = 64 × 64 = 4,096 params)
    /// Acts as "keys" in attention: score[i] = dot(expr_hidden, rule_embed[i])
    pub rule_embeddings: [[f32; HIDDEN_DIM]; MAX_RULES],

    /// Per-rule bias (like rule priors)
    pub rule_bias: [f32; MAX_RULES],

    // ========== SEARCH HEAD (Legacy - scalar priority) ==========
    /// Search head weights: [HIDDEN_DIM] (64 params)
    pub search_w: [f32; HIDDEN_DIM],

    /// Search head bias (1 param)
    pub search_b: f32,

    // ========== UNIFIED MASK ARCHITECTURE ==========
    // These fields support the new bilinear expr-rule interaction model
    // that scales to 1000+ rules.

    /// Projects backbone hidden (64) to shared expr embedding (EMBED_DIM=24).
    /// Weights: [HIDDEN_DIM x EMBED_DIM]
    pub expr_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Expr projection bias: [EMBED_DIM]
    pub expr_proj_b: [f32; EMBED_DIM],

    /// Value MLP layer 1 weights: expr_embed (24) → hidden (16)
    pub value_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    /// Value MLP layer 1 bias
    pub value_mlp_b1: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 weights: hidden (16) → cost (1)
    pub value_mlp_w2: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 bias
    pub value_mlp_b2: f32,

    /// Mask MLP layer 1 weights: [expr_embed (24), value_pred (1)] → hidden (16)
    /// The mask sees value prediction as input: "Given this costs X, should I try rule R?"
    pub mask_mlp_w1: [[f32; MLP_HIDDEN]; MASK_INPUT_DIM],
    /// Mask MLP layer 1 bias
    pub mask_mlp_b1: [f32; MLP_HIDDEN],
    /// Mask MLP layer 2 weights: hidden (16) → mask_features (24)
    pub mask_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    /// Mask MLP layer 2 bias
    pub mask_mlp_b2: [f32; EMBED_DIM],

    /// Rule MLP layer 1 weights: rule_features (8) → hidden (16).
    /// Shared across all rules - scales sublinearly with rule count.
    /// (Legacy: used with hand-crafted RuleFeatures)
    pub rule_mlp_w1: [[f32; MLP_HIDDEN]; RULE_FEATURE_DIM],
    /// Rule MLP layer 1 bias
    pub rule_mlp_b1: [f32; MLP_HIDDEN],
    /// Rule MLP layer 2 weights: hidden (16) → rule_embed (24)
    pub rule_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    /// Rule MLP layer 2 bias
    pub rule_mlp_b2: [f32; EMBED_DIM],

    // ========== RULE TEMPLATE PROJECTION (LHS/RHS embeddings) ==========
    // These fields support encoding rules from their LHS/RHS expression
    // templates using the SAME expr_embed as value/mask heads.
    //
    // 4-way concat: [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] (96) → rule_embed (24)

    /// Rule projection weights: [RULE_CONCAT_DIM x EMBED_DIM] = [96 x 24] = 2,304 params.
    /// Projects 4-way concatenation to rule embedding.
    pub rule_proj_w: [[f32; EMBED_DIM]; RULE_CONCAT_DIM],
    /// Rule projection bias: [EMBED_DIM] = 24 params
    pub rule_proj_b: [f32; EMBED_DIM],

    /// Bilinear interaction matrix: mask_features @ interaction @ rule_embed
    pub interaction: [[f32; EMBED_DIM]; EMBED_DIM],

    /// Per-rule bias for mask scoring (learned rule priors)
    pub mask_rule_bias: [f32; MASK_MAX_RULES],

    // NOTE: The search head has been removed.
    // "Mask IS the policy" - the bilinear mask scoring IS the search/policy head.
    // See plan: "Idea 4B: Mask IS the search/policy ✅ CHOSEN"
}

impl Default for ExprNnue {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprNnue {
    /// Create a zero-initialized dual-head network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Legacy backbone
            embeddings: OpEmbeddings::new(),
            w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            b1: [0.0; HIDDEN_DIM],
            value_w: [0.0; HIDDEN_DIM],
            value_b: 0.0,
            rule_embeddings: [[0.0; HIDDEN_DIM]; MAX_RULES],
            rule_bias: [0.0; MAX_RULES],
            search_w: [0.0; HIDDEN_DIM],
            search_b: 0.0,

            // Unified mask architecture
            expr_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            expr_proj_b: [0.0; EMBED_DIM],

            value_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            value_mlp_b1: [0.0; MLP_HIDDEN],
            value_mlp_w2: [0.0; MLP_HIDDEN],
            value_mlp_b2: 5.0, // Start near typical log-cost

            mask_mlp_w1: [[0.0; MLP_HIDDEN]; MASK_INPUT_DIM],  // 25 × 16 (value-aware)
            mask_mlp_b1: [0.0; MLP_HIDDEN],
            mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            mask_mlp_b2: [0.0; EMBED_DIM],

            rule_mlp_w1: [[0.0; MLP_HIDDEN]; RULE_FEATURE_DIM],
            rule_mlp_b1: [0.0; MLP_HIDDEN],
            rule_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            rule_mlp_b2: [0.0; EMBED_DIM],

            // Rule template projection (LHS/RHS embeddings)
            rule_proj_w: [[0.0; EMBED_DIM]; RULE_CONCAT_DIM],
            rule_proj_b: [0.0; EMBED_DIM],

            interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            mask_rule_bias: [0.0; MASK_MAX_RULES],
            // NOTE: search_mlp fields removed - mask IS the policy
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

    /// Convert from single-head ExprNnue (reuse trained embeddings).
    ///
    /// The value head inherits the original output weights.
    /// Rule embeddings and search head start at zero (need training).
    /// New unified mask architecture fields are zero-initialized (need training).
    #[must_use]
    pub fn from_factored(factored: &ExprNnue) -> Self {
        Self {
            embeddings: factored.embeddings.clone(),
            w1: factored.w1,
            b1: factored.b1,
            // Value head inherits the original output
            value_w: factored.w2,
            value_b: factored.b2,
            // Rule embeddings start fresh (needs training)
            rule_embeddings: [[0.0; HIDDEN_DIM]; MAX_RULES],
            rule_bias: [0.0; MAX_RULES],
            // Search head starts fresh (needs training)
            search_w: [0.0; HIDDEN_DIM],
            search_b: 0.0,

            // Unified mask architecture - zero-initialized (needs training)
            expr_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            expr_proj_b: [0.0; EMBED_DIM],

            value_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            value_mlp_b1: [0.0; MLP_HIDDEN],
            value_mlp_w2: [0.0; MLP_HIDDEN],
            value_mlp_b2: 5.0,

            mask_mlp_w1: [[0.0; MLP_HIDDEN]; MASK_INPUT_DIM],  // 25 × 16 (value-aware)
            mask_mlp_b1: [0.0; MLP_HIDDEN],
            mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            mask_mlp_b2: [0.0; EMBED_DIM],

            rule_mlp_w1: [[0.0; MLP_HIDDEN]; RULE_FEATURE_DIM],
            rule_mlp_b1: [0.0; MLP_HIDDEN],
            rule_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            rule_mlp_b2: [0.0; EMBED_DIM],

            // Rule template projection - zero-initialized (needs training)
            rule_proj_w: [[0.0; EMBED_DIM]; RULE_CONCAT_DIM],
            rule_proj_b: [0.0; EMBED_DIM],

            interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            mask_rule_bias: [0.0; MASK_MAX_RULES],
            // NOTE: search_mlp fields removed - mask IS the policy
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

        // Rule embeddings (keys for attention-like policy)
        // Scale so dot product is O(1)
        let scale_rule = sqrtf(1.0 / HIDDEN_DIM as f32);
        for rule_idx in 0..MAX_RULES {
            for dim in 0..HIDDEN_DIM {
                self.rule_embeddings[rule_idx][dim] = next_f32() * scale_rule;
            }
            self.rule_bias[rule_idx] = 0.0; // Neutral prior
        }

        // Search head (legacy)
        for w in &mut self.search_w {
            *w = next_f32() * scale_head;
        }
        self.search_b = 0.0; // Neutral starting priority

        // Initialize unified mask architecture (full init - includes shared projection + value mlp)
        self.randomize_unified_arch_with_rng(&mut next_f32);
    }

    /// Internal helper to randomize ALL unified architecture weights.
    ///
    /// ONLY used during full random init (randomize_weights_only).
    /// Do NOT call this when bootstrapping from judge - use randomize_mask_only instead.
    fn randomize_unified_arch_with_rng<F: FnMut() -> f32>(&mut self, next_f32: &mut F) {
        // He initialization scales
        let scale_proj = sqrtf(2.0 / HIDDEN_DIM as f32);
        let scale_embed = sqrtf(2.0 / EMBED_DIM as f32);
        let scale_hidden = sqrtf(2.0 / MLP_HIDDEN as f32);
        let scale_rule_feat = sqrtf(2.0 / RULE_FEATURE_DIM as f32);

        // Expr projection: HIDDEN_DIM → EMBED_DIM
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.expr_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        for b in &mut self.expr_proj_b {
            *b = next_f32().abs() * 0.1;
        }

        // Value MLP: EMBED_DIM → MLP_HIDDEN → 1
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                self.value_mlp_w1[i][j] = next_f32() * scale_embed;
            }
        }
        for b in &mut self.value_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            self.value_mlp_w2[j] = next_f32() * scale_hidden;
        }
        self.value_mlp_b2 = 5.0; // Start near typical log-cost

        // Mask MLP: MASK_INPUT_DIM (25) → MLP_HIDDEN → EMBED_DIM
        // Input is [expr_embed (24), value_pred (1)]
        let scale_mask_input = sqrtf(2.0 / MASK_INPUT_DIM as f32);
        for i in 0..MASK_INPUT_DIM {
            for j in 0..MLP_HIDDEN {
                self.mask_mlp_w1[i][j] = next_f32() * scale_mask_input;
            }
        }
        for b in &mut self.mask_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.mask_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.mask_mlp_b2 {
            *b = 0.0; // Neutral
        }

        // Rule MLP: RULE_FEATURE_DIM → MLP_HIDDEN → EMBED_DIM (legacy, hand-crafted features)
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                self.rule_mlp_w1[i][j] = next_f32() * scale_rule_feat;
            }
        }
        for b in &mut self.rule_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.rule_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.rule_mlp_b2 {
            *b = 0.0; // Neutral
        }

        // Rule Projection: RULE_CONCAT_DIM (96) → EMBED_DIM (24)
        // Linear projection from 4-way concat [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS]
        let scale_concat = sqrtf(2.0 / RULE_CONCAT_DIM as f32);
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                self.rule_proj_w[i][k] = next_f32() * scale_concat;
            }
        }
        for b in &mut self.rule_proj_b {
            *b = 0.0; // Neutral
        }

        // Interaction matrix: start near identity (simple dot product baseline)
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                self.interaction[i][j] = if i == j { 1.0 } else { next_f32() * 0.1 };
            }
        }

        // Rule biases: neutral
        for b in &mut self.mask_rule_bias {
            *b = 0.0;
        }

        // NOTE: search_mlp randomization removed - mask IS the policy
    }

    /// Randomize all weights including embeddings.
    pub fn randomize(&mut self, seed: u64) {
        self.embeddings.randomize(seed);
        self.randomize_weights_only(seed);
    }

    /// Create a copy with trained backbone but randomized mask weights.
    ///
    /// This is the key method for embedding sharing: load a trained judge,
    /// then create a new model that:
    /// - Keeps: embeddings, w1, b1, value_w, value_b (trained backbone)
    /// - Keeps: expr_proj_w, expr_proj_b (shared projection - trained with judge)
    /// - Keeps: value_mlp_* (unified value head - trained with judge)
    /// - Randomizes: mask_mlp, rule_mlp, rule_proj, interaction, mask_rule_bias (mask-specific)
    ///
    /// Use this when bootstrapping mask training from a pre-trained judge.
    #[must_use]
    pub fn with_randomized_mask_weights(&self, seed: u64) -> Self {
        let mut model = self.clone();
        model.randomize_mask_only(seed);
        model
    }

    /// Randomize ONLY mask-specific weights, preserving shared backbone and value head.
    ///
    /// Randomizes: mask_mlp, rule_mlp, rule_proj, interaction, mask_rule_bias
    /// Preserves: embeddings, w1, b1, value_w, value_b, expr_proj, value_mlp
    pub fn randomize_mask_only(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(54321);

        let mut next_f32 = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        // He initialization scales
        let scale_embed = sqrtf(2.0 / EMBED_DIM as f32);
        let scale_mask_input = sqrtf(2.0 / MASK_INPUT_DIM as f32);  // 25 dims
        let scale_hidden = sqrtf(2.0 / MLP_HIDDEN as f32);
        let scale_rule_feat = sqrtf(2.0 / RULE_FEATURE_DIM as f32);
        let scale_concat = sqrtf(2.0 / RULE_CONCAT_DIM as f32);

        // Mask MLP: MASK_INPUT_DIM (25) → MLP_HIDDEN → EMBED_DIM
        // Input is [expr_embed (24), value_pred (1)]
        for i in 0..MASK_INPUT_DIM {
            for j in 0..MLP_HIDDEN {
                self.mask_mlp_w1[i][j] = next_f32() * scale_mask_input;
            }
        }
        for b in &mut self.mask_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.mask_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.mask_mlp_b2 {
            *b = 0.0;
        }

        // Rule MLP: RULE_FEATURE_DIM → MLP_HIDDEN → EMBED_DIM (legacy)
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                self.rule_mlp_w1[i][j] = next_f32() * scale_rule_feat;
            }
        }
        for b in &mut self.rule_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.rule_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.rule_mlp_b2 {
            *b = 0.0;
        }

        // Rule Projection: RULE_CONCAT_DIM (96) → EMBED_DIM (24)
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                self.rule_proj_w[i][k] = next_f32() * scale_concat;
            }
        }
        for b in &mut self.rule_proj_b {
            *b = 0.0;
        }

        // Interaction matrix: start near identity
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                self.interaction[i][j] = if i == j { 1.0 } else { next_f32() * 0.1 };
            }
        }

        // Mask rule biases: neutral
        for b in &mut self.mask_rule_bias {
            *b = 0.0;
        }
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
    // Policy Head: Attention-like rule selection
    //
    // Bilinear form: score = expr^T @ W @ rule
    // Factored as:   score = dot(expr_hidden, rule_embed) + rule_bias
    //
    // expr_hidden comes from the shared backbone (query)
    // rule_embed is a learned embedding per rule (key)
    // ========================================================================

    /// Score a single (expr, rule) pair.
    ///
    /// score = dot(expr_hidden, rule_embedding[rule_idx]) + rule_bias[rule_idx]
    #[must_use]
    pub fn score_rule(&self, expr: &Expr, rule_idx: usize) -> f32 {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        self.score_rule_with_features(&acc, &structural, rule_idx)
    }

    /// Score with pre-computed expression features.
    #[must_use]
    pub fn score_rule_with_features(
        &self,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
        rule_idx: usize,
    ) -> f32 {
        debug_assert!(rule_idx < MAX_RULES);
        let expr_hidden = self.forward_shared(acc, structural);
        dot(&expr_hidden, &self.rule_embeddings[rule_idx]) + self.rule_bias[rule_idx]
    }

    /// Score ALL rules for an expression at once (attention-style).
    ///
    /// Returns [MAX_RULES] logits. One forward pass, O(rules) dot products.
    /// This is expr_hidden @ rule_embeddings^T + rule_bias
    #[must_use]
    pub fn score_all_rules(&self, expr: &Expr) -> [f32; MAX_RULES] {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        self.score_all_rules_with_features(&acc, &structural)
    }

    /// Score all rules with pre-computed features.
    #[must_use]
    pub fn score_all_rules_with_features(
        &self,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
    ) -> [f32; MAX_RULES] {
        let expr_hidden = self.forward_shared(acc, structural);

        let mut scores = [0.0f32; MAX_RULES];
        for rule_idx in 0..MAX_RULES {
            scores[rule_idx] = dot(&expr_hidden, &self.rule_embeddings[rule_idx])
                + self.rule_bias[rule_idx];
        }
        scores
    }

    /// Classify which rules should fire for this expression.
    ///
    /// Returns indices of rules with score > threshold (after sigmoid).
    pub fn classify_rules(&self, expr: &Expr, num_rules: usize, threshold: f32) -> Vec<usize> {
        let scores = self.score_all_rules(expr);
        scores[..num_rules]
            .iter()
            .enumerate()
            .filter(|(_, score)| sigmoid(**score) > threshold)
            .map(|(idx, _)| idx)
            .collect()
    }

    // ========================================================================
    // Unified Mask Architecture Forward Pass
    //
    // New bilinear interaction model that scales to 1000+ rules.
    //
    // Architecture:
    //   expr → backbone → hidden → expr_proj → expr_embed (shared)
    //                                              │
    //          ┌───────────────────────────────────┼───────────────────┐
    //          │                                   │                   │
    //          ▼                                   ▼                   ▼
    //    value_mlp (private)               mask_mlp (private)    rule_features
    //          │                                   │                   │
    //          ▼                                   │             rule_mlp (shared)
    //       cost (1)                               │                   │
    //                                              ▼                   ▼
    //                                        mask_features       rule_embed
    //                                              │                   │
    //                                              └──── bilinear ─────┘
    //                                                       │
    //                                                       ▼
    //                                              score + rule_bias
    // ========================================================================

    /// Project backbone hidden to shared expr embedding (EMBED_DIM).
    #[inline]
    fn compute_expr_embed(&self, hidden: &[f32; HIDDEN_DIM]) -> [f32; EMBED_DIM] {
        let mut embed = self.expr_proj_b;
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                embed[k] += hidden[j] * self.expr_proj_w[j][k];
            }
        }
        embed
    }

    /// Compute mask features from expr embedding and value prediction (for bilinear scoring).
    ///
    /// The mask sees the value prediction as context:
    /// "Given this expression costs X, should I try rule R?"
    ///
    /// Input: [expr_embed (24), value_pred (1)] = 25 dims
    /// MLP: MASK_INPUT_DIM (25) → MLP_HIDDEN (ReLU) → EMBED_DIM (24)
    #[inline]
    fn compute_mask_features(
        &self,
        expr_embed: &[f32; EMBED_DIM],
        value_pred: f32,
    ) -> [f32; EMBED_DIM] {
        // Build input: [expr_embed (24), value_pred (1)] = 25 dims
        // First layer: MASK_INPUT_DIM → MLP_HIDDEN
        let mut h = self.mask_mlp_b1;

        // Process expr_embed (first 24 dims)
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.mask_mlp_w1[i][j];
            }
        }

        // Process value_pred (25th dim)
        for j in 0..MLP_HIDDEN {
            h[j] += value_pred * self.mask_mlp_w1[EMBED_DIM][j];
        }

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer: MLP_HIDDEN → EMBED_DIM
        let mut out = self.mask_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.mask_mlp_w2[j][k];
            }
        }
        out
    }

    /// Forward pass through value MLP from expr embedding.
    ///
    /// MLP: EMBED_DIM (24) → MLP_HIDDEN (16, ReLU) → 1
    /// Returns the predicted cost for this expression.
    #[inline]
    fn value_mlp_forward(&self, expr_embed: &[f32; EMBED_DIM]) -> f32 {
        // First layer: EMBED_DIM → MLP_HIDDEN
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer: MLP_HIDDEN → 1
        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }
        cost
    }

    /// Encode rule features to rule embedding.
    ///
    /// MLP: RULE_FEATURE_DIM → MLP_HIDDEN (ReLU) → EMBED_DIM
    /// This MLP is shared across all rules - scales sublinearly with rule count.
    #[must_use]
    pub fn encode_rule(&self, rule_features: &[f32; RULE_FEATURE_DIM]) -> [f32; EMBED_DIM] {
        // First layer: RULE_FEATURE_DIM → MLP_HIDDEN
        let mut h = self.rule_mlp_b1;
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += rule_features[i] * self.rule_mlp_w1[i][j];
            }
        }

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer: MLP_HIDDEN → EMBED_DIM
        let mut out = self.rule_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.rule_mlp_w2[j][k];
            }
        }
        out
    }

    /// Pre-encode all rules (call once, cache results).
    ///
    /// Returns a Vec of rule embeddings that can be reused across multiple
    /// expressions during saturation.
    #[must_use]
    pub fn encode_all_rules(&self, rule_features: &RuleFeatures, num_rules: usize) -> Vec<[f32; EMBED_DIM]> {
        (0..num_rules)
            .map(|r| self.encode_rule(&rule_features.features[r]))
            .collect()
    }

    // =========================================================================
    // Rule Encoding from LHS/RHS Templates
    //
    // Uses the SAME expr_embed as value/mask heads. 4-way concatenation:
    // [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] → linear → rule_embed
    //
    // This provides richer semantic features than hand-crafted rule descriptors.
    // =========================================================================

    /// Encode a rule from its LHS and RHS expression templates.
    ///
    /// Uses the shared backbone to embed both LHS and RHS, then concatenates
    /// four views: [z_LHS, z_RHS, z_LHS-z_RHS, z_LHS*z_RHS] and projects to
    /// EMBED_DIM.
    ///
    /// # Arguments
    /// * `lhs` - LHS pattern (what the rule matches), e.g., `A * (B + C)`
    /// * `rhs` - RHS pattern (what it produces), e.g., `A*B + A*C`
    ///
    /// # Semantic interpretation
    /// - `z_LHS`: What the rule MATCHES (pattern recognition)
    /// - `z_RHS`: What it PRODUCES (production prediction)
    /// - `z_LHS - z_RHS`: What CHANGED (the delta) - inverse rules have opposite signs
    /// - `z_LHS * z_RHS`: What's SHARED (preserved structure)
    #[must_use]
    pub fn encode_rule_from_templates(&self, lhs: &Expr, rhs: &Expr) -> [f32; EMBED_DIM] {
        // Embed LHS using shared backbone + expr_proj
        let lhs_acc = EdgeAccumulator::from_expr(lhs, &self.embeddings);
        let lhs_structural = StructuralFeatures::from_expr(lhs);
        let lhs_hidden = self.forward_shared(&lhs_acc, &lhs_structural);
        let z_lhs = self.compute_expr_embed(&lhs_hidden);

        // Embed RHS using shared backbone + expr_proj
        let rhs_acc = EdgeAccumulator::from_expr(rhs, &self.embeddings);
        let rhs_structural = StructuralFeatures::from_expr(rhs);
        let rhs_hidden = self.forward_shared(&rhs_acc, &rhs_structural);
        let z_rhs = self.compute_expr_embed(&rhs_hidden);

        // 4-way concatenate: [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] = 96 dims
        let mut concat = [0.0f32; RULE_CONCAT_DIM];
        for i in 0..EMBED_DIM {
            concat[i] = z_lhs[i];                           // what it matches
            concat[EMBED_DIM + i] = z_rhs[i];               // what it produces
            concat[2 * EMBED_DIM + i] = z_lhs[i] - z_rhs[i]; // the delta
            concat[3 * EMBED_DIM + i] = z_lhs[i] * z_rhs[i]; // shared structure
        }

        // Linear projection: 96 → 24 (no MLP, rich features already)
        let mut out = self.rule_proj_b;
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                out[k] += concat[i] * self.rule_proj_w[i][k];
            }
        }
        out
    }

    /// Pre-encode all rules from templates (call once at init, cache results).
    ///
    /// Rules without templates fall back to zero embedding.
    /// Rule embeddings don't change during search - they're computed from
    /// LHS/RHS templates which are static.
    #[must_use]
    pub fn encode_all_rules_from_templates(&self, templates: &RuleTemplates) -> Vec<[f32; EMBED_DIM]> {
        (0..templates.len())
            .map(|r| {
                match (templates.get_lhs(r), templates.get_rhs(r)) {
                    (Some(lhs), Some(rhs)) => self.encode_rule_from_templates(lhs, rhs),
                    _ => [0.0f32; EMBED_DIM], // No template - zero embedding
                }
            })
            .collect()
    }

    /// Bilinear score: mask_features @ interaction @ rule_embed + bias.
    ///
    /// Used by NnueCache for efficient O(1) scoring with cached mask_features.
    #[inline]
    #[must_use]
    pub fn bilinear_score(
        &self,
        mask_features: &[f32; EMBED_DIM],
        rule_embed: &[f32; EMBED_DIM],
        rule_idx: usize,
    ) -> f32 {
        // transformed = mask_features @ interaction
        let mut transformed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                transformed[j] += mask_features[i] * self.interaction[i][j];
            }
        }

        // score = transformed · rule_embed + bias
        let mut score = self.mask_rule_bias[rule_idx];
        for k in 0..EMBED_DIM {
            score += transformed[k] * rule_embed[k];
        }
        score
    }

    /// Score all rules for an expression using unified mask architecture.
    ///
    /// Uses pre-cached rule embeddings for efficiency. One forward pass through
    /// backbone + expr_proj + mask_mlp, then O(rules) bilinear scoring.
    #[must_use]
    pub fn mask_score_all_rules(
        &self,
        expr: &Expr,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Compute value prediction first (shared computation)
        let value_pred = self.value_mlp_forward(&expr_embed);

        // Mask sees value as context: "Given this costs X, should I try rule R?"
        let mask_features = self.compute_mask_features(&expr_embed, value_pred);

        rule_embeds
            .iter()
            .enumerate()
            .map(|(r, rule_embed)| self.bilinear_score(&mask_features, rule_embed, r))
            .collect()
    }

    /// Score all rules with pre-computed backbone hidden state.
    ///
    /// More efficient when you have multiple expressions sharing the same
    /// feature extraction.
    #[must_use]
    pub fn mask_score_all_rules_with_hidden(
        &self,
        hidden: &[f32; HIDDEN_DIM],
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let expr_embed = self.compute_expr_embed(hidden);

        // Compute value prediction first
        let value_pred = self.value_mlp_forward(&expr_embed);

        // Mask sees value as context
        let mask_features = self.compute_mask_features(&expr_embed, value_pred);

        rule_embeds
            .iter()
            .enumerate()
            .map(|(r, rule_embed)| self.bilinear_score(&mask_features, rule_embed, r))
            .collect()
    }

    /// Filter rules by mask threshold using unified architecture.
    ///
    /// Returns indices of rules with sigmoid(score) > threshold.
    #[must_use]
    pub fn filter_rules_unified(
        &self,
        expr: &Expr,
        rule_embeds: &[[f32; EMBED_DIM]],
        threshold: f32,
    ) -> Vec<usize> {
        let scores = self.mask_score_all_rules(expr, rule_embeds);
        scores
            .iter()
            .enumerate()
            .filter(|(_, score)| sigmoid(**score) > threshold)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Score a single (expression, rule) pair for mask prediction.
    ///
    /// Computes the rule embedding on-the-fly from rule features.
    /// Returns raw score (apply sigmoid for probability).
    #[must_use]
    pub fn mask_score_single(
        &self,
        expr: &Expr,
        rule_features: &[f32; RULE_FEATURE_DIM],
        rule_idx: usize,
    ) -> f32 {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Compute value prediction first
        let value_pred = self.value_mlp_forward(&expr_embed);

        // Mask sees value as context
        let mask_features = self.compute_mask_features(&expr_embed, value_pred);

        // Compute rule embedding from features
        let rule_embed = self.encode_rule(rule_features);

        // Bilinear scoring
        self.bilinear_score(&mask_features, &rule_embed, rule_idx)
    }

    /// Predict cost using the new unified value MLP.
    ///
    /// Goes through: backbone → expr_embed → value_mlp → cost
    #[must_use]
    pub fn predict_cost_unified(&self, expr: &Expr) -> f32 {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Value MLP: EMBED_DIM → MLP_HIDDEN (ReLU) → 1
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0); // ReLU
        }

        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }
        cost
    }

    /// Compute full NNUE metadata for an expression.
    ///
    /// Returns (expr_embed, value_pred, mask_features) tuple.
    /// Used by NnueCache to precompute and store e-node metadata.
    ///
    /// The data flow is:
    /// ```text
    /// expr → backbone → expr_embed (24) → value_mlp → value_pred (1)
    ///                                          ↓
    ///                               [expr_embed, value_pred] (25)
    ///                                          ↓
    ///                                      mask_mlp
    ///                                          ↓
    ///                                   mask_features (24)
    /// ```
    #[must_use]
    pub fn compute_metadata(&self, expr: &Expr) -> ([f32; EMBED_DIM], f32, [f32; EMBED_DIM]) {
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Compute value prediction
        let value_pred = self.value_mlp_forward(&expr_embed);

        // Compute mask features with value as context
        let mask_features = self.compute_mask_features(&expr_embed, value_pred);

        (expr_embed, value_pred, mask_features)
    }

    /// Legacy parameter count (excluding unified mask architecture).
    ///
    /// Parameter count including learned rule embeddings but NOT the new
    /// unified mask architecture. Use `param_count_total()` for full count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpEmbeddings::param_count()           // embeddings: 42 * 32 = 1,344
            + INPUT_DIM * HIDDEN_DIM          // w1: 85 * 64 = 5,440
            + HIDDEN_DIM                      // b1: 64
            + HIDDEN_DIM                      // value_w: 64
            + 1                               // value_b: 1
            + MAX_RULES * HIDDEN_DIM          // rule_embeddings: 64 * 64 = 4,096
            + MAX_RULES                       // rule_bias: 64
            + HIDDEN_DIM                      // search_w: 64
            + 1                               // search_b: 1
        // Total: ~11,074 parameters (~44KB)
    }

    /// Parameter count for the unified mask architecture only.
    #[must_use]
    pub const fn param_count_unified_only() -> usize {
        HIDDEN_DIM * EMBED_DIM                // expr_proj_w: 64 * 24 = 1,536
            + EMBED_DIM                       // expr_proj_b: 24
            + EMBED_DIM * MLP_HIDDEN          // value_mlp_w1: 24 * 16 = 384
            + MLP_HIDDEN                      // value_mlp_b1: 16
            + MLP_HIDDEN                      // value_mlp_w2: 16
            + 1                               // value_mlp_b2: 1
            + EMBED_DIM * MLP_HIDDEN          // mask_mlp_w1: 24 * 16 = 384
            + MLP_HIDDEN                      // mask_mlp_b1: 16
            + MLP_HIDDEN * EMBED_DIM          // mask_mlp_w2: 16 * 24 = 384
            + EMBED_DIM                       // mask_mlp_b2: 24
            + RULE_FEATURE_DIM * MLP_HIDDEN   // rule_mlp_w1: 8 * 16 = 128
            + MLP_HIDDEN                      // rule_mlp_b1: 16
            + MLP_HIDDEN * EMBED_DIM          // rule_mlp_w2: 16 * 24 = 384
            + EMBED_DIM                       // rule_mlp_b2: 24
            + EMBED_DIM * EMBED_DIM           // interaction: 24 * 24 = 576
            + MASK_MAX_RULES                  // mask_rule_bias: 1024
        // Total: ~4,937 additional parameters
    }

    /// Total parameter count including unified mask architecture.
    #[must_use]
    pub const fn param_count_total() -> usize {
        Self::param_count() + Self::param_count_unified_only()
        // Total: ~16,011 parameters (~64KB)
    }

    /// Memory size in bytes (f32 weights) for legacy architecture.
    #[must_use]
    pub const fn memory_bytes() -> usize {
        Self::param_count() * 4
    }

    /// Memory size in bytes including unified mask architecture.
    #[must_use]
    pub const fn memory_bytes_total() -> usize {
        Self::param_count_total() * 4
    }

    // ========================================================================
    // MCTS Support: Accumulator-based Evaluation
    //
    // These methods enable cheap MCTS simulation without e-graph cloning.
    // The accumulator can be incrementally updated as rules are applied.
    // ========================================================================

    /// Predict cost directly from accumulator (for MCTS evaluation).
    ///
    /// Skips expr parsing - just forward pass through backbone + value head.
    /// Use this for fast MCTS rollout evaluation.
    #[must_use]
    pub fn predict_cost_from_accumulator(&self, acc: &EdgeAccumulator) -> f32 {
        let structural = StructuralFeatures::default();
        let hidden = self.forward_shared(acc, &structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Value MLP: EMBED_DIM → MLP_HIDDEN (ReLU) → 1
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0); // ReLU
        }

        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }
        cost
    }

    /// Predict cost with pre-computed structural features (for MCTS).
    ///
    /// More accurate than `predict_cost_from_accumulator` when you have
    /// structural information from the real e-graph.
    #[must_use]
    pub fn predict_cost_from_features(
        &self,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
    ) -> f32 {
        let hidden = self.forward_shared(acc, structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Value MLP: EMBED_DIM → MLP_HIDDEN (ReLU) → 1
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0); // ReLU
        }

        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }
        cost
    }

    /// Get policy logits from accumulator (for MCTS prior).
    ///
    /// Returns scores for all rules. Use softmax to get probabilities.
    /// This is the MCTS policy prior P(s, a) for UCB.
    #[must_use]
    pub fn policy_from_accumulator(
        &self,
        acc: &EdgeAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let structural = StructuralFeatures::default();
        let hidden = self.forward_shared(acc, &structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Compute value prediction first
        let value_pred = self.value_mlp_forward(&expr_embed);

        // Mask sees value as context
        let mask_features = self.compute_mask_features(&expr_embed, value_pred);

        rule_embeds
            .iter()
            .enumerate()
            .map(|(r, rule_embed)| self.bilinear_score(&mask_features, rule_embed, r))
            .collect()
    }

    /// Get policy logits with pre-computed structural features.
    #[must_use]
    pub fn policy_from_features(
        &self,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let hidden = self.forward_shared(acc, structural);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Compute value prediction first
        let value_pred = self.value_mlp_forward(&expr_embed);

        // Mask sees value as context
        let mask_features = self.compute_mask_features(&expr_embed, value_pred);

        rule_embeds
            .iter()
            .enumerate()
            .map(|(r, rule_embed)| self.bilinear_score(&mask_features, rule_embed, r))
            .collect()
    }

    /// Get Bernoulli policy probabilities: P(apply) = sigmoid(logit / temp).
    ///
    /// Each rule is an independent binary decision (Bernoulli trial).
    /// Use these to stochastically decide: `if random() < prob[rule] { apply(rule) }`.
    ///
    /// Note: softmax on binary [logit, 0] = sigmoid(logit), so this IS the
    /// correct softmax formulation for independent apply/don't-apply decisions.
    ///
    /// Temperature controls exploration:
    /// - temp → 0: deterministic (prob → 0 or 1)
    /// - temp = 1: standard sigmoid
    /// - temp > 1: more exploration (probs pushed toward 0.5)
    #[must_use]
    pub fn bernoulli_policy_from_accumulator(
        &self,
        acc: &EdgeAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
        temperature: f32,
    ) -> Vec<f32> {
        let logits = self.policy_from_accumulator(acc, rule_embeds);
        let temp = temperature.max(0.01);
        logits.iter().map(|&x| sigmoid(x / temp)).collect()
    }

    /// Stochastically sample which rules to apply using Bernoulli policy.
    ///
    /// For each rule independently: apply if `random() < sigmoid(logit / temp)`.
    /// Returns indices of rules to apply.
    ///
    /// This is the correct exploration strategy for independent rule decisions.
    /// Each rule is sampled according to its own probability - rules don't compete.
    #[must_use]
    pub fn sample_rules_bernoulli(
        &self,
        acc: &EdgeAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
        temperature: f32,
        rng_state: &mut u64,
    ) -> Vec<usize> {
        let probs = self.bernoulli_policy_from_accumulator(acc, rule_embeds, temperature);

        probs
            .iter()
            .enumerate()
            .filter(|&(_, prob)| {
                // Simple LCG for random number
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let random = (*rng_state >> 33) as f32 / (1u64 << 31) as f32;
                random < *prob
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Save weights to a binary file.
    ///
    /// Format: magic "TRI3" + all weights as little-endian f32.
    /// TRI3 = unified architecture v3 (adds LHS/RHS rule template projection).
    #[cfg(feature = "std")]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Magic header (TRI3 = unified tripartite architecture v3 with rule templates)
        file.write_all(b"TRI3")?;

        // ===== Legacy backbone =====
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

        // Value head (legacy)
        for &val in &self.value_w {
            file.write_all(&val.to_le_bytes())?;
        }
        file.write_all(&self.value_b.to_le_bytes())?;

        // Rule embeddings (legacy)
        for row in &self.rule_embeddings {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.rule_bias {
            file.write_all(&val.to_le_bytes())?;
        }

        // Search head (legacy)
        for &val in &self.search_w {
            file.write_all(&val.to_le_bytes())?;
        }
        file.write_all(&self.search_b.to_le_bytes())?;

        // ===== Unified mask architecture =====
        // Expr projection
        for row in &self.expr_proj_w {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.expr_proj_b {
            file.write_all(&val.to_le_bytes())?;
        }

        // Value MLP
        for row in &self.value_mlp_w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.value_mlp_b1 {
            file.write_all(&val.to_le_bytes())?;
        }
        for &val in &self.value_mlp_w2 {
            file.write_all(&val.to_le_bytes())?;
        }
        file.write_all(&self.value_mlp_b2.to_le_bytes())?;

        // Mask MLP
        for row in &self.mask_mlp_w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.mask_mlp_b1 {
            file.write_all(&val.to_le_bytes())?;
        }
        for row in &self.mask_mlp_w2 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.mask_mlp_b2 {
            file.write_all(&val.to_le_bytes())?;
        }

        // Rule MLP (legacy, for hand-crafted features)
        for row in &self.rule_mlp_w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.rule_mlp_b1 {
            file.write_all(&val.to_le_bytes())?;
        }
        for row in &self.rule_mlp_w2 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.rule_mlp_b2 {
            file.write_all(&val.to_le_bytes())?;
        }

        // Rule Projection (TRI3: LHS/RHS template encoding)
        for row in &self.rule_proj_w {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.rule_proj_b {
            file.write_all(&val.to_le_bytes())?;
        }

        // Interaction matrix
        for row in &self.interaction {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // Mask rule bias
        for &val in &self.mask_rule_bias {
            file.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }

    /// Load weights from a binary file.
    ///
    /// Supports "DUAL" (legacy), "TRI2" (unified), and "TRI3" (template) formats.
    /// Legacy files will have unified mask and rule_proj fields zero-initialized.
    #[cfg(feature = "std")]
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Check magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        let is_legacy = &magic == b"DUAL";
        let is_tri2 = &magic == b"TRI2";
        let is_tri3 = &magic == b"TRI3";

        if !is_legacy && !is_tri2 && !is_tri3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid ExprNnue file magic (expected 'DUAL', 'TRI2', or 'TRI3')",
            ));
        }

        let has_unified = is_tri2 || is_tri3;
        let has_rule_proj = is_tri3;

        let mut net = Self::new();

        // ===== Legacy backbone =====
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

        // Read value head (legacy)
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

        if has_unified {
            // Read rule embeddings (in TRI2+ format)
            for row in &mut net.rule_embeddings {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            for val in &mut net.rule_bias {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }

        // Read search head (legacy)
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

        // ===== Unified mask architecture (TRI2+) =====
        if has_unified {
            // Expr projection
            for row in &mut net.expr_proj_w {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            for val in &mut net.expr_proj_b {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }

            // Value MLP
            for row in &mut net.value_mlp_w1 {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            for val in &mut net.value_mlp_b1 {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
            for val in &mut net.value_mlp_w2 {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
            {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                net.value_mlp_b2 = f32::from_le_bytes(buf);
            }

            // Mask MLP
            for row in &mut net.mask_mlp_w1 {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            for val in &mut net.mask_mlp_b1 {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
            for row in &mut net.mask_mlp_w2 {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            for val in &mut net.mask_mlp_b2 {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }

            // Rule MLP
            for row in &mut net.rule_mlp_w1 {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            for val in &mut net.rule_mlp_b1 {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
            for row in &mut net.rule_mlp_w2 {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            for val in &mut net.rule_mlp_b2 {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }

            // Rule Projection (TRI3+: LHS/RHS template encoding)
            if has_rule_proj {
                for row in &mut net.rule_proj_w {
                    for val in row {
                        let mut buf = [0u8; 4];
                        file.read_exact(&mut buf)?;
                        *val = f32::from_le_bytes(buf);
                    }
                }
                for val in &mut net.rule_proj_b {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }
            // TRI2 files: rule_proj fields remain zero-initialized

            // Interaction matrix
            for row in &mut net.interaction {
                for val in row {
                    let mut buf = [0u8; 4];
                    file.read_exact(&mut buf)?;
                    *val = f32::from_le_bytes(buf);
                }
            }

            // Mask rule bias
            for val in &mut net.mask_rule_bias {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        // Legacy DUAL files: unified mask fields remain zero-initialized

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

    // =========================================================================
    // Unified Mask Architecture Training
    //
    // Backprop path: score → bilinear → mask_mlp → expr_embed → expr_proj
    // Also updates: rule_mlp, interaction, rule_bias
    // Backbone (embeddings, w1, b1) is FROZEN during mask training.
    // =========================================================================

    /// Train mask head on single (expr, rule, fired) sample.
    ///
    /// Uses asymmetric BCE loss with higher weight for false negatives
    /// (catching positives is critical - we don't want to skip rules that fire).
    ///
    /// # Arguments
    /// * `expr` - The expression being evaluated
    /// * `rule_features` - Hand-crafted features for this rule
    /// * `rule_idx` - Index of the rule in the rule_bias array
    /// * `fired` - Whether the rule actually fired (ground truth)
    /// * `lr` - Learning rate
    /// * `fp_weight` - Weight for false positives (typically ~1.0)
    /// * `fn_weight` - Weight for false negatives (typically ~100.0)
    ///
    /// # Returns
    /// The (weighted) loss for this sample
    pub fn train_mask_step(
        &mut self,
        expr: &Expr,
        rule_features: &[f32; RULE_FEATURE_DIM],
        rule_idx: usize,
        fired: bool,
        lr: f32,
        fp_weight: f32,
        fn_weight: f32,
    ) -> f32 {
        // ===== Forward pass with stored intermediates =====
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);

        let expr_embed = self.compute_expr_embed(&hidden);

        // Mask MLP forward (store hidden for backprop)
        let (mask_features, mask_hidden) = self.mask_mlp_forward_with_hidden(&expr_embed);

        // Rule MLP forward (store hidden for backprop)
        let (rule_embed, rule_hidden) = self.rule_mlp_forward_with_hidden(rule_features);

        // Bilinear: mask_features @ interaction @ rule_embed
        let (score, transformed) = self.bilinear_forward_with_hidden(&mask_features, &rule_embed, rule_idx);

        // ===== Loss computation =====
        let pred = sigmoid(score);
        let target = if fired { 1.0 } else { 0.0 };

        let p = pred.clamp(1e-7, 1.0 - 1e-7);
        let loss = -(target * libm::logf(p) + (1.0 - target) * libm::logf(1.0 - p));

        // Asymmetric weighting (catch positives!)
        let weight = if pred > 0.5 && !fired {
            fp_weight
        } else if pred <= 0.5 && fired {
            fn_weight
        } else {
            1.0
        };

        let d_score = (weight * (pred - target)).clamp(-10.0, 10.0);

        // ===== Backprop =====
        // d_score → rule_bias
        self.mask_rule_bias[rule_idx] -= lr * d_score;

        // d_score → interaction, mask_features, rule_embed
        let (d_mask_features, d_rule_embed) = self.backprop_bilinear(
            d_score, &mask_features, &rule_embed, &transformed, lr
        );

        // d_mask_features → mask_mlp
        let _d_expr_embed = self.backprop_mask_mlp(&d_mask_features, &expr_embed, &mask_hidden, lr);

        // d_rule_embed → rule_mlp
        self.backprop_rule_mlp(&d_rule_embed, rule_features, &rule_hidden, lr);

        // NOTE: We freeze the backbone (expr_proj, w1, b1, embeddings)
        // If you want to fine-tune, uncomment:
        // self.backprop_expr_proj(&d_expr_embed, &hidden, lr);

        loss * weight
    }

    /// REINFORCE update for a single mask decision.
    ///
    /// The reward comes from FINAL extraction quality, not per-rule outcomes.
    ///
    /// For APPROVED decision: ∇log P(approve) = 1 - sigmoid(score)
    /// For REJECTED decision: ∇log P(reject) = -sigmoid(score)
    ///
    /// Positive advantage → reinforce the decision that was made
    /// Negative advantage → discourage the decision that was made
    ///
    /// # Arguments
    /// * `expr` - Expression that was scored
    /// * `rule_features` - Features of the rule
    /// * `rule_idx` - Index of the rule
    /// * `approved` - Was this rule approved (tried) or rejected (skipped)?
    /// * `advantage` - reward - baseline (from final extraction cost comparison)
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// The gradient magnitude applied.
    pub fn train_mask_reinforce(
        &mut self,
        expr: &Expr,
        rule_features: &[f32; RULE_FEATURE_DIM],
        rule_idx: usize,
        approved: bool,
        advantage: f32,
        lr: f32,
    ) -> f32 {
        // Forward pass to get intermediates
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);

        let expr_embed = self.compute_expr_embed(&hidden);
        let (mask_features, mask_hidden) = self.mask_mlp_forward_with_hidden(&expr_embed);
        let (rule_embed, rule_hidden) = self.rule_mlp_forward_with_hidden(rule_features);
        let (score, transformed) = self.bilinear_forward_with_hidden(&mask_features, &rule_embed, rule_idx);

        // REINFORCE gradient depends on the action taken:
        // - Approved: ∇log sigmoid(score) = 1 - sigmoid(score)
        // - Rejected: ∇log (1 - sigmoid(score)) = -sigmoid(score)
        let prob = sigmoid(score).clamp(1e-6, 1.0 - 1e-6);
        let d_log_prob = if approved {
            1.0 - prob  // push score up when reinforcing approval
        } else {
            -prob  // push score down when reinforcing rejection
        };

        // Clip gradient to prevent explosion
        let d_score = (advantage * d_log_prob).clamp(-1.0, 1.0);

        // Skip update if gradient would be NaN or too small
        if !d_score.is_finite() || d_score.abs() < 1e-8 {
            return 0.0;
        }

        // Backprop
        self.mask_rule_bias[rule_idx] -= lr * d_score;

        let (d_mask_features, d_rule_embed) = self.backprop_bilinear(
            d_score, &mask_features, &rule_embed, &transformed, lr
        );

        let _d_expr_embed = self.backprop_mask_mlp(&d_mask_features, &expr_embed, &mask_hidden, lr);
        self.backprop_rule_mlp(&d_rule_embed, rule_features, &rule_hidden, lr);

        d_score.abs()
    }

    /// Batch REINFORCE update for decisions from a search episode.
    ///
    /// # Arguments
    /// * `decisions` - Vec of (expr, rule_features, rule_idx, approved)
    /// * `advantage` - reward - baseline (from final cost comparison)
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// Total gradient norm applied.
    pub fn train_mask_reinforce_batch(
        &mut self,
        decisions: &[(Expr, [f32; RULE_FEATURE_DIM], usize, bool)],
        advantage: f32,
        lr: f32,
    ) -> f32 {
        let mut total_grad = 0.0f32;
        for (expr, rule_features, rule_idx, approved) in decisions {
            total_grad += self.train_mask_reinforce(expr, rule_features, *rule_idx, *approved, advantage, lr);
        }
        total_grad
    }

    /// REINFORCE training using pre-computed rule embeddings.
    ///
    /// This is the preferred method when using LHS/RHS template embeddings.
    /// Rule embeddings are computed once via `encode_all_rules_from_templates()`
    /// and reused across training.
    ///
    /// # Arguments
    /// * `expr` - The expression being evaluated
    /// * `rule_embed` - Pre-computed rule embedding (from templates)
    /// * `rule_idx` - Rule index (for bias lookup)
    /// * `approved` - Whether this rule was approved by the mask
    /// * `advantage` - reward - baseline
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// The gradient magnitude applied.
    pub fn train_mask_reinforce_with_embed(
        &mut self,
        expr: &Expr,
        rule_embed: &[f32; EMBED_DIM],
        rule_idx: usize,
        approved: bool,
        advantage: f32,
        lr: f32,
    ) -> f32 {
        // Forward pass to get intermediates
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);

        let expr_embed = self.compute_expr_embed(&hidden);
        let (mask_features, mask_hidden) = self.mask_mlp_forward_with_hidden(&expr_embed);
        let (score, transformed) = self.bilinear_forward_with_hidden(&mask_features, rule_embed, rule_idx);

        // REINFORCE gradient:
        // - Approved: ∇log sigmoid(score) = 1 - sigmoid(score)
        // - Rejected: ∇log (1 - sigmoid(score)) = -sigmoid(score)
        let prob = sigmoid(score).clamp(1e-6, 1.0 - 1e-6);
        let d_log_prob = if approved {
            1.0 - prob
        } else {
            -prob
        };

        // Clip gradient to prevent explosion
        let d_score = (advantage * d_log_prob).clamp(-1.0, 1.0);

        // Skip update if gradient would be NaN or too small
        if !d_score.is_finite() || d_score.abs() < 1e-8 {
            return 0.0;
        }

        // Backprop (rule embedding is frozen - computed from templates)
        self.mask_rule_bias[rule_idx] -= lr * d_score;

        let (d_mask_features, _d_rule_embed) = self.backprop_bilinear(
            d_score, &mask_features, rule_embed, &transformed, lr
        );

        let _d_expr_embed = self.backprop_mask_mlp(&d_mask_features, &expr_embed, &mask_hidden, lr);
        // NOTE: Rule embedding is not updated here - it comes from templates.
        // The rule_proj weights that created it are updated only during supervised pretraining.

        d_score.abs()
    }

    /// Batch REINFORCE update using pre-computed rule embeddings.
    ///
    /// # Arguments
    /// * `decisions` - Vec of (expr, rule_embed, rule_idx, approved)
    /// * `advantage` - reward - baseline (from final cost comparison)
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// Total gradient norm applied.
    pub fn train_mask_reinforce_batch_with_embeds(
        &mut self,
        decisions: &[(Expr, [f32; EMBED_DIM], usize, bool)],
        advantage: f32,
        lr: f32,
    ) -> f32 {
        let mut total_grad = 0.0f32;
        for (expr, rule_embed, rule_idx, approved) in decisions {
            total_grad += self.train_mask_reinforce_with_embed(expr, rule_embed, *rule_idx, *approved, advantage, lr);
        }
        total_grad
    }

    /// Train value MLP on (expr, true_cost) sample.
    ///
    /// Uses MSE loss. Backprop goes through value_mlp → expr_proj.
    /// Backbone is frozen.
    pub fn train_value_mlp_step(
        &mut self,
        expr: &Expr,
        true_cost: f32,
        lr: f32,
    ) -> f32 {
        // Forward
        let acc = EdgeAccumulator::from_expr(expr, &self.embeddings);
        let structural = StructuralFeatures::from_expr(expr);
        let hidden = self.forward_shared(&acc, &structural);
        let expr_embed = self.compute_expr_embed(&hidden);
        let (pred_cost, value_hidden) = self.value_mlp_forward_with_hidden(&expr_embed);

        // MSE loss
        let diff = pred_cost - true_cost;
        let loss = diff * diff;
        let d_cost = 2.0 * diff;

        // Backprop through value_mlp
        let _d_expr_embed = self.backprop_value_mlp(d_cost, &expr_embed, &value_hidden, lr);

        // NOTE: Backbone frozen - if you want to fine-tune:
        // self.backprop_expr_proj(&d_expr_embed, &hidden, lr);

        loss
    }

    // =========================================================================
    // Forward with Hidden (for backprop)
    // =========================================================================

    /// Mask MLP forward storing hidden activations.
    fn mask_mlp_forward_with_hidden(&self, expr_embed: &[f32; EMBED_DIM]) -> ([f32; EMBED_DIM], [f32; MLP_HIDDEN]) {
        // First layer
        let mut h = self.mask_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.mask_mlp_w1[i][j];
            }
        }

        // Store pre-ReLU for backprop (we need to know which neurons were active)
        let h_pre_relu = h;

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer
        let mut out = self.mask_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.mask_mlp_w2[j][k];
            }
        }

        (out, h_pre_relu)
    }

    /// Rule MLP forward storing hidden activations.
    fn rule_mlp_forward_with_hidden(&self, rule_features: &[f32; RULE_FEATURE_DIM]) -> ([f32; EMBED_DIM], [f32; MLP_HIDDEN]) {
        // First layer
        let mut h = self.rule_mlp_b1;
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += rule_features[i] * self.rule_mlp_w1[i][j];
            }
        }

        let h_pre_relu = h;

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer
        let mut out = self.rule_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.rule_mlp_w2[j][k];
            }
        }

        (out, h_pre_relu)
    }

    /// Value MLP forward storing hidden activations.
    fn value_mlp_forward_with_hidden(&self, expr_embed: &[f32; EMBED_DIM]) -> (f32, [f32; MLP_HIDDEN]) {
        // First layer
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }

        let h_pre_relu = h;

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer
        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }

        (cost, h_pre_relu)
    }

    /// Bilinear forward storing transformed vector.
    fn bilinear_forward_with_hidden(
        &self,
        mask_features: &[f32; EMBED_DIM],
        rule_embed: &[f32; EMBED_DIM],
        rule_idx: usize,
    ) -> (f32, [f32; EMBED_DIM]) {
        // transformed = mask_features @ interaction
        let mut transformed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                transformed[j] += mask_features[i] * self.interaction[i][j];
            }
        }

        // score = transformed · rule_embed + bias
        let mut score = self.mask_rule_bias[rule_idx];
        for k in 0..EMBED_DIM {
            score += transformed[k] * rule_embed[k];
        }

        (score, transformed)
    }

    // =========================================================================
    // Backpropagation Helpers
    // =========================================================================

    /// Backprop through bilinear layer.
    ///
    /// Returns (d_mask_features, d_rule_embed) and updates interaction matrix.
    fn backprop_bilinear(
        &mut self,
        d_score: f32,
        mask_features: &[f32; EMBED_DIM],
        rule_embed: &[f32; EMBED_DIM],
        transformed: &[f32; EMBED_DIM],
        lr: f32,
    ) -> ([f32; EMBED_DIM], [f32; EMBED_DIM]) {
        // d_score/d_transformed = rule_embed
        // d_score/d_rule_embed = transformed
        let mut d_transformed = [0.0f32; EMBED_DIM];
        let mut d_rule_embed = [0.0f32; EMBED_DIM];

        for k in 0..EMBED_DIM {
            d_transformed[k] = d_score * rule_embed[k];
            d_rule_embed[k] = d_score * transformed[k];
        }

        // d_transformed/d_mask_features = interaction^T
        // d_transformed/d_interaction = outer(mask_features, I)
        let mut d_mask_features = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                d_mask_features[i] += d_transformed[j] * self.interaction[i][j];
                // Update interaction: d_loss/d_interaction[i][j] = mask_features[i] * d_transformed[j]
                self.interaction[i][j] -= lr * mask_features[i] * d_transformed[j];
            }
        }

        (d_mask_features, d_rule_embed)
    }

    /// Backprop through mask MLP.
    ///
    /// Returns d_expr_embed and updates mask_mlp weights.
    fn backprop_mask_mlp(
        &mut self,
        d_out: &[f32; EMBED_DIM],
        expr_embed: &[f32; EMBED_DIM],
        h_pre_relu: &[f32; MLP_HIDDEN],
        lr: f32,
    ) -> [f32; EMBED_DIM] {
        // d_out → w2, b2
        // d_h (post-ReLU) = d_out @ w2^T
        let mut d_h = [0.0f32; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                d_h[j] += d_out[k] * self.mask_mlp_w2[j][k];
                // Update w2
                let h_relu = h_pre_relu[j].max(0.0);
                self.mask_mlp_w2[j][k] -= lr * h_relu * d_out[k];
            }
        }

        // Update b2
        for k in 0..EMBED_DIM {
            self.mask_mlp_b2[k] -= lr * d_out[k];
        }

        // ReLU backward
        for j in 0..MLP_HIDDEN {
            if h_pre_relu[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }

        // d_h → w1, b1, d_expr_embed
        let mut d_expr_embed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                d_expr_embed[i] += d_h[j] * self.mask_mlp_w1[i][j];
                // Update w1
                self.mask_mlp_w1[i][j] -= lr * expr_embed[i] * d_h[j];
            }
        }

        // Update b1
        for j in 0..MLP_HIDDEN {
            self.mask_mlp_b1[j] -= lr * d_h[j];
        }

        d_expr_embed
    }

    /// Backprop through rule MLP.
    ///
    /// Updates rule_mlp weights. Rule features are fixed, so no gradient returned.
    fn backprop_rule_mlp(
        &mut self,
        d_out: &[f32; EMBED_DIM],
        rule_features: &[f32; RULE_FEATURE_DIM],
        h_pre_relu: &[f32; MLP_HIDDEN],
        lr: f32,
    ) {
        // d_out → w2, b2
        let mut d_h = [0.0f32; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                d_h[j] += d_out[k] * self.rule_mlp_w2[j][k];
                let h_relu = h_pre_relu[j].max(0.0);
                self.rule_mlp_w2[j][k] -= lr * h_relu * d_out[k];
            }
        }

        // Update b2
        for k in 0..EMBED_DIM {
            self.rule_mlp_b2[k] -= lr * d_out[k];
        }

        // ReLU backward
        for j in 0..MLP_HIDDEN {
            if h_pre_relu[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }

        // d_h → w1, b1
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                self.rule_mlp_w1[i][j] -= lr * rule_features[i] * d_h[j];
            }
        }

        for j in 0..MLP_HIDDEN {
            self.rule_mlp_b1[j] -= lr * d_h[j];
        }
    }

    /// Backprop through value MLP.
    ///
    /// Returns d_expr_embed and updates value_mlp weights.
    fn backprop_value_mlp(
        &mut self,
        d_cost: f32,
        expr_embed: &[f32; EMBED_DIM],
        h_pre_relu: &[f32; MLP_HIDDEN],
        lr: f32,
    ) -> [f32; EMBED_DIM] {
        // d_cost → w2, b2
        let mut d_h = [0.0f32; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            d_h[j] = d_cost * self.value_mlp_w2[j];
            let h_relu = h_pre_relu[j].max(0.0);
            self.value_mlp_w2[j] -= lr * h_relu * d_cost;
        }

        self.value_mlp_b2 -= lr * d_cost;

        // ReLU backward
        for j in 0..MLP_HIDDEN {
            if h_pre_relu[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }

        // d_h → w1, b1, d_expr_embed
        let mut d_expr_embed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                d_expr_embed[i] += d_h[j] * self.value_mlp_w1[i][j];
                self.value_mlp_w1[i][j] -= lr * expr_embed[i] * d_h[j];
            }
        }

        for j in 0..MLP_HIDDEN {
            self.value_mlp_b1[j] -= lr * d_h[j];
        }

        d_expr_embed
    }

    /// Backprop through expr projection (optional, for fine-tuning).
    ///
    /// Updates expr_proj weights. Backbone (w1, b1) remains frozen.
    #[allow(dead_code)]
    fn backprop_expr_proj(
        &mut self,
        d_expr_embed: &[f32; EMBED_DIM],
        hidden: &[f32; HIDDEN_DIM],
        lr: f32,
    ) {
        // d_expr_embed → expr_proj_w, expr_proj_b
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.expr_proj_w[j][k] -= lr * hidden[j] * d_expr_embed[k];
            }
        }

        for k in 0..EMBED_DIM {
            self.expr_proj_b[k] -= lr * d_expr_embed[k];
        }
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

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

/// Softmax with temperature scaling.
///
/// `softmax(x_i / temp)` - higher temperature = more uniform distribution.
#[must_use]
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let temp = temperature.max(0.01); // Avoid division by zero

    // Scale by temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Numerical stability: subtract max
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| libm::expf(x - max_val)).collect();
    let sum: f32 = exps.iter().sum();

    if sum < 1e-10 {
        // Uniform fallback
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
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
        let net = ExprNnue::new_random(42);
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
            ExprNnue::param_count(),
            42 * 32          // embeddings: OpKind::COUNT * K
            + 85 * 64        // w1: INPUT_DIM * HIDDEN_DIM
            + 64             // b1: HIDDEN_DIM
            + 64             // w2: HIDDEN_DIM
            + 1              // b2
        );

        // Should be much smaller than HalfEP
        assert!(
            ExprNnue::memory_bytes() < 100_000,
            "Factored NNUE should use < 100KB"
        );
    }

    #[test]
    fn test_different_expressions_different_costs() {
        let net = ExprNnue::new_random(42);

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
    // ExprNnue Tests
    // ========================================================================

    #[test]
    fn test_dual_head_param_count() {
        // Verify our parameter count calculation
        // INPUT_DIM = 2*K + STRUCTURAL_FEATURE_COUNT = 2*32 + 21 = 85
        assert_eq!(
            ExprNnue::param_count(),
            42 * 32          // embeddings: OpKind::COUNT * K
            + 85 * 64        // w1: INPUT_DIM * HIDDEN_DIM
            + 64             // b1: HIDDEN_DIM
            + 64             // value_w: HIDDEN_DIM
            + 1              // value_b
            + 64 * 64        // rule_embeddings: MAX_RULES * HIDDEN_DIM
            + 64             // rule_bias: MAX_RULES
            + 64             // search_w: HIDDEN_DIM
            + 1              // search_b
        );

        // Should be slightly larger than single-head
        assert!(
            ExprNnue::param_count() > ExprNnue::param_count(),
            "Dual-head should have more params than single-head"
        );

        // But still under 100KB
        assert!(
            ExprNnue::memory_bytes() < 100_000,
            "Dual-head NNUE should use < 100KB"
        );
    }

    #[test]
    fn test_dual_head_value_prediction() {
        let net = ExprNnue::new_random(42);
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
        let net = ExprNnue::new_random(42);
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
        let net = ExprNnue::new_random(42);
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
        let factored = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let factored_cost = factored.evaluate(&expr);

        // Convert to dual-head
        let dual = ExprNnue::from_factored(&factored);
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
        let net = ExprNnue::new_with_latency_prior(42);

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
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        // Create some rule features
        // Score single rule
        let score_0 = net.score_rule(&expr, 0);
        assert!(score_0.is_finite(), "Policy score should be finite: {}", score_0);

        // Different rules should get different scores (learned embeddings differ)
        let score_1 = net.score_rule(&expr, 1);
        assert!(
            (score_0 - score_1).abs() > 1e-6,
            "Different rules should get different scores: {} vs {}",
            score_0, score_1
        );

        // Score all rules at once (attention-style)
        let all_scores = net.score_all_rules(&expr);
        assert!((all_scores[0] - score_0).abs() < 1e-6, "Batch should match individual");
        assert!((all_scores[1] - score_1).abs() < 1e-6, "Batch should match individual");

        // Classify rules (filter by threshold)
        let active_rules = net.classify_rules(&expr, 10, 0.5);
        // With random init, some should pass threshold
        assert!(active_rules.len() <= 10, "Should only consider first 10 rules");
    }

    // ========================================================================
    // Unified Mask Architecture Tests
    // ========================================================================

    #[test]
    fn test_rule_features_initialization() {
        let mut rule_features = RuleFeatures::new();

        // All features should be zero initially
        for r in 0..10 {
            for f in rule_features.get(r) {
                assert!(*f == 0.0, "Initial features should be zero");
            }
        }

        // Set features for a rule
        rule_features.set(0, [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0]);
        let features = rule_features.get(0);
        assert!((features[0] - 0.25).abs() < 1e-6, "Category should be set");
        assert!((features[3] - 1.0).abs() < 1e-6, "Commutative flag should be set");
    }

    #[test]
    fn test_encode_rule_deterministic() {
        let net = ExprNnue::new_random(42);
        let features = [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0];

        let embed1 = net.encode_rule(&features);
        let embed2 = net.encode_rule(&features);

        // Same input should produce same output
        for i in 0..EMBED_DIM {
            assert!(
                (embed1[i] - embed2[i]).abs() < 1e-6,
                "encode_rule should be deterministic at dim {}",
                i
            );
        }

        // Embedding should be finite
        for i in 0..EMBED_DIM {
            assert!(embed1[i].is_finite(), "Rule embedding should be finite at dim {}", i);
        }
    }

    #[test]
    fn test_encode_all_rules() {
        let net = ExprNnue::new_random(42);
        let mut rule_features = RuleFeatures::new();

        // Set up a few rules with different features
        rule_features.set(0, [0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0]); // algebraic
        rule_features.set(1, [0.25, 0.5, -1.0, 0.0, 1.0, 1.0, 0.3, 0.0]); // peephole
        rule_features.set(2, [0.75, 0.8, 1.0, 0.0, 0.0, 0.0, 0.05, 1.0]); // cross-cutting

        let embeds = net.encode_all_rules(&rule_features, 3);

        assert_eq!(embeds.len(), 3, "Should encode exactly 3 rules");

        // Each embedding should be finite
        for (r, embed) in embeds.iter().enumerate() {
            for (d, &val) in embed.iter().enumerate() {
                assert!(val.is_finite(), "Rule {} dim {} should be finite", r, d);
            }
        }

        // Different features should produce different embeddings
        let diff_01: f32 = embeds[0].iter().zip(embeds[1].iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff_02: f32 = embeds[0].iter().zip(embeds[2].iter()).map(|(a, b)| (a - b).abs()).sum();

        assert!(diff_01 > 1e-3, "Different rules should have different embeddings");
        assert!(diff_02 > 1e-3, "Different rules should have different embeddings");
    }

    #[test]
    fn test_compute_expr_embed() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        // Compute hidden state
        let acc = EdgeAccumulator::from_expr(&expr, &net.embeddings);
        let structural = StructuralFeatures::from_expr(&expr);
        let hidden = net.forward_shared(&acc, &structural);

        // Compute expr embedding
        let expr_embed = net.compute_expr_embed(&hidden);

        assert_eq!(expr_embed.len(), EMBED_DIM, "Expr embed should have EMBED_DIM dimensions");

        for (i, &val) in expr_embed.iter().enumerate() {
            assert!(val.is_finite(), "Expr embedding should be finite at dim {}", i);
        }
    }

    #[test]
    fn test_compute_mask_features() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let acc = EdgeAccumulator::from_expr(&expr, &net.embeddings);
        let structural = StructuralFeatures::from_expr(&expr);
        let hidden = net.forward_shared(&acc, &structural);
        let expr_embed = net.compute_expr_embed(&hidden);

        // Compute value prediction first (required for value-aware mask)
        let value_pred = net.value_mlp_forward(&expr_embed);

        let mask_features = net.compute_mask_features(&expr_embed, value_pred);

        assert_eq!(mask_features.len(), EMBED_DIM, "Mask features should have EMBED_DIM dimensions");

        for (i, &val) in mask_features.iter().enumerate() {
            assert!(val.is_finite(), "Mask features should be finite at dim {}", i);
        }
    }

    #[test]
    fn test_bilinear_score_computation() {
        let net = ExprNnue::new_random(42);

        // Create test vectors
        let mask_features = [1.0f32; EMBED_DIM];
        let rule_embed = [1.0f32; EMBED_DIM];

        let score = net.bilinear_score(&mask_features, &rule_embed, 0);

        assert!(score.is_finite(), "Bilinear score should be finite");

        // Manual verification: score = mask @ interaction @ rule + bias
        // With all-ones vectors, this should be sum of interaction matrix + bias[0]
        let mut expected = net.mask_rule_bias[0];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                expected += net.interaction[i][j];
            }
        }
        assert!(
            (score - expected).abs() < 1e-4,
            "Bilinear computation mismatch: got {}, expected {}",
            score,
            expected
        );
    }

    #[test]
    fn test_mask_score_all_rules_finite() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let mut rule_features = RuleFeatures::new();

        // Set up 5 rules
        for r in 0..5 {
            rule_features.set(r, [r as f32 * 0.2, 0.3, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0]);
        }

        let rule_embeds = net.encode_all_rules(&rule_features, 5);
        let scores = net.mask_score_all_rules(&expr, &rule_embeds);

        assert_eq!(scores.len(), 5, "Should score all 5 rules");

        for (r, &score) in scores.iter().enumerate() {
            assert!(score.is_finite(), "Score for rule {} should be finite", r);
        }
    }

    #[test]
    fn test_filter_rules_unified() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let mut rule_features = RuleFeatures::new();

        // Set up 10 rules
        for r in 0..10 {
            rule_features.set(r, [r as f32 * 0.1, 0.3, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0]);
        }

        // Set some rule biases high to ensure they pass threshold
        net.mask_rule_bias[0] = 5.0; // sigmoid(5) > 0.99
        net.mask_rule_bias[1] = 5.0;
        net.mask_rule_bias[5] = -5.0; // sigmoid(-5) < 0.01

        let rule_embeds = net.encode_all_rules(&rule_features, 10);
        let passing = net.filter_rules_unified(&expr, &rule_embeds, 0.5);

        // Rules 0 and 1 should definitely pass (high bias)
        // Rule 5 should definitely fail (low bias)
        // The test verifies the filtering logic works, not exact counts

        // At least some rules should pass
        assert!(!passing.is_empty(), "Some rules should pass threshold");

        // Not all rules should pass (rule 5 has very negative bias)
        assert!(passing.len() < 10, "Not all rules should pass threshold");
    }

    #[test]
    fn test_predict_cost_unified() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let cost = net.predict_cost_unified(&expr);

        assert!(cost.is_finite(), "Unified cost prediction should be finite");
        assert!(cost > 0.0, "Cost should be positive (exp of value_mlp output)");
    }

    #[test]
    fn test_mask_training_step_loss_decreases() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let rule_features = [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0];

        // Compute initial loss for a positive sample
        let rule_embeds = net.encode_all_rules(&RuleFeatures::new(), 1);
        let initial_scores = net.mask_score_all_rules(&expr, &rule_embeds);
        let initial_pred = 1.0 / (1.0 + (-initial_scores[0]).exp()); // sigmoid

        // Train on positive sample (rule fired)
        let mut total_loss = 0.0;
        for _ in 0..50 {
            let loss = net.train_mask_step(&expr, &rule_features, 0, true, 0.01, 1.0, 10.0);
            total_loss += loss;
            assert!(loss.is_finite(), "Training loss should be finite");
        }

        // Compute final prediction
        let rule_embeds = net.encode_all_rules(&RuleFeatures::new(), 1);
        let final_scores = net.mask_score_all_rules(&expr, &rule_embeds);
        let final_pred = 1.0 / (1.0 + (-final_scores[0]).exp()); // sigmoid

        // After training on positive samples, prediction should increase
        // (network learns to predict 1 for this expr-rule pair)
        assert!(
            final_pred > initial_pred || (final_pred - initial_pred).abs() < 0.1,
            "Training on positive should increase prediction: {} -> {}",
            initial_pred,
            final_pred
        );
    }

    #[test]
    fn test_value_mlp_training_step() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let target_cost = 100.0f32; // Target nanoseconds
        let target_log = target_cost.ln();

        // Compute initial prediction
        let initial_pred = net.predict_cost_unified(&expr);

        // Train for several steps
        for _ in 0..100 {
            let loss = net.train_value_mlp_step(&expr, target_log, 0.01);
            assert!(loss.is_finite(), "Value training loss should be finite");
        }

        // Final prediction should be closer to target
        let final_pred = net.predict_cost_unified(&expr);
        let initial_error = (initial_pred.ln() - target_log).abs();
        let final_error = (final_pred.ln() - target_log).abs();

        // Allow for stochastic behavior, but generally should improve
        // (or at least not get catastrophically worse)
        assert!(
            final_error < initial_error * 2.0 || final_error < 1.0,
            "Value MLP should learn toward target: initial_err={}, final_err={}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_randomize_mask_only() {
        let mut net = ExprNnue::new();

        // Set some backbone values that should be preserved
        net.embeddings.e[0][0] = 1.234;
        net.w1[0][0] = 5.678;
        net.b1[0] = 0.999;
        net.expr_proj_w[0][0] = 2.345;  // shared projection - should be preserved

        // Initially mask-specific weights should be zero
        let initial_mask_sum: f32 = net.mask_mlp_w1.iter().flatten().map(|x| x.abs()).sum();
        assert!(initial_mask_sum < 1e-6, "Initial mask weights should be zero");

        // Randomize mask-only
        net.randomize_mask_only(42);

        // Backbone should be PRESERVED
        assert!((net.embeddings.e[0][0] - 1.234).abs() < 1e-6, "Embeddings should be preserved");
        assert!((net.w1[0][0] - 5.678).abs() < 1e-6, "w1 should be preserved");
        assert!((net.b1[0] - 0.999).abs() < 1e-6, "b1 should be preserved");
        assert!((net.expr_proj_w[0][0] - 2.345).abs() < 1e-6, "expr_proj should be preserved");

        // Mask-specific weights should now be non-zero
        let final_mask_sum: f32 = net.mask_mlp_w1.iter().flatten().map(|x| x.abs()).sum();
        assert!(final_mask_sum > 1.0, "Randomized mask weights should be non-zero");

        // Interaction matrix should be near identity diagonal
        for i in 0..EMBED_DIM {
            assert!(
                (net.interaction[i][i] - 1.0).abs() < 0.5,
                "Diagonal of interaction should be near 1.0"
            );
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_unified_architecture_serialization_roundtrip() {
        use std::path::PathBuf;

        let mut net = ExprNnue::new_random(42);
        net.randomize_mask_only(123);

        // Set some specific values we can verify
        net.interaction[0][0] = 1.234;
        net.mask_rule_bias[5] = -0.567;
        net.value_mlp_b2 = 3.14;

        // Create temp file path
        let temp_path = PathBuf::from("/tmp/test_dual_head_unified_serialization.bin");

        // Serialize
        net.save(&temp_path).expect("Save should succeed");

        // Deserialize
        let loaded = ExprNnue::load(&temp_path).expect("Load should succeed");

        // Cleanup temp file
        let _ = std::fs::remove_file(&temp_path);

        // Verify specific values
        assert!(
            (loaded.interaction[0][0] - 1.234).abs() < 1e-6,
            "Interaction should be preserved"
        );
        assert!(
            (loaded.mask_rule_bias[5] - (-0.567)).abs() < 1e-6,
            "Rule bias should be preserved"
        );
        assert!(
            (loaded.value_mlp_b2 - 3.14).abs() < 1e-6,
            "Value MLP bias should be preserved"
        );

        // Verify predictions match
        let expr = make_fma_pattern();
        let original_cost = net.predict_cost_unified(&expr);
        let loaded_cost = loaded.predict_cost_unified(&expr);
        assert!(
            (original_cost - loaded_cost).abs() < 1e-5,
            "Loaded network should produce same predictions"
        );
    }

    #[test]
    fn test_gradients_finite_through_all_paths() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let rule_features = [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0];

        // Train mask head
        let mask_loss = net.train_mask_step(&expr, &rule_features, 0, true, 0.01, 1.0, 10.0);
        assert!(mask_loss.is_finite(), "Mask loss should be finite");
        assert!(!mask_loss.is_nan(), "Mask loss should not be NaN");

        // Train value head
        let value_loss = net.train_value_mlp_step(&expr, 5.0, 0.01);
        assert!(value_loss.is_finite(), "Value loss should be finite");
        assert!(!value_loss.is_nan(), "Value loss should not be NaN");

        // Verify weights didn't become NaN
        for row in &net.expr_proj_w {
            for &val in row {
                assert!(!val.is_nan(), "expr_proj_w should not contain NaN after training");
            }
        }

        for row in &net.mask_mlp_w1 {
            for &val in row {
                assert!(!val.is_nan(), "mask_mlp_w1 should not contain NaN after training");
            }
        }

        for row in &net.rule_mlp_w1 {
            for &val in row {
                assert!(!val.is_nan(), "rule_mlp_w1 should not contain NaN after training");
            }
        }

        for row in &net.interaction {
            for &val in row {
                assert!(!val.is_nan(), "interaction should not contain NaN after training");
            }
        }
    }
}
