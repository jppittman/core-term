//! # Factored Embedding NNUE Architecture
//!
//! An O(ops) alternative to the O(ops²) HalfEP feature encoding.
//!
//! ## The Problem
//!
//! HalfEP features encode all (perspective_op, descendant_op, depth, path) tuples:
//! - 50 ops → 50² × 8 × 256 = 5.1M possible features
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

extern crate alloc;

use alloc::vec::Vec;
use libm::{logf, sqrtf};

use crate::egraph::Rewrite;
use crate::egraph::cost::latency_prior_cycles;
pub use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::kind::OpMap;

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension per operation.
///
/// Each operation gets a K-dimensional learned embedding. The accumulator
/// stores 2K values: K for parent roles, K for child roles.
pub const K: usize = 32;

/// Number of scalar features appended to each accumulator.
///
/// For the edge tower (`w1`, [`INPUT_DIM`]) these four slots carry the
/// variance histogram (const / frame / scanline / pixel fractions) — see
/// [`EdgeAccumulator::extraction_input`], the single place that builds them.
/// `nnue::guide`'s graph tower reuses this constant for the same reason
/// (log2-compressed search-resource scalars), so it stays `pub` rather than
/// `pub(crate)`.
pub const SCALAR_FEATURE_COUNT: usize = 4;

/// Total input dimension to the hidden layer:
/// 4K (dual accumulator: 2K flat + 2K depth-encoded) + 4 scalars
/// (the variance histogram — see [`EdgeAccumulator::extraction_input`]).
pub const INPUT_DIM: usize = 4 * K + SCALAR_FEATURE_COUNT;

/// Maximum arity for child-index encoding.
/// Effective depth = `depth * MAX_ARITY + child_index`, where child_index ∈ [0, MAX_ARITY).
/// This breaks sibling symmetry: left and right children of the same parent get different PEs.
pub const MAX_ARITY: usize = 3;

/// Maximum effective depth for learned depth embeddings.
/// Child-index encoding triples the effective depth range: a tree of real depth 63
/// with ternary nodes → `63*3+2 = 191 < 192`. Depths beyond this are clamped.
pub const MAX_DEPTH: usize = 192;

/// Hidden layer size.
pub const HIDDEN_DIM: usize = 64;

// ============================================================================
// Shared Embedding Constants
// ============================================================================

/// Embedding dimension for the expr/rule/graph embedding space. Shared by the
/// live extraction head (`expr_proj`, `value_mlp`) and by `nnue::guide`'s
/// saturation head (mask MLP, rule projection) — one embedding space, two
/// downstream heads.
pub const EMBED_DIM: usize = 32;

/// Hidden dimension for private per-head MLPs.
pub const MLP_HIDDEN: usize = 16;

// ============================================================================
// Rule Templates (LHS/RHS Expression Templates)
// ============================================================================

/// Rule templates: LHS and RHS expressions for each rule.
///
/// These use the SAME expr_embed as extraction/saturation heads, enabling the model
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
/// Arena-backed rule templates: one [`ArenaRuleTemplate`] per rule index.
///
/// `None` slots are rules that define no structural template (or only one
/// side). Built from the [`Rewrite`] trait via [`RuleTemplates::build`], which
/// reads each rule's LHS/RHS directly into a per-rule [`ExprArena`].
#[derive(Clone, Default)]
pub struct RuleTemplates {
    /// One optional arena-backed template per rule, indexed by rule_idx.
    pub rules: Vec<Option<ArenaRuleTemplate>>,
}

impl RuleTemplates {
    /// Create empty templates.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create templates for a given number of rules (all `None` initially).
    #[must_use]
    pub fn with_capacity(num_rules: usize) -> Self {
        Self {
            rules: (0..num_rules).map(|_| None).collect(),
        }
    }

    /// Build and store the LHS/RHS template for `rule` at `rule_idx`, reading
    /// them directly from the [`Rewrite`] trait into a per-rule arena.
    ///
    /// Only stored when the rule defines BOTH sides (legacy semantics).
    pub fn build(&mut self, rule_idx: usize, rule: &dyn Rewrite) {
        if rule_idx >= self.rules.len() {
            self.rules.resize_with(rule_idx + 1, || None);
        }
        let tmpl = ArenaRuleTemplate::from_rule(rule);
        if tmpl.lhs.is_some() && tmpl.rhs.is_some() {
            self.rules[rule_idx] = Some(tmpl);
        }
    }

    /// Get the arena-backed template for a rule, if defined.
    #[must_use]
    pub fn get(&self, rule_idx: usize) -> Option<&ArenaRuleTemplate> {
        self.rules.get(rule_idx).and_then(|o| o.as_ref())
    }

    /// Number of rule slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Check if a rule has a template defined.
    #[must_use]
    pub fn has_templates(&self, rule_idx: usize) -> bool {
        self.get(rule_idx).is_some()
    }

    /// Returns `true` if any template (LHS or RHS) has `op` as its root op.
    #[must_use]
    pub fn has_root_op(&self, op: OpKind) -> bool {
        self.rules
            .iter()
            .flatten()
            .any(|t| t.lhs_op == Some(op) || t.rhs_op == Some(op))
    }

    /// Build a precomputed O(1) set of root ops appearing in any template.
    #[must_use]
    pub fn root_op_set(&self) -> OpMap<bool> {
        let mut set = OpMap::splat(false);
        for t in self.rules.iter().flatten() {
            if let Some(op) = t.lhs_op {
                set[op] = true;
            }
            if let Some(op) = t.rhs_op {
                set[op] = true;
            }
        }
        set
    }
}

// ============================================================================
// Arena Rule Templates
// ============================================================================

/// A single rule stored as two subtrees inside one shared [`ExprArena`].
///
/// `lhs` and `rhs` are roots inside `arena`. Either may be `None` when the
/// corresponding side was not provided by the rule.
#[derive(Clone)]
pub struct ArenaRuleTemplate {
    /// Shared arena holding both the LHS and RHS subtrees.
    pub arena: ExprArena,
    /// Root of the LHS pattern, or `None`.
    pub lhs: Option<ExprId>,
    /// Root of the RHS pattern, or `None`.
    pub rhs: Option<ExprId>,
    /// Precomputed: LHS root op kind (if LHS is not a bare Var).
    pub lhs_op: Option<OpKind>,
    /// Precomputed: RHS root op kind (if RHS is not a bare Var).
    pub rhs_op: Option<OpKind>,
}

impl ArenaRuleTemplate {
    /// Build the LHS/RHS templates of `rule` directly into a fresh arena.
    #[must_use]
    pub fn from_rule(rule: &dyn Rewrite) -> Self {
        let mut arena = ExprArena::with_capacity(16);
        let lhs = rule.lhs_template(&mut arena);
        let rhs = rule.rhs_template(&mut arena);

        let lhs_op = lhs.and_then(|id| {
            if matches!(arena.node(id), ExprNode::Var(_)) {
                None
            } else {
                Some(arena.kind(id))
            }
        });
        let rhs_op = rhs.and_then(|id| {
            if matches!(arena.node(id), ExprNode::Var(_)) {
                None
            } else {
                Some(arena.kind(id))
            }
        });

        Self {
            arena,
            lhs,
            rhs,
            lhs_op,
            rhs_op,
        }
    }
}

/// Arena-backed rule template storage for the mask head.
pub struct ArenaRuleTemplates {
    /// One arena-backed template per rule, indexed by rule_idx.
    pub arenas: Vec<ArenaRuleTemplate>,
    /// Precomputed O(1) op-membership set (same semantics as `root_op_set()`).
    pub root_op_set: OpMap<bool>,
}

impl ArenaRuleTemplates {
    /// Convert [`RuleTemplates`] into dense arena form (one entry per rule).
    #[must_use]
    pub fn from_rule_templates(templates: &RuleTemplates) -> Self {
        let mut arenas = Vec::with_capacity(templates.len());
        let mut root_op_set = OpMap::splat(false);

        for slot in &templates.rules {
            let tmpl = match slot {
                Some(t) => t.clone(),
                None => ArenaRuleTemplate {
                    arena: ExprArena::new(),
                    lhs: None,
                    rhs: None,
                    lhs_op: None,
                    rhs_op: None,
                },
            };
            if let Some(op) = tmpl.lhs_op {
                root_op_set[op] = true;
            }
            if let Some(op) = tmpl.rhs_op {
                root_op_set[op] = true;
            }
            arenas.push(tmpl);
        }

        Self {
            arenas,
            root_op_set,
        }
    }

    /// Number of rules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.arenas.len()
    }

    /// `true` if there are no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.arenas.is_empty()
    }
}

// ============================================================================
// Operation Embeddings
// ============================================================================

/// Learned dense embeddings for each operation type.
///
/// Each of the [`OpMap::LEN`] operations gets a K-dimensional embedding
/// vector.
/// These are the primary learned parameters that capture semantic
/// similarity between operations.
#[derive(Clone)]
pub struct OpEmbeddings {
    /// E[op][i] = i-th dimension of op's embedding.
    /// Stored as one K-vector per op: `OpMap::LEN * K` = 50 × 32 = 1,600 floats.
    pub e: OpMap<[f32; K]>,
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
            e: OpMap::splat([0.0; K]),
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
    ///
    /// Dimension 0 = latency, squashed to `[0, 1]` from the shared
    /// [`latency_prior_cycles`] cycle table (source of truth, also used by
    /// `egraph::cost::CostModel::latency_prior`) via
    /// `ln(1+cycles) / ln(1+1000)`.
    ///
    /// The squash is logarithmic, not linear: the 2026-08-10 re-measurement
    /// of the cycle table spread real ops across 3..=196 cycles (Pow's
    /// lowered form is 196, not the old 12), so the previous linear `/20`
    /// clamp would pin every op from Rsqrt (21) to Pow (196) at 1.0 —
    /// indistinguishable from each other *and* from Dwrt's prohibitive 1000.
    /// The log curve keeps both ends discriminable: Add 0.23, Sqrt 0.40,
    /// Sin 0.62, Pow 0.77, Dwrt 1.0. Affects fresh initializations only;
    /// trained weight files are untouched.
    pub fn init_with_latency_prior(&mut self, seed: u64) {
        // Dwrt's deliberately-prohibitive 1000-cycle entry maps to exactly
        // 1.0; everything real lands strictly below it.
        const LATENCY_CEILING_CYCLES: f32 = 1000.0;

        let mut rng_state = seed.wrapping_add(1);
        let small_scale = 0.1; // Small noise for other dimensions

        let cycles_of = latency_prior_cycles();
        for op in OpKind::all() {
            // Dimension 0: latency prior, log-squashed from the shared cycle
            // table.
            let cycles = cycles_of[op] as f32;
            let squashed = logf(1.0 + cycles) / logf(1.0 + LATENCY_CEILING_CYCLES);
            self.e[op][0] = squashed.min(1.0);

            // Dimensions 1..K: small random for learning interactions
            for dim in 1..K {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                self.e[op][dim] = (uniform * 2.0 - 1.0) * small_scale;
            }
        }
    }

    /// Randomize embeddings in place (fully random, no priors).
    pub fn randomize(&mut self, seed: u64) {
        let scale = sqrtf(2.0 / K as f32);
        let mut rng_state = seed.wrapping_add(1);

        for op in OpKind::all() {
            for dim in 0..K {
                // LCG for no_std compatibility
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                // Convert to [-1, 1] and scale
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                let centered = uniform * 2.0 - 1.0;
                self.e[op][dim] = centered * scale;
            }
        }
    }

    /// Get embedding for an operation.
    #[inline]
    #[must_use]
    pub fn get(&self, op: OpKind) -> &[f32; K] {
        &self.e[op]
    }

    /// Total parameter count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpMap::<[f32; K]>::LEN * K
    }
}

// ============================================================================
// Sinusoidal Depth Encoding (Fixed Positional Encoding for AST Depth)
// ============================================================================

/// Precomputed sinusoidal positional encoding table.
///
/// Fixed (not learned) — zero parameters, zero serialization, zero gradients.
/// The downstream weights in w1 learn how to USE the rotation; the encoding
/// itself is a deterministic function of depth.
///
/// Each depth level gets a K-dimensional vector where:
///   PE[d][2i]   = sin(d / 10000^(2i/K))
///   PE[d][2i+1] = cos(d / 10000^(2i/K))
///
/// Used via Hadamard product: `E[op] ⊙ PE[depth]`
/// This binds depth to operation without destroying magnitude —
/// additive encoding (E + PE) would decouple in the commutative sum.
static DEPTH_PE: [[f32; K]; MAX_DEPTH] = {
    let mut table = [[0.0f32; K]; MAX_DEPTH];
    let mut depth = 0;
    while depth < MAX_DEPTH {
        let mut dim = 0;
        while dim < K {
            // 10000^(2*(dim/2)/K) computed via exp/log in const context
            // We use a simpler geometric series: base = 10000^(1/K) ≈ 1.318
            // freq = base^(-dim_pair) where dim_pair = 2*(dim/2)
            let dim_pair = 2 * (dim / 2);
            // Approximate: 10000^(dim_pair/K) via repeated squaring in f64
            // For const context, we compute the exponent directly.
            let exponent: f64 = (dim_pair as f64) / (K as f64);
            // 10000^exponent via exp(exponent * ln(10000))
            // ln(10000) ≈ 9.210340371976184
            let log_base: f64 = 9.210340371976184;
            let divisor: f64 = const_exp(exponent * log_base);
            let angle: f64 = (depth as f64) / divisor;
            // sin/cos via Taylor series (const-compatible)
            if dim % 2 == 0 {
                table[depth][dim] = const_sin(angle) as f32;
            } else {
                table[depth][dim] = const_cos(angle) as f32;
            }
            dim += 1;
        }
        depth += 1;
    }
    table
};

/// Const-compatible exp(x) via Taylor series (18 terms, accurate to ~1e-15).
const fn const_exp(x: f64) -> f64 {
    let mut result: f64 = 1.0;
    let mut term: f64 = 1.0;
    let mut i: u32 = 1;
    while i <= 18 {
        term *= x / (i as f64);
        result += term;
        i += 1;
    }
    result
}

/// Const-compatible sin(x) via Taylor series.
/// Reduces x to [-pi, pi] first for accuracy.
const fn const_sin(x: f64) -> f64 {
    // Reduce to [-pi, pi]
    let pi: f64 = 3.141592653589793;
    let two_pi: f64 = 6.283185307179586;
    let mut r = x;
    // Simple modular reduction (good enough for small positive x)
    while r > pi {
        r -= two_pi;
    }
    while r < -pi {
        r += two_pi;
    }
    // Taylor: sin(r) = r - r^3/6 + r^5/120 - ...
    let mut result: f64 = 0.0;
    let mut term: f64 = r;
    let r2 = r * r;
    let mut i: u32 = 0;
    while i < 12 {
        result += term;
        term *= -r2 / (((2 * i + 2) * (2 * i + 3)) as f64);
        i += 1;
    }
    result
}

/// Const-compatible cos(x) via Taylor series.
const fn const_cos(x: f64) -> f64 {
    let pi: f64 = 3.141592653589793;
    let two_pi: f64 = 6.283185307179586;
    let mut r = x;
    while r > pi {
        r -= two_pi;
    }
    while r < -pi {
        r += two_pi;
    }
    let mut result: f64 = 0.0;
    let mut term: f64 = 1.0;
    let r2 = r * r;
    let mut i: u32 = 0;
    while i < 12 {
        result += term;
        term *= -r2 / (((2 * i + 1) * (2 * i + 2)) as f64);
        i += 1;
    }
    result
}

/// Look up the sinusoidal positional encoding for a given depth.
/// Depths beyond MAX_DEPTH are clamped.
#[inline]
pub fn depth_pe(depth: u32) -> &'static [f32; K] {
    &DEPTH_PE[depth.min((MAX_DEPTH - 1) as u32) as usize]
}

// ============================================================================
// Edge Accumulator (Dual: Flat + Depth-Encoded)
// ============================================================================

/// Dual accumulator for edge-based feature extraction.
///
/// Split into two physically distinct representations:
///
/// - **Flat half (0..2K):** `Σ E[parent]` and `Σ E[child]`.
///   Pure throughput — the network knows exactly how many of each operation
///   exist. A `cos` always contributes its full embedding regardless of depth.
///
/// - **Depth-encoded half (2K..4K):** `Σ (E[parent] ⊙ PE[depth])` and
///   `Σ (E[child] ⊙ PE[depth])`. Pure geometry — the Hadamard product with
///   sinusoidal positional encoding binds each operation to its tree position
///   without destroying its magnitude. The network sees ILP constraints and
///   pipeline bottlenecks.
///
/// Both halves support O(1) incremental updates via vector addition/subtraction.
#[derive(Clone)]
pub struct EdgeAccumulator {
    /// Contiguous accumulator values.
    /// - `[0..K]`:     flat parent sum (throughput)
    /// - `[K..2K]`:    flat child sum (throughput)
    /// - `[2K..3K]`:   depth-encoded parent sum (geometry)
    /// - `[3K..4K]`:   depth-encoded child sum (geometry)
    pub values: [f32; 4 * K],

    /// Edge count (O(1) additive scalar).
    pub edge_count: u32,

    /// Node count (O(1) additive scalar).
    pub node_count: u32,

    /// E-graph node budget for this trajectory (how many nodes the saturator may create).
    /// Carried for saturation-head experiments; NOT fed to the extraction head
    /// (see [`EdgeAccumulator::extraction_input`]).
    pub node_budget: u32,

    /// Epoch budget for this trajectory (max saturation epochs).
    /// Carried for saturation-head experiments; NOT fed to the extraction head.
    pub epoch_budget: u32,

    // -- Variance features (fed to extraction head) --
    /// Fraction of nodes that are compile-time constants (variance = {}).
    pub variance_frac_const: f32,

    /// Fraction of nodes that are frame-uniform (variance ⊆ {Z, W}, no X or Y).
    pub variance_frac_frame: f32,

    /// Fraction of nodes that are scanline-uniform (have Y but no X).
    pub variance_frac_scanline: f32,

    /// Fraction of nodes that are pixel-varying (have X).
    pub variance_frac_pixel: f32,
}

impl Default for EdgeAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeAccumulator {
    /// Create a zero-initialized dual accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: [0.0; 4 * K],
            edge_count: 0,
            node_count: 0,
            node_budget: 0,
            epoch_budget: 0,
            variance_frac_const: 0.0,
            variance_frac_frame: 0.0,
            variance_frac_scanline: 0.0,
            variance_frac_pixel: 0.0,
        }
    }

    /// Reset to zero state.
    ///
    /// Budget fields are intentionally NOT reset — they are trajectory-level
    /// properties that should persist across epoch rebuilds.
    pub fn reset(&mut self) {
        self.values = [0.0; 4 * K];
        self.edge_count = 0;
        self.node_count = 0;
    }

    /// The extraction head's network input vector — the SINGLE feature
    /// construction point shared by deployment
    /// ([`ExprNnue::forward_expr_only`], called by
    /// `IncrementalExtractor::extract_choices_only`) and training
    /// (`pixelflow-pipeline`'s `forward_cached`).
    ///
    /// Layout:
    /// - `[0 .. 4K)`: the dual accumulator, scaled by `1/sqrt(node_count)`
    ///   (prevents variance explosion from summing N embedding vectors).
    /// - `[4K .. 4K+4)`: the variance histogram — fractions of nodes that are
    ///   const / frame-uniform / scanline-uniform / pixel-varying.
    ///
    /// Round 0 of the 2026-08 workflow found the trainer feeding
    /// log2-compressed count/budget scalars into slots `4K..4K+4` while
    /// deployment fed variance fractions into the same `w1` rows — two
    /// meanings for one weight, a −0.29 log-ns deployed bias. Any future
    /// feature must be added HERE, where both paths inherit it at once.
    #[must_use]
    pub fn extraction_input(&self) -> [f32; INPUT_DIM] {
        let mut input = [0.0f32; INPUT_DIM];

        let scale = if self.node_count > 0 {
            1.0 / sqrtf(self.node_count as f32)
        } else {
            1.0
        };
        for (slot, &val) in input.iter_mut().zip(self.values.iter()) {
            *slot = val * scale;
        }

        input[4 * K] = self.variance_frac_const;
        input[4 * K + 1] = self.variance_frac_frame;
        input[4 * K + 2] = self.variance_frac_scanline;
        input[4 * K + 3] = self.variance_frac_pixel;
        input
    }

    /// Add a single edge contribution (both flat and depth-encoded).
    ///
    /// Flat half: raw embedding addition (preserves magnitude).
    /// Depth half: complex multiplication — each pair `(2f, 2f+1)` represents
    /// `(real, imaginary)` for frequency `f`. PE stores `sin` at even, `cos` at
    /// odd indices. Complex: `(emb_re + j·emb_im) × (cos + j·sin)`.
    #[inline]
    pub fn add_edge(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: u32,
    ) {
        let pe = depth_pe(depth);
        self.add_edge_with_pe(emb, parent_op, child_op, pe);
    }

    /// Add a single edge with caller-provided PE (used by InstructionWindow).
    #[inline]
    pub fn add_edge_with_pe(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        pe: &[f32; K],
    ) {
        let parent_emb = emb.get(parent_op);
        let child_emb = emb.get(child_op);

        // Flat half: raw sum (unchanged)
        for i in 0..K {
            self.values[i] += parent_emb[i];
            self.values[K + i] += child_emb[i];
        }

        // Depth-encoded half: complex multiply
        // Each pair (2f, 2f+1) represents (real, imaginary) for frequency f.
        // PE stores sin at even, cos at odd indices.
        // Complex: (emb_re + j·emb_im) × (cos + j·sin)
        for f in 0..K / 2 {
            let sin_d = pe[2 * f];
            let cos_d = pe[2 * f + 1];

            let p_re = parent_emb[2 * f];
            let p_im = parent_emb[2 * f + 1];
            self.values[2 * K + 2 * f] += p_re * cos_d - p_im * sin_d;
            self.values[2 * K + 2 * f + 1] += p_re * sin_d + p_im * cos_d;

            let c_re = child_emb[2 * f];
            let c_im = child_emb[2 * f + 1];
            self.values[3 * K + 2 * f] += c_re * cos_d - c_im * sin_d;
            self.values[3 * K + 2 * f + 1] += c_re * sin_d + c_im * cos_d;
        }
        self.edge_count += 1;
    }

    /// Remove a single edge contribution (for incremental updates).
    #[inline]
    pub fn remove_edge(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: u32,
    ) {
        let pe = depth_pe(depth);
        self.remove_edge_with_pe(emb, parent_op, child_op, pe);
    }

    /// Remove a single edge with caller-provided PE (used by InstructionWindow).
    #[inline]
    pub fn remove_edge_with_pe(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        pe: &[f32; K],
    ) {
        let parent_emb = emb.get(parent_op);
        let child_emb = emb.get(child_op);

        // Flat half: raw subtract
        for i in 0..K {
            self.values[i] -= parent_emb[i];
            self.values[K + i] -= child_emb[i];
        }

        // Depth-encoded half: complex multiply (subtract)
        for f in 0..K / 2 {
            let sin_d = pe[2 * f];
            let cos_d = pe[2 * f + 1];

            let p_re = parent_emb[2 * f];
            let p_im = parent_emb[2 * f + 1];
            self.values[2 * K + 2 * f] -= p_re * cos_d - p_im * sin_d;
            self.values[2 * K + 2 * f + 1] -= p_re * sin_d + p_im * cos_d;

            let c_re = child_emb[2 * f];
            let c_im = child_emb[2 * f + 1];
            self.values[3 * K + 2 * f] -= c_re * cos_d - c_im * sin_d;
            self.values[3 * K + 2 * f + 1] -= c_re * sin_d + c_im * cos_d;
        }
        self.edge_count = self.edge_count.saturating_sub(1);
    }

    // ========================================================================
    // DAG-Aware Accumulator Construction (the ONE walker)
    // ========================================================================

    /// Build an accumulator from any [`CostDag`] view.
    ///
    /// This is the ONLY function that turns an expression DAG into extraction
    /// head features, no matter whether the DAG lives in an [`ExprArena`]
    /// (training labels, DEV evaluation, rule templates) or in an e-graph with
    /// extraction choices (deployment). The 2026-08 round-0 audit found the
    /// trainer and the extractor building accumulators with different edge
    /// policies — no register-reload edges and zero variance fractions at
    /// train time — which biased the deployed head by −0.29 log-ns and
    /// compressed its prediction range to 45% of true. Routing both paths
    /// through one walker makes that divergence structurally unrepresentable:
    /// there is no second edge policy left to drift.
    ///
    /// Edge policy (matches what the JIT emits):
    /// - The first reference to a node is its computation edge
    ///   `(parent_op, child_op)` at the referencing slot's effective depth.
    /// - Every later reference is a register reload: a single
    ///   `(parent_op, Var)` edge — shared subexpressions become let-bindings,
    ///   so the DAG is not tree-bloated.
    /// - Nodes with no recorded choice contribute nothing (speculative
    ///   extraction candidates may reference not-yet-backfilled classes).
    ///
    /// # Panics
    ///
    /// Panics if a node id is out of bounds, or if the view classifies
    /// variance for some expanded nodes but not all of them (a half-populated
    /// histogram would silently feed the model garbage fractions).
    fn from_cost_dag<D: CostDag>(dag: &D, emb: &OpEmbeddings) -> Self {
        let mut acc = Self::new();
        let bound = dag.id_bound();
        let mut expanded = alloc::vec![false; bound];
        // Tracks which child nodes have already received their computation
        // edge. The first reference is a computation edge; every later
        // reference is a register reload (a single var_ref edge).
        let mut edge_emitted = alloc::vec![false; bound];
        // Variance counters
        let mut n_const: u32 = 0;
        let mut n_frame: u32 = 0;
        let mut n_scanline: u32 = 0;
        let mut n_pixel: u32 = 0;
        let mut variance_classified: u32 = 0;
        // Stack: (node id, depth). Reused children scratch buffer.
        let mut stack: Vec<(u32, u32)> = alloc::vec![(dag.root(), 0)];
        let mut children: Vec<u32> = Vec::new();

        while let Some((id, depth)) = stack.pop() {
            let idx = id as usize;
            assert!(
                idx < bound,
                "from_cost_dag: node id {id} out of bounds (bound={bound})"
            );

            // Always increment node_count on first expansion.
            // Subsequent visits to a shared node only add var_ref edges.
            if expanded[idx] {
                continue;
            }
            expanded[idx] = true;

            children.clear();
            let Some(parent_op) = dag.resolve(id, &mut children) else {
                continue; // No recorded choice — contributes nothing.
            };
            acc.node_count += 1;

            // Classify this node's variance if analysis is available
            if let Some(v) = dag.variance(id) {
                variance_classified += 1;
                if v.is_const() {
                    n_const += 1;
                } else if v.is_x_invariant() && !v.depends_on_y() {
                    // Frame-uniform: depends only on Z/W (no X, no Y)
                    n_frame += 1;
                } else if v.is_x_invariant() {
                    // Scanline-uniform: depends on Y (but not X)
                    n_scanline += 1;
                } else {
                    // Pixel-varying: depends on X
                    n_pixel += 1;
                }
            }

            // One edge per child slot; leaves have no slots and add no edges.
            for (child_idx, &child) in children.iter().enumerate() {
                // `None`-tolerant unlike the parent resolve above: on
                // SPECULATIVE, in-progress `choices` during
                // `extract_choices_only`'s candidate search, a tentative swap
                // is scored *before* it is accepted, and children introduced
                // by that (possibly-rejected) swap are only backfilled if the
                // swap wins. Skip the edge rather than fabricating one from a
                // guessed node.
                let Some(child_op) = dag.child_kind(child) else {
                    continue;
                };

                let eff_depth = depth * MAX_ARITY as u32 + (child_idx.min(MAX_ARITY - 1)) as u32;

                if edge_emitted[child as usize] {
                    // Shared reuse: a register reload, not a recomputation.
                    acc.add_var_ref_edges(emb, parent_op, eff_depth, 1);
                } else {
                    edge_emitted[child as usize] = true;
                    acc.add_edge(emb, parent_op, child_op, eff_depth);
                }

                // Push child for expansion (guarded by `expanded`).
                stack.push((child, depth + 1));
            }
        }

        // Populate variance histogram fractions. Either every expanded node
        // was classified or none were — a mixed view is a bug, not a mode.
        assert!(
            variance_classified == 0 || variance_classified == acc.node_count,
            "from_cost_dag: variance classified for {variance_classified} of {} nodes — \
             a partially-populated histogram would silently corrupt the features",
            acc.node_count
        );
        if variance_classified > 0 && acc.node_count > 0 {
            let total = acc.node_count as f32;
            acc.variance_frac_const = n_const as f32 / total;
            acc.variance_frac_frame = n_frame as f32 / total;
            acc.variance_frac_scanline = n_scanline as f32 / total;
            acc.variance_frac_pixel = n_pixel as f32 / total;
        }

        acc
    }

    /// Build the accumulator from an arena DAG, in the DEPLOYMENT
    /// representation (register-reload edges for shared nodes, variance
    /// histogram populated).
    ///
    /// This is the training-side entry point: labels are minted by JIT-timing
    /// the arena, and the JIT let-binds shared `ExprId`s, so featurizing the
    /// same sharing the same way keeps features faithful to the measured
    /// object — and identical to what [`Self::from_dag_choices_with_variance`]
    /// produces for the equivalent e-graph DAG (both are thin adapters over
    /// [`Self::from_cost_dag`]; see `train_and_deploy_feature_paths_agree` in
    /// `egraph::extract`).
    ///
    /// # Panics
    ///
    /// Panics if the subtree contains `ExprNode::Param` — substitute
    /// parameters before costing.
    #[must_use]
    pub fn from_arena_dag(arena: &ExprArena, root: ExprId, emb: &OpEmbeddings) -> Self {
        let variance = pixelflow_ir::variance::compute_arena_variance(arena);
        Self::from_cost_dag(
            &ArenaCostDag {
                arena,
                root,
                variance,
            },
            emb,
        )
    }

    /// Build accumulator from an [`Extraction`](crate::egraph::extract::Extraction)
    /// with DAG-aware sharing (no variance histogram — prefer
    /// [`Self::from_dag_choices_with_variance`]).
    pub fn from_dag_choices(
        extraction: &crate::egraph::extract::Extraction<'_>,
        emb: &OpEmbeddings,
    ) -> Self {
        Self::from_dag_choices_with_variance(extraction, emb, false)
    }

    /// Build accumulator from an [`Extraction`](crate::egraph::extract::Extraction),
    /// optionally populating the variance histogram — the DEPLOYMENT-side
    /// adapter over [`Self::from_cost_dag`].
    ///
    /// When `with_variance` is set, variance is computed recursively over
    /// the extraction's CHOSEN nodes
    /// ([`Extraction::chosen_variance`](crate::egraph::extract::Extraction::chosen_variance)),
    /// not the class-wide meet `DepsAnalysis` computes — see P1(c) in
    /// docs/plans/2026-08-17-cost-model-domain.md. Once a rewrite merges a
    /// pixel-varying node into a class alongside a constant one, the
    /// class-wide meet reports CONST regardless of which node the
    /// extraction actually chose; recursing over the chosen nodes instead
    /// keeps this identical to what [`Self::from_arena_dag`] computes on
    /// the exact arena the extraction emits.
    pub fn from_dag_choices_with_variance(
        extraction: &crate::egraph::extract::Extraction<'_>,
        emb: &OpEmbeddings,
        with_variance: bool,
    ) -> Self {
        // Computed once and shared by `resolve`/`child_kind` (structure) and
        // `chosen_variance` (variance) below, so both walk the identical
        // Shl/Shr-pinned view `choices_to_arena` will materialise — see
        // `ChoicesCostDag::pinned`'s doc comment for why splitting these
        // across two different choice views is a train/deploy skew, not
        // just a redundant computation.
        let pinned = extraction.pinned_choices();
        let variance = with_variance.then(|| extraction.chosen_variance(&pinned));
        Self::from_cost_dag(
            &ChoicesCostDag {
                extraction,
                pinned,
                variance,
            },
            emb,
        )
    }

    /// Add N var-reference edges (representing register loads of a shared value).
    ///
    /// Each var-ref edge is `(Var, parent_op)` at the given depth. This tells the
    /// extraction head: "this parent loads a let-bound value N times."
    pub fn add_var_ref_edges(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        depth: u32,
        count: u32,
    ) {
        for _ in 0..count {
            self.add_edge(emb, parent_op, OpKind::Var, depth);
        }
    }

    /// Remove N var-reference edges (inverse of `add_var_ref_edges`).
    pub fn remove_var_ref_edges(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        depth: u32,
        count: u32,
    ) {
        for _ in 0..count {
            self.remove_edge(emb, parent_op, OpKind::Var, depth);
        }
    }
}

// ============================================================================
// CostDag: the single view both feature paths walk
// ============================================================================

/// A DAG of chosen expression nodes, as the extraction head featurizes it.
///
/// Implemented by exactly two adapters — [`ArenaCostDag`] (training / rule
/// templates) and [`ChoicesCostDag`] (deployment) — and consumed by exactly
/// one walker, [`EdgeAccumulator::from_cost_dag`]. Ids must be canonical:
/// structurally shared subexpressions present the same id on every reference,
/// because sharing is what the reload-edge policy keys on.
trait CostDag {
    /// Exclusive upper bound on node ids.
    fn id_bound(&self) -> usize;

    /// Canonical id of the root node.
    fn root(&self) -> u32;

    /// Resolve the chosen representation of `id`: append its canonical child
    /// ids to `out` (nothing for leaves) and return its op kind. `None` when
    /// the node has no recorded choice (speculative extraction candidates).
    ///
    /// # Panics
    ///
    /// Panics when a recorded choice is malformed (e.g. out-of-bounds node
    /// index) — that is a broken invariant, not a tolerable state.
    fn resolve(&self, id: u32, out: &mut Vec<u32>) -> Option<OpKind>;

    /// Tolerant kind lookup for child edges: `None` when the child has no
    /// recorded (in-bounds) choice, in which case the edge is skipped rather
    /// than fabricated.
    fn child_kind(&self, id: u32) -> Option<OpKind>;

    /// Variance of the node, when variance analysis is available. Must be
    /// `Some` for every node or for none (asserted by the walker).
    fn variance(&self, id: u32) -> Option<pixelflow_ir::Variance>;
}

/// Training-side [`CostDag`]: an [`ExprArena`] subtree.
///
/// Sharing is by `ExprId` — exactly the sharing the JIT's let-binding emitter
/// sees when this same arena is benchmarked for its label, so the reload-edge
/// policy describes the measured object.
struct ArenaCostDag<'a> {
    arena: &'a ExprArena,
    root: ExprId,
    /// Per-`ExprId` variance from `pixelflow_ir::variance::compute_arena_variance`.
    variance: Vec<pixelflow_ir::Variance>,
}

impl CostDag for ArenaCostDag<'_> {
    fn id_bound(&self) -> usize {
        self.arena.len()
    }

    fn root(&self) -> u32 {
        self.root.0
    }

    fn resolve(&self, id: u32, out: &mut Vec<u32>) -> Option<OpKind> {
        let eid = ExprId(id);
        if let ExprNode::Param(i) = self.arena.node(eid) {
            panic!("ExprNode::Param({i}) reached NNUE cost model — substitute params first");
        }
        for child in self.arena.children(eid) {
            out.push(child.0);
        }
        Some(self.arena.kind(eid))
    }

    fn child_kind(&self, id: u32) -> Option<OpKind> {
        // `arena.kind` maps Param to Const; the child itself is expanded (and
        // `resolve` panics) right after, so a Param still fails loudly.
        Some(self.arena.kind(ExprId(id)))
    }

    fn variance(&self, id: u32) -> Option<pixelflow_ir::Variance> {
        Some(self.variance[id as usize])
    }
}

/// Deployment-side [`CostDag`]: an [`Extraction`](crate::egraph::extract::Extraction)
/// (an e-graph plus a validated, well-founded choice function), as produced
/// by `IncrementalExtractor::extract_choices_only`.
struct ChoicesCostDag<'a> {
    extraction: &'a crate::egraph::extract::Extraction<'a>,
    /// [`Extraction::pinned_choices`] — the same `Shl`/`Shr` count
    /// substitution `choices_to_arena` applies, computed once so `resolve`
    /// and `child_kind` walk the DAG `choices_to_arena` will actually
    /// materialise rather than whatever node the extraction chose for a
    /// count class. Using the raw (unpinned) `extraction.choice` here would
    /// let this walker descend into a count class's non-`Const` alternative
    /// — inflating `node_count`/`edge_count` with nodes `choices_to_arena`
    /// never emits, and (with `variance` below keyed by the pinned view)
    /// tripping `from_cost_dag`'s "half-populated histogram" assertion the
    /// moment a shift's count class holds a value-equal varying form.
    pinned: Vec<Option<usize>>,
    /// Chosen-node variance, indexed by canonical e-class id — computed over
    /// the SAME `pinned` view via
    /// [`Extraction::chosen_variance`](crate::egraph::extract::Extraction::chosen_variance).
    /// `None` when the caller didn't ask for variance features.
    variance: Option<Vec<Option<pixelflow_ir::Variance>>>,
}

impl ChoicesCostDag<'_> {
    fn kind_of(node: &crate::egraph::ENode) -> OpKind {
        use crate::egraph::ENode;
        match node {
            ENode::Var(_) => OpKind::Var,
            ENode::Const(_) => OpKind::Const,
            ENode::Buffer(_) => OpKind::Buffer,
            ENode::Op { op, .. } => op.kind(),
        }
    }

    /// The pinned choice recorded for `class`'s canonical id, if any.
    fn pinned_choice(&self, class: crate::egraph::EClassId) -> Option<usize> {
        self.pinned.get(class.0 as usize).copied().flatten()
    }
}

impl CostDag for ChoicesCostDag<'_> {
    fn id_bound(&self) -> usize {
        self.extraction.egraph().num_classes()
    }

    fn root(&self) -> u32 {
        self.extraction.root().0
    }

    fn resolve(&self, id: u32, out: &mut Vec<u32>) -> Option<OpKind> {
        use crate::egraph::{EClassId, ENode};
        let egraph = self.extraction.egraph();
        let canonical = egraph.find(EClassId(id));
        let node_idx = self.pinned_choice(canonical)?;
        let nodes = egraph.nodes(canonical);
        let node = nodes.get(node_idx).unwrap_or_else(|| {
            panic!(
                "from_dag_choices: node_idx {} out of bounds for e-class {} (has {} nodes)",
                node_idx,
                canonical.0,
                nodes.len()
            )
        });
        if let ENode::Op { children, .. } = node {
            for &child in children {
                out.push(egraph.find(child).0);
            }
        }
        Some(Self::kind_of(node))
    }

    fn child_kind(&self, id: u32) -> Option<OpKind> {
        use crate::egraph::EClassId;
        let egraph = self.extraction.egraph();
        let canonical = egraph.find(EClassId(id));
        let node_idx = self.pinned_choice(canonical)?;
        egraph.nodes(canonical).get(node_idx).map(Self::kind_of)
    }

    fn variance(&self, id: u32) -> Option<pixelflow_ir::Variance> {
        use crate::egraph::EClassId;
        let canonical = self.extraction.egraph().find(EClassId(id));
        self.variance
            .as_ref()
            .and_then(|v| v.get(canonical.0 as usize).copied().flatten())
    }
}

// ============================================================================
// Extraction (Value) Head
// ============================================================================

/// `ExprNnue`: the shared backbone plus the extraction (value) head.
///
/// ## Architecture
///
/// ```text
/// expr → OpEmbeddings → EdgeAccumulator → hidden [64] → trunk → expr_proj → expr_embed [32] → value_mlp → cost
/// ```
///
/// `expr_embed → value_mlp (32→16→1)` predicts log-nanosecond cost.
///
/// This struct is deliberately extraction-only: it is the live NNUE
/// checkpoint format ("TRIF" — see [`Self::save`]), and every param in it is
/// trained by `bootstrap_extraction_head`. The saturation (mask/policy) head
/// that used to share this struct — inert, zero non-test callers, Phase-3
/// gated — now lives behind [`crate::nnue::guide::SaturationGuide`], which
/// holds its own weights and reads this struct's shared trunk (`trunk_w`,
/// `trunk_b`), embeddings, and `expr_proj`/`forward_expr_only` by reference
/// rather than duplicating them.
#[derive(Clone)]
pub struct ExprNnue {
    // ========== SHARED (Expression Backbone) ==========
    /// Learned embeddings for each operation (50 × 32 = 1,600 params)
    pub embeddings: OpEmbeddings,

    /// Hidden layer weights: [INPUT_DIM][HIDDEN_DIM] (132 × 64 = 8,448 params)
    pub w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// Hidden layer biases: [HIDDEN_DIM] (64 params)
    pub b1: [f32; HIDDEN_DIM],

    /// Shared trunk weights: [HIDDEN_DIM][HIDDEN_DIM] (64 x 64 = 4,096 params).
    /// Read by `nnue::guide`'s graph tower too (see `apply_trunk`), so this
    /// remains "the deep conceptual representation" shared across both heads
    /// even though the mask/graph tower's own weights moved out.
    pub trunk_w: [[f32; HIDDEN_DIM]; HIDDEN_DIM],
    /// Shared trunk biases: [HIDDEN_DIM] (64 params).
    pub trunk_b: [f32; HIDDEN_DIM],

    /// Projects backbone hidden (64) to the shared expr embedding (`EMBED_DIM`=32).
    /// Weights: [HIDDEN_DIM x EMBED_DIM]
    pub expr_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Expr projection bias: [EMBED_DIM]
    pub expr_proj_b: [f32; EMBED_DIM],

    /// Value MLP layer 1 weights: expr_embed (32) → hidden (16)
    pub value_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    /// Value MLP layer 1 bias
    pub value_mlp_b1: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 weights: hidden (16) → cost (1)
    pub value_mlp_w2: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 bias
    pub value_mlp_b2: f32,
}

impl Default for ExprNnue {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprNnue {
    /// Create a zero-initialized extraction head.
    #[must_use]
    pub fn new() -> Self {
        Self {
            embeddings: OpEmbeddings::new(),
            w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            b1: [0.0; HIDDEN_DIM],

            trunk_w: [[0.0; HIDDEN_DIM]; HIDDEN_DIM],
            trunk_b: [0.0; HIDDEN_DIM],

            expr_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            expr_proj_b: [0.0; EMBED_DIM],

            value_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            value_mlp_b1: [0.0; MLP_HIDDEN],
            value_mlp_w2: [0.0; MLP_HIDDEN],
            value_mlp_b2: 5.0, // Start near typical log-cost
        }
    }

    /// Create a randomly initialized extraction head.
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

    /// Randomize only network weights, not embeddings.
    pub fn randomize_weights_only(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(12345);

        let mut next_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
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

        // Shared trunk: identity + small noise (near-identity preserves tower signal)
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                self.trunk_w[i][j] = if i == j { 1.0 } else { 0.0 } + next_f32() * 0.01;
            }
        }
        for b in &mut self.trunk_b {
            *b = 0.0;
        }

        // Initialize the shared projection + value MLP.
        self.randomize_extraction_head_with_rng(&mut next_f32);
    }

    /// Internal helper to randomize the expr projection + value MLP.
    ///
    /// ONLY used during full random init (`randomize_weights_only`).
    fn randomize_extraction_head_with_rng<F: FnMut() -> f32>(&mut self, next_f32: &mut F) {
        // He initialization scales
        let scale_proj = sqrtf(2.0 / HIDDEN_DIM as f32);
        let scale_embed = sqrtf(2.0 / EMBED_DIM as f32);
        let scale_hidden = sqrtf(2.0 / MLP_HIDDEN as f32);

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
    }

    /// Randomize all weights including embeddings.
    pub fn randomize(&mut self, seed: u64) {
        self.embeddings.randomize(seed);
        self.randomize_weights_only(seed);
    }

    /// Apply the shared trunk: `HIDDEN_DIM -> HIDDEN_DIM` with ReLU.
    ///
    /// Both this struct's edge tower and `nnue::guide`'s graph tower pass
    /// through this layer before reaching their task-specific projection
    /// heads — `pub(crate)` rather than private so the guide module (same
    /// crate, different module) can share it instead of duplicating it.
    /// Initialized near-identity so the trunk preserves tower signal until
    /// training pulls it away.
    #[inline]
    pub(crate) fn apply_trunk(&self, tower_output: &[f32; HIDDEN_DIM]) -> [f32; HIDDEN_DIM] {
        let mut out = self.trunk_b;
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                out[j] += tower_output[i] * self.trunk_w[i][j];
            }
        }
        for h in &mut out {
            *h = h.max(0.0);
        }
        out
    }

    /// Extraction head with pre-computed accumulator.
    ///
    /// More efficient when you already have the accumulator. This is the
    /// DEPLOYED entry point: `IncrementalExtractor` scores every candidate
    /// through it, and the trainer's forward pass
    /// (`pixelflow-pipeline`'s `forward_cached`) consumes the identical
    /// [`EdgeAccumulator::extraction_input`] vector, so the function being
    /// trained is the function being called.
    #[must_use]
    pub fn predict_log_cost_with_features(&self, acc: &EdgeAccumulator) -> f32 {
        let hidden = self.forward_expr_only(acc);
        let expr_embed = self.compute_expr_embed(&hidden);
        self.value_mlp_forward(&expr_embed)
    }

    /// Forward pass through the shared backbone on
    /// [`EdgeAccumulator::extraction_input`] features.
    ///
    /// The extraction head uses expression structure plus the variance
    /// histogram ONLY — no search-resource scalars (node_budget,
    /// epoch_budget, edge_count, node_count). Those exist for the saturation
    /// head's graph tower ([`Self::forward_graph`]), which needs to reason
    /// about search resources; execution cost depends only on what the
    /// expression computes.
    pub fn forward_expr_only(&self, acc: &EdgeAccumulator) -> [f32; HIDDEN_DIM] {
        let input = acc.extraction_input();
        let mut hidden = self.b1;

        for (i, &val) in input.iter().enumerate() {
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += val * self.w1[i][j];
            }
        }

        // ReLU activation
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Shared trunk
        let hidden = self.apply_trunk(&hidden);

        hidden
    }

    /// Project backbone hidden to shared expr embedding (EMBED_DIM).
    #[inline]
    pub fn compute_expr_embed(&self, hidden: &[f32; HIDDEN_DIM]) -> [f32; EMBED_DIM] {
        let mut embed = self.expr_proj_b;
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                embed[k] += hidden[j] * self.expr_proj_w[j][k];
            }
        }
        embed
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

    /// Total parameter count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpEmbeddings::param_count()           // embeddings: 50 * 32 = 1,600
            + INPUT_DIM * HIDDEN_DIM          // w1: 132 * 64 = 8,448
            + HIDDEN_DIM                      // b1: 64
            // shared trunk
            + HIDDEN_DIM * HIDDEN_DIM         // trunk_w: 64 * 64 = 4,096
            + HIDDEN_DIM                      // trunk_b: 64
            // expr_proj
            + HIDDEN_DIM * EMBED_DIM          // expr_proj_w: 64 * 32 = 2,048
            + EMBED_DIM                       // expr_proj_b: 32
            // value MLP
            + EMBED_DIM * MLP_HIDDEN          // value_mlp_w1: 32 * 16 = 512
            + MLP_HIDDEN                      // value_mlp_b1: 16
            + MLP_HIDDEN                      // value_mlp_w2: 16
            + 1 // value_mlp_b2: 1
    }

    /// Memory size in bytes (f32 weights).
    #[must_use]
    pub const fn memory_bytes() -> usize {
        Self::param_count() * 4
    }

    // NOTE: the MCTS-support policy samplers (`policy_from_accumulator`,
    // `policy_from_features`, `bernoulli_policy_from_accumulator`,
    // `sample_rules_bernoulli`) were deleted 2026-08-10: no callers anywhere
    // in the workspace, and they were the last consumers of the old
    // `forward_shared` path that fed log2 count/budget scalars into the same
    // `w1` slots the extraction head uses for the variance histogram — the
    // exact train/deploy skew this round removed. The saturation head (now
    // `nnue::guide`, its own weights entirely) scores rules through its own
    // graph tower, not this struct's edge tower.

    /// Save weights to a binary file.
    ///
    /// Format: magic "TRIF" + all weights as little-endian f32.
    /// TRIF: **live-only** — this checkpoint holds exactly the params
    /// [`Self::param_count`] counts (the extraction head + shared backbone,
    /// 16,897 params / 67,588 bytes). Earlier formats (TRIE and before) also
    /// serialized the saturation head's mask/rule/graph weights — untrained
    /// noise that made up roughly half the file by byte count and drifted
    /// toward zero every training run under weight decay with no gradient
    /// ever opposing it, since nothing trained them. Op embeddings are one
    /// row per op, in `OpKind`'s order.
    #[cfg(feature = "std")]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::io::BufWriter::with_capacity(256 * 1024, std::fs::File::create(path)?);

        // Magic header. Bump it whenever these bytes change meaning — including
        // when `OpKind`'s order changes, since the embedding rows follow it.
        file.write_all(b"TRIF")?;

        // ===== Backbone =====
        // Embeddings
        for row in self.embeddings.e.as_slice() {
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

        // Shared trunk
        for row in &self.trunk_w {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.trunk_b {
            file.write_all(&val.to_le_bytes())?;
        }

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

        Ok(())
    }

    /// Load weights from an in-memory byte slice.
    ///
    /// Used by the compiler to load weights embedded via `include_bytes!`.
    #[cfg(feature = "std")]
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        Self::load_from_reader(std::io::Cursor::new(bytes))
    }

    /// Load weights from a binary file.
    ///
    /// Only supports "TRIF" format. Older formats (TRIE and earlier) require retrain.
    #[cfg(feature = "std")]
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::io::BufReader::with_capacity(256 * 1024, std::fs::File::open(path)?);
        Self::load_from_reader(file)
    }

    #[cfg(feature = "std")]
    fn load_from_reader<R: std::io::Read>(mut file: R) -> std::io::Result<Self> {
        // Check magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        // TRIF: live-only — extraction head + shared backbone ONLY. Op
        //       embeddings are one row per op, in `OpKind`'s order. That
        //       order is `pixelflow-ir`'s to change, and changing it silently
        //       re-attaches every row to a different operation — so a change
        //       there means a new magic here.
        // TRIE: incompatible — also serialized the saturation head's
        //       mask/rule/graph weights (untrained noise, since nothing
        //       trained them — see the 2026-08-17 cost-model domain-model
        //       reorganization). Those params now live behind
        //       `nnue::guide::SaturationGuide`, with their own (unused until
        //       Phase 3) checkpoint format — retrain the extraction head from
        //       this file's raw bytes is not possible; a TRIE file is simply
        //       a different, incompatible layout.
        // TRID: incompatible — op embeddings are indexed by `OpKind::index()`,
        //       and the pre-2026-08-02 discriminants had gaps at 17/31/39. The
        //       shapes still match, so a TRID file would load without error and
        //       silently attribute every op past a gap to its neighbour.
        // TRIC: incompatible — pre-shared-trunk architecture
        // TRIB: incompatible — GRAPH_ACC_DIM was 3K (96), GRAPH_INPUT_DIM was 100
        // TRIA: incompatible — had mask_rule_bias[1024] instead of mask_bias_proj[32]
        // TRI5-TRI9: incompatible — EMBED_DIM was 24, all weight shapes differ
        if &magic != b"TRIF" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Incompatible ExprNnue format {:?}. Expected 'TRIF'. {}",
                    std::str::from_utf8(&magic).unwrap_or("????"),
                    if &magic == b"TRIE" {
                        "'TRIE' predates the live-only checkpoint split (2026-08-17): it also \
                         carries the (untrained) saturation-head weights this format dropped. \
                         Retrain required."
                    } else if &magic == b"TRID" {
                        "'TRID' predates the dense op numbering; its embeddings are \
                         attributed to the wrong operations. Retrain required."
                    } else {
                        "Old formats (TRIE, TRIC, TRIB, TRIA, TRI5-TRI9) require retrain."
                    }
                ),
            ));
        }

        let mut net = Self::new();

        // ===== Backbone =====
        // Embeddings
        for row in net.embeddings.e.as_mut_slice() {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }

        // Hidden layer (w1 is now [INPUT_DIM][HIDDEN_DIM] = [130][64])
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

        // Shared trunk
        for row in &mut net.trunk_w {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.trunk_b {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

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

        Ok(net)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::sync::Arc;

    // ========================================================================
    // ExprNnue Tests
    // ========================================================================

    #[test]
    fn param_count_should_exceed_10k_and_memory_should_stay_under_200kb() {
        // Param count should include the whole backbone + value head.
        let count = ExprNnue::param_count();
        // Backbone (embeddings + w1 + b1 + trunk) ~13,808, plus expr_proj +
        // value_mlp ~3,089 = 16,897 live params (67,588 bytes) — see
        // `Self::save`'s doc for the split from the saturation head's
        // (separately-checkpointed) weights.
        assert!(count > 10_000, "Should have >10k params, got {}", count);
        assert!(
            ExprNnue::memory_bytes() < 200_000,
            "NNUE should use < 200KB, got {} bytes",
            ExprNnue::memory_bytes()
        );
    }

    // ========================================================================
    // Complex PE + Child-Index Encoding Tests
    // ========================================================================

    #[test]
    fn edge_accumulator_add_then_remove_should_return_to_zero() {
        let emb = OpEmbeddings::new_random(42);
        let mut acc = EdgeAccumulator::new();

        // Add several edges at various depths
        acc.add_edge(&emb, OpKind::Add, OpKind::Var, 0);
        acc.add_edge(&emb, OpKind::Mul, OpKind::Const, 3);
        acc.add_edge(&emb, OpKind::Div, OpKind::Sqrt, 10);
        acc.add_edge(&emb, OpKind::Sub, OpKind::Neg, 50);

        // Remove them in reverse order (shouldn't matter for additivity)
        acc.remove_edge(&emb, OpKind::Sub, OpKind::Neg, 50);
        acc.remove_edge(&emb, OpKind::Div, OpKind::Sqrt, 10);
        acc.remove_edge(&emb, OpKind::Mul, OpKind::Const, 3);
        acc.remove_edge(&emb, OpKind::Add, OpKind::Var, 0);

        for (i, &v) in acc.values.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "Complex PE roundtrip failed at index {i}: got {v}, expected ~0.0"
            );
        }
        assert_eq!(acc.edge_count, 0);
    }
}
