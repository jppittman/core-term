//! Cost model for e-graph extraction.
//!
//! The cost model controls which equivalent expression the e-graph extracts.
//! It uses `OpKind` from `pixelflow-ir` as the canonical operation enumeration.
//!
//! # Architecture
//!
//! The module provides two levels of abstraction:
//!
//! 1. **`CostFunction` trait**: Pluggable interface for any cost estimator
//! 2. **`CostModel` struct**: Hardcoded O(1) lookup table based on OpKind
//!
//! This allows the e-graph extraction to use either:
//! - Fast hardcoded costs (`CostModel`)
//! - Learned neural costs (`ExprNnue` from pixelflow-nnue)
//! - Custom domain-specific cost models

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::node::ENode;
use pixelflow_ir::OpKind;
use pixelflow_ir::kind::OpMap;

// ============================================================================
// Latency Prior — single source of truth
// ============================================================================

/// Handcrafted per-op cycle-latency estimates, indexed by `OpKind::index()`.
///
/// This is the ONE place these numbers are allowed to live. Both the static
/// [`CostModel`] (via [`CostModel::latency_prior`]) and the NNUE's embedding
/// initialization (`nnue::factored::OpEmbeddings::init_with_latency_prior`)
/// derive their costs from this table so the two representations cannot
/// drift apart. If you're tempted to hand-tune a number in one place,
/// change it here instead.
///
/// Handcrafted cycle estimates, one per op.
///
/// Written as an exhaustive `match` rather than a positional per-op array.
/// The array form aligned to the discriminants only
/// by convention, nothing checked the convention, and it drifted: while the
/// discriminants had gaps the table was written densely, so 13 of 50 ops read
/// their neighbour's cycle count. `Dwrt` came back 10 instead of 1000 — cheap
/// enough for extraction to pick an unlowered derivative, which is exactly what
/// the 1000 exists to prevent — and `Shr`, which `expand_log2` emits, came back
/// 1000 instead of 1.
///
/// A `match` cannot drift: adding an op is a compile error until it is priced.
///
/// # Measured basis (2026-08-10, Apple M2 Max, aarch64 NEON JIT)
///
/// The non-trivial entries below are **measured, not guessed**: serial chains
/// of K=8 vs K=32 applications of each op, JIT-compiled through
/// `compile_arena_dag` (so transcendentals are timed in their
/// `expand_transcendentals` lowered form — the form this table is actually
/// pricing) and timed under `BenchMode::Latency`; the per-stage slope cancels
/// call overhead exactly. Units are normalized so `Add = 4` (the table's
/// historical unit; measured FADD slope 0.87ns/stage ≈ 3 real cycles at
/// ~3.4GHz). Two independent runs agreed within 3% on every corrected entry.
/// Protocol: `pixelflow-pipeline/examples/measure_latency_prior.rs`.
///
/// The headline correction: `Pow` was priced 12 — cheaper than a hardware
/// `Sqrt` at 15 — but `expand_transcendentals` lowers `Pow(a,b)` to
/// `exp2(b·log2 a)`, two bit-manipulation polynomial kernels. Measured: 196
/// (and the measurement is internally consistent: Log2 121 + Mul 5 + Exp2 69
/// ≈ 196, likewise Ln ≈ Log2+Mul, Asin ≈ Atan2+Sqrt+Mul+Sub). With the old
/// table, extraction preferred `Pow(x, 0.5)` over `Sqrt(x)` — a measured
/// 2.8x kernel slowdown shipped by the DEFAULT cost model.
///
/// Also corrected by the same measurement: `Recip` (10 → 16) and `Rsqrt`
/// (5 → 21) are *slower* serially than the hardware `Div` (11) and `Sqrt`
/// (15) they approximate on this backend — the NEON lowering is
/// estimate + Newton steps, a serial chain, not a cheap single instruction.
///
/// Caveat: measured on one machine (aarch64 NEON). The lowered subgraphs are
/// the same shape on x86, so the ordering is expected to transfer, but the
/// exact ratios are host-specific — see `cargo xtask isa-matrix`.
#[must_use]
pub fn latency_prior_cycles() -> OpMap<usize> {
    OpMap::from_fn(|op| match op {
        OpKind::Var => 0,     // free
        OpKind::Const => 0,   // free
        OpKind::Add => 4,     // measured 4.0 (anchor)
        OpKind::Sub => 4,     // measured 4.0
        OpKind::Mul => 5,     // measured 5.3
        OpKind::Div => 11,    // measured 11.3 (was 15)
        OpKind::Neg => 3,     // measured 3.1 (was 1)
        OpKind::Sqrt => 15,   // measured 14.5
        OpKind::Rsqrt => 21,  // measured 21.5 — estimate + NR chain (was 5)
        OpKind::Abs => 3,     // measured 3.1 (was 1)
        OpKind::Min => 3,     // measured 2.7 (was 4)
        OpKind::Max => 3,     // measured 2.7 (was 4)
        OpKind::MulAdd => 5,  // fused; measured 5.5, kept at Mul parity
        OpKind::Recip => 16,  // measured 16.0 — estimate + NR chain (was 10)
        OpKind::Floor => 4,   // measured ~4
        OpKind::Ceil => 4,    // measured 4.3
        OpKind::Round => 4,   // measured 4.3
        OpKind::Sin => 70,    // measured 70.6 (was 10)
        OpKind::Cos => 75,    // measured 74.8 (was 10)
        OpKind::Tan => 87,    // measured 86.8 (was 10)
        OpKind::Asin => 103,  // measured 102.8 (was 10)
        OpKind::Acos => 103,  // measured 102.9 (was 10)
        OpKind::Atan => 79,   // measured 79.3 (was 10)
        OpKind::Exp => 75,    // measured 74.7 (was 10)
        OpKind::Exp2 => 69,   // measured 69.1 (was 10)
        OpKind::Ln => 128,    // measured 127.6 (was 10)
        OpKind::Log2 => 122,  // measured 122.0 (was 10)
        OpKind::Log10 => 134, // measured 133.6 (was 10)
        OpKind::Atan2 => 79,  // measured 78.8 (was 10)
        OpKind::Pow => 196,   // measured 196.2 ≈ Log2 + Mul + Exp2 (was 12)
        OpKind::Lt => 3,
        OpKind::Le => 3,
        OpKind::Gt => 3,
        OpKind::Ge => 3,
        OpKind::Eq => 3,
        OpKind::Ne => 3,
        OpKind::Select => 4,
        OpKind::Tuple => 0,      // free (structural)
        OpKind::TruncToInt => 1, // cvttps2dq
        OpKind::IntToFloat => 1, // cvtdq2ps
        OpKind::IAdd => 1,       // paddd
        OpKind::Shl => 1,
        OpKind::Shr => 1,
        OpKind::BitAnd => 1,
        OpKind::BitOr => 1,
        OpKind::Dwrt => 1000,
        OpKind::Buffer => 0,     // leaf, free
        OpKind::Gather => 10,    // memory read
        OpKind::RawGather => 10, // primitive memory read
        OpKind::Reduce => 0,     // lowered (unrolled) before costing
    })
}

// ============================================================================
// Cost Function Trait
// ============================================================================

/// Trait for pluggable cost functions in e-graph extraction.
///
/// Implementors provide a cost estimate for ENodes, enabling different
/// cost models (hardcoded, learned neural, domain-specific) to be used
/// interchangeably during extraction.
///
/// # Contract
///
/// - `node_cost` returns a cost in arbitrary units (lower is better)
/// - Leaves (Var, Const) should typically return 0
/// - Costs should be consistent: same input → same output
///
/// # Example
///
/// ```ignore
/// // Using the hardcoded cost model
/// let costs = CostModel::default();
/// let (tree, cost) = extract(&egraph, root, &costs);
///
/// // Using a learned neural cost model
/// let nnue = ExprNnue::load("model.bin")?;
/// let (tree, cost) = extract(&egraph, root, &nnue);
/// ```
pub trait CostFunction {
    /// Estimate the cost of a single ENode given its parent context.
    ///
    /// This is the atomic operation cost, NOT including children.
    /// The extraction algorithm sums child costs separately.
    ///
    /// `parent` is the OpKind of the operation using this result.
    /// This allows for 'sliding window' optimizations (e.g. FMA detection).
    fn node_cost(&self, node: &ENode, parent: Option<OpKind>) -> usize;

    /// Get the cost of an operation by OpKind (optional, for interop).
    fn cost_by_kind(&self, op: OpKind, parent: Option<OpKind>) -> usize {
        panic!("CostFunction::cost_by_kind not implemented");
    }
}

/// Cost model indexed by OpKind.
///
/// Uses [`OpMap`] internally for O(1) lookup.
/// Includes depth penalty for compile-time optimization.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Cost per operation.
    costs: OpMap<usize>,
    /// Depth at which to start applying penalties.
    pub depth_threshold: usize,
    /// Penalty per depth level beyond threshold.
    pub depth_penalty: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CostModel {
    /// Create a cost model seeded with the handcrafted latency-prior cycle
    /// table ([`latency_prior_cycles`]).
    ///
    /// This is the default: an all-zero cost model makes every expression
    /// "free" and extraction degenerates to an arbitrary tie-break, so
    /// zero-cost is never a useful baseline for real extraction. Use
    /// [`CostModel::zero`] explicitly if you actually want all-zero costs
    /// (e.g. to test structural properties independent of cost).
    pub fn new() -> Self {
        Self::latency_prior()
    }

    /// Create a cost model from the handcrafted latency-prior cycle table.
    ///
    /// Source of truth: [`latency_prior_cycles`], shared with
    /// `nnue::factored::OpEmbeddings::init_with_latency_prior` so the static
    /// and learned cost models cannot drift apart.
    pub fn latency_prior() -> Self {
        Self {
            costs: latency_prior_cycles(),
            depth_threshold: 1024, // Effectively disabled
            depth_penalty: 0,
        }
    }

    /// Create an all-zero cost model.
    ///
    /// Every expression costs nothing, so extraction can't distinguish
    /// equivalent forms on cost alone. Only useful for tests that check
    /// structural extraction behavior (DAG sharing, cycle handling, etc.)
    /// independent of any particular cost table.
    pub fn zero() -> Self {
        Self {
            costs: OpMap::splat(0),
            depth_threshold: 1024, // Effectively disabled
            depth_penalty: 0,
        }
    }

    /// Create with aggressive depth penalty for complex kernels.
    pub fn shallow() -> Self {
        Self {
            depth_threshold: 16,
            depth_penalty: 500,
            ..Self::new()
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get cost for an OpKind.
    #[inline]
    pub fn cost(&self, op: OpKind) -> usize {
        self.costs[op]
    }

    /// Set cost for an OpKind.
    #[inline]
    pub fn set_cost(&mut self, op: OpKind, cost: usize) {
        self.costs[op] = cost;
    }

    /// Get the raw costs array.
    pub fn costs(&self) -> &OpMap<usize> {
        &self.costs
    }

    /// Get mutable reference to costs array.
    pub fn costs_mut(&mut self) -> &mut OpMap<usize> {
        &mut self.costs
    }

    /// Calculate the hinge penalty for a given depth.
    #[inline]
    pub fn depth_cost(&self, depth: usize) -> usize {
        if depth > self.depth_threshold {
            (depth - self.depth_threshold) * self.depth_penalty
        } else {
            0
        }
    }

    /// Get cost for an ENode.
    ///
    /// Uses `op.kind()` to convert at the boundary from `&dyn Op` to `OpKind`.
    pub fn node_op_cost(&self, node: &ENode) -> usize {
        match node {
            // Buffer is a leaf like Var/Const: the cost of the read lives on
            // the Gather that consumes it.
            ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => 0,
            // `Dwrt` is the internal autodiff marker. It is rewritten away by
            // the chain rule; a surviving one is the (not-yet-wired) jet
            // fallback. Either way extraction must never choose it, so it is
            // prohibitively expensive regardless of the learned weight table.
            ENode::Op { op, .. } if op.kind() == OpKind::Dwrt => usize::MAX / 4,
            ENode::Op { op, .. } => self.cost(op.kind()),
        }
    }

    /// Get cost by operation name (for backward compatibility).
    ///
    /// # Panics
    ///
    /// Panics if `name` does not match a known `OpKind`. Silently mapping
    /// an unrecognized name to `Add`'s cost would let typos and stale
    /// callers pass through with a wrong-but-plausible number — fail loud
    /// instead.
    pub fn cost_by_name(&self, name: &str) -> usize {
        let op = OpKind::from_name(name)
            .unwrap_or_else(|| panic!("CostModel::cost_by_name: unknown op name {name:?}"));
        self.cost(op)
    }

    // =========================================================================
    // Persistence
    // =========================================================================

    /// Save cost model to a TOML file.
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut contents = String::from("# Learned cost model weights\n");
        contents.push_str("# Generated from SIMD benchmark measurements\n\n");

        for (op, cost) in self.costs.iter() {
            contents.push_str(&format!("{} = {}\n", op.name(), cost));
        }

        contents.push_str(&format!("\ndepth_threshold = {}\n", self.depth_threshold));
        contents.push_str(&format!("depth_penalty = {}\n", self.depth_penalty));

        fs::write(path, contents)
    }

    /// Load cost model from a TOML file.
    ///
    /// Starts from [`CostModel::zero`] (not [`CostModel::latency_prior`]) so
    /// the returned model reflects exactly what's in the file — mixing in
    /// the latency prior for keys the file doesn't mention would silently
    /// blend two cost sources together.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be read, or if a `key = value`
    /// line has a value that fails to parse as `usize`. A malformed line is
    /// a real misconfiguration; silently skipping it would let a typo'd
    /// weight file produce a model that looks valid but isn't.
    pub fn load_toml<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        let mut model = Self::zero();

        for (lineno, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let Some((key, value)) = line.split_once('=') else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("cost model TOML line {}: missing '=': {line:?}", lineno + 1),
                ));
            };
            let key = key.trim();
            let value = value.trim();
            let v = value.parse::<usize>().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "cost model TOML line {}: value {value:?} for key {key:?} is not a usize: {e}",
                        lineno + 1
                    ),
                )
            })?;

            if key == "depth_threshold" {
                model.depth_threshold = v;
            } else if key == "depth_penalty" {
                model.depth_penalty = v;
            } else if let Some(op) = OpKind::from_name(key) {
                // O(1) lookup via OpKind::from_name
                model.costs[op] = v;
            } else {
                // Unrecognized op name: this is an external, evolving file
                // format (older/newer OpKind sets), so we don't hard-fail —
                // but we don't stay silent either.
                eprintln!(
                    "warning: cost model TOML line {}: unknown key {key:?}, ignoring",
                    lineno + 1
                );
            }
        }

        Ok(model)
    }

    /// Try to load from a standard location, falling back to the latency
    /// prior if none is found.
    ///
    /// Each candidate location is tried in order. A location that simply
    /// doesn't exist is expected (most of these are optional overrides) and
    /// is skipped quietly; a location that exists but fails to *parse* is a
    /// real misconfiguration and is reported loudly (`eprintln!`) before
    /// moving on, so a typo'd cost file never fails silently into a
    /// different cost model without a trace.
    pub fn load_or_default() -> Self {
        // Check environment variable first. If the user explicitly set
        // PIXELFLOW_COST_MODEL, a missing/unparsable file is always loud —
        // they asked for this specific file.
        if let Ok(path) = std::env::var("PIXELFLOW_COST_MODEL") {
            match Self::load_toml(&path) {
                Ok(model) => return model,
                Err(e) => eprintln!(
                    "warning: PIXELFLOW_COST_MODEL={path:?} failed to load ({e}); falling back"
                ),
            }
        }

        // Try user config directory.
        if let Some(home) = std::env::var_os("HOME") {
            let config_path = Path::new(&home).join(".config/pixelflow/cost_model.toml");
            match Self::load_toml(&config_path) {
                Ok(model) => return model,
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    eprintln!(
                        "warning: cost model {config_path:?} exists but failed to load ({e}); falling back"
                    );
                }
                Err(_) => {} // not found: expected, this is an optional override
            }
        }

        // Try workspace data directory (for development).
        let workspace_paths = [
            "pixelflow-ml/data/learned_cost_model.toml",
            "../pixelflow-ml/data/learned_cost_model.toml",
        ];
        for path in workspace_paths {
            match Self::load_toml(path) {
                Ok(model) => return model,
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    eprintln!(
                        "warning: cost model {path:?} exists but failed to load ({e}); falling back"
                    );
                }
                Err(_) => {} // not found: expected, this is an optional override
            }
        }

        // No override found anywhere: use the handcrafted latency prior.
        // This is a loud, intentional default — NOT the old all-zero
        // fallback, which made every op "free" and was useless for
        // extraction. If you need a genuinely all-zero model, use
        // `CostModel::zero()` explicitly.
        Self::latency_prior()
    }

    // =========================================================================
    // Interop
    // =========================================================================

    /// Create from HashMap (for backward compatibility).
    ///
    /// Starts from [`CostModel::zero`], not [`CostModel::latency_prior`] —
    /// same reasoning as [`CostModel::load_toml`]: the caller handed us an
    /// explicit map, so the result should reflect exactly that map, not a
    /// blend with the handcrafted prior.
    pub fn from_map(costs: &HashMap<String, usize>) -> Self {
        let mut model = Self::zero();
        for (key, &value) in costs {
            if key == "depth_threshold" {
                model.depth_threshold = value;
            } else if key == "depth_penalty" {
                model.depth_penalty = value;
            } else if let Some(op) = OpKind::from_name(key) {
                // O(1) lookup via OpKind::from_name
                model.costs[op] = value;
            }
            // Unknown keys are silently ignored (external data format)
        }
        model
    }

    /// Convert to HashMap for interop.
    pub fn to_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        for (op, cost) in self.costs.iter() {
            map.insert(op.name().to_string(), *cost);
        }
        map.insert("depth_threshold".to_string(), self.depth_threshold);
        map.insert("depth_penalty".to_string(), self.depth_penalty);
        map
    }
}

// ============================================================================
// CostFunction Implementation for CostModel
// ============================================================================

impl CostFunction for CostModel {
    fn node_cost(&self, node: &ENode, _parent: Option<OpKind>) -> usize {
        self.node_op_cost(node)
    }

    fn cost_by_kind(&self, op: OpKind, _parent: Option<OpKind>) -> usize {
        self.cost(op)
    }
}

#[cfg(test)]
mod every_op_is_priceable {
    use super::{CostModel, latency_prior_cycles};
    use pixelflow_ir::kind::{OpKind, OpMap};

    /// `cost`/`set_cost` subscript a positional per-op array with
    /// `OpKind::index()`. That is only total while the discriminants are dense.
    ///
    /// They were not, until 2026-08-02: `Gather`/`RawGather`/`Reduce` sat past
    /// three gaps and indexed 50..=52 into a 50-slot array. Nothing caught it
    /// because `arena_to_egraph` refuses `Buffer` leaves before extraction
    /// runs, and every `Gather` has one — so the panic was reachable only by
    /// pricing an op the front end happened never to hand over. Pricing the
    /// whole enum here does not depend on that accident holding.
    /// The prices that are load-bearing rather than advisory.
    ///
    /// `Dwrt` is priced prohibitively so extraction never selects an unlowered
    /// derivative — `arena_to_schedule` panics on one that reaches codegen, and
    /// that panic is supposed to be unreachable. When the table was positional
    /// and `index()` was sparse, `Dwrt` came back 10, which is cheaper than
    /// `Div`. `Shr` got Dwrt's 1000 in the same shift, which taught the
    /// optimizer to avoid the shifts `expand_log2` is built from.
    #[test]
    fn the_prohibitive_prices_are_actually_prohibitive() {
        let cycles = latency_prior_cycles();

        assert_eq!(cycles[OpKind::Dwrt], 1000, "Dwrt must stay unselectable");
        assert!(
            cycles[OpKind::Dwrt] > 100 * cycles[OpKind::Mul],
            "Dwrt at {} is not prohibitive next to Mul at {}",
            cycles[OpKind::Dwrt],
            cycles[OpKind::Mul]
        );

        // The bit-manipulation atoms exp/log lower into are single
        // instructions and must be priced like it.
        for op in [
            OpKind::Shl,
            OpKind::Shr,
            OpKind::BitAnd,
            OpKind::BitOr,
            OpKind::IAdd,
            OpKind::TruncToInt,
            OpKind::IntToFloat,
        ] {
            assert_eq!(cycles[op], 1, "{op:?} is a single instruction");
        }
    }

    /// Pricing is total: every op has a cost, and asking for one cannot fail.
    ///
    /// `Gather`/`RawGather`/`Reduce` are named outright rather than left to
    /// the walk over `all()`. They sit at the end of the table, which is where
    /// an op goes missing from a table-filling loop without anyone noticing,
    /// so naming them tests the walk as much as the pricing.
    #[test]
    fn every_op_can_be_priced() {
        let mut model = CostModel::latency_prior();

        for op in [OpKind::Gather, OpKind::RawGather, OpKind::Reduce] {
            let _ = model.cost(op);
            model.set_cost(op, 1);
        }

        for op in OpKind::all() {
            let _ = model.cost(op);
            model.set_cost(op, 1);
        }
    }

    /// `CostModel::zero()` exists specifically so extraction tests can check
    /// structural behavior (DAG sharing, cycle handling) independent of any
    /// cost table — see its doc comment. That property only holds if every
    /// op is actually priced at 0; if `zero()` ever returned the
    /// `latency_prior` table instead (an easy mix-up, since both build a
    /// `CostModel { costs, .. }` with the same shape), every extraction test
    /// built on "cost is irrelevant here" would start silently depending on
    /// real cycle counts instead.
    #[test]
    fn zero_prices_every_op_at_zero_cost() {
        let model = CostModel::zero();
        for op in OpKind::all() {
            assert_eq!(
                model.cost(op),
                0,
                "{op:?} is not zero-priced under CostModel::zero()"
            );
        }
    }

    /// `cost_by_name` used to hand-roll its own `&str -> OpKind` table,
    /// separate from `OpKind::from_name`, and that table's whitelist simply
    /// stopped at `tuple` — every op past it (memory/lattice/bit-manip ops
    /// included) was unreachable by name even though `cost`/`set_cost`
    /// already handled it fine by value. Delegating to `OpKind::from_name`
    /// means `cost_by_name` covers exactly what `cost` covers, nothing more
    /// or less — check that against a sample spanning the enum, not just the
    /// ops the old table happened to list.
    #[test]
    fn cost_by_name_matches_cost_for_every_sampled_op() {
        let model = CostModel::latency_prior();
        for name in [
            "sin",
            "log10",
            "pow",
            "lt",
            "select",
            "tuple",
            "dwrt",
            "buffer",
            "gather",
            "raw_gather",
            "reduce",
            "iadd",
            "shl",
            "trunc_to_int",
        ] {
            let op = OpKind::from_name(name).unwrap_or_else(|| panic!("unknown op name {name:?}"));
            assert_eq!(
                model.cost_by_name(name),
                model.cost(op),
                "cost_by_name({name:?}) should match cost(OpKind::{op:?})"
            );
        }
    }
}

#[cfg(test)]
mod cost_model_accessors {
    use super::{CostFunction, CostModel};
    use crate::egraph::ops::op_from_kind;
    use crate::egraph::{EGraph, ENode};
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity};

    /// `set_cost` followed by `cost` for the same op should observe the
    /// value just written, and must not disturb any other op's price —
    /// otherwise `set_cost`/`cost` could be indexing different slots
    /// without any test noticing.
    #[test]
    fn set_cost_then_cost_returns_the_value_just_set_without_disturbing_other_ops() {
        let mut model = CostModel::latency_prior();
        let mul_before = model.cost(OpKind::Mul);

        model.set_cost(OpKind::Add, 777);

        assert_eq!(model.cost(OpKind::Add), 777);
        assert_eq!(
            model.cost(OpKind::Mul),
            mul_before,
            "set_cost(Add, _) must not leak into Mul's price"
        );
    }

    /// `costs()` is a read-only view onto the same table `cost()` reads
    /// from, seeded by the latency prior — not a stub or a freshly
    /// allocated all-zero map.
    #[test]
    fn costs_accessor_reflects_the_latency_prior_table() {
        let model = CostModel::latency_prior();
        assert_eq!(model.costs()[OpKind::Sin], model.cost(OpKind::Sin));
        assert_eq!(model.costs()[OpKind::Add], model.cost(OpKind::Add));
    }

    /// `costs_mut()` must hand back a view into the model's own table:
    /// writing through it should be visible via `cost()` afterward.
    #[test]
    fn costs_mut_edits_are_visible_through_cost() {
        let mut model = CostModel::latency_prior();
        model.costs_mut()[OpKind::Div] = 321;
        assert_eq!(model.cost(OpKind::Div), 321);
    }

    /// `shallow()` keeps the latency-prior op costs (it only overrides the
    /// depth penalty fields), and its depth parameters must differ from
    /// `new()`'s effectively-disabled defaults or the two constructors
    /// would be indistinguishable.
    #[test]
    fn shallow_keeps_latency_prior_op_costs_but_tightens_the_depth_penalty() {
        let shallow = CostModel::shallow();
        let latency_prior = CostModel::latency_prior();

        assert_eq!(shallow.cost(OpKind::Sin), latency_prior.cost(OpKind::Sin));
        assert_eq!(shallow.depth_threshold, 16);
        assert_eq!(shallow.depth_penalty, 500);
        assert_ne!(shallow.depth_threshold, CostModel::new().depth_threshold);
    }

    /// At or below the threshold, depth carries no penalty at all.
    #[test]
    fn depth_cost_is_zero_at_and_below_the_threshold() {
        let model = CostModel::shallow(); // depth_threshold=16, depth_penalty=500
        assert_eq!(model.depth_cost(16), 0);
        assert_eq!(model.depth_cost(10), 0);
    }

    /// Past the threshold, the penalty scales linearly with how far past —
    /// picked so `+`/`-`/`*`/`/` substitutions on either operator each
    /// produce a different, wrong number instead of coincidentally
    /// agreeing with `(depth - threshold) * penalty`.
    #[test]
    fn depth_cost_charges_penalty_per_level_past_the_threshold() {
        let model = CostModel::shallow(); // depth_threshold=16, depth_penalty=500
        assert_eq!(model.depth_cost(18), 1000); // (18 - 16) * 500
    }

    /// `Var`/`Const`/`Buffer` are leaves; the cost of reading a `Buffer` is
    /// charged to the `Gather` that consumes it, so all three price at
    /// zero regardless of what the op-cost table says.
    ///
    /// `Buffer` is asserted alongside the other two rather than left to the
    /// shared match arm: runtime arenas do contain buffer leaves, so if that
    /// arm ever started taking the table price, the buffer *and* its
    /// consuming gather would both be charged and extraction could change
    /// its choices.
    #[test]
    fn node_op_cost_should_price_var_const_and_buffer_leaves_at_zero() {
        let model = CostModel::latency_prior();
        assert_eq!(model.node_op_cost(&ENode::Var(0)), 0);
        assert_eq!(model.node_op_cost(&ENode::constant(2.0)), 0);

        let decl = BufferDecl {
            id: BufferIdentity::mint(),
            width: 8,
            height: 4,
        };
        assert_eq!(model.node_op_cost(&ENode::Buffer(decl)), 0);
    }

    /// `Dwrt` is the unlowered-autodiff marker and must never look cheap to
    /// extraction, however the op-cost table happens to price it — so
    /// `node_op_cost` overrides the table for it specifically.
    #[test]
    fn node_op_cost_makes_a_dwrt_node_prohibitively_expensive() {
        let model = CostModel::latency_prior();
        let op = op_from_kind(OpKind::Dwrt).expect("Dwrt has an Op impl");
        let node = ENode::Op {
            op,
            children: vec![],
        };
        assert_eq!(model.node_op_cost(&node), usize::MAX / 4);
    }

    /// An ordinary op node (not Dwrt, not a leaf) prices straight from the
    /// op-cost table, not from the Dwrt override path.
    #[test]
    fn node_op_cost_prices_an_ordinary_op_node_from_the_cost_table() {
        let mut egraph = EGraph::new();
        let lhs = egraph.add(ENode::constant(1.0));
        let rhs = egraph.add(ENode::constant(2.0));
        let model = CostModel::latency_prior();
        let op = op_from_kind(OpKind::Add).expect("Add has an Op impl");
        let node = ENode::Op {
            op,
            children: vec![lhs, rhs],
        };
        assert_eq!(model.node_op_cost(&node), model.cost(OpKind::Add));
    }

    /// The `CostFunction` trait impl for `CostModel` is a thin delegation
    /// layer; `node_cost` must route to `node_op_cost` rather than return a
    /// constant. The node is an `Add` (cost 4) rather than a leaf (cost 0)
    /// so that a delegation which quietly returned zero is distinguishable
    /// from one that works.
    #[test]
    fn node_cost_trait_method_delegates_to_node_op_cost() {
        let mut egraph = EGraph::new();
        let lhs = egraph.add(ENode::constant(1.0));
        let rhs = egraph.add(ENode::constant(2.0));
        let op = op_from_kind(OpKind::Add).expect("Add has an Op impl");
        let node = ENode::Op {
            op,
            children: vec![lhs, rhs],
        };

        let model = CostModel::latency_prior();
        assert_eq!(
            CostFunction::node_cost(&model, &node, None),
            model.node_op_cost(&node)
        );
        assert_ne!(CostFunction::node_cost(&model, &node, None), 0);
    }

    /// `CostFunction::cost_by_kind` has a default body (`panic!`) for
    /// implementors that don't provide their own — documented as "not
    /// implemented" rather than silently returning a made-up number.
    /// `CostModel` overrides it (tested above); this exercises the trait's
    /// own default via a minimal second implementor.
    #[test]
    #[should_panic(expected = "not implemented")]
    fn cost_by_kind_default_trait_method_panics_when_not_overridden() {
        struct NoOverride;
        impl CostFunction for NoOverride {
            fn node_cost(&self, _node: &ENode, _parent: Option<OpKind>) -> usize {
                0
            }
        }

        let _ = CostFunction::cost_by_kind(&NoOverride, OpKind::Add, None);
    }

    /// Likewise `cost_by_kind` must route to `cost`, not return a constant.
    #[test]
    fn cost_by_kind_trait_method_delegates_to_cost() {
        let model = CostModel::latency_prior();
        assert_eq!(
            CostFunction::cost_by_kind(&model, OpKind::Log2, None),
            model.cost(OpKind::Log2)
        );
    }
}

#[cfg(test)]
mod persistence {
    use super::CostModel;
    use pixelflow_ir::OpKind;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Builds a path under the OS temp dir that's unique per call, so
    /// parallel test threads never collide on the same file.
    fn unique_temp_path(tag: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "pixelflow-cost-model-test-{tag}-{}-{n}.toml",
            std::process::id()
        ))
    }

    /// A model saved to TOML and loaded back must reproduce every op cost
    /// and both depth-penalty fields — not just return a freshly
    /// constructed default. The round trip also exercises the file's
    /// header comment lines and blank line, which must be skipped rather
    /// than rejected as malformed `key = value` lines.
    #[test]
    fn save_toml_then_load_toml_round_trips_costs_and_depth_fields() {
        let path = unique_temp_path("roundtrip");

        let mut original = CostModel::latency_prior();
        original.set_cost(OpKind::Add, 111);
        original.set_cost(OpKind::Sin, 222);
        original.depth_threshold = 9;
        original.depth_penalty = 13;

        original.save_toml(&path).expect("save_toml should succeed");
        let loaded = CostModel::load_toml(&path).expect("load_toml should succeed");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.cost(OpKind::Add), 111);
        assert_eq!(loaded.cost(OpKind::Sin), 222);
        assert_eq!(loaded.cost(OpKind::Mul), original.cost(OpKind::Mul));
        assert_eq!(loaded.depth_threshold, 9);
        assert_eq!(loaded.depth_penalty, 13);
    }

    /// A key the file never mentions is not silently filled in from the
    /// latency prior — `load_toml` starts from an all-zero model, so an
    /// omitted op stays at 0. Asserting the omitted op specifically is what
    /// separates that from a `load_toml` that returned a default model,
    /// where the omitted op would carry its latency-prior price instead.
    #[test]
    fn load_toml_leaves_unmentioned_ops_at_zero_rather_than_the_latency_prior() {
        let path = unique_temp_path("sparse");
        std::fs::write(&path, "add = 99\n").expect("write test fixture");

        let loaded = CostModel::load_toml(&path).expect("load_toml should succeed");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.cost(OpKind::Add), 99);
        assert_eq!(loaded.cost(OpKind::Mul), 0);
    }

    /// A line with no `=` is a malformed cost file and must be reported,
    /// not skipped or silently ignored.
    #[test]
    fn load_toml_rejects_a_line_with_no_equals_sign() {
        let path = unique_temp_path("no-equals");
        std::fs::write(&path, "this line has no equals sign\n").expect("write test fixture");

        let result = CostModel::load_toml(&path);
        let _ = std::fs::remove_file(&path);

        assert!(result.is_err(), "a line without '=' must be rejected");
    }

    /// A value that doesn't parse as `usize` (including a negative number)
    /// is a malformed cost file and must be reported, not silently
    /// defaulted to some other cost.
    #[test]
    fn load_toml_rejects_a_value_that_does_not_parse_as_usize() {
        let path = unique_temp_path("bad-value");
        std::fs::write(&path, "add = not_a_number\n").expect("write test fixture");

        let result = CostModel::load_toml(&path);
        let _ = std::fs::remove_file(&path);

        assert!(result.is_err(), "a non-usize value must be rejected");
    }

    /// `to_map` then `from_map` must round-trip both op costs and the two
    /// depth-penalty fields through the `HashMap<String, usize>` interop
    /// format — not just produce a fresh default model.
    #[test]
    fn to_map_then_from_map_round_trips_costs_and_depth_fields() {
        let mut original = CostModel::latency_prior();
        original.set_cost(OpKind::Exp, 55);
        original.depth_threshold = 3;
        original.depth_penalty = 7;

        let map: HashMap<String, usize> = original.to_map();
        let restored = CostModel::from_map(&map);

        assert_eq!(restored.cost(OpKind::Exp), 55);
        assert_eq!(restored.depth_threshold, 3);
        assert_eq!(restored.depth_penalty, 7);
    }

    /// The assertion half of
    /// [`load_or_default_should_return_the_model_named_by_the_env_var_override`],
    /// run in a child process that was spawned with `PIXELFLOW_COST_MODEL`
    /// already set. `#[ignore]` keeps it out of the ordinary sweep, where
    /// the variable is unset and there would be nothing to assert.
    #[test]
    #[ignore = "spawned by its parent test with PIXELFLOW_COST_MODEL set"]
    fn env_var_override_child() {
        let model = CostModel::load_or_default();

        assert_eq!(
            model.cost(OpKind::Add),
            42,
            "the env var names a file with `add = 42`; a default model would not have it"
        );
        assert_eq!(
            model.cost(OpKind::Mul),
            0,
            "load_toml starts from zero, not the latency prior"
        );
    }

    /// `load_or_default` must actually load and return the file named by
    /// `PIXELFLOW_COST_MODEL` when it's set — not a freshly constructed
    /// default model, which would happen to look identical for every op
    /// this test doesn't check.
    ///
    /// The override is exercised in a **child process** rather than by
    /// mutating this one's environment. `std::env::set_var` is unsafe
    /// because it requires that no other thread touch the environment
    /// concurrently, and the test harness runs tests on parallel threads —
    /// a mutex private to one test cannot establish that, and
    /// `load_or_default` itself goes on to read `HOME`. Spawning gives the
    /// child an environment nothing races on, so there is no `unsafe` here
    /// at all.
    #[test]
    fn load_or_default_should_return_the_model_named_by_the_env_var_override() {
        let path = unique_temp_path("env-override");
        std::fs::write(&path, "add = 42\n").expect("write test fixture");

        const CHILD: &str = "egraph::cost::persistence::env_var_override_child";

        let output = std::process::Command::new(
            std::env::current_exe().expect("locate this test binary to re-run a single test"),
        )
        .args(["--exact", CHILD, "--ignored"])
        .env("PIXELFLOW_COST_MODEL", &path)
        .output()
        .expect("spawn the child test process");

        let _ = std::fs::remove_file(&path);

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Checked before `status.success()`, and the reason this is an
        // `output()` rather than a `status()`: libtest exits 0 when a filter
        // matches nothing, so a `CHILD` path that drifts out of date would
        // make the success assertion below pass without ever running the
        // assertions it is standing in for.
        assert!(
            stdout.contains("1 passed"),
            "expected the child to run exactly one test; if `{CHILD}` no longer names it, \
             this test is asserting nothing. Child stdout:\n{stdout}"
        );
        assert!(
            output.status.success(),
            "{CHILD} failed under PIXELFLOW_COST_MODEL={}. Child stdout:\n{stdout}",
            path.display()
        );
    }
}
