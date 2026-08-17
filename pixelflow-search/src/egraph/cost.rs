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
#[must_use]
pub fn latency_prior_cycles() -> OpMap<usize> {
    OpMap::from_fn(|op| match op {
        OpKind::Var => 0,   // free
        OpKind::Const => 0, // free
        OpKind::Add => 4,
        OpKind::Sub => 4,
        OpKind::Mul => 5,
        OpKind::Div => 15,
        OpKind::Neg => 1,
        OpKind::Sqrt => 15,
        OpKind::Rsqrt => 5, // fast approximation
        OpKind::Abs => 1,
        OpKind::Min => 4,
        OpKind::Max => 4,
        OpKind::MulAdd => 5, // fused
        OpKind::Recip => 10,
        OpKind::Floor => 4,
        OpKind::Ceil => 4,
        OpKind::Round => 4,
        OpKind::Sin => 10,
        OpKind::Cos => 10,
        OpKind::Tan => 10,
        OpKind::Asin => 10,
        OpKind::Acos => 10,
        OpKind::Atan => 10,
        OpKind::Exp => 10,
        OpKind::Exp2 => 10,
        OpKind::Ln => 10,
        OpKind::Log2 => 10,
        OpKind::Log10 => 10,
        OpKind::Atan2 => 10,
        OpKind::Pow => 12,
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
