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
//! - Learned neural costs (`FactoredNnue` from pixelflow-nnue)
//! - Custom domain-specific cost models

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::node::ENode;
use pixelflow_ir::OpKind;

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
/// let nnue = FactoredNnue::load("model.bin")?;
/// let (tree, cost) = extract(&egraph, root, &nnue);
/// ```
pub trait CostFunction {
    /// Estimate the cost of a single ENode.
    ///
    /// This is the atomic operation cost, NOT including children.
    /// The extraction algorithm sums child costs separately.
    fn node_cost(&self, node: &ENode) -> usize;

    /// Get the cost of an operation by OpKind (optional, for interop).
    ///
    /// Default implementation panics - only needed for models that
    /// support direct OpKind lookup.
    fn cost_by_kind(&self, _op: OpKind) -> usize {
        panic!("CostFunction::cost_by_kind not implemented for this cost model");
    }
}

/// Cost model indexed by OpKind.
///
/// Uses `[usize; OpKind::COUNT]` internally for O(1) lookup.
/// Includes depth penalty for compile-time optimization.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Cost per operation, indexed by `OpKind::index()`.
    costs: [usize; OpKind::COUNT],
    /// Depth at which to start applying penalties.
    pub depth_threshold: usize,
    /// Penalty per depth level beyond threshold.
    pub depth_penalty: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl CostModel {
    /// Create with default costs (reasonable x86-64 SIMD estimates).
    pub fn with_defaults() -> Self {
        let mut costs = [4usize; OpKind::COUNT]; // Default to medium cost

        // Leaves (no cost)
        costs[OpKind::Var.index()] = 0;
        costs[OpKind::Const.index()] = 0;
        costs[OpKind::Tuple.index()] = 0;

        // Cheap (1 cycle)
        costs[OpKind::Neg.index()] = 1;
        costs[OpKind::Abs.index()] = 1;

        // Medium (4-5 cycles)
        costs[OpKind::Add.index()] = 4;
        costs[OpKind::Sub.index()] = 4;
        costs[OpKind::Mul.index()] = 5;
        costs[OpKind::Min.index()] = 4;
        costs[OpKind::Max.index()] = 4;

        // Expensive (12-15 cycles)
        costs[OpKind::Div.index()] = 15;
        costs[OpKind::Sqrt.index()] = 15;
        costs[OpKind::Recip.index()] = 5;
        costs[OpKind::Rsqrt.index()] = 5;

        // Fused (same as components or cheaper)
        costs[OpKind::MulAdd.index()] = 5;
        costs[OpKind::MulRsqrt.index()] = 6;

        // Rounding (cheap)
        costs[OpKind::Floor.index()] = 2;
        costs[OpKind::Ceil.index()] = 2;
        costs[OpKind::Round.index()] = 2;
        costs[OpKind::Fract.index()] = 3;

        // Transcendentals (expensive)
        costs[OpKind::Sin.index()] = 20;
        costs[OpKind::Cos.index()] = 20;
        costs[OpKind::Tan.index()] = 25;
        costs[OpKind::Asin.index()] = 25;
        costs[OpKind::Acos.index()] = 25;
        costs[OpKind::Atan.index()] = 20;
        costs[OpKind::Atan2.index()] = 25;
        costs[OpKind::Exp.index()] = 15;
        costs[OpKind::Exp2.index()] = 12;
        costs[OpKind::Ln.index()] = 15;
        costs[OpKind::Log2.index()] = 12;
        costs[OpKind::Log10.index()] = 15;
        costs[OpKind::Pow.index()] = 30;
        costs[OpKind::Hypot.index()] = 20;

        // Comparison (cheap)
        costs[OpKind::Lt.index()] = 1;
        costs[OpKind::Le.index()] = 1;
        costs[OpKind::Gt.index()] = 1;
        costs[OpKind::Ge.index()] = 1;
        costs[OpKind::Eq.index()] = 1;
        costs[OpKind::Ne.index()] = 1;

        // Control flow
        costs[OpKind::Select.index()] = 2;
        costs[OpKind::Clamp.index()] = 4;

        Self {
            costs,
            depth_threshold: 32,
            depth_penalty: 100,
        }
    }

    /// Create with FMA optimization enabled.
    pub fn with_fma() -> Self {
        let mut model = Self::with_defaults();
        model.costs[OpKind::MulAdd.index()] = 5;
        model
    }

    /// Create with fast reciprocal/rsqrt.
    pub fn with_fast_rsqrt() -> Self {
        let mut model = Self::with_defaults();
        model.costs[OpKind::Rsqrt.index()] = 4;
        model.costs[OpKind::Recip.index()] = 4;
        model
    }

    /// Create with all optimizations.
    pub fn fully_optimized() -> Self {
        let mut model = Self::with_defaults();
        model.costs[OpKind::MulAdd.index()] = 5;
        model.costs[OpKind::Rsqrt.index()] = 4;
        model.costs[OpKind::Recip.index()] = 4;
        model
    }

    /// Create with custom depth threshold.
    pub fn with_depth_limit(threshold: usize, penalty: usize) -> Self {
        Self {
            depth_threshold: threshold,
            depth_penalty: penalty,
            ..Self::with_defaults()
        }
    }

    /// Create with aggressive depth penalty for complex kernels.
    pub fn shallow() -> Self {
        Self {
            depth_threshold: 16,
            depth_penalty: 500,
            ..Self::fully_optimized()
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get cost for an OpKind.
    #[inline]
    pub fn cost(&self, op: OpKind) -> usize {
        self.costs[op.index()]
    }

    /// Set cost for an OpKind.
    #[inline]
    pub fn set_cost(&mut self, op: OpKind, cost: usize) {
        self.costs[op.index()] = cost;
    }

    /// Get the raw costs array.
    pub fn costs(&self) -> &[usize; OpKind::COUNT] {
        &self.costs
    }

    /// Get mutable reference to costs array.
    pub fn costs_mut(&mut self) -> &mut [usize; OpKind::COUNT] {
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
    pub fn node_op_cost(&self, node: &ENode) -> usize {
        match node {
            ENode::Var(_) | ENode::Const(_) => 0,
            ENode::Op { op, .. } => self.cost_by_name(op.name()),
        }
    }

    /// Get cost by operation name (for backward compatibility).
    pub fn cost_by_name(&self, name: &str) -> usize {
        // Map name to OpKind
        let op = match name {
            "var" => OpKind::Var,
            "const" => OpKind::Const,
            "add" => OpKind::Add,
            "sub" => OpKind::Sub,
            "mul" => OpKind::Mul,
            "div" => OpKind::Div,
            "neg" => OpKind::Neg,
            "sqrt" => OpKind::Sqrt,
            "rsqrt" => OpKind::Rsqrt,
            "abs" => OpKind::Abs,
            "min" => OpKind::Min,
            "max" => OpKind::Max,
            "mul_add" => OpKind::MulAdd,
            "mul_rsqrt" => OpKind::MulRsqrt,
            "recip" => OpKind::Recip,
            "floor" => OpKind::Floor,
            "ceil" => OpKind::Ceil,
            "round" => OpKind::Round,
            "fract" => OpKind::Fract,
            "sin" => OpKind::Sin,
            "cos" => OpKind::Cos,
            "tan" => OpKind::Tan,
            "asin" => OpKind::Asin,
            "acos" => OpKind::Acos,
            "atan" => OpKind::Atan,
            "atan2" => OpKind::Atan2,
            "exp" => OpKind::Exp,
            "exp2" => OpKind::Exp2,
            "ln" => OpKind::Ln,
            "log2" => OpKind::Log2,
            "log10" => OpKind::Log10,
            "pow" => OpKind::Pow,
            "hypot" => OpKind::Hypot,
            "lt" => OpKind::Lt,
            "le" => OpKind::Le,
            "gt" => OpKind::Gt,
            "ge" => OpKind::Ge,
            "eq" => OpKind::Eq,
            "ne" => OpKind::Ne,
            "select" => OpKind::Select,
            "clamp" => OpKind::Clamp,
            "tuple" => OpKind::Tuple,
            _ => return self.costs[OpKind::Add.index()], // Default for unknown
        };
        self.costs[op.index()]
    }

    // =========================================================================
    // Persistence
    // =========================================================================

    /// Save cost model to a TOML file.
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut contents = String::from("# Learned cost model weights\n");
        contents.push_str("# Generated from SIMD benchmark measurements\n\n");

        for i in 0..OpKind::COUNT {
            if let Some(op) = OpKind::from_index(i) {
                contents.push_str(&format!("{} = {}\n", op.name(), self.costs[i]));
            }
        }

        contents.push_str(&format!("\ndepth_threshold = {}\n", self.depth_threshold));
        contents.push_str(&format!("depth_penalty = {}\n", self.depth_penalty));

        fs::write(path, contents)
    }

    /// Load cost model from a TOML file.
    pub fn load_toml<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        let mut model = Self::with_defaults();

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                if let Ok(v) = value.parse::<usize>() {
                    if key == "depth_threshold" {
                        model.depth_threshold = v;
                    } else if key == "depth_penalty" {
                        model.depth_penalty = v;
                    } else {
                        // Try to parse as OpKind name
                        for i in 0..OpKind::COUNT {
                    if let Some(op) = OpKind::from_index(i)
                        && op.name() == key
                    {
                        model.costs[i] = v;
                        break;
                            }
                        }
                    }
                }
            }
        }

        Ok(model)
    }

    /// Try to load from a standard location, falling back to fully_optimized.
    pub fn load_or_default() -> Self {
        // Check environment variable first
        if let Ok(path) = std::env::var("PIXELFLOW_COST_MODEL")
            && let Ok(model) = Self::load_toml(&path)
        {
            return model;
        }

        // Try user config directory
        if let Some(home) = std::env::var_os("HOME") {
            let config_path = Path::new(&home).join(".config/pixelflow/cost_model.toml");
            if let Ok(model) = Self::load_toml(&config_path) {
                return model;
            }
        }

        // Try workspace data directory (for development)
        let workspace_paths = [
            "pixelflow-ml/data/learned_cost_model.toml",
            "../pixelflow-ml/data/learned_cost_model.toml",
        ];
        for path in workspace_paths {
            if let Ok(model) = Self::load_toml(path) {
                return model;
            }
        }

        // Default to hardcoded optimized settings
        Self::fully_optimized()
    }

    // =========================================================================
    // Interop
    // =========================================================================

    /// Create from HashMap (for backward compatibility).
    pub fn from_map(costs: &HashMap<String, usize>) -> Self {
        let mut model = Self::with_defaults();
        for (key, &value) in costs {
            if key == "depth_threshold" {
                model.depth_threshold = value;
            } else if key == "depth_penalty" {
                model.depth_penalty = value;
            } else {
                for i in 0..OpKind::COUNT {
                    if let Some(op) = OpKind::from_index(i)
                        && op.name() == key
                    {
                        model.costs[i] = value;
                        break;
                    }
                }
            }
        }
        model
    }

    /// Convert to HashMap for interop.
    pub fn to_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        for i in 0..OpKind::COUNT {
            if let Some(op) = OpKind::from_index(i) {
                map.insert(op.name().to_string(), self.costs[i]);
            }
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
    fn node_cost(&self, node: &ENode) -> usize {
        self.node_op_cost(node)
    }

    fn cost_by_kind(&self, op: OpKind) -> usize {
        self.cost(op)
    }
}
