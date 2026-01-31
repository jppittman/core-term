//! Cost model for e-graph extraction.
//!
//! The cost model controls which equivalent expression the e-graph extracts.
//! It includes:
//! - **Operation costs**: How expensive each op is at runtime
//! - **Depth penalty**: Hinge penalty for type nesting beyond threshold
//!
//! The depth penalty prevents compilation blowup by making deep type trees
//! expensive, encouraging the extractor to prefer shallower forms or boxing.

use super::node::ENode;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Configurable cost model for operation costs and depth penalties.
///
/// # Depth Penalty (Hinge Function)
///
/// When expression depth exceeds `depth_threshold`, a penalty is added:
/// ```text
/// penalty = max(0, depth - depth_threshold) * depth_penalty
/// ```
///
/// This encourages the e-graph to extract shallower expressions when possible,
/// preventing exponential type blowup during compilation.
#[derive(Clone, Debug)]
pub struct CostModel {
    // === Operation costs (runtime) ===
    pub add: usize,
    pub sub: usize,
    pub mul: usize,
    pub div: usize,
    pub neg: usize,
    pub sqrt: usize,
    pub recip: usize,
    pub rsqrt: usize,
    pub abs: usize,
    pub min: usize,
    pub max: usize,
    pub mul_add: usize,

    // === Depth penalty (compile time) ===
    /// Depth threshold before penalty kicks in.
    /// Default: 32 (reasonable for most expressions)
    pub depth_threshold: usize,

    /// Penalty per level beyond threshold (hinge slope).
    /// Default: 100 (makes deep trees very expensive)
    pub depth_penalty: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            add: 4,
            sub: 4,
            mul: 5,
            div: 15,
            neg: 1,
            sqrt: 15,
            recip: 5,
            rsqrt: 5,
            abs: 1,
            min: 4,
            max: 4,
            mul_add: 10,
            // Depth penalty defaults
            depth_threshold: 32,
            depth_penalty: 100,
        }
    }
}

impl CostModel {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn with_fma() -> Self {
        Self {
            mul_add: 5,
            ..Self::default()
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn with_fast_rsqrt() -> Self {
        Self {
            rsqrt: 4,
            recip: 4,
            ..Self::default()
        }
    }

    pub fn fully_optimized() -> Self {
        Self {
            mul_add: 5,
            recip: 4,
            rsqrt: 4,
            ..Self::default()
        }
    }

    /// Create a cost model with custom depth threshold.
    pub fn with_depth_limit(threshold: usize, penalty: usize) -> Self {
        Self {
            depth_threshold: threshold,
            depth_penalty: penalty,
            ..Self::default()
        }
    }

    /// Create a cost model that aggressively penalizes depth.
    /// Useful for complex kernels that would otherwise OOM the compiler.
    pub fn shallow() -> Self {
        Self {
            depth_threshold: 16,
            depth_penalty: 500,
            ..Self::fully_optimized()
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_map(costs: &HashMap<String, usize>) -> Self {
        let mut model = Self::default();
        if let Some(&c) = costs.get("add") {
            model.add = c;
        }
        if let Some(&c) = costs.get("sub") {
            model.sub = c;
        }
        if let Some(&c) = costs.get("mul") {
            model.mul = c;
        }
        if let Some(&c) = costs.get("div") {
            model.div = c;
        }
        if let Some(&c) = costs.get("neg") {
            model.neg = c;
        }
        if let Some(&c) = costs.get("recip") {
            model.recip = c;
        }
        if let Some(&c) = costs.get("sqrt") {
            model.sqrt = c;
        }
        if let Some(&c) = costs.get("rsqrt") {
            model.rsqrt = c;
        }
        if let Some(&c) = costs.get("abs") {
            model.abs = c;
        }
        if let Some(&c) = costs.get("min") {
            model.min = c;
        }
        if let Some(&c) = costs.get("max") {
            model.max = c;
        }
        if let Some(&c) = costs.get("mul_add") {
            model.mul_add = c;
        }
        if let Some(&c) = costs.get("depth_threshold") {
            model.depth_threshold = c;
        }
        if let Some(&c) = costs.get("depth_penalty") {
            model.depth_penalty = c;
        }
        model
    }

    /// Calculate the hinge penalty for a given depth.
    ///
    /// Returns 0 if depth <= threshold, otherwise (depth - threshold) * penalty.
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

    /// Get cost by operation name.
    pub fn cost_by_name(&self, name: &str) -> usize {
        match name {
            "add" => self.add,
            "sub" => self.sub,
            "mul" => self.mul,
            "div" => self.div,
            "neg" => self.neg,
            "recip" => self.recip,
            "sqrt" => self.sqrt,
            "rsqrt" => self.rsqrt,
            "abs" => self.abs,
            "min" => self.min,
            "max" => self.max,
            "mul_add" => self.mul_add,
            "select" | "clamp" => self.add,
            "tuple" => 0,
            _ => self.add, // Default for functions like sin, cos, etc.
        }
    }

    /// Save cost model to a TOML file.
    ///
    /// File format:
    /// ```toml
    /// # Learned cost model from SIMD benchmarks
    /// add = 4
    /// sub = 4
    /// mul = 5
    /// ...
    /// ```
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let contents = format!(
            r#"# Learned cost model weights
# Generated from SIMD benchmark measurements

# Operation costs (relative to fastest operation)
add = {}
sub = {}
mul = {}
div = {}
neg = {}
sqrt = {}
recip = {}
rsqrt = {}
abs = {}
min = {}
max = {}
mul_add = {}

# Depth penalty (compile-time optimization)
depth_threshold = {}
depth_penalty = {}
"#,
            self.add,
            self.sub,
            self.mul,
            self.div,
            self.neg,
            self.sqrt,
            self.recip,
            self.rsqrt,
            self.abs,
            self.min,
            self.max,
            self.mul_add,
            self.depth_threshold,
            self.depth_penalty,
        );
        fs::write(path, contents)
    }

    /// Load cost model from a TOML file.
    ///
    /// Falls back to defaults for missing fields.
    pub fn load_toml<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        let mut model = Self::default();

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                if let Ok(v) = value.parse::<usize>() {
                    match key {
                        "add" => model.add = v,
                        "sub" => model.sub = v,
                        "mul" => model.mul = v,
                        "div" => model.div = v,
                        "neg" => model.neg = v,
                        "sqrt" => model.sqrt = v,
                        "recip" => model.recip = v,
                        "rsqrt" => model.rsqrt = v,
                        "abs" => model.abs = v,
                        "min" => model.min = v,
                        "max" => model.max = v,
                        "mul_add" => model.mul_add = v,
                        "depth_threshold" => model.depth_threshold = v,
                        "depth_penalty" => model.depth_penalty = v,
                        _ => {} // Ignore unknown keys
                    }
                }
            }
        }

        Ok(model)
    }

    /// Try to load from a standard location, falling back to fully_optimized.
    ///
    /// Checks in order:
    /// 1. `PIXELFLOW_COST_MODEL` environment variable
    /// 2. `~/.config/pixelflow/cost_model.toml`
    /// 3. `<workspace>/pixelflow-ml/data/learned_cost_model.toml`
    /// 4. Falls back to `fully_optimized()`
    pub fn load_or_default() -> Self {
        // Check environment variable first
        if let Ok(path) = std::env::var("PIXELFLOW_COST_MODEL")
            && let Ok(model) = Self::load_toml(&path) {
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

    /// Convert to HashMap for interop with other systems.
    pub fn to_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        map.insert("add".to_string(), self.add);
        map.insert("sub".to_string(), self.sub);
        map.insert("mul".to_string(), self.mul);
        map.insert("div".to_string(), self.div);
        map.insert("neg".to_string(), self.neg);
        map.insert("sqrt".to_string(), self.sqrt);
        map.insert("recip".to_string(), self.recip);
        map.insert("rsqrt".to_string(), self.rsqrt);
        map.insert("abs".to_string(), self.abs);
        map.insert("min".to_string(), self.min);
        map.insert("max".to_string(), self.max);
        map.insert("mul_add".to_string(), self.mul_add);
        map.insert("depth_threshold".to_string(), self.depth_threshold);
        map.insert("depth_penalty".to_string(), self.depth_penalty);
        map
    }
}
