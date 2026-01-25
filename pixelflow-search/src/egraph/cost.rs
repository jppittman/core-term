//! Cost model for e-graph extraction.
//!
//! The cost model controls which equivalent expression the e-graph extracts.
//! It includes:
//! - **Operation costs**: How expensive each op is at runtime
//! - **Depth penalty**: Hinge penalty for type nesting beyond threshold
//!
//! The depth penalty prevents compilation blowup by making deep type trees
//! expensive, encouraging the extractor to prefer shallower forms or boxing.

use std::collections::HashMap;
use super::node::ENode;

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
        if let Some(&c) = costs.get("add") { model.add = c; }
        if let Some(&c) = costs.get("sub") { model.sub = c; }
        if let Some(&c) = costs.get("mul") { model.mul = c; }
        if let Some(&c) = costs.get("div") { model.div = c; }
        if let Some(&c) = costs.get("neg") { model.neg = c; }
        if let Some(&c) = costs.get("recip") { model.recip = c; }
        if let Some(&c) = costs.get("sqrt") { model.sqrt = c; }
        if let Some(&c) = costs.get("rsqrt") { model.rsqrt = c; }
        if let Some(&c) = costs.get("abs") { model.abs = c; }
        if let Some(&c) = costs.get("min") { model.min = c; }
        if let Some(&c) = costs.get("max") { model.max = c; }
        if let Some(&c) = costs.get("mul_add") { model.mul_add = c; }
        if let Some(&c) = costs.get("depth_threshold") { model.depth_threshold = c; }
        if let Some(&c) = costs.get("depth_penalty") { model.depth_penalty = c; }
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

    pub fn node_op_cost(&self, node: &ENode) -> usize {
        match node {
            ENode::Var(_) | ENode::Const(_) => 0,
            ENode::Add(_, _) => self.add,
            ENode::Sub(_, _) => self.sub,
            ENode::Mul(_, _) => self.mul,
            ENode::Div(_, _) => self.div,
            ENode::Neg(_) => self.neg,
            ENode::Recip(_) => self.recip,
            ENode::Sqrt(_) => self.sqrt,
            ENode::Rsqrt(_) => self.rsqrt,
            ENode::Abs(_) => self.abs,
            ENode::Min(_, _) => self.min,
            ENode::Max(_, _) => self.max,
            ENode::MulAdd(_, _, _) => self.mul_add,
            ENode::Select(_, _, _) | ENode::Clamp(_, _, _) => self.add,
            ENode::Tuple(_) => 0,
            _ => self.add, // Default for functions like sin, cos, etc.
        }
    }
}
