//! Cost model for e-graph extraction.

use std::collections::HashMap;
use super::node::ENode;

/// Configurable cost model for operation costs.
#[derive(Clone, Debug)]
pub struct CostModel {
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
        model
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
