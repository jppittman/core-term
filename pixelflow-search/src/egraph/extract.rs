//! Extraction: Pull out a concrete expression tree from an e-graph.

use alloc::vec::Vec;
use alloc::boxed::Box;
use super::cost::CostModel;
use super::graph::EGraph;
use super::node::{EClassId, ENode, Op};

/// A concrete expression tree extracted from an e-graph.
#[derive(Clone, Debug)]
pub enum ExprTree {
    Var(u8),
    Const(f32),
    Add(Box<ExprTree>, Box<ExprTree>),
    Sub(Box<ExprTree>, Box<ExprTree>),
    Mul(Box<ExprTree>, Box<ExprTree>),
    Div(Box<ExprTree>, Box<ExprTree>),
    Neg(Box<ExprTree>),
    Recip(Box<ExprTree>),
    Sqrt(Box<ExprTree>),
    Rsqrt(Box<ExprTree>),
    Abs(Box<ExprTree>),
    Min(Box<ExprTree>, Box<ExprTree>),
    Max(Box<ExprTree>, Box<ExprTree>),
    MulAdd(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Floor(Box<ExprTree>),
    Ceil(Box<ExprTree>),
    Round(Box<ExprTree>),
    Fract(Box<ExprTree>),
    Sin(Box<ExprTree>),
    Cos(Box<ExprTree>),
    Tan(Box<ExprTree>),
    Asin(Box<ExprTree>),
    Acos(Box<ExprTree>),
    Atan(Box<ExprTree>),
    Exp(Box<ExprTree>),
    Exp2(Box<ExprTree>),
    Ln(Box<ExprTree>),
    Log2(Box<ExprTree>),
    Log10(Box<ExprTree>),
    Atan2(Box<ExprTree>, Box<ExprTree>),
    Pow(Box<ExprTree>, Box<ExprTree>),
    Hypot(Box<ExprTree>, Box<ExprTree>),
    Lt(Box<ExprTree>, Box<ExprTree>),
    Le(Box<ExprTree>, Box<ExprTree>),
    Gt(Box<ExprTree>, Box<ExprTree>),
    Ge(Box<ExprTree>, Box<ExprTree>),
    Eq(Box<ExprTree>, Box<ExprTree>),
    Ne(Box<ExprTree>, Box<ExprTree>),
    Select(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Clamp(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Tuple(Vec<ExprTree>),
}

// Leaf enum not needed if integrated, but keeping for compatibility if referenced
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Leaf {
    Var(u8),
    Const(f32),
}

impl ExprTree {
    pub fn var(idx: u8) -> Self { Self::Var(idx) }
    pub fn constant(val: f32) -> Self { Self::Const(val) }

    pub fn node_count(&self) -> usize {
        match self {
            Self::Var(_) | Self::Const(_) => 1,
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) | Self::Div(a, b) |
            Self::Min(a, b) | Self::Max(a, b) | Self::Atan2(a, b) | Self::Pow(a, b) |
            Self::Hypot(a, b) | Self::Lt(a, b) | Self::Le(a, b) | Self::Gt(a, b) |
            Self::Ge(a, b) | Self::Eq(a, b) | Self::Ne(a, b) => 1 + a.node_count() + b.node_count(),
            Self::Neg(a) | Self::Recip(a) | Self::Sqrt(a) | Self::Rsqrt(a) |
            Self::Abs(a) | Self::Floor(a) | Self::Ceil(a) | Self::Round(a) |
            Self::Fract(a) | Self::Sin(a) | Self::Cos(a) | Self::Tan(a) |
            Self::Asin(a) | Self::Acos(a) | Self::Atan(a) | Self::Exp(a) |
            Self::Exp2(a) | Self::Ln(a) | Self::Log2(a) | Self::Log10(a) => 1 + a.node_count(),
            Self::MulAdd(a, b, c) | Self::Select(a, b, c) | Self::Clamp(a, b, c) => {
                1 + a.node_count() + b.node_count() + c.node_count()
            }
            Self::Tuple(elems) => 1 + elems.iter().map(|e| e.node_count()).sum::<usize>(),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Self::Var(_) | Self::Const(_) => 1,
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) | Self::Div(a, b) |
            Self::Min(a, b) | Self::Max(a, b) | Self::Atan2(a, b) | Self::Pow(a, b) |
            Self::Hypot(a, b) | Self::Lt(a, b) | Self::Le(a, b) | Self::Gt(a, b) |
            Self::Ge(a, b) | Self::Eq(a, b) | Self::Ne(a, b) => 1 + a.depth().max(b.depth()),
            Self::Neg(a) | Self::Recip(a) | Self::Sqrt(a) | Self::Rsqrt(a) |
            Self::Abs(a) | Self::Floor(a) | Self::Ceil(a) | Self::Round(a) |
            Self::Fract(a) | Self::Sin(a) | Self::Cos(a) | Self::Tan(a) |
            Self::Asin(a) | Self::Acos(a) | Self::Atan(a) | Self::Exp(a) |
            Self::Exp2(a) | Self::Ln(a) | Self::Log2(a) | Self::Log10(a) => 1 + a.depth(),
            Self::MulAdd(a, b, c) | Self::Select(a, b, c) | Self::Clamp(a, b, c) => {
                1 + a.depth().max(b.depth()).max(c.depth())
            }
            Self::Tuple(elems) => 1 + elems.iter().map(|e| e.depth()).max().unwrap_or(0),
        }
    }

    pub fn cost(&self, costs: &CostModel) -> usize {
        match self {
            Self::Var(_) | Self::Const(_) => 0,
            Self::Add(a, b) => costs.cost_by_name("add") + a.cost(costs) + b.cost(costs),
            Self::Sub(a, b) => costs.cost_by_name("sub") + a.cost(costs) + b.cost(costs),
            Self::Mul(a, b) => costs.cost_by_name("mul") + a.cost(costs) + b.cost(costs),
            Self::Div(a, b) => costs.cost_by_name("div") + a.cost(costs) + b.cost(costs),
            Self::Neg(a) => costs.cost_by_name("neg") + a.cost(costs),
            Self::MulAdd(a, b, c) => costs.cost_by_name("mul_add") + a.cost(costs) + b.cost(costs) + c.cost(costs),
            Self::Select(a, b, c) => costs.cost_by_name("select") + a.cost(costs) + b.cost(costs) + c.cost(costs),
            Self::Tuple(elems) => elems.iter().map(|e| e.cost(costs)).sum(),
            // Fallback for others (could be comprehensive, but this covers core)
            _ => 1 + match self {
                Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) | Self::Div(a, b) |
                Self::Min(a, b) | Self::Max(a, b) | Self::Atan2(a, b) | Self::Pow(a, b) |
                Self::Hypot(a, b) | Self::Lt(a, b) | Self::Le(a, b) | Self::Gt(a, b) |
                Self::Ge(a, b) | Self::Eq(a, b) | Self::Ne(a, b) => a.cost(costs) + b.cost(costs),
                Self::Neg(a) | Self::Recip(a) | Self::Sqrt(a) | Self::Rsqrt(a) |
                Self::Abs(a) | Self::Floor(a) | Self::Ceil(a) | Self::Round(a) |
                Self::Fract(a) | Self::Sin(a) | Self::Cos(a) | Self::Tan(a) |
                Self::Asin(a) | Self::Acos(a) | Self::Atan(a) | Self::Exp(a) |
                Self::Exp2(a) | Self::Ln(a) | Self::Log2(a) | Self::Log10(a) => a.cost(costs),
                Self::MulAdd(a, b, c) | Self::Select(a, b, c) | Self::Clamp(a, b, c) => {
                    a.cost(costs) + b.cost(costs) + c.cost(costs)
                }
                Self::Tuple(elems) => elems.iter().map(|e| e.cost(costs)).sum(),
                Self::Var(_) | Self::Const(_) => 0,
            }
        }
    }

    pub fn eval(&self, env: &[f32]) -> f32 {
        match self {
            Self::Var(idx) => env.get(*idx as usize).copied().unwrap_or(0.0),
            Self::Const(val) => *val,
            Self::Add(a, b) => a.eval(env) + b.eval(env),
            Self::Sub(a, b) => a.eval(env) - b.eval(env),
            Self::Mul(a, b) => a.eval(env) * b.eval(env),
            Self::Div(a, b) => a.eval(env) / b.eval(env),
            Self::Neg(a) => -a.eval(env),
            Self::Sqrt(a) => a.eval(env).sqrt(),
            Self::Rsqrt(a) => 1.0 / a.eval(env).sqrt(), // Approx
            Self::Recip(a) => 1.0 / a.eval(env),
            Self::Abs(a) => a.eval(env).abs(),
            Self::Min(a, b) => a.eval(env).min(b.eval(env)),
            Self::Max(a, b) => a.eval(env).max(b.eval(env)),
            Self::MulAdd(a, b, c) => a.eval(env).mul_add(b.eval(env), c.eval(env)),
            // ... implement others as needed for tests ...
            _ => 0.0,
        }
    }

    // Compatibility constructors for macros
    pub fn add(a: Self, b: Self) -> Self { Self::Add(Box::new(a), Box::new(b)) }
    pub fn sub(a: Self, b: Self) -> Self { Self::Sub(Box::new(a), Box::new(b)) }
    pub fn mul(a: Self, b: Self) -> Self { Self::Mul(Box::new(a), Box::new(b)) }
    pub fn div(a: Self, b: Self) -> Self { Self::Div(Box::new(a), Box::new(b)) }
    pub fn neg(a: Self) -> Self { Self::Neg(Box::new(a)) }
    pub fn sqrt(a: Self) -> Self { Self::Sqrt(Box::new(a)) }
    pub fn rsqrt(a: Self) -> Self { Self::Rsqrt(Box::new(a)) }
    pub fn recip(a: Self) -> Self { Self::Recip(Box::new(a)) }
    pub fn abs(a: Self) -> Self { Self::Abs(Box::new(a)) }
    pub fn min(a: Self, b: Self) -> Self { Self::Min(Box::new(a), Box::new(b)) }
    pub fn max(a: Self, b: Self) -> Self { Self::Max(Box::new(a), Box::new(b)) }
    pub fn mul_add(a: Self, b: Self, c: Self) -> Self { Self::MulAdd(Box::new(a), Box::new(b), Box::new(c)) }
}

/// Extract the minimum-cost expression tree from an e-class.
pub fn extract(egraph: &EGraph, root: EClassId, costs: &CostModel) -> (ExprTree, usize) {
    use alloc::collections::BTreeSet;

    let mut best_cost = alloc::vec![None; egraph.num_classes()];
    let mut best_node = alloc::vec![None; egraph.num_classes()];

    fn compute_best(
        egraph: &EGraph,
        class: EClassId,
        costs: &CostModel,
        best_cost: &mut Vec<Option<usize>>,
        best_node: &mut Vec<Option<usize>>,
        visiting: &mut BTreeSet<u32>,
    ) -> usize {
        let canonical = egraph.find(class);
        if let Some(cost) = best_cost[canonical.0 as usize] {
            return cost;
        }
        if !visiting.insert(canonical.0) {
            return usize::MAX / 2;
        }

        let nodes = egraph.nodes(canonical);
        let mut min_cost = usize::MAX;
        let mut min_idx = 0;

        for (idx, node) in nodes.iter().enumerate() {
            let node_cost = costs.node_op_cost(node); // Assuming this works for all variants

            // Add children costs
            let children_cost = node.children().iter().map(|&child| {
                compute_best(egraph, child, costs, best_cost, best_node, visiting)
            }).sum::<usize>();

            let total = node_cost.saturating_add(children_cost);

            if total < min_cost {
                min_cost = total;
                min_idx = idx;
            }
        }

        visiting.remove(&canonical.0);
        best_cost[canonical.0 as usize] = Some(min_cost);
        best_node[canonical.0 as usize] = Some(min_idx);
        min_cost
    }

    let mut visiting = BTreeSet::new();
    let total_cost = compute_best(egraph, root, costs, &mut best_cost, &mut best_node, &mut visiting);

    fn build_tree(
        egraph: &EGraph,
        class: EClassId,
        best_node: &[Option<usize>],
    ) -> ExprTree {
        let canonical = egraph.find(class);
        let node_idx = best_node[canonical.0 as usize].expect("best_node not computed");
        let node = &egraph.nodes(canonical)[node_idx];

        match node {
            ENode::Var(idx) => ExprTree::Var(*idx),
            ENode::Const(bits) => ExprTree::Const(f32::from_bits(*bits)),
            ENode::Add(a, b) => ExprTree::Add(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Sub(a, b) => ExprTree::Sub(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Mul(a, b) => ExprTree::Mul(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Div(a, b) => ExprTree::Div(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Neg(a) => ExprTree::Neg(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Recip(a) => ExprTree::Recip(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Sqrt(a) => ExprTree::Sqrt(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Rsqrt(a) => ExprTree::Rsqrt(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Abs(a) => ExprTree::Abs(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Min(a, b) => ExprTree::Min(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Max(a, b) => ExprTree::Max(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::MulAdd(a, b, c) => ExprTree::MulAdd(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node)), Box::new(build_tree(egraph, *c, best_node))),
            ENode::Floor(a) => ExprTree::Floor(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Ceil(a) => ExprTree::Ceil(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Round(a) => ExprTree::Round(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Fract(a) => ExprTree::Fract(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Sin(a) => ExprTree::Sin(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Cos(a) => ExprTree::Cos(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Tan(a) => ExprTree::Tan(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Asin(a) => ExprTree::Asin(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Acos(a) => ExprTree::Acos(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Atan(a) => ExprTree::Atan(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Exp(a) => ExprTree::Exp(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Exp2(a) => ExprTree::Exp2(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Ln(a) => ExprTree::Ln(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Log2(a) => ExprTree::Log2(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Log10(a) => ExprTree::Log10(Box::new(build_tree(egraph, *a, best_node))),
            ENode::Atan2(a, b) => ExprTree::Atan2(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Pow(a, b) => ExprTree::Pow(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Hypot(a, b) => ExprTree::Hypot(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Lt(a, b) => ExprTree::Lt(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Le(a, b) => ExprTree::Le(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Gt(a, b) => ExprTree::Gt(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Ge(a, b) => ExprTree::Ge(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Eq(a, b) => ExprTree::Eq(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Ne(a, b) => ExprTree::Ne(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node))),
            ENode::Select(a, b, c) => ExprTree::Select(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node)), Box::new(build_tree(egraph, *c, best_node))),
            ENode::Clamp(a, b, c) => ExprTree::Clamp(Box::new(build_tree(egraph, *a, best_node)), Box::new(build_tree(egraph, *b, best_node)), Box::new(build_tree(egraph, *c, best_node))),
            ENode::Tuple(elems) => ExprTree::Tuple(elems.iter().map(|e| build_tree(egraph, *e, best_node)).collect()),
        }
    }

    let tree = build_tree(egraph, root, &best_node);
    (tree, total_cost)
}
