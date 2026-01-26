//! Extraction: Pull out a concrete expression tree from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model.

use alloc::vec::Vec;
use super::cost::CostModel;
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Op;

/// A concrete expression tree extracted from an e-graph.
///
/// Unlike ENode (which uses EClassId children), this has direct child ownership.
/// This is the output of extraction - a single concrete expression from the e-class.
#[derive(Clone, Debug)]
pub enum ExprTree {
    /// Leaf nodes
    Leaf(Leaf),
    /// Operation with child subtrees
    Op {
        op: &'static dyn Op,
        children: Vec<ExprTree>,
    },
}

/// Leaf nodes in an expression tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Leaf {
    /// Variable with index (0=X, 1=Y, 2=Z, 3=W, etc.)
    Var(u8),
    /// Constant value
    Const(f32),
}

impl ExprTree {
    /// Create a variable.
    pub fn var(idx: u8) -> Self {
        Self::Leaf(Leaf::Var(idx))
    }

    /// Create a constant.
    pub fn constant(val: f32) -> Self {
        Self::Leaf(Leaf::Const(val))
    }

    /// Count total nodes in the tree.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Op { children, .. } => 1 + children.iter().map(|c| c.node_count()).sum::<usize>(),
        }
    }

    /// Compute depth of the tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Op { children, .. } => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    // Constructor helpers for common operations
    pub fn add(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Add,
            children: alloc::vec![a, b],
        }
    }

    pub fn sub(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Sub,
            children: alloc::vec![a, b],
        }
    }

    pub fn mul(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Mul,
            children: alloc::vec![a, b],
        }
    }

    pub fn div(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Div,
            children: alloc::vec![a, b],
        }
    }

    pub fn neg(a: Self) -> Self {
        Self::Op {
            op: &super::ops::Neg,
            children: alloc::vec![a],
        }
    }

    pub fn sqrt(a: Self) -> Self {
        Self::Op {
            op: &super::ops::Sqrt,
            children: alloc::vec![a],
        }
    }

    pub fn abs(a: Self) -> Self {
        Self::Op {
            op: &super::ops::Abs,
            children: alloc::vec![a],
        }
    }

    pub fn min(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Min,
            children: alloc::vec![a, b],
        }
    }

    pub fn max(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Max,
            children: alloc::vec![a, b],
        }
    }

    pub fn mul_add(a: Self, b: Self, c: Self) -> Self {
        Self::Op {
            op: &super::ops::MulAdd,
            children: alloc::vec![a, b, c],
        }
    }

    /// Compute the cost of this expression tree using the given cost model.
    pub fn cost(&self, costs: &CostModel) -> usize {
        match self {
            Self::Leaf(_) => 0,  // Variables and constants are free
            Self::Op { op, children } => {
                let op_cost = costs.cost_by_name(op.name());
                let children_cost: usize = children.iter().map(|c| c.cost(costs)).sum();
                op_cost + children_cost
            }
        }
    }

    /// Evaluate the expression tree with given variable values.
    ///
    /// env[0] = X, env[1] = Y, env[2] = Z, env[3] = W
    pub fn eval(&self, env: &[f32]) -> f32 {
        match self {
            Self::Leaf(Leaf::Var(idx)) => env.get(*idx as usize).copied().unwrap_or(0.0),
            Self::Leaf(Leaf::Const(val)) => *val,
            Self::Op { op, children } => {
                let name = op.name();
                match name {
                    "add" => {
                        children.iter().map(|c| c.eval(env)).sum()
                    }
                    "sub" => {
                        if children.len() == 2 {
                            children[0].eval(env) - children[1].eval(env)
                        } else {
                            0.0
                        }
                    }
                    "mul" => {
                        children.iter().map(|c| c.eval(env)).product()
                    }
                    "div" => {
                        if children.len() == 2 {
                            children[0].eval(env) / children[1].eval(env)
                        } else {
                            0.0
                        }
                    }
                    "neg" => {
                        if children.len() == 1 {
                            -children[0].eval(env)
                        } else {
                            0.0
                        }
                    }
                    "sqrt" => {
                        if children.len() == 1 {
                            children[0].eval(env).sqrt()
                        } else {
                            0.0
                        }
                    }
                    "abs" => {
                        if children.len() == 1 {
                            children[0].eval(env).abs()
                        } else {
                            0.0
                        }
                    }
                    "recip" => {
                        if children.len() == 1 {
                            1.0 / children[0].eval(env)
                        } else {
                            0.0
                        }
                    }
                    "rsqrt" => {
                        if children.len() == 1 {
                            1.0 / children[0].eval(env).sqrt()
                        } else {
                            0.0
                        }
                    }
                    "min" => {
                        children.iter().map(|c| c.eval(env)).fold(f32::INFINITY, f32::min)
                    }
                    "max" => {
                        children.iter().map(|c| c.eval(env)).fold(f32::NEG_INFINITY, f32::max)
                    }
                    "mul_add" => {
                        if children.len() == 3 {
                            children[0].eval(env).mul_add(children[1].eval(env), children[2].eval(env))
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        // Unknown operation, return 0
                        0.0
                    }
                }
            }
        }
    }
}

/// Extract the minimum-cost expression tree from an e-class.
///
/// Uses dynamic programming: cost(class) = min over all nodes in class.
pub fn extract(egraph: &EGraph, root: EClassId, costs: &CostModel) -> (ExprTree, usize) {
    use alloc::collections::BTreeSet;

    let mut best_cost = alloc::vec![None; egraph.num_classes()];
    let mut best_node = alloc::vec![None; egraph.num_classes()];

    // Bottom-up DP: compute best cost for each class
    fn compute_best(
        egraph: &EGraph,
        class: EClassId,
        costs: &CostModel,
        best_cost: &mut Vec<Option<usize>>,
        best_node: &mut Vec<Option<usize>>,
        visiting: &mut BTreeSet<u32>,  // cycle detection
    ) -> usize {
        let canonical = egraph.find(class);

        if let Some(cost) = best_cost[canonical.0 as usize] {
            return cost;
        }

        // Cycle detection: if we're already visiting this class, return high cost
        if !visiting.insert(canonical.0) {
            return usize::MAX / 2;  // Return high cost to avoid this path
        }

        let nodes = egraph.nodes(canonical);
        let mut min_cost = usize::MAX;
        let mut min_idx = 0;

        for (idx, node) in nodes.iter().enumerate() {
            let node_cost = match node {
                ENode::Var(_) | ENode::Const(_) => costs.node_op_cost(node), // Use cost model
                ENode::Op { op, children } => {
                    // Check for self-referential children (created by identity rules)
                    if children.iter().any(|&c| egraph.find(c) == canonical) {
                        usize::MAX / 2  // Skip self-referential nodes
                    } else {
                        let op_cost = costs.cost_by_name(op.name());
                        let children_cost: usize = children
                            .iter()
                            .map(|&child| compute_best(egraph, child, costs, best_cost, best_node, visiting))
                            .sum();
                        op_cost + children_cost
                    }
                }
            };

            if node_cost < min_cost {
                min_cost = node_cost;
                min_idx = idx;
            }
        }

        visiting.remove(&canonical.0);
        best_cost[canonical.0 as usize] = Some(min_cost);
        best_node[canonical.0 as usize] = Some(min_idx);
        min_cost
    }

    // Compute costs bottom-up
    let mut visiting = BTreeSet::new();
    let total_cost = compute_best(egraph, root, costs, &mut best_cost, &mut best_node, &mut visiting);

    // Top-down: build the tree
    fn build_tree(
        egraph: &EGraph,
        class: EClassId,
        best_node: &[Option<usize>],
    ) -> ExprTree {
        let canonical = egraph.find(class);
        let node_idx = best_node[canonical.0 as usize].expect("best_node not computed");
        let node = &egraph.nodes(canonical)[node_idx];

        match node {
            ENode::Var(idx) => ExprTree::Leaf(Leaf::Var(*idx)),
            ENode::Const(bits) => ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))),
            ENode::Op { op, children } => {
                let child_trees = children
                    .iter()
                    .map(|&child| build_tree(egraph, child, best_node))
                    .collect();
                ExprTree::Op {
                    op: *op,
                    children: child_trees,
                }
            }
        }
    }

    let tree = build_tree(egraph, root, &best_node);
    (tree, total_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_tree_node_count() {
        let x = ExprTree::var(0);
        assert_eq!(x.node_count(), 1);

        let sum = ExprTree::add(ExprTree::var(0), ExprTree::var(1));
        assert_eq!(sum.node_count(), 3); // Add + X + Y
    }

    #[test]
    fn test_expr_tree_depth() {
        let x = ExprTree::var(0);
        assert_eq!(x.depth(), 1);

        let sum = ExprTree::add(ExprTree::var(0), ExprTree::var(1));
        assert_eq!(sum.depth(), 2);

        // (X + Y) * Z
        let nested = ExprTree::mul(sum, ExprTree::var(2));
        assert_eq!(nested.depth(), 3);
    }

    #[test]
    fn test_extract_simple() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));

        let costs = CostModel::default();
        let (tree, cost) = extract(&egraph, x, &costs);

        assert!(matches!(tree, ExprTree::Leaf(Leaf::Var(0))));
        assert_eq!(cost, 0); // Leaf nodes (Var/Const) have cost 0
    }

    #[test]
    fn test_extract_with_ops() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let costs = CostModel::default();
        let (tree, _cost) = extract(&egraph, sum, &costs);

        assert!(matches!(tree, ExprTree::Op { .. }));
        assert_eq!(tree.node_count(), 3); // Add + X + Y
    }
}
