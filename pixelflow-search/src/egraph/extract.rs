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
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Add,
            children: alloc::vec![a, b],
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Sub,
            children: alloc::vec![a, b],
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn mul(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Mul,
            children: alloc::vec![a, b],
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn div(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Div,
            children: alloc::vec![a, b],
        }
    }

    #[allow(clippy::should_implement_trait)]
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

    // Cap for cycle/self-referential costs - high but not astronomical
    const CYCLE_COST: usize = 1_000_000;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Iterative bottom-up cost computation using topological order
    // We use a work stack to avoid recursion
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)]; // (class, children_processed)
    let mut on_stack: BTreeSet<u32> = BTreeSet::new();

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        // Already computed
        if best_cost[canonical.0 as usize].is_some() {
            continue;
        }

        if !children_done {
            // First visit: push self back (to process after children), then push children
            if !on_stack.insert(canonical.0) {
                // Cycle detected - don't cache, parent will handle with high cost
                continue;
            }

            stack.push((canonical, true)); // Come back after children

            // Push all children that need processing
            for node in egraph.nodes(canonical) {
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        let child_canonical = egraph.find(child);
                        if best_cost[child_canonical.0 as usize].is_none() {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            // Second visit: all children are computed, now compute this class
            on_stack.remove(&canonical.0);

            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                let node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) => costs.node_op_cost(node),
                    ENode::Op { op, children } => {
                        // Check for self-referential children
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.cost_by_name(op.name());
                            let children_cost: usize = children
                                .iter()
                                .map(|&child| {
                                    let c = egraph.find(child);
                                    best_cost[c.0 as usize].unwrap_or(CYCLE_COST)
                                })
                                .sum();
                            op_cost.saturating_add(children_cost)
                        }
                    }
                };

                if node_cost < min_cost {
                    min_cost = node_cost;
                    min_idx = idx;
                }
            }

            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);

    // Phase 2: Iterative top-down tree construction
    // Use a stack of (class, partially_built_tree_slot)
    enum BuildTask {
        Visit(EClassId),
        Complete { canonical: u32, op: &'static dyn super::ops::Op, num_children: usize },
    }

    let mut build_stack: Vec<BuildTask> = vec![BuildTask::Visit(root)];
    let mut result_stack: Vec<ExprTree> = Vec::new();
    let mut building: BTreeSet<u32> = BTreeSet::new();

    while let Some(task) = build_stack.pop() {
        match task {
            BuildTask::Visit(class) => {
                let canonical = egraph.find(class);

                // Cycle detection
                if !building.insert(canonical.0) {
                    result_stack.push(ExprTree::Leaf(Leaf::Const(0.0)));
                    continue;
                }

                let node_idx = best_node[canonical.0 as usize].unwrap_or(0);
                let node = &egraph.nodes(canonical)[node_idx];

                match node {
                    ENode::Var(idx) => {
                        building.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Var(*idx)));
                    }
                    ENode::Const(bits) => {
                        building.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))));
                    }
                    ENode::Op { op, children } => {
                        // Push completion task, then visit children in reverse order
                        build_stack.push(BuildTask::Complete {
                            canonical: canonical.0,
                            op: *op,
                            num_children: children.len(),
                        });
                        for &child in children.iter().rev() {
                            build_stack.push(BuildTask::Visit(child));
                        }
                    }
                }
            }
            BuildTask::Complete { canonical, op, num_children } => {
                building.remove(&canonical);
                // Pop children from result stack (they're in correct order now)
                let start = result_stack.len().saturating_sub(num_children);
                let child_trees: Vec<ExprTree> = result_stack.drain(start..).collect();
                result_stack.push(ExprTree::Op { op, children: child_trees });
            }
        }
    }

    let tree = result_stack.pop().unwrap_or(ExprTree::Leaf(Leaf::Const(0.0)));
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
