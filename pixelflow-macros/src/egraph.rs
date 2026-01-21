//! # E-Graph: Equality Saturation for Expression Optimization
//!
//! A dependency-free implementation of e-graphs for algebraic simplification.
//!
//! E-graphs (equivalence graphs) compactly represent many equivalent expressions
//! simultaneously. By applying rewrite rules to saturation, we discover all
//! algebraically equivalent forms and extract the optimal one.
//!
//! ## Key Concepts
//!
//! - **E-class**: An equivalence class of expressions that are all equal
//! - **E-node**: An expression node where children point to e-classes (not nodes)
//! - **Union-find**: Efficiently tracks which e-classes have been merged
//! - **Saturation**: Apply rules until no new equivalences are discovered
//!
//! ## Example
//!
//! ```ignore
//! use pixelflow_macros::egraph::{EGraph, ENode};
//!
//! let mut egraph = EGraph::new();
//!
//! // Build: (x + 0) - x
//! let x = egraph.add(ENode::Var(0));
//! let zero = egraph.add(ENode::Const(0.0));
//! let x_plus_0 = egraph.add(ENode::Add(x, zero));
//! let result = egraph.add(ENode::Sub(x_plus_0, x));
//!
//! // Saturate with rewrite rules
//! egraph.saturate();
//!
//! // Extract simplest form: should be 0
//! let simplified = egraph.extract(result);
//! ```
//!
//! ## References
//!
//! - Willsey et al., "egg: Fast and Extensible Equality Saturation" (POPL 2021)

use std::collections::HashMap;

/// Identifier for an equivalence class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EClassId(u32);

impl EClassId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// An expression node in the e-graph.
///
/// Children reference e-classes, not specific nodes. This is what allows
/// an e-graph to represent exponentially many equivalent expressions compactly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ENode {
    /// Variable reference by index (0 = X, 1 = Y, 2 = Z, 3 = W)
    Var(u8),
    /// Floating-point constant (stored as bits for hashing/comparison)
    Const(u32),
    /// Addition: left + right
    Add(EClassId, EClassId),
    /// Subtraction: left - right
    Sub(EClassId, EClassId),
    /// Multiplication: left * right
    Mul(EClassId, EClassId),
    /// Division: left / right
    Div(EClassId, EClassId),
    /// Negation: -inner
    Neg(EClassId),
    /// Square root
    Sqrt(EClassId),
    /// Reciprocal square root (fast approximation)
    Rsqrt(EClassId),
    /// Absolute value
    Abs(EClassId),
    /// Minimum of two values
    Min(EClassId, EClassId),
    /// Maximum of two values
    Max(EClassId, EClassId),
    /// Fused multiply-add: a * b + c
    MulAdd(EClassId, EClassId, EClassId),
}

impl ENode {
    /// Create a constant node from f32.
    pub fn constant(val: f32) -> Self {
        ENode::Const(val.to_bits())
    }

    /// Extract f32 value from a Const node.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ENode::Const(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    /// Check if this is a constant with the given value.
    pub fn is_const(&self, val: f32) -> bool {
        self.as_f32() == Some(val)
    }

    /// Update children to their canonical representatives.
    fn canonicalize(&mut self, egraph: &EGraph) {
        match self {
            ENode::Var(_) | ENode::Const(_) => {}
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b) => {
                *a = egraph.find(*a);
                *b = egraph.find(*b);
            }
            ENode::Neg(a) | ENode::Sqrt(a) | ENode::Rsqrt(a) | ENode::Abs(a) => {
                *a = egraph.find(*a);
            }
            ENode::MulAdd(a, b, c) => {
                *a = egraph.find(*a);
                *b = egraph.find(*b);
                *c = egraph.find(*c);
            }
        }
    }

    /// Get all child e-class IDs.
    fn children(&self) -> Vec<EClassId> {
        match self {
            ENode::Var(_) | ENode::Const(_) => vec![],
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b) => vec![*a, *b],
            ENode::Neg(a) | ENode::Sqrt(a) | ENode::Rsqrt(a) | ENode::Abs(a) => vec![*a],
            ENode::MulAdd(a, b, c) => vec![*a, *b, *c],
        }
    }
}

/// An equivalence class containing multiple equivalent e-nodes.
#[derive(Clone, Debug, Default)]
struct EClass {
    /// All e-nodes that are equivalent (belong to this class).
    nodes: Vec<ENode>,
}

/// The e-graph: a data structure for equality saturation.
pub struct EGraph {
    /// Storage for all e-classes.
    classes: Vec<EClass>,
    /// Union-find parent pointers. `parent[i]` points to parent of class i.
    parent: Vec<EClassId>,
    /// Memo table for hash-consing: canonical node -> e-class.
    memo: HashMap<ENode, EClassId>,
    /// Worklist for propagating merges.
    worklist: Vec<EClassId>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    /// Create a new empty e-graph.
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::new(),
            worklist: Vec::new(),
        }
    }

    /// Find the canonical representative of an e-class (with path compression).
    pub fn find(&self, id: EClassId) -> EClassId {
        let mut current = id;
        while self.parent[current.index()] != current {
            current = self.parent[current.index()];
        }
        current
    }

    /// Find with path compression (mutable version).
    fn find_mut(&mut self, id: EClassId) -> EClassId {
        let mut current = id;
        let mut path = Vec::new();

        // Find root
        while self.parent[current.index()] != current {
            path.push(current);
            current = self.parent[current.index()];
        }

        // Path compression
        for node in path {
            self.parent[node.index()] = current;
        }

        current
    }

    /// Add an e-node to the graph, returning its e-class ID.
    ///
    /// If an equivalent node already exists (hash-consing), returns the
    /// existing e-class. Otherwise creates a new e-class.
    pub fn add(&mut self, mut node: ENode) -> EClassId {
        // Canonicalize children
        node.canonicalize(self);

        // Check memo for existing equivalent
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }

        // Create new e-class
        let id = EClassId(self.classes.len() as u32);
        self.classes.push(EClass {
            nodes: vec![node.clone()],
        });
        self.parent.push(id);
        self.memo.insert(node, id);

        id
    }

    /// Merge two e-classes, asserting they are equivalent.
    ///
    /// Returns the canonical ID of the merged class.
    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let a = self.find_mut(a);
        let b = self.find_mut(b);

        if a == b {
            return a;
        }

        // Union by rank (smaller id becomes child)
        let (parent, child) = if a.0 < b.0 { (a, b) } else { (b, a) };

        self.parent[child.index()] = parent;

        // Merge nodes from child into parent
        let child_nodes = std::mem::take(&mut self.classes[child.index()].nodes);
        self.classes[parent.index()].nodes.extend(child_nodes);

        // Add to worklist for re-canonicalization
        self.worklist.push(parent);

        parent
    }

    /// Rebuild the e-graph after unions (re-canonicalize memo).
    fn rebuild(&mut self) {
        while let Some(id) = self.worklist.pop() {
            let id = self.find(id);

            // Re-canonicalize all nodes in this class
            let nodes = std::mem::take(&mut self.classes[id.index()].nodes);
            let mut new_nodes = Vec::new();

            for mut node in nodes {
                node.canonicalize(self);

                // Check if this canonical form already exists elsewhere
                if let Some(&existing) = self.memo.get(&node) {
                    let existing = self.find(existing);
                    if existing != id {
                        // Need to merge
                        self.union(id, existing);
                    }
                } else {
                    self.memo.insert(node.clone(), id);
                }

                new_nodes.push(node);
            }

            self.classes[id.index()].nodes = new_nodes;
        }
    }

    /// Get all nodes in an e-class.
    pub fn nodes(&self, id: EClassId) -> &[ENode] {
        let id = self.find(id);
        &self.classes[id.index()].nodes
    }

    /// Check if an e-class contains a specific constant.
    pub fn contains_const(&self, id: EClassId, val: f32) -> bool {
        self.nodes(id).iter().any(|n| n.is_const(val))
    }

    /// Apply all rewrite rules once, returning number of new equivalences found.
    fn apply_rules(&mut self) -> usize {
        let mut unions = 0;
        let num_classes = self.classes.len();

        for class_idx in 0..num_classes {
            let class_id = EClassId(class_idx as u32);
            let class_id = self.find(class_id);

            // Clone nodes to avoid borrow issues
            let nodes: Vec<ENode> = self.classes[class_id.index()].nodes.clone();

            for node in &nodes {
                // Apply each rewrite rule
                if let Some(new_id) = self.apply_rule(class_id, node) {
                    if self.find(class_id) != self.find(new_id) {
                        self.union(class_id, new_id);
                        unions += 1;
                    }
                }
            }
        }

        self.rebuild();
        unions
    }

    /// Apply rewrite rules to a single node, returning equivalent e-class if found.
    fn apply_rule(&mut self, _class_id: EClassId, node: &ENode) -> Option<EClassId> {
        match node {
            // x + 0 -> x
            ENode::Add(a, b) => {
                if self.contains_const(*b, 0.0) {
                    return Some(*a);
                }
                if self.contains_const(*a, 0.0) {
                    return Some(*b);
                }
                // x + x -> 2 * x
                if self.find(*a) == self.find(*b) {
                    let two = self.add(ENode::constant(2.0));
                    return Some(self.add(ENode::Mul(two, *a)));
                }
                None
            }

            // x - 0 -> x
            // x - x -> 0
            ENode::Sub(a, b) => {
                if self.contains_const(*b, 0.0) {
                    return Some(*a);
                }
                if self.find(*a) == self.find(*b) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                // 0 - x -> -x
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::Neg(*b)));
                }
                None
            }

            // x * 0 -> 0
            // x * 1 -> x
            // x * -1 -> -x
            ENode::Mul(a, b) => {
                if self.contains_const(*a, 0.0) || self.contains_const(*b, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                if self.contains_const(*b, 1.0) {
                    return Some(*a);
                }
                if self.contains_const(*a, 1.0) {
                    return Some(*b);
                }
                if self.contains_const(*b, -1.0) {
                    return Some(self.add(ENode::Neg(*a)));
                }
                if self.contains_const(*a, -1.0) {
                    return Some(self.add(ENode::Neg(*b)));
                }
                None
            }

            // x / 1 -> x
            // 0 / x -> 0 (assuming x != 0)
            // x / x -> 1 (assuming x != 0)
            ENode::Div(a, b) => {
                if self.contains_const(*b, 1.0) {
                    return Some(*a);
                }
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                if self.find(*a) == self.find(*b) {
                    // x / x = 1 (assuming non-zero)
                    return Some(self.add(ENode::constant(1.0)));
                }
                // x / sqrt(y) -> x * rsqrt(y)
                for child_node in self.nodes(*b) {
                    if let ENode::Sqrt(inner) = child_node {
                        let inner = *inner;
                        let rsqrt = self.add(ENode::Rsqrt(inner));
                        return Some(self.add(ENode::Mul(*a, rsqrt)));
                    }
                }
                None
            }

            // --x -> x
            ENode::Neg(a) => {
                for child_node in self.nodes(*a) {
                    if let ENode::Neg(inner) = child_node {
                        return Some(*inner);
                    }
                }
                // -0 -> 0
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                None
            }

            // sqrt(0) -> 0
            // sqrt(1) -> 1
            ENode::Sqrt(a) => {
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                if self.contains_const(*a, 1.0) {
                    return Some(self.add(ENode::constant(1.0)));
                }
                None
            }

            // rsqrt(1) -> 1
            ENode::Rsqrt(a) => {
                if self.contains_const(*a, 1.0) {
                    return Some(self.add(ENode::constant(1.0)));
                }
                None
            }

            // |0| -> 0
            // |-x| -> |x|
            ENode::Abs(a) => {
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                for child_node in self.nodes(*a) {
                    if let ENode::Neg(inner) = child_node {
                        return Some(self.add(ENode::Abs(*inner)));
                    }
                    // |c| -> c if c >= 0
                    if let Some(c) = child_node.as_f32() {
                        if c >= 0.0 {
                            return Some(*a);
                        } else {
                            return Some(self.add(ENode::constant(-c)));
                        }
                    }
                }
                None
            }

            // min(x, x) -> x
            ENode::Min(a, b) => {
                if self.find(*a) == self.find(*b) {
                    return Some(*a);
                }
                None
            }

            // max(x, x) -> x
            ENode::Max(a, b) => {
                if self.find(*a) == self.find(*b) {
                    return Some(*a);
                }
                None
            }

            // a * b + c: check if c = 0 -> a * b
            ENode::MulAdd(a, b, c) => {
                if self.contains_const(*c, 0.0) {
                    return Some(self.add(ENode::Mul(*a, *b)));
                }
                if self.contains_const(*a, 0.0) || self.contains_const(*b, 0.0) {
                    return Some(*c);
                }
                if self.contains_const(*a, 1.0) {
                    return Some(self.add(ENode::Add(*b, *c)));
                }
                if self.contains_const(*b, 1.0) {
                    return Some(self.add(ENode::Add(*a, *c)));
                }
                None
            }

            _ => None,
        }
    }

    /// Run equality saturation until no new equivalences are found or max iterations.
    pub fn saturate(&mut self) {
        self.saturate_with_limit(100)
    }

    /// Run equality saturation with iteration limit.
    pub fn saturate_with_limit(&mut self, max_iters: usize) {
        for _ in 0..max_iters {
            let unions = self.apply_rules();
            if unions == 0 {
                break;
            }
        }
    }

    /// Extract the simplest expression from an e-class.
    ///
    /// Uses a simple cost model: constants and vars are free, operations cost 1,
    /// and we minimize total cost.
    pub fn extract(&self, root: EClassId) -> ENode {
        let root = self.find(root);

        // Cost table: e-class id -> (cost, best_node)
        let mut costs: HashMap<EClassId, (usize, ENode)> = HashMap::new();

        // Iterate until costs stabilize (simple fixpoint)
        for _ in 0..self.classes.len() {
            let mut changed = false;

            for idx in 0..self.classes.len() {
                let id = EClassId(idx as u32);
                let id = self.find(id);

                for node in &self.classes[id.index()].nodes {
                    let cost = self.node_cost(node, &costs);
                    let current = costs.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX);

                    if cost < current {
                        costs.insert(id, (cost, node.clone()));
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        costs
            .get(&root)
            .map(|(_, node)| node.clone())
            .unwrap_or(ENode::Const(0))
    }

    /// Compute cost of a node given current best costs for children.
    fn node_cost(&self, node: &ENode, costs: &HashMap<EClassId, (usize, ENode)>) -> usize {
        let get_cost = |id: EClassId| -> usize {
            let id = self.find(id);
            costs.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX / 2)
        };

        match node {
            ENode::Var(_) | ENode::Const(_) => 0,
            _ => {
                let child_cost: usize = node.children().iter().map(|&c| get_cost(c)).sum();
                child_cost.saturating_add(1)
            }
        }
    }

    /// Extract a full expression tree from an e-class.
    pub fn extract_tree(&self, root: EClassId) -> ExprTree {
        let root = self.find(root);
        let best_node = self.extract(root);
        self.node_to_tree(&best_node)
    }

    fn node_to_tree(&self, node: &ENode) -> ExprTree {
        match node {
            ENode::Var(v) => ExprTree::Var(*v),
            ENode::Const(bits) => ExprTree::Const(f32::from_bits(*bits)),
            ENode::Add(a, b) => ExprTree::Add(
                Box::new(self.extract_tree(*a)),
                Box::new(self.extract_tree(*b)),
            ),
            ENode::Sub(a, b) => ExprTree::Sub(
                Box::new(self.extract_tree(*a)),
                Box::new(self.extract_tree(*b)),
            ),
            ENode::Mul(a, b) => ExprTree::Mul(
                Box::new(self.extract_tree(*a)),
                Box::new(self.extract_tree(*b)),
            ),
            ENode::Div(a, b) => ExprTree::Div(
                Box::new(self.extract_tree(*a)),
                Box::new(self.extract_tree(*b)),
            ),
            ENode::Neg(a) => ExprTree::Neg(Box::new(self.extract_tree(*a))),
            ENode::Sqrt(a) => ExprTree::Sqrt(Box::new(self.extract_tree(*a))),
            ENode::Rsqrt(a) => ExprTree::Rsqrt(Box::new(self.extract_tree(*a))),
            ENode::Abs(a) => ExprTree::Abs(Box::new(self.extract_tree(*a))),
            ENode::Min(a, b) => ExprTree::Min(
                Box::new(self.extract_tree(*a)),
                Box::new(self.extract_tree(*b)),
            ),
            ENode::Max(a, b) => ExprTree::Max(
                Box::new(self.extract_tree(*a)),
                Box::new(self.extract_tree(*b)),
            ),
            ENode::MulAdd(a, b, c) => ExprTree::MulAdd(
                Box::new(self.extract_tree(*a)),
                Box::new(self.extract_tree(*b)),
                Box::new(self.extract_tree(*c)),
            ),
        }
    }
}

/// A concrete expression tree (for extraction results).
#[derive(Clone, Debug, PartialEq)]
pub enum ExprTree {
    /// Variable by index
    Var(u8),
    /// Constant value
    Const(f32),
    /// Addition
    Add(Box<ExprTree>, Box<ExprTree>),
    /// Subtraction
    Sub(Box<ExprTree>, Box<ExprTree>),
    /// Multiplication
    Mul(Box<ExprTree>, Box<ExprTree>),
    /// Division
    Div(Box<ExprTree>, Box<ExprTree>),
    /// Negation
    Neg(Box<ExprTree>),
    /// Square root
    Sqrt(Box<ExprTree>),
    /// Reciprocal square root
    Rsqrt(Box<ExprTree>),
    /// Absolute value
    Abs(Box<ExprTree>),
    /// Minimum
    Min(Box<ExprTree>, Box<ExprTree>),
    /// Maximum
    Max(Box<ExprTree>, Box<ExprTree>),
    /// Fused multiply-add
    MulAdd(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
}

impl ExprTree {
    /// Evaluate the expression tree with given variable values.
    pub fn eval(&self, vars: &[f32; 4]) -> f32 {
        match self {
            ExprTree::Var(i) => vars[*i as usize],
            ExprTree::Const(c) => *c,
            ExprTree::Add(a, b) => a.eval(vars) + b.eval(vars),
            ExprTree::Sub(a, b) => a.eval(vars) - b.eval(vars),
            ExprTree::Mul(a, b) => a.eval(vars) * b.eval(vars),
            ExprTree::Div(a, b) => a.eval(vars) / b.eval(vars),
            ExprTree::Neg(a) => -a.eval(vars),
            ExprTree::Sqrt(a) => a.eval(vars).sqrt(),
            ExprTree::Rsqrt(a) => 1.0 / a.eval(vars).sqrt(),
            ExprTree::Abs(a) => a.eval(vars).abs(),
            ExprTree::Min(a, b) => a.eval(vars).min(b.eval(vars)),
            ExprTree::Max(a, b) => a.eval(vars).max(b.eval(vars)),
            ExprTree::MulAdd(a, b, c) => a.eval(vars) * b.eval(vars) + c.eval(vars),
        }
    }

    /// Count the number of operations in the tree.
    pub fn cost(&self) -> usize {
        match self {
            ExprTree::Var(_) | ExprTree::Const(_) => 0,
            ExprTree::Neg(a) | ExprTree::Sqrt(a) | ExprTree::Rsqrt(a) | ExprTree::Abs(a) => {
                1 + a.cost()
            }
            ExprTree::Add(a, b)
            | ExprTree::Sub(a, b)
            | ExprTree::Mul(a, b)
            | ExprTree::Div(a, b)
            | ExprTree::Min(a, b)
            | ExprTree::Max(a, b) => 1 + a.cost() + b.cost(),
            ExprTree::MulAdd(a, b, c) => 1 + a.cost() + b.cost() + c.cost(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_zero() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let x_plus_0 = eg.add(ENode::Add(x, zero));

        eg.saturate();

        // x + 0 should be equivalent to x
        assert_eq!(eg.find(x_plus_0), eg.find(x));
    }

    #[test]
    fn test_mul_zero() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let x_times_0 = eg.add(ENode::Mul(x, zero));

        eg.saturate();

        // x * 0 should be equivalent to 0
        assert_eq!(eg.find(x_times_0), eg.find(zero));
    }

    #[test]
    fn test_mul_one() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));
        let x_times_1 = eg.add(ENode::Mul(x, one));

        eg.saturate();

        // x * 1 should be equivalent to x
        assert_eq!(eg.find(x_times_1), eg.find(x));
    }

    #[test]
    fn test_sub_self() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let x_minus_x = eg.add(ENode::Sub(x, x));
        let zero = eg.add(ENode::constant(0.0));

        eg.saturate();

        // x - x should be equivalent to 0
        assert_eq!(eg.find(x_minus_x), eg.find(zero));
    }

    #[test]
    fn test_div_self() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let x_div_x = eg.add(ENode::Div(x, x));
        let one = eg.add(ENode::constant(1.0));

        eg.saturate();

        // x / x should be equivalent to 1
        assert_eq!(eg.find(x_div_x), eg.find(one));
    }

    #[test]
    fn test_double_neg() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::Neg(x));
        let neg_neg_x = eg.add(ENode::Neg(neg_x));

        eg.saturate();

        // --x should be equivalent to x
        assert_eq!(eg.find(neg_neg_x), eg.find(x));
    }

    #[test]
    fn test_complex_simplification() {
        let mut eg = EGraph::new();

        // Build: (x + 0) * 1 - x
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let one = eg.add(ENode::constant(1.0));

        let x_plus_0 = eg.add(ENode::Add(x, zero));
        let times_1 = eg.add(ENode::Mul(x_plus_0, one));
        let minus_x = eg.add(ENode::Sub(times_1, x));

        eg.saturate();

        // Should simplify to 0
        let simplified = eg.extract_tree(minus_x);
        assert_eq!(simplified, ExprTree::Const(0.0));
    }

    #[test]
    fn test_div_sqrt_to_mul_rsqrt() {
        let mut eg = EGraph::new();

        // Build: x / sqrt(y)
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sqrt_y = eg.add(ENode::Sqrt(y));
        let div = eg.add(ENode::Div(x, sqrt_y));

        eg.saturate();

        // Should be equivalent to x * rsqrt(y)
        let rsqrt_y = eg.add(ENode::Rsqrt(y));
        let mul = eg.add(ENode::Mul(x, rsqrt_y));

        assert_eq!(eg.find(div), eg.find(mul));
    }

    #[test]
    fn test_extract_simplest() {
        let mut eg = EGraph::new();

        // Build: ((x * 1) + 0) - 0
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let one = eg.add(ENode::constant(1.0));

        let x_times_1 = eg.add(ENode::Mul(x, one));
        let plus_0 = eg.add(ENode::Add(x_times_1, zero));
        let minus_0 = eg.add(ENode::Sub(plus_0, zero));

        eg.saturate();

        // Extract should give us just x (Var(0))
        let tree = eg.extract_tree(minus_0);
        assert_eq!(tree, ExprTree::Var(0));
    }

    #[test]
    fn test_x_plus_x() {
        let mut eg = EGraph::new();

        // Build: x + x
        let x = eg.add(ENode::Var(0));
        let x_plus_x = eg.add(ENode::Add(x, x));

        eg.saturate();

        // Should be equivalent to 2 * x
        let two = eg.add(ENode::constant(2.0));
        let two_times_x = eg.add(ENode::Mul(two, x));

        assert_eq!(eg.find(x_plus_x), eg.find(two_times_x));
    }

    #[test]
    fn test_eval_tree() {
        let tree = ExprTree::Add(
            Box::new(ExprTree::Mul(
                Box::new(ExprTree::Var(0)),
                Box::new(ExprTree::Var(0)),
            )),
            Box::new(ExprTree::Mul(
                Box::new(ExprTree::Var(1)),
                Box::new(ExprTree::Var(1)),
            )),
        );

        // x^2 + y^2 at (3, 4) = 9 + 16 = 25
        let result = tree.eval(&[3.0, 4.0, 0.0, 0.0]);
        assert!((result - 25.0).abs() < 1e-6);
    }
}
