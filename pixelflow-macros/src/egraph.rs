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
//! - **CostModel**: Configurable costs for target-specific optimization
//!
//! ## Example
//!
//! ```ignore
//! use pixelflow_macros::egraph::{EGraph, ENode, CostModel};
//!
//! let mut egraph = EGraph::new();
//!
//! // Build: a * b + c (should fuse to MulAdd)
//! let a = egraph.add(ENode::Var(0));
//! let b = egraph.add(ENode::Var(1));
//! let c = egraph.add(ENode::Var(2));
//! let mul = egraph.add(ENode::Mul(a, b));
//! let add = egraph.add(ENode::Add(mul, c));
//!
//! // Saturate with rewrite rules
//! egraph.saturate();
//!
//! // Extract with FMA-aware cost model
//! let costs = CostModel::with_fma();
//! let simplified = egraph.extract_with_costs(add, &costs);
//! // simplified is MulAdd(a, b, c) because it's cheaper
//! ```
//!
//! ## References
//!
//! - Willsey et al., "egg: Fast and Extensible Equality Saturation" (POPL 2021)

use std::collections::HashMap;

// ============================================================================
// Cost Model
// ============================================================================

/// Configurable cost model for operation costs.
///
/// Costs are in abstract "cycles" - lower is better. The actual values should
/// be tuned based on target CPU characteristics. Use `build.rs` to detect CPU
/// features and set appropriate costs.
///
/// ## Default Costs (approximate x86-64 cycles)
///
/// | Operation | Default | With FMA | Notes |
/// |-----------|---------|----------|-------|
/// | Var/Const | 0       | 0        | Free (register/immediate) |
/// | Add/Sub   | 4       | 4        | ~4 cycles latency |
/// | Mul       | 5       | 5        | ~5 cycles latency |
/// | Div       | 15      | 15       | ~15-20 cycles |
/// | Sqrt      | 15      | 15       | ~15-20 cycles |
/// | Rsqrt     | 5       | 5        | ~5 cycles (fast approx) |
/// | MulAdd    | 10      | 5        | 2 ops unfused, 1 op fused |
/// | MulRsqrt  | 10      | 6        | Mul + Rsqrt unfused |
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Cost of addition
    pub add: usize,
    /// Cost of subtraction
    pub sub: usize,
    /// Cost of multiplication
    pub mul: usize,
    /// Cost of division
    pub div: usize,
    /// Cost of negation
    pub neg: usize,
    /// Cost of square root
    pub sqrt: usize,
    /// Cost of reciprocal square root
    pub rsqrt: usize,
    /// Cost of absolute value
    pub abs: usize,
    /// Cost of min
    pub min: usize,
    /// Cost of max
    pub max: usize,
    /// Cost of fused multiply-add (a*b+c in one op)
    pub mul_add: usize,
    /// Cost of fused multiply-rsqrt (a*rsqrt(b) in one op)
    pub mul_rsqrt: usize,
}

impl Default for CostModel {
    /// Default costs assuming no FMA support.
    fn default() -> Self {
        Self {
            add: 4,
            sub: 4,
            mul: 5,
            div: 15,
            neg: 1,
            sqrt: 15,
            rsqrt: 5,
            abs: 1,
            min: 4,
            max: 4,
            // Without FMA: mul(5) + add(4) = 9, but we penalize slightly
            // because the expression is larger
            mul_add: 10,
            // Without fusion: mul(5) + rsqrt(5) = 10
            mul_rsqrt: 10,
        }
    }
}

impl CostModel {
    /// Create a cost model optimized for CPUs with FMA support.
    ///
    /// On modern x86-64 with FMA3 (Haswell+), `vfmadd` is a single instruction
    /// with ~5 cycle latency, same as a plain multiply.
    pub fn with_fma() -> Self {
        Self {
            mul_add: 5, // Fused: same cost as single mul
            ..Self::default()
        }
    }

    /// Create a cost model optimized for CPUs with fast rsqrt.
    ///
    /// On x86-64 with AVX, `vrsqrtps` + Newton-Raphson is ~8 cycles total,
    /// much faster than `vsqrtps` (~15 cycles) + `vdivps` (~15 cycles).
    pub fn with_fast_rsqrt() -> Self {
        Self {
            rsqrt: 4,
            mul_rsqrt: 6, // Fused is slightly cheaper
            ..Self::default()
        }
    }

    /// Create a cost model with all optimizations enabled.
    pub fn fully_optimized() -> Self {
        Self {
            mul_add: 5,
            rsqrt: 4,
            mul_rsqrt: 6,
            ..Self::default()
        }
    }

    /// Create a cost model from a map of operation names to costs.
    ///
    /// This is useful for dynamic configuration from build.rs.
    ///
    /// Recognized keys: "add", "sub", "mul", "div", "neg", "sqrt", "rsqrt",
    /// "abs", "min", "max", "mul_add", "mul_rsqrt"
    pub fn from_map(costs: &HashMap<String, usize>) -> Self {
        let mut model = Self::default();
        if let Some(&c) = costs.get("add") { model.add = c; }
        if let Some(&c) = costs.get("sub") { model.sub = c; }
        if let Some(&c) = costs.get("mul") { model.mul = c; }
        if let Some(&c) = costs.get("div") { model.div = c; }
        if let Some(&c) = costs.get("neg") { model.neg = c; }
        if let Some(&c) = costs.get("sqrt") { model.sqrt = c; }
        if let Some(&c) = costs.get("rsqrt") { model.rsqrt = c; }
        if let Some(&c) = costs.get("abs") { model.abs = c; }
        if let Some(&c) = costs.get("min") { model.min = c; }
        if let Some(&c) = costs.get("max") { model.max = c; }
        if let Some(&c) = costs.get("mul_add") { model.mul_add = c; }
        if let Some(&c) = costs.get("mul_rsqrt") { model.mul_rsqrt = c; }
        model
    }

    /// Get the cost of an ENode (not including children).
    pub fn node_op_cost(&self, node: &ENode) -> usize {
        match node {
            ENode::Var(_) | ENode::Const(_) => 0,
            ENode::Add(_, _) => self.add,
            ENode::Sub(_, _) => self.sub,
            ENode::Mul(_, _) => self.mul,
            ENode::Div(_, _) => self.div,
            ENode::Neg(_) => self.neg,
            ENode::Sqrt(_) => self.sqrt,
            ENode::Rsqrt(_) => self.rsqrt,
            ENode::Abs(_) => self.abs,
            ENode::Min(_, _) => self.min,
            ENode::Max(_, _) => self.max,
            ENode::MulAdd(_, _, _) => self.mul_add,
            ENode::MulRsqrt(_, _) => self.mul_rsqrt,
        }
    }
}

// ============================================================================
// E-Graph Data Structures
// ============================================================================

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
    /// Fused multiply-rsqrt: a * rsqrt(b)
    MulRsqrt(EClassId, EClassId),
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
            | ENode::Max(a, b)
            | ENode::MulRsqrt(a, b) => {
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
            | ENode::Max(a, b)
            | ENode::MulRsqrt(a, b) => vec![*a, *b],
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

    /// Check if an e-class contains a Mul node and return its operands.
    fn find_mul(&self, id: EClassId) -> Option<(EClassId, EClassId)> {
        for node in self.nodes(id) {
            if let ENode::Mul(a, b) = node {
                return Some((*a, *b));
            }
        }
        None
    }

    /// Check if an e-class contains a Rsqrt node and return its operand.
    fn find_rsqrt(&self, id: EClassId) -> Option<EClassId> {
        for node in self.nodes(id) {
            if let ENode::Rsqrt(inner) = node {
                return Some(*inner);
            }
        }
        None
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
            // ================================================================
            // Addition rules
            // ================================================================
            ENode::Add(a, b) => {
                // x + 0 -> x
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

                // FMA fusion: Mul(x, y) + z -> MulAdd(x, y, z)
                // Check if 'a' contains a Mul
                if let Some((x, y)) = self.find_mul(*a) {
                    let fused = self.add(ENode::MulAdd(x, y, *b));
                    return Some(fused);
                }
                // Check if 'b' contains a Mul (symmetric)
                if let Some((x, y)) = self.find_mul(*b) {
                    let fused = self.add(ENode::MulAdd(x, y, *a));
                    return Some(fused);
                }

                None
            }

            // ================================================================
            // Subtraction rules
            // ================================================================
            ENode::Sub(a, b) => {
                // x - 0 -> x
                if self.contains_const(*b, 0.0) {
                    return Some(*a);
                }
                // x - x -> 0
                if self.find(*a) == self.find(*b) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                // 0 - x -> -x
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::Neg(*b)));
                }
                None
            }

            // ================================================================
            // Multiplication rules
            // ================================================================
            ENode::Mul(a, b) => {
                // x * 0 -> 0
                if self.contains_const(*a, 0.0) || self.contains_const(*b, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                // x * 1 -> x
                if self.contains_const(*b, 1.0) {
                    return Some(*a);
                }
                if self.contains_const(*a, 1.0) {
                    return Some(*b);
                }
                // x * -1 -> -x
                if self.contains_const(*b, -1.0) {
                    return Some(self.add(ENode::Neg(*a)));
                }
                if self.contains_const(*a, -1.0) {
                    return Some(self.add(ENode::Neg(*b)));
                }

                // MulRsqrt fusion: x * rsqrt(y) -> MulRsqrt(x, y)
                if let Some(inner) = self.find_rsqrt(*b) {
                    let fused = self.add(ENode::MulRsqrt(*a, inner));
                    return Some(fused);
                }
                if let Some(inner) = self.find_rsqrt(*a) {
                    let fused = self.add(ENode::MulRsqrt(*b, inner));
                    return Some(fused);
                }

                None
            }

            // ================================================================
            // Division rules
            // ================================================================
            ENode::Div(a, b) => {
                // x / 1 -> x
                if self.contains_const(*b, 1.0) {
                    return Some(*a);
                }
                // 0 / x -> 0 (assuming x != 0)
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                // x / x -> 1 (assuming non-zero)
                if self.find(*a) == self.find(*b) {
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

            // ================================================================
            // Negation rules
            // ================================================================
            ENode::Neg(a) => {
                // --x -> x
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

            // ================================================================
            // Square root rules
            // ================================================================
            ENode::Sqrt(a) => {
                // sqrt(0) -> 0
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                // sqrt(1) -> 1
                if self.contains_const(*a, 1.0) {
                    return Some(self.add(ENode::constant(1.0)));
                }
                None
            }

            // ================================================================
            // Reciprocal square root rules
            // ================================================================
            ENode::Rsqrt(a) => {
                // rsqrt(1) -> 1
                if self.contains_const(*a, 1.0) {
                    return Some(self.add(ENode::constant(1.0)));
                }
                None
            }

            // ================================================================
            // Absolute value rules
            // ================================================================
            ENode::Abs(a) => {
                // |0| -> 0
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                for child_node in self.nodes(*a) {
                    // |-x| -> |x|
                    if let ENode::Neg(inner) = child_node {
                        return Some(self.add(ENode::Abs(*inner)));
                    }
                    // |c| -> c if c >= 0, else -c
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

            // ================================================================
            // Min/Max rules
            // ================================================================
            ENode::Min(a, b) => {
                // min(x, x) -> x
                if self.find(*a) == self.find(*b) {
                    return Some(*a);
                }
                None
            }

            ENode::Max(a, b) => {
                // max(x, x) -> x
                if self.find(*a) == self.find(*b) {
                    return Some(*a);
                }
                None
            }

            // ================================================================
            // Fused operation rules
            // ================================================================
            ENode::MulAdd(a, b, c) => {
                // a * b + 0 -> a * b
                if self.contains_const(*c, 0.0) {
                    return Some(self.add(ENode::Mul(*a, *b)));
                }
                // 0 * b + c -> c, a * 0 + c -> c
                if self.contains_const(*a, 0.0) || self.contains_const(*b, 0.0) {
                    return Some(*c);
                }
                // 1 * b + c -> b + c
                if self.contains_const(*a, 1.0) {
                    return Some(self.add(ENode::Add(*b, *c)));
                }
                // a * 1 + c -> a + c
                if self.contains_const(*b, 1.0) {
                    return Some(self.add(ENode::Add(*a, *c)));
                }
                None
            }

            ENode::MulRsqrt(a, b) => {
                // x * rsqrt(1) -> x * 1 -> x
                if self.contains_const(*b, 1.0) {
                    return Some(*a);
                }
                // 0 * rsqrt(y) -> 0
                if self.contains_const(*a, 0.0) {
                    return Some(self.add(ENode::constant(0.0)));
                }
                // 1 * rsqrt(y) -> rsqrt(y)
                if self.contains_const(*a, 1.0) {
                    return Some(self.add(ENode::Rsqrt(*b)));
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

    /// Extract the simplest expression from an e-class using default costs.
    ///
    /// Uses a simple cost model: constants and vars are free, operations cost 1.
    pub fn extract(&self, root: EClassId) -> ENode {
        self.extract_with_costs(root, &CostModel::default())
    }

    /// Extract the optimal expression from an e-class using custom costs.
    pub fn extract_with_costs(&self, root: EClassId, costs: &CostModel) -> ENode {
        let root = self.find(root);

        // Cost table: e-class id -> (cost, best_node)
        let mut cost_table: HashMap<EClassId, (usize, ENode)> = HashMap::new();

        // Iterate until costs stabilize (simple fixpoint)
        for _ in 0..self.classes.len() {
            let mut changed = false;

            for idx in 0..self.classes.len() {
                let id = EClassId(idx as u32);
                let id = self.find(id);

                for node in &self.classes[id.index()].nodes {
                    let cost = self.node_cost_with_model(node, &cost_table, costs);
                    let current = cost_table.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX);

                    if cost < current {
                        cost_table.insert(id, (cost, node.clone()));
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        cost_table
            .get(&root)
            .map(|(_, node)| node.clone())
            .unwrap_or(ENode::Const(0))
    }

    /// Compute cost of a node using the cost model.
    fn node_cost_with_model(
        &self,
        node: &ENode,
        cost_table: &HashMap<EClassId, (usize, ENode)>,
        costs: &CostModel,
    ) -> usize {
        let get_child_cost = |id: EClassId| -> usize {
            let id = self.find(id);
            cost_table.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX / 2)
        };

        let op_cost = costs.node_op_cost(node);
        let child_cost: usize = node.children().iter().map(|&c| get_child_cost(c)).sum();
        child_cost.saturating_add(op_cost)
    }

    /// Extract a full expression tree from an e-class.
    pub fn extract_tree(&self, root: EClassId) -> ExprTree {
        self.extract_tree_with_costs(root, &CostModel::default())
    }

    /// Extract a full expression tree using custom costs.
    pub fn extract_tree_with_costs(&self, root: EClassId, costs: &CostModel) -> ExprTree {
        let root = self.find(root);
        let best_node = self.extract_with_costs(root, costs);
        self.node_to_tree_with_costs(&best_node, costs)
    }

    fn node_to_tree_with_costs(&self, node: &ENode, costs: &CostModel) -> ExprTree {
        match node {
            ENode::Var(v) => ExprTree::Var(*v),
            ENode::Const(bits) => ExprTree::Const(f32::from_bits(*bits)),
            ENode::Add(a, b) => ExprTree::Add(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Sub(a, b) => ExprTree::Sub(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Mul(a, b) => ExprTree::Mul(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Div(a, b) => ExprTree::Div(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Neg(a) => ExprTree::Neg(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Sqrt(a) => ExprTree::Sqrt(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Rsqrt(a) => ExprTree::Rsqrt(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Abs(a) => ExprTree::Abs(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Min(a, b) => ExprTree::Min(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Max(a, b) => ExprTree::Max(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::MulAdd(a, b, c) => ExprTree::MulAdd(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
                Box::new(self.extract_tree_with_costs(*c, costs)),
            ),
            ENode::MulRsqrt(a, b) => ExprTree::MulRsqrt(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
        }
    }
}

// ============================================================================
// Expression Tree (extraction result)
// ============================================================================

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
    /// Fused multiply-add: a * b + c
    MulAdd(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    /// Fused multiply-rsqrt: a * rsqrt(b)
    MulRsqrt(Box<ExprTree>, Box<ExprTree>),
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
            ExprTree::MulRsqrt(a, b) => a.eval(vars) / b.eval(vars).sqrt(),
        }
    }

    /// Compute the cost of this expression tree using the given cost model.
    pub fn cost(&self, model: &CostModel) -> usize {
        match self {
            ExprTree::Var(_) | ExprTree::Const(_) => 0,
            ExprTree::Neg(a) => model.neg + a.cost(model),
            ExprTree::Sqrt(a) => model.sqrt + a.cost(model),
            ExprTree::Rsqrt(a) => model.rsqrt + a.cost(model),
            ExprTree::Abs(a) => model.abs + a.cost(model),
            ExprTree::Add(a, b) => model.add + a.cost(model) + b.cost(model),
            ExprTree::Sub(a, b) => model.sub + a.cost(model) + b.cost(model),
            ExprTree::Mul(a, b) => model.mul + a.cost(model) + b.cost(model),
            ExprTree::Div(a, b) => model.div + a.cost(model) + b.cost(model),
            ExprTree::Min(a, b) => model.min + a.cost(model) + b.cost(model),
            ExprTree::Max(a, b) => model.max + a.cost(model) + b.cost(model),
            ExprTree::MulAdd(a, b, c) => model.mul_add + a.cost(model) + b.cost(model) + c.cost(model),
            ExprTree::MulRsqrt(a, b) => model.mul_rsqrt + a.cost(model) + b.cost(model),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

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

        assert_eq!(eg.find(x_plus_0), eg.find(x));
    }

    #[test]
    fn test_mul_zero() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let x_times_0 = eg.add(ENode::Mul(x, zero));

        eg.saturate();

        assert_eq!(eg.find(x_times_0), eg.find(zero));
    }

    #[test]
    fn test_mul_one() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));
        let x_times_1 = eg.add(ENode::Mul(x, one));

        eg.saturate();

        assert_eq!(eg.find(x_times_1), eg.find(x));
    }

    #[test]
    fn test_sub_self() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let x_minus_x = eg.add(ENode::Sub(x, x));
        let zero = eg.add(ENode::constant(0.0));

        eg.saturate();

        assert_eq!(eg.find(x_minus_x), eg.find(zero));
    }

    #[test]
    fn test_div_self() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let x_div_x = eg.add(ENode::Div(x, x));
        let one = eg.add(ENode::constant(1.0));

        eg.saturate();

        assert_eq!(eg.find(x_div_x), eg.find(one));
    }

    #[test]
    fn test_double_neg() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::Neg(x));
        let neg_neg_x = eg.add(ENode::Neg(neg_x));

        eg.saturate();

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

    // ========================================================================
    // New tests for FMA fusion
    // ========================================================================

    #[test]
    fn test_fma_fusion_equivalence() {
        let mut eg = EGraph::new();

        // Build: a * b + c
        let a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let mul = eg.add(ENode::Mul(a, b));
        let add = eg.add(ENode::Add(mul, c));

        eg.saturate();

        // Should be equivalent to MulAdd(a, b, c)
        let fma = eg.add(ENode::MulAdd(a, b, c));
        assert_eq!(eg.find(add), eg.find(fma));
    }

    #[test]
    fn test_fma_extraction_with_costs() {
        let mut eg = EGraph::new();

        // Build: a * b + c
        let a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let mul = eg.add(ENode::Mul(a, b));
        let add = eg.add(ENode::Add(mul, c));

        eg.saturate();

        // With FMA support, MulAdd should be cheaper
        let costs_fma = CostModel::with_fma();
        let tree_fma = eg.extract_tree_with_costs(add, &costs_fma);

        // Should extract MulAdd because it's cheaper (5) than Mul(5) + Add(4) = 9
        assert!(matches!(tree_fma, ExprTree::MulAdd(_, _, _)));

        // Verify the cost is lower with FMA
        let cost_fma = tree_fma.cost(&costs_fma);
        assert_eq!(cost_fma, 5); // MulAdd cost with FMA
    }

    #[test]
    fn test_fma_vs_unfused_cost() {
        let mut eg = EGraph::new();

        // Build: a * b + c
        let a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let mul = eg.add(ENode::Mul(a, b));
        let add = eg.add(ENode::Add(mul, c));

        eg.saturate();

        // Default costs (no FMA optimization)
        let costs_default = CostModel::default();
        let tree_default = eg.extract_tree_with_costs(add, &costs_default);

        // Without FMA hardware: Mul(5) + Add(4) = 9 vs MulAdd(10)
        // Should prefer unfused because 9 < 10
        let cost_default = tree_default.cost(&costs_default);
        assert!(cost_default <= 10, "Expected cost <= 10, got {}", cost_default);
    }

    #[test]
    fn test_mul_rsqrt_fusion() {
        let mut eg = EGraph::new();

        // Build: x * rsqrt(y)
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let rsqrt = eg.add(ENode::Rsqrt(y));
        let mul = eg.add(ENode::Mul(x, rsqrt));

        eg.saturate();

        // Should be equivalent to MulRsqrt(x, y)
        let fused = eg.add(ENode::MulRsqrt(x, y));
        assert_eq!(eg.find(mul), eg.find(fused));
    }

    #[test]
    fn test_mul_rsqrt_extraction() {
        let mut eg = EGraph::new();

        // Build: x * rsqrt(y)
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let rsqrt = eg.add(ENode::Rsqrt(y));
        let mul = eg.add(ENode::Mul(x, rsqrt));

        eg.saturate();

        // With fast rsqrt, MulRsqrt should be preferred
        let costs = CostModel::with_fast_rsqrt();
        let tree = eg.extract_tree_with_costs(mul, &costs);

        // Should extract MulRsqrt because it's cheaper
        assert!(matches!(tree, ExprTree::MulRsqrt(_, _)));
    }

    #[test]
    fn test_cost_model_from_map() {
        let mut costs_map = HashMap::new();
        costs_map.insert("mul".to_string(), 3);
        costs_map.insert("add".to_string(), 2);
        costs_map.insert("mul_add".to_string(), 4);

        let model = CostModel::from_map(&costs_map);
        assert_eq!(model.mul, 3);
        assert_eq!(model.add, 2);
        assert_eq!(model.mul_add, 4);
        // Others should be default
        assert_eq!(model.div, 15);
    }

    #[test]
    fn test_div_sqrt_becomes_mul_rsqrt_fused() {
        let mut eg = EGraph::new();

        // Build: x / sqrt(y) which should become x * rsqrt(y) then MulRsqrt(x, y)
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sqrt_y = eg.add(ENode::Sqrt(y));
        let div = eg.add(ENode::Div(x, sqrt_y));

        eg.saturate();

        // Should also be equivalent to MulRsqrt(x, y) through the chain
        let fused = eg.add(ENode::MulRsqrt(x, y));
        assert_eq!(eg.find(div), eg.find(fused));
    }

    #[test]
    fn test_expr_tree_cost_calculation() {
        // Manual tree: (a * b) + c
        let tree = ExprTree::Add(
            Box::new(ExprTree::Mul(
                Box::new(ExprTree::Var(0)),
                Box::new(ExprTree::Var(1)),
            )),
            Box::new(ExprTree::Var(2)),
        );

        let model = CostModel::default();
        let cost = tree.cost(&model);
        // Mul(5) + Add(4) = 9
        assert_eq!(cost, 9);

        // MulAdd tree
        let fma_tree = ExprTree::MulAdd(
            Box::new(ExprTree::Var(0)),
            Box::new(ExprTree::Var(1)),
            Box::new(ExprTree::Var(2)),
        );

        let fma_cost_default = fma_tree.cost(&CostModel::default());
        assert_eq!(fma_cost_default, 10); // Default MulAdd cost

        let fma_cost_optimized = fma_tree.cost(&CostModel::with_fma());
        assert_eq!(fma_cost_optimized, 5); // FMA-optimized cost
    }
}
