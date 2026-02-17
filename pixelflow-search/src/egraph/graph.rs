//! The E-Graph data structure and operations.

use std::collections::HashMap;

use super::algebra::{
    AddNeg, Cancellation, Canonicalize, InverseAnnihilation, Involution, MulRecip,
};
use super::cost::CostModel;
use super::extract::ExprTree;
use super::node::{EClassId, ENode};
use super::ops;
use super::rewrite::{Rewrite, RewriteAction};
use super::rules::{Annihilator, Commutative, Distributive, Identity};

/// A potential rewrite target: (rule, e-class, node within class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RewriteTarget {
    /// Index into the e-graph's rule list
    pub rule_idx: usize,
    /// The e-class to apply the rule to
    pub class_id: EClassId,
    /// The node within the e-class that the rule should try to match
    pub node_idx: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EClass {
    pub(crate) nodes: Vec<ENode>,
}

pub struct EGraph {
    pub(crate) classes: Vec<EClass>,
    pub(crate) parent: Vec<EClassId>,
    memo: HashMap<ENode, EClassId>,
    worklist: Vec<EClassId>,
    /// Rules are shared via Arc so EGraph can be cloned for search branching.
    rules: std::sync::Arc<Vec<Box<dyn Rewrite>>>,
    pub match_counts: HashMap<String, usize>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EGraph {
    fn clone(&self) -> Self {
        Self {
            classes: self.classes.clone(),
            parent: self.parent.clone(),
            memo: self.memo.clone(),
            worklist: self.worklist.clone(),
            rules: self.rules.clone(), // Arc clone - cheap, shares rules
            match_counts: self.match_counts.clone(),
        }
    }
}

impl EGraph {
    pub fn new() -> Self {
        // Build rules first, then wrap in Arc
        let rules = Self::create_algebraic_rules();
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::new(),
            worklist: Vec::new(),
            rules: std::sync::Arc::new(rules),
            match_counts: HashMap::new(),
        }
    }

    /// Create the standard algebraic rewrite rules.
    fn create_algebraic_rules() -> Vec<Box<dyn Rewrite>> {
        vec![
            Canonicalize::<AddNeg>::new(),
            Involution::<AddNeg>::new(),
            Cancellation::<AddNeg>::new(),
            InverseAnnihilation::<AddNeg>::new(),
            Canonicalize::<MulRecip>::new(),
            Involution::<MulRecip>::new(),
            Cancellation::<MulRecip>::new(),
            InverseAnnihilation::<MulRecip>::new(),
            Commutative::new(&ops::Add),
            Commutative::new(&ops::Mul),
            Commutative::new(&ops::Min),
            Commutative::new(&ops::Max),
            Distributive::new(&ops::Mul, &ops::Add),
            Distributive::new(&ops::Mul, &ops::Sub),
            // Domain-specific fusion rules (FmaFusion, RecipSqrt) should be added
            // by the domain layer (pixelflow-macros) using add_rule(), not here.
            // Identity rules: x + 0 = x, x * 1 = x
            Identity::new(&ops::Add),
            Identity::new(&ops::Mul),
            // Annihilator rules: x * 0 = 0
            Annihilator::new(&ops::Mul),
        ]
    }

    /// Add a custom rule (only works before cloning).
    ///
    /// Note: This will panic if the EGraph has been cloned.
    pub fn add_rule(&mut self, rule: Box<dyn Rewrite>) {
        std::sync::Arc::get_mut(&mut self.rules)
            .expect("Cannot add rules after EGraph has been cloned")
            .push(rule);
    }

    pub fn register_algebraic_rules(&mut self) {
        // =================================================================
        // InversePair-derived rules: One trait, four rules each
        // =================================================================

        // BINARY SEARCH: First half disabled
        // AddNeg: Addition and Negation are inverses
        // self.add_rule(Canonicalize::<AddNeg>::new());        // a - b → a + neg(b)
        // self.add_rule(Involution::<AddNeg>::new());          // neg(neg(x)) → x
        // self.add_rule(Cancellation::<AddNeg>::new());        // (x + a) - a → x
        // self.add_rule(InverseAnnihilation::<AddNeg>::new()); // x + neg(x) → 0

        // MulRecip: Multiplication and Reciprocal are inverses
        // self.add_rule(Canonicalize::<MulRecip>::new());        // a / b → a * recip(b)
        // self.add_rule(Involution::<MulRecip>::new());          // recip(recip(x)) → x
        // self.add_rule(Cancellation::<MulRecip>::new());        // (x * a) / a → x
        // self.add_rule(InverseAnnihilation::<MulRecip>::new()); // x * recip(x) → 1

        // =================================================================
        // Non-InversePair rules
        // =================================================================

        // Commutativity
        // self.add_rule(Commutative::new(&ops::Add));
        // self.add_rule(Commutative::new(&ops::Mul));
        // self.add_rule(Commutative::new(&ops::Min));
        // self.add_rule(Commutative::new(&ops::Max));

        // Identity
        // self.add_rule(Identity::new(&ops::Add));
        // self.add_rule(Identity::new(&ops::Mul));

        // Annihilator
        // self.add_rule(Annihilator::new(&ops::Mul));

        // BINARY SEARCH: Second subdivision - disable Idempotent and Distributive
        // Idempotence
        // self.add_rule(Idempotent::new(&ops::Min));
        // self.add_rule(Idempotent::new(&ops::Max));

        // Distributivity
        // self.add_rule(Distributive::new(&ops::Mul, &ops::Add));
        // self.add_rule(Distributive::new(&ops::Mul, &ops::Sub));

        // BINARY SEARCH: Third subdivision - disable Factor
        // Factoring
        // self.add_rule(Factor::new(&ops::Add, &ops::Mul));
        // self.add_rule(Factor::new(&ops::Sub, &ops::Mul));

        // BINARY SEARCH: Disable all rules to test baseline
        // Structural / Fusion
        // self.add_rule(Box::new(RecipSqrt)); // 1/sqrt(x) → rsqrt(x)
        // self.add_rule(Box::new(FmaFusion)); // a * b + c → mul_add(a, b, c)
    }

    pub fn find(&self, id: EClassId) -> EClassId {
        let mut current = id;
        while self.parent[current.index()] != current {
            current = self.parent[current.index()];
        }
        current
    }

    fn find_mut(&mut self, id: EClassId) -> EClassId {
        let mut current = id;
        let mut path = Vec::new();
        while self.parent[current.index()] != current {
            path.push(current);
            current = self.parent[current.index()];
        }
        for node in path {
            self.parent[node.index()] = current;
        }
        current
    }

    fn canonicalize_node(&self, node: &mut ENode) {
        match node {
            ENode::Var(_) | ENode::Const(_) => {}
            ENode::Op { children, .. } => {
                for child in children {
                    *child = self.find(*child);
                }
            }
        }
    }

    pub fn add(&mut self, mut node: ENode) -> EClassId {
        self.canonicalize_node(&mut node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        let id = EClassId(self.classes.len() as u32);
        self.classes.push(EClass {
            nodes: vec![node.clone()],
        });
        self.parent.push(id);
        self.memo.insert(node, id);
        id
    }

    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let a = self.find_mut(a);
        let b = self.find_mut(b);
        if a == b {
            return a;
        }
        let (parent, child) = if a.0 < b.0 { (a, b) } else { (b, a) };
        self.parent[child.index()] = parent;
        let child_nodes = std::mem::take(&mut self.classes[child.index()].nodes);
        self.classes[parent.index()].nodes.extend(child_nodes);
        self.worklist.push(parent);
        parent
    }

    fn rebuild(&mut self) {
        while let Some(id) = self.worklist.pop() {
            let id = self.find(id);
            let nodes = std::mem::take(&mut self.classes[id.index()].nodes);
            let mut new_nodes = Vec::new();
            for mut node in nodes {
                self.canonicalize_node(&mut node);
                if let Some(&existing) = self.memo.get(&node) {
                    let existing = self.find(existing);
                    if existing != id {
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

    pub fn nodes(&self, id: EClassId) -> &[ENode] {
        let id = self.find(id);
        &self.classes[id.index()].nodes
    }

    /// Get the number of registered rewrite rules.
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Get the number of e-classes.
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get the total number of nodes across all e-classes.
    pub fn node_count(&self) -> usize {
        self.classes.iter().map(|c| c.nodes.len()).sum()
    }

    /// Insert an expression tree into the e-graph, returning the root e-class.
    pub fn add_expr(&mut self, tree: &ExprTree) -> EClassId {
        use super::extract::Leaf;

        match tree {
            ExprTree::Leaf(Leaf::Var(v)) => self.add(ENode::Var(*v)),
            ExprTree::Leaf(Leaf::Const(c)) => self.add(ENode::constant(*c)),

            ExprTree::Op { op, children } => {
                // Recursively add all children first
                let child_ids: Vec<_> = children.iter().map(|c| self.add_expr(c)).collect();

                // Create ENode::Op with the operation and children
                let node = ENode::Op {
                    op: *op,
                    children: child_ids,
                };

                self.add(node)
            }
        }
    }

    /// Get a rule by index.
    pub fn rule(&self, idx: usize) -> Option<&dyn Rewrite> {
        self.rules.get(idx).map(|r| r.as_ref())
    }

    /// Find all actual rewrite matches in the E-graph.
    ///
    /// Returns only targets where the rule actually matches (produces an action).
    /// Much more efficient than enumerating all combinations - only scores real matches.
    pub fn find_rewrite_matches(&self) -> Vec<RewriteTarget> {
        let mut matches = Vec::new();

        for (rule_idx, rule) in self.rules.iter().enumerate() {
            for class_idx in 0..self.classes.len() {
                let class_id = EClassId(class_idx as u32);
                let class_id = self.find(class_id);
                let nodes = &self.classes[class_id.index()].nodes;

                for (node_idx, node) in nodes.iter().enumerate() {
                    // Check if rule matches this node
                    if rule.apply(self, class_id, node).is_some() {
                        matches.push(RewriteTarget {
                            rule_idx,
                            class_id,
                            node_idx,
                        });
                    }
                }
            }
        }

        matches
    }

    /// Apply a single rule to a specific (class, node) pair.
    ///
    /// Returns true if the rule matched and produced a change.
    /// This is used by guided search to apply rules one at a time.
    pub fn apply_single_rule(
        &mut self,
        rule_idx: usize,
        class_id: EClassId,
        node_idx: usize,
    ) -> bool {
        let Some(rule) = self.rules.get(rule_idx) else {
            return false;
        };

        let class_id = self.find(class_id);
        let nodes = self.classes[class_id.index()].nodes.clone();
        let Some(node) = nodes.get(node_idx) else {
            return false;
        };

        let Some(action) = rule.apply(self, class_id, node) else {
            return false;
        };

        // Apply the action
        let changed = match action {
            RewriteAction::Union(target_id) => {
                if self.find(class_id) != self.find(target_id) {
                    self.union(class_id, target_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Create(new_node) => {
                let new_id = self.add(new_node);
                if self.find(class_id) != self.find(new_id) {
                    self.union(class_id, new_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Distribute {
                outer,
                inner,
                a,
                b,
                c,
            } => {
                // A * (B + C) → A*B + A*C
                let ab_node = ENode::Op {
                    op: outer,
                    children: vec![a, b],
                };
                let ab_id = self.add(ab_node);
                let ac_node = ENode::Op {
                    op: outer,
                    children: vec![a, c],
                };
                let ac_id = self.add(ac_node);
                let result_node = ENode::Op {
                    op: inner,
                    children: vec![ab_id, ac_id],
                };
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Factor {
                outer,
                inner,
                common,
                unique_l,
                unique_r,
            } => {
                // A*B + A*C → A * (B + C)
                let sum_node = ENode::Op {
                    op: outer,
                    children: vec![unique_l, unique_r],
                };
                let sum_id = self.add(sum_node);
                let result_node = ENode::Op {
                    op: inner,
                    children: vec![common, sum_id],
                };
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Canonicalize {
                target,
                inverse,
                a,
                b,
            } => {
                // a - b → a + neg(b)
                let inv_node = ENode::Op {
                    op: inverse,
                    children: vec![b],
                };
                let inv_id = self.add(inv_node);
                let target_node = ENode::Op {
                    op: target,
                    children: vec![a, inv_id],
                };
                let target_id = self.add(target_node);
                if self.find(class_id) != self.find(target_id) {
                    self.union(class_id, target_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Associate { op, a, b, c } => {
                // (a op b) op c → a op (b op c)
                let bc_node = ENode::Op {
                    op,
                    children: vec![b, c],
                };
                let bc_id = self.add(bc_node);
                let result_node = ENode::Op {
                    op,
                    children: vec![a, bc_id],
                };
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    true
                } else {
                    false
                }
            }
        };

        if changed {
            self.rebuild();
        }
        changed
    }

    pub fn contains_const(&self, id: EClassId, val: f32) -> bool {
        self.nodes(id).iter().any(|n| n.is_const(val))
    }

    pub fn saturate(&mut self) {
        self.saturate_with_limit(100)
    }

    pub fn saturate_with_limit(&mut self, max_iters: usize) {
        for _ in 0..max_iters {
            let unions = self.apply_rules();
            if unions == 0 {
                break;
            }
        }
    }

    /// Apply all rewrite rules once and return the number of changes made.
    ///
    /// This is used by best-first search to expand a state by one step.
    pub fn apply_rules_once(&mut self) -> usize {
        self.apply_rules()
    }

    fn apply_rules(&mut self) -> usize {
        let mut unions = 0;
        let mut updates = Vec::new();

        let num_classes = self.classes.len();
        for class_idx in 0..num_classes {
            let class_id = EClassId(class_idx as u32);
            let class_id = self.find(class_id);
            let nodes: Vec<ENode> = self.classes[class_id.index()].nodes.clone();

            for node in &nodes {
                for rule in self.rules.iter() {
                    if let Some(action) = rule.apply(self, class_id, node) {
                        updates.push((class_id, action));
                        *self
                            .match_counts
                            .entry(rule.name().to_string())
                            .or_insert(0) += 1;
                    }
                }
            }
        }

        for (class_id, action) in updates {
            match action {
                RewriteAction::Union(target_id) => {
                    if self.find(class_id) != self.find(target_id) {
                        self.union(class_id, target_id);
                        unions += 1;
                    }
                }
                RewriteAction::Create(new_node) => {
                    let new_id = self.add(new_node);
                    if self.find(class_id) != self.find(new_id) {
                        self.union(class_id, new_id);
                        unions += 1;
                    }
                }
                RewriteAction::Distribute {
                    outer,
                    inner,
                    a,
                    b,
                    c,
                } => {
                    let ab_node = ENode::Op {
                        op: outer,
                        children: vec![a, b],
                    };
                    let ab_id = self.add(ab_node);
                    let ac_node = ENode::Op {
                        op: outer,
                        children: vec![a, c],
                    };
                    let ac_id = self.add(ac_node);
                    let result_node = ENode::Op {
                        op: inner,
                        children: vec![ab_id, ac_id],
                    };
                    let result_id = self.add(result_node);
                    if self.find(class_id) != self.find(result_id) {
                        self.union(class_id, result_id);
                        unions += 1;
                    }
                }
                RewriteAction::Factor {
                    outer,
                    inner,
                    common,
                    unique_l,
                    unique_r,
                } => {
                    let sum_node = ENode::Op {
                        op: outer,
                        children: vec![unique_l, unique_r],
                    };
                    let sum_id = self.add(sum_node);
                    let result_node = ENode::Op {
                        op: inner,
                        children: vec![common, sum_id],
                    };
                    let result_id = self.add(result_node);
                    if self.find(class_id) != self.find(result_id) {
                        self.union(class_id, result_id);
                        unions += 1;
                    }
                }
                RewriteAction::Canonicalize {
                    target,
                    inverse,
                    a,
                    b,
                } => {
                    let inv_node = ENode::Op {
                        op: inverse,
                        children: vec![b],
                    };
                    let inv_id = self.add(inv_node);
                    let target_node = ENode::Op {
                        op: target,
                        children: vec![a, inv_id],
                    };
                    let target_id = self.add(target_node);
                    if self.find(class_id) != self.find(target_id) {
                        self.union(class_id, target_id);
                        unions += 1;
                    }
                }
                RewriteAction::Associate { op, a, b, c } => {
                    let bc_node = ENode::Op {
                        op,
                        children: vec![b, c],
                    };
                    let bc_id = self.add(bc_node);
                    let result_node = ENode::Op {
                        op,
                        children: vec![a, bc_id],
                    };
                    let result_id = self.add(result_node);
                    if self.find(class_id) != self.find(result_id) {
                        self.union(class_id, result_id);
                        unions += 1;
                    }
                }
            }
        }

        self.rebuild();
        unions
    }

    pub fn extract_with_costs(&self, root: EClassId, costs: &CostModel) -> ENode {
        let root = self.find(root);
        let mut cost_table: HashMap<EClassId, (usize, ENode)> = HashMap::new();
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

    fn node_cost_with_model(
        &self,
        node: &ENode,
        cost_table: &HashMap<EClassId, (usize, ENode)>,
        costs: &CostModel,
    ) -> usize {
        let get_child_cost = |id: EClassId| {
            let id = self.find(id);
            cost_table
                .get(&id)
                .map(|(c, _)| *c)
                .unwrap_or(usize::MAX / 4)
        };
        let op_cost = costs.node_op_cost(node);
        let child_cost = node
            .children()
            .iter()
            .fold(0usize, |acc, &c| acc.saturating_add(get_child_cost(c)));
        child_cost.saturating_add(op_cost)
    }

    /// Extract the minimum-cost expression tree from an e-class.
    pub fn extract_tree_with_costs(&self, root: EClassId, costs: &CostModel) -> ExprTree {
        let (tree, _cost) = super::extract::extract(self, root, costs);
        tree
    }

    /// Extract the best expression tree and its cost.
    pub fn extract_best(&self, root: EClassId, costs: &CostModel) -> (ExprTree, usize) {
        super::extract::extract(self, root, costs)
    }

    /// Extract a DAG with sharing information from an e-class.
    ///
    /// Unlike `extract_tree_with_costs`, this tracks which e-classes are used
    /// multiple times, enabling codegen to emit let-bindings for shared subexprs.
    ///
    /// # Example
    ///
    /// For `sin(X) * sin(X) + sin(X)`:
    /// - Tree extraction would compute sin(X) three times
    /// - DAG extraction marks sin(X) as shared, enabling: `let __0 = X.sin(); __0 * __0 + __0`
    pub fn extract_dag_with_costs(
        &self,
        root: EClassId,
        costs: &CostModel,
    ) -> super::extract::ExtractedDAG {
        super::extract::extract_dag(self, root, costs)
    }

    /// Extract up to N different equivalent expressions from an e-class.
    ///
    /// This returns different structural representations of the same value,
    /// useful for training data generation where we want to benchmark
    /// multiple equivalent forms.
    ///
    /// # Arguments
    ///
    /// * `root` - The e-class to extract variants from
    /// * `n` - Maximum number of variants to extract
    /// * `costs` - Cost model for building expression trees
    ///
    /// # Returns
    ///
    /// A vector of distinct ExprTree representations, up to `n` elements.
    pub fn extract_variants(&self, root: EClassId, n: usize, costs: &CostModel) -> Vec<ExprTree> {
        let root = self.find(root);
        let nodes = &self.classes[root.index()].nodes;

        // Use a set to deduplicate by debug string (since ExprTree contains f32 which doesn't impl Eq)
        let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut variants = Vec::with_capacity(n);

        for node in nodes.iter() {
            if variants.len() >= n {
                break;
            }

            // Build tree for this particular node (not the "best" one)
            let tree = self.node_to_tree_variant(node, costs);

            // Deduplicate by debug representation
            let key = format!("{:?}", tree);
            if seen_keys.insert(key) {
                variants.push(tree);
            }
        }

        variants
    }

    /// Convert a specific ENode to a tree, recursively extracting children.
    ///
    /// Unlike extraction, this preserves the specific node structure at this level,
    /// but uses optimal extraction for children.
    fn node_to_tree_variant(&self, node: &ENode, costs: &CostModel) -> ExprTree {
        use super::extract::Leaf;

        match node {
            ENode::Var(v) => ExprTree::Leaf(Leaf::Var(*v)),
            ENode::Const(bits) => ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))),
            ENode::Op { op, children } => {
                // Recursively extract optimal children
                let child_trees = children
                    .iter()
                    .map(|&child| self.extract_tree_with_costs(child, costs))
                    .collect();
                ExprTree::Op {
                    op: *op,
                    children: child_trees,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_add() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::Op {
            op: &ops::Neg,
            children: vec![x],
        });
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, neg_x],
        });
        eg.saturate();
        let zero = eg.add(ENode::constant(0.0));
        assert_eq!(eg.find(sum), eg.find(zero));
    }

    #[test]
    fn test_inverse_mul() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let recip_x = eg.add(ENode::Op {
            op: &ops::Recip,
            children: vec![x],
        });
        let product = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, recip_x],
        });
        eg.saturate();
        let one = eg.add(ENode::constant(1.0));
        assert_eq!(eg.find(product), eg.find(one));
    }

    #[test]
    fn test_complex_inverse() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let five = eg.add(ENode::constant(5.0));
        let prod = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, five],
        });
        let div = eg.add(ENode::Op {
            op: &ops::Div,
            children: vec![prod, x],
        });
        eg.saturate();
        assert_eq!(eg.find(div), eg.find(five));
    }

    #[test]
    fn test_nested_subtraction() {
        // a - (b - c) should equal a - b + c
        // Test: 10 - (6 - 2) = 10 - 4 = 6
        let mut eg = EGraph::new();
        let a = eg.add(ENode::constant(10.0)); // a = 10
        let b = eg.add(ENode::constant(6.0)); // b = 6
        let c = eg.add(ENode::constant(2.0)); // c = 2

        // Build a - (b - c)
        let b_minus_c = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![b, c],
        }); // 6 - 2 = 4
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![a, b_minus_c],
        }); // 10 - 4 = 6

        eg.saturate();

        // Extract and verify
        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(result, &costs);
        let val = tree.eval(&[0.0; 4]);

        // Should be 6.0, not something else
        assert!(
            (val - 6.0).abs() < 0.001,
            "10 - (6 - 2) should be 6.0, got {}",
            val
        );
    }

    #[test]
    fn test_mul_sub_pattern() {
        // This is the problematic pattern from discriminant:
        // d*d - (c - r) where d=4, c=16, r=1
        // = 16 - (16 - 1) = 16 - 15 = 1
        let mut eg = EGraph::new();
        let d = eg.add(ENode::constant(4.0));
        let c_sq = eg.add(ENode::constant(16.0));
        let r_sq = eg.add(ENode::constant(1.0));

        // d * d = 16
        let d_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![d, d],
        });
        // c_sq - r_sq = 15
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![c_sq, r_sq],
        });
        // d_sq - inner_sub = 16 - 15 = 1
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![d_sq, inner_sub],
        });

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Extracted tree: {:?}", tree);
        let val = tree.eval(&[0.0; 4]);

        assert!(
            (val - 1.0).abs() < 0.001,
            "16 - (16 - 1) should be 1.0, got {}",
            val
        );
    }

    #[test]
    fn test_mul_sub_pattern_with_vars() {
        // Same pattern but with variables:
        // x*x - (y - z) where x=4, y=16, z=1
        // = 16 - (16 - 1) = 16 - 15 = 1
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0)); // Will be 4
        let y = eg.add(ENode::Var(1)); // Will be 16
        let z = eg.add(ENode::Var(2)); // Will be 1

        // x * x = 16
        let x_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, x],
        });
        // y - z = 15
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![y, z],
        });
        // x_sq - inner_sub = 16 - 15 = 1
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x_sq, inner_sub],
        });

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Extracted tree with vars: {:?}", tree);
        let val = tree.eval(&[4.0, 16.0, 1.0, 0.0]);

        assert!(
            (val - 1.0).abs() < 0.001,
            "16 - (16 - 1) should be 1.0, got {}",
            val
        );
    }

    #[test]
    fn test_mul_sub_pattern_with_fma() {
        // Same pattern but with FMA costs (what the kernel! macro uses)
        // x*x - (y - z) where x=4, y=16, z=1
        // = 16 - (16 - 1) = 16 - 15 = 1
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0)); // Will be 4
        let y = eg.add(ENode::Var(1)); // Will be 16
        let z = eg.add(ENode::Var(2)); // Will be 1

        // x * x = 16
        let x_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, x],
        });
        // y - z = 15
        let inner_sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![y, z],
        });
        // x_sq - inner_sub = 16 - 15 = 1
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x_sq, inner_sub],
        });

        eg.saturate();

        // Use fully_optimized costs like the kernel! macro does
        let costs = CostModel::fully_optimized();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Extracted tree with FMA costs: {:?}", tree);
        let val = tree.eval(&[4.0, 16.0, 1.0, 0.0]);

        assert!(
            (val - 1.0).abs() < 0.001,
            "16 - (16 - 1) with FMA should be 1.0, got {}",
            val
        );
    }

    #[test]
    fn test_discriminant_structure() {
        // Match the actual discriminant structure:
        // d_dot_c² - (c_sq - r_sq)
        // where c_sq = a² + b² and r_sq = r² (3 scalar vars to fit in eval's 4-slot array)
        // Using d=4, a=0, b=4, r=1
        // d_sq = 16
        // c_sq = 0 + 16 = 16
        // r_sq = 1
        // discriminant = 16 - (16 - 1) = 16 - 15 = 1
        let mut eg = EGraph::new();
        let d = eg.add(ENode::Var(0)); // d = 4
        let a = eg.add(ENode::Var(1)); // a = 0
        let b = eg.add(ENode::Var(2)); // b = 4
        let r = eg.add(ENode::Var(3)); // r = 1

        // d_sq = d * d = 16
        let d_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![d, d],
        });

        // c_sq = a*a + b*b = 0 + 16 = 16
        let a_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![a, a],
        });
        let b_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![b, b],
        });
        let c_sq = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![a_sq, b_sq],
        });

        // r_sq = r * r = 1
        let r_sq = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![r, r],
        });

        // inner = c_sq - r_sq = 15
        let inner = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![c_sq, r_sq],
        });

        // result = d_sq - inner = 1
        let result = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![d_sq, inner],
        });

        eg.saturate();

        let costs = CostModel::fully_optimized();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Discriminant tree: {:?}", tree);
        // d=4, a=0, b=4, r=1
        let val = tree.eval(&[4.0, 0.0, 4.0, 1.0]);
        eprintln!("Discriminant value: {}", val);

        assert!(
            (val - 1.0).abs() < 0.001,
            "discriminant should be 1.0, got {}",
            val
        );
    }

    #[test]
    fn test_depth_penalty_calculation() {
        // Test the hinge penalty function
        let costs = CostModel::with_depth_limit(5, 100);

        // Below threshold: no penalty
        assert_eq!(costs.depth_cost(0), 0);
        assert_eq!(costs.depth_cost(5), 0);

        // Above threshold: linear penalty
        assert_eq!(costs.depth_cost(6), 100);
        assert_eq!(costs.depth_cost(7), 200);
        assert_eq!(costs.depth_cost(10), 500);
    }

    #[test]
    fn test_shallow_cost_model() {
        // Shallow model should have aggressive depth penalty
        let costs = CostModel::shallow();
        assert_eq!(costs.depth_threshold, 16);
        assert_eq!(costs.depth_penalty, 500);

        // Penalty kicks in after 16
        assert_eq!(costs.depth_cost(16), 0);
        assert_eq!(costs.depth_cost(17), 500);
        assert_eq!(costs.depth_cost(20), 2000);
    }

    #[test]
    fn test_depth_aware_extraction() {
        // Build a deep expression: ((((x + 1) + 1) + 1) + 1)
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));

        let mut current = x;
        for _ in 0..10 {
            current = eg.add(ENode::Op {
                op: &ops::Add,
                children: vec![current, one],
            });
        }

        eg.saturate();

        // Extract with default costs (high threshold)
        let default_costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(current, &default_costs);
        let val = tree.eval(&[5.0, 0.0, 0.0, 0.0]);
        assert!(
            (val - 15.0).abs() < 0.001,
            "5 + 10*1 should be 15.0, got {}",
            val
        );

        // Extract with shallow costs (low threshold)
        // The result should still be mathematically correct
        let shallow_costs = CostModel::with_depth_limit(3, 1000);
        let tree2 = eg.extract_tree_with_costs(current, &shallow_costs);
        let val2 = tree2.eval(&[5.0, 0.0, 0.0, 0.0]);
        assert!(
            (val2 - 15.0).abs() < 0.001,
            "shallow extraction should still be 15.0, got {}",
            val2
        );
    }

    #[test]
    fn test_extract_variants() {
        // Build x + 0, which should have multiple equivalent forms after saturation
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, zero],
        });

        eg.saturate();

        let costs = CostModel::default();
        let variants = eg.extract_variants(sum, 5, &costs);

        // Should have at least one variant
        assert!(!variants.is_empty(), "Should extract at least one variant");

        // All variants should evaluate to the same value
        let test_val = 42.0;
        let expected = test_val; // x + 0 = x
        for (i, tree) in variants.iter().enumerate() {
            let val = tree.eval(&[test_val, 0.0, 0.0, 0.0]);
            assert!(
                (val - expected).abs() < 0.001,
                "Variant {} should evaluate to {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_extract_variants_multiple_forms() {
        // Build x * 1, which should simplify to x
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));
        let prod = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, one],
        });

        eg.saturate();

        let costs = CostModel::default();
        let variants = eg.extract_variants(prod, 10, &costs);

        // Should have multiple variants (x*1, 1*x, x, etc.)
        eprintln!("Found {} variants for x*1:", variants.len());
        for (i, v) in variants.iter().enumerate() {
            eprintln!("  {}: {:?}", i, v);
        }

        // All should evaluate to x
        for tree in &variants {
            let val = tree.eval(&[7.5, 0.0, 0.0, 0.0]);
            assert!((val - 7.5).abs() < 0.001);
        }
    }
}
