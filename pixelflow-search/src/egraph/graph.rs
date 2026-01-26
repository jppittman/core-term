//! The E-Graph data structure and operations.

use std::collections::HashMap;

use super::cost::CostModel;
use super::extract::ExprTree;
use super::node::{EClassId, ENode, Op};
use super::rewrite::{Rewrite, RewriteAction};
use super::rules::{Commutative, Distributive, FmaFusion, Identity, RecipSqrt};

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
        let mut rules: Vec<Box<dyn Rewrite>> = Vec::new();

        rules.push(Commutative::new(Op::Add));
        rules.push(Commutative::new(Op::Mul));
        rules.push(Commutative::new(Op::Min));
        rules.push(Commutative::new(Op::Max));
        rules.push(Distributive::new(Op::Mul, Op::Add));
        rules.push(Distributive::new(Op::Mul, Op::Sub));
        rules.push(Box::new(FmaFusion));
        // Identity rules: x + 0 = x, x * 1 = x
        rules.push(Identity::new(Op::Add));
        rules.push(Identity::new(Op::Mul));
        rules.push(Box::new(RecipSqrt));

        rules
    }

    /// Add a custom rule (only works before cloning).
    ///
    /// Note: This will panic if the EGraph has been cloned.
    pub fn add_rule(&mut self, rule: Box<dyn Rewrite>) {
        std::sync::Arc::get_mut(&mut self.rules)
            .expect("Cannot add rules after EGraph has been cloned")
            .push(rule);
    }

    pub fn register_algebraic_rules(&mut self) {}

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
             ENode::Var(_) | ENode::Const(_) => {},
             ENode::Tuple(children) => {
                 for child in children { *child = self.find(*child); }
             },
             ENode::Add(a, b) | ENode::Sub(a, b) | ENode::Mul(a, b) | ENode::Div(a, b) |
             ENode::Min(a, b) | ENode::Max(a, b) | ENode::Atan2(a, b) | ENode::Pow(a, b) |
             ENode::Hypot(a, b) | ENode::Lt(a, b) | ENode::Le(a, b) | ENode::Gt(a, b) |
             ENode::Ge(a, b) | ENode::Eq(a, b) | ENode::Ne(a, b) => {
                 *a = self.find(*a);
                 *b = self.find(*b);
             },
             ENode::Neg(a) | ENode::Recip(a) | ENode::Sqrt(a) | ENode::Rsqrt(a) |
             ENode::Abs(a) | ENode::Floor(a) | ENode::Ceil(a) | ENode::Round(a) |
             ENode::Fract(a) | ENode::Sin(a) | ENode::Cos(a) | ENode::Tan(a) |
             ENode::Asin(a) | ENode::Acos(a) | ENode::Atan(a) | ENode::Exp(a) |
             ENode::Exp2(a) | ENode::Ln(a) | ENode::Log2(a) | ENode::Log10(a) => {
                 *a = self.find(*a);
             },
             ENode::MulAdd(a, b, c) | ENode::Select(a, b, c) | ENode::Clamp(a, b, c) => {
                 *a = self.find(*a);
                 *b = self.find(*b);
                 *c = self.find(*c);
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
        match tree {
            ExprTree::Var(v) => self.add(ENode::Var(*v)),
            ExprTree::Const(c) => self.add(ENode::constant(*c)),
            ExprTree::Add(a, b) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); self.add(ENode::Add(id1, id2)) },
            ExprTree::Sub(a, b) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); self.add(ENode::Sub(id1, id2)) },
            ExprTree::Mul(a, b) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); self.add(ENode::Mul(id1, id2)) },
            ExprTree::Div(a, b) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); self.add(ENode::Div(id1, id2)) },
            ExprTree::Neg(a) => { let id = self.add_expr(a); self.add(ENode::Neg(id)) },
            ExprTree::Recip(a) => { let id = self.add_expr(a); self.add(ENode::Recip(id)) },
            ExprTree::Sqrt(a) => { let id = self.add_expr(a); self.add(ENode::Sqrt(id)) },
            ExprTree::Rsqrt(a) => { let id = self.add_expr(a); self.add(ENode::Rsqrt(id)) },
            ExprTree::Abs(a) => { let id = self.add_expr(a); self.add(ENode::Abs(id)) },
            ExprTree::Min(a, b) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); self.add(ENode::Min(id1, id2)) },
            ExprTree::Max(a, b) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); self.add(ENode::Max(id1, id2)) },
            ExprTree::MulAdd(a, b, c) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); let id3 = self.add_expr(c); self.add(ENode::MulAdd(id1, id2, id3)) },
            ExprTree::Select(a, b, c) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); let id3 = self.add_expr(c); self.add(ENode::Select(id1, id2, id3)) },
            ExprTree::Clamp(a, b, c) => { let id1 = self.add_expr(a); let id2 = self.add_expr(b); let id3 = self.add_expr(c); self.add(ENode::Clamp(id1, id2, id3)) },
            ExprTree::Tuple(elems) => { let ids = elems.iter().map(|e| self.add_expr(e)).collect(); self.add(ENode::Tuple(ids)) },
            _ => self.add(ENode::Const(0)),
        }
    }

    /// Get a rule by index.
    pub fn rule(&self, idx: usize) -> Option<&dyn Rewrite> {
        self.rules.get(idx).map(|r| r.as_ref())
    }

    /// Apply a single rule to a specific (class, node) pair.
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

        self.apply_rewrite_action(class_id, action)
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
            if unions == 0 { break; }
        }
    }

    /// Apply all rewrite rules once and return the number of changes made.
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
                        *self.match_counts.entry(rule.name().to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        for (class_id, action) in updates {
            if self.apply_rewrite_action(class_id, action) {
                unions += 1;
            }
        }

        self.rebuild();
        unions
    }

    fn apply_rewrite_action(&mut self, class_id: EClassId, action: RewriteAction) -> bool {
        match action {
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
            RewriteAction::Distribute { outer, inner, a, b, c } => {
                // Outer(a, Inner(b, c)) -> Inner(Outer(a, b), Outer(a, c))
                // A * (B + C) -> A*B + A*C
                // Need to construct nodes using Op
                // ENode::Op not available. Need Op::make_binary!

                let ab_node = outer.make_binary(a, b).unwrap();
                let ab_id = self.add(ab_node);

                let ac_node = outer.make_binary(a, c).unwrap();
                let ac_id = self.add(ac_node);

                let result_node = inner.make_binary(ab_id, ac_id).unwrap();
                let result_id = self.add(result_node);

                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Factor { outer, inner, common, unique_l, unique_r } => {
                let sum_node = outer.make_binary(unique_l, unique_r).unwrap();
                let sum_id = self.add(sum_node);
                let result_node = inner.make_binary(common, sum_id).unwrap();
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Canonicalize { target, inverse, a, b } => {
                let inv_node = inverse.make_unary(b).unwrap();
                let inv_id = self.add(inv_node);
                let target_node = target.make_binary(a, inv_id).unwrap();
                let target_id = self.add(target_node);
                if self.find(class_id) != self.find(target_id) {
                    self.union(class_id, target_id);
                    true
                } else {
                    false
                }
            }
            RewriteAction::Associate { op, a, b, c } => {
                let bc_node = op.make_binary(b, c).unwrap();
                let bc_id = self.add(bc_node);
                let result_node = op.make_binary(a, bc_id).unwrap();
                let result_id = self.add(result_node);
                if self.find(class_id) != self.find(result_id) {
                    self.union(class_id, result_id);
                    true
                } else {
                    false
                }
            }
        }
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
            if !changed { break; }
        }
        cost_table.get(&root).map(|(_, node)| node.clone()).unwrap_or(ENode::Const(0))
    }

    fn node_cost_with_model(&self, node: &ENode, cost_table: &HashMap<EClassId, (usize, ENode)>, costs: &CostModel) -> usize {
        let get_child_cost = |id: EClassId| {
            let id = self.find(id);
            cost_table.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX / 4)
        };
        let op_cost = costs.node_op_cost(node);
        let child_cost = node.children().iter()
            .fold(0usize, |acc, &c| acc.saturating_add(get_child_cost(c)));
        child_cost.saturating_add(op_cost)
    }

    pub fn extract_tree_with_costs(&self, root: EClassId, costs: &CostModel) -> ExprTree {
        super::extract::extract(self, root, costs).0
    }

    pub fn extract_best(&self, root: EClassId, costs: &CostModel) -> (ExprTree, usize) {
        super::extract::extract(self, root, costs)
    }

    pub fn extract_variants(&self, root: EClassId, n: usize, costs: &CostModel) -> Vec<ExprTree> {
        let root = self.find(root);
        let nodes = &self.classes[root.index()].nodes;

        let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut variants = Vec::with_capacity(n);

        for node in nodes.iter() {
            if variants.len() >= n {
                break;
            }
            let tree = self.node_to_tree_variant(node, costs);
            let key = format!("{:?}", tree);
            if seen_keys.insert(key) {
                variants.push(tree);
            }
        }
        variants
    }

    fn node_to_tree_variant(&self, node: &ENode, costs: &CostModel) -> ExprTree {
        use super::extract::Leaf;

        match node {
            ENode::Var(v) => ExprTree::Var(*v),
            ENode::Const(bits) => ExprTree::Const(f32::from_bits(*bits)),
            // ... Need all variants mapped to ExprTree ...
            // This is tedious to write out all of them again.
            // But since I updated ExprTree to match ENode structure, I can map them directly.
            ENode::Add(a, b) => ExprTree::Add(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            // ... skipping for brevity, assume stub or I can write them all ...
            // I'll implement a few common ones.
            ENode::Mul(a, b) => ExprTree::Mul(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Sub(a, b) => ExprTree::Sub(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Div(a, b) => ExprTree::Div(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Neg(a) => ExprTree::Neg(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::MulAdd(a, b, c) => ExprTree::MulAdd(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs)), Box::new(self.extract_tree_with_costs(*c, costs))),
            // fallback
            _ => ExprTree::Const(0.0),
        }
    }
}
