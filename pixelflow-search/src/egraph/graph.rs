//! The E-Graph data structure and operations.

use std::collections::HashMap;

use super::algebra::{AddNeg, MulRecip, Canonicalize, Cancellation, InverseAnnihilation, Involution};
use super::cost::CostModel;
use super::extract::ExprTree;
use super::node::{EClassId, ENode, Op};
use super::rewrite::{Rewrite, RewriteAction};
use super::rules::{Annihilator, Commutative, Distributive, Factor, FmaFusion, Idempotent, Identity, RecipSqrt};

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

        // InversePair-derived rules
        rules.push(Canonicalize::<AddNeg>::new());
        rules.push(Involution::<AddNeg>::new());
        rules.push(Cancellation::<AddNeg>::new());
        rules.push(InverseAnnihilation::<AddNeg>::new());
        rules.push(Canonicalize::<MulRecip>::new());
        rules.push(Involution::<MulRecip>::new());
        rules.push(Cancellation::<MulRecip>::new());
        rules.push(InverseAnnihilation::<MulRecip>::new());

        // Non-InversePair rules
        rules.push(Commutative::new(Op::Add));
        rules.push(Commutative::new(Op::Mul));
        rules.push(Commutative::new(Op::Min));
        rules.push(Commutative::new(Op::Max));
        rules.push(Identity::new(Op::Add));
        rules.push(Identity::new(Op::Mul));
        rules.push(Annihilator::new(Op::Mul));
        rules.push(Idempotent::new(Op::Min));
        rules.push(Idempotent::new(Op::Max));
        rules.push(Distributive::new(Op::Mul, Op::Add));
        rules.push(Distributive::new(Op::Mul, Op::Sub));
        rules.push(Factor::new(Op::Add, Op::Mul));
        rules.push(Factor::new(Op::Sub, Op::Mul));
        rules.push(Box::new(RecipSqrt));
        rules.push(Box::new(FmaFusion));

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

    pub fn register_algebraic_rules(&mut self) {
        // =================================================================
        // InversePair-derived rules: One trait, four rules each
        // =================================================================

        // AddNeg: Addition and Negation are inverses
        self.add_rule(Canonicalize::<AddNeg>::new());        // a - b → a + neg(b)
        self.add_rule(Involution::<AddNeg>::new());          // neg(neg(x)) → x
        self.add_rule(Cancellation::<AddNeg>::new());        // (x + a) - a → x
        self.add_rule(InverseAnnihilation::<AddNeg>::new()); // x + neg(x) → 0

        // MulRecip: Multiplication and Reciprocal are inverses
        self.add_rule(Canonicalize::<MulRecip>::new());        // a / b → a * recip(b)
        self.add_rule(Involution::<MulRecip>::new());          // recip(recip(x)) → x
        self.add_rule(Cancellation::<MulRecip>::new());        // (x * a) / a → x
        self.add_rule(InverseAnnihilation::<MulRecip>::new()); // x * recip(x) → 1

        // =================================================================
        // Non-InversePair rules
        // =================================================================

        // Commutativity
        self.add_rule(Commutative::new(Op::Add));
        self.add_rule(Commutative::new(Op::Mul));
        self.add_rule(Commutative::new(Op::Min));
        self.add_rule(Commutative::new(Op::Max));

        // Identity
        self.add_rule(Identity::new(Op::Add));
        self.add_rule(Identity::new(Op::Mul));

        // Annihilator
        self.add_rule(Annihilator::new(Op::Mul));

        // Idempotence
        self.add_rule(Idempotent::new(Op::Min));
        self.add_rule(Idempotent::new(Op::Max));

        // Distributivity
        self.add_rule(Distributive::new(Op::Mul, Op::Add));
        self.add_rule(Distributive::new(Op::Mul, Op::Sub));

        // Factoring
        self.add_rule(Factor::new(Op::Add, Op::Mul));
        self.add_rule(Factor::new(Op::Sub, Op::Mul));

        // Structural / Fusion
        self.add_rule(Box::new(RecipSqrt)); // 1/sqrt(x) → rsqrt(x)
        self.add_rule(Box::new(FmaFusion)); // a * b + c → mul_add(a, b, c)
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
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b)
            | ENode::Atan2(a, b)
            | ENode::Pow(a, b)
            | ENode::Hypot(a, b)
            | ENode::Lt(a, b)
            | ENode::Le(a, b)
            | ENode::Gt(a, b)
            | ENode::Ge(a, b)
            | ENode::Eq(a, b)
            | ENode::Ne(a, b) => {
                *a = self.find(*a);
                *b = self.find(*b);
            }
            ENode::Neg(a)
            | ENode::Recip(a)
            | ENode::Sqrt(a)
            | ENode::Rsqrt(a)
            | ENode::Abs(a)
            | ENode::Floor(a)
            | ENode::Ceil(a)
            | ENode::Round(a)
            | ENode::Fract(a)
            | ENode::Sin(a)
            | ENode::Cos(a)
            | ENode::Tan(a)
            | ENode::Asin(a)
            | ENode::Acos(a)
            | ENode::Atan(a)
            | ENode::Exp(a)
            | ENode::Exp2(a)
            | ENode::Ln(a)
            | ENode::Log2(a)
            | ENode::Log10(a) => {
                *a = self.find(*a);
            }
            ENode::MulAdd(a, b, c) | ENode::Select(a, b, c) | ENode::Clamp(a, b, c) => {
                *a = self.find(*a);
                *b = self.find(*b);
                *c = self.find(*c);
            }
            ENode::Tuple(elems) => {
                for e in elems {
                    *e = self.find(*e);
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
        match tree {
            ExprTree::Var(v) => self.add(ENode::Var(*v)),
            ExprTree::Const(c) => self.add(ENode::constant(*c)),
            ExprTree::Add(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Add(a_id, b_id))
            }
            ExprTree::Sub(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Sub(a_id, b_id))
            }
            ExprTree::Mul(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Mul(a_id, b_id))
            }
            ExprTree::Div(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Div(a_id, b_id))
            }
            ExprTree::Neg(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Neg(a_id))
            }
            ExprTree::Recip(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Recip(a_id))
            }
            ExprTree::Sqrt(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Sqrt(a_id))
            }
            ExprTree::Rsqrt(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Rsqrt(a_id))
            }
            ExprTree::Abs(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Abs(a_id))
            }
            ExprTree::Min(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Min(a_id, b_id))
            }
            ExprTree::Max(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Max(a_id, b_id))
            }
            ExprTree::MulAdd(a, b, c) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                let c_id = self.add_expr(c);
                self.add(ENode::MulAdd(a_id, b_id, c_id))
            }
            ExprTree::Floor(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Floor(a_id))
            }
            ExprTree::Ceil(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Ceil(a_id))
            }
            ExprTree::Round(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Round(a_id))
            }
            ExprTree::Fract(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Fract(a_id))
            }
            ExprTree::Sin(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Sin(a_id))
            }
            ExprTree::Cos(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Cos(a_id))
            }
            ExprTree::Tan(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Tan(a_id))
            }
            ExprTree::Asin(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Asin(a_id))
            }
            ExprTree::Acos(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Acos(a_id))
            }
            ExprTree::Atan(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Atan(a_id))
            }
            ExprTree::Exp(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Exp(a_id))
            }
            ExprTree::Exp2(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Exp2(a_id))
            }
            ExprTree::Ln(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Ln(a_id))
            }
            ExprTree::Log2(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Log2(a_id))
            }
            ExprTree::Log10(a) => {
                let a_id = self.add_expr(a);
                self.add(ENode::Log10(a_id))
            }
            ExprTree::Atan2(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Atan2(a_id, b_id))
            }
            ExprTree::Pow(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Pow(a_id, b_id))
            }
            ExprTree::Hypot(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Hypot(a_id, b_id))
            }
            ExprTree::Lt(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Lt(a_id, b_id))
            }
            ExprTree::Le(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Le(a_id, b_id))
            }
            ExprTree::Gt(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Gt(a_id, b_id))
            }
            ExprTree::Ge(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Ge(a_id, b_id))
            }
            ExprTree::Eq(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Eq(a_id, b_id))
            }
            ExprTree::Ne(a, b) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                self.add(ENode::Ne(a_id, b_id))
            }
            ExprTree::Select(a, b, c) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                let c_id = self.add_expr(c);
                self.add(ENode::Select(a_id, b_id, c_id))
            }
            ExprTree::Clamp(a, b, c) => {
                let a_id = self.add_expr(a);
                let b_id = self.add_expr(b);
                let c_id = self.add_expr(c);
                self.add(ENode::Clamp(a_id, b_id, c_id))
            }
            ExprTree::Tuple(elems) => {
                let elem_ids: Vec<_> = elems.iter().map(|e| self.add_expr(e)).collect();
                self.add(ENode::Tuple(elem_ids))
            }
        }
    }

    /// Get a rule by index.
    pub fn rule(&self, idx: usize) -> Option<&dyn Rewrite> {
        self.rules.get(idx).map(|r| r.as_ref())
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
            RewriteAction::Distribute { outer, inner, a, b, c } => {
                if let Some(ab_node) = outer.make_binary(a, b) {
                    let ab_id = self.add(ab_node);
                    if let Some(ac_node) = outer.make_binary(a, c) {
                        let ac_id = self.add(ac_node);
                        if let Some(result_node) = inner.make_binary(ab_id, ac_id) {
                            let result_id = self.add(result_node);
                            if self.find(class_id) != self.find(result_id) {
                                self.union(class_id, result_id);
                                return true;
                            }
                        }
                    }
                }
                false
            }
            RewriteAction::Factor { outer, inner, common, unique_l, unique_r } => {
                if let Some(sum_node) = outer.make_binary(unique_l, unique_r) {
                    let sum_id = self.add(sum_node);
                    if let Some(result_node) = inner.make_binary(common, sum_id) {
                        let result_id = self.add(result_node);
                        if self.find(class_id) != self.find(result_id) {
                            self.union(class_id, result_id);
                            return true;
                        }
                    }
                }
                false
            }
            RewriteAction::Canonicalize { target, inverse, a, b } => {
                if let Some(inv_node) = inverse.make_unary(b) {
                    let inv_id = self.add(inv_node);
                    if let Some(target_node) = target.make_binary(a, inv_id) {
                        let target_id = self.add(target_node);
                        if self.find(class_id) != self.find(target_id) {
                            self.union(class_id, target_id);
                            return true;
                        }
                    }
                }
                false
            }
            RewriteAction::Associate { op, a, b, c } => {
                if let Some(bc_node) = op.make_binary(b, c) {
                    let bc_id = self.add(bc_node);
                    if let Some(result_node) = op.make_binary(a, bc_id) {
                        let result_id = self.add(result_node);
                        if self.find(class_id) != self.find(result_id) {
                            self.union(class_id, result_id);
                            return true;
                        }
                    }
                }
                false
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
            if unions == 0 { break; }
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
                        *self.match_counts.entry(rule.name().to_string()).or_insert(0) += 1;
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
                RewriteAction::Distribute { outer, inner, a, b, c } => {
                    if let Some(ab_node) = outer.make_binary(a, b) {
                        let ab_id = self.add(ab_node);
                        if let Some(ac_node) = outer.make_binary(a, c) {
                            let ac_id = self.add(ac_node);
                            if let Some(result_node) = inner.make_binary(ab_id, ac_id) {
                                let result_id = self.add(result_node);
                                if self.find(class_id) != self.find(result_id) {
                                    self.union(class_id, result_id);
                                    unions += 1;
                                }
                            }
                        }
                    }
                }
                RewriteAction::Factor { outer, inner, common, unique_l, unique_r } => {
                    if let Some(sum_node) = outer.make_binary(unique_l, unique_r) {
                        let sum_id = self.add(sum_node);
                        if let Some(result_node) = inner.make_binary(common, sum_id) {
                            let result_id = self.add(result_node);
                            if self.find(class_id) != self.find(result_id) {
                                self.union(class_id, result_id);
                                unions += 1;
                            }
                        }
                    }
                }
                RewriteAction::Canonicalize { target, inverse, a, b } => {
                    if let Some(inv_node) = inverse.make_unary(b) {
                        let inv_id = self.add(inv_node);
                        if let Some(target_node) = target.make_binary(a, inv_id) {
                            let target_id = self.add(target_node);
                            if self.find(class_id) != self.find(target_id) {
                                self.union(class_id, target_id);
                                unions += 1;
                            }
                        }
                    }
                }
                RewriteAction::Associate { op, a, b, c } => {
                    if let Some(bc_node) = op.make_binary(b, c) {
                        let bc_id = self.add(bc_node);
                        if let Some(result_node) = op.make_binary(a, bc_id) {
                            let result_id = self.add(result_node);
                            if self.find(class_id) != self.find(result_id) {
                                self.union(class_id, result_id);
                                unions += 1;
                            }
                        }
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
        let root = self.find(root);
        // Use depth-aware extraction
        self.extract_tree_with_depth(root, costs, 0)
    }

    /// Extract the best expression tree and its cost.
    pub fn extract_best(&self, root: EClassId, costs: &CostModel) -> (ExprTree, usize) {
        let tree = self.extract_tree_with_costs(root, costs);
        let cost = tree.cost(costs);
        (tree, cost)
    }

    /// Extract tree with depth tracking for penalty calculation.
    ///
    /// Uses the iterative extraction algorithm to find optimal per-e-class
    /// representations, then builds the tree while tracking depth.
    /// The depth_penalty in CostModel affects the total cost calculation
    /// but the iterative algorithm chooses representations globally.
    fn extract_tree_with_depth(&self, root: EClassId, costs: &CostModel, _depth: usize) -> ExprTree {
        let root = self.find(root);
        let best_node = self.extract_with_costs(root, costs);
        self.node_to_tree_iterative(&best_node, costs)
    }

    /// Convert a node to a tree using iterative extraction for children.
    fn node_to_tree_iterative(&self, node: &ENode, costs: &CostModel) -> ExprTree {
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
            ENode::Recip(a) => ExprTree::Recip(Box::new(self.extract_tree_with_costs(*a, costs))),
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
            ENode::Floor(a) => ExprTree::Floor(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Ceil(a) => ExprTree::Ceil(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Round(a) => ExprTree::Round(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Fract(a) => ExprTree::Fract(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Sin(a) => ExprTree::Sin(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Cos(a) => ExprTree::Cos(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Tan(a) => ExprTree::Tan(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Asin(a) => ExprTree::Asin(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Acos(a) => ExprTree::Acos(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Atan(a) => ExprTree::Atan(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Exp(a) => ExprTree::Exp(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Exp2(a) => ExprTree::Exp2(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Ln(a) => ExprTree::Ln(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Log2(a) => ExprTree::Log2(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Log10(a) => ExprTree::Log10(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Atan2(a, b) => ExprTree::Atan2(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Pow(a, b) => ExprTree::Pow(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Hypot(a, b) => ExprTree::Hypot(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Lt(a, b) => ExprTree::Lt(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Le(a, b) => ExprTree::Le(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Gt(a, b) => ExprTree::Gt(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Ge(a, b) => ExprTree::Ge(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Eq(a, b) => ExprTree::Eq(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Ne(a, b) => ExprTree::Ne(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Select(a, b, c) => ExprTree::Select(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
                Box::new(self.extract_tree_with_costs(*c, costs)),
            ),
            ENode::Clamp(a, b, c) => ExprTree::Clamp(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
                Box::new(self.extract_tree_with_costs(*c, costs)),
            ),
            ENode::Tuple(elems) => ExprTree::Tuple(
                elems.iter().map(|&e| self.extract_tree_with_costs(e, costs)).collect()
            ),
        }
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
    /// Unlike `node_to_tree_iterative`, this uses the optimal extraction for
    /// children, but preserves the specific node structure at this level.
    fn node_to_tree_variant(&self, node: &ENode, costs: &CostModel) -> ExprTree {
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
            ENode::Recip(a) => ExprTree::Recip(Box::new(self.extract_tree_with_costs(*a, costs))),
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
            ENode::Floor(a) => ExprTree::Floor(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Ceil(a) => ExprTree::Ceil(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Round(a) => ExprTree::Round(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Fract(a) => ExprTree::Fract(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Sin(a) => ExprTree::Sin(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Cos(a) => ExprTree::Cos(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Tan(a) => ExprTree::Tan(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Asin(a) => ExprTree::Asin(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Acos(a) => ExprTree::Acos(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Atan(a) => ExprTree::Atan(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Exp(a) => ExprTree::Exp(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Exp2(a) => ExprTree::Exp2(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Ln(a) => ExprTree::Ln(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Log2(a) => ExprTree::Log2(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Log10(a) => ExprTree::Log10(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Atan2(a, b) => ExprTree::Atan2(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Pow(a, b) => ExprTree::Pow(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Hypot(a, b) => ExprTree::Hypot(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Lt(a, b) => ExprTree::Lt(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Le(a, b) => ExprTree::Le(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Gt(a, b) => ExprTree::Gt(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Ge(a, b) => ExprTree::Ge(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Eq(a, b) => ExprTree::Eq(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Ne(a, b) => ExprTree::Ne(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
            ),
            ENode::Select(a, b, c) => ExprTree::Select(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
                Box::new(self.extract_tree_with_costs(*c, costs)),
            ),
            ENode::Clamp(a, b, c) => ExprTree::Clamp(
                Box::new(self.extract_tree_with_costs(*a, costs)),
                Box::new(self.extract_tree_with_costs(*b, costs)),
                Box::new(self.extract_tree_with_costs(*c, costs)),
            ),
            ENode::Tuple(elems) => ExprTree::Tuple(
                elems.iter().map(|&e| self.extract_tree_with_costs(e, costs)).collect()
            ),
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
        let neg_x = eg.add(ENode::Neg(x));
        let sum = eg.add(ENode::Add(x, neg_x));
        eg.saturate();
        let zero = eg.add(ENode::constant(0.0));
        assert_eq!(eg.find(sum), eg.find(zero));
    }

    #[test]
    fn test_inverse_mul() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let recip_x = eg.add(ENode::Recip(x));
        let product = eg.add(ENode::Mul(x, recip_x));
        eg.saturate();
        let one = eg.add(ENode::constant(1.0));
        assert_eq!(eg.find(product), eg.find(one));
    }

    #[test]
    fn test_complex_inverse() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let five = eg.add(ENode::constant(5.0));
        let prod = eg.add(ENode::Mul(x, five));
        let div = eg.add(ENode::Div(prod, x));
        eg.saturate();
        assert_eq!(eg.find(div), eg.find(five));
    }

    #[test]
    fn test_nested_subtraction() {
        // a - (b - c) should equal a - b + c
        // Test: 10 - (6 - 2) = 10 - 4 = 6
        let mut eg = EGraph::new();
        let a = eg.add(ENode::constant(10.0));  // a = 10
        let b = eg.add(ENode::constant(6.0));   // b = 6
        let c = eg.add(ENode::constant(2.0));   // c = 2

        // Build a - (b - c)
        let b_minus_c = eg.add(ENode::Sub(b, c));   // 6 - 2 = 4
        let result = eg.add(ENode::Sub(a, b_minus_c));  // 10 - 4 = 6

        eg.saturate();

        // Extract and verify
        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(result, &costs);
        let val = tree.eval(&[0.0; 4]);

        // Should be 6.0, not something else
        assert!((val - 6.0).abs() < 0.001, "10 - (6 - 2) should be 6.0, got {}", val);
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
        let d_sq = eg.add(ENode::Mul(d, d));
        // c_sq - r_sq = 15
        let inner_sub = eg.add(ENode::Sub(c_sq, r_sq));
        // d_sq - inner_sub = 16 - 15 = 1
        let result = eg.add(ENode::Sub(d_sq, inner_sub));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Extracted tree: {:?}", tree);
        let val = tree.eval(&[0.0; 4]);

        assert!((val - 1.0).abs() < 0.001, "16 - (16 - 1) should be 1.0, got {}", val);
    }

    #[test]
    fn test_mul_sub_pattern_with_vars() {
        // Same pattern but with variables:
        // x*x - (y - z) where x=4, y=16, z=1
        // = 16 - (16 - 1) = 16 - 15 = 1
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));  // Will be 4
        let y = eg.add(ENode::Var(1));  // Will be 16
        let z = eg.add(ENode::Var(2));  // Will be 1

        // x * x = 16
        let x_sq = eg.add(ENode::Mul(x, x));
        // y - z = 15
        let inner_sub = eg.add(ENode::Sub(y, z));
        // x_sq - inner_sub = 16 - 15 = 1
        let result = eg.add(ENode::Sub(x_sq, inner_sub));

        eg.saturate();

        let costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Extracted tree with vars: {:?}", tree);
        let val = tree.eval(&[4.0, 16.0, 1.0, 0.0]);

        assert!((val - 1.0).abs() < 0.001, "16 - (16 - 1) should be 1.0, got {}", val);
    }

    #[test]
    fn test_mul_sub_pattern_with_fma() {
        // Same pattern but with FMA costs (what the kernel! macro uses)
        // x*x - (y - z) where x=4, y=16, z=1
        // = 16 - (16 - 1) = 16 - 15 = 1
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));  // Will be 4
        let y = eg.add(ENode::Var(1));  // Will be 16
        let z = eg.add(ENode::Var(2));  // Will be 1

        // x * x = 16
        let x_sq = eg.add(ENode::Mul(x, x));
        // y - z = 15
        let inner_sub = eg.add(ENode::Sub(y, z));
        // x_sq - inner_sub = 16 - 15 = 1
        let result = eg.add(ENode::Sub(x_sq, inner_sub));

        eg.saturate();

        // Use fully_optimized costs like the kernel! macro does
        let costs = CostModel::fully_optimized();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Extracted tree with FMA costs: {:?}", tree);
        let val = tree.eval(&[4.0, 16.0, 1.0, 0.0]);

        assert!((val - 1.0).abs() < 0.001, "16 - (16 - 1) with FMA should be 1.0, got {}", val);
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
        let d = eg.add(ENode::Var(0));  // d = 4
        let a = eg.add(ENode::Var(1));  // a = 0
        let b = eg.add(ENode::Var(2));  // b = 4
        let r = eg.add(ENode::Var(3));  // r = 1

        // d_sq = d * d = 16
        let d_sq = eg.add(ENode::Mul(d, d));

        // c_sq = a*a + b*b = 0 + 16 = 16
        let a_sq = eg.add(ENode::Mul(a, a));
        let b_sq = eg.add(ENode::Mul(b, b));
        let c_sq = eg.add(ENode::Add(a_sq, b_sq));

        // r_sq = r * r = 1
        let r_sq = eg.add(ENode::Mul(r, r));

        // inner = c_sq - r_sq = 15
        let inner = eg.add(ENode::Sub(c_sq, r_sq));

        // result = d_sq - inner = 1
        let result = eg.add(ENode::Sub(d_sq, inner));

        eg.saturate();

        let costs = CostModel::fully_optimized();
        let tree = eg.extract_tree_with_costs(result, &costs);
        eprintln!("Discriminant tree: {:?}", tree);
        // d=4, a=0, b=4, r=1
        let val = tree.eval(&[4.0, 0.0, 4.0, 1.0]);
        eprintln!("Discriminant value: {}", val);

        assert!((val - 1.0).abs() < 0.001, "discriminant should be 1.0, got {}", val);
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
            current = eg.add(ENode::Add(current, one));
        }

        eg.saturate();

        // Extract with default costs (high threshold)
        let default_costs = CostModel::default();
        let tree = eg.extract_tree_with_costs(current, &default_costs);
        let val = tree.eval(&[5.0, 0.0, 0.0, 0.0]);
        assert!((val - 15.0).abs() < 0.001, "5 + 10*1 should be 15.0, got {}", val);

        // Extract with shallow costs (low threshold)
        // The result should still be mathematically correct
        let shallow_costs = CostModel::with_depth_limit(3, 1000);
        let tree2 = eg.extract_tree_with_costs(current, &shallow_costs);
        let val2 = tree2.eval(&[5.0, 0.0, 0.0, 0.0]);
        assert!((val2 - 15.0).abs() < 0.001, "shallow extraction should still be 15.0, got {}", val2);
    }

    #[test]
    fn test_extract_variants() {
        // Build x + 0, which should have multiple equivalent forms after saturation
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let sum = eg.add(ENode::Add(x, zero));

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
                i, expected, val
            );
        }
    }

    #[test]
    fn test_extract_variants_multiple_forms() {
        // Build x * 1, which should simplify to x
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));
        let prod = eg.add(ENode::Mul(x, one));

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
