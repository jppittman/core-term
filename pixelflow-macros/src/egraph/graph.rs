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
    rules: Vec<Box<dyn Rewrite>>,
    pub match_counts: HashMap<String, usize>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    pub fn new() -> Self {
        let mut eg = Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::new(),
            worklist: Vec::new(),
            rules: Vec::new(),
            match_counts: HashMap::new(),
        };
        eg.register_algebraic_rules();
        eg
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rewrite>) {
        self.rules.push(rule);
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

    fn apply_rules(&mut self) -> usize {
        let mut unions = 0;
        let mut updates = Vec::new();

        let num_classes = self.classes.len();
        for class_idx in 0..num_classes {
            let class_id = EClassId(class_idx as u32);
            let class_id = self.find(class_id);
            let nodes: Vec<ENode> = self.classes[class_id.index()].nodes.clone();

            for node in &nodes {
                for rule in &self.rules {
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
        let best_node = self.extract_with_costs(root, costs);
        self.node_to_tree_with_costs(&best_node, costs)
    }

    fn node_to_tree_with_costs(&self, node: &ENode, costs: &CostModel) -> ExprTree {
        match node {
            ENode::Var(v) => ExprTree::Var(*v),
            ENode::Const(bits) => ExprTree::Const(f32::from_bits(*bits)),
            ENode::Add(a, b) => ExprTree::Add(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Sub(a, b) => ExprTree::Sub(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Mul(a, b) => ExprTree::Mul(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Div(a, b) => ExprTree::Div(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Neg(a) => ExprTree::Neg(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Recip(a) => ExprTree::Recip(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Sqrt(a) => ExprTree::Sqrt(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Rsqrt(a) => ExprTree::Rsqrt(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Abs(a) => ExprTree::Abs(Box::new(self.extract_tree_with_costs(*a, costs))),
            ENode::Min(a, b) => ExprTree::Min(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Max(a, b) => ExprTree::Max(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::MulAdd(a, b, c) => ExprTree::MulAdd(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs)), Box::new(self.extract_tree_with_costs(*c, costs))),
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
            ENode::Atan2(a, b) => ExprTree::Atan2(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Pow(a, b) => ExprTree::Pow(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Hypot(a, b) => ExprTree::Hypot(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Lt(a, b) => ExprTree::Lt(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Le(a, b) => ExprTree::Le(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Gt(a, b) => ExprTree::Gt(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Ge(a, b) => ExprTree::Ge(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Eq(a, b) => ExprTree::Eq(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Ne(a, b) => ExprTree::Ne(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
            ENode::Select(a, b, c) => ExprTree::Select(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs)), Box::new(self.extract_tree_with_costs(*c, costs))),
            ENode::Clamp(a, b, c) => ExprTree::Clamp(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs)), Box::new(self.extract_tree_with_costs(*c, costs))),
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
}
