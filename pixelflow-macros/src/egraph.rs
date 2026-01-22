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
//! ## References
//!
//! - Willsey et al., "egg: Fast and Extensible Equality Saturation" (POPL 2021)

use std::collections::HashMap;

// ============================================================================
// Cost Model
// ============================================================================

/// Configurable cost model for operation costs.
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
    /// Cost of reciprocal (1/x)
    pub recip: usize,
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
            mul_rsqrt: 10,
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
            mul_rsqrt: 6,
            ..Self::default()
        }
    }

    pub fn fully_optimized() -> Self {
        Self {
            mul_add: 5,
            recip: 4,
            rsqrt: 4,
            mul_rsqrt: 6,
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
        if let Some(&c) = costs.get("mul_rsqrt") { model.mul_rsqrt = c; }
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
            ENode::MulRsqrt(_, _) => self.mul_rsqrt,
            // Pass-through unary ops (use sqrt cost as reasonable default)
            ENode::Floor(_)
            | ENode::Ceil(_)
            | ENode::Round(_)
            | ENode::Fract(_)
            | ENode::Sin(_)
            | ENode::Cos(_)
            | ENode::Tan(_)
            | ENode::Asin(_)
            | ENode::Acos(_)
            | ENode::Atan(_)
            | ENode::Exp(_)
            | ENode::Exp2(_)
            | ENode::Ln(_)
            | ENode::Log2(_)
            | ENode::Log10(_) => self.sqrt,
            // Pass-through binary ops
            ENode::Atan2(_, _) | ENode::Pow(_, _) | ENode::Hypot(_, _) => self.sqrt,
            // Comparisons (cheap)
            ENode::Lt(_, _)
            | ENode::Le(_, _)
            | ENode::Gt(_, _)
            | ENode::Ge(_, _)
            | ENode::Eq(_, _)
            | ENode::Ne(_, _) => self.add,
            // Ternary
            ENode::Select(_, _, _) | ENode::Clamp(_, _, _) => self.add,
        }
    }
}

// ============================================================================
// Algebraic Traits
// ============================================================================

/// Identifies the kind of operation for generic algebraic rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Recip,
    Min,
    Max,
    Sqrt,
    Rsqrt,
    Abs,
    MulAdd,
    MulRsqrt,
    // Add other pass-throughs if needed for generic handling
}

impl Op {
    /// Is this operation commutative? (a op b == b op a)
    pub fn is_commutative(&self) -> bool {
        matches!(self, Op::Add | Op::Mul | Op::Min | Op::Max)
    }

    /// Is this operation associative? ((a op b) op c == a op (b op c))
    pub fn is_associative(&self) -> bool {
        matches!(self, Op::Add | Op::Mul | Op::Min | Op::Max)
    }

    /// Identity element for this operation (a op I == a)
    pub fn identity(&self) -> Option<f32> {
        match self {
            Op::Add => Some(0.0),
            Op::Mul => Some(1.0),
            _ => None,
        }
    }

    /// Annihilator element for this operation (a op Z == Z)
    pub fn annihilator(&self) -> Option<f32> {
        match self {
            Op::Mul => Some(0.0),
            _ => None,
        }
    }

    /// Idempotence (a op a == a)
    pub fn is_idempotent(&self) -> bool {
        matches!(self, Op::Min | Op::Max)
    }

    /// Does this operation have an inverse?
    /// Returns (inverse_op, identity_element).
    pub fn inverse(&self) -> Option<(Op, f32)> {
        match self {
            Op::Add => Some((Op::Neg, 0.0)),
            Op::Mul => Some((Op::Recip, 1.0)),
            _ => None,
        }
    }
}

// ============================================================================
// Rewrite Rules
// ============================================================================

pub trait Rewrite {
    fn name(&self) -> &str;
    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<EClassId>;
}

pub struct SelfInverse {
    op: Op,
}

impl SelfInverse {
    pub fn new(op: Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for SelfInverse {
    fn name(&self) -> &str { "self-inverse" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<EClassId> {
        // Match Outer: op(inner)
        if node.op() != Some(self.op) { return None; }
        
        // Only valid for unary ops
        let children = node.children();
        if children.len() != 1 { return None; }
        let inner_id = children[0];
        
        // Search the E-Class of the child to see if it contains the same Op
        for inner_node in egraph.nodes(inner_id) {
            if inner_node.op() == Some(self.op) {
                let inner_children = inner_node.children();
                if inner_children.len() == 1 {
                    // Found op(op(x)) -> return x
                    return Some(inner_children[0]);
                }
            }
        }
        None
    }
}

pub struct Cancellation {
    outer: Op, // The "Inverse" op (Sub, Div)
    inner: Op, // The "Constructor" op (Add, Mul)
}

impl Cancellation {
    pub fn new(outer: Op, inner: Op) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Cancellation {
    fn name(&self) -> &str { "binary-cancellation" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<EClassId> {
        // 1. Match Outer: e.g., Sub(numerator, canceller)
        if node.op() != Some(self.outer) { return None; }
        let (numerator, canceller) = node.binary_operands()?;

        // 2. Search inner class for the pair op: e.g., look for Add(a, b) inside 'numerator'
        for inner_node in egraph.nodes(numerator) {
            if inner_node.op() == Some(self.inner) {
                if let Some((a, b)) = inner_node.binary_operands() {
                    // Check logic: (A + B) - B ==> A
                    
                    // Case 1: The right operand matches the canceller
                    // (a op b) inv_op b -> a
                    if egraph.find(b) == egraph.find(canceller) {
                        return Some(a);
                    }
                    
                    // Case 2: The left operand matches (if inner is commutative)
                    // (b op a) inv_op b -> a
                    if self.inner.is_commutative() && egraph.find(a) == egraph.find(canceller) {
                        return Some(b);
                    }
                }
            }
        }
        None
    }
}

// ============================================================================
// E-Graph Data Structures
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EClassId(u32);

impl EClassId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ENode {
    // === Core ===
    Var(u8),
    Const(u32),

    // === Arithmetic (optimizable) ===
    Add(EClassId, EClassId),
    Sub(EClassId, EClassId),
    Mul(EClassId, EClassId),
    Div(EClassId, EClassId),
    Neg(EClassId),
    Recip(EClassId),
    Sqrt(EClassId),
    Rsqrt(EClassId),
    Abs(EClassId),
    Min(EClassId, EClassId),
    Max(EClassId, EClassId),
    MulAdd(EClassId, EClassId, EClassId),
    MulRsqrt(EClassId, EClassId),

    // === Pass-through unary ===
    Floor(EClassId),
    Ceil(EClassId),
    Round(EClassId),
    Fract(EClassId),
    Sin(EClassId),
    Cos(EClassId),
    Tan(EClassId),
    Asin(EClassId),
    Acos(EClassId),
    Atan(EClassId),
    Exp(EClassId),
    Exp2(EClassId),
    Ln(EClassId),
    Log2(EClassId),
    Log10(EClassId),

    // === Pass-through binary ===
    Atan2(EClassId, EClassId),
    Pow(EClassId, EClassId),
    Hypot(EClassId, EClassId),

    // === Comparisons ===
    Lt(EClassId, EClassId),
    Le(EClassId, EClassId),
    Gt(EClassId, EClassId),
    Ge(EClassId, EClassId),
    Eq(EClassId, EClassId),
    Ne(EClassId, EClassId),

    // === Selection/Ternary ===
    Select(EClassId, EClassId, EClassId),
    Clamp(EClassId, EClassId, EClassId),
}

impl ENode {
    pub fn constant(val: f32) -> Self {
        ENode::Const(val.to_bits())
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ENode::Const(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    pub fn is_const(&self, val: f32) -> bool {
        self.as_f32() == Some(val)
    }

    pub fn op(&self) -> Option<Op> {
        match self {
            ENode::Add(_, _) => Some(Op::Add),
            ENode::Sub(_, _) => Some(Op::Sub),
            ENode::Mul(_, _) => Some(Op::Mul),
            ENode::Div(_, _) => Some(Op::Div),
            ENode::Neg(_) => Some(Op::Neg),
            ENode::Recip(_) => Some(Op::Recip),
            ENode::Min(_, _) => Some(Op::Min),
            ENode::Max(_, _) => Some(Op::Max),
            ENode::Sqrt(_) => Some(Op::Sqrt),
            ENode::Rsqrt(_) => Some(Op::Rsqrt),
            ENode::Abs(_) => Some(Op::Abs),
            ENode::MulAdd(_, _, _) => Some(Op::MulAdd),
            ENode::MulRsqrt(_, _) => Some(Op::MulRsqrt),
            _ => None,
        }
    }

    pub fn binary_operands(&self) -> Option<(EClassId, EClassId)> {
        match self {
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b)
            | ENode::MulRsqrt(a, b)
            | ENode::Atan2(a, b)
            | ENode::Pow(a, b)
            | ENode::Hypot(a, b)
            | ENode::Lt(a, b)
            | ENode::Le(a, b)
            | ENode::Gt(a, b)
            | ENode::Ge(a, b)
            | ENode::Eq(a, b)
            | ENode::Ne(a, b) => Some((*a, *b)),
            _ => None,
        }
    }

    fn canonicalize(&mut self, egraph: &EGraph) {
        match self {
            ENode::Var(_) | ENode::Const(_) => {}
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b)
            | ENode::MulRsqrt(a, b)
            | ENode::Atan2(a, b)
            | ENode::Pow(a, b)
            | ENode::Hypot(a, b)
            | ENode::Lt(a, b)
            | ENode::Le(a, b)
            | ENode::Gt(a, b)
            | ENode::Ge(a, b)
            | ENode::Eq(a, b)
            | ENode::Ne(a, b) => {
                *a = egraph.find(*a);
                *b = egraph.find(*b);
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
                *a = egraph.find(*a);
            }
            ENode::MulAdd(a, b, c) | ENode::Select(a, b, c) | ENode::Clamp(a, b, c) => {
                *a = egraph.find(*a);
                *b = egraph.find(*b);
                *c = egraph.find(*c);
            }
        }
    }

    fn children(&self) -> Vec<EClassId> {
        match self {
            ENode::Var(_) | ENode::Const(_) => vec![],
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b)
            | ENode::MulRsqrt(a, b)
            | ENode::Atan2(a, b)
            | ENode::Pow(a, b)
            | ENode::Hypot(a, b)
            | ENode::Lt(a, b)
            | ENode::Le(a, b)
            | ENode::Gt(a, b)
            | ENode::Ge(a, b)
            | ENode::Eq(a, b)
            | ENode::Ne(a, b) => vec![*a, *b],
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
            | ENode::Log10(a) => vec![*a],
            ENode::MulAdd(a, b, c) | ENode::Select(a, b, c) | ENode::Clamp(a, b, c) => vec![*a, *b, *c],
        }
    }
}

#[derive(Clone, Debug, Default)]
struct EClass {
    nodes: Vec<ENode>,
}

pub struct EGraph {
    classes: Vec<EClass>,
    parent: Vec<EClassId>,
    memo: HashMap<ENode, EClassId>,
    worklist: Vec<EClassId>,
    rules: Vec<Box<dyn Rewrite>>,
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
        };
        eg.register_algebraic_rules();
        eg
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rewrite>) {
        self.rules.push(rule);
    }

    pub fn register_algebraic_rules(&mut self) {
        // Unary Inverses: A(A(x)) = x
        self.add_rule(SelfInverse::new(Op::Neg));   // -(-x) -> x
        self.add_rule(SelfInverse::new(Op::Recip)); // 1/(1/x) -> x

        // Binary Inverses: (x op a) inv_op a -> x
        // (x + a) - a -> x
        self.add_rule(Cancellation::new(Op::Sub, Op::Add)); 
        // (x * a) / a -> x
        self.add_rule(Cancellation::new(Op::Div, Op::Mul));
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

    pub fn add(&mut self, mut node: ENode) -> EClassId {
        node.canonicalize(self);
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
                node.canonicalize(self);
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

    fn get_coeff(&self, id: EClassId) -> (f32, EClassId) {
        let id = self.find(id);
        for node in self.nodes(id) {
            match node {
                ENode::Mul(a, b) => {
                    if let Some(ka) = self.as_const(*a) { return (ka, *b); }
                    if let Some(kb) = self.as_const(*b) { return (kb, *a); }
                }
                ENode::Neg(a) => return (-1.0, *a),
                _ => {}
            }
        }
        (1.0, id)
    }

    fn as_const(&self, id: EClassId) -> Option<f32> {
        let id = self.find(id);
        for node in self.nodes(id) {
            if let Some(val) = node.as_f32() { return Some(val); }
        }
        None
    }

    fn find_mul(&self, id: EClassId) -> Option<(EClassId, EClassId)> {
        for node in self.nodes(id) {
            if let ENode::Mul(a, b) = node { return Some((*a, *b)); }
        }
        None
    }

    fn find_add(&self, id: EClassId) -> Option<(EClassId, EClassId)> {
        for node in self.nodes(id) {
            if let ENode::Add(a, b) = node { return Some((*a, *b)); }
        }
        None
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
                // Apply trait-based rules
                for rule in &self.rules {
                    if let Some(new_id) = rule.apply(self, class_id, node) {
                        if self.find(class_id) != self.find(new_id) {
                            updates.push((class_id, new_id));
                        }
                    }
                }

                // Apply existing hardcoded rules
                if let Some(new_id) = self.apply_rule(class_id, node) {
                    if self.find(class_id) != self.find(new_id) {
                        updates.push((class_id, new_id));
                    }
                }
            }
        }
        
        for (a, b) in updates {
            if self.find(a) != self.find(b) {
                self.union(a, b);
                unions += 1;
            }
        }
        
        self.rebuild();
        unions
    }

    fn apply_rule(&mut self, _class_id: EClassId, node: &ENode) -> Option<EClassId> {
        if let Some(op) = node.op() {
            // Commutativity
            if op.is_commutative() {
                if let Some((a, b)) = node.binary_operands() {
                    let swapped = match op {
                        Op::Add => ENode::Add(b, a),
                        Op::Mul => ENode::Mul(b, a),
                        Op::Min => ENode::Min(b, a),
                        Op::Max => ENode::Max(b, a),
                        _ => unreachable!(),
                    };
                    let swapped_id = self.add(swapped);
                    if self.find(swapped_id) != self.find(_class_id) { return Some(swapped_id); }
                }
            }

            // Identity
            if let Some(id_val) = op.identity() {
                if let Some((a, b)) = node.binary_operands() {
                    if self.contains_const(b, id_val) { return Some(a); }
                    if self.contains_const(a, id_val) { return Some(b); }
                }
            }

            // Annihilator
            if let Some(zero_val) = op.annihilator() {
                if let Some((a, b)) = node.binary_operands() {
                    if self.contains_const(a, zero_val) || self.contains_const(b, zero_val) {
                        return Some(self.add(ENode::constant(zero_val)));
                    }
                }
            }

            // Idempotence
            if op.is_idempotent() {
                if let Some((a, b)) = node.binary_operands() {
                    if self.find(a) == self.find(b) { return Some(a); }
                }
            }

            // Inverse rules: x op inv(x) -> identity
            if let Some((inv_op, ident)) = op.inverse() {
                if let Some((a, b)) = node.binary_operands() {
                    // Check if a is inv(b) or b is inv(a)
                    for node_a in self.nodes(a) {
                        if node_a.op() == Some(inv_op) && node_a.children().get(0) == Some(&b) {
                            return Some(self.add(ENode::constant(ident)));
                        }
                    }
                    for node_b in self.nodes(b) {
                        if node_b.op() == Some(inv_op) && node_b.children().get(0) == Some(&a) {
                            return Some(self.add(ENode::constant(ident)));
                        }
                    }
                }
            }
        }

        match node {
            ENode::Sub(a, b) => {
                if self.find(*a) == self.find(*b) { return Some(self.add(ENode::constant(0.0))); }
                let neg_b = self.add(ENode::Neg(*b));
                return Some(self.add(ENode::Add(*a, neg_b)));
            }
            ENode::Div(a, b) => {
                if self.find(*a) == self.find(*b) { return Some(self.add(ENode::constant(1.0))); }
                let recip_b = self.add(ENode::Recip(*b));
                return Some(self.add(ENode::Mul(*a, recip_b)));
            }
            ENode::Neg(a) => {
                for child in self.nodes(*a) {
                    if let ENode::Neg(inner) = child { return Some(*inner); }
                }
                let minus_one = self.add(ENode::constant(-1.0));
                let mul = self.add(ENode::Mul(minus_one, *a));
                if self.find(mul) != self.find(_class_id) { return Some(mul); }
                None
            }
            ENode::Recip(a) => {
                for child in self.nodes(*a) {
                    if let ENode::Recip(inner) = child { return Some(*inner); }
                    if let ENode::Sqrt(inner) = child { return Some(self.add(ENode::Rsqrt(*inner))); }
                }
                None
            }
            ENode::Add(a, b) => {
                // Associativity
                if let Some((x, y)) = self.find_add(*a) {
                    let y_plus_z = self.add(ENode::Add(y, *b));
                    let assoc = self.add(ENode::Add(x, y_plus_z));
                    if self.find(assoc) != self.find(_class_id) { return Some(assoc); }
                }
                if let Some((y, z)) = self.find_add(*b) {
                    let x_plus_y = self.add(ENode::Add(*a, y));
                    let assoc = self.add(ENode::Add(x_plus_y, z));
                    if self.find(assoc) != self.find(_class_id) { return Some(assoc); }
                }
                // Factoring
                let (ca, va) = self.get_coeff(*a);
                let (cb, vb) = self.get_coeff(*b);
                if self.find(va) == self.find(vb) {
                    let sum_c_node = self.add(ENode::constant(ca + cb));
                    return Some(self.add(ENode::Mul(sum_c_node, va)));
                }
                // FMA fusion
                if let Some((x, y)) = self.find_mul(*a) { return Some(self.add(ENode::MulAdd(x, y, *b))); }
                if let Some((x, y)) = self.find_mul(*b) { return Some(self.add(ENode::MulAdd(x, y, *a))); }
                None
            }
            ENode::MulAdd(a, _b, c) => {
                if self.contains_const(*a, 0.0) { return Some(*c); }
                None
            }
            _ => None,
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
            cost_table.get(&id).map(|(c, _)| *c).unwrap_or(usize::MAX / 2)
        };
        let op_cost = costs.node_op_cost(node);
        let child_cost: usize = node.children().iter().map(|&c| get_child_cost(c)).sum();
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
            ENode::MulRsqrt(a, b) => ExprTree::MulRsqrt(Box::new(self.extract_tree_with_costs(*a, costs)), Box::new(self.extract_tree_with_costs(*b, costs))),
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

// ============================================================================
// Expression Tree
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
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
    MulRsqrt(Box<ExprTree>, Box<ExprTree>),
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
}

#[cfg(test)]
impl ExprTree {
    pub fn eval(&self, vars: &[f32; 4]) -> f32 {
        match self {
            ExprTree::Var(i) => vars[*i as usize],
            ExprTree::Const(c) => *c,
            ExprTree::Add(a, b) => a.eval(vars) + b.eval(vars),
            ExprTree::Sub(a, b) => a.eval(vars) - b.eval(vars),
            ExprTree::Mul(a, b) => a.eval(vars) * b.eval(vars),
            ExprTree::Div(a, b) => a.eval(vars) / b.eval(vars),
            ExprTree::Neg(a) => -a.eval(vars),
            ExprTree::Recip(a) => 1.0 / a.eval(vars),
            ExprTree::Sqrt(a) => a.eval(vars).sqrt(),
            ExprTree::Rsqrt(a) => 1.0 / a.eval(vars).sqrt(),
            ExprTree::Abs(a) => a.eval(vars).abs(),
            ExprTree::Min(a, b) => a.eval(vars).min(b.eval(vars)),
            ExprTree::Max(a, b) => a.eval(vars).max(b.eval(vars)),
            ExprTree::MulAdd(a, b, c) => a.eval(vars) * b.eval(vars) + c.eval(vars),
            ExprTree::MulRsqrt(a, b) => a.eval(vars) / b.eval(vars).sqrt(),
            _ => todo!("Implement eval for more variants if needed"),
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
        // x * 5 / x -> 5
        let x = eg.add(ENode::Var(0));
        let five = eg.add(ENode::constant(5.0));
        let prod = eg.add(ENode::Mul(x, five));
        let div = eg.add(ENode::Div(prod, x));
        eg.saturate();
        assert_eq!(eg.find(div), eg.find(five));
    }
}
