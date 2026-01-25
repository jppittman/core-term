//! Integration between E-Graph and the Search infrastructure.
//!
//! This module implements the Category and Morphism traits for e-graph states,
//! enabling MCTS-guided exploration of rewrite sequences.
//!
//! # Design
//!
//! The e-graph presents a challenge for standard MCTS because:
//! - E-graphs are **mutable** (saturation adds equivalences)
//! - Rewrites don't create new graphs, they modify the existing one
//! - The "state" is implicitly the entire e-graph
//!
//! We solve this by:
//! 1. Using `ExprTree` (extracted expressions) as immutable states
//! 2. Defining `RewriteMove` as the morphism (rewrite at a path)
//! 3. Re-inserting into e-graph to find new equivalences
//!
//! This gives us: ExprTree --[rewrite]--> ExprTree' (via e-graph roundtrip)

use std::hash::{Hash, Hasher};

use crate::egraph::{CostModel, EGraph, ENode, EClassId, ExprTree};
use crate::search::algebra::{Category, Morphism, Valuation};

// ============================================================================
// State: Expression Tree
// ============================================================================

/// A search state is an extracted expression tree.
/// This is immutable and can be hashed/compared.
#[derive(Clone, Debug)]
pub struct SearchState {
    /// The expression tree representing this state.
    pub tree: ExprTree,
    /// Cached cost for quick access.
    pub cost: usize,
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        // Compare by tree structure
        self.tree == other.tree
    }
}

impl Eq for SearchState {}

impl Hash for SearchState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tree.hash(state);
    }
}

impl SearchState {
    /// Create a new search state from an expression tree.
    pub fn new(tree: ExprTree, costs: &CostModel) -> Self {
        let cost = tree.cost(costs);
        Self { tree, cost }
    }

    /// Create from an e-graph root, extracting the optimal tree.
    pub fn from_egraph(eg: &EGraph, root: EClassId, costs: &CostModel) -> Self {
        let tree = eg.extract_tree_with_costs(root, costs);
        Self::new(tree, costs)
    }
}

// ============================================================================
// Morphism: Rewrite Move
// ============================================================================

/// A rewrite move specifies where and what to rewrite.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RewriteMove {
    /// Path to the subexpression (0 = left, 1 = right, 2 = third child).
    pub path: Vec<u8>,
    /// Index of the rewrite rule to apply (implicit in the e-graph).
    /// For now, we just re-saturate and extract.
    pub rule_hint: u8,
}

impl RewriteMove {
    pub fn new(path: Vec<u8>) -> Self {
        Self { path, rule_hint: 0 }
    }

    pub fn root() -> Self {
        Self { path: vec![], rule_hint: 0 }
    }
}

impl Morphism for RewriteMove {
    type Object = SearchState;

    fn apply(&self, source: &Self::Object) -> Self::Object {
        // The "move" is to re-insert into e-graph, saturate, and extract.
        // This finds all equivalent forms and returns the cheapest.
        let costs = CostModel::fully_optimized();

        let mut eg = EGraph::new();
        let root = insert_tree(&mut eg, &source.tree);
        eg.saturate();

        SearchState::from_egraph(&eg, root, &costs)
    }
}

// ============================================================================
// Category: E-Graph Search Space
// ============================================================================

/// The category of e-graph states with rewrite morphisms.
pub struct EGraphCategory {
    /// Cost model for extraction.
    pub costs: CostModel,
    /// Maximum number of moves to enumerate per state.
    pub max_moves: usize,
}

impl Default for EGraphCategory {
    fn default() -> Self {
        Self {
            costs: CostModel::fully_optimized(),
            max_moves: 50,
        }
    }
}

impl Category for EGraphCategory {
    type Object = SearchState;
    type Morphism = RewriteMove;

    fn hom(&self, object: &Self::Object) -> Vec<Self::Morphism> {
        // Enumerate possible rewrite points in the tree.
        // Each node is a potential rewrite target.
        let mut moves = Vec::new();
        enumerate_paths(&object.tree, &mut vec![], &mut moves, self.max_moves);
        moves
    }

    fn is_terminal(&self, object: &Self::Object) -> bool {
        // Terminal if no rewrites improve cost, or tree is a leaf.
        matches!(object.tree, ExprTree::Var(_) | ExprTree::Const(_))
    }
}

/// Enumerate paths to all nodes in a tree.
fn enumerate_paths(tree: &ExprTree, path: &mut Vec<u8>, moves: &mut Vec<RewriteMove>, max: usize) {
    if moves.len() >= max {
        return;
    }

    // Add current node as a rewrite target
    moves.push(RewriteMove::new(path.clone()));

    // Recurse into children
    match tree {
        ExprTree::Var(_) | ExprTree::Const(_) => {}

        // Unary operations
        ExprTree::Neg(a) | ExprTree::Recip(a) | ExprTree::Sqrt(a)
        | ExprTree::Rsqrt(a) | ExprTree::Abs(a) | ExprTree::Floor(a)
        | ExprTree::Ceil(a) | ExprTree::Round(a) | ExprTree::Fract(a)
        | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
        | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a)
        | ExprTree::Exp(a) | ExprTree::Exp2(a) | ExprTree::Ln(a)
        | ExprTree::Log2(a) | ExprTree::Log10(a) => {
            path.push(0);
            enumerate_paths(a, path, moves, max);
            path.pop();
        }

        // Binary operations
        ExprTree::Add(a, b) | ExprTree::Sub(a, b) | ExprTree::Mul(a, b)
        | ExprTree::Div(a, b) | ExprTree::Min(a, b) | ExprTree::Max(a, b)
        | ExprTree::Atan2(a, b) | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b)
        | ExprTree::Lt(a, b) | ExprTree::Le(a, b) | ExprTree::Gt(a, b)
        | ExprTree::Ge(a, b) | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
            path.push(0);
            enumerate_paths(a, path, moves, max);
            path.pop();
            path.push(1);
            enumerate_paths(b, path, moves, max);
            path.pop();
        }

        // Ternary operations
        ExprTree::MulAdd(a, b, c) | ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
            path.push(0);
            enumerate_paths(a, path, moves, max);
            path.pop();
            path.push(1);
            enumerate_paths(b, path, moves, max);
            path.pop();
            path.push(2);
            enumerate_paths(c, path, moves, max);
            path.pop();
        }

        ExprTree::Tuple(elems) => {
            for (i, elem) in elems.iter().enumerate() {
                path.push(i as u8);
                enumerate_paths(elem, path, moves, max);
                path.pop();
            }
        }
    }
}

// ============================================================================
// Valuation: Cost-Based Evaluation
// ============================================================================

/// A simple cost-based valuation using the CostModel.
pub struct CostValuation {
    pub costs: CostModel,
}

impl Default for CostValuation {
    fn default() -> Self {
        Self {
            costs: CostModel::fully_optimized(),
        }
    }
}

impl Valuation<EGraphCategory> for CostValuation {
    type Value = f64;

    fn eval(&self, object: &SearchState) -> Self::Value {
        // Lower cost is better, so negate for MCTS (which maximizes).
        // We use 1/(cost+1) to keep values bounded and positive.
        1.0 / (object.cost as f64 + 1.0)
    }
}

// ============================================================================
// Helper: Insert ExprTree into EGraph
// ============================================================================

/// Insert an ExprTree into an EGraph, returning the root EClassId.
pub fn insert_tree(eg: &mut EGraph, tree: &ExprTree) -> EClassId {
    match tree {
        ExprTree::Var(i) => eg.add(ENode::Var(*i)),
        ExprTree::Const(c) => eg.add(ENode::constant(*c)),

        // Unary operations
        ExprTree::Neg(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Neg(a_id)) }
        ExprTree::Recip(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Recip(a_id)) }
        ExprTree::Sqrt(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Sqrt(a_id)) }
        ExprTree::Rsqrt(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Rsqrt(a_id)) }
        ExprTree::Abs(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Abs(a_id)) }
        ExprTree::Floor(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Floor(a_id)) }
        ExprTree::Ceil(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Ceil(a_id)) }
        ExprTree::Round(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Round(a_id)) }
        ExprTree::Fract(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Fract(a_id)) }
        ExprTree::Sin(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Sin(a_id)) }
        ExprTree::Cos(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Cos(a_id)) }
        ExprTree::Tan(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Tan(a_id)) }
        ExprTree::Asin(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Asin(a_id)) }
        ExprTree::Acos(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Acos(a_id)) }
        ExprTree::Atan(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Atan(a_id)) }
        ExprTree::Exp(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Exp(a_id)) }
        ExprTree::Exp2(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Exp2(a_id)) }
        ExprTree::Ln(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Ln(a_id)) }
        ExprTree::Log2(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Log2(a_id)) }
        ExprTree::Log10(a) => { let a_id = insert_tree(eg, a); eg.add(ENode::Log10(a_id)) }

        // Binary operations
        ExprTree::Add(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Add(a_id, b_id))
        }
        ExprTree::Sub(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Sub(a_id, b_id))
        }
        ExprTree::Mul(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Mul(a_id, b_id))
        }
        ExprTree::Div(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Div(a_id, b_id))
        }
        ExprTree::Min(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Min(a_id, b_id))
        }
        ExprTree::Max(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Max(a_id, b_id))
        }
        ExprTree::Atan2(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Atan2(a_id, b_id))
        }
        ExprTree::Pow(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Pow(a_id, b_id))
        }
        ExprTree::Hypot(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Hypot(a_id, b_id))
        }
        ExprTree::Lt(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Lt(a_id, b_id))
        }
        ExprTree::Le(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Le(a_id, b_id))
        }
        ExprTree::Gt(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Gt(a_id, b_id))
        }
        ExprTree::Ge(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Ge(a_id, b_id))
        }
        ExprTree::Eq(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Eq(a_id, b_id))
        }
        ExprTree::Ne(a, b) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            eg.add(ENode::Ne(a_id, b_id))
        }

        // Ternary operations
        ExprTree::MulAdd(a, b, c) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            let c_id = insert_tree(eg, c);
            eg.add(ENode::MulAdd(a_id, b_id, c_id))
        }
        ExprTree::Select(a, b, c) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            let c_id = insert_tree(eg, c);
            eg.add(ENode::Select(a_id, b_id, c_id))
        }
        ExprTree::Clamp(a, b, c) => {
            let a_id = insert_tree(eg, a);
            let b_id = insert_tree(eg, b);
            let c_id = insert_tree(eg, c);
            eg.add(ENode::Clamp(a_id, b_id, c_id))
        }

        ExprTree::Tuple(elems) => {
            let ids: Vec<_> = elems.iter().map(|e| insert_tree(eg, e)).collect();
            eg.add(ENode::Tuple(ids))
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
    fn test_search_state_from_egraph() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::Add(x, y));
        eg.saturate();

        let costs = CostModel::default();
        let state = SearchState::from_egraph(&eg, sum, &costs);

        assert!(state.cost > 0);
    }

    #[test]
    fn test_rewrite_move_apply() {
        // Create a state: x + 0
        let tree = ExprTree::Add(
            Box::new(ExprTree::Var(0)),
            Box::new(ExprTree::Const(0.0)),
        );
        let costs = CostModel::default();
        let state = SearchState::new(tree, &costs);

        // Apply rewrite (which saturates and extracts)
        let mv = RewriteMove::root();
        let new_state = mv.apply(&state);

        // Should simplify x + 0 → x
        assert!(new_state.cost <= state.cost);
    }

    #[test]
    fn test_category_hom() {
        let tree = ExprTree::Add(
            Box::new(ExprTree::Var(0)),
            Box::new(ExprTree::Mul(
                Box::new(ExprTree::Var(1)),
                Box::new(ExprTree::Var(2)),
            )),
        );
        let costs = CostModel::default();
        let state = SearchState::new(tree, &costs);

        let category = EGraphCategory::default();
        let moves = category.hom(&state);

        // Should have moves for: root, left(x), right(y*z), right-left(y), right-right(z)
        assert!(moves.len() >= 5);
    }

    #[test]
    fn test_cost_valuation() {
        let tree = ExprTree::Var(0);
        let costs = CostModel::default();
        let state = SearchState::new(tree, &costs);

        let val = CostValuation::default();
        let score = val.eval(&state);

        // Score should be positive and bounded
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }
}
