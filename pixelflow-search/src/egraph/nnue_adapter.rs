//! Adapter to convert between e-graph expressions and NNUE expression trees.
//!
//! Bridges the gap between `pixelflow_search::egraph` types and `pixelflow_search::nnue` types
//! for feature extraction and training data generation.
//!
//! ## Dual-Head NNUE Integration
//!
//! This module provides adapters for using [`DualHeadNnue`] with the e-graph:
//!
//! - [`NnueCostAdapter`]: Implements `CostFunction` using the value head for extraction
//! - [`expr_tree_to_nnue`]: Converts `ExprTree` to NNUE `Expr` for prediction

use crate::egraph::{EClassId, EGraph, ENode, ops};
use crate::egraph::extract::{ExprTree, Leaf};
use crate::egraph::cost::CostFunction;
use crate::egraph::node::ENode as ENodeRef;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use crate::nnue::{Expr, OpKind, DualHeadNnue};

/// Convert e-graph `Op` to `OpKind`.
///
/// Uses the `Op::kind()` method which delegates to the canonical `OpKind`.
#[inline]
pub fn op_to_nnue(op: &dyn crate::egraph::ops::Op) -> OpKind {
    op.kind()
}

/// Convert `OpKind` to e-graph `Op` reference.
///
/// Delegates to `ops::op_from_kind`.
#[inline]
pub fn nnue_to_op(kind: OpKind) -> Option<&'static dyn crate::egraph::ops::Op> {
    ops::op_from_kind(kind)
}

/// Extract a concrete `Expr` from an e-class.
/// Picks the first representative (a real implementation would use cost-based extraction).
pub fn eclass_to_expr(egraph: &EGraph, class: EClassId) -> Expr {
    let node = &egraph.nodes(class)[0];

    match node {
        ENode::Var(idx) => Expr::Var(*idx),
        ENode::Const(bits) => Expr::Const(f32::from_bits(*bits)),
        ENode::Op { op, children } => {
            let op_kind = op_to_nnue(*op);
            match children.len() {
                1 => Expr::Unary(op_kind, Box::new(eclass_to_expr(egraph, children[0]))),
                2 => Expr::Binary(
                    op_kind,
                    Box::new(eclass_to_expr(egraph, children[0])),
                    Box::new(eclass_to_expr(egraph, children[1])),
                ),
                3 => Expr::Ternary(
                    op_kind,
                    Box::new(eclass_to_expr(egraph, children[0])),
                    Box::new(eclass_to_expr(egraph, children[1])),
                    Box::new(eclass_to_expr(egraph, children[2])),
                ),
                _ => panic!("Unsupported arity {} for op {}", children.len(), op.name()),
            }
        }
    }
}

/// Convert an `ExprTree` to a NNUE `Expr` for feature extraction.
pub fn expr_tree_to_nnue(tree: &ExprTree) -> Expr {
    match tree {
        ExprTree::Leaf(Leaf::Var(i)) => Expr::Var(*i),
        ExprTree::Leaf(Leaf::Const(c)) => Expr::Const(*c),
        ExprTree::Op { op, children } => {
            let op_kind = op_to_nnue(*op);
            match children.len() {
                1 => Expr::Unary(op_kind, Box::new(expr_tree_to_nnue(&children[0]))),
                2 => Expr::Binary(
                    op_kind,
                    Box::new(expr_tree_to_nnue(&children[0])),
                    Box::new(expr_tree_to_nnue(&children[1])),
                ),
                3 => Expr::Ternary(
                    op_kind,
                    Box::new(expr_tree_to_nnue(&children[0])),
                    Box::new(expr_tree_to_nnue(&children[1])),
                    Box::new(expr_tree_to_nnue(&children[2])),
                ),
                0 => Expr::Const(0.0), // Empty children = constant
                _ => {
                    // For tuple or other variadic ops, just use first element
                    expr_tree_to_nnue(&children[0])
                }
            }
        }
    }
}

// ============================================================================
// NNUE Cost Adapter for Extraction
// ============================================================================

/// Adapter that implements `CostFunction` using `DualHeadNnue`'s value head.
///
/// This allows the e-graph extraction to use learned neural cost predictions
/// instead of hardcoded lookup tables.
///
/// ## Usage
///
/// ```ignore
/// let nnue = DualHeadNnue::new_with_latency_prior(42);
/// let adapter = NnueCostAdapter::new(&nnue);
/// let (tree, cost) = extract(&egraph, root, &adapter);
/// ```
///
/// ## How It Works
///
/// When `node_cost` is called on an `ENode`, the adapter:
/// 1. Builds an `Expr` from the node (recursively for children)
/// 2. Calls `predict_cost()` on the value head
/// 3. Returns the cost as a usize (scaled for integer comparison)
///
/// The value head predicts cost in nanoseconds (log scale internally).
pub struct NnueCostAdapter<'a> {
    /// Reference to the dual-head NNUE network.
    nnue: &'a DualHeadNnue,
    /// Scale factor for converting float cost to usize.
    /// Higher values give more precision but risk overflow.
    scale: f32,
}

impl<'a> NnueCostAdapter<'a> {
    /// Create a new adapter with the given NNUE model.
    #[must_use]
    pub fn new(nnue: &'a DualHeadNnue) -> Self {
        Self {
            nnue,
            scale: 1000.0, // 1000x scale: 1.5ns → cost 1500
        }
    }

    /// Create with a custom scale factor.
    #[must_use]
    pub fn with_scale(nnue: &'a DualHeadNnue, scale: f32) -> Self {
        Self { nnue, scale }
    }
}

impl<'a> CostFunction for NnueCostAdapter<'a> {
    fn node_cost(&self, node: &ENodeRef) -> usize {
        // For leaves, use standard zero cost
        match node {
            ENodeRef::Var(_) | ENodeRef::Const(_) => 0,
            ENodeRef::Op { op, children: _ } => {
                // For non-leaf nodes, we can't recurse into children here
                // (CostFunction is per-node, not per-tree)
                // So we use the op's latency prior as an approximation
                // The full tree cost is computed by the extraction algorithm
                // which sums node_cost over all nodes.
                //
                // Get the op's base cost from the NNUE embeddings
                let op_kind = op.kind();
                let op_idx = op_kind.index();

                // Use dimension 0 of the embedding (the latency prior)
                let latency = self.nnue.embeddings.e[op_idx][0];

                // Scale and convert to usize
                // Latencies are normalized to [0, 1], multiply by typical cycle count
                let cycles = latency * 20.0; // Max ~20 cycles
                (cycles * self.scale) as usize
            }
        }
    }

    fn cost_by_kind(&self, op: OpKind) -> usize {
        let op_idx = op.index();
        let latency = self.nnue.embeddings.e[op_idx][0];
        let cycles = latency * 20.0;
        (cycles * self.scale) as usize
    }
}

/// Predict full expression cost using the value head.
///
/// This is for cases where you have an `ExprTree` and want the
/// neural network's full prediction (not just per-node costs).
pub fn predict_tree_cost(tree: &ExprTree, nnue: &DualHeadNnue) -> f32 {
    let expr = expr_tree_to_nnue(tree);
    nnue.predict_cost(&expr)
}

/// Predict search priority using the search head.
///
/// Returns a priority value where lower = higher priority (expand first).
pub fn predict_tree_priority(tree: &ExprTree, nnue: &DualHeadNnue) -> i64 {
    let expr = expr_tree_to_nnue(tree);
    nnue.predict_priority(&expr)
}

// ============================================================================
// Neural Extraction (uses full predict_log_cost, not per-node summation)
// ============================================================================

/// Extract the minimum-cost expression using the neural network's full prediction.
///
/// Unlike the standard `extract()` which sums per-node costs (assuming sequential
/// execution), this evaluates complete candidate trees with `predict_log_cost()`.
/// This captures ILP, critical path, and other structural features that per-node
/// costs miss.
///
/// ## Algorithm
///
/// Bottom-up dynamic programming, but instead of summing child costs, we:
/// 1. For each node in an e-class, build the full subtree (using best choices for children)
/// 2. Call `predict_log_cost()` on that subtree
/// 3. Pick the node with lowest predicted cost
///
/// This is more expensive than additive extraction (O(nodes × tree_size) vs O(nodes)),
/// but gives ILP-aware cost estimates.
pub fn extract_neural(egraph: &EGraph, root: EClassId, nnue: &DualHeadNnue) -> (ExprTree, f32) {
    use alloc::collections::BTreeSet;

    const CYCLE_COST: f32 = 1_000_000.0;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<f32>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Bottom-up - compute best node for each e-class
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)];
    let mut on_stack: BTreeSet<u32> = BTreeSet::new();

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        if best_cost[canonical.0 as usize].is_some() {
            continue;
        }

        if !children_done {
            if !on_stack.insert(canonical.0) {
                continue;
            }

            stack.push((canonical, true));

            // Push children
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
            on_stack.remove(&canonical.0);

            let nodes = egraph.nodes(canonical);
            let mut min_cost = f32::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                // Build the full subtree for this node choice
                let subtree = build_subtree_with_choices(egraph, canonical, idx, &best_node);

                // Check for cycles
                if subtree.is_none() {
                    continue;
                }
                let subtree = subtree.unwrap();

                // Use the FULL neural network prediction
                let expr = expr_tree_to_nnue(&subtree);
                let cost = nnue.predict_log_cost(&expr);

                if cost < min_cost {
                    min_cost = cost;
                    min_idx = idx;
                }
            }

            if min_cost == f32::MAX {
                min_cost = CYCLE_COST;
            }

            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(CYCLE_COST);

    // Phase 2: Build final tree using best choices
    let tree = build_tree_with_choices(egraph, root, &best_node);

    (tree, total_cost)
}

/// Build a subtree using a specific node choice for the root, and best choices for descendants.
/// Uses iterative approach to avoid stack overflow.
fn build_subtree_with_choices(
    egraph: &EGraph,
    class: EClassId,
    node_idx: usize,
    best_node: &[Option<usize>],
) -> Option<ExprTree> {
    use alloc::collections::BTreeSet;

    // Task: (class, node_idx, parent_info)
    // parent_info: None for root, Some((parent_idx_in_result, child_slot))
    enum BuildTask {
        Visit { class: EClassId, node_idx: usize },
        Complete { op: &'static dyn crate::egraph::ops::Op, num_children: usize },
    }

    let mut stack: Vec<BuildTask> = vec![BuildTask::Visit { class, node_idx }];
    let mut result_stack: Vec<ExprTree> = Vec::new();
    let mut visiting: BTreeSet<u32> = BTreeSet::new();

    while let Some(task) = stack.pop() {
        match task {
            BuildTask::Visit { class, node_idx } => {
                let canonical = egraph.find(class);

                // Cycle detection
                if !visiting.insert(canonical.0) {
                    return None;
                }

                let node = &egraph.nodes(canonical)[node_idx];

                match node {
                    ENode::Var(idx) => {
                        visiting.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Var(*idx)));
                    }
                    ENode::Const(bits) => {
                        visiting.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))));
                    }
                    ENode::Op { op, children } => {
                        // Push completion task, then visit children in reverse
                        stack.push(BuildTask::Complete { op: *op, num_children: children.len() });
                        for &child in children.iter().rev() {
                            let child_canonical = egraph.find(child);
                            let child_node_idx = best_node[child_canonical.0 as usize].unwrap_or(0);
                            stack.push(BuildTask::Visit { class: child, node_idx: child_node_idx });
                        }
                    }
                }
            }
            BuildTask::Complete { op, num_children } => {
                let start = result_stack.len().saturating_sub(num_children);
                let child_trees: Vec<ExprTree> = result_stack.drain(start..).collect();
                result_stack.push(ExprTree::Op { op, children: child_trees });
            }
        }
    }

    result_stack.pop()
}

/// Build the final tree using best node choices for all e-classes.
/// Uses iterative approach to avoid stack overflow.
fn build_tree_with_choices(
    egraph: &EGraph,
    root: EClassId,
    best_node: &[Option<usize>],
) -> ExprTree {
    let root_canonical = egraph.find(root);
    let root_node_idx = best_node[root_canonical.0 as usize].unwrap_or(0);

    // Just delegate to the subtree builder
    build_subtree_with_choices(egraph, root, root_node_idx, best_node)
        .unwrap_or_else(|| ExprTree::Leaf(Leaf::Const(0.0)))
}

// ============================================================================
// Beam Search Extraction
// ============================================================================

/// Extract using beam search with full tree evaluation at each step.
///
/// Unlike bottom-up DP which assumes optimal substructure (broken for neural costs),
/// beam search evaluates full trees from root at each decision point, keeping
/// the top-k candidates.
///
/// ## Algorithm
///
/// 1. Start with root e-class, try all node choices → k candidates
/// 2. For each candidate, find next unassigned e-class (BFS from root)
/// 3. Expand: try all node choices for that e-class
/// 4. Evaluate full trees (using choice 0 for still-unassigned classes)
/// 5. Keep top-k by neural cost
/// 6. Repeat until all reachable e-classes are assigned
///
/// ## Parameters
///
/// - `beam_width`: Number of candidates to keep at each step (k)
pub fn extract_beam(
    egraph: &EGraph,
    root: EClassId,
    nnue: &DualHeadNnue,
    beam_width: usize,
) -> (ExprTree, f32) {
    let num_classes = egraph.num_classes();

    // A candidate is a partial assignment of e-class → node index
    // None means "not yet decided, use default (0)"
    type Choices = Vec<Option<usize>>;

    // Initialize beam with all choices for root e-class
    let root_canonical = egraph.find(root);
    let root_nodes = egraph.nodes(root_canonical);

    let mut beam: Vec<(Choices, f32)> = Vec::with_capacity(beam_width);

    for node_idx in 0..root_nodes.len() {
        let mut choices: Choices = vec![None; num_classes];
        choices[root_canonical.0 as usize] = Some(node_idx);

        // Evaluate full tree with this choice
        if let Some(tree) = build_tree_with_partial_choices(egraph, root, &choices) {
            let expr = expr_tree_to_nnue(&tree);
            let cost = nnue.predict_log_cost(&expr);
            beam.push((choices, cost));
        }
    }

    // Sort and truncate to beam width
    beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    beam.truncate(beam_width);

    // If beam is empty, the e-graph has issues - panic with details
    if beam.is_empty() {
        panic!(
            "extract_beam: no valid trees from root {:?}. E-graph has {} classes, root has {} nodes",
            root, num_classes, root_nodes.len()
        );
    }

    // Iteratively expand until no more unassigned e-classes
    loop {
        // Find the next unassigned e-class reachable from root (any candidate)
        let next_class = find_next_unassigned(egraph, root, &beam[0].0);

        let next_class = match next_class {
            Some(c) => c,
            None => break, // All reachable classes assigned
        };

        let next_canonical = egraph.find(next_class);
        let next_nodes = egraph.nodes(next_canonical);

        // Expand beam: for each candidate, try all choices for next_class
        let mut new_beam: Vec<(Choices, f32)> = Vec::new();

        for (choices, _old_cost) in &beam {
            for node_idx in 0..next_nodes.len() {
                let mut new_choices = choices.clone();
                new_choices[next_canonical.0 as usize] = Some(node_idx);

                // Evaluate full tree
                if let Some(tree) = build_tree_with_partial_choices(egraph, root, &new_choices) {
                    let expr = expr_tree_to_nnue(&tree);
                    let cost = nnue.predict_log_cost(&expr);
                    new_beam.push((new_choices, cost));
                }
            }
        }

        if new_beam.is_empty() {
            break; // No valid expansions (shouldn't happen)
        }

        // Sort and truncate
        new_beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        new_beam.truncate(beam_width);
        beam = new_beam;
    }

    // Best candidate is beam[0]
    let (best_choices, best_cost) = &beam[0];
    let tree = build_tree_with_partial_choices(egraph, root, best_choices)
        .unwrap_or_else(|| ExprTree::Leaf(Leaf::Const(0.0)));

    (tree, *best_cost)
}

/// Find the next unassigned e-class reachable from root in BFS order.
fn find_next_unassigned(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> Option<EClassId> {
    use alloc::collections::VecDeque;
    use alloc::collections::BTreeSet;

    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut queue: VecDeque<EClassId> = VecDeque::new();

    queue.push_back(root);

    while let Some(class) = queue.pop_front() {
        let canonical = egraph.find(class);

        if !visited.insert(canonical.0) {
            continue;
        }

        // If this class is unassigned, return it
        if choices[canonical.0 as usize].is_none() {
            return Some(canonical);
        }

        // Otherwise, add its children to the queue
        let node_idx = choices[canonical.0 as usize].unwrap_or(0);
        let node = &egraph.nodes(canonical)[node_idx];

        if let ENode::Op { children, .. } = node {
            for &child in children {
                queue.push_back(child);
            }
        }
    }

    None
}

/// Build a tree using partial choices (None → use node 0).
fn build_tree_with_partial_choices(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> Option<ExprTree> {
    use alloc::collections::BTreeSet;

    enum BuildTask {
        Visit { class: EClassId },
        Complete { op: &'static dyn crate::egraph::ops::Op, num_children: usize },
        PopPath { class_id: u32 },  // Remove from path when done with subtree
    }

    let mut stack: Vec<BuildTask> = vec![BuildTask::Visit { class: root }];
    let mut result_stack: Vec<ExprTree> = Vec::new();
    let mut on_path: BTreeSet<u32> = BTreeSet::new();  // Current path, not all visited

    while let Some(task) = stack.pop() {
        match task {
            BuildTask::Visit { class } => {
                let canonical = egraph.find(class);

                // Cycle detection: are we already on the current path?
                if on_path.contains(&canonical.0) {
                    return None;  // Cycle in this specific tree
                }

                // Use choice if assigned, else default to 0
                let node_idx = choices[canonical.0 as usize].unwrap_or(0);
                let nodes = egraph.nodes(canonical);

                if node_idx >= nodes.len() {
                    return None; // Invalid choice
                }

                let node = &nodes[node_idx];

                match node {
                    ENode::Var(idx) => {
                        result_stack.push(ExprTree::Leaf(Leaf::Var(*idx)));
                    }
                    ENode::Const(bits) => {
                        result_stack.push(ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))));
                    }
                    ENode::Op { op, children } => {
                        // Add to path before visiting children
                        on_path.insert(canonical.0);

                        // Pop from path after all children are done
                        stack.push(BuildTask::PopPath { class_id: canonical.0 });
                        stack.push(BuildTask::Complete { op: *op, num_children: children.len() });

                        for &child in children.iter().rev() {
                            stack.push(BuildTask::Visit { class: child });
                        }
                    }
                }
            }
            BuildTask::Complete { op, num_children } => {
                let start = result_stack.len().saturating_sub(num_children);
                let child_trees: Vec<ExprTree> = result_stack.drain(start..).collect();
                result_stack.push(ExprTree::Op { op, children: child_trees });
            }
            BuildTask::PopPath { class_id } => {
                on_path.remove(&class_id);
            }
        }
    }

    result_stack.pop()
}

// ============================================================================
// Expression Conversion
// ============================================================================

/// Insert an `Expr` into the e-graph, returning the root e-class.
pub fn expr_to_egraph(expr: &Expr, egraph: &mut EGraph) -> EClassId {
    match expr {
        Expr::Var(idx) => egraph.add(ENode::Var(*idx)),
        Expr::Const(val) => egraph.add(ENode::Const(val.to_bits())),
        Expr::Binary(kind, a, b) => {
            let a_class = expr_to_egraph(a, egraph);
            let b_class = expr_to_egraph(b, egraph);

            // Handle MulRsqrt decomposition
            if *kind == OpKind::MulRsqrt {
                let rsqrt_b = egraph.add(ENode::Op {
                    op: &ops::Rsqrt,
                    children: vec![b_class],
                });
                return egraph.add(ENode::Op {
                    op: &ops::Mul,
                    children: vec![a_class, rsqrt_b],
                });
            }

            let op_ref = nnue_to_op(*kind)
                .unwrap_or_else(|| panic!("Unsupported binary op: {:?}", kind));
            egraph.add(ENode::Op {
                op: op_ref,
                children: vec![a_class, b_class],
            })
        }
        Expr::Unary(kind, a) => {
            let a_class = expr_to_egraph(a, egraph);
            let op_ref = nnue_to_op(*kind)
                .unwrap_or_else(|| panic!("Unsupported unary op: {:?}", kind));
            egraph.add(ENode::Op {
                op: op_ref,
                children: vec![a_class],
            })
        }
        Expr::Ternary(kind, a, b, c) => {
            let a_class = expr_to_egraph(a, egraph);
            let b_class = expr_to_egraph(b, egraph);
            let c_class = expr_to_egraph(c, egraph);
            let op_ref = nnue_to_op(*kind)
                .unwrap_or_else(|| panic!("Unsupported ternary op: {:?}", kind));
            egraph.add(ENode::Op {
                op: op_ref,
                children: vec![a_class, b_class, c_class],
            })
        }
        Expr::Nary(kind, children) => {
            let child_classes: Vec<_> = children.iter()
                .map(|c| expr_to_egraph(c, egraph))
                .collect();
            // Currently only Tuple is supported as Nary
            match kind {
                OpKind::Tuple => {
                    // Tuple has no e-graph representation - flatten to first element or panic
                    if child_classes.is_empty() {
                        panic!("expr_to_egraph: empty Tuple not supported");
                    }
                    // For now, just return the first child (tuples aren't fully supported in e-graph yet)
                    child_classes[0]
                }
                _ => panic!("Unsupported n-ary op type: {:?}", kind),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expr_equals(a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Var(i), Expr::Var(j)) => i == j,
            (Expr::Const(x), Expr::Const(y)) => (x - y).abs() < 1e-6,
            (Expr::Unary(op1, a1), Expr::Unary(op2, a2)) => op1 == op2 && expr_equals(a1, a2),
            (Expr::Binary(op1, a1, b1), Expr::Binary(op2, a2, b2)) => {
                op1 == op2 && expr_equals(a1, a2) && expr_equals(b1, b2)
            }
            (Expr::Ternary(op1, a1, b1, c1), Expr::Ternary(op2, a2, b2, c2)) => {
                op1 == op2 && expr_equals(a1, a2) && expr_equals(b1, b2) && expr_equals(c1, c2)
            }
            (Expr::Nary(op1, c1), Expr::Nary(op2, c2)) => {
                op1 == op2 && c1.len() == c2.len() &&
                c1.iter().zip(c2.iter()).all(|(x, y)| expr_equals(x, y))
            }
            _ => false,
        }
    }

    #[test]
    fn test_op_to_nnue_roundtrip() {
        let ops_to_test: &[&dyn crate::egraph::ops::Op] = &[
            &ops::Add,
            &ops::Sub,
            &ops::Mul,
            &ops::Div,
            &ops::Neg,
            &ops::Min,
            &ops::Max,
            &ops::Sqrt,
            &ops::Rsqrt,
            &ops::Abs,
            &ops::MulAdd,
        ];
        for op in ops_to_test {
            let nnue_op = op_to_nnue(*op);
            let back = nnue_to_op(nnue_op);
            assert!(back.is_some(), "Roundtrip failed for {}", op.name());
            assert_eq!(back.unwrap().name(), op.name(), "Roundtrip failed for {}", op.name());
        }
    }

    #[test]
    fn test_eclass_to_expr_leaf() {
        let mut egraph = EGraph::new();
        let var_class = egraph.add(ENode::Var(0));
        let expr = eclass_to_expr(&egraph, var_class);
        assert!(matches!(expr, Expr::Var(0)));
    }

    #[test]
    fn test_roundtrip_simple() {
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let mut egraph = EGraph::new();
        let class = expr_to_egraph(&expr, &mut egraph);
        let recovered = eclass_to_expr(&egraph, class);
        assert!(
            expr_equals(&expr, &recovered),
            "Roundtrip failed for simple binary expression"
        );
    }

    #[test]
    fn test_roundtrip_nested() {
        // (x * 2.0) + y
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Var(1)),
        );
        let mut egraph = EGraph::new();
        let class = expr_to_egraph(&expr, &mut egraph);
        let recovered = eclass_to_expr(&egraph, class);
        assert!(
            expr_equals(&expr, &recovered),
            "Roundtrip failed for nested expression"
        );
    }

    // ========================================================================
    // NnueCostAdapter Tests
    // ========================================================================

    #[test]
    fn test_nnue_cost_adapter_basic() {
        let nnue = DualHeadNnue::new_with_latency_prior(42);
        let adapter = NnueCostAdapter::new(&nnue);

        // Var nodes should have zero cost
        let var_node = ENode::Var(0);
        assert_eq!(adapter.node_cost(&var_node), 0, "Var should have zero cost");

        // Const nodes should have zero cost
        let const_node = ENode::Const(1.0f32.to_bits());
        assert_eq!(adapter.node_cost(&const_node), 0, "Const should have zero cost");
    }

    #[test]
    fn test_nnue_cost_adapter_ops() {
        let nnue = DualHeadNnue::new_with_latency_prior(42);
        let adapter = NnueCostAdapter::new(&nnue);

        // Create an Add node (should have non-zero cost)
        let add_node = ENode::Op {
            op: &ops::Add,
            children: vec![],
        };
        let add_cost = adapter.node_cost(&add_node);
        assert!(add_cost > 0, "Add should have positive cost: {}", add_cost);

        // Div should be more expensive than Add
        let div_node = ENode::Op {
            op: &ops::Div,
            children: vec![],
        };
        let div_cost = adapter.node_cost(&div_node);
        assert!(div_cost > add_cost, "Div ({}) should cost more than Add ({})", div_cost, add_cost);
    }

    #[test]
    fn test_predict_tree_cost() {
        let nnue = DualHeadNnue::new_with_latency_prior(42);

        // Simple tree: x + y
        let tree = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Var(1)),
            ],
        };

        let cost = predict_tree_cost(&tree, &nnue);
        assert!(cost.is_finite(), "Cost should be finite");
        assert!(cost > 0.0, "Cost should be positive");
    }

    #[test]
    fn test_predict_tree_priority() {
        let nnue = DualHeadNnue::new_random(42);

        let tree = ExprTree::Op {
            op: &ops::Mul,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Var(1)),
            ],
        };

        let priority = predict_tree_priority(&tree, &nnue);
        // Just check it returns a reasonable value
        assert!(priority.abs() < i64::MAX / 2, "Priority should be reasonable");
    }

    #[test]
    fn test_nnue_cost_by_kind() {
        let nnue = DualHeadNnue::new_with_latency_prior(42);
        let adapter = NnueCostAdapter::new(&nnue);

        // Var/Const should be cheap
        let var_cost = adapter.cost_by_kind(OpKind::Var);
        let const_cost = adapter.cost_by_kind(OpKind::Const);
        assert_eq!(var_cost, 0, "Var cost should be 0");
        assert_eq!(const_cost, 0, "Const cost should be 0");

        // Div should be expensive
        let div_cost = adapter.cost_by_kind(OpKind::Div);
        assert!(div_cost > 0, "Div should have positive cost");

        // Sqrt should be expensive
        let sqrt_cost = adapter.cost_by_kind(OpKind::Sqrt);
        assert!(sqrt_cost > 0, "Sqrt should have positive cost");
    }
}
