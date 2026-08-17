//! Extraction: materialise a concrete arena expression from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model and materialises it as an
//! [`pixelflow_ir::ExprArena`].

use super::cost::{CostFunction, CostModel};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use crate::nnue::{EdgeAccumulator, ExprNnue};
use alloc::vec::Vec;

/// Incremental 3-pass neural extractor.
///
/// Instead of recomputing the full `EdgeAccumulator` from scratch for each
/// candidate, performs an O(Δ) swap: remove old subtree edges, add new.
/// This makes each candidate evaluation O(subtree_size) instead of O(whole_tree).
///
/// ## Algorithm
///
/// - **Pass 1 (Bootstrap)**: extract shallowest tree (minimum AST node count per
///   e-class). This is fast and gives a reasonable starting point.
/// - **Passes 2-10 (Refine)**: for each active e-class, try alternative nodes
///   using O(Δ) accumulator updates. Accept if strictly lower cost.
///   Repeat until fixpoint or `MAX_PASSES` (10) reached.
pub struct IncrementalExtractor<'a> {
    nnue: &'a ExprNnue,
    top_k: usize,
}

impl<'a> IncrementalExtractor<'a> {
    /// Create a new incremental extractor.
    ///
    /// `top_k` bounds how many alternative nodes per e-class are evaluated
    /// during refinement passes (default 8 is a good trade-off).
    pub fn new(nnue: &'a ExprNnue, top_k: usize) -> Self {
        Self { nnue, top_k }
    }

    /// Run the extraction refinement loop and return only `(cost, choices)`.
    ///
    /// The `choices` vector maps canonical e-class ID to the chosen node index.
    /// Call [`choices_to_arena`] to materialise the extracted DAG.
    pub fn extract_choices_only(
        &self,
        egraph: &EGraph,
        root_class: EClassId,
    ) -> (f32, Vec<Option<usize>>) {
        const MAX_PASSES: usize = 10;

        // Pass 1: Bootstrap with a well-founded choice per reachable e-class.
        // For unmerged classes this is the original expression's node; where
        // saturation merges reordered node lists it is the first admissible
        // node instead — "node 0 everywhere" is cyclic on merged graphs (see
        // `backfill_well_founded`), and refinement below then improves
        // whatever valid start this provides.
        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        backfill_well_founded(egraph, root_class, &mut choices);

        // Run variance analysis once — O(n) over e-graph, provides
        // per-e-class coordinate dependency info to the extraction head.
        let variance_analysis = super::deps::DepsAnalysis::analyze(egraph);

        // Build initial DAG-aware accumulator. Sharing-aware by construction:
        // shared subexpressions are counted once (computation) plus one
        // var_ref edge (register load) per later reference.
        let current_acc = EdgeAccumulator::from_dag_choices_with_variance(
            egraph,
            root_class,
            &choices,
            &self.nnue.embeddings,
            Some(&variance_analysis),
        );
        let mut current_cost = self.nnue.predict_log_cost_with_features(&current_acc);

        // Refinement passes: for each e-class, try ALL alternatives (up to top_k),
        // accept the BEST improvement (not first). Repeat until fixpoint or max passes.
        //
        // DAG-aware: each swap may change ref counts (new children may be shared
        // differently), so we rebuild the accumulator from scratch for each candidate.
        // This is O(reachable_classes) per candidate, same as the old tree-based path,
        // but now sharing-aware. True incremental updates can be added later.
        for _pass in 0..MAX_PASSES {
            let active = self.get_active_classes(egraph, root_class, &choices);
            let mut improved = false;

            for &class in &active {
                let canonical = egraph.find(class);
                let nodes = egraph.nodes(canonical);
                if nodes.len() <= 1 {
                    continue;
                }

                let current_node_idx = choices[canonical.0 as usize].unwrap_or_else(|| {
                    panic!(
                        "extract_choices_only: e-class {} is active (reachable from root) \
                         but has no recorded choice — backfill_well_founded should have \
                         populated every class returned by get_active_classes",
                        canonical.0
                    )
                });
                let candidates_to_try = nodes.len().min(self.top_k);

                // Best-improvement: evaluate ALL candidates, pick the cheapest.
                //
                // Each candidate is evaluated on a COMPLETE choice state: the
                // swap applied AND the newly reachable subtree backfilled,
                // then cycle-checked. The previous shape checked the swap
                // with unfilled classes modeled as node 0 and only backfilled
                // after acceptance — so the state it verified was not the
                // state it adopted. Cloning is O(classes), the same order as
                // the accumulator rebuild below, so this costs nothing
                // asymptotically and removes the drift.
                let mut best_swap_cost = current_cost;
                let mut best_swap: Option<Vec<Option<usize>>> = None;

                for node_idx in 0..candidates_to_try {
                    if node_idx == current_node_idx {
                        continue;
                    }

                    // Skip self-referential candidates (would create cycles).
                    if let ENode::Op { children, .. } = &nodes[node_idx] {
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            continue;
                        }
                    }

                    let mut test_choices = choices.clone();
                    test_choices[canonical.0 as usize] = Some(node_idx);
                    if let Some(ENode::Op { children, .. }) = egraph.nodes(canonical).get(node_idx)
                    {
                        for &child in children {
                            backfill_well_founded(egraph, child, &mut test_choices);
                        }
                    }
                    // The backfilled subtrees are well-founded internally,
                    // but the swapped node can still close a loop through
                    // classes that already held choices — reject those.
                    if choices_have_cycle_from(egraph, root_class, &test_choices) {
                        continue;
                    }

                    let test_acc = EdgeAccumulator::from_dag_choices_with_variance(
                        egraph,
                        root_class,
                        &test_choices,
                        &self.nnue.embeddings,
                        Some(&variance_analysis),
                    );
                    let test_cost = self.nnue.predict_log_cost_with_features(&test_acc);

                    if test_cost < best_swap_cost {
                        best_swap_cost = test_cost;
                        best_swap = Some(test_choices);
                    }
                }

                if let Some(swapped) = best_swap {
                    // Adopt EXACTLY the state that was cycle-checked and
                    // scored — no post-acceptance re-derivation.
                    choices = swapped;
                    current_cost = best_swap_cost;
                    improved = true;
                }
            }

            if !improved {
                break; // Fixpoint
            }
        }

        (current_cost, choices)
    }

    /// Walk the current best tree and collect active (reachable) e-class IDs.
    fn get_active_classes(
        &self,
        egraph: &EGraph,
        root: EClassId,
        choices: &[Option<usize>],
    ) -> Vec<EClassId> {
        use alloc::collections::BTreeSet;

        let mut active = Vec::new();
        let mut visited = BTreeSet::new();
        let mut stack = vec![root];

        while let Some(class) = stack.pop() {
            let canonical = egraph.find(class);
            if !visited.insert(canonical.0) {
                continue;
            }

            active.push(canonical);

            let node_idx = choices[canonical.0 as usize].unwrap_or_else(|| {
                panic!(
                    "get_active_classes: e-class {} reachable from root has no recorded \
                     choice — extract_choices_only must call backfill_well_founded \
                     transitively before invoking get_active_classes",
                    canonical.0
                )
            });
            let nodes = egraph.nodes(canonical);
            if node_idx < nodes.len() {
                if let ENode::Op { children, .. } = &nodes[node_idx] {
                    for &child in children {
                        stack.push(child);
                    }
                }
            }
        }

        active
    }
}

/// Fill in a **well-founded** choice for every e-class reachable from
/// `start` that doesn't yet have a recorded choice.
///
/// This restores the invariant relied on throughout `extract_choices_only`
/// and its helpers (`get_active_classes`, the refinement loop, and
/// `choices_to_arena`): every e-class reachable from the root via the
/// *currently chosen* nodes has a `Some` entry in `choices`, and the choice
/// graph is a DAG.
///
/// The previous version filled `Some(0)` (the "original/first" node) — which
/// is only the original expression's node for an UNMERGED class. After
/// saturation unions, a class's node list is a merge, so two classes can each
/// hold a node referencing the other at index 0 and the node-0 choice
/// function is CYCLIC. A cyclic bootstrap is unrecoverable downstream: every
/// refinement swap fails the cycle check (the pre-existing cycle is reachable
/// no matter what is swapped) and the cyclic set flows to `choices_to_arena`
/// (observed: a full-DEV bench run died by OOM there; the restart-DFS
/// `break_choice_cycles` repair did not terminate on the same graph).
///
/// So the fill is constructed well-founded instead of repaired after the
/// fact: Kahn-style admission. A class is admitted once ANY of its nodes has
/// all children admitted (leaves admit immediately; classes that already hold
/// a choice count as admitted, since their subtrees are complete by this same
/// invariant). The admitted node is recorded as the choice. Every well-formed
/// e-graph admits all reachable classes — each class was created holding a
/// node whose children existed before it, so creation order is a topological
/// witness — and a class that never admits is a corrupt graph, reported by
/// panic rather than papered over.
///
/// # Panics
///
/// Panics if some reachable class cannot be admitted (the e-graph holds a
/// class none of whose nodes has admissible children — a structurally corrupt
/// graph, never a rewrite judgment call).
fn backfill_well_founded(egraph: &EGraph, start: EClassId, choices: &mut [Option<usize>]) {
    let num_classes = choices.len();

    // Scope: canonical classes reachable from `start` through EVERY node of
    // each not-yet-chosen class. Classes with an existing choice are complete
    // subtrees and stop the walk.
    let mut scope_pos: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut scope: Vec<u32> = Vec::new();
    let mut stack = alloc::vec![egraph.find(start)];
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes || scope_pos[idx].is_some() || choices[idx].is_some() {
            continue;
        }
        scope_pos[idx] = Some(scope.len());
        scope.push(canonical.0);
        for node in egraph.nodes(canonical) {
            if let ENode::Op { children, .. } = node {
                for &child in children {
                    stack.push(egraph.find(child));
                }
            }
        }
    }
    if scope.is_empty() {
        return;
    }

    // Per (scope class, node): how many child references still await
    // admission. Duplicate children count once per occurrence. Reverse edges
    // record which (class, node) counters to decrement when a class admits.
    let mut pending: Vec<Vec<usize>> = Vec::with_capacity(scope.len());
    let mut reverse: Vec<Vec<(usize, usize)>> = alloc::vec![Vec::new(); scope.len()];
    let mut ready: Vec<u32> = Vec::new();
    for (pos, &cid) in scope.iter().enumerate() {
        let canonical = EClassId(cid);
        let nodes = egraph.nodes(canonical);
        let mut per_node = Vec::with_capacity(nodes.len());
        let mut any_ready = false;
        for (node_idx, node) in nodes.iter().enumerate() {
            let mut count = 0usize;
            if let ENode::Op { children, .. } = node {
                for &child in children {
                    let child_idx = egraph.find(child).0 as usize;
                    if let Some(child_pos) = scope_pos[child_idx] {
                        count += 1;
                        reverse[child_pos].push((pos, node_idx));
                    }
                }
            }
            if count == 0 {
                any_ready = true;
            }
            per_node.push(count);
        }
        pending.push(per_node);
        if any_ready {
            ready.push(cid);
        }
    }

    let mut admitted = 0usize;
    let mut queue: alloc::collections::VecDeque<u32> = ready.into_iter().collect();
    while let Some(cid) = queue.pop_front() {
        let idx = cid as usize;
        if choices[idx].is_some() {
            continue; // Admitted through an earlier queue entry.
        }
        let pos = scope_pos[idx].expect("queued class is in scope");
        let node_idx = pending[pos]
            .iter()
            .position(|&count| count == 0)
            .expect("queued class has a zero-pending node");
        choices[idx] = Some(node_idx);
        admitted += 1;
        for &(parent_pos, parent_node) in &reverse[pos] {
            pending[parent_pos][parent_node] -= 1;
            if pending[parent_pos][parent_node] == 0 {
                queue.push_back(scope[parent_pos]);
            }
        }
    }

    assert!(
        admitted == scope.len(),
        "backfill_well_founded: {} of {} reachable e-classes cannot be given a well-founded \
         choice — the e-graph holds classes none of whose nodes has admissible children, which \
         is structural corruption, not a rewrite outcome (root {})",
        scope.len() - admitted,
        scope.len(),
        start.0
    );
}

/// Extract directly into an [`pixelflow_ir::ExprArena`].
#[must_use]
pub fn extract_neural_to_arena(
    egraph: &EGraph,
    root: EClassId,
    nnue: &ExprNnue,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId, f32) {
    let extractor = IncrementalExtractor::new(nnue, 8);
    let (cost, choices) = extractor.extract_choices_only(egraph, root);
    let (arena, root_id) = choices_to_arena(egraph, root, &choices);
    (arena, root_id, cost)
}

/// Check whether the current extraction choices contain a cycle reachable from `root`.
fn choices_have_cycle_from(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> bool {
    let num_classes = egraph.num_classes();
    let mut color: Vec<u8> = alloc::vec![0; num_classes];
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes {
            continue;
        }

        if children_done {
            color[idx] = 2;
            continue;
        }

        match color[idx] {
            1 => return true,
            2 => continue,
            _ => {}
        }

        color[idx] = 1;
        stack.push((canonical, true));

        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or(0);
        if let Some(ENode::Op { children, .. }) = egraph.nodes(canonical).get(node_idx) {
            for &child in children.iter().rev() {
                stack.push((child, false));
            }
        }
    }

    false
}

/// Make an existing choice function well-founded, keeping every recorded
/// choice that participates in no cycle.
///
/// The bottom-up DP can record mutual cycles (class 68 picks `neg(69)` while
/// class 69 picks `neg(68)`): `CYCLE_COST` penalizes only SELF-references, so
/// two merged classes can each cheapest-pick a node through the other. The
/// previous repair (`break_choice_cycles`, a restart-DFS that broke one cycle
/// per pass) did not terminate on saturated FULL-tier graphs — its Strategy-3
/// fallback can leave the cycle intact, and the restart then rediscovers the
/// same cycle forever (observed live on two DEV kernels, each pinning a core
/// for minutes before the run was killed).
///
/// So, like [`backfill_well_founded`], the repair is a construction rather
/// than a patch loop — Kahn admission in two interleaved phases:
///
/// 1. **Drain**: admit every class whose RECORDED node has all children
///    admitted. Acyclic regions of the DP's choice graph are admitted here
///    unchanged, so their cost-optimal selection is kept.
/// 2. **Unstick**: when the drain stalls before every class is admitted, the
///    remaining classes all sit on cycles (or behind them). Admit ONE ready
///    class through its first admissible node — rewriting its choice — and
///    return to draining. Each unstick admits a class, so the loop is bounded
///    by the class count; total work is linear in e-graph edges.
///
/// # Panics
///
/// Panics if admission exhausts with classes left over — a class none of
/// whose nodes has admissible children is a structurally corrupt e-graph
/// (every well-formed class holds a creation-order witness node).
fn repair_choices_well_founded(egraph: &EGraph, root: EClassId, choices: &mut [Option<usize>]) {
    let num_classes = choices.len();

    // Scope: every canonical class reachable from `root` through ANY node —
    // the repair may switch a class to a node whose children the recorded
    // graph never visited, so the full downward closure is the safe scope.
    let mut scope_pos: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut scope: Vec<u32> = Vec::new();
    let mut stack = alloc::vec![egraph.find(root)];
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes || scope_pos[idx].is_some() {
            continue;
        }
        scope_pos[idx] = Some(scope.len());
        scope.push(canonical.0);
        for node in egraph.nodes(canonical) {
            if let ENode::Op { children, .. } = node {
                for &child in children {
                    stack.push(egraph.find(child));
                }
            }
        }
    }
    if scope.is_empty() {
        return;
    }

    // Pending child-admissions per (scope class, node); reverse edges say
    // which counters an admission decrements. Duplicate children count per
    // occurrence.
    let mut pending: Vec<Vec<usize>> = Vec::with_capacity(scope.len());
    let mut reverse: Vec<Vec<(usize, usize)>> = alloc::vec![Vec::new(); scope.len()];
    for (pos, &cid) in scope.iter().enumerate() {
        let nodes = egraph.nodes(EClassId(cid));
        let mut per_node = Vec::with_capacity(nodes.len());
        for (node_idx, node) in nodes.iter().enumerate() {
            let mut count = 0usize;
            if let ENode::Op { children, .. } = node {
                for &child in children {
                    let child_pos = scope_pos[egraph.find(child).0 as usize]
                        .expect("child of a scope class is in scope");
                    count += 1;
                    reverse[child_pos].push((pos, node_idx));
                }
            }
            per_node.push(count);
        }
        pending.push(per_node);
    }

    let mut admitted: Vec<bool> = alloc::vec![false; scope.len()];
    let mut admitted_count = 0usize;
    // Classes whose RECORDED node is ready (drain phase pulls from here).
    let mut recorded_ready: Vec<usize> = Vec::new();
    // Classes with ANY ready node (unstick phase pulls from here).
    let mut any_ready: Vec<usize> = Vec::new();
    for (pos, &cid) in scope.iter().enumerate() {
        let recorded = choices[cid as usize];
        for (node_idx, &count) in pending[pos].iter().enumerate() {
            if count == 0 {
                if recorded == Some(node_idx) {
                    recorded_ready.push(pos);
                }
                any_ready.push(pos);
            }
        }
    }

    // Admit `pos` through `node_idx`, propagating readiness to parents.
    let mut admit = |pos: usize,
                     node_idx: usize,
                     admitted: &mut Vec<bool>,
                     admitted_count: &mut usize,
                     recorded_ready: &mut Vec<usize>,
                     any_ready: &mut Vec<usize>,
                     pending: &mut Vec<Vec<usize>>,
                     choices: &mut [Option<usize>]| {
        admitted[pos] = true;
        *admitted_count += 1;
        choices[scope[pos] as usize] = Some(node_idx);
        for &(parent_pos, parent_node) in &reverse[pos] {
            pending[parent_pos][parent_node] -= 1;
            if pending[parent_pos][parent_node] == 0 && !admitted[parent_pos] {
                if choices[scope[parent_pos] as usize] == Some(parent_node) {
                    recorded_ready.push(parent_pos);
                }
                any_ready.push(parent_pos);
            }
        }
    };

    while admitted_count < scope.len() {
        // Phase 1 — drain: keep recorded choices wherever they admit.
        let mut progressed = false;
        while let Some(pos) = recorded_ready.pop() {
            if admitted[pos] {
                continue;
            }
            let node_idx = choices[scope[pos] as usize]
                .expect("recorded_ready holds only classes with a recorded choice");
            admit(
                pos,
                node_idx,
                &mut admitted,
                &mut admitted_count,
                &mut recorded_ready,
                &mut any_ready,
                &mut pending,
                choices,
            );
            progressed = true;
        }
        if admitted_count == scope.len() {
            break;
        }
        // Phase 2 — unstick: everything left sits on or behind a cycle of
        // recorded choices. Rewrite ONE class to its first admissible node.
        while let Some(pos) = any_ready.pop() {
            if admitted[pos] {
                continue;
            }
            let node_idx = pending[pos]
                .iter()
                .position(|&count| count == 0)
                .expect("any_ready holds only classes with a zero-pending node");
            admit(
                pos,
                node_idx,
                &mut admitted,
                &mut admitted_count,
                &mut recorded_ready,
                &mut any_ready,
                &mut pending,
                choices,
            );
            progressed = true;
            break;
        }
        assert!(
            progressed,
            "repair_choices_well_founded: {} of {} reachable e-classes cannot be admitted — \
             the e-graph holds classes none of whose nodes has admissible children, which is \
             structural corruption, not a rewrite outcome (root {})",
            scope.len() - admitted_count,
            scope.len(),
            root.0
        );
    }
}

/// Extract the minimum-cost arena expression from an e-class.
///
/// Uses dynamic programming: cost(class) = min over all nodes in class.
///
/// # Type Parameter
///
/// The cost function can be any type implementing `CostFunction`:
/// - `CostModel` for hardcoded costs
/// - Neural cost models (e.g., `ExprNnue` via adapter)
/// - Custom domain-specific cost functions
pub fn extract<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId, usize) {
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
                let this_node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => {
                        costs.node_cost(node, None)
                    }
                    ENode::Op { children, .. } => {
                        // Check for self-referential children
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_cost(node, None);
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

                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                }
            }

            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);

    // Repair any mutual cycles in the choice graph before building the tree.
    repair_choices_well_founded(egraph, root, &mut best_node);

    let (arena, root_id) = choices_to_arena(egraph, root, &best_node);
    (arena, root_id, total_cost)
}

// ============================================================================
// DAG-Aware Reference Counting (for NNUE extraction)
// ============================================================================

/// Count how many times each canonical e-class is referenced by the current
/// extraction choices, walking from `root`.
///
/// A class with `ref_count > 1` is referenced by multiple parents and should
/// be treated as shared (let-bound) in the DAG. The function uses `expanded`
/// tracking so each e-class is recursed into only once, but its count is
/// incremented every time it is referenced.
///
/// Returns a `Vec<u32>` indexed by canonical e-class ID.
pub fn compute_ref_counts(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> Vec<u32> {
    let num_classes = egraph.num_classes();
    let mut counts: Vec<u32> = alloc::vec![0u32; num_classes];
    let mut expanded: Vec<bool> = alloc::vec![false; num_classes];
    let mut stack: Vec<EClassId> = alloc::vec![root];

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes {
            continue;
        }

        counts[idx] += 1;

        // Only recurse into children on first visit (DAG, not tree).
        if !expanded[idx] {
            expanded[idx] = true;
            if let Some(node_idx) = choices[idx] {
                let nodes = egraph.nodes(canonical);
                if node_idx < nodes.len() {
                    if let ENode::Op { children, .. } = &nodes[node_idx] {
                        for &child in children {
                            stack.push(child);
                        }
                    }
                }
            }
        }
    }

    counts
}

/// Build an `ExtractedDAG` from NNUE extraction choices + reference counts.
///
/// Bridges the NNUE hill-climbing extractor (which produces per-e-class choices)
/// with DAG codegen (which needs `ExtractedDAG` with sharing info for let-bindings).
pub fn build_extracted_dag_from_choices(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
    ref_counts: &[u32],
) -> ExtractedDAG {
    let canonical_root = egraph.find(root);

    // Shared e-classes: ref_count > 1
    let shared: Vec<(EClassId, usize)> = ref_counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 1)
        .map(|(i, c)| (EClassId(i as u32), *c as usize))
        .collect();

    // Topological schedule: shared classes before their dependents (post-order).
    let mut schedule = Vec::new();
    let mut visited = alloc::vec![false; egraph.num_classes()];

    fn topo_walk(
        egraph: &EGraph,
        class: EClassId,
        choices: &[Option<usize>],
        ref_counts: &[u32],
        visited: &mut Vec<bool>,
        schedule: &mut Vec<EClassId>,
    ) {
        let canonical = egraph.find(class);
        let idx = canonical.index();
        if idx >= visited.len() || visited[idx] {
            return;
        }
        visited[idx] = true;

        if let Some(node_idx) = choices.get(idx).copied().flatten() {
            if let Some(node) = egraph.nodes(canonical).get(node_idx) {
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        topo_walk(egraph, child, choices, ref_counts, visited, schedule);
                    }
                }
            }
        }

        if ref_counts.get(idx).copied().unwrap_or(0) > 1 {
            schedule.push(canonical);
        }
    }

    topo_walk(
        egraph,
        root,
        choices,
        ref_counts,
        &mut visited,
        &mut schedule,
    );

    ExtractedDAG {
        root: canonical_root,
        shared,
        schedule,
        choices: choices.to_vec(),
        total_cost: 0,
    }
}

// ============================================================================
// Arena-Direct Extraction (EGraph → ExprArena)
// ============================================================================

/// Walk extraction choices and materialise directly into an [`pixelflow_ir::ExprArena`].
///
/// Each reachable e-class maps to exactly one [`pixelflow_ir::ExprId`]. Shared
/// e-classes naturally share `ExprId`s (DAG output — nodes are not duplicated).
///
/// ## Algorithm
///
/// Iterative post-order traversal with a `Vec<Option<ExprId>>` cache indexed by
/// canonical e-class id:
///
/// - If an e-class already has a cached `ExprId`, reuse it (O(1), `ExprId` is `Copy`).
/// - Otherwise push children for visiting (in reverse so they are processed
///   left-to-right), then push a `Complete` task for the current e-class.
/// - On `Complete`: pop the children `ExprId`s from the result stack, push a new
///   node into the arena, and record the `ExprId` in the cache.
///
/// Post-order guarantees nodes are appended in topological order (children before
/// parents), which is a requirement of [`pixelflow_ir::ExprArena`].
/// Re-pin every `Shl`/`Shr` count child to a `Const` representative.
///
/// The emitter lowers shifts to hardware immediates, so the count child MUST
/// extract as a `Const`. But a count's e-class can legitimately hold
/// arithmetic as well — a reachable `4 + 4` folds into the same class as `8`
/// — and extraction picks by COST, so a cost model that prices the `Add`
/// lower (a learned one, or any future retuning) hands codegen a non-constant
/// child and it panics. Substituting a `Const` from the same class is sound
/// by definition: same class means equal value.
///
/// # Panics
///
/// Panics if a shift-count class holds no `Const` at all. That cannot arise
/// from a well-formed arena (the count entered as a literal) and would panic
/// in the emitter regardless — failing here names the real cause.
fn pin_shift_counts(egraph: &EGraph, choices: &[Option<usize>]) -> alloc::vec::Vec<Option<usize>> {
    let mut pinned = choices.to_vec();
    for idx in 0..egraph.num_classes() {
        let canonical = egraph.find(EClassId(idx as u32));
        if canonical.0 as usize != idx {
            continue;
        }
        let Some(node_idx) = pinned.get(idx).and_then(|o| *o) else {
            continue;
        };
        let Some(ENode::Op { op, children }) = egraph.nodes(canonical).get(node_idx) else {
            continue;
        };
        if !matches!(
            op.kind(),
            pixelflow_ir::OpKind::Shl | pixelflow_ir::OpKind::Shr
        ) {
            continue;
        }
        let Some(&count) = children.get(1) else {
            continue;
        };
        let count_class = egraph.find(count);
        let ci = count_class.0 as usize;
        let count_nodes = egraph.nodes(count_class);
        if let Some(chosen) = pinned.get(ci).and_then(|o| *o)
            && matches!(count_nodes.get(chosen), Some(ENode::Const(_)))
        {
            continue;
        }
        let const_idx = count_nodes
            .iter()
            .position(|n| matches!(n, ENode::Const(_)))
            .unwrap_or_else(|| {
                panic!(
                    "pin_shift_counts: shift-count e-class {ci} holds no Const; \
                     the emitter's immediate-only shift lowering cannot be met"
                )
            });
        pinned[ci] = Some(const_idx);
    }
    pinned
}

pub fn choices_to_arena(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    use pixelflow_ir::{ExprArena, ExprId};

    // Shifts must reach codegen with a constant count — see `pin_shift_counts`.
    let pinned = pin_shift_counts(egraph, choices);
    let choices: &[Option<usize>] = &pinned;

    enum Task {
        /// Visit an e-class: push it to the result stack if cached, otherwise
        /// schedule children + a Complete task.
        Visit(EClassId),
        /// All children of this e-class have been processed; pop their ExprIds,
        /// push a new arena node, and cache the result.
        Complete { canonical_id: u32, node_idx: usize },
    }

    let num_classes = egraph.num_classes();
    let mut arena = ExprArena::with_capacity(num_classes);
    // Cache: canonical e-class id → ExprId (None = not yet visited).
    let mut id_map: Vec<Option<ExprId>> = alloc::vec![None; num_classes];
    // DFS color per canonical class: 0 = unvisited, 1 = on the current path
    // (children scheduled, Complete pending). Re-entering a gray class means
    // the choice graph reaches a class through its own descendants — a CYCLE.
    // Without this check the walk re-schedules the cycle forever and the
    // process dies by OOM instead of an error (observed: a full-DEV bench run
    // SIGKILLed at 2.7GB inside this loop). A cyclic choice set is an
    // extractor bug and must be reported as one, loudly, with the class id.
    let mut color: Vec<u8> = alloc::vec![0; num_classes];
    // Buffer identity → declared slot in the output arena. Distinct e-classes
    // can carry the same identity only if their decls differ, which is a
    // corrupt graph (one memory, two extents) — assert, never alias silently.
    let mut buffer_slots: alloc::collections::BTreeMap<
        pixelflow_ir::arena::BufferIdentity,
        (
            pixelflow_ir::arena::BufferId,
            pixelflow_ir::arena::BufferDecl,
        ),
    > = alloc::collections::BTreeMap::new();
    let mut result_stack: Vec<ExprId> = Vec::new();
    let mut task_stack: Vec<Task> = alloc::vec![Task::Visit(root)];

    while let Some(task) = task_stack.pop() {
        match task {
            Task::Visit(class) => {
                let canonical = egraph.find(class);
                let idx = canonical.0 as usize;

                // Already materialised — reuse without any clone (ExprId is Copy).
                if let Some(cached_id) = id_map.get(idx).and_then(|o| *o) {
                    result_stack.push(cached_id);
                    continue;
                }

                // No recorded choice for a reachable e-class means the extractor
                // that produced `choices` violated the invariant that every class
                // reachable from `root` (via chosen nodes) has an entry — e.g. a
                // saturation-introduced child that wasn't transitively backfilled.
                // Silently materialising node 0 here would paper over that bug by
                // emitting a node that may not even be the reachable/consistent
                // variant. Panic loudly instead so the extractor bug gets fixed
                // at the source rather than surfacing as a subtly wrong kernel.
                let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
                    panic!(
                        "choices_to_arena: e-class {} is reachable from root {} but has \
                         no recorded extraction choice — the extractor that produced \
                         `choices` must guarantee every reachable e-class has Some(idx)",
                        idx, root.0
                    )
                });

                let nodes = egraph.nodes(canonical);
                assert!(
                    node_idx < nodes.len(),
                    "choices_to_arena: node_idx {} out of bounds ({}) for e-class {}",
                    node_idx,
                    nodes.len(),
                    idx
                );
                let node = &nodes[node_idx];

                match node {
                    ENode::Var(var_idx) => {
                        let expr_id = arena.push_var(*var_idx);
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Const(bits) => {
                        let expr_id = arena.push_const(f32::from_bits(*bits));
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Buffer(decl) => {
                        // One slot per distinct identity: e-classes already
                        // dedupe equal decls, so a repeat identity here means
                        // two decls disagreeing on extents.
                        let buf_id = match buffer_slots.get(&decl.id) {
                            Some(&(buf_id, prior)) => {
                                assert!(
                                    prior == *decl,
                                    "choices_to_arena: BufferIdentity declared twice with \
                                     different extents ({prior:?} vs {decl:?})"
                                );
                                buf_id
                            }
                            None => {
                                let buf_id = arena.declare_buffer(*decl);
                                buffer_slots.insert(decl.id, (buf_id, *decl));
                                buf_id
                            }
                        };
                        let expr_id = arena.push_buffer(buf_id);
                        if idx < id_map.len() {
                            id_map[idx] = Some(expr_id);
                        }
                        result_stack.push(expr_id);
                    }
                    ENode::Op { children, .. } => {
                        assert!(
                            color[idx] != 1,
                            "choices_to_arena: extraction choices are CYCLIC — e-class {} is \
                             reached again through its own chosen descendants (root {}). The \
                             extractor that produced these choices must guarantee a \
                             well-founded choice DAG; materializing this one would loop until \
                             the process is OOM-killed",
                            idx,
                            root.0
                        );
                        color[idx] = 1;
                        // Schedule completion after children are processed.
                        task_stack.push(Task::Complete {
                            canonical_id: canonical.0,
                            node_idx,
                        });
                        // Push children in reverse so they are popped left-to-right.
                        for &child in children.iter().rev() {
                            task_stack.push(Task::Visit(child));
                        }
                    }
                }
            }

            Task::Complete {
                canonical_id,
                node_idx,
            } => {
                let idx = canonical_id as usize;

                // Another branch may have filled the cache between scheduling this
                // Complete and executing it (diamond sharing). Reuse if so.
                if let Some(cached_id) = id_map.get(idx).and_then(|o| *o) {
                    result_stack.push(cached_id);
                    continue;
                }

                let canonical = EClassId(canonical_id);
                let nodes = egraph.nodes(canonical);
                let node = &nodes[node_idx];

                let ENode::Op { op, children } = node else {
                    // Leaves are handled in Visit; reaching here would be a bug.
                    panic!(
                        "choices_to_arena: Complete task for non-Op node (e-class {})",
                        canonical_id
                    );
                };

                let arity = children.len();
                let start = result_stack.len().checked_sub(arity).unwrap_or_else(|| {
                    panic!(
                        "choices_to_arena: result_stack underflow (arity={}, len={}, e-class={})",
                        arity,
                        result_stack.len(),
                        canonical_id
                    )
                });
                let child_ids: Vec<pixelflow_ir::ExprId> = result_stack.drain(start..).collect();

                let op_kind = op.kind();

                let expr_id = match arity {
                    0 => arena.push_const(0.0), // Degenerate zero-arity Op — treat as 0.
                    1 => arena.push_unary(op_kind, child_ids[0]),
                    2 => arena.push_binary(op_kind, child_ids[0], child_ids[1]),
                    3 => arena.push_ternary(op_kind, child_ids[0], child_ids[1], child_ids[2]),
                    _ => arena.push_nary(op_kind, &child_ids),
                };

                if idx < id_map.len() {
                    id_map[idx] = Some(expr_id);
                }
                result_stack.push(expr_id);
            }
        }
    }

    let root_id = result_stack
        .pop()
        .unwrap_or_else(|| panic!("choices_to_arena: empty result stack after traversal"));
    (arena, root_id)
}

// ============================================================================
// DAG-Aware Extraction
// ============================================================================

/// Result of DAG-aware extraction with sharing information.
///
/// Unlike regular extraction which produces a tree, this tracks:
/// - Which e-classes are used multiple times (candidates for let-binding)
/// - The topological order for emission (dependencies first)
/// - The best node choice per e-class
///
/// # Example
///
/// For `sin(X) * sin(X) + sin(X)`:
/// - E-class containing `sin(X)` is used 3 times
/// - DAG extraction identifies this for let-binding
/// - Codegen emits: `let __0 = X.sin().eval(__p); (__0 * __0 + __0).eval(__p)`
#[derive(Clone, Debug)]
pub struct ExtractedDAG {
    /// The root e-class of the expression.
    pub root: EClassId,

    /// E-classes used more than once: (class_id, use_count).
    /// These are candidates for let-binding in codegen.
    pub shared: Vec<(EClassId, usize)>,

    /// Topological order for emission (dependencies before dependents).
    /// Shared e-classes appear before e-classes that use them.
    pub schedule: Vec<EClassId>,

    /// Best node choice per e-class (indexed by canonical e-class ID).
    pub choices: Vec<Option<usize>>,

    /// Total cost of the extracted expression.
    pub total_cost: usize,
}

impl ExtractedDAG {
    /// Check if an e-class is shared (used more than once).
    pub fn is_shared(&self, class: EClassId) -> bool {
        self.shared.iter().any(|(id, _)| *id == class)
    }

    /// Get the use count for an e-class.
    pub fn use_count(&self, class: EClassId) -> usize {
        self.shared
            .iter()
            .find(|(id, _)| *id == class)
            .map(|(_, count)| *count)
            .unwrap_or(1)
    }

    /// Get the index of the best node for an e-class.
    pub fn best_node_idx(&self, class: EClassId) -> Option<usize> {
        self.choices.get(class.0 as usize).and_then(|o| *o)
    }
}

/// Extract a DAG with sharing information from an e-class.
///
/// This is the DAG-aware version of `extract()`. It returns structural
/// information about sharing that codegen can use to emit let-bindings.
///
/// # Arguments
///
/// * `egraph` - The e-graph to extract from
/// * `root` - The root e-class
/// * `costs` - The cost function for choosing best nodes
///
/// # Returns
///
/// An `ExtractedDAG` containing:
/// - Best node per e-class
/// - Shared e-classes (for let-binding)
/// - Topological order for emission
pub fn extract_dag<C: CostFunction>(egraph: &EGraph, root: EClassId, costs: &C) -> ExtractedDAG {
    use alloc::collections::BTreeSet;

    const CYCLE_COST: usize = 1_000_000;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Compute best node per e-class (same as regular extraction)
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
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                let this_node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => {
                        costs.node_cost(node, None)
                    }
                    ENode::Op { children, .. } => {
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_cost(node, None);
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

                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                }
            }

            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);

    // Repair any mutual cycles in the choice graph before counting refs.
    repair_choices_well_founded(egraph, root, &mut best_node);

    // Phase 2: Count references to each e-class in the extracted DAG
    let mut ref_counts: Vec<usize> = alloc::vec![0; num_classes];
    count_refs_recursive(egraph, root, &best_node, &mut ref_counts);

    // Phase 3: Identify shared e-classes (count > 1)
    let shared: Vec<(EClassId, usize)> = ref_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 1)
        .map(|(idx, count)| (EClassId(idx as u32), *count))
        .collect();

    // Phase 4: Topological sort for emission order
    let schedule = toposort_dag(egraph, root, &best_node, &shared);

    ExtractedDAG {
        root: egraph.find(root),
        shared,
        schedule,
        choices: best_node,
        total_cost,
    }
}

/// Count references to each e-class in the extracted expression.
///
/// Uses iterative traversal with explicit stack to avoid thread stack overflow.
fn count_refs_recursive(
    egraph: &EGraph,
    class: EClassId,
    best_node: &[Option<usize>],
    ref_counts: &mut [usize],
) {
    let mut stack: Vec<EClassId> = alloc::vec![class];

    while let Some(cls) = stack.pop() {
        let canonical = egraph.find(cls);
        ref_counts[canonical.0 as usize] += 1;

        // Only recurse on first visit to count true structural refs
        if ref_counts[canonical.0 as usize] == 1 {
            if let Some(node_idx) = best_node[canonical.0 as usize] {
                let node = &egraph.nodes(canonical)[node_idx];
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        stack.push(child);
                    }
                }
            }
        }
    }
}

/// Topological sort of e-classes for emission order.
///
/// Returns e-classes in order such that dependencies come before dependents.
/// Shared e-classes are prioritized to appear early.
///
/// Uses iterative post-order traversal to avoid thread stack overflow.
fn toposort_dag(
    egraph: &EGraph,
    root: EClassId,
    best_node: &[Option<usize>],
    shared: &[(EClassId, usize)],
) -> Vec<EClassId> {
    use alloc::collections::BTreeSet;

    let shared_set: BTreeSet<u32> = shared.iter().map(|(id, _)| id.0).collect();
    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut result = Vec::new();

    // Iterative post-order: (class, children_pushed)
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        if visited.contains(&canonical.0) {
            continue;
        }

        if !children_done {
            stack.push((canonical, true));

            if let Some(node_idx) = best_node.get(canonical.0 as usize).and_then(|o| *o) {
                let node = &egraph.nodes(canonical)[node_idx];
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        let child_can = egraph.find(child);
                        if !visited.contains(&child_can.0) {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            visited.insert(canonical.0);

            if shared_set.contains(&canonical.0) {
                result.push(canonical);
            }
        }
    }

    // Add root if not already included
    let root_canonical = egraph.find(root);
    if !result.iter().any(|id| *id == root_canonical) {
        result.push(root_canonical);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_simple() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));

        let costs = CostModel::default();
        let (arena, root, cost) = extract(&egraph, x, &costs);

        assert_eq!(arena.len(), 1);
        assert_eq!(root.0, 0);
        assert_eq!(cost, 0); // Leaf nodes (Var/Const) have cost 0
    }

    #[test]
    fn extract_with_ops() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let costs = CostModel::default();
        let (arena, root, _cost) = extract(&egraph, sum, &costs);

        assert_eq!(arena.len(), 3); // Add + X + Y
        assert_eq!(root.0, 2);
    }

    #[test]
    fn extract_latency_prior_picks_cheaper_equivalent_form() {
        // x + x and x * 2 are equivalent, but under the latency-prior cost
        // model Add (4 cycles) is cheaper than Mul (5 cycles), so once the
        // two forms are unioned into one e-class, extraction must pick the
        // Add form.
        //
        // This is the extraction-side counterpart to the existing
        // NNUE latency-prior tests: it exercises `CostModel::latency_prior`
        // (the static cost table), not the neural model.
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let two = egraph.add(ENode::constant(2.0));

        let x_plus_x = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, x],
        });
        let x_times_2 = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, two],
        });

        egraph.union(x_plus_x, x_times_2);

        let costs = CostModel::latency_prior();
        assert!(
            costs.cost(pixelflow_ir::OpKind::Add) < costs.cost(pixelflow_ir::OpKind::Mul),
            "test assumes Add is strictly cheaper than Mul in the latency prior"
        );

        let (arena, root, cost) = extract(&egraph, egraph.find(x_plus_x), &costs);

        // Cheapest form is `x + x`: Add(4) + Var(0) + Var(0) = 4.
        assert_eq!(cost, costs.cost(pixelflow_ir::OpKind::Add));

        let root_node = arena.node(root);
        assert!(
            matches!(
                root_node,
                pixelflow_ir::arena::ExprNode::Binary(pixelflow_ir::OpKind::Add, _, _)
            ),
            "extraction with the latency-prior cost model should pick the Add form, got {root_node:?}"
        );
    }

    // ========================================================================
    // Cyclic choice sets (2026-08 round 1: full-DEV bench OOM)
    // ========================================================================

    /// An e-graph whose merged root class holds a node referencing a class
    /// that references it back: x unioned with neg(neg(x)). Returns
    /// (egraph, merged_root, inner_neg_class).
    fn cyclic_capable_egraph() -> (EGraph, EClassId, EClassId) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let n1 = egraph.add(ENode::Op {
            op: &super::super::ops::Neg,
            children: alloc::vec![x],
        });
        let n2 = egraph.add(ENode::Op {
            op: &super::super::ops::Neg,
            children: alloc::vec![n1],
        });
        let merged = egraph.union(x, n2); // neg(neg(x)) = x
        egraph.rebuild();
        (egraph, merged, n1)
    }

    /// Node index of the Neg op inside a class, if present.
    fn neg_index(egraph: &EGraph, class: EClassId) -> Option<usize> {
        egraph
            .nodes(egraph.find(class))
            .iter()
            .position(|n| matches!(n, ENode::Op { .. }))
    }

    #[test]
    #[should_panic(expected = "CYCLIC")]
    fn choices_to_arena_refuses_a_cyclic_choice_set() {
        // Before the gray-marking assert, this walk re-scheduled the cycle
        // forever: a full-DEV bench run grew to 2.7GB and died by SIGKILL
        // with zero diagnostics. The cycle must be a loud extractor
        // accusation instead.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));
        let _ = choices_to_arena(&egraph, merged, &choices);
    }

    #[test]
    fn neural_bootstrap_survives_a_merge_reordered_class() {
        // The real round-1 failure: the old bootstrap picked node index 0
        // everywhere, but after saturation merges "node 0" of two classes
        // can reference each other — a CYCLIC bootstrap. Every refinement
        // swap then failed `choices_have_cycle_from` (the cycle is reachable
        // no matter what is swapped), so the cyclic bootstrap flowed to
        // `choices_to_arena` unrepaired. `backfill_well_founded` must return
        // a well-founded choice set for ANY node ordering, which
        // materializes without panicking.
        let (egraph, merged, _n1) = cyclic_capable_egraph();
        let nnue = ExprNnue::new_with_latency_prior(42);
        let extractor = IncrementalExtractor::new(&nnue, 8);
        let (_cost, choices) = extractor.extract_choices_only(&egraph, merged);
        let (arena, root) = choices_to_arena(&egraph, merged, &choices);
        // The merged class IS x, so whatever form was chosen must evaluate
        // like x — and with cycles broken toward leaves, the only consistent
        // forms are Var(0) or neg-chains over it inside one arena.
        assert!(arena.len() >= 1);
        assert!(root.0 < arena.len() as u32);
    }

    #[test]
    fn repair_keeps_acyclic_choices_and_terminates_on_a_recorded_cycle() {
        // The static DP's failure shape: a recorded mutual cycle (merged
        // class picks Neg(n1), n1 picks Neg(merged)). The old restart-DFS
        // breaker could rediscover the same cycle forever — two DEV kernels
        // pinned a core for minutes. The repair must terminate, produce a
        // well-founded set, and leave already-acyclic choices alone.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));

        repair_choices_well_founded(&egraph, merged, &mut choices);
        assert!(
            !choices_have_cycle_from(&egraph, merged, &choices),
            "repair must leave a well-founded choice set"
        );
        // Materialization is the proof of well-foundedness.
        let (arena, root) = choices_to_arena(&egraph, merged, &choices);
        assert!(root.0 < arena.len() as u32);

        // And a set that is ALREADY acyclic passes through untouched.
        let mut acyclic: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        backfill_well_founded(&egraph, merged, &mut acyclic);
        let before = acyclic.clone();
        repair_choices_well_founded(&egraph, merged, &mut acyclic);
        assert_eq!(
            before, acyclic,
            "an acyclic choice function must be kept verbatim (drain phase only)"
        );
    }

    // ========================================================================
    // DAG Extraction Tests
    // ========================================================================

    #[test]
    fn extract_dag_simple() {
        // X + Y: no sharing
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, sum, &costs);

        assert!(
            dag.shared.is_empty(),
            "X + Y should have no shared subexprs"
        );
        assert_eq!(dag.root, egraph.find(sum));
    }

    #[test]
    fn extract_dag_shared_subexpr() {
        // X * X: X is used twice
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, x], // X used twice!
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, x_squared, &costs);

        // X should be marked as shared (used 2 times)
        assert!(!dag.shared.is_empty(), "X * X should have X as shared");
        assert!(dag.is_shared(x), "X should be shared");
        assert_eq!(dag.use_count(x), 2);
    }

    #[test]
    fn extract_dag_triple_use() {
        // sin(X) * sin(X) + sin(X): sin(X) used 3 times
        // We simulate this structure without actual sin
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        // Simulate sin(X) as sqrt(X) for test purposes
        let sin_x = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![x],
        });
        let sin_x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sin_x, sin_x],
        });
        let result = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![sin_x_squared, sin_x],
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, result, &costs);

        // sin_x should be shared (used 3 times: twice in Mul, once in Add)
        assert!(
            dag.is_shared(sin_x),
            "sqrt(X) should be shared (used 3 times)"
        );
        assert_eq!(dag.use_count(sin_x), 3);

        // Schedule should have sin_x before the operations that use it
        let sin_x_idx = dag.schedule.iter().position(|&id| id == egraph.find(sin_x));
        assert!(sin_x_idx.is_some(), "sin_x should be in schedule");
    }

    #[test]
    fn extract_dag_nested_sharing() {
        // (X + Y) * (X + Y): (X + Y) is shared
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });
        let product = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sum, sum], // sum used twice
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, product, &costs);

        // (X + Y) should be shared
        assert!(dag.is_shared(sum), "(X + Y) should be shared");
        assert_eq!(dag.use_count(sum), 2);
    }

    // ========================================================================
    // compute_ref_counts Tests
    // ========================================================================

    #[test]
    fn compute_ref_counts_no_sharing() {
        // X + Y: no sharing
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(sum).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let rc = compute_ref_counts(&egraph, sum, &choices);
        assert_eq!(
            rc[egraph.find(sum).0 as usize],
            1,
            "root should have ref_count 1"
        );
        assert_eq!(
            rc[egraph.find(x).0 as usize],
            1,
            "X should have ref_count 1"
        );
        assert_eq!(
            rc[egraph.find(y).0 as usize],
            1,
            "Y should have ref_count 1"
        );
    }

    #[test]
    fn compute_ref_counts_shared() {
        // X * X: X is used twice
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(x_squared).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let rc = compute_ref_counts(&egraph, x_squared, &choices);
        assert_eq!(rc[egraph.find(x_squared).0 as usize], 1, "root ref_count");
        assert_eq!(
            rc[egraph.find(x).0 as usize],
            2,
            "X should have ref_count 2"
        );
    }

    #[test]
    fn compute_ref_counts_triple_use() {
        // sqrt(X) * sqrt(X) + sqrt(X): sqrt(X) referenced 3 times
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let sqrt_x = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![x],
        });
        let product = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sqrt_x, sqrt_x],
        });
        let result = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![product, sqrt_x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(result).0 as usize] = Some(0);
        choices[egraph.find(product).0 as usize] = Some(0);
        choices[egraph.find(sqrt_x).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let rc = compute_ref_counts(&egraph, result, &choices);
        assert_eq!(
            rc[egraph.find(sqrt_x).0 as usize],
            3,
            "sqrt(X) should have ref_count 3"
        );
        assert_eq!(
            rc[egraph.find(x).0 as usize],
            1,
            "X should have ref_count 1 (only 1 parent)"
        );
    }

    #[test]
    fn dag_accumulator_handles_shared_subexpressions() {
        use crate::nnue::{EdgeAccumulator, ExprNnue};

        // sin(X) * sin(X): tree has 2x sin edges, DAG has 1x sin + 1x var_ref
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let sqrt_x = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![x],
        });
        let product = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sqrt_x, sqrt_x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(product).0 as usize] = Some(0);
        choices[egraph.find(sqrt_x).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let nnue = ExprNnue::new_with_latency_prior(42);

        // DAG accumulator
        let dag_acc =
            EdgeAccumulator::from_dag_choices(&egraph, product, &choices, &nnue.embeddings);

        assert_eq!(dag_acc.node_count, 3, "DAG acc should count 3 unique nodes");
        assert_eq!(
            dag_acc.edge_count, 3,
            "shared reuse should contribute a var_ref edge"
        );
    }

    // =========================================================================
    // Train/deploy feature-path equivalence (2026-08 round-0 skew guard)
    // =========================================================================

    /// The trainer featurizes arenas (`EdgeAccumulator::from_arena_dag`); the
    /// extractor featurizes e-graph choices
    /// (`EdgeAccumulator::from_dag_choices_with_variance`). Both are thin
    /// adapters over one walker, and this test pins that: for the same DAG —
    /// shared subexpressions, shared leaves, mixed variance — the two paths
    /// must produce bit-identical features and identical predictions. If a
    /// future change gives either path its own edge policy, scalar slots, or
    /// variance classification, this fails.
    #[test]
    fn train_and_deploy_feature_paths_agree() {
        use crate::nnue::{EdgeAccumulator, ExprNnue};
        use pixelflow_ir::{ExprArena, OpKind};

        // Arena: (sin(Z * 0.3) * (X + Y) + sin(Z * 0.3)) + Y * 0.3
        // - sin(Z * 0.3) is SHARED (register reload on the second reference)
        // - Y and 0.3 are shared leaves (leaf reload policy)
        // - variance mix: const (0.3), frame (Z chain), scanline (Y, Y*0.3),
        //   pixel (everything touching X)
        let mut arena = ExprArena::new();
        let z = arena.push_var(2);
        let c = arena.push_const(0.3);
        let zm = arena.push_binary(OpKind::Mul, z, c);
        let sin = arena.push_unary(OpKind::Sin, zm);
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let xy = arena.push_binary(OpKind::Add, x, y);
        let m = arena.push_binary(OpKind::Mul, sin, xy);
        let a = arena.push_binary(OpKind::Add, m, sin);
        let yc = arena.push_binary(OpKind::Mul, y, c);
        let root = arena.push_binary(OpKind::Add, a, yc);

        // The SAME DAG as an e-graph (sharing preserved node for node).
        use crate::egraph::ops;
        let mut eg = EGraph::new();
        let ez = eg.add(ENode::Var(2));
        let ec = eg.add(ENode::constant(0.3));
        let ezm = eg.add(ENode::Op {
            op: &ops::Mul,
            children: alloc::vec![ez, ec],
        });
        let esin = eg.add(ENode::Op {
            op: &ops::Sin,
            children: alloc::vec![ezm],
        });
        let ex = eg.add(ENode::Var(0));
        let ey = eg.add(ENode::Var(1));
        let exy = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ex, ey],
        });
        let em = eg.add(ENode::Op {
            op: &ops::Mul,
            children: alloc::vec![esin, exy],
        });
        let ea = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![em, esin],
        });
        let eyc = eg.add(ENode::Op {
            op: &ops::Mul,
            children: alloc::vec![ey, ec],
        });
        let eroot = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ea, eyc],
        });

        let mut choices: Vec<Option<usize>> = alloc::vec![None; eg.num_classes()];
        backfill_well_founded(&eg, eroot, &mut choices);

        let nnue = ExprNnue::new_with_latency_prior(7);
        let variance = crate::egraph::deps::DepsAnalysis::analyze(&eg);
        let deploy = EdgeAccumulator::from_dag_choices_with_variance(
            &eg,
            eroot,
            &choices,
            &nnue.embeddings,
            Some(&variance),
        );
        let train = EdgeAccumulator::from_arena_dag(&arena, root, &nnue.embeddings);

        assert_eq!(train.node_count, deploy.node_count, "node_count");
        assert_eq!(train.edge_count, deploy.edge_count, "edge_count");
        for i in 0..train.values.len() {
            assert_eq!(
                train.values[i], deploy.values[i],
                "accumulator values[{i}] diverge between the train-path and \
                 deploy-path feature builders"
            );
        }
        assert_eq!(train.variance_frac_const, deploy.variance_frac_const);
        assert_eq!(train.variance_frac_frame, deploy.variance_frac_frame);
        assert_eq!(train.variance_frac_scanline, deploy.variance_frac_scanline);
        assert_eq!(train.variance_frac_pixel, deploy.variance_frac_pixel);

        // Pin the variance semantics themselves, not just path symmetry:
        // 11 nodes — const {0.3}; frame {Z, Z*0.3, sin}; scanline {Y, Y*0.3};
        // pixel {X, X+Y, mul, add, root}.
        assert_eq!(train.node_count, 11);
        assert!((train.variance_frac_const - 1.0 / 11.0).abs() < 1e-6);
        assert!((train.variance_frac_frame - 3.0 / 11.0).abs() < 1e-6);
        assert!((train.variance_frac_scanline - 2.0 / 11.0).abs() < 1e-6);
        assert!((train.variance_frac_pixel - 5.0 / 11.0).abs() < 1e-6);

        // And the predictions extraction would act on are identical.
        let p_train = nnue.predict_log_cost_with_features(&train);
        let p_deploy = nnue.predict_log_cost_with_features(&deploy);
        assert_eq!(
            p_train, p_deploy,
            "identical features must produce identical predicted log-cost"
        );
        assert!(p_train.is_finite());
    }

    // =========================================================================
    // choices_to_arena tests
    // =========================================================================

    /// X + Y should produce an arena with exactly 3 nodes: Var(0), Var(1), Add.
    #[test]
    fn choices_to_arena_simple() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let add = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(add).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let (arena, root_id) = choices_to_arena(&egraph, add, &choices);

        assert_eq!(arena.len(), 3, "X + Y should have exactly 3 arena nodes");
        // Root should be the last node (post-order: X, Y, Add)
        assert_eq!(root_id.0, 2, "root ExprId should be 2 (the Add node)");
    }

    /// X * X should produce an arena with exactly 2 nodes: Var(0) and Mul.
    /// The shared Var(0) e-class must reuse one ExprId rather than being duplicated.
    #[test]
    fn choices_to_arena_shared() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let mul = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, x],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(mul).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let (arena, root_id) = choices_to_arena(&egraph, mul, &choices);

        assert_eq!(
            arena.len(),
            2,
            "X * X should have exactly 2 arena nodes (X shared)"
        );
        assert_eq!(root_id.0, 1, "root ExprId should be 1 (the Mul node)");
    }

    /// Direct extraction and explicit `choices_to_arena` should agree for tree-shaped inputs.
    #[test]
    fn extract_matches_choices_to_arena() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let add = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(add).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let (arena, root_id) = choices_to_arena(&egraph, add, &choices);
        let (extracted_arena, extracted_root, _cost) = extract(&egraph, add, &CostModel::default());
        assert_eq!(arena.len(), extracted_arena.len());
        assert_eq!(root_id, extracted_root);
    }
}
