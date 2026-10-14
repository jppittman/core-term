//! Extraction: materialise a concrete arena expression from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model and materialises it as an
//! [`pixelflow_ir::ExprArena`].

use super::and_or::{Candidate, Child, Frozen, Live, LiveTable, Problem};
use super::cost::CostFunction;
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;
use pixelflow_ir::{Children, ExprArena, ExprId, Ir, LatticeShape, Shape, Variance};

/// A witnessed choice map: an e-graph, a root e-class, and a well-founded
/// choice function from every e-class reachable from `root` to the node
/// index selected for it.
///
/// This is the reranker's state, the one choice map the search edits: a swap
/// changes one class's entry and asks whether the map still names a
/// term. "Well-founded" means every reachable e-class has a recorded choice
/// and the choice graph is acyclic (bottom-up realizable) — an unvalidated
/// `Vec<Option<usize>>` can loop forever when walked (a real 2.7GB OOM), and
/// a call site that forgets to backfill it produces that bug silently.
/// `Extraction::from_backfill` establishes it as part of construction, and
/// `Extraction::try_swap` preserves it.
///
/// What extraction *returns* is not this type: a settlement makes its term
/// directly, and a `Selection` is the term. `Extraction::to_selection`
/// converts, and the conversion cannot fail for a value of this type. See
/// docs/plans/2026-08-17-cost-model-domain.md §1 "Extraction (J2)".
pub struct Extraction<'g> {
    egraph: &'g EGraph,
    root: EClassId,
    choices: Vec<Option<usize>>,
}

impl<'g> Extraction<'g> {
    /// The hand-built path's smart constructor: fills any e-class
    /// reachable from `root` still lacking a choice via
    /// [`backfill_well_founded`], then seals the result. Tests and research
    /// harnesses that pin specific node choices come in here.
    ///
    /// # Panics
    ///
    /// Panics if the result is cyclic. A well-founded backfill cannot
    /// itself introduce a cycle — reaching this means the caller sealed a
    /// choice state that was never cycle-checked, which is a bug at the
    /// call site, not a recoverable outcome (NO SILENT FAILURES).
    pub(crate) fn from_backfill(
        egraph: &'g EGraph,
        root: EClassId,
        mut choices: Vec<Option<usize>>,
    ) -> Self {
        let root = egraph.find(root);
        backfill_well_founded(egraph, root, &mut choices);
        assert!(
            !choices_have_cycle_from(egraph, root, &choices),
            "Extraction::from_backfill: choice graph is cyclic after backfill for root {} — \
             a well-founded backfill cannot itself introduce a cycle; the caller sealed a \
             state that was never cycle-checked",
            root.0
        );
        Self {
            egraph,
            root,
            choices,
        }
    }

    /// The e-graph this extraction selects nodes from.
    pub fn egraph(&self) -> &'g EGraph {
        self.egraph
    }

    /// The extraction's (canonical) root e-class.
    pub fn root(&self) -> EClassId {
        self.root
    }

    /// The chosen node index for `class`, if `class` is reachable from
    /// [`Self::root`]. `None` for classes outside the extraction.
    pub fn choice(&self, class: EClassId) -> Option<usize> {
        let idx = self.egraph.find(class).0 as usize;
        self.choices.get(idx).copied().flatten()
    }

    /// Variance histogram (fraction const / frame-uniform / scanline-uniform
    /// / pixel-varying) of the CHOSEN nodes, not the class-wide fact
    /// (`EGraph::variance`, an intersection over every member).
    ///
    /// Materialises the term once and classifies that arena — P1(c) of
    /// docs/plans/2026-08-17-cost-model-domain.md: once a rewrite merges a
    /// pixel-varying node into a class alongside a constant one, the
    /// class-wide fact reports CONST regardless of which node the
    /// extraction actually chose, so only the materialised DAG describes
    /// what was picked.
    #[must_use]
    pub fn chosen_variance(&self) -> [f32; crate::nnue::factored::SCALAR_FEATURE_COUNT] {
        let (arena, _root) = self.to_selection().to_arena(self.egraph);
        crate::nnue::factored::variance_histogram(&arena)
    }

    /// The term this extraction names.
    ///
    /// # Panics
    ///
    /// If the map is not well-founded, which a value of this type cannot be:
    /// reaching the panic is a bug in the constructor or the swap that made
    /// it, not a recoverable outcome.
    pub(crate) fn to_selection(&self) -> Selection {
        Selection::of_map(self.egraph, self.root, &self.choices).unwrap_or_else(|refused| {
            panic!(
                "Extraction::to_selection: the map is not well-founded from root {}: {refused}",
                self.root.0
            )
        })
    }

    /// Attempt to build a candidate extraction that swaps `class`'s choice
    /// to `node_idx`, backfilling any newly-introduced children and
    /// rejecting (returning `None`) if the swap closes a cycle through
    /// already-chosen classes.
    ///
    /// This is [`IncrementalExtractor::extract_choices_only`]'s
    /// per-candidate constructor — unlike [`Extraction::from_backfill`], a
    /// cycle here is a normal search outcome (reject this candidate, try
    /// another), not a bug.
    ///
    /// ## Acyclicity check scope, and the one case it does NOT match
    /// `choices_have_cycle_from` bit-for-bit
    ///
    /// The check is scoped to `canonical`'s own new outgoing edges
    /// ([`choices_have_cycle_through`]) rather than re-walking the whole
    /// tree from `root`: this swap changes exactly one vertex's outgoing
    /// edge — `canonical`'s — plus adds fresh backfilled subtrees hanging
    /// off `node_idx`'s children, each internally acyclic by construction
    /// ([`backfill_well_founded`] never revisits a class that already has a
    /// choice, so a backfilled region can only ever terminate by joining
    /// the pre-existing tree, not by cutting back into it). PROVIDED
    /// `canonical` is itself currently reachable from `root`, a graph
    /// mutated at exactly one vertex's outgoing edge gains a cycle if and
    /// only if that vertex becomes reachable from itself through the new
    /// edge, so checking forward reachability from `node_idx`'s children
    /// back to `canonical` is equivalent to (but bounded by the
    /// forward-reachable set from the new children, not the whole
    /// root-reachable tree, unlike) a full re-walk from root.
    ///
    /// That proviso can fail: `extract_choices_only`'s refinement loop
    /// takes its `active` class list once per pass and then mutates
    /// `current_extraction` as it goes, so a class visited later in the
    /// same pass can, by the time its own `try_swap` call runs, no longer
    /// be root-reachable at all — an earlier accepted swap in the same
    /// pass severed the only path to it. For such a `canonical`, this
    /// check and a full `choices_have_cycle_from(root)` walk can disagree
    /// (this one may reject a "cycle" the root walk would call
    /// unreachable-hence-irrelevant, or the reverse). That disagreement
    /// never reaches an observable output, though: every consumer that
    /// scores or materialises a candidate walks forward from `root` only
    /// ([`Extraction::to_selection`], which is what a [`Reranker`]'s score is
    /// computed from), so a `canonical` unreachable from root is invisible
    /// to it regardless of what this function decides; accept or reject,
    /// the candidate's score is bit-identical to `current_cost` and it
    /// never wins the refinement loop's strict `<` comparison.
    pub(crate) fn try_swap(&self, class: EClassId, node_idx: usize) -> Option<Self> {
        let canonical = self.egraph.find(class);
        let mut choices = self.choices.clone();
        choices[canonical.0 as usize] = Some(node_idx);

        let new_children = operation_operands(self.egraph.nodes(canonical).get(node_idx));

        for &child in new_children {
            backfill_well_founded(self.egraph, child, &mut choices);
        }

        // A leaf swap (Var/Const/Buffer, or an Op with no children) adds no
        // outgoing edge at all, so it cannot possibly close a cycle — only
        // removes `canonical`'s old edges, which can only ever break a
        // cycle, never create one. Skip the walk entirely rather than
        // paying for a reachability check with nothing to find.
        if !new_children.is_empty() {
            let has_cycle =
                choices_have_cycle_through(self.egraph, canonical, new_children, &choices);
            if has_cycle {
                return None;
            }
        }

        Some(Self {
            egraph: self.egraph,
            root: self.root,
            choices,
        })
    }
}

// ============================================================================
// Swap-refinement search (Reranker seam)
// ============================================================================
//
// The extraction-head program that used to drive this search with a trained
// NNUE tied the static table on schedule-free kernels (workshop paper on
// branch `claude/workshop-writeup`, PR #1072, closed without merging — not
// in this tree; see docs/plans/2026-09-01-schedule-cost-model-denotation.md)
// — but JP's ruling on the
// program's SHAPE was narrower than "delete everything it touched":
// "I don't think what we have was the correct shape. I think it's right as
// an idea. [...] Egraph extraction is the place where code gen's schedule
// choice is going to go." What was wrong was a bag-of-edges MLP that
// predicted TOTAL cost and tried to replace the additive table outright;
// what was right is this local search — swap one e-class's choice, rescore,
// keep strict improvements — as the reranking primitive a future
// non-additive cost model needs. So the search stays, generic over a
// [`Reranker`] seam with no implementation shipped: the NNUE-specific
// scoring is deleted with the program that trained it, not the search that
// used to call it.

/// A pluggable scoring function for [`IncrementalExtractor`]'s
/// swap-refinement search. Lower is better — the search treats this as a
/// cost, not a probability or utility, and accepts a candidate only on a
/// strict improvement.
///
/// No implementation ships in this crate. This trait is the seam a future
/// residual reranker over the additive latency-prior table plugs into; only
/// test-only rerankers exist today (see `egraph::extract::tests`).
pub trait Reranker {
    /// Score the DAG `extraction` selects, materialised as `arena` — the
    /// arena `extraction`'s term materializes as, computed once per
    /// candidate by the search loop.
    fn score(&self, extraction: &Extraction<'_>, arena: &pixelflow_ir::ExprArena) -> f64;
}

/// Incremental swap-refinement extractor, generic over a [`Reranker`].
///
/// - **Pass 1 (Bootstrap)**: extract a well-founded starting choice per
///   e-class via [`Extraction::from_backfill`].
/// - **Passes 2..=`MAX_PASSES` (Refine)**: for each active e-class, try
///   alternative nodes (up to `top_k`) via [`Extraction::try_swap`],
///   rescoring each candidate through `reranker`. Accept the single best
///   strict improvement per class per pass. Repeat until a pass makes no
///   improvement (fixpoint) or `MAX_PASSES` is reached.
pub struct IncrementalExtractor<'a, R: Reranker + ?Sized> {
    reranker: &'a R,
    top_k: usize,
}

impl<'a, R: Reranker + ?Sized> IncrementalExtractor<'a, R> {
    /// `top_k` bounds how many alternative nodes per e-class are evaluated
    /// during refinement passes.
    pub fn new(reranker: &'a R, top_k: usize) -> Self {
        Self { reranker, top_k }
    }

    /// Run the extraction refinement loop and return `(score, extraction)`.
    ///
    /// [`Extraction::to_selection`] on the result is the extracted DAG.
    pub fn extract_choices_only<'g>(
        &self,
        egraph: &'g EGraph,
        root_class: EClassId,
    ) -> (f64, Extraction<'g>) {
        const MAX_PASSES: usize = 10;

        // Pass 1: Bootstrap with a well-founded choice per reachable e-class.
        // For unmerged classes this is the original expression's node; where
        // saturation merges reordered node lists it is the first admissible
        // node instead — refinement below then improves whatever valid
        // start this provides.
        let num_classes = egraph.num_classes();
        let choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut current_extraction = Extraction::from_backfill(egraph, root_class, choices);
        let mut current_cost = self.score(&current_extraction);

        // Refinement passes: for each e-class, try ALL alternatives (up to
        // top_k), accept the BEST improvement (not first). Repeat until
        // fixpoint or max passes.
        for _pass in 0..MAX_PASSES {
            let active = self.get_active_classes(&current_extraction);
            let mut improved = false;

            for &class in &active {
                let canonical = egraph.find(class);
                let nodes = egraph.nodes(canonical);
                if nodes.len() <= 1 {
                    continue;
                }

                let current_node_idx = current_extraction.choice(canonical).unwrap_or_else(|| {
                    panic!(
                        "extract_choices_only: e-class {} is active (reachable from root) \
                         but has no recorded choice — backfill_well_founded should have \
                         populated every class returned by get_active_classes",
                        canonical.0
                    )
                });
                let candidates_to_try = nodes.len().min(self.top_k);

                // Best-improvement: evaluate ALL candidates, pick the
                // cheapest. Each candidate is evaluated on a COMPLETE choice
                // state (the swap applied AND the newly reachable subtree
                // backfilled, cycle-checked via `Extraction::try_swap`), so
                // the state scored is the state adopted.
                let mut best_swap_cost = current_cost;
                let mut best_swap: Option<Extraction<'g>> = None;

                for (node_idx, node) in nodes.iter().enumerate().take(candidates_to_try) {
                    if node_idx == current_node_idx {
                        continue;
                    }

                    // Skip self-referential candidates (would create cycles).
                    {
                        let children = node.children_slice();
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            continue;
                        }
                    }

                    // Rejects candidates that close a cycle through classes
                    // that already held choices.
                    let Some(candidate) = current_extraction.try_swap(canonical, node_idx) else {
                        continue;
                    };

                    let test_cost = self.score(&candidate);
                    if test_cost < best_swap_cost {
                        best_swap_cost = test_cost;
                        best_swap = Some(candidate);
                    }
                }

                if let Some(swapped) = best_swap {
                    // Adopt EXACTLY the state that was cycle-checked and
                    // scored — no post-acceptance re-derivation.
                    current_extraction = swapped;
                    current_cost = best_swap_cost;
                    improved = true;
                }
            }

            if !improved {
                break; // Fixpoint
            }
        }

        (current_cost, current_extraction)
    }

    /// Materialise `extraction` and hand it to `self.reranker`.
    fn score(&self, extraction: &Extraction<'_>) -> f64 {
        let (arena, _root) = extraction.to_selection().to_arena(extraction.egraph());
        self.reranker.score(extraction, &arena)
    }

    /// Walk the current best tree and collect active (reachable) e-class IDs.
    fn get_active_classes(&self, extraction: &Extraction<'_>) -> Vec<EClassId> {
        let egraph = extraction.egraph();
        let root = extraction.root();

        let mut active = Vec::new();
        let mut visited: alloc::vec::Vec<bool> = alloc::vec![false; egraph.num_classes()];
        let mut stack = vec![root];

        while let Some(class) = stack.pop() {
            let canonical = egraph.find(class);
            let idx = canonical.0 as usize;
            if visited[idx] {
                continue;
            }
            visited[idx] = true;

            active.push(canonical);

            let node_idx = extraction.choice(canonical).unwrap_or_else(|| {
                panic!(
                    "get_active_classes: e-class {} reachable from root has no recorded \
                     choice — extract_choices_only must call backfill_well_founded \
                     transitively before invoking get_active_classes",
                    canonical.0
                )
            });
            let nodes = egraph.nodes(canonical);
            if node_idx < nodes.len() {
                {
                    let children = nodes[node_idx].children_slice();
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
/// `Extraction::to_selection`): every e-class reachable from the root via the
/// *currently chosen* nodes has a `Some` entry in `choices`, and the choice
/// graph is a DAG.
///
/// The previous version filled `Some(0)` (the "original/first" node) — which
/// is only the original expression's node for an UNMERGED class. After
/// saturation unions, a class's node list is a merge, so two classes can each
/// hold a node referencing the other at index 0 and the node-0 choice
/// function is CYCLIC. A cyclic bootstrap is unrecoverable downstream: every
/// refinement swap fails the cycle check (the pre-existing cycle is reachable
/// no matter what is swapped) and the cyclic set flows to materialization
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
            {
                let children = (node).children_slice();
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
            {
                let children = (node).children_slice();
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

        // `unwrap_or(0)` here is an ANALYSIS default, not an identity
        // sentinel, and the difference is why it stays: a class with no
        // recorded choice contributes no edge to the extracted DAG, so
        // walking its node 0 can only add phantom edges — which can only
        // report a cycle that is not there (a rejected swap, i.e. cost),
        // never miss one that is. Nothing materialised comes out of here;
        // the one place a missing choice would become a wrong *term* is
        // `Selection::of_map`, which refuses it instead.
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or(0);
        for &child in operation_operands(egraph.nodes(canonical).get(node_idx))
            .iter()
            .rev()
        {
            stack.push((child, false));
        }
    }

    false
}

/// The operand classes of an operation node, a shift's one operand included;
/// none for any other node, which is what the reachability walks below follow.
fn operation_operands(node: Option<&ENode>) -> &[EClassId] {
    match node {
        Some(ENode::Op { children, .. }) => children,
        Some(ENode::Shift { operand, .. }) => core::slice::from_ref(operand),
        _ => &[],
    }
}

/// Check whether `canonical` is forward-reachable from `new_children` by
/// following `choices`. Used by [`Extraction::try_swap`] as the equivalent,
/// but far cheaper, replacement for a full [`choices_have_cycle_from`]
/// re-walk from root — see that method's doc comment for why scoping the
/// check to the one changed vertex's new edges is sound. Plain reachability
/// (no gray/black coloring) is enough here, unlike
/// [`choices_have_cycle_from`]: we are not distinguishing "cycle" from
/// "revisited via legitimate DAG sharing" for the whole tree, only asking
/// whether ANY forward path from the new edges leads back to `canonical` —
/// which is exactly what a cycle through the swapped vertex means.
///
/// Bounded by the size of the forward-reachable set from `new_children`,
/// not `egraph.num_classes()`.
fn choices_have_cycle_through(
    egraph: &EGraph,
    canonical: EClassId,
    new_children: &[EClassId],
    choices: &[Option<usize>],
) -> bool {
    let num_classes = choices.len();
    let mut visited: alloc::collections::BTreeSet<u32> = alloc::collections::BTreeSet::new();
    let mut stack: Vec<EClassId> = new_children.to_vec();

    while let Some(class) = stack.pop() {
        let c = egraph.find(class);
        if c == canonical {
            return true;
        }
        let idx = c.0 as usize;
        if idx >= num_classes || !visited.insert(c.0) {
            continue;
        }
        // Same analysis default as `choices_have_cycle_from`, safe for the
        // same reason: extra edges can only over-report reachability, which
        // costs a rejected swap and never a wrong term.
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or(0);
        stack.extend_from_slice(operation_operands(egraph.nodes(c).get(node_idx)));
    }

    false
}

// ============================================================================
// Selection
// ============================================================================

/// The term an extraction names, as the DAG it is: one entry per distinct
/// chosen e-class, children before parents.
///
/// Denotation: a well-founded choice of one e-node per class, restricted to
/// the classes the root reaches through the nodes chosen. Everything a caller
/// asks of it is one fold over [`Self::picks`] in order: [`Self::price`] and
/// [`Self::to_arena`] each read an operand's value at a position that was
/// filled earlier in the same pass. Neither walks the e-graph to find order or
/// reachability, and neither can meet a missing choice or a cycle, because
/// neither is representable here.
///
/// The order is the left-to-right DFS post-order of the chosen DAG, first
/// occurrence winning. It is *an* order every fold needs and *the* order the
/// materializer needs: [`pixelflow_ir::ExprArena`] numbers its nodes (and
/// numbers buffer and uniform slots) in the order they are embedded, so a
/// different topological order (settle order, for one) is a different arena
/// and different bytes. The one place an unwell-founded choice can be refused
/// is the constructor that takes a hand-built map, [`Selection::of_map`].
///
/// Graph-free on purpose — an [`Optimized`](super::Optimized) owns one and is
/// cloned and stored beside graphs it does not borrow — so the folds that
/// need a node's payload take the e-graph, and check the entry they read still
/// has the shape it was selected with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Selection {
    /// The canonical root class; the last pick.
    root: EClassId,
    picks: Vec<Pick>,
    /// The picks' operands, concatenated: for each operand slot of the chosen
    /// node, the position in [`Self::picks`] of the class it names, always
    /// earlier than the pick that names it. `Mul(a, a)` has `a` twice.
    operands: Vec<usize>,
}

/// One class and the e-node chosen for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pick {
    class: EClassId,
    /// Index into the class's e-nodes.
    idx: usize,
    /// This pick's operands are `operands[previous.operands_end..operands_end]`.
    operands_end: usize,
}

/// Why a hand-built choice map names no term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// A class the root reaches through the chosen nodes has no choice.
    Missing(EClassId),
    /// A class's choice is past its last e-node.
    OutOfBounds { class: EClassId, idx: usize },
    /// A class reaches itself through the nodes chosen: no finite term.
    Cyclic(EClassId),
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Missing(class) => write!(
                f,
                "e-class {} is reachable from the root but has no recorded choice",
                class.0
            ),
            Self::OutOfBounds { class, idx } => {
                write!(f, "e-class {} chose node {idx}, past its last", class.0)
            }
            Self::Cyclic(class) => write!(
                f,
                "extraction choices are CYCLIC: e-class {} is reached again through its own \
                 chosen descendants, so the choice names no finite term",
                class.0
            ),
        }
    }
}

/// A DFS post-order under construction: the picks emitted so far and where
/// each emitted class sits.
struct PostOrder {
    picks: Vec<Pick>,
    operands: Vec<usize>,
}

impl PostOrder {
    fn new() -> Self {
        Self {
            picks: Vec::new(),
            operands: Vec::new(),
        }
    }

    /// Emit `class` with `operand_positions` as its operands; returns its
    /// position.
    fn emit(
        &mut self,
        class: EClassId,
        idx: usize,
        operand_positions: impl Iterator<Item = usize>,
    ) -> usize {
        self.operands.extend(operand_positions);
        self.picks.push(Pick {
            class,
            idx,
            operands_end: self.operands.len(),
        });
        self.picks.len() - 1
    }

    fn finish(self, root: EClassId) -> Selection {
        Selection {
            root,
            picks: self.picks,
            operands: self.operands,
        }
    }
}

impl Selection {
    /// The selection a settlement made: `pick` is the node index each settled
    /// class settled on.
    ///
    /// Walks the problem's own rows, so it needs no class-id lookup beyond
    /// matching an operand to its row, and no check: a class settles strictly
    /// after the children of the candidate it settles on, so a child is
    /// emitted before its parent, and the `expect`s below are the assertion
    /// that settlement said so.
    fn settled(problem: Problem<'_>, pick: impl Fn(Live) -> Option<usize>) -> Self {
        let (egraph, graph) = (problem.egraph(), problem.graph());
        let mut order = PostOrder::new();
        let mut position: LiveTable<Option<usize>> = graph.table(None);
        let mut entered = graph.table(false);
        let root = graph.root_live();
        entered[root] = true;
        // (class, how many of its candidate's distinct children are entered).
        let mut stack = alloc::vec![(root, 0usize)];

        while let Some(&mut (live, ref mut entered_children)) = stack.last_mut() {
            let idx = pick(live).expect("a class a chosen candidate needs is settled");
            let candidate = graph
                .candidate(live, idx)
                .expect("a settled pick is one of its class's e-nodes");
            let row = graph.children(candidate);
            if let Some(child) = row.get(*entered_children) {
                *entered_children += 1;
                if !entered[child.live] {
                    entered[child.live] = true;
                    stack.push((child.live, 0));
                }
                continue;
            }
            let class = candidate.class();
            let operands = egraph.nodes(class)[idx]
                .children_slice()
                .iter()
                .map(|&operand| {
                    let operand = egraph.find(operand);
                    let child = row
                        .iter()
                        .find(|child| graph.class(child.live) == operand)
                        .expect("a candidate's row names every class its e-node does");
                    position[child.live].expect("a settled class's children are emitted before it")
                });
            position[live] = Some(order.emit(class, idx, operands));
            stack.pop();
        }
        order.finish(graph.root())
    }

    /// The selection a hand-built choice map names, or why it names none.
    ///
    /// For a map nothing has vouched for: a test's, a research harness's, a
    /// reranker's search state. Extraction does not come through here — a
    /// settlement makes its [`Selection`] directly — and the differential
    /// assertion in `costed` checks the two constructions agree.
    ///
    /// Entries for classes the root does not reach through the chosen nodes
    /// are ignored, so a backfilled or stale map is as good as an exact one.
    pub(crate) fn of_map(
        egraph: &EGraph,
        root: EClassId,
        choices: &[Option<usize>],
    ) -> Result<Self, Refused> {
        let root = egraph.find(root);
        let mut order = PostOrder::new();
        let mut position: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let mut entered: Vec<bool> = alloc::vec![false; egraph.num_classes()];
        entered[root.index()] = true;
        // (class, chosen node index, how many operands are entered).
        let mut stack: Vec<(EClassId, usize, usize)> = Vec::new();
        let chosen = |class: EClassId| -> Result<usize, Refused> {
            let idx = choices
                .get(class.index())
                .copied()
                .flatten()
                .ok_or(Refused::Missing(class))?;
            (idx < egraph.nodes(class).len())
                .then_some(idx)
                .ok_or(Refused::OutOfBounds { class, idx })
        };
        stack.push((root, chosen(root)?, 0));

        while let Some(&mut (class, idx, ref mut entered_operands)) = stack.last_mut() {
            let operands = egraph.nodes(class)[idx].children_slice();
            if let Some(&operand) = operands.get(*entered_operands) {
                *entered_operands += 1;
                let child = egraph.find(operand);
                if !entered[child.index()] {
                    entered[child.index()] = true;
                    stack.push((child, chosen(child)?, 0));
                }
                continue;
            }
            // An operand entered but not yet emitted is an ancestor on the
            // stack: the choices close a loop through it.
            let mut positions = Vec::with_capacity(operands.len());
            for &operand in operands {
                let child = egraph.find(operand);
                positions.push(position[child.index()].ok_or(Refused::Cyclic(child))?);
            }
            position[class.index()] = Some(order.emit(class, idx, positions.into_iter()));
            stack.pop();
        }
        Ok(order.finish(root))
    }

    /// The canonical root class.
    pub(crate) fn root(&self) -> EClassId {
        self.root
    }

    /// How many classes the term has: its distinct nodes.
    pub(crate) fn len(&self) -> usize {
        self.picks.len()
    }

    /// The classes the term reaches and the node index chosen for each,
    /// children before parents.
    pub(crate) fn picks(&self) -> impl Iterator<Item = (EClassId, usize)> + '_ {
        self.picks.iter().map(|pick| (pick.class, pick.idx))
    }

    /// The node chosen at `position`, checked against `egraph`.
    ///
    /// A selection is read beside the graph it was made from; the arity check
    /// is what notices it is not, since an index into another graph's class
    /// is in bounds more often than it is the same node.
    fn node<'g>(&self, egraph: &'g EGraph, position: usize) -> &'g ENode {
        let Pick { class, idx, .. } = self.picks[position];
        let node = egraph.nodes(class).get(idx).unwrap_or_else(|| {
            panic!(
                "Selection: e-class {} has no node {idx} in this e-graph — the selection was \
                 made from another graph, or this one was rebuilt since",
                class.0
            )
        });
        assert_eq!(
            node.children_slice().len(),
            self.operands_of(position).len(),
            "Selection: the node chosen for e-class {} has a different arity in this e-graph \
             — the selection was made from another graph, or this one was rebuilt since",
            class.0
        );
        node
    }

    fn operands_of(&self, position: usize) -> &[usize] {
        let start = match position {
            0 => 0,
            _ => self.picks[position - 1].operands_end,
        };
        &self.operands[start..self.picks[position].operands_end]
    }

    /// This selection as a choice map over `num_classes` classes: `Some` for
    /// the classes it reaches.
    pub(crate) fn to_map(&self, num_classes: usize) -> Vec<Option<usize>> {
        let mut map = alloc::vec![None; num_classes];
        for (class, idx) in self.picks() {
            map[class.index()] = Some(idx);
        }
        map
    }

    /// Materialise the term as an arena, and the id of its root.
    ///
    /// One node per class, in pick order, so a class reached twice is one
    /// `ExprId` used twice. A shift is written the way the arena spells it,
    /// its count as a `Const` second operand emitted just before it.
    pub(crate) fn to_arena(&self, egraph: &EGraph) -> (ExprArena, ExprId) {
        let mut arena = ExprArena::with_capacity(self.len());
        let mut ids: Vec<ExprId> = Vec::with_capacity(self.len());
        let mut operand_ids: Vec<ExprId> = Vec::new();
        for position in 0..self.len() {
            operand_ids.clear();
            operand_ids.extend(
                self.operands_of(position)
                    .iter()
                    .map(|&operand| ids[operand]),
            );
            let shape = match self.node(egraph, position) {
                ENode::Var(var) => Shape::Var(*var),
                ENode::Const(bits) => Shape::Const(f32::from_bits(*bits)),
                // One slot per distinct identity, and the assertion that a
                // repeat identity agrees on extents, both live in
                // `ExprArena`'s `embed`: declaring a buffer is what the
                // destination representation does, not what the selection does.
                ENode::Buffer(decl) => Shape::Buffer(*decl),
                // Same rule: the decl (identity and default) is redeclared by
                // the destination arena, one slot per identity.
                ENode::Uniform(decl) => Shape::Uniform(*decl),
                // The slot index is the whole node; it means the same thing in
                // the destination arena, and the builder substitutes it there.
                ENode::Param(slot) => Shape::Param(*slot),
                // A unit comes back as its name: its body is optimized by
                // itself and linked after extraction
                // (`pixelflow_ir::passes::link`).
                ENode::Ref { key, .. } => Shape::Ref(*key),
                ENode::Op { op, .. } => Shape::Op(*op, Children::Many(&operand_ids)),
                ENode::Shift {
                    direction, count, ..
                } => {
                    operand_ids.push(arena.embed(Shape::Const(count.literal())));
                    Shape::Op(direction.op(), Children::Many(&operand_ids))
                }
                ENode::Reduce { fold, .. } => Shape::Reduce {
                    fold: *fold,
                    body: operand_ids[0],
                },
            };
            ids.push(arena.embed(shape));
        }
        let root = *ids.last().expect("a selection holds at least its root");
        (arena, root)
    }

    /// The price of the term, in both shapes that matter.
    ///
    /// Prices *this* term and minimizes nothing: no node is reconsidered. A
    /// node's op cost is weighted by [`LatticeShape::evals`] of the variance
    /// of the form chosen below it, so a subexpression that depends on
    /// nothing is priced once per call and an X-dependent one once per
    /// sample. Leaves are free (`CostModel::node_op_cost`), so under
    /// [`LatticeShape::POINT`] `ChoiceCost::dag` equals the latency-prior
    /// cost of the arena [`Self::to_arena`] builds from the same term.
    pub(crate) fn price<C: CostFunction>(
        &self,
        egraph: &EGraph,
        pricing: &Pricing<'_, C>,
    ) -> ChoiceCost {
        let Pricing { costs, shape } = *pricing;
        let mut tree: Vec<usize> = Vec::with_capacity(self.len());
        let mut variance: Vec<Variance> = Vec::with_capacity(self.len());
        let mut dag = 0usize;
        for position in 0..self.len() {
            let node = self.node(egraph, position);
            let operands = self.operands_of(position);
            let node_variance = node.variance(|child| {
                let child = egraph.find(child);
                let at = operands
                    .iter()
                    .find(|&&operand| self.picks[operand].class == child)
                    .expect("a chosen node's operands are picks");
                variance[*at]
            });
            let own = weighted_own(costs, node, shape.evals(node_variance));
            // A fold evaluates its body once per index — see `fold_body_multiple`.
            let per_child = fold_body_multiple(node);
            // Saturating throughout: a tree cost is exponential in the sharing
            // it refuses to price, so it reaches the ceiling on real inputs,
            // and a fold whose monoid has no combiner is priced
            // `usize::MAX / 4` (`CostModel::node_op_cost`).
            let below = operands
                .iter()
                .map(|&operand| {
                    usize::try_from((tree[operand] as u64).saturating_mul(per_child))
                        .unwrap_or(usize::MAX)
                })
                .fold(0usize, usize::saturating_add);
            tree.push(own.saturating_add(below));
            variance.push(node_variance);
            dag = dag.saturating_add(own);
        }
        ChoiceCost {
            tree: *tree.last().expect("a selection holds at least its root"),
            dag,
        }
    }
}

// ============================================================================
// DAG-Aware Extraction
// ============================================================================

/// What an extraction settled on: the node chosen for every class it
/// settled, the price of the term the root reaches through them, and which
/// objective found it.
///
/// The term itself is a `Selection`, not readable outside the crate. The
/// settlement that chose it made it once, and pricing and materializing are
/// folds over it. In the crate, `ExtractedDAG::to_arena` materializes it;
/// outside, the only materializer is [`extract`], pending a decision on a
/// public `ExtractedDAG::to_arena`. [`Self::choices`] is the settlement's
/// record, kept for the callers that read it.
#[derive(Clone, Debug)]
pub struct ExtractedDAG {
    /// The root e-class of the expression.
    pub root: EClassId,

    /// The node settled on for each class settlement reached, by canonical
    /// e-class id: a superset of the term, which holds only the classes the
    /// root reaches through the nodes chosen. Nothing materializes from this
    /// field; editing it does not change the term.
    pub choices: Vec<Option<usize>>,

    /// The term: the classes the root reaches through [`Self::choices`],
    /// children before parents.
    pub(super) selection: Selection,

    /// **Tree** cost of the term: every child summed at every use, so
    /// sharing is never priced. This is the objective the
    /// extraction DP minimizes, and it is *not* what the emitted kernel pays
    /// — see [`Self::dag_cost`], which is the number a caller asking "what
    /// will this kernel cost?" wants. On `shader:julia_set` the two are
    /// ~1.4e7 and 716 (`docs/results/2026-09-02-extraction-gap.md`).
    ///
    /// Read from the term returned, so it describes the term this struct
    /// returns (#1111). Before that fix it was the DP's own table total after
    /// a repair had rewritten picks under it, which named a different term on
    /// 132 of 302 measured kernels.
    pub total_cost: usize,

    /// **DAG** cost of the term: each distinct chosen e-class priced once,
    /// which is what the emitted kernel pays. The term materializes as one
    /// arena node per reachable e-class and codegen let-binds the shared
    /// ones, so under
    /// [`LatticeShape::POINT`] this equals the latency-prior cost of that
    /// arena — the property every measurement in this repo assumes when it
    /// re-costs the materialized arena instead of reading a field here.
    ///
    /// The DP does not minimize this (#1116); it is the honest price of what
    /// the DP happened to choose.
    pub dag_cost: usize,

    /// Which objective the term came from, and what the
    /// sharing-aware pass cost to find out. A number quoted from this struct
    /// without its objective is a number from an unknown extractor.
    pub report: ExtractionReport,
}

/// Which of `extract_dag_scoped`'s objectives produced the returned term.
///
/// The two-objective no-regression property (the sharing-aware term is
/// returned only when it is cheaper by true [`ChoiceCost::dag`], else the
/// tree term) holds for [`Self::Shared`] and [`Self::TreeCheaper`], where
/// both objectives ran. It is **not attempted** for [`Self::TreeOnly`]: the
/// sharing-aware pass was abandoned at [`SHARED_DAG_PASS_BYTE_BUDGET`] and
/// the tree term is all there is. That case is loud by construction — it
/// is a variant, not a silently identical `Vec<Option<usize>>` — so a
/// measurement above the budget can never be quoted as if it were on the
/// production objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractionObjective {
    /// Both objectives ran; the sharing-aware term was cheaper.
    Shared,
    /// Both objectives ran; the tree term was no dearer (ties go to it).
    TreeCheaper,
    /// Only the tree objective ran: the sharing-aware pass's reach sets
    /// outgrew [`SHARED_DAG_PASS_BYTE_BUDGET`] and it was abandoned.
    TreeOnly,
    /// The choices were supplied from outside the two-objective DP — a
    /// `Reranker`'s. Neither objective's pass ran.
    External,
}

impl ExtractionObjective {
    /// The name the telemetry record and the measurement harnesses print.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::TreeCheaper => "tree_cheaper",
            Self::TreeOnly => "tree_only",
            Self::External => "external",
        }
    }
}

/// What `shared_dag_dp_pass` cost, whether or not it finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedPassStats {
    /// Classes reachable from the root — the set the pass's reach sets
    /// range over. On a saturated glyph a third of the e-graph.
    pub live_classes: usize,
    /// Bytes the reach sets held when the pass ended: at its completion,
    /// or at the point it crossed [`SHARED_DAG_PASS_BYTE_BUDGET`] and was
    /// abandoned (then the first total above the budget). A deterministic
    /// function of the e-graph, so two hosts report the same number.
    pub reach_bytes: usize,
}

/// The objective behind an [`ExtractedDAG`], with the sharing-aware pass's
/// accounting beside it. Carried by [`ExtractedDAG::report`] and
/// [`Optimized::extraction`](super::Optimized::extraction), and emitted by
/// the `saturation-telemetry` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractionReport {
    /// Which term was returned.
    pub objective: ExtractionObjective,
    /// The sharing-aware pass's accounting; `None` only under
    /// [`ExtractionObjective::External`], where no pass ran.
    pub shared_pass: Option<SharedPassStats>,
    /// What the winning arm's DP believed it had chosen, against the term it
    /// actually named. `None` only under [`ExtractionObjective::External`],
    /// where no DP ran. See [`ClaimAudit`].
    pub audit: Option<ClaimAudit>,
}

impl ExtractionReport {
    /// Choices supplied from outside the DP: no objective, no pass.
    #[must_use]
    pub fn external() -> Self {
        Self {
            objective: ExtractionObjective::External,
            shared_pass: None,
            audit: None,
        }
    }
}

/// Which column of [`ChoiceCost`] a DP arm's own minimized value is on.
///
/// The two arms of `extract_dag_scoped` minimize *different quantities* —
/// `tree_dp_pass` a tree cost, `shared_dag_dp_pass` a DAG cost — and a
/// claim read off one arm's table means nothing beside the other column. The
/// scale is carried rather than inferred from [`ExtractionObjective`] because
/// a number whose units live in a comment is a number something will
/// eventually compare wrongly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostScale {
    /// [`ChoiceCost::tree`] — every child summed at every use.
    Tree,
    /// [`ChoiceCost::dag`] — each distinct chosen e-class priced once.
    Dag,
}

impl CostScale {
    /// The column of `cost` this scale names.
    #[must_use]
    pub fn of(self, cost: ChoiceCost) -> usize {
        match self {
            Self::Tree => cost.tree,
            Self::Dag => cost.dag,
        }
    }

    /// The name the measurement harnesses print.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tree => "tree",
            Self::Dag => "dag",
        }
    }
}

/// The winning DP arm's own minimized value at the root, and the scale it is
/// on — the objective the extractor **minimized**, beside the price the
/// kernel **pays** ([`ExtractedDAG::dag_cost`]).
///
/// Those have to be the same number. `settle_in_cost_order` settles a class
/// strictly after the children of the candidate it settles on, so its map is
/// well-founded, nothing downstream rewrites a pick, and every reach set is
/// final before a parent reads it. When they are not the same number the
/// extractor is minimizing something no one pays — which is what the DFS
/// post-order it replaced did, claiming **281** for a chrome term costing
/// **4,564,003,324** at a 50,000-class cap
/// (`docs/results/2026-09-08-cse-mispricing.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimAudit {
    /// The winning arm's minimized value at the root, read from its own DP.
    pub claimed: usize,
    /// Which of [`ChoiceCost`]'s columns [`Self::claimed`] is comparable to.
    pub scale: CostScale,
}

impl ClaimAudit {
    /// The signed error of the claim against the term actually returned:
    /// `claimed - actual` on [`Self::scale`]'s column. Negative means the DP
    /// believed the term cheaper than it is — the direction in which more
    /// graph buys a worse choice.
    #[must_use]
    pub fn signed_error(&self, cost: ChoiceCost) -> i128 {
        self.claimed as i128 - self.scale.of(cost) as i128
    }
}

impl ExtractedDAG {
    /// Materialise the term as an arena, in `egraph`, the graph it was
    /// extracted from.
    pub(crate) fn to_arena(&self, egraph: &EGraph) -> (ExprArena, ExprId) {
        self.selection.to_arena(egraph)
    }

    /// The two reported costs as the pair they are.
    #[must_use]
    pub fn cost(&self) -> ChoiceCost {
        ChoiceCost {
            tree: self.total_cost,
            dag: self.dag_cost,
        }
    }
}

/// The variance of one e-node, given the variance already chosen for the
/// classes below it — [`ENode::variance`], the transfer function the
/// e-graph's class fact is seeded by, fed the chosen forms instead. A fold's
/// bound index drops out, which is what makes `Σ_i f(i)` frame-uniform when
/// `f` reads nothing but the index, and therefore hoistable out of the pixel
/// loop.
///
/// A node reading its own class (a cycle) counts as fully
/// varying — the conservative direction, since it can only make a form look
/// more expensive, never less.
fn node_variance(
    egraph: &EGraph,
    node: &ENode,
    best_var: &[Variance],
    canonical: EClassId,
) -> Variance {
    if node
        .children_slice()
        .iter()
        .any(|&child| egraph.find(child) == canonical)
    {
        return Variance::ALL;
    }
    node.variance(|child| best_var[egraph.find(child).index()])
}

/// The cost of one *settled* extraction, in both of the shapes that matter.
///
/// The two numbers differ by exactly how much sharing the choice function
/// induces, and the gap is not cosmetic: on `shader:julia_set` the tree cost
/// is ~1.4e7 against a DAG cost of 716, a 20,000x sharing ratio
/// (`docs/results/2026-09-02-extraction-gap.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChoiceCost {
    /// Every child summed at every use — **sharing is never priced**, so a
    /// subterm reached ten times is paid ten times. This is the objective
    /// `extract_dag_scoped`'s DP minimizes, which is why it is reported:
    /// comparing the DP against a reference means comparing this number.
    pub tree: usize,

    /// Each distinct chosen e-class priced **once** — what the emitted kernel
    /// actually pays, since the term materializes as exactly one arena node
    /// per reachable e-class and codegen let-binds the shared ones.
    ///
    /// A caller asking "what will this kernel cost?" wants this number.
    pub dag: usize,
}

/// The production extraction of `root`, priced at [`LatticeShape::POINT`].
pub fn extract_dag<C: CostFunction>(egraph: &EGraph, root: EClassId, costs: &C) -> ExtractedDAG {
    extract_dag_scoped(egraph, root, costs, LatticeShape::POINT)
}

/// The arena [`extract_dag`] selects, with the tree cost it reports: the
/// production extractor's choices through the production seal.
///
/// Not a second extractor: it owns no extraction. It exists as the
/// materializer for a caller holding a saturated graph and a
/// [`CostFunction`] the [`CostModel`](super::cost::CostModel) table cannot
/// express, which [`Optimized::to_arena`](super::optimizer::Optimized::to_arena)
/// cannot take (`pixelflow-codegen`'s `halve_fold_jit` prices loops out).
/// Its name and shape are pending the decision on a public materializer for
/// an [`ExtractedDAG`]; in-crate callers use [`extract_dag`] and
/// `ExtractedDAG::to_arena` directly.
pub fn extract<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId, usize) {
    let dag = extract_dag(egraph, root, costs);
    let (arena, arena_root) = dag.to_arena(egraph);
    (arena, arena_root, dag.total_cost)
}

/// [`extract_dag`], pricing each node by how often the lattice evaluates it.
///
/// The cost of a program is not the cost of its text but of its execution:
/// a node's op cost is multiplied by [`LatticeShape::evals`] of the variance
/// of the form chosen for it, so a subexpression that depends on nothing (a
/// uniform's arithmetic) is priced once per call, one that depends only on Y
/// once per row, and one that touches X once per sample. Every extent is
/// known at compile time, so the weight is a count rather than an ordinal
/// preference — except inside a surviving fold, where `evals` prices a node
/// that reads the binder as per-sample and without the fold's trip count.
///
/// That single change is what makes extraction the thing that *decides* the
/// factorization: given `(X + u) + u` for a uniform `u`, and its
/// reassociation `X + (u + u)`, both two adds, the second leaves one of them
/// outside the pixel loop and is therefore cheaper by a factor of the frame —
/// which loop-invariant code motion after the fact can only discover, never
/// choose between.
///
/// [`LatticeShape::POINT`] weights everything by one, so `extract_dag`'s
/// behavior is unchanged.
///
/// # The objective (#1116)
///
/// The DP this used to be summed each child's `best_cost`, which is a
/// **tree** cost: a class used ten times was charged ten times in the
/// objective and emitted once in the kernel, so the thing minimized was not
/// the thing paid. On `shader:julia_set` the two numbers were ~1.4e7 and 716.
///
/// The fix is to carry, alongside each class's cost, **the set of classes
/// its chosen sub-DAG contains**, and to price that set — each member once.
/// A parent unions its children's sets, so a class two siblings both reach is
/// paid for once, and `Mul(a, a)` pays for `a` once. At every class the number
/// being minimized is then the true DAG cost of the sub-DAG rooted there, and
/// at the root it is exactly [`ExtractedDAG::dag_cost`]: the objective and the
/// price are the same quantity. See [`shared_dag_dp_pass`].
///
/// It remains an approximation of the extraction optimum, which is NP-hard —
/// the choice is still made greedily bottom-up, so a locally dear class that
/// would have paid for itself upstream is still passed over. What it is not is
/// an approximation of the *objective*: nothing here charges for sharing.
///
/// Both objectives are run and **the cheaper term by true `dag_cost` wins**,
/// ties going to the tree pass. So the returned DAG cost can only be lower
/// than the pre-#1116 extractor's, never higher — no-regression is structural
/// rather than empirical, at the price of a second DP pass over a graph
/// extraction walks once per compile. The one exception is loud: the shared
/// pass holds its reach sets under [`SHARED_DAG_PASS_BYTE_BUDGET`], and a
/// graph whose sets outgrow it gets the tree term with
/// [`ExtractionObjective::TreeOnly`] in [`ExtractedDAG::report`] — never
/// the same `Vec` under a different objective.
pub fn extract_dag_scoped<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    extract_dag_over(
        Frozen::new(egraph, root).problem(),
        &Pricing { costs, shape },
    )
}

/// [`extract_dag_scoped`] over a problem already read from its e-graph: what
/// a caller that extracts the same saturated graph at several shapes calls,
/// so the live set, the candidates and the parent index are built once and
/// each shape adds its weights and its settlement.
pub(crate) fn extract_dag_over<C: CostFunction>(
    problem: Problem<'_>,
    pricing: &Pricing<'_, C>,
) -> ExtractedDAG {
    let (egraph, root) = (problem.egraph(), problem.root());
    let Pricing { costs, shape } = *pricing;
    let at = Rooted { egraph, root };
    let tree = costed(
        at,
        tree_dp_pass(problem, &mut Dp::production(costs, shape)),
        CostScale::Tree,
        pricing,
    );
    let pass = shared_dag_dp_pass(
        problem,
        &mut Dp::production(costs, shape),
        SHARED_DAG_PASS_BYTE_BUDGET,
    );
    let stats = Some(pass.stats);
    let Some(dp) = pass.outcome else {
        return assemble(tree, ExtractionObjective::TreeOnly, stats);
    };
    let shared = costed(at, dp, CostScale::Dag, pricing);
    // The comparison is between two *re-costed* terms, never between the two
    // DPs' own tables: those are on different scales (`CostScale`). Both sides
    // here are `Selection::price` of a settled term under the same `costs`
    // and the same `shape`, so the min is like against like.
    if shared.cost.dag < tree.cost.dag {
        return assemble(shared, ExtractionObjective::Shared, stats);
    }
    assemble(tree, ExtractionObjective::TreeCheaper, stats)
}

/// The most memory `shared_dag_dp_pass` may hold in reach sets before it
/// gives up and `extract_dag_scoped` returns the tree term as
/// [`ExtractionObjective::TreeOnly`].
///
/// The pass's memory is the sum of its reach sets, each held in whichever
/// form is smaller (see `Reach`), so the worst case is the dense bound
/// `live_classes² / 8` bytes — a chain, where every class reaches every
/// class below it — and real kernels sit far under it: the reach sets are
/// the sub-DAGs of the chosen terms, and most of a saturated glyph's live
/// classes are variants deep inside one Bézier segment with a sub-DAG of a
/// few hundred classes. Calibrated on the 2026-09-08 class-cap sweep
/// (`docs/results/2026-09-08-class-cap-sweep.md`): the number is set so
/// that no DEV kernel at the shipped `classical` cap comes near it and the
/// dense worst case is still bounded to a size a glyph bake can hold
/// transiently — the sets are allocated once per extraction and dropped at
/// its end. A budget in bytes rather than classes because bytes are what
/// the gate protects, and the class count was a proxy that fired on the
/// whole e-graph while the pass was sized by the third of it the root
/// reaches.
pub const SHARED_DAG_PASS_BYTE_BUDGET: usize = 256 << 20;

/// What a term is priced under: the cost function and the lattice that weights
/// each node by how often it is evaluated.
pub(crate) struct Pricing<'a, C> {
    pub(crate) costs: &'a C,
    pub(crate) shape: LatticeShape,
}

/// The class a term is extracted from, in the graph that holds it.
#[derive(Clone, Copy)]
pub(crate) struct Rooted<'a> {
    pub(crate) egraph: &'a EGraph,
    pub(crate) root: EClassId,
}

/// A settled term, the cost of it, and what the DP that produced it claimed
/// it was worth.
struct CostedChoices {
    choices: Vec<Option<usize>>,
    selection: Selection,
    cost: ChoiceCost,
    audit: ClaimAudit,
}

/// The two terms [`extract_dag_scoped`] chooses between: `(tree, shared)`.
///
/// `tree` is the pre-#1116 extractor, arithmetic for arithmetic, and exists
/// so the A/B that justifies the sharing-aware objective can be run against
/// the thing it replaced rather than against a remembered number.
#[cfg(test)]
pub(crate) fn extract_dag_objectives<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> (ExtractedDAG, ExtractedDAG) {
    (
        extract_dag_tree_arm(egraph, root, costs, shape),
        extract_dag_shared_arm(egraph, root, costs, shape),
    )
}

/// The tree-cost arm on its own — the pre-#1116 extractor, and the control
/// the objective A/B and its cost measurement are run against.
#[cfg(test)]
pub(crate) fn extract_dag_tree_arm<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    let frozen = Frozen::new(egraph, root);
    let term = costed(
        Rooted { egraph, root },
        tree_dp_pass(frozen.problem(), &mut Dp::production(costs, shape)),
        CostScale::Tree,
        &Pricing { costs, shape },
    );
    // `shared_pass: None` — the pass was not run, as opposed to run and
    // abandoned, which is what production's `TreeOnly` carries.
    assemble(term, ExtractionObjective::TreeOnly, None)
}

/// The sharing-aware arm on its own (#1116). Runs the pass to completion
/// whatever it costs — an A/B that silently swapped its arm for the tree
/// term above some size would be measuring nothing — so a graph that would
/// exceed [`SHARED_DAG_PASS_BYTE_BUDGET`] in production is the caller's
/// memory to spend here.
#[cfg(test)]
pub(crate) fn extract_dag_shared_arm<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
) -> ExtractedDAG {
    let pass = shared_dag_dp_pass(
        Frozen::new(egraph, root).problem(),
        &mut Dp::production(costs, shape),
        usize::MAX,
    );
    let choices = pass
        .outcome
        .expect("an unbounded shared pass cannot run out of budget");
    let term = costed(
        Rooted { egraph, root },
        choices,
        CostScale::Dag,
        &Pricing { costs, shape },
    );
    assemble(term, ExtractionObjective::Shared, Some(pass.stats))
}

/// Price the term a settlement selected.
///
/// The settlement made the [`Selection`] itself: a class settles strictly
/// after the children of the candidate it settles on, so the term is
/// well-founded by construction and no pick changes after the DP made it.
///
/// The cost is of the term being RETURNED, not of the DP's running table:
/// the DP's claim and the returned term's price are separate numbers (#1111),
/// and this fold is what keeps them from being confused.
fn costed<C: CostFunction>(
    at: Rooted<'_>,
    dp: DpOutcome,
    scale: CostScale,
    pricing: &Pricing<'_, C>,
) -> CostedChoices {
    let DpOutcome {
        choices,
        selection,
        root_cost,
    } = dp;
    // Two constructions of one value — the settlement's, over its own rows,
    // and the hand-built map's, over the e-graph with its checks — so each
    // is the other's differential test on every extraction any test performs.
    debug_assert_eq!(
        Ok(&selection),
        Selection::of_map(at.egraph, at.root, &choices).as_ref(),
        "the selection a settlement made and the one its settled map names are different terms"
    );
    let cost = selection.price(at.egraph, pricing);
    let audit = ClaimAudit {
        claimed: root_cost,
        scale,
    };
    // The DP minimizes `claimed`; the caller pays `cost`. `settle_in_cost_order`
    // settles a class strictly after the children of the candidate it settles
    // on, so nothing rewrites a pick and every reach set is final before its
    // parent reads it — they are the same term, so they must be the same
    // number. An extractor whose objective differs from the price of what it
    // returns is optimizing something no one pays, which is exactly what the
    // DFS post-order did (docs/results/2026-09-08-cse-mispricing.md).
    //
    // A `debug_assert` rather than a test: this way every extraction any test
    // in the workspace performs is a self-consistency check, which is the only
    // way to cover corpora this crate cannot name. The defect this closes was
    // found by it firing on core-term's terminal scene.
    debug_assert_eq!(
        audit.claimed,
        audit.scale.of(cost),
        "extraction claim/price mismatch on the {:?} scale: the DP settled the root at \
         {claimed}, but the term its map names costs tree {tree} / dag {dag} — the objective \
         and the price have come apart",
        audit.scale,
        claimed = audit.claimed,
        tree = cost.tree,
        dag = cost.dag,
    );
    CostedChoices {
        choices,
        selection,
        cost,
        audit,
    }
}

/// The result of an extraction: the priced term and the objective that found
/// it.
fn assemble(
    costed: CostedChoices,
    objective: ExtractionObjective,
    shared_pass: Option<SharedPassStats>,
) -> ExtractedDAG {
    let CostedChoices {
        choices,
        selection,
        cost,
        audit,
    } = costed;
    ExtractedDAG {
        root: selection.root(),
        choices,
        selection,
        total_cost: cost.tree,
        dag_cost: cost.dag,
        report: ExtractionReport {
            objective,
            shared_pass,
            audit: Some(audit),
        },
    }
}

/// One node's weighted own cost under `shape`.
fn weighted_own<C: CostFunction>(costs: &C, node: &ENode, weight: u64) -> usize {
    usize::try_from((costs.node_cost(node, None) as u64).saturating_mul(weight))
        .unwrap_or(usize::MAX)
}

/// **How many times `node`'s children are evaluated per evaluation of `node`.**
///
/// One, for everything except a fold: `⊕_{[lo,hi)} f` evaluates `f` once per
/// index, and codegen emits a surviving fold as a loop that runs its body
/// exactly that many times (`pixelflow_ir::passes::legalize`). Pricing the
/// body once would tell the extractor a 34-piece fold costs what one piece
/// costs, which is how an unpriced fold turns the e-graph's unrolling
/// (`PeelFold`, `HalveFold`) off — it would keep every fold, unconditionally,
/// because folding would always look free.
///
/// This is the multiplier the fold's own [`CostModel::node_op_cost`] arm
/// deliberately leaves out: a node's cost cannot see its children's, and this
/// is the one place that number is in hand. The two halves sum to
/// [`CostModel::fold_cost`], the one formula for a fold's price.
///
/// The trip count is *local to the fold node*, which is what makes it exact.
/// A per-binder-slot table would not work: `PeelFold` rewrites
/// `⊕_{[lo,hi)} f` to `f(hi-1) ⊕ ⊕_{[lo,hi-1)} f`, keeping the same binder
/// and the same body e-class, so after saturation one slot carries folds of
/// many different lengths over one shared body and no single number is right
/// for it.
fn fold_body_multiple(node: &ENode) -> u64 {
    match node {
        ENode::Reduce { fold, .. } => u64::from(fold.len()),
        _ => 1,
    }
}

/// The two policy knobs the DP passes read, plus the trace they write.
///
/// Grouped rather than passed as four more arguments, and monomorphized
/// rather than dispatched: production instantiates it as
/// `Dp<C, Insertion, ()>`, where both `T` and `R` are ZSTs whose methods are
/// empty, so the emitted passes are byte-identical to the ones that took
/// `(costs, shape)` alone. Every knob is a *type*, so a research arm is a
/// second `impl` rather than a mode flag threaded through the inner loop.
pub(crate) struct Dp<'a, C, T, R> {
    costs: &'a C,
    shape: LatticeShape,
    ties: T,
    rec: R,
}

impl<'a, C> Dp<'a, C, Insertion, ()> {
    /// Production's instance: ties to insertion order, nothing recorded.
    pub(crate) fn production(costs: &'a C, shape: LatticeShape) -> Self {
        Self {
            costs,
            shape,
            ties: Insertion,
            rec: (),
        }
    }
}

impl<'a, C, T, R> Dp<'a, C, T, R> {
    /// A research instance: the same DP under another tie-break, writing to
    /// `rec`.
    #[cfg(any(test, feature = "provenance-journal"))]
    pub(crate) fn new(costs: &'a C, shape: LatticeShape, ties: T, rec: R) -> Self {
        Self {
            costs,
            shape,
            ties,
            rec,
        }
    }

    /// The trace the passes wrote, once they are done with it.
    #[cfg(any(test, feature = "provenance-journal"))]
    pub(crate) fn into_recorder(self) -> R {
        self.rec
    }
}

/// Which of two nodes a class keeps when the DP prices them **equally**.
///
/// The strict `<` the passes have always used answers this implicitly —
/// the earlier index wins, and index is insertion order — which is why the
/// `'8'` bisect saw extraction move under a semantically-null change to the
/// input. Naming the decision makes the alternative a second `impl` instead
/// of a fork of the pass.
pub(crate) trait TieBreak {
    /// At equal cost, does `challenger` displace the `incumbent` node index
    /// in `class`?
    fn prefer(&self, egraph: &EGraph, class: EClassId, incumbent: usize, challenger: usize)
    -> bool;
}

/// Production: never — the first admissible node stands, so the choice is
/// the e-graph's insertion order.
#[derive(Clone, Copy)]
pub(crate) struct Insertion;

impl TieBreak for Insertion {
    #[inline]
    fn prefer(&self, _: &EGraph, _: EClassId, _: usize, _: usize) -> bool {
        false
    }
}

/// The research arm: a total order on the node's own content, so a tie is
/// broken by what the node *is* rather than by when it was inserted.
///
/// Key, ascending: leaf/op tag, then the leaf's payload or the `OpKind`
/// ordinal, then arity, then the canonical child ids. Two distinct nodes in
/// one class always differ somewhere in that key (a class holds no
/// duplicates after `rebuild`), so the order is total and the result is
/// independent of insertion order.
#[derive(Clone, Copy)]
#[cfg(any(test, feature = "provenance-journal"))]
pub(crate) struct Canonical;

/// `node`'s position in [`Canonical`]'s order.
#[cfg(any(test, feature = "provenance-journal"))]
fn canonical_key(egraph: &EGraph, node: &ENode) -> (u8, u128, usize, Vec<u32>) {
    let children: Vec<u32> = node
        .children_slice()
        .iter()
        .map(|&c| egraph.find(c).0)
        .collect();
    match node {
        ENode::Var(i) => (0, u128::from(*i), 0, children),
        ENode::Const(bits) => (1, u128::from(*bits), 0, children),
        ENode::Buffer(_) => (2, 0, 0, children),
        ENode::Uniform(_) => (3, 0, 0, children),
        ENode::Param(i) => (4, u128::from(*i), 0, children),
        ENode::Op { op, .. } => (5, *op as u128, children.len(), children),
        // The fold *is* the discriminating part: two folds over one body
        // differ only in their metadata, so that is what orders them.
        ENode::Reduce { fold, .. } => (6, fold.to_bits(), children.len(), children),
        ENode::Ref { key, .. } => (7, u128::from(key.bits()), 0, children),
        // A shift orders among the operations by its op, as it did when it
        // was one; its count, which tells two shifts of one operand apart,
        // takes the arity's place (a shift has one operand).
        ENode::Shift {
            direction, count, ..
        } => (5, direction.op() as u128, count.bits() as usize, children),
    }
}

#[cfg(any(test, feature = "provenance-journal"))]
impl TieBreak for Canonical {
    fn prefer(
        &self,
        egraph: &EGraph,
        class: EClassId,
        incumbent: usize,
        challenger: usize,
    ) -> bool {
        let nodes = egraph.nodes(class);
        canonical_key(egraph, &nodes[challenger]) < canonical_key(egraph, &nodes[incumbent])
    }
}

/// What a DP pass writes about each candidate it priced, for the research
/// harness that asks *why* a class went the way it did.
///
/// Production's instance is `()`, whose methods are empty and inline away —
/// the pass allocates nothing and branches nowhere for a trace nobody reads.
pub(crate) trait StageRecorder {
    /// One candidate priced: its DP cost and its weighted own cost.
    fn candidate(&mut self, class: EClassId, idx: usize, cost: usize, own: usize);
    /// The candidate the class settled on.
    fn settled(&mut self, class: EClassId, idx: usize);
}

impl StageRecorder for () {
    #[inline]
    fn candidate(&mut self, _: EClassId, _: usize, _: usize, _: usize) {}
    #[inline]
    fn settled(&mut self, _: EClassId, _: usize) {}
}

/// A class is settled when its cheapest **admissible** candidate has every
/// child settled — Knuth's AND-OR generalisation of Dijkstra, and the
/// denotation both DP passes below compute.
///
/// The passes used to walk one DFS post-order, which cannot express that. A
/// class whose child was still on the stack got priced at a sentinel and was
/// never revisited, so on a saturated graph — commutativity alone closes
/// cycles — a large fraction of classes carried no opinion at all, and the
/// pick fell to a repair whose job was well-foundedness, not cost. 29 % of
/// the frontier classes holding the witnesses of
/// `docs/results/2026-09-08-extraction-witnesses.md`, and 73 % of them on the
/// shaders, were decided that way; the sentinel is gone with the traversal
/// that needed it.
///
/// Settling in cost order is exact whenever a candidate costs at least as
/// much as each of its children, which both passes satisfy: the tree pass
/// adds its children's costs to a non-negative own cost, and the shared pass
/// prices the union of its children's reach sets, a superset of each of
/// them. A candidate that mentions its own class is never admissible, and a
/// class none of whose candidates ever becomes admissible is one no
/// well-founded term reaches — [`settle_in_cost_order`] refuses to return
/// with the root in that state rather than inventing a choice for it.
trait Settling {
    /// Price `candidate`, whose e-node is `node`. Every child class is
    /// settled, and none of them is the candidate's own.
    fn price(&mut self, candidate: Candidate, node: &ENode) -> usize;

    /// Which of two candidates priced **equally** the class keeps.
    fn prefer(&self, class: EClassId, incumbent: usize, challenger: usize) -> bool;

    /// Settle the candidate's class on `candidate`, whose e-node is `node`,
    /// priced at `cost`. Returning `false` abandons the pass.
    fn settle(&mut self, candidate: Candidate, node: &ENode, cost: usize) -> bool;
}

/// What a settlement tracks beside the problem it settles: the candidates it
/// is still waiting on, the cheapest priced so far for each class, and the
/// classes ready to be settled, cheapest first.
struct Settlement<'a, S> {
    problem: Problem<'a>,
    pricer: &'a mut S,
    /// Unsettled distinct child classes per candidate, by
    /// [`AndOrGraph::slot`](super::and_or::AndOrGraph::slot).
    waiting: Vec<usize>,
    /// The cheapest candidate priced so far for each live class: its cost
    /// and index.
    best: LiveTable<Option<(usize, usize)>>,
    /// Ordered by cost, then by the class's **e-graph id** (the live index
    /// rides along and decides nothing, the id being unique). Equal-cost
    /// classes settle in id order, and under `Insertion` that settle order is
    /// what picks the incumbent, so the key must stay the id the e-graph
    /// gave and not the position the class holds in the root's reach.
    heap: BinaryHeap<Reverse<(usize, u32, Live)>>,
}

impl<'a, S: Settling> Settlement<'a, S> {
    fn new(problem: Problem<'a>, pricer: &'a mut S) -> Self {
        let graph = problem.graph();
        Self {
            problem,
            pricer,
            waiting: graph.child_counts(),
            best: graph.table(None),
            heap: BinaryHeap::new(),
        }
    }

    /// Price one candidate and let it take its class's incumbent if it is
    /// cheaper, or equal and preferred.
    fn relax(&mut self, candidate: Candidate) {
        let (class, live, idx) = (candidate.class(), candidate.live(), candidate.idx());
        let cost = self
            .pricer
            .price(candidate, &self.problem.egraph().nodes(class)[idx]);
        let takes = match self.best[live] {
            None => true,
            Some((incumbent_cost, incumbent)) => {
                cost < incumbent_cost
                    || (cost == incumbent_cost && self.pricer.prefer(class, incumbent, idx))
            }
        };
        if !takes {
            return;
        }
        self.best[live] = Some((cost, idx));
        self.heap.push(Reverse((cost, class.0, live)));
    }

    fn run(mut self) -> Option<DpOutcome> {
        let (egraph, graph) = (self.problem.egraph(), self.problem.graph());
        let mut settled = graph.table(false);
        let mut choice: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];

        for live in graph.lives() {
            for candidate in graph.candidates(live) {
                if self.waiting[graph.slot(candidate)] == 0 {
                    self.relax(candidate);
                }
            }
        }

        while let Some(Reverse((cost, _, live))) = self.heap.pop() {
            if settled[live] {
                continue;
            }
            let (incumbent_cost, incumbent) =
                self.best[live].expect("a class in the heap has been priced at least once");
            if incumbent_cost != cost {
                // A superseded entry: the class has since been priced cheaper,
                // and that entry is still in the heap.
                continue;
            }
            let chosen = graph
                .candidate(live, incumbent)
                .expect("a priced candidate is one of its class's e-nodes");
            if !self.pricer.settle(
                chosen,
                &egraph.nodes(chosen.class())[incumbent],
                incumbent_cost,
            ) {
                return None;
            }
            settled[live] = true;
            choice[chosen.class().index()] = Some(incumbent);

            for parent in graph.parents(live) {
                if settled[parent.live()] {
                    continue;
                }
                let w = &mut self.waiting[graph.slot(parent)];
                assert!(
                    *w != 0,
                    "settle_in_cost_order: candidate {} of e-class {} was decremented past its \
                     child count — the parent index and the child counts disagree",
                    parent.idx(),
                    parent.class().0
                );
                *w -= 1;
                if *w == 0 {
                    self.relax(parent);
                }
            }
        }

        let root = graph.root_live();
        assert!(
            settled[root],
            "settle_in_cost_order: root e-class {} has no well-founded term — every candidate of \
             every class it reaches sits behind a cycle, which is structural corruption rather than \
             a rewrite outcome",
            graph.root().0
        );
        let root_cost = self.best[root].expect("a settled class has been priced").0;
        let selection = Selection::settled(self.problem, |live| choice[graph.class(live).index()]);
        Some(DpOutcome {
            choices: choice,
            selection,
            root_cost,
        })
    }
}

/// Settle every live class in increasing cost order, or abandon.
///
/// `None` is [`Settling::settle`] asking to stop — the shared pass over its
/// memory budget. Otherwise every class some well-founded term reaches has a
/// choice, and the map is acyclic **by construction**: a class is settled
/// strictly after the children of the candidate it settles on, so no repair
/// stage is required to make the result materialisable.
///
/// A candidate that names its own class waits on a child that settles only
/// after the class does, and a settled class's parent entries are skipped,
/// so such a candidate is never priced and needs no case of its own.
///
/// The [`DpOutcome`] carries the root's settled cost beside the map: that is
/// the value this driver *minimized*, and it is the only number that can be
/// checked against the price of the term the map names.
fn settle_in_cost_order<S: Settling>(problem: Problem<'_>, pricer: &mut S) -> Option<DpOutcome> {
    Settlement::new(problem, pricer).run()
}

/// A DP pass's choice map beside the value its own table holds at the root —
/// the number the pass *minimized*, before anything downstream re-costs the
/// term. Kept together because quoting either without the other is how a
/// claim gets mistaken for a price.
pub(crate) struct DpOutcome {
    /// Every class the pass settled and the node it settled on, by class id:
    /// the settled record, a superset of the selection.
    pub(crate) choices: Vec<Option<usize>>,
    /// The term the root settled on.
    pub(crate) selection: Selection,
    pub(crate) root_cost: usize,
}

/// The pre-#1116 DP: cheapest node per class where a child costs its whole
/// subtree, at every use. The control arm of the objective A/B, and the
/// floor [`extract_dag_scoped`] never returns worse than.
pub(crate) fn tree_dp_pass<C: CostFunction, T: TieBreak, R: StageRecorder>(
    problem: Problem<'_>,
    dp: &mut Dp<'_, C, T, R>,
) -> DpOutcome {
    let mut pricer = TreePricer {
        problem,
        dp,
        cost: problem.graph().table(None),
        // The variance of the form chosen for each class, which is what its
        // scope — and so its weight — is read from. Carried in the same DP
        // as the cost because the two determine each other: a child's
        // variance sets its parent's weight, and a parent's weight is part
        // of what makes one child's form worth choosing over another's.
        var: alloc::vec![Variance::CONST; problem.egraph().num_classes()],
    };
    settle_in_cost_order(problem, &mut pricer)
        .expect("the tree pass has no budget and never abandons")
}

/// [`tree_dp_pass`]'s pricing: a candidate costs its own weighted cost plus
/// each child's settled cost, summed at every use.
struct TreePricer<'a, 'c, C, T, R> {
    problem: Problem<'a>,
    dp: &'a mut Dp<'c, C, T, R>,
    /// The settled cost of each live class.
    cost: LiveTable<Option<usize>>,
    /// By e-class id, the table `node_variance` reads.
    var: Vec<Variance>,
}

impl<C: CostFunction, T: TieBreak, R: StageRecorder> Settling for TreePricer<'_, '_, C, T, R> {
    fn price(&mut self, candidate: Candidate, node: &ENode) -> usize {
        let node_var = node_variance(self.problem.egraph(), node, &self.var, candidate.class());
        let own = weighted_own(self.dp.costs, node, self.dp.shape.evals(node_var));
        let cost = match node {
            ENode::Var(_)
            | ENode::Const(_)
            | ENode::Buffer(_)
            | ENode::Uniform(_)
            | ENode::Param(_)
            | ENode::Ref { .. } => own,
            // Saturating fold, not `.sum()`: a child's own cost can already
            // sit at a prohibitive sentinel (the `usize::MAX / 4`
            // `CostModel::node_op_cost` gives a fold whose monoid has no
            // combiner), so a node with several such children overflows a
            // plain `usize` sum.
            ENode::Op { .. } | ENode::Shift { .. } | ENode::Reduce { .. } => {
                let per_child = fold_body_multiple(node);
                own.saturating_add(
                    self.problem
                        .graph()
                        .children(candidate)
                        .iter()
                        .map(|child| {
                            let sub = self.cost[child.live]
                                .expect("a priced candidate's children are settled");
                            usize::try_from((sub as u64).saturating_mul(per_child))
                                .unwrap_or(usize::MAX)
                                .saturating_mul(child.uses)
                        })
                        .fold(0usize, usize::saturating_add),
                )
            }
        };
        self.dp
            .rec
            .candidate(candidate.class(), candidate.idx(), cost, own);
        cost
    }

    fn prefer(&self, class: EClassId, incumbent: usize, challenger: usize) -> bool {
        self.dp
            .ties
            .prefer(self.problem.egraph(), class, incumbent, challenger)
    }

    fn settle(&mut self, candidate: Candidate, node: &ENode, cost: usize) -> bool {
        // Read the children's variances before writing this class's, which
        // `node_variance` never consults for an admissible candidate.
        let node_var = node_variance(self.problem.egraph(), node, &self.var, candidate.class());
        self.cost[candidate.live()] = Some(cost);
        self.var[candidate.class().index()] = node_var;
        self.dp.rec.settled(candidate.class(), candidate.idx());
        true
    }
}

/// The sharing-aware DP (#1116): cheapest node per class where the cost of a
/// candidate is the cost of **the set of classes its sub-DAG contains**, each
/// member priced once.
///
/// Same driver as [`tree_dp_pass`]; the only change is what a candidate
/// costs. Each class carries the set of classes its chosen sub-DAG reaches
/// (a [`Reach`]), and a candidate unions its children's sets, adding a
/// class's own cost the first time that class enters the union. Two siblings
/// that both reach `sin(X)` therefore pay for it once, which is what the
/// emitted kernel does: the term materializes as one node per
/// reachable class and codegen let-binds the shared ones.
///
/// A union is taken by stamping: every member of every child's set is
/// visited once, and a per-class epoch mark says whether it has been seen
/// under this candidate, so the cost of a candidate is the size of its
/// children's sets, not the size of the graph. Memory is the sum of the
/// sets, each in the smaller of its two forms, and is held under `budget`:
/// the first class whose set would carry the total past it ends the pass
/// with [`SharedPassOutcome::choices`] `None`, and the caller returns the
/// tree term as [`ExtractionObjective::TreeOnly`]. The dense form bounds
/// the worst case at `live² / 8` bytes (the 2026-09-08 memory profile's
/// measured quadratic); real kernels hold a small fraction of that because
/// most live classes are variants deep inside one sub-DAG.
pub(crate) fn shared_dag_dp_pass<C: CostFunction, T: TieBreak, R: StageRecorder>(
    problem: Problem<'_>,
    dp: &mut Dp<'_, C, T, R>,
    budget: usize,
) -> SharedPassOutcome {
    let graph = problem.graph();
    let live = graph.len();
    let mut pricer = SharedPricer {
        problem,
        dp,
        var: alloc::vec![Variance::CONST; problem.egraph().num_classes()],
        sets: ReachSets {
            reach: graph.table_from_fn(|_| None),
            own: graph.table(0),
            stamp: graph.table(0),
            epoch: 0,
            scratch: Vec::new(),
            words: live.div_ceil(REACH_WORD_BITS),
            bytes: 0,
            budget,
        },
    };
    let outcome = settle_in_cost_order(problem, &mut pricer);
    SharedPassOutcome {
        outcome,
        stats: SharedPassStats {
            live_classes: live,
            reach_bytes: pricer.sets.bytes,
        },
    }
}

/// The reach sets, indexed by [`Live`], and the budget they are held
/// under.
struct ReachSets {
    /// `reach[i]` is the set of classes the chosen sub-DAG at live class `i`
    /// contains, itself included — `None` until the class settles.
    reach: LiveTable<Option<Reach>>,
    /// The weighted own cost of each live class's chosen node: what a union
    /// pays when that class first enters it.
    own: LiveTable<usize>,
    /// `stamp[i] == epoch` iff live class `i` has entered the union being
    /// taken for the current candidate. One epoch per candidate; never
    /// cleared, so a union costs its members and nothing else.
    stamp: LiveTable<usize>,
    epoch: usize,
    scratch: Vec<u32>,
    words: usize,
    bytes: usize,
    budget: usize,
}

impl ReachSets {
    /// Union the children's reach sets into `scratch`, returning what those
    /// classes cost with each member paid once.
    ///
    /// `children` is a candidate's row in the [`AndOrGraph`](super::and_or::AndOrGraph).
    /// A leaf's row is empty, so it reaches nothing without a case of its own.
    fn union_below(&mut self, children: &[Child]) -> usize {
        self.epoch += 1;
        self.scratch.clear();
        let mut below = 0usize;
        for &Child { live: ci, .. } in children {
            // Taken and put back: the closure below needs `self` mutably
            // while the set is read, and a set is never its own member's.
            let set = self.reach[ci]
                .take()
                .expect("a priced candidate's children are settled");
            set.for_each(|member| {
                let m = Live::from_index(member as usize);
                if self.stamp[m] != self.epoch {
                    self.stamp[m] = self.epoch;
                    below = below.saturating_add(self.own[m]);
                    self.scratch.push(member);
                }
            });
            self.reach[ci] = Some(set);
        }
        below
    }
}

/// [`shared_dag_dp_pass`]'s pricing.
struct SharedPricer<'a, 'c, C, T, R> {
    problem: Problem<'a>,
    dp: &'a mut Dp<'c, C, T, R>,
    var: Vec<Variance>,
    sets: ReachSets,
}

impl<C: CostFunction, T: TieBreak, R: StageRecorder> Settling for SharedPricer<'_, '_, C, T, R> {
    fn price(&mut self, candidate: Candidate, node: &ENode) -> usize {
        let node_var = node_variance(self.problem.egraph(), node, &self.var, candidate.class());
        let own = weighted_own(self.dp.costs, node, self.dp.shape.evals(node_var));
        let below = self
            .sets
            .union_below(self.problem.graph().children(candidate));
        let cost = own.saturating_add(below);
        self.dp
            .rec
            .candidate(candidate.class(), candidate.idx(), cost, own);
        cost
    }

    fn prefer(&self, class: EClassId, incumbent: usize, challenger: usize) -> bool {
        self.dp
            .ties
            .prefer(self.problem.egraph(), class, incumbent, challenger)
    }

    fn settle(&mut self, candidate: Candidate, node: &ENode, _cost: usize) -> bool {
        let node_var = node_variance(self.problem.egraph(), node, &self.var, candidate.class());
        let own = weighted_own(self.dp.costs, node, self.dp.shape.evals(node_var));
        // Rebuild the winner's union rather than carrying one per class in
        // flight: the driver prices candidates for many unsettled classes
        // before any of them settles, so a stored winner set would be a
        // second copy of the frontier's reach.
        let _ = self
            .sets
            .union_below(self.problem.graph().children(candidate));
        let me = candidate.live();
        self.sets.scratch.push(
            u32::try_from(me.index()).expect("a live index is below the e-graph's u32 class count"),
        );
        let set = Reach::smaller_of(&self.sets.scratch, self.sets.words);
        self.sets.bytes = self.sets.bytes.saturating_add(set.bytes());
        if self.sets.bytes > self.sets.budget {
            return false;
        }
        self.sets.reach[me] = Some(set);
        self.sets.own[me] = own;
        self.var[candidate.class().index()] = node_var;
        self.dp.rec.settled(candidate.class(), candidate.idx());
        true
    }
}

/// What [`shared_dag_dp_pass`] returns: its choice map when it finished
/// under budget, and its accounting either way.
pub(crate) struct SharedPassOutcome {
    /// `None` when the reach sets crossed the byte budget and the pass was
    /// abandoned.
    pub(crate) outcome: Option<DpOutcome>,
    pub(crate) stats: SharedPassStats,
}

const REACH_WORD_BITS: usize = u64::BITS as usize;

/// One live class's reach set — the live indices of the classes its chosen
/// sub-DAG contains — in whichever of two forms is smaller.
///
/// The sparse form is the members themselves, four bytes each; the dense
/// form is one bit per live class. [`Reach::smaller_of`] picks per set, so
/// a leaf's set is four bytes and the root's is a bitset, and the total
/// held by [`shared_dag_dp_pass`] is never above the dense bound and is far
/// below it on any graph that is not a chain.
enum Reach {
    Sparse(Vec<u32>),
    Dense(Vec<u64>),
}

impl Reach {
    /// `members` as the smaller of the two forms over `words` dense words.
    /// `members` must be distinct.
    fn smaller_of(members: &[u32], words: usize) -> Self {
        let sparse_bytes = std::mem::size_of_val(members);
        let dense_bytes = words * core::mem::size_of::<u64>();
        if sparse_bytes <= dense_bytes {
            return Self::Sparse(members.to_vec());
        }
        let mut bits = alloc::vec![0u64; words];
        for &m in members {
            let m = m as usize;
            bits[m / REACH_WORD_BITS] |= 1u64 << (m % REACH_WORD_BITS);
        }
        Self::Dense(bits)
    }

    fn bytes(&self) -> usize {
        match self {
            Self::Sparse(v) => v.len() * core::mem::size_of::<u32>(),
            Self::Dense(v) => v.len() * core::mem::size_of::<u64>(),
        }
    }

    fn for_each(&self, mut f: impl FnMut(u32)) {
        match self {
            Self::Sparse(v) => v.iter().copied().for_each(f),
            Self::Dense(v) => {
                for (w, &word) in v.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        f((w * REACH_WORD_BITS + bit) as u32);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::egraph::and_or::AndOrGraph;
    use crate::egraph::cost::CostModel;
    use crate::egraph::shift::{ShiftCount, ShiftDirection};
    use pixelflow_ir::OpKind;

    /// What the reference pass prices an unresolved child at — above any
    /// weighted cost a real program can reach (a flat 1_000_000 was safely
    /// above every *unweighted* cost; weighting by a frame's sample count
    /// clears that by orders of magnitude). Production settlement prices no
    /// cycle, so this is a property of the reference alone.
    const CYCLE_COST: usize = usize::MAX / 4;

    /// Scope-weighted extraction un-fuses an FMA to move work out of the loop.
    ///
    /// One e-class holds two equal forms of `X + 2Z`: the fused
    /// `MulAdd(2, Z, X)` — one instruction — and `X + (Z + Z)`, which is two.
    /// Priced by text the FMA wins and always should. Priced against a
    /// 256×256 frame it loses: its single instruction runs at every one of
    /// the 65 536 samples, while the pair runs one add per sample and lifts
    /// `Z + Z` out of the loop entirely, because `Z` does not vary across it.
    ///
    /// No saturation here — both forms are placed in the class by hand — so
    /// this is a property of the cost model alone, and the choice flipping
    /// with the lattice is the whole claim of scope-weighted extraction: the
    /// factorization is *chosen*, not discovered afterwards by a hoisting
    /// pass that arrives once the FMA has already welded `Z` into the
    /// per-sample expression.
    #[test]
    fn scope_weighting_unfuses_an_fma_to_hoist_the_z_term() {
        use pixelflow_ir::OpKind;

        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let z = egraph.add(ENode::Var(2));
        let two = egraph.add(ENode::Const(2.0f32.to_bits()));

        let fused = egraph.add(ENode::apply(OpKind::MulAdd, &[two, z, x]));
        let z_plus_z = egraph.add(ENode::apply(OpKind::Add, &[z, z]));
        let unfused = egraph.add(ENode::apply(OpKind::Add, &[x, z_plus_z]));
        let root = egraph.union(fused, unfused);
        egraph.rebuild();

        let costs = CostModel::latency_prior();
        let chosen = |shape| {
            let dag = extract_dag_scoped(&egraph, root, &costs, shape);
            let idx = dag.choices[egraph.find(root).0 as usize].expect("root is chosen");
            match &egraph.nodes(egraph.find(root))[idx] {
                ENode::Op { op, .. } => *op,
                other => panic!("root should be an op, got {other:?}"),
            }
        };

        assert_eq!(
            chosen(LatticeShape::POINT),
            OpKind::MulAdd,
            "with no lattice the fused form is one instruction against two"
        );
        assert_eq!(
            chosen(LatticeShape::new([256, 256])),
            OpKind::Add,
            "over a frame the fused form pays for Z at every sample"
        );
    }

    /// A chain of `n` distinct-const `Add` nodes: exactly `n + 1` live
    /// classes with no saturation (each `add()` is a fresh, non-folding
    /// node), and the adversarial shape for the reach sets — every class
    /// reaches every class below it, so the sets are as large as they can
    /// be. What the tree-only test above the old class-count gate used, and
    /// what the byte budget is bounded on.
    fn add_chain(n: usize) -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let mut cur = egraph.add(ENode::Var(0));
        for i in 0..n {
            let c = egraph.add(ENode::constant(i as f32 + 1.0));
            cur = egraph.add(ENode::apply(pixelflow_ir::OpKind::Add, &[cur, c]));
        }
        egraph.rebuild();
        (egraph, cur)
    }

    /// The shared pass abandons itself at its byte budget and says so: the
    /// choice map is `None`, the stats carry the first total over the
    /// budget, and `extract_dag_scoped` — whose budget is the production
    /// constant — reports the objective it actually used rather than
    /// returning the tree term under a shared label.
    #[test]
    fn shared_pass_over_budget_is_abandoned_loudly() {
        const CHAIN: usize = 2_000;
        let (egraph, root) = add_chain(CHAIN);
        let costs = CostModel::latency_prior();

        let full = shared_dag_dp_pass(
            Frozen::new(&egraph, root).problem(),
            &mut Dp::production(&costs, LatticeShape::POINT),
            usize::MAX,
        );
        let full_choices = full
            .outcome
            .as_ref()
            .expect("unbounded pass finishes")
            .choices
            .clone();
        assert_eq!(full.stats.live_classes, 2 * CHAIN + 1);
        // The dense bound, plus one word of rounding per set.
        let live = full.stats.live_classes;
        let dense_bound = live * live.div_ceil(REACH_WORD_BITS) * 8;
        assert!(
            full.stats.reach_bytes <= dense_bound,
            "reach sets ({}) must never exceed the dense bound ({dense_bound})",
            full.stats.reach_bytes
        );

        let budget = full.stats.reach_bytes / 2;
        let cut = shared_dag_dp_pass(
            Frozen::new(&egraph, root).problem(),
            &mut Dp::production(&costs, LatticeShape::POINT),
            budget,
        );
        assert!(
            cut.outcome.is_none(),
            "a pass over budget returns no choices"
        );
        assert!(
            cut.stats.reach_bytes > budget,
            "stats carry the total that crossed the budget ({} vs {budget})",
            cut.stats.reach_bytes
        );
        assert_eq!(cut.stats.live_classes, full.stats.live_classes);

        // Under the production budget this chain fits, and the report says
        // which term came back and what it cost to know.
        let scoped = extract_dag_scoped(&egraph, root, &costs, LatticeShape::POINT);
        let stats = scoped
            .report
            .shared_pass
            .expect("production extraction always runs the pass");
        assert_eq!(stats, full.stats);
        assert!(
            full.stats.reach_bytes <= SHARED_DAG_PASS_BYTE_BUDGET,
            "fixture must fit the production budget for this half of the test"
        );
        assert_ne!(scoped.report.objective, ExtractionObjective::TreeOnly);
        assert_ne!(scoped.report.objective, ExtractionObjective::External);
        let shared = extract_dag_shared_arm(&egraph, root, &costs, LatticeShape::POINT);
        assert_eq!(shared.choices, full_choices);
    }

    /// Sparse and dense reach sets are one set: a union taken through
    /// either form visits the same members, so a class whose set flips
    /// form (the chain's upper half) prices its children identically.
    #[test]
    fn reach_forms_agree() {
        let words = 4;
        let few: Vec<u32> = vec![3, 200, 77];
        let many: Vec<u32> = (0..words as u32 * 40).collect();
        let sparse = Reach::smaller_of(&few, words);
        let dense = Reach::smaller_of(&many, words);
        assert!(matches!(sparse, Reach::Sparse(_)));
        assert!(matches!(dense, Reach::Dense(_)));
        assert_eq!(sparse.bytes(), few.len() * 4);
        assert_eq!(dense.bytes(), words * 8);
        let mut seen = Vec::new();
        sparse.for_each(|m| seen.push(m));
        assert_eq!(seen, few);
        seen.clear();
        dense.for_each(|m| seen.push(m));
        assert_eq!(seen, many);
    }

    /// The sharing-aware DP as it shipped before the reach sets went hybrid
    /// **and** before the pass settled in cost order: one dense bitset per
    /// live class, one DFS post-order, `CYCLE_COST` for a class whose child
    /// is still on the stack. Kept verbatim as the reference the budgeted
    /// pass is held to — its results are the ones every committed extraction
    /// row was taken with
    /// (`docs/results/2026-09-07-egraph-off-vs-on-real-shaders-rows`).
    ///
    /// Its agreement with the shipped pass is therefore conditional, and the
    /// condition is checked rather than assumed: the fixpoint reproduces
    /// this reference exactly **wherever this reference had an opinion**, so
    /// the caller asserts the fixture holds no class it could only price at
    /// the sentinel. Without that assertion the comparison would quietly go
    /// vacuous the day a fixture grew a cycle.
    fn dense_reference_pass<C: CostFunction>(
        egraph: &EGraph,
        root: EClassId,
        costs: &C,
        shape: LatticeShape,
    ) -> (Vec<Option<usize>>, usize) {
        const BITS: usize = usize::BITS as usize;
        let num_classes = egraph.num_classes();
        let order = AndOrGraph::of(egraph, root).classes().to_vec();
        let live = order.len();
        let words = live.div_ceil(BITS);
        let mut compact: Vec<u32> = alloc::vec![u32::MAX; num_classes];
        for (i, c) in order.iter().enumerate() {
            compact[c.0 as usize] = i as u32;
        }
        let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut best_var: Vec<Variance> = alloc::vec![Variance::CONST; num_classes];
        let mut best_own: Vec<usize> = alloc::vec![0; live];
        let mut reach: Vec<usize> = alloc::vec![0; live * words];
        let mut scratch: Vec<usize> = alloc::vec![0; words];
        for canonical in order.iter().copied() {
            let me = compact[canonical.0 as usize] as usize;
            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;
            let mut min_var = Variance::CONST;
            let mut min_own = 0usize;
            for (idx, node) in nodes.iter().enumerate() {
                let node_var = node_variance(egraph, node, &best_var, canonical);
                let own = weighted_own(costs, node, shape.evals(node_var));
                let this_node_cost = match node {
                    ENode::Var(_)
                    | ENode::Const(_)
                    | ENode::Buffer(_)
                    | ENode::Uniform(_)
                    | ENode::Param(_)
                    | ENode::Ref { .. } => own,
                    ENode::Op { .. } | ENode::Shift { .. } | ENode::Reduce { .. } => {
                        let children = node.children_slice();
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            scratch.fill(0);
                            let mut below = 0usize;
                            let mut unresolved = false;
                            for &child in children.iter() {
                                let c = egraph.find(child).0 as usize;
                                if best_cost[c].is_none() {
                                    unresolved = true;
                                    break;
                                }
                                let base = compact[c] as usize * words;
                                for w in 0..words {
                                    let fresh = reach[base + w] & !scratch[w];
                                    if fresh == 0 {
                                        continue;
                                    }
                                    scratch[w] |= fresh;
                                    let mut bits = fresh;
                                    while bits != 0 {
                                        let bit = bits.trailing_zeros() as usize;
                                        bits &= bits - 1;
                                        below = below.saturating_add(best_own[w * BITS + bit]);
                                    }
                                }
                            }
                            if unresolved {
                                CYCLE_COST
                            } else {
                                own.saturating_add(below)
                            }
                        }
                    }
                };
                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                    min_var = node_var;
                    min_own = own;
                }
            }
            let base = me * words;
            reach[base..base + words].fill(0);
            {
                let children = nodes[min_idx].children_slice();
                if min_cost != CYCLE_COST {
                    for &child in children.iter() {
                        let cbase = compact[egraph.find(child).0 as usize] as usize * words;
                        for w in 0..words {
                            reach[base + w] |= reach[cbase + w];
                        }
                    }
                }
            }
            reach[base + me / BITS] |= 1usize << (me % BITS);
            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
            best_var[canonical.0 as usize] = min_var;
            best_own[me] = min_own;
        }
        let root_cost = best_cost[egraph.find(root).0 as usize].expect("the root is settled");
        (best_node, root_cost)
    }

    /// An SDF-shaped arena with real sharing (the same op mix as
    /// `egraph_profile`'s), saturated by the production optimizer under a
    /// small explicit budget: a graph with variants, cycles and shared
    /// classes, as the passes meet them in production.
    fn saturated_sdf_egraph(target_nodes: usize) -> (EGraph, EClassId) {
        use pixelflow_ir::{ExprArena, OpKind};
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let c = arena.push_const(0.37);
        let dx = arena.push_binary(OpKind::Sub, x, c);
        let dy = arena.push_binary(OpKind::Sub, y, c);
        let dx2 = arena.push_binary(OpKind::Mul, dx, dx);
        let mut cur = arena.push_ternary(OpKind::MulAdd, dy, dy, dx2);
        let mut step = 0usize;
        while arena.len() < target_nodes {
            cur = match step % 8 {
                0 => {
                    let inside = arena.push_binary(OpKind::Lt, cur, c);
                    arena.push_ternary(OpKind::If, inside, dx2, cur)
                }
                1 => arena.push_unary(OpKind::Sqrt, cur),
                2 => arena.push_binary(OpKind::Mul, cur, dx),
                3 => arena.push_binary(OpKind::Add, cur, dy),
                4 => arena.push_ternary(OpKind::MulAdd, cur, c, dx2),
                5 => arena.push_binary(OpKind::Max, cur, dx),
                6 => arena.push_binary(OpKind::Sub, cur, c),
                _ => arena.push_binary(OpKind::Mul, cur, cur),
            };
            step += 1;
        }
        let mut optimizer =
            super::super::Optimizer::production().budget(super::super::Budget::Explicit {
                iterations: 4,
                minted: 3_000,
                applications: Some(20_000),
            });
        let mut egraph = optimizer.egraph();
        let root =
            super::super::insert(&arena, cur, &mut egraph, super::super::Vocabulary::Runtime)
                .expect("the SDF arena is representable");
        let optimized = optimizer.run(&mut egraph, root);
        assert!(
            optimized.stats.classes > 500,
            "fixture must saturate into a real graph ({} classes)",
            optimized.stats.classes
        );
        (egraph.into_graph(), root)
    }

    /// How many classes the pre-fixpoint DP left at the cycle sentinel: a
    /// class every one of whose candidates mentions itself or a class the
    /// single DFS post-order had not yet priced.
    fn sentinel_priced(egraph: &EGraph, root: EClassId) -> Vec<EClassId> {
        let order = AndOrGraph::of(egraph, root).classes().to_vec();
        let mut priced: Vec<bool> = alloc::vec![false; egraph.num_classes()];
        let mut sentinel = Vec::new();
        for &class in &order {
            let all_cyclic = egraph.nodes(class).iter().all(|node| match node {
                ENode::Op { children, .. } => children
                    .iter()
                    .any(|&c| egraph.find(c) == class || !priced[egraph.find(c).0 as usize]),
                _ => false,
            });
            if all_cyclic {
                sentinel.push(class);
            }
            priced[class.0 as usize] = true;
        }
        sentinel
    }

    /// The defect the fixpoint exists to remove: the post-order DP priced a
    /// class in a cycle at the sentinel and expressed no preference, so the
    /// pick fell to a repair that had no cost model.
    /// The fixpoint prices it — `neg(neg(x)) = x` means the `Neg` class's
    /// only child is the class that holds it, and settling in cost order
    /// reaches the child first.
    #[test]
    fn the_fixpoint_prices_a_class_the_post_order_dp_left_at_the_sentinel() {
        #[derive(Default)]
        struct Prices(Vec<(EClassId, usize)>);
        impl StageRecorder for Prices {
            fn candidate(&mut self, class: EClassId, _idx: usize, cost: usize, _own: usize) {
                self.0.push((class, cost));
            }
            fn settled(&mut self, _: EClassId, _: usize) {}
        }

        let (egraph, merged, n1) = cyclic_capable_egraph();
        let n1c = egraph.find(n1);
        assert!(
            sentinel_priced(&egraph, merged).contains(&n1c),
            "the fixture must hold a class the post-order DP could only price at the sentinel"
        );

        let costs = CostModel::latency_prior();
        let mut dp = Dp::new(&costs, LatticeShape::POINT, Insertion, Prices::default());
        let choices = tree_dp_pass(Frozen::new(&egraph, merged).problem(), &mut dp).choices;
        let priced = dp.into_recorder().0;

        assert!(
            priced.iter().any(|&(c, _)| c == n1c),
            "the class the post-order DP left at the sentinel was never priced"
        );
        assert!(
            priced.iter().all(|&(_, cost)| cost < CYCLE_COST),
            "the fixpoint priced a candidate at the cycle sentinel: {priced:?}"
        );
        assert!(
            choices[n1c.0 as usize].is_some(),
            "the class was priced but not settled"
        );
    }

    /// Kahn admission over the chosen edges: a class is admitted once every
    /// class its chosen node names is, and a well-founded map admits every
    /// class that holds a pick.
    ///
    /// This is a different walk from `Selection::of_map`, which follows only
    /// the picks reachable from the root. It reads the whole map, so a pick
    /// off the chosen subgraph (which the reranker's `try_swap` still reads)
    /// is covered too.
    fn assert_every_pick_is_admitted(egraph: &EGraph, choices: &[Option<usize>], what: &str) {
        let picks: Vec<(usize, Vec<usize>)> = choices
            .iter()
            .enumerate()
            .filter_map(|(i, pick)| {
                let class = EClassId(i as u32);
                let node = &egraph.nodes(class)[(*pick)?];
                let children = node
                    .children_slice()
                    .iter()
                    .map(|&c| egraph.find(c).index());
                Some((egraph.find(class).index(), children.collect()))
            })
            .collect();
        let mut admitted = alloc::vec![false; egraph.num_classes()];
        // A child with no pick is never admitted, so a pick that names one is
        // reported by the final assertion rather than skipped.
        loop {
            let before = admitted.iter().filter(|&&a| a).count();
            for (class, children) in &picks {
                if children.iter().all(|&c| admitted[c]) {
                    admitted[*class] = true;
                }
            }
            if admitted.iter().filter(|&&a| a).count() == before {
                break;
            }
        }
        for (class, _) in &picks {
            assert!(
                admitted[*class],
                "{what}: class {class} holds a pick that no admission order reaches"
            );
        }
    }

    /// A choice map out of either DP pass is well-founded: the hand-built
    /// constructor accepts it and names the term the pass selected, and every
    /// pick in it is admitted.
    #[test]
    fn the_dp_map_is_well_founded() {
        let costs = CostModel::latency_prior();
        for (nodes, shape) in [
            (64, LatticeShape::POINT),
            (256, LatticeShape::new([32, 32])),
        ] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            let tree = tree_dp_pass(
                Frozen::new(&egraph, root).problem(),
                &mut Dp::production(&costs, shape),
            );
            let shared = shared_dag_dp_pass(
                Frozen::new(&egraph, root).problem(),
                &mut Dp::production(&costs, shape),
                usize::MAX,
            )
            .outcome
            .expect("unbounded");
            for (arm, what) in [(tree, "tree arm"), (shared, "shared arm")] {
                assert_every_pick_is_admitted(&egraph, &arm.choices, what);
                assert_eq!(
                    Selection::of_map(&egraph, root, &arm.choices),
                    Ok(arm.selection),
                    "{what}: the settled map names a different term than the settlement made"
                );
            }
        }
    }

    /// The research tie-break seam changes nothing under production's
    /// instance.
    ///
    /// `Ties::Insertion` is the `impl` the shipped extractor uses, so
    /// `witness::extract_under` with it must return the same choice map and
    /// the same cost as `extract_dag_scoped` — on a saturated graph, at a
    /// frame shape, where ties are dense (the 2026-09-08 witness run found
    /// 47–81 % of live classes tied). Without this, a refactor of either DP
    /// pass could silently move production's extraction and only the
    /// research harness would see it.
    #[cfg(feature = "provenance-journal")]
    #[test]
    fn insertion_tie_break_is_productions_extraction() {
        use crate::egraph::witness::{Ties, extract_under};
        let costs = CostModel::latency_prior();
        for (nodes, shape) in [
            (64, LatticeShape::POINT),
            (256, LatticeShape::new([32, 32])),
        ] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            let production = extract_dag_scoped(&egraph, root, &costs, shape);
            let (choices, cost) = extract_under(&egraph, root, &costs, shape, Ties::Insertion);
            assert_eq!(
                choices, production.choices,
                "{nodes} nodes at {shape:?}: the Insertion tie-break moved production's choices"
            );
            assert_eq!(
                (cost.tree, cost.dag),
                (production.total_cost, production.dag_cost),
                "{nodes} nodes at {shape:?}: the Insertion tie-break moved production's cost"
            );
        }
    }

    /// FNV-1a over the choices as a stream of words: a class with no choice
    /// is `0`, a choice of node `i` is `i + 1`.
    fn choice_digest(choices: &[Option<usize>]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        choices.iter().fold(FNV_OFFSET, |hash, choice| {
            let word = choice.map_or(0, |i| i as u64 + 1);
            word.to_le_bytes()
                .iter()
                .fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(FNV_PRIME))
        })
    }

    /// **Settlement order is pinned, not only compared.** Equal-cost classes
    /// settle in `EClassId` order and, under `Insertion`, the settle order is
    /// what picks the incumbent, so a change to how the passes are set up
    /// (their live set, candidate rows, parent index) that reorders
    /// settlement moves a choice on a tie-dense graph without moving any
    /// cost. Every other test here compares two paths that share that setup,
    /// so they move together; these digests were taken before the setup was
    /// lifted into [`AndOrGraph`] and do not.
    ///
    /// One digest per (graph, shape) and per arm: the tree arm, the shared
    /// arm, and the arm production returns.
    #[test]
    fn settlement_order_is_pinned() {
        let costs = CostModel::latency_prior();
        let observed: Vec<(usize, LatticeShape, [u64; 3])> = [
            (64, LatticeShape::POINT),
            (256, LatticeShape::POINT),
            (256, LatticeShape::new([32, 32])),
        ]
        .into_iter()
        .map(|(nodes, shape)| {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            let (tree, shared) = extract_dag_objectives(&egraph, root, &costs, shape);
            let scoped = extract_dag_scoped(&egraph, root, &costs, shape);
            let digests = [&tree, &shared, &scoped].map(|dag| choice_digest(&dag.choices));
            (nodes, shape, digests)
        })
        .collect();
        let recorded: [[u64; 3]; 3] = SETTLEMENT_DIGESTS;
        for ((nodes, shape, digests), expected) in observed.iter().zip(recorded) {
            assert_eq!(
                *digests, expected,
                "{nodes} nodes at {shape:?}: settlement picked other nodes than it did when the \
                 digests were recorded (tree, shared, production)"
            );
        }
    }

    /// Recorded at the commit before the passes shared an [`AndOrGraph`].
    const SETTLEMENT_DIGESTS: [[u64; 3]; 3] = [
        [
            0x0126_79f9_2714_6ff2,
            0xf9c8_412f_a72c_3fe0,
            0xf9c8_412f_a72c_3fe0,
        ],
        [
            0xf9d6_e836_a077_a4e6,
            0xf9d6_e836_a077_a4e6,
            0xf9d6_e836_a077_a4e6,
        ],
        [
            0xf142_5de0_5476_17e6,
            0x4cc2_eca9_a5d6_21e4,
            0xf142_5de0_5476_17e6,
        ],
    ];

    /// The budgeted hybrid-set pass and the dense pass it replaced choose
    /// the same node in every class of a saturated graph — same objective,
    /// same tie-breaking, priced through sparse and dense sets alike —
    /// under both the point and a frame lattice. This is what lets the
    /// committed extraction rows stand as this pass's regression baseline.
    #[test]
    fn shared_pass_matches_the_dense_reference_on_a_saturated_graph() {
        let costs = CostModel::latency_prior();
        for (nodes, shape) in [
            (64, LatticeShape::POINT),
            (256, LatticeShape::POINT),
            (256, LatticeShape::new([32, 32])),
        ] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            assert!(
                sentinel_priced(&egraph, root).is_empty(),
                "{nodes} nodes: the reference could only price some class at the sentinel, so \
                 it has no opinion there and this comparison would be vacuous — see \
                 `the_fixpoint_prices_a_class_the_post_order_dp_left_at_the_sentinel`"
            );
            let (reference_choices, reference_cost) =
                dense_reference_pass(&egraph, root, &costs, shape);
            let pass = shared_dag_dp_pass(
                Frozen::new(&egraph, root).problem(),
                &mut Dp::production(&costs, shape),
                usize::MAX,
            );
            let live = pass.stats.live_classes;
            assert!(live > 100, "{nodes} nodes: only {live} live classes");
            let dp = pass.outcome.expect("unbounded");
            assert_eq!(
                dp.choices, reference_choices,
                "{nodes} nodes at {shape:?}: the hybrid pass disagrees with the dense reference"
            );
            // #1229 pinned the choices; the claim beside them is the number
            // the pass minimized, and a sparse/dense split that agreed on the
            // map while disagreeing on its price would be a mispricing this
            // test was blind to.
            assert_eq!(
                dp.root_cost, reference_cost,
                "{nodes} nodes at {shape:?}: the hybrid pass and the dense reference price the \
                 same map differently"
            );
            let dense_bytes = live * live.div_ceil(REACH_WORD_BITS) * 8;
            assert!(
                pass.stats.reach_bytes <= dense_bytes,
                "{nodes} nodes: hybrid sets ({}) above the dense bound ({dense_bytes})",
                pass.stats.reach_bytes
            );
        }
    }

    #[test]
    fn extract_simple() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, x, &costs);
        let cost = dag.total_cost;
        let (arena, root) = dag.to_arena(&egraph);

        assert_eq!(arena.len(), 1);
        assert_eq!(root.0, 0);
        assert_eq!(cost, 0); // Leaf nodes (Var/Const) have cost 0
    }

    #[test]
    fn extract_with_ops() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::apply(OpKind::Add, &[x, y]));

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, sum, &costs);
        let (arena, root) = dag.to_arena(&egraph);

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

        let x_plus_x = egraph.add(ENode::apply(OpKind::Add, &[x, x]));
        let x_times_2 = egraph.add(ENode::apply(OpKind::Mul, &[x, two]));

        egraph.union(x_plus_x, x_times_2);

        let costs = CostModel::latency_prior();
        assert!(
            costs.cost(pixelflow_ir::OpKind::Add) < costs.cost(pixelflow_ir::OpKind::Mul),
            "test assumes Add is strictly cheaper than Mul in the latency prior"
        );

        let dag = extract_dag(&egraph, egraph.find(x_plus_x), &costs);
        let cost = dag.total_cost;
        let (arena, root) = dag.to_arena(&egraph);

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
    // Swap-refinement search (Reranker seam)
    // ========================================================================

    /// A trivial, test-only [`Reranker`]: the sum of `costs.cost(op)` over
    /// every node the candidate's materialised arena contains — the same
    /// additive latency-prior table [`extract_dag`] minimizes, just summed
    /// over the arena instead of folded bottom-up through the e-graph. No
    /// sharing discount (each arena node is already deduped by
    /// `Selection::to_arena`, so this is a DAG cost, not a tree cost) — fine
    /// for the sharing-free graphs these tests build.
    struct TableReranker<'a> {
        costs: &'a CostModel,
    }

    impl Reranker for TableReranker<'_> {
        fn score(&self, _extraction: &Extraction<'_>, arena: &pixelflow_ir::ExprArena) -> f64 {
            let mut total = 0.0f64;
            for i in 0..arena.len() {
                let id = pixelflow_ir::ExprId(i as u32);
                total += self.costs.cost(arena.kind(id)) as f64;
            }
            total
        }
    }

    #[test]
    fn swap_search_reproduces_extract_dags_choice_on_a_cost_ambiguous_class() {
        // Same setup as `extract_latency_prior_picks_cheaper_equivalent_form`:
        // x+x and x*2 are unioned, so the class has a genuine choice to make.
        // The bootstrap pass may start on either node; the swap-refinement
        // loop must walk to the same cheaper form `extract_dag` reaches via
        // its (unrelated) bottom-up DP — same table, same answer, two
        // different search strategies.
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let two = egraph.add(ENode::constant(2.0));
        let x_plus_x = egraph.add(ENode::apply(OpKind::Add, &[x, x]));
        let x_times_2 = egraph.add(ENode::apply(OpKind::Mul, &[x, two]));
        let merged = egraph.union(x_plus_x, x_times_2);
        egraph.rebuild();

        let costs = CostModel::latency_prior();
        let dag = extract_dag(&egraph, merged, &costs);

        let reranker = TableReranker { costs: &costs };
        let extractor = IncrementalExtractor::new(&reranker, 8);
        let (search_cost, extraction) = extractor.extract_choices_only(&egraph, merged);
        let (arena, root) = extraction.to_selection().to_arena(&egraph);

        assert_eq!(
            search_cost, dag.total_cost as f64,
            "swap-refinement search must reach the same additive cost as extract_dag's DP"
        );
        let root_node = arena.node(root);
        assert!(
            matches!(
                root_node,
                pixelflow_ir::arena::ExprNode::Binary(pixelflow_ir::OpKind::Add, _, _)
            ),
            "swap search should have converged on the Add form, got {root_node:?}"
        );
    }

    #[test]
    fn swap_search_bootstrap_survives_a_merge_reordered_class() {
        // The real round-1 failure this search once hit: the old bootstrap
        // picked node index 0 everywhere, but after saturation merges "node
        // 0" of two classes can reference each other — a CYCLIC bootstrap.
        // `Extraction::from_backfill` (which the search's bootstrap pass
        // uses) must return a well-founded choice set for ANY node
        // ordering, which materialises without panicking regardless of
        // what the reranker says.
        let (egraph, merged, _n1) = cyclic_capable_egraph();
        let costs = CostModel::latency_prior();
        let reranker = TableReranker { costs: &costs };
        let extractor = IncrementalExtractor::new(&reranker, 8);
        let (_cost, extraction) = extractor.extract_choices_only(&egraph, merged);
        let (arena, root) = extraction.to_selection().to_arena(&egraph);
        assert!(!arena.is_empty());
        assert!(root.0 < arena.len() as u32);
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
        let n1 = egraph.add(ENode::apply(OpKind::Neg, &[x]));
        let n2 = egraph.add(ENode::apply(OpKind::Neg, &[n1]));
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

    /// The hand-built map that closes a loop: `merged` picks `Neg(n1)` and
    /// `n1` picks `Neg(merged)`.
    fn cyclic_choice_set() -> (EGraph, EClassId, Vec<Option<usize>>) {
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));
        (egraph, merged, choices)
    }

    /// Before a cycle was refused, materializing one re-scheduled it forever:
    /// a full-DEV bench run grew to 2.7GB and died by SIGKILL with zero
    /// diagnostics. A map that names no finite term is refused, with the
    /// class it loops through.
    #[test]
    fn a_selection_is_refused_for_a_cyclic_choice_set() {
        let (egraph, merged, choices) = cyclic_choice_set();
        let refused = Selection::of_map(&egraph, merged, &choices)
            .expect_err("a cyclic choice set names no term");
        assert!(matches!(refused, Refused::Cyclic(_)), "{refused}");
    }

    #[test]
    #[should_panic(expected = "CYCLIC")]
    fn an_extraction_refuses_to_select_a_cyclic_choice_set() {
        let (egraph, merged, choices) = cyclic_choice_set();
        let extraction = Extraction {
            egraph: &egraph,
            root: egraph.find(merged),
            choices,
        };
        let _ = extraction.to_selection();
    }

    #[test]
    fn a_selection_is_refused_for_an_incomplete_or_out_of_range_choice_set() {
        let (egraph, merged, mut choices) = cyclic_choice_set();
        let m = egraph.find(merged);
        choices[m.index()] = None;
        assert_eq!(
            Selection::of_map(&egraph, merged, &choices),
            Err(Refused::Missing(m))
        );
        choices[m.index()] = Some(egraph.nodes(m).len());
        assert_eq!(
            Selection::of_map(&egraph, merged, &choices),
            Err(Refused::OutOfBounds {
                class: m,
                idx: egraph.nodes(m).len()
            })
        );
    }

    #[test]
    #[should_panic(expected = "cyclic")]
    fn extraction_constructors_refuse_a_cyclic_choice_set() {
        // `from_backfill` refuses a cyclic vector up front.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));
        let _ = Extraction::from_backfill(&egraph, merged, choices);
    }

    /// Prices `Var` above two `Neg`s, so on the cyclic-capable graph the
    /// mutual pick (`merged` takes `Neg(n1)`, `n1` takes `Neg(merged)`) is
    /// what a cost-greedy walk would prefer to the leaf.
    struct VarAboveTwoNegs;

    impl CostFunction for VarAboveTwoNegs {
        fn node_cost(&self, node: &ENode, _parent: Option<pixelflow_ir::OpKind>) -> usize {
            match node {
                ENode::Var(_) => VAR_ABOVE_TWO_NEGS,
                ENode::Op { .. } | ENode::Shift { .. } | ENode::Reduce { .. } => 1,
                ENode::Const(_)
                | ENode::Buffer(_)
                | ENode::Uniform(_)
                | ENode::Param(_)
                | ENode::Ref { .. } => 0,
            }
        }
    }

    const VAR_ABOVE_TWO_NEGS: usize = 100;

    /// The cyclic-capable graph (`neg(neg(x)) = x`) is the one shape a map
    /// can go wrong on, and with `Var` dearer than the pair of `Neg`s the
    /// mutual pick is the cheaper-looking one. Settlement admits `Neg(n1)`
    /// only after `n1`, which needs the merged class, so the leaf is the pick
    /// and its cost is the one reported.
    #[test]
    fn production_extraction_of_a_cyclic_capable_graph_picks_the_leaf() {
        let (egraph, merged, _n1) = cyclic_capable_egraph();
        let var_idx = egraph
            .nodes(merged)
            .iter()
            .position(|n| matches!(n, ENode::Var(_)))
            .expect("merged class holds Var(0)");
        let dag = extract_dag(&egraph, merged, &VarAboveTwoNegs);
        assert_eq!(dag.choices[egraph.find(merged).index()], Some(var_idx));
        assert_eq!(dag.total_cost, VAR_ABOVE_TWO_NEGS);
        assert_eq!(dag.dag_cost, VAR_ABOVE_TWO_NEGS);
        let (arena, root) = dag.to_arena(&egraph);
        assert!(matches!(
            arena.node(root),
            pixelflow_ir::arena::ExprNode::Var(_)
        ));
    }

    // ========================================================================
    // A shared class is one node of the term
    // ========================================================================

    /// A class used several times is one pick, so one arena node and one
    /// price: the sharing an emitted kernel's let-bindings come from.
    #[test]
    fn a_class_used_many_times_is_one_node_of_the_term() {
        let costs = CostModel::default();
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::apply(OpKind::Add, &[x, y]));
        // (X + Y) * (X + Y)
        let product = egraph.add(ENode::apply(OpKind::Mul, &[sum, sum]));
        // sqrt(X) * sqrt(X) + sqrt(X): used three times
        let sqrt_x = egraph.add(ENode::apply(OpKind::Sqrt, &[x]));
        let sqrt_squared = egraph.add(ENode::apply(OpKind::Mul, &[sqrt_x, sqrt_x]));
        let triple = egraph.add(ENode::apply(OpKind::Add, &[sqrt_squared, sqrt_x]));

        // (root, distinct classes the term holds)
        for (root, distinct) in [(sum, 3), (product, 4), (sqrt_squared, 3), (triple, 4)] {
            let dag = extract_dag(&egraph, root, &costs);
            assert_eq!(dag.root, egraph.find(root));
            assert_eq!(
                dag.selection.len(),
                distinct,
                "distinct classes of the term"
            );
            assert_eq!(
                dag.to_arena(&egraph).0.len(),
                distinct,
                "one node per class"
            );
        }
    }

    // =========================================================================
    // Train/deploy feature-path equivalence (2026-08 round-0 skew guard)
    // =========================================================================

    #[test]
    fn edge_trace_records_a_reload_for_a_shared_subexpression() {
        use crate::nnue::EdgeTrace;

        // sqrt(X) * sqrt(X): tree has 2x sqrt edges, DAG has 1x sqrt + 1x reload
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let sqrt_x = egraph.add(ENode::apply(OpKind::Sqrt, &[x]));
        let product = egraph.add(ENode::apply(OpKind::Mul, &[sqrt_x, sqrt_x]));

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(product).0 as usize] = Some(0);
        choices[egraph.find(sqrt_x).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, product, choices);
        let (term, term_root) = extraction.to_selection().to_arena(&egraph);
        let trace = EdgeTrace::from_arena_dag(&term, term_root);

        assert_eq!(trace.node_count(), 3, "3 unique nodes");
        assert_eq!(
            trace.edges().len(),
            3,
            "shared reuse should contribute a reload edge"
        );
        let reloads = trace
            .edges()
            .iter()
            .filter(|e| {
                e.parent == pixelflow_ir::OpKind::Mul && e.child == pixelflow_ir::OpKind::Var
            })
            .count();
        assert_eq!(reloads, 1);
    }

    // =========================================================================
    // Arena/extraction edge-walk equivalence (2026-08 round-0 skew guard)
    // =========================================================================

    /// The term a selection materializes as must record the edge stream of
    /// the arena it denotes, built by hand: the DAG — shared subexpressions,
    /// shared leaves — survives materialization with its sharing intact, so
    /// the training-side walk of a hand-built arena and the deploy-side walk
    /// of an extracted one cannot disagree about it.
    #[test]
    fn a_materialized_selection_records_the_edge_stream_of_its_arena() {
        use crate::nnue::EdgeTrace;
        use pixelflow_ir::{ExprArena, OpKind};

        // Arena: (sin(Z * 0.3) * (X + Y) + sin(Z * 0.3)) + Y * 0.3
        // - sin(Z * 0.3) is SHARED (register reload on the second reference)
        // - Y and 0.3 are shared leaves (leaf reload policy)
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
        let mut eg = EGraph::new();
        let ez = eg.add(ENode::Var(2));
        let ec = eg.add(ENode::constant(0.3));
        let ezm = eg.add(ENode::apply(OpKind::Mul, &[ez, ec]));
        let esin = eg.add(ENode::apply(OpKind::Sin, &[ezm]));
        let ex = eg.add(ENode::Var(0));
        let ey = eg.add(ENode::Var(1));
        let exy = eg.add(ENode::apply(OpKind::Add, &[ex, ey]));
        let em = eg.add(ENode::apply(OpKind::Mul, &[esin, exy]));
        let ea = eg.add(ENode::apply(OpKind::Add, &[em, esin]));
        let eyc = eg.add(ENode::apply(OpKind::Mul, &[ey, ec]));
        let eroot = eg.add(ENode::apply(OpKind::Add, &[ea, eyc]));

        let choices: Vec<Option<usize>> = alloc::vec![None; eg.num_classes()];
        let extraction = Extraction::from_backfill(&eg, eroot, choices);

        let from_arena = EdgeTrace::from_arena_dag(&arena, root);
        let (term, term_root) = extraction.to_selection().to_arena(&eg);
        let from_term = EdgeTrace::from_arena_dag(&term, term_root);

        assert_eq!(from_arena.node_count(), 11, "11 distinct nodes");
        assert_eq!(
            from_arena, from_term,
            "the materialized term and the hand-built arena must record the identical \
             edge stream"
        );

        // A shift materializes in the arena's spelling, `Shl(x, Const)`, so
        // its stream is that arena's, count included.
        let mut shifted = ExprArena::new();
        let sx = shifted.push_var(0);
        let count = shifted.push_const(0.0);
        let shl = shifted.push_binary(OpKind::Shl, sx, count);
        let from_shifted = EdgeTrace::from_arena_dag(&shifted, shl);

        let mut shift_graph = EGraph::new();
        let shift_x = shift_graph.add(ENode::Var(0));
        let shift_node = shift_graph.add(ENode::Shift {
            direction: ShiftDirection::Left,
            count: ShiftCount::from_literal(0.0).expect("0 is a count"),
            operand: shift_x,
        });
        let extraction = Extraction::from_backfill(
            &shift_graph,
            shift_node,
            alloc::vec![None; shift_graph.num_classes()],
        );
        let (term, term_root) = extraction.to_selection().to_arena(&shift_graph);
        assert_eq!(
            from_shifted,
            EdgeTrace::from_arena_dag(&term, term_root),
            "a shift's stream is the arena spelling's, count included"
        );
    }

    // =========================================================================
    // Extraction::chosen_variance (2026-09-01: denotation kept, accumulator
    // shape deleted — see the module doc comment above
    // `crate::nnue::factored::variance_histogram`)
    // =========================================================================

    /// A known const/frame/scanline/pixel mix, built once as an arena and
    /// once as the equivalent e-graph, must classify identically: the term an
    /// extraction materializes is the DAG that was built by hand —
    /// mirroring `a_materialized_selection_records_the_edge_stream_of_its_arena`
    /// above.
    #[test]
    fn a_chosen_term_classifies_a_known_variance_mix_as_its_arena_does() {
        use crate::nnue::factored::variance_histogram;
        use pixelflow_ir::{ExprArena, OpKind};

        // Add(Add(Const(2.0), W), Add(Y, X)):
        // - Const(2.0)        -> const
        // - W (var 3)         -> frame     (no X, no Y)
        // - Add(Const, W)     -> frame
        // - Y (var 1)         -> scanline  (no X)
        // - X (var 0)         -> pixel
        // - Add(Y, X)         -> pixel     (depends on X)
        // - root Add          -> pixel     (depends on X)
        // 1 const, 2 frame, 1 scanline, 3 pixel of 7 nodes.
        let mut arena = ExprArena::new();
        let c = arena.push_const(2.0);
        let w = arena.push_var(3);
        let frame_sum = arena.push_binary(OpKind::Add, c, w);
        let y = arena.push_var(1);
        let x = arena.push_var(0);
        let xy = arena.push_binary(OpKind::Add, y, x);
        let root = arena.push_binary(OpKind::Add, frame_sum, xy);

        let mut eg = EGraph::new();
        let ec = eg.add(ENode::constant(2.0));
        let ew = eg.add(ENode::Var(3));
        let efs = eg.add(ENode::apply(OpKind::Add, &[ec, ew]));
        let ey = eg.add(ENode::Var(1));
        let ex = eg.add(ENode::Var(0));
        let exy = eg.add(ENode::apply(OpKind::Add, &[ey, ex]));
        let eroot = eg.add(ENode::apply(OpKind::Add, &[efs, exy]));

        let choices: Vec<Option<usize>> = alloc::vec![None; eg.num_classes()];
        let extraction = Extraction::from_backfill(&eg, eroot, choices);

        let from_arena = variance_histogram(&arena);
        let from_extraction = extraction.chosen_variance();

        assert_eq!(
            from_arena, from_extraction,
            "the chosen term and the hand-built arena must classify identically"
        );
        assert_eq!(
            from_arena,
            [1.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 3.0 / 7.0],
            "known const/frame/scanline/pixel mix: {from_arena:?}"
        );
        let _ = root; // arena root; classification is over every node.
    }

    // =========================================================================
    // Selection::to_arena tests
    // =========================================================================

    /// X + Y should produce an arena with exactly 3 nodes: Var(0), Var(1), Add.
    #[test]
    fn a_selection_materializes_in_post_order() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let add = egraph.add(ENode::apply(OpKind::Add, &[x, y]));

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(add).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, add, choices);
        let (arena, root_id) = extraction.to_selection().to_arena(&egraph);

        assert_eq!(arena.len(), 3, "X + Y should have exactly 3 arena nodes");
        // Root should be the last node (post-order: X, Y, Add)
        assert_eq!(root_id.0, 2, "root ExprId should be 2 (the Add node)");
    }

    /// A shift reaches the arena as the arena spells it, its count a `Const`
    /// emitted at the shift, after its operand and before it, where a walk
    /// of the term meets them. A literal another node names too is the same
    /// arena node, whichever use reaches it first.
    #[test]
    fn a_shift_materializes_with_its_count_at_its_position_and_shares_the_literal() {
        use pixelflow_ir::{ExprArena, OpKind};

        // root = Add(Shift(x, 3), Mul(y, three)): `three` is an ordinary
        // operand later, and a count first.
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let three = eg.add(ENode::constant(3.0));
        let shifted = eg.add(ENode::Shift {
            direction: ShiftDirection::Left,
            count: ShiftCount::from_literal(3.0).expect("3 is a count"),
            operand: x,
        });
        let product = eg.add(ENode::apply(OpKind::Mul, &[y, three]));
        let root = eg.add(ENode::apply(OpKind::Add, &[shifted, product]));

        let extraction = Extraction::from_backfill(&eg, root, alloc::vec![None; eg.num_classes()]);
        let (arena, arena_root) = extraction.to_selection().to_arena(&eg);

        let mut expected = ExprArena::new();
        let ex = expected.push_var(0);
        let e3 = expected.push_const(3.0);
        let eshl = expected.push_binary(OpKind::Shl, ex, e3);
        let ey = expected.push_var(1);
        let emul = expected.push_binary(OpKind::Mul, ey, e3);
        let eroot = expected.push_binary(OpKind::Add, eshl, emul);
        assert_eq!(arena_root, eroot);
        assert_eq!(
            arena.nodes().collect::<Vec<_>>(),
            expected.nodes().collect::<Vec<_>>(),
            "the count is emitted at the shift, and the Mul reuses it"
        );
    }

    /// A shift under a fold body materializes with its count too, and a
    /// literal the fold's sibling names is the same arena node.
    #[test]
    fn a_shift_under_a_fold_body_materializes_with_its_count() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};
        use pixelflow_ir::{ExprArena, OpKind};

        let binder = Binder::from_slot(0).expect("a live binder");
        let fold = Fold::new(Monoid::SUM, binder, 0..4);

        // root = Add(Reduce(Shift(x, 3)), three)
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let three = eg.add(ENode::constant(3.0));
        let shifted = eg.add(ENode::Shift {
            direction: ShiftDirection::Left,
            count: ShiftCount::from_literal(3.0).expect("3 is a count"),
            operand: x,
        });
        let reduce = eg.add(ENode::Reduce {
            fold,
            body: shifted,
        });
        let root = eg.add(ENode::apply(OpKind::Add, &[reduce, three]));

        let extraction = Extraction::from_backfill(&eg, root, alloc::vec![None; eg.num_classes()]);
        let (arena, arena_root) = extraction.to_selection().to_arena(&eg);

        let mut expected = ExprArena::new();
        let ex = expected.push_var(0);
        let e3 = expected.push_const(3.0);
        let eshl = expected.push_binary(OpKind::Shl, ex, e3);
        let ereduce = expected.push_reduce(fold, eshl);
        let eroot = expected.push_binary(OpKind::Add, ereduce, e3);
        assert_eq!(arena_root, eroot);
        assert_eq!(
            arena.nodes().collect::<Vec<_>>(),
            expected.nodes().collect::<Vec<_>>(),
            "the count is the Const under the fold and beside it"
        );
    }

    /// X * X should produce an arena with exactly 2 nodes: Var(0) and Mul.
    /// The shared Var(0) e-class must reuse one ExprId rather than being duplicated.
    #[test]
    fn a_shared_class_materializes_once() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let mul = egraph.add(ENode::apply(OpKind::Mul, &[x, x]));

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(mul).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, mul, choices);
        let (arena, root_id) = extraction.to_selection().to_arena(&egraph);

        assert_eq!(
            arena.len(),
            2,
            "X * X should have exactly 2 arena nodes (X shared)"
        );
        assert_eq!(root_id.0, 1, "root ExprId should be 1 (the Mul node)");
    }

    /// A hand-built full choice map sealed by `from_backfill` materializes the
    /// same arena as the production extractor's term, for a tree-shaped
    /// input.
    #[test]
    fn a_backfilled_hand_map_materializes_what_production_extracts() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let add = egraph.add(ENode::apply(OpKind::Add, &[x, y]));

        let num_classes = egraph.num_classes();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; num_classes];
        choices[egraph.find(add).0 as usize] = Some(0);
        choices[egraph.find(x).0 as usize] = Some(0);
        choices[egraph.find(y).0 as usize] = Some(0);

        let extraction = Extraction::from_backfill(&egraph, add, choices);
        let (arena, root_id) = extraction.to_selection().to_arena(&egraph);
        let dag = extract_dag(&egraph, add, &CostModel::default());
        let (extracted_arena, extracted_root) = dag.to_arena(&egraph);
        assert_eq!(arena.len(), extracted_arena.len());
        assert_eq!(root_id, extracted_root);
    }

    // ========================================================================
    // The reported cost describes the returned term (#1111)
    // ========================================================================

    /// Latency-prior DAG cost of a materialized arena: every reachable
    /// operation priced once, leaves free — the independent statement of
    /// what [`ExtractedDAG::dag_cost`] claims, computed from the arena
    /// instead of from the choices. Deliberately a second implementation:
    /// the point of `dag_cost_equals_the_materialized_arenas_cost` is that
    /// two walks over two representations agree.
    fn arena_dag_cost(
        arena: &pixelflow_ir::ExprArena,
        root: pixelflow_ir::ExprId,
        costs: &CostModel,
    ) -> usize {
        use pixelflow_ir::arena::ExprNode;
        let mut seen = alloc::vec![false; arena.len()];
        let mut stack = alloc::vec![root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let kind = match arena.node(id) {
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => None,
                ExprNode::Unary(k, _)
                | ExprNode::Binary(k, _, _)
                | ExprNode::Ternary(k, _, _, _) => Some(k),
                other => panic!("unexpected extracted node {other:?}"),
            };
            if let Some(k) = kind {
                total = total.saturating_add(costs.cost(k));
            }
            stack.extend(arena.children(id));
        }
        total
    }

    /// `sin(X) * sin(X) + sin(X)` — one `Sin` reached three times, so the
    /// tree and DAG costs of the same term are genuinely different numbers.
    /// One e-class, two forms: `sin(X) * sin(X)`, which reuses one `Sin`,
    /// and `ln(X)`, which reuses nothing. The tree objective charges the
    /// `Sin` twice and picks `ln`; the emitted kernel would then compute a
    /// `Ln` where a `Mul` over an already-live `Sin` was cheaper.
    ///
    /// This is #1116 in four e-nodes.
    fn sharing_vs_flat_egraph() -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::apply(OpKind::Sin, &[x]));
        let squared = egraph.add(ENode::apply(OpKind::Mul, &[s, s]));
        let flat = egraph.add(ENode::apply(OpKind::Ln, &[x]));
        egraph.union(squared, flat);
        egraph.rebuild();
        let root = egraph.find(squared);
        (egraph, root)
    }

    #[test]
    fn the_sharing_objective_picks_the_form_that_reuses_a_subterm() {
        let (egraph, root) = sharing_vs_flat_egraph();
        let costs = CostModel::latency_prior();

        let sin = costs.cost(pixelflow_ir::OpKind::Sin);
        let mul = costs.cost(pixelflow_ir::OpKind::Mul);
        let ln = costs.cost(pixelflow_ir::OpKind::Ln);
        // The fixture only says anything if `ln` sits strictly between the
        // DAG price of the shared form and its tree price. Assert that here
        // rather than let a cost-table refresh quietly defuse the test.
        assert!(
            mul + sin < ln && ln < mul + sin + sin,
            "fixture defused by the cost table: mul {mul} + sin {sin} vs ln {ln}"
        );

        let (tree, shared) = extract_dag_objectives(&egraph, root, &costs, LatticeShape::POINT);
        assert_eq!(
            tree.dag_cost, ln,
            "the tree objective charges the shared Sin twice and takes the Ln"
        );
        assert_eq!(
            shared.dag_cost,
            mul + sin,
            "the sharing objective prices the Sin once and takes the Mul"
        );
        assert_eq!(
            extract_dag(&egraph, root, &costs).dag_cost,
            mul + sin,
            "extract_dag returns the cheaper of the two by DAG cost"
        );
    }

    /// The no-regression property `extract_dag_scoped` is built on: it
    /// returns the cheaper of the two objectives by true DAG cost, so it can
    /// never be worse than the extractor it replaced.
    #[test]
    fn extract_dag_is_never_worse_than_the_tree_objective_it_replaced() {
        let costs = CostModel::latency_prior();
        for (label, (egraph, root)) in [
            ("shared_sin", shared_sin_egraph()),
            ("sharing_vs_flat", sharing_vs_flat_egraph()),
        ] {
            let (tree, _) = extract_dag_objectives(&egraph, root, &costs, LatticeShape::POINT);
            let chosen = extract_dag(&egraph, root, &costs);
            assert!(
                chosen.dag_cost <= tree.dag_cost,
                "{label}: extract_dag returned {} against the tree arm's {}",
                chosen.dag_cost,
                tree.dag_cost
            );
        }
    }

    /// With nothing shared, the two objectives are the same function, so the
    /// sharing pass must not perturb an unshared kernel at all.
    #[test]
    fn the_two_objectives_agree_when_nothing_is_shared() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let a = egraph.add(ENode::apply(OpKind::Sin, &[x]));
        let b = egraph.add(ENode::apply(OpKind::Sqrt, &[a]));
        egraph.rebuild();
        let costs = CostModel::latency_prior();
        let (tree, shared) = extract_dag_objectives(&egraph, b, &costs, LatticeShape::POINT);
        assert_eq!(tree.dag_cost, shared.dag_cost);
        assert_eq!(tree.total_cost, shared.total_cost);
        assert_eq!(tree.choices, shared.choices);
    }

    fn shared_sin_egraph() -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::apply(OpKind::Sin, &[x]));
        let sq = egraph.add(ENode::apply(OpKind::Mul, &[s, s]));
        let root = egraph.add(ENode::apply(OpKind::Add, &[sq, s]));
        egraph.rebuild();
        (egraph, root)
    }

    /// The property every measurement in this repo assumes when it re-costs
    /// the materialized arena rather than reading the extraction's own field
    /// (`runtime.rs`'s `arena_cost`, `ir_bridge.rs`'s namesake, #1101's
    /// harness): `dag_cost` IS that number, so the workaround and the field
    /// agree.
    #[test]
    fn dag_cost_equals_the_materialized_arenas_cost() {
        let (egraph, root) = shared_sin_egraph();
        let costs = CostModel::latency_prior();
        let dag = extract_dag(&egraph, root, &costs);

        let (arena, arena_root) = dag.to_arena(&egraph);

        assert_eq!(
            dag.dag_cost,
            arena_dag_cost(&arena, arena_root, &costs),
            "ExtractedDAG::dag_cost must equal the latency-prior cost of the arena \
             the same term materializes as"
        );
    }

    /// The two reported numbers are not two spellings of one quantity: the
    /// DP minimizes tree cost, the kernel pays DAG cost, and on a term with
    /// any sharing they differ. (`shader:julia_set` is the extreme: ~1.4e7
    /// against 716 — `docs/results/2026-09-02-extraction-gap.md`.)
    #[test]
    fn tree_cost_prices_a_shared_subterm_once_per_use_and_dag_cost_once() {
        let (egraph, root) = shared_sin_egraph();
        let costs = CostModel::latency_prior();
        let dag = extract_dag(&egraph, root, &costs);

        let sin = costs.cost(pixelflow_ir::OpKind::Sin);
        let mul = costs.cost(pixelflow_ir::OpKind::Mul);
        let add = costs.cost(pixelflow_ir::OpKind::Add);
        assert!(sin > 0, "the fixture needs a Sin that costs something");

        assert_eq!(
            dag.dag_cost,
            sin + mul + add,
            "the DAG cost pays the one shared Sin once"
        );
        assert_eq!(
            dag.total_cost,
            3 * sin + mul + add,
            "the tree cost pays it at each of its three uses"
        );
    }

    /// A cost function that prices `Sin` above `CYCLE_COST`, so a class
    /// holding a self-referential node alongside a `Sin` has a self-reference
    /// that a DP pricing cycles at that sentinel would prefer. The only way
    /// to reach that with an off-the-shelf table would be a lattice big
    /// enough to saturate the weighting — this says the same thing in one
    /// line.
    struct SinAboveTheCycleSentinel;

    impl CostFunction for SinAboveTheCycleSentinel {
        fn node_cost(&self, node: &ENode, _parent: Option<pixelflow_ir::OpKind>) -> usize {
            match node {
                ENode::Op { op, .. } if *op == pixelflow_ir::OpKind::Sin => usize::MAX / 2,
                ENode::Op { .. } | ENode::Shift { .. } | ENode::Reduce { .. } => 1,
                ENode::Var(_)
                | ENode::Const(_)
                | ENode::Buffer(_)
                | ENode::Uniform(_)
                | ENode::Param(_)
                | ENode::Ref { .. } => 0,
            }
        }
    }

    /// `sin(X)` unioned with `neg(sin(X))`, so one class holds both `Sin(x)`
    /// and a `Neg` whose child is that same class. Returns
    /// (egraph, the merged class, index of `Sin` in it, index of `Neg`).
    fn self_referential_pick_egraph() -> (EGraph, EClassId, usize, usize) {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::apply(OpKind::Sin, &[x]));
        let n = egraph.add(ENode::apply(OpKind::Neg, &[s]));
        let merged = egraph.union(s, n);
        egraph.rebuild();

        let canonical = egraph.find(merged);
        let nodes = egraph.nodes(canonical);
        let sin_idx = nodes
            .iter()
            .position(|nd| matches!(nd, ENode::Op { op, .. } if *op == pixelflow_ir::OpKind::Sin))
            .expect("merged class holds Sin(x)");
        let neg_idx = nodes
            .iter()
            .position(|nd| matches!(nd, ENode::Op { op, .. } if *op == pixelflow_ir::OpKind::Neg))
            .expect("merged class holds Neg(merged)");
        (egraph, canonical, sin_idx, neg_idx)
    }

    /// #1111: the reported cost is the cost of the term that comes back.
    ///
    /// The merged class holds `Sin` (priced `usize::MAX / 2`) and a `Neg`
    /// whose only child is the class itself. Settlement never admits a
    /// candidate that names its own class, so `Sin` is picked outright and
    /// both reported numbers are `Sin`'s — not `CYCLE_COST`, which names a
    /// term nobody receives.
    #[test]
    fn a_self_referential_candidate_is_never_the_pick_however_dear_the_alternative() {
        let (egraph, merged, sin_idx, neg_idx) = self_referential_pick_egraph();
        assert_ne!(sin_idx, neg_idx);

        let mut self_referential: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        self_referential[merged.0 as usize] = Some(neg_idx);
        assert!(
            choices_have_cycle_from(&egraph, merged, &self_referential),
            "the fixture's Neg must name its own class, or the test pins nothing"
        );

        let costs = SinAboveTheCycleSentinel;
        let sin_node = &egraph.nodes(merged)[sin_idx];
        assert!(
            costs.node_cost(sin_node, None) > usize::MAX / 4,
            "the premise: Sin is dearer than CYCLE_COST, so pricing the self-reference \
             at it would have won"
        );

        let dag = extract_dag(&egraph, merged, &costs);

        assert_eq!(
            dag.choices[merged.0 as usize],
            Some(sin_idx),
            "the self-referential Neg must not be picked"
        );
        assert_eq!(
            dag.total_cost,
            usize::MAX / 2,
            "the reported tree cost must be Sin's, the node actually returned"
        );
        assert_eq!(
            dag.dag_cost,
            usize::MAX / 2,
            "and so must the DAG cost — one op, reached once"
        );

        // The whole term still materializes.
        let (arena, arena_root) = dag.to_arena(&egraph);
        assert!(matches!(
            arena.node(arena_root),
            pixelflow_ir::arena::ExprNode::Unary(pixelflow_ir::OpKind::Sin, _)
        ));
    }

    /// A cyclic map names no term, so it has no price: pricing is a fold over
    /// a [`Selection`], and the map is refused before there is one. (Costing
    /// the DP's raw table is exactly the #1111 bug.)
    #[test]
    fn a_cyclic_choice_map_has_no_price() {
        let (egraph, merged, _sin_idx, neg_idx) = self_referential_pick_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        choices[merged.0 as usize] = Some(neg_idx);
        assert_eq!(
            Selection::of_map(&egraph, merged, &choices),
            Err(Refused::Cyclic(merged))
        );
    }
    // -----------------------------------------------------------------
    // The extractor's objective against the price of what it returns.
    // The hypothesis these pin (JP, 2026-09-08): "the DP's internal cost for
    // the choice map it selects does not equal the true `dag_cost` of the
    // term that map materializes, and the error grows with graph size."
    // It was true of the DFS post-order DP — chrome at a 50,000-class cap
    // claimed 281 for a term costing 4,564,003,324. See
    // docs/results/2026-09-08-cse-mispricing.md.
    // -----------------------------------------------------------------

    /// The claim a DP arm reports is the price of the term it returns, on
    /// that arm's own scale — across sizes and shapes, so a divergence that
    /// only appears on bigger graphs is caught here rather than inferred from
    /// a corpus run.
    #[test]
    fn the_dp_claim_prices_the_term_the_arm_returns() {
        let costs = CostModel::latency_prior();
        for nodes in [64usize, 256, 1024, 4096] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            for shape in [LatticeShape::POINT, LatticeShape::new([256, 256])] {
                let (tree, shared) = extract_dag_objectives(&egraph, root, &costs, shape);
                for (label, dag) in [("tree arm", &tree), ("shared arm", &shared)] {
                    let audit = dag.report.audit.expect("both arms run a DP");
                    assert_eq!(
                        audit.claimed,
                        audit.scale.of(dag.cost()),
                        "{label} at {nodes} nodes / {shape:?}: claimed {} on the {:?} scale \
                         but the term costs tree {} / dag {}",
                        audit.claimed,
                        audit.scale,
                        dag.total_cost,
                        dag.dag_cost,
                    );
                }
                // The sharing-aware arm's scale is the one the kernel pays,
                // so its claim IS `dag_cost` — the property #1116 bought and
                // the post-order traversal took back.
                let audit = shared.report.audit.expect("the shared arm runs a DP");
                assert_eq!(audit.scale, CostScale::Dag);
                assert_eq!(audit.claimed, shared.dag_cost);
                assert_eq!(audit.signed_error(shared.cost()), 0);
            }
        }
    }

    /// `extract_dag_scoped` chooses between the arms on **one** scale — the
    /// true `dag_cost` of each term — never on the arms' own DP tables, which
    /// are on different scales and can be saturated besides.
    #[test]
    fn the_arms_are_compared_on_the_price_not_on_their_claims() {
        let costs = CostModel::latency_prior();
        for nodes in [64usize, 256, 1024, 4096] {
            let (egraph, root) = saturated_sdf_egraph(nodes);
            for shape in [LatticeShape::POINT, LatticeShape::new([256, 256])] {
                let (tree, shared) = extract_dag_objectives(&egraph, root, &costs, shape);
                let scoped = extract_dag_scoped(&egraph, root, &costs, shape);
                assert_eq!(
                    scoped.dag_cost,
                    tree.dag_cost.min(shared.dag_cost),
                    "{nodes} nodes at {shape:?}: production returned {} where the cheaper arm \
                     costs tree {} / shared {}",
                    scoped.dag_cost,
                    tree.dag_cost,
                    shared.dag_cost
                );
                // Ties go to the tree arm, and only ties: a `Shared` verdict
                // means the sharing-aware term was *strictly* cheaper.
                match scoped.report.objective {
                    ExtractionObjective::Shared => {
                        assert!(shared.dag_cost < tree.dag_cost);
                        assert_eq!(scoped.choices, shared.choices);
                    }
                    ExtractionObjective::TreeCheaper => {
                        assert!(tree.dag_cost <= shared.dag_cost);
                        assert_eq!(scoped.choices, tree.choices);
                    }
                    other => panic!("{nodes} nodes: unexpected objective {other:?}"),
                }
            }
        }
    }

    /// The tree arm's DP objective **saturates** — its claim reaches
    /// `usize::MAX` on a graph this size, at which point every candidate ties
    /// at the ceiling and the settling keeps whichever the tie-break prefers.
    /// A degenerate objective, pinned so the fact is a test rather than a
    /// surprise.
    ///
    /// It cannot degrade production: the arms are chosen between by true
    /// `dag_cost` (the test above), so a saturated tree claim buys the tree
    /// arm nothing. The pin is what keeps that reasoning honest — if the
    /// comparison ever moved onto the arms' own tables, this says what it
    /// would be comparing.
    #[test]
    fn the_tree_arms_objective_saturates_on_a_real_sized_graph() {
        let costs = CostModel::latency_prior();
        let (egraph, root) = saturated_sdf_egraph(256);
        let tree = extract_dag_tree_arm(&egraph, root, &costs, LatticeShape::POINT);
        let audit = tree.report.audit.expect("the tree arm runs a DP");
        assert_eq!(audit.scale, CostScale::Tree);
        assert!(
            audit.claimed > 1e18 as usize,
            "the tree objective was expected at or near its ceiling, not {}",
            audit.claimed
        );
        assert_eq!(audit.claimed, tree.total_cost);
        // ...while the price of that same term is an ordinary number.
        assert!(
            tree.dag_cost < 1_000_000,
            "dag_cost should be a real number, not a ceiling: {}",
            tree.dag_cost
        );
    }

    /// A class two paths reach is priced **once** by the sharing-aware DP —
    /// not twice (the tree objective's error) and not zero times (a reach set
    /// that lost a member, which is what the post-order DP did to every class
    /// it left at the sentinel). Checked against a hand-computed sum so the
    /// test does not restate the implementation.
    #[test]
    fn the_shared_dp_prices_a_doubly_reached_class_exactly_once() {
        // sin(X) * sin(X): the `sin` class is reached by both children.
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let s = egraph.add(ENode::apply(OpKind::Sin, &[x]));
        let root = egraph.add(ENode::apply(OpKind::Mul, &[s, s]));
        egraph.rebuild();
        let costs = CostModel::latency_prior();
        let shape = LatticeShape::POINT;
        let dp = shared_dag_dp_pass(
            Frozen::new(&egraph, root).problem(),
            &mut Dp::production(&costs, shape),
            usize::MAX,
        )
        .outcome
        .expect("unbounded");

        let own = |c: EClassId| -> usize {
            weighted_own(
                &costs,
                &egraph.nodes(c)[dp.choices[c.0 as usize].unwrap()],
                1,
            )
        };
        assert_eq!(
            dp.root_cost,
            own(root) + own(s) + own(x),
            "the shared class must enter the sum once"
        );
        // The tree objective is the same sum with `sin(X)` charged twice.
        let tree = tree_dp_pass(
            Frozen::new(&egraph, root).problem(),
            &mut Dp::production(&costs, shape),
        );
        assert_eq!(tree.root_cost, own(root) + 2 * (own(s) + own(x)));
    }

    /// Over [`SHARED_DAG_PASS_BYTE_BUDGET`] the pass returns no map and the
    /// caller reports `TreeOnly` **on the `Tree` scale** — never a
    /// shared-priced answer under a tree label.
    #[test]
    fn the_budget_fallback_reports_the_tree_scale_it_actually_used() {
        const CHAIN: usize = 2_000;
        let (egraph, root) = add_chain(CHAIN);
        let costs = CostModel::latency_prior();
        let shape = LatticeShape::POINT;
        let full = shared_dag_dp_pass(
            Frozen::new(&egraph, root).problem(),
            &mut Dp::production(&costs, shape),
            usize::MAX,
        );
        let budget = full.stats.reach_bytes / 2;
        assert!(
            shared_dag_dp_pass(
                Frozen::new(&egraph, root).problem(),
                &mut Dp::production(&costs, shape),
                budget
            )
            .outcome
            .is_none()
        );

        let tree = extract_dag_tree_arm(&egraph, root, &costs, shape);
        assert_eq!(tree.report.objective, ExtractionObjective::TreeOnly);
        let audit = tree.report.audit.expect("the tree arm runs a DP");
        assert_eq!(audit.scale, CostScale::Tree);
        assert_eq!(audit.claimed, tree.total_cost);
    }

    /// The smallest e-graph whose DFS post-order asks a DP to price a class
    /// before any form of it exists — and, before extraction was settled in
    /// cost order, the smallest one on which the sharing-aware pass returned
    /// a term costing more than it claimed.
    ///
    /// ```text
    /// R = { Mul(Z, W) , Cos(C) }     Z = { Neg(P) }     W = Var(2)
    /// C = { Cos(D)    , Sin(P) }     P = { Sqrt(C) }    D = Var(1)
    /// ```
    ///
    /// `C` and `P` are mutually reachable, so a DFS entering `C` through
    /// `Sin(P)` opens `P` while `C` is still on the stack. The post-order DP
    /// settled `P` there, priced it at the cycle sentinel and gave it a reach
    /// set of `{P}` — and `C`, which only the chosen term's `P` reaches, went
    /// unpaid.
    fn blind_dfs_egraph() -> (EGraph, EClassId) {
        let mut egraph = EGraph::new();
        let d = egraph.add(ENode::Var(1));
        let w = egraph.add(ENode::Var(2));
        let c0 = egraph.add(ENode::apply(OpKind::Cos, &[d]));
        let p = egraph.add(ENode::apply(OpKind::Sqrt, &[c0]));
        let c1 = egraph.add(ENode::apply(OpKind::Sin, &[p]));
        let c = egraph.union(c0, c1);
        let z = egraph.add(ENode::apply(OpKind::Neg, &[p]));
        let r0 = egraph.add(ENode::apply(OpKind::Mul, &[z, w]));
        let r1 = egraph.add(ENode::apply(OpKind::Cos, &[c]));
        let root = egraph.union(r0, r1);
        egraph.rebuild();
        (egraph, root)
    }

    /// The claim/price identity on the graph a DFS order could not price —
    /// exactly where it used to fail, and the premise is asserted rather than
    /// remembered, so a fixture that stops exercising the case says so.
    #[test]
    fn the_claim_is_exact_on_the_graph_a_dfs_order_could_not_price() {
        let costs = CostModel::latency_prior();
        let (egraph, root) = blind_dfs_egraph();

        // Premise: some class of this graph has no form whose children the
        // DFS post-order settles before it.
        let mut settled = alloc::vec![false; egraph.num_classes()];
        let mut blind = 0usize;
        for &class in AndOrGraph::of(&egraph, root).classes() {
            let has_form = egraph.nodes(class).iter().any(|node| match node {
                ENode::Op { children, .. } => {
                    children.iter().all(|&c| settled[egraph.find(c).0 as usize])
                }
                _ => true,
            });
            if !has_form {
                blind += 1;
            }
            settled[class.0 as usize] = true;
        }
        assert!(blind > 0, "premise: the DFS order is blind on this fixture");

        for shape in [LatticeShape::POINT, LatticeShape::new([1920, 1080])] {
            let dag = extract_dag_scoped(&egraph, root, &costs, shape);
            let audit = dag.report.audit.expect("a DP ran");
            assert_eq!(
                audit.claimed,
                audit.scale.of(dag.cost()),
                "at {shape:?}: claimed {} against a term costing tree {} / dag {}",
                audit.claimed,
                dag.total_cost,
                dag.dag_cost
            );
            let term = Selection::of_map(&egraph, root, &dag.choices)
                .expect("the settled map names the term");
            assert_eq!(
                term.price(
                    &egraph,
                    &Pricing {
                        costs: &costs,
                        shape
                    }
                ),
                dag.cost(),
                "the reported pair must be the price of the returned term"
            );
        }
    }
}
