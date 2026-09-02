//! Extraction: materialise a concrete arena expression from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model and materialises it as an
//! [`pixelflow_ir::ExprArena`].

use super::cost::{CostFunction, CostModel};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use alloc::vec::Vec;

/// A witnessed selection: an e-graph, a root e-class, and a well-founded
/// choice function from every e-class reachable from `root` to the node
/// index selected for it.
///
/// "Well-founded" means: every reachable e-class has a recorded choice, and
/// the choice graph is acyclic (bottom-up realizable). Those two properties
/// are exactly what let a choice function be materialised at all — an
/// unvalidated `Vec<Option<usize>>` can loop forever when walked (a real
/// 2.7GB OOM, see `choices_to_arena`'s doc comment), and a call site that
/// forgets to repair or backfill it produces that bug silently.
///
/// `Extraction` makes the bug class unrepresentable: the only ways to
/// obtain a value of this type are [`Extraction::from_dp`] (wraps
/// [`repair_choices_well_founded`]) and [`Extraction::from_backfill`]
/// (wraps [`backfill_well_founded`]) — both establish well-foundedness as
/// part of construction, so a bare unvalidated vector can never cross into
/// [`choices_to_arena`] or the edge walker
/// ([`crate::nnue::EdgeTrace::from_extraction`]), which accept only
/// `&Extraction`. See docs/plans/2026-08-17-cost-model-domain.md §1
/// "Extraction (J2)".
pub struct Extraction<'g> {
    egraph: &'g EGraph,
    root: EClassId,
    choices: Vec<Option<usize>>,
}

impl<'g> Extraction<'g> {
    /// The DP path's smart constructor: makes `choices` well-founded via
    /// [`repair_choices_well_founded`] (resolving any mutual cycles the
    /// bottom-up DP recorded — `CYCLE_COST` penalizes only
    /// self-references, so two merged classes can each cheapest-pick a
    /// node through the other), then seals the result.
    pub(crate) fn from_dp(
        egraph: &'g EGraph,
        root: EClassId,
        mut choices: Vec<Option<usize>>,
    ) -> Self {
        let root = egraph.find(root);
        repair_choices_well_founded(egraph, root, &mut choices);
        Self {
            egraph,
            root,
            choices,
        }
    }

    /// The hand-built path's smart constructor: fills any e-class
    /// reachable from `root` still lacking a choice via
    /// [`backfill_well_founded`], then seals the result. Tests and research
    /// harnesses that pin specific node choices come in here; production
    /// extraction comes in through [`Extraction::from_dp`].
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

    /// Read-only view of the raw choice vector, indexed by canonical
    /// e-class id. Still only reachable through a sealed `Extraction`.
    pub(crate) fn choices(&self) -> &[Option<usize>] {
        &self.choices
    }

    /// The choice vector [`choices_to_arena`] will actually materialise:
    /// every `Shl`/`Shr` count child re-pinned to a `Const` representative of
    /// its class ([`pin_shift_counts`]), because the emitter's shift lowering
    /// requires an immediate and a count class can legitimately hold
    /// arithmetic that is value-equal to a constant without being one
    /// (e.g. `Y - Y` merged with `Const(0)`).
    ///
    /// Any consumer that walks an `Extraction` — not just
    /// [`choices_to_arena`] itself — must walk this view, not [`Self::choices`]
    /// directly: describing the raw (possibly non-`Const`) choice for a
    /// count class would describe a DAG that is not the one actually
    /// compiled. `ChoicesCostDag` in `crate::nnue::factored` takes this view
    /// rather than re-deriving it (one definition, imported, not restated).
    pub(crate) fn pinned_choices(&self) -> Vec<Option<usize>> {
        pin_shift_counts(self.egraph, self.root, &self.choices)
    }

    /// Variance histogram (fraction const / frame-uniform / scanline-uniform
    /// / pixel-varying) of the CHOSEN nodes, not the class-wide meet
    /// [`super::DepsAnalysis`] would compute over the whole e-graph.
    ///
    /// Materialises the choice function once via [`choices_to_arena`] and
    /// classifies that arena — P1(c) of
    /// docs/plans/2026-08-17-cost-model-domain.md: once a rewrite merges a
    /// pixel-varying node into a class alongside a constant one, the
    /// class-wide meet reports CONST regardless of which node the
    /// extraction actually chose, so only the materialised DAG describes
    /// what was picked.
    #[must_use]
    pub fn chosen_variance(&self) -> [f32; crate::nnue::factored::SCALAR_FEATURE_COUNT] {
        let (arena, _root) = choices_to_arena(self);
        crate::nnue::factored::variance_histogram(&arena)
    }

    /// Unwrap into the raw choice vector.
    ///
    /// Kept for legacy raw-vector consumers (`ExtractionPolicy::choices`,
    /// `compute_ref_counts`, `build_extracted_dag_from_choices`) that
    /// predate this type; new code should consume `&Extraction` instead.
    pub(crate) fn into_choices(self) -> Vec<Option<usize>> {
        self.choices
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
    /// ([`choices_to_arena`], which is what a [`Reranker`]'s score is
    /// computed from), so a `canonical` unreachable from root is invisible
    /// to it regardless of what this function decides; accept or reject,
    /// the candidate's score is bit-identical to `current_cost` and it
    /// never wins the refinement loop's strict `<` comparison.
    pub(crate) fn try_swap(&self, class: EClassId, node_idx: usize) -> Option<Self> {
        let canonical = self.egraph.find(class);
        let mut choices = self.choices.clone();
        choices[canonical.0 as usize] = Some(node_idx);

        let new_children: &[EClassId] = match self.egraph.nodes(canonical).get(node_idx) {
            Some(ENode::Op { children, .. }) => children,
            _ => &[],
        };

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
// NNUE tied the static table on schedule-free kernels
// (docs/paper/2026-08-egraph-nnue-parity.md) — but JP's ruling on the
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
    /// same `(arena, root)` pair [`choices_to_arena`] would return for
    /// `extraction`, computed once per candidate by the search loop.
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
pub struct IncrementalExtractor<'a, R: Reranker> {
    reranker: &'a R,
    top_k: usize,
}

impl<'a, R: Reranker> IncrementalExtractor<'a, R> {
    /// `top_k` bounds how many alternative nodes per e-class are evaluated
    /// during refinement passes.
    pub fn new(reranker: &'a R, top_k: usize) -> Self {
        Self { reranker, top_k }
    }

    /// Run the extraction refinement loop and return `(score, extraction)`.
    ///
    /// Call [`choices_to_arena`] on the returned [`Extraction`] to
    /// materialise the extracted DAG.
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
        let (arena, _root) = choices_to_arena(extraction);
        self.reranker.score(extraction, &arena)
    }

    /// Walk the current best tree and collect active (reachable) e-class IDs.
    fn get_active_classes(&self, extraction: &Extraction<'_>) -> Vec<EClassId> {
        let egraph = extraction.egraph();
        let root = extraction.root();
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
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or(0);
        if let Some(ENode::Op { children, .. }) = egraph.nodes(c).get(node_idx) {
            for &child in children {
                stack.push(child);
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

    // Seals `best_node` into an `Extraction`, repairing any mutual cycles
    // the DP recorded before the tree is built.
    let extraction = Extraction::from_dp(egraph, root, best_node);
    let (arena, root_id) = choices_to_arena(&extraction);
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
/// Scoped to classes reachable from `root` via the ORIGINAL (unpinned)
/// `choices` — the same traversal [`choices_to_arena`] performs once pinning
/// has settled. `choices` can
/// (and, on a graph whose choices were built up by several backfill
/// passes, routinely does) hold `Some` entries for classes no longer
/// reachable from `root` under the CURRENT choice function — a backfill
/// only ever adds entries, never retracts a stale one from an earlier
/// candidate. Walking `0..egraph.num_classes()`
/// unconditionally, as this used to, re-derives and re-pins every one of
/// those stale entries even though nothing downstream ever reads them
/// (`choices_to_arena` only ever visits classes reachable
/// from `root`) — pure wasted work on a saturated e-graph's full class
/// count, not the reachable subtree's.
///
/// Using unpinned `choices` (rather than the pins already decided so far in
/// this same walk) to decide which children to descend into is deliberately
/// a superset of the classes [`choices_to_arena`] will actually visit once
/// pinning is final: pinning a count class to a `Const` can only ever REMOVE
/// reachability (a `Const` has no children to recurse into), never add it,
/// so this walk's reachable set is never missing a class the final pinned
/// tree needs. The decision written into `pinned[ci]` for any one count
/// class does not depend on visitation order either — it is "keep the
/// existing choice if already `Const`, else the class's first `Const`
/// node", the same answer no matter which of possibly several referencing
/// `Shl`/`Shr` nodes is processed first — so interleaving the decision with
/// the traversal (instead of a full resolve-then-walk pass) cannot change
/// the returned vector's values, only which unreachable entries are left
/// untouched (they pass through from `choices` unread either way).
///
/// # Panics
///
/// Panics if a shift-count class holds no `Const` at all. That cannot arise
/// from a well-formed arena (the count entered as a literal) and would panic
/// in the emitter regardless — failing here names the real cause.
fn pin_shift_counts(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> alloc::vec::Vec<Option<usize>> {
    let num_classes = choices.len();
    let mut pinned = choices.to_vec();
    let mut visited: alloc::vec::Vec<bool> = alloc::vec![false; num_classes];
    let mut stack: Vec<EClassId> = alloc::vec![egraph.find(root)];

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.0 as usize;
        if idx >= num_classes || visited[idx] {
            continue;
        }
        visited[idx] = true;

        // Deliberately reads the ORIGINAL (unpinned) choice, not `pinned`,
        // to decide what this class's traversal children are — see the
        // "superset" reasoning in the doc comment above.
        let Some(node_idx) = choices.get(idx).and_then(|o| *o) else {
            continue;
        };
        let Some(ENode::Op { op, children }) = egraph.nodes(canonical).get(node_idx) else {
            continue;
        };
        if matches!(
            op.kind(),
            pixelflow_ir::OpKind::Shl | pixelflow_ir::OpKind::Shr
        ) && let Some(&count) = children.get(1)
        {
            let count_class = egraph.find(count);
            let ci = count_class.0 as usize;
            let count_nodes = egraph.nodes(count_class);
            let already_const = pinned
                .get(ci)
                .and_then(|o| *o)
                .is_some_and(|chosen| matches!(count_nodes.get(chosen), Some(ENode::Const(_))));
            if !already_const {
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
        }

        for &child in children {
            stack.push(child);
        }
    }

    pinned
}

pub fn choices_to_arena(
    extraction: &Extraction<'_>,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    use pixelflow_ir::{ExprArena, ExprId};

    let egraph = extraction.egraph();
    let root = extraction.root();

    // Shifts must reach codegen with a constant count — see
    // `Extraction::pinned_choices` / `pin_shift_counts`.
    let pinned = extraction.pinned_choices();
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
    // Swap-refinement search (Reranker seam)
    // ========================================================================

    /// A trivial, test-only [`Reranker`]: the sum of `costs.cost(op)` over
    /// every node the candidate's materialised arena contains — the same
    /// additive latency-prior table [`extract_dag`] minimizes, just summed
    /// over the arena instead of folded bottom-up through the e-graph. No
    /// sharing discount (each arena node is already deduped by
    /// `choices_to_arena`, so this is a DAG cost, not a tree cost) — fine
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
        let x_plus_x = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, x],
        });
        let x_times_2 = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, two],
        });
        let merged = egraph.union(x_plus_x, x_times_2);
        egraph.rebuild();

        let costs = CostModel::latency_prior();
        let dag = extract_dag(&egraph, merged, &costs);

        let reranker = TableReranker { costs: &costs };
        let extractor = IncrementalExtractor::new(&reranker, 8);
        let (search_cost, extraction) = extractor.extract_choices_only(&egraph, merged);
        let (arena, root) = choices_to_arena(&extraction);

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
        let (arena, root) = choices_to_arena(&extraction);
        assert!(arena.len() >= 1);
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
        // accusation instead. `Extraction`'s own constructors now refuse a
        // cyclic choice set before this point is ever reached (see
        // `extraction_constructors_refuse_a_cyclic_choice_set`) — this test
        // exercises `choices_to_arena`'s own belt-and-suspenders check by
        // constructing the `Extraction` directly (private-field literal,
        // valid from within this module), bypassing the smart constructors
        // on purpose.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));
        let extraction = Extraction {
            egraph: &egraph,
            root: egraph.find(merged),
            choices,
        };
        let _ = choices_to_arena(&extraction);
    }

    #[test]
    #[should_panic(expected = "cyclic")]
    fn extraction_constructors_refuse_a_cyclic_choice_set() {
        // The type-level guarantee J2 adds: a cyclic choice vector can no
        // longer become an `Extraction` at all, so the 2.7GB-OOM class of
        // bug can't reach `choices_to_arena` in the first place.
        let (egraph, merged, n1) = cyclic_capable_egraph();
        let mut choices: Vec<Option<usize>> = alloc::vec![None; egraph.num_classes()];
        let m = egraph.find(merged).0 as usize;
        let i = egraph.find(n1).0 as usize;
        choices[m] = Some(neg_index(&egraph, merged).expect("merged class holds Neg(n1)"));
        choices[i] = Some(neg_index(&egraph, n1).expect("n1 holds Neg(x)"));
        let _ = Extraction::from_backfill(&egraph, merged, choices);
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
        let extraction = Extraction {
            egraph: &egraph,
            root: egraph.find(merged),
            choices,
        };
        let (arena, root) = choices_to_arena(&extraction);
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

    // =========================================================================
    // Train/deploy feature-path equivalence (2026-08 round-0 skew guard)
    // =========================================================================

    #[test]
    fn edge_trace_records_a_reload_for_a_shared_subexpression() {
        use crate::nnue::EdgeTrace;

        // sqrt(X) * sqrt(X): tree has 2x sqrt edges, DAG has 1x sqrt + 1x reload
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

        let extraction = Extraction::from_backfill(&egraph, product, choices);
        let trace = EdgeTrace::from_extraction(&extraction);

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

    /// The arena walk (`EdgeTrace::from_arena_dag`) and the e-graph walk
    /// (`EdgeTrace::from_extraction`) are thin adapters over one walker, and
    /// this test pins that: for the same DAG — shared subexpressions, shared
    /// leaves — the two paths must record the identical edge stream. If a
    /// future change gives either path its own edge policy, this fails.
    #[test]
    fn arena_and_extraction_walks_record_the_same_edge_stream() {
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

        let choices: Vec<Option<usize>> = alloc::vec![None; eg.num_classes()];
        let extraction = Extraction::from_backfill(&eg, eroot, choices);

        let from_arena = EdgeTrace::from_arena_dag(&arena, root);
        let from_egraph = EdgeTrace::from_extraction(&extraction);

        assert_eq!(from_arena.node_count(), 11, "11 distinct nodes");
        assert_eq!(
            from_arena, from_egraph,
            "the arena walk and the e-graph walk must record the identical edge stream"
        );

        // ---------------------------------------------------------------
        // Shift-count pinning (review thread on PR #1019): a Shl/Shr count
        // e-class can legitimately hold both a Const and a value-equal
        // varying-shaped alternative. `choices_to_arena` always pins that
        // child to the Const representative (`pin_shift_counts` — the
        // emitter's shift lowering requires an immediate), so if the
        // extraction chose the varying node, the e-graph walk must describe
        // the PINNED arena, not the raw choice — otherwise it records nodes
        // `choices_to_arena` never emits.
        // ---------------------------------------------------------------
        struct ShlOp;
        impl crate::egraph::ops::Op for ShlOp {
            fn kind(&self) -> OpKind {
                OpKind::Shl
            }
        }

        // The arena `choices_to_arena` will actually materialise: pinning
        // always wins, so the compiled form is `Shl(X, Const(0))` no matter
        // which node the count class's extraction chose.
        let mut arena3 = ExprArena::new();
        let x3 = arena3.push_var(0);
        let zero3 = arena3.push_const(0.0);
        let shl3 = arena3.push_binary(OpKind::Shl, x3, zero3);
        let from_arena3 = EdgeTrace::from_arena_dag(&arena3, shl3);

        let mut eg3 = EGraph::new();
        let ex3 = eg3.add(ENode::Var(0));
        let ey3 = eg3.add(ENode::Var(1));
        let esub3 = eg3.add(ENode::Op {
            op: &ops::Sub,
            children: alloc::vec![ey3, ey3],
        });
        let econst3 = eg3.add(ENode::constant(0.0));
        let count3 = eg3.union(esub3, econst3); // Sub(Y, Y) = 0, same class as Const(0)
        eg3.rebuild();
        let eshl3 = eg3.add(ENode::Op {
            op: &ShlOp,
            children: alloc::vec![ex3, count3],
        });

        // Choose the varying Sub(Y, Y) node for the count class, not the
        // Const — exactly the scenario `pin_shift_counts` exists to correct
        // at emission time.
        let canonical_count3 = eg3.find(count3);
        let sub_idx3 = eg3
            .nodes(canonical_count3)
            .iter()
            .position(|n| matches!(n, ENode::Op { .. }))
            .expect("merged count class holds the Sub node");
        let mut choices3: Vec<Option<usize>> = alloc::vec![None; eg3.num_classes()];
        choices3[eg3.find(eshl3).0 as usize] = Some(0);
        choices3[canonical_count3.0 as usize] = Some(sub_idx3);
        choices3[eg3.find(ex3).0 as usize] = Some(0);
        choices3[eg3.find(ey3).0 as usize] = Some(0);
        let extraction3 = Extraction::from_backfill(&eg3, eshl3, choices3);

        let from_egraph3 = EdgeTrace::from_extraction(&extraction3);

        assert_eq!(
            from_arena3, from_egraph3,
            "the e-graph walk must not walk into Sub(Y, Y) once the count is pinned to \
             Const(0), or its stream will disagree with the arena choices_to_arena emits"
        );
    }

    // =========================================================================
    // Extraction::chosen_variance (2026-09-01: denotation kept, accumulator
    // shape deleted — see the module doc comment above
    // `crate::nnue::factored::variance_histogram`)
    // =========================================================================

    /// A known const/frame/scanline/pixel mix, built once as an arena and
    /// once as the equivalent e-graph, must classify identically through
    /// both entry points — mirroring
    /// `arena_and_extraction_walks_record_the_same_edge_stream` above.
    #[test]
    fn arena_and_extraction_classify_a_known_variance_mix_identically() {
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

        use crate::egraph::ops;
        let mut eg = EGraph::new();
        let ec = eg.add(ENode::constant(2.0));
        let ew = eg.add(ENode::Var(3));
        let efs = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ec, ew],
        });
        let ey = eg.add(ENode::Var(1));
        let ex = eg.add(ENode::Var(0));
        let exy = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![ey, ex],
        });
        let eroot = eg.add(ENode::Op {
            op: &ops::Add,
            children: alloc::vec![efs, exy],
        });

        let choices: Vec<Option<usize>> = alloc::vec![None; eg.num_classes()];
        let extraction = Extraction::from_backfill(&eg, eroot, choices);

        let from_arena = variance_histogram(&arena);
        let from_extraction = extraction.chosen_variance();

        assert_eq!(
            from_arena, from_extraction,
            "arena and extraction must classify the same DAG identically"
        );
        assert_eq!(
            from_arena,
            [1.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 3.0 / 7.0],
            "known const/frame/scanline/pixel mix: {from_arena:?}"
        );
        let _ = root; // arena root; classification is over every node.
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

        let extraction = Extraction::from_backfill(&egraph, add, choices);
        let (arena, root_id) = choices_to_arena(&extraction);

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

        let extraction = Extraction::from_backfill(&egraph, mul, choices);
        let (arena, root_id) = choices_to_arena(&extraction);

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

        let extraction = Extraction::from_backfill(&egraph, add, choices);
        let (arena, root_id) = choices_to_arena(&extraction);
        let (extracted_arena, extracted_root, _cost) = extract(&egraph, add, &CostModel::default());
        assert_eq!(arena.len(), extracted_arena.len());
        assert_eq!(root_id, extracted_root);
    }
}
