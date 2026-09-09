//! The extraction seam: which term the optimizer takes out of a saturated
//! e-graph.
//!
//! An extractor is a function `(EGraph, root, costs, shape) -> ExtractedDAG`
//! minimizing [`ExtractedDAG::dag_cost`] — each distinct chosen e-class
//! priced once, which is what the emitted kernel pays. The value it returns
//! is a **choice map**: one node index per reachable e-class. Two extractors
//! are equal iff their choice maps are, which is why the identity claim for
//! [`Greedy`] is a test rather than a hope.
//!
//! Denotation and the boundary decisions:
//! `docs/plans/2026-09-08-extractor-trait.md`.
//!
//! Three implementations ship:
//!
//! - [`Greedy`] — production. The tree DP and the sharing-aware DP, each
//!   repaired and costed, cheaper wins, ties to tree.
//! - [`Beam`] — [`Greedy`]'s sharing-aware pass with `k` seats per e-class
//!   instead of one. The first non-learned test of whether the class-cap
//!   regression (`docs/results/2026-09-08-class-cap-sweep.md`) is a search
//!   **width** failure.
//! - [`Reranked`] — the swap-refinement search over a
//!   [`Reranker`](super::Reranker). It was already an extractor, inlined
//!   into `Optimizer::run` as the `Some` arm of a two-arm match; naming it
//!   leaves one dispatch point where there were two.

use alloc::boxed::Box;
use alloc::vec::Vec;
use pixelflow_ir::{LatticeShape, Variance};

use super::cost::CostFunction;
use super::extract::{
    self, CostedChoices, ExtractedDAG, ExtractionObjective, ExtractionReport, IncrementalExtractor,
    Reranker, SHARED_DAG_PASS_BYTE_BUDGET, SharedPassSettled,
};
use super::graph::EGraph;
use super::node::{EClassId, ENode};

/// Take a term out of a saturated e-graph.
///
/// Generic in the cost function rather than taking `&dyn CostFunction`:
/// [`CostFunction::node_cost`] is called once per *(class, node)* in each
/// pass — millions of times on a 100k-class graph — so a trait object there
/// would pay a virtual call per node, which is the shape CLAUDE.md's fold
/// rule is against. `Optimizer` holds a `Box<dyn Extractor<CostModel>>`
/// instead: the extractor is chosen once and dispatched once per extraction,
/// and inside the impl the cost model is concrete.
pub trait Extractor<C: CostFunction + ?Sized> {
    /// The chosen term, with the schedule, sharing and cost that describe
    /// it, and the [`ExtractionObjective`] it was actually produced under.
    fn extract(
        &self,
        egraph: &EGraph,
        root: EClassId,
        costs: &C,
        shape: LatticeShape,
    ) -> ExtractedDAG;
}

/// Production's extractor: the tree DP and the sharing-aware DP, cheaper by
/// true DAG cost wins, ties to tree.
///
/// This is a name for [`extract::extract_dag_scoped`], not a second copy of
/// it — every existing caller still reaches the free function directly, and
/// "`Greedy` is byte-identical to production" is therefore structural.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Greedy;

impl<C: CostFunction> Extractor<C> for Greedy {
    fn extract(
        &self,
        egraph: &EGraph,
        root: EClassId,
        costs: &C,
        shape: LatticeShape,
    ) -> ExtractedDAG {
        extract::extract_dag_scoped(egraph, root, costs, shape)
    }
}

/// The swap-refinement search over a [`Reranker`], as an extractor.
///
/// Reports [`ExtractionObjective::External`]: the reranker minimizes its own
/// `f64` score, not [`ExtractedDAG::dag_cost`], so the cost this returns is
/// the honest price of a term chosen under a different objective.
pub struct Reranked {
    reranker: Box<dyn Reranker>,
    top_k: usize,
}

/// How many alternatives per e-class the reranking search evaluates.
/// Matches the value the swap-refinement search shipped with.
pub const RERANK_TOP_K: usize = 4;

impl Reranked {
    /// `top_k` bounds how many alternative nodes per e-class the refinement
    /// loop evaluates.
    ///
    /// # Panics
    ///
    /// If `top_k` is zero — a refinement loop that may consider no
    /// alternative is not a search.
    #[must_use]
    pub fn new(reranker: Box<dyn Reranker>, top_k: usize) -> Self {
        assert!(top_k > 0, "Reranked::new: top_k must be at least 1");
        Self { reranker, top_k }
    }
}

impl<C: CostFunction> Extractor<C> for Reranked {
    fn extract(
        &self,
        egraph: &EGraph,
        root: EClassId,
        costs: &C,
        shape: LatticeShape,
    ) -> ExtractedDAG {
        let choices = IncrementalExtractor::new(self.reranker.as_ref(), self.top_k)
            .extract_choices_only(egraph, root)
            .1
            .into_choices();
        let cost = extract::cost_of_choices(egraph, root, &choices, costs, shape);
        extract::assemble(
            egraph,
            root,
            CostedChoices { choices, cost },
            ExtractionReport::external(),
        )
    }
}

// ============================================================================
// Beam
// ============================================================================

/// [`Greedy`]'s sharing-aware pass with `k` seats per e-class.
///
/// The sharing-aware DP walks classes in post-order and keeps **one** choice
/// per class: the node minimizing `own + Σ own over the union of its
/// children's sub-DAGs`. The union is what makes it exact — a subterm
/// entering from two children is counted once — and the single seat is what
/// makes it greedy: a class settles before any parent that might have
/// preferred a different settlement.
///
/// `Beam` keeps `k`. A beam state is a **partial choice map** — the sorted
/// `(class, node, own)` triples of exactly the classes in its sub-DAG — and
/// its cost is the sum of the own column, i.e. exactly the DAG cost of that
/// sub-DAG. Nothing is estimated.
///
/// Two states merge only if they **agree** on every class they share; a
/// disagreement makes the pair inadmissible and it is dropped, because a
/// state must name one term. That a state always exists is structural rather
/// than lucky: seat 0 of every class is the **anchor**, the state naming the
/// DP's own choices, and the DP assigns one node per class globally, so
/// merging anchors across siblings can never conflict. The same fact makes
/// `Beam::width(1)` reproduce the DP exactly, and hence
/// [`Greedy`] — [`Self::extract`] folds against the same tree arm with the
/// same tie-break.
///
/// Seat 0 costing a seat is deliberate and is the width's price: `k` seats
/// are the anchor plus the `k - 1` cheapest other states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Beam {
    width: usize,
}

impl Beam {
    /// `k` seats per e-class. `k = 1` is [`Greedy`].
    ///
    /// # Panics
    ///
    /// If `k` is zero.
    #[must_use]
    pub fn width(k: usize) -> Self {
        assert!(k > 0, "Beam::width: k must be at least 1");
        Self { width: k }
    }

    /// The configured width.
    #[must_use]
    pub fn k(self) -> usize {
        self.width
    }
}

impl<C: CostFunction> Extractor<C> for Beam {
    fn extract(
        &self,
        egraph: &EGraph,
        root: EClassId,
        costs: &C,
        shape: LatticeShape,
    ) -> ExtractedDAG {
        let tree = extract::repaired_and_costed(
            egraph,
            root,
            extract::tree_dp_pass(egraph, root, costs, shape),
            costs,
            shape,
        );
        let pass =
            extract::shared_dag_dp_pass(egraph, root, costs, shape, SHARED_DAG_PASS_BYTE_BUDGET);
        let stats = Some(pass.stats);
        let Some(settled) = pass.settled else {
            // Identical to `Greedy`: the sharing-aware pass never produced a
            // table, so there is nothing for the beam to widen.
            return extract::assemble(
                egraph,
                root,
                tree,
                ExtractionReport {
                    objective: ExtractionObjective::TreeOnly,
                    shared_pass: stats,
                },
            );
        };

        let choices = beam_pass(egraph, root, costs, shape, &settled, self.width);
        let shared = extract::repaired_and_costed(egraph, root, choices, costs, shape);
        if shared.cost.dag < tree.cost.dag {
            return extract::assemble(
                egraph,
                root,
                shared,
                ExtractionReport {
                    objective: ExtractionObjective::Shared,
                    shared_pass: stats,
                },
            );
        }
        extract::assemble(
            egraph,
            root,
            tree,
            ExtractionReport {
                objective: ExtractionObjective::TreeCheaper,
                shared_pass: stats,
            },
        )
    }
}

/// One class of a beam state's partial choice map.
///
/// `own` is the *weighted* own cost of `node` under the variance its own
/// sub-DAG induces — the number a parent pays once when this class first
/// enters its union.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Member {
    /// Compact id: position in the post-order.
    class: u32,
    node: u32,
    own: u64,
}

/// A partial choice map over exactly one class's chosen sub-DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    /// Sorted by [`Member::class`], distinct, self included and last.
    members: Vec<Member>,
    /// The DP's selection cost: the sum of the own column, or
    /// [`extract::CYCLE_COST`] for a cycle-priced class.
    cost: usize,
    var: Variance,
    node: u32,
}

impl State {
    /// The canonical order the beam prunes and ties break by. Never
    /// insertion order: that would make the chosen term depend on
    /// `EGraph::nodes`'s layout, and a kernel would build differently on two
    /// hosts.
    fn key(&self) -> (usize, u32, &[Member]) {
        (self.cost, self.node, self.members.as_slice())
    }

    fn bytes(&self) -> usize {
        self.members.len() * core::mem::size_of::<Member>()
    }
}

/// A partial merge of some prefix of one node's children.
#[derive(Clone, Debug)]
struct Acc {
    members: Vec<Member>,
    below: u64,
    var: Variance,
}

/// Merge two partial choice maps.
///
/// `None` when they disagree — assign different nodes to the same class.
/// Returns the merged map and the sum of its own column.
///
/// # Panics
///
/// If two maps agree on a class's node but disagree on its own cost. That
/// cannot happen — a member set is downward closed, so agreeing on a class
/// means agreeing on its whole sub-DAG and therefore on its variance — and
/// if it ever does, the beam's cost column has stopped meaning DAG cost.
fn merge(a: &[Member], b: &[Member]) -> Option<(Vec<Member>, u64)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut sum: u64 = 0;
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        if x.class < y.class {
            sum = sum.saturating_add(x.own);
            out.push(x);
            i += 1;
        } else if y.class < x.class {
            sum = sum.saturating_add(y.own);
            out.push(y);
            j += 1;
        } else {
            if x.node != y.node {
                return None;
            }
            assert_eq!(
                x.own, y.own,
                "beam merge: e-class {} chosen as node {} by both maps but priced {} and {} — \
                 a member set is downward closed, so equal choices must carry equal cost",
                x.class, x.node, x.own, y.own
            );
            sum = sum.saturating_add(x.own);
            out.push(x);
            i += 1;
            j += 1;
        }
    }
    for &m in &a[i..] {
        sum = sum.saturating_add(m.own);
        out.push(m);
    }
    for &m in &b[j..] {
        sum = sum.saturating_add(m.own);
        out.push(m);
    }
    Some((out, sum))
}

/// The beam's own byte budget, per seat.
///
/// A beam state carries the same members the DP's reach sets do, plus the
/// node index and the own cost, so `width` seats cost `width` times
/// [`SHARED_DAG_PASS_BYTE_BUDGET`]'s worth of members at four times the
/// bytes each. Crossing it **panics** rather than quietly returning the DP's
/// term labelled as the beam's: `Beam` is never production, so a width too
/// large for an input is a fact about the measurement, and a measurement
/// that silently ran a different extractor is worse than one that stopped.
fn beam_budget(width: usize) -> usize {
    SHARED_DAG_PASS_BYTE_BUDGET.saturating_mul(width)
}

/// The beam's choice map: the DP's table, overwritten by the cheapest
/// complete state at the root.
///
/// Classes the chosen sub-DAG does not contain keep the DP's choice — they
/// are unreachable through the returned map and so never priced, but a map
/// with holes in it is not something the repair and cost passes should have
/// to reason about.
fn beam_pass<C: CostFunction>(
    egraph: &EGraph,
    root: EClassId,
    costs: &C,
    shape: LatticeShape,
    settled: &SharedPassSettled,
    width: usize,
) -> Vec<Option<usize>> {
    let SharedPassSettled {
        choices: dp_choices,
        order,
        compact,
        ..
    } = settled;
    let budget = beam_budget(width);
    let live = order.len();

    // Filled in `order`, exactly as the DP fills its own: a class not yet
    // reached is `CONST`, which is what makes width 1 reproduce the DP's
    // variance-weighted own costs rather than a post-hoc version of them.
    let mut beam_var: Vec<Variance> = alloc::vec![Variance::CONST; egraph.num_classes()];
    let mut beams: Vec<Vec<State>> = Vec::with_capacity(live);
    let mut held_bytes: usize = 0;

    for (me, canonical) in order.iter().copied().enumerate() {
        let nodes = egraph.nodes(canonical);
        let anchor = anchor_state(egraph, canonical, me, settled, &beams, compact, nodes);
        let mut free: Vec<State> = Vec::new();

        if width > 1 {
            for (idx, node) in nodes.iter().enumerate() {
                expand_node(
                    ExpandCtx {
                        egraph,
                        costs,
                        shape,
                        compact,
                        beams: &beams,
                        beam_var: &beam_var,
                        width,
                    },
                    canonical,
                    me,
                    idx as u32,
                    node,
                    &mut free,
                );
            }
            free.sort_by(|a, b| a.key().cmp(&b.key()));
            free.dedup();
            free.retain(|s| s != &anchor);
            free.truncate(width - 1);
        }

        beam_var[canonical.0 as usize] = anchor.var;
        held_bytes = held_bytes
            .saturating_add(anchor.bytes())
            .saturating_add(free.iter().map(State::bytes).sum::<usize>());
        assert!(
            held_bytes <= budget,
            "Beam::width({width}): beam states hold {held_bytes} bytes, over the {budget}-byte \
             budget, at live class {me} of {live}. Narrow the width or the class cap — this \
             extractor will not quietly return the greedy term under a beam's name."
        );

        let mut seats = Vec::with_capacity(1 + free.len());
        seats.push(anchor);
        seats.append(&mut free);
        beams.push(seats);
    }

    let root_me = compact[egraph.find(root).0 as usize] as usize;
    assert!(
        root_me < live,
        "beam_pass: the root is not in its own post-order"
    );
    let best = beams[root_me]
        .iter()
        .min_by(|a, b| a.key().cmp(&b.key()))
        .expect("beam_pass: every class holds at least its anchor");

    let mut choices = dp_choices.clone();
    for m in &best.members {
        choices[order[m.class as usize].0 as usize] = Some(m.node as usize);
    }
    choices
}

/// The state naming the DP's own choices at `me`.
///
/// Built from the settled table and the anchors already in `beams`, not
/// recomputed: the DP's own costs are authoritative, and rebuilding them
/// would be a second implementation of the same recurrence. The equality
/// assertion is the pin that says the two representations still agree.
fn anchor_state(
    egraph: &EGraph,
    canonical: EClassId,
    me: usize,
    settled: &SharedPassSettled,
    beams: &[Vec<State>],
    compact: &[u32],
    nodes: &[ENode],
) -> State {
    let idx = settled.choices[canonical.0 as usize]
        .expect("anchor_state: the sharing-aware DP chose a node for every live class");
    let cost = settled.cost[canonical.0 as usize]
        .expect("anchor_state: the sharing-aware DP costed every live class");
    let own = settled.own[me] as u64;
    let me_member = Member {
        class: me as u32,
        node: idx as u32,
        own,
    };
    let var = settled.var[canonical.0 as usize];

    // A cycle-priced class reaches only itself, as it does in the DP.
    if cost == extract::CYCLE_COST {
        return State {
            members: alloc::vec![me_member],
            cost,
            var,
            node: idx as u32,
        };
    }

    let mut members: Vec<Member> = Vec::new();
    let mut below: u64 = 0;
    if let ENode::Op { children, .. } = &nodes[idx] {
        for &child in children.iter() {
            let ci = compact[egraph.find(child).0 as usize] as usize;
            let (merged, sum) = merge(&members, &beams[ci][0].members)
                .expect("anchor_state: anchors name one term per class and cannot conflict");
            members = merged;
            below = sum;
        }
    }
    members.push(me_member);
    let total = usize::try_from(below.saturating_add(own)).unwrap_or(usize::MAX);
    assert_eq!(
        total, cost,
        "anchor_state: e-class {} reassembles to {total} but the sharing-aware DP priced it \
         {cost} — the beam's member sets and the DP's reach sets have drifted",
        canonical.0
    );
    State {
        members,
        cost,
        var,
        node: idx as u32,
    }
}

/// Everything [`expand_node`] reads. A struct because the list is six deep
/// and none of it is this function's own state.
struct ExpandCtx<'a, C: CostFunction> {
    egraph: &'a EGraph,
    costs: &'a C,
    shape: LatticeShape,
    compact: &'a [u32],
    beams: &'a [Vec<State>],
    beam_var: &'a [Variance],
    width: usize,
}

/// Every state `node` admits at class `canonical`, appended to `out`.
fn expand_node<C: CostFunction>(
    ctx: ExpandCtx<'_, C>,
    canonical: EClassId,
    me: usize,
    idx: u32,
    node: &ENode,
    out: &mut Vec<State>,
) {
    let ExpandCtx {
        egraph,
        costs,
        shape,
        compact,
        beams,
        beam_var,
        width,
    } = ctx;

    // A leaf, a self-reference, or a child still on the DFS stack: one state,
    // itself alone, priced exactly as the DP prices it — the variance read
    // out of the in-flight `beam_var`, not a settled table.
    let alone = |cost_is_cycle: bool| {
        let var = extract::node_variance(egraph, node, beam_var, canonical);
        let own = extract::weighted_own(costs, node, shape.evals(var)) as u64;
        State {
            members: alloc::vec![Member {
                class: me as u32,
                node: idx,
                own
            }],
            cost: if cost_is_cycle {
                extract::CYCLE_COST
            } else {
                usize::try_from(own).unwrap_or(usize::MAX)
            },
            var,
            node: idx,
        }
    };

    let ENode::Op { children, .. } = node else {
        out.push(alone(false));
        return;
    };
    if children.iter().any(|&c| egraph.find(c) == canonical) {
        out.push(alone(true));
        return;
    }
    for &child in children.iter() {
        let c = egraph.find(child);
        let ci = compact[c.0 as usize];
        assert!(
            ci != u32::MAX,
            "expand_node: e-class {} is a child of a live class but the post-order did not \
             enumerate it — the two traversals have drifted",
            c.0
        );
        // Still on the DFS stack — the same back edge the DP prices at the
        // cycle sentinel.
        if ci as usize > me {
            out.push(alone(true));
            return;
        }
    }

    // Merge the children's beams left to right, pruning to `width` between
    // steps: `O(children × width²)` merges rather than `width^children`.
    // Seat 0 of every accumulator is the all-anchor prefix, so the next step
    // always has at least one admissible pair.
    let mut anchor_acc = Acc {
        members: Vec::new(),
        below: 0,
        var: Variance::CONST,
    };
    let mut free: Vec<Acc> = Vec::new();
    for &child in children.iter() {
        let ci = compact[egraph.find(child).0 as usize] as usize;
        let cbeam = &beams[ci];
        let next_anchor = merge_acc(&anchor_acc, &cbeam[0])
            .expect("expand_node: anchors name one term per class and cannot conflict");
        let mut next_free: Vec<Acc> = Vec::new();
        for a in core::iter::once(&anchor_acc).chain(free.iter()) {
            for s in cbeam {
                let Some(m) = merge_acc(a, s) else { continue };
                next_free.push(m);
            }
        }
        next_free.sort_by(|a, b| acc_key(a).cmp(&acc_key(b)));
        next_free.dedup_by(|a, b| acc_key(a) == acc_key(b));
        next_free.retain(|a| acc_key(a) != acc_key(&next_anchor));
        next_free.truncate(width.saturating_sub(1));
        anchor_acc = next_anchor;
        free = next_free;
    }

    for acc in core::iter::once(&anchor_acc).chain(free.iter()) {
        let own = extract::weighted_own(costs, node, shape.evals(acc.var)) as u64;
        let mut members = acc.members.clone();
        members.push(Member {
            class: me as u32,
            node: idx,
            own,
        });
        out.push(State {
            cost: usize::try_from(acc.below.saturating_add(own)).unwrap_or(usize::MAX),
            members,
            var: acc.var,
            node: idx,
        });
    }
}

fn merge_acc(a: &Acc, s: &State) -> Option<Acc> {
    let (members, below) = merge(&a.members, &s.members)?;
    Some(Acc {
        members,
        below,
        var: a.var.union(s.var),
    })
}

fn acc_key(a: &Acc) -> (u64, &[Member]) {
    (a.below, a.members.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::cost::CostModel;
    use crate::egraph::ops::op_from_kind;
    use crate::egraph::optimizer::Optimizer;
    use alloc::vec;
    use pixelflow_ir::OpKind;

    fn op(egraph: &mut EGraph, kind: OpKind, children: Vec<EClassId>) -> EClassId {
        egraph.add(ENode::Op {
            op: op_from_kind(kind).expect("modelled op"),
            children,
        })
    }

    /// A handful of saturated graphs with real sharing in them: enough
    /// structure that the two DP objectives disagree, small enough that the
    /// whole matrix runs in a unit test.
    fn cases() -> Vec<(&'static str, EGraph, EClassId)> {
        let mut out = Vec::new();

        let build: Vec<(&'static str, fn(&mut EGraph) -> EClassId)> = vec![
            ("fma_chain", |g| {
                let x = g.add(ENode::Var(0));
                let y = g.add(ENode::Var(1));
                let mut acc = x;
                for i in 0..6 {
                    let c = g.add(ENode::constant(i as f32 + 1.5));
                    let m = op(g, OpKind::Mul, vec![acc, c]);
                    acc = op(g, OpKind::Add, vec![m, y]);
                }
                acc
            }),
            ("shared_sin", |g| {
                let x = g.add(ENode::Var(0));
                let s = op(g, OpKind::Sin, vec![x]);
                let sq = op(g, OpKind::Mul, vec![s, s]);
                let c = op(g, OpKind::Cos, vec![x]);
                let cq = op(g, OpKind::Mul, vec![c, c]);
                let sum = op(g, OpKind::Add, vec![sq, cq]);
                op(g, OpKind::Mul, vec![sum, s])
            }),
            ("length_sq", |g| {
                let x = g.add(ENode::Var(0));
                let y = g.add(ENode::Var(1));
                let z = g.add(ENode::Var(2));
                let xx = op(g, OpKind::Mul, vec![x, x]);
                let yy = op(g, OpKind::Mul, vec![y, y]);
                let zz = op(g, OpKind::Mul, vec![z, z]);
                let a = op(g, OpKind::Add, vec![xx, yy]);
                let b = op(g, OpKind::Add, vec![a, zz]);
                let r = op(g, OpKind::Sqrt, vec![b]);
                let inv = op(g, OpKind::Recip, vec![r]);
                op(g, OpKind::Mul, vec![b, inv])
            }),
            ("exp_ring", |g| {
                let x = g.add(ENode::Var(0));
                let y = g.add(ENode::Var(1));
                let e = op(g, OpKind::Exp, vec![x]);
                let l = op(g, OpKind::Ln, vec![e]);
                let n = op(g, OpKind::Neg, vec![l]);
                let s = op(g, OpKind::Add, vec![n, y]);
                let m = op(g, OpKind::Mul, vec![s, e]);
                op(g, OpKind::Sub, vec![m, x])
            }),
        ];

        for (name, f) in build {
            let mut egraph = Optimizer::production().egraph();
            let root = f(&mut egraph);
            egraph.rebuild();
            let _ = egraph.saturate_budgeted(6, 4_000, Some(20_000));
            out.push((name, egraph, root));
        }
        out
    }

    fn shapes() -> [LatticeShape; 2] {
        [LatticeShape::POINT, LatticeShape::new([64, 64])]
    }

    fn same(a: &ExtractedDAG, b: &ExtractedDAG, what: &str) {
        assert_eq!(a.choices, b.choices, "{what}: choice maps differ");
        assert_eq!(a.dag_cost, b.dag_cost, "{what}: dag_cost differs");
        assert_eq!(a.total_cost, b.total_cost, "{what}: total_cost differs");
        assert_eq!(a.report, b.report, "{what}: extraction report differs");
        assert_eq!(a.schedule, b.schedule, "{what}: schedule differs");
        assert_eq!(a.shared, b.shared, "{what}: sharing differs");
    }

    /// `Greedy` is a name for production's extractor, not a second copy of
    /// it. The pin is here so a later edit that "moves" the chain into the
    /// impl has to keep the equality.
    #[test]
    fn greedy_is_production() {
        let costs = CostModel::latency_prior();
        for (name, egraph, root) in cases() {
            for shape in shapes() {
                let a = Greedy.extract(&egraph, root, &costs, shape);
                let b = extract::extract_dag_scoped(&egraph, root, &costs, shape);
                same(&a, &b, &alloc::format!("{name} @ {shape:?}"));
            }
        }
    }

    /// Width is the only variable between `Beam` and `Greedy`: at one seat
    /// the beam holds nothing but its anchor, and the anchor is the DP's own
    /// choice, so the two return the same bytes.
    #[test]
    fn beam_at_width_one_is_greedy() {
        let costs = CostModel::latency_prior();
        for (name, egraph, root) in cases() {
            for shape in shapes() {
                let a = Beam::width(1).extract(&egraph, root, &costs, shape);
                let b = Greedy.extract(&egraph, root, &costs, shape);
                same(&a, &b, &alloc::format!("{name} @ {shape:?}"));
            }
        }
    }

    /// The anchor is in every beam, so whatever width finds is at worst what
    /// the DP found. A width that could lose would make the measurement
    /// unreadable — a mixed table of wins and losses says nothing about
    /// whether search width is the missing axis.
    #[test]
    fn beam_never_loses_to_greedy() {
        let costs = CostModel::latency_prior();
        for (name, egraph, root) in cases() {
            for shape in shapes() {
                let greedy = Greedy.extract(&egraph, root, &costs, shape);
                for k in [2usize, 4, 8, 16] {
                    let beam = Beam::width(k).extract(&egraph, root, &costs, shape);
                    assert!(
                        beam.dag_cost <= greedy.dag_cost,
                        "{name} @ {shape:?}: Beam({k}) cost {} above Greedy's {}",
                        beam.dag_cost,
                        greedy.dag_cost
                    );
                }
            }
        }
    }

    /// Every state the beam returns names a term that can be materialized:
    /// the choice map is well-founded and its recomputed cost is the one
    /// reported.
    #[test]
    fn beam_returns_a_costable_term() {
        let costs = CostModel::latency_prior();
        for (name, egraph, root) in cases() {
            for k in [1usize, 4, 16] {
                let dag = Beam::width(k).extract(&egraph, root, &costs, LatticeShape::POINT);
                let recost = extract::cost_of_choices(
                    &egraph,
                    root,
                    &dag.choices,
                    &costs,
                    LatticeShape::POINT,
                );
                assert_eq!(
                    recost.dag, dag.dag_cost,
                    "{name}: Beam({k}) reported a cost its own choice map does not have"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "k must be at least 1")]
    fn width_zero_is_not_a_search() {
        let _ = Beam::width(0);
    }
}
