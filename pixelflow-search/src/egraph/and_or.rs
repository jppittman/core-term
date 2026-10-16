//! The extraction problem, lifted off the e-graph it was read from.
//!
//! Extraction solves an AND-OR graph over the classes a root reaches: a class
//! is an OR (it is realised by *one* of its e-nodes), an e-node is an AND (it
//! needs *all* of its distinct child classes). Which classes are live, what
//! each one's candidates are, and who depends on whom are facts about the
//! saturated graph alone. A lattice shape adds only weights (what each
//! candidate costs per evaluation) and a settlement that uses them, so the
//! facts are built once per saturated graph and every arm at every shape reads
//! the same value ([`AndOrGraph`]).
//!
//! Every table here is *live*-sized and indexed by a class's position in
//! [`AndOrGraph::classes`], not by its [`EClassId`]: the root reaches about a
//! third of a saturated glyph's classes, and a table sized by the whole graph
//! pays for the rest in every zeroing and every copy. The two index spaces
//! are two types ([`Live`] and [`EClassId`]), and a table keyed by liveness is
//! a [`LiveTable`], which a class id cannot index.
//!
//! The graph is only meaningful beside the e-graph it was read from: a class
//! merged or a node added afterwards leaves it describing something else.
//! [`Frozen`] is the one place the two are paired, so a [`Problem`] (what
//! extraction takes) cannot name a graph and a problem that were not read
//! from one another: the pair owns the e-graph it read, or borrows it for as
//! long as the problem lives, and no code holds a way to change it.

use super::graph::EGraph;
use super::node::EClassId;
use super::provenance::ENodeId;
use alloc::vec::Vec;
use core::borrow::Borrow;
use core::ops::{Index, IndexMut};

/// The position a class holds in the root's reach: the index into
/// [`AndOrGraph::classes`].
///
/// Its own type so that a table keyed by liveness and one keyed by
/// [`EClassId`] cannot be read with each other's index: both are `usize`
/// underneath, both are below the e-graph's class count, and the wrong one
/// reads a plausible value, which is a different selection and not a panic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Live(usize);

impl Live {
    /// The position as a number, for an encoding keyed by position (a bit set
    /// over the reach).
    pub(crate) fn index(self) -> usize {
        self.0
    }

    /// The class at position `index`: the inverse of [`Self::index`], for
    /// decoding such an encoding.
    pub(crate) fn from_index(index: usize) -> Self {
        Self(index)
    }
}

/// A table with one entry per live class, indexed by [`Live`] and nothing
/// else.
pub(crate) struct LiveTable<T>(Vec<T>);

impl<T> Index<Live> for LiveTable<T> {
    type Output = T;

    fn index(&self, live: Live) -> &T {
        &self.0[live.0]
    }
}

impl<T> IndexMut<Live> for LiveTable<T> {
    fn index_mut(&mut self, live: Live) -> &mut T {
        &mut self.0[live.0]
    }
}

/// One distinct child class of a candidate, and how many of the candidate's
/// operand slots name it.
///
/// `Mul(a, a)` has one child, `a`, used twice. A *tree* price pays for a
/// child once per use and a *DAG* price once however often it is used, so
/// both are one fold over these rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Child {
    pub(crate) live: Live,
    pub(crate) uses: usize,
}

/// One e-node of one live class, by position: what a pass prices and a
/// settlement chooses.
///
/// Only [`AndOrGraph`] makes one, so its class, position and index agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Candidate {
    class: EClassId,
    live: Live,
    idx: usize,
}

impl Candidate {
    pub(crate) fn class(self) -> EClassId {
        self.class
    }

    pub(crate) fn live(self) -> Live {
        self.live
    }

    /// Position among the class's e-nodes, in the order `egraph.nodes(class)`
    /// yields them, which is also what a choice map records.
    pub(crate) fn idx(self) -> usize {
        self.idx
    }
}

/// The shape-independent half of an extraction: the live classes, each
/// class's candidates with their distinct children, and the parent index.
///
/// The orders are part of the value, because settlement is order-sensitive
/// where costs tie, and a tie decides which node a class keeps:
///
/// - **classes** are the root's reach in DFS post-order, so the initial
///   relaxation visits them in the order it always has;
/// - **candidates** of a class are in e-node order;
/// - **children** of a candidate are distinct, in first-occurrence order;
/// - **parents** of a class are in the order the building loop meets them:
///   by parent class, then candidate, then child.
///
/// A candidate that names its own class is not special. Its self-edge is a
/// child it waits on that cannot settle before it does, so it never becomes
/// admissible, with no case for it here or in settlement.
pub(crate) struct AndOrGraph {
    root: EClassId,
    /// The root's reach in DFS post-order; a class's index here is its
    /// [`Live`]. The root is last.
    classes: Vec<EClassId>,
    /// Candidates of live class `i` are the slots
    /// `first_candidate[i]..first_candidate[i + 1]`.
    first_candidate: Vec<usize>,
    /// Children of the candidate in slot `k` are
    /// `children[first_child[k]..first_child[k + 1]]`.
    first_child: Vec<usize>,
    children: Vec<Child>,
    /// The node of the candidate in slot `k`, by id: the `k`-th tag of the
    /// member walk that numbered the slots. A candidate is a position (what a
    /// choice map records) and an id (what a pick holds); this is the map from
    /// one to the other.
    node_of_slot: Vec<ENodeId>,
    /// Candidates that name live class `i` are
    /// `parents[first_parent[i]..first_parent[i + 1]]`.
    first_parent: Vec<usize>,
    parents: Vec<Parent>,
}

/// A candidate that needs some class, by position: its own class and its
/// index in that class's e-nodes.
#[derive(Clone, Copy)]
struct Parent {
    live: Live,
    idx: usize,
}

impl AndOrGraph {
    /// The problem `egraph` poses for extraction from `root`.
    ///
    /// A bare graph carries no proof of which e-graph it was read from; what
    /// extraction takes is a [`Problem`], which only [`Frozen`] makes.
    #[must_use]
    pub(crate) fn of(egraph: &EGraph, root: EClassId) -> Self {
        let root = egraph.find(root);
        let classes = reach_in_post_order(egraph, root);
        assert_eq!(
            classes.last(),
            Some(&root),
            "AndOrGraph: the walk from the root must finish on the root, which the last class is \
             taken to be"
        );
        let live_of = LiveIndex::new(egraph.num_classes(), &classes);

        let mut first_candidate = Vec::with_capacity(classes.len() + 1);
        let mut first_child = alloc::vec![0];
        let mut children: Vec<Child> = Vec::new();
        let mut parent_count = alloc::vec![0usize; classes.len()];
        let mut row: Vec<Child> = Vec::new();
        let mut node_of_slot: Vec<ENodeId> = Vec::new();
        for &class in &classes {
            first_candidate.push(first_child.len() - 1);
            for tag in egraph.tags(class) {
                let node = egraph.node(tag);
                node_of_slot.push(tag);
                row.clear();
                // `children_slice`, not an `Op`'s operands: a `Reduce`'s body
                // is a child too. A row that skipped it once swallowed a
                // fold's body, so the shared pass priced a fold as its
                // combiner chain alone while the term's price charged for the
                // body, the DP settled on a claim below the price of the term
                // it named, and the claim/price audit fired.
                for &child in node.children_slice() {
                    let live = live_of.position(egraph.find(child));
                    match row.iter_mut().find(|c| c.live == live) {
                        Some(seen) => seen.uses += 1,
                        None => row.push(Child { live, uses: 1 }),
                    }
                }
                for child in &row {
                    parent_count[child.live.0] += 1;
                }
                children.extend_from_slice(&row);
                first_child.push(children.len());
            }
        }
        first_candidate.push(first_child.len() - 1);

        let mut first_parent = Vec::with_capacity(classes.len() + 1);
        first_parent.push(0);
        for count in &parent_count {
            first_parent.push(first_parent.last().copied().unwrap_or(0) + count);
        }
        let mut next_parent = first_parent.clone();
        let mut parents = alloc::vec![
            Parent {
                live: Live(0),
                idx: 0
            };
            children.len()
        ];
        for live in 0..classes.len() {
            let slots = first_candidate[live]..first_candidate[live + 1];
            for (idx, slot) in slots.enumerate() {
                for child in &children[first_child[slot]..first_child[slot + 1]] {
                    parents[next_parent[child.live.0]] = Parent {
                        live: Live(live),
                        idx,
                    };
                    next_parent[child.live.0] += 1;
                }
            }
        }

        // This value outlives the extraction that built it (a cache holds it
        // for the process), so the slack a grown vector keeps is not freed
        // with a transient.
        let mut classes = classes;
        classes.shrink_to_fit();
        first_candidate.shrink_to_fit();
        first_child.shrink_to_fit();
        children.shrink_to_fit();
        node_of_slot.shrink_to_fit();

        Self {
            root,
            classes,
            first_candidate,
            first_child,
            children,
            node_of_slot,
            first_parent,
            parents,
        }
    }

    /// The canonical class extraction starts from.
    pub(crate) fn root(&self) -> EClassId {
        self.root
    }

    /// The root's position: the last class in post-order, which is the one
    /// the walk started from.
    pub(crate) fn root_live(&self) -> Live {
        Live(self.classes.len() - 1)
    }

    /// The live classes, in DFS post-order from the root.
    #[cfg(test)]
    pub(crate) fn classes(&self) -> &[EClassId] {
        &self.classes
    }

    /// How many classes the root reaches.
    pub(crate) fn len(&self) -> usize {
        self.classes.len()
    }

    /// Every live class, in post-order: the order the initial relaxation
    /// visits them.
    pub(crate) fn lives(&self) -> impl Iterator<Item = Live> + use<> {
        (0..self.classes.len()).map(Live)
    }

    /// The e-graph class at position `live`.
    pub(crate) fn class(&self, live: Live) -> EClassId {
        self.classes[live.0]
    }

    /// A table with `value` for every live class.
    pub(crate) fn table<T: Clone>(&self, value: T) -> LiveTable<T> {
        LiveTable(alloc::vec![value; self.classes.len()])
    }

    /// A table with `f(live)` for every live class.
    pub(crate) fn table_from_fn<T>(&self, f: impl FnMut(Live) -> T) -> LiveTable<T> {
        LiveTable(self.lives().map(f).collect())
    }

    /// Candidate `idx` of live class `live`, or `None` past the class's last
    /// e-node.
    pub(crate) fn candidate(&self, live: Live, idx: usize) -> Option<Candidate> {
        let count = self.first_candidate[live.0 + 1] - self.first_candidate[live.0];
        (idx < count).then(|| Candidate {
            class: self.classes[live.0],
            live,
            idx,
        })
    }

    /// The candidates of live class `live`, in e-node order.
    pub(crate) fn candidates(&self, live: Live) -> impl Iterator<Item = Candidate> + '_ {
        let count = self.first_candidate[live.0 + 1] - self.first_candidate[live.0];
        (0..count).map(move |idx| Candidate {
            class: self.classes[live.0],
            live,
            idx,
        })
    }

    /// How many candidates there are in all: the length of a table keyed by
    /// [`Self::slot`].
    #[cfg(test)]
    pub(crate) fn slot_count(&self) -> usize {
        self.first_child.len() - 1
    }

    /// `candidate`'s position among all candidates: its row in a table of
    /// per-candidate state.
    pub(crate) fn slot(&self, candidate: Candidate) -> usize {
        self.first_candidate[candidate.live.0] + candidate.idx
    }

    /// The node `candidate` is, by id.
    pub(crate) fn node_id(&self, candidate: Candidate) -> ENodeId {
        self.node_of_slot[self.slot(candidate)]
    }

    /// The distinct classes `candidate` needs, in first-occurrence order.
    pub(crate) fn children(&self, candidate: Candidate) -> &[Child] {
        let slot = self.slot(candidate);
        &self.children[self.first_child[slot]..self.first_child[slot + 1]]
    }

    /// How many distinct classes each candidate needs, by [`Self::slot`]:
    /// the counters a settlement counts down.
    pub(crate) fn child_counts(&self) -> Vec<usize> {
        self.first_child.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// The candidates that need live class `live`, in the order the build
    /// loop met them.
    pub(crate) fn parents(&self, live: Live) -> impl Iterator<Item = Candidate> + '_ {
        self.parents[self.first_parent[live.0]..self.first_parent[live.0 + 1]]
            .iter()
            .map(|p| Candidate {
                class: self.classes[p.live.0],
                live: p.live,
                idx: p.idx,
            })
    }
}

/// An e-graph and the [`AndOrGraph`] read from it, held together.
///
/// `G` is the e-graph itself or a handle to one (`&EGraph`, `Arc<EGraph>`),
/// and nothing here lends out a way to change it, so the graph stays the one
/// the problem was read from for as long as this value lives. A borrowed
/// graph cannot be merged into while its problem is alive; an owned one is
/// not reachable for a merge at all.
pub(crate) struct Frozen<G> {
    egraph: G,
    graph: AndOrGraph,
}

impl<G: Borrow<EGraph>> Frozen<G> {
    /// Read the problem `egraph` poses for extraction from `root`.
    pub(crate) fn new(egraph: G, root: EClassId) -> Self {
        let graph = AndOrGraph::of(egraph.borrow(), root);
        Self { egraph, graph }
    }

    /// The pair, as what extraction takes.
    pub(crate) fn problem(&self) -> Problem<'_> {
        Problem {
            egraph: self.egraph.borrow(),
            graph: &self.graph,
        }
    }
}

/// An extraction problem and the e-graph it is about: what every pass takes
/// where it would take an `&EGraph` and an `&AndOrGraph`, which nothing
/// would have tied together. Only [`Frozen`] makes one.
#[derive(Clone, Copy)]
pub(crate) struct Problem<'a> {
    egraph: &'a EGraph,
    graph: &'a AndOrGraph,
}

impl<'a> Problem<'a> {
    pub(crate) fn egraph(self) -> &'a EGraph {
        self.egraph
    }

    pub(crate) fn graph(self) -> &'a AndOrGraph {
        self.graph
    }

    /// The canonical class extraction starts from.
    pub(crate) fn root(self) -> EClassId {
        self.graph.root
    }
}

/// Position in the root's reach for each class id, for the build only.
struct LiveIndex(Vec<Option<Live>>);

impl LiveIndex {
    fn new(num_classes: usize, classes: &[EClassId]) -> Self {
        let mut index = alloc::vec![None; num_classes];
        for (live, class) in classes.iter().enumerate() {
            index[class.index()] = Some(Live(live));
        }
        Self(index)
    }

    /// The position of `class`, which the DFS that enumerated the reach has
    /// already visited: every canonical child of a live class is live.
    fn position(&self, class: EClassId) -> Live {
        self.0[class.index()].unwrap_or_else(|| {
            panic!(
                "AndOrGraph: e-class {} is a child of a live class but was not enumerated as live",
                class.index()
            )
        })
    }
}

/// The classes reachable from `root`, in DFS post-order.
///
/// Iterative, with an explicit stack, so a deep term cannot overflow the
/// thread's. A class met again while its children are still being walked is a
/// cycle (commutativity alone closes them on a saturated graph) and is
/// skipped, which is why a class can appear before a descendant that reaches
/// it back.
fn reach_in_post_order(egraph: &EGraph, root: EClassId) -> Vec<EClassId> {
    let num_classes = egraph.num_classes();
    let mut order: Vec<EClassId> = Vec::new();
    let mut done = alloc::vec![false; num_classes];
    let mut on_stack = alloc::vec![false; num_classes];
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.index();

        if done[idx] {
            continue;
        }
        if children_done {
            on_stack[idx] = false;
            done[idx] = true;
            order.push(canonical);
            continue;
        }
        if on_stack[idx] {
            continue;
        }
        on_stack[idx] = true;
        stack.push((canonical, true));
        for node in egraph.nodes(canonical) {
            for &child in node.children_slice() {
                if !done[egraph.find(child).index()] {
                    stack.push((child, false));
                }
            }
        }
    }

    order
}
