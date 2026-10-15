//! The E-Graph data structure and operations.

use rustc_hash::FxHashMap as HashMap;

use super::live::LiveClasses;
use super::node::{EClassId, ENode};
use super::ops;
use super::provenance::{ENodeId, Provenance};
use super::rewrite::RewriteAction;
use alloc::collections::BTreeMap;
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{KernelKey, Variance};

mod runner;
pub use runner::{
    ApplicationMask, HARD_MINTED_LIMIT, Limits, MaskScope, RewriteTarget, Saturation,
    SaturationStats, SaturationStop,
};

#[derive(Clone, Debug, Default)]
pub(crate) struct EClass {
    pub(crate) nodes: Vec<ENode>,
    /// Stable identity per node, parallel to `nodes`: `tags[i]` names
    /// `nodes[i]`. Must be kept in lockstep with `nodes` through every
    /// mutation (union's extend, rebuild's take/canonicalize/extend) — see
    /// `provenance` module docs.
    pub(crate) tags: Vec<ENodeId>,
}

/// The store: classes, the union-find over them, the memo that hash-conses
/// their nodes, the per-class facts, and the graph's own mutation log.
///
/// It holds no rules and counts no applications — those belong to the runner
/// that sweeps it ([`Saturation`]). What a store does know about rewriting is
/// the change log the runner's dirty tracking reads
/// (`mutation_counter`, `last_changed`) and, when one is
/// kept, the journal that attributes its mutations.
#[derive(Clone)]
pub struct EGraph {
    pub(crate) classes: Vec<EClass>,
    pub(crate) parent: Vec<EClassId>,
    memo: HashMap<ENode, EClassId>,
    /// The classes [`Self::class_ids`] yields, as a set: a rule scan reads it
    /// instead of asking `find` of every class ever minted. Written only by
    /// [`Self::track_liveness`], which restates [`Self::is_live`] for one
    /// class and is called where a class is minted, merged or drained
    /// ([`Self::add`], [`Self::union`], [`Self::rebuild_budgeted`]). Checked
    /// against `class_ids` whenever a rebuild finishes, so a site that forgets
    /// to call it fails loudly instead of dropping classes from every scan.
    live: LiveClasses,
    worklist: Vec<EClassId>,
    /// `queued[id.index()]` is `true` while `id` has an entry on `worklist`
    /// awaiting `rebuild_budgeted`. Guards `union`'s push: several unions
    /// landing on the same parent in one round used to enqueue it once per
    /// union, and `rebuild_budgeted` paid a full canonicalize-and-memo-probe
    /// pass over that class's node vector per redundant entry even though
    /// only the state after the *last* union in the round matters. Parallel
    /// to `classes`/`parent`, grown alongside them in `add`.
    queued: Vec<bool>,
    /// Global monotonic counter minting `ENodeId`s in `add()`.
    next_enode_id: u64,
    /// The constant each class is known to equal, as f32 bits, indexed by
    /// class id — maintained independently of `EClass::nodes` on purpose.
    /// `rebuild` drains a class's nodes with `mem::take` and only THEN
    /// performs congruence unions, so a guard that scanned the node vector
    /// saw an empty class and waved contradictory merges through at exactly
    /// the moment congruence closure does its work. The fact must outlive the
    /// nodes.
    const_fact: Vec<Option<u32>>,
    /// Which binders each class may depend on, indexed by class id — the
    /// per-class variance fact (docs/archive/plans/2026-09-23-an-integral-is-a-fold.md
    /// §1, §3), kept beside `const_fact` and out of `EClass::nodes` for the
    /// same reason: a guard read mid-`rebuild` must not see a drained class.
    ///
    /// **Law:** `var_fact[C] ⊇ var(⟦C⟧)` — the fact over-approximates the
    /// variance of the one function every member of `C` denotes, so a bit
    /// absent from the fact is a binder the class is provably constant along.
    ///
    /// - **Seeded** in [`EGraph::add`] by [`ENode::variance`], the node's
    ///   transfer function, over its children's facts *as they stand then*.
    ///   Each child's fact over-approximates the child, and the transfer is
    ///   monotone, so the seed over-approximates the node.
    /// - **Merged** in [`EGraph::union`] by *intersection*. Both classes
    ///   denote the same function and both facts over-approximate it, so a
    ///   bit either one lacks is a bit the function lacks. A fact only ever
    ///   shrinks from its seed and never grows up from `∅` — the greatest
    ///   fixpoint, not the least — which is why a cycle cannot certify an
    ///   invariance: a class holding `x` and `x + 0` got each member's fact
    ///   from facts that already existed, never from an assumption about
    ///   itself.
    /// - **Never repaired upward.** A union that shrinks a child's fact does
    ///   not revisit the parents seeded from the larger one. A stale parent
    ///   fact is still a superset, so a rule reading it may miss an
    ///   opportunity but can never fire wrongly; soundness needs no repair.
    var_fact: Vec<Variance>,
    /// The units this graph may hold as leaves, with the variance each
    /// carries — told to it by the runtime tier's unit walk
    /// ([`EGraph::admit_unit`]) before the term is inserted. A
    /// [`ENode::Ref`] is only its key, so this is where its variance is read
    /// from. A unit's body is not in this graph, so its variance cannot be
    /// computed here; and a
    /// reference the graph was not told about is one [`insert`](super::insert)
    /// declines, which is what keeps every other path — the macro tier,
    /// research tools — exactly as it was (docs/language/2026-09-25-the-language-is-kernel.md
    /// §4, O1).
    units: BTreeMap<KernelKey, Variance>,
    /// Unions REFUSED because they would assert two numerically unequal
    /// constants equal — a proved falsehood the graph declines to absorb.
    /// Distinct (bits, bits) pairs, kept for reporting and tests; see
    /// [`EGraph::union`].
    refused_const_unions: Vec<(u32, u32)>,
    /// Global monotonic counter, bumped once per class-content mutation
    /// (`add`'s memo miss, `union`'s merge) — never reset, never bumped by
    /// anything that only re-canonicalizes existing content (`nodes()`
    /// already resolves through `find`, so rewriting a child's stored id to
    /// its current canonical form during rebuild surfaces nothing a lookup
    /// couldn't already see). See [`Self::class_is_dirty`].
    mutation_counter: u64,
    /// `last_changed[c.index()]` is the [`Self::mutation_counter`] value as
    /// of the most recent time canonical class `c`'s own node list gained
    /// content — set at creation in `add` and bumped in `union` when `c`
    /// survives as the merge's parent. Parallel to `classes`, grown
    /// alongside it.
    last_changed: Vec<u64>,
    /// The graph's mutation log, or `None` when the graph was built without
    /// one ([`Self::without_journal`]).
    ///
    /// Origins and merges are recorded as they happen — `add` and `union` are
    /// the only writers — so the log cannot be switched on after the fact,
    /// and a graph built without it has nothing to query rather than an
    /// empty or misattributed answer. The runner writes its cursor (the
    /// current step and the application being executed); see
    /// [`Provenance::open_application`].
    journal: Option<Provenance>,
}

/// Why a research walk over the journal refused a graph that keeps none.
#[cfg(feature = "provenance-journal")]
const NO_JOURNAL: &str = "this graph was built without a journal (Saturation::without_journal, or \
     an Optimizer with no observer attached) — a derivation walk has nothing to read";

/// [`EGraph::class_is_dirty`] checks this many hops of a class's forward
/// neighborhood (the class itself, its direct children, and their
/// children) against a rule's last-swept generation.
///
/// This is the audited maximum matching depth across every rule in
/// [`super::all_rules`] — how many levels beyond the single node handed to
/// `Rewrite::apply` a rule inspects via `egraph.nodes()` on a child (or
/// grandchild) e-class. `math/trig.rs`'s `Pythagorean` and
/// `ReverseAngleAddition` are the only rules that reach a grandchild
/// (through `extract_trig_arg`/`extract_squared_trig`/`extract_sin_cos_mul`);
/// every other rule in the production set — all of `math/algebra.rs`,
/// `math/parity.rs`, `math/exp.rs`, `math/power.rs`, `math/fusion.rs`, and
/// `egraph/derivative.rs`'s `ChainRule` — is depth 0 or 1. A dirty-tracking
/// scheme that only propagated 1 hop would be UNSOUND for those two rules:
/// a class could become newly matchable when a grandchild changes without
/// its own node list, or its direct children's, changing at all, and a
/// 1-hop scheme would silently never re-check it — under-saturation that no
/// correctness test can see, only a comparison against the un-skipped
/// extraction cost can. The runtime tier's `FactorFold` (outside
/// `all_rules`, in `fold_rules`) is depth 2 as well: it reads a fold's body
/// class and then its operands' variance facts, and a fact changes only when
/// its own class is unioned, which bumps that class's `last_changed`.
///
/// This is a single, uniform, crate-wide constant rather than a per-rule
/// depth precisely so a future rule cannot silently exceed it the way a
/// per-rule value could go stale unnoticed: every rule is checked to this
/// depth regardless of what it actually needs, so a new rule that matches
/// deeper than this is caught the same way `Pythagorean` would be — by
/// `docs/archive/plans/2026-09-08-egraph-cpu-memory-profile.md`'s methodology
/// (compare extracted cost with dirty-tracking on vs. off), not by silent
/// trust in a number nobody re-audits. Raise it if that audit is ever
/// redone and finds a deeper rule; do not lower it without redoing the
/// audit above.
const DIRTY_TRACKING_MAX_DEPTH: u8 = 2;

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    /// Create an empty store.
    ///
    /// With `provenance-journal` the mutation log is kept, so every direct
    /// caller keeps what it had. Rules are the runner's: see
    /// [`Saturation::over`].
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            parent: Vec::new(),
            memo: HashMap::default(),
            live: LiveClasses::default(),
            worklist: Vec::new(),
            queued: Vec::new(),
            next_enode_id: 0,
            const_fact: Vec::new(),
            var_fact: Vec::new(),
            units: BTreeMap::new(),
            refused_const_unions: Vec::new(),
            mutation_counter: 0,
            last_changed: Vec::new(),
            journal: Some(Provenance::new()),
        }
    }

    /// The same store with its mutation log turned off.
    ///
    /// A choice made when the graph is built: an origin or a merge recorded
    /// without the application that caused it reads as a seed or a congruence
    /// union, so a log that could be flipped mid-run would hand out wrong
    /// answers rather than missing ones. Production builds its graphs this
    /// way, and a cached graph then carries no log at all.
    #[must_use]
    pub fn without_journal(mut self) -> Self {
        assert!(
            self.classes.is_empty(),
            "a journal is chosen when the graph is built: this graph already holds {} classes, \
             whose births it would have to forget",
            self.classes.len(),
        );
        self.journal = None;
        self
    }

    /// Advance the journal's step, if one is kept. The runner's cursor: the
    /// one place the store is told something about the sweep that is
    /// mutating it.
    fn begin_step(&mut self) {
        #[cfg(feature = "provenance-journal")]
        if let Some(journal) = &mut self.journal {
            journal.begin_step();
        }
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

    /// `node` with every operand class replaced by its canonical class: the
    /// form the memo keys on, and the form two nodes are compared in.
    ///
    /// Every operand, a fold's body included. A graph between a `union` and
    /// the `rebuild` that follows it holds nodes naming merged-away classes,
    /// and only this copy can be compared with another.
    pub(crate) fn canonical_node(&self, mut node: ENode) -> ENode {
        for child in node.children_slice_mut() {
            *child = self.find(*child);
        }
        node
    }

    /// Tell this graph that `key` names a **unit** varying as `variance` says,
    /// so a reference to it inserts as an opaque leaf
    /// ([`ENode::Ref`]) rather than being declined.
    ///
    /// The runtime tier's unit walk calls this once per unit a term reads,
    /// before inserting the term: it is the walk that holds each unit's body,
    /// and so the only place its variance can be computed
    /// (docs/language/2026-09-25-the-language-is-kernel.md §4, O1).
    ///
    /// # Panics
    ///
    /// Panics if `variance` names a binder or a retired axis — a unit is a
    /// closed term over the coordinates and uniforms, and nothing else can be
    /// optimized out of its context, because no rewrite outside it can reach
    /// a binder inside it. Panics too if `key` was admitted before with a
    /// different variance: a key names one referent, so its variance is one
    /// value, and two `Ref` nodes for one key would otherwise differ only in
    /// what they claim about it.
    pub(crate) fn admit_unit(&mut self, key: KernelKey, variance: Variance) {
        assert!(
            variance.without(Variance::COORDS).is_const(),
            "EGraph::admit_unit: {key:?} varies as {variance:?} — a unit must be \
             closed over the coordinates, and this one reads a binder no \
             rewrite outside it could substitute"
        );
        if let Some(admitted) = self.units.insert(key, variance) {
            assert_eq!(
                admitted, variance,
                "EGraph::admit_unit: {key:?} was admitted as varying as {admitted:?} and is \
                 admitted again as {variance:?}"
            );
        }
    }

    /// The variance a unit admitted by [`Self::admit_unit`] carries, or
    /// `None` if `key` was never admitted.
    pub(crate) fn unit_variance(&self, key: KernelKey) -> Option<Variance> {
        self.units.get(&key).copied()
    }

    /// The variance of the unit a [`ENode::Ref`] to `key` names: what
    /// [`Self::unit_variance`] answers, for a node the graph holds.
    ///
    /// # Panics
    ///
    /// Panics if `key` was never admitted. [`Self::add`] refuses a `Ref` to
    /// such a unit, so no node in the graph names one.
    pub(crate) fn ref_variance(&self, key: KernelKey) -> Variance {
        self.unit_variance(key)
            .unwrap_or_else(|| panic!("EGraph: a Ref to {key:?}, which no unit walk admitted"))
    }

    /// Insert `node`, returning the e-class that contains it.
    ///
    /// **Total, and the law depends on it.** The returned class always
    /// contains `node`: either the memo already had it, or this call
    /// allocated a fresh class holding exactly it. There is no size at
    /// which `add` returns something else, because there is nothing else
    /// it could correctly return — this is law L1 (`add` is a homomorphism
    /// from the term algebra onto the e-graph quotient,
    /// `docs/archive/plans/2026-09-02-optimizer-api.md` §1), and L4 (any saturation
    /// policy preserves denotation) is proved from it.
    ///
    /// The class budget therefore is not enforced here. It is enforced
    /// where growth is *decided* — the saturation drivers, each of which
    /// clamps its caller's cap to [`HARD_MINTED_LIMIT`] and stops scanning
    /// before it calls `add`. A driver that stops early leaves a graph
    /// holding fewer equalities than an exhaustive run would; every one it
    /// does hold is still true.
    ///
    /// # Panics
    ///
    /// Panics on an [`ENode::Ref`] to a unit no walk has admitted
    /// (`EGraph::admit_unit`): the node is only a key, and the graph holds
    /// nothing to say what it varies as.
    pub fn add(&mut self, node: ENode) -> EClassId {
        let node = self.canonical_node(node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        // `node` is canonical, so every child indexes its class's live fact.
        // Read before the graph changes: a refusal leaves it as it was.
        let variance = node.variance(
            |key| self.ref_variance(key),
            |child| self.var_fact[child.index()],
        );
        let id = EClassId::from_index(self.classes.len());
        let enode_id = ENodeId(self.next_enode_id);
        self.next_enode_id += 1;
        #[cfg(feature = "provenance-journal")]
        if let Some(journal) = &mut self.journal {
            journal.record_birth(enode_id);
        }
        self.const_fact.push(node.as_f32().map(f32::to_bits));
        self.var_fact.push(variance);
        self.classes.push(EClass {
            nodes: vec![node],
            tags: vec![enode_id],
        });
        self.parent.push(id);
        self.track_liveness(id);
        self.queued.push(false);
        self.mutation_counter += 1;
        self.last_changed.push(self.mutation_counter);
        self.memo.insert(node, id);
        id
    }

    /// Refused constant unions: `(bits, bits)` pairs the graph declined to
    /// assert equal. Non-empty means some kernel hit the folding-vs-algebra
    /// collision — see the refusal block in [`EGraph::union`].
    #[must_use]
    pub fn refused_const_unions(&self) -> &[(u32, u32)] {
        &self.refused_const_unions
    }

    /// The binders `class` may depend on: a superset of what the function it
    /// denotes varies with, so `!variance(c).depends_on(b)` proves `c` is
    /// constant along `b` (see the `var_fact` field for the law and why it
    /// holds without repair).
    ///
    /// The side condition every loop-transformation rule over a fold asks —
    /// `⊕_i (c ⊗ f) = c ⊗ ⊕_i f` iff `i ∉ var(c)` — asked of the class
    /// rather than of one representative.
    #[must_use]
    pub(crate) fn variance(&self, class: EClassId) -> Variance {
        self.var_fact[self.find(class).index()]
    }

    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let a = self.find_mut(a);
        let b = self.find_mut(b);
        if a == b {
            return a;
        }
        // An e-class asserts "these are all equal", so a union placing two
        // numerically UNEQUAL constants in one class would be a proved
        // falsehood — and congruence closure amplifies any proved falsehood
        // into everything-equals-everything, with extraction tie-breaking
        // arbitrarily between unequal constants. The falsehoods are real:
        // constant folding computes f32-truths (`x + 2²⁴ = 2²⁴`) while the
        // algebraic rules compute ℝ-truths (`(x+y)−y = x`), each sound
        // alone. At the collision, REFUSE the union: an under-merged e-graph
        // is still sound — every class still holds only provably-equal
        // terms, and the cost is missed CSE at exactly the collision points
        // — while a false merge is unbounded. First prover keeps the class;
        // the refusal is journaled and reported so ill-conditioned kernels
        // are loud instead of subtly nondeterministic. (Folding constants in
        // f64 is the planned complement: it makes the folder agree with the
        // algebra at f32-observable scales, so collisions become
        // vanishingly rare and this valve becomes a pure detector.)
        //
        // `ca == cb` (not bit equality) deliberately admits ±0.0 — signed
        // zero selection is already platform-unspecified, so an optimizer
        // choosing a sign is within contract, same license as commuting
        // Min/Max over NaN. Bit equality additionally admits identical NaN
        // patterns, which `==` would wrongly reject: an all-ones comparison
        // mask reads as NaN and must be unionable with itself.
        // Read the class-level fact, NOT the node vector: `rebuild` drains
        // nodes with `mem::take` before performing congruence unions, so a
        // node-scanning guard is blind exactly when congruence closure runs.
        let (fa, fb) = (self.const_fact[a.index()], self.const_fact[b.index()]);
        if let (Some(ba), Some(bb)) = (fa, fb) {
            let (ca, cb) = (f32::from_bits(ba), f32::from_bits(bb));
            if ca != cb && ba != bb {
                let pair = (ba, bb);
                if !self.refused_const_unions.contains(&pair) {
                    // Journal ALWAYS; print only on request. This fires
                    // during ordinary kernel compilation — two computation
                    // orders of one real quantity disagreeing by an ulp is
                    // enough — so unconditional stderr would spam every
                    // build. `refused_const_unions()` is the durable record.
                    if std::env::var_os("PIXELFLOW_REPORT_CONST_REFUSALS").is_some() {
                        std::eprintln!(
                            "pixelflow e-graph: refusing a union that would assert \
                             {ca} = {cb} (bits {:#010x} vs {:#010x}) — the rule set \
                             derived contradictory constants (f32 folding vs \
                             algebra at an ill-conditioned input); keeping the \
                             classes separate costs only missed CSE",
                            pair.0,
                            pair.1
                        );
                    }
                    self.refused_const_unions.push(pair);
                }
                return if a.0 < b.0 { a } else { b };
            }
        }
        let (parent, child) = if a.0 < b.0 { (a, b) } else { (b, a) };
        self.parent[child.index()] = parent;
        let child_nodes = std::mem::take(&mut self.classes[child.index()].nodes);
        let child_tags = std::mem::take(&mut self.classes[child.index()].tags);
        self.classes[parent.index()].nodes.extend(child_nodes);
        self.classes[parent.index()].tags.extend(child_tags);
        self.track_liveness(child);
        self.track_liveness(parent);
        // The guard above proved the two facts agree (or that at most one
        // exists), so `or` is a merge, not a choice.
        if self.const_fact[parent.index()].is_none() {
            self.const_fact[parent.index()] = self.const_fact[child.index()];
        }
        // Intersection, not a choice: see `var_fact`'s law.
        self.var_fact[parent.index()] =
            self.var_fact[parent.index()].intersection(self.var_fact[child.index()]);
        // `parent`'s own node list just changed (gained `child`'s nodes) —
        // see `class_is_dirty`/`DIRTY_TRACKING_MAX_DEPTH`.
        self.mutation_counter += 1;
        self.last_changed[parent.index()] = self.mutation_counter;
        if !self.queued[parent.index()] {
            self.queued[parent.index()] = true;
            self.worklist.push(parent);
        }
        #[cfg(feature = "provenance-journal")]
        if let Some(journal) = &mut self.journal {
            journal.record_union(parent, child);
        }
        parent
    }

    /// Rebuild the e-graph completely. Processes the entire worklist.
    pub fn rebuild(&mut self) {
        self.rebuild_budgeted(usize::MAX);
    }

    /// Process up to `budget` worklist items. Each item canonicalizes one
    /// e-class's nodes and deduplicates via the memo table.
    ///
    /// The graph is consistent after each item — partially rebuilt is safe.
    /// Unprocessed classes may have stale canonical forms (rule matching
    /// might miss some equivalences) but won't produce wrong results.
    ///
    /// Returns the number of worklist items remaining.
    pub fn rebuild_budgeted(&mut self, budget: usize) -> usize {
        let mut processed = 0;
        while processed < budget {
            let id = match self.worklist.pop() {
                Some(id) => id,
                None => break,
            };
            processed += 1;
            // Cleared by the slot pushed at (not the class it now
            // canonicalizes to) — matches `union`'s guard, which sets it by
            // the same slot.
            self.queued[id.index()] = false;
            let id = self.find(id);
            let mut nodes = std::mem::take(&mut self.classes[id.index()].nodes);
            // `tags` must stay zipped with `nodes` through this loop: both
            // are compacted in place, a kept node moving from index `read` to
            // `kept` together with its tag, in the order they were drained,
            // so every tag stays pointed at the right node. Compacting the
            // taken buffers (rather than building fresh ones) is what spares
            // this hot loop two allocations and two frees per worklist item.
            let mut tags = std::mem::take(&mut self.classes[id.index()].tags);
            debug_assert_eq!(
                nodes.len(),
                tags.len(),
                "EClass.nodes and EClass.tags must never desync"
            );
            let mut kept = 0;
            for read in 0..nodes.len() {
                let node = self.canonical_node(nodes[read]);
                let tag = tags[read];
                if let Some(&existing) = self.memo.get(&node) {
                    let existing = self.find(existing);
                    if existing != id {
                        // `union` keeps `min(id, existing)`, so this call
                        // can go either way, and the write-back at the
                        // bottom of the loop has to survive both:
                        //   - `id` survives: `union`'s extend() appends
                        //     `existing`'s nodes (and tags) directly onto
                        //     `self.classes[id.index()]`, which mem::take
                        //     just emptied — so the write-back must EXTEND,
                        //     not overwrite, or those are dropped.
                        //   - `existing` survives: `id` is merged away and
                        //     `classes[id.index()]` stops being reachable
                        //     through `find` — so the write-back must target
                        //     `self.find(id)`, or everything still in
                        //     `new_nodes` is orphaned.
                        // This is a rebuild-time
                        // congruence-closure union, not a rule firing, so it
                        // carries no rule_idx in the provenance journal
                        // (no application bracket is open during `rebuild`).
                        self.union(id, existing);

                        // `union` REFUSES a merge that would assert two
                        // unequal constants equal, and a refusal here needs
                        // handling that a rule-driven refusal does not: the
                        // node below would be pushed back into `id` while
                        // `memo` names `existing`, leaving ONE ENode in two
                        // contradictory classes with only one of them
                        // reachable by lookup — later matching or extraction
                        // could then give the same expression different
                        // values. Leave the node with the class memo already
                        // names. `id` loses a node congruent to one it can no
                        // longer be merged with, which is under-merging: it
                        // costs CSE, not correctness.
                        if self.find(id) != self.find(existing) {
                            continue;
                        }
                    }
                } else {
                    self.memo.insert(node, id);
                }
                nodes[kept] = node;
                tags[kept] = tag;
                kept += 1;
            }
            nodes.truncate(kept);
            tags.truncate(kept);
            // Write back to the class `id` CANONICALIZES TO, not to `id`.
            // A mid-loop union() above picks `min(id, existing)` as the
            // surviving parent, so when `existing.0 < id.0` it is `id` that
            // was merged away — and `classes[id.index()]` is then a slot
            // `find` no longer routes to. Every node written there is
            // orphaned: `nodes()`/`tags()` canonicalize first, so nothing
            // can read them again. That is lost extraction alternatives
            // (under-merging: sound, since the surviving class still holds
            // only provably-equal terms) but lost *silently*, which this
            // project forbids.
            //
            // And extend, never assign over a non-empty vector: in the mirror
            // case, where `id` survives, union()'s extend() has already
            // appended `existing`'s nodes and tags onto this very vector
            // (which mem::take emptied above), and an assignment would
            // clobber them. `find(id) == id` there, so one write-back handles
            // both directions.
            let dest = self.find(id);
            restore_after_rebuild(&mut self.classes[dest.index()].nodes, nodes);
            restore_after_rebuild(&mut self.classes[dest.index()].tags, tags);
            self.track_liveness(dest);
        }
        if self.worklist.is_empty() {
            assert!(
                !self.live_has_drifted(),
                "the live-class set has drifted from the graph's canonical classes"
            );
        }
        self.worklist.len()
    }

    /// Number of pending worklist items (classes needing rebuild).
    pub fn pending_rebuilds(&self) -> usize {
        self.worklist.len()
    }

    /// The nodes of the class `id` belongs to, in the order the graph gives
    /// them. The order is semantics, not layout: rules that return on the
    /// first matching node, the node substitution reads and the candidate
    /// extraction keeps on a tie all depend on it.
    ///
    /// - [`add`](Self::add) gives `[node]`.
    /// - [`union`](Self::union) gives the surviving class's nodes, then the
    ///   merged-away class's.
    /// - A [`rebuild`](Self::rebuild) item detaches the nodes of the class it
    ///   processes, and any union inside the item appends to the class as
    ///   `union` does. At the end the kept nodes, in the order they were
    ///   walked, follow whatever the class then holds: `[absorbed] ++ [kept]`
    ///   when the class survives its unions. When the class loses one, it is
    ///   the survivor's prior nodes, then any the survivor absorbs later in
    ///   the same item, then `[kept]`.
    ///
    /// A node whose canonical form collides with a class it cannot merge with
    /// leaves the class, and rebuild deduplicates nothing: two members may be
    /// equal.
    pub fn nodes(&self, id: EClassId) -> &[ENode] {
        let id = self.find(id);
        &self.classes[id.index()].nodes
    }

    /// Get the stable `ENodeId` tags for the canonical class's nodes,
    /// parallel to `nodes(id)` — `tags(id)[i]` names `nodes(id)[i]`.
    pub fn tags(&self, id: EClassId) -> &[ENodeId] {
        let id = self.find(id);
        &self.classes[id.index()].tags
    }

    /// The mutation log (origins, application log, union journal), or `None`
    /// for a graph built [`without_journal`](Self::without_journal). See the
    /// `provenance` module for details.
    pub fn provenance(&self) -> Option<&Provenance> {
        self.journal.as_ref()
    }

    /// The mutation log of a graph that keeps one: for a walk that has
    /// nothing to compute without it (the hindsight labeler, the research
    /// harnesses). [`Self::provenance`] is the question to ask first.
    ///
    /// # Panics
    ///
    /// On a graph built without a journal.
    #[cfg(feature = "provenance-journal")]
    pub fn journal(&self) -> &Provenance {
        self.journal.as_ref().expect(NO_JOURNAL)
    }

    /// Look up the `ENode` for a given stable tag within a class, if it's
    /// still present there. `O(class size)` — classes are small in practice
    /// (equality-saturated e-classes rarely exceed a few dozen nodes).
    pub fn node_for_tag(&self, id: EClassId, tag: ENodeId) -> Option<&ENode> {
        let id = self.find(id);
        let class = &self.classes[id.index()];
        class
            .tags
            .iter()
            .position(|&t| t == tag)
            .map(|i| &class.nodes[i])
    }

    /// Compute the transitive rewrite-application ancestry of a set of
    /// chosen `(EClassId, ENodeId)` pairs (typically the nodes an extraction
    /// pass selected). See [`super::provenance::derivation_ancestors`] for
    /// the exact over-approximation made.
    #[cfg(feature = "provenance-journal")]
    pub fn derivation_ancestors(
        &self,
        chosen_nodes: &[(EClassId, ENodeId)],
    ) -> std::collections::BTreeSet<super::provenance::ApplicationId> {
        let tags_of = |class: EClassId| -> Vec<ENodeId> { self.tags(class).to_vec() };
        let children_of = |tag: ENodeId| -> Vec<EClassId> {
            for class in self.classes.iter() {
                if let Some(idx) = class.tags.iter().position(|&t| t == tag) {
                    return class.nodes[idx].children();
                }
            }
            Vec::new()
        };
        super::provenance::derivation_ancestors(
            &tags_of,
            &children_of,
            self.journal(),
            chosen_nodes,
        )
    }

    /// The tightened counterpart to [`EGraph::derivation_ancestors`] — see
    /// [`super::provenance::derivation_ancestors_tight`] for exactly which
    /// over-approximation axes are narrowed and why the result is still safe
    /// (a subset of `derivation_ancestors`'s result, a superset of the
    /// strict node-on-path bound). Additive: does not change what
    /// `derivation_ancestors` computes, so both can be run on the same
    /// episode for comparison.
    #[cfg(feature = "provenance-journal")]
    pub fn derivation_ancestors_tight(
        &self,
        chosen_nodes: &[(EClassId, ENodeId)],
    ) -> std::collections::BTreeSet<super::provenance::ApplicationId> {
        self.derivation_ancestors_tight_from(chosen_nodes, chosen_nodes)
    }

    /// [`EGraph::derivation_ancestors_tight`] for a walk that starts from a
    /// SUBSET of the extraction — one application's chosen output node, say
    /// — while still pruning with the extraction's complete choice map. See
    /// [`super::provenance::derivation_ancestors_tight`] for why passing
    /// only the seeds as the choice map degrades the walk back to the loose
    /// bound for every class it descends into.
    #[cfg(feature = "provenance-journal")]
    pub fn derivation_ancestors_tight_from(
        &self,
        seeds: &[(EClassId, ENodeId)],
        chosen_nodes: &[(EClassId, ENodeId)],
    ) -> std::collections::BTreeSet<super::provenance::ApplicationId> {
        let tags_of = |class: EClassId| -> Vec<ENodeId> { self.tags(class).to_vec() };
        let children_of = |tag: ENodeId| -> Vec<EClassId> {
            for class in self.classes.iter() {
                if let Some(idx) = class.tags.iter().position(|&t| t == tag) {
                    return class.nodes[idx].children();
                }
            }
            Vec::new()
        };
        let canonical_of = |class: EClassId| -> EClassId { self.find(class) };
        super::provenance::derivation_ancestors_tight(
            super::provenance::GraphView {
                tags_of: &tags_of,
                children_of: &children_of,
                canonical_of: &canonical_of,
            },
            self.journal(),
            super::provenance::TightWalk {
                seeds,
                chosen_nodes,
            },
        )
    }

    /// Get the number of e-class ids minted.
    ///
    /// Ids are never reclaimed — a union repoints a slot rather than freeing
    /// it — so this is the count [`Limits::minted`] bounds, not the number of
    /// live classes ([`Self::class_ids`] yields those).
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Iterate over all canonical e-class IDs.
    ///
    /// Returns an iterator of all e-class IDs that are canonical (i.e., roots
    /// of their union-find tree) and have at least one node.
    pub fn class_ids(&self) -> impl Iterator<Item = EClassId> + '_ {
        (0..self.classes.len())
            .map(EClassId::from_index)
            .filter(|&id| self.is_live(id))
    }

    /// Whether `id` is canonical and holds at least one node: the definition
    /// of a class [`Self::class_ids`] yields.
    fn is_live(&self, id: EClassId) -> bool {
        self.find(id) == id && !self.classes[id.index()].nodes.is_empty()
    }

    /// Restate [`Self::is_live`] for `id` in the [`LiveClasses`] set. The one
    /// writer of that set.
    fn track_liveness(&mut self, id: EClassId) {
        match self.is_live(id) {
            true => self.live.insert(id),
            false => self.live.remove(id),
        }
    }

    /// Whether the [`LiveClasses`] set disagrees with [`Self::class_ids`].
    fn live_has_drifted(&self) -> bool {
        !self.live.iter().eq(self.class_ids())
    }

    /// Get the total number of nodes across all e-classes.
    pub fn node_count(&self) -> usize {
        self.classes.iter().map(|c| c.nodes.len()).sum()
    }

    pub fn contains_const(&self, id: EClassId, val: f32) -> bool {
        self.nodes(id).iter().any(|n| n.is_const(val))
    }

    /// Whether `canonical`'s forward neighborhood — itself, its direct
    /// children, and their children (see [`DIRTY_TRACKING_MAX_DEPTH`]) —
    /// contains anything that changed since `swept`, a
    /// [`Self::mutation_counter`] value from some rule's last complete scan.
    ///
    /// Reads only `EClassId`s and `last_changed` entries, never clones a
    /// node vector or calls a rule's `apply` — this is the cheap check
    /// the runner's rule scan uses to decide whether the
    /// expensive path (clone the class's nodes, run every rule against
    /// every one) is worth paying for at all. Canonicalizes every id it
    /// follows: `parent`/`child` fields on a stored node can lag a union
    /// that happened after the node was written, and `last_changed` is only
    /// ever indexed by canonical class slots.
    fn class_is_dirty(&self, canonical: EClassId, swept: u64) -> bool {
        let canonical = self.find(canonical);
        if self.last_changed[canonical.index()] > swept {
            return true;
        }
        for node in &self.classes[canonical.index()].nodes {
            for &child in node.children_slice() {
                let child = self.find(child);
                if self.last_changed[child.index()] > swept {
                    return true;
                }
                if DIRTY_TRACKING_MAX_DEPTH < 2 {
                    continue;
                }
                for gnode in &self.classes[child.index()].nodes {
                    for &gchild in gnode.children_slice() {
                        if self.last_changed[self.find(gchild).index()] > swept {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Union `a` and `b`, reporting whether the graph actually changed.
    ///
    /// [`EGraph::union`] REFUSES merges that would assert two numerically
    /// unequal constants equal, so "the classes differed beforehand" is not
    /// evidence that anything merged. Counting a refusal as a change made
    /// saturation believe it had made progress, so it rebuilt and re-applied
    /// the same refused rewrite every iteration until its limit.
    fn union_counted(&mut self, a: EClassId, b: EClassId) -> usize {
        if self.find(a) == self.find(b) {
            return 0;
        }
        self.union(a, b);
        usize::from(self.find(a) == self.find(b))
    }

    /// Predict exactly how many new e-classes committing `action` would add
    /// to the graph **as it stands right now** — without mutating anything.
    ///
    /// Every [`RewriteAction`] arm builds a node shape of known structure
    /// (`Distribute`'s two products and a sum, `Instantiate`'s template,
    /// `Differentiate`'s one chain-rule step, …), and the e-graph is
    /// hash-consed, so the e-classes an application would add are exactly
    /// the shape's nodes that are not already in the memo. This walks that
    /// same shape — literally the same code [`EGraph::apply_action`] runs,
    /// via [`NodeSink`] — against a [`GrowthPredictor`] that probes the memo
    /// exactly as [`EGraph::add`] does on a hit, instead of inserting into
    /// it.
    ///
    /// Because the e-graph's raw class-slot count and its raw e-node count
    /// grow in lockstep (see the [`growth`](super::growth) module doc),
    /// "e-classes added" and "e-nodes added" are the same number here, which
    /// is why this returns one `usize` rather than a pair.
    ///
    /// **Only exact against the graph this is called on.** If several
    /// actions found in the same scan (before any of them commit) would
    /// build overlapping nodes, each one's prediction is exact in isolation
    /// but their sum over-counts the batch's real total — safely: the sum is
    /// still an upper bound on real growth, never an under-count, so a
    /// caller budgeting against it (the runner's `sweep_rule`) never
    /// overshoots. Pinned exact per single application — the sense that
    /// matters for a budget decided one application at a time — by the
    /// `assert_eq!` in the runner's `apply_measured`, which calls this
    /// immediately before executing the same action for real.
    pub(crate) fn predicted_growth(&self, action: &RewriteAction) -> usize {
        match action {
            RewriteAction::Union(_) => 0,
            RewriteAction::Create(node) => self.predict(|s| {
                s.make(*node);
            }),
            RewriteAction::Instantiate {
                template,
                entry,
                bindings,
            } => self.predict(|s| {
                instantiate_template(s, template.0.entry_at(*entry), bindings);
            }),
            RewriteAction::Distribute {
                outer,
                inner,
                a,
                b,
                c,
            } => self.predict(|s| {
                distribute_shape(
                    s,
                    Distributive {
                        outer: *outer,
                        inner: *inner,
                    },
                    [*a, *b, *c],
                );
            }),
            RewriteAction::Factor {
                outer,
                inner,
                common,
                unique_l,
                unique_r,
            } => self.predict(|s| {
                factor_shape(
                    s,
                    Distributive {
                        outer: *outer,
                        inner: *inner,
                    },
                    [*common, *unique_l, *unique_r],
                );
            }),
            RewriteAction::Canonicalize {
                target,
                inverse,
                a,
                b,
            } => self.predict(|s| {
                canonicalize_shape(s, *target, *inverse, *a, *b);
            }),
            RewriteAction::Associate { op, a, b, c } => self.predict(|s| {
                associate_shape(s, *op, *a, *b, *c);
            }),
            RewriteAction::ReverseAssociate { op, a, b, c } => self.predict(|s| {
                reverse_associate_shape(s, *op, *a, *b, *c);
            }),
            RewriteAction::OddParity { func, inner } => self.predict(|s| {
                odd_parity_shape(s, *func, *inner);
            }),
            RewriteAction::AngleAddition {
                term1_op1,
                term1_op2,
                term2_op1,
                term2_op2,
                term2_sign,
                a,
                b,
            } => self.predict(|s| {
                angle_addition_shape(
                    s,
                    AngleTerms {
                        term1: Product {
                            on_a: *term1_op1,
                            on_b: *term1_op2,
                        },
                        term2: Product {
                            on_a: *term2_op1,
                            on_b: *term2_op2,
                        },
                        term2_sign: *term2_sign,
                    },
                    [*a, *b],
                );
            }),
            RewriteAction::Homomorphism {
                func,
                target_op,
                a,
                b,
            } => self.predict(|s| {
                homomorphism_shape(s, *func, *target_op, *a, *b);
            }),
            RewriteAction::PowerCombine { base, exp_a, exp_b } => self.predict(|s| {
                power_combine_shape(s, *base, *exp_a, *exp_b);
            }),
            RewriteAction::ReverseAngleAddition { trig_op, a, b } => self.predict(|s| {
                reverse_angle_addition_shape(s, *trig_op, *a, *b);
            }),
            RewriteAction::HalfAngleProduct { x } => self.predict(|s| {
                half_angle_product_shape(s, *x);
            }),
            RewriteAction::Doubling { a } => self.predict(|s| {
                doubling_shape(s, *a);
            }),
            RewriteAction::Halving { a } => self.predict(|s| {
                halving_shape(s, *a);
            }),
            RewriteAction::PowerRecurrence { base, exponent } => self.predict(|s| {
                power_recurrence_shape(s, *base, *exponent);
            }),
            RewriteAction::LogPower {
                log_op,
                base,
                exponent,
            } => self.predict(|s| {
                log_power_shape(s, *log_op, *base, *exponent);
            }),
            RewriteAction::ExpandSquare { a, b } => self.predict(|s| {
                expand_square_shape(s, *a, *b);
            }),
            RewriteAction::DiffOfSquares { a, b } => self.predict(|s| {
                diff_of_squares_shape(s, *a, *b);
            }),
            RewriteAction::Differentiate { inner, var } => self.predict(|s| {
                derivative_shape(s, inner, *var);
            }),
            RewriteAction::Plan(plan) => self.predict(|s| {
                plan_shape(s, plan);
            }),
        }
    }

    /// Run `build` against a fresh [`GrowthPredictor`] over this graph and
    /// report how many e-classes it would have minted. See
    /// [`EGraph::predicted_growth`].
    fn predict(&self, build: impl FnOnce(&mut GrowthPredictor<'_>)) -> usize {
        let mut sink = GrowthPredictor {
            egraph: self,
            pending: HashMap::default(),
        };
        build(&mut sink);
        sink.pending.len()
    }

    /// Apply a rewrite action and return 1 if a union was made, 0 otherwise.
    ///
    /// Internal executor. Rule-driven callers must go through the runner's
    /// `commit` so the journal's attribution stays correct; this function
    /// itself has no notion of "which rule" — it only knows how to execute
    /// the `RewriteAction` variants.
    fn apply_action(&mut self, class_id: EClassId, action: RewriteAction) -> usize {
        match action {
            RewriteAction::Union(target_id) => self.union_counted(class_id, target_id),
            RewriteAction::Instantiate {
                template,
                entry,
                bindings,
            } => {
                let node = template.0.entry_at(entry);
                let result_id = instantiate_template(self, node, &bindings);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Create(new_node) => {
                let new_id = self.add(new_node);
                self.union_counted(class_id, new_id)
            }
            RewriteAction::Distribute {
                outer,
                inner,
                a,
                b,
                c,
            } => {
                let result_id = distribute_shape(self, Distributive { outer, inner }, [a, b, c]);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Factor {
                outer,
                inner,
                common,
                unique_l,
                unique_r,
            } => {
                let result_id = factor_shape(
                    self,
                    Distributive { outer, inner },
                    [common, unique_l, unique_r],
                );
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Canonicalize {
                target,
                inverse,
                a,
                b,
            } => {
                let result_id = canonicalize_shape(self, target, inverse, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Associate { op, a, b, c } => {
                let result_id = associate_shape(self, op, a, b, c);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ReverseAssociate { op, a, b, c } => {
                let result_id = reverse_associate_shape(self, op, a, b, c);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::OddParity { func, inner } => {
                let result_id = odd_parity_shape(self, func, inner);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::AngleAddition {
                term1_op1,
                term1_op2,
                term2_op1,
                term2_op2,
                term2_sign,
                a,
                b,
            } => {
                let terms = AngleTerms {
                    term1: Product {
                        on_a: term1_op1,
                        on_b: term1_op2,
                    },
                    term2: Product {
                        on_a: term2_op1,
                        on_b: term2_op2,
                    },
                    term2_sign,
                };
                let result_id = angle_addition_shape(self, terms, [a, b]);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Homomorphism {
                func,
                target_op,
                a,
                b,
            } => {
                let result_id = homomorphism_shape(self, func, target_op, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::PowerCombine { base, exp_a, exp_b } => {
                let result_id = power_combine_shape(self, base, exp_a, exp_b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ReverseAngleAddition { trig_op, a, b } => {
                let result_id = reverse_angle_addition_shape(self, trig_op, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::HalfAngleProduct { x } => {
                let result_id = half_angle_product_shape(self, x);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Doubling { a } => {
                let result_id = doubling_shape(self, a);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Halving { a } => {
                let result_id = halving_shape(self, a);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::PowerRecurrence { base, exponent } => {
                let result_id = power_recurrence_shape(self, base, exponent);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::LogPower {
                log_op,
                base,
                exponent,
            } => {
                let result_id = log_power_shape(self, log_op, base, exponent);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::ExpandSquare { a, b } => {
                let result_id = expand_square_shape(self, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::DiffOfSquares { a, b } => {
                let result_id = diff_of_squares_shape(self, a, b);
                self.union_counted(class_id, result_id)
            }
            RewriteAction::Differentiate { inner, var } => {
                let deriv_id = derivative_shape(self, &inner, var);
                self.union_counted(class_id, deriv_id)
            }
            RewriteAction::Plan(plan) => {
                let built = plan_shape(self, &plan);
                self.union_counted(class_id, built)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RewriteAction shapes: one definition per action, run twice
// ---------------------------------------------------------------------------
//
// Every `RewriteAction` (besides `Union`, which builds nothing) constructs a
// node shape of fixed structure and then unions the result with the matched
// class. `apply_action` needs to build that shape for real; `predicted_growth`
// needs to know how many of its nodes are new, without building anything.
// Rather than a second, hand-copied spelling of each shape for the predictor
// (exactly the drift `RewriteAction::Instantiate`'s own doc warns about —
// "two spellings of one shape"), every shape below is written once, generic
// over where its nodes land: a [`NodeSink`].

/// Materializes one [`ENode`], returning the e-class that holds it.
///
/// [`EGraph`] is a sink that actually inserts (`add`); [`GrowthPredictor`] is
/// a sink that only probes the memo. Every `RewriteAction`'s node-building
/// shape (below) is written once against this trait, so committing an
/// action for real and predicting how many e-classes it would add
/// ([`EGraph::predicted_growth`]) run the exact same code.
trait NodeSink {
    fn make(&mut self, node: ENode) -> EClassId;

    /// The graph the nodes land in, for the facts a shape is conditioned on.
    fn graph(&self) -> &EGraph;
}

impl NodeSink for EGraph {
    fn make(&mut self, node: ENode) -> EClassId {
        self.add(node)
    }

    fn graph(&self) -> &EGraph {
        self
    }
}

/// A [`NodeSink`] that predicts what [`EGraph::add`] would do without
/// mutating the graph: canonicalize, then check the graph's real memo first
/// and a local `pending` table second — the nodes this same prediction has
/// already "created" earlier in the same shape walk, keyed and deduplicated
/// the identical way the real memo would (e.g. `ExpandSquare`'s `a²` and
/// `b²` degenerate to one shared node when `a == b`, and this catches that
/// exactly as a real `add` would).
///
/// `pending`'s virtual ids are minted the same way [`EGraph::add`] mints
/// real ones — sequentially from `egraph.classes.len()` — so a shape that
/// references its own earlier output builds the identical composite node a
/// real run would look up.
struct GrowthPredictor<'a> {
    egraph: &'a EGraph,
    pending: HashMap<ENode, EClassId>,
}

impl GrowthPredictor<'_> {
    /// [`EGraph::canonical_node`], except a child may be a `pending` id
    /// — an id this same prediction minted for a node earlier in the same
    /// shape walk, one that does not exist in the real graph
    /// (`id.index() >= egraph.classes.len()`) and so is not in `egraph.parent`
    /// at all. `EGraph::find` indexes `parent` directly, so calling it on
    /// such an id is out of bounds rather than merely wrong. A pending id
    /// needs no lookup regardless: nothing has unioned a class that does
    /// not yet exist in the real graph, so it is already its own canonical
    /// form. Only a real child is resolved through the real union-find.
    fn canonicalize(&self, node: &mut ENode) {
        for child in node.children_slice_mut() {
            if child.index() < self.egraph.classes.len() {
                *child = self.egraph.find(*child);
            }
        }
    }
}

impl NodeSink for GrowthPredictor<'_> {
    fn make(&mut self, mut node: ENode) -> EClassId {
        self.canonicalize(&mut node);
        if let Some(&id) = self.egraph.memo.get(&node) {
            return self.egraph.find(id);
        }
        if let Some(&id) = self.pending.get(&node) {
            return id;
        }
        let id = EClassId::from_index(self.egraph.classes.len() + self.pending.len());
        self.pending.insert(node, id);
        id
    }

    fn graph(&self) -> &EGraph {
        self.egraph
    }
}

/// The two operators of a distributive law: `outer` is the root of the term
/// the rule matched and `inner` the operator beneath it. `Distribute` matches
/// `*` over `+` and `Factor` matches `+` over `*`, so the roles are fixed by
/// the match, not by which operator is which.
#[derive(Clone, Copy)]
struct Distributive {
    outer: OpKind,
    inner: OpKind,
}

/// A rebuilt buffer is shrunk when its unused capacity exceeds `1 /
/// REBUILT_SLACK_DIVISOR` of its length.
const REBUILT_SLACK_DIVISOR: usize = 2;

/// Put a rebuilt buffer back into a class slot. A slot that is still empty
/// takes the buffer as is; one a mid-rebuild `union` already appended onto
/// keeps those entries and gains the rebuilt ones after them.
fn restore_after_rebuild<T: Copy>(slot: &mut Vec<T>, mut rebuilt: Vec<T>) {
    if slot.is_empty() {
        // The buffer was grown by pushes and keeps that slack; a class lives
        // for the whole saturation, so slack is peak memory. Only a buffer
        // with a large share unused is worth a realloc to give it back.
        if rebuilt.capacity() - rebuilt.len() > rebuilt.len() / REBUILT_SLACK_DIVISOR {
            rebuilt.shrink_to_fit();
        }
        *slot = rebuilt;
        return;
    }
    slot.extend(rebuilt);
}

/// `Distribute`: `A * (B + C) -> A*B + A*C`, with `ops` = (outer `*`, inner
/// `+`) and `operands` = `[A, B, C]`.
fn distribute_shape<S: NodeSink>(
    sink: &mut S,
    ops: Distributive,
    operands: [EClassId; 3],
) -> EClassId {
    let [a, b, c] = operands;
    let ab_id = sink.make(ENode::apply(ops.outer, &[a, b]));
    let ac_id = sink.make(ENode::apply(ops.outer, &[a, c]));
    sink.make(ENode::apply(ops.inner, &[ab_id, ac_id]))
}

/// `Factor`: `A*B + A*C -> A * (B + C)`, with `ops` = (outer `+`, inner `*`)
/// and `operands` = `[A, B, C]`.
fn factor_shape<S: NodeSink>(sink: &mut S, ops: Distributive, operands: [EClassId; 3]) -> EClassId {
    let [common, unique_l, unique_r] = operands;
    let sum_id = sink.make(ENode::apply(ops.outer, &[unique_l, unique_r]));
    sink.make(ENode::apply(ops.inner, &[common, sum_id]))
}

/// `Canonicalize`: `Sub(a,b) -> Add(a, Neg(b))`.
fn canonicalize_shape<S: NodeSink>(
    sink: &mut S,
    target: OpKind,
    inverse: OpKind,
    a: EClassId,
    b: EClassId,
) -> EClassId {
    let inv_id = sink.make(ENode::apply(inverse, &[b]));
    sink.make(ENode::apply(target, &[a, inv_id]))
}

/// `Associate`: `(a op b) op c -> a op (b op c)`.
fn associate_shape<S: NodeSink>(
    sink: &mut S,
    op: OpKind,
    a: EClassId,
    b: EClassId,
    c: EClassId,
) -> EClassId {
    let bc_id = sink.make(ENode::apply(op, &[b, c]));
    sink.make(ENode::apply(op, &[a, bc_id]))
}

/// `ReverseAssociate`: `a op (b op c) -> (a op b) op c`.
fn reverse_associate_shape<S: NodeSink>(
    sink: &mut S,
    op: OpKind,
    a: EClassId,
    b: EClassId,
    c: EClassId,
) -> EClassId {
    let ab_id = sink.make(ENode::apply(op, &[a, b]));
    sink.make(ENode::apply(op, &[ab_id, c]))
}

/// `OddParity`: `Op(neg(x)) -> neg(Op(x))`.
fn odd_parity_shape<S: NodeSink>(sink: &mut S, func: OpKind, inner: EClassId) -> EClassId {
    let func_id = sink.make(ENode::apply(func, &[inner]));
    sink.make(ENode::apply(OpKind::Neg, &[func_id]))
}

/// `on_a(a) * on_b(b)`: one term of an angle-addition identity.
#[derive(Clone, Copy)]
struct Product {
    on_a: OpKind,
    on_b: OpKind,
}

/// The shape of an angle-addition identity: two products of single-argument
/// functions, joined by `term2_sign`.
#[derive(Clone, Copy)]
struct AngleTerms {
    term1: Product,
    term2: Product,
    term2_sign: crate::math::trig::Sign,
}

/// `AngleAddition`: `sin(a+b) -> sin(a)cos(b) + cos(a)sin(b)` (or the `cos`
/// case, sign-selected).
fn angle_addition_shape<S: NodeSink>(
    sink: &mut S,
    terms: AngleTerms,
    angles: [EClassId; 2],
) -> EClassId {
    use crate::math::trig::Sign;

    let [a, b] = angles;
    let AngleTerms {
        term1,
        term2,
        term2_sign,
    } = terms;

    let t1_left_id = sink.make(ENode::apply(term1.on_a, &[a]));
    let t1_right_id = sink.make(ENode::apply(term1.on_b, &[b]));
    let term1_id = sink.make(ENode::apply(OpKind::Mul, &[t1_left_id, t1_right_id]));

    let t2_left_id = sink.make(ENode::apply(term2.on_a, &[a]));
    let t2_right_id = sink.make(ENode::apply(term2.on_b, &[b]));
    let term2_id = sink.make(ENode::apply(OpKind::Mul, &[t2_left_id, t2_right_id]));

    match term2_sign {
        Sign::Plus => sink.make(ENode::apply(OpKind::Add, &[term1_id, term2_id])),
        Sign::Minus => sink.make(ENode::apply(OpKind::Sub, &[term1_id, term2_id])),
    }
}

/// `Homomorphism`: `f(a ⊕ b) -> f(a) ⊗ f(b)`.
fn homomorphism_shape<S: NodeSink>(
    sink: &mut S,
    func: OpKind,
    target_op: OpKind,
    a: EClassId,
    b: EClassId,
) -> EClassId {
    let func_a_id = sink.make(ENode::apply(func, &[a]));
    let func_b_id = sink.make(ENode::apply(func, &[b]));
    sink.make(ENode::apply(target_op, &[func_a_id, func_b_id]))
}

/// `PowerCombine`: `x^a * x^b -> x^(a+b)`.
fn power_combine_shape<S: NodeSink>(
    sink: &mut S,
    base: EClassId,
    exp_a: EClassId,
    exp_b: EClassId,
) -> EClassId {
    let sum_id = sink.make(ENode::apply(OpKind::Add, &[exp_a, exp_b]));
    sink.make(ENode::apply(OpKind::Pow, &[base, sum_id]))
}

/// `ReverseAngleAddition`: `sin(a)cos(b) + cos(a)sin(b) -> sin(a + b)` (or
/// the `cos` case).
fn reverse_angle_addition_shape<S: NodeSink>(
    sink: &mut S,
    trig_op: OpKind,
    a: EClassId,
    b: EClassId,
) -> EClassId {
    let sum_id = sink.make(ENode::apply(OpKind::Add, &[a, b]));
    sink.make(ENode::apply(trig_op, &[sum_id]))
}

/// `HalfAngleProduct`: `sin(x) * cos(x) -> sin(x + x) / 2`.
fn half_angle_product_shape<S: NodeSink>(sink: &mut S, x: EClassId) -> EClassId {
    let two_x_id = sink.make(ENode::apply(OpKind::Add, &[x, x]));
    let sin_2x_id = sink.make(ENode::apply(OpKind::Sin, &[two_x_id]));
    let two_id = sink.make(ENode::Const(2.0_f32.to_bits()));
    sink.make(ENode::apply(OpKind::Div, &[sin_2x_id, two_id]))
}

/// `Doubling`: `a + a -> 2 * a`.
fn doubling_shape<S: NodeSink>(sink: &mut S, a: EClassId) -> EClassId {
    let two_id = sink.make(ENode::Const(2.0_f32.to_bits()));
    sink.make(ENode::apply(OpKind::Mul, &[two_id, a]))
}

/// `Halving`: `2 * a -> a + a`.
fn halving_shape<S: NodeSink>(sink: &mut S, a: EClassId) -> EClassId {
    sink.make(ENode::apply(OpKind::Add, &[a, a]))
}

/// `PowerRecurrence`: `pow(x, n) -> x * pow(x, n-1)` for integer `n >= 3`.
fn power_recurrence_shape<S: NodeSink>(sink: &mut S, base: EClassId, exponent: i32) -> EClassId {
    let n_minus_1_id = sink.make(ENode::constant((exponent - 1) as f32));
    let pow_id = sink.make(ENode::apply(OpKind::Pow, &[base, n_minus_1_id]));
    sink.make(ENode::apply(OpKind::Mul, &[base, pow_id]))
}

/// `LogPower`: `log(pow(x, n)) -> n * log(x)`.
fn log_power_shape<S: NodeSink>(
    sink: &mut S,
    log_op: OpKind,
    base: EClassId,
    exponent: EClassId,
) -> EClassId {
    let log_x_id = sink.make(ENode::apply(log_op, &[base]));
    sink.make(ENode::apply(OpKind::Mul, &[exponent, log_x_id]))
}

/// `ExpandSquare`: `(a+b)² -> a² + 2ab + b²`.
fn expand_square_shape<S: NodeSink>(sink: &mut S, a: EClassId, b: EClassId) -> EClassId {
    let a2_id = sink.make(ENode::apply(OpKind::Mul, &[a, a]));
    let b2_id = sink.make(ENode::apply(OpKind::Mul, &[b, b]));
    let ab_id = sink.make(ENode::apply(OpKind::Mul, &[a, b]));
    let two_id = sink.make(ENode::constant(2.0));
    let two_ab_id = sink.make(ENode::apply(OpKind::Mul, &[two_id, ab_id]));
    let sum1_id = sink.make(ENode::apply(OpKind::Add, &[a2_id, two_ab_id]));
    sink.make(ENode::apply(OpKind::Add, &[sum1_id, b2_id]))
}

/// `DiffOfSquares`: `a² - b² -> (a+b)(a-b)`.
fn diff_of_squares_shape<S: NodeSink>(sink: &mut S, a: EClassId, b: EClassId) -> EClassId {
    let sum_id = sink.make(ENode::apply(OpKind::Add, &[a, b]));
    let diff_id = sink.make(ENode::apply(OpKind::Sub, &[a, b]));
    sink.make(ENode::apply(OpKind::Mul, &[sum_id, diff_id]))
}

/// Materialize `template`'s subtree at `id`, bottom-up, reading each
/// metavariable leaf from `bindings` rather than creating a node for it —
/// the same convention every hand-written multi-node shape above uses, just
/// driven by a runtime pattern instead of a fixed shape.
///
/// # Panics
///
/// On a `Param`/`Buffer` node (a rewrite RHS template must never contain
/// either), on an `OpKind` that `ops::template_may_name` refuses (a template
/// naming something no rule may name is a bug in the rule, not a runtime
/// condition), on a template node whose child count is not its op's arity
/// (`Operands::new`), or on a metavariable with no binding.
fn instantiate_template<S: NodeSink>(
    sink: &mut S,
    node: pixelflow_ir::Node<'_, pixelflow_ir::expr::ExprData>,
    bindings: &[EClassId],
) -> EClassId {
    use pixelflow_ir::expr::ExprData;

    match *node {
        ExprData::Var(mv) => {
            let mv = mv as usize;
            assert!(
                mv < bindings.len(),
                "instantiate_template: metavariable {mv} has no binding \
                 ({} supplied) — a TemplateRewrite construction bug, since \
                 apply() refuses to instantiate a binding it cannot fill",
                bindings.len()
            );
            bindings[mv]
        }
        ExprData::Const(bits) => sink.make(ENode::constant(f32::from_bits(bits))),
        ExprData::Param(p) => {
            panic!("instantiate_template: Param({p}) in a rewrite RHS template")
        }
        ExprData::Buffer(b) => {
            panic!(
                "instantiate_template: Buffer({}) in a rewrite RHS template",
                b.0
            )
        }
        ExprData::Uniform(u) => {
            panic!(
                "instantiate_template: Uniform({}) in a rewrite RHS template",
                u.0
            )
        }
        ExprData::Ref(k) => {
            panic!("instantiate_template: Ref({k:?}) in a rewrite RHS template")
        }
        ExprData::Reduce(fold) => {
            let body = node
                .children()
                .next()
                .expect("a Reduce template has its body as its one child");
            let body = instantiate_template(sink, body, bindings);
            sink.make(ENode::Reduce { fold, body })
        }
        ExprData::Op(kind) => {
            assert!(
                ops::template_may_name(kind),
                "instantiate_template: a template cannot name {kind:?}"
            );
            let children: Vec<EClassId> = node
                .children()
                .map(|c| instantiate_template(sink, c, bindings))
                .collect();
            sink.make(ENode::apply(kind, &children))
        }
    }
}

/// Build a [`Plan`](super::fold_rules::Plan) — every fold and integration
/// rule's right-hand side — and return the class of its root.
///
/// The plan arrives already computed because computing it needs to *read*
/// the graph (walking a body's classes to substitute a binder, recognizing an
/// integrand), which a [`NodeSink`] cannot do; the rule does that half and
/// this replays it, node by node in build order.
fn plan_shape<S: NodeSink>(sink: &mut S, plan: &super::fold_rules::Plan) -> EClassId {
    use super::fold_rules::{HeadNode, HeadRef};
    let mut planned: Vec<EClassId> = Vec::with_capacity(plan.nodes.len());
    let resolve = |r: HeadRef, planned: &[EClassId]| match r {
        HeadRef::Plan(i) => planned[i as usize],
        HeadRef::Class(c) => c,
    };
    for entry in &plan.nodes {
        let id = match entry {
            HeadNode::Const(bits) => sink.make(ENode::Const(*bits)),
            HeadNode::Op { op, children } => {
                let operands: Vec<EClassId> =
                    children.iter().map(|c| resolve(*c, &planned)).collect();
                sink.make(ENode::apply(*op, &operands))
            }
            HeadNode::Reduce { fold, body } => sink.make(ENode::Reduce {
                fold: *fold,
                body: resolve(*body, &planned),
            }),
            HeadNode::Shift {
                direction,
                count,
                operand,
            } => sink.make(ENode::Shift {
                direction: *direction,
                count: *count,
                operand: resolve(*operand, &planned),
            }),
        };
        planned.push(id);
    }
    resolve(plan.root, &planned)
}

/// Build the e-class of the derivative of `inner` with respect to variable
/// `var`, one chain-rule step deep. Sub-expressions are wrapped in fresh
/// `Dwrt` nodes so equality saturation continues the expansion; the leaf
/// cases (`Var`, `Const`) terminate it. Operators whose derivative is not
/// (yet) known reconstruct the original `Dwrt`, leaving it to survive
/// saturation as the jet fallback.
fn derivative_shape<S: NodeSink>(sink: &mut S, inner: &ENode, var: u8) -> EClassId {
    let (op, children) = match inner {
        ENode::Const(_) => return sink.make(ENode::constant(0.0)),
        ENode::Var(i) => {
            return sink.make(ENode::constant(if *i == var { 1.0 } else { 0.0 }));
        }
        // A Buffer leaf is not a value — it only ever appears as Gather's
        // first child, and no rewrite builds `Dwrt(buffer)`. Reaching one
        // here means the graph is malformed; fail loudly. (`Dwrt(gather)`
        // is the Op arm below: Gather has no derivative table entry, so it
        // reconstructs the Dwrt as the jet fallback.)
        ENode::Buffer(decl) => {
            panic!("derivative_shape: Dwrt applied to a Buffer leaf ({decl:?})")
        }
        // Invariant across the lattice: ∂u/∂x = 0, as for a constant.
        ENode::Uniform(_) => return sink.make(ENode::constant(0.0)),
        // Likewise ∂p/∂x = 0: a builder's scalar is one number for the
        // whole lattice, whichever number it turns out to be.
        ENode::Param(_) => return sink.make(ENode::constant(0.0)),
        // A unit that does not vary along the axis has derivative zero, as
        // a uniform does. One that does keeps its `Dwrt`: a name has no
        // structure to differentiate. The unit walk links every reference
        // a written `Dwrt` reaches before insertion, so the chain rule runs
        // here on every derivative the program wrote; a `Dwrt` over a unit
        // can only come of rewriting a class it shares, and one extraction
        // keeps is lowered by `lower_dwrt` on the linked body, after the
        // link, like any other `Dwrt` the rules did not reach.
        ENode::Ref { key } => {
            if !sink.graph().ref_variance(*key).depends_on(var) {
                return sink.make(ENode::constant(0.0));
            }
            let var_const = sink.make(ENode::constant(var as f32));
            let inner = sink.make(*inner);
            return sink.make(ENode::apply(OpKind::Dwrt, &[inner, var_const]));
        }
        ENode::Op { op, children } => (*op, *children),
        // `d(⊕_k f) = ⊕_k d(f)` is linearity, which holds for `Σ` and for
        // nothing else in the monoid set — `Π` wants the product rule, and
        // `min`/`max` are selections. It is a rule this table does not have,
        // so the `Dwrt` is reconstructed below and survives as the fallback,
        // exactly as `Gather` does. A shift has no derivative rule either
        // (it moves bits, which have no slope), and is kept the same way.
        ENode::Reduce { .. } | ENode::Shift { .. } => {
            let var_const = sink.make(ENode::constant(var as f32));
            let inner = sink.make(*inner);
            return sink.make(ENode::apply(OpKind::Dwrt, &[inner, var_const]));
        }
    };

    let var_const = sink.make(ENode::constant(var as f32));
    // d(child)/dvar as a fresh Dwrt node (saturation expands it later).
    let dwrt = |s: &mut S, c: EClassId| s.make(ENode::apply(OpKind::Dwrt, &[c, var_const]));
    let op2 = |s: &mut S, o: OpKind, a: EClassId, b: EClassId| s.make(ENode::apply(o, &[a, b]));
    let un = |s: &mut S, o: OpKind, a: EClassId| s.make(ENode::apply(o, &[a]));
    let cst = |s: &mut S, v: f32| s.make(ENode::constant(v));

    match op {
        // Linearity: D(a + b) = D(a) + D(b); D(a - b) = D(a) - D(b).
        OpKind::Add | OpKind::Sub => {
            let da = dwrt(sink, children[0]);
            let db = dwrt(sink, children[1]);
            op2(sink, op, da, db)
        }
        OpKind::Neg => {
            let da = dwrt(sink, children[0]);
            un(sink, OpKind::Neg, da)
        }
        // Product rule: D(a*b) = D(a)*b + a*D(b).
        OpKind::Mul => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let t1 = op2(sink, OpKind::Mul, da, b);
            let t2 = op2(sink, OpKind::Mul, a, db);
            op2(sink, OpKind::Add, t1, t2)
        }
        // Fused multiply-add a*b + c: D = D(a)*b + a*D(b) + D(c).
        OpKind::MulAdd => {
            let (a, b, c) = (children[0], children[1], children[2]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let dc = dwrt(sink, c);
            let t1 = op2(sink, OpKind::Mul, da, b);
            let t2 = op2(sink, OpKind::Mul, a, db);
            let prod = op2(sink, OpKind::Add, t1, t2);
            op2(sink, OpKind::Add, prod, dc)
        }
        // Quotient rule: D(a/b) = (D(a)*b - a*D(b)) / (b*b).
        OpKind::Div => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let t1 = op2(sink, OpKind::Mul, da, b);
            let t2 = op2(sink, OpKind::Mul, a, db);
            let num = op2(sink, OpKind::Sub, t1, t2);
            let den = op2(sink, OpKind::Mul, b, b);
            op2(sink, OpKind::Div, num, den)
        }
        // d(sqrt u) = 0.5 * rsqrt(u) * u'.
        OpKind::Sqrt => {
            let u = children[0];
            let du = dwrt(sink, u);
            let half = cst(sink, 0.5);
            let rs = un(sink, OpKind::Rsqrt, u);
            let factor = op2(sink, OpKind::Mul, half, rs);
            op2(sink, OpKind::Mul, factor, du)
        }
        // d(recip u) = -u' / (u*u).
        OpKind::Recip => {
            let u = children[0];
            let du = dwrt(sink, u);
            let ndu = un(sink, OpKind::Neg, du);
            let u2 = op2(sink, OpKind::Mul, u, u);
            op2(sink, OpKind::Div, ndu, u2)
        }
        // d(|u|) = (u / |u|) * u'.
        OpKind::Abs => {
            let u = children[0];
            let du = dwrt(sink, u);
            let au = un(sink, OpKind::Abs, u);
            let sign = op2(sink, OpKind::Div, u, au);
            op2(sink, OpKind::Mul, sign, du)
        }
        // d(rsqrt u) = -0.5 * rsqrt(u) * recip(u) * u'.
        OpKind::Rsqrt => {
            let u = children[0];
            let du = dwrt(sink, u);
            let neg_half = cst(sink, -0.5);
            let rs = un(sink, OpKind::Rsqrt, u);
            let rc = un(sink, OpKind::Recip, u);
            let t = op2(sink, OpKind::Mul, rs, rc);
            let factor = op2(sink, OpKind::Mul, neg_half, t);
            op2(sink, OpKind::Mul, factor, du)
        }
        // Piecewise: derivative of the branch the primal takes. Masks and
        // tie behavior mirror Jet2 (and the runtime `lower_dwrt` pass).
        OpKind::Min => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let mask = op2(sink, OpKind::Lt, a, b);
            sink.make(ENode::apply(OpKind::If, &[mask, da, db]))
        }
        OpKind::Max => {
            let (a, b) = (children[0], children[1]);
            let da = dwrt(sink, a);
            let db = dwrt(sink, b);
            let mask = op2(sink, OpKind::Gt, a, b);
            sink.make(ENode::apply(OpKind::If, &[mask, da, db]))
        }
        // Blend the branch derivatives on the primal mask; the mask itself
        // is not differentiated.
        OpKind::If => {
            let (m, t, f) = (children[0], children[1], children[2]);
            let dt = dwrt(sink, t);
            let df = dwrt(sink, f);
            sink.make(ENode::apply(OpKind::If, &[m, dt, df]))
        }
        // Masks and rounding are step functions: zero almost everywhere.
        OpKind::Lt
        | OpKind::Le
        | OpKind::Gt
        | OpKind::Ge
        | OpKind::Eq
        | OpKind::Ne
        | OpKind::Floor
        | OpKind::Ceil
        | OpKind::Round => sink.make(ENode::constant(0.0)),
        // d(sin u) = cos(u) * u'.
        OpKind::Sin => {
            let u = children[0];
            let du = dwrt(sink, u);
            let c = un(sink, OpKind::Cos, u);
            op2(sink, OpKind::Mul, c, du)
        }
        // d(cos u) = -sin(u) * u'.
        OpKind::Cos => {
            let u = children[0];
            let du = dwrt(sink, u);
            let s = un(sink, OpKind::Sin, u);
            let ns = un(sink, OpKind::Neg, s);
            op2(sink, OpKind::Mul, ns, du)
        }
        // d(tan u) = u' / cos(u)^2.
        OpKind::Tan => {
            let u = children[0];
            let du = dwrt(sink, u);
            let c = un(sink, OpKind::Cos, u);
            let c2 = op2(sink, OpKind::Mul, c, c);
            op2(sink, OpKind::Div, du, c2)
        }
        // d(atan u) = u' / (1 + u*u).
        OpKind::Atan => {
            let u = children[0];
            let du = dwrt(sink, u);
            let one = cst(sink, 1.0);
            let u2 = op2(sink, OpKind::Mul, u, u);
            let den = op2(sink, OpKind::Add, one, u2);
            op2(sink, OpKind::Div, du, den)
        }
        // d(asin u) = u' / sqrt(1 - u*u).
        OpKind::Asin => {
            let u = children[0];
            let du = dwrt(sink, u);
            let one = cst(sink, 1.0);
            let u2 = op2(sink, OpKind::Mul, u, u);
            let diff = op2(sink, OpKind::Sub, one, u2);
            let s = un(sink, OpKind::Sqrt, diff);
            op2(sink, OpKind::Div, du, s)
        }
        // d(acos u) = -u' / sqrt(1 - u*u).
        OpKind::Acos => {
            let u = children[0];
            let du = dwrt(sink, u);
            let one = cst(sink, 1.0);
            let u2 = op2(sink, OpKind::Mul, u, u);
            let diff = op2(sink, OpKind::Sub, one, u2);
            let s = un(sink, OpKind::Sqrt, diff);
            let q = op2(sink, OpKind::Div, du, s);
            un(sink, OpKind::Neg, q)
        }
        // d(exp u) = exp(u) * u'.
        OpKind::Exp => {
            let u = children[0];
            let du = dwrt(sink, u);
            let e = un(sink, OpKind::Exp, u);
            op2(sink, OpKind::Mul, e, du)
        }
        // d(2^u) = 2^u * ln2 * u'.
        OpKind::Exp2 => {
            let u = children[0];
            let du = dwrt(sink, u);
            let e = un(sink, OpKind::Exp2, u);
            let ln2 = cst(sink, core::f32::consts::LN_2);
            let factor = op2(sink, OpKind::Mul, e, ln2);
            op2(sink, OpKind::Mul, factor, du)
        }
        // d(ln u) = u' / u.
        OpKind::Ln => {
            let u = children[0];
            let du = dwrt(sink, u);
            op2(sink, OpKind::Div, du, u)
        }
        // d(log2 u) = u' / (u * ln2).
        OpKind::Log2 => {
            let u = children[0];
            let du = dwrt(sink, u);
            let ln2 = cst(sink, core::f32::consts::LN_2);
            let den = op2(sink, OpKind::Mul, u, ln2);
            op2(sink, OpKind::Div, du, den)
        }
        // d(log10 u) = u' / (u * ln10).
        OpKind::Log10 => {
            let u = children[0];
            let du = dwrt(sink, u);
            let ln10 = cst(sink, core::f32::consts::LN_10);
            let den = op2(sink, OpKind::Mul, u, ln10);
            op2(sink, OpKind::Div, du, den)
        }
        // d(atan2(y, x)) = (x*y' - y*x') / (x² + y²).
        OpKind::Atan2 => {
            let (y, x) = (children[0], children[1]);
            let dy = dwrt(sink, y);
            let dx = dwrt(sink, x);
            let t1 = op2(sink, OpKind::Mul, x, dy);
            let t2 = op2(sink, OpKind::Mul, y, dx);
            let num = op2(sink, OpKind::Sub, t1, t2);
            let x2 = op2(sink, OpKind::Mul, x, x);
            let y2 = op2(sink, OpKind::Mul, y, y);
            let den = op2(sink, OpKind::Add, x2, y2);
            op2(sink, OpKind::Div, num, den)
        }
        // d(f^g) = f^g * (g'*ln f + g*f'/f)  (Jet2's rule).
        OpKind::Pow => {
            let (f, g) = (children[0], children[1]);
            let df = dwrt(sink, f);
            let dg = dwrt(sink, g);
            let lnf = un(sink, OpKind::Ln, f);
            let t1 = op2(sink, OpKind::Mul, dg, lnf);
            let g_over_f = op2(sink, OpKind::Div, g, f);
            let t2 = op2(sink, OpKind::Mul, g_over_f, df);
            let inner = op2(sink, OpKind::Add, t1, t2);
            let p = op2(sink, OpKind::Pow, f, g);
            op2(sink, OpKind::Mul, p, inner)
        }
        // Unknown derivative: reconstruct the Dwrt and let it survive.
        _ => {
            let reconstructed = sink.make(*inner);
            dwrt(sink, reconstructed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::cost::CostModel;
    use crate::egraph::extract::extract_dag;
    use crate::egraph::provenance::{ApplicationId, Origin};
    use crate::egraph::rewrite::Rewrite;
    use crate::egraph::rules::RuleSet;
    use crate::egraph::saturate::SaturationConfig;
    use crate::math::algebra::{
        AddNeg, Annihilator, Cancellation, Canonicalize, Commutative, Distributive, Identity,
        InverseAnnihilation, Involution, MulRecip,
    };

    /// The canonical copy of a node names the surviving class of every operand
    /// the node has, whichever variant holds it: an operation's operands, a
    /// shift's operand and a fold's body. The graph is not rebuilt between
    /// the union and the reads, which is the state an observer can find it in.
    #[test]
    fn a_canonical_node_reads_every_operand_through_find() {
        use pixelflow_ir::ShiftCount;
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut eg = EGraph::new();
        let kept = eg.add(ENode::Var(0));
        let merged = eg.add(ENode::Var(1));
        let survivor = eg.union(kept, merged);
        let fold = Fold::new(
            Monoid::SUM,
            Binder::from_slot(0).expect("a live binder"),
            0..4,
        );
        let shift = |operand| ENode::Shift {
            direction: crate::egraph::ShiftDirection::Right,
            count: ShiftCount::from_literal(8.0).expect("in range"),
            operand,
        };
        let stale_and_canonical = [
            (
                ENode::apply(OpKind::Add, &[merged, kept]),
                ENode::apply(OpKind::Add, &[survivor, survivor]),
            ),
            (shift(merged), shift(survivor)),
            (
                ENode::Reduce { fold, body: merged },
                ENode::Reduce {
                    fold,
                    body: survivor,
                },
            ),
        ];
        for (stale, canonical) in stale_and_canonical {
            assert_eq!(eg.canonical_node(stale), canonical);
        }
    }

    /// Regression test for a `rebuild_budgeted` bug: when canonicalizing the
    /// worklist item `id`'s nodes triggers a union whose surviving parent is
    /// `id` itself, `union()`'s `extend()` appends the merged-in class's
    /// nodes directly onto `classes[id.index()].nodes` (which `rebuild_budgeted`
    /// had just emptied via `mem::take`). The old code then did
    /// `classes[id.index()].nodes = new_nodes`, an outright *assignment* that
    /// clobbered whatever `union()` had just appended — silently dropping
    /// nodes. The fix extends instead of assigning.
    ///
    /// Trigger recipe:
    /// - class 3 (`nc = Neg(c)`, c = class 2) is the worklist item being
    ///   rebuilt (`id = 3`).
    /// - Before nc's rebuild, `union(b, c)` merges class 2 into class 1
    ///   (1 < 2), so canonicalizing `Neg(c)` during nc's rebuild turns it
    ///   into `Neg(b)`.
    /// - `Neg(b)` is already memoized as class 4 (`nb = Neg(b)`, created
    ///   after nc so it has a strictly higher id: 4 > 3).
    /// - Because `id (3) < existing (4)`, `union(3, 4)` picks **3** as the
    ///   surviving parent — reproducing the exact "current worklist item
    ///   survives the merge" case that dropped nodes.
    /// - Class 4 (`nb`) must still hold its node at the moment of that
    ///   inner union, which it does (only class 3's `.nodes` was drained).
    #[test]
    fn rebuild_budgeted_does_not_drop_nodes_when_current_class_survives_union() {
        let mut eg = EGraph::new();

        let _a = eg.add(ENode::Var(0)); // class 0 (unused, keeps ids spaced out)
        let b = eg.add(ENode::Var(1)); // class 1
        let c = eg.add(ENode::Var(2)); // class 2
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 3: memo Neg([2]) -> 3
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 4: memo Neg([1]) -> 4
        let d = eg.add(ENode::Var(3)); // class 5, dummy used only to enqueue `nc`
        let marker = eg.add(ENode::Var(9)); // class 6: a node unique to nb's class

        assert_eq!(nc.index(), 3, "test assumes nc is class 3");
        assert_eq!(nb.index(), 4, "test assumes nb is class 4");

        // Enqueue class 3 (nc) on the worklist without disturbing its node
        // list content: nc (3) < d (5), so nc survives as parent and simply
        // gains an extra Var(3) node — it does not lose anything.
        eg.union(nc, d);
        assert_eq!(eg.pending_rebuilds(), 1);

        // Give nb's class a node with no structural twin anywhere else
        // (`Var(9)`), so its loss is directly observable. nb (4) < marker
        // (6), so nb survives as parent and gains `Var(9)`.
        eg.union(nb, marker);

        // Now merge c into b. b (1) < c (2), so b survives; c's node list
        // (just `Var(2)`) is merged in. This does NOT yet touch nc/nb.
        eg.union(b, c);

        // Rebuild exactly one worklist item. LIFO worklist: the most
        // recently pushed parent (class 1, from union(b,c)) pops first.
        // Process it, then process class 3 (nc) next so its rebuild is the
        // one that triggers the id-survives-merge collision with nb (4).
        //
        // Drain the whole worklist via rebuild() (equivalent to
        // rebuild_budgeted(usize::MAX)) so ordering doesn't need to be
        // hand-tracked — the bug reproduces regardless of pop order, as
        // long as nc's rebuild eventually runs after b/c are merged.
        eg.rebuild();

        // The critical assertion: nb's class (now merged into nc's class,
        // since nc=3 < nb=4 survives) must still contain BOTH nodes that
        // were live before the merge: nc's own `Neg([canonical c])` and
        // nb's `Neg([b])` (same canonical shape, but a distinct ENode
        // instance in the vec before dedup would also be acceptable — the
        // point is the vec must not have been clobbered to something
        // that lost nb's contribution entirely).
        let surviving = eg.find(nc);
        assert_eq!(
            surviving,
            eg.find(nb),
            "nc and nb should have been unioned via the canonicalization collision"
        );

        let nodes = eg.nodes(surviving);
        assert!(
            !nodes.is_empty(),
            "rebuild_budgeted must not silently drop all nodes from the surviving class"
        );

        // The dummy Var(3) node pushed onto nc's class via `union(nc, d)`
        // earlier must have survived.
        assert!(
            nodes.iter().any(|n| matches!(n, ENode::Var(3))),
            "expected Var(3) (pushed before the id-survives merge) to still be present; \
             rebuild_budgeted's overwrite bug would have dropped it"
        );

        // The critical, structurally-unique marker node from nb's class
        // (`Var(9)`) must have survived being merged into nc's class via
        // union()'s extend(). This is exactly the data the overwrite bug
        // silently discarded: union() appended it to
        // `classes[id.index()].nodes` mid-loop, and the old
        // `self.classes[id.index()].nodes = new_nodes` assignment clobbered
        // it because `new_nodes` was built from nc's pre-union node list,
        // which never contained `Var(9)`.
        assert!(
            nodes.iter().any(|n| matches!(n, ENode::Var(9))),
            "expected Var(9) (nb's unique marker, merged in via union()'s extend) \
             to still be present; rebuild_budgeted's overwrite bug drops nodes appended \
             by a mid-loop union() when the current worklist item survives as parent"
        );
    }

    /// The mirror of the test above, and the case its fix left open: when
    /// canonicalizing the worklist item `id`'s nodes triggers a union whose
    /// surviving parent is the OTHER class, `id` itself becomes
    /// non-canonical — and the write-back at the bottom of the loop targets
    /// `classes[id.index()]`, a slot `find` no longer routes to. Every node
    /// `id` still held is orphaned: `EGraph::nodes(id)` canonicalizes first,
    /// so nothing can ever read them again.
    ///
    /// Lost e-nodes are lost extraction alternatives (under-merging, not
    /// unsoundness — every remaining class still holds only provably-equal
    /// terms), but they are lost *silently*, which this project forbids.
    /// The fix writes back to `self.find(id)`, extending rather than
    /// assigning because `union` may already have moved nodes there.
    ///
    /// Trigger recipe — the same shape as the test above with the two `Neg`
    /// classes created in the opposite order, which is all it takes to flip
    /// which side `union`'s `min(a, b)` keeps:
    /// - `nb = Neg(b)` (class 3) is created BEFORE `nc = Neg(c)` (class 4),
    ///   so `memo[Neg([1])] -> 3` and `3 < 4`.
    /// - `union(nc, marker)` gives class 4 a structurally unique `Var(9)`
    ///   and enqueues it for rebuild.
    /// - `union(b, c)` merges class 2 into class 1, so canonicalizing
    ///   `Neg([2])` during class 4's rebuild rewrites it to `Neg([1])`.
    /// - `memo` maps `Neg([1])` to class 3, and `existing (3) < id (4)`, so
    ///   `union(4, 3)` keeps **3** — `id` is merged away mid-loop.
    /// - The write-back then appends `Var(9)` to class 4, which `find` now
    ///   routes past. `nodes(find(nc))` never sees it again.
    #[test]
    fn rebuild_budgeted_does_not_orphan_nodes_when_current_class_is_merged_away() {
        let mut eg = EGraph::new();

        let a = eg.add(ENode::Var(0)); // class 0 (keeps the ids spaced out)
        let b = eg.add(ENode::Var(1)); // class 1
        let c = eg.add(ENode::Var(2)); // class 2
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 3: memo Neg([1]) -> 3
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 4: memo Neg([2]) -> 4
        let marker = eg.add(ENode::Var(9)); // class 5: a node unique to nc's class

        assert_eq!(nb.index(), 3, "test assumes nb is class 3");
        assert_eq!(nc.index(), 4, "test assumes nc is class 4");

        // Give nc's class a node with no structural twin anywhere else
        // (`Var(9)`), so its loss is directly observable, and enqueue nc for
        // rebuild. nc (4) < marker (5), so nc survives as parent here.
        eg.union(nc, marker);
        assert_eq!(eg.pending_rebuilds(), 1);

        // Merge c into b. b (1) < c (2), so b survives. This is what makes
        // `Neg([2])` canonicalize to `Neg([1])` during nc's rebuild.
        eg.union(b, c);

        eg.rebuild();

        // nb (3) is the surviving parent: `union(4, 3)` keeps the lower id.
        let surviving = eg.find(nc);
        assert_eq!(
            surviving,
            eg.find(nb),
            "nc and nb should have been unioned via the canonicalization collision"
        );
        assert_eq!(
            surviving, nb,
            "test assumes `existing` (nb, class 3) is the surviving parent, \
             i.e. the `existing.0 < id.0` branch"
        );

        // The critical assertion: `Var(9)` lived in nc's class when nc was
        // merged away mid-loop, so it must still be reachable through the
        // surviving class. The orphaning bug writes it back to class 4,
        // which `find` routes past — `nodes()` canonicalizes, so it is
        // unreachable from every public entry point from then on.
        let nodes = eg.nodes(surviving);
        assert!(
            nodes.iter().any(|n| matches!(n, ENode::Var(9))),
            "expected Var(9) (nc's unique marker) to still be reachable via \
             nodes(find(nc)); the write-back targeted the merged-away slot \
             classes[{}] instead of classes[{}]. Got: {nodes:?}",
            nc.index(),
            surviving.index()
        );

        // Tags must stay zipped with nodes through the redirected write-back.
        assert_eq!(
            eg.nodes(surviving).len(),
            eg.tags(surviving).len(),
            "EClass.nodes and EClass.tags must never desync"
        );

        // Nothing else got orphaned on the way. Stated as the law the public
        // API actually owes a caller — **every id `add` ever handed out still
        // resolves to a class containing its node** — rather than by reading
        // the merged-away slot directly: a shadow copy is unreachable by
        // definition, so the only honest way to catch one is to check that
        // everything reachable is still there.
        let holds = |eg: &EGraph, id: EClassId, want: &ENode| -> bool {
            eg.nodes(eg.find(id)).iter().any(|n| n == want)
        };
        let neg_b = ENode::apply(OpKind::Neg, &[eg.find(b)]);
        for (id, want) in [
            (a, ENode::Var(0)),
            (b, ENode::Var(1)),
            (c, ENode::Var(2)),
            (marker, ENode::Var(9)),
            (nb, neg_b),
            (nc, neg_b),
        ] {
            assert!(
                holds(&eg, id, &want),
                "id {} no longer resolves to a class holding {want:?}",
                id.index()
            );
        }

        // And the graph must be at rest: a second rebuild is a no-op. The
        // orphaning bug leaves the worklist naming a class whose nodes went
        // somewhere else, so "reachable" was not yet a fixpoint — a state
        // that can look correct until something rebuilds again.
        let before: Vec<ENode> = eg.nodes(surviving).to_vec();
        eg.rebuild();
        assert_eq!(
            eg.find(nc),
            surviving,
            "a second rebuild must not move the surviving class"
        );
        assert_eq!(
            eg.nodes(eg.find(nc)),
            before.as_slice(),
            "rebuild must be idempotent once the worklist is drained"
        );
    }

    /// `union` gives the surviving class's nodes, then the merged-away
    /// class's, whichever argument order the caller used.
    #[test]
    fn union_gives_the_parent_then_the_child() {
        for flipped in [false, true] {
            let mut eg = EGraph::new();
            let parent = eg.add(ENode::Var(0));
            let parent_peer = eg.add(ENode::Var(1));
            let child = eg.add(ENode::Var(2));
            let child_peer = eg.add(ENode::Var(3));
            let parent = eg.union(parent, parent_peer);
            let child = eg.union(child, child_peer);
            assert_eq!(
                eg.nodes(parent),
                [ENode::Var(0), ENode::Var(1)],
                "a union of two classes gives the first then the second"
            );

            let survivor = if flipped {
                eg.union(child, parent)
            } else {
                eg.union(parent, child)
            };
            assert_eq!(survivor, parent, "the smaller id survives");
            assert_eq!(
                eg.nodes(survivor),
                [ENode::Var(0), ENode::Var(1), ENode::Var(2), ENode::Var(3)],
                "parent then child (flipped argument order: {flipped})"
            );
            assert_eq!(eg.nodes(child), eg.nodes(parent));
        }
    }

    /// A rebuild item whose class survives a union made during the item gives
    /// `[absorbed] ++ [kept]`: what the union appended, then the item's own
    /// nodes in the order they were walked. Built as the first regression test
    /// above is: the item is class 3, it absorbs class 4, and 3 < 4.
    #[test]
    fn a_rebuild_item_that_survives_gives_absorbed_then_kept() {
        let mut eg = EGraph::new();
        let _a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 3, the item
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 4, absorbed
        let d = eg.add(ENode::Var(3));
        let marker = eg.add(ENode::Var(9));
        eg.union(nc, d);
        eg.union(nb, marker);
        eg.union(b, c);
        assert_eq!(
            eg.nodes(nc),
            [ENode::apply(OpKind::Neg, &[c]), ENode::Var(3)],
            "the item's nodes before the rebuild, the first still naming c"
        );
        assert_eq!(
            eg.nodes(nb),
            [ENode::apply(OpKind::Neg, &[b]), ENode::Var(9)]
        );

        // The worklist is a stack of 3, 4, 1: class 1 pops, then 4, then 3.
        assert_eq!(eg.pending_rebuilds(), 3);
        assert_eq!(eg.rebuild_budgeted(3), 1, "the union in item 3 queues 3");
        assert_eq!(eg.find(nb), nc, "the item absorbed class 4 and survived");

        let neg_b = ENode::apply(OpKind::Neg, &[b]);
        let expected = [neg_b, ENode::Var(9), neg_b, ENode::Var(3)];
        assert_eq!(
            eg.nodes(nc),
            expected,
            "[absorbed: Neg(b), Var(9)] ++ [kept: Neg(b), Var(3)]"
        );
        eg.rebuild();
        assert_eq!(eg.nodes(nc), expected, "a further rebuild keeps the order");
    }

    /// A rebuild item whose class loses a union made during the item gives
    /// the survivor's prior nodes, then whatever the survivor absorbs later in
    /// the same item, then `[kept]`. The item is class 4. Its first node
    /// canonicalizes into class 3, which survives with a prior member; its
    /// second into class 5, which 3 then absorbs; its third is its own.
    #[test]
    fn a_rebuild_item_that_loses_gives_prior_then_later_absorbed_then_kept() {
        let mut eg = EGraph::new();
        let _a = eg.add(ENode::Var(0));
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let neg_b = eg.add(ENode::apply(OpKind::Neg, &[b])); // class 3, survives
        let neg_c = eg.add(ENode::apply(OpKind::Neg, &[c])); // class 4, the item
        let abs_b = eg.add(ENode::apply(OpKind::Abs, &[b])); // class 5, absorbed later
        let abs_c = eg.add(ENode::apply(OpKind::Abs, &[c]));
        let prior = eg.add(ENode::Var(7));
        let later = eg.add(ENode::Var(8));
        let marker = eg.add(ENode::Var(9));
        eg.union(neg_b, prior);
        eg.union(abs_b, later);
        eg.union(neg_c, abs_c);
        eg.union(neg_c, marker);
        eg.union(b, c);
        assert_eq!(
            [neg_b, neg_c, abs_b].map(EClassId::index),
            [3, 4, 5],
            "the recipe assumes these ids"
        );
        assert_eq!(
            eg.nodes(neg_c),
            [
                ENode::apply(OpKind::Neg, &[c]),
                ENode::apply(OpKind::Abs, &[c]),
                ENode::Var(9)
            ]
        );

        // The worklist is a stack of 3, 5, 4, 1: class 1 pops, then 4.
        assert_eq!(
            eg.rebuild_budgeted(2),
            2,
            "the unions in item 4 queue nothing"
        );
        assert_eq!(eg.find(neg_c), neg_b, "the item lost to class 3");
        assert_eq!(eg.find(abs_b), neg_b, "and 3 absorbed class 5 during it");

        let neg = ENode::apply(OpKind::Neg, &[b]);
        let abs = ENode::apply(OpKind::Abs, &[b]);
        let expected = [
            neg,
            ENode::Var(7), // the survivor's prior nodes
            abs,
            ENode::Var(8), // absorbed later in the item
            neg,
            abs,
            ENode::Var(9), // kept, in the order they were walked
        ];
        assert_eq!(eg.nodes(neg_b), expected);
        eg.rebuild();
        assert_eq!(
            eg.nodes(neg_b),
            expected,
            "a further rebuild keeps the order"
        );
    }

    /// Create an e-graph with standard algebraic rules for testing.
    fn egraph_with_rules() -> Saturation {
        let rules: Vec<Box<dyn Rewrite>> = vec![
            // InversePair rules
            Canonicalize::<AddNeg>::new(),
            Involution::<AddNeg>::new(),
            Cancellation::<AddNeg>::new(),
            InverseAnnihilation::<AddNeg>::new(),
            Canonicalize::<MulRecip>::new(),
            Involution::<MulRecip>::new(),
            Cancellation::<MulRecip>::new(),
            InverseAnnihilation::<MulRecip>::new(),
            // Commutativity
            Commutative::new(OpKind::Add),
            Commutative::new(OpKind::Mul),
            Commutative::new(OpKind::Min),
            Commutative::new(OpKind::Max),
            // Distributivity
            Distributive::new(OpKind::Mul, OpKind::Add),
            Distributive::new(OpKind::Mul, OpKind::Sub),
            // Identity
            Identity::new(OpKind::Add),
            Identity::new(OpKind::Mul),
            // Annihilator
            Annihilator::new(OpKind::Mul),
        ];
        Saturation::over(RuleSet::new(rules))
    }

    #[test]
    fn inverse_add() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let neg_x = eg.add(ENode::apply(OpKind::Neg, &[x]));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, neg_x]));
        SaturationConfig::compatibility(100).run(&mut eg);
        let zero = eg.add(ENode::constant(0.0));
        assert_eq!(eg.find(sum), eg.find(zero));
    }

    #[test]
    fn inverse_mul() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let recip_x = eg.add(ENode::apply(OpKind::Recip, &[x]));
        let product = eg.add(ENode::apply(OpKind::Mul, &[x, recip_x]));
        SaturationConfig::compatibility(100).run(&mut eg);
        let one = eg.add(ENode::constant(1.0));
        assert_eq!(eg.find(product), eg.find(one));
    }

    #[test]
    fn complex_inverse() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let five = eg.add(ENode::constant(5.0));
        let prod = eg.add(ENode::apply(OpKind::Mul, &[x, five]));
        let div = eg.add(ENode::apply(OpKind::Div, &[prod, x]));
        SaturationConfig::compatibility(100).run(&mut eg);
        assert_eq!(eg.find(div), eg.find(five));
    }

    #[test]
    fn nested_subtraction() {
        // a - (b - c) should equal a - b + c
        // Test: 10 - (6 - 2) = 10 - 4 = 6
        let mut eg = egraph_with_rules();
        let a = eg.add(ENode::constant(10.0)); // a = 10
        let b = eg.add(ENode::constant(6.0)); // b = 6
        let c = eg.add(ENode::constant(2.0)); // c = 2

        // Build a - (b - c)
        let b_minus_c = eg.add(ENode::apply(OpKind::Sub, &[b, c])); // 6 - 2 = 4
        let result = eg.add(ENode::apply(OpKind::Sub, &[a, b_minus_c])); // 10 - 4 = 6

        SaturationConfig::compatibility(100).run(&mut eg);

        // Extract and verify structure
        let costs = CostModel::default();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!("Extracted arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn mul_sub_pattern() {
        // This is the problematic pattern from discriminant:
        // d*d - (c - r) where d=4, c=16, r=1
        let mut eg = egraph_with_rules();
        let d = eg.add(ENode::constant(4.0));
        let c_sq = eg.add(ENode::constant(16.0));
        let r_sq = eg.add(ENode::constant(1.0));

        let d_sq = eg.add(ENode::apply(OpKind::Mul, &[d, d]));
        let inner_sub = eg.add(ENode::apply(OpKind::Sub, &[c_sq, r_sq]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[d_sq, inner_sub]));

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::default();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!("Extracted arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn mul_sub_pattern_with_vars() {
        // x*x - (y - z)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));

        let x_sq = eg.add(ENode::apply(OpKind::Mul, &[x, x]));
        let inner_sub = eg.add(ENode::apply(OpKind::Sub, &[y, z]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[x_sq, inner_sub]));

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::default();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!(
            "Extracted arena with vars: root={:?} len={}",
            root,
            arena.len()
        );
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn mul_sub_pattern_with_fma() {
        // Same pattern but with FMA costs (what the kernel! macro uses)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));

        let x_sq = eg.add(ENode::apply(OpKind::Mul, &[x, x]));
        let inner_sub = eg.add(ENode::apply(OpKind::Sub, &[y, z]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[x_sq, inner_sub]));

        SaturationConfig::compatibility(100).run(&mut eg);

        // Use default costs like the kernel! macro does
        let costs = CostModel::new();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!(
            "Extracted arena with FMA costs: root={:?} len={}",
            root,
            arena.len()
        );
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn discriminant_structure() {
        // Match the actual discriminant structure:
        // d_dot_c² - (c_sq - r_sq) where c_sq = a² + b² and r_sq = r²
        let mut eg = egraph_with_rules();
        let d = eg.add(ENode::Var(0));
        let a = eg.add(ENode::Var(1));
        let b = eg.add(ENode::Var(2));
        let r = eg.add(ENode::Var(3));

        let d_sq = eg.add(ENode::apply(OpKind::Mul, &[d, d]));
        let a_sq = eg.add(ENode::apply(OpKind::Mul, &[a, a]));
        let b_sq = eg.add(ENode::apply(OpKind::Mul, &[b, b]));
        let c_sq = eg.add(ENode::apply(OpKind::Add, &[a_sq, b_sq]));
        let r_sq = eg.add(ENode::apply(OpKind::Mul, &[r, r]));
        let inner = eg.add(ENode::apply(OpKind::Sub, &[c_sq, r_sq]));
        let result = eg.add(ENode::apply(OpKind::Sub, &[d_sq, inner]));

        SaturationConfig::compatibility(100).run(&mut eg);

        let costs = CostModel::new();
        let dag = extract_dag(&eg, result, &costs);
        let (arena, root) = dag.to_arena(&eg);
        eprintln!("Discriminant arena: root={:?} len={}", root, arena.len());
        assert!(arena.node_count_subtree(root) > 0);
    }

    #[test]
    fn depth_cost_should_apply_linear_penalty_only_above_threshold() {
        let mut costs = CostModel::new();
        costs.depth_threshold = 5;
        costs.depth_penalty = 100;

        assert_eq!(costs.depth_cost(0), 0);
        assert_eq!(costs.depth_cost(5), 0);

        assert_eq!(costs.depth_cost(6), 100);
        assert_eq!(costs.depth_cost(7), 200);
        assert_eq!(costs.depth_cost(10), 500);
    }

    #[test]
    fn shallow_should_set_aggressive_depth_threshold_and_penalty() {
        let costs = CostModel::shallow();
        assert_eq!(costs.depth_threshold, 16);
        assert_eq!(costs.depth_penalty, 500);

        assert_eq!(costs.depth_cost(16), 0);
        assert_eq!(costs.depth_cost(17), 500);
        assert_eq!(costs.depth_cost(20), 2000);
    }

    #[test]
    fn depth_aware_extraction() {
        // Build a deep expression: ((((x + 1) + 1) + 1) + 1)
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let one = eg.add(ENode::constant(1.0));

        let mut current = x;
        for _ in 0..10 {
            current = eg.add(ENode::apply(OpKind::Add, &[current, one]));
        }

        SaturationConfig::compatibility(100).run(&mut eg);

        // Extract with default costs (high threshold)
        let default_costs = CostModel::default();
        let dag = extract_dag(&eg, current, &default_costs);
        let (arena, root) = dag.to_arena(&eg);
        assert!(arena.node_count_subtree(root) > 0);

        // Extract with shallow costs (low threshold)
        let mut shallow_costs = CostModel::new();
        shallow_costs.depth_threshold = 3;
        shallow_costs.depth_penalty = 1000;
        let dag = extract_dag(&eg, current, &shallow_costs);
        let (arena2, root2) = dag.to_arena(&eg);
        assert!(arena2.node_count_subtree(root2) > 0);
    }

    // ========================================================================
    // Provenance tests
    // ========================================================================

    /// Find the `RewriteTarget` for a given rule name, failing loudly (not
    /// silently) if it isn't present — a missing match means the test setup
    /// is wrong, not that provenance has nothing to check.
    fn find_target(eg: &Saturation, rule_name: &str) -> RewriteTarget {
        eg.find_rewrite_matches()
            .into_iter()
            .find(|t| eg.rules().rule(t.rule_idx).name() == rule_name)
            .unwrap_or_else(|| panic!("no rewrite match found for rule {rule_name:?}"))
    }

    /// Like `find_target`, but further restricted to matches against a
    /// specific (already-canonical) class — needed once a class holds
    /// multiple nodes that all match the same rule name.
    fn find_target_in_class(eg: &Saturation, rule_name: &str, class: EClassId) -> RewriteTarget {
        eg.find_rewrite_matches()
            .into_iter()
            .find(|t| eg.rules().rule(t.rule_idx).name() == rule_name && t.class_id == class)
            .unwrap_or_else(|| {
                panic!("no rewrite match found for rule {rule_name:?} in class {class:?}")
            })
    }

    #[test]
    fn provenance_tiny_expression_one_rewrite_matches_hand_derivation() {
        // x + y, then apply Commutative -> y + x (a fresh Create'd node).
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, y]));

        // Hand-derivation: x, y, and sum are all seeds.
        for &(class, _) in &[(x, "x"), (y, "y"), (sum, "sum")] {
            for &tag in eg.tags(class) {
                assert_eq!(
                    eg.journal().origin(tag),
                    Some(Origin::Seed),
                    "seed node in class {class:?} should be Origin::Seed"
                );
            }
        }
        assert_eq!(eg.journal().recorded_count(), 0);
        assert_eq!(eg.journal().union_count(), 0);

        let target = find_target(&eg, "commutative");
        assert_eq!(target.class_id, eg.find(sum));

        let applied = eg.apply_single_rule(target.rule_idx, target.class_id, target.tag);
        assert!(applied, "commutative rule should have applied to x + y");

        // Hand-derivation: exactly one application recorded, for the
        // "commutative" rule, matched against `sum`'s class.
        assert_eq!(eg.journal().recorded_count(), 1);
        let record = eg.journal().application(ApplicationId(0)).unwrap();
        assert_eq!(record.rule_idx, target.rule_idx);
        assert_eq!(record.match_root, eg.find(sum));
        assert_eq!(eg.rules().rule(record.rule_idx).name(), "commutative");

        // Hand-derivation: the new commuted node (y + x) has Origin::Rule
        // pointing at that one application; it lives in the same class as
        // `sum` post-union.
        let commuted_class = eg.find(sum);
        let rule_origin_tags: Vec<ENodeId> = eg
            .tags(commuted_class)
            .iter()
            .copied()
            .filter(|&t| matches!(eg.journal().origin(t), Some(Origin::Rule(_))))
            .collect();
        assert_eq!(
            rule_origin_tags.len(),
            1,
            "expected exactly one rule-created node in the merged class"
        );
        assert_eq!(
            eg.journal().origin(rule_origin_tags[0]),
            Some(Origin::Rule(ApplicationId(0)))
        );

        // Hand-derivation: exactly one union event, attributed to the
        // commutative rule's index, at the step apply_single_rule ran on.
        assert_eq!(eg.journal().union_count(), 1);
        let union_event = eg.journal().union_events()[0];
        assert_eq!(union_event.rule_idx, Some(target.rule_idx));
    }

    #[test]
    fn provenance_chain_rule_b_consumes_rule_a_product() {
        // Chain: apply Commutative to (x + y) producing (y + x) [rule A],
        // then apply Commutative again to a *different* sum, (y + x) + z,
        // whose match consumes the class that A's product lives in as a
        // child. derivation_ancestors of B's product must include A's
        // application.
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let inner = eg.add(ENode::apply(OpKind::Add, &[x, y])); // x + y
        let outer = eg.add(ENode::apply(OpKind::Add, &[inner, z])); // (x + y) + z

        // Rule A: commute the inner sum (x + y) -> (y + x).
        let target_a = find_target_in_class(&eg, "commutative", eg.find(inner));
        assert!(eg.apply_single_rule(target_a.rule_idx, target_a.class_id, target_a.tag));
        let app_a = ApplicationId(0);

        // Rule B: commute the outer sum ((x+y)+z) -> (z + (x+y)). After
        // rule A, inner's class holds both `x+y` and `y+x`, both of which
        // still match "commutative" — so we must specifically target
        // outer's class rather than take the first "commutative" match.
        let target_b = find_target_in_class(&eg, "commutative", eg.find(outer));
        assert!(eg.apply_single_rule(target_b.rule_idx, target_b.class_id, target_b.tag));
        let app_b = ApplicationId(1);
        assert_eq!(eg.journal().recorded_count(), 2);

        // Find B's produced node: the rule-created node in outer's class
        // whose origin is app_b.
        let outer_class = eg.find(outer);
        let b_product_tag = eg
            .tags(outer_class)
            .iter()
            .copied()
            .find(|&t| eg.journal().origin(t) == Some(Origin::Rule(app_b)))
            .expect("expected a node created by application B in outer's class");

        let ancestors = eg.derivation_ancestors(&[(outer_class, b_product_tag)]);
        assert!(
            ancestors.contains(&app_b),
            "B's own application must be in its own ancestry"
        );
        assert!(
            ancestors.contains(&app_a),
            "A's application (which produced the node B's match consumed as \
             a child of the outer sum) must be included in B's product's ancestry"
        );

        // Sanity: the trace formats without panicking and mentions both
        // applications' rule name.
        let trace = eg.format_derivation_trace(&ancestors);
        assert!(trace.contains("commutative"));
    }

    #[test]
    fn provenance_union_driven_case_includes_congruence_union_event() {
        // Build two structurally-identical-after-canonicalization Neg nodes
        // in different classes purely because their children start out
        // unmerged, then union the children directly (not through a rule).
        //
        // Congruence closure only reconsiders a class's memo key when that
        // class itself is on the rebuild worklist (see rebuild_budgeted) —
        // union(b, c) alone doesn't touch nb/nc's classes. So we also
        // enqueue nc via a harmless union(nc, marker) (nc's numeric id is
        // lower, so nc survives as parent and just gains an extra node);
        // this is the same enqueue trick used in
        // rebuild_budgeted_does_not_drop_nodes_when_current_class_survives_union.
        // When rebuild() then reprocesses nc, canonicalizing Neg([c]) turns
        // it into Neg([b]) (since union(b,c) already ran), which collides
        // with nb's memo entry — a congruence-closure union with
        // rule_idx = None.
        let mut eg = egraph_with_rules();
        let b = eg.add(ENode::Var(1));
        let c = eg.add(ENode::Var(2));
        let nb = eg.add(ENode::apply(OpKind::Neg, &[b]));
        let nc = eg.add(ENode::apply(OpKind::Neg, &[c]));
        let marker = eg.add(ENode::Var(9));

        assert_eq!(eg.journal().union_count(), 0);

        // Enqueue nc on the worklist without disturbing its semantics.
        eg.union(nc, marker);
        // Direct union, not via a rule: rule_idx must be None on the
        // resulting UnionEvent.
        eg.union(b, c);
        eg.rebuild();

        assert_eq!(
            eg.find(nb),
            eg.find(nc),
            "Neg(b) and Neg(c) should have merged"
        );

        // At least one recorded union event must have rule_idx = None
        // (the direct union(b, c) and/or the congruence-closure union that
        // merged nb/nc during rebuild).
        assert!(
            eg.journal()
                .union_events()
                .iter()
                .any(|e| e.rule_idx.is_none()),
            "expected at least one non-rule-driven (congruence/direct) union event"
        );
    }

    /// Overhead measurement: total saturation time and provenance record
    /// counts on a reasonably deep expression. Not a correctness test —
    /// `#[ignore]`d so normal `cargo test` runs stay fast; run explicitly
    /// with `cargo test -p pixelflow-search --release -- --ignored
    /// provenance_overhead`. See module docs on `provenance` for the
    /// numbers observed when this was last run.
    #[test]
    #[ignore]
    fn provenance_overhead_timing() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));

        // A reasonably deep expression: alternating +/-/* chain over x, y.
        let mut current = x;
        for i in 0..40 {
            let op: OpKind = match i % 3 {
                0 => OpKind::Add,
                1 => OpKind::Mul,
                _ => OpKind::Sub,
            };
            current = eg.add(ENode::apply(op, &[current, y]));
        }

        let start = std::time::Instant::now();
        SaturationConfig::compatibility(100).run(&mut eg);
        let elapsed = start.elapsed();

        eprintln!(
            "provenance overhead: saturation took {:?}; origins={} applications={} unions={} classes={}",
            elapsed,
            eg.journal().origin_count(),
            eg.journal().recorded_count(),
            eg.journal().union_count(),
            eg.num_classes(),
        );
    }

    // ------------------------------------------------------------------
    // The laws. `docs/archive/plans/2026-09-02-optimizer-api.md` §1 states each one
    // and names the test that should pin it; these are those tests. They
    // assert the *property*, never the mechanism — a future fix that moves
    // the class budget somewhere else entirely should keep them passing.
    // ------------------------------------------------------------------

    /// Seed a graph whose rules certainly have work to do, and return the
    /// probe ids the law tests watch. `Sub` guarantees a `Canonicalize`
    /// firing (`a - b → a + (-b)`), which is a *node-minting* action — the
    /// kind that used to route through the over-budget hole in `add`.
    fn seed_probes(eg: &mut EGraph) -> Vec<EClassId> {
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let z = eg.add(ENode::Var(2));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, y]));
        let diff = eg.add(ENode::apply(OpKind::Sub, &[x, y]));
        let prod = eg.add(ENode::apply(OpKind::Mul, &[sum, z]));
        let scaled = eg.add(ENode::apply(OpKind::Mul, &[diff, z]));
        vec![x, y, z, sum, diff, prod, scaled]
    }

    /// The pairs of probes the graph currently calls equal.
    fn equal_pairs(eg: &EGraph, probes: &[EClassId]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..probes.len() {
            for j in (i + 1)..probes.len() {
                if eg.find(probes[i]) == eg.find(probes[j]) {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// **L1 — soundness.** `add` is a homomorphism onto the semantic
    /// quotient: the class it names always contains the node it was given,
    /// and no amount of graph growth changes that. Consequently an
    /// over-budget run can discover *fewer* equalities than an exhaustive
    /// one, never a different one.
    ///
    /// This is the regression test for #1105. Before the fix, `add`
    /// answered `EClassId(0)` once the graph reached the hard class limit —
    /// a sentinel, not an insertion — and `RewriteAction`'s node-minting
    /// handlers unioned the matched class against it, asserting that some
    /// unrelated term equals whatever was added first. Both halves are
    /// asserted here: that `add` past the limit still names a class holding
    /// the node, and that a saturation run driven past the limit invents no
    /// equality among terms that are not equal.
    ///
    /// Deliberately mechanism-blind: it says nothing about *how* growth is
    /// refused, only that nothing false comes out.
    #[test]
    fn over_budget_growth_cannot_assert_a_false_equality() {
        let mut eg = egraph_with_rules();

        // Class 0 is the term every false union used to land on, so make it
        // something no probe below is equal to.
        let class_zero = eg.add(ENode::constant(-1.0));
        assert_eq!(class_zero, EClassId(0), "class 0 must be the first add");

        let probes = seed_probes(&mut eg);
        assert!(
            equal_pairs(&eg, &probes).is_empty(),
            "probes start distinct"
        );

        // Pad to one short of the ceiling with distinct constants, so the
        // next rewrite that wants to mint a node is over budget — and so the
        // graph lands *exactly* on the ceiling after the `add` below, which
        // keeps the saturation run below inside the sweep rather than
        // bailing at `saturate_with_limits`'s own pre-sweep check.
        let mut pad = 0.0f32;
        while eg.classes.len() < HARD_MINTED_LIMIT - 1 {
            let _ = eg.add(ENode::constant(pad));
            pad += 1.0;
        }

        // `add` is still total at the ceiling: the class it returns holds
        // the node, and it is not class 0's term.
        let fresh = eg.add(ENode::Var(200));
        assert_eq!(eg.classes.len(), HARD_MINTED_LIMIT);
        assert!(
            matches!(eg.nodes(eg.find(fresh)).first(), Some(ENode::Var(200))),
            "add past the limit returned a class that does not hold the node"
        );
        assert_ne!(
            eg.find(fresh),
            eg.find(class_zero),
            "add past the limit aliased an unrelated class"
        );

        // Drive saturation with a cap no smaller than the graph, so the
        // ceiling — not the caller's budget — is what has to hold.
        let stats = eg.saturate_with_limits(4, usize::MAX, std::time::Duration::from_secs(60));
        eprintln!("over-budget saturation: {stats:?}");

        // The load-bearing assertion: nothing new is equal.
        assert!(
            equal_pairs(&eg, &probes).is_empty(),
            "over-budget saturation invented an equality among distinct terms"
        );
        for (i, &p) in probes.iter().enumerate() {
            assert_ne!(
                eg.find(p),
                eg.find(class_zero),
                "probe {i} was falsely unioned with class 0"
            );
        }
        assert_ne!(eg.find(fresh), eg.find(class_zero));
    }

    /// Seed a graph in which equalities are *derivable* and arrive at
    /// different depths, so that a budget ladder has something to be
    /// monotone about. Returns the probes, several of which the rules
    /// eventually prove congruent.
    fn seed_derivable_probes(eg: &mut EGraph) -> Vec<EClassId> {
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let zero = eg.add(ENode::constant(0.0));
        let one = eg.add(ENode::constant(1.0));

        // x + 0 = x, one Identity firing away.
        let x_plus_zero = eg.add(ENode::apply(OpKind::Add, &[x, zero]));
        // (x + 0) * 1 = x, two firings away.
        let scaled = eg.add(ENode::apply(OpKind::Mul, &[x_plus_zero, one]));
        // x * 0 = 0, via Annihilator.
        let annihilated = eg.add(ENode::apply(OpKind::Mul, &[x, zero]));
        // x + y = y + x, via Commutative.
        let xy = eg.add(ENode::apply(OpKind::Add, &[x, y]));
        let yx = eg.add(ENode::apply(OpKind::Add, &[y, x]));

        vec![x, y, zero, one, x_plus_zero, scaled, annihilated, xy, yx]
    }

    /// **L2 — monotonicity.** Saturation only ever adds equalities. A pair
    /// the graph already calls equal is still equal after a run at any
    /// budget, and the partition at a larger budget refines the partition at
    /// a smaller one.
    ///
    /// Budget truncation is allowed to stop discovery; it is not allowed to
    /// retract a discovery. That is what lets a budget stay a
    /// quality/compile-time dial instead of a correctness one, and it is half
    /// the proof of L4 (any saturation policy preserves denotation) — the
    /// other half is L1, above.
    ///
    /// Two things are asserted, in increasing strength: a hand-made union
    /// that no rule could have derived survives every budget, and the set of
    /// probe pairs the graph calls equal only ever grows as the budget does.
    ///
    /// Note what is *not* asserted, because this rebuild does not do it:
    /// upward congruence. `union` enqueues only the merged class, never its
    /// parents, so `a = b` does not by itself re-canonicalize `f(a)` and
    /// `f(b)` into one class — they merge when a later sweep re-walks them,
    /// or not at all. That is an incompleteness (fewer equalities than an
    /// ideal congruence closure), which is exactly the direction L2 permits;
    /// it is filed as #1106 rather than assumed here.
    #[test]
    fn saturation_at_any_budget_never_removes_an_equality() {
        // Strictly increasing, spanning "cannot even finish one rule" to the
        // ceiling itself.
        let ladder = [1usize, 2, 8, 64, 512, 4096, HARD_MINTED_LIMIT];
        let mut previous: Option<(usize, Vec<(usize, usize)>)> = None;
        let mut ever_found = 0usize;

        for max_minted in ladder {
            let mut eg = egraph_with_rules();
            let probes = seed_derivable_probes(&mut eg);

            // A hand-asserted equality between two terms no rule in this
            // vocabulary relates: only a retraction could ever break it.
            let a = eg.add(ENode::Var(30));
            let b = eg.add(ENode::Var(31));
            eg.union(a, b);
            eg.rebuild();
            assert_eq!(eg.find(a), eg.find(b), "union did not take");

            eg.saturate_with_limits(20, max_minted, std::time::Duration::from_secs(30));

            assert_eq!(
                eg.find(a),
                eg.find(b),
                "budget {max_minted} retracted a hand-made equality"
            );

            // Refinement: every pair equal at a smaller budget is still
            // equal at this one.
            let pairs = equal_pairs(&eg, &probes);
            if let Some((smaller, before)) = &previous {
                for pair in before {
                    assert!(
                        pairs.contains(pair),
                        "budget {max_minted} lost probe pair {pair:?} that budget \
                         {smaller} had found — saturation is not monotone"
                    );
                }
            }
            ever_found = ever_found.max(pairs.len());
            previous = Some((max_minted, pairs));
        }

        // Guard against the ladder asserting nothing: if no budget ever
        // derived an equality, the refinement check above was vacuous.
        assert!(
            ever_found > 0,
            "no budget derived any probe equality — the monotonicity ladder \
             is vacuous and no longer tests L2"
        );
    }
}

#[cfg(test)]
mod variance_fact_tests {
    use super::*;
    use pixelflow_ir::{Binder, Fold, Monoid};

    /// **Free of what any member is free of.** `X − X` is seeded `{X}` from
    /// its children; the graph then learns it equals `0`, and the class is
    /// `{X} ∩ ∅ = ∅`. A parent seeded before the union keeps its larger fact
    /// — no upward repair — which is still a superset of the truth, so a rule
    /// reading it can miss the invariance but never invent one.
    #[test]
    fn a_class_is_free_of_what_any_member_is_free_of() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let x_minus_x = eg.add(ENode::apply(OpKind::Sub, &[x, x]));
        let parent = eg.add(ENode::apply(OpKind::Mul, &[x_minus_x, y]));
        assert_eq!(eg.variance(x_minus_x), Variance::X, "seeded from X and X");
        assert_eq!(eg.variance(parent), Variance::COORDS);

        let zero = eg.add(ENode::constant(0.0));
        eg.union(x_minus_x, zero);
        eg.rebuild();

        assert_eq!(
            eg.variance(x_minus_x),
            Variance::CONST,
            "one member proves the class constant, so the class is"
        );
        assert!(
            eg.variance(parent).depends_on_x(),
            "the parent's seed is stale and says so: still a superset of the \
             truth ({{Y}}), which is all soundness needs"
        );
    }

    /// **The meet is an intersection, not a pick.** A class proved equal to
    /// both a Y-only and an X-only term varies with neither — a popcount
    /// minimum would have kept one of the two.
    #[test]
    fn two_disjoint_members_leave_nothing() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sin_x = eg.add(ENode::apply(OpKind::Sin, &[x]));
        let sin_y = eg.add(ENode::apply(OpKind::Sin, &[y]));
        eg.union(sin_x, sin_y);
        eg.rebuild();
        assert_eq!(eg.variance(sin_x), Variance::CONST);
    }

    /// **A fold frees its binder**, and only its binder: `Σ_i (i · X)` varies
    /// with `X`, its body with both.
    #[test]
    fn a_fold_removes_its_binder() {
        let mut eg = EGraph::new();
        let binder = Binder::from_slot(0).expect("a live binder");
        let i = eg.add(ENode::Var(binder.var()));
        let x = eg.add(ENode::Var(0));
        let body = eg.add(ENode::apply(OpKind::Mul, &[i, x]));
        let fold = eg.add(ENode::Reduce {
            fold: Fold::new(Monoid::SUM, binder, 0..8),
            body,
        });
        assert_eq!(
            eg.variance(body),
            Variance::X.union(Variance::from_var(binder.var()))
        );
        assert_eq!(eg.variance(fold), Variance::X);
    }

    /// **A cycle does not certify invariance.** Once `X + 0` is merged with
    /// `X`, the class holds a member whose child is the class itself. Every
    /// fact in it was seeded from facts that already existed — `X`'s own bit
    /// — and never from an assumption about the class, so the cycle leaves
    /// `{X}` standing. A least fixpoint started from `∅` closes at `∅`
    /// instead: `var(C) = {X} ∩ var(C + 0)` and `var(C + 0) = var(C)`, so
    /// `∅` is a fixpoint, and it is false.
    #[test]
    fn a_cycle_does_not_certify_invariance() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let x_plus_0 = eg.add(ENode::apply(OpKind::Add, &[x, zero]));
        eg.union(x_plus_0, x);
        eg.rebuild();

        let class = eg.find(x);
        assert!(
            eg.nodes(class)
                .iter()
                .any(|n| n.children_slice().iter().any(|&c| eg.find(c) == class)),
            "precondition: the class reaches itself"
        );
        assert_eq!(eg.variance(class), Variance::X);
    }

    /// A journal is chosen when the graph is built: dropping it from a graph
    /// that already holds classes would forget their births.
    #[test]
    #[should_panic(expected = "already holds")]
    fn a_journal_cannot_be_dropped_from_a_populated_graph() {
        let mut egraph = EGraph::new();
        let _x = egraph.add(ENode::Var(0));
        let _journalless = egraph.without_journal();
    }

    /// The fact survives a clone — a search branch reads the same facts its
    /// parent had.
    #[test]
    fn a_clone_keeps_the_facts() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let sum = eg.add(ENode::apply(OpKind::Add, &[x, y]));
        let copy = eg.clone();
        assert_eq!(copy.variance(sum), Variance::COORDS);
    }
}
