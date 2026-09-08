# The rules × nodes filter: a seam before application

JP, 2026-09-08: "something of the shape of my bilinear filter that takes
rules × nodes ⇒ rules × nodes, before application. I want a trait attached
to it, and I want the naive version to be a no-op." This document denotes
the seam; the code in `pixelflow-search/src/egraph/filter.rs` is obliged to
it. The research question it serves is whether a CPU-resident net can scale
the e-graph past its class cap: on real scenes the rules expand 110–465
input nodes 10–40× to the cap in 2–5 rounds, and only ~1.2 % of applications
are strictly load-bearing (~20 % by the loose bound;
docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md). A filter that
keeps the load-bearing `(rule, class)` pairs reaches the same or a better
extraction on a fraction of the budget. Rules are domain-conditional
(Pythagorean identities: useless for Béziers, essential for spherical
harmonics), which is why the filter is bilinear over a rule embedding *and* a
node embedding rather than a per-rule prior — but this document is only
about the seam, and the seam is policy-free.

## The denotation

**The match matrix.** Each saturation round, the e-graph has a sparse matrix
`M : Rule × EClass ⇀ Action`. A cell `M[r, c]` is inhabited exactly when rule
`r` matches some e-node of canonical class `c`, and it holds the
`RewriteAction` that match produced — the rule's right-hand side *with its
bindings already substituted* (a `Union` target, a `Create`d e-node, a
`Distribute {a, b, c}` …). Not a bool and not a count: a bool would lose the
binding, which is the only thing that distinguishes two firings of the same
rule at the same class, and a count would lose which ones. `Rewrite::apply`
returns at most one action per `(rule, e-node)`, so a cell may hold more than
one action when a class has several matching e-nodes; the row for rule `r` is
the list of `(c, action)` pairs in scan order. This is exactly the
`updates: Vec<(EClassId, RewriteAction)>` the production scan already
stages — the matrix type borrows it rather than inventing a second one.

**The filter.** `F : M → M'`, same shape: a function from a sparse
rules × classes matrix to a sparse rules × classes matrix over the *same*
rule set and the *same* class ids. `F` runs **after matching and before
application**: it sees every match the round enumerated, with its binding,
and it decides which of them the loop commits. It may drop cells; it may not
invent an action the rules did not produce (there is no constructor for a
binding it did not receive), so the graph after `F` holds a subset of the
equalities the unfiltered graph would — the same law (L4 in
`egraph/optimizer.rs`) every ordering policy is already covered by.

**Identity.** `Identity(M) = M`. Under `Identity` the loop commits exactly
what it enumerated, in the order it enumerated it, and every observable —
emitted bytes, `dag_cost`, schedule, stop reason, application count, class
count, iteration count — is byte-identical to the loop before the seam
existed. **That is a test
(`pixelflow-pipeline/tests/rules_by_nodes_identity.rs`), not a claim.**

**Budget accounting.** The application counter (`EGraph::applications`, the
one budget dimension every policy shares) is charged **at commit, for M'**.
A dropped cell costs nothing: it is not counted, not recorded in the
provenance journal, and not applied — the same three "nots" the replay mask
already guarantees. The scan enumerates a row only up to the *remaining*
allowance (it cannot know what `F` will keep, and it must bound its own work),
so `|row| ≤ remaining` before `F` and `F` can only shrink it; a filter that
drops most of a row leaves that headroom for the next rule's row. Likewise
the class cap: the scan's `estimated_new_nodes` is an upper bound computed
over M, so a filtered commit can only stay further under the cap. Both bounds
are conservative in the direction of *fewer* applications, never more.

## Which matching semantics the loop has

The wiring depends on one question: within one round, does rule `r+1` match
against rule `r`'s additions, or against the round's starting snapshot?

**It matches against rule `r`'s additions.** `EGraph::saturate_bounded`
(`egraph/graph.rs`) runs, per round, `for rule_idx in 0..n_rules {
batch.apply_rule(rule_idx, …) }`, and `EGraphBatch::apply_rule` is: scan rule
`r` over `canonical_class_ids()`, **commit** its staged actions
(`apply_action_from_rule` — unions and node creations land in `classes`
immediately), then `rebuild_budgeted(256)` — an *interleaved partial rebuild*
of up to 256 worklist items before the next rule's scan starts. Rule `r+1`'s
`canonical_class_ids()` therefore already contains what rule `r` created,
partially re-canonicalized. The round's "one rebuild per iteration" is only
the *drain* on `EGraphBatch::drop`; the graph rule `r+1` sees is not the
round's starting snapshot.

Consequently **M cannot be collected for all rules and filtered once**
without changing what M *is*: rule `r+1`'s row is a function of rule `r`'s
committed row, so collecting every row against the snapshot would enumerate
different matches (and different bindings) than production does, and no
filter — Identity included — could reproduce production's bytes from it. The
only shape that preserves the semantics is the **streamed** one: `F` is
applied to rule `r`'s row at the moment it is fully matched and before it is
committed, then the loop proceeds to rule `r+1`. That is still
rules × nodes → rules × nodes — the matrix is delivered one row at a time
because that is the order the loop *produces* it, and the filter sees the
graph as it stands when that row was matched, which is the same graph the
row's bindings refer to. A whole-matrix filter is representable on top of it
(buffer rows, decide, but then the rows are no longer what production would
have matched), and that is precisely why it is not the seam.

So the trait method takes one row, and "before application, after matching"
is the point between the scan loop and the commit loop inside
`apply_rule_at_index_timed` — the one place in the crate where a row exists
as a value that has been matched but not yet committed.

## What exists that this touches

- `EGraph::saturate_bounded` — the one production loop. Now takes
  `&mut dyn ApplicationFilter`; `saturate_budgeted`/`saturate_with_limits`
  pass `Identity`, and `Optimizer` passes whatever `Optimizer::filter` was
  given (default `Identity`).
- `ApplicationMask` / `Optimizer::mask` — a filter in a weaker form (it
  withholds one application, or every re-derivation of it), but at a
  *different point*: it decides at **commit time**, keyed on the commit-time
  application ordinal, and under `MaskScope::AllMatchingCandidate` it reads
  class content **as of each commit** — which an earlier commit in the same
  row may have changed. A pre-application row filter sees the row's
  pre-commit content, so folding the mask into it would change the
  confluence-aware scope's semantics (and the replay harness's Δ
  measurements). The `Single` scope *is* expressible as a row filter
  (`ordinal = applications + kept_so_far`); folding it is a phase-2 change
  that owes its own identity test on the mask fixtures. Left in place.
- `SaturationGuide::score_candidates` (`nnue/guide/`, `egraph/guided.rs`,
  `egraph/anytime.rs`) — the same idea (score rule × class candidates, with
  rule and node embeddings) at the **wrong point**: a flat candidate list
  scored after a separate enumeration in a separate loop
  (`GuidedEpisode::advance` → `apply_single_rule`, which re-matches), never
  wired into production, with no `Identity` and no byte-identity test. The
  bilinear head (`nnue/guide/bilinear.rs`) is the model that phase 2 ports
  onto `ApplicationFilter`; once it is ported, `SaturationGuide`, the guided
  loop, and `Optimizer::guide` are marked for deletion. Not deleted here.

## Public surface

`egraph::filter::{ApplicationFilter, MatchRow, Identity}` and
`Optimizer::filter(Box<dyn ApplicationFilter>)`. Nothing else. The row's
`RewriteAction`s are already public (`egraph::RewriteAction`); the row is
consumed by value inside the crate and exposed to the filter by `&mut`.

`Box<dyn ApplicationFilter>` rather than a type parameter on `Optimizer`:
the filter is called **once per rule per round** (62 × ≤ 9 calls on the
largest DEV kernel), not per match, and `Optimizer` already holds
`Box<dyn Reranker>` and `Box<dyn SaturationGuide>` the same way. `Identity`
is a ZST, so its `Box` does not allocate; the seam adds no per-round
allocation (the row *is* the scan's existing `updates` vector, moved into
`MatchRow` and back).

## Cost of the seam at Identity

Deterministic: zero additional allocations per round (the row is the
existing vector, moved); one indirect call per rule per round with an empty
body. Clock: Σ glyph bake time, taken at load < 8, must be inside noise —
recorded in the PR.

## Phase 2 (not this change)

Port `BilinearCandidateGuide` onto `ApplicationFilter` (rule embedding ×
node embedding of the matched class, scored per cell, keep the top-k or
above-threshold), train it on `dag_cost`-minted hindsight labels
(docs/plans/2026-09-01-guide-return-to-go.md; tree-cost labels steer toward
unshared terms, docs/results/2026-09-02-guided-regression-bisect.md), and
measure regret-at-budget against `Identity` on DEV only
(docs/plans/2026-09-01-phase3-registration.md). Then delete
`SaturationGuide` and the guided loop.
