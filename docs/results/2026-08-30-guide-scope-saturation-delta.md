# Saturation delta economics: does the Stockfish incrementality argument hold for the Guide? (2026-08-30, revised)

Reproduce:
```
cargo run --release -p pixelflow-pipeline --features training --bin guide_scope_saturation_delta -- \
    --corpus-dir pixelflow-pipeline/data --min-expressions 200 --limit 0 \
    --out docs/results/2026-08-30-guide-scope-saturation-delta.json
```
Harness: `pixelflow-pipeline/src/bin/guide_scope_saturation_delta.rs`. Corpus: all 1,512
expressions in `corpus_dev.bin` (comfortably above the >=200 floor). Every number below is a
deterministic count or a graph-size ratio — no wall-clock timing anywhere. Pre-run check:
`pgrep -fl "bootstrap_extraction_head|bench_extraction_3way"` — a `bootstrap_extraction_head`
publication run was live in another worktree while this file was finalized; this harness is
CPU-light (~20s wall-clock for the full corpus) and does not compete with it meaningfully. Full
run: 1,512 expressions, ~20s wall-clock.

## Revision note (why this version replaces the 2026-08-30 21:19 draft)

The first draft of this report was written against a harness that drove saturation by
**single-stepping**: scan `find_rewrite_matches()` once, then fire every match in that
snapshot serially via `EGraph::apply_single_rule`. Before that draft's session ended, the
harness itself was rewritten (its own module doc explains why — see "A rejected design" below)
to drive saturation via the **actual production algorithm**,
`EGraph::saturate_with_limits(1, max_classes, big_timeout)` called once per round — the same
batched path `EGraph::saturate()` uses in every real caller, just interrupted after each round
instead of run to completion in one call. The source file was edited *after* the JSON that
draft's prose quoted was written, so that draft's headline numbers did not match what the
harness on disk actually computes or reproduces. This version regenerates the JSON from the
current (final) harness and rewrites the numbers throughout; the qualitative conclusions
(incrementality holds, and holds harder than the single-step draft found) are unchanged, but
every specific figure below is different from — and supersedes — the first draft's.

This audit also found the single-step draft's headline needed a second correction independent
of the methodology swap: under the real batched algorithm, the *median* recorded application has
**zero** delta (see below), which the original per-application summary would have silently
reported as "median 0.0%" with no explanation. The harness was extended (this round, additively)
to also report the zero-delta share and the delta distribution conditioned on state-changing
applications only, so the headline is accurate rather than misleadingly degenerate.

**Revision note 2 (2026-09-01, PR review fixes):** an automated review of PR #1067
(`chatgpt-codex-connector`) found two further defects in this harness, both now fixed and both
re-measured rather than assumed away:

1. *Dropped-node attribution.* `rebuild_budgeted` can permanently drop a node from its class when
   canonicalizing it collides with a different, constant-incompatible class (`EGraph::union`'s
   ill-conditioned-kernel refusal, `graph.rs` — a deliberate under-merging valve, not routine
   congruence closure). `Provenance::origins()` is append-only and still names such a node's
   creating application forever, but the node is gone from the final canonical structure this
   harness's `arity_of` map is built from — so the previous code counted it toward `nodes_added`
   while silently defaulting its edge contribution to 0, an inconsistency between the two counts
   for one application. Fixed: such origins are now excluded from both counts consistently, and
   the exclusion count is reported. **Measured impact: 136 refusal events, 1,997 origins excluded
   — 0.044% of the 4.49M total applications.** The concern was valid; the practical effect on the
   headline numbers below is negligible, now shown rather than assumed.
2. *Rejected round vs. quiescence.* This function's own pre-round guard checks *canonical*
   (live-root) class count against `max_classes`, but `saturate_with_limits`'s internal cap checks
   *allocated* class-slot count (always >= canonical count, since it includes classes already
   unioned away). A round could pass this function's guard, then have `saturate_with_limits`
   immediately hit its own stricter cap and do zero work (`stats.iterations == 0`) — which the
   previous `stats.total_unions == 0` check alone could not distinguish from genuine convergence.
   Fixed: a round with `stats.iterations == 0` is now treated as budget-exhausted, not quiescence.
   **Measured impact: 0/1,512 expressions changed classification** — this corpus's
   `max_classes=3000` budget never actually triggered the gap in a way that flipped a result, but
   the harness no longer depends on that being true by luck.

## The question

The extraction-head program refuted NNUE incrementality for **extraction candidates**: sibling
candidates at one e-class differ by a median **44.9%** of their edge multiset
(`docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md`), so an incremental accumulator over
extraction candidates buys ~2x, not the chess-engine ~98%
(`docs/plans/2026-08-17-cost-model-domain.md`, item A6).

The Guide (Phase 3, `docs/plans/2026-07-07-guided-saturation-redesign.md`) scores a **different**
object: not sibling extraction candidates, but the e-graph state as rewrite rules fire during
saturation, which is monotone and append-only. Does the append-only structure actually buy back
the incrementality the extraction study refuted, or does 44.9%-style churn show up here too?

## Headline: yes, and the real (batched) algorithm makes the case even stronger than single-stepping did

1,512 expressions, 4,495,274 rule applications recorded (`Provenance::application_count()`,
i.e. every rule firing the production batched algorithm recorded — including ones that produced
no new state), 6,840 saturation rounds, 0/1,512 expressions hit a budget cap before quiescence
(`max_iterations=100`, `max_classes=3000`).

| Quantity | Value |
|---|---|
| Applications with **zero** node/edge delta (idempotent re-fires — see below) | **91.1%** (4,096,878 / 4,495,274) |
| Per-application edge-delta fraction, all applications (median / p90) | **0.00%** / 0.00% (floored by the 91.1% zero share) |
| Per-application edge-delta fraction, **state-changing applications only** (median / p90) | **0.14%** / 0.79% |
| Trend across a run, state-changing applications only (early / mid / late third, edge-delta median) | 0.18% → 0.06% → 0.05% — **confirms shrinking as the graph grows** |
| Cumulative-vs-incremental work ratio (pooled, Σedges_before / Σedges_added, all applications) | **14,422x** |
| Implied speedup, state-changing applications only (1/median) | **~728x** |
| Evals (candidate matches scored) per committed action | 10.41x |
| Stale rate (candidate scored, produced no committed rewrite action) | 90.4% |

**Read the "state-changing applications only" row as the number comparable to the extraction
study.** The pooled figures (median 0.00%, ratio 14,422x) are not wrong, but they are dominated
by literal no-ops: under the *real* batched algorithm, every already-resolved match a rule
re-offers (see "Why the deltas are this small" below) still gets recorded as a `Provenance`
application with zero nodes created, and 91.1% of the 4.49M recorded applications are exactly
this. That is itself the headline finding, not a nuisance to filter out — it means the majority
of what an incremental accumulator would ever be asked to process costs **literally zero** work,
which is a stronger form of incrementality than "small delta." The conditional row answers "for
the minority of applications that actually change the graph, how big is that change" — median
0.14%, a genuine incremental update — the number to contrast directly against the extraction
study's 44.9%: **state-changing saturation deltas are ~325x smaller than extraction-candidate
deltas**, and then 91.1% of all applications need no update at all.

## Why the deltas are this small: the domain difference is real, not an artifact

The extraction study compares **structurally unrelated sibling subtrees** at a decision point —
two different implementations of the same subexpression, which can differ arbitrarily. A rewrite
APPLICATION during saturation is instead almost always a **local, single-node edit**: of the
398,396 applications that create at least one node, the `nodes_added` distribution is
overwhelmingly 1 (325,458) or 2 (63,232), with 3+ rare (9,301 at 3 nodes, falling off sharply —
the heaviest bucket above 3 is 308 applications creating 7 nodes,
`AngleAddition`/`ExpandSquare`-shaped multi-node rules). A rule that creates at most 1-2 nodes
against a graph that has already grown to hundreds or thousands of nodes is, by construction, a
small fraction of the whole — the append-only structure isn't incidental to this result, it's the
entire explanation. This is also why the conditional trend is monotone (0.18% → 0.06% → 0.05%):
the numerator (one rule's fixed-size footprint) stays roughly constant while the denominator
(current graph size) grows, so the fraction falls as saturation proceeds.

**The 91.1% zero-delta share has a separate, equally structural explanation: idempotent
re-matching.** `commutative` is the clearest example (`pixelflow-search/src/math/algebra.rs`):

```rust
fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
    ...
    let (a, b) = node.binary_operands()?;
    if a == b { return None; }
    Some(RewriteAction::Create(ENode::Op { op: self.op, children: vec![b, a] }))
}
```
There is no check for "is `Add(b, a)` already unioned into this class" — `apply()` always fires
whenever the operands differ. The first firing genuinely merges the swapped form in; every
subsequent scan of the same (now-canonicalized) node still matches, `EGraph::add`'s memo hits the
already-existing node (zero new nodes), and the resulting union is a same-class no-op. Under the
real batched algorithm this happens at much higher volume than a single-stepped scan would
suggest, because every round re-applies *every* live match for *every* rule in one sweep, and
`commutative`/`associative`/`reverse-associative`/`even-negation`/`involution` — the rules whose
`apply()` has this unconditional-refire shape — are exactly the top of the per-rule application
table below (`commutative` alone: 875,458 + 627,324 + 29,120 + 27,598 applications across its
per-operator instances).

**This is a finding about the candidate-generation contract, not a harness bug** (confirmed
independent of the harness rewrite: the same shape was already visible, at smaller volume, in the
single-step draft's stale-rate finding). It directly answers "how many evaluations of
slightly-perturbed states per step, and how many of those evaluations were ever real work": for
this rule library, under the algorithm production actually runs, the large majority of the
candidate pool at every round is not new opportunity, it is idempotent already-applied structure
re-offering itself.

## The eval-count economics: a second, separate cost the accumulator win does not solve

14,422x (or 728x, conditional) incremental savings on *accumulator maintenance* does not mean a
Guide's total per-round work drops by the same factor, because most of what a Guide would score
every round is not a live candidate at all. `evals_per_applied` = 10.41x system-wide (worse at
higher iteration counts: eval_count median 153/round, p90 2,276/round, while applied actions
median is only 10/round) — **90.4%** of scored candidates (4,495,274 evals, 431,787 committed
`SaturationStats::total_unions` actions) produce no committed rewrite. This is the same
phenomenon as the 91.1% zero-delta share above, viewed from the "candidate scored" side instead
of the "application recorded" side (the two numbers differ slightly — 90.4% vs 91.1% — because
`total_unions` counts distinct committed actions per round while the per-application walk counts
individual `Provenance` records; both describe the same idempotent-refire mechanism).

Closing this requires **candidate-level deduplication** (skip a `(rule, canonical class
content)` pair once it has fired and produced no change), not a cheaper per-candidate score —
unchanged from the original draft's conclusion, now backed by a larger, more representative
measurement.

## Method notes (current, batched-algorithm harness)

- **Iteration-level driving uses the real production algorithm, not a reimplementation.** Each
  round calls `EGraph::saturate_with_limits(1, max_classes, big_timeout)` — one outer round of
  the exact batched algorithm `EGraph::saturate()`/`saturate_with_budget` drive in production,
  interrupted after every round instead of run to completion in one call. `big_timeout` (3600s)
  is a backstop never approached; the real stopping conditions are `stats.total_unions == 0`
  (quiescence — a diagnostic condition, not a certified fixpoint; this optimizer is budget-only
  by design), `--max-iterations` (100), or `--max-classes` (3000).
- **Per-application attribution is reconstructed post-hoc from `Provenance`, exactly, with no
  extra e-graph walks during the run**: `Provenance::origins()` (added for this measurement,
  purely additive) maps every `ENodeId` ever minted to the `ApplicationId` that created it (or
  `Seed`); replaying `applications()` in firing order while accumulating each application's
  attributed node/edge counts reconstructs the exact graph size at every point in the run. This
  is exact for `nodes_added`/`edges_added`. It is **not** exact for unions — `UnionEvent` never
  records an `ApplicationId` — so unions are reported per **iteration** instead (exact at that
  granularity) via `Provenance::union_count()` before/after each round, alongside
  `SaturationStats::total_unions` (the narrower "distinct committed rewrite actions" count,
  which excludes congruence-only merges the round's closing rebuild discovers — 0.2% of all
  merges, 1,057/432,844, so this distinction barely matters in practice).
- **A rejected design: single-stepping goes stale fast, but measures the wrong thing.** An
  earlier version of this harness scanned `find_rewrite_matches()` once per round and fired every
  match in that snapshot serially via `EGraph::apply_single_rule` — a 20-expression pilot of that
  design measured a 98.6% stale rate (after the first application in a round, nearly every other
  match in the same stale snapshot no longer applies). That is a real, useful negative result
  about naive one-at-a-time firing without a rescan between decisions, but it is a property of
  that specific firing strategy, not of "standard saturation" — the batched algorithm above is
  what production runs, so it is what this binary reports.
- **Budgets are harness safety valves, not claims about production.** `--max-iterations` (100),
  `--max-classes` (3000) exist only to bound this harness's own wall-clock; every one of them is
  a count, never a duration, and 0/1,512 expressions in this run actually needed either cap
  (all reached `stats.total_unions == 0` quiescence on their own).

## Rule table (top 20 by application count, corpus-wide, current run)

| rule | applications |
|---|---:|
| commutative | 875,458 |
| fma-fusion | 704,289 |
| commutative | 627,324 |
| reverse-associative | 341,608 |
| associative | 315,068 |
| reverse-associative | 256,007 |
| associative | 239,786 |
| factor | 179,492 |
| constant-fold | 166,445 |
| distribute | 152,079 |
| identity | 129,395 |
| even-negation | 102,610 |
| halving | 60,877 |
| involution | 60,400 |
| annihilator | 47,528 |
| doubling | 39,596 |
| commutative | 29,120 |
| commutative | 27,598 |
| power-sqrt | 20,599 |
| even-negation | 14,262 |

(Repeated names are distinct rule instances registered per-op, e.g. `Commutative::new(&ops::Add)`
vs. `Commutative::new(&ops::Mul)` — same `name()`, different `rule_idx`, matching
`guide_headroom`'s convention.)

## Design implications for Phase 3

1. **Build the Guide's state scorer on the incremental `GraphAccumulator`, not a from-scratch
   rebuild — this measurement's strongest possible confirmation.** 91.1% of recorded applications
   need zero accumulator work, and the remainder need a ~0.14%-of-graph-size update (~728x
   cheaper than a rebuild). `add_edge`/`remove_edge` (`accumulator.rs`) are already the right
   shape for this.
2. **Candidate-level deduplication is the more urgent of the two costs, not accumulator
   maintenance.** An O(1)-per-edge incremental accumulator update is still 100% wasted when the
   candidate it scores is one of the 90.4% that produce no committed action. Recommend Phase 3's
   candidate generation track "has this `(rule_idx, canonical class content)` already fired and
   resolved" — a per-class visited-set keyed by a structural fingerprint, invalidated only when
   the class's content actually changes — so idempotent rules like `commutative`/`associative`/
   `even-negation`/`involution` stop being re-offered as live candidates after their first
   (successful) firing.
3. **Report zero-delta share explicitly in any future version of this measurement**, not just a
   pooled median — a median computed over a population that is 91.1% exact zeros is technically
   correct but uninformative on its own; the finding is the 91.1% share itself, not "the median is
   0."

Full per-expression aggregate data: `docs/results/2026-08-30-guide-scope-saturation-delta.json`.
