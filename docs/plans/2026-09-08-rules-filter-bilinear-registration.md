# Registration: the bilinear rules × nodes filter, on real shaders

**Date:** 2026-09-08
**Status:** REGISTERED — written before any filter was trained, before any
filtered run existed. Nothing below is revised after the first trained run
except to append results against the decision rule; a change of parameters
is a new registration that supersedes this one and says why.
**Seam:** `pixelflow-search/src/egraph/filter.rs` (PR #1228,
docs/plans/2026-09-08-rules-by-nodes-filter.md) — `F : M → M'` over the
rules × e-classes match matrix, one row per rule per round, run after
matching and before application; `KeepAll` is the identity and is what
production runs, byte-identically (210 / 210 shipped kernels).
**Authority:** JP, 2026-09-08 — *"take what's in that black box and make the
egraph output better."* Rules are domain-conditional (*"Pythagorean
identities are useless for Béziers and essential for spherical
harmonics"*), which is why the thing under test is bilinear over a rule
embedding **and** an e-class embedding, and why a per-rule prior is a
control rather than a candidate.
**Corpus:** docs/plans/2026-09-07-benchmark-correction.md §B (branch
`claude/benchmark-correction`; the DEV / HELD-OUT split is restated in §2
so this document stands on its own).

## 1. The question

On real scenes the rules expand 110–465 input nodes 10–40× to the class cap
in 2–5 rounds, and only ~1.2 % of applications are strictly load-bearing
(~20 % by the tight bound; docs/results/2026-09-07-egraph-off-vs-on-real-
shaders.md). A filter that keeps the load-bearing `(rule, class)` pairs and
drops the rest reaches the same or a better extraction on a fraction of the
application budget — and, because a dropped cell mints no class, it stays
under the class cap for more rounds, which is the limit the program wants to
scale past. This registration asks whether a CPU-resident bilinear head can
be that filter, on the shipped kernels, against three controls.

## 2. Environment (fixed)

| Item | Value |
|---|---|
| Loop | `EGraph::saturate_bounded` through `Optimizer::filter`, `Budget::Applications(b)` with the production tier's round and class caps as backstops (`Budget::limits`). No clock anywhere in the decision. |
| Rule set | `RuleSet::production()` (62 rules), production order. |
| Cost model | `CostModel::latency_prior()`; extraction `extract_dag_scoped` at the kernel's own `LatticeShape` — the production extractor. Every cost below is **`dag_cost`** of the extracted arena (the sum of the latency prior over the reachable nodes, shared nodes counted once — `egraph_off_on`'s column), never tree cost. |
| DEV | DejaVu Sans Mono glyph bakes (95 printable ASCII × tile 16 and tile 32) plus the three `font_rendering` bench glyphs and `O`@32 wide; the twelve `shader_bench` ports; `bench_scene_psychedelic` (packed, with its clock uniform); the terminal cell grid at 80×24 @2×. That is the `egraph_off_on` corpus minus its two chrome rows: 208 kernels. NotoSansMono-Regular bakes (DEV per §B.2) are added to the *evaluation* if the run completes in the session; they are not in the training set either way. |
| HELD-OUT | NotoSansMono-**Bold** glyph bakes (both tiles, 190) and `bench_scene_chrome` (packed and its red channel). Opened exactly once, at the end, with the model trained on all of DEV; the numbers are reported beside DEV, never pooled, and never used to choose anything. |
| Families | glyph16, glyph32, bench, bench_wide (all DejaVu), shader, psychedelic, cellgrid. For the family-held-out protocol (§5) the four DejaVu families are one fold (**glyph** — glyph32 is byte-identical to glyph16 on 94 / 95 glyphs, so holding one out while training on the other holds nothing out), the shaders are a fold (**shader**), and psychedelic + cellgrid are a fold (**scene**). |
| Per-kernel budget `B` | The applications the *unfiltered production run of that kernel* fires — `Optimized::stats.applications` under `Optimizer::production()`, read per kernel from the same process that then evaluates it. `Identity` at `Budget::Applications(B)` therefore reproduces production by construction, and that is asserted (dag_cost and bytes), not assumed. The registered budgets are `B`, `B/2`, `B/4`, with `B/8` measured for the dual curve only. |
| Determinism | Every arm is a deterministic function of the kernel, the budget and the checkpoint; `UniformRandom` is a hash of `(seed, rule, class, applications-so-far)`, never an RNG. Rows carry the git sha and the ISA tier. |
| Load | The clock is a sign only: the filter's own wall share is reported at 1-minute load < 8 or "not taken". Deterministic columns are the claim. |

## 3. The thing under test, and the controls

All four are `ApplicationFilter`s over the same seam; only the score
differs. Each arm keeps a cell iff its score clears a threshold, and the
threshold is set so the arm keeps the **same fraction ρ of cells over the
training samples** — the keep-rate is the one knob, and it is equal across
the three non-identity arms so the comparison is about *which* cells, not
how many. Registered keep-rates: **ρ = 0.25 (primary)**, ρ = 0.5
(secondary). Thresholds are the `(1 − ρ)` quantiles of each arm's score over
the training samples, computed once and carried in the checkpoint manifest.

| arm | score of cell `(r, c)` | what it can express |
|---|---|---|
| `Identity` (`KeepAll`) | — keeps everything | production |
| `PerRuleRate` | `P_train(load-bearing \| r)` — the per-rule positive rate on the training samples | a per-rule prior only: the same rule is kept or dropped in every context. This is the arm JP says cannot work because rules are domain-conditional |
| `UniformRandom` | hash-uniform in `[0, 1)` of `(seed, r, c, applications so far)` | nothing — the same keep-rate with no information; separates "fewer applications" from "the right applications" |
| **`Bilinear`** | `s(r, c) = m(x_c)ᵀ W e_r + bᵀ e_r` — `SaturationHead::score_candidate` on the existing candidate tower, `e_r` the rule embedding derived from the rule's templates (`BilinearCandidateGuide`'s encoder, unchanged), `x_c` the cell's context (§4) | a rule × context interaction: the same rule ranked differently in different neighborhoods |

**Threshold, not top-k.** Rows arrive streamed (rule `r+1` matches against
rule `r`'s committed row — the seam doc's matching-semantics section), so
"top-k per epoch" would need the epoch's matrix before its first commit,
which changes what the matrix is. A per-cell threshold needs nothing the
seam does not already hand the filter. Registered.

## 4. The cell's features — one definition, train and deploy

A cell is featurized by `nnue::guide::filter::CellContext::observe(graph,
class, episode)` at the moment the filter sees the row — the graph exactly
as the row's bindings refer to it — and the *same function* is what the
recording filter calls when the training samples are minted. That is the
whole point of minting inside the seam rather than from the final graph:
`CandidateFeatures::observe` reads whatever graph it is handed, and the
earlier pipelines handed it the fully-saturated graph, so their features
described a class as it ended, not as it was matched.

The context is what the existing tower already reads, nothing new:
`neighborhood_ops` (the one-hop child ops of every node in the class, with
multiplicity), `match_class_node_count`, `expr_node_count` (the kernel's
reachable input node count — an episode constant), and `budget_fraction` =
applications committed so far ÷ the tier's production application cap
(`SaturationConfig::max_applications`: 20 000 / 80 000 / 200 000) — the
denominator is fixed per tier so a run at `B/2` reads the same units as the
production run the labels came from.

Inference cost is counted, not estimated: the filter reports cells scored,
cells kept, and the multiply-adds it performed (`FilterStats`), and the
evaluation reports the filter's wall share of saturation per family.

## 5. Labels and training

**Labels.** DEV under `Identity` at the production budget with
`--features provenance-journal`; every committed application is labeled by
`EpisodeLabels::compute_strict` **and** `compute_tight` against the
production extraction's choices. The join between a minted cell and its
provenance record is by application ordinal and is asserted (rule and match
root must agree) — a silent mismatch would train on the wrong row.

**Training label: tight.** Strict credit is minted credit and is blind to a
union-only rewrite (docs/results/2026-09-01-strict-label-constant-output-
blindspot.md); a filter trained on it would drop the enabling rewrites —
`commutative(Add)` has 0 strict and 65 912 tight applications on the real
corpus — and the extraction would lose what they enable. Tight is the
narrowest safe bound. Strict is reported beside it, with the fraction of
tight positives strict cannot see.

**Model and loss.** `BilinearTrainer` (the existing head, the existing
backward pass with its gradient check) as a classifier
`P(tight | r, c) = σ(s(r, c))`, weighted BCE with `pos_weight =
negatives / positives` on the training samples, cold start, no warm start.
Samples per kernel are stride-capped so no family dominates by application
count; the cap and the resulting counts are in the results doc.

**Protocol — family-held-out.** Three models, one per fold of §2, each
trained on the *other* folds' DEV kernels; the arm evaluated on a family is
the model that never saw that family. A fourth model trained on all of DEV
is used only for the HELD-OUT opening and for an in-family upper bound that
is reported and not claimed. The intrinsic metric is ROC-AUC and PR-AUC on
the held-out fold's samples, for tight and for strict, beside the same two
numbers for `PerRuleRate` (the per-rule prior as a scorer) — a bilinear
intrinsic that does not beat the per-rule prior's has no business claiming
an extrinsic win.

## 6. The metric (verbatim, pre-committed)

At each registered budget `b ∈ {B, B/2, B/4}` and each family, for every
arm, per kernel: extracted `dag_cost`, emitted machine-code bytes, and
guarded schedule entries (`PIXELFLOW_GUARD_TELEMETRY`), each as a ratio to
`Identity` at the same `b` on the same kernel. Reported as median and
p10 / p90 over the family, with the per-kernel CSV.

**The dual:** applications needed to reach `Identity`'s cost at `B` — per
kernel and arm, the smallest `b ∈ {B/8, B/4, B/2, B}` at which the arm's
`dag_cost` is ≤ `Identity@B`'s, or "never" in the grid.

## 7. Decision rule (pre-committed)

The bilinear filter is a **win on a family** iff, at `B/2`, ρ = 0.25,
under the family-held-out model:

1. median `dag_cost(Bilinear) / dag_cost(Identity)` ≤ **0.95**, and
2. median `bytes(Bilinear) / bytes(Identity)` ≤ **1.00** (bytes no worse), and
3. the HELD-OUT families agree in **sign** — median dag_cost ratio < 1 on
   NotoSansMono-Bold and on chrome, read once at the end with the all-DEV
   model.

Everything else is a **null** and is reported as one. In particular:

- if `UniformRandom` at the same ρ matches `Bilinear` within the family's
  p10 / p90 band, the effect is "fewer applications", not "the right ones",
  and the bilinear claim is null even if clause 1 holds;
- if `PerRuleRate` matches `Bilinear`, the interaction bought nothing over a
  per-rule prior and the bilinear claim is null; that result is reported as
  evidence *for* a per-rule prior on that family, against the prior
  expectation;
- a win that appears at ρ = 0.5 but not at ρ = 0.25 is the secondary result
  and is labeled so.

**What ships.** Whatever the verdict, `Optimizer::production()` stays on
`KeepAll`. If the win is real, the PR ships the trained weights as a
fixture; promotion is JP's call and needs each family's own primary metric
(§B.3 of the benchmark correction: row-prologue guarded fraction and µs/row
for glyphs, ns/px for scenes and shaders, µs/frame for the grid) and the
bytes column, which this registration does not claim.

## 8. Self-censoring

- The intrinsic AUC is reported on the held-out fold only; a within-training
  AUC is a memorization number and does not appear in the verdict.
- The HELD-OUT set is opened once, after every DEV decision is made and
  written down, with the model trained on all of DEV. The opening is
  journaled; NotoSansMono-Bold and chrome are promoted to DEV by it and the
  program owes new held-out members before its next claim.
- No number from the earlier Guide results
  (docs/results/2026-09-01-phase3-at-budget-eval, 2026-09-02-bilinear-guide,
  2026-09-02-guided-regression-bisect) is a prior for this one: they were
  minted on a synthetic corpus in tree units and are retracted in the
  claims ledger.
- The synthetic corpus is not run. A generated kernel is never a headline.
