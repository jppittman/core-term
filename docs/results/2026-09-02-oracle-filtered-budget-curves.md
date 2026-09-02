# Oracle-filtered anytime budget curves, on DAG cost: rule-granularity filtering moves the classical curves *less* than the tree-cost run claimed (2026-09-02)

**Supersedes** `docs/results/2026-08-30-oracle-filtered-budget-curves.{md,csv}` and
`docs/results/2026-09-01-oracle-filtered-budget-curves.{md,csv}`. Both are left in
place with a superseded-by banner; neither has been edited otherwise.

Reproduce:

```
cargo run --release -p pixelflow-search --example oracle_filtered_budget_curves
```

Output CSV: `docs/results/2026-09-02-oracle-filtered-budget-curves.csv`
(6,300 rows = 225 expressions x 2 curves x 14 application-count checkpoints).
Corpus and seed unchanged from both prior runs: 220 synthetic expressions across
10 depth bands (seed 20260830) + the 5 named realistic shaders. Rule library: 62
rules. Grid: `APP_CHECKPOINT_GRID` = 25 ... 204,800 cumulative applications.

---

## Why this re-run exists

#1117 established that `ExtractedDAG::total_cost` is the extraction DP's **tree**
cost — it pays a shared subterm once per use — and that the quantity a regret
curve is asking about is `dag_cost`, what the emitted kernel actually pays. Both
prior runs reported `total_cost` in their `cost` and `regret_pct` columns. The
harness on `main` reads `dag_cost`; the shared curve loop
(`pixelflow_search::egraph::anytime`) reports `ChoiceCost::dag`. This run is the
first to report the corrected quantity.

Three further instrument changes come with the forward-port and are stated once in
[docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md):
the application budget binds mid-scan (`app_actual == app_target` exactly), the
curve runs through `Optimizer` (#1108) rather than a private loop, and the oracle
rule set is derived from the **minimum-cost** checkpoint's extraction rather than
the final one (`extract_dag` is non-monotonic along a curve).

### One defect found and fixed in the course of this re-run

After #1118 the provenance journal is **opt-in**: `Optimizer::production()`
attaches no `Observer`, so an unobserved curve reports `recorded_count() == 0`
and `EpisodeLabels::compute` returns an empty `load_bearing` set. The first
attempt at this re-run therefore produced an oracle arm with **zero rules** —
which does not error, does not warn, and looks exactly like a strong null
result (`live=0` past B=25, a flat curve, and a plausible-looking regret
number). The harness now attaches `observe(Some(KeepJournal))` to the two arms
whose journal it reads, and asserts loudly that a graph which committed
applications recorded some. Numbers below are from the observed run.

---

## Finding 1: the curves still have shape, at the same place

| scope | n | curves with >1 distinct cost | first-to-final gap (median) |
|---|---:|---:|---:|
| ALL | 220 | 130 (59.1%) | 7.14% |
| blitz | 23 | 0 (0.0%) | 0.00% |
| rapid | 97 | 31 (32.0%) | 0.00% |
| classical | 100 | **99 (99.0%)** | 30.46% |

Unchanged in kind from the 09-01 run (which also found 99/100 classical): the
anytime question is a deep-band question. Run end status, unguided arm: 156/220
quiesced, 64 hit the class cap, 0 exhausted the grid, 0 hit the sweep ceiling;
classical median ended-at-applications 10,296 (p90 17,997).

The **level** moved a lot, because tree cost inflates exactly where sharing is
exposed. Classical median regret, unguided arm:

| B | 25 | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 | 6400 | 12800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TREE (09-01) | 81.88 | — | 94.92 | 52.24 | 35.26 | 0.89 | ~0 | ~0 | ~0 | ~0 |
| **DAG (this run)** | **43.81** | 43.65 | **43.24** | **34.46** | **33.79** | **29.18** | 17.94 | 1.36 | 0.13 | **0.00** |

Two things follow. The tree-cost curve's headline non-monotonicity (regret
*rising* from B=25 to B=100, 81.9 -> 94.9) **does not exist** on DAG cost: the
corrected curve is monotone non-increasing at the median across the whole grid.
It was an artifact of the tree metric rewarding whatever sharing saturation
happened to expose at a given checkpoint. And the decay is much slower than the
tree-cost run reported — real regret is still 29% at B=800 and does not reach 0
at the median until B=12,800, an order of magnitude past where the tree-cost run
said the problem was solved. The anytime headroom is *larger* than Round 1
believed, not smaller.

## Finding 2 (the load-bearing one): rule-granularity oracle filtering is now indistinguishable from unguided

Restricting the rule library to the hindsight-load-bearing set `R*` per
expression (median |R*| = 3 / 7 / 22 of 62 rules for blitz/rapid/classical;
min 0, max 42 at the classical tier) leaves the curve essentially untouched.

| classical median regret% | B=25 | B=100 | B=200 | B=400 | B=800 |
|---|---:|---:|---:|---:|---:|
| unguided | 43.81 | 43.24 | 34.46 | 33.79 | 29.18 |
| oracle-filtered | 43.81 | 43.24 | 34.46 | 33.79 | 28.06 |
| **delta** | **0.00** | **0.00** | **0.00** | **0.00** | **-1.12** |

Paired, per expression, classical tier (n=100):

| B | oracle strictly better | strictly worse | identical | mean delta (pts) |
|---|---:|---:|---:|---:|
| 25 | 1 | 0 | 99 | -0.30 |
| 100 | 2 | 0 | 98 | -0.31 |
| 200 | 2 | 0 | 98 | -0.33 |
| 400 | 6 | 0 | 94 | -0.26 |
| 800 | 6 | 0 | 94 | -0.16 |
| 1600 | 9 | 0 | 91 | -0.09 |
| 3200 | 12 | 1 | 87 | -0.16 |

**1,329 of 1,400 classical (expression, checkpoint) cells have a bit-identical
DAG cost across the two arms (94.9%).** The median delta is exactly 0.00 points
at every checkpoint on the grid, and the mean never exceeds a third of a point
before B=400.

Whole-corpus medians tell the same story: ALL-scope unguided 7.69 / 3.79 / 2.05
at B=25/50/100 against oracle 6.55 / 3.50 / 1.91, and 0.00 for both from B=200.

## Verdict on the conclusion that justified candidate-granularity guidance

**It survives, and the correction strengthens it.**

The 09-01 report cited a 94.9% -> 85.4% drop in classical median regret at B=100
as the *entire visible benefit* of a perfect rule-level oracle, and argued that
~9 points was too little to build a Guide on. On DAG cost that 9-point gap is
**not there at all**: unguided and oracle-filtered are both 43.24 at B=100, and
98 of 100 classical expressions extract a bit-identical kernel under the two rule
sets. The gap the tree-cost run reported was itself an artifact of the tree
metric — it was measuring how much sharing each arm happened to expose, not how
good either arm's kernel was.

So the design premise holds a fortiori: dropping non-load-bearing **rules** does
not shorten the path to a good extraction, because the surviving rules re-fire
their idempotent matches and explore in the same order — the work distribution
barely changes. Combined with the scoping round's dedup finding (90.4% of scored
candidates commit nothing), the implication for the Guide is the one the design
revision (§4) already drew, now on the corrected instrument: **the buy-back has
to come from candidate-granularity discrimination — which match, at which
e-class, now or never — not from rule masking.** The Phase 3 registration records
this as non-binding pre-flight context (§8), and nothing in the registration
needs to change.

One caveat worth naming rather than burying: because real regret decays an order
of magnitude more slowly than the tree-cost run reported (29% at B=800, not
~1%), the *room* a candidate-granularity Guide has to work in at production-like
budgets is bigger than Round 1's numbers implied. That raises the ceiling on
what a Guide could win; it does not change what rule filtering cannot win.

## Diagnostics recorded this run

- Non-direct-creator share of load-bearing applications: **98.73%** overall
  (611,582 / 619,426); 87.30% blitz, 97.46% rapid, 98.76% classical. This does
  **not** isolate the class-membership over-approximation specifically — genuine
  transitive-enabler credit and all three named over-approximation axes are
  lumped together (the PR #1067 renaming, kept).
- Shaders in-sample check: 4 of 5 named shaders end at 0.00% regret on both arms
  and are in-sample for their tier; `shader_circle_sdf` (18 nodes, rapid) ends at
  10.81% unguided / 0.00% oracle and is out-of-sample against a synthetic rapid
  range of [0.00%, 0.00%] — the one place in this run where the oracle arm
  visibly beats unguided at the end of a curve.
- One classical expression has |R*| = 0 (its minimum-cost extraction used no
  rewritten node), so its oracle arm is the input expression unmodified. It is
  included in every figure above.
