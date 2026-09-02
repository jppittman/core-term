# How much is the greedy extractor leaving on the table?

Measurement only. Nothing in production changed: same rule set, same `Budget::Production`, same `CostModel::latency_prior()`, same `extract_dag`.

Context: the defects measured here are issue #1111 and §1.L5 of `docs/plans/2026-09-02-optimizer-api.md`; the budget trades whose sequencing they decide are #1101 (numeric-first rule order), #1109 (removing the class-cap break) and #1114 (live-counted class budget). **The decision input is [§ The decision input](#the-decision-input);** everything after it is the evidence.

## Headline

**The measurable half needs no time limit.** Knuth's algorithm gives the exact minimum TREE cost — the objective `extract_dag` names — in polynomial time, on every kernel. Against it, over ALL **302** kernels with nothing excluded:

- `1 - treeopt/greedy` (DAG cost of each returned term): median **0.000%**, Q1 0.000%, Q3 0.000%, p90 **0.403%**, worst **14.557%**.
- The two choosers tie on **230** kernels; greedy is ahead on **4** (possible, and honest to report: the tree-optimal term is optimal for TREE cost, and neither chooser optimizes the DAG cost both are scored by).
- On its own objective, with no DAG argument in play, the DP misses the tree optimum by median **0.000%**, p90 **3.258%**, worst **26.789%**, and attains it on **230/302** kernels.

The NP-hard half, with a time limit:

- Kernels: **302** (206 real dumps, 96 synthetic).
- Exact optimum proved on **89**; **213** UNSOLVED within 4s and excluded from the gap statistic.
- `cost_greedy / cost_exact` on solved kernels: median **1.0000**, Q1 1.0000, Q3 1.0000, p90 **1.0000**, worst **1.1212**.
- Greedy is **exactly optimal on 86 of 89 solved kernels**.
- Certified floor across ALL kernels (an unsolved search still returns a term, which upper-bounds the optimum): median `greedy/best-found` **1.0000**, strictly beaten on **81**.

## The decision input

This section is the part a sequencing decision needs: how big the gap is, which
of the three defects is worth fixing first, and whether that should happen
before or after the three budget trades already on the table — the numeric-first
rule order (#1101), removing the class-cap break (#1109), and the live-counted
class budget (#1114). The defects themselves are #1111 and §1.L5 of
`docs/plans/2026-09-02-optimizer-api.md`.

### 1. The headline gap, and what it is not

There are two optima, and only one of them is measurable at production scale.

| reference | scope | pooled recovery vs greedy | per-kernel |
|---|---|---|---|
| **tree optimum** (Knuth, exact, polynomial, no time limit) | **all 302, nothing excluded** | **0.258%** (real-only 0.284%; shaders 2.62%) | median 0.000%, p90 0.403%, worst **14.557%**; strictly better on **68**, greedy ahead on 4 |
| **DAG optimum** (branch and bound, 4s cap) | **89 of 302 proved** | 0.228% on those 89 | median ratio 1.0000, greedy already exactly optimal on **86 of 89**, worst **1.1212** |
| **certified floor** (cheapest term the search actually exhibited; a valid upper bound on the optimum even when UNSOLVED) | all 302 | 0.286% | strictly beats greedy on **81**; on those 81 median 0.428%, p90 5.727%, max 14.557% |

**The headline number is 0.26% pooled / 14.6% worst case, and it is a floor.**

Three caveats that have to travel with it:

1. **Which optimum.** The 0.26% is against the *tree* optimum — the objective
   `extract_dag` names. The true DAG optimum is lower, but it is NP-hard and
   only closed on 89 kernels.
2. **What fraction was solvable.** Exact DAG extraction stops being computable
   at roughly 100–200 reachable e-classes. A production glyph saturates to a
   median of **1,755** reachable classes, so 213/302 are UNSOLVED and excluded
   from the exact statistic. The exact number is measured where it is
   measurable, which is not where production lives — that is precisely why the
   tree-optimal reference, which has no such limit, is the decision-relevant one.
3. **The size boundary cuts the mechanism split in half.** All 68 kernels where
   the tree optimum beats greedy are **real** (64 glyph, 4 shader) — **zero
   synthetic**. The synthetic ladder is where most of the 89 exact solves live.
   So the two references are describing different populations, and any single
   ratio that averages them is misleading.

For scale: #1114 buys **+2.03% for 2.00x rule applications**. Pooled, 0.26% is
smaller than that. In the tail it is not — 2.62% pooled on shaders, 5.6% on
`shader:julia_set` at the production budget with no budget change at all, and
10.5% on the two `U+004B` glyphs #1114 regresses.

### 2. The mechanism split, and what each fix recovers

| # | defect (§1.L5 / #1111) | share of the proved loss | what fixing it recovers | cost to fix |
|---|---|---|---|---|
| (i) | single DFS, not a fixpoint — a class whose child is `on_stack` is scored `CYCLE_COST` and never revisited | 10 of 195 units (5%) | **0.258% pooled over all 302, p90 0.403%, worst 14.557%, 68 kernels strictly improved** — and it settles both `U+004B` regressions (1132 → 1013) | small: a worklist to fixpoint replacing one DFS |
| (ii) | tree cost, not DAG cost — sharing is never priced | 185 of 195 units (95%) | **0.028% pooled beyond the tree optimum** (real-only 0.013%); 17 kernels improved further | large: NP-hard; needs a DAG-aware heuristic, not the reference |
| (iii) | `total_cost` read at `extract.rs:1636` before `repair_choices_well_founded` mutates the choices at `:1639` | 0 units | **no cost at all** — it is an evidence bug, not a speed bug | trivial: recompute after repair |

**The 95/5 split is a small-kernel statement and does not survive to production
scale.** It is computed only where the branch and bound closed, i.e. on small
e-graphs — where the DFS rarely stumbles and unpriced sharing is the whole
story. Over the full corpus the picture inverts: the tree optimum is worth
0.258% pooled and the entire remaining certified headroom beyond it is 0.028%.
Both statements are true; they describe different kernels.

**Fix (i) first.** It is the largest measurable recovery at production scale, it
is the cheapest of the three, it is what the #1114 regressions actually are, and
it is a prerequisite for (ii) — a DAG-cost DP that is still a single DFS just
produces a different set of `CYCLE_COST`-poisoned classes. Cycle-priced classes
are not rare: **238 of 302** kernels have at least one (median 10, max 284), and
on **203 of 302** the best term the search holds *uses* a class the greedy DP
priced at `CYCLE_COST` — a floor, since on an UNSOLVED kernel that term is only
the seed.

**Do not chase (ii) next.** The measurement's most useful negative result is
that exact DAG extraction is worth **0.028% pooled** beyond the tree optimum on
this corpus. That does not license shipping a branch and bound in the compiler,
and it says the sharing-aware objective is a research direction, not a fix.

**(iii) is worse than "stale", and free to fix.** On the 132 of 302 kernels
where `total_cost` fails to describe the returned term, **not one** is an
ordinary-magnitude discrepancy: **92** report exactly `usize::MAX` (the
`.unwrap_or(usize::MAX)` at `:1636`, taken when the single DFS never resolves
the root class) and **40** report 1x, 2x or 3x `usize::MAX / 4` — the `Dwrt`
sentinel from `cost.rs:292`, summed into a number typed as a cost. On those
kernels `total_cost` is not a slightly-wrong cost; it is arithmetic garbage
wearing a cost's type, and it is what a naive A/B would read.

No shipped result is known to be corrupted by it, and that is worth stating
precisely rather than leaving as a worry: the harnesses that decided #1101,
#1109 and #1114 re-score the returned arena (`arena_static_cost`,
`runtime.rs:1278`) instead of reading `total_cost`, and the one live consumer
that does read it — `guide_headroom.rs:395` — ran on an 800-expression corpus
containing no sentinel value. So (iii) is a live trap, not a live wound. It
should still be fixed first in wall-clock order, because it costs nothing and it
is the reason a future A/B could be silently wrong.

### 3. Sequencing — one sentence

**Fix extraction before taking any of the three budget trades:** the fixpoint
fix is free at run time, recovers more on real kernels (0.284% pooled, 14.6%
worst) than #1114 buys for 2x the rule applications on the kernels where they
overlap, and — decisively — turns #1114's two `U+004B` regressions from
regressions into improvements, which no budget tuning can do.

### 4. The finding that settles it

On `glyph16:U+004B` and `glyph32:U+004B`, two of #1114's four regressions, the
roomier e-graph **provably contains a term costing 1013** while greedy returns
**1132** on it and **1047** on the smaller production graph. A larger e-graph
holds a superset of the equalities, so the optimum over it can only fall; the
graph got strictly better and the chooser gave up 8.1% anyway. **Those two
regressions are extraction failures, not budget failures**, and a chooser that
attained its own stated objective would have reported an improvement there. The
same reference shows `shader:julia_set` leaving **5.6%** on the table at the
production budget with no budget change in play.

The remaining regression, `psychedelic` (766 → 816), is **unresolved, not
exonerated**: no cheaper term was exhibited in the roomier graph inside the time
limit, and at 946 reachable classes it is well past where the exact reference
stops. It does not license the trade and it does not condemn it.

### 5. What a fixed extractor would cost

**Estimate, not a measurement** — the probe does not time the Knuth pass
separately, and nothing here was benchmarked as production code.

The fixpoint version of `extract_dag` is Knuth's algorithm: a Dijkstra over the
AND-OR graph in which each e-node is relaxed once per settled child, i.e.
`O(sum of arities * log V)` with a binary heap, against the current single
DFS's `O(sum of arities)`. A small constant and a log — not an order. What the
run does pin: the Knuth reference completed on **all 302** kernels including the
4,847-class `psychedelic` and the 5,226-class `U+0040` glyph, inside a probe
whose per-kernel wall time was dominated by the 4s branch and bound. It is not
the expensive part.

**The asymmetry is the argument.** Extraction runs **once per compile**.
Saturation runs thousands to a million rule applications per compile — the
guide-headroom corpus is **8,729,067 applications over 800 expressions** (median
194.5, p90 31,197, max 996,047) — and #1114 proposes to double that for +2.03%.
A log factor on a once-per-compile pass, buying 0.28% pooled and up to 14.6% on
individual real kernels, is a different kind of trade from 2x on the pass that
already dominates compile time. **This is the cheap place to spend.**


## The mechanism split

*(Evidence for §2 above: how the decomposition is defined and why it is exact.)*

The loss decomposes exactly, because the middle term is itself an argmin:

```text
greedy - exact = (greedy - tree_optimal) + (tree_optimal - exact)
```

`tree_optimal` is Knuth's algorithm — Dijkstra generalized to AND-OR graphs — which computes the exact minimum TREE cost in polynomial time. That is the objective `extract_dag` *names*, so the first term is the DP failing its own objective (single DFS + `CYCLE_COST`, defects 2 and 3) and the second is that objective being the wrong one (sharing unpriced, defect 1). Both terms are measured as the true DAG cost of the term each chooser returns.

| share | pooled cost units (solved kernels) |
|---|---|
| total loss, greedy - exact | 195 |
| (i) DP not attaining its own tree objective | 10 |
| (ii) tree cost being the wrong objective | 185 |

(iii) **cost reported before repair** (issue #1111, `extract.rs:1636` vs `:1639`): `total_cost` fails to describe the returned term on **132 of 302** kernels. Per-kernel magnitudes are in `reported_delta`.

Cycle handling, separately: the greedy DP priced at least one class at `CYCLE_COST` that the best term the exact search holds nevertheless uses, on **203 of 302** kernels. On an unsolved kernel that term is the seed, so this is a floor.

## Where exactness stops being computable

Minimum-cost extraction with sharing is NP-hard, so the reference is a branch-and-bound over per-class node choices with a per-kernel budget (4s wall, expansion-capped), seeded with the better of the greedy and tree-optimal terms. It reports `UNSOLVED` rather than a truncated answer.

| category | kernels | solved | median reachable classes (solved) | median reachable classes (unsolved) |
|---|---|---|---|---|
| glyph | 190 | 10 | 113 | 1755 |
| shader | 12 | 5 | 104 | 474 |
| psychedelic | 1 | 0 | n/a | 946 |
| cellgrid | 3 | 0 | n/a | 1261 |
| synthetic | 96 | 74 | 26 | 362 |

## Is a bigger e-graph monotonically better?

A larger e-graph holds a superset of the equalities, hence a superset of the extractable terms, so the **exact** optimum over it can only fall. Any kernel whose *greedy* cost rises when the budget is loosened is therefore an extraction failure, not a budget failure.

| kernel | budget | classes | stop | greedy | tree-optimal | best known | status |
|---|---|---|---|---|---|---|---|
| glyph16:U+004B [production] | production | 4578 | ClassCap | 1047 | 1047 | 1047 | UNSOLVED |
| glyph16:U+004B [roomier] | roomier | 16774 | ClassCap | 1132 | 1013 | 1013 | UNSOLVED |
| glyph32:U+004B [production] | production | 4578 | ClassCap | 1047 | 1047 | 1047 | UNSOLVED |
| glyph32:U+004B [roomier] | roomier | 16774 | ClassCap | 1132 | 1013 | 1013 | UNSOLVED |
| psychedelic [production] | production | 4847 | ClassCap | 766 | 766 | 766 | UNSOLVED |
| psychedelic [roomier] | roomier | 15387 | ClassCap | 816 | 816 | 816 | UNSOLVED |
| shader:julia_set [production] | production | 4533 | ClassCap | 716 | 676 | 676 | UNSOLVED |
| shader:julia_set [roomier] | roomier | 17076 | ClassCap | 678 | 676 | 676 | UNSOLVED |

**Verdicts.**

- `glyph16:U+004B`: production greedy 1047, roomier greedy 1132, cheapest term exhibited in the roomier graph 1013 — **extraction failure** — the roomier e-graph demonstrably CONTAINS a cheaper term than production's own output, and the greedy chooser nevertheless returned a more expensive one than it did on the smaller graph.
- `glyph32:U+004B`: production greedy 1047, roomier greedy 1132, cheapest term exhibited in the roomier graph 1013 — **extraction failure** — the roomier e-graph demonstrably CONTAINS a cheaper term than production's own output, and the greedy chooser nevertheless returned a more expensive one than it did on the smaller graph.
- `psychedelic`: production greedy 766, roomier greedy 816, cheapest term exhibited in the roomier graph 816 — greedy regressed, and no cheaper term was exhibited in the roomier graph within the time limit — unresolved, not exonerated.
- `shader:julia_set`: production greedy 716, roomier greedy 678, cheapest term exhibited in the roomier graph 676 — no regression at this budget step.

One caveat, stated because it is easy to misread the table: the **exact** optimum is monotone under a budget increase, but the *tree-optimal* column need not be. Knuth's algorithm minimizes tree cost; the term it picks is then scored here on DAG cost, and a roomier graph can hand it a term that is cheaper as a tree and shares less as a DAG. Where that column rises, it is that effect, not a violated law.


## Worst kernels by objective miss — the DP against its own tree cost

No time limit is involved in this table: Knuth's algorithm is exact and polynomial, so every kernel here is a settled number. `greedy DAG` and `tree-opt DAG` are the same two terms re-scored as DAGs, which is what the emitted code actually costs.

| kernel | category | greedy tree | tree-opt tree | miss | greedy DAG | tree-opt DAG |
|---|---|---|---|---|---|---|
| shader:julia_set | shader | 14151801 | 10360711 | 26.79% | 716 | 676 |
| glyph16:U+006D | glyph | 72339 | 61210 | 15.38% | 9624 | 9611 |
| glyph32:U+006D | glyph | 72339 | 61210 | 15.38% | 9624 | 9611 |
| glyph16:U+0077 | glyph | 46916 | 39718 | 15.34% | 6602 | 6589 |
| glyph32:U+0077 | glyph | 46916 | 39718 | 15.34% | 6602 | 6589 |
| glyph16:U+0032 | glyph | 50145 | 42461 | 15.32% | 6980 | 6967 |
| glyph32:U+0032 | glyph | 50145 | 42461 | 15.32% | 6980 | 6967 |
| glyph16:U+0042 | glyph | 57563 | 48774 | 15.27% | 7970 | 7957 |
| glyph32:U+0042 | glyph | 57563 | 48774 | 15.27% | 7970 | 7957 |
| shader:mandelbrot_distance | shader | 910369 | 775353 | 14.83% | 576 | 527 |
| glyph16:U+0068 | glyph | 34505 | 31276 | 9.36% | 5045 | 5036 |
| glyph32:U+0068 | glyph | 34505 | 31276 | 9.36% | 5045 | 5036 |
| glyph16:U+0052 | glyph | 31606 | 28650 | 9.35% | 4725 | 4716 |
| glyph32:U+0052 | glyph | 31606 | 28650 | 9.35% | 4725 | 4716 |
| glyph16:U+0044 | glyph | 31079 | 28180 | 9.33% | 4506 | 4497 |
| glyph32:U+0044 | glyph | 31079 | 28180 | 9.33% | 4506 | 4497 |
| glyph16:U+0057 | glyph | 140483 | 129208 | 8.03% | 13499 | 13494 |
| glyph32:U+0057 | glyph | 140483 | 129208 | 8.03% | 13499 | 13494 |
| glyph16:U+0030 | glyph | 72707 | 68325 | 6.03% | 9970 | 9926 |
| glyph32:U+0030 | glyph | 72707 | 68325 | 6.03% | 9970 | 9926 |

## Worst kernels by proved ratio

| kernel | category | reachable classes | greedy | tree-opt | exact | greedy/exact |
|---|---|---|---|---|---|---|
| synth_d11_s4 | synthetic | 89 | 1665 | 1665 | 1485 | 1.1212 |
| shader:smooth_min_scene | shader | 160 | 136 | 126 | 126 | 1.0794 |
| synth_d4_s7 | synthetic | 53 | 614 | 614 | 609 | 1.0082 |
| glyph16:U+0020 | glyph | 1 | 0 | 0 | 0 | 1.0000 |
| glyph16:U+002D | glyph | 113 | 133 | 133 | 133 | 1.0000 |
| glyph16:U+0040 | glyph | 5226 | 23527 | 23527 | 23527 | 1.0000 |
| glyph16:U+005F | glyph | 318 | 129 | 129 | 129 | 1.0000 |
| glyph16:U+007C | glyph | 111 | 133 | 133 | 133 | 1.0000 |
| glyph32:U+0020 | glyph | 1 | 0 | 0 | 0 | 1.0000 |
| glyph32:U+002D | glyph | 113 | 133 | 133 | 133 | 1.0000 |
| glyph32:U+0040 | glyph | 5226 | 23527 | 23527 | 23527 | 1.0000 |
| glyph32:U+005F | glyph | 318 | 129 | 129 | 129 | 1.0000 |
| glyph32:U+007C | glyph | 111 | 133 | 133 | 133 | 1.0000 |
| shader:gyroid_slice | shader | 104 | 938 | 938 | 938 | 1.0000 |
| shader:kaleidoscope_fold | shader | 57 | 554 | 554 | 554 | 1.0000 |

## Raw data

`2026-09-02-extraction-gap.csv` / `.json` carry every kernel's row. The two budget knobs are part of the result — a different limit moves the solved/UNSOLVED split and nothing else — so they are written out rather than left to the defaults:

```sh
PIXELFLOW_EXTRACTION_GAP_ARENA_DIR=<dir of .arena dumps> \
  PIXELFLOW_EXTRACTION_GAP_SECS=4 \
  PIXELFLOW_EXTRACTION_GAP_EXPANSIONS=15000000 \
  RUST_MIN_STACK=268435456 \
  cargo test -p pixelflow-search --release --lib -- --ignored extraction_gap_measurement
```

The `.arena` corpus is what the three `#[ignore]`d dumpers produce (`pixelflow-core`'s cell-grid dump, `pixelflow-graphics`'s glyph dump, `pixelflow-pipeline`'s shader/psychedelic dump) — the same 206 files the missing-congruence probe reads.

