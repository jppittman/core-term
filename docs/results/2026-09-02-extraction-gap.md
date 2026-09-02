# How much is the greedy extractor leaving on the table?

Measurement only. Nothing in production changed: same rule set, same `Budget::Production`, same `CostModel::latency_prior()`, same `extract_dag`.

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

## The mechanism split

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

