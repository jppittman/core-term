# The structural gap between the real kernels and the corpora the claims were minted on

**Date:** 2026-09-07 · **Branch:** `claude/corpus-gaps` · **Instrument:** `pixelflow-pipeline/src/bin/corpus_gaps.rs` (new) · **Rows:** `2026-09-07-corpus-structural-gaps.csv` (777 kernels × 90 columns) · **Aggregates:** `.json` · **Script:** `pixelflow-pipeline/scripts/corpus_gaps_aggregate.py` · **PR:** #1212

This is the post-mortem of the *method*, not of any one claim. Every cost-model and Guide result before 2026-09-06 was minted on a synthetic corpus and then failed, in one way or another, on the kernels production actually compiles (86× numeric-first → 0.97× real; the Guide never beating unguided on a real trig kernel; "parity" on a corpus with no schedule alternatives; `dyn_memory_ops` at ρ = 0.98 ranking and 27% paired sign; 33× cost perturbation moving 0 of 190 glyph extractions). The question here is the one underneath all of those: **what do the real kernels have that the generators never produce?** — answered as counts, two populations on the same columns, so the next generator can be checked against the table rather than argued about.

## Populations and columns

**REAL (207):** the 190 DejaVu Sans Mono glyph bakes (95 printable ASCII × tile 16 and tile 32, `production_glyph_arena_dump`), the terminal cell grid at three geometries (80×24 @1×/2×, 120×40 @2×; `dump_production_cell_grid_arenas`), `bench_scene_chrome` and `bench_scene_psychedelic` as the ONE packed kernel `PackedManifold::compile` hands the optimizer at 1920×1080 (`dump_production_scene_arenas`, added in this PR), and the 12 withheld ShaderToy ports from their live `shader_bench` definitions (the 2026-09-02 `shader:*` dumps predate the retirement of the Z/W axes and name `Var(2)`, which the emitter refuses — they were replaced, not repaired).

**SYNTHETIC (570):** every generator a claim was minted on, sampled at the sizes the claims used — `gen_bench_corpus`'s `BwdGenerator` at its 15 size bands (`fused_op_prob = 0`, 24 each = 360), the extraction-objective probe's default-config `BwdGenerator` at depths 4/6/8/11 (24 each = 96, the `synth_d{d}_s{s}` set), `gen_sh_corpus` (48), `gen_bezier_corpus` (48), the five original named production kernels, and `collapse_cost`'s allocation-pressure fixtures (13; `anchored12x40` and `anchored16x64` excluded — see caveats). The Phase-3 / Round-2 binaries sample `gen_bench_corpus`'s output, so the bands cover them. Seed 20260907; the run is deterministic except for `opt_ms`. **Checked, not assumed:** the tool was run twice — once during development (1-minute load 5.9–9.3) and once from the committed binary (load 12.4–19.7) — and every one of the 777 rows and all 1,699 per-scope guard-telemetry lines are byte-identical between the two runs with `opt_ms` removed.

Every kernel goes through production's own sequence — `lower_dwrt`, `expand_reduce`, `Optimizer::production().for_lattice(shape)` under `Budget::Production`, `emit::compile` — and is measured **only in counts**: arena nodes at construction and after structural hash-consing (the splice-duplication factor); tree cost against DAG cost of the lowered input and of the extracted term under `CostModel::latency_prior` (the sharing ratio) plus the DP's own trip-weighted `ChoiceCost`; the select / compare / gather / op-kind census on the hash-consed graph, with arm reach and arm-exclusive fractions per select; classes, applications, iterations and the typed stop reason at production budget, and the per-rule firing histogram from the provenance journal; and from the emitter, bytes, spill slots, LICM-hoisted values, instructions per scope of the collapse nest (frame / row / pixel — the variance classes), trip-weighted memory ops, and the `PIXELFLOW_GUARD_TELEMETRY` line per scope (schedule entries, selects, arm-exclusive entries, guarded entries). Wall clock appears once (`opt_ms`), as a sign only: the committed run's clock was taken at 1-minute load 12.4–19.7 (contended) and is not a claim; the uncontended run (load 5.9–9.3) gave 24.6 ms [16.5, 86.4] real against 0.6 ms [0.2, 22.7] synthetic at the median, the same ordering.

## The gaps, ranked by how much the property is already known to move cost

The ranking uses numbers measured *before* this inventory: the 3.6× on chrome came from the guard schedule (S3b); tree-vs-DAG pricing was 95% of the extraction gap (#1192); trip-weighting took the paired-sign predictor from 78% to 98% (`collapse_cost`); raising the class cap 12× made chrome 15% slower; the `O`@32 glyph runs 33 ns/px at 40 px against 2.0 at 640 because its Y-only prologue runs unconditionally. The inventory's job is to say whether each of those properties exists in the synthetic population at all.

| # | property | real | synthetic | cost evidence it moves |
|---|---|---:|---:|---|
| 1 | selects per 100 nodes · kernels with any select | 7.2 · **95%** | 0 · **0%** | chrome 3.6× (S3b guards) |
| 1 | schedule entries under a guard (median) · selects that got a guard | 88% · 59% | 0% · 12% (8 kernels) | same |
| 2 | extracted tree cost / dag cost (median [p10, p90]) · kernels that share | 5.2 [2.1, 17] · **99%** | 1.0 [1, 3.3] · 44% | 95% of the extraction gap (#1192) |
| 2 | input (lowered) tree / dag | 16.6 (chrome 2,818) | 1.0 | same |
| 3 | schedule entries hoisted out of the pixel loop (frame + row) | **78%** (row 73%) | 41%, bimodal [0, 1] | 78% → 98% sign (trips); `O`@32 16× |
| 3 | LICM-hoisted values (median) | 72 | 1 | same |
| 4 | splice-duplication factor (reachable / hash-cons) | 1.72 glyph · 4.8 grid · 11 psychedelic · **1,082 chrome** | 1.11 | 12× cap = 15% slower; cap binds on the input |
| 4 | stop reason at production budget | **93% ClassCap**, median 2 iterations, `U+0040` **0 iterations** at both tiles | 24% ClassCap, 76% quiesce | same |
| 5 | kernels with a transcendental · Select+compare share of nodes | 4% · 18.3% | **88%** · 0% | 86× numeric-first → 0.97× real |
| 5 | rules that fire on one population only | 0 | **15** (power-*, even/odd-negation, pythagorean, exp/ln cancel) | same |
| 6 | kernels with gathers / buffers / uniforms | 3 / 3 / 1 of 207 | **0** of 570 | `dyn_memory_ops` ρ 0.98 vs 27% sign was fitted with no gathers in the corpus |
| 7 | e-graph input nodes · depth · schedule entries · bytes | 823 · 42 · 1,048 · 7,952 | 29 · 13 · 16 · 768 | every "at scale" claim |
| — | kernels naming `Var(2)`/`Var(3)` (the emitter refuses) | 0 | **45%** (259 / 570; 72% of `bwd_default`) | unmeasurable, not merely unrepresentative |
| — | additive `dag_cost` moved by production saturation (median) | **−4.7%** (glyph −3.1, shader −5.8, grid −8.5, chrome −17.9); 64 kernels *raised* | **−37.5%** (bands −49%) | the "F" delta the plan says was never measured, in the DP's own units |

### 1. Coherent masks, exclusive arms, guards — absent from every generator

Every real family but psychedelic is select-bearing: 95% of real kernels, 43 selects at the median, 7.2 per 100 hash-consed nodes; compares are 10.9% of real nodes and 24 of them per kernel are select masks, with `BitAnd`/`BitOr` (3.5%) combining them. The arms are what makes them guardable: the true arm reaches 61 nodes at the median, the false arm 1 (a constant or the passthrough), and **97.8% of the nodes an arm reaches are exclusive to it** — which is why the emitter's guard analysis, once S3b made it see a contiguous arm, covers 88% of the schedule entries of a median real kernel and 59% of its selects (chrome: 1,228 of 1,232 exclusive entries guarded, 6 of 34 selects; `U+004F`@16: 3,318 of 3,335, 158 of 260). No synthetic generator emits a single `Select` or compare — 0 of 570 — so the guard machinery, the coherence-dependent term of the schedule cost, and the anti-guard direction of `SelectHoistUnary` (round2_rules.rs:588–635, nothing opposing it) are all unexercised by everything the models were trained and validated on. The 8 synthetic kernels with schedule selects are `sh` kernels whose `Max`/`Min` lowered into selects; 12% of *those* got a guard.

*Can a generator produce it?* Structurally, yes: a `Select` whose arms are drawn as disjoint subexpressions gives exclusive arms for free, and a mask that is a compare of a coordinate against constants (`y_lo ≤ Y < y_hi`) gives the *static* coherence fraction the demand-DAG plan (§4) prices from extents. What a random expression cannot produce is the geometry that makes the *dynamic* fraction coherent — a hit test's silhouette, a curve's sign band, a row range of a glyph — because that is a property of the function's level sets, not of its syntax. Coherent masks come from composing a predicate with the arms it guards over a lattice; that is the construction, and the generator has to *be* that construction (a kernel composed from `kernel!` pieces over an extent), not a term drawn from an op distribution.

### 2. Sharing — the DAG the kernel pays for versus the tree the corpus contains

On the lowered input, the tree cost of a real kernel is 16.6× its DAG cost at the median (glyphs; the derivative lowering shares its `Sqrt` and hit-path subterms — lowering grows the hash-consed graph 1.44×), 2,818× for chrome and 3,915× for `mandelbrot_distance`; after extraction the term still shares 5.2× [2.1, 17], and 99% of real kernels share at all. The synthetic median is exactly 1.0: `BwdGenerator` builds trees, and only `sh` (5.5×) and `bezier` (4.2×) — the two families written as *constructions* rather than draws — have any sharing. The extraction-objective result (#1192: pricing sharing closes 95% of the gap) was therefore invisible on the corpus that "validated" tree extraction, and would remain invisible on any refit of it.

*Can a generator produce it?* Only by building DAGs on purpose — reusing a drawn subterm in several places, differentiating a value and keeping the value (`Dwrt` is 6.5% of real nodes and 0% of synthetic), or composing kernels by value so that splice makes copies that hash-consing later folds. Random trees with a "reuse this node" probability would give a sharing ratio; they would not give the *shape* of real sharing, where the shared thing is a hit-path normalization or an antialiasing derivative that the whole kernel hangs from. Composition produces it; sampling does not.

### 3. Variance classes — where the schedule entries live

For a median real kernel, 78% of the emitted schedule sits outside the pixel loop (73% row-scope, 5% frame-scope) and 72 values are LICM-hoisted; the pixel body is 22% of the instructions. This is the property trip-weighting prices and an additive cost cannot see: it is why `static_mem_ops` gets the paired sign right 78% of the time and its trips-weighted form 98%, and why the `O`@32 glyph is 16× slower per pixel at 40 px than at 640 (a Y-only prologue whose cost is per row, not per pixel). The synthetic column is bimodal — median 41%, p10 0, p90 1 — and the mass is an artifact: a `bwd_bands` expression is "hoisted" 35% of the time because a small random expression over `Var(0..4)` is often *entirely* X-invariant, not because it has a row prologue and a pixel body. 45% of synthetic kernels name `Var(2)`/`Var(3)`, coordinate axes production no longer has; the emitter refuses them, so for 259 of 570 the scope split cannot be measured at all, and for the rest the split is a coin flip on which variables the draw happened to include.

*Can a generator produce it?* Yes, if it draws in production's variable vocabulary — two coordinate axes plus uniforms — and deliberately builds a per-row part (a function of `Y` alone: a scanline's `t`, a band predicate) that the per-pixel part consumes. That is a two-level construction, and its parameter is the *ratio* of row work to pixel work, which this table now gives per family (glyph 0.74 / 0.21; scene 0.16 / 0.39; grid 0.29 / 0.71; shader 0.12 / 0.89).

### 4. Splice duplication and the class cap that binds on the input

Real kernels arrive duplicated: the arena `Kernel` composition hands the optimizer is 1.72× its hash-consed size for glyphs, 4.8× for the cell grid, 11× for psychedelic and **1,082× for chrome** (335,411 reachable, 416,579 tree, 310 unique; 465 after lowering). Because the class cap counts classes and the hash-cons happens inside the e-graph, the budget is spent on the input: 93% of real kernels stop on `ClassCap` at a median of **2 iterations**, 38 sit exactly at the cap, and `U+0040` (the largest glyph: 3,424 unique nodes, 5,226 after lowering) is over the 5,000-class cap before the first rewrite — **0 iterations, 0 applications, at both tiles** — so production saturation is a no-op on the kernel that would benefit most, and its 307 spill slots and 51 kB of code are the un-optimized term. The synthetic population quiesces (76%; `bwd_bands` 97%) at 53 classes median, 1.5 classes per input node against 4.6 for glyphs; the 12×-cap experiment (15% slower) and the "more saturation is not the lever" reading (§3 of the production-scale plan) were both made in a regime the synthetic corpus never enters.

*Can a generator produce it?* Duplication is a property of *how the term was built* — `Kernel::at`, `Kernel::sum`, `select` by value splice copies — so a generator that emits a hash-consed arena cannot have it, and one that emits trees has only the trivial 1.0–1.4× that `junkify` rewrites leave behind. The honest fix is on the production side (hash-cons at construction, plan §4 item 1), after which this row disappears from the real column too; until then, any corpus that is to predict production behaviour at the cap must be composed the way production composes.

### 5. Op-kind mass and the rules that fire

Real kernels are `MulAdd` (15.9%), `Select` (7.4%), compares (10.9%), `Sqrt` (4.5%), `Div` (3.3%), `Dwrt` (6.5%), `Max`/`Min` (8.0%), `BitAnd` (3.5%); transcendentals are 0.02% of nodes and appear in 4% of kernels (the shaders). The synthetic population is 88% transcendental-bearing — `Neg` 16.3%, `Abs` 7.9%, `Pow` 5.0%, `Sin`/`Cos`/`Exp`/`Log*`/`Atan2`/`Tan` ~8% — with `Div` at 0.16% and `Sqrt` at 0.45%. The rule histogram follows: fifteen rules fire only on synthetic kernels (`power-*`, `even-negation(Abs|Cos)`, `odd-negation(Sin|Tan)`, `pythagorean`, `exp-ln-cancel`, `log2-exp2-cancel`, …) and none only on real; on real, `identity(Add)` (4.1% of firings, 93% of kernels), `annihilator` (1.4%, 89%) and `canonicalize(Sub|Div)` carry mass that is ≈0 on synthetic — the `x + 0`, `x · 0`, `x − x` seams that composition leaves at splice boundaries and that a generator never writes. This is the structural form of the 86× → 0.97× miss: the numeric-first rules had a corpus made of the functions they rewrite. The `BwdGenerator` op weights were "rebalanced to match ShaderToy distributions"; the real distribution to match is the op column of this table, and it is font geometry and ray-sphere arithmetic, not ShaderToy.

*Can a generator produce it?* The op histogram, trivially — the weights are a parameter. What cannot be sampled into existence is where the ops *sit*: a `Sqrt` on real kernels is a normalization shared by a value and its derivative, a compare is a select mask over a band, a `Div` is a Bézier `t`. Match the histogram alone and the rule-fire table will still disagree, because rules fire on *patterns*.

### 6. Gathers, buffers, uniforms — the terminal's kernel is unrepresented

The cell grid — the one kernel `core-term` runs every frame — has 14 gathers over 2 buffers with integer index arithmetic (`TruncToInt`, `Shl`, `BitAnd`/`BitOr` packing), 4 selects and 17× sharing; psychedelic has the only uniform. That is 3 + 1 kernels of 207 real and 0 of 570 synthetic. `dyn_memory_ops` — ρ = 0.98 across kernels, 27% paired sign — was fitted on a corpus in which no kernel loads memory inside the pixel loop except through spills; the term it was named for never appears in its training data.

*Can a generator produce it?* A buffer declaration plus `Gather` at drawn integer coordinates is one function; the index arithmetic (atlas slot from cell, packed-word channel extraction) is a construction and comes with it or not at all. Given how few production shapes there are, the right move is not to generate this family but to take the real one — at more geometries — as the corpus.

### 7. Scale

A median real kernel is 1,215 arena nodes (823 into the e-graph after lowering and hash-consing), depth 42, 1,048 schedule entries, 7,952 emitted bytes; a median synthetic one is 32 / 29 / 13 / 16 / 768. The largest synthetic band (`bwd_default` d11) reaches 63 unique nodes; the median *glyph* is 734. Nothing in the synthetic population is the size of the smallest glyph family, so the extraction-time, class-count and application-count budgets were all tuned one to two orders of magnitude below the kernels that hit them.

## The F proxy: what production saturation buys, in the DP's units

Priced with the flat latency prior on the lowered input and on the extracted term (same units, no clock), production saturation moves a real kernel's additive `dag_cost` by **−4.7%** at the median (glyphs −3.1%, shaders −5.8%, cell grid −8.5%, psychedelic −4.1%, chrome −17.9%, `mandelbrot_distance` −21.2%) and *raises* it on 64 of 207 — the trip-weighted `ChoiceCost` since `for_lattice` will take a dearer pixel-flat term for a cheaper scheduled one, which is the right trade and exactly what an additive corpus metric cannot see. On the synthetic population the same pass buys **−37.5%** (bands −49%, default −39%, bezier −27%, `sh` −0.2%). The corpora therefore said the e-graph is worth ~8× more than it is on the kernels that ship — the same shape as every other miss on this list — and the clock measurement of F the production-scale plan asks for (§7) should be expected to come in small on glyphs and mid-sized on chrome.

## What to do with the table

1. **Make the real population the corpus** — 207 kernels, at their shapes, are already dumped and loadable (`corpus_gaps --dumps`); the guard/scope/sharing columns are the labels a schedule-valued cost (plan C) is fitted to. Fonts are an instance of the target, not the target: the shader and scene families need more members, and the structural columns above say what a new member must have to count.
2. **A generator that wants to stand in for it must be a construction, not a draw:** two-axis + uniform vocabulary; a row-part and a pixel-part; selects over band and sign predicates with arms built disjointly; values reused and differentiated; op weights fitted to the real column of the op table; composed by value so splice duplication appears (or hash-cons at construction lands first and the row is retired on both sides). `sh_family` and `bezier_family` are already half-way there (sharing 5.5× / 4.2×, 92% / 100% at the cap) and are the templates.
3. **Two engineering items this measurement makes urgent:** hash-cons at construction (plan §4 item 1; the 93%-at-cap / 0-iteration rows are its symptom), and a class cap that counts *rewrite-minted* classes rather than input classes, so `U+0040` gets a saturation at all.
4. **Re-validate every claim in the ledger against the real column**, not a refit of the synthetic one — the domain-shift ablation JP required for the Guide is the same instrument at the corpus level.

## Caveats, stated so they can be checked

- **Tile 16 and tile 32 glyph arenas are structurally identical** (the tile scale is a constant), so the 190 glyph rows are 95 shapes × 2 extents; they differ only in trip-weighted columns. The real population is effectively 95 + 3 + 2 + 12 = 112 distinct shapes.
- The chrome dump is 416,579 tree nodes against the gate's 416,420 printout (335,411 reachable): a 159-node (0.04%) drift between the `cell_grid` restatement and `compile_packed_for`, recorded, not chased.
- Guard telemetry sums the emitter's per-scope lines; `exclusive` counts an entry once per select whose arm holds it, so nested selects can push `exclusive / schedule` above 1 (p90 1.02). `guarded / schedule` does not overlap.
- `collapse_anchored:anchored12x40` did not finish production saturation within the run (a 40-deep add chain over 12 anchors; `anchored16x64` was skipped with it). Both are recorded as an observation, not a row: production's budget is applications-capped, and a kernel that does not reach the cap in minutes is itself a finding about that budget on chain-shaped inputs.
- The 2026-09-02 `shader:*` dumps name `Var(2)`; the live `shader_bench` definitions were used instead. The single-channel hand-transcribed `psychedelic` dump was skipped in favour of the packed four-channel kernel production compiles.
- Synthetic sizes are `gen_bench_corpus`'s bands restated in the binary (they are private to it); the sample is 24 per configuration at one seed. `opt_ms` is a loaded-host clock (load 12–20 in the committed CSV) and is reported as a sign only; the ordering (real ≈ 40× synthetic at the median) held in both runs.
- Nothing here is a clock claim about speed. The ranking column cites clocks measured elsewhere; the inventory's own columns are all deterministic.

## Reproduce

```bash
PIXELFLOW_TELEMETRY_DIR=$D cargo test -p pixelflow-graphics --release --test production_glyph_arena_dump -- --ignored
PIXELFLOW_TELEMETRY_DIR=$D cargo test -p pixelflow-graphics --release -- --ignored dump_production_cell_grid_arenas dump_production_scene_arenas
PIXELFLOW_GUARD_TELEMETRY=1 cargo run --release -p pixelflow-pipeline --bin corpus_gaps -- \
  --dumps $D --out rows.csv --synthetic-n 24 --skip psychedelic \
  --skip collapse_anchored:anchored12x40 --skip collapse_anchored:anchored16x64 2> guards.log
python3 pixelflow-pipeline/scripts/corpus_gaps_aggregate.py rows.csv guards.log docs/results/2026-09-07-corpus-structural-gaps
```

The tables below are generated from `rows.csv` + `guards.log` by that script (the `.csv` is the joined per-kernel table, the `.json` the aggregates).

## Tables

### Side by side: REAL vs SYNTHETIC (median [p10, p90]; fractions are means)

| property | real (n=207) | synthetic (n=570) |
|---|---:|---:|
| arena nodes at construction (reachable) | 1,215 [127.0, 3,469] | 32.0 [12.0, 130.0] |
| nodes after hash-cons | 572.0 [77.0, 1,968] | 29.0 [11.0, 92.0] |
| splice-duplication factor (reachable / hash-cons) | 1.72 [1.66, 1.81] | 1.11 [1, 1.44] |
| fraction of kernels with any duplication | 99% | 79% |
| e-graph input nodes (lowered, hash-consed) | 823.0 [78.0, 3,014] | 29.0 [11.0, 92.0] |
| lowering growth (Dwrt/reduce expansion) | 1.43 [0.874, 1.53] | 1 [1, 1] |
| input tree cost / dag cost (lowered) | 16.6 [4.2, 17.6] | 1 [1, 4.18] |
| extracted tree cost / dag cost | 5.2 [2.12, 17.0] | 1 [1, 3.3] |
| fraction with extracted sharing > 1 | 99% | 44% |
| extracted DAG nodes | 733.0 [63.0, 2,921] | 18.0 [7, 80.0] |
| extracted nodes / e-graph input nodes | 0.958 [0.856, 1.01] | 0.644 [0.476, 0.942] |
| extracted dag_cost vs input dag_cost (%) | -4.66 [-26.7, 2.15] | -37.5 [-76.4, -4.58] |
| selects (hash-consed) | 43.0 [3, 155.0] | 0 [0, 0] |
| selects per 100 nodes | 7.23 [3.95, 7.83] | 0 [0, 0] |
| fraction with any select | 95% | 0% |
| compares | 61.0 [5, 222.0] | 0 [0, 0] |
| compares that are select masks | 24.0 [0, 90.0] | 0 [0, 0] |
| masks used by >1 select | 1 [0, 3] | 0 [0, 0] |
| median true-arm reach (nodes) | 61.0 [44.0, 62.5] | 0 [0, 0] |
| median false-arm reach (nodes) | 1 [1, 1] | 0 [0, 0] |
| arm-exclusive nodes / arm reach | 0.978 [0.963, 0.979] | 0 [0, 0] |
| selects in extracted term | 57.0 [3, 213.0] | 0 [0, 0] |
| schedule entries (emitter) | 1,048 [142.0, 4,093] | 16.0 [0, 296.0] |
| guard telemetry: exclusive entries / schedule | 0.898 [0.613, 1.02] | 0.037 [0, 0.708] |
| guard telemetry: guarded entries / schedule | 0.878 [0.409, 1] | 0 [0, 0.449] |
| guard telemetry: exclusive but unguarded (order refuses) | 0.0355 [0.0063, 0.356] | 0.912 [0.0421, 1] |
| selects in the schedule | 84.0 [7, 310.0] | 0 [0, 8] |
| selects with a non-empty exclusive arm | 60.0 [7, 212.0] | 0 [0, 8] |
| selects that got a guard | 52.0 [2, 160.0] | 0 [0, 4] |
| guarded selects / schedule selects | 0.593 [0.2, 0.627] | 0.118 [0, 0.8] |
| DP objective (trip-weighted) tree/dag | 2.17 [1.16, 5.11] | 1 [1, 3.39] |
| gathers | 0 [0, 0] | 0 [0, 0] |
| fraction with any gather | 1% | 0% |
| buffers | 0 [0, 0] | 0 [0, 0] |
| uniforms | 0 [0, 0] | 0 [0, 0] |
| transcendental nodes | 0 [0, 0] | 4 [0, 12.0] |
| fraction with any transcendental | 4% | 88% |
| depth | 42.0 [25.0, 62.0] | 13.0 [8, 24.0] |
| schedule entries hoisted (frame+row) / all | 0.777 [0.285, 0.836] | 0.415 [0, 1] |
| frame-scope instructions / all | 0.047 [0.013, 0.231] | 0 [0, 0.437] |
| row-scope instructions / all | 0.727 [0.225, 0.797] | 0.062 [0, 1] |
| pixel-scope instructions / all | 0.223 [0.164, 0.715] | 0.585 [0, 1] |
| values LICM hoisted | 72.0 [11.0, 267.0] | 1 [0, 9] |
| spill slots | 0 [0, 3] | 0 [0, 0] |
| emitted bytes | 7,952 [912.0, 28,896] | 768.0 [240.0, 1,984] |
| trip-weighted memory ops | 18,614 [98.0, 104,827] | 0 [0, 1,283] |
| e-classes at stop | 4,684 [2,531, 5,000] | 53.0 [17.0, 4,346] |
| classes per e-graph input node | 5.27 [1.64, 35.0] | 1.75 [1.29, 47.8] |
| fraction where saturation ran 0 iterations (input alone over the cap) | 0 [0, 0] | 0 [0, 0] |
| fraction stopping on ClassCap | 93% | 24% |
| fraction quiescing | 7% | 76% |
| rule applications | 5,422 [1,948, 15,196] | 114.0 [18.0, 8,710] |
| applications per e-graph input node | 7.08 [0.641, 202.3] | 4.64 [1.4, 145.0] |
| iterations | 2 [1, 5] | 3 [2, 6] |
| saturate+extract ms (host loaded; sign only) | 30.1 [18.6, 102.5] | 1 [0.2, 35.5] |

### By group (median)

| group | n | nodes_reachable | nodes_hashcons | splice_factor | input_sharing | ext_sharing | selects | compares | gathers | transcendentals | sched_len | guard_exclusive_frac | guard_guarded_frac | hoist_frac | body_frac | classes | applications | classes_at_cap | quiesced | bytes | ext_nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cellgrid (real) | 3 | 623.0 | 131.0 | 4.76 | 17.3 | 17.1 | 4 | 4 | 14.0 | 0 | 535.0 | 0.329 | 0.329 | 0.286 | 0.714 | 4,582 | 6,952 | 100% | 0% | 2,896 | 121.0 |
| glyph16 (real) | 95 | 1,253 | 734.0 | 1.72 | 16.6 | 5.24 | 54.0 | 78.0 | 0 | 0 | 1,373 | 0.901 | 0.883 | 0.79 | 0.21 | 4,707 | 5,335 | 94% | 6% | 10,752 | 964.0 |
| glyph32 (real) | 95 | 1,253 | 734.0 | 1.72 | 16.6 | 5.24 | 54.0 | 78.0 | 0 | 0 | 1,373 | 0.901 | 0.883 | 0.79 | 0.21 | 4,707 | 5,335 | 94% | 6% | 10,752 | 964.0 |
| scene (real) | 2 | 1,246 | 112.0 | 11.1 | 5.75 | 5.09 | 0 | 0 | 0 | 0 | 444.0 | 0.27 | 0.131 | 0.0642 | 0.388 | 4,676 | 5,531 | 100% | 0% | 2,368 | 112.0 |
| shader (real) | 12 | 62.0 | 50.0 | 1.14 | 3.62 | 3.4 | 0 | 0 | 0 | 2 | 291.0 | 0.313 | 0.269 | 0.231 | 0.885 | 1,822 | 6,977 | 75% | 25% | 1,376 | 49.0 |
| bezier (synthetic) | 48 | 62.0 | 54.0 | 1.43 | 4.24 | 3.25 | 0 | 0 | 0 | 0 | 82.0 | 0 | 0 | 0 | 1 | 3,895 | 7,063 | 100% | 0% | 592.0 | 41.0 |
| bwd_bands (synthetic) | 360 | 22.0 | 20.0 | 1.08 | 1 | 1 | 0 | 0 | 0 | 3 | 8 | 0.0377 | 0 | 0.348 | 0.652 | 31.0 | 53.0 | 3% | 97% | 736.0 | 12.0 |
| bwd_default (synthetic) | 96 | 73.0 | 63.0 | 1.19 | 1.05 | 1.02 | 0 | 0 | 0 | 12.0 | 0 | 0.482 | 0 | 0.484 | 0.021 | 158.0 | 568.0 | 24% | 76% | 1,216 | 40.0 |
| collapse_synth (synthetic) | 13 | 91.0 | 69.0 | 1.33 | 1.02 | 1.07 | 0 | 0 | 0 | 0 | 83.0 | 0 | 0 | 0.92 | 0.08 | 3,818 | 7,094 | 85% | 15% | 412.0 | 43.0 |
| named (synthetic) | 5 | 9 | 9 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 15.0 | 0 | 0 | 0.333 | 0.667 | 84.0 | 406.0 | 40% | 60% | 188.0 | 7 |
| sh (synthetic) | 48 | 119.0 | 92.0 | 1.5 | 5.52 | 5.47 | 0 | 0 | 0 | 6 | 442.0 | 0.585 | 0.316 | 0.576 | 0.426 | 4,621 | 7,018 | 92% | 8% | 2,112 | 89.0 |

### Stop reason under production budget

| population | ClassCap | Quiesced |
|---|---:|---:|
| real | 192 | 15 |
| synthetic | 138 | 432 |

### Op-kind share of hash-consed nodes (pooled)

| op | real | synthetic | ratio real/synth |
|---|---:|---:|---:|
| Mul | 10.25% | 18.46% | 0.56 |
| Const | 13.01% | 15.38% | 0.85 |
| Add | 11.35% | 15.18% | 0.75 |
| MulAdd | 15.89% | 2.62% | 6.06 |
| Neg | 1.24% | 15.60% | 0.08 |
| Abs | 0.12% | 7.56% | 0.02 |
| Select | 7.42% | 0.00% | ∞ |
| Sub | 4.18% | 2.75% | 1.52 |
| Dwrt | 6.45% | 0.00% | ∞ |
| Sqrt | 4.54% | 1.01% | 4.50 |
| Max | 4.58% | 0.91% | 5.00 |
| Var | 0.24% | 4.55% | 0.05 |
| Pow | 0.00% | 4.76% | 0 |
| Ge | 4.55% | 0.00% | ∞ |
| Min | 3.37% | 0.88% | 3.83 |
| BitAnd | 3.47% | 0.00% | ∞ |
| Div | 3.26% | 0.15% | 21.39 |
| Le | 2.94% | 0.00% | ∞ |
| Gt | 2.71% | 0.00% | ∞ |
| Sin | 0.01% | 1.72% | 0.01 |
| Cos | 0.01% | 1.64% | 0.00 |
| Exp | 0.00% | 1.02% | 0.00 |
| Log2 | 0.00% | 0.89% | 0.00 |
| Atan2 | 0.00% | 0.89% | 0.00 |
| Ln | 0.00% | 0.86% | 0.00 |
| Log10 | 0.00% | 0.82% | 0 |
| Tan | 0.00% | 0.81% | 0 |
| Exp2 | 0.00% | 0.80% | 0 |
| Recip | 0.00% | 0.60% | 0 |
| Lt | 0.34% | 0.00% | ∞ |
| Rsqrt | 0.00% | 0.10% | 0 |
| Uniform | 0.00% | 0.03% | 0.08 |
| Gather | 0.02% | 0.00% | ∞ |
| Shl | 0.02% | 0.00% | ∞ |
| TruncToInt | 0.02% | 0.00% | ∞ |
| BitOr | 0.01% | 0.00% | ∞ |
| Floor | 0.01% | 0.00% | ∞ |
| Buffer | 0.00% | 0.00% | ∞ |

### Rule-fire share under production saturation (pooled applications; kernels-with = fraction of kernels the rule fired on)

| rule | real share | real kernels-with | synthetic share | synthetic kernels-with |
|---|---:|---:|---:|---:|
| fma-fusion | 17.22% | 94% | 21.28% | 58% |
| commutative(Add) | 13.74% | 98% | 16.10% | 96% |
| reverse-associative(Add) | 11.34% | 92% | 11.74% | 40% |
| commutative(Mul) | 11.07% | 96% | 12.00% | 73% |
| associative(Add) | 10.53% | 92% | 10.17% | 40% |
| reverse-associative(Mul) | 6.15% | 92% | 7.88% | 43% |
| associative(Mul) | 4.80% | 92% | 6.93% | 42% |
| factor | 4.75% | 93% | 4.05% | 40% |
| distribute | 4.41% | 94% | 3.26% | 40% |
| identity(Add) | 4.06% | 93% | 0.18% | 2% |
| constant-fold | 2.77% | 97% | 1.39% | 57% |
| doubling | 2.35% | 90% | 0.70% | 33% |
| halving | 1.59% | 89% | 1.03% | 33% |
| annihilator | 1.39% | 89% | 0.04% | 1% |
| canonicalize(Sub) | 1.16% | 98% | 0.26% | 23% |
| commutative(Max) | 0.81% | 95% | 0.14% | 22% |
| commutative(Min) | 0.70% | 96% | 0.15% | 20% |
| even-negation(Abs) | 0.00% | 0% | 0.73% | 77% |
| identity(Mul) | 0.32% | 86% | 0.41% | 14% |
| involution(Sub) | 0.01% | 33% | 0.71% | 76% |
| canonicalize(Div) | 0.53% | 94% | 0.03% | 12% |
| inverse-annihilation(Sub) | 0.23% | 43% | 0.01% | 3% |
| power-sqrt | 0.00% | 0% | 0.14% | 43% |
| power-recip | 0.00% | 0% | 0.12% | 44% |
| even-negation(Cos) | 0.00% | 0% | 0.10% | 25% |
| reverse-angle-addition | 0.01% | 2% | 0.05% | 12% |
| half-angle-product | 0.00% | 1% | 0.05% | 9% |
| power-combine | 0.00% | 0% | 0.05% | 11% |
| sin-angle-addition | 0.01% | 2% | 0.03% | 12% |
| recip-sqrt | 0.00% | 0% | 0.03% | 14% |
| cancellation(Sub) | 0.03% | 43% | 0.00% | 0% |
| reverse-associative(Min) | 0.00% | 1% | 0.03% | 5% |
| odd-negation(Sin) | 0.00% | 0% | 0.03% | 8% |
| associative(Min) | 0.00% | 1% | 0.02% | 5% |
| power-rsqrt | 0.00% | 0% | 0.03% | 14% |
| reverse-associative(Max) | 0.00% | 1% | 0.02% | 4% |
| cos-angle-addition | 0.01% | 2% | 0.02% | 9% |
| associative(Max) | 0.00% | 1% | 0.02% | 4% |
| odd-negation(Tan) | 0.00% | 0% | 0.01% | 3% |
| diff-of-squares | 0.00% | 3% | 0.01% | 7% |

Rules firing only on real: none.

Rules firing only on synthetic: even-negation(Abs), power-sqrt, power-recip, even-negation(Cos), power-combine, odd-negation(Sin), power-rsqrt, odd-negation(Tan), cancellation(Div), power-identity, exp-ln-cancel, ln-exp-cancel, pythagorean, exp2-log2-cancel, log2-exp2-cancel.


Total applications: real 1,404,338, synthetic 1,531,861.


### The production scenes and grids, individually

| kernel | nodes_reachable | nodes_hashcons | nodes_lowered_hc | splice_factor | input_sharing | ext_sharing | ext_nodes | dag_cost_delta_pct | selects | compares | gathers | sched_len | sched_selects | guard_exclusive | guard_guarded | guard_guarded_selects | hoist_frac | classes | class_cap | stop | applications | iterations | bytes | spill_slots | opt_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cellgrid:120x40_d2 | 623.0 | 131.0 | 131.0 | 4.76 | 17.3 | 17.1 | 121.0 | -8.49 | 4 | 4 | 14.0 | 535 | 8 | 176 | 176 | 8 | 0.286 | 4,582 | 5,000 | ClassCap | 6,952 | 3 | 2,912 | 8 | 21.2 |
| cellgrid:80x24_d1 | 623.0 | 128.0 | 128.0 | 4.87 | 17.4 | 17.3 | 118.0 | -8.58 | 4 | 4 | 14.0 | 531 | 8 | 176 | 176 | 8 | 0.288 | 4,582 | 5,000 | ClassCap | 6,955 | 3 | 2,848 | 8 | 21.1 |
| cellgrid:80x24_d2 | 623.0 | 131.0 | 131.0 | 4.76 | 17.3 | 17.1 | 121.0 | -8.49 | 4 | 4 | 14.0 | 535 | 8 | 176 | 176 | 8 | 0.286 | 4,582 | 5,000 | ClassCap | 6,952 | 3 | 2,896 | 8 | 20.8 |
| glyph16:U+004A | 1,253 | 734.0 | 1,074 | 1.71 | 16.9 | 7.54 | 1,006 | -4.68 | 54.0 | 79.0 | 0 | 1486 | 108 | 1324 | 1304 | 66 | 0.823 | 4,282 | 5,000 | ClassCap | 3,220 | 1 | 10,752 | 0 | 14.3 |
| glyph32:U+0040 | 5,940 | 3,424 | 5,226 | 1.74 | 17.4 | 17.4 | 5,226 | 0 | 269.0 | 381.0 | 0 | 10669 | 746 | 26500 | 19696 | 190 | 0.0627 | 5,226 | 5,000 | ClassCap | 0 | 0 | 51,408 | 307.0 | 9.1 |
| scene:chrome | 335,411 | 310.0 | 465.0 | 1,082 | 2,818 | 2,061 | 398.0 | -17.9 | 15.0 | 9 | 0 | 797 | 34 | 1232 | 1228 | 6 | 0.0642 | 4,940 | 5,000 | ClassCap | 5,531 | 2 | 3,424 | 22.0 | 50.7 |
| scene:psychedelic | 1,246 | 112.0 | 112.0 | 11.1 | 5.75 | 5.09 | 112.0 | -4.07 | 0 | 0 | 0 | 444 | 4 | 120 | 58 | 2 | 0.612 | 4,676 | 5,000 | ClassCap | 7,319 | 3 | 2,368 | 0 | 29.7 |
| shader:smooth_min_scene | 43.0 | 40.0 | 40.0 | 1.07 | 3.62 | 3.4 | 37.0 | -4.96 | 0 | 0 | 0 | 75 | 0 | 0 | 0 | 0 | 0.275 | 1,215 | 2,000 | ClassCap | 2,757 | 3 | 432.0 | 0 | 13.9 |
| shader:mandelbrot_distance | 152.0 | 128.0 | 128.0 | 1.19 | 3,915 | 1,769 | 96.0 | -21.2 | 20.0 | 5 | 0 | 291 | 39 | 91 | 24 | 6 | 0.0616 | 3,675 | 5,000 | ClassCap | 15,392 | 4 | 1,408 | 0 | 69.2 |
