# Extraction prices sharing (#1116), rebased onto main — the numbers under main's cost table, and at both production shapes

The change is the one measured in [`2026-09-02-extraction-objective.md`](2026-09-02-extraction-objective.md): `extract_dag` charged a shared subterm once per use and the kernel pays for it once, so the DP now carries the set of classes each chosen sub-DAG contains, prices each member once, runs both objectives and returns the cheaper by true `dag_cost` (ties to the tree arm). That run was made under #1134's refreshed latency prior, which was closed unmerged; this is the same probe re-run on **main's** table, plus what the production-scale plan ([`2026-09-06-egraph-at-production-scale.md`](../plans/2026-09-06-egraph-at-production-scale.md) §5.3) now requires of any extraction change: compile time on both production shapes, and which glyph kernels it touched.

Every probe number is deterministic (`CostModel::latency_prior` through `ExtractedDAG::dag_cost`); the per-kernel rows are in `2026-09-06-extraction-objective-rebase.{csv,json}`. Corpus: the 206 production `.arena` dumps regenerated on main by the three dumpers (byte-identical to the 2026-09-02 dumps — the macro consolidation did not move a single glyph arena) plus the 96 synthetic BwdGenerator kernels.

## The probe, on main's table

| | 2026-09-02 (#1134 table) | this run (main's table) |
|---|---:|---:|
| gap closed against the branch-and-bound DAG optimum | 95.9% (187 / 195 units, 3 kernels) | **95.9% (187 / 195 units, 3 kernels)** |
| exactly DAG-optimal on the solved set | 90 → 92 of 93 | **86 → 88 of 89** |
| real kernels improved / unchanged / worse | 55 / 151 / 0 | **55 / 151 / 0** |
| pooled real `dag_cost` | −0.23% | **−0.22%** (2,303 units) |
| best | `shader:mandelbrot_distance` −10.43% | **`shader:mandelbrot_distance` 576 → 518, −10.07%** |
| extraction wall time vs the tree-only extractor | median 2.23×, p90 2.29×, max 2.84× | **median 2.23×, p90 2.33×, max 2.52×, pooled 2.23×** |

The gap kernels: `shader:smooth_min_scene` 136 → 134 (optimum 126, 20% closed), `synth_d4_s7` 614 → 609 and `synth_d11_s4` 1665 → 1485 (both 100%). Four fewer kernels closed under the 4 s branch-and-bound limit than on 2026-09-02 — that limit is wall clock, and the host was at load 6–10 — so the solved set is 89, not 93; the gap fraction is computed within this run.

| group | n | improved | unchanged | worse | best Δ | units saved |
|---|---:|---:|---:|---:|---:|---:|
| cellgrid | 3 | 3 | 0 | 0 | −0.23% | 3 |
| glyph | 190 | 48 | 142 | 0 | −1.91% (`U+004A` 4339 → 4256) | 2,232 |
| psychedelic | 1 | 0 | 1 | 0 | +0.00% | 0 |
| shader | 12 | 4 | 8 | 0 | −10.07% | 68 |
| **all real** | **206** | **55** | **151** | **0** | **−10.07%** | **2,303** |

**Zero regressions is asserted, not observed.** `extract_dag_scoped` returns the minimum over a set that contains the old answer; the probe panics on a real kernel priced worse than the tree arm rather than tabulating it. The arm split is as before — sharing alone is strictly cheaper on 55, ties on 61, and dearer on 90 (`psychedelic` 766 → 1232 if taken alone) — which is why both arms run.

The extraction ratio is nine alternating pairs per kernel timed back to back in one process on one saturated graph, so only the ratio is claimed; the host was at 1-minute load 6–10 for the run (its floor is ~5, held by `fileproviderd`/`bird`/`syspolicyd`).

## Shape A: the scene kernels

`bench_scene_chrome` / `bench_scene_psychedelic`, main and this branch built side by side, alternating, three rounds, median of 9 frames each; 1-minute load 4.4–6.0 (the <4 the plan asks for was not reached in a 15-minute wait — see the floor above). Host: Apple M2 Max, so the default tier is **NEON**; the **SSE2** tier was built for `x86_64-apple-darwin` and run under Rosetta 2 — its bytes, schedule and guard counts are the SSE2 backend's own and deterministic, its compile ms and ns/px are emulated, one sample, and indicative only.

| | main | this branch | Δ |
|---|---:|---:|---:|
| **chrome, NEON** — arena nodes | 416,420 | 416,420 | — |
| compile (saturate + extract + emit), median of 3 | 106.1 ms | 106.7 ms | +0.6 ms |
| emitted code | 3,584 B | 3,424 B | −160 B |
| ns/px, 1 thread / 12 threads | 4.36 / 0.61 | 4.30 / 0.59 | −1.4% / −3% |
| guard telemetry: schedule / selects / guarded / exclusive | 401 / 17 / 642 / 644 | 385 / 17 / 614 / 616 | 16 fewer entries, all 17 selects still guarded |
| **chrome, SSE2 (Rosetta)** — compile | 175.5 ms | 175.3 ms | — |
| emitted code | 9,760 B | 9,221 B | −539 B |
| ns/px, 1 thread | 7.80 | 7.63 | −2% (emulated) |
| guard telemetry | 401 / 17 / 642 / 644 | 385 / 17 / 614 / 616 | as NEON |
| **psychedelic, NEON** — compile, median of 3 | 21.1 ms | 22.9 ms | +1.8 ms |
| emitted code | 2,368 B | 2,368 B | same DAG (probe row unchanged: tree arm wins), same schedule 127 / 1 / 29 / 30 |
| ns/px, 1 thread / 12 threads | 3.67 / 0.51 | 3.69 / 0.51 | noise |
| **psychedelic, SSE2 (Rosetta)** — compile / code | 49.3 ms / 5,584 B | 51.7 ms / 5,584 B | byte count and schedule unchanged |

The chrome scene is the case the plan is about, and the sharing-aware objective **shortens its extracted schedule** (401 → 385 entries) and its code (−160 B NEON, −539 B SSE2) at no compile cost — 106 ms against the ~250 ms budget. The ns/px gain is at the edge of what this host can resolve (the plan's ~10% rule for single kernels), so the claim is the schedule and the bytes, not the clock. The psychedelic shader is a leaf-coloured kernel with no sharing to find; its DAG is unchanged and it pays the second DP pass, +1.8 ms.

## Shape B: the glyph warm

`glyph_compile_report` (new, `pixelflow-pipeline/examples`): NotoSansMono at core-term's 16 pt cell height, both atlas densities (tile 16 and tile 32), the 95 printable-ASCII bakes each, compiled through the `Lattice::bake` path (`compile_as_baked`) once per process, seven alternating processes per build, per-kernel medians; plus one cold `atlas.warm` per density per process.

| | main | this branch | Δ |
|---|---:|---:|---:|
| tile 16: Σ per-kernel median compile, 95 kernels | 4,459 ms | 4,667 ms | **+208 ms (+4.7%), +2.19 ms per kernel** |
| tile 16: per-kernel compile ratio | | median 1.080, p90 1.139, max 1.153 | |
| tile 16: emitted code, Σ | 1,219,668 B | 1,241,268 B | +21,600 B (+1.8%) |
| tile 16: kernels whose code changed | | **55 of 95** | byte Δ from −160 (`U+002B`) to +1,200 (`U+0033`) |
| tile 32 | 4,458 ms → 4,665 ms | 55 of 95 changed, +21,600 B | the same 55; the two tiles differ only in the scaffold |
| cold `atlas.warm`, tile 16, median of 7 | 4,490 ms | 4,694 ms | +204 ms (+4.5%) |
| cold `atlas.warm`, tile 32 | 4,476 ms | 4,693 ms | +217 ms |

Against §5.3's yardstick — "5 ms per kernel adds half a second to startup" — this adds **2.2 ms per kernel, 0.2 s per 95-glyph warm**, on top of a warm that already costs 4.5 s on this host. It is the second DP pass; nothing else in the path changed.

**Which glyphs, and why 55 rather than 48.** The probe extracts at `LatticeShape::POINT` (unweighted) and counts 24 glyphs per tile as improved; production extracts at the bake shape, where a node's cost is weighted by how often the lattice evaluates it, and that objective prefers the sharing arm on 55. Of the 24 POINT-improved tile-16 glyphs, 23 emit different code and one (`U+004E`) emits identically; the other 32 change only under the weighted objective. On every one of the 55 the sharing arm was chosen because its weighted `dag_cost` was strictly lower — that is the only branch that returns it — so these are improvements in the objective production actually minimizes.

**The finding worth keeping.** Where the objective bites on a glyph, the *emitted bytes grow* (`U+0033` 27,824 → 29,024 B, `U+0030` 22,208 → 23,104 B) while the DAG cost falls. The additive prior says the term is cheaper; the emitter's schedule for it is longer. A baked glyph's runtime is a rounding error, so nothing is lost here, but it is one more instance of the plan's §3 point that a function of the DAG does not see the schedule — and the reason the chrome scene, where it matters, was measured on the clock rather than trusted from the table.

## What CI cannot see, and the check it implies

No presubmit job compiles a production scene or a glyph warm. The numbers above came from two examples run by hand against a side-by-side build of main; a change that moved the chrome compile past 250 ms, or added 5 ms to every glyph, would merge green. The check to write is a job that runs `bench_scene_chrome` (compile ms, bytes, guard counts) and `glyph_compile_report` (Σ compile, the byte-hash column) and diffs them against the base commit's — compile time thresholded loosely, byte hashes exactly. `glyph_compile_report` is written so that its rows are that diff.

## What this obliges downstream

Unchanged from the original: every Guide label and every registered Phase-3 constant was minted under tree cost, and that chain restarts from #1116.

## Reproduction

```sh
# the probe (14 min at PIXELFLOW_EXTRACTION_GAP_SECS=4)
PIXELFLOW_EXTRACTION_GAP_ARENA_DIR=<dir of the 206 .arena dumps> \
  PIXELFLOW_EXTRACTION_GAP_SECS=4 PIXELFLOW_EXTRACTION_GAP_EXPANSIONS=15000000 \
  RUST_MIN_STACK=268435456 \
  cargo test -p pixelflow-search --release --lib -- --ignored extraction_objective_measurement
# shape A
PIXELFLOW_GUARD_TELEMETRY=1 cargo run --release -p pixelflow-runtime --example bench_scene_chrome
cargo run --release --target x86_64-apple-darwin -p pixelflow-runtime --example bench_scene_chrome   # SSE2 tier
# shape B, once per build, several processes
cargo run --release -p pixelflow-pipeline --example glyph_compile_report
PIXELFLOW_GLYPH_REPORT_MODE=warm cargo run --release -p pixelflow-pipeline --example glyph_compile_report
```
