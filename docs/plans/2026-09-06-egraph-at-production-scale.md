# The e-graph at production scale: the chrome scene as the case to optimize around

**Audience:** the research arm. **Status:** a use case and a set of measured facts,
not a plan. Everything numeric here was measured during the kernel-with-a-lattice
programme (S1–S4b-2, `2026-09-06-kernel-with-a-lattice.md`) and is reproducible
from the tree at `main`; the recipe for each number is named where it appears.

## 0. The ask in one paragraph

The JIT is now the only tier, and it wins on every kernel we ship. The first
production kernel large enough to hit the e-graph's class cap is the chrome scene
in `bench_scene_chrome`: it saturates for two iterations and stops at 4,913 of 5,000
classes, and **giving it more classes made its code slower** (5,000 → 60,000: 30× the
compile time, 15% slower). So the scaling problem is not "let saturation run
longer"; it is that the additive latency prior plus swap refinement cannot rank a
larger space, on exactly the kernel shape — guarded selects, hoisted invariants,
shared geometry across channels — where cost is non-additive. That is the regime the
schedule-cost denotation names and the Guide programme was designed for, and this
kernel is the first production instance of it. What we want from research is an
extraction (and, second, a guided saturation) that turns headroom into ns/px on this
kernel without moving its compile time, and that leaves the 193 small glyph kernels
untouched.

## 1. The two production shapes

| shape | example | count | size (schedule entries) | what matters |
|---|---|---:|---:|---|
| **A: one scene kernel per frame shape** | chrome scene, psychedelic shader, the terminal's cell grid | one per scene | 401 (chrome), ~200 (psychedelic) | **code quality.** Compiled once per lattice shape (time rides on `W`, so no per-frame recompile), runs every frame, every stripe. Compile is a budget, not a metric: ~220 ms today. |
| **B: many small kernels at startup** | the glyph atlas — `atlas.warm(' '..='~')`, 95 kernels, at every font size | hundreds per session | 37–87 | **compile latency**, summed. A glyph bakes once and is cached; its runtime is a rounding error. |

The measurement corpus (`collapse_cost`, 208 kernels: 193 production glyph bakes, 15
synthetic) is shape B. Every extraction and cost-model result to date was obtained on
shape B. The chrome scene is shape A and is the case this document is about; any
result must hold on both.

## 2. The chrome kernel, exactly

Four channel kernels sharing one geometry — `Ray::through_screen` → `Sphere::hit` →
Householder reflection → the world (checker floor with a box-filtered edge, sky) — with
exact `dx()`/`dy()` partials of the hit point for the checker's antialiasing, packed to
one `u32` inside the kernel and selected on packed words (S3b). Source:
`pixelflow-graphics/src/scene3d.rs`; gate: `pixelflow-runtime/examples/bench_scene_chrome.rs`.

| quantity | value | how measured |
|---|---:|---|
| arena nodes at construction | 416,420 (S3b tree; 429,395 in S3) | `Kernel::parts()` before compile |
| build time (construction, no compile) | 14–27 ms | gate example |
| e-classes after hash-consing at e-graph build | ~4,900 (from ~193k distinct nodes in the S3 telemetry) | `--features saturation-telemetry` |
| tier | classical (node count > 50): 100 iterations, 5,000 classes, 200,000 applications, 300 s ceiling | `SaturationConfig::classical()` |
| stop reason | **class cap, after 2 iterations, 4,913/5,000** | telemetry `stop` field |
| compile (saturate + extract + emit) | 181–229 ms | gate example |
| extracted schedule | 401 entries, 17 selects, 642 entries under a guard | `PIXELFLOW_GUARD_TELEMETRY=1` |
| emitted code | 9,760 B SSE2 / 7,645 B AVX-512 | `PackedManifold::code_bytes()` |
| runtime, 1920×1080, single thread | 12.2 ns/px SSE2 / 4.8 AVX-512 | gate, median of 9 |
| against the retired template tier | 1.13–1.75× faster at every thread count | S3b landing block |

**Where the 416k nodes come from.** Not the derivative pass: `lower_dwrt`
differentiates the DAG with a per-node memo (`pixelflow-ir/src/passes.rs`,
`differentiate`). They come from **construction**: `Kernel` composition splices copies
(`at`, `select`, per-channel `Rgba` operations each re-splice the hit point), so the
tree the e-graph receives is ~40× its own DAG. Hash-consing at e-graph build folds
it, and that fold alone lands within 90 classes of the cap. This is the reason the
cap binds on iteration 2: almost the whole budget is spent representing the input.

## 3. The evidence that more saturation is not the lever

All from the S3 landing block, SSE2, single thread, pre-S3b tree (43.9 ns/px baseline):

| experiment | compile | runtime | reading |
|---|---:|---:|---|
| class cap 5,000 → 60,000 | **30×** | **51 ns/px (+15%)** | more classes, worse extraction |
| `world(select(m, R, D))` (one world, select on the ray) | 0.2 s → 2.0 s; 4.6M nodes | −10% | a construction-shape change, superseded by S3b's one-select colour |
| S3b: one select per colour + guard clustering + cost bound | unchanged | **12.2 ns/px (−72%)** | the win came from the *schedule*, not from saturation |

The third row is the important one. The 3.6× improvement on this kernel came from
making the emitter's guard analysis see a contiguous arm and from bounding when a
guard pays for its branch — schedule decisions the e-graph does not represent and the
additive prior cannot see. `dyn_memory_ops`, the corpus predictor validated on shape
B, reported +40% on that change while the clock improved 7–8%: it trip-weights every
schedule entry as executed and is blind to a skipped range. That is the concrete
failure of an additive cost on this kernel.

## 4. What is engineering, not research — do these first, so research measures against the right baseline

1. **Hash-cons at construction.** The deferred "shared-store" note on `Kernel::sum`:
   one interned arena that composition indexes rather than copies. It removes the
   40× before the e-graph sees it, so the class cap stops binding on the input and
   starts binding on rewrites — which is the only regime in which "raise the cap"
   is a meaningful experiment. Measure: arena node count, build time, classes at
   stop, and the emitted bytes must be identical.
2. **The `Sqrt` derivative rule** pushes a fresh `Rsqrt(u)` estimate chain rather
   than sharing anything with the value's `Sqrt(u)`; every square root on the hit
   path pays it once per partial. A constant factor on the ~0.8× that two partials
   cost today (S4b-2 open items). Measure with `collapse_cost`.
3. **A class-cap diagnosis**, using the provenance journal
   (`--features provenance-journal`): of the ~4,900 classes at stop, how many are
   the hash-consed input, how many are numeric-rule expansions (`recip`/`sqrt`
   families on the ray normalizations) that never reach an extraction, and how many
   are genuine alternatives. This tells research whether the cap is guarding memory
   (its stated purpose, `saturate.rs`) or meaning.

## 5. What to optimize around

Each item has a metric, an oracle, and the document that already denotes it.

**5.1 Extraction under a non-additive cost — the first-order ask.**
The chrome kernel is the candidate for the schedule-cost trigger
(`2026-09-01-schedule-cost-model-denotation.md` §5.2): (a) does its e-graph admit ≥2
extractions with distinct level/guard assignments, and (b) does the measured best-of-k
beat the analytic DP's choice by a geomean clearing ±5%? If (a) and (b) hold, the
`Reranker` seam (`pixelflow-search/src/egraph/extract.rs`) gets its first
implementation with a residual over the latency prior. The residual's first named
term is **mask coherence** — how often a select's mask is uniform per batch — which is
data: the sphere's silhouette mask is uniform in 97% of batches, a glyph's coverage
mask in almost none, and the static bound (`MISPREDICT_PENALTY_CYCLES`) refuses guards
under 16 cycles precisely because it cannot know. Metric: ns/px on `bench_scene_chrome`
at both tiers, compile time unchanged. Oracle: arm C of §5.3 there, measured best of k.

**5.2 Guided saturation on a kernel that hits the cap.**
The Guide programme (`2026-08-31-guide-design-revision.md`) measured a 2.6× oracle
savings bound (labeler) on the shape-B corpus. The chrome kernel is the case where the
budget that binds is classes rather than applications, so the question changes shape:
not "which applications were load-bearing" but "which *classes* were". Ask: the
anytime curve of extracted cost versus classes admitted, with the chrome kernel as the
held-out production instance in the pre-registered Phase 3 experiment. Metric: cost
at a fixed class budget; then ns/px.

**5.3 The compile budget, stated so it can be checked.**
Shape A: a scene compiles at ~220 ms today and a resize recompiles; anything under
~250 ms is invisible, 2 s is not (the `world(select)` row). Shape B: 95 glyph compiles
at startup per font size; a research result that adds 5 ms per kernel adds half a
second to startup. Any change must report both, and the glyph kernels must stay
byte-identical unless the change is meant for them.

## 6. How to measure, and the traps already found

- **Runtime:** `cargo run --release -p pixelflow-runtime --example bench_scene_chrome`,
  SSE2 default and `RUSTFLAGS="-C target-feature=+avx512f,+avx512dq"`, threads
  1 / cores / 12, idle host, load stated. Pixel agreement before timing.
- **Saturation:** `--features saturation-telemetry` writes a JSONL record per
  production saturation (tier, node count, stop reason, classes, applications,
  wall clock, extracted cost) to `$PIXELFLOW_SATURATION_TELEMETRY`.
- **Guards:** `PIXELFLOW_GUARD_TELEMETRY=1` prints, per compiled program, schedule
  entries, selects, arm-exclusive entries, and entries under a guard.
- **Corpus:** `collapse_cost` (pixelflow-pipeline), 208 kernels, both tiers.
  **Compare two builds' rows, never a hash against prose** — two stages recorded
  column hashes that did not reproduce. Per-kernel clock ratios are not trustworthy
  below ~10% on the 4-core host (byte-identical code measured 4× apart between runs,
  A/A floor ~1% within a run); the aggregate sign is the claim.
- **Determinism:** budgets are rule applications, classes, and iterations — never
  wall clock (`2026-09-01-production-budget-determinism.md`). The 300 s safety
  ceiling panics rather than truncating; a panic is a finding, not a knob.

## 7. Pointers

`pixelflow-search/src/egraph/{saturate,extract,cost,provenance,labeler}.rs`;
`pixelflow-search/src/runtime.rs` (`optimize_runtime_arena`: `LowerDwrt` →
`ExpandReduce` → `Saturate::runtime`); `pixelflow-codegen/src/emit/guards.rs`;
`pixelflow-graphics/src/scene3d.rs`; the landing blocks S3, S3b and S4b-2 in
`2026-09-06-kernel-with-a-lattice.md`.
