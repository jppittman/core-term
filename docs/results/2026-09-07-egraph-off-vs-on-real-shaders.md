# Saturation on vs off, on every shipped kernel

The "F: no e-graph" column of [`2026-09-06-egraph-at-production-scale.md`](../plans/2026-09-06-egraph-at-production-scale.md) §7, measured. Every kernel is compiled through the production path (`optimize_runtime_arena` → `relink` → `emit::compile`, the three calls `jit_cache::compile` makes; buffer-free kernels are asserted byte-identical to `Manifold::compile`) twice: `PIXELFLOW_SATURATION=off` runs the `Identity` path (`LowerDwrt`, `ExpandReduce`, no saturation), `on` is production. Two in-harness arms beside them: `cse-only` (the production optimizer at zero rewrite rounds — insert, extract — so hash-consing's share is separated from the rules') and `with-select-hoist` (production's rules plus the three `SelectHoistUnary` rules, which are **not** in `all_rules()`). Deterministic columns are the claim; the clock, when taken, is a sign. Per-kernel rows: the `.csv`/`.json` beside this file.

Host: aarch64 (NEON, 4 lanes), macOS, load average 8–43 throughout (other sessions); **no clock taken** — every column here is deterministic.
**chrome (both kernels) has no `off` row**: with saturation off the emitter panics — `CBZ/CBNZ branch offset out of range (±1MB)` (`pixelflow-codegen/src/emit/aarch64.rs:849`) at a 247,789-entry body schedule (chrome_packed; chrome_R: 84,315). The constructed `Kernel` is a tree of 335,411 reachable nodes (`Kernel` composition does not hash-cons); the e-graph's insertion hash-consing takes it to 465 and the rules to 398. Without the e-graph the chrome scene does not compile on this target.

## Verdict per shader class

`Δ` is on relative to off; negative is saturation winning. `mem ops` is the trip-weighted dynamic memory-op count (`dyn_memory_ops`). `cse` is the zero-round arm: the gap between `off` and `cse` is hash-consing, between `cse` and `on` is the rules.

| class | n | Σ bytes off → cse → on | Σ dag_cost off → cse → on | Σ mem ops off → on | median clock Δ | picture identical | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| cellgrid | 1 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | 42154099 → 5227011 (-87.6%) | - | 1/1 | helps (static) |
| psychedelic | 1 | 10496 → 2176 → 2368 (-77.4%) | 4948 → 860 → 825 (-83.3%) | 36333378 → 4324 (-100.0%) | - | 0/1 | helps (static) |

## Kernels with no `off` row: the un-saturated arena does not compile

Run with `--skip` in `off` mode because the emitter panics on them (the notes above quote the panic). The columns are the `on` row and, where present, the `cse-only` arm.

| kernel | extent | nodes in → cse → on | dag_cost in → cse → on | bytes cse → on | schedule on | guarded/selects on | spills on | saturation (apps / rounds / classes / stop) | compile on (ms) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| chrome_R | 1920×1080 | 114633 → 351 → 300 | 371499 → 1726 → 1418 | 2608 → 2640 | 288 | 433/9 | 23 | 5669 apps / 2 it / 4894 cls / ClassCap | 74.0 |
| chrome_packed | 1920×1080 | 335411 → 465 → 398 | 1090401 → 2031 → 1668 | 3456 → 3424 | 385 | 614/17 | 22 | 5531 apps / 2 it / 4940 cls / ClassCap | 222.9 |

## The headline kernels

| kernel | extent | nodes in → off → cse → on | bytes off → cse → on | dag_cost off → cse → on | schedule off → on | guarded/selects off → on | spills off → on | mem ops off → on | saturation (apps / rounds / classes / stop) | compile off → on (ms; saturation share) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| cellgrid_80x24_d2 | 1600×768 | 667 → 667 → 123 → 117 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | 779 → 217 | 0/0 → 0/0 | 27 → 7 | 42154099 → 5227011 (-87.6%) | 6841 apps / 3 it / 4641 cls / ClassCap | 1.6 → 25.2 (97%) |
| psychedelic_packed | 1920×1080 | 1246 → 1246 → 112 → 112 | 10496 → 2176 → 2368 (-77.4%) | 4948 → 860 → 825 (-83.3%) | 1072 → 127 | 174/6 → 29/1 | 11 → 0 | 36333378 → 4324 (-100.0%) | 7319 apps / 3 it / 4676 cls / ClassCap | 4.9 → 24.7 (96%) |

## Where saturation makes the kernel worse, and which rule

Sorted by bytes Δ. Rule column: the on-extraction's rules from the provenance journal, `strict/tight/fired` — strict credits an application only when its own e-node was chosen (`EpisodeLabels::compute_strict`), tight is `derivation_ancestors_tight`.

| kernel | bytes off → cse → on | dag_cost off → cse → on | mem ops Δ | rules (strict/tight/fired) |
|---|---:|---:|---:|---|
| (none: no kernel grew in bytes or dag_cost) | | | | |

### The largest wins, for the same rule attribution

| kernel | bytes off → cse → on | dag_cost off → cse → on | rules (strict/tight/fired) |
|---|---:|---:|---|
| psychedelic_packed | 10496 → 2176 → 2368 (-77.4%) | 4948 → 860 → 825 (-83.3%) | `fma-fusion` 14/371/1194, `commutative(Mul)` 0/360/1329, `associative(Mul)` 7/226/768, `reverse-associative(Mul)` 1/190/918, `distribute` 4/142/515 |
| cellgrid_80x24_d2 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | `fma-fusion` 20/214/1515, `commutative(Add)` 0/137/861, `associative(Add)` 1/72/631, `reverse-associative(Add)` 0/58/614, `commutative(Mul)` 0/47/757 |

## Which rules are load-bearing on real shaders (all kernels, production run)

| rule | kernels where load-bearing | Σ strict | Σ tight | Σ fired |
|---|---:|---:|---:|---:|
| `fma-fusion` | 2 | 34 | 585 | 2709 |
| `constant-fold` | 2 | 13 | 43 | 116 |
| `associative(Mul)` | 2 | 9 | 237 | 1261 |
| `canonicalize(Sub)` | 2 | 5 | 17 | 63 |
| `distribute` | 2 | 4 | 162 | 1073 |
| `reverse-associative(Add)` | 2 | 3 | 134 | 1110 |
| `associative(Add)` | 2 | 1 | 168 | 1079 |
| `factor` | 2 | 1 | 19 | 1114 |
| `reverse-associative(Mul)` | 2 | 1 | 196 | 1439 |
| `associative(Max)` | 1 | 0 | 4 | 8 |
| `associative(Min)` | 1 | 0 | 3 | 18 |
| `canonicalize(Div)` | 1 | 0 | 6 | 20 |
| `commutative(Add)` | 2 | 0 | 260 | 1348 |
| `commutative(Max)` | 2 | 0 | 8 | 41 |
| `commutative(Min)` | 2 | 0 | 11 | 54 |
| `commutative(Mul)` | 2 | 0 | 407 | 2086 |
| `exp-homomorphism` | 1 | 0 | 5 | 21 |
| `halving` | 1 | 0 | 2 | 41 |
| `idempotent(Min)` | 1 | 0 | 1 | 6 |
| `identity(Mul)` | 2 | 0 | 40 | 414 |
| `reverse-associative(Max)` | 1 | 0 | 3 | 12 |
| `reverse-associative(Min)` | 1 | 0 | 3 | 21 |
| `sin-angle-addition` | 1 | 0 | 3 | 19 |

## `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`)

**Not in production.** The three rules live in `round2_rules::experimental_rules()` and are not part of `all_rules()`, so they fire zero times in every production compile above by construction. The `with-select-hoist` arm adds them to production's rules: fired in **0 of 2** measured kernels (0 applications); tight-load-bearing in **0** (0); strict 0.

| kernel | fired | tight | strict | bytes on → +hoist | dag_cost on → +hoist | guarded/selects on → +hoist | schedule on → +hoist |
|---|---:|---:|---:|---:|---:|---:|---:|
| (never fired on any real shader; bytes identical everywhere) | | | | | | | |

Adding the rule changed the emitted bytes of **0** kernels; Σ guarded-value delta (+hoist − on) over all measured kernels: **+0**.

## Correctness

Same-form: `eval_scalar` of the emitted arena vs the JIT at 256 sampled pixels (a difference here is a JIT bug). Cross-form: `eval_scalar` of the legalized arena as constructed vs the JIT of the on-extraction (what the rewrites moved; divergence at singularities is the algebraic contract, not a defect).

- same-form NaN mismatches over all kernels: on **0**, off **0**; worst same-form |Δ|: on `psychedelic_packed` 0e0 (0 NaN), off `psychedelic_packed` 0e0
- worst cross-form |Δ| (on): `psychedelic_packed` 0e0 (0 NaN)
- full-extent output bit-identical off vs on: **1 of 2** kernels
- packed kernels (on): `cellgrid_80x24_d2`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0); `chrome_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 58 (max byte Δ 1); `psychedelic_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0)
- instrument = production path: `Manifold::compile` bytes differed for 0 kernels []; probe extraction differed from production for 0 kernels []

## Saturation's share of compile time

`optimize_ms` is `optimize_runtime_arena` (legalize + saturate + extract); `emit_ms` is `emit::compile`. Wall clock at the load stated above — a ratio, not a number.

| class | Σ compile off (ms) | Σ compile on (ms) | Σ saturation (on − off optimize) | share of on |
|---|---:|---:|---:|---:|
| cellgrid | 1.6 | 25.2 | 24.4 | 97% |
| psychedelic | 4.9 | 24.7 | 23.6 | 96% |
| chrome_R (on only) | - | 74.0 | 70.8 | - |
| chrome_packed (on only) | - | 222.9 | 217.5 | - |
