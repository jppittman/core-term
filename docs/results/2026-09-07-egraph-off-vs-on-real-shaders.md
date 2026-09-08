# Saturation on vs off, on every shipped kernel

The "F: no e-graph" column of [`2026-09-06-egraph-at-production-scale.md`](../plans/2026-09-06-egraph-at-production-scale.md) §7, measured. Every kernel is compiled through the production path (`optimize_runtime_arena` → `relink` → `emit::compile`, the three calls `jit_cache::compile` makes; buffer-free kernels are asserted byte-identical to `Manifold::compile`) twice: `PIXELFLOW_SATURATION=off` runs the `Identity` path (`LowerDwrt`, `ExpandReduce`, no saturation), `on` is production. Two in-harness arms beside them: `cse-only` (the production optimizer at zero rewrite rounds — insert, extract — so hash-consing's share is separated from the rules') and `with-select-hoist` (production's rules plus the three `SelectHoistUnary` rules, which are **not** in `all_rules()`). Deterministic columns are the claim; the clock, when taken, is a sign. Per-kernel rows: the `.csv`/`.json` beside this file.

Host: aarch64 (NEON, 4 lanes), macOS, load average 8–43 throughout (other sessions); **no clock taken** — every column here is deterministic.
**chrome (both kernels) has no `off` row**: with saturation off the emitter panics — `CBZ/CBNZ branch offset out of range (±1MB)` (`pixelflow-codegen/src/emit/aarch64.rs:849`) at a 247,789-entry body schedule (chrome_packed; chrome_R: 84,315). The constructed `Kernel` is a tree of 335,411 reachable nodes (`Kernel` composition does not hash-cons); the e-graph's insertion hash-consing takes it to 465 and the rules to 398. Without the e-graph the chrome scene does not compile on this target.

## Verdict per shader class

`Δ` is on relative to off; negative is saturation winning. `mem ops` is the trip-weighted dynamic memory-op count (`dyn_memory_ops`). `cse` is the zero-round arm: the gap between `off` and `cse` is hash-consing, between `cse` and `on` is the rules.

| class | n | Σ bytes off → cse → on | Σ dag_cost off → cse → on | Σ mem ops off → on | median clock Δ | picture identical | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| cellgrid | 1 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | 42154099 → 5227011 (-87.6%) | - | 1/1 | helps (static) |
| glyph16 | 95 | 1500068 → 883060 → 846068 (-43.6%) | 745471 → 375881 → 368113 (-50.6%) | 2596166 → 970018 (-62.6%) | - | 1/95 | helps (static) |
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
| glyph16_U0033 | 43408 → 24256 → 22832 (-47.4%) | 21916 → 10737 → 10873 (-50.4%) | `commutative(Mul)` 0/565/661, `commutative(Add)` 0/435/436, `associative(Add)` 42/173/173, `distribute` 40/155/155, `doubling` 0/134/218 |
| glyph16_U0038 | 56288 → 30512 → 29664 (-47.3%) | 28621 → 13849 → 13645 (-52.3%) | `commutative(Mul)` 0/725/853, `commutative(Add)` 0/550/551, `canonicalize(Sub)` 0/132/132, `commutative(Max)` 0/94/94, `commutative(Min)` 0/65/65 |
| glyph16_U0061 | 40544 → 22960 → 21376 (-47.3%) | 20428 → 10044 → 10230 (-49.9%) | `commutative(Mul)` 0/530/619, `commutative(Add)` 0/416/416, `associative(Add)` 52/221/221, `doubling` 1/164/279, `distribute` 50/144/145 |
| glyph16_U0028 | 14528 → 8640 → 7712 (-46.9%) | 7237 → 3607 → 3145 (-56.5%) | `commutative(Mul)` 0/618/1648, `commutative(Add)` 0/360/699, `fma-fusion` 40/344/867, `reverse-associative(Mul)` 7/196/635, `associative(Mul)` 28/130/336 |
| glyph16_U0039 | 43024 → 24448 → 22896 (-46.8%) | 21730 → 10770 → 10892 (-49.9%) | `commutative(Mul)` 0/572/668, `commutative(Add)` 0/428/428, `associative(Add)` 41/173/173, `distribute` 39/157/157, `doubling` 1/138/219 |
| glyph16_U0036 | 42784 → 24208 → 22896 (-46.5%) | 21546 → 10678 → 10849 (-49.6%) | `commutative(Mul)` 0/564/656, `commutative(Add)` 0/420/420, `associative(Add)` 44/180/180, `distribute` 42/149/149, `doubling` 1/139/227 |
| glyph16_U0040 | 56160 → 31936 → 30096 (-46.4%) | 28979 → 14237 → 14070 (-51.4%) | `commutative(Mul)` 0/685/799, `commutative(Add)` 0/569/569, `canonicalize(Sub)` 0/140/140, `commutative(Max)` 0/99/99, `commutative(Min)` 0/69/69 |
| glyph16_U004F | 28208 → 15344 → 15184 (-46.2%) | 14365 → 6901 → 6770 (-52.9%) | `fma-fusion` 82/404/556, `commutative(Mul)` 0/328/426, `associative(Mul)` 14/286/343, `commutative(Add)` 0/259/279, `associative(Add)` 34/139/139 |

## Which rules are load-bearing on real shaders (all kernels, production run)

| rule | kernels where load-bearing | Σ strict | Σ tight | Σ fired |
|---|---:|---:|---:|---:|
| `fma-fusion` | 95 | 2606 | 34145 | 149162 |
| `constant-fold` | 96 | 2292 | 4087 | 24116 |
| `canonicalize(Div)` | 95 | 1702 | 1835 | 3003 |
| `canonicalize(Sub)` | 96 | 1571 | 4283 | 6853 |
| `factor` | 93 | 1545 | 4881 | 38160 |
| `associative(Add)` | 94 | 1462 | 13385 | 99929 |
| `associative(Mul)` | 93 | 1273 | 13858 | 41265 |
| `distribute` | 95 | 870 | 7882 | 36927 |
| `reverse-associative(Mul)` | 93 | 688 | 6217 | 47312 |
| `doubling` | 93 | 98 | 4804 | 15784 |
| `reverse-associative(Add)` | 92 | 57 | 10568 | 109358 |
| `annihilator` | 93 | 0 | 660 | 13339 |
| `associative(Max)` | 1 | 0 | 4 | 8 |
| `associative(Min)` | 1 | 0 | 3 | 18 |
| `cancellation(Sub)` | 39 | 0 | 75 | 249 |
| `commutative(Add)` | 96 | 0 | 31620 | 107160 |
| `commutative(Max)` | 95 | 0 | 2626 | 4833 |
| `commutative(Min)` | 96 | 0 | 2077 | 4450 |
| `commutative(Mul)` | 96 | 0 | 27803 | 74307 |
| `exp-homomorphism` | 1 | 0 | 5 | 21 |
| `halving` | 13 | 0 | 115 | 2527 |
| `idempotent(Min)` | 1 | 0 | 1 | 6 |
| `identity(Add)` | 94 | 0 | 1597 | 39221 |
| `identity(Mul)` | 88 | 0 | 267 | 2288 |
| `inverse-annihilation(Sub)` | 39 | 0 | 39 | 1404 |

## `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`)

**Not in production.** The three rules live in `round2_rules::experimental_rules()` and are not part of `all_rules()`, so they fire zero times in every production compile above by construction. The `with-select-hoist` arm adds them to production's rules: fired in **14 of 97** measured kernels (93 applications); tight-load-bearing in **14** (93); strict 0.

| kernel | fired | tight | strict | bytes on → +hoist | dag_cost on → +hoist | guarded/selects on → +hoist | schedule on → +hoist |
|---|---:|---:|---:|---:|---:|---:|---:|
| glyph16_U0028 | 8 | 8 | 0 | 7712 → 7712 | 3145 → 3145 | 490/25 → 490/25 | 210 → 210 |
| glyph16_U0044 | 8 | 8 | 0 | 8288 → 8288 | 3264 → 3264 | 516/27 → 516/27 | 229 → 229 |
| glyph16_U004A | 8 | 8 | 0 | 8688 → 8688 | 3386 → 3386 | 545/29 → 545/29 | 247 → 247 |
| glyph16_U0050 | 8 | 8 | 0 | 8512 → 8512 | 3306 → 3306 | 524/28 → 524/28 | 237 → 237 |
| glyph16_U0068 | 8 | 8 | 0 | 8944 → 8944 | 3405 → 3405 | 545/30 → 545/30 | 253 → 253 |
| glyph16_U006E | 8 | 8 | 0 | 8816 → 8816 | 3401 → 3401 | 545/30 → 545/30 | 253 → 253 |
| glyph16_U0072 | 8 | 8 | 0 | 8800 → 8800 | 3386 → 3386 | 532/29 → 532/29 | 245 → 245 |
| glyph16_U0075 | 8 | 8 | 0 | 9152 → 9152 | 3423 → 3423 | 552/30 → 552/30 | 254 → 254 |
| glyph16_U0029 | 7 | 7 | 0 | 7072 → 7072 | 2914 → 2914 | 448/23 → 448/23 | 197 → 197 |
| glyph16_U0079 | 6 | 6 | 0 | 7296 → 7296 | 2840 → 2840 | 437/24 → 437/24 | 210 → 210 |
| glyph16_U0066 | 4 | 4 | 0 | 4992 → 4992 | 1959 → 1959 | 298/20 → 298/20 | 166 → 166 |
| glyph16_U006A | 4 | 4 | 0 | 4992 → 4992 | 1964 → 1964 | 293/19 → 293/19 | 161 → 161 |
| glyph16_U006C | 4 | 4 | 0 | 4720 → 4720 | 1867 → 1867 | 278/17 → 278/17 | 146 → 146 |
| glyph16_U0074 | 4 | 4 | 0 | 4992 → 4992 | 1960 → 1960 | 297/20 → 297/20 | 165 → 165 |

Adding the rule changed the emitted bytes of **0** kernels; Σ guarded-value delta (+hoist − on) over all measured kernels: **+0**.

## Correctness

Same-form: `eval_scalar` of the emitted arena vs the JIT at 256 sampled pixels (a difference here is a JIT bug). Cross-form: `eval_scalar` of the legalized arena as constructed vs the JIT of the on-extraction (what the rewrites moved; divergence at singularities is the algebraic contract, not a defect).

- same-form NaN mismatches over all kernels: on **0**, off **0**; worst same-form |Δ|: on `glyph16_U0067` 3.3974647521972656e-6 (0 NaN), off `glyph16_U0067` 3.9637088775634766e-6
- worst cross-form |Δ| (on): `glyph16_U0059` 3.814697265625e-6 (0 NaN)
- full-extent output bit-identical off vs on: **2 of 97** kernels
- packed kernels (on): `cellgrid_80x24_d2`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0); `chrome_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 58 (max byte Δ 1); `psychedelic_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0)
- instrument = production path: `Manifold::compile` bytes differed for 0 kernels []; probe extraction differed from production for 0 kernels []

## Saturation's share of compile time

`optimize_ms` is `optimize_runtime_arena` (legalize + saturate + extract); `emit_ms` is `emit::compile`. Wall clock at the load stated above — a ratio, not a number.

| class | Σ compile off (ms) | Σ compile on (ms) | Σ saturation (on − off optimize) | share of on |
|---|---:|---:|---:|---:|
| cellgrid | 1.6 | 25.2 | 24.4 | 97% |
| glyph16 | 1782.9 | 5818.8 | 4990.9 | 86% |
| psychedelic | 4.9 | 24.7 | 23.6 | 96% |
| chrome_R (on only) | - | 74.0 | 70.8 | - |
| chrome_packed (on only) | - | 222.9 | 217.5 | - |
