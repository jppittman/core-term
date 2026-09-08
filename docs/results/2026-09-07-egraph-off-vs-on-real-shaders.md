# Saturation on vs off, on every shipped kernel

The "F: no e-graph" column of [`2026-09-06-egraph-at-production-scale.md`](../plans/2026-09-06-egraph-at-production-scale.md) §7, measured. Every kernel is compiled through the production path (`optimize_runtime_arena` → `relink` → `emit::compile`, the three calls `jit_cache::compile` makes; buffer-free kernels are asserted byte-identical to `Manifold::compile`) twice: `PIXELFLOW_SATURATION=off` runs the `Identity` path (`LowerDwrt`, `ExpandReduce`, no saturation), `on` is production. Two in-harness arms beside them: `cse-only` (the production optimizer at zero rewrite rounds — insert, extract — so hash-consing's share is separated from the rules') and `with-select-hoist` (production's rules plus the three `SelectHoistUnary` rules, which are **not** in `all_rules()`). Deterministic columns are the claim; the clock, when taken, is a sign. Per-kernel rows: the `.csv`/`.json` beside this file.

Host: aarch64 (NEON, 4 lanes), macOS, load average 8–43 throughout (other sessions); **no clock taken** — every column here is deterministic.
**chrome (both kernels) has no `off` row**: with saturation off the emitter panics — `CBZ/CBNZ branch offset out of range (±1MB)` (`pixelflow-codegen/src/emit/aarch64.rs:849`) at a 247,789-entry body schedule (chrome_packed; chrome_R: 84,315). The constructed `Kernel` is a tree of 335,411 reachable nodes (`Kernel` composition does not hash-cons); the e-graph's insertion hash-consing takes it to 465 and the rules to 398. Without the e-graph the chrome scene does not compile on this target.

## Verdict per shader class

`Δ` is on relative to off; negative is saturation winning. `mem ops` is the trip-weighted dynamic memory-op count (`dyn_memory_ops`). `cse` is the zero-round arm: the gap between `off` and `cse` is hash-consing, between `cse` and `on` is the rules.

| class | n | Σ bytes off → cse → on | Σ dag_cost off → cse → on | Σ mem ops off → on | median clock Δ | picture identical | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| bench | 3 | 73360 → 42256 → 41184 (-43.9%) | 37249 → 18513 → 18527 (-50.3%) | 891017 → 302199 (-66.1%) | - | 0/3 | helps (static) |
| bench_wide | 1 | 28208 → 15344 → 15184 (-46.2%) | 14365 → 6901 → 6770 (-52.9%) | 6242609 → 1751202 (-71.9%) | - | 0/1 | helps (static) |
| cellgrid | 1 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | 42154099 → 5227011 (-87.6%) | - | 1/1 | helps (static) |
| glyph16 | 95 | 1500068 → 883060 → 846068 (-43.6%) | 745471 → 375881 → 368113 (-50.6%) | 2596166 → 970018 (-62.6%) | - | 1/95 | helps (static) |
| glyph32 | 95 | 1500068 → 883060 → 846100 (-43.6%) | 745471 → 375881 → 368113 (-50.6%) | 10359990 → 3570226 (-65.5%) | - | 1/95 | helps (static) |
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
| bench_A_linear | 40×45 | 271 → 271 → 157 → 169 | 2128 → 1744 → 1968 (-7.5%) | 823 → 618 → 670 (-18.6%) | 191 → 71 | 242/7 → 77/7 | 0 → 0 | 0 → 3522 (+0.0%) | 8114 apps / 3 it / 3006 cls / ClassCap | 2.7 → 51.8 (95%) |
| bench_O_quadratic | 40×45 | 1799 → 3655 → 1536 → 1482 | 28208 → 15344 → 15184 (-46.2%) | 14365 → 6901 → 6770 (-52.9%) | 3503 → 401 | 9730/209 → 979/49 | 226 → 0 | 390359 → 117702 (-69.8%) | 3328 apps / 1 it / 4868 cls / ClassCap | 39.1 → 34.7 (55%) |
| bench_O_quadratic_wide | 640×45 | 1799 → 3655 → 1536 → 1482 | 28208 → 15344 → 15184 (-46.2%) | 14365 → 6901 → 6770 (-52.9%) | 3503 → 401 | 9730/209 → 979/49 | 226 → 0 | 6242609 → 1751202 (-71.9%) | 3328 apps / 1 it / 4868 cls / ClassCap | 38.7 → 33.6 (56%) |
| bench_S_complex | 40×45 | 2815 → 5623 → 2463 → 2453 | 43024 → 25168 → 24032 (-44.1%) | 22061 → 10994 → 11087 (-49.7%) | 5359 → 652 | 14850/317 → 1580/77 | 291 → 0 | 500658 → 180975 (-63.9%) | 2485 apps / 1 it / 5000 cls / ClassCap | 56.1 → 42.4 (52%) |
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
| glyph32_U0033 | 43408 → 24256 → 22832 (-47.4%) | 21916 → 10737 → 10873 (-50.4%) | `commutative(Mul)` 0/565/661, `commutative(Add)` 0/435/436, `associative(Add)` 42/173/173, `distribute` 40/155/155, `doubling` 0/134/218 |
| glyph16_U0038 | 56288 → 30512 → 29664 (-47.3%) | 28621 → 13849 → 13645 (-52.3%) | `commutative(Mul)` 0/725/853, `commutative(Add)` 0/550/551, `canonicalize(Sub)` 0/132/132, `commutative(Max)` 0/94/94, `commutative(Min)` 0/65/65 |
| glyph32_U0038 | 56288 → 30512 → 29664 (-47.3%) | 28621 → 13849 → 13645 (-52.3%) | `commutative(Mul)` 0/725/853, `commutative(Add)` 0/550/551, `canonicalize(Sub)` 0/132/132, `commutative(Max)` 0/94/94, `commutative(Min)` 0/65/65 |
| glyph16_U0061 | 40544 → 22960 → 21376 (-47.3%) | 20428 → 10044 → 10230 (-49.9%) | `commutative(Mul)` 0/530/619, `commutative(Add)` 0/416/416, `associative(Add)` 52/221/221, `doubling` 1/164/279, `distribute` 50/144/145 |
| glyph32_U0061 | 40544 → 22960 → 21376 (-47.3%) | 20428 → 10044 → 10230 (-49.9%) | `commutative(Mul)` 0/530/619, `commutative(Add)` 0/416/416, `associative(Add)` 52/221/221, `doubling` 1/164/279, `distribute` 50/144/145 |
| glyph16_U0028 | 14528 → 8640 → 7712 (-46.9%) | 7237 → 3607 → 3145 (-56.5%) | `commutative(Mul)` 0/618/1648, `commutative(Add)` 0/360/699, `fma-fusion` 40/344/867, `reverse-associative(Mul)` 7/196/635, `associative(Mul)` 28/130/336 |
| glyph32_U0028 | 14528 → 8640 → 7712 (-46.9%) | 7237 → 3607 → 3145 (-56.5%) | `commutative(Mul)` 0/618/1648, `commutative(Add)` 0/360/699, `fma-fusion` 40/344/867, `reverse-associative(Mul)` 7/196/635, `associative(Mul)` 28/130/336 |

## Which rules are load-bearing on real shaders (all kernels, production run)

| rule | kernels where load-bearing | Σ strict | Σ tight | Σ fired |
|---|---:|---:|---:|---:|
| `fma-fusion` | 192 | 5364 | 69213 | 298631 |
| `constant-fold` | 194 | 4698 | 8288 | 48460 |
| `canonicalize(Div)` | 193 | 3525 | 3786 | 6120 |
| `canonicalize(Sub)` | 194 | 3240 | 8808 | 13932 |
| `factor` | 188 | 3203 | 10074 | 76259 |
| `associative(Add)` | 190 | 3033 | 27275 | 200396 |
| `associative(Mul)` | 188 | 2580 | 28203 | 82330 |
| `distribute` | 192 | 1794 | 16054 | 73471 |
| `reverse-associative(Mul)` | 188 | 1425 | 12493 | 94259 |
| `doubling` | 190 | 199 | 9979 | 32407 |
| `reverse-associative(Add)` | 186 | 113 | 21467 | 219514 |
| `annihilator` | 190 | 0 | 1345 | 26840 |
| `associative(Max)` | 1 | 0 | 4 | 8 |
| `associative(Min)` | 1 | 0 | 3 | 18 |
| `cancellation(Sub)` | 79 | 0 | 152 | 503 |
| `commutative(Add)` | 194 | 0 | 64314 | 214833 |
| `commutative(Max)` | 192 | 0 | 5418 | 9823 |
| `commutative(Min)` | 194 | 0 | 4269 | 9000 |
| `commutative(Mul)` | 194 | 0 | 56571 | 148516 |
| `exp-homomorphism` | 1 | 0 | 5 | 21 |
| `halving` | 25 | 0 | 228 | 5013 |
| `idempotent(Min)` | 1 | 0 | 1 | 6 |
| `identity(Add)` | 192 | 0 | 3219 | 78765 |
| `identity(Mul)` | 176 | 0 | 499 | 4192 |
| `inverse-annihilation(Sub)` | 79 | 0 | 79 | 2828 |

## `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`)

**Not in production.** The three rules live in `round2_rules::experimental_rules()` and are not part of `all_rules()`, so they fire zero times in every production compile above by construction. The `with-select-hoist` arm adds them to production's rules: fired in **28 of 196** measured kernels (186 applications); tight-load-bearing in **28** (186); strict 0.

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
| glyph32_U0028 | 8 | 8 | 0 | 7712 → 7712 | 3145 → 3145 | 490/25 → 490/25 | 210 → 210 |
| glyph32_U0044 | 8 | 8 | 0 | 8288 → 8288 | 3264 → 3264 | 516/27 → 516/27 | 229 → 229 |
| glyph32_U004A | 8 | 8 | 0 | 8704 → 8704 | 3386 → 3386 | 545/29 → 545/29 | 247 → 247 |
| glyph32_U0050 | 8 | 8 | 0 | 8512 → 8512 | 3306 → 3306 | 524/28 → 524/28 | 237 → 237 |
| glyph32_U0068 | 8 | 8 | 0 | 8944 → 8944 | 3405 → 3405 | 545/30 → 545/30 | 253 → 253 |
| glyph32_U006E | 8 | 8 | 0 | 8816 → 8816 | 3401 → 3401 | 545/30 → 545/30 | 253 → 253 |
| glyph32_U0072 | 8 | 8 | 0 | 8800 → 8800 | 3386 → 3386 | 532/29 → 532/29 | 245 → 245 |
| glyph32_U0075 | 8 | 8 | 0 | 9152 → 9152 | 3423 → 3423 | 552/30 → 552/30 | 254 → 254 |
| glyph16_U0029 | 7 | 7 | 0 | 7072 → 7072 | 2914 → 2914 | 448/23 → 448/23 | 197 → 197 |
| glyph32_U0029 | 7 | 7 | 0 | 7072 → 7072 | 2914 → 2914 | 448/23 → 448/23 | 197 → 197 |
| glyph16_U0079 | 6 | 6 | 0 | 7296 → 7296 | 2840 → 2840 | 437/24 → 437/24 | 210 → 210 |
| glyph32_U0079 | 6 | 6 | 0 | 7296 → 7296 | 2840 → 2840 | 437/24 → 437/24 | 210 → 210 |
| glyph16_U0066 | 4 | 4 | 0 | 4992 → 4992 | 1959 → 1959 | 298/20 → 298/20 | 166 → 166 |
| glyph16_U006A | 4 | 4 | 0 | 4992 → 4992 | 1964 → 1964 | 293/19 → 293/19 | 161 → 161 |
| glyph16_U006C | 4 | 4 | 0 | 4720 → 4720 | 1867 → 1867 | 278/17 → 278/17 | 146 → 146 |
| glyph16_U0074 | 4 | 4 | 0 | 4992 → 4992 | 1960 → 1960 | 297/20 → 297/20 | 165 → 165 |
| glyph32_U0066 | 4 | 4 | 0 | 4992 → 4992 | 1959 → 1959 | 298/20 → 298/20 | 166 → 166 |
| glyph32_U006A | 4 | 4 | 0 | 4992 → 4992 | 1964 → 1964 | 293/19 → 293/19 | 161 → 161 |
| glyph32_U006C | 4 | 4 | 0 | 4720 → 4720 | 1867 → 1867 | 278/17 → 278/17 | 146 → 146 |
| glyph32_U0074 | 4 | 4 | 0 | 4992 → 4992 | 1960 → 1960 | 297/20 → 297/20 | 165 → 165 |

Adding the rule changed the emitted bytes of **0** kernels; Σ guarded-value delta (+hoist − on) over all measured kernels: **+0**.

## Correctness

Same-form: `eval_scalar` of the emitted arena vs the JIT at 256 sampled pixels (a difference here is a JIT bug). Cross-form: `eval_scalar` of the legalized arena as constructed vs the JIT of the on-extraction (what the rewrites moved; divergence at singularities is the algebraic contract, not a defect).

- same-form NaN mismatches over all kernels: on **0**, off **0**; worst same-form |Δ|: on `glyph32_U0072` 5.245208740234375e-6 (0 NaN), off `glyph32_U0072` 5.0067901611328125e-6
- worst cross-form |Δ| (on): `glyph32_U0072` 5.125999450683594e-6 (0 NaN)
- full-extent output bit-identical off vs on: **3 of 196** kernels
- packed kernels (on): `cellgrid_80x24_d2`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0); `chrome_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 58 (max byte Δ 1); `psychedelic_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0)
- instrument = production path: `Manifold::compile` bytes differed for 0 kernels []; probe extraction differed from production for 0 kernels []

## Saturation's share of compile time

`optimize_ms` is `optimize_runtime_arena` (legalize + saturate + extract); `emit_ms` is `emit::compile`. Wall clock at the load stated above — a ratio, not a number.

| class | Σ compile off (ms) | Σ compile on (ms) | Σ saturation (on − off optimize) | share of on |
|---|---:|---:|---:|---:|
| bench | 97.8 | 128.9 | 90.1 | 70% |
| bench_wide | 38.7 | 33.6 | 18.8 | 56% |
| cellgrid | 1.6 | 25.2 | 24.4 | 97% |
| glyph16 | 1782.9 | 5818.8 | 4990.9 | 86% |
| glyph32 | 5166.4 | 8330.8 | 6782.5 | 81% |
| psychedelic | 4.9 | 24.7 | 23.6 | 96% |
| chrome_R (on only) | - | 74.0 | 70.8 | - |
| chrome_packed (on only) | - | 222.9 | 217.5 | - |
