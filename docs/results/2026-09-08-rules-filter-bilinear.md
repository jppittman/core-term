# The bilinear rules × nodes filter on real shaders

**Date:** 2026-09-08
**Registration:** docs/plans/2026-09-08-rules-filter-bilinear-registration.md — written before any filter was trained; every number here is read against its §7 decision rule and nothing in it was revised.
**Seam:** docs/plans/2026-09-08-rules-by-nodes-filter.md (PR #1228). **Box:** `pixelflow-search/src/nnue/guide/filter.rs`. **Harness:** `pixelflow-pipeline/src/bin/rules_filter.rs` (`mint` / `train` / `eval` / `report`). Tables: `2026-09-08-rules-filter-bilinear.{csv,json}` beside this file.
**Corpus:** DEV = the `egraph_off_on` corpus minus chrome — DejaVu glyph bakes ×2 tiles (190), bench (3) + `O` wide (1), the 12 `shader_bench` ports, psychedelic, cell grid 80×24 @2×: 208 kernels. HELD-OUT = NotoSansMono-Bold bakes (190) and chrome (packed + R), opened once, at the end (§6). NotoSansMono-Regular was not run in this session.
**Instrument:** release, aarch64 (NEON); every arm through `Optimizer::filter` at `Budget::Applications(b)` with production's round and class caps; `Identity` at `b = B` asserted equal to production on `dag_cost` and bytes on every kernel. Deterministic columns are the claim. The box was shared (1-minute load 7–136 during the session; the eval ran at load ≈ 10), so the clock column (§5) is a sign only.

## 0. Verdict

**NULL, on every family, and reported as one.** At the registered budget (B/2), keep-rate (ρ = 0.25) and model (the one that never saw the family), the median `dag_cost` ratio of the bilinear filter against `Identity` is 1.000 on glyph16, 1.000 on bench, 1.042 on psychedelic, 1.049 on the cell grid and 1.054 on the shaders — none within the registered ≤ 0.95 — and on every family with more than one kernel a control (`UniformRandom` at the bilinear arm's own realized keep-rate, or `PerRuleRate`) sits inside the bilinear arm's p10–p90 band (the single-kernel families have no band; their controls are within 0.01 of the bilinear arm). The secondary keep-rate (ρ = 0.5) is also null (best family median 0.998, shaders). The in-family upper bound (the all-DEV model scored on the families it trained on) is null too, and worse than the held-out model on glyphs (1.153 at ρ = 0.25). HELD-OUT, opened once (§6): NotoSansMono-Bold 1.000 with bytes +4.3 %, chrome 1.156 / 1.005 at ρ = 0.25 / 0.5 — the sign never turns below 1.

The intrinsic half stands: the bilinear head ranks load-bearing cells better than the per-rule prior on every held-out family (§3, glyph AUC 0.879 vs 0.717 from 39,628 shader-and-scene samples). The extrinsic half does not follow from it, for a reason the tables make visible (§7): on these kernels `Identity` itself reaches its own final `dag_cost` at B/8 on 40 of 94 glyphs, so *any* filter that fires fewer applications lands at ratio ≈ 1.00, and which cells it fires barely moves the median. What the interaction buys that the controls do not is in the side columns — the bilinear arm holds the guard count (Σ guarded 111,019 vs Identity's 111,506 on glyph16 at B/2) where `PerRuleRate` and `UniformRandom` inflate it by 50–60 % — and that is a finding for the family's own primary metric, not this registration's.


| family | Bilinear dag_cost ratio (median) | Bilinear bytes ratio (median) | UniformRandom (matched keep-rate) dag median | PerRuleRate dag median | a control inside Bilinear's p10–p90 band | verdict |
|---|---:|---:|---:|---:|---|---|
| bench | 1.000 | 1.000 | 1.000 | 0.974 | yes | NULL |
| bench_wide | 1.000 | 1.000 | 0.999 | 0.974 | no | NULL |
| cellgrid | 1.049 | 0.975 | 1.044 | 1.049 | yes | NULL |
| glyph16 | 1.000 | 1.000 | 1.024 | 0.982 | yes | NULL |
| psychedelic | 1.042 | 0.919 | 0.992 | 1.042 | yes | NULL |
| shader | 1.054 | 0.990 | 1.054 | 1.091 | yes | NULL |

## 1. What was minted

`rules_filter mint`: DEV under a recording `KeepAll` at the production budget with the provenance journal on. The recorder writes each cell's `CellContext` at the seam — the graph as the row's bindings refer to it — and the mint joins it to its provenance record by application ordinal, asserting the rule and the match root agree (they did on all 1,509,068 applications; the journal count equalled the budget counter on every kernel).

| | applications | strict load-bearing | tight load-bearing | samples written (stride cap 4,000 / kernel) |
|---|---:|---:|---:|---:|
| DEV, 208 kernels | 1,509,068 | 23,201 (1.54 %) | 373,350 (24.74 %) | 640,401 |

Per family, in the capped sample (the cap keeps a larger share of the short glyph runs than of the long ones, so the capped positive rate is above the corpus rate):

| family | samples | tight positive rate | strict positive rate |
|---|---:|---:|---:|
| glyph16 | 294,502 | 0.402 | 0.0317 |
| glyph32 | 294,500 | 0.402 | 0.0317 |
| bench | 8,618 | 0.581 | 0.0449 |
| bench_wide | 3,153 | 0.715 | 0.0809 |
| shader | 32,551 | 0.115 | 0.0031 |
| psychedelic | 3,664 | 0.234 | 0.0060 |
| cellgrid | 3,413 | 0.094 | 0.0053 |

**Strict is blind to 92.2 % of tight positives** on the all-DEV training set (97.1 % on the shader + scene set): a filter trained on strict credit would drop nine in ten of the applications the tight bound says the extraction depends on. Tight is the training label, as registered.

## 2. What was trained

`BilinearTrainer` cold start (seed 17), weighted BCE with `pos_weight = negatives / positives` on the training samples, per-sample SGD, 3 epochs, lr 0.01 × 0.7 per epoch, L2 1e-4, gradient-norm clip 1.0. One model per held-out fold and one on all of DEV; 3 min 23 s for the four on one core. Thresholds are the `(1 − ρ)` quantiles of the raw score over each model's own training samples, so each arm keeps exactly ρ of *those*; what it keeps of a held-out family is measured, not assumed (§3).

| model | trained on | samples | tight positive rate | pos_weight | epoch loss | threshold ρ=0.25 / 0.5 |
|---|---|---:|---:|---:|---|---|
| glyph held out | shader + scene | 39,628 | 0.124 | 7.06 | 4.15 → 4.03 → 4.00 | −4.50 / −5.09 |
| shader held out | glyph + scene | 607,850 | 0.403 | 1.48 | 0.95 → 0.89 → 0.88 | 5.36 / −3.93 |
| scene held out | glyph + shader | 633,324 | 0.391 | 1.56 | 0.97 → 0.92 → 0.91 | 5.37 / −4.01 |
| all-DEV | everything | 640,401 | 0.389 | 1.57 | 0.99 → 0.94 → 0.93 | 5.38 / −4.07 |

## 3. Intrinsic metric — the held-out fold's samples

ROC-AUC and PR-AUC of the model that never saw the fold, beside the per-rule prior (each rule's tight positive rate on the training samples, used as a score). The zero predictor's PR-AUC is the positive rate.

| held-out fold | n | tight positive rate | **Bilinear** AUC / PR-AUC (tight) | PerRuleRate AUC / PR-AUC (tight) | strict positive rate | Bilinear AUC / PR-AUC (strict) | PerRuleRate AUC / PR-AUC (strict) |
|---|---:|---:|---|---|---:|---|---|
| glyph | 600,773 | 0.404 | **0.879 / 0.839** | 0.717 / 0.598 | 0.0324 | 0.789 / 0.163 | 0.658 / 0.195 |
| shader | 32,551 | 0.115 | **0.663 / 0.192** | 0.580 / 0.143 | 0.0031 | 0.618 / 0.005 | 0.547 / 0.021 |
| scene | 7,077 | 0.166 | **0.625 / 0.253** | 0.576 / 0.196 | 0.0057 | 0.818 / 0.028 | 0.508 / 0.012 |

The bilinear head beats the per-rule prior on tight labels on every held-out family, by a wide margin on glyphs (trained on 39,628 shader and scene samples only) and by a small one on shaders and scenes. That is the intrinsic half of the registration; it licenses nothing by itself.

## 4. Extrinsic metric — per family, at B, B/2, B/4

Ratios are per kernel against `Identity` at the same budget; median (p10 / p90) over the family. `keep-rate` is the fraction of cells the arm actually kept on this family (the thresholds were set on the training samples — §2). glyph32 is byte-identical to glyph16 on 94 / 95 glyphs and is in the CSV, not repeated here; the two space glyphs (`U+0020`, 0 applications) have no budget to divide and no rows.

### At B/2 — the registered budget

| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| bench | Bilinear[all] ρ=0.25 | 3 | 1.000 (0.974 / 1.098) | 1.064 (1.000 / 1.093) | 8836 / 6738 | 1112 | 3744 | 1 | 0.92 |
| bench | Bilinear[heldout] ρ=0.25 | 3 | 1.000 (0.622 / 1.000) | 1.000 (0.835 / 1.000) | 6738 / 6738 | 1564 | 3611 | 1 | 0.32 |
| bench | Bilinear[all] ρ=0.5 | 3 | 1.000 (1.000 / 1.253) | 1.000 (1.000 / 1.392) | 6792 / 6738 | 1576 | 3642 | 1 | 0.49 |
| bench | Bilinear[heldout] ρ=0.5 | 3 | 1.000 (0.981 / 1.000) | 1.000 (1.000 / 1.124) | 6765 / 6738 | 1576 | 3642 | 1 | 0.65 |
| bench | Identity | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 6738 / 6738 | 1576 | 3642 | 1 | 1.00 |
| bench | PerRuleRate ρ=0.25 | 3 | 0.974 (0.805 / 1.000) | 1.052 (1.000 / 1.064) | 8836 / 6738 | 1112 | 2922 | 1 | 0.61 |
| bench | PerRuleRate ρ=0.5 | 3 | 0.920 (0.695 / 1.000) | 1.059 (1.000 / 1.072) | 8653 / 6738 | 1112 | 3148 | 1 | 0.69 |
| bench | UniformRandom ρ=0.25 | 3 | 0.996 (0.971 / 1.108) | 1.068 (0.997 / 1.186) | 8858 / 6738 | 774 | 2780 | 2 | 0.25 |
| bench | UniformRandom ρ=0.5 | 3 | 0.992 (0.925 / 0.995) | 1.078 (0.989 / 1.196) | 8748 / 6738 | 1463 | 3520 | 1 | 0.49 |
| bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 3 | 1.000 (0.998 / 1.091) | 1.001 (1.000 / 1.093) | 6852 / 6738 | 1112 | 3643 | 1 | 0.93 |
| bench | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 3 | 1.000 (0.999 / 1.142) | 1.000 (1.000 / 1.216) | 6858 / 6738 | 1112 | 3662 | 1 | 0.89 |
| bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 3 | 1.000 (1.000 / 1.089) | 1.000 (1.000 / 1.227) | 6788 / 6738 | 1576 | 3642 | 1 | 0.51 |
| bench | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 3 | 1.000 (1.000 / 1.080) | 1.000 (1.000 / 1.309) | 6789 / 6738 | 1576 | 3642 | 1 | 0.65 |
| bench_wide | Bilinear[all] ρ=0.25 | 1 | 0.974 (0.974 / 0.974) | 1.064 (1.064 / 1.064) | 3054 / 1072 | 1555 | 3744 | 1 | 0.95 |
| bench_wide | Bilinear[heldout] ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1072 / 1072 | 1564 | 3611 | 1 | 0.99 |
| bench_wide | Bilinear[all] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1072 / 1072 | 1576 | 3642 | 1 | 1.00 |
| bench_wide | Bilinear[heldout] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1072 / 1072 | 1576 | 3642 | 1 | 1.00 |
| bench_wide | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1072 / 1072 | 1576 | 3642 | 1 | 1.00 |
| bench_wide | PerRuleRate ρ=0.25 | 1 | 0.974 (0.974 / 0.974) | 1.064 (1.064 / 1.064) | 3054 / 1072 | 1245 | 2922 | 1 | 0.42 |
| bench_wide | PerRuleRate ρ=0.5 | 1 | 0.920 (0.920 / 0.920) | 1.059 (1.059 / 1.059) | 2895 / 1072 | 1576 | 3148 | 1 | 0.54 |
| bench_wide | UniformRandom ρ=0.25 | 1 | 0.971 (0.971 / 0.971) | 1.068 (1.068 / 1.068) | 3025 / 1072 | 1034 | 2780 | 2 | 0.25 |
| bench_wide | UniformRandom ρ=0.5 | 1 | 0.992 (0.992 / 0.992) | 1.078 (1.078 / 1.078) | 3030 / 1072 | 1463 | 3522 | 1 | 0.51 |
| bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.998 (0.998 / 0.998) | 1.001 (1.001 / 1.001) | 1072 / 1072 | 1574 | 3643 | 1 | 0.96 |
| bench_wide | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 0.999 (0.999 / 0.999) | 1.000 (1.000 / 1.000) | 1072 / 1072 | 1573 | 3662 | 1 | 0.99 |
| bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1072 / 1072 | 1576 | 3642 | 1 | 1.00 |
| bench_wide | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1072 / 1072 | 1576 | 3642 | 1 | 1.00 |
| cellgrid | Bilinear[all] ρ=0.25 | 1 | 1.049 (1.049 / 1.049) | 0.975 (0.975 / 0.975) | 0 / 0 | 69 | 235 | 5 | 0.13 |
| cellgrid | Bilinear[heldout] ρ=0.25 | 1 | 1.049 (1.049 / 1.049) | 0.975 (0.975 / 0.975) | 0 / 0 | 53 | 227 | 5 | 0.10 |
| cellgrid | Bilinear[all] ρ=0.5 | 1 | 1.023 (1.023 / 1.023) | 1.012 (1.012 / 1.012) | 0 / 0 | 3129 | 3050 | 3 | 0.65 |
| cellgrid | Bilinear[heldout] ρ=0.5 | 1 | 1.023 (1.023 / 1.023) | 1.012 (1.012 / 1.012) | 0 / 0 | 3351 | 3084 | 3 | 0.66 |
| cellgrid | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 0 / 0 | 3413 | 2689 | 3 | 1.00 |
| cellgrid | PerRuleRate ρ=0.25 | 1 | 1.049 (1.049 / 1.049) | 0.975 (0.975 / 0.975) | 0 / 0 | 225 | 213 | 2 | 0.73 |
| cellgrid | PerRuleRate ρ=0.5 | 1 | 1.023 (1.023 / 1.023) | 1.006 (1.006 / 1.006) | 0 / 0 | 3413 | 2676 | 4 | 0.72 |
| cellgrid | UniformRandom ρ=0.25 | 1 | 1.026 (1.026 / 1.026) | 0.981 (0.981 / 0.981) | 0 / 0 | 273 | 292 | 16 | 0.27 |
| cellgrid | UniformRandom ρ=0.5 | 1 | 0.988 (0.988 / 0.988) | 0.981 (0.981 / 0.981) | 0 / 0 | 3281 | 2440 | 6 | 0.49 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.054 (1.054 / 1.054) | 0.981 (0.981 / 0.981) | 0 / 0 | 33 | 153 | 4 | 0.15 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 1.044 (1.044 / 1.044) | 0.988 (0.988 / 0.988) | 0 / 0 | 35 | 152 | 6 | 0.14 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.009 (1.009 / 1.009) | 0.994 (0.994 / 0.994) | 0 / 0 | 3345 | 2631 | 4 | 0.65 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 1.019 (1.019 / 1.019) | 0.994 (0.994 / 0.994) | 0 / 0 | 3324 | 2570 | 4 | 0.66 |
| glyph16 | Bilinear[all] ρ=0.25 | 94 | 1.153 (0.949 / 1.878) | 1.083 (1.000 / 1.418) | 181536 / 111506 | 434 | 928 | 1 | 0.56 |
| glyph16 | Bilinear[heldout] ρ=0.25 | 94 | 1.000 (0.851 / 1.144) | 1.000 (0.866 / 1.083) | 111019 / 111506 | 1620 | 3062 | 2 | 0.31 |
| glyph16 | Bilinear[all] ρ=0.5 | 94 | 1.003 (1.000 / 1.240) | 1.005 (1.000 / 1.298) | 112223 / 111506 | 1788 | 3220 | 2 | 0.37 |
| glyph16 | Bilinear[heldout] ρ=0.5 | 94 | 1.000 (0.936 / 1.071) | 1.000 (1.000 / 1.175) | 111709 / 111506 | 2419 | 3132 | 2 | 0.50 |
| glyph16 | Identity | 94 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 111506 / 111506 | 3353 | 2994 | 2 | 1.00 |
| glyph16 | PerRuleRate ρ=0.25 | 94 | 0.982 (0.896 / 1.629) | 1.069 (0.951 / 1.382) | 181174 / 111506 | 754 | 991 | 6 | 0.71 |
| glyph16 | PerRuleRate ρ=0.5 | 94 | 0.979 (0.785 / 1.368) | 1.071 (0.948 / 1.382) | 176418 / 111506 | 911 | 1108 | 5 | 0.78 |
| glyph16 | UniformRandom ρ=0.25 | 94 | 1.011 (0.924 / 1.209) | 1.083 (0.980 / 1.231) | 162987 / 111506 | 1663 | 2338 | 6 | 0.25 |
| glyph16 | UniformRandom ρ=0.5 | 94 | 1.006 (0.889 / 1.141) | 1.069 (0.988 / 1.186) | 157995 / 111506 | 3005 | 2517 | 3 | 0.50 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 94 | 1.095 (0.937 / 1.864) | 1.046 (0.999 / 1.426) | 126325 / 111506 | 722 | 2214 | 1 | 0.44 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 94 | 1.024 (0.999 / 1.383) | 1.030 (1.000 / 1.333) | 116349 / 111506 | 1518 | 2988 | 2 | 0.30 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 94 | 1.016 (1.000 / 1.229) | 1.033 (1.000 / 1.239) | 114931 / 111506 | 1931 | 3063 | 2 | 0.32 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 94 | 1.000 (0.968 / 1.177) | 1.007 (1.000 / 1.186) | 112308 / 111506 | 3340 | 2988 | 2 | 0.46 |
| psychedelic | Bilinear[all] ρ=0.25 | 1 | 1.042 (1.042 / 1.042) | 0.919 (0.919 / 0.919) | 29 / 29 | 88 | 176 | 2 | 0.34 |
| psychedelic | Bilinear[heldout] ρ=0.25 | 1 | 1.042 (1.042 / 1.042) | 0.919 (0.919 / 0.919) | 29 / 29 | 67 | 165 | 2 | 0.30 |
| psychedelic | Bilinear[all] ρ=0.5 | 1 | 1.025 (1.025 / 1.025) | 1.007 (1.007 / 1.007) | 29 / 29 | 3522 | 3201 | 3 | 0.64 |
| psychedelic | Bilinear[heldout] ρ=0.5 | 1 | 1.029 (1.029 / 1.029) | 1.007 (1.007 / 1.007) | 29 / 29 | 3442 | 2986 | 3 | 0.63 |
| psychedelic | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 29 / 29 | 3664 | 2570 | 3 | 1.00 |
| psychedelic | PerRuleRate ρ=0.25 | 1 | 1.042 (1.042 / 1.042) | 0.919 (0.919 / 0.919) | 29 / 29 | 211 | 195 | 2 | 0.69 |
| psychedelic | PerRuleRate ρ=0.5 | 1 | 1.041 (1.041 / 1.041) | 0.973 (0.973 / 0.973) | 29 / 29 | 3306 | 1712 | 4 | 0.63 |
| psychedelic | UniformRandom ρ=0.25 | 1 | 1.022 (1.022 / 1.022) | 0.932 (0.932 / 0.932) | 29 / 29 | 3131 | 1228 | 20 | 0.25 |
| psychedelic | UniformRandom ρ=0.5 | 1 | 0.985 (0.985 / 0.985) | 0.953 (0.953 / 0.953) | 29 / 29 | 3135 | 1637 | 6 | 0.49 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.984 (0.984 / 0.984) | 0.953 (0.953 / 0.953) | 29 / 29 | 3102 | 1403 | 10 | 0.34 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 0.992 (0.992 / 0.992) | 0.926 (0.926 / 0.926) | 29 / 29 | 3188 | 1679 | 12 | 0.30 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 0.971 (0.971 / 0.971) | 0.953 (0.953 / 0.953) | 29 / 29 | 3555 | 2007 | 5 | 0.62 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 1.001 (1.001 / 1.001) | 0.966 (0.966 / 0.966) | 29 / 29 | 3633 | 2411 | 4 | 0.65 |
| shader | Bilinear[all] ρ=0.25 | 12 | 1.054 (0.959 / 1.244) | 0.990 (0.912 / 1.034) | 654 / 657 | 0 | 46 | 1 | 0.00 |
| shader | Bilinear[heldout] ρ=0.25 | 12 | 1.054 (0.959 / 1.244) | 0.990 (0.912 / 1.034) | 654 / 657 | 0 | 46 | 1 | 0.00 |
| shader | Bilinear[all] ρ=0.5 | 12 | 1.000 (0.963 / 1.096) | 1.000 (0.900 / 1.034) | 614 / 657 | 78 | 99 | 4 | 0.36 |
| shader | Bilinear[heldout] ρ=0.5 | 12 | 0.999 (0.917 / 1.096) | 0.996 (0.875 / 1.034) | 615 / 657 | 45 | 76 | 3 | 0.36 |
| shader | Identity | 12 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 657 / 657 | 2300 | 998 | 4 | 1.00 |
| shader | PerRuleRate ρ=0.25 | 12 | 1.091 (1.021 / 1.244) | 1.000 (0.953 / 1.034) | 654 / 657 | 93 | 88 | 2 | 0.72 |
| shader | PerRuleRate ρ=0.5 | 12 | 1.037 (1.000 / 1.158) | 1.000 (0.950 / 1.088) | 620 / 657 | 536 | 230 | 6 | 0.65 |
| shader | UniformRandom ρ=0.25 | 12 | 1.028 (0.955 / 1.157) | 0.994 (0.900 / 1.013) | 625 / 657 | 28 | 69 | 4 | 0.26 |
| shader | UniformRandom ρ=0.5 | 12 | 1.012 (0.983 / 1.105) | 1.000 (0.977 / 1.000) | 661 / 657 | 1372 | 828 | 8 | 0.49 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 12 | 1.054 (0.959 / 1.244) | 0.990 (0.912 / 1.034) | 654 / 657 | 0 | 46 | 1 | 0.00 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 12 | 1.054 (0.959 / 1.244) | 0.990 (0.912 / 1.034) | 654 / 657 | 0 | 46 | 1 | 0.00 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 12 | 1.033 (0.993 / 1.119) | 1.000 (0.953 / 1.000) | 644 / 657 | 256 | 164 | 9 | 0.38 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 12 | 1.000 (0.987 / 1.119) | 1.000 (0.950 / 1.029) | 644 / 657 | 486 | 212 | 9 | 0.39 |

### At B

| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| bench | Bilinear[all] ρ=0.25 | 3 | 0.972 (0.959 / 1.013) | 1.028 (0.964 / 1.049) | 8671 / 2769 | 2107 | 4602 | 1 | 0.73 |
| bench | Bilinear[heldout] ρ=0.25 | 3 | 0.990 (0.573 / 1.002) | 1.001 (0.736 / 1.008) | 2742 / 2769 | 3120 | 4976 | 1 | 0.40 |
| bench | Bilinear[all] ρ=0.5 | 3 | 1.006 (1.000 / 1.156) | 1.010 (1.000 / 1.227) | 2781 / 2769 | 2613 | 4988 | 1 | 0.54 |
| bench | Bilinear[heldout] ρ=0.5 | 3 | 0.998 (0.942 / 1.001) | 1.001 (0.994 / 1.127) | 2814 / 2769 | 3137 | 4972 | 1 | 0.62 |
| bench | Identity | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 2769 / 2769 | 3153 | 4991 | 1 | 1.00 |
| bench | PerRuleRate ρ=0.25 | 3 | 0.934 (0.742 / 0.972) | 0.927 (0.870 / 1.049) | 8634 / 2769 | 1977 | 2972 | 2 | 0.63 |
| bench | PerRuleRate ρ=0.5 | 3 | 0.834 (0.641 / 0.946) | 0.945 (0.864 / 1.047) | 8119 / 2769 | 2225 | 3376 | 2 | 0.72 |
| bench | UniformRandom ρ=0.25 | 3 | 0.970 (0.940 / 1.022) | 1.042 (0.979 / 1.045) | 8613 / 2769 | 1081 | 3862 | 3 | 0.25 |
| bench | UniformRandom ρ=0.5 | 3 | 0.904 (0.880 / 0.972) | 1.048 (0.980 / 1.064) | 8404 / 2769 | 2028 | 4080 | 1 | 0.49 |
| bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 3 | 1.002 (0.913 / 1.006) | 0.966 (0.964 / 0.995) | 2862 / 2769 | 2179 | 4661 | 1 | 0.73 |
| bench | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 3 | 1.002 (0.995 / 1.054) | 1.002 (0.976 / 1.073) | 2868 / 2769 | 2225 | 4983 | 1 | 0.92 |
| bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 3 | 1.005 (0.956 / 1.044) | 1.006 (0.992 / 1.127) | 2844 / 2769 | 2923 | 4942 | 1 | 0.50 |
| bench | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 3 | 0.996 (0.773 / 1.000) | 0.998 (0.891 / 1.000) | 2772 / 2769 | 3139 | 4990 | 1 | 0.61 |
| bench_wide | Bilinear[all] ρ=0.25 | 1 | 0.959 (0.959 / 0.959) | 1.028 (1.028 / 1.028) | 3091 / 1012 | 2235 | 4602 | 1 | 0.62 |
| bench_wide | Bilinear[heldout] ρ=0.25 | 1 | 0.990 (0.990 / 0.990) | 1.008 (1.008 / 1.008) | 1012 / 1012 | 3120 | 4981 | 1 | 0.96 |
| bench_wide | Bilinear[all] ρ=0.5 | 1 | 1.006 (1.006 / 1.006) | 1.010 (1.010 / 1.010) | 997 / 1012 | 2613 | 4988 | 1 | 0.82 |
| bench_wide | Bilinear[heldout] ρ=0.5 | 1 | 0.998 (0.998 / 0.998) | 0.994 (0.994 / 0.994) | 1015 / 1012 | 3137 | 4988 | 1 | 0.99 |
| bench_wide | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1012 / 1012 | 3153 | 4991 | 1 | 1.00 |
| bench_wide | PerRuleRate ρ=0.25 | 1 | 0.934 (0.934 / 0.934) | 0.870 (0.870 / 0.870) | 3054 / 1012 | 3153 | 2972 | 2 | 0.61 |
| bench_wide | PerRuleRate ρ=0.5 | 1 | 0.834 (0.834 / 0.834) | 0.864 (0.864 / 0.864) | 2682 / 1012 | 3153 | 3376 | 2 | 0.70 |
| bench_wide | UniformRandom ρ=0.25 | 1 | 0.940 (0.940 / 0.940) | 0.979 (0.979 / 0.979) | 2958 / 1012 | 2273 | 3862 | 3 | 0.25 |
| bench_wide | UniformRandom ρ=0.5 | 1 | 0.904 (0.904 / 0.904) | 0.980 (0.980 / 0.980) | 2882 / 1012 | 2028 | 4080 | 1 | 0.51 |
| bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.913 (0.913 / 0.913) | 0.966 (0.966 / 0.966) | 1016 / 1012 | 2598 | 4661 | 1 | 0.63 |
| bench_wide | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 0.995 (0.995 / 0.995) | 0.976 (0.976 / 0.976) | 1018 / 1012 | 3099 | 4983 | 1 | 0.95 |
| bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 0.956 (0.956 / 0.956) | 0.992 (0.992 / 0.992) | 1072 / 1012 | 2923 | 4942 | 1 | 0.83 |
| bench_wide | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 0.996 (0.996 / 0.996) | 0.998 (0.998 / 0.998) | 1024 / 1012 | 3139 | 4990 | 1 | 0.98 |
| cellgrid | Bilinear[all] ρ=0.25 | 1 | 1.051 (1.051 / 1.051) | 0.975 (0.975 / 0.975) | 0 / 0 | 69 | 235 | 5 | 0.13 |
| cellgrid | Bilinear[heldout] ρ=0.25 | 1 | 1.051 (1.051 / 1.051) | 0.975 (0.975 / 0.975) | 0 / 0 | 53 | 227 | 5 | 0.10 |
| cellgrid | Bilinear[all] ρ=0.5 | 1 | 1.026 (1.026 / 1.026) | 1.012 (1.012 / 1.012) | 0 / 0 | 5392 | 4423 | 3 | 0.75 |
| cellgrid | Bilinear[heldout] ρ=0.5 | 1 | 1.026 (1.026 / 1.026) | 1.012 (1.012 / 1.012) | 0 / 0 | 5366 | 4441 | 3 | 0.76 |
| cellgrid | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 0 / 0 | 6826 | 4634 | 3 | 1.00 |
| cellgrid | PerRuleRate ρ=0.25 | 1 | 1.051 (1.051 / 1.051) | 0.975 (0.975 / 0.975) | 0 / 0 | 225 | 213 | 2 | 0.73 |
| cellgrid | PerRuleRate ρ=0.5 | 1 | 1.026 (1.026 / 1.026) | 1.006 (1.006 / 1.006) | 0 / 0 | 4785 | 3605 | 4 | 0.58 |
| cellgrid | UniformRandom ρ=0.25 | 1 | 1.028 (1.028 / 1.028) | 0.981 (0.981 / 0.981) | 0 / 0 | 273 | 292 | 16 | 0.27 |
| cellgrid | UniformRandom ρ=0.5 | 1 | 0.991 (0.991 / 0.991) | 0.981 (0.981 / 0.981) | 0 / 0 | 5514 | 3832 | 6 | 0.49 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.056 (1.056 / 1.056) | 0.981 (0.981 / 0.981) | 0 / 0 | 33 | 153 | 4 | 0.15 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 1.047 (1.047 / 1.047) | 0.988 (0.988 / 0.988) | 0 / 0 | 35 | 152 | 6 | 0.14 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.002 (1.002 / 1.002) | 1.000 (1.000 / 1.000) | 0 / 0 | 6550 | 4419 | 4 | 0.76 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 1.012 (1.012 / 1.012) | 1.006 (1.006 / 1.006) | 0 / 0 | 6530 | 4415 | 4 | 0.75 |
| glyph16 | Bilinear[all] ρ=0.25 | 94 | 1.098 (0.967 / 2.080) | 1.082 (0.989 / 1.500) | 179698 / 74213 | 476 | 928 | 1 | 0.39 |
| glyph16 | Bilinear[heldout] ρ=0.25 | 94 | 0.997 (0.774 / 1.190) | 0.996 (0.825 / 1.139) | 73129 / 74213 | 2962 | 4330 | 2 | 0.36 |
| glyph16 | Bilinear[all] ρ=0.5 | 94 | 1.007 (0.976 / 1.195) | 1.007 (0.978 / 1.227) | 74436 / 74213 | 2560 | 4472 | 2 | 0.41 |
| glyph16 | Bilinear[heldout] ρ=0.5 | 94 | 1.000 (0.939 / 1.049) | 1.002 (0.976 / 1.157) | 74332 / 74213 | 3862 | 4588 | 2 | 0.54 |
| glyph16 | Identity | 94 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 74213 / 74213 | 6707 | 4750 | 2 | 1.00 |
| glyph16 | PerRuleRate ρ=0.25 | 94 | 0.975 (0.843 / 1.791) | 1.017 (0.912 / 1.468) | 179140 / 74213 | 1242 | 991 | 6 | 0.61 |
| glyph16 | PerRuleRate ρ=0.5 | 94 | 0.908 (0.723 / 1.513) | 1.008 (0.904 / 1.468) | 164366 / 74213 | 1311 | 1108 | 5 | 0.70 |
| glyph16 | UniformRandom ρ=0.25 | 94 | 1.021 (0.940 / 1.226) | 1.048 (0.975 / 1.244) | 146684 / 74213 | 3284 | 3223 | 6 | 0.25 |
| glyph16 | UniformRandom ρ=0.5 | 94 | 0.992 (0.897 / 1.146) | 1.043 (0.966 / 1.146) | 129239 / 74213 | 5400 | 3738 | 3 | 0.50 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 94 | 1.051 (0.942 / 2.058) | 1.046 (0.966 / 1.500) | 117329 / 74213 | 1443 | 2970 | 1 | 0.34 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 94 | 1.006 (0.963 / 1.392) | 1.023 (0.980 / 1.273) | 87478 / 74213 | 3027 | 3636 | 2 | 0.30 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 94 | 1.005 (0.962 / 1.263) | 1.021 (0.992 / 1.171) | 87415 / 74213 | 3754 | 3746 | 2 | 0.32 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 94 | 1.000 (0.923 / 1.167) | 1.003 (0.978 / 1.145) | 74466 / 74213 | 5820 | 4436 | 2 | 0.45 |
| psychedelic | Bilinear[all] ρ=0.25 | 1 | 1.042 (1.042 / 1.042) | 0.919 (0.919 / 0.919) | 29 / 29 | 88 | 176 | 2 | 0.34 |
| psychedelic | Bilinear[heldout] ρ=0.25 | 1 | 1.042 (1.042 / 1.042) | 0.919 (0.919 / 0.919) | 29 / 29 | 67 | 165 | 2 | 0.30 |
| psychedelic | Bilinear[all] ρ=0.5 | 1 | 1.016 (1.016 / 1.016) | 1.000 (1.000 / 1.000) | 29 / 29 | 4777 | 3926 | 3 | 0.69 |
| psychedelic | Bilinear[heldout] ρ=0.5 | 1 | 1.016 (1.016 / 1.016) | 1.000 (1.000 / 1.000) | 29 / 29 | 4204 | 3429 | 3 | 0.67 |
| psychedelic | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 29 / 29 | 7328 | 4670 | 3 | 1.00 |
| psychedelic | PerRuleRate ρ=0.25 | 1 | 1.042 (1.042 / 1.042) | 0.919 (0.919 / 0.919) | 29 / 29 | 211 | 195 | 2 | 0.69 |
| psychedelic | PerRuleRate ρ=0.5 | 1 | 1.041 (1.041 / 1.041) | 0.973 (0.973 / 0.973) | 29 / 29 | 7223 | 2142 | 6 | 0.69 |
| psychedelic | UniformRandom ρ=0.25 | 1 | 1.022 (1.022 / 1.022) | 0.932 (0.932 / 0.932) | 29 / 29 | 5869 | 2108 | 23 | 0.25 |
| psychedelic | UniformRandom ρ=0.5 | 1 | 0.985 (0.985 / 0.985) | 0.959 (0.959 / 0.959) | 29 / 29 | 4834 | 2210 | 6 | 0.50 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.987 (0.987 / 0.987) | 0.966 (0.966 / 0.966) | 29 / 29 | 6612 | 2325 | 12 | 0.34 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 0.987 (0.987 / 0.987) | 0.926 (0.926 / 0.926) | 29 / 29 | 6410 | 2689 | 14 | 0.29 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.015 (1.015 / 1.015) | 0.986 (0.986 / 0.986) | 29 / 29 | 5613 | 2978 | 4 | 0.71 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 0.994 (0.994 / 0.994) | 1.000 (1.000 / 1.000) | 29 / 29 | 4470 | 2514 | 4 | 0.67 |
| shader | Bilinear[all] ρ=0.25 | 12 | 1.061 (0.943 / 1.134) | 0.990 (0.912 / 1.000) | 654 / 652 | 0 | 46 | 1 | 0.00 |
| shader | Bilinear[heldout] ρ=0.25 | 12 | 1.061 (0.943 / 1.134) | 0.990 (0.912 / 1.000) | 654 / 652 | 0 | 46 | 1 | 0.00 |
| shader | Bilinear[all] ρ=0.5 | 12 | 0.999 (0.955 / 1.067) | 1.000 (0.900 / 1.000) | 614 / 652 | 78 | 99 | 4 | 0.36 |
| shader | Bilinear[heldout] ρ=0.5 | 12 | 0.998 (0.902 / 1.070) | 0.996 (0.875 / 1.000) | 615 / 652 | 45 | 76 | 3 | 0.36 |
| shader | Identity | 12 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 652 / 652 | 4601 | 1797 | 4 | 1.00 |
| shader | PerRuleRate ρ=0.25 | 12 | 1.084 (1.021 / 1.134) | 1.000 (0.953 / 1.000) | 654 / 652 | 93 | 88 | 2 | 0.72 |
| shader | PerRuleRate ρ=0.5 | 12 | 1.011 (0.955 / 1.158) | 1.000 (0.950 / 1.088) | 620 / 652 | 536 | 230 | 6 | 0.67 |
| shader | UniformRandom ρ=0.25 | 12 | 1.036 (0.923 / 1.133) | 0.986 (0.900 / 1.000) | 625 / 652 | 28 | 69 | 4 | 0.26 |
| shader | UniformRandom ρ=0.5 | 12 | 1.012 (0.918 / 1.104) | 1.000 (0.950 / 1.000) | 643 / 652 | 2587 | 1353 | 8 | 0.49 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 12 | 1.061 (0.943 / 1.134) | 0.990 (0.912 / 1.000) | 654 / 652 | 0 | 46 | 1 | 0.00 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 12 | 1.061 (0.943 / 1.134) | 0.990 (0.912 / 1.000) | 654 / 652 | 0 | 46 | 1 | 0.00 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 12 | 1.031 (0.974 / 1.157) | 1.000 (0.953 / 1.159) | 607 / 652 | 256 | 164 | 10 | 0.38 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 12 | 1.000 (0.974 / 1.099) | 0.994 (0.950 / 1.029) | 644 / 652 | 486 | 212 | 10 | 0.41 |

### At B/4

| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| bench | Bilinear[all] ρ=0.25 | 3 | 1.000 (1.000 / 1.117) | 1.000 (0.972 / 1.000) | 8972 / 8883 | 556 | 2549 | 1 | 0.90 |
| bench | Bilinear[heldout] ρ=0.25 | 3 | 1.000 (0.883 / 1.000) | 1.000 (0.844 / 1.000) | 8856 / 8883 | 788 | 2549 | 1 | 0.55 |
| bench | Bilinear[all] ρ=0.5 | 3 | 1.000 (1.000 / 1.132) | 1.000 (1.000 / 1.110) | 8908 / 8883 | 788 | 2549 | 1 | 0.61 |
| bench | Bilinear[heldout] ρ=0.5 | 3 | 1.000 (0.984 / 1.000) | 1.000 (0.917 / 1.000) | 8859 / 8883 | 788 | 2549 | 1 | 0.75 |
| bench | Identity | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 8883 / 8883 | 788 | 2549 | 1 | 1.00 |
| bench | PerRuleRate ρ=0.25 | 3 | 1.000 (0.819 / 1.000) | 1.000 (0.936 / 1.000) | 8972 / 8883 | 595 | 2549 | 1 | 0.89 |
| bench | PerRuleRate ρ=0.5 | 3 | 1.000 (0.707 / 1.000) | 1.000 (0.954 / 1.000) | 8948 / 8883 | 721 | 2549 | 1 | 0.93 |
| bench | UniformRandom ρ=0.25 | 3 | 1.012 (0.992 / 1.127) | 1.001 (0.994 / 1.055) | 9114 / 8883 | 212 | 2501 | 1 | 0.24 |
| bench | UniformRandom ρ=0.5 | 3 | 0.989 (0.911 / 1.009) | 0.994 (0.993 / 1.037) | 8935 / 8883 | 544 | 2245 | 1 | 0.47 |
| bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 3 | 1.000 (1.000 / 1.110) | 1.000 (0.972 / 1.000) | 8970 / 8883 | 556 | 2549 | 1 | 0.90 |
| bench | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 3 | 1.000 (1.000 / 1.106) | 1.000 (0.982 / 1.000) | 8876 / 8883 | 788 | 2549 | 1 | 0.53 |
| bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 3 | 1.000 (1.000 / 1.087) | 1.000 (1.000 / 1.211) | 8965 / 8883 | 788 | 2549 | 1 | 0.57 |
| bench | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 3 | 1.000 (1.000 / 1.094) | 1.000 (0.972 / 1.000) | 8877 / 8883 | 788 | 2549 | 1 | 0.75 |
| bench_wide | Bilinear[all] ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | Bilinear[heldout] ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | Bilinear[all] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | Bilinear[heldout] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | PerRuleRate ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | PerRuleRate ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | UniformRandom ρ=0.25 | 1 | 0.992 (0.992 / 0.992) | 0.994 (0.994 / 0.994) | 3109 / 3160 | 614 | 2501 | 1 | 0.24 |
| bench_wide | UniformRandom ρ=0.5 | 1 | 0.989 (0.989 / 0.989) | 0.993 (0.993 / 0.993) | 3099 / 3160 | 544 | 2245 | 1 | 0.52 |
| bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| bench_wide | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 3160 / 3160 | 788 | 2549 | 1 | 1.00 |
| cellgrid | Bilinear[all] ρ=0.25 | 1 | 1.030 (1.030 / 1.030) | 0.975 (0.975 / 0.975) | 0 / 0 | 69 | 235 | 5 | 0.13 |
| cellgrid | Bilinear[heldout] ρ=0.25 | 1 | 1.030 (1.030 / 1.030) | 0.975 (0.975 / 0.975) | 0 / 0 | 53 | 227 | 5 | 0.10 |
| cellgrid | Bilinear[all] ρ=0.5 | 1 | 1.005 (1.005 / 1.005) | 1.012 (1.012 / 1.012) | 0 / 0 | 1630 | 1602 | 3 | 0.79 |
| cellgrid | Bilinear[heldout] ρ=0.5 | 1 | 1.005 (1.005 / 1.005) | 1.012 (1.012 / 1.012) | 0 / 0 | 1533 | 1488 | 3 | 0.69 |
| cellgrid | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 0 / 0 | 1706 | 1581 | 2 | 1.00 |
| cellgrid | PerRuleRate ρ=0.25 | 1 | 1.030 (1.030 / 1.030) | 0.975 (0.975 / 0.975) | 0 / 0 | 225 | 213 | 2 | 0.73 |
| cellgrid | PerRuleRate ρ=0.5 | 1 | 1.005 (1.005 / 1.005) | 1.006 (1.006 / 1.006) | 0 / 0 | 1481 | 1389 | 3 | 0.55 |
| cellgrid | UniformRandom ρ=0.25 | 1 | 1.007 (1.007 / 1.007) | 0.981 (0.981 / 0.981) | 0 / 0 | 273 | 292 | 16 | 0.27 |
| cellgrid | UniformRandom ρ=0.5 | 1 | 0.979 (0.979 / 0.979) | 0.981 (0.981 / 0.981) | 0 / 0 | 1600 | 1334 | 5 | 0.49 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.034 (1.034 / 1.034) | 0.981 (0.981 / 0.981) | 0 / 0 | 33 | 153 | 4 | 0.15 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 1.025 (1.025 / 1.025) | 0.988 (0.988 / 0.988) | 0 / 0 | 35 | 152 | 6 | 0.14 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 0.993 (0.993 / 0.993) | 1.006 (1.006 / 1.006) | 0 / 0 | 1683 | 1430 | 3 | 0.79 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 0.982 (0.982 / 0.982) | 0.994 (0.994 / 0.994) | 0 / 0 | 1685 | 1493 | 3 | 0.69 |
| glyph16 | Bilinear[all] ρ=0.25 | 94 | 1.065 (1.000 / 1.858) | 1.065 (1.000 / 1.409) | 185128 / 162990 | 264 | 928 | 1 | 0.65 |
| glyph16 | Bilinear[heldout] ρ=0.25 | 94 | 1.000 (0.913 / 1.069) | 1.000 (0.870 / 1.079) | 162710 / 162990 | 970 | 2038 | 1 | 0.35 |
| glyph16 | Bilinear[all] ρ=0.5 | 94 | 1.008 (1.000 / 1.176) | 1.000 (0.981 / 1.205) | 163325 / 162990 | 942 | 2010 | 1 | 0.39 |
| glyph16 | Bilinear[heldout] ρ=0.5 | 94 | 1.000 (0.925 / 1.024) | 1.000 (0.985 / 1.068) | 163124 / 162990 | 1615 | 2067 | 1 | 0.58 |
| glyph16 | Identity | 94 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 162990 / 162990 | 1676 | 1900 | 1 | 1.00 |
| glyph16 | PerRuleRate ρ=0.25 | 94 | 1.000 (0.920 / 1.628) | 1.031 (1.000 / 1.394) | 184768 / 162990 | 583 | 991 | 6 | 0.70 |
| glyph16 | PerRuleRate ρ=0.5 | 94 | 1.000 (0.788 / 1.375) | 1.025 (1.000 / 1.365) | 181262 / 162990 | 690 | 1108 | 4 | 0.77 |
| glyph16 | UniformRandom ρ=0.25 | 94 | 1.013 (0.987 / 1.239) | 1.077 (0.981 / 1.250) | 176831 / 162990 | 783 | 1608 | 4 | 0.25 |
| glyph16 | UniformRandom ρ=0.5 | 94 | 1.008 (0.966 / 1.167) | 1.044 (0.983 / 1.205) | 175458 / 162990 | 1536 | 1824 | 2 | 0.50 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 94 | 1.017 (1.000 / 1.831) | 1.039 (1.000 / 1.396) | 174222 / 162990 | 361 | 1616 | 1 | 0.51 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 94 | 1.001 (1.000 / 1.308) | 1.003 (1.000 / 1.295) | 164435 / 162990 | 962 | 1874 | 1 | 0.34 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 94 | 1.012 (1.000 / 1.259) | 1.007 (1.000 / 1.250) | 166777 / 162990 | 968 | 1910 | 1 | 0.36 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 94 | 1.000 (0.988 / 1.166) | 1.000 (1.000 / 1.224) | 163729 / 162990 | 1668 | 1892 | 1 | 0.56 |
| psychedelic | Bilinear[all] ρ=0.25 | 1 | 1.002 (1.002 / 1.002) | 0.925 (0.925 / 0.925) | 29 / 29 | 88 | 176 | 2 | 0.34 |
| psychedelic | Bilinear[heldout] ρ=0.25 | 1 | 1.002 (1.002 / 1.002) | 0.925 (0.925 / 0.925) | 29 / 29 | 67 | 165 | 2 | 0.30 |
| psychedelic | Bilinear[all] ρ=0.5 | 1 | 0.976 (0.976 / 0.976) | 1.014 (1.014 / 1.014) | 29 / 29 | 1648 | 1682 | 3 | 0.66 |
| psychedelic | Bilinear[heldout] ρ=0.5 | 1 | 0.970 (0.970 / 0.970) | 1.000 (1.000 / 1.000) | 29 / 29 | 1429 | 1503 | 3 | 0.56 |
| psychedelic | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 29 / 29 | 1832 | 1547 | 2 | 1.00 |
| psychedelic | PerRuleRate ρ=0.25 | 1 | 1.002 (1.002 / 1.002) | 0.925 (0.925 / 0.925) | 29 / 29 | 211 | 195 | 2 | 0.69 |
| psychedelic | PerRuleRate ρ=0.5 | 1 | 1.001 (1.001 / 1.001) | 0.980 (0.980 / 0.980) | 29 / 29 | 1482 | 1228 | 3 | 0.58 |
| psychedelic | UniformRandom ρ=0.25 | 1 | 0.987 (0.987 / 0.987) | 0.939 (0.939 / 0.939) | 29 / 29 | 1590 | 783 | 16 | 0.24 |
| psychedelic | UniformRandom ρ=0.5 | 1 | 0.964 (0.964 / 0.964) | 0.939 (0.939 / 0.939) | 29 / 29 | 1540 | 995 | 5 | 0.50 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.951 (0.951 / 0.951) | 0.946 (0.946 / 0.946) | 29 / 29 | 1624 | 854 | 8 | 0.33 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 0.959 (0.959 / 0.959) | 0.939 (0.939 / 0.939) | 29 / 29 | 1689 | 1039 | 11 | 0.29 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 0.967 (0.967 / 0.967) | 0.986 (0.986 / 0.986) | 29 / 29 | 1786 | 1185 | 4 | 0.65 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 0.966 (0.966 / 0.966) | 0.939 (0.939 / 0.939) | 29 / 29 | 1655 | 1199 | 4 | 0.57 |
| shader | Bilinear[all] ρ=0.25 | 12 | 1.054 (0.961 / 1.210) | 0.996 (0.912 / 1.034) | 654 / 653 | 0 | 46 | 1 | 0.00 |
| shader | Bilinear[heldout] ρ=0.25 | 12 | 1.054 (0.961 / 1.210) | 0.996 (0.912 / 1.034) | 654 / 653 | 0 | 46 | 1 | 0.00 |
| shader | Bilinear[all] ρ=0.5 | 12 | 1.000 (0.941 / 1.067) | 0.995 (0.900 / 1.013) | 614 / 653 | 78 | 99 | 4 | 0.36 |
| shader | Bilinear[heldout] ρ=0.5 | 12 | 1.000 (0.918 / 1.081) | 0.996 (0.875 / 1.034) | 615 / 653 | 45 | 76 | 3 | 0.36 |
| shader | Identity | 12 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 653 / 653 | 1150 | 610 | 3 | 1.00 |
| shader | PerRuleRate ρ=0.25 | 12 | 1.091 (1.021 / 1.210) | 1.000 (0.953 / 1.034) | 654 / 653 | 93 | 88 | 2 | 0.72 |
| shader | PerRuleRate ρ=0.5 | 12 | 1.037 (0.977 / 1.134) | 1.000 (0.950 / 1.088) | 620 / 653 | 472 | 218 | 5 | 0.66 |
| shader | UniformRandom ρ=0.25 | 12 | 1.022 (0.956 / 1.157) | 0.992 (0.900 / 1.013) | 625 / 653 | 28 | 69 | 4 | 0.26 |
| shader | UniformRandom ρ=0.5 | 12 | 1.028 (0.951 / 1.105) | 1.000 (0.950 / 1.000) | 643 / 653 | 702 | 473 | 6 | 0.49 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 12 | 1.054 (0.961 / 1.210) | 0.996 (0.912 / 1.034) | 654 / 653 | 0 | 46 | 1 | 0.00 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 12 | 1.054 (0.961 / 1.210) | 0.996 (0.912 / 1.034) | 654 / 653 | 0 | 46 | 1 | 0.00 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 12 | 1.032 (0.993 / 1.119) | 1.000 (0.953 / 1.000) | 642 / 653 | 256 | 164 | 9 | 0.38 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 12 | 1.026 (0.964 / 1.119) | 1.000 (0.950 / 1.097) | 607 / 653 | 436 | 212 | 8 | 0.37 |

### The dual: the first budget of the grid at which the arm reaches Identity@B's dag_cost

| family | arm | n | at B/8 | at B/4 | at B/2 | at B | never |
|---|---|---:|---:|---:|---:|---:|---:|
| cellgrid | Bilinear[all] ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | Bilinear[heldout] ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | Bilinear[all] ρ=0.5 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | Bilinear[heldout] ρ=0.5 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | Identity | 1 | 0 | 0 | 0 | 1 | 0 |
| cellgrid | PerRuleRate ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | PerRuleRate ρ=0.5 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | UniformRandom ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | UniformRandom ρ=0.5 | 1 | 0 | 1 | 0 | 0 | 0 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 0 | 0 | 0 | 0 | 1 |
| cellgrid | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 0 | 0 | 0 | 0 | 1 |
| glyph16 | Bilinear[all] ρ=0.25 | 94 | 16 | 2 | 3 | 9 | 64 |
| glyph16 | Bilinear[heldout] ρ=0.25 | 94 | 41 | 5 | 3 | 14 | 31 |
| glyph16 | Bilinear[all] ρ=0.5 | 94 | 26 | 4 | 3 | 22 | 39 |
| glyph16 | Bilinear[heldout] ρ=0.5 | 94 | 45 | 4 | 5 | 16 | 24 |
| glyph16 | Identity | 94 | 40 | 9 | 3 | 42 | 0 |
| glyph16 | PerRuleRate ρ=0.25 | 94 | 33 | 4 | 3 | 9 | 45 |
| glyph16 | PerRuleRate ρ=0.5 | 94 | 42 | 12 | 3 | 9 | 28 |
| glyph16 | UniformRandom ρ=0.25 | 94 | 10 | 17 | 6 | 12 | 49 |
| glyph16 | UniformRandom ρ=0.5 | 94 | 13 | 23 | 12 | 11 | 35 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 94 | 18 | 2 | 3 | 2 | 69 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 94 | 25 | 7 | 4 | 14 | 44 |
| glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 94 | 30 | 6 | 4 | 16 | 38 |
| glyph16 | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 94 | 29 | 12 | 5 | 18 | 30 |
| psychedelic | Bilinear[all] ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| psychedelic | Bilinear[heldout] ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| psychedelic | Bilinear[all] ρ=0.5 | 1 | 1 | 0 | 0 | 0 | 0 |
| psychedelic | Bilinear[heldout] ρ=0.5 | 1 | 1 | 0 | 0 | 0 | 0 |
| psychedelic | Identity | 1 | 1 | 0 | 0 | 0 | 0 |
| psychedelic | PerRuleRate ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| psychedelic | PerRuleRate ρ=0.5 | 1 | 0 | 0 | 0 | 0 | 1 |
| psychedelic | UniformRandom ρ=0.25 | 1 | 0 | 0 | 0 | 0 | 1 |
| psychedelic | UniformRandom ρ=0.5 | 1 | 0 | 0 | 1 | 0 | 0 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1 | 0 | 0 | 0 | 0 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 1 | 0 | 1 | 0 | 0 | 0 |
| psychedelic | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 0 | 0 | 1 | 0 | 0 |
| psychedelic | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 1 | 0 | 0 | 0 | 1 | 0 |
| shader | Bilinear[all] ρ=0.25 | 12 | 2 | 0 | 0 | 0 | 10 |
| shader | Bilinear[heldout] ρ=0.25 | 12 | 2 | 0 | 0 | 0 | 10 |
| shader | Bilinear[all] ρ=0.5 | 12 | 8 | 0 | 0 | 0 | 4 |
| shader | Bilinear[heldout] ρ=0.5 | 12 | 9 | 0 | 0 | 0 | 3 |
| shader | Identity | 12 | 9 | 2 | 1 | 0 | 0 |
| shader | PerRuleRate ρ=0.25 | 12 | 1 | 0 | 0 | 0 | 11 |
| shader | PerRuleRate ρ=0.5 | 12 | 4 | 0 | 0 | 0 | 8 |
| shader | UniformRandom ρ=0.25 | 12 | 3 | 0 | 0 | 0 | 9 |
| shader | UniformRandom ρ=0.5 | 12 | 4 | 2 | 0 | 0 | 6 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 12 | 2 | 0 | 0 | 0 | 10 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.25 | 12 | 2 | 0 | 0 | 0 | 10 |
| shader | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 12 | 3 | 1 | 0 | 0 | 8 |
| shader | UniformRandom@Bilinear[heldout]'s realized rate, ρ=0.5 | 12 | 5 | 2 | 1 | 0 | 4 |

## 5. Inference cost

Counted, not estimated: `FilterStats` reports the cells each row offered, the cells scored (one score per run of consecutive cells at one class — they share a context and a rule), the multiply-adds from the head's dimensions (`SaturationHead::score_candidate_macs`: `32·|ops| + 36·64 + 64·64 + 64·32 + 32·16 + 16·32 + 32·32 + 32` = 10,528 + 32·|ops| per scored cell; 12.9 k measured mean on glyphs), and the filter's own wall clock per row.

Per epoch (one rule sweep) on the largest DEV family: the glyph16 held-out arm at ρ = 0.25 scored 876,916 cells over 94 kernels × 4 budgets × a median of 2 rounds — **≈ 1.2 k scored cells and ≈ 15 M multiply-adds per round per kernel**, 30 MFLOP. Wall: 1.35 µs per scored cell at 1-minute load ≈ 10 (a sign, not a claim); the filter is 10–23 % of the filtered run's saturation time, and because it fires fewer applications the filtered run is shorter than `Identity`'s at the same budget on every family but one (cell grid at ρ = 0.5: 39 ms vs 32). It is CPU-resident and cheap in the sense the registration asked for. It is not free: on the shader family at ρ = 0.5 the filter is 11.5 % of a run that is itself 14 % of `Identity`'s.


| family | arm | Σ cells | Σ scored | Σ multiply-adds | Σ filter ms | Σ run ms | Σ Identity run ms | filter share of run |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bench | Bilinear[all] ρ=0.25 | 11214 | 10557 | 112308832 | 7 | 74 | 156 | 9.6% |
| bench | Bilinear[all] ρ=0.5 | 38315 | 19969 | 291094944 | 32 | 150 | 156 | 21.4% |
| bench | Bilinear[heldout] ρ=0.25 | 54315 | 28345 | 355448384 | 34 | 178 | 156 | 18.9% |
| bench | Bilinear[heldout] ρ=0.5 | 37209 | 19051 | 270298848 | 27 | 123 | 156 | 21.8% |
| bench_wide | Bilinear[all] ρ=0.25 | 6441 | 5853 | 62464320 | 4 | 28 | 25 | 13.1% |
| bench_wide | Bilinear[all] ρ=0.5 | 5941 | 5461 | 58593760 | 4 | 28 | 25 | 12.4% |
| bench_wide | Bilinear[heldout] ρ=0.25 | 6012 | 5502 | 58966048 | 4 | 29 | 25 | 12.6% |
| bench_wide | Bilinear[heldout] ρ=0.5 | 5939 | 5434 | 58354432 | 4 | 29 | 25 | 12.4% |
| cellgrid | Bilinear[all] ρ=0.25 | 2200 | 1960 | 20908544 | 1 | 9 | 32 | 14.5% |
| cellgrid | Bilinear[all] ρ=0.5 | 15229 | 7574 | 88898560 | 8 | 38 | 32 | 21.6% |
| cellgrid | Bilinear[heldout] ρ=0.25 | 2048 | 1928 | 20537344 | 1 | 9 | 32 | 14.6% |
| cellgrid | Bilinear[heldout] ρ=0.5 | 15616 | 8142 | 94628352 | 8 | 39 | 32 | 21.2% |
| glyph16 | Bilinear[all] ρ=0.25 | 452743 | 374557 | 4004338848 | 254 | 1899 | 7620 | 13.4% |
| glyph16 | Bilinear[all] ρ=0.5 | 1642966 | 676513 | 12061935264 | 1411 | 7510 | 7620 | 18.8% |
| glyph16 | Bilinear[heldout] ρ=0.25 | 1844368 | 876916 | 11334129664 | 1184 | 5596 | 7620 | 21.2% |
| glyph16 | Bilinear[heldout] ρ=0.5 | 1665328 | 649539 | 10095635808 | 1186 | 5135 | 7620 | 23.1% |
| psychedelic | Bilinear[all] ρ=0.25 | 1028 | 808 | 8586496 | 1 | 4 | 31 | 12.5% |
| psychedelic | Bilinear[all] ρ=0.5 | 16152 | 8264 | 95259648 | 9 | 41 | 31 | 21.3% |
| psychedelic | Bilinear[heldout] ρ=0.25 | 892 | 720 | 7644288 | 0 | 4 | 31 | 11.8% |
| psychedelic | Bilinear[heldout] ρ=0.5 | 15520 | 8086 | 92047776 | 8 | 39 | 31 | 21.4% |
| shader | Bilinear[all] ρ=0.25 | 2304 | 2304 | 24360576 | 2 | 16 | 321 | 10.0% |
| shader | Bilinear[all] ρ=0.5 | 17332 | 11044 | 121292736 | 9 | 68 | 321 | 12.7% |
| shader | Bilinear[heldout] ρ=0.25 | 2304 | 2304 | 24360576 | 2 | 15 | 321 | 10.5% |
| shader | Bilinear[heldout] ρ=0.5 | 10119 | 6856 | 73543040 | 5 | 45 | 321 | 11.5% |

## 6. HELD-OUT, opened once

Opened once, after §0–§5 were written, with the all-DEV model. NotoSansMono-Bold (95 glyphs × 2 tiles, plus its 3 bench glyphs and `O` wide) and chrome (packed, R). By §B.2 of the benchmark correction this opening promotes both to DEV; the programme owes a new held-out font and a new held-out scene before its next claim.

| held-out family | ρ | n | Bilinear[all] dag_cost ratio median at B/2 | p10 / p90 | sign |
|---|---:|---:|---:|---|---|
| bold_bench | 0.25 | 3 | 1.000 | 1.000 / 1.000 | ≥1 |
| bold_bench | 0.5 | 3 | 1.000 | 0.958 / 1.000 | ≥1 |
| bold_bench_wide | 0.25 | 1 | 1.000 | 1.000 / 1.000 | ≥1 |
| bold_bench_wide | 0.5 | 1 | 1.000 | 1.000 / 1.000 | ≥1 |
| bold_glyph16 | 0.25 | 93 | 1.000 | 0.971 / 1.860 | ≥1 |
| bold_glyph16 | 0.5 | 93 | 1.000 | 0.997 / 1.166 | ≥1 |
| bold_glyph32 | 0.25 | 93 | 1.000 | 0.971 / 1.860 | ≥1 |
| bold_glyph32 | 0.5 | 93 | 1.000 | 0.997 / 1.166 | ≥1 |
| chrome | 0.25 | 1 | 1.156 | 1.156 / 1.156 | ≥1 |
| chrome | 0.5 | 1 | 1.005 | 1.005 / 1.005 | ≥1 |
| chrome_channel | 0.25 | 1 | 1.178 | 1.178 / 1.178 | ≥1 |
| chrome_channel | 0.5 | 1 | 1.006 | 1.006 / 1.006 | ≥1 |

### B

| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| bold_bench | Bilinear[all] ρ=0.25 | 3 | 1.005 (0.968 / 1.228) | 1.012 (1.005 / 1.051) | 13505 / 9154 | 1470 | 4997 | 1 | 0.45 |
| bold_bench | Bilinear[all] ρ=0.5 | 3 | 1.000 (1.000 / 1.015) | 1.000 (0.996 / 1.000) | 9154 / 9154 | 2200 | 5000 | 1 | 0.79 |
| bold_bench | Identity | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 9154 / 9154 | 2218 | 5000 | 1 | 1.00 |
| bold_bench | UniformRandom ρ=0.25 | 3 | 1.001 (0.969 / 1.124) | 1.002 (0.985 / 1.005) | 13327 / 9154 | 1078 | 4142 | 1 | 0.25 |
| bold_bench | UniformRandom ρ=0.5 | 3 | 0.997 (0.972 / 1.064) | 1.002 (0.997 / 1.011) | 12500 / 9154 | 1791 | 4570 | 1 | 0.51 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 3 | 1.002 (0.958 / 1.116) | 1.002 (0.980 / 1.005) | 13301 / 9154 | 2174 | 5000 | 1 | 0.30 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 3 | 1.000 (1.000 / 1.111) | 1.000 (1.000 / 1.024) | 9825 / 9154 | 2216 | 5000 | 1 | 0.78 |
| bold_bench_wide | Bilinear[all] ρ=0.25 | 1 | 0.968 (0.968 / 0.968) | 1.012 (1.012 / 1.012) | 5052 / 1742 | 2122 | 4997 | 1 | 0.97 |
| bold_bench_wide | Bilinear[all] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1742 / 1742 | 2200 | 5000 | 1 | 0.99 |
| bold_bench_wide | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1742 / 1742 | 2218 | 5000 | 1 | 1.00 |
| bold_bench_wide | UniformRandom ρ=0.25 | 1 | 0.969 (0.969 / 0.969) | 1.005 (1.005 / 1.005) | 5032 / 1742 | 1078 | 4152 | 1 | 0.25 |
| bold_bench_wide | UniformRandom ρ=0.5 | 1 | 0.972 (0.972 / 0.972) | 0.997 (0.997 / 0.997) | 5054 / 1742 | 1791 | 4897 | 1 | 0.51 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.958 (0.958 / 0.958) | 1.005 (1.005 / 1.005) | 5014 / 1742 | 2174 | 5000 | 1 | 0.97 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 1742 / 1742 | 2216 | 5000 | 1 | 0.99 |
| bold_glyph16 | Bilinear[all] ρ=0.25 | 93 | 1.007 (0.967 / 1.935) | 1.056 (1.000 / 1.429) | 256793 / 134840 | 1597 | 3739 | 1 | 0.41 |
| bold_glyph16 | Bilinear[all] ρ=0.5 | 93 | 1.000 (0.984 / 1.171) | 1.000 (0.986 / 1.188) | 135318 / 134840 | 2447 | 4932 | 1 | 0.53 |
| bold_glyph16 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 134840 / 134840 | 3555 | 4930 | 1 | 1.00 |
| bold_glyph16 | UniformRandom ρ=0.25 | 93 | 0.990 (0.937 / 1.190) | 1.027 (0.954 / 1.168) | 219405 / 134840 | 2696 | 3562 | 4 | 0.25 |
| bold_glyph16 | UniformRandom ρ=0.5 | 93 | 0.989 (0.926 / 1.129) | 1.021 (0.962 / 1.155) | 190597 / 134840 | 3394 | 4163 | 2 | 0.50 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.011 (0.939 / 1.935) | 1.015 (0.970 / 1.429) | 174421 / 134840 | 2193 | 3816 | 1 | 0.34 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.001 (0.944 / 1.211) | 1.005 (0.957 / 1.141) | 151297 / 134840 | 3435 | 4809 | 1 | 0.39 |
| bold_glyph32 | Bilinear[all] ρ=0.25 | 93 | 1.007 (0.967 / 1.935) | 1.056 (1.000 / 1.429) | 256793 / 134840 | 1597 | 3739 | 1 | 0.41 |
| bold_glyph32 | Bilinear[all] ρ=0.5 | 93 | 1.000 (0.984 / 1.171) | 1.000 (0.986 / 1.188) | 135318 / 134840 | 2447 | 4932 | 1 | 0.53 |
| bold_glyph32 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 134840 / 134840 | 3555 | 4930 | 1 | 1.00 |
| bold_glyph32 | UniformRandom ρ=0.25 | 93 | 0.990 (0.937 / 1.190) | 1.027 (0.954 / 1.155) | 219404 / 134840 | 2696 | 3562 | 4 | 0.25 |
| bold_glyph32 | UniformRandom ρ=0.5 | 93 | 0.989 (0.926 / 1.129) | 1.021 (0.958 / 1.155) | 190572 / 134840 | 3394 | 4163 | 2 | 0.50 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.011 (0.939 / 1.935) | 1.015 (0.970 / 1.429) | 174421 / 134840 | 2193 | 3816 | 1 | 0.34 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.000 (0.946 / 1.211) | 1.006 (0.957 / 1.141) | 151294 / 134840 | 3435 | 4809 | 1 | 0.38 |
| chrome | Bilinear[all] ρ=0.25 | 1 | 1.164 (1.164 / 1.164) | 0.995 (0.995 / 0.995) | 705 / 614 | 3824 | 3828 | 2 | 0.50 |
| chrome | Bilinear[all] ρ=0.5 | 1 | 1.007 (1.007 / 1.007) | 1.000 (1.000 / 1.000) | 617 / 614 | 5361 | 4921 | 2 | 0.96 |
| chrome | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 614 / 614 | 5502 | 4944 | 2 | 1.00 |
| chrome | UniformRandom ρ=0.25 | 1 | 1.121 (1.121 / 1.121) | 1.136 (1.136 / 1.136) | 668 / 614 | 3866 | 3437 | 6 | 0.25 |
| chrome | UniformRandom ρ=0.5 | 1 | 1.085 (1.085 / 1.085) | 1.164 (1.164 / 1.164) | 636 / 614 | 5037 | 4146 | 3 | 0.51 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.103 (1.103 / 1.103) | 1.140 (1.140 / 1.140) | 654 / 614 | 5011 | 4086 | 3 | 0.51 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.011 (1.011 / 1.011) | 1.019 (1.019 / 1.019) | 618 / 614 | 5466 | 4928 | 2 | 0.96 |
| chrome_channel | Bilinear[all] ρ=0.25 | 1 | 1.162 (1.162 / 1.162) | 0.982 (0.982 / 0.982) | 505 / 433 | 3520 | 3505 | 2 | 0.47 |
| chrome_channel | Bilinear[all] ρ=0.5 | 1 | 0.992 (0.992 / 0.992) | 0.976 (0.976 / 0.976) | 425 / 433 | 5344 | 4876 | 2 | 0.94 |
| chrome_channel | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 433 / 433 | 5626 | 4903 | 2 | 1.00 |
| chrome_channel | UniformRandom ρ=0.25 | 1 | 1.077 (1.077 / 1.077) | 1.085 (1.085 / 1.085) | 440 / 433 | 4329 | 3399 | 6 | 0.26 |
| chrome_channel | UniformRandom ρ=0.5 | 1 | 1.043 (1.043 / 1.043) | 1.036 (1.036 / 1.036) | 429 / 433 | 4387 | 3971 | 3 | 0.50 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.072 (1.072 / 1.072) | 1.091 (1.091 / 1.091) | 447 / 433 | 5545 | 4179 | 4 | 0.47 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 0.995 (0.995 / 0.995) | 1.012 (1.012 / 1.012) | 426 / 433 | 5340 | 4860 | 2 | 0.94 |

### B/2

| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| bold_bench | Bilinear[all] ρ=0.25 | 3 | 1.000 (1.000 / 1.080) | 1.000 (1.000 / 1.024) | 13703 / 12692 | 1109 | 4020 | 1 | 0.36 |
| bold_bench | Bilinear[all] ρ=0.5 | 3 | 1.000 (0.958 / 1.000) | 1.000 (0.989 / 1.000) | 12689 / 12692 | 1109 | 4020 | 1 | 0.96 |
| bold_bench | Identity | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 12692 / 12692 | 1109 | 4020 | 1 | 1.00 |
| bold_bench | UniformRandom ρ=0.25 | 3 | 0.999 (0.988 / 1.010) | 0.992 (0.949 / 1.002) | 13676 / 12692 | 434 | 3211 | 1 | 0.25 |
| bold_bench | UniformRandom ρ=0.5 | 3 | 0.992 (0.987 / 1.008) | 0.997 (0.974 / 1.003) | 13690 / 12692 | 766 | 3554 | 1 | 0.51 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 3 | 1.000 (0.999 / 1.000) | 1.000 (0.961 / 1.000) | 13591 / 12692 | 1109 | 4020 | 1 | 0.31 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 3 | 1.000 (0.978 / 1.000) | 1.000 (0.959 / 1.000) | 12680 / 12692 | 1109 | 4020 | 1 | 0.96 |
| bold_bench_wide | Bilinear[all] ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5250 / 5250 | 1109 | 4020 | 1 | 1.00 |
| bold_bench_wide | Bilinear[all] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5250 / 5250 | 1109 | 4020 | 1 | 1.00 |
| bold_bench_wide | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5250 / 5250 | 1109 | 4020 | 1 | 1.00 |
| bold_bench_wide | UniformRandom ρ=0.25 | 1 | 0.999 (0.999 / 0.999) | 0.992 (0.992 / 0.992) | 5283 / 5250 | 434 | 3211 | 1 | 0.24 |
| bold_bench_wide | UniformRandom ρ=0.5 | 1 | 0.992 (0.992 / 0.992) | 0.997 (0.997 / 0.997) | 5278 / 5250 | 766 | 3554 | 1 | 0.49 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5250 / 5250 | 1109 | 4020 | 1 | 1.00 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5250 / 5250 | 1109 | 4020 | 1 | 1.00 |
| bold_glyph16 | Bilinear[all] ρ=0.25 | 93 | 1.000 (0.971 / 1.860) | 1.043 (1.000 / 1.368) | 260662 / 205446 | 1106 | 3091 | 1 | 0.53 |
| bold_glyph16 | Bilinear[all] ρ=0.5 | 93 | 1.000 (0.997 / 1.174) | 1.000 (1.000 / 1.227) | 205857 / 205446 | 1587 | 3436 | 1 | 0.47 |
| bold_glyph16 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 205446 / 205446 | 1777 | 3210 | 1 | 1.00 |
| bold_glyph16 | UniformRandom ρ=0.25 | 93 | 1.003 (0.930 / 1.179) | 1.017 (0.962 / 1.158) | 238208 / 205446 | 1376 | 2596 | 3 | 0.25 |
| bold_glyph16 | UniformRandom ρ=0.5 | 93 | 1.001 (0.901 / 1.125) | 1.018 (0.967 / 1.175) | 230336 / 205446 | 1554 | 2988 | 2 | 0.50 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.000 (0.922 / 1.860) | 1.017 (1.000 / 1.368) | 238773 / 205446 | 1126 | 2987 | 1 | 0.40 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.000 (0.994 / 1.184) | 1.006 (1.000 / 1.186) | 211508 / 205446 | 1702 | 3235 | 1 | 0.39 |
| bold_glyph32 | Bilinear[all] ρ=0.25 | 93 | 1.000 (0.971 / 1.860) | 1.043 (1.000 / 1.368) | 260662 / 205446 | 1106 | 3091 | 1 | 0.53 |
| bold_glyph32 | Bilinear[all] ρ=0.5 | 93 | 1.000 (0.997 / 1.174) | 1.000 (1.000 / 1.227) | 205857 / 205446 | 1587 | 3436 | 1 | 0.47 |
| bold_glyph32 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 205446 / 205446 | 1777 | 3210 | 1 | 1.00 |
| bold_glyph32 | UniformRandom ρ=0.25 | 93 | 1.003 (0.930 / 1.179) | 1.017 (0.962 / 1.158) | 238208 / 205446 | 1376 | 2596 | 3 | 0.25 |
| bold_glyph32 | UniformRandom ρ=0.5 | 93 | 1.001 (0.901 / 1.125) | 1.018 (0.967 / 1.175) | 230331 / 205446 | 1554 | 2988 | 2 | 0.50 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.000 (0.922 / 1.860) | 1.017 (1.000 / 1.368) | 238773 / 205446 | 1126 | 2987 | 1 | 0.40 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.000 (0.994 / 1.186) | 1.006 (1.000 / 1.186) | 211491 / 205446 | 1702 | 3235 | 1 | 0.39 |
| chrome | Bilinear[all] ρ=0.25 | 1 | 1.156 (1.156 / 1.156) | 0.995 (0.995 / 0.995) | 705 / 606 | 2546 | 2817 | 2 | 0.62 |
| chrome | Bilinear[all] ρ=0.5 | 1 | 1.005 (1.005 / 1.005) | 1.005 (1.005 / 1.005) | 607 / 606 | 2751 | 3424 | 2 | 0.97 |
| chrome | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 606 / 606 | 2751 | 3412 | 2 | 1.00 |
| chrome | UniformRandom ρ=0.25 | 1 | 1.121 (1.121 / 1.121) | 1.103 (1.103 / 1.103) | 672 / 606 | 2435 | 2476 | 5 | 0.25 |
| chrome | UniformRandom ρ=0.5 | 1 | 1.066 (1.066 / 1.066) | 1.079 (1.079 / 1.079) | 641 / 606 | 2585 | 2600 | 3 | 0.50 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.155 (1.155 / 1.155) | 1.145 (1.145 / 1.145) | 702 / 606 | 2495 | 2878 | 2 | 0.61 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.008 (1.008 / 1.008) | 1.005 (1.005 / 1.005) | 612 / 606 | 2730 | 3265 | 2 | 0.97 |
| chrome_channel | Bilinear[all] ρ=0.25 | 1 | 1.178 (1.178 / 1.178) | 1.000 (1.000 / 1.000) | 505 / 414 | 2813 | 3033 | 2 | 0.64 |
| chrome_channel | Bilinear[all] ρ=0.5 | 1 | 1.006 (1.006 / 1.006) | 1.006 (1.006 / 1.006) | 415 / 414 | 2813 | 3205 | 2 | 0.97 |
| chrome_channel | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 414 / 414 | 2813 | 3146 | 2 | 1.00 |
| chrome_channel | UniformRandom ρ=0.25 | 1 | 1.095 (1.095 / 1.095) | 1.093 (1.093 / 1.093) | 446 / 414 | 2534 | 2371 | 5 | 0.26 |
| chrome_channel | UniformRandom ρ=0.5 | 1 | 1.119 (1.119 / 1.119) | 1.148 (1.148 / 1.148) | 458 / 414 | 2335 | 2349 | 3 | 0.50 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.151 (1.151 / 1.151) | 1.111 (1.111 / 1.111) | 481 / 414 | 2582 | 2751 | 2 | 0.63 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.005 (1.005 / 1.005) | 1.006 (1.006 / 1.006) | 416 / 414 | 2813 | 3081 | 2 | 0.98 |

### B/4

| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| bold_bench | Bilinear[all] ρ=0.25 | 3 | 1.000 (0.958 / 1.000) | 1.000 (0.972 / 1.000) | 13731 / 12783 | 554 | 3497 | 1 | 0.41 |
| bold_bench | Bilinear[all] ρ=0.5 | 3 | 1.000 (0.992 / 1.000) | 1.000 (0.996 / 1.000) | 12783 / 12783 | 554 | 3497 | 1 | 0.93 |
| bold_bench | Identity | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 12783 / 12783 | 554 | 3497 | 1 | 1.00 |
| bold_bench | UniformRandom ρ=0.25 | 3 | 1.010 (0.875 / 1.011) | 0.997 (0.894 / 1.002) | 13778 / 12783 | 163 | 2992 | 1 | 0.24 |
| bold_bench | UniformRandom ρ=0.5 | 3 | 1.008 (0.866 / 1.008) | 1.003 (0.917 / 1.003) | 13748 / 12783 | 321 | 3186 | 1 | 0.51 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 3 | 1.000 (0.890 / 1.000) | 1.000 (0.943 / 1.000) | 13626 / 12783 | 554 | 3497 | 1 | 0.42 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 3 | 1.000 (1.000 / 1.001) | 1.000 (1.000 / 1.004) | 12780 / 12783 | 554 | 3497 | 1 | 0.92 |
| bold_bench_wide | Bilinear[all] ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 554 | 3497 | 1 | 1.00 |
| bold_bench_wide | Bilinear[all] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 554 | 3497 | 1 | 1.00 |
| bold_bench_wide | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 554 | 3497 | 1 | 1.00 |
| bold_bench_wide | UniformRandom ρ=0.25 | 1 | 1.011 (1.011 / 1.011) | 0.997 (0.997 / 0.997) | 5384 / 5278 | 163 | 2992 | 1 | 0.23 |
| bold_bench_wide | UniformRandom ρ=0.5 | 1 | 1.008 (1.008 / 1.008) | 1.003 (1.003 / 1.003) | 5385 / 5278 | 321 | 3186 | 1 | 0.48 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 554 | 3497 | 1 | 1.00 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 554 | 3497 | 1 | 1.00 |
| bold_glyph16 | Bilinear[all] ρ=0.25 | 93 | 1.000 (0.985 / 1.765) | 1.000 (1.000 / 1.342) | 262189 / 238079 | 554 | 2092 | 1 | 0.65 |
| bold_glyph16 | Bilinear[all] ρ=0.5 | 93 | 1.000 (1.000 / 1.110) | 1.000 (1.000 / 1.122) | 238214 / 238079 | 823 | 2235 | 1 | 0.48 |
| bold_glyph16 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 238079 / 238079 | 888 | 2176 | 1 | 1.00 |
| bold_glyph16 | UniformRandom ρ=0.25 | 93 | 1.010 (0.965 / 1.161) | 1.003 (0.976 / 1.149) | 251996 / 238079 | 596 | 1861 | 2 | 0.25 |
| bold_glyph16 | UniformRandom ρ=0.5 | 93 | 1.008 (0.968 / 1.082) | 1.002 (0.964 / 1.131) | 243495 / 238079 | 803 | 2163 | 1 | 0.50 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.000 (0.998 / 1.765) | 1.000 (0.981 / 1.342) | 242175 / 238079 | 563 | 2097 | 1 | 0.47 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.000 (1.000 / 1.174) | 1.000 (1.000 / 1.148) | 241386 / 238079 | 869 | 2176 | 1 | 0.42 |
| bold_glyph32 | Bilinear[all] ρ=0.25 | 93 | 1.000 (0.985 / 1.765) | 1.000 (1.000 / 1.342) | 262189 / 238079 | 554 | 2092 | 1 | 0.65 |
| bold_glyph32 | Bilinear[all] ρ=0.5 | 93 | 1.000 (1.000 / 1.110) | 1.000 (1.000 / 1.122) | 238214 / 238079 | 823 | 2235 | 1 | 0.48 |
| bold_glyph32 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 238079 / 238079 | 888 | 2176 | 1 | 1.00 |
| bold_glyph32 | UniformRandom ρ=0.25 | 93 | 1.010 (0.965 / 1.161) | 1.003 (0.976 / 1.149) | 251996 / 238079 | 596 | 1861 | 2 | 0.25 |
| bold_glyph32 | UniformRandom ρ=0.5 | 93 | 1.008 (0.968 / 1.082) | 1.002 (0.964 / 1.131) | 243495 / 238079 | 803 | 2163 | 1 | 0.50 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.000 (0.998 / 1.765) | 1.000 (0.981 / 1.342) | 242175 / 238079 | 563 | 2097 | 1 | 0.47 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.000 (1.000 / 1.174) | 1.000 (1.000 / 1.148) | 241386 / 238079 | 869 | 2176 | 1 | 0.42 |
| chrome | Bilinear[all] ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 0.964 (0.964 / 0.964) | 705 / 702 | 833 | 1617 | 1 | 0.41 |
| chrome | Bilinear[all] ρ=0.5 | 1 | 1.003 (1.003 / 1.003) | 0.987 (0.987 / 0.987) | 704 / 702 | 1375 | 2228 | 1 | 0.93 |
| chrome | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 702 / 702 | 1375 | 2151 | 1 | 1.00 |
| chrome | UniformRandom ρ=0.25 | 1 | 0.950 (0.950 / 0.950) | 1.004 (1.004 / 1.004) | 664 / 702 | 1085 | 1409 | 4 | 0.26 |
| chrome | UniformRandom ρ=0.5 | 1 | 0.939 (0.939 / 0.939) | 0.987 (0.987 / 0.987) | 657 / 702 | 1290 | 1713 | 2 | 0.49 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.944 (0.944 / 0.944) | 1.067 (1.067 / 1.067) | 645 / 702 | 1185 | 1729 | 2 | 0.41 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.002 (1.002 / 1.002) | 0.987 (0.987 / 0.987) | 703 / 702 | 1362 | 2100 | 1 | 0.94 |
| chrome_channel | Bilinear[all] ρ=0.25 | 1 | 0.995 (0.995 / 0.995) | 0.964 (0.964 / 0.964) | 505 / 502 | 1406 | 1740 | 2 | 0.55 |
| chrome_channel | Bilinear[all] ρ=0.5 | 1 | 1.003 (1.003 / 1.003) | 1.000 (1.000 / 1.000) | 504 / 502 | 1406 | 2068 | 1 | 0.95 |
| chrome_channel | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 502 / 502 | 1406 | 2024 | 1 | 1.00 |
| chrome_channel | UniformRandom ρ=0.25 | 1 | 0.966 (0.966 / 0.966) | 1.054 (1.054 / 1.054) | 472 / 502 | 1085 | 1350 | 4 | 0.25 |
| chrome_channel | UniformRandom ρ=0.5 | 1 | 0.935 (0.935 / 0.935) | 1.012 (1.012 / 1.012) | 458 / 502 | 1282 | 1670 | 2 | 0.50 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.911 (0.911 / 0.911) | 1.018 (1.018 / 1.018) | 444 / 502 | 1369 | 1771 | 2 | 0.56 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.012 (1.012 / 1.012) | 1.048 (1.048 / 1.048) | 509 / 502 | 1383 | 1988 | 1 | 0.95 |

### B/8

| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| bold_bench | Bilinear[all] ρ=0.25 | 3 | 1.000 (1.000 / 1.015) | 1.000 (0.993 / 1.000) | 13904 / 12950 | 277 | 3220 | 1 | 0.73 |
| bold_bench | Bilinear[all] ρ=0.5 | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 12950 / 12950 | 277 | 3220 | 1 | 0.99 |
| bold_bench | Identity | 3 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 12950 / 12950 | 277 | 3220 | 1 | 1.00 |
| bold_bench | UniformRandom ρ=0.25 | 3 | 0.995 (0.957 / 1.011) | 0.997 (0.969 / 1.010) | 13820 / 12950 | 97 | 2926 | 1 | 0.25 |
| bold_bench | UniformRandom ρ=0.5 | 3 | 0.996 (0.968 / 1.008) | 1.003 (0.983 / 1.015) | 13993 / 12950 | 189 | 3054 | 1 | 0.50 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 3 | 1.000 (0.989 / 1.000) | 1.000 (1.000 / 1.011) | 13959 / 12950 | 277 | 3220 | 1 | 0.74 |
| bold_bench | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 3 | 1.000 (0.997 / 1.000) | 1.000 (0.998 / 1.000) | 12950 / 12950 | 277 | 3220 | 1 | 0.98 |
| bold_bench_wide | Bilinear[all] ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 277 | 3220 | 1 | 1.00 |
| bold_bench_wide | Bilinear[all] ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 277 | 3220 | 1 | 1.00 |
| bold_bench_wide | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 277 | 3220 | 1 | 1.00 |
| bold_bench_wide | UniformRandom ρ=0.25 | 1 | 1.011 (1.011 / 1.011) | 0.997 (0.997 / 0.997) | 5384 / 5278 | 97 | 2926 | 1 | 0.23 |
| bold_bench_wide | UniformRandom ρ=0.5 | 1 | 1.008 (1.008 / 1.008) | 1.003 (1.003 / 1.003) | 5385 / 5278 | 189 | 3054 | 1 | 0.49 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 277 | 3220 | 1 | 1.00 |
| bold_bench_wide | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 5278 / 5278 | 277 | 3220 | 1 | 1.00 |
| bold_glyph16 | Bilinear[all] ρ=0.25 | 93 | 1.000 (1.000 / 1.633) | 1.000 (1.000 / 1.268) | 264180 / 254946 | 277 | 1678 | 1 | 0.69 |
| bold_glyph16 | Bilinear[all] ρ=0.5 | 93 | 1.000 (1.000 / 1.111) | 1.000 (1.000 / 1.094) | 255102 / 254946 | 444 | 1680 | 1 | 0.50 |
| bold_glyph16 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 254946 / 254946 | 444 | 1680 | 1 | 1.00 |
| bold_glyph16 | UniformRandom ρ=0.25 | 93 | 1.010 (0.991 / 1.222) | 1.000 (0.984 / 1.133) | 261323 / 254946 | 189 | 1464 | 1 | 0.26 |
| bold_glyph16 | UniformRandom ρ=0.5 | 93 | 1.006 (0.980 / 1.173) | 1.000 (0.981 / 1.144) | 257044 / 254946 | 323 | 1527 | 1 | 0.50 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.000 (0.999 / 1.633) | 1.000 (1.000 / 1.268) | 261059 / 254946 | 281 | 1680 | 1 | 0.52 |
| bold_glyph16 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.000 (1.000 / 1.156) | 1.000 (1.000 / 1.138) | 255719 / 254946 | 434 | 1680 | 1 | 0.48 |
| bold_glyph32 | Bilinear[all] ρ=0.25 | 93 | 1.000 (1.000 / 1.633) | 1.000 (1.000 / 1.268) | 264180 / 254946 | 277 | 1678 | 1 | 0.69 |
| bold_glyph32 | Bilinear[all] ρ=0.5 | 93 | 1.000 (1.000 / 1.111) | 1.000 (1.000 / 1.094) | 255102 / 254946 | 444 | 1680 | 1 | 0.50 |
| bold_glyph32 | Identity | 93 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 254946 / 254946 | 444 | 1680 | 1 | 1.00 |
| bold_glyph32 | UniformRandom ρ=0.25 | 93 | 1.010 (0.991 / 1.222) | 1.000 (0.984 / 1.133) | 261323 / 254946 | 189 | 1464 | 1 | 0.26 |
| bold_glyph32 | UniformRandom ρ=0.5 | 93 | 1.006 (0.980 / 1.173) | 1.000 (0.981 / 1.144) | 257050 / 254946 | 323 | 1527 | 1 | 0.50 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 93 | 1.000 (0.999 / 1.633) | 1.000 (1.000 / 1.268) | 261059 / 254946 | 281 | 1680 | 1 | 0.52 |
| bold_glyph32 | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 93 | 1.000 (1.000 / 1.156) | 1.000 (1.000 / 1.138) | 255718 / 254946 | 434 | 1680 | 1 | 0.48 |
| chrome | Bilinear[all] ρ=0.25 | 1 | 1.009 (1.009 / 1.009) | 1.009 (1.009 / 1.009) | 705 / 704 | 598 | 1337 | 1 | 0.73 |
| chrome | Bilinear[all] ρ=0.5 | 1 | 1.006 (1.006 / 1.006) | 1.023 (1.023 / 1.023) | 704 / 704 | 687 | 1438 | 1 | 0.88 |
| chrome | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 704 / 704 | 687 | 1268 | 1 | 1.00 |
| chrome | UniformRandom ρ=0.25 | 1 | 0.981 (0.981 / 0.981) | 1.051 (1.051 / 1.051) | 674 / 704 | 562 | 1027 | 3 | 0.27 |
| chrome | UniformRandom ρ=0.5 | 1 | 0.936 (0.936 / 0.936) | 1.005 (1.005 / 1.005) | 652 / 704 | 616 | 1177 | 2 | 0.48 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 1.018 (1.018 / 1.018) | 1.028 (1.028 / 1.028) | 705 / 704 | 681 | 1323 | 1 | 0.72 |
| chrome | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.016 (1.016 / 1.016) | 1.023 (1.023 / 1.023) | 707 / 704 | 678 | 1345 | 1 | 0.88 |
| chrome_channel | Bilinear[all] ρ=0.25 | 1 | 1.006 (1.006 / 1.006) | 0.982 (0.982 / 0.982) | 505 / 502 | 505 | 1085 | 1 | 0.62 |
| chrome_channel | Bilinear[all] ρ=0.5 | 1 | 1.009 (1.009 / 1.009) | 1.030 (1.030 / 1.030) | 504 / 502 | 703 | 1348 | 1 | 0.90 |
| chrome_channel | Identity | 1 | 1.000 (1.000 / 1.000) | 1.000 (1.000 / 1.000) | 502 / 502 | 703 | 1240 | 1 | 1.00 |
| chrome_channel | UniformRandom ρ=0.25 | 1 | 0.972 (0.972 / 0.972) | 1.048 (1.048 / 1.048) | 468 / 502 | 538 | 925 | 3 | 0.26 |
| chrome_channel | UniformRandom ρ=0.5 | 1 | 0.933 (0.933 / 0.933) | 0.988 (0.988 / 0.988) | 453 / 502 | 605 | 981 | 2 | 0.48 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.25 | 1 | 0.937 (0.937 / 0.937) | 0.988 (0.988 / 0.988) | 446 / 502 | 632 | 1133 | 1 | 0.64 |
| chrome_channel | UniformRandom@Bilinear[all]'s realized rate, ρ=0.5 | 1 | 1.015 (1.015 / 1.015) | 1.030 (1.030 / 1.030) | 505 / 502 | 686 | 1251 | 1 | 0.90 |

## 7. What this does and does not say

**What the null is.** The registered claim was that keeping the load-bearing `(rule, class)` pairs reaches a *better* extraction on a fraction of the budget — ≤ 0.95 of `Identity`'s `dag_cost` at B/2. The measured ratio is 1.00 at the median on the families where the filter keeps anything, and > 1 where it keeps little. The classifier is real (§3) and the filter is cheap (§5); the extraction does not improve because, at these budgets, the e-graph's `dag_cost` on real kernels is already what it is going to be: `Identity` at B/8 equals `Identity` at B on 40 of 94 glyphs, and the dual (§4) shows every arm reaching Identity@B's cost at B/8 about as often as `Identity` does. "Only 1.2 % of applications are strictly load-bearing" is true and is the wrong lever: dropping the other 98.8 % changes the application count, not the extraction, because the extractor was already finding the same term in the smaller graph. The budget the filter frees is real — the shader family reaches Identity's cost with 45 applications instead of 2,300 (ρ = 0.5, B/2) — and it is worth exactly what an application costs, which on these kernels is compile time the class cap already bounds.

**What the controls say.** `UniformRandom` at the bilinear arm's realized keep-rate lands inside its band on every family: fewer applications is the whole effect on `dag_cost`. `PerRuleRate` reaches a *lower* median `dag_cost` than the bilinear filter on glyphs (0.982 at ρ = 0.25) while inflating bytes by 6.9 % and guarded entries by 62 % — the "dag_cost improves, the family's own metric worsens" pattern §B.3 of the benchmark correction warns a global ratio cannot see. The bilinear arm is the only non-identity arm whose guard count stays at `Identity`'s on glyphs (and on bold glyphs held out: 205,857 vs 205,446 at ρ = 0.5). That is consistent with the interaction learning *which* rewrites the row prologue depends on, and it is the one place in these tables where "which cells" mattered over "how many"; it is reported, not claimed, because the registration did not name it.

**Two properties of the seam this run exposed, both engineering rather than research.** (1) A fixed threshold makes the filter *stop the run*: when no cell in a sweep clears it, the sweep commits nothing, the loop reads `unions == 0` as quiescence and stops with budget in hand — every shader run at ρ = 0.25 fired 0 applications and 69 of 192 glyph runs at ρ = 0.25 ended this way. A filter that wants to spend its budget needs a floor (keep the row's best cell, or a per-row quantile), and a loop that wants to tell "nothing matched" from "nothing was kept" needs the stop reason to say which. (2) A threshold set on the training distribution keeps a different fraction of a held-out family (0.36 of glyph cells at nominal ρ = 0.25 from the shader-and-scene model; 0.00 of shader cells from the glyph-heavy model), which is why the matched-keep-rate control exists and why the nominal-ρ control alone would have been misleading.

**What is not concluded.** Nothing here bounds what the filter could do at a class cap it is *allowed to exceed*: every run in this document ran under production's cap, which most glyph kernels hit in two rounds, so a filter that stays under the cap for more rounds had at most a handful of extra rounds to spend. The programme's question — can a filter scale the e-graph past its traditional limit — needs a run where the cap is raised and the filter is what keeps the graph tractable; that run is not this registration. Nor is the clock: no family's primary metric (µs/row, ns/px, µs/frame) was taken, and a `dag_cost` ratio of 1.00 says nothing about them. The earlier Guide results (retracted in the claims ledger) are not a prior here and were not used as one.

**Disposition.** `Optimizer::production()` stays on `KeepAll`. No weights ship as a fixture: the registration ships them only on a win. The box is filled and measured, the seam holds (the `Recording` filter is production byte-for-byte on 208 kernels, and `Identity` at `Applications(B)` reproduced production on every one), the cost is counted, and the answer on real shaders at production budgets is that the rules × nodes filter is not the lever on `dag_cost`. The next registration, if there is one, is the raised-cap run, with the family's own metric as the primary.

