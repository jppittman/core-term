# The bilinear rules × nodes filter on real shaders

**Date:** 2026-09-08
**Registration:** docs/plans/2026-09-08-rules-filter-bilinear-registration.md — written before any filter was trained; every number here is read against its §7 decision rule and nothing in it was revised.
**Seam:** docs/plans/2026-09-08-rules-by-nodes-filter.md (PR #1228). **Box:** `pixelflow-search/src/nnue/guide/filter.rs`. **Harness:** `pixelflow-pipeline/src/bin/rules_filter.rs` (`mint` / `train` / `eval` / `report`). Tables: `2026-09-08-rules-filter-bilinear.{csv,json}` beside this file.
**Corpus:** DEV = the `egraph_off_on` corpus minus chrome — DejaVu glyph bakes ×2 tiles (190), bench (3) + `O` wide (1), the 12 `shader_bench` ports, psychedelic, cell grid 80×24 @2×: 208 kernels. HELD-OUT = NotoSansMono-Bold bakes (190) and chrome (packed + R), opened once, at the end (§6). NotoSansMono-Regular was not run in this session.
**Instrument:** release, aarch64 (NEON); every arm through `Optimizer::filter` at `Budget::Applications(b)` with production's round and class caps; `Identity` at `b = B` asserted equal to production on `dag_cost` and bytes on every kernel. Deterministic columns are the claim. The box was shared (1-minute load 7–136 during the session; the eval ran at load ≈ 10), so the clock column (§5) is a sign only.

## 0. Verdict

VERDICT_PLACEHOLDER

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

EXTRINSIC_PLACEHOLDER

## 5. Inference cost

COST_PLACEHOLDER

## 6. HELD-OUT, opened once

HELDOUT_PLACEHOLDER

## 7. What this does and does not say

DISCUSSION_PLACEHOLDER
