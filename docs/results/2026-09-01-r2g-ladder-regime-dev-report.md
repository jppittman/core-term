# Phase 3 at-budget evaluation on DEV (ablation ladder against the registered claim)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

**Date:** 2026-09-01 · **Registration:** `docs/plans/2026-09-01-phase3-registration.md` (unrevised) · **Authority:** `docs/plans/2026-08-31-guide-design-revision.md` §5

Per-expression rows: `docs/results/2026-09-01-r2g-ladder-regime-dev.jsonl` (208 expressions: classical 208). Skipped: none.

Cost = `CostModel::latency_prior()` `extract_dag` cost, deterministic; no wall-clock in any number below. All arms go through `egraph::anytime::run_anytime_curve_with`; guided arms via `GuidedSaturation` (dedup set carried across checkpoints). Distributions are q1 / median / q3 (p90). Regret reference = empirical best of ALL arms at ANY checkpoint for that expression. Strict-oracle arm: not run (see the harness module doc — no cheap provenance replay exists for it).

## classical band, B = 100 (REGISTERED: Y = 16.3% → median ratio ≤ 0.837; 4B-approach: median gap ≤ 24.2%)

n = 208.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.16% / 42.82% / 17276.91% (p90 47441.3%) | 58.32% / 106.30% / 36320.50% (p90 69646.9%) | 0 | 0.616 | 0.0202 | — |
| (b) unguided @4B | — | — | 0 | 0.29% / 36.90% / 85.80% (p90 29341.6%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.003 / 0.552 / 0.691 (p90 0.794) | 202 / 0 / 6 | -33.48% / -12.45% / 0.29% (p90 1.2%) | 0.57% / 2.17% / 9.86% (p90 29.6%) | 1.000 / 1.001 / 75.528 (p90 530.575) (n=195) | 0.222 | 0.2969 | 3 |
| (d) LinearReturnGuide[return-mse] @B [claim] | 0.003 / 0.521 / 0.670 (p90 0.751) | 200 / 0 / 8 | -36.79% / -18.89% / 0.31% (p90 1.5%) | 0.45% / 1.43% / 5.85% (p90 19.5%) | 1.000 / 1.002 / 104.428 (p90 1873.544) (n=195) | 0.286 | 0.3033 | 3 |

Ladder diagnostics: linear < control on 92 expressions, equal on 24, linear > control on 92. control: quiesced before B on 0 / 208 (ended at 265.250 / 501.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 101; vs production cost better / equal / worse = 20 / 20 / 168. linear: quiesced before B on 0 / 208 (ended at 268.250 / 498.500 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 107; vs production cost better / equal / worse = 29 / 14 / 165. 

**Verdict (B=100):** (d) LinearReturnGuide[return-mse] median ratio vs unguided-at-B = **0.521** against the registered threshold ≤ 0.837 (Y = 16.3%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = -18.89% against ≤ 24.2% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.552, regret 2.17% vs (d) regret 1.43% — (d) **beats** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.06. D_control = -0.0136 (m_control^S = 0.5519, m_control^DEV = 0.5655). D_linear = -0.0159 (m_linear^S = 0.5207, m_linear^DEV = 0.5366). D_control − D_linear = 0.0023 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: holds; both arms D ≤ M_B: holds.

## classical band, B = 200 (REGISTERED: Y = 9.0% → median ratio ≤ 0.910; 4B-approach: median gap ≤ 11.0%)

n = 208.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 10.26% / 57.31% / 169.96% (p90 38114.8%) | 48.73% / 80.39% / 26019.35% (p90 60647.6%) | 0 | 0.630 | 0.0252 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 0.19% / 29.46% (p90 25402.5%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.015 / 0.603 / 0.709 (p90 0.996) | 197 / 4 / 7 | -0.24% / 0.00% / 0.35% (p90 3.6%) | 0.09% / 0.56% / 4.61% (p90 4192.8%) | 0.995 / 1.000 / 1.000 (p90 4.025) (n=189) | 0.367 | 0.1701 | 4 |
| (d) LinearReturnGuide[return-mse] @B [claim] | 0.098 / 0.600 / 0.719 (p90 0.999) | 193 / 4 / 11 | -0.14% / 0.00% / 0.28% (p90 0.9%) | 0.00% / 0.40% / 2.99% (p90 15361.0%) | 0.996 / 1.000 / 1.000 (p90 3.996) (n=189) | 0.418 | 0.1683 | 5 |

Ladder diagnostics: linear < control on 64 expressions, equal on 90, linear > control on 54. control: quiesced before B on 28 / 208 (ended at 265.250 / 501.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 101; vs production cost better / equal / worse = 18 / 47 / 143. linear: quiesced before B on 28 / 208 (ended at 268.250 / 498.500 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 107; vs production cost better / equal / worse = 16 / 58 / 134. 

**Verdict (B=200):** (d) LinearReturnGuide[return-mse] median ratio vs unguided-at-B = **0.600** against the registered threshold ≤ 0.910 (Y = 9.0%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = 0.00% against ≤ 11.0% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.603, regret 0.56% vs (d) regret 0.40% — (d) **beats** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.07. D_control = -0.0964 (m_control^S = 0.6027, m_control^DEV = 0.6991). D_linear = -0.0955 (m_linear^S = 0.6004, m_linear^DEV = 0.6959). D_control − D_linear = -0.0009 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: holds; both arms D ≤ M_B: holds.

## Trig-rule firings (round-1b registration §1.3 / §4)

Cells are `fired (strict-positive, strict rate)` pooled over the set's rows that carry the per-rule-index histogram; `exprs` = expressions on which the arm fired the rule at any point of its run. A rule with 0 firings in every arm on a set has no live match there — a precondition to state before reading D on that set.

### set = classical (n = 208, rows with rule-index histogram = 208)

| idx | rule | arm | @100 | @200 | full run | exprs |
|---:|---|---|---|---|---|---:|
| 20 | doubling | unguided | 0 | 7 (1, 14.3%) | 24132 (55, 0.2%) | 153 |
| 20 | doubling | control | 58 (10, 17.2%) | 255 (36, 14.1%) | 1366 (44, 3.2%) | 149 |
| 20 | doubling | linear | 312 (34, 10.9%) | 391 (34, 8.7%) | 1386 (41, 3.0%) | 150 |
| 30 | odd-negation | unguided | 0 | 13 (2, 15.4%) | 1391 (10, 0.7%) | 86 |
| 30 | odd-negation | control | 47 (5, 10.6%) | 161 (8, 5.0%) | 388 (8, 2.1%) | 84 |
| 30 | odd-negation | linear | 56 (6, 10.7%) | 157 (8, 5.1%) | 372 (8, 2.2%) | 84 |
| 31 | odd-negation | unguided | 0 | 10 (1, 10.0%) | 635 (8, 1.3%) | 54 |
| 31 | odd-negation | control | 23 (6, 26.1%) | 104 (7, 6.7%) | 245 (8, 3.3%) | 54 |
| 31 | odd-negation | linear | 20 (6, 30.0%) | 91 (8, 8.8%) | 229 (8, 3.5%) | 54 |
| 32 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 32 | odd-negation | control | 0 | 0 | 0 | 0 |
| 32 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 33 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 33 | odd-negation | control | 0 | 0 | 0 | 0 |
| 33 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 34 | even-negation | unguided | 0 | 68 (31, 45.6%) | 5035 (341, 6.8%) | 179 |
| 34 | even-negation | control | 602 (316, 52.5%) | 1078 (345, 32.0%) | 1330 (345, 25.9%) | 179 |
| 34 | even-negation | linear | 823 (343, 41.7%) | 1126 (344, 30.6%) | 1389 (344, 24.8%) | 179 |
| 36 | sin-angle-addition | unguided | 0 | 2 (0, 0.0%) | 1696 (2, 0.1%) | 53 |
| 36 | sin-angle-addition | control | 10 (0, 0.0%) | 42 (0, 0.0%) | 210 (1, 0.5%) | 50 |
| 36 | sin-angle-addition | linear | 38 (1, 2.6%) | 69 (1, 1.4%) | 227 (2, 0.9%) | 50 |
| 37 | cos-angle-addition | unguided | 0 | 1 (0, 0.0%) | 1556 (4, 0.3%) | 55 |
| 37 | cos-angle-addition | control | 15 (0, 0.0%) | 73 (2, 2.7%) | 319 (4, 1.3%) | 51 |
| 37 | cos-angle-addition | linear | 38 (0, 0.0%) | 84 (2, 2.4%) | 309 (4, 1.3%) | 50 |
| 38 | reverse-angle-addition | unguided | 0 | 2 (0, 0.0%) | 3571 (18, 0.5%) | 53 |
| 38 | reverse-angle-addition | control | 12 (10, 83.3%) | 40 (17, 42.5%) | 155 (19, 12.3%) | 50 |
| 38 | reverse-angle-addition | linear | 50 (18, 36.0%) | 69 (18, 26.1%) | 174 (19, 10.9%) | 49 |
| 39 | half-angle-product | unguided | 0 | 0 | 2436 (0, 0.0%) | 15 |
| 39 | half-angle-product | control | 0 | 0 | 17 (0, 0.0%) | 4 |
| 39 | half-angle-product | linear | 0 | 0 | 16 (0, 0.0%) | 4 |
| 40 | pythagorean | unguided | 0 | 0 | 137 (0, 0.0%) | 3 |
| 40 | pythagorean | control | 0 | 0 | 0 | 0 |
| 40 | pythagorean | linear | 0 | 0 | 0 | 0 |

## Enabler-starvation diagnostics

### classical (n = 208)

Unguided strict-positive applications: 7681 total, 6546 numeric (non-structural). Of the numeric ones, **5470** (83.6%) have a structural application in their tight derivation ancestry ("structurally enabled"), 530 have a direct child whose chosen node a structural rule created.

| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |
|---|---|---|
| control | 5398 / 6546 (82.5%) | 4325 / 5470 (79.1%) |
| linear | 5407 / 6546 (82.6%) | 4334 / 5470 (79.2%) |

| arm | structural share of applications @100 | @200 | top rules @100 |
|---|---|---|---|
| control | 0.222 | 0.367 | even-negation 7832, commutative 2956, power-sqrt 2174, power-recip 1497, constant-fold 1425, involution 1356, fma-fusion 929, power-rsqrt 581 |
| linear | 0.286 | 0.418 | even-negation 7566, commutative 2602, fma-fusion 2128, power-sqrt 1871, constant-fold 1247, power-recip 1236, associative 688, involution 670 |
| unguided | 0.616 | 0.630 | commutative 17054, involution 5920, constant-fold 3885, canonicalize 756, inverse-annihilation 49 |

- control: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)
- linear: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)

## Production-units context (exact production saturation call per expression)

`production_saturation_probe` = the same function body `optimize_runtime_arena` runs (`config_for_node_count` → `saturate_with_full_budget`), stop reason READ from the loop. Wall-clock is a stop condition of that call only; `timeout` stops are machine-dependent.

### classical (n = 208, probe returned None for 0)

- stop reasons: quiesced 194, timeout 14
- effective B (applications at stop): 1029.750 / 3816.000 / 14656.750 (p90 27399.800) (max 85941)
- share with applications ≥ 100 / 200 / 400 / 800 / 1600: ≥100: 100.0%, ≥200: 100.0%, ≥400: 96.2%, ≥800: 80.8%, ≥1600: 66.3%
- rounds run: 5.000 / 6.000 / 9.000 (p90 10.000); classes at stop: 277.500 / 607.000 / 1909.750 (p90 4534.000)
- production cost / unguided cost@100: 0.004 / 0.521 / 0.670 (p90 0.998)
- production cost / unguided cost@200: 0.317 / 0.595 / 0.718 (p90 1.000)
- production cost / unguided cost@800: 0.982 / 1.000 / 1.000 (p90 1.000)
- production regret vs empirical best: 0.00% / 0.00% / 0.00% (p90 20240.5%)
- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): 0: 4, 25: 16, 200: 6, 400: 43, 800: 51, 1600: 44, 3200: 22, 6400: 7, 12800: 11, 25600: 4
- unguided checkpoint whose app_actual first reaches production's application count: 400: 8, 800: 30, 1600: 27, 3200: 29, 6400: 25, 12800: 24, 25600: 31, 51200: 16, 102400: 5, beyond grid: 13

## Context (not metrics)

- arms: ["unguided", "control", "linear"]
- checkpoint: pixelflow-pipeline/data/guide_checkpoint_r2g_regime_v1.json
- claim_guide: LinearReturnGuide[return-mse]
- corpus: pixelflow-pipeline/data/corpus_dev.bin
- guided_grid: [25, 50, 100, 200, 400, 800] (unguided: [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
- load_at_end: 4:55  up 51 days, 19:28, 2 users, load averages: 8.18 9.41 7.93
- load_at_start: 4:54  up 51 days, 19:27, 2 users, load averages: 10.44 9.93 8.03
- source_rev: cf8814a8b4e83ca117bf69be2be44a3f6ddc449e
- structural_rules: ["commutative", "associative", "reverse-associative", "distribute", "fma-fusion", "identity"]
- train_guide_report: docs/results/2026-09-01-train-guide-report.json
