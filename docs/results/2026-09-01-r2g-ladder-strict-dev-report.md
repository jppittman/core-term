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

Per-expression rows: `docs/results/2026-09-01-r2g-ladder-strict-dev.jsonl` (334 expressions: classical 334). Skipped: none.

Cost = `CostModel::latency_prior()` `extract_dag` cost, deterministic; no wall-clock in any number below. All arms go through `egraph::anytime::run_anytime_curve_with`; guided arms via `GuidedSaturation` (dedup set carried across checkpoints). Distributions are q1 / median / q3 (p90). Regret reference = empirical best of ALL arms at ANY checkpoint for that expression. Strict-oracle arm: not run (see the harness module doc — no cheap provenance replay exists for it).

## classical band, B = 100 (REGISTERED: Y = 16.3% → median ratio ≤ 0.837; 4B-approach: median gap ≤ 24.2%)

n = 334.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.39% / 51.66% / 178.61% (p90 48115.2%) | 54.19% / 96.17% / 28395.31% (p90 75649.2%) | 0 | 0.589 | 0.0212 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 1.10% / 57.51% (p90 22210.4%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.007 / 0.565 / 0.701 (p90 0.841) | 321 / 4 / 9 | -28.68% / 0.00% / 0.21% (p90 1.0%) | 0.00% / 0.64% / 5.63% (p90 24.7%) | 1.000 / 1.000 / 4.784 (p90 202.792) (n=308) | 0.279 | 0.2248 | 4 |
| (d) LinearCandidateGuide @B [claim] | 0.004 / 0.537 / 0.681 (p90 0.797) | 323 / 4 / 7 | -31.27% / 0.00% / 0.00% (p90 0.9%) | 0.00% / 0.38% / 1.97% (p90 10.2%) | 1.000 / 1.000 / 20.613 (p90 388.785) (n=308) | 0.322 | 0.2237 | 3 |

Ladder diagnostics: linear < control on 131 expressions, equal on 134, linear > control on 69. control: quiesced before B on 41 / 334 (ended at 132.250 / 294.000 / 750.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 189; vs production cost better / equal / worse = 23 / 102 / 209. linear: quiesced before B on 39 / 334 (ended at 134.250 / 293.000 / 744.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 199; vs production cost better / equal / worse = 36 / 106 / 192. 

**Verdict (B=100):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.537** against the registered threshold ≤ 0.837 (Y = 16.3%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = 0.00% against ≤ 24.2% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.565, regret 0.64% vs (d) regret 0.38% — (d) **beats** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.06. D_control = -0.0000 (m_control^S = 0.5655, m_control^DEV = 0.5655). D_linear = -0.0000 (m_linear^S = 0.5366, m_linear^DEV = 0.5366). D_control − D_linear = -0.0000 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: holds; both arms D ≤ M_B: holds.

## classical band, B = 200 (REGISTERED: Y = 9.0% → median ratio ≤ 0.910; 4B-approach: median gap ≤ 11.0%)

n = 334.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.00% / 16.89% / 78.93% (p90 29102.3%) | 0.70% / 51.45% / 176.86% (p90 50849.3%) | 0 | 0.596 | 0.0296 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 0.00% / 5.66% (p90 17630.8%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.522 / 0.699 / 1.000 (p90 1.000) | 245 / 71 / 18 | 0.00% / 0.00% / 0.23% (p90 2.8%) | 0.00% / 0.20% / 2.85% (p90 16.5%) | 0.994 / 1.000 / 1.000 (p90 2.063) (n=228) | 0.403 | 0.1422 | 5 |
| (d) LinearCandidateGuide @B [claim] | 0.512 / 0.696 / 1.000 (p90 1.000) | 245 / 71 / 18 | -0.27% / 0.00% / 0.09% (p90 0.4%) | 0.00% / 0.08% / 0.44% (p90 5.8%) | 0.999 / 1.000 / 1.001 (p90 1.998) (n=228) | 0.430 | 0.1460 | 5 |

Ladder diagnostics: linear < control on 114 expressions, equal on 202, linear > control on 18. control: quiesced before B on 126 / 334 (ended at 132.250 / 294.000 / 750.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 189; vs production cost better / equal / worse = 20 / 138 / 176. linear: quiesced before B on 128 / 334 (ended at 134.250 / 293.000 / 744.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 199; vs production cost better / equal / worse = 28 / 150 / 156. 

**Verdict (B=200):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.696** against the registered threshold ≤ 0.910 (Y = 9.0%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = 0.00% against ≤ 11.0% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.699, regret 0.20% vs (d) regret 0.08% — (d) **beats** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.07. D_control = 0.0000 (m_control^S = 0.6991, m_control^DEV = 0.6991). D_linear = -0.0000 (m_linear^S = 0.6959, m_linear^DEV = 0.6959). D_control − D_linear = 0.0001 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: FAILS; both arms D ≤ M_B: holds.

## Trig-rule firings (round-1b registration §1.3 / §4)

Cells are `fired (strict-positive, strict rate)` pooled over the set's rows that carry the per-rule-index histogram; `exprs` = expressions on which the arm fired the rule at any point of its run. A rule with 0 firings in every arm on a set has no live match there — a precondition to state before reading D on that set.

### set = classical (n = 334, rows with rule-index histogram = 334)

| idx | rule | arm | @100 | @200 | full run | exprs |
|---:|---|---|---|---|---|---:|
| 20 | doubling | unguided | 8 (0, 0.0%) | 27 (1, 3.7%) | 31203 (66, 0.2%) | 202 |
| 20 | doubling | control | 162 (17, 10.5%) | 414 (44, 10.6%) | 1796 (54, 3.0%) | 197 |
| 20 | doubling | linear | 355 (32, 9.0%) | 564 (43, 7.6%) | 1835 (50, 2.7%) | 196 |
| 30 | odd-negation | unguided | 11 (3, 27.3%) | 42 (5, 11.9%) | 1685 (14, 0.8%) | 108 |
| 30 | odd-negation | control | 128 (9, 7.0%) | 249 (12, 4.8%) | 492 (12, 2.4%) | 106 |
| 30 | odd-negation | linear | 68 (7, 10.3%) | 194 (10, 5.2%) | 462 (12, 2.6%) | 106 |
| 31 | odd-negation | unguided | 5 (1, 20.0%) | 25 (2, 8.0%) | 706 (9, 1.3%) | 63 |
| 31 | odd-negation | control | 50 (8, 16.0%) | 139 (9, 6.5%) | 283 (10, 3.5%) | 63 |
| 31 | odd-negation | linear | 27 (5, 18.5%) | 96 (8, 8.3%) | 253 (10, 4.0%) | 62 |
| 32 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 32 | odd-negation | control | 0 | 0 | 0 | 0 |
| 32 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 33 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 33 | odd-negation | control | 0 | 0 | 0 | 0 |
| 33 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 34 | even-negation | unguided | 99 (30, 30.3%) | 306 (78, 25.5%) | 6039 (396, 6.6%) | 257 |
| 34 | even-negation | control | 892 (371, 41.6%) | 1392 (400, 28.7%) | 1659 (400, 24.1%) | 257 |
| 34 | even-negation | linear | 1135 (395, 34.8%) | 1490 (399, 26.8%) | 1764 (400, 22.7%) | 257 |
| 36 | sin-angle-addition | unguided | 0 | 9 (0, 0.0%) | 1953 (2, 0.1%) | 64 |
| 36 | sin-angle-addition | control | 28 (0, 0.0%) | 72 (0, 0.0%) | 278 (1, 0.4%) | 61 |
| 36 | sin-angle-addition | linear | 31 (0, 0.0%) | 76 (0, 0.0%) | 286 (1, 0.3%) | 61 |
| 37 | cos-angle-addition | unguided | 1 (0, 0.0%) | 14 (0, 0.0%) | 1766 (5, 0.3%) | 69 |
| 37 | cos-angle-addition | control | 51 (1, 2.0%) | 133 (3, 2.3%) | 412 (5, 1.2%) | 65 |
| 37 | cos-angle-addition | linear | 48 (1, 2.1%) | 129 (3, 2.3%) | 401 (4, 1.0%) | 65 |
| 38 | reverse-angle-addition | unguided | 0 | 11 (1, 9.1%) | 5714 (19, 0.3%) | 64 |
| 38 | reverse-angle-addition | control | 26 (11, 42.3%) | 63 (18, 28.6%) | 197 (20, 10.2%) | 61 |
| 38 | reverse-angle-addition | linear | 36 (16, 44.4%) | 76 (19, 25.0%) | 209 (20, 9.6%) | 61 |
| 39 | half-angle-product | unguided | 0 | 0 | 3171 (0, 0.0%) | 17 |
| 39 | half-angle-product | control | 0 | 0 | 23 (0, 0.0%) | 6 |
| 39 | half-angle-product | linear | 0 | 0 | 23 (0, 0.0%) | 7 |
| 40 | pythagorean | unguided | 0 | 0 | 225 (0, 0.0%) | 5 |
| 40 | pythagorean | control | 0 | 0 | 0 | 0 |
| 40 | pythagorean | linear | 0 | 0 | 0 | 0 |

## Enabler-starvation diagnostics

### classical (n = 334)

Unguided strict-positive applications: 8902 total, 7588 numeric (non-structural). Of the numeric ones, **6290** (82.9%) have a structural application in their tight derivation ancestry ("structurally enabled"), 619 have a direct child whose chosen node a structural rule created.

| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |
|---|---|---|
| control | 6365 / 7588 (83.9%) | 5070 / 6290 (80.6%) |
| linear | 6346 / 7588 (83.6%) | 5051 / 6290 (80.3%) |

| arm | structural share of applications @100 | @200 | top rules @100 |
|---|---|---|---|
| control | 0.279 | 0.403 | even-negation 10168, commutative 6042, involution 3659, power-sqrt 2650, constant-fold 2263, power-recip 1950, fma-fusion 1505, associative 760 |
| linear | 0.322 | 0.430 | even-negation 8268, commutative 6149, constant-fold 4334, fma-fusion 2783, power-sqrt 2516, power-recip 1970, involution 1932, associative 684 |
| unguided | 0.589 | 0.596 | commutative 23880, involution 9981, constant-fold 5128, canonicalize 792, even-negation 702, reverse-associative 106, associative 96, inverse-annihilation 72 |

- control: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)
- linear: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)

## Production-units context (exact production saturation call per expression)

`production_saturation_probe` = the same function body `optimize_runtime_arena` runs (`config_for_node_count` → `saturate_with_full_budget`), stop reason READ from the loop. Wall-clock is a stop condition of that call only; `timeout` stops are machine-dependent.

### classical (n = 334, probe returned None for 0)

- stop reasons: quiesced 316, timeout 18
- effective B (applications at stop): 399.750 / 1671.000 / 9441.000 (p90 24535.400) (max 85895)
- share with applications ≥ 100 / 200 / 400 / 800 / 1600: ≥100: 98.5%, ≥200: 91.6%, ≥400: 74.9%, ≥800: 59.6%, ≥1600: 50.3%
- rounds run: 4.000 / 5.000 / 8.000 (p90 10.000); classes at stop: 139.500 / 329.500 / 1053.250 (p90 4154.600)
- production cost / unguided cost@100: 0.236 / 0.534 / 0.686 (p90 0.896)
- production cost / unguided cost@200: 0.520 / 0.706 / 1.000 (p90 1.000)
- production cost / unguided cost@800: 0.996 / 1.000 / 1.000 (p90 1.000)
- production regret vs empirical best: 0.00% / 0.00% / 0.00% (p90 13630.6%)
- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): 0: 4, 25: 21, 50: 1, 100: 1, 200: 75, 400: 72, 800: 57, 1600: 50, 3200: 27, 6400: 8, 12800: 14, 25600: 4
- unguided checkpoint whose app_actual first reaches production's application count: 100: 6, 200: 24, 400: 51, 800: 47, 1600: 28, 3200: 40, 6400: 27, 12800: 26, 25600: 34, 51200: 19, 102400: 6, beyond grid: 26

## Context (not metrics)

- arms: ["unguided", "control", "linear"]
- checkpoint: pixelflow-pipeline/data/guide_checkpoint_strict_v1.json
- claim_guide: LinearCandidateGuide
- corpus: pixelflow-pipeline/data/corpus_dev.bin
- guided_grid: [25, 50, 100, 200, 400, 800] (unguided: [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
- load_at_end: 19:30  up 51 days, 10:03, 2 users, load averages: 6.35 6.96 9.16
- load_at_start: 19:29  up 51 days, 10:02, 2 users, load averages: 5.14 6.94 9.30
- source_rev: 9bc88578ad17a85a400245e948d6868fd3e9ce9b
- structural_rules: ["commutative", "associative", "reverse-associative", "distribute", "fma-fusion", "identity"]
- train_guide_report: docs/results/2026-09-01-train-guide-report.json
