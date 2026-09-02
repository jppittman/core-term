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

Per-expression rows: `docs/results/2026-09-01-r2g-ladder-strict-sh.jsonl` (95 expressions: classical 95). Skipped: none.

Cost = `CostModel::latency_prior()` `extract_dag` cost, deterministic; no wall-clock in any number below. All arms go through `egraph::anytime::run_anytime_curve_with`; guided arms via `GuidedSaturation` (dedup set carried across checkpoints). Distributions are q1 / median / q3 (p90). Regret reference = empirical best of ALL arms at ANY checkpoint for that expression. Strict-oracle arm: not run (see the harness module doc — no cheap provenance replay exists for it).

## classical band, B = 100 (REGISTERED: Y = 16.3% → median ratio ≤ 0.837; 4B-approach: median gap ≤ 24.2%)

n = 95.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.08% / 0.18% / 5.38% (p90 12.5%) | 15.21% / 20.48% / 32.05% (p90 41.6%) | 0 | 0.906 | 0.0107 | — |
| (b) unguided @4B | — | — | 0 | 11.51% / 17.37% / 28.70% (p90 38.9%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.892 / 0.903 / 0.921 (p90 0.965) | 95 / 0 / 0 | -10.32% / -7.88% / -1.82% (p90 1.0%) | 2.96% / 7.92% / 17.41% (p90 33.2%) | 1.136 / 13.500 / 77.125 (p90 92.600) (n=76) | 0.743 | 0.1541 | 3 |
| (d) LinearCandidateGuide @B [claim] | 0.892 / 0.904 / 0.918 (p90 0.939) | 95 / 0 / 0 | -10.23% / -7.92% / -3.90% (p90 0.9%) | 2.96% / 7.97% / 17.34% (p90 31.4%) | 1.138 / 17.443 / 77.688 (p90 93.100) (n=76) | 0.764 | 0.1466 | 3 |

Ladder diagnostics: linear < control on 30 expressions, equal on 43, linear > control on 22. control: quiesced before B on 0 / 95 (ended at 800.000 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 6; vs production cost better / equal / worse = 3 / 1 / 91. linear: quiesced before B on 0 / 95 (ended at 800.000 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 6; vs production cost better / equal / worse = 3 / 1 / 91. 

**Verdict (B=100):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.904** against the registered threshold ≤ 0.837 (Y = 16.3%) — the Y-clause **FAILS**; median gap vs unguided-at-4B = -7.92% against ≤ 24.2% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.903, regret 7.92% vs (d) regret 7.97% — (d) **does not beat** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.06. D_control = 0.3373 (m_control^S = 0.9028, m_control^DEV = 0.5655). D_linear = 0.3673 (m_linear^S = 0.9039, m_linear^DEV = 0.5366). D_control − D_linear = -0.0300 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: FAILS; both arms D ≤ M_B: FAILS.

## classical band, B = 200 (REGISTERED: Y = 9.0% → median ratio ≤ 0.910; 4B-approach: median gap ≤ 11.0%)

n = 95.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.74% / 2.24% / 7.76% (p90 14.5%) | 15.21% / 20.33% / 31.93% (p90 41.6%) | 0 | 0.913 | 0.0130 | — |
| (b) unguided @4B | — | — | 0 | 10.55% / 15.75% / 24.56% (p90 37.1%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.879 / 0.894 / 0.914 (p90 0.937) | 95 / 0 / 0 | -9.32% / -6.45% / -1.37% (p90 0.6%) | 1.73% / 6.70% / 16.50% (p90 31.3%) | 1.217 / 3.539 / 12.105 (p90 30.507) (n=93) | 0.780 | 0.1020 | 4 |
| (d) LinearCandidateGuide @B [claim] | 0.882 / 0.896 / 0.916 (p90 0.938) | 95 / 0 / 0 | -9.01% / -5.86% / -1.23% (p90 0.6%) | 2.08% / 6.80% / 16.62% (p90 31.3%) | 1.159 / 3.700 / 12.105 (p90 30.507) (n=93) | 0.791 | 0.0991 | 4 |

Ladder diagnostics: linear < control on 3 expressions, equal on 55, linear > control on 37. control: quiesced before B on 1 / 95 (ended at 800.000 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 6; vs production cost better / equal / worse = 3 / 1 / 91. linear: quiesced before B on 1 / 95 (ended at 800.000 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 6; vs production cost better / equal / worse = 3 / 1 / 91. 

**Verdict (B=200):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.896** against the registered threshold ≤ 0.910 (Y = 9.0%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = -5.86% against ≤ 11.0% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.894, regret 6.70% vs (d) regret 6.80% — (d) **does not beat** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.07. D_control = 0.1949 (m_control^S = 0.8940, m_control^DEV = 0.6991). D_linear = 0.2000 (m_linear^S = 0.8959, m_linear^DEV = 0.6959). D_control − D_linear = -0.0051 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: FAILS; both arms D ≤ M_B: FAILS.

## Trig-rule firings (round-1b registration §1.3 / §4)

Cells are `fired (strict-positive, strict rate)` pooled over the set's rows that carry the per-rule-index histogram; `exprs` = expressions on which the arm fired the rule at any point of its run. A rule with 0 firings in every arm on a set has no live match there — a precondition to state before reading D on that set.

### set = classical (n = 95, rows with rule-index histogram = 95)

| idx | rule | arm | @100 | @200 | full run | exprs |
|---:|---|---|---|---|---|---:|
| 20 | doubling | unguided | 0 | 7 (0, 0.0%) | 37314 (0, 0.0%) | 94 |
| 20 | doubling | control | 286 (0, 0.0%) | 588 (0, 0.0%) | 3003 (0, 0.0%) | 93 |
| 20 | doubling | linear | 295 (0, 0.0%) | 575 (0, 0.0%) | 2875 (0, 0.0%) | 93 |
| 30 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 30 | odd-negation | control | 0 | 0 | 0 | 0 |
| 30 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 31 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 31 | odd-negation | control | 0 | 0 | 0 | 0 |
| 31 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 32 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 32 | odd-negation | control | 0 | 0 | 0 | 0 |
| 32 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 33 | odd-negation | unguided | 0 | 0 | 0 | 0 |
| 33 | odd-negation | control | 0 | 0 | 0 | 0 |
| 33 | odd-negation | linear | 0 | 0 | 0 | 0 |
| 34 | even-negation | unguided | 0 | 0 | 0 | 0 |
| 34 | even-negation | control | 0 | 0 | 0 | 0 |
| 34 | even-negation | linear | 0 | 0 | 0 | 0 |
| 36 | sin-angle-addition | unguided | 0 | 0 | 1025 (0, 0.0%) | 94 |
| 36 | sin-angle-addition | control | 257 (0, 0.0%) | 371 (0, 0.0%) | 779 (0, 0.0%) | 93 |
| 36 | sin-angle-addition | linear | 219 (0, 0.0%) | 332 (0, 0.0%) | 733 (0, 0.0%) | 92 |
| 37 | cos-angle-addition | unguided | 0 | 0 | 307 (0, 0.0%) | 51 |
| 37 | cos-angle-addition | control | 22 (0, 0.0%) | 48 (0, 0.0%) | 202 (0, 0.0%) | 48 |
| 37 | cos-angle-addition | linear | 22 (0, 0.0%) | 37 (0, 0.0%) | 176 (0, 0.0%) | 48 |
| 38 | reverse-angle-addition | unguided | 0 | 0 | 1134 (50, 4.4%) | 94 |
| 38 | reverse-angle-addition | control | 134 (0, 0.0%) | 271 (0, 0.0%) | 662 (0, 0.0%) | 93 |
| 38 | reverse-angle-addition | linear | 91 (0, 0.0%) | 226 (0, 0.0%) | 619 (0, 0.0%) | 92 |
| 39 | half-angle-product | unguided | 0 | 0 | 3275 (72, 2.2%) | 94 |
| 39 | half-angle-product | control | 393 (127, 32.3%) | 538 (133, 24.7%) | 1050 (134, 12.8%) | 93 |
| 39 | half-angle-product | linear | 360 (134, 37.2%) | 496 (134, 27.0%) | 985 (134, 13.6%) | 93 |
| 40 | pythagorean | unguided | 0 | 0 | 231 (0, 0.0%) | 19 |
| 40 | pythagorean | control | 0 | 0 | 0 | 0 |
| 40 | pythagorean | linear | 0 | 0 | 0 | 0 |

## Enabler-starvation diagnostics

### classical (n = 95)

Unguided strict-positive applications: 2337 total, 996 numeric (non-structural). Of the numeric ones, **996** (100.0%) have a structural application in their tight derivation ancestry ("structurally enabled"), 59 have a direct child whose chosen node a structural rule created.

| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |
|---|---|---|
| control | 854 / 996 (85.7%) | 854 / 996 (85.7%) |
| linear | 859 / 996 (86.2%) | 859 / 996 (86.2%) |

| arm | structural share of applications @100 | @200 | top rules @100 |
|---|---|---|---|
| control | 0.743 | 0.780 | reverse-associative 2964, associative 2098, fma-fusion 1766, canonicalize 570, constant-fold 551, half-angle-product 393, doubling 286, sin-angle-addition 257 |
| linear | 0.764 | 0.791 | reverse-associative 2981, associative 2404, fma-fusion 1730, constant-fold 547, canonicalize 546, half-angle-product 360, doubling 295, sin-angle-addition 219 |
| unguided | 0.906 | 0.913 | commutative 10792, constant-fold 616, canonicalize 492, associative 15, distribute 12, factor 8, halving 6 |

- control: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)
- linear: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)

## Production-units context (exact production saturation call per expression)

`production_saturation_probe` = the same function body `optimize_runtime_arena` runs (`config_for_node_count` → `saturate_with_full_budget`), stop reason READ from the loop. Wall-clock is a stop condition of that call only; `timeout` stops are machine-dependent.

### classical (n = 95, probe returned None for 0)

- stop reasons: quiesced 95
- effective B (applications at stop): 9811.500 / 10997.000 / 12422.500 (p90 16024.600) (max 19646)
- share with applications ≥ 100 / 200 / 400 / 800 / 1600: ≥100: 100.0%, ≥200: 100.0%, ≥400: 100.0%, ≥800: 100.0%, ≥1600: 100.0%
- rounds run: 8.000 / 8.000 / 9.000 (p90 10.000); classes at stop: 4982.000 / 4990.000 / 4996.000 (p90 4998.600)
- production cost / unguided cost@100: 0.794 / 0.840 / 0.871 (p90 0.896)
- production cost / unguided cost@200: 0.794 / 0.840 / 0.871 (p90 0.897)
- production cost / unguided cost@800: 0.835 / 0.877 / 0.912 (p90 0.976)
- production regret vs empirical best: 0.00% / 0.05% / 5.54% (p90 13.0%)
- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): 0: 23, 400: 1, 1600: 2, 3200: 5, 6400: 16, 12800: 38, 25600: 10
- unguided checkpoint whose app_actual first reaches production's application count: 3200: 1, 12800: 76, 25600: 15, beyond grid: 3

## Context (not metrics)

- arms: ["unguided", "control", "linear"]
- checkpoint: pixelflow-pipeline/data/guide_checkpoint_strict_v1.json
- claim_guide: LinearCandidateGuide
- corpus: pixelflow-pipeline/data/corpus_dev_ood.bin
- guided_grid: [25, 50, 100, 200, 400, 800] (unguided: [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
- load_at_end: 19:31  up 51 days, 10:03, 2 users, load averages: 6.36 6.92 9.08
- load_at_start: 19:30  up 51 days, 10:03, 2 users, load averages: 6.35 6.96 9.16
- source_rev: 9bc88578ad17a85a400245e948d6868fd3e9ce9b
- structural_rules: ["commutative", "associative", "reverse-associative", "distribute", "fma-fusion", "identity"]
- train_guide_report: docs/results/2026-09-01-train-guide-report.json
