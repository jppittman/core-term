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

Per-expression rows: `docs/results/2026-09-01-phase3-round1b-bezier.jsonl` (80 expressions: classical 80). Skipped: none.

Cost = `CostModel::latency_prior()` `extract_dag` cost, deterministic; no wall-clock in any number below. All arms go through `egraph::anytime::run_anytime_curve_with`; guided arms via `GuidedSaturation` (dedup set carried across checkpoints). Distributions are q1 / median / q3 (p90). Regret reference = empirical best of ALL arms at ANY checkpoint for that expression. Strict-oracle arm: not run (see the harness module doc — no cheap provenance replay exists for it).

## classical band, B = 100 (REGISTERED: Y = 16.3% → median ratio ≤ 0.837; 4B-approach: median gap ≤ 24.2%)

n = 80.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 4.30% / 5.60% / 8.81% (p90 9.9%) | 22.36% / 29.55% / 44.81% (p90 72.7%) | 0 | 0.708 | 0.0171 | — |
| (b) unguided @4B | — | — | 0 | 17.11% / 22.68% / 31.82% (p90 58.7%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.910 / 0.910 / 0.949 (p90 0.996) | 80 / 0 / 0 | -3.92% / -2.64% / 0.00% (p90 2.5%) | 13.50% / 17.87% / 35.06% (p90 54.5%) | 1.000 / 1.300 / 1.700 (p90 1.700) (n=80) | 0.750 | 0.0815 | 3 |
| (d) LinearCandidateGuide @B [claim] | 0.910 / 0.910 / 0.949 (p90 0.996) | 80 / 0 / 0 | -3.92% / -3.92% / 0.00% (p90 2.5%) | 13.50% / 17.87% / 35.06% (p90 47.6%) | 1.000 / 1.700 / 1.700 (p90 1.800) (n=80) | 0.759 | 0.0934 | 3 |

Ladder diagnostics: linear < control on 15 expressions, equal on 65, linear > control on 0. control: quiesced before B on 0 / 80 (ended at 744.250 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 0; vs production cost better / equal / worse = 0 / 0 / 80. linear: quiesced before B on 0 / 80 (ended at 730.500 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 0; vs production cost better / equal / worse = 0 / 0 / 80. 

**Verdict (B=100):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.910** against the registered threshold ≤ 0.837 (Y = 16.3%) — the Y-clause **FAILS**; median gap vs unguided-at-4B = -3.92% against ≤ 24.2% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.910, regret 17.87% vs (d) regret 17.87% — (d) **does not beat** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.06. D_control = 0.3443 (m_control^S = 0.9098, m_control^DEV = 0.5655). D_linear = 0.3732 (m_linear^S = 0.9098, m_linear^DEV = 0.5366). D_control − D_linear = -0.0289 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: FAILS; both arms D ≤ M_B: FAILS.

## classical band, B = 200 (REGISTERED: Y = 9.0% → median ratio ≤ 0.910; 4B-approach: median gap ≤ 11.0%)

n = 80.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.00% / 9.85% / 24.42% (p90 24.4%) | 22.26% / 29.55% / 44.81% (p90 58.7%) | 0 | 0.768 | 0.0122 | — |
| (b) unguided @4B | — | — | 0 | 3.19% / 4.12% / 31.82% (p90 58.7%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.910 / 0.910 / 0.980 (p90 1.000) | 60 / 20 / 0 | -0.99% / 0.00% / 13.20% (p90 13.2%) | 13.50% / 17.87% / 30.52% (p90 54.5%) | 0.459 / 0.459 / 1.100 (p90 1.100) (n=45) | 0.826 | 0.0491 | 4 |
| (d) LinearCandidateGuide @B [claim] | 0.857 / 0.885 / 0.926 (p90 1.000) | 60 / 20 / 0 | -0.99% / 0.00% / 6.60% (p90 6.6%) | 8.34% / 11.00% / 30.52% (p90 40.6%) | 0.730 / 0.730 / 1.100 (p90 1.100) (n=45) | 0.822 | 0.0467 | 4 |

Ladder diagnostics: linear < control on 41 expressions, equal on 39, linear > control on 0. control: quiesced before B on 0 / 80 (ended at 744.250 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 0; vs production cost better / equal / worse = 0 / 0 / 80. linear: quiesced before B on 0 / 80 (ended at 730.500 / 800.000 / 800.000 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 0; vs production cost better / equal / worse = 0 / 0 / 80. 

**Verdict (B=200):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.885** against the registered threshold ≤ 0.910 (Y = 9.0%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = 0.00% against ≤ 11.0% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.910, regret 17.87% vs (d) regret 11.00% — (d) **beats** (c).

**Round-1b domain-shift statistic (S = classical):** M_B = 0.07. D_control = 0.2107 (m_control^S = 0.9098, m_control^DEV = 0.6991). D_linear = 0.1896 (m_linear^S = 0.8855, m_linear^DEV = 0.6959). D_control − D_linear = 0.0212 → **H_null**. §1.2 polynomial prediction: both arms D ≤ 0: FAILS; both arms D ≤ M_B: FAILS.

## DEV op-composition stratification (round-1b registration §2)

Population counts over the FULL classical DEV corpus loaded this run (n = 80), by first-matching stratum:

| stratum | n | share |
|---|---:|---:|
| polynomial-only | 80 | 100.0% |

Per-stratum results, over classical rows THIS run actually put through the ablation ladder (may be a small sample — see population counts above for the true stratum sizes):

| stratum | B | n | control ratio med | linear ratio med | D_control | D_linear | D_control − D_linear | M_B | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| polynomial-only | 100 | 80 | 0.910 | 0.910 | 0.3443 | 0.3732 | -0.0289 | 0.06 | H_null |
| polynomial-only | 200 | 80 | 0.910 | 0.885 | 0.2107 | 0.1896 | 0.0212 | 0.07 | H_null |

## Trig-rule firings (round-1b registration §1.3 / §4)

Cells are `fired (strict-positive, strict rate)` pooled over the set's rows that carry the per-rule-index histogram; `exprs` = expressions on which the arm fired the rule at any point of its run. A rule with 0 firings in every arm on a set has no live match there — a precondition to state before reading D on that set.

### set = classical (n = 80, rows with rule-index histogram = 80)

| idx | rule | arm | @100 | @200 | full run | exprs |
|---:|---|---|---|---|---|---:|
| 20 | doubling | unguided | 0 | 0 | 1480 (0, 0.0%) | 20 |
| 20 | doubling | control | 0 | 0 | 360 (0, 0.0%) | 20 |
| 20 | doubling | linear | 0 | 0 | 360 (0, 0.0%) | 20 |
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
| 36 | sin-angle-addition | unguided | 0 | 0 | 0 | 0 |
| 36 | sin-angle-addition | control | 0 | 0 | 0 | 0 |
| 36 | sin-angle-addition | linear | 0 | 0 | 0 | 0 |
| 37 | cos-angle-addition | unguided | 0 | 0 | 0 | 0 |
| 37 | cos-angle-addition | control | 0 | 0 | 0 | 0 |
| 37 | cos-angle-addition | linear | 0 | 0 | 0 | 0 |
| 38 | reverse-angle-addition | unguided | 0 | 0 | 0 | 0 |
| 38 | reverse-angle-addition | control | 0 | 0 | 0 | 0 |
| 38 | reverse-angle-addition | linear | 0 | 0 | 0 | 0 |
| 39 | half-angle-product | unguided | 0 | 0 | 0 | 0 |
| 39 | half-angle-product | control | 0 | 0 | 0 | 0 |
| 39 | half-angle-product | linear | 0 | 0 | 0 | 0 |
| 40 | pythagorean | unguided | 0 | 0 | 0 | 0 |
| 40 | pythagorean | control | 0 | 0 | 0 | 0 |
| 40 | pythagorean | linear | 0 | 0 | 0 | 0 |

### set = polynomial-only (n = 80, rows with rule-index histogram = 80)

| idx | rule | arm | @100 | @200 | full run | exprs |
|---:|---|---|---|---|---|---:|
| 20 | doubling | unguided | 0 | 0 | 1480 (0, 0.0%) | 20 |
| 20 | doubling | control | 0 | 0 | 360 (0, 0.0%) | 20 |
| 20 | doubling | linear | 0 | 0 | 360 (0, 0.0%) | 20 |
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
| 36 | sin-angle-addition | unguided | 0 | 0 | 0 | 0 |
| 36 | sin-angle-addition | control | 0 | 0 | 0 | 0 |
| 36 | sin-angle-addition | linear | 0 | 0 | 0 | 0 |
| 37 | cos-angle-addition | unguided | 0 | 0 | 0 | 0 |
| 37 | cos-angle-addition | control | 0 | 0 | 0 | 0 |
| 37 | cos-angle-addition | linear | 0 | 0 | 0 | 0 |
| 38 | reverse-angle-addition | unguided | 0 | 0 | 0 | 0 |
| 38 | reverse-angle-addition | control | 0 | 0 | 0 | 0 |
| 38 | reverse-angle-addition | linear | 0 | 0 | 0 | 0 |
| 39 | half-angle-product | unguided | 0 | 0 | 0 | 0 |
| 39 | half-angle-product | control | 0 | 0 | 0 | 0 |
| 39 | half-angle-product | linear | 0 | 0 | 0 | 0 |
| 40 | pythagorean | unguided | 0 | 0 | 0 | 0 |
| 40 | pythagorean | control | 0 | 0 | 0 | 0 |
| 40 | pythagorean | linear | 0 | 0 | 0 | 0 |

## Enabler-starvation diagnostics

### classical (n = 80)

Unguided strict-positive applications: 1368 total, 524 numeric (non-structural). Of the numeric ones, **524** (100.0%) have a structural application in their tight derivation ancestry ("structurally enabled"), 264 have a direct child whose chosen node a structural rule created.

| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |
|---|---|---|
| control | 200 / 524 (38.2%) | 200 / 524 (38.2%) |
| linear | 268 / 524 (51.1%) | 268 / 524 (51.1%) |

| arm | structural share of applications @100 | @200 | top rules @100 |
|---|---|---|---|
| control | 0.750 | 0.826 | associative 2254, reverse-associative 1668, fma-fusion 1268, canonicalize 1122, constant-fold 882, commutative 786, distribute 20 |
| linear | 0.759 | 0.822 | associative 2373, reverse-associative 1960, fma-fusion 1348, canonicalize 1082, constant-fold 802, commutative 340, distribute 40, factor 40 |
| unguided | 0.708 | 0.768 | commutative 6627, canonicalize 1765, constant-fold 963 |

- control: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)
- linear: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)

## Production-units context (exact production saturation call per expression)

`production_saturation_probe` = the same function body `optimize_runtime_arena` runs (`config_for_node_count` → `saturate_with_full_budget`), stop reason READ from the loop. Wall-clock is a stop condition of that call only; `timeout` stops are machine-dependent.

### classical (n = 80, probe returned None for 0)

- stop reasons: quiesced 80
- effective B (applications at stop): 11197.000 / 11197.000 / 18007.000 (p90 29740.000) (max 29740)
- share with applications ≥ 100 / 200 / 400 / 800 / 1600: ≥100: 100.0%, ≥200: 100.0%, ≥400: 100.0%, ≥800: 100.0%, ≥1600: 100.0%
- rounds run: 7.000 / 7.000 / 8.500 (p90 10.000); classes at stop: 4239.500 / 4878.000 / 4878.000 (p90 4939.000)
- production cost / unguided cost@100: 0.523 / 0.526 / 0.722 (p90 0.992)
- production cost / unguided cost@200: 0.523 / 0.573 / 0.723 (p90 0.996)
- production cost / unguided cost@800: 0.650 / 0.650 / 0.770 (p90 0.996)
- production regret vs empirical best: 0.00% / 0.00% / 0.00% (p90 0.0%)
- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): 0: 60, 3200: 20
- unguided checkpoint whose app_actual first reaches production's application count: 12800: 45, 25600: 15, 51200: 20

## Context (not metrics)

- arms: ["unguided", "control", "linear"]
- checkpoint: pixelflow-pipeline/data/guide_checkpoint_strict_v1.json
- corpus: pixelflow-pipeline/data/corpus_dev_ood.bin
- guided_grid: [25, 50, 100, 200, 400, 800] (unguided: [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
- load_at_end: 18:11  up 51 days,  8:43, 2 users, load averages: 5.58 6.05 5.94
- load_at_start: 18:10  up 51 days,  8:43, 2 users, load averages: 5.94 6.14 5.97
- source_rev: 14125f105fed58dc61791c0ad8a098b71de8a0ac
- structural_rules: ["commutative", "associative", "reverse-associative", "distribute", "fma-fusion", "identity"]
- train_guide_report: docs/results/2026-09-01-train-guide-report.json
