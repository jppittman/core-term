# Oracle-filtered anytime budget curves, recalibrated: curves have shape; rule-granularity filtering barely moves them (2026-09-01)

> **SUPERSEDED (2026-09-02).** Every `cost` and `regret_pct` number below is the
> extraction DP's **tree** cost standing in for the emitted kernel's cost (#1117). The
> corrected run is
> [docs/results/2026-09-02-oracle-filtered-budget-curves.md](2026-09-02-oracle-filtered-budget-curves.md),
> which supersedes this file. The load-bearing conclusion (Finding 2) **survives** the
> correction and is strengthened by it — the 94.9% -> 85.4% gap quoted below is itself a
> tree-cost artifact and is 0.00 points on DAG cost. Nothing below has been edited.

> **Predates the 2026-09-02 review fixes; a re-run is required.** The oracle rule set is
> now derived from the minimum-cost checkpoint's extraction rather than the final one
> (`extract_dag` is non-monotonic along a curve, and 21 of 225 unguided curves in the
> committed CSV end above their own minimum), so the filtered rule sets — and every
> oracle curve below — change. Nothing here has been edited to match the new code.

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Reproduce:
```
cargo run --release -p pixelflow-search --example oracle_filtered_budget_curves
```
Output CSV: `docs/results/2026-09-01-oracle-filtered-budget-curves.csv` (6,300 rows =
225 expressions × 2 curves × 14 application-count checkpoints).

**Supersedes** `docs/results/2026-08-30-oracle-filtered-budget-curves.md`'s null result. That
run's checkpoint grid (fractions of a per-tier nominal sweep budget) started after 97.8% of
expressions had already ended, so `regret_pct` was 0.0000 in every row. This re-run is the
recalibration that report and the design revision (§2.3) called for: checkpoints are now
denominated in **cumulative rule applications** on the geometric grid
`APP_CHECKPOINT_GRID` (25 … 204,800), and the curve loop is the shared
`pixelflow_search::egraph::anytime::run_anytime_curve` (one definition — the same loop the
Phase 3 baseline and registration use). Same corpus construction as before: 220 synthetic
expressions across 10 depth bands (seed 20260830) + the 5 named realistic shaders.

## Finding 1: the recalibration worked — curves have shape now

Unguided curves with more than one distinct cost across checkpoints: 125/220 overall,
**99/100 classical** (0/23 blitz, 26/97 rapid — consistent with the Phase 3 baseline's
finding that the anytime question is a deep-band question). Classical median regret against
the empirical best-either-curve reference falls 81.9% → 94.9% → 52.2% → 35.3% → 0.9% across
B = 25/100/200/400/800 and is ~0 from B=1600. Non-monotonicity is real (regret can rise
before it falls — B=100 > B=25 at the median): `extract_dag` is heuristic, so growth can
temporarily worsen its answer; this is exactly why the regret reference is the empirical best
over all checkpoints, never the final state.

## Finding 2 (the load-bearing one): oracle rule-filtering ≈ unguided, everywhere

Restricting the rule library to the hindsight-load-bearing rule set (`R*` per expression,
median 3/7/24 of 62 rules for blitz/rapid/classical in the 08-30 run) leaves the anytime
curve almost unchanged at every checkpoint:

| classical median regret% | B=25 | B=100 | B=200 | B=400 | B=800 |
|---|---:|---:|---:|---:|---:|
| unguided | 81.88 | 94.92 | 52.24 | 35.26 | 0.89 |
| oracle-filtered | 80.70 | 85.42 | 52.24 | 33.89 | 0.89 |

The entire visible benefit of a *perfect rule-level* oracle is ~1–10 points of median regret
at B ≤ 400 and nothing after. Dropping non-load-bearing **rules** does not shorten the path
to good extractions, because the surviving rules still re-fire their idempotent matches and
still explore in the same order — the work distribution barely changes. Combined with the
scoping round's dedup finding (90.4% of scored candidates commit nothing), the implication
for the Guide is the design revision §4's, now with direct curve evidence: **the buy-back has
to come from candidate-granularity discrimination (which match, at which e-class, now or
never), not from rule masking.** The Phase 3 registration records this as non-binding
pre-flight context (§8).

Also recorded this run (diagnostics, synthetic corpus): 219/220 quiesced (1 classical hit the
class cap); non-direct-creator share of load-bearing applications 99.26% overall (this
corpus's deeper expressions push it above the 08-30 scoping sample's figure); all 5 named
shaders are in-sample for their tier (every rapid/blitz curve, shader or synthetic, ends at
0% regret at the final checkpoint).
