# Round 2: unguided anytime curves vs rule count

Per-mode sections appended by each agent's binary as it completes its run (docs/plans/2026-09-01-phase3-round2-rule-scaling.md §9). This file did not exist when the mode (iii) section below was written — it was created with mode (iii)'s section only.

## Round 2, mode (iii): unguided anytime curves, |R|=62 vs 62+batch

Agent: rule-batch implementation + unguided curves (docs/plans/2026-09-01-phase3-round2-rule-scaling.md §2.3, §8). Base arm `|R|=62` (`all_rules()`, unchanged production set); batch arm `|R|=95` (`all_rules() + pixelflow_search::math::round2_rules::experimental_rules()`, `+33` new rules, harness-only — never added to `all_rules()`). Same Round-1 400-expression stratified sample (stride 10.36 over 4143 TRAIN+DEV expressions), same shared `run_anytime_curve`/`APP_CHECKPOINT_GRID`, deterministic `CostModel::latency_prior()` — UNGUIDED only, no Guide loaded or trained. Batch arm's per-curve safety ceiling scaled by `95/62` per §5 (still panics if it binds; it did not bind on this run).

### Tier: blitz (n=23)

| B | arm |R| | n | median cost@B | median cost@quiescence | quiescence excl. cycle-cost | cycle-cost hits | median trunc-loss% | median apps-to-quiescence | class_cap hits | quiesced |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 62 (base) | 23 | 78.00 | 78.00 | 78.00 | 0 | 0.000 | 13.0 | 0 | 23 |
| 100 | 95 (+batch) | 23 | 78.00 | 78.00 | 78.00 | 0 | 0.000 | 259.0 | 0 | 23 |
| 200 | 62 (base) | 23 | 78.00 | 78.00 | 78.00 | 0 | 0.000 | 13.0 | 0 | 23 |
| 200 | 95 (+batch) | 23 | 78.00 | 78.00 | 78.00 | 0 | 0.000 | 259.0 | 0 | 23 |

_"cycle-cost hits" = curves whose quiescence cost hit `extract_dag`'s CYCLE_COST sentinel (>= 900,000) — a genuinely self-referential e-class post-saturation, a pre-existing extraction-algorithm behavior verified present under UNCHANGED `all_rules()` alone on large classical expressions (not introduced by `experimental_rules()`; see "quiescence excl. cycle-cost" for the reliable median over the non-cyclic curves only). `median cost@B` (the B=100/200 regret-relevant number) is unaffected: cyclic e-classes only arise at very high application counts, past where B or 4B falls._

Closure-gain at quiescence (`fid(e) = (cost@62 - cost@62+batch)/cost@62` at curve-end, median over the blitz tier expressions with a nonzero, non-cyclic base cost; 0 excluded for a CYCLE_COST sentinel on either arm): **0.000%**, positive for 7/23.

- ended (base, n=23): quiesced=23 class_cap=0 grid_exhausted=0 sweep_ceiling=0 timeout=0 | ended_at_apps median=13 p90=48
- ended (+batch, n=23): quiesced=23 class_cap=0 grid_exhausted=0 sweep_ceiling=0 timeout=0 | ended_at_apps median=259 p90=2170

### Tier: rapid (n=189)

| B | arm |R| | n | median cost@B | median cost@quiescence | quiescence excl. cycle-cost | cycle-cost hits | median trunc-loss% | median apps-to-quiescence | class_cap hits | quiesced |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 62 (base) | 189 | 271.00 | 266.00 | 266.00 | 0 | 0.000 | 80.0 | 0 | 189 |
| 100 | 95 (+batch) | 189 | 276.00 | 286.00 | 263.00 | 14 | 0.000 | 2043.0 | 0 | 189 |
| 200 | 62 (base) | 189 | 266.00 | 266.00 | 266.00 | 0 | 0.000 | 80.0 | 0 | 189 |
| 200 | 95 (+batch) | 189 | 282.00 | 286.00 | 263.00 | 14 | 0.000 | 2043.0 | 0 | 189 |

_"cycle-cost hits" = curves whose quiescence cost hit `extract_dag`'s CYCLE_COST sentinel (>= 900,000) — a genuinely self-referential e-class post-saturation, a pre-existing extraction-algorithm behavior verified present under UNCHANGED `all_rules()` alone on large classical expressions (not introduced by `experimental_rules()`; see "quiescence excl. cycle-cost" for the reliable median over the non-cyclic curves only). `median cost@B` (the B=100/200 regret-relevant number) is unaffected: cyclic e-classes only arise at very high application counts, past where B or 4B falls._

Closure-gain at quiescence (`fid(e) = (cost@62 - cost@62+batch)/cost@62` at curve-end, median over the rapid tier expressions with a nonzero, non-cyclic base cost; 14 excluded for a CYCLE_COST sentinel on either arm): **0.251%**, positive for 97/175.

- ended (base, n=189): quiesced=189 class_cap=0 grid_exhausted=0 sweep_ceiling=0 timeout=0 | ended_at_apps median=80 p90=258
- ended (+batch, n=189): quiesced=189 class_cap=0 grid_exhausted=0 sweep_ceiling=0 timeout=0 | ended_at_apps median=2043 p90=14357

### Tier: classical (n=188)

| B | arm |R| | n | median cost@B | median cost@quiescence | quiescence excl. cycle-cost | cycle-cost hits | median trunc-loss% | median apps-to-quiescence | class_cap hits | quiesced |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 62 (base) | 188 | 14459.50 | 3181.50 | 2621.00 | 25 | 48.467 | 2685.5 | 0 | 187 |
| 100 | 95 (+batch) | 188 | 14459.50 | 1000006.00 | 1689.50 | 100 | 40.611 | 24686.5 | 3 | 183 |
| 200 | 62 (base) | 188 | 7413.50 | 3181.50 | 2621.00 | 25 | 21.922 | 2685.5 | 0 | 187 |
| 200 | 95 (+batch) | 188 | 7648.50 | 1000006.00 | 1689.50 | 100 | 16.941 | 24686.5 | 3 | 183 |

_"cycle-cost hits" = curves whose quiescence cost hit `extract_dag`'s CYCLE_COST sentinel (>= 900,000) — a genuinely self-referential e-class post-saturation, a pre-existing extraction-algorithm behavior verified present under UNCHANGED `all_rules()` alone on large classical expressions (not introduced by `experimental_rules()`; see "quiescence excl. cycle-cost" for the reliable median over the non-cyclic curves only). `median cost@B` (the B=100/200 regret-relevant number) is unaffected: cyclic e-classes only arise at very high application counts, past where B or 4B falls._

Closure-gain at quiescence (`fid(e) = (cost@62 - cost@62+batch)/cost@62` at curve-end, median over the classical tier expressions with a nonzero, non-cyclic base cost; 102 excluded for a CYCLE_COST sentinel on either arm): **0.782%**, positive for 62/86.

- ended (base, n=188): quiesced=187 class_cap=0 grid_exhausted=1 sweep_ceiling=0 timeout=0 | ended_at_apps median=2686 p90=32208
- ended (+batch, n=188): quiesced=183 class_cap=3 grid_exhausted=2 sweep_ceiling=0 timeout=0 | ended_at_apps median=24686 p90=58740

### Per-arm stop-reason detail (raw, ALL tiers combined)

| arm | quiesced | class_cap | grid_exhausted | sweep_ceiling | timeout |
|---|---|---|---|---|---|
| base (|R|=62) | 399 | 0 | 1 | 0 | 0 |
| +batch | 395 | 3 | 2 | 0 | 0 |

Raw per-expression rows: 400 expressions x 14 grid checkpoints x 2 arms, computed but not dumped here (this is a curve/aggregate report, not a raw-data artifact — re-run this binary for the full grid if per-expression rows are needed).

_Grid: `[25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800]`. Stop-name legend: `quiesced` = saturation reached a fixpoint before any budget bound; `class_cap` = hit the tier's `max_classes` memory-protection cap; `grid_exhausted` = still finding unions when the grid's top checkpoint (204800 applications) was reached; `sweep_ceiling`/`timeout` = safety ceilings (would panic if actually hit — listed for completeness, expected 0 in both columns)._
