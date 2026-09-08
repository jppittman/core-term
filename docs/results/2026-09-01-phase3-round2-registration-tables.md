> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

**Classical band (n=188), absolute cost and where the inflation is visible.** `visible@B` = expressions whose cost@B differs from the |R|=62 curve's; `first visible` = smallest grid checkpoint at which any expression's cost differs from |R|=62 (`never` = identical at all 14 checkpoints).

| rule set | \|R\| | fingerprint | cost@100 q1/med/q3 | cost@200 q1/med/q3 | cost@400 med | cost@800 med | curve-end med (excl. cycle) | cycle hits | app_actual@100 med | visible@100 | visible@200 | first visible | apps-to-end med | ended |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `base` | 62 | `e99af8402beaff5d` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | — | — | n/a | 2686 | app_budget=1, quiesced=187 |
| `dup:93` | 93 | `fdd617246eb98590` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3327.5 | 2621.0 | 27 | 113 | 0 | 3 | 200 | 3115 | app_budget=2, quiesced=186 |
| `dup:124` | 124 | `87fefd5a63575175` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | 0 | 0 | never | 4274 | app_budget=1, quiesced=187 |
| `dup:186` | 186 | `37a4c537606a549b` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | 0 | 0 | never | 4803 | app_budget=1, quiesced=187 |
| `dup:248` | 248 | `809a0f52b61fe6c0` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | 0 | 0 | never | 5322 | app_budget=1, quiesced=187 |
| `comp:93` | 93 | `0c3fd6f35f444a59` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2597.0 | 27 | 113 | 0 | 0 | 800 | 2980 | app_budget=1, quiesced=187 |
| `comp:124` | 124 | `521798ae521a0572` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3925.5 | 3331.0 | 2609.0 | 28 | 113 | 0 | 0 | 400 | 3809 | app_budget=1, quiesced=187 |
| `new:95` | 95 | `4f4a4cbd2e4f89cb` | 2952 / 14459.5 / 1002550 | 1639 / 7648.5 / 1000456 | 5270.0 | 3884.0 | 1689.5 | 100 | 113 | 0 | 18 | 200 | 24686 | app_budget=2, class_cap=3, quiesced=183 |

**Classical band, unguided regret U against the unguided-only closure-aware reference at the same |R|, truncation loss L, and Y.** Percentages; regret quartiles are per-expression.

| rule set | \|R\| | U@100 med | p25 | p75 | p90 | U@200 med | p25 | p75 | p90 | L@100 med | Y@100 | L@200 med | Y@200 | closure gain vs 62: med / p90 / >0 / <0 (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base` | 62 | 96.58 | 47.35 | 27840.42 | 75221.25 | 40.49 | 0.00 | 113.73 | 43079.78 | 48.467 | 16.32 | 21.922 | 8.99 | — (reference) |
| `dup:93` | 93 | 96.58 | 47.35 | 25047.55 | 75221.25 | 40.49 | 0.00 | 113.73 | 43032.34 | 48.467 | 16.32 | 21.922 | 8.99 | 0.000 / 0.000 / 0 / 4 (173) |
| `dup:124` | 124 | 96.58 | 47.35 | 27840.42 | 75221.25 | 40.49 | 0.00 | 113.73 | 43079.78 | 48.467 | 16.32 | 21.922 | 8.99 | 0.000 / 0.000 / 0 / 0 (173) |
| `dup:186` | 186 | 96.58 | 47.35 | 27840.42 | 75221.25 | 40.49 | 0.00 | 113.73 | 43079.78 | 48.467 | 16.32 | 21.922 | 8.99 | 0.000 / 0.000 / 0 / 0 (173) |
| `dup:248` | 248 | 96.58 | 47.35 | 27840.42 | 75221.25 | 40.49 | 0.00 | 113.73 | 43079.78 | 48.467 | 16.32 | 21.922 | 8.99 | 0.000 / 0.000 / 0 / 0 (173) |
| `comp:93` | 93 | 92.32 | 46.50 | 27840.42 | 75221.25 | 39.97 | 0.00 | 108.97 | 43079.78 | 48.467 | 16.32 | 21.922 | 8.99 | 0.000 / 0.000 / 0 / 2 (173) |
| `comp:124` | 124 | 92.32 | 46.50 | 27840.42 | 73724.51 | 39.97 | 0.00 | 113.73 | 43079.78 | 48.467 | 16.32 | 21.922 | 8.99 | 0.000 / 0.000 / 1 / 3 (173) |
| `new:95` | 95 | 108.60 | 49.89 | 36981.76 | 80463.09 | 46.95 | 2.18 | 424.41 | 57590.04 | 40.611 | 14.44 | 16.941 | 7.24 | 0.737 / 2.202 / 121 / 21 (173) |

**blitz (n=23) — reported, no claim.**

| rule set | \|R\| | cost@100 med | cost@200 med | U@100 med | L@100 med | apps-to-end med | ended |
|---|---:|---:|---:|---:|---:|---:|---|
| `base` | 62 | 78.0 | 78.0 | 0.00 | 0.000 | 13 | quiesced=23 |
| `dup:93` | 93 | 78.0 | 78.0 | 0.00 | 0.000 | 16 | quiesced=23 |
| `dup:124` | 124 | 78.0 | 78.0 | 0.00 | 0.000 | 42 | quiesced=23 |
| `dup:186` | 186 | 78.0 | 78.0 | 0.00 | 0.000 | 74 | quiesced=23 |
| `dup:248` | 248 | 78.0 | 78.0 | 0.00 | 0.000 | 85 | quiesced=23 |
| `comp:93` | 93 | 78.0 | 78.0 | 0.00 | 0.000 | 13 | quiesced=23 |
| `comp:124` | 124 | 78.0 | 78.0 | 0.00 | 0.000 | 13 | quiesced=23 |
| `new:95` | 95 | 78.0 | 78.0 | 0.00 | 0.000 | 259 | quiesced=23 |

**rapid (n=189) — reported, no claim.**

| rule set | \|R\| | cost@100 med | cost@200 med | U@100 med | L@100 med | apps-to-end med | ended |
|---|---:|---:|---:|---:|---:|---:|---|
| `base` | 62 | 271.0 | 266.0 | 0.00 | 0.000 | 80 | quiesced=189 |
| `dup:93` | 93 | 271.0 | 266.0 | 0.00 | 0.000 | 87 | quiesced=189 |
| `dup:124` | 124 | 271.0 | 266.0 | 0.00 | 0.000 | 157 | quiesced=189 |
| `dup:186` | 186 | 271.0 | 266.0 | 0.00 | 0.000 | 185 | quiesced=189 |
| `dup:248` | 248 | 271.0 | 266.0 | 0.00 | 0.000 | 314 | quiesced=189 |
| `comp:93` | 93 | 271.0 | 266.0 | 0.00 | 0.000 | 80 | quiesced=189 |
| `comp:124` | 124 | 271.0 | 266.0 | 0.00 | 0.000 | 80 | quiesced=189 |
| `new:95` | 95 | 276.0 | 282.0 | 0.00 | 0.000 | 2043 | quiesced=189 |

**Per-mode H1 statistics (classical, from the tables above).**

| mode | grid \|R\| | B | U(\|R\|) | Spearman rho | U(max) - U(62) | Delta1 | H1 direction | H1 effect | Y(\|R\|) | Delta2 | LS slope of U per rule |
|---|---|---:|---|---:|---:|---:|---|---|---|---:|---:|
| (i) | [62, 93, 124, 186, 248] | 100 | [96.585, 96.585, 96.585, 96.585, 96.585] | undefined (all tied) | +0.000 | 28.230 | FAILS | FAILS | [16.32, 16.32, 16.32, 16.32, 16.32] | 0.020 | 0 (exact) |
| (i) | [62, 93, 124, 186, 248] | 200 | [40.489, 40.489, 40.489, 40.489, 40.489] | undefined (all tied) | +0.000 | 12.342 | FAILS | FAILS | [8.99, 8.99, 8.99, 8.99, 8.99] | 0.020 | 0 (exact) |
| (ii) | [62, 93, 124] | 100 | [96.585, 92.322, 92.322] | -0.866 | -4.262 | 28.230 | FAILS | FAILS | [16.32, 16.32, 16.32] | 0.020 | -0.06875 pts/rule |
| (ii) | [62, 93, 124] | 200 | [40.489, 39.966, 39.966] | -0.866 | -0.522 | 12.342 | FAILS | FAILS | [8.99, 8.99, 8.99] | 0.020 | -0.00843 pts/rule |
| (iii) | [62, 95] | 100 | [96.585, 108.599] | 1.000 | +12.014 | 28.230 | holds | FAILS | [16.32, 14.44] | 0.020 | +0.36407 pts/rule |
| (iii) | [62, 95] | 200 | [40.489, 46.945] | 1.000 | +6.456 | 12.342 | holds | FAILS | [8.99, 7.24] | 0.020 | +0.19565 pts/rule |
