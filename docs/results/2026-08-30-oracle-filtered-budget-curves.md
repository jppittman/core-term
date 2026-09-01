# Oracle-filtered anytime budget curves: inconclusive as calibrated (2026-08-30)

Reproduce:
```
cargo run --release -p pixelflow-search --example oracle_filtered_budget_curves
```
Harness: `pixelflow-search/examples/oracle_filtered_budget_curves.rs` (889 lines, complete —
`fn main` closes cleanly, compiles clean under `cargo check`). Output: prints a summary to
stdout and writes a full per-expression/per-curve/per-work-fraction CSV to
`docs/results/2026-08-30-oracle-filtered-budget-curves.csv` (2,250 rows = 225 expressions x 2
curves x 5 work-fraction checkpoints, a complete cross product — 220 synthetic corpus items
across 10 complexity bands + 5 named realistic shaders).

**This report is written from the existing, already-complete CSV only.** A `bootstrap_extraction_head`
publication run was live in another worktree while this audit was in progress, with a
sentinel-guarded abort budget; an attempted re-run of this harness (to capture the console
summary text, which is not persisted to any file) caused a +307.7% regime-change abort in that
run before it was killed at 160/225 expressions. The CSV it would have overwritten was untouched
(the harness only writes output at the very end of `main`), so no data was lost, but this report
does not include the console-only diagnostics (per-tier quiesce-before-nominal-budget percentages,
the shader in-sample check) that were never captured to disk by whichever agent ran this
originally — only what the CSV itself contains. Do not re-run this harness while another
worktree's timed benchmark may be live; check `pgrep -fl "bootstrap_extraction_head|bench_extraction_3way"`
first.

## Methodology, from the harness's own module doc

The kernel language is strongly normalizing and this optimizer is budget-only by design:
saturation spends a budget and stops, full stop, and reaching quiescence
(`stats.total_unions == 0`) is a diagnostic condition, never a certified fixpoint. So there is no
privileged "full saturation" reference to call `C_full` and regret everything else against.
Instead, for each expression, the harness runs **two curves** — **unguided** (all 62 rules) and
**oracle-filtered** (only the rules a hindsight pass over the unguided run's own best-found
extraction marked load-bearing, filtered at whole-rule granularity, not per-application — see the
source's module doc for why match-level replay is not stable across runs) — from empty to a
generous ceiling (3x the expression's own production-tier iteration budget), sampling extraction
cost at standardized work fractions (0.25x/0.5x/1x/2x/3x of that per-tier nominal budget) along
the way. Regret at any checkpoint is `(cost - best_cost_ever_seen) / best_cost_ever_seen`, where
"best ever seen" is the minimum cost either curve reaches at any checkpoint for that expression —
an empirical reference, never a claimed optimum.

## Headline: no regret was observed anywhere in this run — and the reason why is itself the finding

**All 2,250 rows report `regret_pct = 0.0000`.** For every one of the 225 expressions, the
`unguided` and `oracle` curves reach *identical* extraction cost at *every* one of the five
checkpoints. This is not the intended positive result ("oracle-filtered keeps pace with
unguided, cheaply") landing cleanly — it is the checkpoint grid never getting a chance to see a
difference, for two distinct, corpus-measurable reasons:

| Cause | Share of the 225 expressions | What it looks like in the CSV |
|---|---:|---|
| Saturation reaches quiescence (or the safety class-count cap) **at or before the first checkpoint** (25% of nominal budget) | **97.8%** (220/225) | `iteration`, `classes`, `applications`, `cost` are byte-identical across all 5 `frac` rows for that expression+curve |
| Saturation hits `EGraph::saturate_with_limits`'s class-count safety cap (~15,000 classes for the largest, "classical"-tier expressions) within single-digit-to-teens iterations, freezing the extraction for the rest of the curve | **4.9%** (11/225, all "classical" tier, node_count 1,255-8,511) | class count converges to 14,949-14,998 (a ~15k cap) by iteration 4-14, then stays fixed through frac=3.0 |
| Genuinely converges that fast (small/simple expressions, not capped) | remaining ~93% of the 97.8% | e.g. `shader_redundant` (13 nodes): quiesces at iteration 14, cost 17, unchanged through frac=3.0 |

Only 5/225 expressions (2.2%) even show the *iteration count itself* varying across checkpoints
(meaning the loop was still actively running between frac=0.25 and frac=3.0) — and even for those
5, `cost` is flat across every checkpoint (two are already at their minimum extractable cost from
the first sample; the other three are class-capped mid-run). In no case, for any expression at
any work level, does `unguided` cost ever differ from `oracle` cost.

## What this does and does not tell us

**It does not refute the Guide thesis.** Unguided and oracle never *disagreeing* is consistent
with oracle-filtering being "free" at these particular checkpoints, but the checkpoints tested
are all *past* the point where either curve stops changing — so this run cannot distinguish "the
Guide has no headroom to offer" from "this checkpoint grid is entirely on the flat part of both
curves." The `guide_headroom` measurement (a different harness, same session) independently shows
2.6x-34x headroom exists in *which applications were load-bearing*; that headroom is invisible
here because the coarsest checkpoint already comes after saturation has settled.

**It does not confirm the thesis either.** "Does oracle-filtered replay reach forms unguided
never finds" — the specific question this harness was written to answer — has no evidence either
way in this run: the two curves are identical at every sampled point, which could mean "no,
never" or could mean "the sampled points are all too coarse to see it." This measurement is
**inconclusive**, not negative.

**Root cause: the checkpoint grid (`FRACS` x `config_for_node_count`'s per-tier nominal
iteration budget) is calibrated far above what this corpus actually needs to settle.** The
smallest fraction tested, 25% of nominal, is already enough iterations for 97.8% of expressions
to fully quiesce or hit their safety cap. To see an anytime curve with actual shape, the next
attempt at this measurement should checkpoint at **absolute small iteration counts** (1, 2, 3, 4,
..., 20) rather than fractions of an already-generous nominal budget — the same lesson
`guide_headroom` reports independently (all 800 of its expressions quiesced before either budget
cap; saturation is fast relative to the budgets this codebase currently configures for it).
**This recalibration was not attempted in this round** — CPU-heavy re-runs of this harness are
deferred (see the note at the top about the concurrent publication benchmark), and a config-only
change without re-running would not produce trustworthy numbers to report.

## What is usable from this run: the oracle rule-set reduction itself

Independent of the (uninformative) regret numbers, the CSV does confirm the *magnitude* of rule
filtering a perfect oracle would apply, per tier (`rules_allowed` column, oracle curve):

| Tier | n | median oracle rules kept (of 62) |
|---|---:|---:|
| blitz (smallest, node_count 6-10) | 24 | 3 |
| rapid (node_count 11-47) | 101 | 7 |
| classical (node_count 53-8,511) | 100 | 24 |

This is directionally consistent with `guide_headroom`'s finding that larger expressions have
*more* rules earning "load-bearing" credit under the labeler bound (more distinct rewrite shapes
get exercised as expressions grow) even though the *fraction* of applications that matter shrinks.
It is not, on its own, evidence that filtering to this reduced set preserves quality at a smaller
budget — that is exactly the question the flat curves above failed to test.

## Design implications for Phase 3

1. **This measurement needs to be re-run with absolute-iteration checkpointing before it can
   speak to the anytime-curve question.** As calibrated, it is not evidence for or against the
   Guide thesis; treat it as a null result caused by checkpoint-grid miscalibration, not as
   "the Guide has nothing to offer." Recommend this as explicit follow-up work, not blocking Phase
   3's design (the design doc's go/no-go leans on `guide_headroom` and the saturation-delta
   measurement instead, both of which are decisive and use deterministic per-application counts
   rather than budget-fraction checkpoints).
2. **The oracle rule-count-kept table (3/7/24 median by tier) is usable now** as a rough sizing
   prior for how aggressively a Guide might restrict the candidate rule set at each tier, but
   should be corroborated once a properly-calibrated anytime run exists.
3. **Any future version of this harness should record "hit safety cap" as a distinct status from
   "quiesced,"** rather than treating both as the same `stats.total_unions == 0` diagnostic — this
   audit could only tell them apart by cross-referencing class counts against the ~15,000 cap after
   the fact; the harness itself does not currently distinguish them.

Full per-expression, per-curve, per-checkpoint data: `docs/results/2026-08-30-oracle-filtered-budget-curves.csv`.
