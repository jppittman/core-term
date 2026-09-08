# Phase 3 re-run banner: the two instrument changes

**Status:** binding. Every Phase 3 results doc ported forward gets this banner
appended to whatever re-run banner it already carries. Its own banners are
kept as written — this one is an addition, never a rewrite.

Two things about how a Phase 3 anytime curve is *measured* changed between the
Round-1 runs and the forward-ported harness. Neither is a bug fix in the
analysis; both are changes to the instrument, and both mean **a re-run of a
registered number does not reproduce it even if nothing else changed**.

---

## 1. The application budget binds mid-scan, not between sweeps

Round 1 crossed a checkpoint target at **rule-sweep granularity**: the sample
at target `B` was taken at the first between-sweeps point where the cumulative
count was `>= B`, and `AnytimeCheckpoint::app_actual` recorded the exact —
possibly overshooting — count.

`EGraph::saturate_bounded` (#1118) enforces the cap **the moment an
application is about to commit** (`ScanStop::ApplicationBudget`). A live
checkpoint now lands on `app_actual == app_target` exactly.

This is a strictly better instrument: the overshoot was an artifact of where
the check sat, not a property of the arm being measured, and it was larger for
whichever arm happened to have longer sweeps. Two consequences:

- `AnytimeCheckpoint::app_actual` stays in the schema — a curve that always
  equals its target is still worth asserting, and
  `live_checkpoints_land_exactly_on_their_target` asserts it — but it **can no
  longer explain a discrepancy** against a Round-1 artifact. On a Round-1 row
  a gap between `app_target` and `app_actual` was information; on a new row it
  is a bug.
- Any arm whose Round-1 sample was taken after an overshoot was measured with
  slightly more work than its label claimed. The direction of that bias is not
  uniform across arms, so it cannot be corrected after the fact.

## 2. The reported cost is the DAG cost, not the tree cost

Every Round-1 number — the registered `B` and `Y`, the Round-1b domain-shift
deltas, the Round-2 regret-vs-|R| tables, the R2G return labels — was read off
`ExtractedDAG::total_cost`, which is the **tree** cost: every child summed at
every use, sharing never priced.

`ChoiceCost::dag` (#1117) is what the emitted kernel actually pays, since
`choices_to_arena` materializes exactly one arena node per reachable e-class
and codegen let-binds the shared ones. On `shader:julia_set` the two numbers
are **1.4e7 and 716**.

The ported harnesses report `dag`. Both shapes are carried on every
checkpoint (`ChoiceCost { tree, dag }`) so a comparison can always name which
number it is looking at, but:

- The registered constants **port as-is and are not re-derivable** from a
  `dag`-cost re-run. They are correct records of what was measured under the
  instrument of the day.
- The R2G *return* label is a difference of costs. Whether the
  strict-versus-return rank correlation survives the unit change is an **open
  question** this port does not answer.

---

## What is unchanged

The x-axis is still **rule applications**, counted one per action commit
including idempotent re-fires, on `EGraph::application_count` — the budget's
own unconditional denominator, never gated on whether anyone is recording
provenance. The checkpoint grid (`APP_CHECKPOINT_GRID`), the class cap as
environment-not-work, the clamping convention for a run that ended early, and
the regret reference being the empirical best over all checkpoints rather than
the final state: all as registered.
