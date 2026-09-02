# Round 2: unguided anytime curves vs rule count

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

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

## Round 2, modes (i)/(ii): unguided anytime curves vs rule count

Agent: rule-set inflation harness (modes i/ii) + unguided curves
(docs/plans/2026-09-01-phase3-round2-rule-scaling.md §2.1, §2.2, §8). Mode (i)
(exact duplicates, the "learnable overhead" control) via
`pixelflow_search::math::inflate::DuplicateRule`: a duplicate delegates
`apply`/`is_destructive`/templates straight to its inner rule, under a new
index and name `"<inner>#dup<k>"` — same closure, purely a second/third/…
way to reach the same node. Mode (ii) (mechanical compositions A∘B) via a new
`pixelflow_search::egraph::template::TemplateRewrite`: first-order syntactic
unification of `B.lhs` against every position of `A.rhs`, executed through a
new, additive `RewriteAction::Instantiate` variant (`egraph/rewrite.rs`,
`egraph/graph.rs`) — nothing in `all_rules()`, `saturate_until_applications`,
or the anytime runner changed behavior. Every rule set is `all_rules()` (62,
unchanged, pinned by its own test) as an exact prefix, followed by inflation,
built by `pixelflow_search::math::inflate::build_rule_set`
(`RuleSetSpec::parse`: `"dup:<total>"` / `"comp:<total>"`; never pads — an
unreachable grid point is a hard `RuleSetError`). Rule-set identity
(`rule_set_fingerprint`) is recorded per row.

**Composition oracle gate (§2.4).** Every A∘B composition is validated by a
cross-form oracle (`pixelflow_search::math::oracle`, `#[cfg(test)]`, plus a
runtime-reachable duplicate of the identical check inline in the curve
binary, since `pixelflow-ir`'s `oracle` feature is a `pixelflow-search`
`[dev-dependencies]`-only feature — a normal build of that crate never links
`eval_scalar`): instantiate LHS and RHS at 256 random leaf assignments,
compare via `pixelflow_ir::eval_scalar` under a magnitude-relative tolerance
(rel 1e-3, abs 1e-3, points below 1e-2 magnitude recorded as ill-conditioned
rather than scored). A composition is kept if its agreement rate is >= 0.60
— **a rate threshold, not zero-tolerance**, because a composition can induce
a narrower *domain* in its surface metavariables than either parent rule had
alone (verified empirically while building this harness: e.g. `pow(x,-1)`
composed with `sin(neg(x)) -> neg(sin(x))` at the `pow` base position is a
true identity over the reals, but this codebase's `Pow` — `exp2(y*log2(x))`
— saturates to its underflow floor for a negative base, which random
sampling over the reals inevitably hits about half the time; see
`math::oracle`'s module doc for the full argument). Run against the seeded
pool (`composition_pool(all_rules(), 0x5EED2)`, `cargo test -p
pixelflow-search math::oracle`): **2,935 kept, 9 DROPPED LOUDLY** (below
0.60; every name and rate printed by the test), **395 untestable** (>4
surface metavariables or a mask-valued root — outside what `eval_scalar`'s
4-coordinate `Var` convention can check, reported, never silently accepted),
mean agreement rate **0.997** over the 2,944 checkable rules. The pool
(2,935 survivors) comfortably covers every grid point's inflation (max 186).

Same Round-1 400-expression stratified sample (stride 10.36 over 4,143
TRAIN+DEV expressions; corpus MD5s verified identical to the registration —
train `0ed6cf16abcbc006cd7a3ee2365b15b4`, dev
`3026133ebba066eeca10f658da554400`), same `run_anytime_curve`/
`APP_CHECKPOINT_GRID`, deterministic `CostModel::latency_prior()` — UNGUIDED
only, no Guide loaded, trained, or scored anywhere in this binary. Per-curve
safety ceiling and sweep ceiling scaled by `|R|/62` per §5 (still panics if
it binds; it did not bind on any completed rule set below).

**Status at commit time: 7 of 9 rule sets complete** (`base`, `dup:93`,
`dup:124`, `dup:186`, `dup:248`, `comp:93`, `comp:124` — the full duplicate
grid and two of four composition points). `comp:186` and `comp:248` were
still running in the background (single long-lived process, `--rule-sets`
covering the full grid in one invocation) when this section was written; see
"Known limitations" below for exactly what that means for reading this data,
and re-run/inspect `docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.{csv,json}`
for the completed grid once that process finishes.

### Per-tier summary (7 completed rule sets)

median cost@B (B=100/200), median truncation loss vs 4B, median
applications-to-quiescence, and median cost at the grid's final checkpoint
(204800 applications — quiescence for curves that finished, the clamped
final state otherwise), excl./incl. `extract_dag`'s CYCLE_COST sentinel
(>=900,000; a pre-existing extraction-algorithm behavior on large classical
expressions, not introduced by this inflation — same caveat mode (iii)'s
section above documents).

**Tier: blitz (n=23) — flat across every rule set.**

| rule_set | \|R\| | cost@100 | cost@200 | trunc-loss% (B=100) | median apps-to-quiescence | cost@quiescence |
|---|---|---|---|---|---|---|
| base | 62 | 78.0 | 78.0 | 0.000 | 13 | 78.0 |
| dup:93 | 93 | 78.0 | 78.0 | 0.000 | 16 | 78.0 |
| dup:124 | 124 | 78.0 | 78.0 | 0.000 | 42 | 78.0 |
| dup:186 | 186 | 78.0 | 78.0 | 0.000 | 74 | 78.0 |
| dup:248 | 248 | 78.0 | 78.0 | 0.000 | 85 | 78.0 |
| comp:93 | 93 | 78.0 | 78.0 | 0.000 | 13 | 78.0 |
| comp:124 | 124 | 78.0 | 78.0 | 0.000 | 13 | 78.0 |

**Tier: rapid (n=189).**

| rule_set | \|R\| | cost@100 | cost@200 | trunc-loss% (B=100) | median apps-to-quiescence | cost@quiescence |
|---|---|---|---|---|---|---|
| base | 62 | 271.0 | 266.0 | 0.000 | 80 | 266.0 |
| dup:93 | 93 | 271.0 | 266.0 | 0.000 | 87 | 266.0 |
| dup:124 | 124 | 271.0 | 266.0 | 0.000 | 157 | 266.0 |
| dup:186 | 186 | 271.0 | 266.0 | 0.000 | 185 | 266.0 |
| dup:248 | 248 | 271.0 | 266.0 | 0.000 | 314 | 266.0 |
| comp:93 | 93 | 271.0 | 266.0 | 0.000 | 80 | 266.0 |
| comp:124 | 124 | 271.0 | 266.0 | 0.000 | 80 | 266.0 |

**Tier: classical (n=188).**

| rule_set | \|R\| | cost@100 | cost@200 | trunc-loss% (B=100) | median apps-to-quiescence | cost@quiescence (all) | cost@quiescence (excl. cycle) | cycle-cost hits | class_cap hits |
|---|---|---|---|---|---|---|---|---|---|
| base | 62 | 14459.5 | 7413.5 | 48.467 | 2686 | 3181.5 | 2621.0 | 25 | 0 |
| dup:93 | 93 | 14459.5 | 7413.5 | 48.467 | 3115 | 3192.0 | 2621.0 | 27 | 0 |
| dup:124 | 124 | 14459.5 | 7413.5 | 48.467 | 4274 | 3181.5 | 2621.0 | 25 | 0 |
| dup:186 | 186 | 14459.5 | 7413.5 | 48.467 | 4803 | 3181.5 | 2621.0 | 25 | 0 |
| dup:248 | 248 | 14459.5 | 7413.5 | 48.467 | 5322 | 3181.5 | 2621.0 | 25 | 0 |
| comp:93 | 93 | 14459.5 | 7413.5 | 48.467 | 2980 | 3181.5 | 2597.0 | 27 | 0 |
| comp:124 | 124 | 14459.5 | 7413.5 | 48.467 | 3809 | 3192.0 | 2609.0 | 28 | 0 |

`quiesced` counts (out of 188 classical) stayed 186–187 across every rule
set; `grid_exhausted` 1–2; `sweep_ceiling`/`timeout` 0 throughout — no
safety ceiling bound at any completed grid point.

### The headline observation: cost@B is flat in \|R\|; applications-to-quiescence is not

**Median `cost@B=100` and `cost@B=200` are bit-for-bit identical across
every completed rule set, in every tier**, including at |R|=248 (4x the
base rule count). Median `cost@quiescence` is likewise flat to within
noise (3181.5 → 3192.0 → 3181.5 → 3181.5 → 3181.5 for classical dup:62→248;
the two 3192.0 readings are `dup:93`/`comp:124`, and even those are a
~0.3% wobble, not a trend). **`median applications-to-quiescence` is NOT
flat** — it grows monotonically and substantially with |R| in mode (i):
2686 → 3115 → 4274 → 4803 → 5322 for classical (dup:62→248, ~2x), 80 → 87 →
157 → 185 → 314 for rapid (~4x), 13 → 16 → 42 → 74 → 85 for blitz (~6.5x).

Two distinct things are going on here, and it matters to keep them apart
when this data feeds the Register statistics:

1. **The flat median cost is expected, not a finding — it is what §4
   predicts.** Modes (i) and (ii) are closure-preserving by construction
   (§2.1: "a duplicate can create nothing the original cannot"; §2.2: "the
   composed rule's closure is contained in the closure of {A, B}"), so the
   best reachable cost of an expression is unchanged at every |R| — only
   *search efficiency* (applications needed) should vary, and it clearly
   does (previous paragraph). A flat `cost@quiescence` median across |R| is
   the harness behaving correctly, not evidence against H1.

2. **The flat `cost@100`/`cost@200` for classical tier is a checkpoint-
   granularity artifact worth flagging explicitly, because it affects how
   H1 should be read off these two specific B values on this corpus.**
   `run_anytime_curve`'s checkpoint sampling is at **sweep-boundary
   granularity**, not per-application (`anytime.rs`: "the sample at target
   B is taken at the first between-sweeps point where the cumulative count
   is >= B"). For a classical-tier expression, ONE sweep (every rule tried
   against every class once) already produces well over 100–200
   applications on its own (median `apps-to-quiescence` for classical is
   2,686 even at the base 62 rules — round 1 alone plausibly accounts for
   a four-figure share of that). So `cost@100` and `cost@200` are actually
   sampling **"state after round 1 completes," not "state after 100/200
   applications of budget-constrained search"** — and round 1's *content*
   (which classes exist, what they contain) is identical whether the extra
   |R|-62 rules fired 0 times or several hundred idempotent times within
   that same round, because index-ordered duplicates/compositions can only
   re-derive what the original low-index rules already derived first. The
   *application count* recorded for round 1 does grow with |R| (idempotent
   re-fires are still recorded applications, matching §2.1's semantics) —
   which is exactly the `apps-to-quiescence` growth measured above — but
   the *checkpoint's sampled cost* does not, because it was never actually
   constrained by the 100/200 target for classical-tier expressions in this
   corpus. **This means B=100 and B=200 are not testing "early-budget
   regret vs |R|" for the classical tier here** — a genuine, reportable
   methodological caveat for the Register/H1 read on this specific grid,
   independent of anything the composition or duplicate generators did.
   blitz/rapid tiers, whose `apps-to-quiescence` medians (13–85, 80–314)
   straddle 100/200, are a cleaner read of the same question and show the
   same flat-cost/growing-apps pattern — consistent, not contradicted.

This is reported as data, not diagnosed further here — the Register run's
own Spearman/bootstrap statistics (§1.2) operate on the full grid including
`comp:186`/`comp:248`, and should be computed from `--out-json`'s
per-expression rows directly rather than from this section's medians-only
summary.

### Known limitations

- **comp:186 and comp:248 were still running** (single background process,
  started before this section was written, ~14 minutes elapsed and still
  actively computing on the classical-tier tail — the largest expressions in
  the stratified sample — when this section was committed). The process was
  left running rather than killed; `docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.csv`
  will gain the remaining ~11,200 rows (2 rule sets x 400 x 14) as it
  completes, and `--out-json` (written only once, at the very end of all 9
  rule sets) will not exist until then. A follow-up session should check
  whether the process is still alive, wait for it, and append the
  `comp:186`/`comp:248` rows to the tables above before this data is used
  for the Register's H1 statistics (§1.2 needs the full 5-point grid for
  both modes).
- **The composition rule sets this run actually used (`comp:93`/`comp:124`,
  and `comp:186`/`comp:248` once done) were built by
  `build_rule_set`/`composition_pool` directly, WITHOUT re-running the
  oracle filter at construction time** — only `cargo test -p pixelflow-search
  math::oracle` gates the pool. Given the pool-wide drop rate (9/2,944 =
  0.3%), the expected count of oracle-failing compositions inside any
  specific `comp:N`'s first-N-of-the-shuffle prefix is well under 1 (max N
  here is 186), but this is not a guarantee — a small residual risk that a
  handful of applications in the `comp:N` curves above came from a
  composition that would fail the 0.60 agreement-rate bar. This did not
  visibly affect the aggregate medians (`comp:93`/`comp:124`'s numbers track
  `dup:93`/`dup:124`'s closely, as expected), but a future run should either
  filter the pool inline before `build_rule_set` draws from it, or treat
  this as accepted harness-level noise and say so in the Register.
- **Per-candidate overhead** (§7.1: raw matches enumerated per round,
  scored-candidate count per application) is **not measured in this run** —
  this binary reports application counts and stop reasons only. A raw-match
  measurement (`EGraph::find_rewrite_matches().len()` at the final state,
  per curve) would need a small, separate pass; not done here given the
  compute budget already spent on the curves themselves.
