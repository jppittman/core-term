# Phase 3 Round 2 registration v2: regret at budget vs rule count, interleaved order

**SUPERSEDES** `docs/plans/2026-09-01-phase3-round2-registration.md` — **change:** inflated rules
are INTERLEAVED into the sweep order by a fixed seed, and every budget is now reported in SWEEPS
as well as applications. **Reason:** under v1's appended order, H1 was unobservable by
construction — 0/188 classical expressions differed in cost@B for ANY inflated set at B ∈ {100,
200} (v1 §6.1), because a sweep-major order cannot reach an appended rule (index ≥ 62) before
every one of the 62 production rules has swept once, and on classical expressions one pass over
the 62-rule prefix alone already exceeds B = 100 (v1: median `app_actual` at the B = 100
checkpoint is 113). v1's own §6 "what a superseding registration would have to change" named
interleaving as option (b) and adopted nothing (correct discipline for that document). This
document adopts it.

**Date:** 2026-09-01
**Status:** REGISTERED — §4/§5/§6 are filled from a completed Register run (§11 Entry 1,
2026-09-01): 8 of 9 grid points (`comp:186`/`comp:248` did not complete within the |R|-scaled
safety ceiling on two independent attempts — §4). **H1 FAILS on this grid, decisively, in every
mode measured** — not unobservable (v1's outcome under append order), but a large, consistent
effect in the direction opposite H1's prediction (§6). Nothing in §5 is copied from v1: v1's
numbers were computed under the order this document replaces, and are not valid for a different
sweep order (a different rule order changes which application lands at which checkpoint, hence
`app_actual`, cost@B, and every downstream statistic). Per the rule inherited from v1, §5 as filled
here may not be revised except by append (§11) or a further superseding registration; a guided run
at |R| > 62 may use this document as its registration for the 8 completed points, but not for mode
(ii) past |R|=124 (comp:186/comp:248 remain unregistered).
**Authority:** `docs/plans/2026-08-31-guide-design-revision.md` (§5 protocol);
`docs/plans/2026-09-01-phase3-registration.md` (Round 1 — FROZEN; B, Y's formula, the grid, the
curve runner, and the reference convention are inherited from it verbatim, via v1);
`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` (Round 2 design — H1/H2 statistics, modes,
what Register may and may not fix, §6 — carried verbatim except where this document states a
change); `docs/plans/2026-09-01-phase3-round2-registration.md` (v1 — superseded by this document;
its §H1 result, §10 proof discipline, and §12 reproduction shape are the template for this one);
JP's directive, verbatim (2026-09-01): "supersede round 2 with interleaved rules, go." JP's scope
cut, verbatim, same message: *"I thought this learned egraph structure could be more broadly
applicable to all kinds of complex cpu decision making. We need to cut scope though... 'think
about that later, worry zero about it now.'"* — this document and its harness build the smallest
thing that answers the registered question for THIS compiler; a generality-motivated abstraction
belongs in a later document, not here (see the "Parked" note at the end of §2).

## 0. What changed in the harness (mechanism, committed by this commit)

- **`pixelflow-search/src/math/inflate.rs`**: `RuleSetSpec` gains an explicit `order: RuleOrder`
  field. `RuleOrder::Append` reproduces v1 exactly (base prefix, inflation appended, both
  unshuffled) — kept as a named, harness-selectable order, not deleted. `RuleOrder::Interleave(seed)`
  is v2's default: `build_rule_set` builds the same base+inflation content v1 built, then applies a
  seeded Fisher–Yates shuffle (`fisher_yates`, the same primitive v1 already used to permute the
  composition pool) to the WHOLE base+inflation list. **The `|R| = 62` point is never reordered
  under any order** — `build_rule_set` returns `all_rules()` verbatim whenever the requested total
  is 62, so that point stays byte-identical to Round 1 and to v1, regardless of which order the
  rest of the grid uses (pinned by `math::inflate::tests::the_62_rule_point_is_never_reordered`).
- **Spec grammar**: `"<mode>:<total>"` (e.g. `"dup:124"`) now defaults to
  `Interleave(DEFAULT_INTERLEAVE_SEED)`; `"<mode>:<total>:append"` reproduces v1;
  `"<mode>:<total>:interleave:<seed>"` picks an explicit seed (decimal or `0x`-prefixed hex). See
  §1 below for the exact seed this registration uses and the CLI syntax.
- **`rule_set_fingerprint`** is unchanged code (it already hashes rule names in Vec order, so a
  different Vec order — which is exactly what interleaving produces — already fingerprints
  differently; no change to the function was needed). Pinned by
  `math::inflate::tests::append_and_interleave_fingerprint_differently_for_the_same_content` (two
  orders of the identical rule multiset must fingerprint differently) and
  `math::inflate::tests::v2_grid_fingerprints_are_pinned` (the exact fingerprint of every realized
  v2 grid point, both orders, hardcoded — §2's table below is read from this test, not invented).
- **`pixelflow-pipeline/src/bin/phase3_round2_unguided_curves.rs`**: every curve row now also
  carries `sweeps_actual` (cumulative sweeps STARTED through the checkpoint, summed over checkpoint
  segments — the anytime runner's own counter, just not written before; §0.1 says what it does
  and does not measure), `evals_actual` (cumulative rule-match ATTEMPTS at the
  checkpoint — new: `EGraph::total_evals()`, accumulated at the one site every rule-application
  path funnels through, `apply_rule_at_index_timed`), and `apps_per_sweep` (applications recorded
  by exactly one full sweep of this expression's rule set, measured by a separate one-sweep probe
  run before the real curve — a throwaway e-graph, discarded immediately, never sharing state with
  the measured curve). §0.1–§0.2 below say what each buys.

### 0.1 Sweep-denominated budgets

v1's own finding was that B = 100 lands "inside the first sweep of the production prefix" for a
majority of classical expressions — i.e. B is *below one sweep* at |R| = 62 already, and further
below one sweep as |R| grows (a larger rule set's first sweep costs more applications). Reporting
B only in applications hid this. `apps_per_sweep` (per expression, per rule set) lets every
reported `app_actual` be re-expressed as `app_actual / apps_per_sweep` sweeps, so a reader can see
directly whether a given (mode, |R|, B) point is sub-sweep, one sweep, or several — the aggregate
table in the filled §4 reports both axes side by side at every grid point, not just applications.

**What `sweeps_actual` measures — corrected after reading the Register run's data (§11 Entry 2),
before the first version of this paragraph could mislead anyone.** The anytime runner takes each
checkpoint as its own segment: `EGraph::saturate_until_applications` starts a fresh pass at rule
index 0 for every segment, cuts that pass between rules when the cumulative application count
reaches the target, and counts the pass as one `iteration` whether or not it completed.
`sweeps_actual` is the running sum of those counts — **passes started, not sweeps completed** — and
is therefore ≥ the checkpoint ordinal by construction. In the Register run it equals the ordinal
almost everywhere (median 1/2/3/4/5 at targets 25/50/100/200/400 for every rule set, including
`dup:248`, where one full pass is ~995 applications and three completed passes by B = 100 are
impossible). It is NOT the sweep-denominated budget. The registered sweep denomination is
`B / apps_per_sweep` from the one-sweep probe (§4.1); `sweeps_actual` stays in the CSV as the raw
runner counter only. Two consequences a reader needs:

- (a) At every point where `B / apps_per_sweep < 1`, B = 100 is spent as three partial passes
  (segments 25, 50, 100) that each begin at the FRONT of the rule vector, so the rules the seed
  placed early in the interleaved vector receive most of the budget (including their idempotent
  re-fires, which count as applications — §1) and rules near the tail may never be visited at that
  B. Under `Interleave` the front is a seeded random mixture of base and inflated rules — exactly
  why inflated rules are now reachable (§4) — but WHICH rules sit there is a property of the one
  seed `0x2026_0901`. A seed-sensitivity check is not part of this registration; its absence is a
  stated limitation, not something to be fixed quietly later.
- (b) At |R| = 62, `B / apps_per_sweep ≈ 1.0`: Round 1's B = 100 point and this document's `base`
  point are each roughly one front-to-back pass over `all_rules()` in production order.

### 0.2 The §7.1 overhead precondition, measured for the unguided arm

Design §7.1 requires that a Guide's per-round cost stay flat in |R| — measured there as "scored
candidates per recorded application." Unguided saturation has no scorer, so the stand-in this
document registers is **raw matches enumerated per application**: `evals_actual / app_actual` at
B, per (mode, |R|). This is the precondition's unguided half — the guided half (scored-candidate
count) requires `GuidedEpisodeStats` on `GuidedSaturation`, which does not exist yet (v1 §9 point 3;
still true here) and is out of this document's scope. `sweeps_at` and `evals_at` are both raw,
per-checkpoint, per-expression columns in the CSV (never only an aggregate), so "raw matches
enumerated per round" (`evals_actual` delta / `sweeps` delta between two checkpoints) is
reconstructable exactly, not estimated.

## 1. Environment (fixed; inherited from v1/Round 1 unless stated)

| Item | Value |
|---|---|
| Corpus | Unchanged: `gen_bench_corpus --target 4000 --seed 42`; train MD5 `0ed6cf16abcbc006cd7a3ee2365b15b4`, dev `3026133ebba066eeca10f658da554400`. FINAL not opened. |
| Sample (H1) | Unchanged: Round 1's 400-expression size-stratified TRAIN+DEV sample (stride 10.36 over 4,143): blitz 23, rapid 189, **classical 188** (TRAIN 154 + DEV 34). |
| Sample (H2) | Unchanged: full DEV classical band, n = 334. |
| Budgets | Unchanged: classical **B = 100 (primary), B = 200 (secondary)**. Now ALSO reported in sweeps at every point (§0.1) — no new budget values, a new way of reading the same ones. |
| Cost model | Unchanged: `CostModel::latency_prior()`, deterministic. |
| Work axis | Unchanged: cumulative recorded rule applications, idempotent re-fires included. Sweeps and raw match-attempts (`evals_actual`) are reported ALONGSIDE, never substituted as the budget axis. |
| Checkpoint grid | Unchanged: `APP_CHECKPOINT_GRID`, 14 points, 25…204800. |
| **Sweep order (the change)** | Rule-major, as before, but the rule VECTOR each sweep iterates is now base+inflation **shuffled together** by a fixed seed for every inflated set, rather than base-prefix-then-inflation-appended. **Seed: `DEFAULT_INTERLEAVE_SEED = 0x2026_0901`** (`pixelflow_search::math::inflate::DEFAULT_INTERLEAVE_SEED`), stated once here and reused by every default-order spec string. The \|R\| = 62 point is exempt (§0). The append order remains available (`:append` suffix) for any comparison back to v1. |
| Class cap | Unchanged: `config_for_node_count(node_count).max_classes`. |
| Safety ceilings | Unchanged: per-curve wall-clock `300 s × \|R\|/62`, sweep ceiling `10,000 × \|R\|/62`; PANIC when they bind. |
| Source rev | Harness at the commit that adds this document, branch `claude/phase3-round2`. `inflate.rs` gains `RuleOrder`/`DEFAULT_INTERLEAVE_SEED`/order-aware `build_rule_set`; `phase3_round2_unguided_curves` gains `sweeps_actual`/`evals_actual`/`apps_per_sweep` columns; `EGraph` gains `total_evals()`; `AppBudgetSaturationStats` and `AnytimeCheckpoint` gain an `evals`/`evals_actual` field. No behavior change to `all_rules()`, to the `:append` order, or to any existing public function's return value for inputs that existed before this commit. |
| Production | Unchanged: `all_rules()` = 62, pinned by test. |

## 2. The |R| grid, rule-set identity under the new order, and what is parked

Rule-set identity is `rule_set_fingerprint` as before (now order-sensitive by construction, not by
a change to the function — see §0). Every fingerprint below is read from
`math::inflate::tests::v2_grid_fingerprints_are_pinned`, which hardcodes and re-checks these exact
values on every test run — this table is not hand-computed.

| Mode | spec (default order) | \|R\| | fingerprint (interleave, seed `0x20260901`) | fingerprint (`:append`, = v1's committed value) |
|---|---|---:|---|---|
| shared | `base` | 62 | `e99af8402beaff5d` (unshuffled under every order — §0) | same |
| (i) | `dup:93` | 93 | `83e610e33e782a68` | `fdd617246eb98590` |
| (i) | `dup:124` | 124 | `b207aa331bb625ab` | `87fefd5a63575175` |
| (i) | `dup:186` | 186 | `3a00c565900b48e6` | `37a4c537606a549b` |
| (i) | `dup:248` | 248 | `43c43d764ef7f76b` | `809a0f52b61fe6c0` |
| (ii) | `comp:93` | 93 | `904ceec9b110e89e` | `0c3fd6f35f444a59` |
| (ii) | `comp:124` | 124 | `a7600e5942f0baa5` | `521798ae521a0572` |
| (ii) | `comp:186` | 186 | `9e9bf3a4458a3045` | `ff65cfbabc95a6cf` |
| (ii) | `comp:248` | 248 | `b89d841eada63c13` | `dfc176cd60c7124f` |
| (iii) | `new:95` | 95 | `113cca49c99cc850` | `4f4a4cbd2e4f89cb` |

The `:append` fingerprints for `dup:93/124/186/248` and `comp:93/124` reproduce v1's §2 table
exactly (same code path, unchanged) — confirmed by
`math::inflate::tests::v2_grid_fingerprints_are_pinned`. `comp:186`/`comp:248`'s `:append`
fingerprints are newly computed here: v1 never finished writing curve rows for these two points
(v1 §2.2 — the process producing the committed CSV did not survive), but the rule sets themselves
were always fully specified and cheap to fingerprint (no curve run needed for a fingerprint); they
are included for completeness and because the interleaved-order Register run below intends to
realize the full 9-point grid this time. `new:95`'s `:append` fingerprint reproduces
`phase3_round2_new_rules.rs`'s original `all_rules() + experimental_rules()` splice byte-for-byte
(same content, same order) — mode (iii) now goes through the same `RuleSetSpec` path as the other
two modes, per the "Mode (iii) update" note above.

**Mode (iii) update (JP's follow-up directive, same date):** the paragraph below described this
document's FIRST commit, which left mode (iii) out. JP's next instruction explicitly asked for
`new:95` on the same interleaved-order grid, so a later commit on this branch gave mode (iii) its
own `InflationMode::NewRules` variant and `build_new_rule_set` in `inflate.rs` — `"new:95"` now
goes through the same `RuleSetSpec`/`build_rule_set`/`RuleOrder` path as `dup`/`comp`, exactly
like the two existing modes, rather than staying a separate hand-rolled `all_rules() +
experimental_rules()` splice in `phase3_round2_new_rules.rs`. `NEW_RULES_GRID = &[95]` is
mode (iii)'s one point (`62 + experimental_rules().len()`, pinned by
`math::inflate::tests::new_rules_grid_matches_experimental_rules_len` against a future edit to the
batch silently drifting the registered count); order applies to it exactly as §0 describes for
`dup`/`comp` (the 62-point never shuffled, `Interleave(DEFAULT_INTERLEAVE_SEED)` the default,
`:append` reproducing v1's `all_rules() + experimental_rules()` splice byte-for-byte). This
supersedes the "out of scope"/"parked" framing directly below, which is kept for the historical
record of the design's first pass rather than deleted. v1's mode (iii) finding (H1 fails, +12.0
points at B=100, entirely a reference/fidelity effect — v1 §6 point 4) was measured under append
order and is superseded by §4/§6's `new:95` row below, for the same reason every other mode's v1
number is superseded (§3's "what changes under interleaving").

**Original paragraph, kept for the record — no longer current, see above.** Mode (iii)'s harness
(`round2_rules::experimental_rules()`, `phase3_round2_new_rules.rs`) does not build rule sets
through `RuleSetSpec`/`build_rule_set` — it extends `all_rules()` directly with a fixed batch — so
`RuleOrder` does not apply to it as written. v1's mode (iii) finding (H1 fails, +12.0 points at
B=100, entirely a reference/fidelity effect — v1 §6 point 4) is UNCHANGED by this document and is
not re-registered here. **Parked, per JP's scope cut:** whether mode (iii) should also gain an
interleaved-order option is a question about a different harness, not required to answer THIS
round's H1/H2 questions, and is not addressed by this commit.

**Also parked, per the same scope cut:** nothing in this document's mechanism (a rule-set builder
with a pluggable sweep order) is generalized beyond `pixelflow-search`'s inflation harness. The
broader idea — that learned e-graph structure or a sweep-order policy could apply to other complex
CPU decision-making beyond this compiler — is explicitly not built, not abstracted toward, and not
mentioned again below. If it matters later, it earns its own document then.

## 3. Reference and regret conventions (binding, unchanged from v1/design)

Carried verbatim from v1 §3 — the closure-aware reference, the regret formula, and the cost-at-B
convention do not depend on sweep order and are not superseded by this document:

- `ref(e, |R|)` is the **closure-aware** reference at the same |R|: the minimum cost over every
  checkpoint of every arm at that |R| for expression e (design §1.1). Never pooled across |R|.
- In this Register, and for H1 for all time, the reference is the unguided-only version
  `ref_U(e, |R|) = min over the 14 checkpoints of the unguided curve at |R|`.
- H2's statistics (`Q`, `G`) use the two-arm reference at each |R| when guided data exist; `Q(|R|)`
  itself is a cost ratio and needs no reference.
- Regret `= (cost@B − ref) / ref`; a positive cost against a zero reference is infinite regret,
  never 0%.
- Cost at B is the checkpoint whose `app_target` = B (first between-rules point with cumulative
  applications ≥ B); `app_actual` is recorded and reported, now alongside `app_actual /
  apps_per_sweep` (§0.1).
- The `CYCLE_COST` extraction sentinel convention is unchanged from v1 §3.

**What changes under interleaving, stated so it is not rediscovered by surprise:** because the
rule VECTOR order differs from v1, `app_actual` at a given `app_target` can differ from v1's value
for the same expression and the same |R| — a checkpoint may now land after a different subset of
rules has swept, because inflated rules are no longer confined to the tail of each sweep. This is
the intended effect (§0), not a bug; it is exactly why `ref_U`, cost@B, and every statistic below
must be recomputed under this order rather than read off v1's tables.

## 4. Measured interleaved-order curves — RUN, 8 of 9 grid points

Register run, `phase3_round2_unguided_curves --release`, same 400-expression sample, same CSV
shape plus the three new columns. Full tables (absolute cost, `visible@100`/`visible@200`/`first
visible`, sweeps/`evals_actual` overhead, regret/truncation-loss/closure-gain, per-tier, H1
statistics) live in `docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.md` — not
retranscribed here to avoid a second, driftable copy; this section states what changed and points
there.

**`comp:186` and `comp:248` did not complete.** Two independent attempts each ran `comp:186` to
curve 200/400 then hit `anytime.rs`'s wall-clock safety ceiling panic at the same ~25-expression
window, under different system load (16.45 then 12.35) — treated as a genuine compute wall for at
least one expression at `|R|=186`, not contention, per the results doc's "Grid coverage" section.
Mode (ii)'s grid below is therefore `[62, 93, 124]` only, not the full 5 points modes (i)/(iii)
mirror; `comp:186`/`comp:248` remain open, exactly as they did in v1 (§2), for a different
proximate reason (v1's process didn't survive; v2's hits the scaled ceiling deterministically).

**The prediction stated in advance was correct, and understated:** `visible@100`/`visible@200` are
nonzero in every mode, including mode (i) — but not merely nonzero: **159–188 of 188** classical
expressions differ from `base` at B=100 for every one of the 7 completed inflated sets (`new:95`
is 188/188 — literally every classical expression), versus v1's committed 0/188 for any set. This
is the direct before/after evidence §6 draws on.

### 4.1 B in sweeps at every point (the sweep-denominated statement)

`apps_per_sweep` = applications recorded by one full pass of the rule vector on a throwaway e-graph
of the expression (§0), median over the 188 classical expressions; `B in sweeps` = B / that
median. The `sweeps_actual` runner counter is deliberately not used here (§0.1). Source: the
results doc's "Sweeps and match-enumeration overhead" table, which is computed from the raw CSV.

| rule set | \|R\| | apps_per_sweep (med) | B=100 in sweeps | B=200 in sweeps | app_actual@100 (med) | reading |
|---|---:|---:|---:|---:|---:|---|
| `base` | 62 | 99.5 | 1.01 | 2.01 | 113 | B=100 ≈ one full pass of `all_rules()` in production order |
| `dup:93` | 93 | 142.0 | 0.70 | 1.41 | 102 | below one sweep at B=100 |
| `dup:124` | 124 | 300.0 | 0.33 | 0.67 | 106 | below one sweep at both B |
| `dup:186` | 186 | 502.0 | 0.20 | 0.40 | 104 | below one sweep at both B |
| `dup:248` | 248 | 995.5 | 0.10 | 0.20 | 108 | below one sweep at both B |
| `comp:93` | 93 | 87.0 | 1.15 | 2.30 | 104 | ≈ one sweep at B=100 |
| `comp:124` | 124 | 89.5 | 1.12 | 2.23 | 109 | ≈ one sweep at B=100 |
| `comp:186` / `comp:248` | 186 / 248 | — | — | — | — | not realized (§4) |
| `new:95` | 95 | 291.5 | 0.34 | 0.69 | 108 | below one sweep at both B |

Two things the table makes visible. First, "B in sweeps" is not a function of |R| alone: a `dup`
sweep costs 1.4× / 3.0× / 5.0× / 10× base at 1.5× / 2× / 3× / 4× the rules (superlinear — each
duplicate's re-fires also grow the graph the next rule scans), while a `comp` sweep costs LESS than
base (87–89.5 vs 99.5; compositions match rarely), so the column must be read per mode. Second,
every completed inflated point except `comp:*` is sub-sweep at B = 100, several by a wide margin;
§6's H1 movement is therefore a statement about what the first partial pass(es) of an interleaved
vector find, not about saturation across the whole rule set.

**Round 1 note (as this document's brief requires):** the `base` row is why Round 1's B = 100
guided-vs-unguided win is largely rule-list order within the first sweep — B = 100 is 1.01 sweeps
of `all_rules()`, so which of the 62 rules come first decides which fire at all; Round 1's
discussion already says this, and the table above is the quantified version.

**Append-only provision for `comp:186` / `comp:248` (v1 §2.2's form, carried forward):** their rule
sets are fully specified (pool seed `0x5EED2`, inflation prefix lengths 124 / 186, interleave seed
`0x2026_0901`, fingerprints in §2) and their unguided curves may be appended under §11 as data,
under this registration's constants, provided that no guided run at |R| ∈ {186, 248} of mode (ii)
precedes them. If appended, the only registered numbers that change are mode (ii)'s rows in §5.2
(Y at 186 / 248), §5.4 (Δ2, whose |R|max becomes 248), §5.6, and §6 (ρ, U(max) − U(62), verdict
recomputed over five points); Δ1, B, the sample, the reference convention, and every mode (i) /
(iii) number stay fixed. Realizing them within §1's |R|-scaled safety ceiling failed on two
attempts (§4); raising that ceiling is a change to §1 and is a superseding change for those two
points, stated here so that it cannot be made quietly.

## 5. Registered constants — from the interleaved-order Register run (§11 Entry 1)

Per the binding rule inherited from v1 (§6 of the design, carried forward), these are fixed ONLY
from unguided data on the interleaved-order harness, committed before any guided run at |R| > 62
under this order, and — now committed — may not be revised except by append (§11) or by a further
superseding registration. Mode (ii) below is fixed only through |R|=124 (`comp:186`/`comp:248`
incomplete, §4) — its numbers are registered for the points that exist, not extrapolated to the
missing two.

### 5.1 Inherited, not re-derived
B = 100 / 200 (classical); Y's formula `Y = 1 − (1 + L/2)/(1 + L)`; ε = 0.005; the 0.02 floor on
Δ2; the pool seed `0x5EED2` (which compositions are IN the prefix — unchanged, orthogonal to
sweep order, §0); the bootstrap seed `42`; the oracle seed `0xC0FF_EE42`; the sample; the grid;
the reference convention (§3); the H2 statistics (design §1.3). **New this document:** the
interleave seed `0x2026_0901` (§1), now also part of "what is fixed, not re-derived" once §4/§5
are filled.

### 5.2 Y(|R|) per mode and B

| Mode | \|R\| | L@100 | **Y@100** | L@200 | **Y@200** |
|---|---:|---:|---:|---:|---:|
| (i) | 93 | 20.910 | 8.65 | 0.181 | 0.09 |
| (i) | 124 | 15.597 | 6.75 | 5.088 | 2.42 |
| (i) | 186 | 2.565 | 1.25 | 1.047 | 0.52 |
| (i) | 248 | 10.996 | 4.95 | 6.997 | 3.27 |
| (ii) | 93 | 24.604 | 9.87 | 0.175 | 0.09 |
| (ii) | 124 | 14.371 | 6.28 | 2.684 | 1.31 |
| (ii) | 186, 248 | — | — | — | — (missing, §4) |
| (iii) | 95 | −0.000 | −0.00 | 0.000 | 0.00 |

(`base`, |R|=62: L@100 = 48.467, Y@100 = 16.32; L@200 = 21.922, Y@200 = 8.99 — the reference row,
restated here since every Δ2 below subtracts it.)

### 5.3 Δ1 — H1's minimum effect (from |R| = 62 only)

95% bootstrap CI of the median unguided regret at |R| = 62 on the 188 classical expressions
(10,000 resamples, seed 42, order-statistic 2.5/97.5 percentiles), run fresh under the interleaved
harness (the |R| = 62 rule set and its curve are byte-identical to v1's — §0 — but the CI is
recomputed rather than copied, per v1's own discipline).

| B | median U(62) | CI | **Δ1** |
|---:|---:|---|---:|
| 100 | 0.9658 (96.58%) | [0.7312, 1.2958] | **0.2823 (28.23 pts)** |
| 200 | 0.4049 (40.49%) | [0.2570, 0.5039] | **0.1234 (12.34 pts)** |

### 5.4 Δ2 — H2's minimum effect, per mode

`Δ2 = max(0.02, Y(|R|max) − Y(62))`, |R|max = the largest COMPLETED point per mode (§4):

| Mode | \|R\|max | Y(\|R\|max) − Y(62) @100 | **Δ2@100** | Y(\|R\|max) − Y(62) @200 | **Δ2@200** |
|---|---:|---:|---:|---:|---:|
| (i) | 248 | 4.95 − 16.32 = −11.37 pts | 0.020 (floor; negative) | 3.27 − 8.99 = −5.72 pts | 0.020 (floor; negative) |
| (ii) | 124 (not 248 — §4) | 6.28 − 16.32 = −10.04 pts | 0.020 (floor; negative) | 1.31 − 8.99 = −7.68 pts | 0.020 (floor; negative) |
| (iii) | 95 | −0.00 − 16.32 = −16.32 pts | 0.020 (floor; negative) | 0.00 − 8.99 = −8.99 pts | 0.020 (floor; negative) |

Every mode's raw `Y(|R|max) − Y(62)` is negative (Y falls, it does not rise), so every Δ2 sits at
the 0.02 floor — the same outcome §6 reaches from U directly: the effect is real and large, in the
opposite direction from what Δ2 (built to test "does Y rise enough") is shaped to detect.

### 5.5 The H2 slope-ratio threshold

Same derivation rule as v1 §5.5 (evidence only when the 95% bootstrap CI of `slope_U` excludes
zero and `slope_U > 0`). Every completed mode's least-squares `slope_U` (results doc, H1 table) is
**negative** — mode (i)@100: −0.268 pts/rule; (i)@200: −0.036; (ii)@100: −0.885; (ii)@200: −0.311;
(iii)@100: −1.923; (iii)@200: +0.351 (the lone exception, a 2-point line, not a fit with a CI). The
precondition `slope_U > 0` fails everywhere it can be checked, so the H2 slope-ratio threshold is
**not evidence-bearing on this grid** — it was designed to interpret a rising slope, and finds a
falling one instead.

### 5.6 Per-candidate Guide overhead — the §7.1 flatness check

Unguided half (§0.2), median `evals_actual / app_actual` at B, per (mode, |R|) — full table in the
results doc's "Sweeps and match-enumeration overhead" section. Against the base (|R|=62) values
(31.20 @100, 39.54 @200), every inflated point's ratio is 2.0–3.3× the base value at B=100 (e.g.
`dup:248` 78.85/31.20 = 2.53×; `comp:124` 100.60/31.20 = 3.23×; `new:95` 69.33/31.20 = 2.22×) —
**every completed inflated point exceeds design §7.1's "≤ 2× its value at |R| = 62" flatness
threshold at B=100**, though several (e.g. `dup:186` 91.81/31.20 = 2.94×, `new:95`'s ratio falls to
1.57× at B=200) sit close to or under it at B=200. This is the raw-match-enumeration precondition
only (§0.2) — a Guide's SCORED-candidate overhead is a separate, still-unmeasured quantity pending
`GuidedEpisodeStats` (unchanged from v1 §9 point 3) — but it is worth flagging now: whatever guided
mechanism eventually runs at these |R| values inherits a raw enumeration cost per application that
is already 2–3× the |R|=62 baseline before any scoring is added.

**Registered threshold, as numbers:** flat ⇔ median `evals_actual / app_actual` at B is ≤ 2× its
|R| = 62 value, i.e. **≤ 62.40 at B = 100 and ≤ 79.08 at B = 200**. Measured:

| rule set | \|R\| | evals/app @100 | × base | flat @100? | evals/app @200 | × base | flat @200? |
|---|---:|---:|---:|---|---:|---:|---|
| `base` | 62 | 31.20 | 1.00 | — | 39.54 | 1.00 | — |
| `dup:93` | 93 | 75.37 | 2.42 | no | 79.54 | 2.01 | no (by 0.46) |
| `dup:124` | 124 | 78.07 | 2.50 | no | 76.37 | 1.93 | yes |
| `dup:186` | 186 | 91.81 | 2.94 | no | 92.67 | 2.34 | no |
| `dup:248` | 248 | 78.85 | 2.53 | no | 74.75 | 1.89 | yes |
| `comp:93` | 93 | 85.77 | 2.75 | no | 92.42 | 2.34 | no |
| `comp:124` | 124 | 100.60 | 3.22 | no | 102.79 | 2.60 | no |
| `comp:186` / `comp:248` | 186 / 248 | — | — | not realized | — | — | not realized |
| `new:95` | 95 | 69.33 | 2.22 | no | 61.98 | 1.57 | yes |

Verdict: the unguided half of the §7.1 precondition **fails at B = 100 at every completed inflated
point** and at B = 200 at all but `dup:124`, `dup:248`, `new:95`. Whichever guided mechanism
consumes this registration must report its scored-candidate count against these same rows.

## 6. H1 verdict on this grid

Per design §1.2 and v1's own discipline, H1 is entirely an unguided measurement, computed here
under the interleaved order, before any guided run at |R| > 62 exists under this registration.

| Mode | grid \|R\| | B | U(\|R\|) (%) | differing from `base` at B=100 / B=200 (of 188), per inflated point | Spearman ρ | U(max) − U(62) | Δ1 | U(max) − U(62) ≥ +Δ1? | direction | effect | **verdict** |
|---|---|---:|---|---|---:|---:|---:|---|---|---|---|
| (i) | 62,93,124,186,248 | 100 | 96.59, 43.76, 41.12, 15.44, 38.67 | 171/147, 186/157, 179/178, 186/160 | −0.900 | −57.92 pts | 28.23 | no (−2.05·Δ1) | FAILS | FAILS | **H1 FAILS (opposite direction)** |
| (i) | 62,93,124,186,248 | 200 | 40.49, 6.69, 25.44, 9.47, 27.02 | (same points as above) | −0.100 | −13.46 pts | 12.34 | no (−1.09·Δ1) | FAILS | FAILS | **H1 FAILS** |
| (ii) | 62,93,124 (incomplete, §4) | 100 | 96.59, 60.55, 41.69 | 159/151, 184/148 | −1.000 | −54.89 pts | 28.23 | no (−1.94·Δ1) | FAILS | FAILS | **H1 FAILS**, grid incomplete |
| (ii) | 62,93,124 (incomplete, §4) | 200 | 40.49, 25.60, 21.22 | (same points as above) | −1.000 | −19.27 pts | 12.34 | no (−1.56·Δ1) | FAILS | FAILS | **H1 FAILS**, grid incomplete |
| (iii) | 62,95 | 100 | 96.59, 33.11 | 188/186 | −1.000 | −63.47 pts | 28.23 | no (−2.25·Δ1) | FAILS | FAILS | **H1 FAILS** |
| (iii) | 62,95 | 200 | 40.49, 52.06 | (same point as above) | +1.000 (2-pt) | +11.57 pts | 12.34 | **no (+0.94·Δ1 — inside the noise floor)** | holds (2-pt) | FAILS | **H1 FAILS** (effect below Δ1; a 2-point ρ is not evidence of a trend) |

**H1 fails on this grid, decisively, in every mode measured — but not by being unobservable (v1's
outcome).** U(|R|) falls sharply and monotonically-in-the-large as |R| grows in modes (i) and (ii)
(ρ ≤ −0.90 wherever more than 2 points exist); mode (iii)'s single step also moves in the
"wrong" direction at B=100. At B = 100 every |U(max) − U(62)| clears Δ1 (by 1.9–2.3×: −57.9,
−54.9, −63.5 pts against Δ1 = 28.2); at B = 200 modes (i)/(ii) clear it more narrowly (1.1× and
1.6×) and mode (iii)'s lone positive step (+11.6 pts) sits BELOW Δ1 = 12.3, i.e. inside the noise
floor. (An earlier draft of this paragraph and the §11 Entry 1 commit message said "2–5×"; that
was an arithmetic overstatement and is corrected here, not silently.) So at the primary budget
this is not a noise-floor call — it is a real, large effect, opposite in sign to H1's prediction
that unguided regret at fixed B should RISE with |R|; at the secondary budget the same sign holds
in modes (i)/(ii) at a smaller margin, and mode (iii) is uninformative. **Reading:** more rules, swept in an order where they can
actually be reached inside the budget, let unguided saturation find a cheaper form FASTER on
these classical expressions than the base 62-rule set does at the same B — the added search
surface is net productive within B=100–200 applications on this sample, not merely more haystack
around the same needle. This directly falsifies the premise a Guide's B=100/200 advantage would
need to grow FROM (H2 part 2, §7) on this grid: if unguided search itself gets closer to the
reference as |R| grows, the guided/unguided gap `Q(|R|)` has less room to widen, not more. §7
below restates this precisely as what remains testable for H2.

**Honest fallback fires, per §8:** this is v1's own contingency ("if H1 fails here too, §6 is the
deliverable... exactly as v1's fallback fired"), now realized with an observable, decisive result
instead of v1's null-by-construction one. The capacity finding is: unguided saturation absorbs
rule-count growth well within a 100–200 application budget on this sample, at least through
|R|=248 (modes i) and |R|=95 (mode iii); mode (ii)'s ceiling is unmeasured past |R|=124 (§4).
**This is a finding about capacity, not a reason to touch the grid again:** the budget is not
binding at these rule counts on this corpus under an order that can reach the added rules. A third
grid, a third budget, or a third sweep order would be a third registration, and nothing in this
document's data motivates one.

## 6b. Confound: rule order (2026-09-01, appended after §11 Entry 2)

**§6's H1-FAILS reading conflates two different changes and mode (i) proves it.** Mode (i)
(`dup:*`) adds *exact duplicates* of existing rules — a duplicate has the identical LHS/RHS
template and closure as its original (§0 of this document, `DuplicateRule` delegates `apply`
verbatim), so it can never let saturation reach a node the base 62 rules could not already reach.
Its closure is unchanged by construction. Yet §6's own table shows U(|R|) falling just as sharply
under mode (i) as under modes (ii)/(iii) — U(62) = 96.59% → `dup:93` = 43.76% → `dup:124` = 41.12%
→ `dup:186` = 15.44% → `dup:248` = 38.67% at B = 100 (ρ = −0.900) — which cannot be a *closure*
effect (duplicates add zero closure) and therefore cannot be a genuine `|R|` effect either. What
changed between the `base` point and every `dup:*` point is not "more distinguishable rules," it is
*where the 62 real rules sit inside the swept vector*: `Interleave(seed)` Fisher-Yates-shuffles the
whole base+duplicate list together, so the seed relocates each of the 62 base rules to a new
position, and B = 100 is sub-sweep at every inflated point (§4.1: 1.01 sweeps at `base`/|R|=62,
falling to 0.10 sweeps at `dup:248`) — so which rules the budget actually reaches is decided almost
entirely by where the shuffle happened to put them, not by how many total rules exist. Production's
`all_rules()` order puts algebra/structural rules first and the high-yield numeric-fusion rules
(`power-rsqrt`, `power-recip`, `recip-sqrt`, `power-sqrt` — the four highest TRAIN strict-positive
rates in `docs/results/2026-09-01-train-guide-report.md`, 13–18%) near indices 51–60, so `base`'s
own B=100 pass already runs out before reaching most of them; any shuffle that moves even one of
those four earlier reaches a cheaper rewrite sooner, at any `|R|`. Modes (ii)/(iii) inherit the
identical confound (ρ = −1.000, −1.000) for the same structural reason — their inflation is also
shuffled into the front of the vector by the same seeded Fisher-Yates — but mode (i) is the clean
proof, because it is the one mode where "more rules" is verifiably not the explanation.

**Consequence for this document, and the production implication that follows from it.** v2's H1
verdict ("H1 fails, decisively, unguided regret at fixed B falls as `|R|` grows") is **superseded by
v3 as a *reading*, not retracted as *data*** — every measured number in §4/§5/§6 (the curves, the
fingerprints, `U(|R|)`, ρ, Δ1/Δ2) stands exactly as run and is not rerun or revised; what v3 changes
is the causal attribution, by holding order fixed via `RuleOrder::OrderMatchedBase` (§0 below) and
re-measuring the `|R|` effect against a reference that already reflects the seed's reordering of the
base 62, rather than against the unshuffled `base` point this document used. This is exactly the
gap v2's own §0.1(a) flagged as unmeasured ("a seed-sensitivity check is not part of this
registration; its absence is a stated limitation") — v3 closes it. It also surfaces a **production
quick win** independent of any inflated-rule-count question at all: if a *fixed* reordering of the
existing 62 rules — one that front-loads the high-yield numeric rules `all_rules()`'s module-major
order defers — recovers a meaningful share of Round 1's Guide-vs-unguided B=100 win at zero runtime
cost (no scoring, no learned model), that is a change to `all_rules()`'s own order, a decision for
JP to make on v3's data, not a claim this document makes or a change this branch makes to
production (`all_rules()` is unchanged by every commit on `claude/phase3-round2`, per the branch's
binding rules).

## 7. What remains testable for H2 under this Register

**H2 status: UNTESTED.** No guided run exists at any |R| > 62 (§10), so nothing in this document
evaluates H2; its statistics, thresholds, and the reading rule under an H1 failure (v1 §7, design
§1.3) stand unchanged and are neither weakened nor pre-judged by §6.

Structure carried verbatim from v1 §7 (H2 part 1/part 3 fully testable at every point; part 2
requires the Guide's advantage to grow, live in modes (ii)/(iii) and impossible-by-construction in
mode (i) exactly as v1 argued — sweep order does not change that argument, only the numbers it is
evaluated against). Thresholds `1 − Y(|R|)` (§5.2), the headroom an unguided curve leaves for a
guided arm to close at each point:

| Mode | \|R\| | 1−Y@100 | 1−Y@200 |
|---|---:|---:|---:|
| shared | 62 | 83.68% | 91.01% |
| (i) | 93 | 91.35% | 99.91% |
| (i) | 124 | 93.25% | 97.58% |
| (i) | 186 | 98.75% | 99.48% |
| (i) | 248 | 95.05% | 96.73% |
| (ii) | 93 | 90.13% | 99.91% |
| (ii) | 124 | 93.72% | 98.69% |
| (iii) | 95 | ~100.00% | ~100.00% |

Every inflated point's headroom is HIGHER than |R|=62's, not lower — the same direction as §6's
finding (unguided gets closer to the reference as |R| grows within this budget), read through the
truncation-loss lens instead of the regret lens. `new:95` leaves essentially no headroom at either
B — consistent with `L@100`/`L@200` both being ~0 in §5.2 (unguided saturation at |R|=95 is
already within noise of its own quiescent cost by B=100 on this sample). Whatever H2 part 2 (the
Guide's advantage growing with |R|) would need to show, it needs to show it against a shrinking
window, not a growing one, on this grid.

## 8. Gates

**Accept gate (per mode):** unchanged from v1 §8 — H1 AND H2 hold on DEV classical (n = 334) at
B = 100, full per-expression distributions reported; FINAL required for publication. **Does not
fire for any mode on this grid** — H1 fails everywhere measured (§6), so the accept gate's
precondition is not met for modes (i), (ii, incomplete), or (iii).

**Kill gate (per mode):** unchanged from v1 §8 — H2 part 3 failing at any |R| point on DEV, after
one clean re-mint/re-train, stops that mode. Not evaluated by this document — this Register run is
unguided-only (§9), so H2 part 3 (a guided-arm statistic) has no data here to fail or pass; the
kill gate is a question for whichever future guided run consumes this registration.

**Honest fallback: FIRES.** H1 fails on this grid (§6), so §6 is the deliverable — a large,
decisive capacity finding (unguided regret falls, not rises, with |R| — past Δ1 in every mode at
B = 100, past it more narrowly in modes (i)/(ii) at B = 200, and inside the noise floor only for
mode (iii)'s single B = 200 step — §6), rather than v1's null-by-construction non-finding. This is the SAME fallback shape v1
predicted for itself, realized for the first time with actual signal to read.

## 9. Protocol prerequisites for step 3 (must exist before any guided run at |R| > 62) — unchanged from v1 §9

None of v1's five prerequisites (`gen_strict_labels --rule-set`, fingerprint-checked
`train_guide`/Guide loaders, `GuidedEpisodeStats`, a curves binary whose guided arms reuse
`run_anytime_curve_with`, the |R| = 62 guided anchor re-run) exist yet on this branch at this
commit. This document's commit adds only: the interleave order and its tests, the three new CSV
columns, `EGraph::total_evals()`, and this doc skeleton — no guided-path code. §10 states the proof
obligation this document inherits from v1: before any guided run at |R| > 62 under THIS
registration, a grep proof analogous to v1 §10 must show no such run exists yet, run fresh (v1's
own §10 proof is about v1's fingerprints and does not cover this document's interleaved-order
fingerprints).

## 10. Proof that no guided run at |R| > 62 exists under this registration

**Updated after the Register run (§4–§6 now filled):** the first bullet below is no longer
literally true as v1 originally wrote it ("no result file... mentions any of them yet") — the
Register run's OWN unguided curve CSV/JSON now legitimately carry these fingerprints, because that
is the run this document commits. The bullet is restated below to say what actually still holds:
no GUIDED artifact carries them.

- `git grep` for every interleaved-order fingerprint in §2 finds them in exactly the expected
  unguided places: `pixelflow-search/src/math/inflate.rs` (the pinning test),
  `docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.{csv,json,md}`, and
  `docs/results/2026-09-01-phase3-round2-registration-v2.json` (the stats output) — every hit is
  this Register run's own unguided data or this document. No hit is a Guide checkpoint, a training
  label file, or anything under a `train_guide`/`gen_strict_labels` output path.
- `git grep -l -E 'RuleOrder::Interleave|DEFAULT_INTERLEAVE_SEED'` (verified fresh for this entry)
  matches only `inflate.rs` (the type/constant's own definition and its pinning tests) and this
  document — `phase3_round2_unguided_curves.rs` selects `Interleave` only indirectly, through
  `RuleSetSpec::parse`'s default (a bare `"dup:93"` string, no literal `RuleOrder::Interleave` or
  `DEFAULT_INTERLEAVE_SEED` token in that file) — nothing under a Guide training or checkpoint
  path either way.
- No Guide checkpoint or label artifact anywhere in `docs/results/` carries any fingerprint from
  §2's interleave column — the only Guide artifacts on the branch remain Round 1's, at |R| = 62,
  predating this document (same three files v1 §10 named); this Register run added zero Guide
  artifacts.
- `phase3_round2_unguided_curves` imports no `nnue::guide` symbol (unchanged from v1 and from this
  document's earlier commits; the Register run's diff to that binary was the sweeps/evals columns
  and the probe function only — nothing guided-path-shaped was added to realize §4).

**Fresh proof, run at the commit that finalizes this registration (§11 Entry 2), on
`claude/phase3-round2`, verbatim:**

```text
$ for fp in 83e610e33e782a68 b207aa331bb625ab 3a00c565900b48e6 43c43d764ef7f76b \
           904ceec9b110e89e a7600e5942f0baa5 9e9bf3a4458a3045 b89d841eada63c13 113cca49c99cc850; do
    printf '%s: ' $fp; git grep -l "$fp" -- . ':!docs/plans/2026-09-01-phase3-round2-registration-v2.md' | tr '\n' ' '; echo; done
83e610e33e782a68: docs/results/2026-09-01-phase3-round2-registration-v2.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.md
b207aa331bb625ab: (same four unguided files)
3a00c565900b48e6: (same four unguided files)
43c43d764ef7f76b: (same four unguided files)
904ceec9b110e89e: (same four unguided files)
a7600e5942f0baa5: (same four unguided files)
9e9bf3a4458a3045:                                   <- comp:186 interleaved: no file anywhere (never realized)
b89d841eada63c13:                                   <- comp:248 interleaved: no file anywhere (never realized)
113cca49c99cc850: (same four unguided files)

$ git grep -l -E 'RuleOrder::Interleave|DEFAULT_INTERLEAVE_SEED'
docs/plans/2026-09-01-phase3-round2-registration-v2.md
docs/results/journal.jsonl                          <- the unguided run's own journal record
pixelflow-search/src/math/inflate.rs

$ grep -c 'nnue::guide\|GuidedSaturation' pixelflow-pipeline/src/bin/phase3_round2_unguided_curves.rs
0

$ grep -rl 'math::inflate\|inflate::' pixelflow-pipeline/src/bin/
pixelflow-pipeline/src/bin/phase3_round2_new_rules.rs
pixelflow-pipeline/src/bin/phase3_round2_unguided_curves.rs   <- the only two binaries that can build an inflated set; neither links a Guide

$ sed -n 618,626p pixelflow-pipeline/src/bin/phase3_at_budget_eval.rs   # the guided harness
    let mut stepper = GuidedSaturation::new(guide, embeds);
    let out = run_anytime_curve_with(
        input.arena,
        input.root,
        all_rules(),                                <- hard-wired to |R| = 62; no rule-set argument exists
```

Why `inflate.rs` is absent from the plain-hex hits above even though it pins every fingerprint:
`v2_grid_fingerprints_are_pinned` stores them as underscored `u64` literals
(`0x83e6_10e3_3e78_2a68`, `0x9e9b_f3a4_458a_3045`, `0xb89d_841e_ada6_3c13`, …), which a
16-hex-character grep does not match. So the proof reads: the seven realized interleaved
fingerprints occur, as data, in exactly the four unguided output files of this run and nowhere
else; the two never-realized ones (`comp:186`, `comp:248`) occur as data nowhere at all — their
only occurrence on the branch is the test's pin, which is how §2 knows their values without a
curve ever having been run. No file under `docs/results/` that names a Guide, a checkpoint, a
label set, or `phase3_at_budget_eval` contains any of the nine.

## 11. Results appended against the gates

(Append-only, as in v1.)

**Entry 1 (2026-09-01, this commit).** Interleaved-order Register run, `phase3_round2_unguided_curves
--release`, 400-expression sample, 8 of 9 grid points completed (`comp:186`/`comp:248` did not —
§4). Full numbers: `docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.md` (tables + narrative),
`.csv`/`.json` (raw per-expression-per-checkpoint rows), `docs/results/2026-09-01-phase3-round2
-registration-v2.json` (aggregate stats, this document's §4–§6 source). **H1 verdict: FAILS in
every mode measured, decisively (§6)** — not unobservable (v1's outcome), but a real, large effect
in the direction opposite H1's prediction: unguided regret at B=100/200 FALLS as |R| grows on this
sample, for every completed inflated set. The honest fallback (§8) fires: §6's capacity finding is
the deliverable. `comp:186`/`comp:248` remain open grid points (mode (ii) incomplete past |R|=124),
documented, not padded or extrapolated.

**Entry 2 (2026-09-01, the commit that finalizes this registration).** No new data. Fills the
registration items Entry 1 left in the results doc only or left implicit: §4.1 (B in sweeps at
every point, the Round 1 rule-order note, the append-only provision for `comp:186`/`comp:248` in
v1 §2.2's form), §5.6 (the §7.1 threshold as numbers, per-point pass/fail), §6 (per-point count of
classical expressions differing from `base` at B = 100 / 200, an explicit `≥ +Δ1?` column, and a
correction: the earlier "clears Δ1 by 2–5×" was an overstatement — the true margins are 1.9–2.3×
at B = 100, 1.1–1.6× at B = 200 for modes (i)/(ii), and mode (iii)'s B = 200 step is below Δ1),
§7 (H2 explicitly UNTESTED, rule stands), §10 (the grep proof run fresh, verbatim). Also corrects
§0's description of `sweeps_actual` (§0.1): it counts passes STARTED per checkpoint segment, not
completed sweeps, and equals the checkpoint ordinal almost everywhere in the data; the registered
sweep denomination is `B / apps_per_sweep` (§4.1). Registered constants (§5.2–§5.5) are unchanged.

**Note on the stats script (resolved by this commit):** `round2_register_stats.py`'s `HEADER` now
includes the three new columns (`sweeps_actual`, `evals_actual`, `apps_per_sweep`) and computes
`sweeps_actual_at_B`/`evals_per_app_at_B`/`apps_per_sweep` per rule set, rendered in the results
doc's "Sweeps and match-enumeration overhead" table — the schema gap this section originally
flagged is closed as part of this same commit, not deferred further.

## 12. Reproduction

The commands actually run for Entry 1 (§11) — updated from the pre-run skeleton this section
originally sketched, to match what was executed rather than what was planned:

```bash
# unguided curves, modes (i)/(ii)/(iii), interleaved order (the default) — 8 of 9 grid points
# (comp:186/comp:248 excluded — they do not complete, see §4; run separately below)
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv \
    --out-json docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.json \
    --rule-sets base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,new:95

# comp:186/comp:248 — expect the |R|-scaled wall-clock safety ceiling to panic (§4); this is the
# correct, honest outcome (CLAUDE.md "fail loud, never silently truncate"), not a bug to route
# around. Two independent runs of this exact command both panicked at the same point.
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv /tmp/comp186_248.csv --out-json /tmp/comp186_248.json \
    --rule-sets comp:186,comp:248

# fingerprint + order-sensitivity + 62-never-reordered guarantees, incl. new:95's NewRules variant
cargo test -p pixelflow-search math::inflate -- --nocapture

# every registered number in §4/§5/§6 and the results doc's tables
python3 pixelflow-pipeline/scripts/round2_register_stats.py \
    --csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv \
    --expect base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,new:95 \
    --modes 'i=base,dup:93,dup:124,dup:186,dup:248;ii=base,comp:93,comp:124;iii=base,new:95' \
    --out-json docs/results/2026-09-01-phase3-round2-registration-v2.json \
    --out-md /tmp/round2v2_final_tables.md   # embedded into the results .md's "Tables" section
```

Not run: the `:append`-order comparison over the full grid (v1 reproduction at every point) — the
results doc's per-point `:append` fingerprints (§2) and v1's own committed numbers already give the
before/after comparison this document's §4/§6 draw on; a full second 8-set curve run under
`:append` would reproduce v1's original run (already committed) rather than add new information,
so it was not spent from this run's time budget.
