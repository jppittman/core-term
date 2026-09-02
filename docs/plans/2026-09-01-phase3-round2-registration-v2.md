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
**Status:** DESIGN SKELETON — this document commits the mechanism (interleaved order, seed,
sweep-denominated reporting, the §7.1 overhead measurement) and the statistical machinery carried
over from v1 UNCHANGED. It does **not** yet commit numbers. Every quantity that v1 computed from
its (now-superseded) unguided curves — §4's measured tables, §5's registered constants (Δ1, Δ2,
Y(|R|) per point, the H1 verdict, the overhead thresholds) — is marked **TBD** below and is filled
by a Register run of the interleaved-order harness on the same 400-expression sample, following
v1's exact procedure. Nothing in §5 may be copied from v1: v1's numbers were computed under the
order this document replaces, and are not valid for a different sweep order (a different rule
order can change which application lands at which checkpoint, hence `app_actual`, cost@B, and
every downstream statistic). Until §5 is filled and committed, no guided run at |R| > 62 may use
this document as its registration — the same rule v1 stated for itself.
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
  carries `sweeps_actual` (cumulative completed sweeps at the checkpoint — already computed by the
  anytime runner, just not written before), `evals_actual` (cumulative rule-match ATTEMPTS at the
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
| **Sweep order (the change)** | Rule-major, as before, but the rule VECTOR each sweep iterates is now base+inflation **shuffled together** by a fixed seed for every inflated set, rather than base-prefix-then-inflation-appended. **Seed: `DEFAULT_INTERLEAVE_SEED = 0x2026_0901`** (`pixelflow_search::math::inflate::DEFAULT_INTERLEAVE_SEED`), stated once here and reused by every default-order spec string. The |R| = 62 point is exempt (§0). The append order remains available (`:append` suffix) for any comparison back to v1. |
| Class cap | Unchanged: `config_for_node_count(node_count).max_classes`. |
| Safety ceilings | Unchanged: per-curve wall-clock `300 s × |R|/62`, sweep ceiling `10,000 × |R|/62`; PANIC when they bind. |
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

The `:append` fingerprints for `dup:93/124/186/248` and `comp:93/124` reproduce v1's §2 table
exactly (same code path, unchanged) — confirmed by
`math::inflate::tests::v2_grid_fingerprints_are_pinned`. `comp:186`/`comp:248`'s `:append`
fingerprints are newly computed here: v1 never finished writing curve rows for these two points
(v1 §2.2 — the process producing the committed CSV did not survive), but the rule sets themselves
were always fully specified and cheap to fingerprint (no curve run needed for a fingerprint); they
are included for completeness and because the interleaved-order Register run below intends to
realize the full 9-point grid this time.

**Mode (iii) is out of scope for this document.** Mode (iii)'s harness
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

## 4. Measured interleaved-order curves — TBD

**Not yet run.** This section is filled by the Register run: `phase3_round2_unguided_curves` over
the full 9-set default-order grid (`base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,comp:186,
comp:248` — the same default strings as v1's invocation; the interleaved order is now what those
strings mean, per §0), on the same 400-expression sample, writing the same CSV shape as v1 plus
the three new columns (§0). Table shapes to fill, mirroring v1 §4 exactly (absolute cost, `visible
@B`/`visible@200`/`first visible`, sweep-denominated `app_actual` alongside applications,
`evals_actual`-derived matches-per-application) plus v1 §4's regret/truncation-loss/closure-gain
table — **TBD**.

The expected qualitative difference from v1, stated in advance so it is a prediction and not a
post-hoc read: under interleaving, `visible@100`/`visible@200` should be nonzero even in mode (i)
(duplicates), because a copy can now be swept before the useful rules that dominate v1's
first-sweep budget — this is the observation the mechanism exists to make visible. Whether the
direction and magnitude clear Δ1 (§5.3, TBD) is exactly what the Register run measures; it is not
assumed here.

## 5. Registered constants — TBD from the interleaved-order Register run

Everything in this section is **TBD**. Per the binding rule inherited from v1 (§6 of the design,
carried forward), Register may fix these ONLY from unguided data on the interleaved-order harness,
committed before any guided run at |R| > 62 under this order, and — once committed — may not be
revised except by append or by a further superseding registration.

### 5.1 Inherited, not re-derived
B = 100 / 200 (classical); Y's formula `Y = 1 − (1 + L/2)/(1 + L)`; ε = 0.005; the 0.02 floor on
Δ2; the pool seed `0x5EED2` (which compositions are IN the prefix — unchanged, orthogonal to
sweep order, §0); the bootstrap seed `42`; the oracle seed `0xC0FF_EE42`; the sample; the grid;
the reference convention (§3); the H2 statistics (design §1.3). **New this document:** the
interleave seed `0x2026_0901` (§1), now also part of "what is fixed, not re-derived" once §4/§5
are filled.

### 5.2 Y(|R|) per mode and B — TBD

| Mode | \|R\| | L@100 | **Y@100** | L@200 | **Y@200** |
|---|---:|---:|---:|---:|---:|
| (i) | 93, 124, 186, 248 | TBD | TBD | TBD | TBD |
| (ii) | 93, 124, 186, 248 | TBD | TBD | TBD | TBD |

### 5.3 Δ1 — H1's minimum effect (from |R| = 62 only) — TBD

95% bootstrap CI of the median unguided regret at |R| = 62 on the 188 classical expressions
(10,000 resamples, seed 42, order-statistic 2.5/97.5 percentiles) — same procedure as v1 §5.3, run
fresh because it is measured, not inherited (the |R| = 62 point's rule set is unchanged, but a
fresh Register run is the discipline v1 itself used rather than reusing a prior run's number).

| B | median U(62) | CI | **Δ1** |
|---:|---:|---|---:|
| 100 | TBD | TBD | TBD |
| 200 | TBD | TBD | TBD |

### 5.4 Δ2 — H2's minimum effect, per mode — TBD

`Δ2 = max(0.02, Y(|R|max) − Y(62))`, computed once §5.2 exists.

### 5.5 The H2 slope-ratio threshold — TBD

Same derivation rule as v1 §5.5 (the ratio is evidence only when the 95% bootstrap CI of
`slope_U` excludes zero and `slope_U > 0`); recomputed from this document's own `slope_U`, not
copied from v1's (v1's `slope_U` was measured under append order and is not informative about the
interleaved order's slope).

### 5.6 Per-candidate Guide overhead — the §7.1 flatness check — TBD

Unguided half (§0.2): median and quartiles of `evals_actual / app_actual` at B, per (mode, |R|),
against the same "≤ 2× its value at |R| = 62" threshold design §7.1 pre-commits. Guided half
remains out of scope pending `GuidedEpisodeStats` (unchanged from v1 §9 point 3).

## 6. H1 verdict on this grid — TBD, recorded only after the interleaved-order Register run exists

Per design §1.2 and v1's own discipline, H1 is entirely an unguided measurement, computed here
under the interleaved order, before any guided run at |R| > 62 exists under this registration. The
table shape is v1 §6's (mode, grid, B, U(|R|), Spearman ρ, U(max) − U(62), Δ1, direction, effect,
verdict) — **TBD**, filled together with §4/§5, never partially.

## 7. What remains testable for H2 under this Register — carried forward, TBD numbers

Structure carried verbatim from v1 §7 (H2 part 1/part 3 fully testable at every point; part 2
requires the Guide's advantage to grow, live in modes (ii)/(iii) and impossible-by-construction in
mode (i) exactly as v1 argued — sweep order does not change that argument, only the numbers it is
evaluated against). Thresholds (`1 − Y` at each |R|) are **TBD** pending §5.2.

## 8. Gates (structure fixed, numbers TBD)

**Accept gate (per mode):** unchanged from v1 §8 — H1 AND H2 hold on DEV classical (n = 334) at
B = 100, full per-expression distributions reported; FINAL required for publication.

**Kill gate (per mode):** unchanged from v1 §8 — H2 part 3 failing at any |R| point on DEV, after
one clean re-mint/re-train, stops that mode.

**Honest fallback:** unchanged in shape — if H1 fails here too, §6 is the deliverable (capacity
finding plus absolute-cost/closure-gain columns), exactly as v1's fallback fired. Whether it fires
again under interleaving is precisely what §4/§6 will show; it is not assumed.

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

## 10. Proof that no guided run at |R| > 62 exists under this registration at this commit

- `git grep` for every interleaved-order fingerprint in §2 (`83e610e33e782a68`, `b207aa331bb625ab`,
  `3a00c565900b48e6`, `43c43d764ef7f76b`, `904ceec9b110e89e`, `a7600e5942f0baa5`,
  `9e9bf3a4458a3045`, `b89d841eada63c13`) outside `pixelflow-search/src/math/inflate.rs` (the test
  that pins them) returns nothing — no result file, label dataset, or checkpoint under
  `docs/results/` mentions any of them yet, because none has been produced.
- `git grep -l -E 'RuleOrder::Interleave|DEFAULT_INTERLEAVE_SEED'` outside `inflate.rs` matches
  only `phase3_round2_unguided_curves.rs`'s doc comment (referring to the default spec string
  meaning, not constructing a `RuleOrder` value directly) and this document.
- No Guide checkpoint or label artifact anywhere in `docs/results/` carries any fingerprint from
  §2's interleave column — the only Guide artifacts on the branch remain Round 1's, at |R| = 62,
  predating this document (same three files v1 §10 named).
- `phase3_round2_unguided_curves` imports no `nnue::guide` symbol (unchanged from v1; this
  commit's diff to that file is the three new columns and the probe function, nothing
  guided-path-shaped).

## 11. Results appended against the gates

(Append-only, as in v1. First entry: the interleaved-order Register run's full §4/§5/§6 numbers,
committed together, before any guided run at |R| > 62 under this registration.)

**Note on the stats script (honest, not silent):** `round2_register_stats.py` asserts its input
CSV's header EXACTLY (`die()` on any mismatch, by design — no silent misread of a changed schema).
The three new columns (§0) mean the CSV this document's harness now writes does not match that
assertion as-is. This commit does not update the script — filling §4/§5/§6 from the new CSV is
Register-phase work, and updating `HEADER`/the column reads to match is part of that work, not
assumed done here. Running the §12 command against today's script will fail loud at the header
check, which is the correct behavior for an unreconciled schema change, not a bug to route around.

## 12. Reproduction

```bash
# unguided curves, modes (i)/(ii), interleaved order (the new default) — full 9-set grid
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2v2-unguided-vs-rulecount-modes-i-ii.csv \
    --out-json docs/results/2026-09-01-round2v2-unguided-vs-rulecount-modes-i-ii.json
# same grid, v1's append order, for a direct before/after comparison at identical rule content
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2v2-unguided-vs-rulecount-modes-i-ii-append.csv \
    --out-json docs/results/2026-09-01-round2v2-unguided-vs-rulecount-modes-i-ii-append.json \
    --rule-sets base,dup:93:append,dup:124:append,dup:186:append,dup:248:append,comp:93:append,comp:124:append,comp:186:append,comp:248:append
# fingerprint + order-sensitivity + 62-never-reordered guarantees
cargo test -p pixelflow-search math::inflate -- --nocapture
# every registered number (once the CSV above exists) — v1's script, same schema plus the new
# sweeps_actual/evals_actual/apps_per_sweep columns; --expect/--modes unchanged from v1's invocation
python3 pixelflow-pipeline/scripts/round2_register_stats.py \
    --csv docs/results/2026-09-01-round2v2-unguided-vs-rulecount-modes-i-ii.csv \
    --expect base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,comp:186,comp:248 \
    --modes 'i=base,dup:93,dup:124,dup:186,dup:248;ii=base,comp:93,comp:124,comp:186,comp:248' \
    --out-json docs/results/2026-09-01-phase3-round2-registration-v2.json \
    --out-md docs/results/2026-09-01-phase3-round2-registration-v2-tables.md
```
