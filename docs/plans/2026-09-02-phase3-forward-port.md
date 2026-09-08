# Phase 3 forward-port: inventory and map

**Status:** Ported. Phase A (the `pixelflow-search` core), Phase B
(`pixelflow-pipeline`'s harnesses) and the docs are all in; A10 is
deliberately left out, with its reason — see §6.
**Base:** `origin/main` @ `40c96ece` (#1118).
**Sources (READ-ONLY):** `claude/phase3-guide` (#1084), `claude/phase3-domain-shift`
(#1091), `claude/phase3-label-constfold` (#1095), `claude/phase3-r2g` (#1096),
`claude/phase3-round2` (#1088), `claude/phase3-context` (unpushed).

All six branch off `2e82cdc2`. Main has since landed 13 PRs that rewrote exactly
the files they touch (`graph.rs` +883, `extract.rs` +633, `runtime.rs` +1483,
`saturate.rs` +139, `provenance.rs` +101, `extraction.rs` **deleted**, and four
new modules: `optimizer.rs`, `rules.rs`, `rule_order.rs`, `telemetry.rs`).
`git rebase origin/main` conflicts on the first of 40 commits, which is why this
is a forward-port.

---

## 0. Shape of the stack

The six branches are **not** six independent diffs. They share a 23-commit trunk
(the Round-1 Guide surface, `2e82cdc2..3db50bd7`), then fork:

```
2e82cdc2 (branch base)
  └── TRUNK (23c) ── Round 1: anytime curves, candidate features, strict labels,
      │              guided saturation, linear Guide, at-budget eval
      ├── phase3-guide      (+3c)  review fixes: ENodeId-keyed targets, train/deploy candidate parity
      ├── phase3-context    (+3c)  WIP snapshot: candidate-context design, coverage table, extract.rs factoring
      ├── phase3-round2    (+15c)  rule-count scaling: |R| inflation, new-rule batch, v2/v3 registrations
      └── ROUND-1B (7c) ── domain shift: sh/bezier OOD corpora, DEV stratification
          ├── phase3-domain-shift (+2c) nine review findings + quiescence test
          ├── phase3-label-constfold (+2c) strict-label constant-output blind spot
          └── phase3-r2g       (+10c) return-to-go labels, counterfactual replay, ordering-diversity guides
```

**Two branches diverged on the same file.** `phase3-guide` and
`phase3-domain-shift` each refined `egraph/saturate.rs` after the trunk, in
different directions (~305 diff lines between the two resulting files).
`phase3-guide`'s version is the newer of the two on the overlapping hunks (it
names a matched node by `ENodeId` instead of `node_idx`, and re-reads the clock
before declaring quiescence). The forward-port must **merge**, not pick: the
domain-shift fixes to `gen_sh_corpus`, `guide_linear`, and `phase3_at_budget_eval`
are not in `phase3-guide` at all.

Volumes, against the branch base:

| branch | code (`pixelflow-search` + `pixelflow-pipeline` + `scripts`) | docs/results |
|---|---:|---:|
| phase3-guide | 11,005 | 17,562 |
| phase3-domain-shift | 13,569 | 28,858 |
| phase3-label-constfold | 13,738 | 28,759 |
| phase3-r2g | 20,835 | 46,891 |
| phase3-round2 | 15,334 | 292,590 |
| phase3-context | 10,818 | 18,326 |
| **union (est.)** | **~29,500** | **~300,000** |

The docs number is dominated by round-2's raw CSV sweeps (224k lines in four
files). Those port byte-for-byte and cost no review.

---

## 1. THE MAP

Legend: **(a) DELETE** — main provides it; the port removes the branch's copy.
**(b) SEAM** — slots onto a named main seam, with the change stated.
**(c) NEW** — genuinely new surface main has no equivalent for.

### 1.1 `pixelflow-search` — saturation and budget

| item | src | verdict | notes |
|---|---|---|---|
| `SaturationStop` enum (`graph.rs`, 18 ln) | trunk | **(a) DELETE** | Main's `SaturationStop` is a strict superset (adds `ApplicationBudget`). Re-export from main. |
| `ScanStop::ApplicationBudget` plumbing | trunk | **(a) DELETE** | Main has `ScanStop::{Completed,ClassCap,Deadline,ApplicationBudget}`. |
| `AppBudgetSaturationStats` (11 ln) | trunk | **(a) DELETE** | Main's `SaturationStats { iterations, total_unions, stop }` + unconditional `EGraph::application_count() -> u64`. |
| `EGraph::saturate_until_applications(max_apps, iters, classes, timeout)` (80 ln) | trunk | **(a) DELETE → (b) SEAM** | ⚠️ **Main has no such function.** Replaced by `EGraph::saturate_budgeted(iters, classes, Some(apps))` or `Optimizer::budget(Budget::Applications(n))`. **Two semantic changes — see §2.1.** |
| `runtime::saturate_for_production` + `ProductionSaturation` (95 ln) | trunk | **(a) DELETE** | This existed to stop `production_saturation_probe` from being a drifting copy of the production step. `Optimizer::production()` *is* that seam now. |
| `runtime::production_saturation_probe` | trunk | **(b) SEAM** | Rewrite as `Optimizer::production().run(&mut eg, root, n)` reading `OptimizerStats`. Reads `extraction.total_cost` today → **must read `dag_cost`** (#1117). |
| `env_extraction_policy` / `ExtractionPolicy` | trunk (via runtime.rs) | **(a) DELETE** | `extraction.rs` deleted on main; `CostModel::latency_prior()` is `Optimizer::production()`'s default. |
| `anytime.rs` (366 + 47 ln): `APP_CHECKPOINT_GRID`, `AnytimeCurve`, `AnytimeStepper`, `run_anytime_curve{,_with}` | trunk + guide | **(b) SEAM, shrinks** | The stepper becomes repeated `Optimizer::budget(Budget::Applications(delta)).run(..)` on one `EGraph`. The per-curve wall-clock safety ceiling that panics on `SaturationStop::Timeout` becomes `Optimizer::hard_ceiling(d)`, which panics by construction — so the guide branch's "re-read the clock before declaring `Quiesced`" fix (`e9517bd7`+`49632f96` hunk) becomes **unnecessary and is deleted**: `saturate_budgeted` takes no clock and can never report `Timeout`. |
| Wall-clock `hard_timeout` fields threaded through the harnesses | trunk, round1b | **(b) SEAM** | `SaturationConfig::hard_timeout` is demoted to a panicking `safety_ceiling` (#1118). `phase3_at_budget_eval`'s `hard_timeout_ms` JSON field is now meaningless as a *stop reason*; keep it only as a reported ceiling, never as a metric. |
| `EGraph::saturate_with_limits` callers (`gen_strict_labels`, `guide_headroom`, `tightened_labeler_rank`, `guide_scope_saturation_delta`) | trunk | **no change** | Main kept `saturate_with_limits(iters, classes, timeout)`. Compiles as-is. |

### 1.2 `pixelflow-search` — the Guide

| item | src | verdict | notes |
|---|---|---|---|
| `SaturationGuide` trait (v2 candidate-local contract) | trunk | **(c) NEW** | See §3.1 — **this is the API gap.** |
| `CandidateSummary` | trunk | **(c) NEW** | Carries `rule_idx: usize` → must become `RuleId` (§2.2). |
| `egraph/candidate.rs` (446 ln): `CandidateKey`, `ClassContentKey`, `CandidateFeatures`, `Firing`, `REGISTERED_PRIMARY_BUDGET_APPLICATIONS = 100` | trunk | **(c) NEW** | Nothing on main resembles it. `CandidateKey.rule_idx` → `RuleId`. The constant is a **frozen registration** — port byte-identical. |
| `saturate::GuidedSaturation` + `saturate_guided_until_applications` (599 + 158/171 ln) | trunk + guide + domain-shift | **(c) NEW, merged** | Two divergent refinements (§0). Re-express its budget loop on `Budget::Applications`; keep its dedup/scoring/ordering logic. |
| `nnue/guide/mod.rs` v1→v2 contract revision (311 ln) | trunk | **(c) NEW** | Deletes `GraphSummary`/`RuleCandidate` wrappers, keeps `accumulator.rs` (segregated roadmap seam). Main still has the v1 shape. |
| `nnue/guide/scoring.rs` candidate tower (360 ln) | trunk | **(c) NEW** | Additive to main's `scoring.rs`. |
| `nnue/guide/linear.rs` (473 + 144 + 375 ln): `LinearCandidateGuide`, `PerRuleRateGuide` | trunk + guide + r2g | **(c) NEW** | Checkpoint keys `w_rule` **positionally** — the G5 violation (§2.2). |
| `nnue/guide/diversity.rs` (224 ln): ordering-diversity guides | r2g | **(c) NEW** | 12 ordering policies for trajectory minting. |
| `EGraph::apply_single_rule(rule_idx, class_id, node_idx)` → `(.., tag: ENodeId)` | guide | **(b) SEAM, small** | Main keeps `node_idx` but already stores `EClass::tags: Vec<ENodeId>` unconditionally and exposes `tags()` / `node_for_tag()`. The fix (+ `RewriteTarget.node_idx` → `tag`) is ~40 lines and works with `provenance-journal` **off**. |

### 1.3 `pixelflow-search` — provenance and labels

| item | src | verdict | notes |
|---|---|---|---|
| `UnionEvent.application_id: Option<ApplicationId>` | trunk | **(c) NEW** | Additive; gate behind `provenance-journal`. |
| `derivation_ancestors_tight` / `_tight_from` | trunk + guide | **(c) NEW** | Additive; gate. |
| `EpisodeLabels::compute_tight` / `compute_strict` (+184 ln) | trunk | **(c) NEW** | Main's `labeler.rs` is unchanged from the branch base but is now `#[cfg(feature = "provenance-journal")]`. Slots in under the gate. |
| labeler output-class recount (+354 ln, no new pub items) | label-constfold | **(c) NEW** | Internal + tests. |
| `provenance().application_count()` | trunk, everywhere | **(b) SEAM** | Renamed to `recorded_count()` on main; the **budget denominator** is now unconditional `EGraph::application_count() -> u64`. Every harness call site must pick the right one — the counter is per-action-commit including no-ops on both, so the x-axis is preserved. |
| `ApplicationRecord` gains `rule: Option<RuleId>`, `minted: Range<u64>`, `unions: usize`, `changed()`, `minted_count()` | **main (#1118)** | **opportunity** | The strict/tight bounds re-derive "did this application change anything" by hand. Main now records it. Possible net shrink in `labeler.rs`; verify before assuming. |
| Harness use of the journal at all | all | **(b) SEAM** | `provenance-journal` is default-OFF (#1118). `pixelflow-pipeline` already enables it unconditionally; `pixelflow-search`'s own tests/examples do too via the self-dependency. **New harness code must not leak it into `pixelflow-compiler`/`-codegen`/`-runtime`/`core-term`** — `scripts/check-provenance-journal-scope.sh` is the gate. |

### 1.4 `pixelflow-search` — Round 2 rule scaling

| item | src | verdict | notes |
|---|---|---|---|
| `RewriteAction` +11 harness-only variants; `TemplateArena` | round2 | **(c) NEW, question it** | Grows a production enum by 11 variants for a harness. `egraph/template.rs` (577 ln) executes them. Consider a separate harness `Rewrite` impl over `Instantiate` alone. |
| `math/inflate.rs` (1,342 ln), `math/oracle.rs` (447), `math/round2_rules.rs` (1,659) | round2 | **(c) NEW** | The `\|R\|`-inflation machinery + 20-ish genuinely-new rules + a numeric oracle. |
| "three v3 base-only `RuleOrder` variants" (`f22eddb4`) | round2 | **(a) DELETE** | ⚠️ **Main already ships `egraph/rule_order.rs`** with `RuleOrder::{Production, NumericFirst, Shuffled(u64)}` and `NUMERIC_FIRST_ORDER: [usize; 62]` — *derived from round-2 v3's own §6b finding*. Main's version is the survivor. |
| Ad-hoc `EGraph::with_rules(inflated_vec)` | round2 | **(b) SEAM** | `RuleSet::new(vec)` + `Optimizer::rules(set)`; `fingerprint()` then names the arm. |
| `EGraphBatch::total_evals()` | round2 | **(c) NEW, trivial** | |
| Round-2 v2/v3 registrations + results (4 CSVs, 224k ln) | round2 | **port as-is** | **Frozen artifacts.** Re-run banners included. |

### 1.5 `pixelflow-search` — Return-to-go / counterfactual replay

| item | src | verdict | notes |
|---|---|---|---|
| `MaskScope`, `ApplicationMask`, `EGraph::saturate_until_applications_observed`, `last_replay_mask_skips` (435 ln) | r2g | **(c) NEW** | Hooks `apply_action_from_rule` at the exact line main increments `self.applications` — clean insertion point, works with the journal off. **No `Optimizer` seam exists for it** (§3.2). |

### 1.6 `pixelflow-search` — extraction

| item | src | verdict | notes |
|---|---|---|---|
| `extract::class_costs_and_choices` (factor phase 1 out of `extract_dag`) | context | **(b) SEAM, redo** | Main rewrote `extract.rs` (+633: `extract_dag_scoped`, `repair_choices_well_founded`, `cost_of_choices`, `LatticeShape`, `dag_cost`). The factoring is still the right shape; the code must be re-derived, not re-applied. `phase3-context` is an explicit WIP snapshot ("pre-rebase") — treat it as a design, not a patch. |
| `ExtractedDAG::total_cost` reads | trunk, r2g | **(b) SEAM — the big one** | See §2.3. |

### 1.7 `pixelflow-pipeline`

| item | src | verdict | notes |
|---|---|---|---|
| `phase3_unguided_baseline` (514 + 38 + 20 ln) | trunk, guide, domain-shift | **(c) NEW** | |
| `tightened_labeler_rank` (711 + 59) | trunk, guide | **(c) NEW** | |
| `gen_strict_labels` (774 + 28) | trunk, guide | **(c) NEW** | |
| `train_guide` (1,429 + 221 + 14) + `training/guide_linear.rs` (343 + 211 + 31) | trunk, guide, domain-shift | **(c) NEW** | Checkpoint writer — G5 site. |
| `skew_test_linear_guide` (295 + 379 + 4) | trunk, r2g, domain-shift | **(c) NEW** | Mandatory train/deploy skew test. |
| `eval_control_guides` (235 + 69) | trunk, guide | **(c) NEW** | |
| `phase3_at_budget_eval` (1,820 + 948 + 205 + 146 + 97) | trunk, round1b, guide, r2g, domain-shift | **(c) NEW, four-way merge** | The single most-contended file: five tails touch it. |
| `gen_sh_corpus` (319 + 45), `gen_bezier_corpus` (991), `training/sh_family.rs` (670 + 6) | round1b, domain-shift | **(c) NEW** | |
| `training/split.rs` `dev_families` (+193) | round1b | **(b) SEAM, additive** | ⚠️ Main **already has** `training/split.rs` (1,173 ln) and `corpus_split.toml`. Port the +193 delta only, not the file. |
| `strict_label_output_class_recount` (503) | label-constfold | **(c) NEW** | |
| `gen_r2g_trajectories` (1,775), `counterfactual_credit` (1,748), `train_guide_r2g` (1,430), `r2g_spread_vs_budget` (420), `training/r2g.rs` (941), `scripts/r2g_ladder_join.py` (342) | r2g | **(c) NEW** | |
| `phase3_round2_unguided_curves` (455), `phase3_round2_new_rules` (644), two `round2_register_stats*.py` (954) | round2 | **(c) NEW** | |
| `guide_coverage_table` (963) | context | **(c) NEW, WIP** | |
| `guide_headroom_tight` removal (`21d5a4db`) | trunk | **(a) DELETE — already done** | Main has no such binary. Drop the commit. |
| `docs/results/2026-09-01-train-guide-report.md` | trunk | **(b) 8-line delta only** | ⚠️ **Main already has this file.** Port only the guide branch's re-run banner. |

---

## 2. The real work — flagged items

### 2.1 `saturate_until_applications` is gone, and its replacement stops at a different place

Main's `saturate_bounded` sets `application_cap = current + n` and enforces it
**deep inside the scan**, the moment an application is about to commit
(`ScanStop::ApplicationBudget`). The Phase 3 anytime curve was specified to
cross checkpoints at **rule-sweep granularity**:

> Checkpoint targets are crossed at rule-sweep granularity: the sample at target
> `B` is taken at the first between-sweeps point where the cumulative count is
> `>= B`, and `AnytimeCheckpoint::app_actual` records the exact count.
> — `egraph/anytime.rs`, trunk

On main, `app_actual == app_target` exactly; there is no overshoot to record.
**This changes the instrument.** It is a strictly better instrument — the
overshoot was an artifact — but it means a re-run does not reproduce the
registered curves even if nothing else changed. Two consequences:

1. `AnytimeCheckpoint::app_actual` stays in the schema (a curve that always
   equals its target is still worth asserting), but it can no longer explain a
   discrepancy against a Round-1 artifact.
2. Every results doc's re-run banner must name this, not just #1117.

Second change: `saturate_budgeted` takes **no clock**, so `SaturationStop::Timeout`
is unreachable from it. The anytime runner's fail-loud ceiling moves to
`Optimizer::hard_ceiling(d)`, which panics — same semantics, one fewer stop
reason to handle, and it deletes the guide branch's clock-recheck fix.

Also note `Budget::Production` now carries `max_applications` (20,000 / 80,000 /
200,000 per tier). Any ported code that spells its own "production budget" must
use `Budget::Production`, never restate the caps.

### 2.2 Positional `rule_idx` — 18 files, and two of them are checkpoints

`RuleId` / `RuleSet` / `Fingerprint` landed on main (#1108). `rule_idx: usize`
survives in the branches at:

- **`nnue/guide/linear.rs`** — `LinearCandidateGuide`'s `w_rule` vector is
  indexed by `rule_idx`, with a length check against the *live* rule table. The
  checkpoint already does the right thing for **ops** (`op_names` → `op_index`
  with a loud disagreement check) and the wrong thing for **rules**. A
  same-length reorder of `all_rules()` repoints every weight and nothing is the
  wrong length. Fix: key `w_rule` by `rule_label`, and write
  `RuleSet::fingerprint()` into the checkpoint; the loader refuses a mismatch.
- **`PerRuleRateGuide::from_train_guide_report`** — reads `rule_idx` out of a
  JSON report. Same fix.
- `egraph/candidate.rs::CandidateKey.rule_idx`, `nnue::guide::CandidateSummary.rule_idx`.
- Per-rule firing histograms and per-rule base-rate tables in
  `phase3_at_budget_eval`, `train_guide`, `gen_strict_labels`,
  `tightened_labeler_rank`, `eval_control_guides`, `skew_test_linear_guide`,
  `counterfactual_credit`, `gen_r2g_trajectories`, `train_guide_r2g`,
  `training/{guide_linear,r2g}.rs`, `strict_label_output_class_recount`.

`RuleSet::id_of(idx)` / `index_of(id)` / `label_of(idx)` are the translation
seam; `EGraph::rule_id(idx)` gives it from a graph. Main's `UnionEvent` still
carries `rule_idx: Option<usize>`, so the boundary is `ApplicationRecord.rule`,
which is already a `RuleId`.

**This is the largest mechanical piece of the port and the one that must not be
deferred**: every JSON artifact the port re-mints is keyed wrong until it lands.

### 2.3 Every Phase 3 cost number was measured on TREE cost

`ExtractedDAG::total_cost` is the **tree** cost — every child summed at every
use, sharing never priced. `dag_cost` (#1117) is what the emitted kernel pays.
On `shader:julia_set` the two are **1.4e7 and 716**.

`total_cost` is read in the branches at:
`egraph/anytime.rs` (the anytime curve's y-axis — i.e. **the metric the whole
program is registered on**), `runtime.rs` (`production_saturation_probe`),
`graph.rs` (r2g), `counterfactual_credit.rs`, `gen_r2g_trajectories.rs`.

Consequences:

- Every ported harness reads `dag_cost`. No exceptions.
- The registered `B` and `Y` in `2026-09-01-phase3-registration.md`, the round-1b
  domain-shift deltas, the round-2 regret-vs-|R| tables, and the R2G return
  labels were all computed on tree cost. **They port as-is and are not
  re-derivable from a `dag_cost` re-run.** Their re-run banners must say so
  explicitly, alongside §2.1's granularity change.
- The R2G *return* label is a difference of costs. Whether the strict-vs-return
  rank correlation result survives the unit change is an open question the port
  does not answer.

### 2.4 Smaller flags

- `provenance().application_count()` → `recorded_count()` (journal) vs
  `EGraph::application_count()` (budget denominator, unconditional).
- `egraph/extraction.rs` deleted → `env_extraction_policy` gone.
- `HARD_CLASS_LIMIT` is a ceiling now, not a sentinel (#1107) — harnesses that
  passed `usize::MAX` classes get `HARD_CLASS_LIMIT` and a truthful `ClassCap`.
- `apply_single_rule` **panics** at `HARD_CLASS_LIMIT` on main.
- `PIXELFLOW_SATURATION_CEILING_MS` exists and **panics on an unparsable value**;
  harnesses that set it must set it correctly.

---

## 3. What CANNOT be expressed on the new seams — the API gaps

### 3.1 GAP 1 (blocking): `Optimizer` has no `.guide()`

`docs/plans/2026-09-02-optimizer-api.md` §G1 specifies:

> `guide: Option<Box<dyn SaturationGuide>>` … `.guide(Some(LinearCandidateGuide::load(p)?))`

and §2.3 quotes the trait verbatim from `claude/phase3-guide`. **#1108 did not
ship it.** Main's `Optimizer` has `rules`, `budget`, `cost`, `for_lattice`,
`rerank`, `observe`, `hard_ceiling`, `no_ceiling`, `fingerprint`, `rule_set`,
`egraph`, `run` — and no guide. `pixelflow-search` contains no `SaturationGuide`
trait, no `CandidateSummary`, and no guided rewrite loop anywhere.

So the premise that motivated this port — "the research surface was designed
drop-in; re-applying it onto the seams should be *smaller* than the old diff" —
**does not hold for the single most important item.** The Guide is not a
`Some(..)` on an existing field; it is the field, the trait, the candidate
module, and the loop, all new (~1,500 lines: `candidate.rs` 446, `guide/mod.rs`
v2 311, `guide/scoring.rs` tower 360, `GuidedSaturation` ~350, the field + `run`
branch ~50).

That is not a reason to stop — the surface is exactly what the doc specified and
it lands additively — but JP should know that G1 was **specified and then left
out of #1108**, and that the "drop-in" framing of this port came from the design
doc, not from main.

Design note for the port: `Optimizer::run` does saturate-then-extract in one
call. Guided saturation needs to interleave scoring with sweeps. The cleanest
shape on the existing seams is `Optimizer::run` dispatching to a guided loop when
`guide.is_some()`, with `Budget::Applications` as the stop — no second entry
point, which is the whole point of #1108.

### 3.2 GAP 2 (small): no seam for the counterfactual replay mask

`ApplicationMask` / `MaskScope` withhold a named application ordinal (and,
under `AllMatchingCandidate`, every later application sharing its `CandidateKey`)
so the harness can measure `Δ_a = R(τ\a,B) − R(τ,B)`. It is a *policy* in exactly
the sense `Reranker` and `Observer` are, and it belongs as
`Optimizer::mask(Option<ApplicationMask>)`. Today it can only exist as an
`EGraph`-level harness method (`saturate_until_applications_observed`), which
reintroduces a second saturation entry point — precisely what #1085/#1108 removed.
Small, but it should be a field, not a parallel loop.

### 3.3 Not a gap, but worth stating

The anytime **stepper** *is* expressible: repeated
`Optimizer::budget(Budget::Applications(delta)).run(&mut eg, root, n)` on one
`EGraph`, because `Budget::Applications(n)` resolves to a **delta from the
current count**, not an absolute target, and `EGraph::application_count()` is
cumulative and unconditional. The port should say this out loud in `anytime.rs`,
because the delta-vs-absolute reading is easy to get backwards and would silently
produce a curve whose x-axis is a prefix sum of itself.

---

## 4. Size estimate

| area | old diff | ported (est.) | Δ |
|---|---:|---:|---:|
| `pixelflow-search` deletes (§1.1) | — | — | **−430** |
| `pixelflow-search` new seam work (`.guide()`, mask field, Fingerprint plumbing) | — | — | **+250** |
| `pixelflow-search` remainder | ~9,600 | ~9,600 | 0 |
| `pixelflow-pipeline` RuleId migration (§2.2) | — | — | **+350** |
| `pixelflow-pipeline` `dag_cost` swap (§2.3), `Budget::Applications` (§2.1) | — | — | **−20** |
| `pixelflow-pipeline` remainder | ~19,000 | ~19,000 | 0 |
| **total code** | **~29,500** | **~29,650** | **+150 (+0.5%)** |

**Verdict: a wash — not larger, not the shrink the premise predicted.** The
deletes are real (§1.1 collapses the whole parallel budget/stop machinery onto
main's) but they are almost exactly cancelled by the RuleId migration and by the
Guide seam main does not have. The port proceeds; the premise's "smaller" claim
was resting on G1, and G1 is not there.

---

## 5. Ordered work list

### Phase A — `pixelflow-search` core (must land first, one PR per bullet where possible)

1. **Port `egraph/candidate.rs`** as-is except `CandidateKey.rule_idx: RuleId`.
   Freeze `REGISTERED_PRIMARY_BUDGET_APPLICATIONS = 100` byte-identical.
2. **Port the `SaturationGuide` v2 contract** (`guide/mod.rs` revision,
   `CandidateSummary` with `RuleId`, `guide/scoring.rs` tower). No loop yet.
3. **Add `Optimizer::guide(Option<Box<dyn SaturationGuide>>)`** and the guided
   branch in `run`, budgeted by `Budget::Applications`. Merge `phase3-guide`'s
   and `phase3-domain-shift`'s divergent `saturate.rs` refinements; keep
   `phase3-guide`'s `ENodeId`-keyed `RewriteTarget` / `apply_single_rule`; **drop**
   its clock-recheck fix (no clock exists). Pin L4 with the optimizer-api doc's
   proposed test: one arena, N guides, byte-identical extracted arenas.
4. **Port `anytime.rs`** on `Budget::Applications` + `hard_ceiling`. Read
   `dag_cost`. Document the delta-not-absolute budget reading (§3.3) and the
   sweep→mid-scan granularity change (§2.1).
5. **Port `provenance.rs` / `labeler.rs` additions** (`UnionEvent.application_id`,
   `derivation_ancestors_tight{,_from}`, `compute_tight`, `compute_strict`, the
   output-class recount) under `provenance-journal`. First check whether main's
   new `ApplicationRecord::{changed, unions, minted}` lets `compute_strict`
   shrink.
6. **Rewrite `production_saturation_probe`** on `Optimizer::production()`;
   delete `saturate_for_production` / `ProductionSaturation`.
7. **Port `nnue/guide/linear.rs`** with `w_rule` keyed by `rule_label` and a
   `Fingerprint` in the checkpoint that the loader refuses to mismatch. Then
   `guide/diversity.rs`.
8. **Add `Optimizer::mask(Option<ApplicationMask>)`** (§3.2) and port
   `MaskScope`/`ApplicationMask` behind it instead of a second saturation entry.
9. **Round-2 machinery** (`template.rs`, `math/{inflate,oracle,round2_rules}.rs`)
   on `RuleSet::new` + `Optimizer::rules`. **Delete** the v3 `RuleOrder`
   variants — main's `rule_order.rs` supersedes them. Reconsider the 11
   `RewriteAction` variants before committing to them.
10. **`extract::class_costs_and_choices`** re-derived against main's rewritten
    `extract.rs` (from `phase3-context`'s design, not its patch).

**Gate after each:** `cargo build --workspace`, `cargo test --workspace`,
`clippy --workspace --all-targets -D warnings`, `fmt`,
`cargo check -p pixelflow-ir --no-default-features`,
`cargo check -p pixelflow-search --no-default-features`,
`scripts/check-provenance-journal-scope.sh`.
**Plus:** a before/after byte-identical-extraction check on the real kernels —
production leaves `guide`/`mask` `None`, so nothing may move.

### Phase B — `pixelflow-pipeline`

11. `training/split.rs` `dev_families` **delta only** (+193) onto main's file;
    `corpus_split.toml` `families = ["sh","bezier"]` delta.
12. `gen_strict_labels`, `tightened_labeler_rank`, `phase3_unguided_baseline`,
    `eval_control_guides` — with `RuleId` keys and `dag_cost`.
13. `train_guide` + `training/guide_linear.rs` + `skew_test_linear_guide` (merge
    the `phase3-domain-shift` and `phase3-r2g` tails).
14. `phase3_at_budget_eval` — the five-way merge. Do this last among the Round-1
    binaries; it is the most contended file in the stack.
15. Round-1b corpora: `gen_sh_corpus`, `gen_bezier_corpus`, `training/sh_family.rs`.
16. `strict_label_output_class_recount`.
17. R2G: `training/r2g.rs`, `gen_r2g_trajectories`, `counterfactual_credit`,
    `train_guide_r2g`, `r2g_spread_vs_budget`, `scripts/r2g_ladder_join.py`.
18. Round 2: `phase3_round2_unguided_curves`, `phase3_round2_new_rules`, the two
    stats scripts.
19. `guide_coverage_table` (WIP — port last, or leave on the branch).

### Phase C — docs

20. Port every registration **byte-identical**:
    `2026-09-01-phase3-registration.md`,
    `2026-09-01-phase3-round1b-domain-shift-registration.md`, the Round-2
    registration + v2 + v3. Never edit a registered constant.
21. Port every results doc and raw artifact **as-is, including its existing
    re-run banner**. Append — do not rewrite — one new banner noting the two
    instrument changes (§2.1 sweep→mid-scan granularity, §2.3 tree→DAG cost) that
    make the numbers non-reproducible by a post-port re-run.
22. `docs/plans/2026-09-01-guide-candidate-context.md` and
    `2026-09-01-guide-return-to-go.md` port as designs.


---

## 6. Port status (2026-09-02)

### Phase A — landed

| item | commit | notes |
|---|---|---|
| A1 `egraph/candidate.rs` | `feat: candidate-local features, keyed by rule identity` | `CandidateKey.rule: RuleId`, `Firing.rule: RuleId`. `REGISTERED_PRIMARY_BUDGET_APPLICATIONS = 100` byte-identical. Also adds the neutrality harness (below). |
| A2 `SaturationGuide` v2 | `feat: the SaturationGuide v2 contract` | `guide/mod.rs` revision + `guide/scoring.rs` candidate tower. `CandidateSummary.rule: RuleId`. `GraphAccumulator` and `mask_score_all_rules_graph` kept as the segregated whole-graph seam. |
| A3 `Optimizer::guide` + the guided loop | `feat: Optimizer::guide` | `egraph/guided.rs`. Merged from `phase3-guide` (ENodeId-keyed `RewriteTarget`) and `phase3-domain-shift` (all targets under one key attempted; quiescence test). Clock re-check dropped. L4 test included. |
| A4 `egraph/anytime.rs` | `feat: the anytime curve, on Budget::Applications and dag_cost` | Shrank: the `AnytimeStepper` trait, `UnguidedStepper`, `AnytimeStep` and `run_anytime_curve_with` all collapse into one function over `&mut Optimizer`. |
| A5 provenance / labeler | `feat: the strict and tight credit bounds` | `UnionEvent.application_id`, `derivation_ancestors_tight{,_from}`, `compute_tight`, `compute_strict`, all under `provenance-journal`. |
| A6 `production_saturation_probe` | — | **Nothing to port.** Main already deleted `saturate_for_production`/`ProductionSaturation`/`env_extraction_policy`; `runtime.rs`'s telemetry harness is already on `Optimizer::production()`. §1.1's "(a) DELETE" verdict was right and main got there first. |
| A7 `nnue/guide/linear.rs` | `feat: the cold-start linear Guide and its per-rule control arm` | `w_rule` keyed by `RuleId`, gated by `Fingerprint`. **No `serde_json` dependency** — parsing moved to `pixelflow-pipeline`, refusal kept at the constructor. |
| A8 `Optimizer::mask` | `feat: Optimizer::mask` | `MaskScope`/`ApplicationMask`/`last_replay_mask_skips` behind an optimizer field, not `saturate_until_applications_observed`. Checked before the application counter increments, so a skipped ordinal goes to the next candidate. |

### Phase A — the rest

| item | commit | notes |
|---|---|---|
| A9 Round-2 machinery | `feat: Round 2 rule-count scaling` | `egraph/template.rs`, `math/{inflate,oracle,round2_rules}.rs`, on `RuleSet::new` + `Optimizer::rules`. §1.4's two open questions are answered below. |
| A10 `extract::class_costs_and_choices` | — | **Deliberately not landed.** See below. |

#### A9's two open questions, answered

**The 11 harness-only `RewriteAction` variants are gone.** `Instantiate` — an
RHS pattern arena plus its bindings — is already the generic executor a
mechanical composition `A∘B` needs, and every one of the 33 rules in
`math::round2_rules` already spells its RHS exactly once, as the
`rhs_template` its cross-form oracle test reads. A bespoke variant per shape
was a second spelling of that shape with no test on it. `RewriteAction` grows
by **one** variant, `apply_action` by one arm, and all 25 oracle tests pass
unchanged. `DivByLiteral` is the single rule whose RHS constant depends on the
matched literal, so it builds its three-node pattern at match time; its
`rhs_template` still spells the shape, with a representative literal, for the
oracle.

**`RuleOrder::StaticReorder` delegates to main's `egraph::rule_order`,** which
is where v3's §6b finding was adopted; the two 62-entry permutations were
byte-identical, and now there is one. The other two v3 variants do **not**
delegate, and §1.4's "main's version is the survivor" is only true of
`NumericFirst`:

- `inflate::RuleOrder::Shuffled(seed)` is not `rule_order::RuleOrder::Shuffled(seed)`.
  The two draw from different PRNGs (SplitMix64 vs xorshift64\*), so the same
  seed names a different permutation. `inflate`'s shuffles with the same
  generator `Interleave` and `OrderMatchedBase` use, which is the entire point
  of a base-only control **at the same seed** — routing it to main's builder
  would silently redefine the v3 seed-sensitivity arm. Both are documented as
  separate objects at both sites.
- `OrderMatchedBase(seed, total)` has no equivalent on main at all, and is the
  reference `ΔU(p)` is defined against.

Likewise `inflate::rule_set_fingerprint` is **not** replaced by
`RuleSet::fingerprint()`. They digest different things (rule *names* in Vec
order vs `RuleId`s, i.e. `rule_label` including specialization), and
`rule_set_fingerprint`'s values are quoted verbatim in the v2 registration's
grid table — a frozen registered constant. Both are documented at the
definition, and `v2_grid_fingerprints_are_pinned` still passes.

#### A10, and why it is not landed

`extract::class_costs_and_choices` factors phase 1 out of `extract_dag`, and
`phase3-context` is explicit that its consumer — `RoundSnapshot` /
`CandidateContext` — "is a sibling agent's work, landing separately". It has
not landed. `guide_coverage_table`, the only `phase3-context` binary ported
here, is a JSONL analysis tool and never calls it.

Re-deriving the factoring against main's rewritten `extract_dag_scoped` is
straightforward (the phase-1 boundary is still a single bottom-up DP followed
by `repair_choices_well_founded`), but it would land a `pub(crate)` seam with
no caller. It belongs in the PR that brings `RoundSnapshot`, where the shape
of the tables it must return is decided by an actual consumer rather than
guessed from a pre-rebase snapshot.

### Phase B — landed

| item | commit | notes |
|---|---|---|
| B11 `training/split.rs` `dev_families` | `feat: DEV-only out-of-distribution families` | Delta only, onto main's file. |
| B12 `gen_strict_labels`, `tightened_labeler_rank`, `phase3_unguided_baseline`, `eval_control_guides` | `feat: the strict-label dataset, its rank re-measurement, …` | `GuideCheckpoint` moves to `training::guide_linear`; `Fingerprint::from_raw` added so a loader outside the crate can name a vocabulary. |
| B13 `train_guide`, `training/guide_linear.rs`, `skew_test_linear_guide` | `feat: train_guide` + `feat: return-to-go` | The skew test lands with B17, which its `--model return` half depends on. |
| B14 `phase3_at_budget_eval` | `feat: the at-budget ablation ladder, five tails merged` | Both run-config guards kept; `production_saturation_probe` replaced by the harness running `Optimizer::production()` itself. |
| B15 `gen_sh_corpus`, `gen_bezier_corpus`, `training/sh_family.rs` | `feat: the sh and bezier out-of-distribution DEV families` | Unchanged — neither generator touches the e-graph. |
| B16 `strict_label_output_class_recount` | `feat: the strict label's constant-output blind spot` | |
| B17 R2G | `feat: return-to-go — trajectories, counterfactual credit, and the return head` | `LinearReturnGuide` on the shared `LinearWeights`; `ReturnObjective` is an enum, not the checkpoint's string. |
| B18 Round 2 harnesses | `feat: Round 2 rule-count scaling` | `phase3_round2_{unguided_curves,new_rules}` + both stats scripts. |
| B19 `guide_coverage_table` | `feat: guide_coverage_table` | Still a proxy over the current record schema, as its own module doc says. |

### Two seams Phase A did not have, added by Phase B

- **`Optimizer::observe(Some(Box::new(KeepJournal)))`.** `Optimizer` records
  provenance only for an observer (#1118), but four harnesses read the whole
  journal off the graph *after* the run, as a graph, in an order the stream
  cannot give them. `KeepJournal` is the observer that consumes nothing; its
  presence is the request. Without it those harnesses silently minted zero
  labels — which is how it was found.
- **`Optimizer::guided_keys_seen()`.** The guided episode's resolved-key
  count, which `phase3_at_budget_eval` reports per arm and which is not
  derivable from the application count (a key can be scored and then fail to
  fire).

### Docs

- The registrations and the two designs are ported **byte-identical**
  (`md5` checked). No registered constant is edited — including
  `phase3_at_budget_eval`'s `TRIG_RULE_IDX`, which keeps its eleven indices
  and gains a `TRIG_RULE_LABELS` companion plus a startup check that the two
  still name the same rules.
- 110 results artifacts port byte-identical, existing re-run banners intact,
  and each results doc gains **one** new banner naming the two instrument
  changes. `docs/results/2026-09-02-phase3-instrument-changes.md` states them
  once.
- `docs/results/journal.jsonl` gains the 18 records these runs appended, in
  timestamp order.

### The neutrality proof


`pixelflow-search/src/runtime.rs`'s `production_equivalence::production_extraction_digest`
(ignored; `PIXELFLOW_EQUIV_DIR`/`PIXELFLOW_EQUIV_OUT`) replays the #1110
dumpers' arenas through `optimize_runtime_arena_uncached` — the production
entry point — and digests the extracted term. Run over
`/private/tmp/classcap_corpus`: **206 kernels** (95 glyphs at each of two
densities, three cell grids, twelve shaders, psychedelic), **0 bailouts**,
**diverged = 0** at A1, A3, A8 and A7 against the pre-port baseline, and
again after **A9 and the whole of Phase B** (the digest is byte-identical to
the pre-A9 baseline over all 206).

That is the strong form of the claim: L4 says a lever cannot change what the
extracted term *means*; this says it did not change the *term*. Every lever
added here is an `Option` that `Optimizer::production()` leaves `None`.

### One design note, stated because it is a deviation

The task framing asked for the guided path to "reuse `saturate_bounded`".
It does not, and cannot: `saturate_bounded` is a rule-then-class sweep and the
guided path is enumerate-score-order-apply. Those are different traversals by
construction — the difference *is* the lever — so folding one into the other
would either change the unguided path (breaking the diverged=0 result above)
or amount to a branch at the top of one function.

What is genuinely shared, and what the "one loop" rule is actually about, is
kept: there is **one public entry point** (`Optimizer::run`); both traversals
resolve their limits through the same `Budget`/`Limits`; both report the same
`SaturationStats` and the same `SaturationStop` vocabulary; both are clockless;
and every mutation the guided path makes goes through
`EGraph::apply_single_rule` into the same `apply_action_from_rule` the sweep
uses, so the application counter — the budget's denominator — means one thing.
A Guide never constructs a `RewriteAction`, never calls `union`, never touches
`const_fact`, and never holds a `&mut EGraph`. That is the property L4 rests
on, and it is what lets this PR owe a quality measurement instead of a
correctness suite.
