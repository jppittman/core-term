# Phase 3 Round 2 registration v3: the |R| effect and the order effect, separated

**SUPERSEDES** `docs/plans/2026-09-01-phase3-round2-registration-v2.md`'s **H1 READING — not its
data.** Every curve, fingerprint, and statistic v2 measured (§4/§5/§6 of that document) stands
exactly as run and is not rerun or revised by this document. **Reason (v2 §6b, appended
2026-09-01):** v2's H1 verdict compared each inflated point `U(p)` against the unshuffled `base`
(production-order) point. Mode (i) (`dup:*`) proves that comparison is confounded: a duplicate
rule delegates its `apply`/templates verbatim to its original (`DuplicateRule`, `inflate.rs`), so
it can never let saturation reach a node the base 62 rules could not already reach — mode (i)'s
closure is unchanged by construction. Yet v2's own table shows `U(|R|)` falling under mode (i) just
as sharply as under modes (ii)/(iii) (96.59% → 43.76% → 41.12% → 15.44% → 38.67% at B=100, ρ =
−0.900) — which cannot be a closure/rule-count effect, so it is not evidence that `|R|` itself moves
`U`. What changed between `base` and every inflated point is that `Interleave(seed)`
Fisher-Yates-shuffles the *whole* base+inflation vector together, relocating each of the 62 base
rules to a new position — and B=100 is sub-sweep at every inflated point (v2 §4.1: 1.01 sweeps at
`base`, 0.10 sweeps at `dup:248`), so which rules the budget reaches is decided mostly by where the
shuffle put them, not by how many total rules exist. v3 separates the two effects: it measures the
`|R|` effect against a reference that holds order fixed (`RuleOrder::OrderMatchedBase`, already
committed on this branch), and measures the order effect on its own, as an independent registered
finding, rather than leaving it latent inside every inflated-point comparison.

**Date:** 2026-09-01
**Status:** SKELETON. This document fixes v3's grid, references, statistics, seeds, and gates.
**No data has been collected under it yet** — every constant marked **TBD** is fixed only by an
unguided Register run on this harness (§5), committed before any guided run at `|R| > 62` uses this
document, per the binding rule inherited from v1/v2.

**Authority:** carries forward v2's full authority chain (`docs/plans/2026-08-31-guide-design-revision.md`
§5 protocol; `docs/plans/2026-09-01-phase3-registration.md` — Round 1, FROZEN; `docs/plans/2026-09-01-phase3-round2-rule-scaling.md`
— Round 2 design) plus `docs/plans/2026-09-01-phase3-round2-registration-v2.md` (v2 — H1 *reading*
superseded by this document, §6b's argument and every measured number otherwise inherited
verbatim). Orchestrator finding on the v2 result (this branch's task header, 2026-09-01, restated
in v2 §6b): mode (i) cannot change closure, so the interleaved-order drop in `U(|R|)` is a
confound between rule order and rule count; a static reorder of `all_rules()` may itself capture
much of Round 1's Guide-vs-unguided win at zero runtime cost — the production quick-win candidate
this document's §4 registers a finding about, without changing `all_rules()` itself (out of this
branch's scope, per its binding rules).

## 0. Mechanism (already committed — `pixelflow-search/src/math/inflate.rs`, prior commit on this
branch)

Three new `RuleOrder` variants, all base-62-only (`spec.mode == None`; a caller that reaches
`apply_order` with one of them on an inflated, `total > 62` build gets a loud panic, never a silent
no-op — CLAUDE.md's no-silent-failures rule):

- **`RuleOrder::OrderMatchedBase(seed, total)`** — the base-62 rules, reordered to the exact
  relative order they occupy inside `Interleave(seed)` of the `total`-rule inflated set (the
  subsequence of that interleaved vector restricted to base indices). **Proven mode-independent**
  by `order_matched_base_is_the_interleaved_subsequence_and_mode_independent` (`inflate.rs`):
  `fisher_yates`'s swap sequence is a function of vector LENGTH only, never content, and
  `DUPLICATE_GRID`/`COMPOSITION_GRID` share the same total↔inflation-count mapping at every shared
  grid point (93/124/186/248), so `OrderMatchedBase(s, t)` needs no mode argument — `dup:t` and
  `comp:t` under `Interleave(s)` yield an identical base-62 subsequence. This is the reference §3
  below measures every inflated point against.
- **`RuleOrder::Shuffled(seed)`** — base-62 fully shuffled by `seed`, no inflation content at all.
  Isolates seed sensitivity of the order effect on its own (§4).
- **`RuleOrder::StaticReorder(StaticReorderKind::NumericFirst)`** — base-62 ordered by descending
  TRAIN strict-positive rate (`docs/results/2026-09-01-train-guide-report.md`'s per-rule table),
  ties broken by ascending production index; pinned as `NUMERIC_FIRST_ORDER: [usize; 62]`
  (`inflate.rs`). The production quick-win candidate (§4).

CLI spec grammar (`RuleSetSpec::parse`, already committed): `"base:matched:<seed>:<total>"`,
`"base:shuffled:<seed>"`, `"base:static:numeric-first"` — `<seed>` decimal or `0x`-prefixed hex, no
underscores. Every one of these builds a 62-rule set; `rule_set_fingerprint` distinguishes all four
base-62 orders (production, matched, shuffled, static) from each other, pinned by
`base_only_orders_fingerprint_differently_from_each_other_and_from_production`.

**No production behavior change.** `all_rules()` is untouched by this branch's every commit — these
are harness-selectable orders a research binary opts into, exactly like `Append`/`Interleave`
before them. Adopting `StaticReorder(NumericFirst)` (or any reorder) as `all_rules()`'s own order is
a separate, unmade decision for JP, on this document's future data — not something this branch does.

## 1. Environment

Inherited from v2 §1 verbatim except where stated: same corpus, same 400-expression H1 sample (188
classical), same B = 100 (primary) / 200 (secondary), same `CostModel::latency_prior()`, same
application-count work axis (sweeps/evals reported alongside, per v2 §0.1/§0.2), same 14-point
checkpoint grid, same class cap, same safety ceilings (`comp:186`/`comp:248` remain unattempted here
— §2), same source-rev discipline. **New:** the three orders above; `DEFAULT_INTERLEAVE_SEED =
0x2026_0901` is reused, unchanged, as the seed `OrderMatchedBase` matches against (it must be the
same seed v2's `Interleave(seed)` points were built with, or `ΔU` would compare against the wrong
reference). Three additional seeds are pre-committed for the order-effect measurement (§4, §5.2):
**`SEED_A = 1`, `SEED_B = 2`, `SEED_C = 3`** — chosen for nothing but being fixed, small, and stated
here before any `Shuffled(*)` curve is run, exactly as the interleave seed was in v2 §1.

## 2. Grid

Same points as v2's realized grid (v2 §4): mode (i) `{62, 93, 124, 186, 248}`, mode (ii) `{62, 93,
124}` (`comp:186`/`comp:248` did not complete under v2's safety ceiling and are **not
re-attempted by this document** — same wall, out of scope here), mode (iii) `{62, 95}`. This
document adds no new inflated points; it adds a new *reference* (§3) and a new *independent
measurement* (§4) over the same grid.

## 3. H1 (v3): the `|R|` effect, order held fixed

For every inflated point `p = (mode, |R|)` at the registered interleave seed `s =
DEFAULT_INTERLEAVE_SEED`:

```
ΔU(p) = U(p) − U(OrderMatchedBase(s, |p|))
```

where `U(p)` is v2's already-measured median unguided regret at B for the inflated point (v2 §4/§6
data, unchanged), and `U(OrderMatchedBase(s, |p|))` is a **new** unguided curve — same 188-expression
classical sample, same B, same reference convention (§3 below) — measured on the base-62 rules
swept in exactly the order they occupy inside `Interleave(s)` at that `|p|`. This holds sweep order
fixed between the two terms: both curves see "the base rules in interleaved position order at this
|R|," and the only thing that differs is whether the inflated rules are present at all. `ΔU`
therefore isolates the `|R|` effect from the order effect v2 §6b identified.

**H1(v3):** `ΔU(p)` grows with `|R|` — Spearman `ρ` of `ΔU` over each mode's grid, **and** `ΔU` at
the largest completed `|R|` point clears **+Δ1(v3)** (§5.2; the same derivation rule as v2 §5.3 —
95% bootstrap CI of the median `ΔU` at the smallest inflated point per mode, 10,000 resamples, seed
42 — but computed on `ΔU`, not raw `U`, since the quantity being tested for a minimum detectable
effect has changed). Both conditions are evaluated **per mode**, exactly as v2 §6 did for `U`.

**What this predicts, stated in advance so it is not rediscovered by surprise:** mode (i) is exact
duplicates — its closure is identical to `base`'s at every |R| — so if v2 §6b's confound argument is
correct, mode (i)'s `ΔU` should sit near zero at every point (no room for the added rules to help or
hurt, once order is matched), unlike v2's raw `U(|R|)`, which fell sharply. Modes (ii)/(iii) DO add
real closure (mechanical compositions, genuinely new rules) and are where a nonzero, `|R|`-growing
`ΔU` — if it exists — would be a genuine capacity/coverage effect, not an order artifact. This
document does not assert either outcome; §8 registers whichever the data show.

| Mode | grid `\|R\|` | B | `U(p)` (v2, unchanged) | `U(`OrderMatchedBase`)` (TBD) | `ΔU(p)` (TBD) | Spearman ρ (TBD) | `ΔU`(max) (TBD) | ≥ +Δ1(v3)? (TBD) | **verdict** |
|---|---|---:|---|---|---|---:|---:|---|---|
| (i) | 62,93,124,186,248 | 100 | 96.59, 43.76, 41.12, 15.44, 38.67 | TBD | TBD | TBD | TBD | TBD | TBD |
| (i) | 62,93,124,186,248 | 200 | 40.49, 6.69, 25.44, 9.47, 27.02 | TBD | TBD | TBD | TBD | TBD | TBD |
| (ii) | 62,93,124 | 100 | 96.59, 60.55, 41.69 | TBD | TBD | TBD | TBD | TBD | TBD |
| (ii) | 62,93,124 | 200 | 40.49, 25.60, 21.22 | TBD | TBD | TBD | TBD | TBD | TBD |
| (iii) | 62,95 | 100 | 96.59, 33.11 | TBD | TBD | TBD | TBD | TBD | TBD |
| (iii) | 62,95 | 200 | 40.49, 52.06 | TBD | TBD | TBD | TBD | TBD | TBD |

## 4. The order effect, as its own registered finding

Independent of `|R|` entirely — every rule set measured here is 62 rules. Two sub-measurements,
both on the 188-expression classical sample, both at B = 100/200, both against `ref_U(e, 62)`
(unchanged from v2 §3 — every set here IS the |R|=62 point, so the closure-aware reference is the
min over these curves' own checkpoints together with `base`'s, exactly as v2 §3 defines it for any
fixed |R|):

- **Seed sensitivity:** `U(Shuffled(SEED_A))`, `U(Shuffled(SEED_B))`, `U(Shuffled(SEED_C))` — three
  independently-shuffled base-62 orders, no inflation. If these three disagree substantially, v2's
  single interleave seed (`0x2026_0901`) was not a representative draw and §3's `ΔU` (which is
  anchored to that one seed) needs the caveat spelled out, not silently trusted.
- **Static reorder vs production order:** `U(StaticReorder(NumericFirst))` vs `U(base)` (production
  `all_rules()` order, v2's already-measured |R|=62 point) — the quick-win candidate, measured
  directly rather than inferred.

For every one of the four rule sets in this section, report the **count of the 188 classical
expressions whose extracted cost@B differs from `base`'s** at B=100 and B=200 — the same
"differing" convention v2 §4 used to state that inflated points were reachable at all, here used to
state how much of the sample a pure reorder (no rule added or removed) touches.

| Rule set | B=100 median U (TBD) | B=200 median U (TBD) | differing from `base` @100/@200 (of 188, TBD) |
|---|---:|---:|---|
| `base` (production order, = v2's |R|=62 point) | 96.59 | 40.49 | — (reference row) |
| `Shuffled(SEED_A=1)` | TBD | TBD | TBD |
| `Shuffled(SEED_B=2)` | TBD | TBD | TBD |
| `Shuffled(SEED_C=3)` | TBD | TBD | TBD |
| `StaticReorder(NumericFirst)` | TBD | TBD | TBD |

**Reading rule, stated in advance:** if `StaticReorder(NumericFirst)`'s median U at B=100 sits
substantially below `base`'s 96.59% — on the order of the per-rule-control's 0.565 the orchestrator
finding cites, or better — that is the production quick-win registered as data, not yet as a
production change (§0). If the three `Shuffled` seeds spread widely, the order effect itself is
seed-sensitive and `ΔU` (§3) inherits that sensitivity; if they cluster, `0x2026_0901` was not an
outlier draw and §3's single-seed `ΔU` is a reasonable stand-in for "a typical shuffle's" effect.

## 5. Registered constants

Same binding rule inherited from v1/v2: fixed **only** from unguided data on this harness,
**committed before any guided run at `|R| > 62` uses this document's `OrderMatchedBase` reference**,
and — once filled — may not be revised except by append or a further superseding registration.

### 5.1 Inherited, not re-derived
B = 100/200; `Y`'s formula; ε = 0.005; the 0.02 Δ2 floor; composition pool seed `0x5EED2`;
bootstrap seed `42`; oracle seed `0xC0FF_EE42`; the 400-expression sample and its stratification;
the 14-point checkpoint grid; the closure-aware reference convention (v2 §3, design §1.1); the H2
statistics (design §1.3); `DEFAULT_INTERLEAVE_SEED = 0x2026_0901` (v2 §1, reused as the seed
`OrderMatchedBase` and §3's `ΔU` are anchored to).

### 5.2 New this document
- Order-effect seeds: `SEED_A = 1`, `SEED_B = 2`, `SEED_C = 3` (§1) — pre-committed here, before any
  `Shuffled(*)` curve is run.
- **Δ1(v3): TBD** — 95% bootstrap CI of median `ΔU` at the smallest inflated point per mode (same
  resampling protocol as v2 §5.3), filled once §3's Register run exists.
- Every other v3-specific number in §3/§4's tables: TBD, filled by the same run.

## 6. Gates

Same accept/kill/honest-fallback shape as v2 §8, evaluated against v3's `ΔU` statistic (§3) in
place of v2's raw `U` statistic wherever this document supersedes v2's reading. **Not yet
evaluable** — no data exists under this document (§ Status). Recorded here as structure only:

- **Accept gate (per mode):** H1(v3) (§3) AND H2 (design §1.3, unchanged) hold on DEV classical
  (n=334) at B=100.
- **Kill gate (per mode):** H2 part 3 failing at any `|R|` point on DEV, after one clean
  re-mint/re-train.
- **Honest fallback:** if `ΔU` shows no `|R|`-growing effect in any mode once order is held fixed,
  that is itself the deliverable — v2 §6b's confound argument would then be the WHOLE explanation
  for v2's raw `U(|R|)` finding, with nothing left over once order is controlled for. This is not
  presumed; §8 records whichever way the data land.

## 7. Reproduction (skeleton — commands to run when this Register run happens)

```bash
# §3 — OrderMatchedBase references, one 62-rule curve per inflated |R| point
# on the SAME seed those points were built with (0x20260901 = DEFAULT_INTERLEAVE_SEED
# = 539363585 decimal — the CLI form takes plain hex or decimal, no underscores)
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-order-matched-base-v3.csv \
    --out-json docs/results/2026-09-01-round2-order-matched-base-v3.json \
    --rule-sets base:matched:0x20260901:93,base:matched:0x20260901:124,base:matched:0x20260901:186,base:matched:0x20260901:248,base:matched:0x20260901:95

# §4 — order effect in isolation: 3 shuffled seeds + the static reorder, vs base
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-order-effect-v3.csv \
    --out-json docs/results/2026-09-01-round2-order-effect-v3.json \
    --rule-sets base,base:shuffled:1,base:shuffled:2,base:shuffled:3,base:static:numeric-first

# fingerprint + mode-independence + pinned-order guarantees for the three new RuleOrder variants
cargo test -p pixelflow-search math::inflate -- --nocapture
```

`round2_register_stats.py` (or a v3-specific successor, if the `ΔU` statistic needs a script change
the existing one doesn't already support) computes §3/§4's tables from the two CSVs above.

## 8. Results appended against the gates

(Append-only, as in v1/v2. Empty — no Register run has been made under this document yet.)
