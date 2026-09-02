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
**Status:** REGISTERED. §3–§5.2's Register run is complete (§8, Entry 1) — every constant
previously marked **TBD** is now fixed. Committed before any guided run at `|R| > 62` uses this
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

`ΔU` is undefined at `|R|=62` itself (that point is never reordered — §0/§2 invariant — so it has
no `OrderMatchedBase` counterpart); rows below cover the 7 realized inflated points only.

| Mode | grid `\|R\|` | B | `U(p)` (v2, unchanged) | `U(`OrderMatchedBase`)` | `ΔU(p)` | Spearman ρ | `ΔU`(max) | ≥ +Δ1(v3)? | **verdict** |
|---|---|---:|---|---|---|---:|---:|---|---|
| (i) | 93,124,186,248 | 100 | 43.76, 41.12, 15.44, 38.67 | 60.55, 43.88, 13.85, 4.33 | -16.79, -2.76, +1.59, **+34.34** | **1.000** | +34.34 | YES (Δ1=0.05) | **HOLDS** (direction + effect) |
| (i) | 93,124,186,248 | 200 | 6.69, 25.44, 9.47, 27.02 | 24.82, 16.89, 4.70, 3.36 | -18.13, +8.55, +4.76, **+23.66** | 0.800 | +23.66 | YES (Δ1=4.82) | **HOLDS** (direction + effect) |
| (ii) | 93,124 | 100 | 60.55, 41.69 | 60.55, 43.88 | 0.00, -2.19 | -1.000 (n=2, degenerate) | -2.19 | trivially, Δ1≈0 | direction FAILS; only 2 points, not a real trend read |
| (ii) | 93,124 | 200 | 25.60, 21.22 | 24.82, 16.89 | +0.78, +4.33 | +1.000 (n=2, degenerate) | +4.33 | trivially, Δ1≈0 | direction FAILS (n=2 rho uninformative); effect clears a near-zero Δ1 |
| (iii) | 95 | 100 | 33.11 | 48.25 | -15.14 | undefined (n=1) | -15.14 | NO (Δ1=11.45) | direction undefined; effect FAILS |
| (iii) | 95 | 200 | 52.06 | 27.52 | +24.54 | undefined (n=1) | +24.54 | YES (Δ1=16.74) | direction undefined; effect HOLDS |

**Reading.** Mode (i) is the only mode with enough grid points (4) for a Spearman read, and it
holds cleanly at both budgets — but the sign is *increasing* regret with `|R|`, not decreasing:
`ΔU` climbs from -16.8% at `dup:93` to +34.3% at `dup:248` (B=100). Since mode (i) cannot change
what saturation can reach (exact duplicates, closure identical to `base` at every point), this is
not a capacity effect — it is application-budget dilution: a fixed *count* of applications buys a
shrinking share of productive (non-duplicate) matches as redundant duplicate slots are added to
every sweep, even with the 62 real rules' relative sweep position held fixed via
`OrderMatchedBase`. This is the opposite of §3's stated advance prediction (mode (i) `ΔU` sitting
near zero) — the prediction assumed holding order fixed would leave nothing for `|R|` to move;
what it misses is that `|R|` still governs how much of a fixed *application* budget each sweep
pass costs, independent of order or closure. Modes (ii) (2 points) and (iii) (1 point) do not have
enough grid points to read a trend at all — mode (ii)'s two-point Spearman ρ is ±1 by construction
regardless of effect size, and both modes' `ΔU` values are small-to-moderate and inconsistent in
sign between B=100/B=200. Full numbers, per-expression pairing, and the bootstrap detail:
`docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.{json,md}`.

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

| Rule set | B=100 median U | B=200 median U | differing from `base` @100/@200 (of 188) |
|---|---:|---:|---|
| `base` (production order, = v2's |R|=62 point) | 96.58 | 40.49 | — (reference row) |
| `Shuffled(SEED_A=1)` | 43.74 | 23.57 | 175 / 150 |
| `Shuffled(SEED_B=2)` | 46.19 | 25.70 | 186 / 143 |
| `Shuffled(SEED_C=3)` | 26.28 | 1.49 | 186 / 151 |
| `StaticReorder(NumericFirst)` | **1.12** | **0.44** | 186 / 140 |

**Reading (the prediction above, evaluated).** `StaticReorder(NumericFirst)`'s median U sits at
1.12% (B=100) — not merely "substantially below" `base`'s 96.58%, but ~86x smaller, far past the
per-rule-control's 0.565 benchmark the orchestrator finding cited, and better than every one of
the three random shuffles too (26–46%). This is the production quick-win registered as data, not
yet as a production change (§0). The three `Shuffled` seeds do spread meaningfully (26–46% at
B=100, 1.5–25.7% at B=200) — `0x2026_0901` (v2's single interleave seed, itself equivalent to a
4th random draw once inflation content is stripped out — see the seed-sensitivity addendum below)
was not an outlier in the sense of falling outside this range, but the range itself is wide enough
that a single seed's `ΔU` (§3) should be read with that spread as its uncertainty, not as an exact
value. Every tested order — random or static — still beats `base` by a wide margin, so the
*existence and direction* of the order effect is not seed-fragile even though a specific `ΔU`
number is. `rapid`/`blitz` bands: U=0.00% at every order tested (differing counts nonzero, so the
orders do change costs somewhere in the sample — the differences just don't survive to that
budget's regret at those node-count scales); reported for completeness, no claim drawn from them.

**Seed sensitivity of an inflated point (addendum, not in the original skeleton).** `dup:124` and
`comp:93` re-run under two additional interleave seeds (`1`, `2`) alongside the registered seed —
classical band:

| Rule set | seed | U@100 | U@200 |
|---|---|---:|---:|
| `dup:124` | `0x20260901` (registered) | 41.12 | 25.44 |
| `dup:124:interleave:1` | 1 | 37.80 | 32.37 |
| `dup:124:interleave:2` | 2 | 50.33 | 22.62 |
| `comp:93` | `0x20260901` (registered) | 60.55 | 25.60 |
| `comp:93:interleave:1` | 1 | 39.95 | 18.77 |
| `comp:93:interleave:2` | 2 | 38.70 | 13.07 |

Spread (range) across the 3 seeds: `dup:124` 12.5 pts @100 / 9.8 pts @200; `comp:93` 21.9 pts @100
/ 12.5 pts @200 — real, on the same order as the base-62 spread above, another sign that a single
inflated point's `U` (and thus §3's `ΔU`, anchored to one seed) carries seed noise of this
magnitude. Full numbers: `docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.{json,md}`.

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
- **Δ1(v3)**, per mode, at the smallest inflated point, computed on PAIRED per-expression
  `regret_p(e) − regret_matched(e)` (10,000 resamples, seed 42, same protocol as v2 §5.3):
  - Mode (i) at `dup:93`: B=100 half-width **0.05 pts** (median 0.00, CI [-0.09, 0.00]); B=200
    **4.82 pts** (median -4.21).
  - Mode (ii) at `comp:93`: B=100 half-width **0.00 pts** (median 0.00, CI [0.00, 0.00] — the
    composition rules barely fire within B=100 at |R|=93, so `dup:93`/`comp:93` are nearly
    identical to their `OrderMatchedBase` reference for almost every expression); B=200 **0.00
    pts** (same).
  - Mode (iii) at `new:95`: B=100 half-width **11.45 pts** (median -0.18, CI [-21.22, +1.67]);
    B=200 **16.74 pts** (median +6.12).
- Every other v3-specific number in §3/§4's tables: filled by this run (§8, Entry 1).

## 6. Gates

Same accept/kill/honest-fallback shape as v2 §8, evaluated against v3's `ΔU` statistic (§3) in
place of v2's raw `U` statistic wherever this document supersedes v2's reading.

- **Accept gate (per mode):** H1(v3) (§3) AND H2 (design §1.3, unchanged) hold on DEV classical
  (n=334) at B=100. **Not evaluated here** — this Register run is TRAIN+DEV-sample unguided data
  only (§1, inherited from v2), matching v1/v2's own scope; no DEV-only re-split or guided run was
  performed under this document. H2 remains UNTESTED, as it was left in v2 §11 Entry 2.
- **Kill gate (per mode):** H2 part 3 failing at any `|R|` point on DEV — not triggered; H2 was not
  evaluated (above).
- **Honest fallback — this is what fired.** `ΔU` shows an `|R|`-growing effect only in mode (i)
  (§3), and it is a dilution cost (increasing regret), not a coverage gain — the opposite sign
  from what would make `|R|` itself a reason to add rules. Modes (ii)/(iii) have too few grid
  points to assert a trend either way. So: v2 §6b's confound argument is essentially the WHOLE
  explanation for v2's raw `U(|R|)` finding — order dominates, and the residual `|R|` effect
  (mode i) is small and adverse relative to it (§4's order effect: 50–95 points at B=100 from a
  reorder alone, vs mode (i)'s +34.3-point `ΔU` maximum). This is the honest-fallback outcome
  §6 anticipated, now recorded as data (§8, Entry 1).

## 7. Reproduction (commands actually run for Entry 1, §8)

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

# §4 addendum — seed sensitivity of an inflated point: dup:124 and comp:93 under
# 2 additional interleave seeds (the registered-seed rows are reused from v2's CSV)
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-seed-sensitivity-v3.csv \
    --out-json docs/results/2026-09-01-round2-seed-sensitivity-v3.json \
    --rule-sets dup:124:interleave:1,dup:124:interleave:2,comp:93:interleave:1,comp:93:interleave:2

# fingerprint + mode-independence + pinned-order guarantees for the three new RuleOrder variants
cargo test -p pixelflow-search math::inflate -- --nocapture

# raw-row union under one header (the .csv this document's numbers trace back to)
{ head -1 docs/results/2026-09-01-round2-order-matched-base-v3.csv; \
  tail -n +2 docs/results/2026-09-01-round2-order-matched-base-v3.csv; \
  tail -n +2 docs/results/2026-09-01-round2-order-effect-v3.csv; \
  tail -n +2 docs/results/2026-09-01-round2-seed-sensitivity-v3.csv; \
} > docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.csv

# aggregate stats — ΔU, the order table, seed spread (round2_register_stats.py's
# per_rule_set/quartiles/bootstrap machinery, imported not re-derived; new script
# because base_rs in the shared tool is hard-wired to "base", and §3 needs an
# arbitrary per-point reference)
python3 pixelflow-pipeline/scripts/round2_register_stats_v3.py \
    --v2-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv \
    --matched-csv docs/results/2026-09-01-round2-order-matched-base-v3.csv \
    --order-csv docs/results/2026-09-01-round2-order-effect-v3.csv \
    --seed-csv docs/results/2026-09-01-round2-seed-sensitivity-v3.csv \
    --out-json docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.json \
    --out-md docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.md
```

## 8. Results appended against the gates

(Append-only, as in v1/v2.)

**Entry 1 (2026-09-01, this commit).** Unguided Register run under this document, `phase3_round2
_unguided_curves --release`, same 400-expression sample as v1/v2 (188 classical). v2's 8 already-
measured rule sets (`base`, `dup:93/124/186/248`, `comp:93/124`, `new:95`) are **not re-run** —
reused byte-identical from `docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv` (the
loader in `round2_register_stats_v3.py` asserts byte-identity wherever a `(rule_set, expr,
checkpoint)` key appears in more than one input file, which it does for `base` here, and it did
not die). 3 new binary runs, 13 new rule-set curves total: 5 `OrderMatchedBase` references (§3),
5 order-effect sets (`base` re-measured + 3 `Shuffled` seeds + `StaticReorder(NumericFirst)`, §4),
4 seed-sensitivity curves (`dup:124`/`comp:93` × 2 additional seeds, §4 addendum). All at
`|R| ∈ {62, 93, 124}` — no run in this document touches `|R| ≥ 186`, so none of it revisits the
`comp:186`/`comp:248` wall (§2) or v2's safety-ceiling panics.

**H1(v3) verdict: mode (i) HOLDS (direction + effect, both budgets) — but in the dilution
direction, not the coverage direction §3 flagged as the alternative to watch for. Modes (ii)/(iii)
are underdetermined (2 and 1 grid points).** Full tables: §3/§4 above. **§6's honest fallback
fired**: order dominates and the residual `|R|` effect, where measurable, is small and adverse.

**Production quick-win, measured directly:** `StaticReorder(NumericFirst)` — descending TRAIN
strict-positive rate, ties by production index, zero runtime cost, no rule added or removed —
cuts median unguided regret at B=100 from `base`'s 96.58% to 1.12% (188-expression classical
sample), beating every random shuffle tested too. This is data, not a production change (§0) —
adopting it as `all_rules()`'s order is JP's decision on this data, out of this branch's scope.

**Files:** `docs/results/2026-09-01-round2-order-matched-base-v3.{csv,json}`, `docs/results/2026
-09-01-round2-order-effect-v3.{csv,json}`, `docs/results/2026-09-01-round2-seed-sensitivity-v3.
{csv,json}` (the three raw per-run outputs); `docs/results/2026-09-01-round2-unguided-vs-rulecount
-v3.{csv,json,md}` (the union CSV and the aggregate stats + narrative this document's numbers are
drawn from); `pixelflow-pipeline/scripts/round2_register_stats_v3.py` (the aggregation script, new
this commit).
