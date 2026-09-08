# Phase 3 Round 2 registration: regret at budget vs rule count

**Date:** 2026-09-01
**Status:** REGISTERED — computed from UNGUIDED data only and committed before any guided run
exists at |R| > 62 (§10 is the proof). Per the design's own rule
(`docs/plans/2026-09-01-phase3-round2-rule-scaling.md`, "Status"), nothing above §11 may be
revised after the first guided run at |R| > 62; results are appended under §11 only. A different
grid, budget, sample, or sweep order is a **superseding registration**, recorded as such with the
reason, never an edit here.
**Authority:** `docs/plans/2026-08-31-guide-design-revision.md` (§5 protocol);
`docs/plans/2026-09-01-phase3-registration.md` (Round 1 — FROZEN; B, Y's formula, the grid, the
curve runner, and the reference convention are inherited from it verbatim);
`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` (Round 2 design — H1/H2 statistics, modes,
what Register may and may not fix, §6); JP's thesis of 2026-09-01, quoted in the design.
**Data sources (unguided only):**
- `docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.csv` — 400 curves × 14
  checkpoints × 7 rule sets (`base`, `dup:93/124/186/248`, `comp:93/124`), written by
  `phase3_round2_unguided_curves` (commit `8edca86a`).
- `docs/results/2026-09-01-round2-unguided-vs-rulecount-mode-iii.csv` — 400 curves × 14
  checkpoints × 2 rule sets (`base`, `new:95`), written by `phase3_round2_new_rules --out-csv`
  (this commit; its `base` rows are byte-identical to the file above's — checked by the script).
- `docs/results/2026-09-01-round2-compositions.json` — the ordered first 186 entries of the seeded
  composition pool with each entry's oracle status (from the `composition_prefix_oracle_census`
  test), and the pool-wide oracle census.
- `docs/results/2026-09-01-phase3-round2-registration.json` + `-tables.md` — every number below,
  produced by `pixelflow-pipeline/scripts/round2_register_stats.py` from the two CSVs (§12 gives
  the exact invocation). The script asserts the unguided CSV schema and refuses anything else.
- Narrative summaries by the harness agents: `docs/results/2026-09-01-round2-unguided-vs-rulecount.md`.

## 1. Environment (fixed; inherited from Round 1 unless stated)

| Item | Value |
|---|---|
| Corpus | Round 1's: `gen_bench_corpus --target 4000 --seed 42`; train MD5 `0ed6cf16abcbc006cd7a3ee2365b15b4`, dev `3026133ebba066eeca10f658da554400` (verified identical by the harness before every run). FINAL (`corpus_final.bin`) not opened. |
| Sample (H1) | Round 1's 400-expression size-stratified TRAIN+DEV sample (stride 10.36 over 4,143): blitz 23, rapid 189, **classical 188** (TRAIN 154 + DEV 34). Same names in every rule set of every mode. |
| Sample (H2) | Full DEV classical band, n = 334 (design §5) — unguided DEV-classical curves at every |R| point are produced in step 3 by the same binaries; they are unguided data and do not change anything registered here. |
| Budgets | classical **B = 100 (primary), B = 200 (secondary)** — Round 1 §4, unchanged. Rapid/blitz: reported, no claim. |
| Cost model | `CostModel::latency_prior()`, deterministic. No wall-clock in any metric. |
| Work axis | Cumulative recorded rule applications, idempotent re-fires included (Round 1 §1). |
| Checkpoint grid | `APP_CHECKPOINT_GRID` = 25 … 204800 (14 points), unchanged. **No grid truncation** — the largest completed rule set (|R| = 248, mode (i)) finished the full grid; no safety ceiling bound anywhere (`sweep_ceiling` = `timeout` = 0 on every completed set). |
| Curve runner | `pixelflow_search::egraph::anytime::run_anytime_curve` — the ONE definition. Checkpoints are crossed at **rule granularity** (`saturate_until_applications` checks the budget between rules of a rule-major sweep; `app_actual` records the exact count). Median `app_actual` at B = 100 on classical is 113 (q1 105, q3 136) at every |R|. |
| Sweep order (binding, and load-bearing for §6) | Rule-major: for each sweep, rules in index order, each over all classes. Every inflated rule set is `all_rules()` (62, production order) as an exact prefix, inflation appended at indices ≥ 62 (design §2). Consequence: an appended rule cannot fire before every one of the 62 has swept once. |
| Class cap | `config_for_node_count(node_count).max_classes`, identical across arms and |R|. |
| Safety ceilings | Per-curve wall-clock `300 s × |R|/62`, sweep ceiling `10,000 × |R|/62`; both PANIC when they bind; neither bound. |
| Source rev | Harness at `8edca86a` (modes i/ii) and `9fccf02d` (mode iii) on `claude/phase3-round2`; this registration's commit adds `--out-csv` to the mode (iii) binary, the census test, and the stats script — no behavior change to any curve. |
| Production | `all_rules()` = 62, pinned by test; nothing in this workflow changes it. |

## 2. The |R| grid actually realized, and rule-set identity

Rule-set identity is `rule_set_fingerprint` (names in index order, hashed); it is written into
every CSV row and every guided artifact must carry the same value or be refused.

| Mode | spec | \|R\| | fingerprint | inflation content | validated count |
|---|---|---:|---|---|---|
| shared | `base` | 62 | `e99af8402beaff5d` | `all_rules()` | 62 (production suite) |
| (i) | `dup:93` | 93 | `fdd617246eb98590` | +31 `DuplicateRule` copies of even indices 0, 2, …, 60 (`#dup1`) | 93 — a duplicate delegates to its original; no oracle needed |
| (i) | `dup:124` | 124 | `87fefd5a63575175` | +62: one full cycle | 124 |
| (i) | `dup:186` | 186 | `37a4c537606a549b` | +124: two cycles | 186 |
| (i) | `dup:248` | 248 | `809a0f52b61fe6c0` | +186: three cycles | 248 |
| (ii) | `comp:93` | 93 | `0c3fd6f35f444a59` | +31: pool prefix [0, 31) | 62 + 27 kept / **1 dropped** / 3 untestable |
| (ii) | `comp:124` | 124 | `521798ae521a0572` | +62: pool prefix [0, 62) | 62 + 56 kept / **1 dropped** / 5 untestable |
| (ii) | `comp:186` | 186 | — | +124: pool prefix [0, 124) | 62 + 102 kept / **2 dropped** / 20 untestable — **curves not realized** (§2.2) |
| (ii) | `comp:248` | 248 | — | +186: pool prefix [0, 186) | 62 + 154 kept / **2 dropped** / 30 untestable — **curves not realized** (§2.2) |
| (iii) | `new:95` | **95** | `4f4a4cbd2e4f89cb` | +33: `round2_rules::experimental_rules()` | 95 — 33/33 pass the per-family oracle tests; 24 families |

### 2.1 Mode (iii): 33 indices, not 31

The design's running total said 31 indices; its own table lists 28 N-rows plus five per-op extra
indices (N14b, N14c, N26b, N27b, N24r), which is **33** — the 31 was an arithmetic undercount, and
the implemented batch follows the table. The mode (iii) pair is therefore **|R| = 62 vs 95**, not
62 vs 93. All 33 pass their cross-form oracle tests
(`cargo test -p pixelflow-search round2_rules`, 25/25); none were dropped. The names, in index
order 62..95: min-max-duality-min/max, min-max-absorption-min/max, min-translate, max-translate,
min-scaled-by-nonneg-literal, min-max-distributive, abs-as-max, max-self-neg-as-abs,
select-same-branch, select-lt-to-min, select-lt-to-max, select-hoist-neg/abs/sqrt,
compare-flip-lt, tan-definition, tan-fusion, exp-as-exp2, ln-as-log2, log10-as-log2,
sqrt-product, rsqrt-square-as-recip, normalize-as-sqrt, recip-product, recip-of-product,
fma-unfuse, fma-mul-identity, fma-add-identity, neg-distributes-add, neg-distributes-mul,
div-by-literal.

### 2.2 Mode (ii): the pool, its gate, and the two unrealized points

- Pool: `composition_pool(all_rules(), 0x5EED2)` = **3,339** compositions from the 30 templated
  production rules (first-order unification of `B.lhs` against `A.rhs` at the root and every
  proper subterm; identity/no-op/duplicate filters per design §2.2). Oracle census over the whole
  pool (256 points/rule, seed `0xC0FF_EE42`, agreement-rate gate 0.60): **2,935 kept, 9 dropped,
  395 untestable** (> 4 metavariables or mask-rooted), mean agreement 0.997.
- **The realized `comp:N` sets are raw prefixes of the seeded pool, not of the oracle-filtered
  pool.** `build_rule_set` draws `pool[..inflation]` without re-running the oracle (the oracle is a
  `#[cfg(test)]` gate). The census above therefore matters per prefix, and it is recorded exactly:
  `exp-homomorphism∘exp-homomorphism@[0,0]` (pool index 13) is in every realized prefix and fails
  the gate (agreement below 0.60); `reverse-angle-addition∘exp-homomorphism@[0,0]` (index 114)
  joins it at 186/248. Both compose through `exp` at a nested position: on the oracle's sampling
  domain (leaves in [0.25, 4]) `exp(exp(x + y) + b)` saturates (`exp` saturates rather than
  overflowing, CLAUDE.md), which the oracle's conditioning check does not detect because the
  saturated value is finite — a domain-narrowing divergence of the same kind the oracle's doc
  describes, not a unifier error (the composition of two valid rewrites is valid over the reals
  by the design's §2.2 argument). **Pre-committed treatment:** the realized sets stand as
  fingerprinted, drops and untestables included, and every guided run at these |R| uses the SAME
  fingerprinted sets — the mode (ii) arm is "compositions as generated", and the per-prefix
  census above is the disclosure. Re-drawing the prefixes from the filtered pool would change the
  rule sets, invalidate these unguided curves, and is a superseding registration.
- **`comp:186` and `comp:248` were not realized.** The modes (i)/(ii) binary ran all nine sets in
  one invocation; the process that produced the committed CSV ended after `comp:124` without
  writing the last two sets (the previous session's background process did not survive; a second
  attempt in this session was still inside `comp:186`'s classical tail when this document was
  committed — wall-clock is not a metric and is not why they are excluded; they are excluded
  because their rows do not exist). **The mode (ii) grid registered here is {62, 93, 124}.** Their
  rule sets are fully specified (seed, prefix lengths, census) and their unguided curves may be
  appended under §11 as data, under this registration's constants, provided that no guided run at
  |R| ∈ {186, 248} of mode (ii) precedes them; §5 says which registered numbers would then be
  computed for those points (Y and Δ2 depend on them) — everything else stays fixed.

## 3. Reference and regret conventions (binding)

- `ref(e, |R|)` is the **closure-aware** reference at the same |R|: the minimum cost over every
  checkpoint of every arm at that |R| for expression e (design §1.1). Never pooled across |R|.
- **In this Register, and for H1 for all time, the reference is the unguided-only version**
  `ref_U(e, |R|) = min over the 14 checkpoints of the unguided curve at |R|`. H1's verdict (§6) is
  computed against `ref_U` and is recorded now, before any guided run; it does not get recomputed
  against a two-arm reference later (the design fixes H1 as an unguided measurement, §1.2).
- H2's statistics (`Q`, `G`) use the two-arm reference at each |R| when the guided data exist;
  `Q(|R|)` itself is a cost ratio and needs no reference.
- Regret `= (cost@B − ref) / ref`; a positive cost against a zero reference is infinite regret,
  never 0%. No classical reference is zero in any realized set (`ref_zero_count` = 0).
- Cost at B is the checkpoint whose `app_target` = B (first between-rules point with cumulative
  applications ≥ B); `app_actual` is recorded and reported.
- The `CYCLE_COST` extraction sentinel (≥ 900,000; `extract_dag` on a self-referential class,
  pre-existing under `all_rules()` alone) is excluded from "curve end" cost medians (reported
  as "excl. cycle" with the hit count) and from the closure-gain column; it never affects cost@B
  or cost@4B on this sample (no B or 4B checkpoint hits it).

## 4. Measured unguided curves (the tables every number below is read from)

Generated by the script from the raw rows (`docs/results/2026-09-01-phase3-round2-registration-tables.md`
is the same content, regenerated on append).

**Classical band (n=188), absolute cost and where the inflation is visible.** `visible@B` =
expressions whose cost@B differs from the |R|=62 curve's; `first visible` = smallest grid
checkpoint at which any expression's cost differs from |R|=62 (`never` = identical at all 14).

| rule set | \|R\| | cost@100 q1/med/q3 | cost@200 q1/med/q3 | cost@400 med | cost@800 med | curve-end med (excl. cycle) | cycle hits | app_actual@100 med | visible@100 | visible@200 | first visible | apps-to-end med | ended |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `base` | 62 | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | — | — | n/a | 2686 | app_budget=1, quiesced=187 |
| `dup:93` | 93 | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3327.5 | 2621.0 | 27 | 113 | 0 | 3 | 200 | 3115 | app_budget=2, quiesced=186 |
| `dup:124` | 124 | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | 0 | 0 | never | 4274 | app_budget=1, quiesced=187 |
| `dup:186` | 186 | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | 0 | 0 | never | 4803 | app_budget=1, quiesced=187 |
| `dup:248` | 248 | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | 0 | 0 | never | 5322 | app_budget=1, quiesced=187 |
| `comp:93` | 93 | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2597.0 | 27 | 113 | 0 | 0 | 800 | 2980 | app_budget=1, quiesced=187 |
| `comp:124` | 124 | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3925.5 | 3331.0 | 2609.0 | 28 | 113 | 0 | 0 | 400 | 3809 | app_budget=1, quiesced=187 |
| `new:95` | 95 | 2952 / 14459.5 / 1002550 | 1639 / 7648.5 / 1000456 | 5270.0 | 3884.0 | 1689.5 | 100 | 113 | 0 | 18 | 200 | 24686 | app_budget=2, class_cap=3, quiesced=183 |

(`app_budget` in "ended" is the runner's label for a curve still live at the grid's top
checkpoint — "grid_exhausted" in the harness summaries.) The q3 of cost@B sits on the cycle
sentinel for a quarter of the classical sample: those expressions' *early* extraction is cyclic
and resolves later; it is why every registered statistic is a median with its quartiles, and why
the p75 regret column below is in the tens of thousands of percent.

**Classical band, unguided regret U against `ref_U` at the same |R|, truncation loss L, and Y.**
Percentages; regret quartiles are per-expression. Closure gain is design §4.2's
`fid(e) = (ref_U(e, 62) − ref_U(e, |R|)) / ref_U(e, 62)` over the 173 expressions with a finite,
non-cyclic base reference.

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

**blitz (n=23) and rapid (n=189) — reported, no claim** (Round 1 §4: nothing to buy back there).
Median cost@100 / cost@200 is 78 / 78 (blitz) and 271 / 266 (rapid) at every |R| of modes (i)
and (ii); mode (iii) rapid is 276 / 282. Median regret and truncation loss are 0.000 everywhere.
Median applications-to-end grow with |R|: blitz 13 → 16 → 42 → 74 → 85 (dup), 13 flat (comp),
259 (new); rapid 80 → 87 → 157 → 185 → 314 (dup), 80 flat (comp), 2,043 (new). Full rows in the
tables file.

**Mode (iii) closure gain at curve end** (the harness's quiescence-cost version, results doc):
blitz 0.000% (7/23 positive), rapid 0.251% (97/175), classical 0.782% (62/86, 102 excluded for a
sentinel on either arm); the reference-based version above (classical, 173 expressions):
median +0.737%, p90 +2.20%, positive 121, negative 21. The batch quadruples the classical cycle-
sentinel rate at curve end (25 → 100 of 188) and is ~9× slower to quiesce (2,686 → 24,686 median
applications), with 3 class-cap hits (base: 0).

## 5. Registered constants

Everything in this section is fixed from the tables in §4 and cannot move after a guided run.

### 5.1 Inherited, not re-derived
B = 100 / 200 (classical); Y's formula `Y = 1 − (1 + L/2)/(1 + L)`; ε = 0.005; the 0.02 floor on
Δ2; the seeds (`0x5EED2` pool, `42` bootstrap, `0xC0FF_EE42` oracle); the sample; the grid; the
reference convention (§3); the H2 statistics (design §1.3).

### 5.2 Y(|R|) per mode and B

| Mode | \|R\| | L@100 | **Y@100** | 1 − Y@100 | L@200 | **Y@200** | 1 − Y@200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| (i), (ii) — every realized point | 62, 93, 124, 186, 248 | 48.467% | **16.32%** | 0.8368 | 21.922% | **8.99%** | 0.9101 |
| (iii) | 62 | 48.467% | **16.32%** | 0.8368 | 21.922% | **8.99%** | 0.9101 |
| (iii) | 95 | 40.611% | **14.44%** | 0.8556 | 16.941% | **7.24%** | 0.9276 |

Y is identical at every closure-preserving point because cost@B is per-expression identical to
|R| = 62 there (§6) and cost@4B differs for at most 3 of 188 expressions (`comp:124`, `dup:93`),
which leaves the median L unchanged. Round 1's 16.3% / 9.0% are reproduced to the reported digit.

### 5.3 Δ1 — H1's minimum effect (from |R| = 62 only)

95% bootstrap CI of the median unguided regret at |R| = 62 on the 188 classical expressions:
10,000 resamples with replacement, `random.Random(42)`, order-statistic 2.5/97.5 percentiles.

| B | median U(62) | CI | **Δ1 = half-width** |
|---:|---:|---|---:|
| 100 | 96.585% | [73.120%, 129.580%] | **28.230 points** |
| 200 | 40.489% | [25.704%, 50.389%] | **12.342 points** |

Reading: a shift in the median unguided regret smaller than 28.2 points (B = 100) is inside the
median's own sampling noise on this sample and does not count as "unguided regret grows".

### 5.4 Δ2 — H2's minimum effect, per mode

`Δ2 = max(0.02, Y(|R|max) − Y(62))`:

| Mode | Y(|R|max) − Y(62) at B=100 | **Δ2** | at B=200 | **Δ2** |
|---|---:|---:|---:|---:|
| (i) | 0.000 | **0.020** | 0.000 | **0.020** |
| (ii), realized grid (|R|max = 124) | 0.000 | **0.020** | 0.000 | **0.020** |
| (iii) | −0.019 | **0.020** | −0.017 | **0.020** |

If `comp:186`/`comp:248` are appended (§2.2), Δ2 for mode (ii) is recomputed from *their* Y by the
same formula; with cost@B per-expression identical to base at every realized point, the
pre-registered expectation is that it stays at the 0.02 floor, but it is computed, not assumed.

### 5.5 The H2 slope-ratio threshold — derivation from the unguided spread, and its verdict

The design reports `slope_G / slope_U` (least-squares slopes of the guided and unguided median
regrets against |R|) and gates on the three Q-tests, not on the ratio. This Register fixes the
ratio's reading rule from the unguided spread, in advance:

1. `slope_U` is fitted over the realized grid; its 95% bootstrap CI (same resampling as Δ1,
   applied to the per-expression regrets at every grid point jointly) is the "unguided spread"
   of the slope. Registered values: **mode (i): slope_U = 0 exactly** at both B (every median
   is bit-identical); **mode (ii): −0.069 points/rule (B=100), −0.008 (B=200)** — negative;
   **mode (iii): +0.364 points/rule (B=100), +0.196 (B=200)** over two points.
2. Rule fixed here: the slope ratio is **defined as evidence only when the 95% bootstrap CI of
   `slope_U` excludes zero and `slope_U > 0`**; then H2's slope form is "the upper 95% bound of
   `slope_G / slope_U` is < 1". When the CI contains zero or `slope_U ≤ 0`, the ratio is
   **undefined** (a ratio to a null or negative slope says nothing about "degrades slower") and
   H2 rests entirely on the Q-tests.
3. Verdict on this Register's grid: modes (i) and (ii) — `slope_U` is 0 or negative, so the ratio
   is **undefined and will be reported as such**, not computed. Mode (iii) — two grid points; a
   two-point slope has no bootstrap CI worth the name and the ratio is likewise reported, not
   gated. No mode on this grid has a gateable slope ratio. This is a consequence of §6, recorded
   now so it cannot be re-read later.

### 5.6 Per-candidate Guide overhead — the §7.1 flatness check

Measured in step 3 from `GuidedEpisodeStats` (to be added to `GuidedSaturation`, design §8; it
does not exist yet, §9): **scored candidates per recorded application at B**, per (mode, |R|).
Threshold, pre-committed from the design: at every grid point of every mode, scored-per-application
must be **≤ 2× its value at |R| = 62** in the same mode. Growth beyond 2× at any point flags the
Guide's per-round cost as a real scaling problem regardless of what H2 says. Raw matches per
application are expected to grow ~|R|/62 and are reported, not gated. The unguided analogue is
already visible: median applications-to-quiescence grows 2686 → 3115 → 4274 → 4803 → 5322 in
mode (i) (~2× at 4× the rules, classical), the sweep's own cost.

## 6. H1 verdict on this grid — recorded now, from unguided data

Per design §1.2, H1 is entirely an unguided measurement and its verdict is recorded before any
guided run. Both parts must hold; the tables are §4.

| Mode | grid | B | U(\|R\|) (%) | Spearman ρ | U(max) − U(62) | Δ1 | direction | effect | **H1** |
|---|---|---:|---|---:|---:|---:|---|---|---|
| (i) | 62, 93, 124, 186, 248 | 100 | 96.585 ×5 | undefined (all tied) | +0.000 | 28.230 | fails | fails | **FAILS** |
| (i) | same | 200 | 40.489 ×5 | undefined (all tied) | +0.000 | 12.342 | fails | fails | **FAILS** |
| (ii) | 62, 93, 124 | 100 | 96.585, 92.322, 92.322 | −0.866 | −4.262 | 28.230 | fails | fails | **FAILS** |
| (ii) | same | 200 | 40.489, 39.966, 39.966 | −0.866 | −0.522 | 12.342 | fails | fails | **FAILS** |
| (iii) | 62, 95 | 100 | 96.585, 108.599 | +1.0 (2 points: degenerate) | +12.014 | 28.230 | holds (trivially) | fails | **FAILS** |
| (iii) | same | 200 | 40.489, 46.945 | +1.0 (degenerate) | +6.456 | 12.342 | holds (trivially) | fails | **FAILS** |

**H1 fails in every mode on this grid.** The honest fallback the design pre-registered (§6,
"Honest fallback") applies: *the deliverable for H1 is this finding plus the fidelity table.* The
grid is not tuned after the fact; a new grid is a superseding registration. What the data say,
precisely — each point below is a measurement, not an interpretation added after the fact:

1. **At B = 100, no classical expression's cost differs from |R| = 62 in any mode (`visible@100`
   = 0 / 188, all seven inflated sets).** At B = 200 it is 0–3 of 188 in modes (i)/(ii) and 18 in
   mode (iii). The mechanism is the sweep order (§1): inflation is appended at indices ≥ 62 and
   the sweep is rule-major, so an appended rule cannot fire until all 62 production rules have
   swept once — and on a classical expression the first pass over the 62 alone produces more
   than 100 applications (median `app_actual` at the B = 100 checkpoint is 113, i.e. the
   checkpoint lands *inside the first sweep of the production prefix*). **The registered budgets
   are below the point at which any appended rule is reached.** This is a capacity measurement:
   at |R| ≤ 248 with prefix-appended inflation, budget B ∈ {100, 200} is not binding *on the
   inflation* — it is binding on the same 62-rule first sweep at every |R|.
2. **Mode (i) is null at every checkpoint, not only at B.** For `dup:124`, `dup:186`, `dup:248`
   the cost curve is bit-identical to |R| = 62 at all 14 checkpoints for all 188 expressions
   (`first visible` = never), while applications-to-quiescence doubles. Copies only ever re-fire
   idempotently after the prefix has done all state-changing work in sweep 1, so the e-graph
   *content* at every between-rules boundary is the same and only the application count differs.
   The design predicted H1 "close to analytic" in this mode and said a failure "would indicate a
   harness bug"; the failure is instead the prediction's unstated premise — that copies consume
   budget before B — which the rule-major order falsifies. The harness is doing exactly what its
   contract says (determinism test, identical `base` rows across two binaries and two sessions).
   `dup:93` (even-index copies, half cycle) shows a 3-expression wobble at B = 200 and a few
   later checkpoints (extraction-heuristic sensitivity to rebuild interleaving; the closure gain
   is −, not +, for 4 expressions): reported, not a trend.
3. **Mode (ii)'s U *decreases* (96.585 → 92.322) with cost@100 per-expression identical to
   base — the shift is entirely the reference.** `ref_U(e, |R|)` moved for 2–4 expressions
   (closure gain negative: the composition set's best-any-checkpoint cost is slightly *worse*,
   `first visible` 800 / 400), and on a regret distribution this wide (p25 47%, p75 27,840%) a
   handful of moved references shift the median. That is a measured decrease of U under the
   registered statistic, and the design says a decrease in mode (ii) "is a genuine falsification
   of H1 for this mode and is reported as such" — it is, with the mechanism stated: it is a
   reference effect, not a search-efficiency effect, and it is below Δ1 in magnitude.
4. **Mode (iii)'s U *increases* (+12.0 points at B = 100) with cost@100 per-expression identical
   to base — also entirely the reference, in the other direction.** The new rules reach a cheaper
   form later on the curve for 121 of 173 expressions (closure gain median +0.74%, p90 +2.2%), so
   the same cost@100 is further from a lower `ref_U(e, 95)`. This is precisely the §4 subtlety
   the closure-aware reference exists to make visible ("the optimum moves under mode (iii)"): the
   regret column rises because fidelity rose, not because search got worse — and at B = 200 the
   batch *is* reached in 18 expressions and the median cost@200 is worse (7,648.5 vs 7,413.5),
   the first budget at which mode (iii)'s search cost shows. The effect (+12.0) is below Δ1
   (28.2), so H1 fails on the pre-committed test; the two columns (regret vs absolute cost /
   closure gain) are reported together as the design requires, never merged.

Blitz and rapid show the same pattern with no claim attached: cost@B flat in |R|, applications
to quiescence growing 4–6.5× in mode (i).

**What a superseding registration would have to change, stated as options and not chosen here:**
(a) budgets at or beyond the production prefix's first-sweep application count on classical
(where appended rules are reachable); (b) inflation interleaved into the sweep order rather than
appended (which changes the production-prefix invariant the design chose deliberately); (c) a
budget axis in *sweeps* or *matched candidates* rather than applications. Each is a different
experiment. None is adopted by this document.

## 7. What remains testable for H2 under this Register (guided runs not yet run)

H1's failure does not void H2; it changes what each part of H2 can mean, and that reading is
fixed here before any guided number exists:

- **H2 part 1 (direction, `Q(|R|_{k+1}) ≤ Q(|R|_k) + 0.005`) and part 3 (`Q(|R|) ≤ 1 − Y(|R|)` at
  every point)** are fully testable at every realized point. With unguided cost@B per-expression
  constant in |R| in modes (i)/(ii), `Q(|R|)` measures the Guide alone: whether more candidates
  per round (4× at |R| = 248 in mode (i), all worthless) degrade its ordering within the same
  100/200 applications. **This is exactly the "learnable overhead" claim of design §2.1**, and the
  pre-committed reading stands: flat `Q` = the Guide learned the copies are worthless; rising `Q`
  = it did not, pointing at per-index label scarcity (design §7.2).
- **H2 part 2 (`Q(62) − Q(|R|max) ≥ Δ2 = 0.02`)** requires the Guide's advantage to *grow*. In
  mode (i) that is impossible by construction (same closure, same useful candidates, plus copies),
  so part 2 cannot hold in mode (i) on this grid and **mode (i)'s H2 verdict is recorded as "part 2
  not attainable; parts 1 and 3 carry the mode"** — stated now, not after seeing Q. In mode (ii)
  compositions are shortcuts (one application for two), so the Guide *can* gain; part 2 is live.
  In mode (iii) the Guide can reach the new cheaper forms within budget where the unguided sweep
  cannot (it has not reached index 62 by B = 100); part 2 is live and is the fidelity arm's
  question.
- Thresholds in force at every point: `1 − Y` = 0.8368 (B=100) / 0.9101 (B=200) at every
  closure-preserving point; 0.8556 / 0.9276 at `new:95`. Round 1's anchor `Q(62)` = 0.537 / 0.696
  is re-measured in this round's harness on DEV classical (n = 334) and must reproduce within its
  bootstrap CI or the round stops (design §7.4).
- **Slope ratio:** undefined in every mode on this grid (§5.5); reported as such.

## 8. Gates (numbers filled in; wording per design §6)

**Accept gate (per mode):** H1 AND H2 hold on DEV classical (n = 334) at B = 100 with the
constants above, full per-expression distributions reported; publication additionally requires
FINAL's classical band (family-held-out, n ≥ 30). **On this grid no mode can pass the accept gate,
because H1 has already failed in every mode.** What step 3 can still produce is H2's verdict per
mode (§7) and the §5.6 overhead measurement — publishable as "H1 null at these budgets; Guide
scaling in |R| measured", never as the thesis.

**Kill gate (per mode):** H2 part 3 fails at any |R| point on DEV (`Q(|R|) > 1 − Y(|R|)`) after
one clean re-mint/re-train at that point → stop for that mode and record it: the Guide as built
does not hold Round 1's claim as rules are added.

**Honest fallback (in force for H1):** §6 is the H1 deliverable — the capacity finding with the
absolute-cost and closure-gain columns. Nothing here permits extending the grid after the fact.

## 9. Protocol prerequisites for step 3 (must exist before any guided run at |R| > 62)

None of the following exists on the branch at this commit (checked by grep, §10); each is design
§8's API and is a hard requirement, not a nicety:

1. `gen_strict_labels --rule-set <spec>` minting under `build_rule_set(spec)`, writing the
   fingerprint into the report JSON and the JSONL header; labels re-minted at EVERY |R| point of
   every mode from unguided saturation under that exact rule set (design §7.3); no label file
   reused across points or modes.
2. `train_guide` refusing a mixed-fingerprint dataset and writing the fingerprint into the
   checkpoint; `LinearCandidateGuide::load` / `PerRuleRateGuide::from_train_guide_report` taking
   the expected fingerprint and erroring on mismatch; the `w_rule` table with exactly |R| entries.
3. `GuidedEpisodeStats` on `GuidedSaturation` (raw matches, deduped, scored, applications, rounds)
   for §5.6.
4. A curves binary whose guided arms reuse `run_anytime_curve_with` unchanged and whose Register
   mode refuses any guided arm — the existing `phase3_round2_unguided_curves` is that Register mode
   today; the guided arms are additive.
5. The |R| = 62 guided anchor re-run in the same harness (design §7.4).

Every guided artifact carries the rule-set fingerprint from §2, and the analysis refuses any row
whose fingerprint is not in §2's table.

## 10. Proof that no guided run at |R| > 62 exists at this commit

Run on `claude/phase3-round2` at the commit that adds this document:

- `git ls-files | grep -i round2` lists exactly: the design, this registration, the two curve
  binaries, `round2_rules.rs`, the two unguided CSVs, the compositions JSON, the stats JSON /
  tables, and the results narrative. No checkpoint, label dataset, or eval JSONL mentions a
  Round 2 rule set.
- `git grep -l -E 'dup:[0-9]+|comp:[0-9]+|new:95|experimental_rules|build_rule_set'` outside the
  three harness source files and the design doc hits only the two unguided CSVs, the two unguided
  binaries, the results narrative, and `math/mod.rs` (module declarations).
- Every inflated-set fingerprint (`fdd617246eb98590`, `87fefd5a63575175`, `37a4c537606a549b`,
  `809a0f52b61fe6c0`, `0c3fd6f35f444a59`, `521798ae521a0572`, `4f4a4cbd2e4f89cb`) occurs in
  `docs/results/` only in the two unguided CSVs.
- The only Guide checkpoint/label artifacts on the branch (`2026-09-01-strict-label-dataset.json`,
  `2026-09-01-train-guide-report.{json,md}`, `2026-09-01-phase3-at-budget-eval.*`) are Round 1's,
  at |R| = 62, and predate the Round 2 harness.
- Neither Round 2 binary links a Guide: `phase3_round2_unguided_curves` and
  `phase3_round2_new_rules` import no `nnue::guide` symbol (grep for `guided|checkpoint|
  LinearCandidateGuide|PerRuleRateGuide` in both files matches only the doc comments saying so).

## 11. Results appended against the gates

(Append-only. Nothing above this line changes. First entry: the `comp:186`/`comp:248` unguided
rows, if and when they exist, with their §4 table rows, Y, Δ2, and the mode (ii) H1 row over the
full 5-point grid — before any guided run at those points.)

## 12. Reproduction

```bash
# unguided curves, modes (i)/(ii) — the full nine-set grid in one invocation (comp:186/248 pending)
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.csv \
    --out-json docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.json
# unguided curves, mode (iii)
cargo run --release -p pixelflow-pipeline --features training --bin phase3_round2_new_rules -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-md /dev/null --out-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-mode-iii.csv
# composition census (ordered prefix + oracle status; JSON lines on stderr)
cargo test --release -p pixelflow-search composition_prefix_oracle_census -- --nocapture
# every registered number
python3 pixelflow-pipeline/scripts/round2_register_stats.py \
    --csv docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.csv \
    --csv docs/results/2026-09-01-round2-unguided-vs-rulecount-mode-iii.csv \
    --expect base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,new:95 \
    --modes 'i=base,dup:93,dup:124,dup:186,dup:248;ii=base,comp:93,comp:124;iii=base,new:95' \
    --out-json docs/results/2026-09-01-phase3-round2-registration.json \
    --out-md docs/results/2026-09-01-phase3-round2-registration-tables.md
```
