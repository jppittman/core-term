# Phase 3 round 1b pre-registration: does the Guide's advantage survive domain shift toward trig-dominant kernels?

**Date:** 2026-09-01
**Status:** REGISTERED — committed before any stratified or out-of-distribution evaluation
exists. No number in §1–§6 may be revised after the first such run; results are appended in
§7 against the gates below. A different statistic, threshold, family, or size band is a new
registration recorded as superseding this one.
**Authority:** `docs/plans/2026-09-01-phase3-registration.md` (Round 1, unchanged and still
binding: budgets in rule applications, the one curve definition
`egraph::anytime::run_anytime_curve`, deterministic cost regret under
`CostModel::latency_prior()`, no timing in any metric, FINAL untouched, the registered
classical tiers B = 100 (primary) / 200 (secondary)).
**Model under test:** `pixelflow-pipeline/data/guide_checkpoint_strict_v1.json` **exactly as
Round 1 trained it**. No retraining, no checkpoint change, no label change. This round evaluates
the Round-1 model; it does not improve it.
**Data sources (all pre-existing, none guided-OOD):**
- `docs/results/2026-09-01-phase3-at-budget-eval.{md,json,jsonl}` (Round 1, PR #1084) — the
  DEV-overall reference medians and the within-DEV variability M is fixed from.
- `docs/results/2026-09-01-train-guide-report.{md,json}` — per-rule TRAIN/DEV strict-positive
  rates (the control arm's whole model).

## 0. The question

JP, 2026-09-01, verbatim: *"The pythagorean identities are useless when trying to evaluate
bezier curves, but i sure as shit want them firing when we're doing spherical harmonics."*

Round 1 found the registered claim holds on DEV (linear Guide median ratio 0.537 vs
unguided-at-100), but the control arm — `PerRuleRateGuide`, a per-rule lookup of GLOBAL TRAIN
strict-positive rates with no expression context — reaches 0.565. The median is carried by a
global prior. On this corpus every trig identity has a strict-positive rate between 0% and 1%
(§4), so the global prior says "never fire trig identities". That is right on average and wrong
on exactly the kernels JP names. The candidate-local linear model's `neighborhood_op_hist`
feature (`pixelflow-search/src/egraph/candidate.rs::neighborhood_ops` — the one-hop child ops of
the matched class, `log1p(count)` per `OpKind`) is the only thing in either arm that can see
that a `half-angle-product` match sits inside a sea of `Sin`/`Cos`/`Mul` rather than a sea of
`Add`/`Mul`. Whether it learned to use that is the question.

## 1. Hypotheses and the pre-committed statistic

Notation. For arm A ∈ {control, linear} and registered tier B ∈ {100, 200}:

- `m_A^DEV(B)` = Round 1's median over all 334 DEV classical expressions of
  `cost_A@B / cost_unguided@B`. Quoted from
  `docs/results/2026-09-01-phase3-at-budget-eval.json` (`tiers[].arms[].ratio_vs_unguided_at_b.median`):

  | B | control | linear |
  |---:|---:|---:|
  | 100 | **0.5655** | **0.5366** |
  | 200 | 0.6991 | 0.6959 |

- `m_A^S(B)` = the same median computed on an evaluation set S (a DEV stratum from §2 or an
  OOD family from §3), by the same harness, same arms, same grid, same cost model.
- **`D_A(S, B) = m_A^S(B) − m_A^DEV(B)`** — how much worse (positive) or better (negative) arm A
  does on S than on DEV overall. A ratio is already normalised against unguided-at-B on the same
  expression, so D is a shift in the arm's *advantage*, not in absolute cost.

**H_shift (JP's reading):** under shift toward trig-dominant expressions, the per-rule control's
advantage collapses or inverts while the candidate-local linear model degrades less.
**Accepted at tier B on set S iff `D_control − D_linear > M_B` AND `D_control > 0`.**

**H_null (the corpus-coverage reading):** both arms move together — the corpus taught the linear
model the same global prior through its features, and the fix is training-distribution coverage,
not architecture. **Accepted iff `|D_control − D_linear| ≤ M_B`** (whatever the common sign of D).

**H_inv (named now so it cannot be invented later):** `D_linear − D_control > M_B` — the
neighborhood feature actively misleads under shift. Accepting it means the linear model's tail
advantage on DEV was not context-awareness.

**Primary test:** S = the `sh` family (§3a), B = 100. **Secondary tests, reported alongside:**
S = `sh` at B = 200; S = the DEV `trig-heavy` stratum (§2) at both tiers. A verdict is claimed
only on a set with n ≥ 30 classical expressions; smaller sets are reported as underpowered.

### 1.1 The margin M, fixed from Round 1's own within-DEV variability

Three candidate scales were computed from the Round-1 per-expression rows
(`2026-09-01-phase3-at-budget-eval.jsonl`, classical, ratio vs unguided-at-B), before any OOD
data existed:

| scale | B=100 | B=200 | source |
|---|---:|---:|---|
| half the per-expression IQR of the ratio (control / linear) | 0.347 / 0.338 | 0.239 / 0.244 | `ratio_vs_unguided_at_b.{q1,q3}`: control q1 0.0072, q3 0.7005; linear q1 0.0041, q3 0.6809 |
| bootstrap SE of the overall median (n = 334, 2,000 resamples) | 0.018 / 0.018 | 0.019 / 0.019 | recomputed from the jsonl |
| **largest \|D_control − D_linear\| across the 8 DEV families** (f00–f07, n = 35–46 each, no shift) | **0.057** (f03) | **0.063** (f00) | recomputed from the jsonl by family label `dev_bXX_fYY` |

Half the per-expression IQR is the scale the task suggested and it is rejected here, with the
number stated: it is 0.35 because q1 ≈ 0.005 — a quarter of DEV classical expressions are ones
where unguided-at-100 is catastrophically bad and either Guide's ratio is near zero — so it
measures the truncation-loss tail, not the noise in a difference of subset medians, and would
make H_shift unfalsifiable at any plausible effect size. The statistic being tested is a
*family-level* difference of medians, so its natural null scale is how much that same
statistic already swings between DEV families with **no** domain shift. The largest such swing
on the books is 0.057 at B=100 and 0.063 at B=200. Rounded up to the next 0.01:

**M_100 = 0.06, M_200 = 0.07.**

An effect must exceed the largest within-distribution family swing Round 1 already produced.
This is deliberately conservative (a max over 8, not a quantile); it is ~3× the bootstrap SE of
a full-DEV median and ~1.5–2× what a subset of 60–100 expressions would show.

### 1.2 The Bézier prediction (pre-committed)

On the `bezier` family (§3b) and on the DEV `polynomial-only` stratum (§2), the global prior is
right — there are no trig identities to suppress — so **both arms are predicted to be at or
better than DEV-overall: `D_A ≤ 0` (point prediction) and in any case `D_A ≤ +M_B` for both
arms.** If either arm shows `D_A > M_B` on `bezier`, its DEV benefit was not (only) the global
rule prior: something in the polynomial shape — most plausibly the single-firing dedup closure
losing to `associative`/`distribute`/`fma-fusion` re-fires that polynomials need many of — is
binding, and that is a Round-2 finding about the dedup design, not about rule ordering.

### 1.3 Descriptive diagnostics (reported, not gated)

On every set: per-arm count of applications of the trig rules (§4, rule indices 30, 31, 34,
36–40) within the first B, unguided's count of the same, strict precision at B, and the
head-to-head counts (linear < / = / > control). On `sh` specifically: the number of expressions
on which `pythagorean` (idx 40) is applied at all by any arm at any checkpoint — Round 1 never
observed it firing (§4), so whether the family even *offers* the match is a precondition the
report must state before interpreting D.

## 2. DEV stratification by op composition (mechanical rule)

Applied to every DEV classical expression already evaluated in Round 1 (the 334 rows of the
jsonl; no re-evaluation is needed for the stratified numbers — only the per-expression op
counts, read from `corpus_dev.bin`). Op groups over `pixelflow_ir::OpKind`:

| group | ops |
|---|---|
| TRIG | `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Atan2` |
| TRANS | `Exp`, `Exp2`, `Ln`, `Log2`, `Log10`, `Pow` |
| ROOT | `Sqrt`, `Rsqrt`, `Recip` |
| POLY | `Add`, `Sub`, `Mul`, `MulAdd`, `Neg` |

Let `ops` be the multiset of `OpKind`s over the arena's non-leaf nodes (leaves `Var`, `Const`,
`Param`, `Buffer` excluded); `n_nodes` is the same node count the harness already writes
(`node_count`, arena length). The stratum is the **first** matching row:

1. `polynomial-only` iff every op ∈ POLY.
2. `trig-heavy` iff **|TRIG ops| ≥ 3** (absolute count).
3. `transcendental-heavy` iff |TRANS ops| ≥ 3.
4. `sqrt-recip-heavy` iff |ROOT ops| ≥ 3.
5. `mixed` otherwise.

The one rule for trig is the absolute count, not the 15%-of-nodes alternative, and the reason
is recorded: every trig identity in the library needs ≥ 2 trig nodes to match
(`sin·cos`, `sin² + cos²`, `sin·cos + cos·sin`), so the *count* is what determines whether the
suppressed rules have anything to fire on; a 15% threshold on a > 50-node classical expression
demands ≥ 8 trig nodes, which the ShaderToy-weighted generator (trig ops are 21/187 ≈ 11% of
draws, `pixelflow-search/src/nnue/mod.rs::shader_weight`) rarely produces, and would leave the
stratum underpowered by construction. Stratum sizes are unknown at registration time and are
reported with the results; any stratum with n < 30 is underpowered and gets no verdict. The
DEV strata carry a caveat the OOD families do not: a generator-drawn "trig-heavy" expression
has ≥ 3 trig nodes with *random* arguments, so the identities may still have nothing to match
— which is exactly why §3a exists.

## 3. Out-of-distribution families (DEV-only, never TRAIN)

Both are **named** families entered through the split manifest, the way the named production
kernels enter FINAL (`corpus_split.toml` `[final] kernels`): a new `[dev] families = ["sh",
"bezier"]` key, with `SplitManifest::validate` extended to reject a `families` key under
`[train]` outright and a named family appearing in more than one tier. The fence is checked in
the direction that matters for a model that is already trained: every OOD expression's
`FenceKey` is probed against `corpus_train.bin`'s keys and **any** collision is a hard, named
error (a structural duplicate of a TRAIN expression is a leak, not a hygiene event). The
numeric quarantine applies unchanged. Names are `dev_sh_NNNNN` / `dev_bezier_NNNNN`.

**Size band (both families):** node count > 50 (the registered classical band, so B = 100/200
apply unchanged), targeting 51–400 nodes. **Count (both families):** 60–100 distinct
expressions after structural dedup and quarantine, from seeded coefficient/degree/form
variation; fewer than 30 survivors is a failed generation, not a small sample.

### 3a. `sh` — real spherical harmonics in spherical coordinates

**Parameterisation (fixed):** θ = `X` (`Var(0)`), φ = `Y` (`Var(1)`) — the coordinate variables
*are* the angles. Not the Cartesian route (normalise a direction, then polynomials in
nx, ny, nz, as `pixelflow-ml/src/graphics.rs::project` does): that form contains no trig at
all and would not test the hypothesis; and not `atan2`/`acos` of a direction, which adds
inverse-trig ops no rule in the library addresses and hides the `sin`/`cos` structure the
identities match on.

**Basis (real SH in spherical coordinates, Condon–Shortley phase omitted — the graphics
convention of Green, "Spherical Harmonic Lighting: The Gritty Details" (2003), §"The Real
Spherical Harmonics", y_l^m = √2·K_l^m·cos(mφ)·P_l^m(cosθ) for m > 0,
√2·K_l^m·sin(|m|φ)·P_l^{|m|}(cosθ) for m < 0, K_l^0·P_l^0(cosθ) for m = 0; the rows below are
those products with the associated Legendre polynomials written out, matching the standard
table of real spherical harmonics through l = 4; the Cartesian polynomial forms in Sloan,
"Stupid Spherical Harmonics Tricks" (2008) Appendix A2 are the same functions and are
deliberately NOT used, per the parameterisation note above):**

| l | m | Y_l^m(θ, φ) |
|---:|---:|---|
| 0 | 0 | ½·√(1/π) |
| 1 | −1 | √(3/4π)·sinθ·sinφ |
| 1 | 0 | √(3/4π)·cosθ |
| 1 | 1 | √(3/4π)·sinθ·cosφ |
| 2 | −2 | ¼·√(15/π)·sin²θ·sin2φ |
| 2 | −1 | ½·√(15/π)·sinθ·cosθ·sinφ |
| 2 | 0 | ¼·√(5/π)·(3cos²θ − 1) |
| 2 | 1 | ½·√(15/π)·sinθ·cosθ·cosφ |
| 2 | 2 | ¼·√(15/π)·sin²θ·cos2φ |
| 3 | −3 | ¼·√(35/2π)·sin³θ·sin3φ |
| 3 | −2 | ¼·√(105/π)·sin²θ·cosθ·sin2φ |
| 3 | −1 | ¼·√(21/2π)·sinθ·(5cos²θ − 1)·sinφ |
| 3 | 0 | ¼·√(7/π)·(5cos³θ − 3cosθ) |
| 3 | 1 | ¼·√(21/2π)·sinθ·(5cos²θ − 1)·cosφ |
| 3 | 2 | ¼·√(105/π)·sin²θ·cosθ·cos2φ |
| 3 | 3 | ¼·√(35/2π)·sin³θ·cos3φ |
| 4 | −4 | 3/16·√(35/π)·sin⁴θ·sin4φ |
| 4 | −3 | ¾·√(35/2π)·sin³θ·cosθ·sin3φ |
| 4 | −2 | ⅜·√(5/π)·sin²θ·(7cos²θ − 1)·sin2φ |
| 4 | −1 | ¾·√(5/2π)·sinθ·(7cos³θ − 3cosθ)·sinφ |
| 4 | 0 | 3/16·√(1/π)·(35cos⁴θ − 30cos²θ + 3) |
| 4 | 1..4 | as m = −1..−4 with cos(mφ) in place of sin(mφ) |

Numeric normalisation constants are baked as `Const` (the folder handles them; they are not
what is under test).

**Forms (seeded, each expression draws one):**

- `sh-direct`: the multiples written as `Sin(m·φ)` / `Cos(m·φ)` (a `Mul(Const(m), Y)` argument).
  Angle addition can only reach these through `doubling`/`halving` (rule idx 20/21,
  `2x ↔ x + x`); this is the form where the *enabler* rule matters.
- `sh-expanded`: the multiples expanded in sinφ, cosφ — sin2φ = 2·sinφ·cosφ,
  cos2φ = cos²φ − sin²φ, sin3φ = 3sinφ − 4sin³φ, cos3φ = 4cos³φ − 3cosφ,
  sin4φ = 4·sinφ·cosφ·(2cos²φ − 1), cos4φ = 8cos⁴φ − 8cos²φ + 1 — the form in which
  `half-angle-product` (sinφ·cosφ → sin2φ/2), `reverse-angle-addition`, and `pythagorean`
  (cos²φ − sin²φ with a shared sin²φ + cos²φ elsewhere) have live matches.
- `sh-power`: band energies Σ_m (Y_l^m)² for l = 1 and l = 2 (the rotation-invariant quantity SH
  lighting actually uses), e.g. l = 1: (3/4π)·(sin²θ·sin²φ + cos²θ + sin²θ·cos²φ) — the purest
  `pythagorean` bait in the set, reducible to a constant.

**Expressions:** for each seeded draw, L ∈ {2, 3, 4}, coefficients c_lm ~ U(−1, 1), and the
expression is either (i) the SH-basis dot product f(θ, φ) = Σ_{l ≤ L} Σ_m c_lm·Y_l^m(θ, φ), or
(ii) the product of two independent such sums with L ≤ 2 (an irradiance-style
lighting × transfer product), or (iii) a band energy (sh-power) added to a dot product so it
clears the size band. Node counts: an L = 2 direct sum is ~50–70 nodes, L = 3 ~100–150,
L = 4 ~200–350, expanded forms larger; draws below 51 nodes are discarded and reported.

### 3b. `bezier` — Bézier / de Casteljau evaluation, polynomial-only

**Parameterisation (fixed):** curve parameter t = `X`; the second coordinate `Y` enters as a
point coordinate (squared-distance forms) or as the second patch parameter. Every expression
is POLY-only (`Add`, `Sub`, `Mul`, `MulAdd`, `Neg` plus `Const`/`Var`): no `Div`, no `Sqrt`
(so "distance" is squared distance), so the family lands in the `polynomial-only` stratum by
construction and no trig or transcendental rule can match anything in it.

**Forms (seeded, each expression draws one):**

- `bezier-bernstein`: degree n ∈ {3, 4} curve in Bernstein form,
  B(t) = Σ_i C(n,i)·(1−t)^(n−i)·t^i·P_i, control points P_i ~ U(−2, 2)² (cubic
  1,3,3,1; quartic 1,4,6,4,1), evaluated as the squared distance from the point (`Y`, c₀) to
  the curve point: E = (B_x(X) − Y)² + (B_y(X) − c₀)². ~60 nodes cubic, ~85 quartic.
- `bezier-casteljau`: the same E with each component evaluated by nested lerps
  `lerp(a, b, t) = a + (b − a)·t` (6 lerps for cubic, 10 for quartic), the form `fma-fusion`
  and `factor`/`distribute` have the most to do with.
- `bezier-patch`: tensor-product bicubic patch z(X, Y) = Σ_i Σ_j P_ij·B_i(X)·B_j(Y), 16 control
  heights ~ U(−2, 2), Bernstein form in both parameters; ~150–200 nodes.

## 4. The rules the global prior suppresses

`pixelflow_search::egraph::all_rules()` = `math::all_rules()`: algebra (idx 0–29), parity
(30–35), trig (36–40), exp (41–46), power (47–58), fusion (59–60), derivative (61). Trig-related
rules and their Round-1 strict-positive rates (`2026-09-01-train-guide-report.json`,
`per_rule[]`; a rule absent from that table fired zero times in both splits):

| idx | rule | form | TRAIN fired / rate | DEV fired / rate | control-arm score |
|---:|---|---|---:|---:|---:|
| 30 | `odd-negation` (Sin) | sin(−x) → −sin x | 568 / 1.41% | 182 / 0.55% | 0.0141 |
| 31 | `odd-negation` (Tan) | tan(−x) → −tan x | 309 / 0.32% | 97 / 1.03% | 0.0032 |
| 32 | `odd-negation` (Asin) | asin(−x) → −asin x | 0 / — | 0 / — | 0.0 |
| 33 | `odd-negation` (Atan) | atan(−x) → −atan x | 0 / — | 0 / — | 0.0 |
| 34 | `even-negation` (Cos) | cos(−x) → cos x | 2,440 / 5.53% | 641 / 6.24% | 0.0553 |
| 36 | `sin-angle-addition` | sin(a+b) → sin a·cos b + cos a·sin b | 381 / 0.26% | 109 / **0.00%** | 0.0026 |
| 37 | `cos-angle-addition` | cos(a+b) → cos a·cos b − sin a·sin b | 340 / 0.88% | 103 / **0.00%** | 0.0088 |
| 38 | `reverse-angle-addition` | sin a·cos b + cos a·sin b → sin(a+b) | 563 / 1.07% | 134 / 0.75% | 0.0107 |
| 39 | `half-angle-product` | sin x·cos x → sin(2x)/2 | 221 / 0.45% | 110 / **0.00%** | 0.0045 |
| 40 | **`pythagorean`** | sin²x + cos²x → 1 | **0 / —** | **0 / —** | **0.0** |
| 20 | `doubling` (enabler) | x + x ↔ 2x — the only path from `Sin(2·φ)` to angle addition | 6,616 / 0.50% | 1,222 / 0.66% | 0.0050 |

(idx 35, `even-negation` on `Abs`, shares a name with 34 but is not trig and is excluded.)

For scale: the control arm's top-scored rules are `power-rsqrt` 0.181, `power-recip` 0.161,
`recip-sqrt` 0.165, `power-sqrt` 0.135, `even-negation` 0.055 — every trig identity is scored
10–60× below them, and `pythagorean` — the rule the question is literally about — has a
control-arm score of exactly 0.0 because it never fired in 4,143 expressions, so it sorts with
`commutative` and `distribute` at the bottom of every candidate list. In the linear model its
rule-identity weight received no gradient either; whatever it scores under shift comes entirely
from the shared neighborhood/budget/size features. That is the sharpest possible version of the
question, and the `sh-power` form exists to ask it.

## 5. Accept / kill: what each outcome means

| outcome (primary: `sh`, B=100) | reading | consequence for Round 2 | consequence for training-distribution design |
|---|---|---|---|
| **H_shift** (`D_control − D_linear > 0.06`, `D_control > 0`) | The per-rule prior is a DEV artefact; candidate-local context is load-bearing off-distribution even though it is not at the DEV median. | The linear architecture stays; Round 2 is the tightened-label stage (design decision #1, option 3) on the same features. The ladder's control arm is re-weighted: it is a *DEV* baseline, not a general one, and every future round reports the OOD gap alongside the DEV median. | Coverage still matters but is not the fix: add `sh`-like structure to TRAIN (a new TRAIN family, disjoint seeds, `bezier`/`sh` themselves stay DEV) to lift the trig rules' base rates, and expect the linear model to gain more from it than the control. |
| **H_null** (`\|D_control − D_linear\| ≤ 0.06`) | The features learned the global prior. The neighborhood histogram had no trig-positive examples to learn from (§4: ≤ 1% positives, `pythagorean` zero), so it could not. | Architecture is not the lever; do not add capacity. Round 2 = Round 1's recipe on a corpus whose TRAIN families include trig-structured expressions, then re-run this registration's test unchanged. If H_null still holds *after* coverage, that is the architecture verdict. | The corpus is the object of design: the generator's independent op draws never produce `sin(a+b)`, `sin x·cos x`, or `sin² + cos²` with *shared arguments*; a structure-aware family (SH, rotations, Fourier sums) is required in TRAIN, and the per-rule positive rates in §4 are the acceptance metric for it (target: every trig rule with ≥ 100 TRAIN positives). |
| **H_inv** (`D_linear − D_control > 0.06`) | The neighborhood feature is actively harmful off-distribution — it learned a spurious correlate of the DEV tail. | Round 2 must ablate: re-evaluate the checkpoint with the op-histogram features zeroed (a masking pass at inference, no retraining) to confirm the feature is the cause before anything else is changed. | Same coverage fix as H_null, plus a held-out *family-of-structure* (not just family-of-seed) split rule going forward. |
| both arms **improve** on `sh` (`D_A < −0.06` for both) | Trig-dominant kernels are *easier* at budget B than DEV overall (fewer live rewrites), regardless of the prior. | The question was ill-posed at this budget; report it and move the shift test to the tier where `sh` has truncation loss. | No change. |
| Bézier prediction fails (`D_A > 0.07` for either arm) | The DEV benefit was not only the rule prior. | Dedup-closure investigation (Round 1 §"Where the residual regret lives") is promoted ahead of labels. | No change. |

Kill semantics are unchanged from Round 1: this round is an evaluation and cannot fire or
advance the 5-clean-rounds kill gate. It can only redirect Round 2 as tabled above.

## 6. Execution plan (what gets built, in order; none of it is evaluation)

1. `SplitManifest`: `[dev] families = [...]`; validator rejects `families` under `[train]` and
   cross-tier duplicates. Unit tests for both rejections.
2. Two builders, `sh` and `bezier`, producing `(name, ExprArena, ExprId)` from a seed; entered
   through the same fence + quarantine path the named FINAL kernels use; TRAIN-fence collision
   is a hard error. Written to a separate `corpus_dev_ood.bin` so Round 1's `corpus_dev.bin`
   MD5 (`3026133ebba066eeca10f658da554400`) is untouched.
3. Stratifier: the §2 rule as one function over an arena, with a test per row of the rule.
4. `phase3_at_budget_eval`: a `--corpus` file flag (default unchanged), a `--strata` output
   grouping the per-expression jsonl by §2 stratum, and the §1.3 trig-rule counts per arm.
   Same arms, same grid, same checkpoint, same report JSON — plus a per-set table of
   `m_A^S`, `D_A`, `D_control − D_linear`, n, and the §1 verdict against `M_B`.
5. Run; append §7 here; write `docs/results/2026-09-01-phase3-round1b-domain-shift.md`.

## 7. Results appended against the gates

(none yet — appended only after the run, per this document's own rule)

### 7.1 Run of 2026-09-01 (`docs/results/2026-09-01-phase3-round1b-domain-shift.md`)

Checkpoint unchanged; DEV re-run reproduces Round 1 exactly (D_A^DEV = 0.0000 at both tiers).
`sh`: 95 classical of 100 generated; `bezier`: 80 / 80. Every set below has n ≥ 30.

| set | B | m_control^S | m_linear^S | D_control | D_linear | D_control − D_linear | M_B | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`sh` (primary)** | **100** | 0.9028 | 0.9039 | +0.3373 | +0.3673 | **−0.0300** | 0.06 | **H_null** |
| `sh` | 200 | 0.8940 | 0.8959 | +0.1949 | +0.2000 | −0.0051 | 0.07 | H_null |
| `bezier` | 100 | 0.9098 | 0.9098 | +0.3443 | +0.3732 | −0.0289 | 0.06 | H_null; §1.2 prediction **FAILS** (both arms, both forms) |
| `bezier` | 200 | 0.9098 | 0.8855 | +0.2107 | +0.1896 | +0.0212 | 0.07 | H_null; §1.2 prediction **FAILS** |
| DEV `trig-heavy` (287 / 334 — the stratum is DEV) | 100 | 0.5701 | 0.5399 | +0.0046 | +0.0033 | +0.0013 | 0.06 | H_null |
| DEV `trig-heavy` | 200 | 0.6724 | 0.6654 | −0.0267 | −0.0305 | +0.0038 | 0.07 | H_null |
| DEV `transcendental-heavy` (47) | 100 | 0.5089 | 0.5197 | −0.0566 | −0.0169 | −0.0398 | 0.06 | H_null |
| DEV `transcendental-heavy` | 200 | 1.0000 | 1.0000 | +0.3009 | +0.3041 | −0.0032 | 0.07 | H_null |

**Primary verdict: H_null.** H_shift and H_inv rejected on every set. The §1.2 Bézier prediction
fails for both arms by the same +0.34 as `sh` moved: the shift is the unguided baseline's regime on
structured kernels (flat, unconverged curve — truncation loss +0.18% / +5.6% between B and 4B against
20% / 30% regret; production quiesces at ~11,000 applications), not rule suppression. Both guided arms
still improve every OOD expression and beat unguided-at-4B at B=100 on both families, ending the
800-application grid un-quiesced with 8–18% regret left.

§1.3 diagnostics on `sh`: `half-angle-product` fires under BOTH Guides within the first 100
applications (393 / 360 pooled, 32.3% / 37.2% strict-positive — the best-paying rule of the prefix;
unguided fires it 0 times in its first 200) — the per-rule prior is an ordering, not a filter, and
0.0045 > 0.0 (structural) already ranks it first on a trig-dense pool. `pythagorean` matches on 19 `sh`
expressions (unguided: 231 firings), is fired by neither Guide (score exactly 0.0), and is
strict-positive 0 / 231 even for unguided (0 / 225 on DEV) — referred to the labeling stream as a
possible constant-output blind spot of the strict label. Consequence per §5: H_null row (coverage, not
capacity) **plus** the Bézier-fails row (dedup-closure / budget-regime investigation promoted ahead of
labels).
