# Pre-registration: does a rule-by-context interaction buy a domain-conditional advantage the additive Guide cannot represent?

**Date:** 2026-09-02
**Status:** REGISTERED — committed before any bilinear Guide, any training run, or any
guided evaluation of one exists. No number in §3–§9 may be revised after the first guided
run under this registration. Results are appended in §11 against the gates below. A
different statistic, threshold, family, or size band is a NEW registration recorded as
superseding this one.

**Authority (inherited, unchanged, still binding):**
- `docs/plans/2026-09-01-phase3-registration.md` — budgets in **rule applications**
  (`Budget::Applications` / `Optimizer::guide`), the one curve definition
  (`egraph::anytime::run_anytime_curve`), deterministic cost under
  `CostModel::latency_prior()`, **no timing in any metric**, FINAL untouched, the
  registered classical tiers B = 100 (primary) / 200 (secondary), and §7's 5-clean-round
  training kill gate.
- `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` — the statistic `D`,
  the margins **M_100 = 0.06 / M_200 = 0.07**, the `sh` and `bezier` OOD families, the
  family-held-out fence discipline.

Neither of those documents is edited by this one. This registration cites them and adds a
new claim; it does not move their gates.

**Instrument (changed since both, binding here):** every cost in this registration is
`ExtractedDAG::dag_cost` (`ChoiceCost::dag`, #1117), and the application budget binds
mid-scan (#1118). See `docs/results/2026-09-02-phase3-instrument-changes.md`. **The
registered reference medians in the round-1b registration were measured on tree cost and
are not reproducible under this instrument** — so every `m_A^DEV` used here is re-measured
in this round, under this instrument, for every arm including the frozen ones. §3.2 states
what that does to the inherited margin and how it is handled without loosening the gate.

---

## 0. Why this question is being re-asked

JP, 2026-09-01, verbatim: *"The pythagorean identities are useless when trying to evaluate
bezier curves, but i sure as shit want them firing when we're doing spherical harmonics."*

Round 1b tested exactly that and returned **H_null** on `sh` at B=100
(`docs/results/2026-09-01-phase3-round1b-domain-shift.md`: D_control = +0.337,
D_linear = +0.367, difference **−0.030** against M_100 = 0.06) — the control arm and the
candidate-feature arm moved together under domain shift. Its recorded reading was
"architecture is not the lever; do not add capacity."

This registration re-opens the question on one specific ground, established in §1 before
any new measurement: **the model class every one of those measurements ran is provably
incapable of representing the hypothesis it was testing.** That is a fact about
`LinearCandidateGuide`'s functional form, not a fact about the data, and it was not known
when round 1b was registered.

## 1. The representational result (established, not hypothesised)

`LinearCandidateGuide::score_candidates` computes `LinearWeights::logit`
(`pixelflow-search/src/nnue/guide/linear.rs`):

```text
s(r, x) = bias
        + w_rule[r]
        + w_budget       * budget_fraction
        + w_match_class  * ln(1 + match_class_node_count)
        + w_neighborhood * ln(1 + |neighborhood_ops|)
        + w_expr_size    * ln(1 + expr_node_count)
        + Σ_op w_op[op]  * ln(1 + count_op(neighborhood_ops))
```

Exactly one term names the rule, and it names nothing else. Every other term is a function
of the candidate's context `x` alone. The score is therefore **additively separable**,
`s(r, x) = w_rule[r] + g(x)`, and the rule-by-context second difference vanishes
identically:

```text
[s(r1, x) − s(r2, x)] − [s(r1, x') − s(r2, x')] = 0        for all r1, r2, x, x'
```

**What that forbids.** No context can reorder two rules against each other:
`w_rule[r1] − w_rule[r2]` is the entire answer, in every expression, at every point in
saturation. The model can say "this match site looks promising" — `g(x)` varies freely —
but it applies that judgement identically to every rule. It cannot say "fire the
Pythagorean identity *here* and not *there*" while leaving other rules where they were.
That sentence is JP's question, and it is outside the model class.

**What it still permits, stated so the claim is not overstated.** Two candidates in one
batch generally have *different* contexts (different match sites), so the additive model
can and does reorder *candidates*. What is fixed is the rule ranking *at a fixed context*,
and equivalently the second difference above. "One global rule ranking" means that, and
nothing stronger.

**What the budget term can and cannot do.** It is the one term that looks like it carries
search state, and under the live consumer it carries none. `egraph::guided` reads the
application ordinal **once per round**, before walking that round's matches, so every
`CandidateSummary` in one `score_candidates` batch shares a `budget_fraction`;
`expr_node_count` is likewise an episode constant. A term constant across a batch adds the
same number to every score, and the consumer sorts descending with no threshold anywhere.
The budget term moves the absolute score and cannot move a decision.

**The contrast.** `SaturationHead::score_candidate` ends in `bilinear_score`, which is
`m(x)ᵀ W r + bᵀ r`: linear in the rule embedding with **context-dependent coefficients**.
Its rule-pair difference is `(m(x)ᵀW + bᵀ)(r1 − r2)`, which depends on `x`. That is a
genuine interaction.

**Pinned as tests** in `pixelflow-search/src/nnue/guide/scoring/representation.rs`, all
five passing at registration time, all over the *same* two candidate sets through the
*same* `SaturationGuide::score_candidates` seam, so the only thing varying is the model
class:

| test | what it pins |
|---|---|
| `additive_guide_should_induce_the_same_rule_order_in_every_context` | the induced rule order is identical in a trig-shaped and a polynomial-shaped neighborhood (with a non-vacuity assertion that the contexts do reach the model) |
| `additive_guide_should_have_an_identically_zero_rule_by_context_interaction` | the second difference is zero, and equals `w_rule[r1] − w_rule[r2]` |
| `additive_guide_budget_term_should_shift_every_score_equally_and_reorder_nothing` | the budget term shifts every score by `w_budget·Δ` and reorders nothing |
| `bilinear_head_should_reorder_two_rules_between_two_contexts` | a hand-set weight matrix under which the same two rules **swap** between the two contexts; second difference nonzero |
| `both_model_classes_score_the_same_candidate_sets_through_one_trait` | both arms are `SaturationGuide`s over identical inputs |

The bilinear test uses **hand-set** weights. This is a statement about what the functional
form can express. It is **not** a claim that training finds such weights — that is exactly
what §3 registers and §7 gates.

## 2. What this implies about the prior results — and what it does not

Stated carefully, because the temptation to overclaim here is large.

**It does not invalidate round 1 or round 1b.** Both measured what they said they measured.
Round 1's linear-ties-control (0.537 vs 0.565 at B=100) and round 1b's H_null are correct
records of how `LinearCandidateGuide` behaved.

**It changes what those results are evidence *for*.** Round 1b's H_null was written as "the
corpus taught the linear model the same global prior through its features, and the fix is
training-distribution coverage, not architecture." §1 shows a second explanation is equally
consistent with the identical numbers: **the model could not have expressed a
context-conditional rule preference even if the corpus had taught it one.** Two hypotheses,
one observation:

- *H_context-does-not-matter*: rule value genuinely is context-independent at this budget,
  and a global prior is the right model.
- *H_model-cannot-use-context*: rule value is context-dependent, and the measured model
  class has zero capacity for it.

**The round-1b data cannot distinguish these**, because under the second hypothesis the
additive model produces exactly the H_null signature — both arms collapsing to a rule
prior — for a reason having nothing to do with the corpus. Consistency is not causation:
nothing here says the interaction *is* the missing piece, only that its absence was never
tested. That is the ambiguity this registration exists to resolve, in one direction or the
other.

Two nearby facts, recorded because they bear on the prior and are not part of the claim:

- Round 1's control arm (`PerRuleRateGuide`, a pure rule lookup) *beat* the candidate-feature
  arm at B=100. Under §1, the additive arm's rule ordering **is** a rule lookup — its extra
  features can only add a rule-blind offset. That the two arms nearly tie on the median is
  the predicted consequence of the functional form, not a surprise about the features.
- The re-run oracle curves on `dag_cost`
  (`docs/results/2026-09-02-oracle-filtered-budget-curves.md`) find rule-granularity oracle
  filtering **indistinguishable from unguided** through B=400 (delta 0.00 at every tier) —
  i.e. even a *hindsight-perfect* per-rule signal buys nothing. Anything a rule-ordering
  Guide can win must therefore be candidate-granular, which is the granularity §1 shows the
  additive model has no rule-conditional access to. Independently, that same re-run finds
  headroom **larger** than round 1 believed (classical median regret still 29.18% at B=800,
  reaching 0 at the median only near B=12,800), so a null here is not a null for want of
  room.

## 3. The claim and the pre-committed statistic

### 3.1 Notation (inherited verbatim from the round-1b registration §1)

For arm `A` and registered tier `B ∈ {100, 200}`:

- `m_A^S(B)` = the median over the expressions of set `S` of
  `dag_cost_A@B / dag_cost_unguided@B`, by the one harness, same arms, same grid, same cost
  model, same seed.
- **`D_A(S, B) = m_A^S(B) − m_A^DEV(B)`** — how much worse (positive) or better (negative)
  arm `A` does on `S` than on DEV-overall. Lower is better. `D_unguided ≡ 0` by
  construction.

Both `m_A^S` and `m_A^DEV` are measured **in this round, on `dag_cost`**, for every arm.
No round-1 constant is reused as a reference median.

### 3.2 The margin

The gate uses the **inherited, frozen** margins **M_100 = 0.06, M_200 = 0.07** (round-1b
registration §1.1: the largest `|D_control − D_linear|` across the 8 DEV families with no
shift, rounded up). Freezing them is what makes the gate ungameable after seeing data.

They were calibrated on the *tree-cost* instrument. That is stated, not hidden, and handled
by a pre-commitment that can only make acceptance **harder**: the same family-swing scale is
recomputed on this round's `dag_cost` per-expression rows (max over the 8 DEV families of
`|D_linear − D_bilinear|` with no shift) and **reported**. If the recomputed swing exceeds
the inherited `M_B`, the round is declared **underpowered at that tier and no verdict is
claimed** — the inherited margin is never raised to match, and never lowered.

### 3.3 Hypotheses

**H_repr (the claim): the bilinear arm shows a domain-conditional advantage the additive
arm cannot represent.** Accepted at tier `B` iff **all** of:

1. `D_linear(sh, B) − D_bilinear(sh, B) > M_B` — the bilinear degrades strictly less than
   the frozen additive arm under shift toward trig-dominant kernels, by more than the
   within-distribution family swing;
2. `|D_linear(bezier, B) − D_bilinear(bezier, B)| ≤ M_B` — and it does **not** show the same
   advantage where there is no trig structure to condition on. This conjunct is what makes
   the claim *domain-conditional* rather than "more capacity is better", and it is
   pre-committed here so it cannot be dropped later;
3. `m_bilinear^DEV(B) ≤ m_linear^DEV(B) + M_B` — it did not buy `sh` by wrecking DEV;
4. `n ≥ 30` on the set, and the §3.2 recomputed-swing check passes at that tier.

**H_form-null (the honest null): the functional form is not the bottleneck.** Accepted iff
`|D_linear(sh, B) − D_bilinear(sh, B)| ≤ M_B`. See §8 for what this means and what it
directs next; it is a real, publishable answer, not a failure.

**H_capacity (named now so it cannot be invented later): a general capacity win, not a
domain-conditional one.** `D_linear − D_bilinear > M_B` on **both** `sh` and `bezier`. This
is a genuine finding and it is **not** H_repr: it says added capacity helps everywhere, which
the Pythagorean-identity question does not predict and the additive model's failure does not
explain.

**H_worse:** `D_bilinear(sh, B) − D_linear(sh, B) > M_B` — the interaction actively hurts
under shift. Accepting it means the added capacity found something in TRAIN that does not
transfer.

**Primary test:** `S = sh`, `B = 100`. **Secondary, reported alongside:** `sh` at `B = 200`;
`bezier` at both tiers; the DEV `trig-heavy` and `polynomial-only` strata (round-1b
registration §2's mechanical rule, unchanged) at both tiers.

## 4. Arms (four, fixed)

| arm | what it is | status |
|---|---|---|
| `unguided` | `Optimizer` with `guide: None` — the denominator of every ratio | reference; `D ≡ 0` |
| `PerRuleRateGuide` | the zero-candidate-local-information control: each rule's TRAIN strict-positive rate, nothing else (`docs/results/2026-09-01-train-guide-report.json`) | **frozen**, unchanged |
| `LinearCandidateGuide` | `pixelflow-pipeline/data/guide_checkpoint_strict_v1.json` **exactly as round 1 trained it** | **frozen**, unchanged — no retraining, no relabelling |
| `BilinearCandidateGuide` | new: `SaturationHead`'s `forward_candidate → compute_candidate_embed → compute_mask_features → bilinear_score` tower, with `encode_rule`'s `[LHS \| RHS \| LHS−RHS \| LHS⊙RHS]` rule encoding, trained under §9 | the thing under test |

The frozen R2G checkpoints are not arms in this registration; they answer a different
question (`docs/plans/2026-09-01-guide-return-to-go.md`) and are neither retrained nor
re-evaluated here.

The bilinear arm's **only** licensed difference from the additive arm is the functional
form. Same candidate features (`CandidateSummary`, unchanged), same label source, same
TRAIN split, same budget denominator
(`REGISTERED_PRIMARY_BUDGET_APPLICATIONS`). Any other difference introduced during training
is a confound that must be recorded in §11 and disqualifies the H_repr verdict.

## 5. Evaluation sets (three, all DEV-side, family-held-out)

| set | source | role |
|---|---|---|
| DEV classical | the existing DEV split's classical band | the reference `m_A^DEV(B)` every `D` is taken against |
| `sh` | `dev_sh_NNNNN`, the round-1b registration §3a real spherical harmonics in spherical coordinates | **primary** |
| `bezier` | `dev_bezier_NNNNN`, round-1b §3b | reported the same way; the H_repr conjunct 2 / H_capacity discriminator |

**Fence, checked in the direction that matters:** every evaluation expression's `FenceKey` is
probed against `corpus_train.bin`'s keys and **any** collision is a hard, named error. TRAIN
families train; DEV, `sh`, and `bezier` only evaluate. **FINAL is not opened by this
registration and is not touched.** A set with `n < 30` classical expressions is reported as
underpowered and gets no verdict.

## 6. Budget tiers

`B = 100` (primary) and `B = 200` (secondary), inherited unchanged from the phase-3
registration §4 for the classical band. Budgets are **recorded rule applications**
(`Budget::Applications`), enforced mid-scan, never wall-clock and never a sweep counter. The
`budget_fraction` feature's denominator stays `REGISTERED_PRIMARY_BUDGET_APPLICATIONS = 100`
at both tiers — evaluating at B=200 must not silently change what the feature *means*.

## 7. Gates

**Accept (H_repr):** §3.3's four conjuncts hold at `B = 100` on `sh`, reported with the full
per-expression distribution (median, quartiles, p90, per-expression CSV) — never a bare
median, never a geomean alone. The `B = 200` result is reported alongside as secondary. The
report must state `n` for every set.

**Null (H_form-null):** recorded with the same completeness as an accept, and §8's directions
followed. A null here is a result, and it is a *sharper* result than round 1b's, because
§1 removes the "the model could not have expressed it" explanation from the table.

**Kill:** inherited from the phase-3 registration §7, unchanged — if after **5 clean training
rounds** the bilinear arm cannot beat unguided-at-B at all on DEV's classical band (median
per-expression ratio < 1.0 at either registered tier), stop and record it. No unbounded
iteration, no round 6.

**Disqualification (any verdict void):** a fence collision; any arm's cost read from
`total_cost` rather than `dag_cost`; any timing entering any metric; a training-set change
between arms; a modified frozen checkpoint; a failed skew test (§9); or a production digest
change (§10).

## 8. The honest null, stated in advance

If `|D_linear(sh, 100) − D_bilinear(sh, 100)| ≤ M_100`, the registered reading is:

> **The functional form is not the bottleneck.** Giving the Guide a rule-by-context
> interaction, trained on the same features and the same labels, does not produce a
> domain-conditional advantage. Combined with §1 — which establishes that the additive
> model could not have expressed one — this rules out the model-class explanation for round
> 1b's H_null and leaves two live places to look:
>
> 1. **The features.** `neighborhood_ops` is a one-hop bag of child `OpKind`s of the matched
>    class. It may simply not carry "this expression is spherical harmonics." A 1-hop
>    multiset is topology-blind by construction.
> 2. **The label.** The strict load-bearing bit is a hindsight property of one saturation
>    trajectory; if it is close to rule-marginal in practice, no functional form over it can
>    be context-conditional, because the *target* is not.
>
> It would specifically **not** license "context does not matter" as a conclusion. Two
> model classes over one feature set and one label is not a test of whether context matters;
> it is a test of whether *these* features and *this* label carry it.

An H_capacity verdict directs somewhere else again: the win is real but is not the
Pythagorean-identity story, and the next question is which candidates it reorders, not which
domains.

## 9. Training and skew protocol (binding)

- **Supervised on hindsight labels only.** No RL, no critic, no REINFORCE path, categorically
  (phase-3 registration §7, restated).
- **TRAIN families only.** DEV, `sh`, and `bezier` are never trained on, never used for
  early stopping, never used for hyperparameter selection. Model selection happens on a
  TRAIN-internal holdout or it does not happen.
- **Mandatory bit-exact train/deploy skew test** for the new checkpoint, in the discipline
  `skew_test_linear_guide` established: the deployed `SaturationGuide::score_candidates`
  score is compared against the trainer's own forward pass on held-out records and must agree
  to 1e-6. **No guided evaluation run may be started before that test passes**, and its
  artifact is committed alongside the results.
- **Vocabulary fingerprint refused on mismatch**, as `LinearWeights::check_vocabulary` does:
  a checkpoint trained against a different rule set is a load failure, never a warning and
  never a default.

## 10. No production behavior change

Production saturation leaves `guide: None` (`Optimizer::production` sets no guide, no
reranker, no mask) and is untouched by this registration. This is **proved, not asserted**:
the production extraction digest (`pixelflow-search/src/runtime.rs::production_extraction_digest`,
#1121 — the `#[ignore]`d test run over the 206 real `.arena` dumps with
`PIXELFLOW_EQUIV_DIR` / `PIXELFLOW_EQUIV_OUT`) is captured before and after every change
made under this registration and the two TSVs must be **byte-identical**. It asserts the
stronger thing a research lever needs: not merely that the *meaning* is unchanged, but that
the extracted *term* is. A digest change disqualifies the round (§7) until it is explained
and reverted.

## 11. Results appended against the gates

*(Empty at registration. Nothing may be written here until §9's skew test passes and the
guided runs have been executed under the §7 gates.)*
