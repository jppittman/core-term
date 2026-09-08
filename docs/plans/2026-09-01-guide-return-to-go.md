# Guide return-to-go: hindsight return as the training target, counterfactual replay as the validation of credit

**Date:** 2026-09-01
**Status:** DESIGN + PRE-REGISTRATION — written before any mint, model, or evaluation exists. §1–§6 are
the denotation and the obligations on the Build phase; §7 is the pre-registered comparison and may
not be revised after the first R2G training run except to append results. A departure is a
revision of this document, not a silent divergence in code.
**Branch:** `claude/phase3-r2g` (from `claude/phase3-domain-shift`, Round 1b's harness).
**Authority (unchanged, binding):** `docs/plans/2026-09-01-phase3-registration.md` (Round 1 — budgets
in recorded rule applications; the ONE curve definition `egraph::anytime::run_anytime_curve[_with]`;
deterministic cost under `CostModel::latency_prior()`; no timing in any metric, safety ceilings panic
if they bind; FINAL untouched; classical tiers B = 100 primary / 200 secondary);
`docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` (Round 1b — the `sh`/`bezier`
OOD families, the margins M_100 = 0.06 / M_200 = 0.07, the H_null verdict);
`docs/plans/2026-08-31-guide-design-revision.md` (§0 framing, §2.2 dedup, §4 candidate-local
features, §5 protocol). **This document supersedes that revision's §3 label semantics for
training** (strict-bit cold start, tightened-labeler refinement): the strict/tight/loose bounds
remain as *proxies scored against a counterfactual truth* (§4), not as training targets.
**Consumes when landed:** `docs/plans/2026-09-01-guide-candidate-context.md` (branch
`claude/phase3-context`) — its `CandidateContext` features enter the same linear model through the
same `CandidateFeatures::observe`; nothing here depends on them.

## 0. The redirect, and what changed in the reading of credit

JP, 2026-09-01, verbatim: *"I feel like we're going in circles. This is what I was trying to do with
my transformers and my critic label creation... Decision transformer. Then you were like, oh that's
dumb we don't need that. This isn't chess, we can figure out exactly how to assign credit."* And:
*"linear first is good discipline."*

The corrected position. Hindsight provenance (`derivation_ancestors`, `EpisodeLabels::compute*`)
gives exact **ancestry** — which applications' output nodes sit on, or feed, the extracted derivation
path. It does not give exact **credit**. Credit is counterfactual: *what would the extraction at
budget B have cost had this application not fired.* Every "load-bearing" bound this program has
trained on or proposed was a hand-drawn approximation of that counterfactual — loose (38% median
load-bearing), tight (19%), strict (3%), and the strict-by-output-class variant proposed after Round
1b — and the evidence that the approximation is wrong in the direction that matters is now on the
books:

- Round 1b (PR #1091, `docs/results/2026-09-01-phase3-round1b-domain-shift.md`): `pythagorean`
  fires 231 times across 19 `sh` expressions under unguided saturation and is strict-positive **0 /
  456** overall, because its output is the literal constant `1`, a node that already exists, so the
  firing is a union whose provenance is not the application. The strict bit cannot see the identity
  JP named.
- Same report: the strict-bit linear Guide and the per-rule-rate control move **together** under
  domain shift (D_control − D_linear = −0.030 at B = 100 on `sh`, inside M; the same on `bezier`,
  which has no trig at all). The features had nothing to learn from a label that scores the
  structural/enabling class at 0%.

So the training target moves from a hand-drawn bit to the quantity the bit was approximating: the
**hindsight return-to-go** of the trajectory the application belonged to — the decision-transformer
objective — with a **linear model first**, and **counterfactual replay** as the validation that the
learned credit is real. The transformer over the trajectory is a named later rung (§6), not built
here. Scope cut stands: the smallest thing that answers the question for this compiler; no
generality abstractions.

## 1. Denotation

### 1.1 Trajectory

Fix an expression `e` (an `ExprArena` root) and the rule library `R = all_rules()` (62 rules). An
**ordering policy** `π` is anything that, given the current e-graph and the round's enumerated
matches, decides the order in which matches are applied. A **trajectory**

    τ = ⟨a_1, a_2, …, a_T⟩

is the ordered sequence of *recorded* rewrite-rule applications `π` produces on `e`, from the initial
e-graph (`EGraph::with_rules(R)` + `add_arena(e)`) until the cumulative recorded-application count
reaches the budget `B` or the run ends (quiescence / class cap). Each `a_t` is identified by its
`ApplicationId` (ordinal `t`, the position) and by the `(rule_idx, match)` it applied. Two mechanics
exist and both are kept exactly as Round 1 runs them:

- **Guided-family policies** run through `GuidedSaturation::until_applications`
  (`pixelflow-search/src/egraph/saturate.rs`): per round, enumerate matches, **dedup before scoring**
  on `CandidateKey = (rule_idx, canonical class content)` against the keys already resolved in this
  episode, batch-score the survivors once, apply in descending score order with the budget checked
  after every application, mark a key resolved only when an application was actually recorded. The
  ordering policy is the `SaturationGuide` that produces the scores; every guided-family policy below
  is one.
- **Unguided** is the rule-major sweep `EGraph::saturate_until_applications`: rules in index order,
  every match of a rule applied in class-then-node order, budget checked between rules, no dedup
  (idempotent re-fires are recorded like everything else — 91% of recorded applications).

`B` is denominated in recorded applications, per the registration. The checkpoint semantics at `B`
are the harness's: a guided trajectory's cost at `B` is read at exactly `B` recorded applications
(per-application budget check) or at its end if it ended earlier; unguided's is read at the first
between-rules point with count ≥ `B` (`app_actual` recorded).

### 1.2 Return

The **return** of a trajectory at budget `B` is the latency-prior cost of the extraction at `B`,
expressed as regret against the expression's best-known cost so that returns are comparable across
expressions:

    c_τ(B)  = extract_dag(egraph after τ up to B, root, latency_prior).total_cost
    c*_e    = min over every trajectory τ' of e (all K in §2) and every checkpoint on τ' of c_τ'(·)
              — the Round-1 empirical-best convention, extended over the K trajectories; the
              unguided trajectory contributes its full APP_CHECKPOINT_GRID through quiescence/cap
    R(τ,B)  = ln( c_τ(B) / c*_e )        (log-regret; 0 = the trajectory reached the best known)

Log rather than ratio because the per-expression ratio tail is heavy (Round 1 reports p90 regret at
the guided grid end of 8.5e3% on a handful of tiny-best expressions) and an MSE target with that tail
is a target on the outliers; `ln` is monotone, so nothing about ranking changes. **`c*_e = 0`**: a
positive cost against a zero reference is infinite loss (registration §6 convention); such
expressions are **excluded from training** and counted in the mint report (`zero_best_excluded`),
and `c_τ(B) = c*_e = 0` is `R = 0`. Ratio-vs-unguided-at-B, the registered evaluation statistic, is
untouched by any of this; `R` is the training quantity.

### 1.3 Return-to-go label

The episode has one terminal reward (the cost at `B`) and no intermediate reward, so the return-to-go
at every position `t ≤ B` equals the trajectory's return. The **return-to-go label** of application
`a_t ∈ τ` is

    y_{a_t} = R(τ, B)        attached to  ( CandidateFeatures at a_t , budget position t )

where `CandidateFeatures` is the object `CandidateFeatures::observe` produced for that candidate in
the round it was enumerated (the deploy-time feature, see §1.5) and the budget position enters as
`budget_fraction` (round-start ordinal / `REGISTERED_PRIMARY_BUDGET_APPLICATIONS`, unchanged), with
the exact ordinal `t` also recorded. An application with `t > B` carries no label for that `B`
(`R(τ,B)` is already determined before it fires).

Two labels are minted per application, `return_b100` and `return_b200` (the two registered tiers; a
trajectory to 200 contains its own 100-prefix). **The training target is the expression-centered
return**

    ŷ_{a} = R(τ,B) − R̄_e(B),      R̄_e(B) = mean over the K trajectories of e of R(τ_k, B)

because a candidate-local linear model has no expression identity to learn the expression's absolute
regret level from; that level is nuisance variance that centering removes, leaving the part of the
return that varies with the *ordering* — which is exactly the credit signal. The raw `R` is written
alongside and trained on as an ablation (§3.4). Both are `null` where `t > B` or `c*_e = 0`.

**This is the decision-transformer objective**, stated plainly: every token in a trajectory is
labelled with the trajectory's return-to-go, and the model learns the return associated with taking
an action in a context at a timestep. Two substitutions make it the linear-first version. (i) The
sequence context — the transformer's attention over the preceding tokens — is replaced by
**candidate-local context** (rule identity, matched-class content, one-hop neighborhood ops,
matched-class size, expression size) plus the **budget position as the timestep**; a linear scorer
over these cannot represent a sequence, so nothing is lost by not feeding it one, and the sequence
context is precisely what the transformer rung (§6) would add. (ii) At deploy the model is used in
the inverse direction: rather than conditioning on a target return to emit an action, it scores
every available action by its predicted return and the loop takes them best-first. What the model
learns is *credit by association* — the average return of trajectories in which a candidate like
this appeared at a position like this. That is a learned credit, not an assumed one, which is why §4
exists: the counterfactual replay is what checks it against the truth.

### 1.4 Why a within-step ranking loss is vacuous here (and what "within a step" has to mean)

Every application in one trajectory carries the same label. A pairwise ranking loss "within a step"
of one trajectory has no signal by construction. Ranking signal exists only **across trajectories of
the same expression**: candidate `a` chosen at position bucket `p` in `τ_i` versus candidate `b`
chosen at the same bucket in `τ_j`, ordered by `R(τ_i) − R(τ_j)`. §3.4 defines the ranking ablation
that way. This is the observable difference between the R2G objective and a per-step advantage: the
label is a property of the *ordering*, and only ordering diversity (§2) makes it informative.

### 1.5 Observation time (skew-freedom)

The guided-family records are minted **from the live guided loop**: the `CandidateSummary` the Guide
was actually handed in that round is what is written, with the `ApplicationId` it received when it
fired. There is no post-hoc replay against a final graph and no per-round approximation of the
ordinal beyond the one the deployed loop itself makes. The unguided trajectory has no rounds in the
guided sense; its records are observed at the start of each per-rule batch inside the sweep
(`Batch::apply_rule` enumerates rule `r`'s matches against the live graph, then applies them), with
the ordinal = application count at that batch start — the finest "round start" the batched path has.
That asymmetry exists only for unguided-origin records and is stated here rather than hidden;
Round 1's `gen_strict_labels` observed everything against the *final* graph, which the new mint
never does.

## 2. Ordering diversity is the dataset

Without variation in return across trajectories of the same expression there is no credit signal:
if every ordering of `e` reaches the same cost at `B`, every record of `e` has `ŷ = 0` and the
model learns nothing about `e`'s candidates. Diversity is therefore the dataset's design, not a
safety valve. Per expression, `K = 12` trajectories:

| # | policy (`Policy` name in the records) | mechanics | why it is in the set |
|---|---|---|---|
| 1 | `unguided` | rule-major sweep, no dedup | the production-shaped ordering; also supplies `c*_e`'s deep grid |
| 2 | `per-rule` | `PerRuleRateGuide` from `2026-09-01-train-guide-report.json` | the Round-1 control; a global-prior ordering |
| 3 | `strict-v1` | `LinearCandidateGuide` on the FROZEN `guide_checkpoint_strict_v1.json` | the Round-1 policy, off-policy data for R2G |
| 4–9 | `random:<seed>`, 6 seeds | `UniformRandomGuide { seed }`: score = splitmix64(seed ⊕ candidate fingerprint) → `[0,1)`; a fixed pseudo-random preference over candidate types, deterministic and platform-independent (no RNG state, no interior mutability) | the ordering distribution's bulk; every rule gets budget in some seed |
| 10–12 | `mix:strict-v1:1/4:<s1>`, `mix:strict-v1:1/4:<s2>`, `mix:strict-v1:1/2:<s1>` | `RankMixGuide { base, eps, seed }`: score = (1−ε)·(base's within-round rank, normalized to `[0,1]`) + ε·u, `u` the same hash-uniform as above | local perturbations of the best known policy — trajectories that share a prefix with `strict-v1` and diverge, which is where per-candidate credit is sharpest |

`EpsilonMix { numerator, denominator }` is the same type the context design (§6.1 there) names for
deploy-time ε; its deploy-time interleave semantics are not needed for a mint and are not built
here. `RankMixGuide` mixes on **ranks**, not raw logits, so ε means the same thing regardless of the
base Guide's score scale.

**Justification of K and n_rand from expression count and mint cost.** TRAIN classical expressions
number ≈ 1,600 (the classical band is 188/400 of the size-stratified sample of 4,143 TRAIN+DEV;
DEV holds 334 of them). Round 1b's harness ran 334 DEV expressions × (unguided to quiescence at a
median 2,073 applications, two guided arms to 800, the production probe) in ~1–2 minutes of
wall-clock (`load_at_start`/`load_at_end` in the report contexts — context, not a metric), i.e.
≈ 0.3 s per expression with three trajectories, dominated by the unguided deep grid. Eleven further
guided-family trajectories to `B = 200` add well under a second per expression. The mint over
`--train-limit 800` stride-sampled TRAIN classical expressions (Round 1 minted 415) is therefore
tens of minutes, and its record count is bounded by 800 × 12 × 200 = **1.92 M** rows (each
trajectory contributes at most 200 labelled applications), about 3× Round 1's 660 k. `n_rand = 6`
is the smallest count at which every rule index in a typical 50-survivor round is placed in the top
quartile by at least one seed with high probability (each seed independently ranks a given
candidate in the top quartile with p = 1/4; 1 − (3/4)^6 = 0.82), and six random orderings plus three
mixtures already give 9 distinct orderings per expression beyond the three named ones — more seeds
buy diminishing spread at linear cost. Fixed here; not tuned after seeing results.

**The per-expression return spread is a dataset statistic and a gate.** For every expression and
each `B`: `spread_e(B) = max_k R(τ_k,B) − min_k R(τ_k,B)` and the number of distinct values. The
mint report records the quantiles of `spread_e`, the share of expressions with `spread_e(100) = 0`
(carrying no credit signal), the share of *records* from such expressions, and the per-policy mean
of `R`. Zero-spread expressions stay in the training set (their `ŷ = 0` records say "these
candidates do not move the outcome at this budget", which is a true statement the model may as well
fit) but are reported. **Pre-registered dataset gate:** if more than 50% of TRAIN classical
expressions have `spread_e(100) = 0`, the finding is that ordering does not move the outcome at the
primary tier on this corpus; record it, do not train, and take that finding to §7's kill row.

**Split discipline.** TRAIN records come from `corpus_train.bin`'s classical band, fence-checked
through `SplitManifest` by parsed family name exactly as `gen_strict_labels` does. DEV records
(`corpus_dev.bin`, the 334 classical) and OOD records (`corpus_dev_ood.bin`, `sh` 95 / `bezier` 80)
are minted with the same binary into separate files and are never trained on; the DEV file feeds the
skew test and held-out regression metrics, and every loaded entry is `FenceKey`-checked against
TRAIN with a collision as a hard error (Round 1b's rule).

## 2b. Round 3: measuring spread where the budget binds (2026-09-01)

Round 2's dataset gate FIRED (§9's appended results: 67.2% of TRAIN classical `spread_e(100) = 0`)
and the counterfactual credit test found the linear R2G Guide ties the strict bit everywhere
(ρ = −0.004 vs the counterfactual, `docs/results/2026-09-01-guide-return-to-go.md`). JP's reading:
that mint inherited `gen_strict_labels`' `--max-expr-nodes 250` / `--max-classes 2000` tractability
filter — a regime where `B = 100` is near quiescence for most surviving expressions, and dedup
collapses every guided ordering onto the same committed set (`bezier` showed an exact 11-way tie).
Production runs classical kernels to a median ≈ 1,671 applications and the 5,000-class production
cap binds on ≥ 67.6% of real kernels (PR #1087) — that regime was never in Round 1/2's training
data. This round re-mints with the size filter LIFTED, the class cap raised to production's
classical tier, and returns reported at six budgets (`100..3200`, `BUDGET_LADDER`) instead of two,
specifically to find OUT whether spread — and, more precisely, spread AMONG the guided policies
(the credit signal a learned Guide could improve; unguided-vs-guided spread is just
ordering+dedup) — exists anywhere in the regime production actually runs in.

**Selection-rule test (registered before reading the results below):** a `(B, node-count band)`
cell enters the training regime iff `zero_spread_guided_share < 0.5` for that cell — i.e. at least
half its expressions show non-zero spread AMONG the 11 guided orderings. `qualifies` in
`r2g_spread_vs_budget`'s own output is exactly this test, per cell.

Full numbers: `docs/results/2026-09-01-r2g-spread-vs-budget.{md,csv,json}`; mint provenance:
`docs/results/2026-09-01-r2g-trajectory-mint-full.{json,md}`.

### 2b.1 Run

`gen_r2g_trajectories --tier all --train-limit 0 --dev-limit 0 --max-expr-nodes 0` (size filter
LIFTED — the full population, not a stride sample; `--train-limit`/`--dev-limit 0` mean "all"),
`--max-classes` unset → resolved to production's classical tier, **5,000**, read from
`config_for_node_count(usize::MAX)` (not hardcoded — see the flag's own doc). Budget ladder
`100,200,400,800,1600,3200`. No expression hit the per-expression 10-minute wall-clock ceiling
(`skipped_wallclock: 0` on every split) and the run-level `|R|`-scaled panic did not fire. Mint
wall-clock: 1,306.1s (~21.8 min) for all four splits combined.

| Split | Expressions | Zero-best excluded | Trajectories | Applications |
|---|---:|---:|---:|---:|
| TRAIN | 3,356 | 3 | 40,272 | 18,565,784 |
| DEV | 783 | 1 | 9,396 | 3,544,477 |
| sh | 100 | 0 | 1,200 | 3,652,201 |
| bezier | 80 | 0 | 960 | 2,504,034 |

**No expression in TRAIN, DEV, `sh`, or `bezier` exceeds 1,000 arena nodes even with the size
filter fully lifted** — the `>1000` node-count band is empty for every split (`r2g_spread_vs_budget`
never emits it). This is itself a finding, not a null result: production's classical-tier tail
(p90 23,799 / max 85,900 applications at stop, `docs/results/2026-09-01-phase3-at-budget-eval.md`)
is not explained by a handful of unusually large expression TREES — it comes from expressions in
the 251–1,000 node range needing many more rewrite ROUNDS than round 1/2's `--max-classes 2000` /
B≤200 mint ever gave them room to take, not from trees themselves growing past 1,000 nodes. The
corpus's node-count ceiling, not the class cap or the budget ladder, is what bounds this round's
coverage of production's tail (p90/max reach ~24k/86k applications; this ladder tops out at 3,200).

### 2b.2 Selection-rule test results

Applying `zero_spread_guided_share < 0.5` (the pre-registered test) per `(B, band)` cell:

| Set | Qualifying cells (B, band) | n per cell |
|---|---|---|
| **TRAIN** | (100, 101-250), (200, 101-250), (100, 251-1000), (200, 251-1000), (400, 251-1000) | 540, 540, 529, 529, 529 |
| **DEV** | (100, 101-250), (200, 101-250), (100, 251-1000), (200, 251-1000), (400, 251-1000) | 116, 116, 92, 92, 92 |
| `sh` | ALL 18 cells (3 bands × 6 budgets) | 46–100 |
| `bezier` | 15 of 18 cells (all but B∈{100,200} band 51-100/all) | 19–80 |

**TRAIN and DEV agree exactly on which cells qualify** — the same two bands, the same budget
cutoffs, independently on two disjoint expression populations (different generator families,
`corpus_split.toml`'s family-holdout). That agreement is the strongest evidence in this round that
the effect is real rather than a sampling artifact of either split.

**The pattern, read against `zero_spread_guided_share` directly** (not just the yes/no test):

- **Band 51-100 and the pooled "all" row never qualify on TRAIN/DEV, at any budget.** At this size,
  expressions are already near-quiescent by B=100 (43.4%/42.1% zero-spread-among-guided at B=100
  even before any spread has a chance to develop, climbing past 90% by B=400) — this is round 2's
  original finding, now localized: it was never a property of "classical expressions" in general,
  it is specifically what happens to SMALL classical expressions.
- **Band 101-250 qualifies only at B∈{100,200}.** `zero_spread_guided_share` is 11.1%/36.1% (TRAIN)
  and 7.8%/42.2% (DEV) there, then jumps to 74.1%/85.3% at B=400 — the credit signal exists but is
  gone by the third rung of the ladder. `unguided_differs_share` at B=100 in this band is 94.6%
  (TRAIN) / 97.4% (DEV) — much higher than the pooled `all`-band figure (44.1%/40.0%) — confirming
  this band specifically, not the population at large, is where unguided departs furthest from the
  guided family (ordering+dedup is doing real work here, on top of the guided-vs-guided signal).
- **Band 251-1000 qualifies through B=400 (0.0% / 0.0% / 19.1–21.7% zero-spread-among-guided at
  B=100/200/400), then collapses to 54–58% by B=800.** This is the band carrying the strongest,
  widest signal — spread-among-guided is exactly 0% at B=100/200 (EVERY expression in this band
  has SOME guided policy disagree with some other), and unguided-differs is 100.0% at every budget
  through B=400. `ApplicationBudget` is the ONLY stop reason at B=100/200 for TRAIN's 251-1000 band
  (6,348 of 6,348 trajectory-checkpoints, i.e. all 529 expressions × 12 policies — zero quiesce),
  dropping to 5,863/6,348 at B=400 — these expressions are genuinely budget-bound, not quiescing
  early; the spread only closes once budget stops binding.
- `sh`/`bezier` qualify almost everywhere, confirming round 2's own reading: the synthetic
  TRAIN/DEV generator's structural motifs converge under saturation in a way the two OOD families
  do not, independent of node-count band.

### 2b.3 Selection rule (registered here, before any training run)

**The training regime is: node-count band ∈ {101-250, 251-1000}, budget ∈ {100, 200} for
101-250 and ∈ {100, 200, 400} for 251-1000** — i.e. exactly the five TRAIN/DEV-agreeing cells in
§2b.2's table. A record enters training iff its source expression's arena node count falls in one
of those two bands AND its label budget is one of that band's qualifying budgets; everything
else (band 51-100 at any budget, band 101-1000 past its budget cutoff, and the pooled/unbanded
population) is excluded from the training regime as a zero-signal population by this test, even
though `spread_report`'s existing dataset gate would still admit it. Label budget for training:
B=100 (both bands qualify there with the widest margin from zero) unless the higher-budget
ablation named in §3.4 is run.

**This is a narrower regime than "TRAIN classical," and it does not cover production's tail** (no
`>1000`-node expression exists in this corpus at all, §2b.1) — training restricted to it answers
"can a Guide learn anything in the region where guided orderings disagree," not "does a Guide
generalize to the full production distribution." Extending coverage past 1,000 nodes is a corpus-
design question (§2b.1's redirect), separate from this selection rule.

Selection made from this measurement alone, before training; §7's registered comparison and Round
1/2's kill-gate accounting are unaffected until a training run against this regime reports back.

## 3. The model

### 3.1 Features

Exactly the Round-1 encoding, produced by the one constructor `CandidateFeatures::observe` and
encoded by `pixelflow_pipeline::training::guide_linear::to_sample` — nothing is added or renamed:
`w_rule[rule_idx]` (rule one-hot), `Σ_op w_op[op] · ln(1 + hist[op])` over the one-hop
`neighborhood_op_hist`, `w_budget · budget_fraction`, `w_match_class · ln(1 + match_class_node_count)`,
`w_neighborhood · ln(1 + neighborhood_op_count)`, `w_expr_size · ln(1 + expr_node_count)`, plus
bias. Rule embeddings from templates remain out of scope. When the context features land, they enter
this same linear form through the same constructor; that is a separate registration.

### 3.2 Objective: MSE on the centered log-regret (primary)

    L = mean over labelled records of ( f(x_a) − ŷ_a )²,   ŷ_a = centered return_b100

Chosen over the ranking loss because (i) deploy needs a scalar to sort and MSE gives one directly,
calibrated across expressions by the centering; (ii) it needs no pair sampling and no choice of
position bucket, so there is one fewer hand-drawn decision in the primary result; (iii) it is the
DT objective's own loss shape (regress the return). Trainer mechanics unchanged from `train_guide`:
cold start at zero, online SGD with inverse-time decay, L2 on weights not bias, per-step gradient
clipping on `dL/dz = 2(f − ŷ)`, seeded shuffle. No class weighting (there are no classes).

### 3.3 Deploy

`LinearReturnGuide` implements `SaturationGuide` with `score = −f(x)`: `GuidedSaturation` sorts
descending, so the candidate with the **lowest predicted return fires first**. Same dedup, same one
batch-score per round, same per-application budget check — the loop is untouched; only the scorer
differs. Its forward pass is the same field-for-field formula as `LinearCandidateGuide` (one private
`LinearWeights` inside `nnue/guide/linear.rs` serves both public types, so the formula exists once
in the deployed crate and once in the trainer, which is the pair the skew test compares). The
checkpoint carries `"objective": "return-mse"`; `LinearReturnGuide::load` refuses a checkpoint
without it and `LinearCandidateGuide::load` refuses one with it — cross-loading a regression head as
a logit head is a loud error, never a silently reversed ordering. The frozen Round-1 checkpoint has
no `objective` field and loads exactly as before.

**Skew test, mandatory:** `skew_test_linear_guide --model return` runs the trainer's `Model::logit`
and `LinearReturnGuide::score_candidates` on ≥ 1,000 DEV records of the new mint and requires
`|f_trainer + score_deployed| ≤ 1e-6` on every record (the sign is the one deliberate difference).
The same `black_box` fencing keeps release builds bit-exact.

### 3.4 Ablations (reported next to the primary, never in its place)

| ablation | what changes | question it answers |
|---|---|---|
| raw target | train on `return_b100`, not the centered one | does centering carry the result, or is the expression level harmless noise? |
| B = 200 target | train on `centered_b200` | is the credit tier-specific? (evaluated at both tiers either way) |
| pairwise ranking | pairs `(a ∈ τ_i, b ∈ τ_j)` of the same expression, `i ≠ j`, same `budget_fraction` decile, logistic loss on `sign(R(τ_j) − R(τ_i))·(f(x_a) − f(x_b))`, 8 pairs per record, seeded | does a rank objective beat the regression at the ranking task the loop actually performs? |

## 4. Counterfactual credit as validation

### 4.1 The truth

For an application `a` at ordinal `t_a` in a trajectory `τ`, the **leave-one-out credit** is

    Δ_a = R(τ \ a, B) − R(τ, B)

where `τ \ a` **replays the same ordering policy with `a` masked**: at the ordinal where `a` would
have been recorded, that `(rule, match)` is skipped and *not* recorded, so the budget slot it would
have consumed is spent on the next candidate in the same order; everything after is whatever the
policy does on the counterfactual graph. `Δ_a > 0` means removing `a` made the outcome worse — `a`
was genuinely helpful at this budget; `Δ_a < 0` means `a` was actively wasteful (its slot was worth
more to the next candidate); `Δ_a = 0` means `a` did not matter. Deterministic: the unguided sweep is
already pinned bit-identical across runs (`unguided_mode_is_bit_identical_to_before_this_change`),
and the mask is keyed on the recorded ordinal, so `τ \ a` is a pure function of `(e, a)`.

This is one counterfactual (leave-one-out), not a Shapley value: it does not average over the
orderings `a` could have appeared in and it charges interactions to whichever application is
removed. It is enough here because the question is whether a proxy — the model's score or a
hand-drawn bit — ranks applications the way *removing them* ranks them, on the very trajectory the
proxy was computed for; a Shapley credit would answer a different question (a's average marginal
value across coalitions) at a cost exponential in `T`. Shapley is parked (§6).

### 4.2 The sample S

- Expressions: **≥ 30 `sh`** (from `corpus_dev_ood.bin`, `dev_sh_*`, classical) and **≥ 30 DEV
  classical** (`corpus_dev.bin`, stride-sampled over the 334), never TRAIN.
- Trajectory: the **unguided** trajectory of each, observed in-sweep (§1.5), run through `B = 100`
  (primary) with the same grid semantics as the harness.
- Applications: **≥ 20 per expression**, sampled uniformly over ordinal `t ∈ [1, B]` **from the
  state-changing recorded applications** (those whose apply committed a node or a union). The
  no-op re-fires (91% of the sweep) have `Δ = 0` by construction and would make the statistic a
  tie-count; their share is reported, not sampled. An expression with fewer than 20 state-changing
  applications in `[1, B]` contributes all of them and is flagged.
- Cost: ≥ 60 expressions × ≥ 20 replays to `B = 100` — about 1,200 short unguided runs, well under a
  minute at the harness's measured rate (context, not a metric; the per-curve safety ceiling panics
  if it binds).

### 4.3 The proxies, scored against Δ

For each sampled `a`, with `A_t` = the other applications recorded in the **same sweep** as `a`
(same `ApplicationRecord::step`; the set the sweep spent budget on around `a` — "alternatives at that
step" for a rule-major sweep, across rules, not just `a`'s own rule):

| proxy | score for `a` | expected sign vs Δ |
|---|---|---|
| **R2G model (the claim)** | `adv_a = mean_{b ∈ A_t} f(x_b) − f(x_a)` — predicted return of the alternatives minus `a`'s (positive = `a` predicted better) | + |
| strict-v1 linear Guide | `adv_a = logit(x_a) − mean_{b ∈ A_t} logit(x_b)` | + |
| per-rule rate | same, with the rate table | + |
| loose bound | `EpisodeLabels::compute` bit for `a` | + |
| tight bound | `EpisodeLabels::compute_tight` bit | + |
| strict bound | `EpisodeLabels::compute_strict` bit | + |
| strict-by-output-class | NEW `EpisodeLabels::compute_strict_by_output_class`: positive iff the canonical class of `a`'s output — any node with `Origin::Rule(a)`, or either side of a `UnionEvent` caused by `a` — contains a node on the extracted derivation path (credits `pythagorean`'s union into an existing `1`) | + |

The bounds are computed on the unguided run **at B** (labels over the applications recorded ≤ B,
against the extraction at the B checkpoint) as the primary column, because that is the extraction
`R(τ,B)` is defined by; the full-saturation version (what Round 1 trained on) is the secondary
column. **Statistic:** Spearman ρ between proxy score and `Δ_a`, average-rank ties, pooled over S
and per set (`sh`, DEV); for a 0/1 bound this is the point-biserial correlation. Reported with the
share of sampled applications with `Δ_a ≠ 0` (the sample's power) and a 2,000-resample bootstrap CI
over expressions. **The hand-drawn bounds are ranked against the truth here rather than argued.**

## 5. What is built (all additive; production `saturate*` untouched; minimal public API)

Full entry-point table in §8. In order:

1. `pixelflow-search`: `UniformRandomGuide`, `RankMixGuide`, `EpsilonMix`; `LinearReturnGuide` with
   the shared private `LinearWeights`; `GuidedSaturation::with_observer` (called once per *recorded*
   application with the `CandidateSummary` that was scored, the `ApplicationId`, the round ordinal,
   and `changed`); `EGraph::saturate_until_applications_observed` (in-sweep per-rule-batch
   observation) with an `ApplicationMask` (skip the application that would receive ordinal `t`,
   `None` = the existing function, which delegates to it — the bit-identity test pins that);
   `EpisodeLabels::compute_strict_by_output_class`; an `AnytimeStepper` for the observed/masked
   unguided sweep so every curve still goes through `run_anytime_curve_with`.
2. `pixelflow-pipeline`: `training/r2g.rs` (`Policy`, `R2gRecord` — a strict superset of
   `guide_linear::Record`'s fields so `to_sample` applies unchanged — `TrajectoryRow`, `log_regret`,
   centering); `gen_r2g_trajectories`; `train_guide --objective return-mse|return-rank --target
   centered|raw --label-b 100|200`; `skew_test_linear_guide --model return`;
   `r2g_counterfactual`; `phase3_at_budget_eval --r2g-checkpoint` adding the `r2g` arm and the §7
   tables. Each lands with tests; `cargo check -p pixelflow-ir --no-default-features` and the
   unguided bit-identity test run before every commit.
3. Mint (TRAIN, DEV, `sh`, `bezier`), report the spread gate, train primary, skew test, ablations,
   counterfactual, at-budget ladder, append §7.1 here and write
   `docs/results/<date>-phase3-r2g.md`.

## 6. Parked (named rungs, with the condition that pulls each forward)

- **Transformer over the trajectory** (JP's original design): the same R2G label, with the sequence
  of `(candidate, position)` tokens as context. **Pull forward when** the candidate-local family
  plateaus against the counterfactual truth — i.e. the linear R2G model (and, once landed, the
  context-feature linear model) beats every hand-drawn bound on §4's Spearman but that Spearman
  itself stays low (< 0.3 pooled) while `Δ` has spread: the remaining credit is then in what came
  before, which only a sequence model can see.
- **Shapley credit**: the average marginal value over orderings. Pull forward only if leave-one-out
  `Δ` proves too interaction-dominated to rank proxies (many large |Δ| that flip sign across
  neighbouring ordinals); expensive, and §4.1 says why one counterfactual is enough for the present
  question.
- **On-policy iteration**: re-minting from the R2G Guide's own trajectories and retraining. The
  ε-mixtures here are perturbations of the *frozen* strict-v1 policy — off-policy data, not a loop.
  On-policy iteration is not entered until an off-policy R2G model has cleared §7; the context
  design's §7 (no compile-time adaptation) stands regardless.

## 7. Pre-registered comparison (numbers from Round 1 / 1b only; fixed before any R2G run)

**Arms:** `unguided`, `control` (`PerRuleRateGuide`), `linear` (Round-1 strict-bit
`LinearCandidateGuide`, frozen `guide_checkpoint_strict_v1.json`, MD5
`dcc79b59cfe00bc62df031924382e279`), **`r2g`** (`LinearReturnGuide`, `guide_checkpoint_r2g_v1.json`).
**Sets:** DEV classical (n = 334, `corpus_dev.bin` MD5 `3026133ebba066eeca10f658da554400`), `sh`
(n = 95), `bezier` (n = 80) (`corpus_dev_ood.bin` MD5 `0c7cbe710c50175afb3cd91f60960b64`).
**Tiers:** B = 100 / 200. **Statistic:** `m_A^S(B)` = median over the set of per-expression
`cost_A@B / cost_unguided@B`, by the same harness, grid, cost model, and class cap as PR #1091.
**Margins:** M_100 = 0.06, M_200 = 0.07 (Round 1b §1.1).

Reference medians (PR #1091, `docs/results/2026-09-01-phase3-round1b-domain-shift.md`):

| set | B | control | strict-bit linear | R2G must reach |
|---|---:|---:|---:|---|
| `sh` | 100 | 0.9028 | **0.9039** | `m_r2g < 0.9039 − 0.06 = 0.8439` |
| `sh` | 200 | 0.8940 | **0.8959** | `m_r2g < 0.8959 − 0.07 = 0.8259` |
| DEV | 100 | 0.5655 | 0.5366 | `m_r2g ≤ 0.5366 + 0.06 = 0.5966` |
| DEV | 200 | 0.6991 | 0.6959 | `m_r2g ≤ 0.6959 + 0.07 = 0.7659` |
| `bezier` | 100 | 0.9098 | 0.9098 | reported; no claim |
| `bezier` | 200 | 0.9098 | 0.8855 | reported; no claim |

**Claim A (ordering).** On `sh`, the R2G Guide's median ratio vs unguided-at-B is lower than the
strict-bit Guide's by more than M_B at **both** tiers, and on DEV it is not worse than the
strict-bit Guide's by more than M_B at either tier. Reported with the full per-expression
distribution (quartiles, p90, head-to-head counts r2g < / = / > linear), never a median alone.

**Claim B (credit).** On S (§4.2), the R2G model's advantage–Δ Spearman exceeds every hand-drawn
bound's (loose, tight, strict, strict-by-output-class; at-B column) pooled **and** on each of the
two subsets. The bootstrap CI of each difference is reported as evidence, not as the gate.

**Readings, pre-committed:**

| outcome | reading | next |
|---|---|---|
| A and B hold | the target was the bottleneck; credit is learnable from ordering diversity with candidate-local features | context features (`claude/phase3-context`) enter the R2G objective; FINAL is opened only at a publication run under Round 1's accept gate |
| A holds, B fails | the Guide improved for a reason the counterfactual does not see at B = 100 (e.g. credit realized past B, or through no-op-free dedup closure) | re-run §4 at B = 200 and on a guided-family trajectory before believing A; report both |
| A fails on `sh` but B holds | credit is learned but the linear scorer cannot act on it at this budget | context features next; the ordering result is reported as null |
| **R2G ties the strict bit everywhere** (`\|m_r2g − m_linear\| ≤ M_B` on every set and tier) | **the target was not the bottleneck** | say so; the next lever is context features (`claude/phase3-context`), not another label |
| R2G worse than the strict bit on DEV by > M_B | the centered-return regression is misspecified for this feature set | run the §3.4 ablations before any other change; if none recovers, record the negative |
| dataset gate fires (§2) | ordering does not move the outcome at B = 100 on TRAIN classical | no training; the finding is the deliverable and the budget-regime investigation Round 1b promoted takes precedence |

Kill semantics: this round trains one new checkpoint from a new label and counts as one of the five
clean rounds of Round 1's kill gate (the label source changed; the gates did not).

### 7.1 Results appended against the gates

(none yet — appended only after the run, per this document's own rule)

## 8. Build-phase entry points (the api_notes, in one place)

| Crate / file | Item | Kind |
|---|---|---|
| `pixelflow-search/src/nnue/guide/linear.rs` | private `LinearWeights { load(&Value, path) -> Result<_, CheckpointError>, logit(&CandidateSummary) -> f32 }` shared by `LinearCandidateGuide` (unchanged behaviour; refuses `"objective": "return-*"`) and NEW `pub struct LinearReturnGuide` (`load(&Path)` requires `objective ∈ {"return-mse","return-rank"}`; `score_candidates = −logit`) | additive |
| `pixelflow-search/src/nnue/guide/diversity.rs` (new) | `pub struct UniformRandomGuide { seed: u64 }`; `pub struct EpsilonMix { numerator: u16, denominator: u16 }`; `pub struct RankMixGuide<'a> { base: &'a dyn SaturationGuide, eps: EpsilonMix, seed: u64 }`; both `impl SaturationGuide`; scores are `splitmix64(seed ⊕ fnv1a64(rule_idx, budget_fraction bits, match_class_node_count, expr_node_count, neighborhood op hist))` mapped to `[0,1)` | additive |
| `pixelflow-search/src/egraph/saturate.rs` | `pub struct ApplicationObservation<'a> { application: ApplicationId, round: u32, changed: bool, summary: &'a CandidateSummary }`; `GuidedSaturation::with_observer(self, FnMut(ApplicationObservation<'_>)) -> Self` (invoked once per recorded application) | additive |
| `pixelflow-search/src/egraph/graph.rs` | `pub struct ApplicationMask { skip_ordinal: u64 }`; `pub struct SweepObservation { application: ApplicationId, sweep: usize, changed: bool, features: CandidateFeatures }`; `pub fn saturate_until_applications_observed(&mut self, max_total, max_iters, max_classes, timeout, mask: Option<ApplicationMask>, hook: impl FnMut(SweepObservation)) -> AppBudgetSaturationStats`; the existing `saturate_until_applications` delegates with `None` and a no-op hook (bit-identity test pins it) | additive; production untouched |
| `pixelflow-search/src/egraph/anytime.rs` | `pub struct ObservedUnguidedStepper<F> { mask, hook }` implementing `AnytimeStepper` | additive |
| `pixelflow-search/src/egraph/labeler.rs` | `EpisodeLabels::compute_strict_by_output_class(egraph, root, choices) -> Self` | additive |
| `pixelflow-pipeline/src/training/r2g.rs` (new) | `pub enum Policy { Unguided, PerRule, StrictV1, Random { seed }, Mix { base: Box<Policy>, eps: EpsilonMix, seed } }` + `Display`/`FromStr` (`unguided`, `per-rule`, `strict-v1`, `random:<seed>`, `mix:strict-v1:<n>/<d>:<seed>`); `pub struct R2gRecord` (all `guide_linear::Record` fields + `trajectory_id: u32, policy: String, round_ordinal: u32, application_ordinal: u64, changed: bool, candidate_key_fingerprint: u64, cost_b100: u64, cost_b200: u64, expr_best_cost: u64, return_b100: Option<f32>, return_b200: Option<f32>, centered_b100: Option<f32>, centered_b200: Option<f32>`); `pub struct TrajectoryRow { expr_name, tier, trajectory_id, policy, app_actual_b100, cost_b100, app_actual_b200, cost_b200, ended, ended_at_apps, return_b100, return_b200 }`; `pub fn log_regret(cost: u64, best: u64) -> Option<f32>`; `pub fn spread_report(rows: &[TrajectoryRow]) -> SpreadReport` | additive |
| `pixelflow-pipeline/src/bin/gen_r2g_trajectories.rs` (new) | `--corpus-dir`, `--manifest`, `--corpus <file>` (OOD), `--name-prefix`, `--tier train\|dev\|ood`, `--train-limit 800` (stride over classical), `--n-rand 6`, `--mix 1/4:1,1/4:2,1/2:1`, `--strict-checkpoint`, `--train-guide-report`, `--label-b 100,200`, `--out-records <jsonl>`, `--out-trajectories <jsonl>`, `--report-json`, `--report-md`; outputs `pixelflow-pipeline/data/r2g_{train,dev,sh,bezier}.jsonl`, `r2g_trajectories_{…}.jsonl`, `docs/results/<date>-r2g-dataset.{json,md}` (spread quantiles, zero-spread shares, per-policy mean return, `zero_best_excluded`, fence result, `uptime` context) | binary |
| `pixelflow-pipeline/src/bin/train_guide.rs` | `--objective strict-bce\|return-mse\|return-rank` (default `strict-bce`, unchanged behaviour), `--target centered\|raw`, `--label-b 100\|200`, `--pairs-per-record 8`; checkpoint gains `objective`, `target`, `label_b`, `dataset_fnv1a64`, `policies`, `k_trajectories`; default out `pixelflow-pipeline/data/guide_checkpoint_r2g_v1.json` when objective is return-*; report gains DEV MSE, DEV Spearman(f, ŷ) per expression pooled, and per-rule mean predicted return | changed trainer |
| `pixelflow-pipeline/src/bin/skew_test_linear_guide.rs` | `--model strict\|return`; return mode checks `\|Model::logit + LinearReturnGuide::score\| ≤ 1e-6` over ≥ 1,000 `r2g_dev.jsonl` records; writes `docs/results/<date>-skew-test-r2g.json` | mandatory |
| `pixelflow-pipeline/src/bin/r2g_counterfactual.rs` (new) | `--corpus-dir`, `--corpus`, `--name-prefix dev_sh_`, `--n-expr 30`, `--n-apps 20`, `--budget 100`, `--r2g-checkpoint`, `--strict-checkpoint`, `--train-guide-report`, `--seed`, `--out-jsonl`, `--out-json`, `--out-md`; one row per sampled application (`expr_name, set, ordinal, sweep, rule_idx, rule_name, changed, cost_with, cost_without, delta, adv_r2g, adv_strict_v1, adv_per_rule, bit_loose_at_b, bit_tight_at_b, bit_strict_at_b, bit_strict_class_at_b, bit_*_full`); aggregate Spearman per proxy per set + pooled, nonzero-Δ share, bootstrap CIs; `docs/results/<date>-r2g-counterfactual.{jsonl,json,md}` | binary |
| `pixelflow-pipeline/src/bin/phase3_at_budget_eval.rs` | `--r2g-checkpoint <path>` adds arm `r2g` (`ARM_NAMES` gains `"r2g"`); the §7 table (`m_r2g − m_linear` against `M_B`, per set and tier, head-to-head counts) in the JSON/MD; `docs/results/<date>-phase3-r2g-{dev,sh,bezier}.{jsonl,json}` + `-report.md`, combined `<date>-phase3-r2g-comparison.{json,csv,md}`; journal record `phase3_r2g` | evaluation |

| `pixelflow-pipeline/src/training/r2g.rs` (round 3) | `pub const BUDGET_LADDER: [usize; 6] = [100, 200, 400, 800, 1600, 3200]`; `pub struct CheckpointRow { budget, app_actual, cost, stop: String, return_val: Option<f32> }`; `TrajectoryRow` gains `expr_node_count: usize` and `checkpoints: Vec<CheckpointRow>` (both `#[serde(default)]`, additive — `app_actual_b100`/`cost_b100`/etc. unchanged) | additive |
| `pixelflow-pipeline/src/bin/gen_r2g_trajectories.rs` (round 3) | `--max-expr-nodes 0` lifts the size filter; `--max-classes` becomes `Option<usize>`, resolved from `pixelflow_search::egraph::config_for_node_count(usize::MAX).max_classes` when unset; `--budgets` (default `100,200,400,800,1600,3200`, first two entries pinned to 100/200); per-expression `PER_EXPR_WALLCLOCK_CEILING` (10 min, post-hoc — excludes and loudly reports an over-ceiling expression rather than writing partial output); run-level `\|R\|`-scaled wall-clock assert (panics if the aggregate run is far slower than every per-expression skip explains) | changed binary, additive flags |
| `pixelflow-pipeline/src/bin/r2g_spread_vs_budget.rs` (new, round 3) | `--data-dir`, `--tiers`, `--out-json`, `--out-csv`, `--out-md`; reads `r2g_trajectories_{tier}.jsonl`'s `checkpoints` field and reports, per `(tier, budget, node-count band ∈ {51-100,101-250,251-1000,>1000,all})`: `zero_spread_all_share` (12-way), `zero_spread_guided_share` (11-way, excludes `unguided` — the credit signal), `qualifies` (`zero_spread_guided_share < 0.5`), spread quartiles, `unguided_differs_share`, a typed stop-reason histogram; `docs/results/2026-09-01-r2g-spread-vs-budget.{json,csv,md}` | binary |

Everything in this table is additive to the crate surfaces named; no existing public item changes
signature; `pub(crate)` stays `pub(crate)`.

## 9. Revision log

- **2026-09-01, first commit:** denotation (§1), diversity as dataset (§2), linear R2G model (§3),
  counterfactual validation (§4), build list (§5), parked rungs (§6), pre-registered comparison (§7),
  entry points (§8).

Appended 2026-09-01 after the run; §0–§7 above are unrevised. Full report:
`docs/results/2026-09-01-guide-return-to-go.md` (+ `.json`, `.csv`).

- **Dataset gate (§2): FIRED** — 67.2 % of TRAIN classical expressions have `spread_e(100) = 0`
  (DEV 68.0 %, `sh` 0 %, `bezier` 25 %). The registered action was "do not train"; training and
  evaluation proceeded on the orchestrator's explicit direction as the exploratory completion of the
  round, scored against every gate as written.
- **Model (§3):** `LinearReturnGuide`, `--target centered --label-b 100 --loss mse`, lr 1e-4 / clip 1
  (the defaults diverged: TRAIN loss 0.871 vs a zero-predictor floor of 0.705; the sweep was
  selected on TRAIN loss only). TRAIN final MSE 0.7047 vs floor 0.7054; DEV MSE 0.4946 vs 0.4945,
  DEV Spearman 0.099. Skew test PASS, 0/5000, max diff 0.000e0. Checkpoint
  `guide_checkpoint_r2g_v1.json` MD5 `73b7db7bf75d13c94824f7826830a021`.
- **Claim B (credit): FAILS.** S = 1,095 applications (600 `sh`, 495 DEV), Δ 92.4 % zero / 5.6 % > 0 /
  2.0 % < 0. Spearman vs Δ, pooled: **R2G −0.004** [−0.062, +0.051]; strict-v1 linear 0.170; per-rule
  0.182; loose/tight undefined (all true); **strict 0.389** [0.304, 0.477]; strict-by-output-class 0.005.
  Paired-bootstrap Δρ vs R2G excludes zero for strict, per-rule and strict-v1 on the pooled sample.
- **Claim A (ordering): FAILS on `sh`** (`m_r2g` 0.9084 vs < 0.8439 at B = 100; 0.9017 vs < 0.8259 at
  B = 200), **holds on DEV** (0.5493 ≤ 0.5966; 0.7137 ≤ 0.7659). `|m_r2g − m_linear| ≤ M_B` on all six
  cells (+0.013, +0.018, +0.005, +0.006, −0.053, −0.029). Head-to-head r2g < / = / > strict-bit: DEV
  25/102/207 and 10/183/141; `sh` 21/1/73 and 5/15/75; `bezier` 60/20/0 and 19/61/0.
- **Pre-committed row that fires: "R2G ties the strict bit everywhere" → the target was not the
  bottleneck.** `pythagorean`: 0 firings under R2G and under the strict bit at every checkpoint on
  `sh`; absent from S (unguided reaches it only past B = 200), so its Δ at B = 100 is unmeasurable.
- **Next (per the row):** context features (`claude/phase3-context`), gated by
  `counterfactual_credit --r2g-checkpoint/--strict-checkpoint` (Spearman vs Δ) before any ladder
  run. §6's transformer pull-forward condition is not met (the linear model beat no bound).
- Kill-gate accounting: one clean round consumed (Round 1, Round 1b, this round).

**Round 3 (2026-09-01), "measure spread where the budget binds":** full §2b above. Re-mint with
`--max-expr-nodes 0` (size filter lifted) and `--max-classes` resolved from production's own
`config_for_node_count` (5,000) rather than the round-1/2 `--max-classes 2000` inherited from
`gen_strict_labels`; budget ladder `100..3200` in place of B∈{100,200}. TRAIN 3,356 / DEV 783 / `sh`
100 / `bezier` 80 expressions, no wall-clock skips. Finding: TRAIN and DEV agree exactly on five
qualifying `(B, band)` cells — 101-250 nodes at B∈{100,200}, 251-1000 nodes at B∈{100,200,400} —
where guided-among-themselves spread is non-zero for ≥50% of expressions; every other cell
(band 51-100, the pooled population, and both bands past their budget cutoff) does not qualify.
No expression in any split exceeds 1,000 nodes, so this corpus does not reach production's
measured tail (p90/max ~24k/86k applications) regardless of node-count filtering — a corpus-design
gap, not a training-target gap. Selection rule registered in §2b.3: training regime restricted to
the five agreeing cells. Data: `docs/results/2026-09-01-r2g-spread-vs-budget.{md,csv,json}`,
`docs/results/2026-09-01-r2g-trajectory-mint-full.{json,md}`.

**Round 4 (2026-09-01), "train inside the selected regime, and fix the credit instrument":** the
first training run against §2b.3's regime, plus the confluence-aware second mask mode §4.1 was
missing. Full numbers: `docs/results/2026-09-01-guide-r2g-spread-first.{md,json,csv}`.

- **§1.3 was never enforced by the mint.** `gen_r2g_trajectories` attaches a trajectory's return to
  every application it records, including ones firing thousands of applications after the checkpoint
  the return was read at. Round 1/2's ladder topped out at B = 200 so the damage was bounded; round
  3's ladder reaches 3,200, so the great majority of records carried a label for a measurement they
  came after. `train_guide_r2g --enforce-label-ordinal` (new, default off so pre-round-3 runs
  reproduce bit-for-bit) is what enforces the registration. Of 18,565,784 TRAIN records read,
  17,283,072 are rejected by the band + ordinal filters; 1,282,712 train.
- **Result: trainable, still nearly flat.** DEV Spearman(predicted, realized) 0.2375 (round 2:
  0.0990), but DEV MSE 1.132686 against a zero-predictor floor of 1.133602 — **0.081% of the label
  variance**. A linear model on these features does not read this credit signal.
- **Ladder (B = 100/200; DEV classical restricted to the band, `sh`/`bezier` whole):** the
  regime-trained R2G no longer ties the strict bit. Per-expression head-to-head at B = 100 — DEV band
  101-1000 **70 / 119 / 19** (R2G better / strict better / tie), `sh` **43 / 45 / 7**, `bezier`
  **60 / 0 / 20** (median cost ratio 0.942). It loses in distribution, ties on `sh`, wins on
  `bezier`.
- **Credit instrument, the round's most load-bearing finding.** `MaskScope::AllMatchingCandidate`
  masks the seed and every later application sharing its `(rule_idx, canonical matched-class
  content)`, so an alternative re-derivation cannot silently restore what leave-one-out removed.
  On the round-3 sample (n = 1,095): **77 of 1,012 (7.6%) leave-one-out Δ = 0 applications become
  Δ > 0**; 435/1,095 multi-masks skipped more than the seed (mean 1.57, max 6). On that sample every
  proxy ranks better against the confluence-aware truth — `strict` 0.389 → 0.725, `strict_v1_linear`
  0.170 → 0.345, `per_rule_rate` 0.182 → 0.267, R2G(regime) 0.103 → 0.249. Inside the band
  (n = 3,000, 90/2,525 = 3.6% flip) the two truths *separate the bounds from the models*: `strict`
  still rises (0.510 → 0.565) while every model proxy falls (R2G 0.199 → 0.145, `per_rule_rate`
  0.288 → 0.202) — a learned score that tracks leave-one-out better than the confluence-aware Δ is
  fitting the instrument's zero-inflation, not the credit. **Round 3's ρ table — including its
  R2G ρ = −0.004 — was computed against a truth biased toward zero.** The strict bit remains the
  best predictor under both masks and on both samples.
- Kill-gate accounting: unchanged (this round trains a new checkpoint against a registered regime
  and reports it; no clause is claimed or waived).
