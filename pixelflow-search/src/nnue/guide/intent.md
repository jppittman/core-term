# pixelflow-search/src/nnue/guide

**Is.** The saturation Guide: an ordering policy over the rule applications
available this round (DENOTATION §8, law 4, guide neutrality). It may only
order and truncate; it mints no nodes and performs no unions, so it changes
cost and compile time, never meaning. Trained offline; inert in production.

## Owns
- The seam: `SaturationGuide`, one batched call per round,
  `score_candidates(&[CandidateSummary]) -> Vec<f32>`, higher fires first.
- `CandidateSummary`, all a guide may know of a candidate, built only by `::new`.
- The deployed policies, and the refusal to deploy weights against a rule
  vocabulary they were not trained on (`GuideError`, `Fingerprint`), and the
  head's private representation (`scoring.rs`, `accumulator.rs`).

## Never
- A guide is never handed an e-graph: `CandidateSummary` holds owned values,
  no `Saturation` or `EClassId`, and `score_candidates` takes `&self`.
- Nothing here builds a `RewriteAction`, calls `union` or touches `const_fact`;
  the only mutation is `Saturation::apply_single_rule`, in `egraph/guided.rs`.
- No weight is keyed by rule position: `RuleId` plus the vocabulary
  `Fingerprint`, so a same-length reorder is refused, not repointed.
- No file, JSON, clock or RNG state; weights arrive as parsed values and
  `UniformRandomGuide` hashes `(seed, candidate)`, so rescoring is stable.
- No caller outside `pixelflow-search` and `pixelflow-pipeline`;
  `Optimizer::production()` sets no guide.

## Contract
`mod.rs`: `SaturationGuide`, `CandidateSummary`, `Guide` (random head).
`linear`: `LinearWeights`, `LinearCandidateGuide`, `LinearReturnGuide`,
`PerRuleRateGuide` (control arm), `GuideError`. `bilinear`: `BilinearWeights`,
`BilinearCandidateGuide`, `BilinearTrainer`. `diversity`: `UniformRandomGuide`,
`uniform_unit_score`. `accumulator`, `scoring` (with `backward`) are private.

## Enforced by
- Types: the trait signature is the neutrality argument. `egraph/guided.rs`
  asserts one finite score per candidate and sorts stably, so a fixed guide
  gives a fixed run; its tests pin dedup before scoring and the budget as a delta.
- `linear.rs` `a_same_length_reorder_is_refused`; `bilinear.rs`
  `new_should_refuse_weights_from_a_different_rule_vocabulary`.
- `scoring/representation.rs`: the additive guide's rule-by-context
  interaction is identically zero; the bilinear head's is not.

## Behind
- No test pins law 4. `egraph/guided.rs:23` cites an L4 test with a reversing
  guide and `egraph/optimizer.rs:34` cites `laws.rs`; neither exists
  (2026-10-15), and `optimizer-api` §6 lists `tests/optimizer_laws.rs` as
  deleted. The crate intent records the same gap.
- `mod.rs` still reads "Phase 3 — wiring in progress". The Guide's value
  claims were retracted 2026-09-07 (ledger L047, L087); it is an instrument
  until `benchmark-correction` §E item 5 is taken in DAG units.
- `accumulator.rs` and `scoring.rs` carry `#![allow(dead_code)]`: tests-only
  callers, the whole-graph Judge seam (J10, ROADMAP-ADMITTED).

## Decided in
`docs/DENOTATION.md` §8, `docs/plans/2026-09-02-optimizer-api.md` §1 (L4),
`docs/plans/2026-07-07-guided-saturation-redesign.md`,
`docs/plans/2026-08-31-guide-design-revision.md` §4, `docs/plans/2026-09-01-guide-return-to-go.md`,
`docs/plans/2026-09-02-bilinear-guide-registration.md` §4, `docs/plans/2026-09-07-benchmark-correction.md` §E.
