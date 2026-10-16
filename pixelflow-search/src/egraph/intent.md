# pixelflow-search/src/egraph

**Is.** The optimizer's mechanism (DENOTATION §8): an e-graph whose classes are sets of terms equal in
meaning, grown by rewrite rules under a budget counted in rule applications, and read out by extraction
under a cost. `Optimizer::run` is its one entry; the five laws of §8 are its contract.

## Owns
- The store and the runner: `EGraph` (hash-consing, `union`, `rebuild`; a node arena whose index is the
  node's id and its provenance tag, a member-id list per class and the live set the store restates at every
  write to one, the per-class constant and variance facts, the per-kind class index a rule's scan reads) and
  `Saturation`, the store paired with its `RuleSet` and every count denominated in that sweep. Rule identity (`RuleId`, `Fingerprint`) and admission (`Vocabulary`, `template_may_name`).
- The budget: `Budget` resolves to `Limits` (iterations, minted ids, applications); `SaturationConfig`
  picks the tier from the input.
- Extraction and cost: the tree and shared DP arms and their arbiter, `Selection`. `cost(If) = cost(m) +
  (cost(a)+cost(b))/2 + cost(select)`; a fold costs `len·body + (len−1)·combine`; a surviving `Dwrt`
  costs `LEGALIZATION_PRICE`. The fold rewrites and the chain rule are here; other rule bodies are `crate::math`.
- The guided loop, provenance journal, labeler and growth telemetry: research, off in production.

## Never
- The sweep reads no clock: `Limits` holds no duration and `saturate_budgeted` takes none. The one
  clock read is `Optimizer::run`'s assertion afterwards.
- A guide orders and never mutates: `SaturationGuide::score_candidates` sees summaries, and `guided.rs`
  commits only through `apply_single_rule`.
- Unequal constants are never unioned: `EGraph::union` refuses, recording `refused_const_unions`.
- No rule names `Gather`, `Reduce`, `Shl` or `Shr`: a fold is its own node, a shift's count a `ShiftCount`.
- No estimate of how often a mask is true enters a cost. No register, lane, width or ISA tier is named.
- The journal is not in the budget's denominator: `Saturation::application_count` is unconditional.

## Contract
`mod.rs` declares the modules and re-exports `Optimizer`, `Budget`, `Optimized`, `EGraph`, `Saturation`,
`Limits`, `SaturationStop`, `insert`, `Declined`, `Vocabulary`, `Rewrite`, `RewriteAction`, `RuleSet`,
`SaturationConfig`, `CostModel`, `CostFunction`, `extract_dag`, `ExtractedDAG`, `Provenance`.

## Enforced by
- Types: `Selection::of_map` refuses `Missing`, `OutOfBounds`, `Cyclic`; `LiveTable` cannot be indexed
  by an `EClassId`; `template_may_name` is an exhaustive match, so a new `OpKind` fails the build.
- L1, L2: `graph.rs` tests `over_budget_growth_cannot_assert_a_false_equality` and
  `saturation_at_any_budget_never_removes_an_equality`. L3 well-foundedness: `the_dp_map_is_well_founded`.
- Budget: `tests/production_budget_determinism.rs`, `tests/saturation_stop.rs`,
  `tests/saturation_ceiling_env.rs`; constants: `tests/fold_exactness.rs`; rules and vocabulary:
  `tests/root_shapes.rs`, `tests/ir_insert.rs`; cost: `tests/latency_prior_regression.rs`.
- `scripts/check-provenance-journal-scope.sh` (the journal types exist only under `provenance-journal`);
  desloppify `control-plane-64-bit` (`EClassId` is `u32`, its measurement in `node.rs`).

## Behind
- `optimizer.rs` line 34 cites a `laws.rs` that is not in the tree; `guided.rs` cites an L4 test with a
  reversing guide that does not exist; the plan's `optimizer_laws.rs` was deleted. L4 has no test.
- L5's argmin half is violated and two docs lag the code: DENOTATION §8 and the `Optimizer::cost` doc say
  sharing is never priced, but `extract_dag_over` also runs `shared_dag_dp_pass` and keeps the cheaper by
  `dag_cost`. Both arms are greedy (BACKLOG E5); a fold ties its unrolling and insertion order decides
  (E6); `Reranker` has no implementation outside tests.
- No test pins `cost(If)` directly; `SETTLEMENT_DIGESTS` in `extract.rs` holds it.
- `Vocabulary` still admits `Gather` and `ENode::Buffer`; the language has no bound buffers
  (kernels-evaluate-kernels, 2026-10-15). `mod.rs` lists an `extraction` module that does not exist and
  says `all_rules` has 42 rules; `rule_order.rs` asserts 62.
- The store's nodes are an arena and node ids are the provenance tags (CL 3b), and extraction names the
  node it picks by id (CL 3c; choice maps still record positions). Membership is still a per-class
  `Vec<ENodeId>`: CLs 4 and 7 (membership as a list, freeze drops saturation state) are open.

## Decided in
`docs/DENOTATION.md` §3, §4, §8; `docs/archive/plans/2026-09-02-optimizer-api.md`, `2026-09-01-production-budget-determinism.md`,
`2026-09-04-ir-as-a-trait.md`, `2026-09-08-extraction-witnesses.md`, `2026-09-09-a-fold-is-a-node.md`,
`2026-10-15-kernels-evaluate-kernels.md` §1; `docs/compiler/2026-10-15-the-egraph-is-an-arena.md`.
