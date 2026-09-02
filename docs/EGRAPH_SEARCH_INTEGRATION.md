# E-Graph Search Integration

> Rewritten 2026-08-05. The previous version of this doc described `best_first.rs` (A*)
> and `guided.rs` (MCTS) as existing-but-unintegrated; both were deleted in the July 2026
> cleanup along with the rest of the RL guidance stack.

## What the compiler does today

`pixelflow-compiler/src/optimize.rs` builds an e-graph from the expression AST, saturates
it with the full rule set (`pixelflow_search::math::all_rules()`, 62 rules — count is
machine-checked by `all_rules_count` in `pixelflow-search/src/math/mod.rs`), and extracts
the minimum-cost DAG. Extraction cost comes from `CostModel::latency_prior()`, the only
policy; a learned (NNUE) extraction head was tried, tied the table on schedule-free kernels,
and had its shape deleted in 2026-09 — its seam and denotation are kept (see
[NNUE_INTEGRATION_STATUS.md](NNUE_INTEGRATION_STATUS.md) and
[plans/2026-09-01-schedule-cost-model-denotation.md](plans/2026-09-01-schedule-cost-model-denotation.md)).

Saturation entry points: `EGraph::saturate` / `saturate_with_limit`
(`pixelflow-search/src/egraph/graph.rs`) and the budget-tracking
`saturate_with_budget` (`pixelflow-search/src/egraph/saturate.rs`).

There is no A*, MCTS, or learned rule-masking in the compiler path. Guided saturation is a
research direction, not a shipped feature; its supervision substrate (rule provenance +
hindsight labeling, `provenance.rs` / `labeler.rs`) exists and is the basis of the plan in
[2026-07-07-guided-saturation-redesign.md](plans/2026-07-07-guided-saturation-redesign.md)
and the workflow in
[2026-08-05-egraph-nnue-research-workflow.md](plans/2026-08-05-egraph-nnue-research-workflow.md).
