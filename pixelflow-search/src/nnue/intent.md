# pixelflow-search/src/nnue

**Is.** Not a stage of `P`. The vocabulary learned programs read the optimizer
through, kept after the extraction head's shape was deleted (2026-09-01): an op
table seeded from the latency prior, a DAG as an ordered edge multiset, a
variance histogram, a backward generator of equal-term pairs. A learned
extraction cost model is not a concept; the `Reranker` seam is (DENOTATION §8
law 5, §11). `guide/` carries its own `intent.md`.

## Owns
- `OpEmbeddings`, the crate's only learned op table: dimension 0 is
  `latency_prior_cycles()` log-squashed, so an untrained op shares a scale with
  the table the extractor prices by.
- The edge stream. `walk_cost_dag` is the only function that turns a DAG into
  `CostEdge`s: a node's first reference is its edge, later ones `parent -> Var`.
- `variance_histogram`, behind `Extraction::chosen_variance`.
- `BwdGenerator` and `RuleTemplates`: mint a term, then junkify it with rule
  templates run backward.

## Never
- No `impl Reranker` and no price for an extraction: the policy is
  `CostModel::latency_prior()`, in `egraph/`.
- No second walker: `EdgeSink` and `CostDag` are private, so a consumer takes a
  recorded `EdgeTrace` and cannot choose an edge policy.
- No weights file or env switch: nothing in `src/` reads `PIXELFLOW_NNUE_WEIGHTS`
  outside a test guard.

## Contract
`mod.rs`: `factored`, `guide`; re-exports `CostEdge`, `EdgeTrace`, `OpEmbeddings`,
`PeSlot`, `RuleTemplates`; defines `pattern_match_arena`, `BwdGenerator`.

## Enforced by
- Types: `PeSlot` is built only by clamping, so a value is a valid depth row.
- `nnue::factored::tests`, `nnue::tests::bwd_generator_*`, and in `extract.rs`
  `a_materialized_selection_records_the_edge_stream_of_its_arena` and
  `a_chosen_term_classifies_a_known_variance_mix_as_its_arena_does`.
- Nothing enforces the first Never line; the one `impl Reranker` is a test
  type, `TableReranker`, found by grep.

## Behind
- `Reranker` has no implementation and nothing calls `Optimizer::rerank`. Its
  premise, a schedule cost in extraction, moved to the CPU backend on 2026-10-15;
  the seam wants revisiting (kernels-evaluate-kernels §3).
- `EdgeTrace`, `CostEdge`, `PeSlot` are read only by tests in `extract.rs`
  (2026-10-15).
- `variance_histogram` has four buckets over X and Y; a node depending only on a
  fold binder lands in "frame-uniform", though `Variance` is a set of binders
  (DENOTATION §7). `extract.rs` imports it from here, so extraction depends on
  this directory.
- `BwdGenConfig::num_vars` defaults to 4, "X,Y,Z,W"; indices 2 and 3 are no
  longer axes (DENOTATION §11). Its one non-test consumer is the `corpus_gaps`
  bin of `pixelflow-pipeline`, and the corpus has no acceptance criterion since
  the quarantine gate went (CLAUDE.md).

## Decided in
`docs/compiler/2026-09-01-schedule-cost-model-denotation.md` (table plus residual
stands; premise retracted), `docs/archive/plans/2026-09-01-dead-code-with-ideas.md`,
`docs/archive/plans/2026-08-17-cost-model-domain.md`, `docs/DENOTATION.md` §8 and §11,
`docs/language/2026-10-15-kernels-evaluate-kernels.md` §3.
