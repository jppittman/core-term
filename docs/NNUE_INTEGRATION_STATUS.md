# NNUE Integration Status

> **Closed 2026-09-01.** The extraction-head program this doc tracked — a learned NNUE cost
> model for e-graph extraction, opt-in behind a weights-file env var — was closed as an
> honest negative (the static latency table ties it; every lever made it worse:
> [`paper/2026-08-egraph-nnue-parity.md`](paper/2026-08-egraph-nnue-parity.md)) and
> deleted per JP's ruling ("delete. program closed, history belongs in vcs."). The value
> head, its checkpoint format, the incremental extractor, the trainer and the 3-way gate are
> in git history, not in the tree. What survived is what the successor program (the
> saturation Guide, [`plans/2026-08-31-guide-design-revision.md`](plans/2026-08-31-guide-design-revision.md))
> reuses by denotation: op embeddings seeded from the latency prior, the typed edge stream,
> the graph accumulator and scoring head, provenance/labeling, and the bench harness.
>
> Rewritten 2026-08-05. The previous version of this doc described systems deleted in the
> July 2026 cleanup (`train_unified`, the Guide/self-play loop, `critic_server.py`) as extant.

## What exists today

- **Compiler integration** (`pixelflow-compiler/src/optimize.rs`, `pixelflow-search/src/runtime.rs`):
  extraction cost is the static `CostModel::latency_prior()`, chosen by both tiers through
  `egraph::env_extraction_policy()` — the one seam a future policy would be chosen at.
- **Op embeddings and the typed edge stream** (`pixelflow-search/src/nnue/factored.rs`):
  `OpEmbeddings` (dimension 0 seedable from the latency table) and `EdgeTrace`/`CostEdge`,
  the parent→child edge walk over an arena or an e-graph extraction — reused by the Guide.
- **Corpus minting** (`gen_bench_corpus`, `-p pixelflow-pipeline --features training`) via the
  JIT bench harness (`pixelflow-pipeline/src/jit_bench.rs`), with the shared quarantine,
  tiered split and sentinel-normalized labels. `measure_latency_prior` re-derives the table.
- **Provenance substrate** for guided-saturation research: rule provenance
  (`pixelflow-search/src/egraph/provenance.rs`) and hindsight labeling
  (`pixelflow-search/src/egraph/labeler.rs`).

## What no longer exists

The AlphaZero-style self-play/critic/REINFORCE loop (removed July 2026 after a four-agent
audit found it methodologically unsound), `train_unified`, `GuideNnue`, and all RL training
binaries. Post-mortem and replacement architecture:
[2026-07-07-guided-saturation-redesign.md](plans/2026-07-07-guided-saturation-redesign.md).

## Where this is going

The research workflow toward a publishable extraction-head result (harness repairs, split
discipline, iteration loop):
[2026-08-05-egraph-nnue-research-workflow.md](plans/2026-08-05-egraph-nnue-research-workflow.md).
