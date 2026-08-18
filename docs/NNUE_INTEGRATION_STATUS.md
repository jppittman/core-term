# NNUE Integration Status

> Rewritten 2026-08-05. The previous version of this doc described systems deleted in the
> July 2026 cleanup (`train_unified`, the Guide/self-play loop, `critic_server.py`) as extant.

## What exists today

- **Extraction head** (`ExprNnue`, `pixelflow-search/src/nnue/factored.rs`): predicts
  expression execution cost from an `EdgeAccumulator` built over `ExprArena`. Consumed by
  e-graph extraction (`pixelflow-search/src/egraph/extract.rs`,
  `extract_neural_to_arena`).
- **Compiler integration** (`pixelflow-compiler/src/optimize.rs`): the DEFAULT extraction
  cost model is the static `CostModel::latency_prior()`. The NNUE extraction head is
  **opt-in only**, via the `PIXELFLOW_NNUE_WEIGHTS` env var read at proc-macro expansion
  time; bad weights hard-fail the build.
- **Training** (supervised, offline): `gen_bench_corpus` mints (expression, measured-ns)
  pairs via the JIT bench harness (`pixelflow-pipeline/src/jit_bench.rs`);
  `bootstrap_extraction_head` regresses the head on them. Both require
  `-p pixelflow-pipeline --features training`.
- **Gate**: `bench_extraction_3way` compares no-swap vs latency prior vs NNUE end-to-end.
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
