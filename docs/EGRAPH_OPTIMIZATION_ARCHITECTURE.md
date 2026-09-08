# E-Graph Optimization Architecture

**Classification:** Historical/superseded (annotated 2026-09-01, not rewritten).
The NNUE-scored `IncrementalExtractor` step and the `unified_backward.rs` gradient path
this document describes belonged to the extraction-head program, whose shape was deleted
on 2026-09-01 after it tied the static table (workshop paper on branch
`claude/workshop-writeup`, PR #1072, closed without merging — not in this tree; see
`plans/2026-09-01-schedule-cost-model-denotation.md` for the citations and numbers in-repo);
extraction is the static latency prior only (`CLAUDE.md`, "Compiler Pipeline"), and the
swap search survives behind an unimplemented `Reranker` seam
(`plans/2026-09-01-schedule-cost-model-denotation.md`). The live learned program is the
saturation Guide (`plans/2026-08-31-guide-design-revision.md`).
The rest of this file is left as it was written.

This was the optimization and training spine as of mid-2026.

## Canonical Representation

`pixelflow-ir` owns the expression IR. Arena-backed expressions are the canonical form used for rule templates, extraction output, and the active training path.

## Optimization Path

1. The compiler or pipeline inserts an expression into `pixelflow-search::egraph::EGraph`.
2. Rewrite rules from `pixelflow-search::math` saturate the graph.
3. `IncrementalExtractor` evaluates extraction candidates with `EgraphOptimizationModel`.
4. The result is exported as arena IR and handed back to the compiler, JIT, or benchmark path.

## Model

`EgraphOptimizationModel` has one shared backbone and two heads:

- `ExtractionHead`: predicts log-cost for extracted expressions
- `SaturationHead`: scores rewrite applications from graph state plus rule embeddings

Rule embeddings come from arena-backed `RuleTemplate` values, not from a separate legacy expression format.

## Training Path

The live training loop is in `pixelflow-pipeline`:

- [`train_unified.rs`](../pixelflow-pipeline/src/bin/train_unified.rs): outer loop
- [`self_play.rs`](../pixelflow-pipeline/src/training/self_play.rs): trajectory generation
- [`unified_backward.rs`](../pixelflow-pipeline/src/training/unified_backward.rs): analytical gradient path

The critic boundary is the unified trajectory format in [`unified.rs`](../pixelflow-pipeline/src/training/unified.rs).

## What Is Gone

These are not part of the current architecture:

- best-first search
- guided-search sidecars
- tree/adapter bridges in the live path
- split legacy vs arena rule-template APIs
- judge/guide naming as separate top-level systems
