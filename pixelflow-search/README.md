# pixelflow-search

`pixelflow-search` is PixelFlow's arena-native algebraic optimization research crate. Its
implemented core builds an e-graph from `pixelflow-ir::ExprArena`, applies rewrite rules under
explicit budgets, records rule provenance, and extracts an equivalent arena with a pluggable
cost function.

Learned rule guidance (the saturation Guide) is an experiment with decision gates—not the
default optimizer and not a completed training product. A learned extraction cost model was
tried and tied the static table on schedule-free kernels (see below); the static latency prior
is the extraction policy, and the learned model's seam (`Reranker`) is kept for a future
schedule-cost residual.

## Status at a glance

| Capability | Status |
|---|---|
| Arena ↔ e-graph conversion | Implemented |
| Algebraic, transcendental, fusion, and derivative rewrites | Implemented and tested for soundness |
| Time/class/iteration-bounded saturation | Implemented |
| Static latency-prior extraction | Implemented; compiler default |
| DAG-aware extraction and arena reconstruction | Implemented |
| Rule-application provenance and hindsight labels | Implemented |
| Learned (NNUE) extraction cost model | Shape deleted 2026-09 (tied the table on schedule-free kernels); `Reranker` seam kept, denotation in `docs/plans/2026-09-01-schedule-cost-model-denotation.md` |
| Trained provenance guide for choosing rewrites | Research plan; thesis experiment not yet completed |
| Beam/lookahead search over rewrite applications | Conditional future work |

The obsolete critic/REINFORCE dual-head architecture is retained in historical documents for
rationale, but it is not the live training path. See
[`guided-saturation-redesign.md`](../docs/plans/2026-07-07-guided-saturation-redesign.md).

## Deterministic optimization path

```text
ExprArena
   │
   ▼
EGraph::add_arena
   │
   ▼
budgeted rewrite saturation
   │
   ▼
CostModel / other CostFunction
   │
   ▼
extracted ExprArena
```

`EGraph::new()` deliberately contains no rules. Applications supply a rule set explicitly,
usually `all_rules()`, so a test or compiler phase can constrain the algebra it permits.

```rust
use pixelflow_ir::{ExprArena, OpKind};
use pixelflow_search::egraph::{CostModel, EGraph, all_rules, saturate_for_extraction};

let mut arena = ExprArena::new();
let x = arena.push_var(0);
let zero = arena.push_const(0.0);
let root = arena.push_binary(OpKind::Add, x, zero);

let node_count = arena.node_count_subtree(root);

let mut egraph = EGraph::with_rules(all_rules());
let root_class = egraph.add_arena(&arena, root);
saturate_for_extraction(&mut egraph, node_count);

let costs = CostModel::default();
let (optimized, optimized_root, estimated_cost) =
    egraph.extract_best(root_class, &costs);

assert!(optimized.node_count_subtree(optimized_root) > 0);
assert!(estimated_cost <= costs.cost(OpKind::Add));
```

`saturate_for_extraction()` is the production entry point: it sizes the iteration, e-class,
and wall-clock budget from `node_count`, so a three-node expression is not given a budget
meant for a hundred-node one. It is the only entry point that decides a budget.

`EGraph::saturate_with_limits()` is the other one — the same loop with the three limits
supplied by the caller, for tests and experiments that must pin an exact stopping policy
rather than inherit the size-tiered one. Those call sites spell the budget as
`SaturationConfig::compatibility(rounds)` instead of repeating literals.

## Cost model

`CostFunction` is the extraction interface. The standard `CostModel` is an O(1) table indexed
by `OpKind`, with optional depth penalties.

`CostModel::default()` and `CostModel::latency_prior()` use the same handcrafted per-operation
latency prior. This is the compiler's normal extraction policy. The values are estimates, not
portable benchmark results; calibration and learned alternatives belong to the research
tooling.

Both compiler tiers (the `kernel!` macros and runtime-built kernels) choose the policy through
one seam, `egraph::env_extraction_policy()`, which returns the latency prior. A learned
extraction head sat behind that seam from 2026-07 to 2026-09 and was closed as an honest
negative — the static table tied it and every lever made it worse (the workshop paper on
branch `claude/workshop-writeup`, PR #1072, closed without merging — not in this tree; see
[`docs/plans/2026-09-01-schedule-cost-model-denotation.md`](../docs/plans/2026-09-01-schedule-cost-model-denotation.md)
for the citations and numbers in-repo). It is deleted, not disabled.

Extraction selects an expression from an already-built e-graph. It should not be confused
with a learned policy that decides which rewrite applications to admit during saturation.

## Provenance and labels

The e-graph records three append-only kinds of evidence:

- the origin of each stable e-node identity;
- each rule application and the nodes it matched or created;
- union events between e-classes.

After extraction, `EpisodeLabels` walks the winning derivation and labels applications as
load-bearing or unused for that episode. `run_episode` packages saturation, static extraction,
and hindsight labeling for a supplied rule set.

This machinery supports rule audits today. Its proposed use as supervised training data for a
Guide remains research; provenance itself does not prove that a trained guide improves the
optimizer.

Run the checked-in report over its small illustrative corpus:

```bash
cargo run --release -p pixelflow-search --example rule_report
```

The report is corpus-conditioned. Do not generalize rule rankings from its five example
kernels to production workloads without a representative corpus.

## Guided-saturation research

The current research question is whether a learned rule filter can keep a large rewrite
library inside a fixed budget while preserving the extraction quality of smaller full
saturation. The plan deliberately separates stages:

1. Provenance and hindsight labels — landed.
2. Static latency-prior baseline — landed and default.
3. Greedy guide trained from provenance — not yet validated.
4. Lookahead search — considered only if the greedy experiment exposes missed enabling chains.

Each stage has a stop condition. The repository does not claim that guided saturation, an
AlphaZero-like loop, or a learned cost model has beaten the static baseline.

## Modules

| Module | Purpose |
|---|---|
| `egraph` | Graph, rewrites, saturation, extraction, provenance, and labeling |
| `math` | Algebraic, calculus, parity, exponential, and trigonometric rewrite families |
| `nnue` | Op embeddings, the typed edge stream, the saturation Guide, and the backward expression generator |

Training corpora, JIT measurement, and extraction comparisons live in `pixelflow-pipeline`
rather than in this crate. Commands documented here are limited to targets that currently
exist; older `pixelflow-compiler` training examples were removed with the superseded RL path.

## Test

```bash
cargo test -p pixelflow-search
```

Rewrite-soundness tests evaluate both sides through the IR's reference semantics. Optimization
results and benchmark reports remain point-in-time evidence, not compatibility guarantees.

## Related documents

- [`guided-saturation-redesign.md`](../docs/plans/2026-07-07-guided-saturation-redesign.md) —
  current research plan and decision gates
- [`docs/results/`](../docs/results/) — rule and extraction experiment reports
- [`EGRAPH_OPTIMIZATION_ARCHITECTURE.md`](../docs/EGRAPH_OPTIMIZATION_ARCHITECTURE.md) —
  superseded critic/REINFORCE architecture, retained for history

## License

[Apache License 2.0](../LICENSE.md)
