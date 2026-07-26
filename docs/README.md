# PixelFlow Documentation

This page is the landing page for repository documentation. A document's classification
describes how it should be read today; directory names alone do not establish currency.

## Classification vocabulary

- **Current architecture** — describes the system as it exists now.
- **Plan of record** — approved or active direction for work not yet fully implemented.
- **Experiment/result** — an observation, audit, benchmark, investigation, or research
  reference; it is evidence rather than a promise about current architecture.
- **Historical/superseded** — retained for rationale or archaeology, but not authoritative.
- **Active bug** — a known unresolved defect or quality gap.
- **Fixed bug** — a defect report whose described fault has been corrected.

Classification is separate from design workflow status. **Every future plan and design must
use exactly one of `Draft`, `Review`, `Approved`, or `Implemented` in its metadata**, as
specified by [`templates/DESIGN_DOC.md`](templates/DESIGN_DOC.md). Use `Supersedes` or
`Superseded by` metadata to record lineage rather than creating new status words.

## Current architecture

- [`designs/2026-07-24-totality-and-the-cost-model.md`](designs/2026-07-24-totality-and-the-cost-model.md)
  — design-of-record axiom for the total kernel language. Some consequences it describes
  (typed discrete fields and cost re-denotation) remain plan-of-record work.
- [`designs/KERNELS_AND_LATTICES.md`](designs/KERNELS_AND_LATTICES.md) — implemented kernel
  and lattice substrate. Its milestone list is a point-in-time snapshot; use the newer
  kernel-unification plan for consumer migration status.
- [`designs/2026-07-25-two-level-ir-and-backend-completeness.md`](designs/2026-07-25-two-level-ir-and-backend-completeness.md)
  — audit of the lowering/backend boundary and the first landed instruction/assembler split;
  follow-up work is identified explicitly in the document.
- [`designs/pty-actor-troupe.md`](designs/pty-actor-troupe.md) — implemented PTY actor wiring.
- [`STYLE.md`](STYLE.md) — current coding and review conventions.

## Plan of record

- [`plans/2026-07-20-kernel-unification.md`](plans/2026-07-20-kernel-unification.md) — the
  active migration from type-level combinator emission to arena-backed `Kernel` values. Its
  phase annotations distinguish landed slices from future work.
- [`plans/2026-07-07-guided-saturation-redesign.md`](plans/2026-07-07-guided-saturation-redesign.md)
  — supervised guided-saturation research direction. Provenance and the static latency prior
  are landed; a trained guide and guided-saturation thesis test are not.
- [`designs/actor-scheduler-mealy-transducer.md`](designs/actor-scheduler-mealy-transducer.md)
  and [`designs/pixelflow-runtime-engine-mesh-migration.md`](designs/pixelflow-runtime-engine-mesh-migration.md)
  — draft actor/runtime designs. The scheduler primitives have landed; the runtime mesh
  migration has not.
- [`designs/LATTICE_EVAL.md`](designs/LATTICE_EVAL.md),
  [`designs/lattice-scheduling-types.md`](designs/lattice-scheduling-types.md),
  [`designs/REDUCTIONS_AND_FOLDS.md`](designs/REDUCTIONS_AND_FOLDS.md),
  [`designs/ML_AND_LINEAR_ALGEBRA.md`](designs/ML_AND_LINEAR_ALGEBRA.md), and
  [`designs/ML_AUTODIFF_PIPELINE.md`](designs/ML_AUTODIFF_PIPELINE.md) — related language
  and scheduling designs that remain directional rather than descriptions of completed code.

## Experiments and results

- [`results/`](results/) — recorded extraction, rewrite-rule, JIT-cost, and actor benchmark
  results. These are point-in-time evidence; heed corrections in each report.
- [`COMPILER_ANALYSIS.md`](COMPILER_ANALYSIS.md),
  [`COMPILER_OPPORTUNITIES.md`](COMPILER_OPPORTUNITIES.md),
  [`KERNEL_PARAM_LIMIT_INVESTIGATION.md`](KERNEL_PARAM_LIMIT_INVESTIGATION.md), and
  [`function-namespace-audit.md`](function-namespace-audit.md) — analyses and audits.
- [`designs/2026-07-23-jit-orthodoxy-survey.md`](designs/2026-07-23-jit-orthodoxy-survey.md),
  [`AUTODIFF_RENDERING.md`](AUTODIFF_RENDERING.md), and the papers and paper notes in this
  directory — research context, not implementation contracts.
- [`designs/BRAINSTORM_VARIANCE_EGRAPH.md`](designs/BRAINSTORM_VARIANCE_EGRAPH.md) and
  [`GNN_REWRITE_GUIDANCE_VISION.md`](GNN_REWRITE_GUIDANCE_VISION.md) — exploratory ideas.

## Historical or superseded

- [`EGRAPH_OPTIMIZATION_ARCHITECTURE.md`](EGRAPH_OPTIMIZATION_ARCHITECTURE.md) — obsolete
  critic/REINFORCE training spine. Its “current” wording predates and is superseded by the
  guided-saturation redesign; do not use it as the live architecture.
- [`SEARCH_PIPELINE_DESIGN.md`](SEARCH_PIPELINE_DESIGN.md) — unshipped MCTS/REINFORCE
  interface proposal, superseded by the guided-saturation redesign.
- [`NNUE_TRAINING_RECIPE.md`](NNUE_TRAINING_RECIPE.md),
  [`plans/2026-02-25-unified-training-design.md`](plans/2026-02-25-unified-training-design.md),
  and [`plans/2026-02-25-unified-training-plan.md`](plans/2026-02-25-unified-training-plan.md)
  — obsolete critic/REINFORCE training path.
- [`designs/nnue-training-pipeline.md`](designs/nnue-training-pipeline.md) — earlier curriculum
  and best-first proposal.
- [`NNUE_INTEGRATION_STATUS.md`](NNUE_INTEGRATION_STATUS.md) and
  [`EGRAPH_SEARCH_INTEGRATION.md`](EGRAPH_SEARCH_INTEGRATION.md) — point-in-time status
  audits; use the current architecture and guided-saturation documents above instead.
- [`designs/actor-scheduler-supervisor-migration.md`](designs/actor-scheduler-supervisor-migration.md)
  — explicitly superseded by the Mealy-transducer design.
- [`designs/2026-07-23-lower-realize-boundary.md`](designs/2026-07-23-lower-realize-boundary.md)
  — transitional `Lower`/`realize` boundary, superseded by the JIT-first `Kernel` course
  correction recorded in the kernel-unification plan.
- [`plans/2025-02-21-kernel-jit-feature-parity-design.md`](plans/2025-02-21-kernel-jit-feature-parity-design.md),
  [`plans/2025-02-21-kernel-jit-feature-parity.md`](plans/2025-02-21-kernel-jit-feature-parity.md),
  and [`superpowers/`](superpowers/) — completed or superseded implementation-era plans.
- [`FLAT_CONTEXT_TUPLE_PROTOTYPE.md`](FLAT_CONTEXT_TUPLE_PROTOTYPE.md),
  [`MESSAGE_CUJ_COVERAGE.md`](MESSAGE_CUJ_COVERAGE.md), and
  [`designs/compiler-architecture-2026.md`](designs/compiler-architecture-2026.md) — retained
  point-in-time proposals; verify their claims against current code before use.

## Bugs

### Active bugs

- [`bugs/2026-07-15-pty-fork-malloc-deadlock.md`](bugs/2026-07-15-pty-fork-malloc-deadlock.md)
  — diagnosed and explicitly not yet fixed.
- [`bugs/2026-07-20-test-quality-audit.md`](bugs/2026-07-20-test-quality-audit.md) — mixed
  audit report with unresolved test-quality findings; classified active until those gaps are
  dispositioned.

### Fixed bugs

- [`bugs/2026-07-21-openpty-not-thread-safe.md`](bugs/2026-07-21-openpty-not-thread-safe.md)
  — concurrent-test contract fix.
- [`bugs/2026-07-22-trig-chebyshev-coefficients-wrong.md`](bugs/2026-07-22-trig-chebyshev-coefficients-wrong.md)
  — trigonometric approximation and regression-test fixes.
