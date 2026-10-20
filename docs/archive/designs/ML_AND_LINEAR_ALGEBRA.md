> **Archived 2026-10-15 — status `Historical`.** Superseded by `docs/language/2026-07-24-totality-and-the-cost-model.md`. Twelve-line April stub on functional ML as pure linear maps; self-declared Historical, owns no open BACKLOG item, and its idea lives in the totality design doc. Kept for rationale; not authoritative about the tree.

> **Historical (Apr 2026), and a stub — five lines, never developed.** The idea it
> gestures at is stated properly in
> [`../../language/2026-07-24-totality-and-the-cost-model.md`](../../language/2026-07-24-totality-and-the-cost-model.md)
> ("Think denotationally, not in vectors"), which is the design of record. Nothing
> depends on this page.

# Functional Machine Learning and Linear Algebra

## Denotational Design for Neural Networks
Instead of graphs and stateful execution, we represent mathematical models as pure functions:

- A Matrix is a linear map `A -> B`.
- A Neural Network layer is a generic function `(A -> B) + Bias -> NonLinearity`.