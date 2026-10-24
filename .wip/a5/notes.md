# A5 working notes

A5 of docs/language/2026-10-20-a-bound-is-a-value.md: `<const N: usize>` is deleted, and with it
§3.4's structural path (`Count::Structural`, `StructuralRange`, `instantiated_range`, `FoldRange`,
`Holes`, `OPEN_RANGE`, `HOLE_STRIDE`, `ExpansionFold::Hole`, the lowering's `Param`-as-hole).
No production user. Before A6 because a structural bound lowered as a `Param` leaf is
`Variance::ALL` and L3 refuses it.

Base: main 7e9531e (A4 merged, #1448). Gates in ../gates (README.md).

This directory is dropped (`git rm -r .wip`) before the PR is marked ready.

## State
- [ ] design: harness re-verify at 7e9531e, measure base, spec, critiques
- [ ] rulings
- [ ] implement
- [ ] review rounds
- [ ] close: drop .wip, ready, merge when green
