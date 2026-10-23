# Reduce-site checklist at base (3bfdaded): what was read

Run: `python3 -I .wip/a3/gen/checklist.py`, exit 0.

```
grep: 155 lines, 151 sites (deduped by (file, line))
already-correct  39   changed-by-A3  19   left  13   tests  60   n/a  5   A4  15   A5  0
classified 151      UNCLASSIFIED 0      STALE 0
```

classes.json has 142 rows; five keys carry a count above 1 (`node_cost` in `extract.rs`,
`subtree_eq`, `collapse_shape`, and two lattice tests).

## Agreement with spec section 5

- All 151 table rows joined to a grep hit at the stated `file:line`; no table row without a hit,
  no hit without a row. The class counts equal the spec's (39, 19, 13, 60, 5, 15).
- Every non-test site (91) was read in context. No class was changed: none disagrees.
- Mechanical check of the `tests` class: all 60 rows lie in a `#[cfg(test)]` module or under
  `tests/` (`pict_rewrite_tests.rs` is `#[cfg(test)] mod pict_rewrite_tests` in `math/mod.rs`).
  The compiler's `lower.rs` hits are all in its test module, so A5 is empty, as the spec says.
- Classified by purpose, not by location (agreed with the spec): `cell_grid.rs` `dump_arena`
  sits in a `#[cfg(test)]` module but is a header-bits serializer (A4);
  `production_glyph_arena_dump.rs` `dump_arena` is under `tests/` and is the same (A4);
  `glyph_saturation_cost.rs` `has_fold` and `glyph_phase_split.rs` `trip_counts` are examples
  (already-correct, left).

## Differences that are not classifications

- The spec's "In" column names the enclosing item loosely (`definition`, `Fold` doc, `ExprNode`,
  `Shape<`, `insert (Visit)`, `PlanBuilder::reduce`, `EmptyFold::apply`, `ENode::variance`). The
  script's key is the bare enclosing fn, else `enum X` / `impl ...`, else `<top>`; so
  `fold.rs:343` and `kind.rs:204` (doc comments above top-level items) are `<top>`, the enum
  definitions are `enum ExprNode`, `enum Shape`, `enum HeadNode`, `enum ENode`,
  `enum ScheduledOp`, and the methods are `reduce`, `apply`, `variance`.
- Base behaviour behind the changed-by-A3 reasons, read at base: `free_index` and both arms of
  `compute_arena_variance` and `ENode::variance` use the body alone (value bounds ignored);
  `lowest_free_binder` walks `children()` (so it reaches the value bounds) but marks fold
  binders only, not binder `Var` reads; `extract.rs` `price` already passes every operand's
  settled cost to `below`, which uses the first and ignores the rest; `known(lo)`/`known(hi)` in
  `lower.rs` and `to_bits` refuse a value bound by name.
- `training/corpus.rs` `write_node` still says "before A3's format version" at base; the spec's
  reword to A4 is the B1 change.

## Rustdoc base

See `rustdoc-base.md`: codegen with `-D warnings` passes at base; pixelflow-ir 38 warnings (35
keys), pixelflow-search 28 warnings (27 keys).
