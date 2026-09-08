# Executing the glyph-as-a-fold rewrite

**Date:** 2026-09-09
**Status:** Plan. Staged, each stage independently landable and gated by the
same test.
**Author:** JP (direction), Claude (draft)
**Executes:**
[2026-09-09-a-glyph-is-a-circle.md](2026-09-09-a-glyph-is-a-circle.md)
**Replaces:** the per-piece arena construction in
`pixelflow-graphics/src/fonts/loop_blinn.rs`

---

## The gate, first

`pixelflow-graphics/tests/loop_blinn_winding.rs` is an independent `f64`
winding oracle sharing no code or constants with the kernel. It is green
today at 9/9 and it is the gate for every stage below. **No stage lands
without it.**

That test is why this rewrite is safe to do at all: the denotation is
unchanged, so a rewrite that passes the oracle computed the same function.
Every other pin (`freetype_oracle`, `text_union_identity`,
`kernel_glyph_optimize`) is a secondary signal and any change to one must be
argued, not renumbered.

## Two unknowns that gate the design

Both are being answered before S1 starts, because either one could change the
shape.

**U1 — does a buffer read compose with a reduce binder?** The denotation
needs `table.at(&constant(col), &binder)`. `.at` substitutes `Var(0)`/`Var(1)`;
reduce binders live in a reserved slot range (`REDUCE_BINDER_BASE`). If `.at`
cannot reach a binder, or the emitter cannot lower a `Buffer` read inside a
`Reduce` body, the table has to arrive some other way and S1's shape changes.

**U2 — is the read a single tap?** `fonts/cache.rs` reads a bound buffer
through a bilinear sampler: four taps and a blend. For a table at exact
integer indices that is four times the loads for a lookup. If no direct
addressing mode exists, that cost is real and has to be priced into S1 rather
than discovered in S3.

## S0 — the table

**Deliverable:** a host-side row layout and a `Kernel` that reads it.

Row `i` is piece `i`. Columns are its coefficients. One row layout serves both
a line and a quadratic — a line is a quadratic whose sliver term is the
monoid identity — so there is no per-kind branch in the body, only data.

Provisional layout (widths to be confirmed once U1/U2 land):

| col | meaning | line | quad |
|---|---|---|---|
| 0–3 | chord `a.x, a.y, b.x, b.y` | the segment | the chord |
| 4 | `dx/dy` | crossing slope | crossing slope |
| 5 | `direction` (±1) | crossing sign | crossing sign |
| 6–11 | `(u, v)` contramap, six numbers | — | the sliver |
| 12 | sliver `sign` (±1) | **0** | ±1 |
| 13 | chord-bound deviation | 0 | curve's stray |

A line's sliver term is `0 · mask`, which is the sum's identity, so it costs
arithmetic but no correctness. Whether that is cheaper than a second body
under a second reduce is a **measurement**, deferred to S3.

**Gate:** a unit test that bakes a two-row table and reads both rows back
exactly, under a `sum_over`.

## S1 — the body

**Deliverable:** `glyph()` as one `Kernel::over` body, per-glyph extent.

Per-glyph extent (`n` = that glyph's piece count), not font-wide. That gets
every deletion below with no padding waste, and leaves the padding question
to S3 where it can be measured instead of assumed.

Port, unchanged in meaning:

- `crossing_term` → reads cols 0–5 at the binder
- `sliver_term` → reads cols 6–12; `{v ≥ u²} ∩ {v ≤ u}`, no triangle
- `distance_to` → capsule bound vs implicit, cols 0–4 and 13
- `bounds` → `w₋` and `w₋ + dir` against zero, naming `winding` once

**Deletes at this stage:**

- `min_of` — `Kernel::min_over` is the fold
- `may_be_interior`, `chord_winding` — one body names `winding` once; the N
  unrolled copies are structurally identical and pixel-only, so CSE collapses
  them
- the f64/kernel-constant duplication in `Piece`

**Gate:** `loop_blinn_winding` 9/9. Plus `freetype_oracle`'s pins unchanged
(`KNOWN_ORPHAN_TEXELS = 0`, `TEXELS_WE_MISS_FAST = 3`).

**Expected regression, and it is not a defect:** the optimizer can no longer
fold through a gathered coefficient, so any test asserting the `Dwrt`
normalisation collapses to a compile-time constant will fail. That assertion
is *true of constants and false of data*, and the fix is ask B (hoist
binder-only work out of the pixel loops), not a revert. Whichever test holds
it gets restated to assert the hoist, or marked with the ask.

## S2 — delete the domain-side pruning

**Deliverable:** `cells`, `contour_bounds`, and the `Union` plumbing in
`text.rs` gone; one kernel per glyph.

Blocked on **ask A** — demand as a DAG property
([2026-09-07](2026-09-07-demand-is-a-dag-property.md)). Until a compiler can
prove "this binder range contributes nothing here", deleting `cells` trades
correctness-preserving speed for nothing.

So S2 is **not** scheduled with S1. What S1 must do is leave it possible:
the pruning must stay a wrapper around the body, never a fact the body
depends on.

## S3 — one program for the font

**Deliverable:** font-wide extent, table padded with monoid identities (`0`
for the sum, `+∞` for the min), so the compiled program is the same for every
glyph and a glyph is a `UniformBlock`-style table write.

This is a **trade, not a win**, and it must be measured: fewer compiles
against evaluation of rows that contribute nothing. The padding waste is
exactly what ask A removes, so S3 is worth much more after §A than before it.

Measure: distinct `jit_cache::entry_count()` deltas over a font bake, compile
wall clock, and collapse wall clock, both ways.

## Order and parallelism

```
U1, U2  ──▶  S0  ──▶  S1  ──▶  gate green  ──▶  land
                                    │
                                    ├──▶ S3 (measure, then decide)
                                    └──▶ S2 (blocked on ask A)
```

S0 and S1 are one thread of work — S1 cannot start until the table reads.
S2 and S3 are independent of each other and both wait on S1.

## What must not change

Carried forward from the denotation, and each one already cost something to
learn:

- **The winding is never approximated.** Splitting, padding, pruning: none
  may change an integer. Only the ramp is a tunable.
- **A distance is not negative.** The chord bound goes below zero on a
  curve's chord; unclamped it makes `½ − d` exceed one.
- **`|f|/‖∇f‖` alone is not a distance.** It underestimates at high
  curvature — a texel a full pixel outside `O`@7px read as an edge texel.
  Keep the capsule bound and the de Casteljau split that makes it tight.
- **Geometry stays on the host.** Affine maps are applied to control points
  before a kernel exists.
- **Pruning is exact or it is not pruning.** A piece dropped must be provably
  zero there, not merely small.
