# One conditional, three lowerings

**Date:** 2026-09-08
**Status:** Draft. Nothing built.
**Author:** JP (direction), Claude (draft)
**Supersedes:** [2026-09-07-demand-is-a-dag-property.md](2026-09-07-demand-is-a-dag-property.md),
whose §1–§2 survive as the *analysis* this plan needs and whose §9–§10
remain the record of the glyph investigation. What that plan got wrong is
its framing: it treated demand as a codegen concern replacing per-select
guard machinery. It is a language concern, and the machinery it proposed to
replace is the third of three lowerings, not the whole subject.
**Refines:** §5 of [2026-09-06-lattice-is-the-index.md](2026-09-06-lattice-is-the-index.md)
("guards and index ranges are one thing"), which observed the identity and
did not say what to do about it.

**Decisions it records (JP, 2026-09-08):**

> *"Claude.md is wrong about select and you should fix it. Select contains
> an if."*

> *"I don't understand why union/rectangles can't just be built with two
> selects. I had a beautiful example of a circular kernel that was just a
> select. That's how I imagined dispatch working."*

---

## 1. The thesis

**`Union` is a missing compiler capability that leaked into the public
API.**

A rectangle is two selects. A circle is one select. Someone needed a
select's mask to become a region the loop could be split on, the compiler
could not derive one, so a parallel domain mechanism was built by hand and
handed to the caller. L3 did this for text, L4 did it again for the cell
grid's border, and the font loader did a third variant as `Support`. Three
hand-written spellings of one thing the compiler should derive.

`IndexRange` is the leak's carrier rather than the leak itself: it also
denotes a caller-selected collapse band, which is a legitimate public
concept the compiler has no business deriving. §5 separates the two.

The language has **one conditional**: `Select`. The compiler should have
**three lowerings** of it.

| what the compiler can prove about the mask | lowering | cost outside the region |
|---|---|---|
| it implies an index range | **domain split** — the region gets the select, the complement gets the false arm | the false arm alone, usually a fill |
| it is uniform over a row | **hoist and guard** the row prologue | one any-true test per row |
| nothing static | **blend** — what `Select` does today | full |

These are not three mechanisms. They are one denotation lowered against
three levels of static knowledge, and a single kernel routinely uses all
three at different nodes.

**Lowering 1 is a split, not a narrowed loop, and the distinction is a
correctness one.** `Select(m, a, b)` is defined at *every* index in the
extent. A derived range bounds where `m` can be nonzero, so outside it the
select's value is `b` — not nothing. Narrowing the loop and stopping there
would leave the complement holding whatever the destination was
initialized to, which is the value only when `b` happens to equal it. So
the lowering emits two disjoint programs: the select over the region, and
`b` specialized over the complement. The saving is that `b` alone is
usually a constant, so the complement is a fill rather than an evaluation
— and when it equals the destination's initialization it costs nothing at
all.

That is exactly what `IndexRange::paint_complement` already does, and why
this plan derives that machinery rather than deleting it: L3 and L4 wrote
the split by hand because the compiler could not derive the region, not
because the split was the wrong shape.

### Why this was invisible

Until #1208, both style docs said the *denotation* of `Select` was
branchless and its meaning carried one case. Under that reading, "which
values does this arm serve" is an artifact of codegen — so the natural
place to compute it is next to the emitter, per select. That is where it
was built, and it did not survive contact (superseded plan, §9).

`Select(m, a, b)` **means** `if m then a else b`. Two cases. An arm's
condition is a fact about the DAG, and the region a value is observed over
is a property to be *read*, not reconstructed.

## 2. The worked example that fixes the shape

```rust
let disc = kernel!(|cx: f32, cy: f32, r: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).lt(r * r)
});
disc.select(&fg, &bg)
```

The mask is a circle. It is not a rectangle and never will be. But it
**implies** one:

```
dx² ≤ dx² + dy² < r²   ⟹   |X − cx| < |r|   ⟹   X ∈ (cx − |r|, cx + |r|)
```

and symmetrically in Y. So the compiler restricts the select to the
bounding box, fills the corners with `bg`, and blends inside. One select,
lowering 1 composed with lowering 3.

**The `|r|` is not a typo, it is the shape of the obligation.** `r` is an
unconstrained `f32`: `r = -5` gives `r * r = 25`, a mask true inside radius
5, while `(cx − r, cx + r)` is reversed and excludes every index the mask
admits. Narrowing is unsound, so the derivation is over the *value range*
the uniform can take, never over the symbol as written. A derivation that
reads the source syntactically and forgets that `r` can be negative
produces a range that deletes pixels — which is why D1's gate is a
containment check and not an equality against a hand-written answer (§8).

That composition is the whole design. A mask does not have to *be* a range
to yield one; it only has to imply one. **Weakening a predicate to
something it implies is always sound**, which the superseded plan already
states as its first constraint. The converse is the whole risk: every
unsound step in this plan is a narrowing, and narrowing deletes pixels
silently.

## 3. The missing piece: mask ⟹ index range

This is the actual work, and it is smaller and better defined than
"replace the guard machinery."

Given a mask node, produce a rectangle in lattice indices that **contains
every index where the mask can be nonzero**. A superset is always
admissible; the full extent is the trivial answer and the correct one when
nothing can be proven.

Two tiers, and the split matters because of when the answer is available:

- **Compile-time.** The mask depends only on coordinates and constants.
  The range is baked into the emitted loop nest. `X < 40`, a glyph's
  support, the circle above.
- **Bind-time.** The mask depends on a uniform. The range cannot be
  compiled in, because `Manifold::compile` sees only the extent and the
  block is bound later. It is computed when the block is written.

L4 already hit this and named it: the cell grid's range "is a function of
its per-frame metric, not of the shape it was compiled at," which is why
`Scene` gained a variant. That finding is a constraint on this design, not
an inconvenience — and it is also why the bind-time tier must exist rather
than being deferred.

### How to derive it

A cheap exact path, then a general sound one:

1. **Symbolic, for axis-aligned literals.** `X < c`, `Y >= c`, and
   conjunctions and disjunctions of them read directly off the DAG. This
   covers `in_grid`, a glyph's support, and every range L3 and L4 write by
   hand today. Exact, no search.
2. **Interval evaluation, for everything else.** Evaluate the mask in
   interval arithmetic over a candidate rectangle. If the result interval
   is provably all-zero, the rectangle is excluded. Bisect to tighten.

   "Sound by construction" holds only if every transfer function encloses
   what the **target backend** actually computes, and this codebase
   deliberately makes that a nontrivial requirement. `Recip`/`Rsqrt` are
   estimates (~12–14 bits on x86, `FRECPE` plus one step on aarch64);
   `Min`/`Max`, `Gt`/`Ge`, and `Round` diverge between targets on NaN and
   on ties; `MulAdd` rounds once or twice depending on ISA level. An
   interval computed in ideal reals — or in the build host's `f32` — can
   prove a mask all-zero near a boundary where the emitted code makes it
   true, and because exclusion *narrows* the visited set that is a deleted
   pixel, not a looser box.

   So the transfer functions round outward and are target-aware, and any
   op whose target behavior is not bounded yields the full interval — the
   trivial answer, which costs a larger rectangle and nothing else. This is
   the same rule as the folder's: the build host is not the target, and a
   value that depends on which machine ran the analysis is simply wrong.

The circle falls to (2) and yields its exact bounding box. Precision beyond
that is a **compile-budget knob and must be documented as one** — never a
correctness bound, and never a tuned constant standing in for a derivation.
That discipline is not optional here: this codebase has now had three
constants (`EXTENT_SLOP`, `CLUSTER_ROUNDS_PER_SELECT`, `DISC_BAND_ULPS`)
whose stated derivations did not survive measurement.

A mask that reads a bound buffer does **not** categorically fall to
lowering 3. `(X < 40) & sample(buf, X, Y)` implies `X < 40` whatever the
buffer holds, and discarding that conjunct forces full-frame blending on
exactly the masks that combine geometry with an atlas — the glyph case.
The derivation weakens away the buffer-dependent operand and keeps every
coordinate-only consequence; a conjunction's range is the intersection of
its conjuncts' ranges, and an unbounded conjunct contributes the full
extent rather than poisoning the result. Only a mask with *no*
coordinate-only consequence falls to lowering 3, which is what happens
today.

## 4. Why demand stays an optimization, not a semantics

If arms were only *defined* on their side of the mask, widening a predicate
would be unsound and demand would be load-bearing for correctness. It is
not, and the reason is worth stating because it is what makes the whole
design safe:

**Gathers clamp.** `floor` then clamp to the declared extents, on both the
AVX2 and AVX-512 lowerings. An arm evaluated outside its region reads an
in-bounds value which the blend then discards. Arms are total.

The one residual is NaN, which can survive a `min`/`max` clamp on aarch64
(CLAUDE.md's divergence table: `FMIN`/`FMAX` propagate). That is why a
skipped region's parks hold **zero, never a mask** — the rule from the
superseded plan §2, which stands unchanged and becomes load-bearing here.

## 5. What this subsumes

- **`Union` leaves the public API**, and with it the *derived-region*
  meaning of `IndexRange`. They survive as the compiler's internal
  representation of what lowering 1 produces, not as something a caller
  writes.
- **`IndexRange` itself stays public, because it means two things.** Its
  other denotation is a **caller-selected band**: which samples of a frame
  a caller wants collapsed, in `CellGridPackedFrame::collapse_rows` and
  `CellGridFrame::collapse_channel_rows`, and one per worker stripe in
  `render_bands` (`pixelflow-graphics/src/render/{cell_grid,scene}.rs`,
  `pixelflow-core/src/lattice/cell_grid.rs`). That has nothing to do with
  `Select` — it is a work partition, chosen by whoever is calling, and no
  compiler derives it.

  Two meanings on one type is the thing CLAUDE.md says to pay for at the
  moment you notice it, so: **the derived region and the requested band are
  different types**, and D3 splits them rather than hiding one. The band
  keeps the public name it has today; the derived region is internal and
  never appears in a signature. That they are both rectangles is not a
  reason to spell them the same, and spelling them the same is what made it
  possible to write "`IndexRange` leaves the public API" and not notice it
  would break `collapse_rows`.
- **`text_union` should not exist.** `text()` returns a chain of selects
  over glyph supports; the compiler produces what L3 hand-built, including
  per-region code. "Region known statically → domain split" done properly
  *means* emitting, per region, only the values demanded there. That is
  per-region compilation; it is not a separate capability.
- **L4's border stops being hand-written, not deleted.** `grid_range` +
  `intersect` + `paint_complement` *is* lowering 1 applied to
  `in_grid.select(blended, default_bg)` — the select L4 spelled out by
  hand. The complement fill is the split's second half and stays; what goes
  is the caller having to write it.
- **`Support` stops being its own concept.** A glyph's support is the range
  implied by its own mask.

The measure of success is subtraction: four hand-written mechanisms become
one construct with three lowerings.

## 6. What survives from the superseded plan

Its §1–§2 are the analysis this design needs once regions exist:

- The **demand predicate** (DNF over mask literals, clause-capped, widened
  to `true` on overflow) is how you determine which values a region serves
  once you have regions. It is the machinery of lowering 2.
- The **producer-superset invariant** (`demand(u) ⊇ demand(v)` for every
  producer edge) and the sort that follows from it.
- **A guard is a region of equal demand**, with `MISPREDICT_PENALTY_CYCLES`
  unchanged as the one profitability bound.
- **Scope projection** — dropping an X-dependent literal in the row
  prologue and guarding on the Y-only remainder.
- **Parks hold zero**, per §4 above.

**The select short-circuit is not going anywhere.** It is lowering 2, it
keeps its bound and its park rule, and demand strictly widens its coverage:
a value read by the true arm of `S₁` and the false arm of `S₂` has demand
`m₁ ∨ ¬m₂` — refused as "not exclusive" today, guardable under this design.

## 7. What is dropped

**The `O`@32 telemetry gate ("no worse than 2,801 guarded entries") is
withdrawn.** That number was measured with C1's guard work applied, and
that work was reverted in #1187 — the branch shipped test-only. It names a
state that never merged, and it measures one lowering of one case, which is
the framing this plan replaces.

The replacement gate is compositional and needs no baseline:

- `text()` as a select chain produces what `text_union` produces today,
  and `text_union` can be deleted.
- L4's border falls out of `in_grid.select(..)` without the caller writing
  `grid_range`/`paint_complement` — the complement fill is still emitted,
  it is just derived.
- A glyph's `Support` is derived rather than declared.
- The circle of §2 skips its corners.

Each is a subtraction with an observable before and after. Per-row and
per-pixel numbers are then consequences to measure, not targets to clear.

## 8. Stages

**D1 — mask ⟹ index range, compile-time tier, symbolic only.** §3 path (1).
No lowering yet: derive the range and check it.

Gate, and the order matters. The **soundness** check is containment,
checked directly: collapse the mask over the full extent and assert every
nonzero index falls inside the derived range. That is the property the
lowering actually depends on, it fails for the right reason, and it does
not care whether any hand-written range is correct. The **usefulness**
check is the comparison against `grid_range`, a glyph's `Support`, and the
column cuts `text_cells` computes — a derived range that contains the mask
but equals the full extent is sound and worthless, and this is what
catches that.

Checking only the agreement would make D1 a change-detector against three
answers whose correctness is assumed; #1187 is the standing reminder that a
hand-written answer in this area can be wrong. Production behaviour is
unchanged either way.

**D2 — lowering 1, compile-time.** Emit the split: the select over the
derived range, the false arm over the complement. Gate: `text()` as a
select chain matches `text_union()` bit-for-bit and does not regress
`pixelflow_text_sizes`; `glyph_atlas_golden` unmoved. A differential check
over the whole extent — split versus unsplit — is what proves the
complement half, since a wrong complement is invisible to any test that
only looks inside the region.

**D3 — bind-time tier**, and the split of `IndexRange` into a derived
region and a requested band. Ranges from uniform-dependent masks, computed
when the block is written. Gate: L4's cell grid stops *calling*
`grid_range` and `paint_complement` — both are derived — with its three
identity hashes unchanged, and `collapse_rows`/`collapse_channel_rows`
keep taking a caller-chosen band.

**D4 — interval evaluation.** §3 path (2). Gate: the circle's bounding box
is exact for both signs of `r`; precision is a documented budget knob; a
mask whose only coordinate-free operand reads a buffer still yields its
coordinate-only range; and every op without a target-aware outward bound
yields the full interval rather than a tighter wrong one.

**D5 — lowering 2 on the general predicate.** The superseded plan's §1–§2,
now as the third case rather than the whole subject.

`Union` and the derived-region type leave the public API at D3, once
nothing writes one by hand. `IndexRange`-as-a-band stays; D3 is where the
two meanings become two types.

## 9. Constraints

- **Widening is always sound; narrowing never is.** Any derived range may
  be replaced by a larger one, up to the full extent. Every unsoundness
  available to this design is a narrowing, and a narrowing deletes pixels
  without failing.
- **A restricted region never restricts the output domain.** The select is
  defined at every index; lowering 1 splits the domain, it does not shrink
  it.
- **The analysis runs on the build host; the code runs on the target.**
  Interval bounds round outward and are target-aware, and an op whose
  target behavior is not bounded yields the full interval. Same rule as
  `ConstantFold`'s.
- **A derived range is over the values a uniform can take**, never over the
  symbol as written.
- **No tuned constant stands in for a bound.** Interval precision is a
  compile-budget knob and says so. `MISPREDICT_PENALTY_CYCLES` keeps its
  existing derivation.
- **`Select`'s value semantics do not change.** Which lowering is chosen
  changes what is computed, never what is selected.
- **A skipped region leaves zero in its parks.** Never a mask.
- **The frame prologue is never guarded.** It runs once per call.

## 10. Non-goals

- Deriving non-rectangular regions. A triangular or circular *domain* is
  GPU-shaped thinking: the triangle in Loop-Blinn exists because GPUs
  rasterize triangles, and this is a pull renderer with nothing to
  tessellate for. A circle yields a bounding box and blends inside it.
- Profile-guided coherence. The dynamic half of "is this guard worth it"
  is the schedule-cost residual and is not this plan.
- Changing `Select`'s NaN or bit-pattern semantics.
- The glyph waist defect. Still open, still pinned by the oracle, unrelated
  to this design.
