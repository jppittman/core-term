# A glyph is a circle that cannot read its own sign

**Date:** 2026-09-09
**Status:** Denotation. No code. The implementation in
`pixelflow-graphics/src/fonts/loop_blinn.rs` deviates from this in four
places, each one a compiler ask rather than a fact about glyphs.
**Author:** JP (direction), Claude (draft)
**Follows:** [2026-09-08-loop-blinn-glyph.md](2026-09-08-loop-blinn-glyph.md),
which is what was built. This is what it should have been.

---

## The whole thing

A circle is easy because it arrives as a polynomial whose **sign** is the
answer and whose **magnitude** is the distance:

```rust
let sdf = r.sub(&(x.mul(&x).add(&y.mul(&y))).sqrt());
let coverage = half.add(&sdf).clamp(&zero, &one);
```

That is an antialiased circle, and it is the whole program. A glyph is the
same program. It differs in exactly one way, and it is worth saying the
difference precisely, because everything else in this document follows from
it:

> A glyph's boundary is many pieces, so no single polynomial has both
> properties. Its **magnitude** is still available — the distance to the
> nearest piece, a minimum. Its **sign** is not, because the nearest piece
> does not know which side of the *outline* you are on. The winding number
> is the repair, and it is the only reason it appears.

So:

```
sign      = a sum   over every piece      →  ℤ, exact
magnitude = a min   over boundary pieces  →  pixels, soft
coverage  = clamp(½ + sign · magnitude, 0, 1)
```

Two folds and a clamp. In the language:

```rust
pub fn glyph(outline: &Outline) -> Kernel {
    let pieces = outline.pieces();

    let winding = Kernel::sum(&pieces.map(Piece::crossing));
    let sign = winding.abs().ge(&one).select(&one, &minus_one);

    let distance = Kernel::min(&pieces.map(|p| {
        p.bounds(&winding).select(&p.distance(), &far)
    }));

    half.add(&sign.mul(&distance)).clamp(&zero, &one)
}
```

Nine lines. Everything below is about the distance between those nine lines
and the eight hundred that are actually checked in.

## The three pieces

A **piece** is a line, or a quadratic. Each contributes to both folds.

```
crossing(line a→b)  =  (a.y ≤ Y < b.y) ∧ (x_at(Y) > X)   →  ±1 else 0
crossing(quad)      =  crossing(chord)  +  sliver
sliver              =  (v ≤ u) ∧ (v ≥ u²)                →  ±1 else 0
```

`(u, v)` is an affine contramap of `(X, Y)` — six numbers per curve, computed
on the host — chosen so the arc lands on the one parabola `v = u²`. That is
Loop and Blinn's construction and it does exactly one job: it makes a curved
edge as cheap to ask about as a straight one, by turning *where is the curve*
into *what is the sign of a formula*.

It needs no fence beyond `v ≤ u`, the chord's own half-plane. `u² ≤ v ≤ u`
holds nowhere but `u ∈ [0, 1]`, so the crescent is bounded by its own
inequality. A GPU spends a rasterized triangle on this. Two comparisons do
it here.

`bounds(piece)` — whether a drawn edge is a *boundary* rather than an edge
buried under other ink — is a question the sum already answers. With `w₋` the
winding less this piece's own term, the two sides of the piece are `w₋` and
`w₋ + dir`. They differ by one, so they are never both zero, and the piece
separates ink from no-ink exactly when one of them is.

That is the entire denotation. There is no cell, no union, no index range, no
bounding box, no reference point, and no host-side pruning in it.

## Four places the code says something else

Each of these is the host proving a theorem the compiler could prove, and
then encoding the proof in a data structure. None of them is about glyphs.

### 1. `min` is not a fold

`Kernel::sum(&[Kernel])` exists and is variadic. `Kernel::min` is binary, so
the code ships `min_of`, a hand-rolled balanced tree carrying this comment:

> Every `Kernel` combinator copies its arena, so `acc = acc.min(&d)` in a
> loop copies a growing arena `n` times: quadratic.

That is a fact about `ExprArena` living in a font file, and it is CLAUDE.md's
own warning about itself — a convention written in a comment. Measured cost
of not knowing it: 4.7 million arena nodes and seven seconds to *construct*
the kernel for a 26-character string.

**Ask.** Either a variadic form for every associative op, or — subtractively,
and better — an arena that shares instead of copying, which retires
`Kernel::sum`'s special case as well.

### 2. Naming a value twice should not cost twice

`bounds(piece)` names `winding` once per piece. Naming it N times copies it N
times, so the code does not ask: it precomputes, on the host, which pieces
*could* be interior — a winding test at each piece's midpoint against the
other contours — and only those pay.

The answer happens to be tiny (on this font, two glyphs of 189: `Ç` and `ç`,
where the cedilla runs under the C), so the workaround is cheap and the
mechanism is invisible. That is what makes it worth naming: it is not a
performance fix, it is the same fact computed twice in two languages, and the
host's copy will drift.

**Ask.** The same one as §1. `winding` is one node; each query is two
subtractions and a compare.

### 3. Demand is a property of the DAG, not of the domain

`cells` cuts the frame into rectangles and gives each an `IndexRange` in a
`Union` carrying only the pieces that reach it. Per-contour bounding boxes
drop whole characters. Both are *exact* — nothing is approximated — and both
are the host proving one of these:

| the fact | what proves it |
|---|---|
| a crossing term is zero outside its chord's Y band | interval analysis on one node |
| a sliver term is zero outside its arc's box | interval analysis on one node |
| a *whole contour's* contribution to the sum is zero outside its box | the summands form a closed cycle, so the crossings cancel |

The first two are ordinary range propagation. The third is the interesting
one, and the reason this is worth writing down: it is not a property of any
node, it is a property of a **sum whose terms trace a closed loop**. A ray
crosses a closed contour an even number of times and the signs cancel. No
per-node analysis can see that; something has to know the summands are a
cycle.

This is [2026-09-07-demand-is-a-dag-property.md](2026-09-07-demand-is-a-dag-property.md)
and a glyph is its worked example — including the hard case. If it lands,
`cells` is deleted, one kernel is both the elegant form and the fast one, and
`Union`'s explicit ranges stop being a thing a *caller* computes.

Note what does **not** motivate this, because the earlier plan said it did:
the sliver needs no domain-side fence. `{v ≥ u²} ∩ {v ≤ u}` is bounded on its
own. The extent is a performance device throughout, never a correctness one.

### 4. Hoist to the collapse, not just the row

The one deviation that is not yet in the code, and the one that decides
whether a glyph can be *data*.

Every per-piece number is currently a `Const`, folded into the fragment. Make
them `Uniform` and the arena becomes structurally identical across cells with
the same piece pattern — and `jit_cache` keys on structure, with a uniform's
*position* in the key and never its value. One compiled program serves many
cells; a glyph becomes a `UniformBlock` write.

The objection is `Dwrt`. Every distance is divided by the gradient magnitude
of its own edge function, so the ramp is one *screen* pixel through any
enclosing warp. With constant coefficients that whole normalisation folds to
a compile-time multiply. With uniforms it does not fold, and it looks like a
`rsqrt` per linear form per pixel.

It is not. **A uniform is constant over a collapse.** `‖∇f‖` is invariant in
X and Y both — not row-invariant, whole-lattice-invariant — so it belongs in
a prologue that runs once per collapse, computed from the block. The cost is
a load per use, which is what a uniform costs anyway.

**Ask.** A hoisting level above the row. The scanline's failure was LICM
hoisting *too eagerly* to the row prologue — Y-only work run for every
segment on every row. This is the same mechanism asked for one level up,
where it is unambiguously right: the value cannot change within a collapse,
because nothing in a collapse can write it.

## What this buys, stated so it can be checked

If all four land, `loop_blinn.rs` is the nine lines at the top plus the piece
constructors, and:

- `min_of` deletes (§1)
- `may_be_interior` and `chord_winding` delete (§2)
- `cells`, `contour_bounds`, `meets_triangle`, and the `Union` plumbing in
  `text.rs` delete (§3)
- `TEXT_CELL` — a constant currently chosen by argument rather than
  measurement — stops existing to be chosen (§3)
- a font is one program and a glyph is a block write (§4)

That is roughly 800 lines to roughly 150, with no behaviour change, and every
deleted line is a proof the compiler should have been doing.

## What is genuinely ours and stays

Not everything ugly is the compiler's fault, and the difference matters.

- **The distance to a quadratic is a cubic root.** It is not computed. The
  code takes the larger of a capsule bound and the implicit's own
  `|f|/‖∇f‖`, and splits a curve by de Casteljau until the bound is tight.
  That is a numerical trade, deliberately made, and it is *ours*.
- **The winding is never approximated.** Splitting, dropping, pruning: none
  of them may change an integer. Only the ramp is a tunable. This is the
  separation that made the `'8'` waist defect unexpressible, and it is a
  design commitment, not an artifact.
- **Geometry belongs on the host.** Every affine map is applied to control
  points before a kernel exists. Quadratics are closed under affine maps, so
  nothing is lost, and a kernel's constants are its own frame's numbers.
  This is not something a smarter compiler should take back.
