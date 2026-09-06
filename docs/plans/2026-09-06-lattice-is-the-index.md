# A lattice is the index; everything else is a contramap or a uniform

**Date:** 2026-09-06
**Status:** Draft
**Author:** JP (direction), Claude (draft)
**Refines:** [2026-09-06-kernel-with-a-lattice.md](2026-09-06-kernel-with-a-lattice.md),
which defines a lattice as "extents and origin". This plan removes the
origin.
**Depends on:** [2026-09-06-uniform-slot-identity.md](2026-09-06-uniform-slot-identity.md)
for the leaf that two of the origin's jobs become.
**Decision it records (JP, 2026-09-06):** *"I don't think we should have
allowed lattices themselves to live in different coordinate spaces and
forced all of the translation to be contramaps plus uniforms."*

---

## The shape

CLAUDE.md states the law that makes a buffer *be* a manifold rather than
merely back one:

```
index(collapse(f)) = f
```

`Lattice` today is `{ extent: [u32; 4], origin: [f32; 4] }`. With an origin
`o`, `collapse(f)(i) = f(o + i)`, so `index ∘ collapse = f ∘ shift`, and the
law holds only if `index` also knows `o`. At that point the lattice is no
longer the representing object. It is an index carrying a coordinate frame,
and the frame is exactly the thing that should have been precomposed onto
`f`.

The correction is one removal:

- A **lattice** is an extent. Its points are the indices `[0, wₓ) × [0, w_y)`
  and nothing else. It is the representable functor's index, and the law
  holds without a side condition.
- A **contramap** is where a coordinate frame lives. Pixel-center sampling
  is `f ∘ (+½)`. A placed glyph is `f ∘ (−pos)`. A DPI scale is `f ∘ (·s)`.
  The kernel already has `Kernel::at`; the lattice stops competing with it.
- A **uniform** is where a per-call scalar lives. Time, cursor phase, light
  direction. The lattice's Z and W "axes" have extent 1 in every production
  call in the tree: an axis that never varies is not an axis.
- A **sub-lattice** is an index range: rows `y₀ .. y₀ + n` of a plane. This
  is the one thing that legitimately lives on the domain side, and
  `PlaneRegion::rows(width, y0: usize, rows)` already spells it as an index
  rather than a coordinate. The plan makes that the only spelling.

```
kernel  ──compile(extent)──▶  manifold  ──collapse(index range, uniforms)──▶  buffer
                                                       │
                                        coordinates enter here only through
                                        the kernel's own contramaps
```

**What the machine code already does.** `Lattice::bake` compiles on
`LatticeShape::new(self.extent)` alone and hands `origin` to the emitted
kernel at runtime as the `Point4` base coordinate. `PlaneRegion::on_slice(z, w)`
does the same for time. So the origin is *operationally* a uniform today.
What is wrong is the type: one `[f32; 4]` field conflates a coordinate frame,
two uniforms, and an index restriction, and breaks the law on paper while
the emitted code quietly respects it. This plan changes types and call
sites; it does not change what the JIT emits for a frame.

## Why now: the measurements that led here

All on this host (AVX-512, `target-cpu=native`), 2026-09-06, on
`0ad8a1c0` unless noted.

**The psychedelic shader, 1920×1080, whole frame.** `Lattice::bake` is 2.86×
faster than `kernel!` (1.84 vs 5.26 ns/px). Two of its three sines are
functions of `W` alone. LLVM, handed the fully inlined loop, does not hoist
them (hand-folding them to constants took the templates tier from 5.52 to
2.57 ns/px). The JIT hoists them because the lattice tells it `W` is
invariant. That is the whole win, and it is a win about *knowing which
values do not vary*, which is what a uniform is. Today that knowledge is
smuggled in as `origin[3]` on an axis of extent 1.

**Text as `Kernel::sum` is O(n) per pixel.** `text()` sums n translated
glyph kernels. A sum has no mask, so nothing in it can be guarded, and every
pixel evaluates every glyph:

| chars | sum, ns/px | balanced select tree, ns/px |
|---|---|---|
| 5 | 87.8 | 62.7 |
| 10 | 171.1 | 59.9 |
| 26 | 297.7 | 104.4 |

The tree is O(log n) and bit-identical. But it is the *range* encoding of an
extent: every pixel still visits the tree. The disjointness of the glyphs,
the one fact that makes text cheap, is nowhere in the program. The sum is
denotationally correct and structurally blind, which is exactly the failure
CLAUDE.md warns about: the invariant lived in the author's head. (The
`text_sizes/50` benchmark case also only renders 26 characters, because
`take(50)` on a 26-char string is 26; its curve is not the flattening it
appears to be.)

**A curved glyph is all row prologue.** `O` at 32 px: 33.1 ns/px on a 40-px
lattice, 2.0 ns/px on a 640-px one, a 16.5× ratio that is the batch-count
ratio. Solving the two widths: per-pixel body cost is ≈0, per-row cost is
≈1.3 µs. Every segment's root-finding is Y-only, so LICM hoists it to the
row prologue, and there it runs unconditionally for all 24 segments on
every row, because the hoist moved it out from under the `in_y` select that
would have skipped it. Guard telemetry on the row prologue reads
`selects=60 guarded=0`; the emitter's comment says "the prologues run once,
so a branch buys nothing there", which is true of the frame prologue and
false of the row prologue. That is a compiler finding, recorded in §5; it
is adjacent to this plan because a domain-side extent is what would make it
moot for a grid and not for a glyph.

**The terminal already does it right.** See §2.

## 1. Where the origin's four jobs go

| job today | example | becomes | mechanism |
|---|---|---|---|
| time / frame parameter | `Lattice::frame(w, h, z)`, `PlaneRegion::on_slice(z, w)`, shaders reading `W` | a **uniform** | uniform-slot-identity plan |
| pixel-center convention | font bench `origin: [0.5, 0.5, …]` | a **contramap** | `kernel.at(X + ½, Y + ½, …)` at the call site that owns the convention |
| point evaluation | `Lattice::point(x, y, z, w)` | a **contramap** over a 1×1 extent | `kernel.at(x, y, z, w)` then `collapse` at `[1, 1]`; or `eval_at`, which already takes a `Point4` |
| row band | `PlaneRegion::rows(width, y0, rows)` | an **index range** | unchanged in meaning; the type stops offering an `f32` alternative |

Z and W leave `Lattice` entirely. A kernel that reads `Z` or `W` today reads
a uniform tomorrow, and the variance analysis that already classifies
"invariant in X and Y" classifies it the same way with nothing to relearn.

## 2. The terminal is the worked example, and the BSP is the counterexample

`CellGrid` (`pixelflow-core/src/lattice/cell_grid.rs`) renders the terminal
as one program over the device-pixel frame at origin zero:

- `col = floor(x / cell_w)`, `lx = x − col·cell_w`: a contramap from pixel
  to (cell, offset), written as kernel arithmetic.
- `cells.at(col·10 + field, row)`: a contramap into the per-cell buffer.
  Which glyph is in which cell is **data**. A keystroke writes ten floats.
- `atlas.at(u0 + lx·density, v0 + ly·density)`: a contramap into atlas
  texel space through the bilinear sampler.

The retired `spatial_bsp` (deleted in `0ad8a1c0`) put which-glyph-where in
tree *structure*, with boxed leaves and per-batch descent:

| | `spatial_bsp` | `CellGrid` |
|---|---|---|
| programs | one per leaf, boxed | one, fused |
| which-glyph-where | tree structure | buffer data |
| routing | tree descent per batch, early-exit when lanes agree | `floor(x / cell_w)` |
| change one char | rebuild the tree | write ten floats |
| cost per pixel | O(log n) descent + leaf | O(1): two floors, four gathers, four taps |
| requires a grid | no | yes |

The BSP's generality is what killed fusion: arbitrary bounds with arbitrary
leaves means `dyn` leaves, and nothing optimizes across them. The grid
exploits the one fact the BSP refused to assume, that routing is arithmetic
rather than search. For a terminal, data beats structure. `CellGrid` is this
plan, written before the plan, by someone who understood the law.

Two places it still carries the old mistake at a smaller scale:

- **Geometry is baked as constants.** `cell_w`, `cols`, `density`, `frame_w`
  are `Kernel::constant`, so a resize recompiles. The module's own doc
  apologizes: the runtime renders "without any DPI contramap", and the scene
  "bakes its display scale in here". DPI is a contramap; cell geometry is a
  block of uniforms. With both, a resize is a parameter write. (Loop bounds
  stay static, per the uniforms plan; `frame_w × frame_h` is the extent,
  not a uniform.)
- **`in_grid.select(blended, default_bg)`.** The frame's extent, encoded in
  the range as a mask. The frame *is* the lattice; the border is the
  region outside the grid's sub-lattice, and it should be its own
  collapse over its own index range, not a per-pixel blend.

## 3. Extents on the domain

An extent is a contramap along an *inclusion* `ι : L' ↪ L`. That map is
partial on the frame, and a partial function in a total language has two
encodings:

- **Totalize it into the range.** `select(mask, value, 0)`. Every pixel
  evaluates; the mask says whether the answer counts. The guard
  (`cluster_select_arms`) recovers the domain encoding after the fact when
  the mask is coherent and the arm is expensive enough. This is what the
  font code does, and what the select-tree `text()` would do.
- **Restrict the index.** Don't ask outside `L'`. In a pull renderer the
  lattice does the asking, so this is a contramap on the *index* side: an
  index range. No mask, no guard, no profitability bound, no coherence
  question. The loop does not go there.

These agree on values where both are defined and differ on who visits what.
The index range already exists (`PlaneRegion`). What does not exist is a
lattice that is a **union of ranges**: `bake(⊔ Lᵢ, K)` distributing to one
loop nest per summand, and `bake(⊔ Lᵢ, Kᵢ)` with a kernel per summand, which
is a draw list. `CellGrid` is the uniform special case, where the union is a
grid and the kernel is shared. Text with a proportional font, sprites, and
any scene of placed things are the general case, and today they have no
lattice to pull from except the whole frame, which is why the only place
left to put the extent is the range.

**Overlap is what the select is for.** A union of *disjoint* ranges needs no
select. Pieces that genuinely overlap, where a pixel really must choose,
are where a select tree is the right encoding, and where the guard
machinery earns its keep. The two are not competing designs; they are the
disjoint and overlapping cases of one denotation, and a scene should be
able to say which it is.

**"One program" is preserved.** A sum of ranges is one program whose loop
nest is shaped by the summands: an outer loop over placed regions, an inner
loop over each region's pixels. That is a draw-call list inlined into a
single kernel, and it is the loop shape `CellGrid` already emits for the
uniform case.

## 4. Loop–Blinn is the shape of a glyph kernel over a domain extent

`ttf_curve_analytical.rs` is a scanline rasterizer written as a kernel:
per segment, solve for where the curve crosses `y = Y`, then compare `X` to
the crossing. It is inherently row-shaped, which is why all of its cost
hoists into the row prologue (§ "Why now"). It is correct without an extent,
because the winding number is global, and slow without one.

Loop–Blinn asks a per-pixel question instead. Assign `(u, v) = (0,0), (½,0),
(1,1)` to a quadratic's control points, interpolate across the control
triangle, and the curve is the zero set of `u² − v`. One affine map and one
`MulAdd` per pixel; nothing Y-only to hoist; the glyph interior is an
ordinary triangulation of the control polygon. Its antialiasing is
`f / ‖∇f‖`, which is the gradient-normalized ramp the font code already
computes, and `Dwrt` would derive `∇(u² − v)` through the affine map with
no new code.

The property that matters for this plan: **outside its control triangle,
`u² − v` is not slow, it is wrong.** It is the implicit of the infinite
parabola. Loop–Blinn is only meaningful *bounded*, so for it the extent is
not an optimization but the denotation. On the GPU that bound is the
rasterizer: the fragment shader never runs outside the triangle. In
pixelflow that bound is a domain-side extent, and the triangulation is the
BSP over which each pixel finds its one triangle. It is the first
formulation under which "a compiled glyph keeps up with an atlas read" is
plausible: a few guarded selects plus eight ops against a gather's latency.
It was not plausible at forty-eight Y-only ops per segment per row.

This section records the direction, not a commitment. TrueType is
quadratics only, which is the easy case; the triangulation and the
overlapping-control-triangle subdivision are real host-side geometry.

## 5. Compiler finding: exclusivity does not survive the partition

Recorded here because §3's domain-side extent is what makes it moot for
grids, and §4 is what makes it moot for glyphs; until both land, this is
where `O`'s 33 ns/px goes.

The emitter computes two facts about a kernel and never joins them:

- `partition_by_scope` takes `variance` and the binder mask, nothing else.
  It moves Y-only values into the row prologue with no memory of which
  select arm they were exclusive to.
- `analyze_select_guards` runs at emission on `allocation.body().schedule()`,
  both call sites. By then the Y-only values are gone from the body, so the
  arm it sees is the X-dependent remainder (the 9- and 23-op guards in the
  telemetry). The row prologue never gets `cluster_select_arms` at all, per
  a comment that reasons about the frame prologue and applies it to both.

The set the row prologue should compute *under a guard* is
**arm-exclusive ∧ Y-only**, guarded by a mask that is itself Y-only and
hoists alongside. Every input to that derivation is already computed. A
synthetic Y-only select with an expensive exclusive arm *does* get guarded in
the row prologue today (telemetry `guarded=37` of 45; timing ratio 3.2× vs
1.9× control), so the machinery works; it is the font kernel's arms that lose
exclusivity because the hoist strips it. `guards.rs` already says exclusivity
is "a property of the schedule, not of allocation"; the refinement is that it
is a property of the DAG, computed once before partitioning, and the scopes
inherit it.

Two honest limits: it is as tight as the mask the font wrote (for quads,
`disc ≥ 0` is a half-plane; a host-computed `y_min ≤ Y < y_max` is the tight
mask, and interval-bounding a quadratic is the font's job, not the
emitter's); and coherence remains a data property, ideal here only because a
row-uniform mask cannot diverge within a row.

## 6. Stages

Each stage is independently landable and leaves every benchmark's emitted
bytes unchanged unless it says otherwise.

**L1 — Z and W become uniforms.** Depends on the uniforms plan's leaf.
`Lattice::extent` becomes `[u32; 2]`; `Lattice::frame(w, h)` loses its `z`;
`PlaneRegion::on_slice` takes uniform values instead of coordinates. Kernels
reading `Kernel::z()` / `Kernel::w()` read a uniform slot. Identity gate: the
psychedelic and chrome packed programs' emitted bytes are unchanged (the
value already arrives at runtime; only its spelling moves).

**L2 — `origin` is removed.** `Lattice` is `{ extent }`. The pixel-center
`0.5` moves to a contramap at the two call sites that use it (`font_rendering`
bench, glyph cache). `Lattice::point` becomes `eval_at`. `Lattice::scanline`
becomes a row-range collapse. Identity gate: every golden unchanged.

**L3 — index ranges get a union.** `Lattice` (or a new `Domain`) is a sum of
index ranges; `collapse` distributes. Disjoint by construction, checked at
build. `text()` for a monospace font becomes a union of per-glyph ranges
with a kernel per range, and its per-pixel cost stops depending on string
length. This is the stage with a measurable target: `pixelflow_text_sizes/26`
from 4.12 ms toward the FreeType-shaped 55 µs, and the `text_sizes` bench
fixed to render the number of characters its label claims.

**L4 — `CellGrid` geometry becomes uniforms; DPI becomes a contramap.**
Resize is a parameter write. The `in_grid` select becomes a border range.
Depends on L1 and L3.

**C1 — exclusivity survives the partition** (§5). Independent of L1–L4;
lands whenever. Gate: `O` at 40 px moves from 33 ns/px toward the 2 ns/px
its 640-px number already shows, with the font's tight `y_min/y_max` mask
supplied alongside.

**G1 — Loop–Blinn glyph kernel** (§4). Depends on L3 for the domain-side
triangle extent. Not scheduled.

## 7. Constraints

- **Loop bounds stay static.** An extent is compiled against; a uniform can
  be a gather index but never an extent. Unchanged from the uniforms plan.
- **The law is the test.** `index(collapse(f)) = f` with no shift and no side
  condition, pinned by a test that fails if anyone reintroduces a coordinate
  onto the index.
- **No coordinate on the domain side.** A sub-lattice is a `usize` range. If
  a call site wants a coordinate, it wants a contramap, and the type says so
  by having nowhere else to put it.
- **The select stays.** For overlapping pieces it is the right encoding and
  the guard machinery is right to exist. This plan removes the cases where a
  select was standing in for a domain extent, not the select.

## 8. Non-goals

- Changing the emitted code for any frame-shaped bake. L1 and L2 are type
  changes over a runtime that already behaves this way.
- Proportional-font layout, kerning, shaping. L3's union is the primitive
  those would compose over; they are not this plan.
- Cubic Béziers in G1. TrueType is quadratic; CFF is later.
- Retiring the range-encoded extent. Overlap is real and the select is its
  encoding.
