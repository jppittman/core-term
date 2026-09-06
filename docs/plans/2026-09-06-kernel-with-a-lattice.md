# A kernel with a lattice is the only evaluation API

**Date:** 2026-09-06
**Status:** plan of record (project direction, stated 2026-09-06; shape
corrected the same day — see "The shape")
**Supersedes, in scope:** the "Surface" lane of
[2026-07-28-jit-performance-parity.md](2026-07-28-jit-performance-parity.md);
completes P6 of [2026-07-20-kernel-unification.md](2026-07-20-kernel-unification.md)
for evaluation and states its end state.

## The shape

Three objects, one verb, and the same vocabulary the language has always
had:

- A **kernel** is the description: an arena with a root, built with the
  combinator vocabulary — `X`, `Y`, arithmetic, `.at`, `.select`, `.sqrt`,
  `over`, `Dwrt`. `pixelflow_ir::Kernel` already carries it.
- A **manifold** is a kernel **compiled at a lattice shape**: the thing you
  can sample over a domain. It is compiled IR (today's `JitManifold`),
  specialized on the extents it was compiled for, held behind the global
  compile cache.
- A **lattice** is the domain: extents and origin.
- The one verb is **collapse**: tabulate a manifold over a lattice. The
  compiler owns everything inside it — the loop nest, invariant hoisting,
  the pack, register allocation — and the runtime calls it **one collapse
  call per stripe**, stripes across threads.

```
kernel  ──compile(shape)──▶  manifold  ──collapse(lattice)──▶  buffer
```

`Lattice::bake(&Kernel)` is `collapse(compile(k))` fused and can stop being
a separate name once `collapse` takes a compiled manifold.

**What this corrects.** The first draft of this plan kept the `Manifold`
*trait* — `eval(point) -> value`, one SIMD batch at a time — as "the
algebra's semantics" and deleted `Lattice::collapse` because it consumed
that trait. That had the word bound to the wrong level. The batch-shaped
trait is exactly what forces a Rust loop around the JIT and the per-batch
ABI cost; a compiled kernel cannot be a good implementor of it, and can be
a very good implementor of a manifold interface defined at the *lattice*
level. So: `collapse` is the right verb aimed at the wrong type, and it is
retargeted, not deleted. The trait, the ZST combinator library that
implements it, the `kernel!` LLVM tier that computes with it, and `Lower`
(which was to bridge them to IR and was implemented for two types) are the
**legacy tier**, on their way out because `Kernel` already has the same
vocabulary and the JIT already beats them.

**Per-batch evaluation is not an API.** The only way a consumer turns a
kernel into numbers is to compile it at a lattice's shape and collapse it
over that lattice.

## Why: the 2× is the boundary, and the compiler already wins without it

Measured 2026-09-06 on this host, the psychedelic shader (four `exp`, five
`sin`, six divides, three channels), 1920×1080, release, single thread,
against the monomorphized `kernel!` expression template:

| path | tier | templates | JIT | JIT vs templates |
|---|---|---:|---:|---|
| per-batch `eval`, one C call per SIMD batch | SSE2 | 22.8 ns/px | 25.0 | 1.10× slower |
| (`kernel_jit!` through the old `bench_psychedelic`) | AVX-512 | 3.39 | 6.71 | **1.98× slower** |
| one collapse call per plane (`Lattice::bake`) | SSE2 | 22.3 | 9.7 | **2.3× faster** |
| | AVX2 | 6.4 | 4.2 | 1.5× faster |
| | AVX-512 | 3.58 | 2.44 | 1.5× faster |

Same shader, same lattice, results agree to 5e-4. The per-batch path pays
the ABI — every vector register caller-saved, every invariant recomputed —
and that is the whole 2×. The collapse path beats the templates because it
hoists: three of the four `exp` and two of the five `sin` are
row-invariant, computed once per row where LLVM recomputes them per batch.
At 2.44 ns/px on AVX-512 the kernel spends ~107 cycles per 16-pixel batch,
of which six `vdivps zmm` are ~96: the compiler is on the division
roofline for this expression. The repo's FTZ/DAZ guard and LLVM's fast-math
flags move neither side.

Production is already on the collapse path: the terminal's cell grid bakes
one packed collapse kernel per stripe; glyphs bake through `Lattice::bake`.

## Inventory: every per-batch surface, and what it becomes

| surface | becomes |
|---|---|
| `kernel_jit!` → `__JitWrapper: Manifold` (calls the JIT per batch) | **landed (S1):** `kernel_jit!` returns a `Kernel`. |
| `JitManifold::eval_row` / `eval_at` / `eval_grid` | **landed (S1):** deleted; `call_collapse` is the collapse driver's one entry. |
| `Lattice::collapse_with` / `collapse_scalar` / `collapse_axis` / `ReduceOp` | **landed (S1):** deleted; a reduction is a binder inside the kernel (`Kernel::sum_over`). |
| `Lattice::collapse(&impl Manifold)` — the batch-shaped trait | **retargeted:** `Lattice::collapse(&Manifold)` takes a *compiled* manifold and is the one tabulate verb, one collapse call per stripe. `Lattice::bake(&Kernel)` = `collapse(compile(k))`. Its three remaining callers (`CachedGlyph` tests) move when a compiled manifold can bind buffers (below). |
| `Scene::Surface(Arc<dyn Manifold<Output = Discrete>>)`, `rasterize`, `render_parallel`, `render_work_stealing`, `execute_stripe`, `materialize_discrete*`, `Discrete::pack` in Rust | **S2:** a scene is a compiled manifold with four channels — four channel kernels compiled at the frame's shape with the pack inside (`×255`, clamp, `trunc_to_int`, `shl`, `bitor`, all IR), rendered one collapse per stripe with work stealing. The cell grid is an instance. The HiDPI contramap becomes `Kernel::at`. |
| `Manifold` trait (`eval(point)`), the ZST combinators, `kernel!`'s LLVM tier, `Lower` | **S4, the legacy tier:** retired once nothing consumes them. `Kernel` is the language; the compiled manifold is the object. |

**No colour in the language.** `pixelflow-core` and `pixelflow-ir` know
fields, lattices and integer/bit ops — not RGBA, byte lanes, or pixel
formats. A colour output is the colour-cube idea in the JIT: four channel
kernels in `[0, 1]`, packed at the frame boundary by integer IR ops the
*graphics* crate composes. The packed program, its shifts and the
`Pixel`-format mapping live in `pixelflow-graphics`; the cell grid keeps its
lattice geometry and channel kernels in core, not the pack.

**The language is a DAG.** No iteration binder. A fixed-count iteration is
unrolled at kernel construction into an ordinary DAG the e-graph can CSE
across (the same move `expand_reduce` makes for `over`); a trip count that
must change is a recompile through the cache that already keys on shape.
Nothing that cannot be written as a finite unrolled DAG is a kernel.

## Stages

**S1 — the JIT has no per-batch entry. Landed 2026-09-06 (#1172).**
`kernel_jit!` is a `Kernel`; the `JitManifold` evaluators and the lattice's
reduction family are gone; every caller bakes; `bench_psychedelic` times one
whole frame on both sides, then again under `FastMathGuard`. Net −630
lines; `pixelflow-codegen/src/emit/` untouched. Three findings:

1. **`Lower` is implemented for two types in the tree, `f32` and
   `Kernel`.** The lower/realize design's per-generator impls were never
   landed; nothing built from combinators lowers compositionally. Under the
   corrected shape this is not a gap to fill: there is nothing to lower
   when kernels are already IR, and `Lower` retires with the tier.
2. **`Lattice::bake` refuses an arena that binds memory** while the packed
   frame binds buffers at call time. A compiled manifold must be able to
   bind buffers for fields as well as packed pixels; that is S2's
   generalized program exposed for one channel, and `Lattice::bake` is its
   buffer-free instance. `CachedGlyph` (a `Gather` over bound coverage)
   moves then, and `collapse` is retargeted then.
3. **Outside core there is no other way to turn a `Manifold` into
   numbers** (`Field::store` is `pub(crate)`). Intended.

Bench (this host, whole frame, single thread, median of 5): SSE2 `kernel!`
31.9–35.0 vs bake 11.8–13.3 ns/px (2.6–2.9×); AVX-512 5.5–5.7 vs 4.0–4.3
(1.3–1.4×). The old ten-scanline baseline had flattered the templates ~20%.

**S2 — a scene is a compiled manifold with four channels.** The packed
program generalized from cell-grid geometry to any four channel kernels plus
shifts, living in graphics; `Scene::CellGrid` an instance; a bound-buffer
bake for fields exposed from the same program; runtime fixtures stop
constructing `Scene::Surface`; the stripe loop writes straight into the
frame (no staging, no per-row copy) and pulls small stripes work-stealing
from a pool sized to the cores; FTZ/DAZ on the workers. Gate: the
psychedelic shader through `Scene::render`, threads 1/4/12, both tiers —
first measurement of the recovered prototype: packed beats the surface
path 2.3–2.5× on SSE2 but scales only 1.8× on 4 cores at AVX-512 and loses
when oversubscribed, which is the staging copy and the fixed stripes, not
the compiler. Acceptance: packed never slower than surface at any thread
count; ≥ 3× on 4 cores on both tiers; no loss at 12 threads on 4 cores.

**S3 — scene3d as kernel constructors. Landed 2026-09-06.**
`pixelflow_graphics::scene3d` is plain functions over small structs —
`Ray::through_screen`, `Sphere::hit`, `Plane::hit`, `Hit::select`,
`checker`, `sky` — each producing `Kernel`s and a scene producing four of
them, compiled through S2's `PackedProgram` into `Scene::Packed`. No
`Manifold` impls, no `kernel!`, no jet domain, no `Discrete`. The jet tier
moves to `scene3d_surface` for S4 to delete, with two consumers left on it:
the S3 gate's "before" side and `subdivision_autodiff`, whose
`SubdivisionGeometry` evaluates a Catmull-Clark limit surface in `Jet3` and
has no kernel form. The six CI-named contracts render through the packed
lane; `mullet_vs_3channel_comparison` is replaced by
`four_channels_share_one_geometry`.

**The paragraph above this one was wrong about the code: scene3d is
analytic and there is no ray march.** Nothing in the module iterates ("No
iteration. Nesting is occlusion."): the sphere's `t` is the quadratic's
near root with an epsilon under the radical for grazing rays, the plane's
is a division, and reflection is a Householder step. So S3 needed no
unrolled march and no language feature; every geometry is a closed-form
kernel, and every derivative comes from `Kernel::dx()/dy()`, which
differentiates screen → direction → hit → reflection → hit symbolically.
A march, if a scene ever wants one, is `n` steps of an ordinary Rust `for`
at construction and still a DAG when it reaches the compiler.

Five findings, most of them about what the jet tier's derivatives were
actually doing:

1. **The hit mask was the discriminant, laundered through a derivative
   test.** `Surface` accepted a hit on `t > 0` plus `|∇t|² < 10⁸`; for a
   miss the sphere's `Field::sqrt` (`sqrt_fast`) selects 0, whose
   derivative rule `1/(2√u)` is then infinite, and that is what rejected
   the miss. Stated directly as `disc > 0` here, which also keeps the
   silhouette off `NaN > 0` — unordered and therefore *true* on x86, false
   on aarch64 (CLAUDE.md's `Gt` row). A mask read off `t` would have lit
   the whole frame on one target and nothing on the other.
2. **The jet tier's antialiasing filter was ~540 pixels wide.** Its jets
   were seeded *after* the pixel-to-screen remap, so `|∂P/∂screen|` was per
   *normalized* screen unit — half the frame height. That is why its
   distant floor showed whole flipped cells (coverage → 0 selects the
   neighbour's colour outright) while its near floor smeared checker edges
   over a hundred pixels. `Hit::footprint` is one pixel, which is what a
   footprint means, and the difference is 52% of the pixels in two goldens.
3. **The jet tier's reflection is off a non-unit normal.** It normalizes
   the tangent frame's cross product with `n_len_sq.sqrt().rsqrt()` — which
   is `|n|^-½`, not `|n|^-1` — so its "unit normal" has length `√|n|` and
   `D − 2(D·N)N` is a reflection only where the screen-to-surface map has
   unit area scale. That is very likely what the hand-tuned `2/|cos θ|`
   curvature factor was compensating for, and it is why the gate's mirror
   row disagrees over the whole sphere rather than at its edges. The packed
   lane reflects off the sphere's analytic normal;
   `a_reflection_off_a_sphere_is_a_unit_ray` pins `|R| = 1` wherever the
   sphere is hit, to 2.5% — loose only because `t = b − √(b² − c)` loses its
   leading digits at the grazing rim, which is worth a stable quadratic form
   some day.
4. **A checker filter must wash out to grey, not to its neighbour.**
   `coverage = clamp(d/f, 0, 1)` sends a pixel *on* an edge to the
   neighbour's colour and a surface whose footprint exceeds a cell to
   whichever cell its centre lands in, so cells swap across every boundary
   and a grazing floor flickers between whole cells. A box filter covers `½
   + d/f` of the cell it is in, and with that the two reflective goldens
   pass at every ISA level; with the old form they failed AVX2 by 1.10% of
   pixels and AVX-512 by 1.61%, over the golden helper's 1% platform-noise
   budget, at whole-cell flips.
5. **Time is a coordinate, not a rebuild.** `animated_sphere` rebuilt its
   scene per frame with `t` baked in as a constant, which under kernels is
   a ~200 ms JIT compile per frame. The sphere's centre is now
   `sin(W)·amplitude` and the frame binds its timestamp with
   `PackedFrame::on_slice` (`PlaneRegion` gained the `(z, w)` plane its
   band lies in, private and set by constructor). This is the
   uniform-shaped need of
   [2026-09-06-uniform-slot-identity.md](2026-09-06-uniform-slot-identity.md)
   met by a coordinate: it costs nothing because the scene already had a
   free axis, and it does not generalize to a second uniform.

**The gate** (`pixelflow-runtime/examples/bench_scene_chrome.rs`), chrome
sphere at 1920×1080, median of 9 frames after 3 warm-ups, on an otherwise
idle 4-core host (load ≈ 1.0 from the session itself):

| tier | threads | surface (ns/px) | packed (ns/px) | packed/surface |
|---|---:|---:|---:|---|
| SSE2 | 1 | 13.89 | 43.30 | 0.32× |
| SSE2 | 4 | 4.27 | 11.86 | 0.36× |
| SSE2 | 12 | 4.27 | 11.43 | 0.37× |
| AVX-512 | 1 | 6.08 | 15.61 | 0.39× |
| AVX-512 | 4 | 2.14 | 4.05 | 0.53× |
| AVX-512 | 12 | 2.40 | 4.08 | 0.59× |

Compile cost: 429,395 arena nodes over the four channels, built in 27 ms
and compiled in 211 ms (SSE2; 229 ms at AVX-512) to 8,881 bytes of code
(6,131 at AVX-512). One compile, not one per frame. No saturation
safety-ceiling panic.

Agreement, before either lane is timed. The checker has to come out of any
row that is meant to check something else, because the filter width is one
of the things that changed; so the geometry row is a **matte** sphere over
a matte floor under the sky, and it agrees within 2/255 on **99.48%** of
pixels (max channel delta 102, at the silhouette) — silhouette, horizon,
sky and pack are the same picture. The reflection gets its own checker-free
row, a sphere mirroring the sky alone, which differs on 2.83% of pixels
against a sphere covering 2.93% of the frame: the whole sphere, by up to
63/255, which is finding 4 below and not an antialiasing difference at all.
As shipped, the two lanes differ on 51.2% of pixels, which is finding 2 and
is the point.

**The acceptance criterion "packed never slower than surface" is not met,
and the reason is structural rather than a compiler defect.** Decomposed on
this host, SSE2, single thread:

- Floor and sky alone: surface 9.7, packed 18.6 ns/px. With a *constant*
  filter width the packed kernel is 10.3 — so everything except the
  derivatives is at parity, and exact symbolic AA costs ~8 ns/px where the
  jet tier's forward-mode jets cost ~2. Symbolic `dx()`/`dy()` of a hit
  point re-enters every division and square root through the quotient and
  `0.5·rsqrt(u)·u'` rules, where a jet carries `(val, dx, dy)` through each
  op once.
- The chrome scene doubles that: `select(hit, world(R), world(D))` computes
  both worlds in every lane, while the combinator tier's `Select` evaluates
  a branch **only when some lane in the batch needs it** — the sphere
  covers 2.9% of the frame, so the surface lane skips the reflected world
  in ~97% of batches. The packed kernel *has* the equivalent — the emitter
  guards a `Select`'s arm-exclusive schedule range with a branch on the
  mask's uniformity (`pixelflow-codegen/src/emit/guards.rs`) — and it does
  not fire here, for a reason that is in the schedule analysis and not in
  the language: a value is guardable only if **every consumer lies inside
  that one select's arm**, and a colour is four selects sharing one mask.
  The reflected world's geometry (its `t`, hit point, checker cell,
  footprint) feeds all four channels' arms, so from any single select's
  view it is shared, and only the few per-channel scalar ops at the leaves
  are ever skipped. The old tier never had this problem because its colours
  flowed as one packed word through one `Select`.

Two things that look like levers and are not, both measured: raising the
runtime saturation class cap (telemetry says the chrome kernel hits
`class_cap` after 2 iterations at 4,913 of 5,000 classes — hash-consing
alone folds 193k nodes into ~4.9k) from 5,000 to 60,000 costs 30× the
compile time and makes the kernel **15% slower** (43 → 51 ns/px); and
rewriting the scene as `world(select(m, R, D))` — identical per lane, since
a world is a function of its ray — buys 10% while taking the arena from
429k to 4.6M nodes and the compile from 0.2 s to 2.0 s, because the
derivative of a select is a select of both branches' derivatives.

Two things that are levers, neither measured yet, and a stage for them:

**S3b — a choice between colours is one select.** Two fixes, general and
local, and both should land:

1. *Language, typed:* `Bits::select(mask, a, b)`. `Select` is already a
   bitwise blend on every backend, so a select over packed words is the
   same instruction with an honest type. A colour in graphics becomes a
   tree — four channel kernels at the leaves, a choice between colours at
   the nodes — that `packed_kernel` lowers by packing each leaf and
   selecting on the words. `Hit::select` then produces **one** select whose
   arms are whole packed colours, geometry included, and the existing
   per-select guard skips the reflected world exactly as the old tier did.
   The pack stays in graphics; core still knows no colour.
2. *Codegen, general:* guard analysis over a **group** of selects with the
   same mask — exclusivity against the union of the group's arms, the
   group's arm-exclusive ops clustered contiguously in the schedule ahead
   of the first select — so any kernel that selects the same condition
   several times (every multi-channel scene, every masked field) skips
   what it can without being restructured for it.

The derivative cost is the other half and is measured separately: symbolic
`dx()`/`dy()` of a hit point costs ~4× a forward-mode jet on this host, and
whether that is the derivative rules re-entering `Recip`/`Rsqrt` estimate
chains, or the class cap starving CSE of the expanded derivative, is a
question for `collapse_cost` (docs/plans/2026-09-01-schedule-cost-model-denotation.md §9),
not for a guess. Gate for S3b: the chrome scene, packed ≥ surface at every
thread count on both tiers — the acceptance S3 did not meet — and S4 does
not begin until it is.

**S4 — the legacy tier retires.** `Scene::Surface`, `execute_stripe`,
`rasterize`, `render_parallel`, `render_work_stealing`,
`materialize_discrete*`, `Discrete::pack`, the `Manifold<Output = Discrete>`
bound on rendering, the Jet3 `Manifold` impls, then the `Manifold` trait
itself, the ZST combinator library, `kernel!`'s LLVM backend and `Lower`.
`JitManifold` takes the name `Manifold`. `Lattice::collapse` takes the
compiled type and `bake` folds into it. The inventory is empty.

## Constraints

- **Subtract before you add.** Every row of the inventory is a deletion with
  a migration, not a new path beside an old one.
- **No batch-shaped evaluation entry.** After S4 the only function that
  takes a manifold and produces numbers is `Lattice::collapse`, and its
  argument is compiled IR.
- **The pack is IR.** No Rust-side `from_f32_scaled` on a render path.
- **One measurement per stage**, on the same shader, before and after.
