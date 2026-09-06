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

**S3 — scene3d as unrolled kernels.** Ray direction from the screen
coordinate, sphere tracing as `n` unrolled SDF steps, normals via `Dwrt`,
reflection and sky as field arithmetic, each producing four channel kernels;
the examples switch over. Gate: the chrome sphere at the examples' frame
size through `Scene::render` before and after, both tiers, and the compile
cost of the unrolled march at the examples' step count, recorded.

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
