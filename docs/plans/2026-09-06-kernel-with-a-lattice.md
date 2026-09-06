# A kernel with a lattice is the only evaluation API

**Date:** 2026-09-06
**Status:** plan of record (project direction, stated 2026-09-06)
**Supersedes, in scope:** the "Surface" lane of
[2026-07-28-jit-performance-parity.md](2026-07-28-jit-performance-parity.md);
completes P6 of [2026-07-20-kernel-unification.md](2026-07-20-kernel-unification.md)
for evaluation.

## The direction

**Per-batch evaluation is not an API.** The only way a consumer turns a
kernel into numbers is to hand the compiler a **kernel with a lattice** and
receive the tabulated result: `Lattice::bake(&Kernel) -> DiscreteManifold`
for a field, and its packed-pixel form for a frame. The compiler owns the
loop nest, the invariant hoisting, the pixel pack, and the register
allocation across all of it. Rust-side loops that call something once per
SIMD batch — a `Manifold::eval` per sixteen pixels across an `extern "C"`
boundary, a rasterizer walking a `dyn Manifold` — are the thing this
removes.

Parallelism stays where it is: a frame is baked **one collapse call per
stripe** (a band of rows), stripes across threads, as `render_cell_grid`
already does. Within a stripe the kernel is the whole program.

`Manifold::eval` itself is not what goes. It is the algebra's semantics —
the `kernel!` tier computes with it, and the combinator library is defined
by it — and it stays as the definition of the language. What goes is every
place a *consumer* reached for it to get a buffer.

## Why: the 2× is the boundary, and the compiler already wins without it

Measured 2026-09-06 on this host, the psychedelic shader (four `exp`, four
`sin`, three channels), 1920 wide, release builds, against the
monomorphized `kernel!` expression template:

| path | tier | templates | JIT | JIT vs templates |
|---|---|---:|---:|---|
| per-batch `eval`, one C call per SIMD batch | SSE2 | 22.8 ns/px | 25.0 | 1.10× slower |
| (`kernel_jit!` through `bench_psychedelic`) | AVX-512 | 3.39 | 6.71 | **1.98× slower** |
| one collapse call per plane (`Lattice::bake`) | SSE2 | 22.6 | 9.6 | **2.35× faster** |
| | AVX2 | 6.4 | 4.2 | 1.5× faster |
| | AVX-512 | 3.69 | 2.63 | 1.4× faster |

Same shader, same lattice, results agree to 5e-4 (FMA and reassociation).
The per-batch path pays the ABI — every vector register caller-saved,
every invariant recomputed — and that is the whole 2×. The collapse path
beats the templates because it hoists: three of the four `exp` and two of
the four `sin` are row-invariant, and the kernel computes them once per row
where LLVM recomputes them per batch. Nothing in codegen needs to be fixed
for the JIT to be competitive; the per-batch path needs to stop existing.

Production is already there: the terminal's cell grid bakes through one
packed collapse kernel per stripe; glyphs bake through `Lattice::bake`.
`kernel_jit!` has one caller in the tree, the benchmark.

## Inventory: every per-batch surface, and what replaces it

| surface | replaced by |
|---|---|
| `kernel_jit!` → `__JitWrapper: Manifold` (calls the JIT per batch) | `kernel_jit!` returns a **`Kernel`** (zero params) or a builder returning one. `Lower` impl unnecessary — a `Kernel` splices. Its `Manifold` impl is deleted. |
| `JitManifold::eval_row` / `eval_at` / `eval_grid` | deleted. Callers bake: `Lattice::point(..).bake`, `Lattice::index(n).bake`, `Lattice::frame(..).bake`. `call_collapse` stays as the one entry the collapse driver uses. |
| `Lattice::collapse(&impl Manifold)` (and `collapse_with` / `collapse_scalar` / `collapse_axis` if nothing but tests call them) | deleted. A `kernel!` struct lowers (`Lower`, already emitted by the macro) and bakes; hand-written test manifolds become arena kernels. Reductions are `Kernel::over` inside the kernel. |
| `Scene::Surface(Arc<dyn Manifold<Output = Discrete>>)`, `rasterize`, `render_parallel`, `render_work_stealing`, `execute_stripe`, `materialize_discrete*` | `Scene` is a **packed program over the frame lattice**: `CellGridPackedProgram`/`Frame` generalized to any four channel kernels plus byte-lane shifts, with the pack inside the kernel (`packed_kernel` already does this), rendered by `render_cell_grid`'s stripe loop. `Scene::CellGrid` becomes an instance. The HiDPI contramap in `render_coordinator` becomes `Kernel::at` — precomposition in the language, not a Rust wrapper. |
| the AVX-512/SSE2 `Discrete::pack` in Rust | the pack is IR (`trunc_to_int`, `shl`, `bitor`), as the cell grid already does. |

## What does not lower yet, and the decision (made 2026-09-06)

Hand-written combinators with `Manifold` but no `Lower` cannot be baked:
`Color`/`ColorCube`, `spatial_bsp`, `baked.rs`, and **`scene3d`** — the
ray-marched chrome spheres in `chrome_sphere`, `animated_sphere`,
`chrome_asm`, and `scene3d_test`. Ray marching is a fixed-count iteration
`x_{n+1} = f(x_n)`, and the IR has no iteration construct.

**Decision: the language is a DAG, and it stays one.** No iteration binder.
Finite iteration is *unrolled at kernel construction*: `f` applied `n` times
in Rust while building the arena is `n` compositions, an ordinary DAG the
e-graph can CSE and fold across (the same move `expand_reduce` makes for
`over`). A trip count that must change is a different kernel — **recompile**,
through the same cache that already keys on shape. Anything that cannot be
written as a finite unrolled DAG is not a kernel and does not belong in the
language.

**Decision: no colour in the language.** `pixelflow-core` and `pixelflow-ir`
know fields, lattices, and integer/bit ops; they do not know RGBA, byte
lanes, or pixel formats. A colour output is the **colour-cube idea in the
JIT**: four channel kernels (R, G, B, A, each a field in `[0, 1]`), packed
into a `u32` at the frame boundary by IR integer ops the *graphics* crate
composes (`×255`, clamp, `trunc_to_int`, `shl`, `bitor`). The packed program,
its shifts, and the `Pixel`-format mapping live in `pixelflow-graphics`;
what the cell grid keeps in `pixelflow-core` is its lattice geometry and the
channel kernels over its buffers, not the pack.

So S3 is a **rewrite of scene3d as kernel constructors**: ray direction from
the screen coordinate, sphere tracing as `n` unrolled SDF steps, normals via
`Dwrt` (the IR's derivative, already lowered by `lower_dwrt`), reflection and
sky as field arithmetic, each producing four channel kernels. The Jet3-based
`Manifold` impls go with the rasterizer.

## Stages

**S1 — the JIT has no per-batch entry.** `kernel_jit!` returns a `Kernel`;
`JitManifold::eval_*` deleted; `Lattice::collapse*` deleted; every caller
migrated to a bake. `bench_psychedelic` keeps its template baseline as a
loop *inside the benchmark* (a benchmark owns its loop; that is not an API)
and measures the JIT through `Lattice::bake`. Byte-identical emitted code
(`emit::compile` untouched); every suite green on every tier.

**S2 — a scene is a packed kernel over the frame.** The generalized packed
program; `Scene::CellGrid` an instance of it; runtime tests' `ColorCube`
scenes become packed constant kernels; the HiDPI `At` becomes `Kernel::at`.
`Scene::Surface` remains only for scene3d. Gate: the psychedelic shader
rendered as a `Scene` through `Scene::render`, before and after, on SSE2
and AVX-512 — the after must match the collapse row of the table above.

**S3 — scene3d as unrolled kernels** (decision above). Ray marching,
reflection, sky and checker become `Kernel` constructors with a static step
count, producing four channel kernels rendered as a packed scene; the
examples switch over; then `Scene::Surface`, `execute_stripe`, `rasterize`,
`render_parallel`, `render_work_stealing`, `materialize_discrete*`,
`Discrete::pack`, the `Manifold<Output = Discrete>` bound on rendering, and
the Jet3 `Manifold` impls are deleted, and the inventory is empty. Gate: the
chrome sphere at the examples' frame size, `Scene::render` before and after,
both tiers — and the compile cost of the unrolled march at the step count
the examples use, recorded.


## S1 — landed 2026-09-06, with three findings for the inventory

Landed: `kernel_jit!` is a `Kernel`; `JitManifold::eval_row`/`eval_at`/
`eval_grid` are gone (`call_collapse` is the collapse driver's one entry);
`collapse_with`/`collapse_scalar`/`collapse_axis`/`ReduceOp` are gone (a
reduction is a binder inside the kernel — `Kernel::sum_over`); every caller
bakes; `bench_psychedelic` times one 1920×1080 frame on both sides, then
again under `FastMathGuard`. Net −630 lines. `pixelflow-codegen/src/emit/`
untouched.

**`Lattice::collapse` survives, with exactly three callers**, all
`CachedGlyph` tests in `fonts/cache.rs`, and the reasons it survives are
gaps in this plan rather than in the work:

1. **`Lower` is implemented for two types in the whole tree: `f32` and
   `Kernel`.** The lower/realize design's "one impl per generator beside
   each `Manifold` impl" — coordinates, the op ZSTs, `Select`, `At`,
   comparisons, `Let` — was never landed. Nothing built from combinators
   lowers compositionally; only macro output and `Kernel` values do. The
   inventory's "a `kernel!` struct lowers" is true; "hand-written
   combinators lower trivially" was not, for anything but constants.
2. **`Lattice::bake` refuses an arena that binds memory**
   (`arena.buffers().is_empty()` is asserted). `CachedGlyph`'s coverage is a
   bound buffer sampled by `Gather`. The cell grid bakes bound buffers today
   through its packed frame (`CellGridPackedFrame::bake_packed_rows` binds
   slots at call time), so the capability exists, but only on the packed
   path. A bake that takes bound buffers is the same shape the packed
   program already has; S2's generalized program should expose it for
   fields, not only for packed pixels, and `Lattice::bake` should be the
   buffer-free instance of it.
3. **Outside `pixelflow-core` there is no other way to turn a `Manifold`
   into numbers** (`Field::store` is `pub(crate)`). Deleting `collapse`
   removes that capability from consumer crates entirely — the intent — so a
   blocked caller cannot keep a local loop the way the codegen tests did.

A `Kernel` twin of `CachedGlyph::eval` was deliberately not written: it
would test a definition the rasterizer does not run, the shared-definition
trap `CLAUDE.md` names. `collapse`'s rustdoc says why it is still there.

Bench (this host, single thread, whole frame, median of 5; the host was
~1.5× slower than the reference table that day and both agents saw the same
offset): SSE2 `kernel!` 31.9–35.0 vs bake 11.8–13.3 ns/px (2.6–2.9×);
AVX-512 5.5–5.7 vs 4.0–4.3 (1.3–1.4×). FTZ/DAZ moves neither. The old
ten-scanline baseline had flattered the templates by ~20% (a warm working
set); whole-frame is the honest number.

## Constraints

- **Subtract before you add.** Every row of the inventory is a deletion with
  a migration, not a new path beside an old one.
- **No `Manifold` bound on any evaluation entry.** After S1 the only
  functions that take a `Manifold` and produce numbers are the algebra's own
  `eval` and the benchmark's private loop.
- **The pack is IR.** No Rust-side `from_f32_scaled` on a render path.
- **One measurement per stage**, on the same shader, before and after.
