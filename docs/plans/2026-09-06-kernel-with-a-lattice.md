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

## What does not lower yet, and the decision it forces

Hand-written combinators with `Manifold` but no `Lower` cannot be baked:
`Color`/`ColorCube` (trivially lowerable, four channel constants),
`spatial_bsp`, `baked.rs`, and **`scene3d`** — the ray-marched chrome
spheres in `chrome_sphere`, `animated_sphere`, `chrome_asm`, and
`scene3d_test`. Ray marching is a fixed-count iteration `x_{n+1} = f(x_n)`,
and the IR has no iteration binder: `over` folds an index with a monoid,
which is a domain, not a dynamics
([kernel-unification P9](2026-07-20-kernel-unification.md) names the gap).

Two ways to close it, and the choice is the project's:

1. **Add the binder.** `Iterate { n, body }` — a fold whose monoid is
   function composition, static trip count, distributed (unrolled) for small
   `n` and factored (looped) for large, exactly the distribute/factor rule
   in [2026-09-01-loop-aware-codegen.md](2026-09-01-loop-aware-codegen.md).
   The e-graph already holds `Reduce` as a binder e-node; this is the
   second. Then scene3d lowers, and the last per-batch path is deleted.
2. **Retire the demos.** Delete scene3d and its examples with the rasterizer.

Until that is decided, `Scene::Surface` and `execute_stripe` remain as the
**only** per-batch remnant, documented as such, with no new consumer
permitted. Everything else in the inventory goes now.

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

**S3 — the iteration binder, or the retirement** (decision above). Then
`Scene::Surface`, `execute_stripe`, `rasterize`, `render_parallel`,
`render_work_stealing`, `materialize_discrete*`, `Discrete::pack` and the
`Manifold<Output = Discrete>` bound on rendering are deleted, and the
inventory is empty.

## Constraints

- **Subtract before you add.** Every row of the inventory is a deletion with
  a migration, not a new path beside an old one.
- **No `Manifold` bound on any evaluation entry.** After S1 the only
  functions that take a `Manifold` and produce numbers are the algebra's own
  `eval` and the benchmark's private loop.
- **The pack is IR.** No Rust-side `from_f32_scaled` on a render path.
- **One measurement per stage**, on the same shader, before and after.
