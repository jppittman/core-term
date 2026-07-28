# JIT Performance Parity: Scoping

**Date:** 2026-07-28
**Status:** Scoping (pre-implementation)
**Context:** feature parity (P6 of docs/plans/2026-07-20-kernel-unification.md)
is close; this doc scopes the path back to *performance* parity with the
monomorphized expression-template path the JIT replaced. Prior art:
docs/designs/2026-07-25-two-level-ir-and-backend-completeness.md (Parts 2b and
3 remain the correct deep-dive; some of its line references have drifted).

## The headline findings

1. **The loop-in-JIT kernel already exists and nobody calls it.**
   `compile_collapse_avx512` + `emit_avx512_collapse_loop`
   (pixelflow-ir/src/backend/emit/mod.rs) runs the full Stage-1 lowering
   pipeline, wraps `emit_dag_body` in an internal loop, sets the spill frame up
   once, stores straight to an output buffer, keeps `rdi` = ctx so gather
   works, and is tested against the interpreter (a 32-wide matmul in one
   call). It is AVX-512-only, 1-D (Y/Z/W zeroed), takes X from a
   caller-materialized `xs` array instead of generating it as an induction
   value, has no tail handling, and has zero production callers. The "`over`"
   work is generalization and wiring, not invention.

2. **Runtime-built `Kernel`s never see the optimizer.** `kernel!`/`kernel_jit!`
   run e-graph saturation (FMA fusion, CSE) at macro-expansion time, but
   `Lattice::bake(&Kernel)` and `jit_cache::compile_cached` call
   `compile_arena_dag` directly — no saturation, no CSE, no FMA fusion. Every
   runtime-composed kernel (glyphs, `Kernel::over`, spliced compositions) gets
   raw emission. Compounding it, transcendental expansion builds its Chebyshev
   polynomials from bare `Add`/`Mul` on the theory that "re-fusing is the
   optimizer's job" — and the optimizer never runs on this path. This may be a
   larger parity gap than call overhead. **Measure before committing to
   loop-in-JIT as the first win.**

## Current call path and per-call overhead

One JIT call computes one SIMD batch (one pixel per lane); the Rust loop in
`Lattice::collapse` (pixelflow-core/src/lattice/mod.rs) iterates. Costs per
batch:

- Indirect call through a fn pointer re-loaded from
  `Arc<JitManifold> → ExecutableCode` every batch; opaque to the caller's
  scheduler.
- The `extern "C"` boundary makes every vector register caller-saved
  (SysV/AAPCS64), so all loop-carried SIMD state (`x_field`, `step`, Y/Z/W)
  spills and reloads around every call. This dominant cost is invisible in the
  JIT'd bytes.
- Loop-invariant recomputation: Y/Z/W-only subexpressions re-execute per
  batch. Variance analysis + hoisting exist (`variance.rs`,
  `arena_to_hoisted_schedule`) but only feed the aarch64 scanline path, which
  nothing in production uses.
- Constant re-materialization per call (AVX-512: stack round-trip per
  occurrence; aarch64: literal-pool re-anchor).
- `Lattice::collapse` additionally stores each batch to a scratch array and
  `copy_from_slice`s it into the row — an extra store+memcpy per batch that the
  collapse kernel's direct `emit_store` already avoids. (Trivially fixable
  independent of everything else.)

Where the loops live: `Lattice::collapse`/`fold_lanes`/`collapse_axis*`
(pixelflow-core/src/lattice/mod.rs) — production; the combinator rasterizer's
`execute_stripe` (pixelflow-graphics/src/render/rasterizer/mod.rs) is the
monomorphized loop we're chasing. Pathological sites calling
`Lattice::point(..).bake(..)` per single pixel exist in tests.

## The `over` plan (loop-in-JIT)

**Rule of construction (2026-07-28, project direction): one implementation per
job, replacement in place.** The new thing takes the existing good name; the
superseded thing is deleted in the same change, never left beside it as a
slow parallel path.

**Landed 2026-07-28:**

1. `IsaBackend::emit_collapse_loop` — each backend (SSE2/AVX2/AVX-512/NEON)
   wraps the shared `emit_dag_body` in its own ~60-line loop scaffold. X is an
   induction value (caller passes lane-sequential `x0`; the kernel steps it by
   the batch width), Y/Z/W are loop-invariant arguments, coordinate state
   lives in stack slots above the body's spill frame, the ctx register stays
   reserved for gather. `compile_collapse` replaced `compile_collapse_avx512`
   in place; the ABI is `CollapseKernelFn(ctx, out, groups, x0, y, z, w)`.
2. `Lattice::bake` tabulates through it — one call per row instead of one per
   batch, row tails through a one-batch scratch. The mode-tagged
   `jit_cache::compile_collapse_cached` shares compiles.
3. The superseded kernels are gone (−3,240 lines): both scanline compile
   entries and their hand-rolled loops, the x86 Sethi-Ullman tree-walk
   emitter, `ScanlineJitManifold`/`ScanlineKernelFn`, the aarch64
   v8-v15 hoisted-prologue machinery, `pixelflow-core/src/jit.rs`, and the
   two pipeline benches that measured the deleted path.
   `tests/collapse_loop.rs` pins collapse output bit-exact against the
   per-batch kernel (and the interpreter) across arithmetic, selects,
   transcendentals, gather, reduce, and forced spills.

**Remaining, in order:**

1. **`Reduce`'s loop lowering.** `expand_reduce` statically unrolls; a large
   extent needs a real loop. Per the rule above this is a *replacement* of
   the unroll, not a sibling keyed on extent — small trip counts may still
   unroll, but as a decision inside the one lowering, not as two lowerings.
   Semantic note: `over` is `⊕_i` (eliminates a dimension); the render loop
   is `for i {{ out[i] = f(i) }}` — the collapse scaffold is the second shape,
   a `Reduce` loop is the first, and they can share the backend loop
   primitives.
2. **The Y loop (2D collapse).** One call per frame instead of per row; the
   scaffold grows an outer loop stepping the Y slot. This is also where LICM
   lands: `variance.rs` (kept for exactly this) identifies Y/Z/W-only
   subexpressions to hoist out of the inner loop — aarch64 can pin them in
   v8-v15 with a save/restore prologue, x86 uses stack slots.
3. **The e-graph gap** (finding 2 above) — unchanged, and measurement still
   comes first: if runtime-composed kernels are dominated by missing FMA/CSE,
   a bounded optimization pass on the bake path may buy more than any
   further loop work.

## Measurement plan (do this first)

- `pixelflow-runtime/examples/bench_psychedelic.rs` is the JIT-vs-LLVM parity
  number (`kernel_raw!` vs `kernel!` vs `kernel_jit!`, ns/pixel).
- Add a criterion bench in `pixelflow-ir/benches/` (crate has none): N ×
  `KernelFn` vs one `CollapseKernelFn` on the same arena — isolates call
  overhead from everything else. `pixelflow-graphics/benches/font_rendering.rs`
  (bake on a cached compile) already measures the production win end to end.
- To size finding 2 (missing optimization) separately: bake the same
  expression via `kernel_jit!` (e-graph'd at macro time) and via runtime
  `Kernel` composition (raw), and diff ns/pixel. If the gap is large, an
  optimization pass on the bake path (bounded-budget saturation, or at
  minimum a peephole FMA/CSE pass over the arena) may buy more than the loop.

## Other emit-path gaps (flagged, unfixed)

- AVX2 register budget cut from 6 to 4 workspace-wide to paper over
  `emit_gather_scalar`'s temp usage; the source names the fix (red-zone
  spill of the two temps).
- AVX-512 constant materialization: `mov [rsp-4], imm32` + `vbroadcastss` per
  occurrence, no pool, no dedup.
- `Select` compiles to short-circuit branches with a horizontal mask reduction
  per select; worth measuring against a straight blend inside a tight loop.
- No `vzeroupper` anywhere; SSE2/scanline paths use legacy SSE encodings — an
  AVX↔SSE transition hazard when the Rust caller is compiled with AVX. Flagged,
  unmeasured.
- Per-batch fn-pointer reload from `Arc` in `RealizedKernel::eval` (hoistable
  in the collapse-loop world; irrelevant once the loop is inside the JIT).
