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

Ordered so each step is independently landable:

1. **Second lowering strategy for `Reduce`.** `expand_reduce` always statically
   unrolls; a 1920-wide domain needs a loop, not 1920 body copies. Choose
   loop-vs-unroll by extent (or an explicit scope tag) in `lowering.rs` — one
   place, all backends inherit it. Semantic gap to design around: `over` is
   `⊕_i` (eliminates a dimension); the render loop is `for i { out[i] = f(i) }`
   (iterates and stores). Either a monoid-less domain binder or a
   store-tagged `Reduce`.
2. **Two `IsaBackend` primitives** (scalar counter compare-and-branch, ~5
   lines per ISA) replace the three existing hand-encoded loops (aarch64
   hoisted scanline, x86 scanline, AVX-512 collapse).
3. **Generalize the collapse entry** off AVX-512: synthesize X as an induction
   value (`base + lane_seq`, vector-step add — deletes the `xs` array), add the
   Y loop (per-row Y in a register, enabling the existing hoist machinery as
   true LICM), add tail handling. Keep `rdi`/`x0` = ctx — the scanline ABI
   repurposes the ctx register for `xs` and therefore can never gather;
   the collapse ABI is the one to grow. Note the one ABI asymmetry: aarch64
   has 8 callee-saved vector regs for hoisted values; x86-64 has none (hoisting
   must be stack-slot-based there).
4. **Wire `Lattice::collapse`/`bake`** to the collapse entry (per-mode cache
   key in `jit_cache`; today it refuses to cache `Buffer` arenas at all, which
   now matters — `BilinearSampler` compiles one per glyph).
5. **Delete the superseded machinery:** the x86 Sethi-Ullman tree-walk emitter
   (second codegen strategy, no spilling, no CSE), both hand-rolled scanline
   loops + `ScanlineKernelFn` (SSE2-only — 4× narrower than the per-batch path
   on an AVX-512 build), and `pixelflow-core/src/jit.rs` (an abandoned
   inline-cache stripe evaluator with zero callers).

Caveats the scanline paths teach: they skip Stage-1 lowering (`sin`, `Dwrt`,
`Reduce` silently unsupported) and hard-error on `Nary`. The generalized
collapse entry must keep running the full lowering chain, as
`compile_collapse_avx512` already does.

## Measurement plan (do this first)

- `pixelflow-pipeline/src/bin/bench_hoisted_scanline.rs` is already the
  per-call-overhead benchmark (collapse-loop vs per-batch on 1920×1080);
  needs an x86-64 lane and de-gating from `training`.
- `pixelflow-runtime/examples/bench_psychedelic.rs` is the JIT-vs-LLVM parity
  number (`kernel_raw!` vs `kernel!` vs `kernel_jit!`, ns/pixel).
- Add a criterion bench in `pixelflow-ir/benches/` (crate has none): N ×
  `KernelFn` vs one `CollapseKernelFn` on the same arena — isolates call
  overhead from everything else; the acceptance metric for the `over` work.
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
