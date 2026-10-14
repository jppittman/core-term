# Tests check the as-if contract

**Status:** accepted (JP, 2026-10-14). Supersedes the location rule of
`docs/plans/2026-10-08-selection-is-a-phase.md` §0.6.

## What a compiler promises

JP, 2026-10-13: a compiler "takes what you wrote and it's going to generate
some assembly that is behaviorally correct and, in our case, hopefully quite
fast across a variety of different inputs." And: "you don't want to be
asserting the behavior of the optimizer."

That is the C standard's as-if rule (C99 §5.1.2.3): an implementation owes
the program's observable behaviour and nothing about how it produces it. GCC's
own suite shows both sides. Its torture tests compile one program at every
optimization level, run it and compare the output; they check the promise.
Its `scan-assembler` tests grep the emitted code; they are its notorious
change detectors.

So a test here is a **law about the contract, checked against an independent
oracle**. It names something a caller relies on, it survives every refactor
that keeps the promise, and it fails when the promise breaks.

## The contract (tested)

1. **Values.** A compiled kernel's output equals its meaning, against an
   oracle that shares no code with the compiler (Rust's `f32`, `libm`),
   within the documented per-op tolerance and the per-target rows of
   CLAUDE.md "Floating point at the edges". Range is hard; precision is a
   tunable; outside a documented domain the answer is NaN.
2. **Refusals.** What is rejected is rejected by name, never miscompiled.
3. **No crash.** Big frames, register pressure, deep nesting: observed as
   correct output, not as a count of what the allocator did.
4. **Determinism.** The same kernel on the same target compiles to the same
   kernel (compile twice, compare), never to a pinned byte string.
5. **The ABI and the public API.** `fn(ctx, out, pitch)`; what `compile`,
   `CompiledKernel` and `jit_cache` document.

External specifications are contracts too: the image goldens, and an encoder
test of one instruction against the ISA manual's encoding.

## Not the contract (measured)

Speed, compile time and code size are measured by benchmarks; code size gets
at most a sanity ceiling there. Never asserted in a test: instruction counts,
loads, stores, spills, rematerializations, what was hoisted, frame size,
which instruction was selected, which register was assigned, or the shape of
any intermediate structure (schedules, scopes, ownership regions, blocks).
`EmitTraffic` is an instrument the benchmarks read, not a test target.

A slowdown nothing fails is a gap in CI ("CI is the gate"), and the fix is a
benchmark check, not a unit test.

A performance claim is tested as an outcome, against a baseline, never as a
mechanism. JP: GCC might assert that a trigonometric benchmark built with
fast-math outperforms the same benchmark without it, and that is not
brittle; it does not test "the inner workings of the register allocator ...
when you enable fast math", or look for CPU flags. Here the two
differentials below are timed as well as compared: the optimized corpus is
no slower than its `kernel_raw!` twin, and AVX-512 no slower than AVX2 on
the same host. Both are relative, on one machine at one time, so host noise
largely cancels. Compile time and a code-size ceiling are absolute
measurements with regression thresholds. All of it lives in the benchmark
gate.

## Where a test lives

Location is not the criterion; what the test asserts is. An in-file test is
fine when it states a law of that module's documented contract — `asm.rs` is
a pure function from program to binary accepting exactly the legal programs;
`Pass::then` is associative — and would survive any refactor keeping that
law. A test of an intermediate structure with no stated law is a change
detector wherever it lives, and is deleted. A detector that guarded a real
property by proxy is replaced by the property.

## The suite this points at

Torture-style differentials over generated kernels as well as hand-written
ones, each against the Rust oracle:

- **Across tiers.** AVX2, AVX-512 and NEON agree within the documented
  per-target rows.
- **Across optimization.** `kernel!` against `kernel_raw!` (the `Identity`
  optimizer): the e-graph is tested without one assertion about what it
  chose. The documented divergences (`min`/`max` on NaN) are written into the
  comparison, not discovered by it.

Generated kernels come from a small hand-written generator over the arena
builder, not a new dependency (subtract before you add).
