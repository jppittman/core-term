# Estrin beats Horner on the critical path and loses in the loop — so PixelFlow's polynomials should stay Horner

**Date:** 2026-09-06
**Question:** `passes::horner_step` emits every transcendental polynomial as a serial chain of `MulAdd`s — `a₀ + x(a₁ + x(a₂ + …))`. Horner minimizes operations but maximizes dependency depth. Estrin's scheme evaluates the same polynomial as a `log₂ n`-deep tree over explicit powers `x², x⁴, x⁸, …`, trading `log₂ n` extra multiplies for `O(log n)` depth. Is that trade worth taking for `sin`/`cos`/`exp2`/`log2`/`atan`?
**Answer:** **No, at the degrees PixelFlow uses.** Estrin is 1.2–4.0× faster when a single evaluation is serialized, and the collapse loop erases that advantage by converting a latency problem into a throughput problem: measured as marginal cost per degree, Horner's gets ~2× cheaper inside the loop while Estrin's gets *more expensive*, because in the loop it is charged for ops rather than depth. At AVX-512 the two meet exactly (slope ratio 1.03 — throughput-bound), and once neither schedule is latency-limited the only thing left to compare is op count, where Estrin is strictly worse by its `log₂ n` squarings. In production's regime (one call per contiguous row, the kernel's own loop supplying the independent work) Estrin is a **tie to 11% slower** at every ISA level for every production polynomial. It only wins once the chain outgrows what cross-iteration ILP can hide: degree ≥ 12 on SSE2, ≥ 24 on AVX2, and never within degree ≤ 32 on AVX-512. Production's polynomials are degree 4–9. Two further findings fall out: Estrin introduces an **underflow hazard Horner does not have** (up to 30× slower, fixed entirely by `FastMathGuard`, which nothing in the render path currently holds), and the extraction cost model prefers Horner **for the wrong reason** — it cannot see schedule at all, and is therefore wrong by up to 1.9× on the cases where Estrin does win.
**Reproduction:** `cargo run --release -p pixelflow-pipeline --example horner_vs_estrin`, at each ISA level per `xtask isa-matrix`'s flags. Host: 4-vCPU Intel Xeon @ 2.80GHz (avx512f/dq/vl, fma), Linux, shared VM. Numbers below are medians of 4–5 clean runs × 5 in-session repetitions × the harness's own median-of-20 samples. Runs that tripped `BenchSession`'s sentinel regime-change abort were discarded, not averaged in — roughly a third of runs on this host.

## 1. What was measured

`pixelflow_pipeline::poly` emits both schedules from one ascending coefficient list. The Horner arm is production's shape: `MulAdd(acc, x, c)` per step, exactly what `passes::horner_step` emits. Estrin pairs coefficients into `aᵢ + x·aᵢ₊₁`, then folds pairs against `x²`, `x⁴`, … Both are checked against each other in CI (`poly::tests::estrin_agrees_with_horner`, and `estrin_is_shallower_and_wider_than_horner` pins the structural claim the timings then price).

Three regimes, because the answer differs in each:

| mode | what it does | what it answers |
|---|---|---|
| `Latency` | each evaluation's output feeds the next | the critical path, with nothing to overlap |
| `Throughput` | independent evaluations, one call each | ILP across a call boundary production does not pay |
| `Scanline` (new) | one call per 64-group row; the emitted loop supplies the independent evaluations over one contiguous, ascending address stream | **production** — the shape `Lattice`'s collapse runs |

`BenchMode::Scanline` was added to `jit_bench` for this. It passes `rows = 1`: a single contiguous run of 64 SIMD groups, X advancing lane-sequentially into one ascending address stream — a CPU rasterizer's straight line, not GPU-style tiling. (`TileSlice`'s `rows`/`row_skip_bytes` exist so a caller can land such rows inside a strided destination; `Lattice::collapse_into` passes `full_groups` per row and the scalar tail as the skip. The extra row does not belong in an arithmetic comparison: a row advance costs the same in both arms and only dilutes the ratio.) The other two modes bracket reality from opposite sides; neither is the thing itself, and the gap between them turned out to be the entire answer.

## 2. Results

Every cell is **Horner ÷ Estrin**: above 1.00 means Estrin is faster. Latency/throughput/scanline columns are overhead-adjusted (audit M1); `scanline (raw)` is end-to-end, which is what a shipping decision is made on.

**Read the AVX-512 `latency` column with care, and not per row.** `BenchMode::Latency`'s chaining apparatus — a 4×`LANES`-wide `Point4` stored to the stack, a call, and a store→load round trip, all serial — costs ~30ns per call at 16 lanes. That is the same order as the kernel, so every degree below ~16 reads ~30ns regardless of the polynomial (`n=4` and `n=8` measure 30.25 and 30.10ns/call: four extra serial FMAs for free, which is not a thing that happens). The overhead subtraction removes most of it and leaves a small difference of large numbers, which is why those cells are noisy and non-monotonic. §3's slope table is the same measurement without that defect, and it is what the mechanism should be read from. The `scanline` columns are unaffected — their fixed cost is 0.37ns/lane, an order below the kernel.


### SSE2 baseline (4 lanes, 16 regs, no FMA — `MulAdd` is `mulps`+`addps`) — 5 clean runs

| poly | n | spills H/E | latency | throughput | scanline (adj) | scanline (raw) | scanline (raw) range |
|---|---|---|---|---|---|---|---|
| *degree sweep* | | | | | | | |
| sweep | 4 | 0/0 | 1.33 | 0.91 | 0.99 | 0.99 | 0.99–1.00 |
| sweep | 6 | 0/0 | 1.73 | 0.93 | 0.79 | 0.89 | 0.89–0.90 |
| sweep | 8 | 0/0 | 1.99 | 1.02 | 0.92 | 0.96 | 0.95–1.03 |
| sweep | 9 | 0/0 | 1.91 | 0.89 | 1.04 | 1.03 | 0.92–1.06 |
| sweep | 12 | 0/0 | 2.55 | 1.07 | 1.35 | 1.24 | 1.15–1.34 |
| sweep | 16 | 0/0 | 2.82 | 1.43 | 1.53 | 1.39 | 1.35–1.43 |
| sweep | 24 | 0/4 | 3.69 | 1.76 | 1.92 | 1.75 | 1.74–1.81 |
| sweep | 32 | 0/8 | 3.97 | 2.06 | 2.03 | 1.88 | 1.83–2.00 |
| *production polynomials* | | | | | | | |
| atan | 4 | 0/0 | 1.31 | 0.91 | 0.99 | 0.99 | 0.99–1.00 |
| sin/cos | 6 | 0/0 | 1.73 | 0.93 | 0.79 | 0.89 | 0.89–0.90 |
| exp2 | 6 | 0/0 | 1.74 | 0.93 | 0.79 | 0.89 | 0.89–0.90 |
| log2 | 9 | 0/0 | 1.90 | 0.89 | 0.93 | 0.96 | 0.91–1.06 |
| *underflow hazard (arg ≈ 1e-9)* | | | | | | | |
| hazard | 8 | 0/0 | 2.00 | 0.06 | 0.13 | — | 0.12–0.14 |
| hazard | 16 | 0/0 | 0.82 | 0.11 | 0.17 | — | 0.16–0.18 |
| hazard | 32 | 0/8 | 0.64 | 0.19 | 0.22 | — | 0.21–0.24 |
| *same, under FastMathGuard (FTZ/DAZ)* | | | | | | | |
| hazard+ftz | 8 | 0/0 | 2.01 | 1.02 | 0.93 | — | 0.92–1.06 |
| hazard+ftz | 16 | 0/0 | 2.84 | 1.40 | 1.50 | — | 1.47–1.56 |
| hazard+ftz | 32 | 0/8 | 4.00 | 2.22 | 2.00 | — | 1.96–2.20 |

### AVX2+FMA (8 lanes, 16 regs) — 5 clean runs

| poly | n | spills H/E | latency | throughput | scanline (adj) | scanline (raw) | scanline (raw) range |
|---|---|---|---|---|---|---|---|
| *degree sweep* | | | | | | | |
| sweep | 4 | 0/0 | 1.15 | 0.36 | 0.98 | 1.00 | 0.99–1.00 |
| sweep | 6 | 0/0 | 1.29 | 0.73 | 0.68 | 0.93 | 0.93–0.93 |
| sweep | 8 | 0/0 | 1.65 | 0.78 | 0.82 | 0.95 | 0.95–0.96 |
| sweep | 9 | 0/0 | 1.60 | 0.70 | 0.83 | 0.94 | 0.91–0.95 |
| sweep | 12 | 0/0 | 1.84 | 0.88 | 0.90 | 0.95 | 0.69–0.96 |
| sweep | 16 | 0/0 | 2.17 | 1.10 | 0.99 | 1.00 | 0.97–1.06 |
| sweep | 24 | 0/4 | 2.72 | 1.36 | 1.25 | 1.17 | 0.53–1.18 |
| sweep | 32 | 0/8 | 3.13 | 1.40 | 1.41 | 1.31 | 0.67–1.31 |
| *production polynomials* | | | | | | | |
| atan | 4 | 0/0 | 1.15 | 0.69 | 0.89 | 1.00 | 0.95–1.00 |
| sin/cos | 6 | 0/0 | 1.29 | 0.68 | 0.52 | 0.93 | 0.86–0.93 |
| exp2 | 6 | 0/0 | 1.29 | 0.73 | 0.66 | 0.93 | 0.91–0.93 |
| log2 | 9 | 0/0 | 1.60 | 0.81 | 0.83 | 0.94 | 0.91–0.95 |
| *underflow hazard (arg ≈ 1e-9)* | | | | | | | |
| hazard | 8 | 0/0 | 1.65 | 0.01 | 0.03 | — | 0.03–0.03 |
| hazard | 16 | 0/0 | 2.15 | 0.04 | 0.09 | — | 0.07–0.09 |
| hazard | 32 | 0/8 | 3.13 | 0.08 | 0.17 | — | 0.16–0.17 |
| *same, under FastMathGuard (FTZ/DAZ)* | | | | | | | |
| hazard+ftz | 8 | 0/0 | 1.65 | 0.83 | 0.83 | — | 0.67–3.25 |
| hazard+ftz | 16 | 0/0 | 2.16 | 1.06 | 1.02 | — | 0.95–1.02 |
| hazard+ftz | 32 | 0/8 | 3.13 | 1.41 | 1.42 | — | 1.41–1.45 |

### AVX-512F/DQ (16 lanes, 32 regs) — 4 clean runs

| poly | n | spills H/E | latency | throughput | scanline (adj) | scanline (raw) | scanline (raw) range |
|---|---|---|---|---|---|---|---|
| *degree sweep* | | | | | | | |
| sweep | 4 | 0/0 | 1.22 | 1.04 | 1.02 | 1.01 | 0.91–1.02 |
| sweep | 6 | 0/0 | 1.07 | 0.99 | 0.97 | 0.99 | 0.98–0.99 |
| sweep | 8 | 0/0 | 0.84 | 1.00 | 1.06 | 1.02 | 1.02–1.03 |
| sweep | 9 | 0/0 | 0.84 | 1.00 | 0.99 | 1.00 | 0.99–1.01 |
| sweep | 12 | 0/0 | 0.95 | 1.00 | 1.05 | 1.02 | 0.86–1.06 |
| sweep | 16 | 0/0 | 0.91 | 1.03 | 0.93 | 0.96 | 0.88–1.04 |
| sweep | 24 | 0/0 | 1.02 | 0.84 | 0.84 | 0.89 | 0.89–1.06 |
| sweep | 32 | 0/0 | 1.58 | 1.10 | 0.91 | 0.93 | 0.93–1.11 |
| *production polynomials* | | | | | | | |
| atan | 4 | 0/0 | 1.19 | 1.00 | 0.98 | 1.00 | 1.00–1.08 |
| sin/cos | 6 | 0/0 | 1.46 | 0.99 | 0.97 | 0.99 | 0.86–0.99 |
| exp2 | 6 | 0/0 | 1.17 | 1.00 | 0.97 | 0.99 | 0.91–1.02 |
| log2 | 9 | 0/0 | 0.80 | 1.00 | 0.99 | 1.00 | 1.00–1.19 |
| *underflow hazard (arg ≈ 1e-9)* | | | | | | | |
| hazard | 8 | 0/0 | 0.68 | 0.06 | 0.06 | — | 0.05–0.06 |
| hazard | 16 | 0/0 | 0.76 | 0.08 | 0.07 | — | 0.07–0.08 |
| hazard | 32 | 0/0 | 1.57 | 0.08 | 0.09 | — | 0.08–0.09 |
| *same, under FastMathGuard (FTZ/DAZ)* | | | | | | | |
| hazard+ftz | 8 | 0/0 | 0.88 | 1.00 | 1.06 | — | 0.72–1.06 |
| hazard+ftz | 16 | 0/0 | 0.72 | 1.04 | 0.93 | — | 0.80–1.07 |
| hazard+ftz | 32 | 0/0 | 1.57 | 0.97 | 0.91 | — | 0.91–1.70 |

## 3. Why the loop erases the win

The critical path is real and behaves exactly as theory says: at SSE2 a degree-32 Horner chain takes 4.0× longer than the Estrin tree when evaluations are serialized, and the advantage grows monotonically with degree. That column is not wrong — it is answering a question production does not ask.

The collapse kernel evaluates a **row**: `groups × LANES` independent points, produced by the emitted loop with X advancing lane-sequentially into contiguous memory. Consecutive iterations have no data dependence, so the out-of-order engine has a second, third, and fourth evaluation available to fill exactly the stalls Estrin restructures the polynomial to remove.

The cleanest way to see this is the **marginal cost of one more degree**, differenced across `n = 16 → 32` so the shared prelude and the mode's fixed cost cancel exactly (the protocol `measure_latency_prior` uses). Medians of 6 clean runs, ns per call:

| level | mode | Horner | Estrin | H ÷ E |
|---|---|---|---|---|
| SSE2 | Latency | 2.476 | 0.368 | 6.72 |
| SSE2 | **Scanline** | **1.271** | **0.506** | **2.51** |
| AVX2 | Latency | 1.190 | 0.207 | 5.75 |
| AVX2 | **Scanline** | **0.643** | **0.368** | **1.75** |
| AVX-512 | Latency | 0.974 | 0.428 | 2.27 |
| AVX-512 | **Scanline** | **0.567** | **0.549** | **1.03** |

Read down each pair rather than across:

- **Horner's marginal degree gets cheaper in the loop** — 2.476 → 1.271 at SSE2, 1.190 → 0.643 at AVX2, 0.974 → 0.567 at AVX-512. Consistently a little under half. That is the neighbouring iteration paying for the chain.
- **Estrin's marginal degree gets *more expensive*** — 0.368 → 0.506, 0.207 → 0.368, 0.428 → 0.549. Nothing regressed: in the loop it is no longer being charged for depth, it is being charged for **ops**, and it has more of them.

The two curves therefore converge, and how far they converge is the whole result. At AVX-512 they meet: **1.03**, equal marginal cost per degree. Equal slope means the machine is **throughput-bound** — it is retiring instructions at its own rate, dependency structure no longer constrains anything, and the only thing that can separate two schedules is how many instructions each contains. Estrin contains strictly more (its `log₂ n` squarings), so in that regime it can only be equal or worse, which is exactly the 0.89–1.02 measured. Its extra ops are ~2–3 instructions out of ~10 at production's degrees, hence a ratio that reads as a tie rather than a visible loss.

At SSE2 and AVX2 the convergence is only partial (2.51 and 1.75): with 16 registers the engine cannot keep enough iterations in flight to hide the whole chain, latency stays partly exposed, and past some degree that residue exceeds Estrin's extra ops. That is the crossover, at 12 and 24 respectively.

So the answer is not that AVX-512 is a wash for unclear reasons. **The loop converts a latency problem into a throughput problem, and AVX-512 — 16 lanes, 32 registers, no spills at any degree measured — converts it completely.** The register story is visible in the `spills` column: Estrin spills 4 slots at degree 24 and 8 at degree 32 on both 16-register targets, and never on AVX-512. Those spilling configurations are also the only ones whose timings are unstable — the AVX2 degree-24 scanline ratio ranges 0.53–1.18 across runs, against 0.93–0.96 for the non-spilling degree-6 row. **Once Estrin spills it is both slower and erratic**, which is the shape of a schedule that has run out of the resource it was trading for.

## 4. The underflow hazard is new, and belongs to Estrin alone

Horner never forms a power of `x`. Its accumulator stays within a small factor of `a₀` for any argument, because every step folds a coefficient back in. Estrin computes `x²`, `x⁴`, `x⁸`, `x¹⁶` explicitly, and for a small argument those underflow: at `arg ≈ 1e-9`, `x⁸ ≈ 1e-72` is not representable. An underflowing multiply on x86 is a microcode assist, not an instruction.

The cost, scanline mode, same kernels, one constant changed:

| level | degree 8 | degree 16 | degree 32 |
|---|---|---|---|
| SSE2 | 7.7× slower | 5.9× | 4.5× |
| AVX2 | 33× | 11× | 5.9× |
| AVX-512 | 17× | 14× | 11× |

Holding a `FastMathGuard` (FTZ/DAZ) removes it completely — every hazard row returns to its nominal value, within noise. That guard exists, is exported from `pixelflow-core`, and **is held by nothing outside `pixelflow-core/benches`**: no `pixelflow-runtime` driver, no `core-term` frame loop, no glyph bake. Today's Horner expansions do not need it, which is why nothing has noticed. Any future schedule that forms explicit powers does, and so does any user kernel that does — this is a live gap independent of the Horner/Estrin question.

## 5. Accuracy

Peak-relative error against an `f64` Horner reference over 1024 arguments in `[0, 1)`, measured on the JIT's own arithmetic (so FMA rounding is included, which no scalar oracle reproduces):

| degree | Horner | Estrin |
|---|---|---|
| 4–8 | 4e-8 – 5e-8 | 3e-8 – 6e-8 |
| 9–16 | 4e-8 – 5e-8 | 8e-8 – 9e-8 |
| 24–32 | 4e-8 | 1e-7 |

Estrin costs roughly a factor of two in the worst case and stays inside single-precision noise. This is a *precision* difference, which CLAUDE.md's "Floating point at the edges" puts on the table; the range guarantees are untouched, since they come from the reduction and the `Select`, not from the polynomial's schedule. Accuracy is not what decides this question.

## 6. What the cost model sees

`CostModel::latency_prior` sums per-node costs. It therefore prices the two schedules by op count alone and prefers Horner everywhere — by 5 table-cycles at degree 4 (26 vs 31), 20 at degree 32 (166 vs 186), the extra squarings. Read as a **critical path** through the same table, the same weights say the opposite: 26 vs 21 cycles at degree 4, 166 vs 36 at degree 32.

§3 says exactly when each reading is the right one, and it is not a matter of taste. **An additive cost model is precisely correct in a throughput-bound regime** — where marginal costs are equal, total cost really is the sum of the parts, and AVX-512's 1.03 slope ratio says the machine is there. It is wrong exactly where latency stays exposed: SSE2's 2.51 and AVX2's 1.75 mean a degree of Horner and a degree of Estrin are not the same purchase, and no sum over nodes can say so. So extraction's preference matches the measurement for production's polynomials, and is wrong by up to 1.9× on SSE2 at degree ≥ 12 — not because the table's entries are off, but because additivity itself is a claim about the machine that holds at one ISA level and fails at another. This is a concrete worked example for `docs/plans/2026-09-01-schedule-cost-model-denotation.md`: a pair of extractions with identical semantics, an op-count ordering, a measured ordering that inverts, and an inversion point that moves with ISA level and register file size. It is precisely the residual that document specifies and the `Reranker` seam is held open for.

Note also that the e-graph could not choose Estrin today even if the model wanted it: `horner_step`'s `MulAdd` nodes are created in `passes`, *after* saturation, and there is no rewrite rule that reassociates a polynomial. Adopting Estrin would be a change to the expansion in `pixelflow-ir/src/passes.rs`, not to the optimizer.

## 7. Conclusion

Keep Horner. It is the right schedule for degree 4–9 in a loop that already has the parallelism Estrin manufactures, it needs fewer registers, it is more accurate, and it cannot underflow. `horner_step`'s existing justification — accuracy, and one `MulAdd` per step — survives; this adds the throughput argument it did not previously have.

Revisit if any of these change:

- a polynomial reaches degree ≥ 12 **and** targets a 16-register x86 level (SSE2 or AVX2);
- a polynomial lands somewhere genuinely serialized, where the scanline loop cannot supply neighbouring work — a reduction body, or a dependent chain of transcendentals;
- codegen starts giving the e-graph schedules to choose between, at which point this pair is a ready-made regression fixture for the schedule-cost residual.

Independently of all of that: **something in the render path should hold a `FastMathGuard`**, or the guard should be deleted as dead code. It currently is neither held nor removed.
