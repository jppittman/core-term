# Measurement-integrity audit: benchmark / label-minting harness

**Date:** 2026-08-05
**Scope:** `pixelflow-pipeline/src/jit_bench.rs`, `bin/bench_extraction_3way.rs`,
`bin/bench_jit_corpus.rs`, `bin/gen_bench_corpus.rs`, `bin/bootstrap_extraction_head.rs`,
`src/training/episodes.rs`, `.github/workflows/benchmark_regression.yaml`.
**Threat model:** the e-graph extractor + NNUE is *selected* on numbers this harness produces
(bootstrap labels, episode speedups, the 3-way gate verdict). Anywhere the harness diverges
from real cost, gradient descent will find it. Companion plan:
`docs/plans/2026-08-05-egraph-nnue-research-workflow.md` (Phase 0.3 tracks the fixes).

---

## HIGH severity

### H1. Correctness gate is a single point (training paths) / 8 easy points with 0.2 absolute tolerance (gate path) — a wrong-but-fast kernel can pass

- `jit_bench.rs:279-282` — the bench evaluates at **one** coordinate `(0.5, 0.7, 1.3, -0.2)`,
  splatted across lanes. `BenchResult.output` (`:243-245`) therefore holds 4 *identical* lanes;
  `check_equivalence` (`:250-263`) looks like a 4-lane check but is a **1-point** check.
- `training/episodes.rs:206-207` — episode-level rewrite verification is exactly that 1-point
  check, `EQUIV_EPSILON = 1e-3`. Any unsound rewrite (or extraction of a non-equivalent e-node)
  that coincidentally agrees at that one point mints a training episode with a fake speedup
  label.
- `bench_extraction_3way.rs:59-68` — the gate's GRID is 8 points, all in `[-1.2, 1.2]`: no
  large magnitudes, no near-zero div/sqrt (explicitly avoided per the comment), no denormals,
  no NaN/Inf. `ABS_TOL = 2e-1` (`:76`), `REL_TOL = 0.05` (`:78`). Kernel outputs here are O(1),
  so anything wrong by less than 0.2 *everywhere on those 8 points* passes — e.g. a truncated
  transcendental approximation, a dropped low-order term, or `x·rsqrt(d)` with fewer Newton
  steps. That is precisely the shape of "cheat" a cost-model-guided extractor is rewarded for
  finding.

**Exploit path:** NNUE picks an e-node whose lowering is cheaper-but-sloppier; gate passes at
tol 0.2; geomean improves; the Phase-2 gate declares NNUE the winner on numbers that partially
reflect accuracy loss, not speed.

**Fix (minimal):** (a) evaluate on a randomized 64+ point grid including magnitude sweeps
(1e-4 … 1e4, negatives, ±0.0) per kernel domain; (b) evaluate the *arena* in a scalar f64
interpreter as the reference (see H2); (c) tighten to relative tolerance ~1e-3 except on
documented transcendental-lowering divergences, which should be per-op allowances, not a global
0.2.

### H2. The equivalence reference is the same JIT backend as the thing under test

`bench_extraction_3way.rs:331-336` compiles the no-swap arena with `compile_arena_dag` and uses
it as ground truth (`compile_check_and_bench`, `:288-303`). A JIT emit bug that affects both
forms identically is invisible; more subtly, a *systematic* backend behavior (e.g. an op that
lowers wrong for both) validates itself. Also note the code object that is correctness-checked
is **not** the code object that is timed: `benchmark_jit_arena` at `:305` recompiles from
scratch (`jit_bench.rs:406`). Deterministic today, but the check and the measurement are
formally on different artifacts.

**Fix:** interpret the arena in scalar `f64` (independent of the JIT) as the reference; time
the same `ExecutableCode` you checked (a `benchmark_exec_code`-taking entry point already
exists privately at `jit_bench.rs:266`).

### H3. Fixed constant inputs + independent back-to-back calls: the harness measures throughput under perfect ILP, not latency, at one input value

`jit_bench.rs:31-134` (`eval100!`) issues 100 independent calls; inputs are built **once**
before warmup (`:279-282`, `:322-342`) and never vary. `black_box` protects each *output*, and
the opaque JIT fn pointer protects against rustc const-prop/DCE (that part is sound — see "done
right"). But at the CPU level nothing serializes call N+1 on call N's result, so out-of-order
execution overlaps calls: long-dependency-chain kernels are systematically under-billed
relative to their real single-evaluation latency, and high-latency/high-throughput instructions
(div, sqrt) hide their latency. Meanwhile the single fixed input means data-dependent cost
never appears (no denormal stalls, one select path, perfectly warm everything).

**Corruption path:** labels teach the NNUE that deep chains are cheap; the extractor then
prefers chain-y forms; the 3-way gate — built on the same harness — confirms the preference.
Self-consistent, invisible, and wrong for the production use (per-pixel varying coordinates in
a scanline).

**Fix:** feed the previous output back into one input lane (`x = black_box(prev_out)`) to
serialize the chain — or measure both modes and label with both; additionally cycle inputs
through a small buffer of varied coordinates (which also widens H1's effective check to 100
points for free).

### H4. No core pinning, no QoS class, no sentinel — Apple Silicon P/E placement and thermal drift can multiply labels by 2-3x with zero detection

No `pthread_set_qos_class_self_np`, no affinity, no calibration kernel anywhere in
`pixelflow-pipeline/src`. `bootstrap_extraction_head.rs:123-159` benchmarks 50k expressions in
one process over hours; an E-core placement or thermal throttle mid-run shifts every subsequent
label. Median-of-20 (`jit_bench.rs:301-303`) defends *within* one expression, not *between*
expressions — and between-expression consistency is exactly what a regression target needs.
There is no reference kernel measured periodically, so a poisoned corpus is indistinguishable
from a clean one.

**Fix (small, high value):** (a) set QoS to `QOS_CLASS_USER_INTERACTIVE` at bench-process
start; (b) benchmark one fixed sentinel kernel every N expressions, write its ns alongside each
batch, and hard-fail (loudly) if it drifts >10% from the run's opening calibration. The single
best paper-methodology addition.

### H5. Timer quantization at the measurement floor; the anti-noise path exists but is never used

`mach_absolute_time` granularity is one tick = 41.67ns (documented at `jit_bench.rs:182-185`).
With `repeat_batches = 1` (the only value any caller uses — `benchmark_jit_arena`,
`jit_bench.rs:391-393`), one timed sample is 100 evals, so a 1-2ns kernel yields a sample of
~3-7 ticks: per-eval resolution ~0.42ns, i.e. **20-40% quantization error on the smallest
kernels**, which are most of the tiny-band corpus (`gen_bench_corpus.rs:50-76`).
`benchmark_jit_arena_repeated` (`jit_bench.rs:395-408`) with `repeat_batches=100` was built
exactly for this ("validation-only heavier path") but no label-minting caller invokes it.

**Fix:** auto-scale `repeat_batches` until `median_total >= ~100 ticks` (≈4µs); a 5-line change
in `benchmark_exec_code`.

---

## MEDIUM severity

### M1. Constant call overhead is inside every measurement and never subtracted — log transform turns it into a floor the NNUE learns

Each eval includes fn-pointer call/ret + register setup (~1-2ns). Not modeled, not subtracted.
For 10-18-node kernels near the floor, cost differences between candidate extractions are
compressed under a shared additive constant, and `log_ns` (`jit_bench.rs:517-522`) makes the
distortion nonlinear precisely where the extractor makes most of its decisions ("everything
small costs log(overhead)"). **Fix:** measure an identity kernel (`ret x`) per run, subtract
its per-eval time (or record it as a feature/normalizer), and sentinel it per H4.

### M2. Non-random censoring of failures from training data

- `bootstrap_extraction_head.rs:129-132` and `:202-204`: `bench.ns <= 0.0` is dropped as
  `jit_failed`. But `validate_median` documents zero as **valid** for constant-folded
  expressions (`jit_bench.rs:224-225`) — the fastest expressions are censored from training, so
  the model never sees the floor. Also `Err(_)` at `:214` discards the error entirely
  (synthetic path prints nothing per-failure; corpus path prints only first 5, `:215-218`).
- `bench_jit_corpus.rs:68-75`: failures `continue` with a count.
- `training/episodes.rs:185-191, 207-221`: compile failures and equivalence failures return
  `None` — episodes where rewriting *breaks things* never enter the dataset, so the policy is
  never penalized for driving expressions into failure regions. The same censored-failure
  pathology the July post-mortem cited for killing the RL loop, surviving in miniature.

**Fix:** write every exclusion (name, reason, error string) to a structured sidecar JSONL;
assert exclusion-rate < threshold at end of run; for equivalence failures in episodes, consider
keeping them as maximal-cost labels.

### M3. 3-way gate: fixed sequential ordering, no interleaving, no policy repeats, ±5% decision band

`bench_extraction_3way.rs:326-399` (`evaluate_kernel`): per kernel the order is always no-swap
bench → heavy saturation (`:344-346`, CPU-intensive, perturbs thermal/frequency state) → static
bench (`:362`) → nnue bench (`:375`). No A/B/A/B interleaving, each policy measured once. The
gate verdict threshold is ±5% geomean (`:650-670`) — the same order of magnitude as drift on
small kernels under H4/H5. **Fix:** interleave the three policies round-robin with 3+ repeats
each, take per-policy medians, randomize order per repeat.

### M4. No lower plausibility bound (dependency-chain floor)

`validate_median` (`jit_bench.rs:226-237`) rejects only NaN/negative/>1s (`MAX_PLAUSIBLE_NS`,
`:140`). Nothing catches "faster than physically possible": a JIT bug emitting an early `ret`,
a mis-scaled timebase, or an unroll miscount would produce a sub-floor number that passes
validation and — if the output happens to match at the single check point (H1) — mints a label.
**Fix:** compute a per-expression lower bound (op count × min reciprocal throughput, or even
`0.05ns × node_count`) and reject measurements below it as harness bugs.

### M5. Variance is measured then thrown away

`jit_bench.rs:292-303` collects 20 samples, sorts, keeps `times[10]`, discards the spread.
`BenchResult` (`:240-245`) has no dispersion field; `bench_jit_corpus.rs:79-84` writes only
`timing_ns`; training weighs a jittery 1ns label equally with a rock-stable 500ns label.
**Fix:** add IQR (or MAD) to `BenchResult`, persist it in the corpus JSONL and samples, and
either inverse-variance-weight the regression loss or reject samples with IQR/median above a
bound.

### M6. CI regression check runs on shared GitHub runners with a ratcheting baseline

`.github/workflows/benchmark_regression.yaml:19,41` — `cargo bench --workspace` on
`ubuntu-latest` (shared, noisy, heterogeneous hardware across runs); `:59-60` alert at 115%
with `fail-on-alert: true`; `:64` baseline auto-advances on every push. Two failure modes:
(a) shared-runner noise regularly exceeds 15% → flaky red PRs train people to ignore the gate;
(b) the ratchet: a sequence of 10-14% regressions each re-baselines and accumulates unbounded
total regression with no alert. **Fix:** benchmark base and head **in the same job on the same
runner** and compare the ratio; keep a long-horizon fixed anchor (e.g. compare against the
30-day-old baseline too).

---

## LOW severity / hygiene

- **L1. Stale Finder-copy duplicates are live Cargo targets.**
  `pixelflow-pipeline/src/bin/bench_extraction_3way 2.rs`, `bench_jit_compile_cost 2.rs`, and
  `src/training/episodes 2.rs` are older versions (the 3-way copy lacks the AVX2 `run_scalar`
  variant — on an AVX2 host it would exercise a wrong-ABI path). `src/bin/*.rs` auto-discovery
  compiles them as separate binaries; running the stale gate binary by tab-completion accident
  reports stale numbers. Delete all three.
- **L2. Geomean can silently ingest `ln(0)`.** `bench_extraction_3way.rs:422-428`:
  `validate_median` allows `ns == 0.0`; one zero makes the geomean 0/-inf with no error. Assert
  `v > 0.0` before `ln`.
- **L3. Unescaped JSON.** `bench_jit_corpus.rs:79-84` writes `expression` text into JSONL via
  `format!` with no escaping — a quote/backslash in `arena_to_kernel_code` output corrupts the
  file silently. Use `serde_json`.
- **L4. "Validation" MAE is training data.** `bootstrap_extraction_head.rs:322-334` computes
  MAE on the *last 1000 training samples* and prints it as a quality signal; there is no
  held-out split anywhere in the binary. Add a real held-out split (dedup machinery already
  exists in `gen_bench_corpus.rs:262-308`).
- **L5. Warmup vs DVFS.** `WARMUP_ITERS = 64` (`jit_bench.rs:18`) is ~0.1-1µs of work — enough
  for icache/branch state, far too short to ramp frequency from idle; the first expressions of
  a cold corpus run are measured pre-ramp. The sentinel in H4 subsumes this (calibrate after a
  deliberate multi-ms burn-in).

---

## What the harness already does RIGHT (methodology-section material)

1. **rustc-level DCE/const-prop is genuinely closed.** The measured code is behind a runtime
   JIT fn pointer (`jit_bench.rs:277,317`) rustc cannot see through, and every call result is
   `black_box`ed (`:33-133`). Inputs not being re-black_boxed per iteration is a CPU-level
   issue (H3), not a compiler-elimination hole.
2. **Correct mach timebase handling**: `mach_timebase_info` conversion with the 125/3 tick
   documented and empirically verified, u128 intermediate against 50-day overflow
   (`jit_bench.rs:180-198`); `CLOCK_MONOTONIC_RAW` on Linux (`:200-210`), immune to NTP slew.
3. **Loop-counter bias eliminated by full unrolling**, with the additive-cost-model rationale
   written down (`:20-30`).
4. **Median-of-20 with garbage rejection**: `validate_median` refuses NaN/negative/absurd
   (`:226-237`); `log_ns` asserts on NaN and clamps (`:516-522`).
5. **Failures are excluded, not averaged in**: the 3-way bench records per-kernel failures and
   drops them from the geomean explicitly (`bench_extraction_3way.rs:18-22, 362-386, 601-603`),
   and a no-swap compile failure is a loud panic, not a skip (`:331-337`).
6. **A correctness gate exists at all** in both the gate bench and episode minting (loud
   `REWRITE BUG` dump, `episodes.rs:207-221`) — many published harnesses in this genre have
   none.
7. **Extraction overhead is measured separately** from kernel runtime (`extract_us_*`,
   `bench_extraction_3way.rs:349-360`), so policy compile-time cost never contaminates the
   kernel-speed comparison.
8. **Structural dedup of the corpus** (`gen_bench_corpus.rs:262-308, 348`) prevents
   duplicate-expression leakage; fixed seeds make corpus generation reproducible.
9. **CI hygiene**: PRs compare against history without mutating it; only postsubmit advances
   the baseline (`benchmark_regression.yaml:62-64`); raw bencher output is archived.
10. **Compile cost has its own harness** (101-sample median, warmup, fresh-vs-reused mmap
    paths, `jit_bench.rs:417-503`) rather than being folded into kernel timings.

## Top-3 concrete actions by leverage

1. **Sentinel + QoS pinning** (H4) — one reference kernel every N expressions, hard-fail on
   drift; ~50 lines, protects every label ever minted.
2. **Scalar-f64 interpreter reference + widened randomized check grid, tightened tolerance**
   (H1/H2) — closes the actual reward-hacking channel for the extractor.
3. **Auto-scale `repeat_batches` to a tick floor + record IQR + subtract identity-kernel
   overhead** (H5/M1/M5) — turns the small-kernel end of the label distribution from ±30%
   quantized noise into usable regression targets.
