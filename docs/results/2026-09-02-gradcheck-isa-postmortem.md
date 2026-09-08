# The Guide gradient check is not an ISA divergence — it is a finite difference outside its own error budget

**Date:** 2026-09-02
**Question:** Two merged commits (09fd8758 = the bilinear Guide #1124, ddb51a91 = the bisect docs #1128) each failed postsubmit on `ISA matrix (SSE2/AVX2/AVX-512) -- build+lint+test`, both on `nnue::guide::scoring::backward::tests::numerical_gradient_check_candidate_tower_including_every_scalar_row`, both with `candidate_w1[5][7]: analytic 0.97299886 vs numeric 1.0299683` (5.7% off a 2% tolerance). Auto-revert PRs #1126/#1130 were opened. Is this the documented `MulAdd` one-vs-two-rounding divergence, a real gradient bug that aarch64 masks, or a step-size problem? And why did presubmit not catch it?
**Answer:** **A step-size problem, independent of ISA.** The analytic gradient is correct and reproduces to 1.3 ppm across every build tested; the *numeric* side moves by 12.5%. At `H = 1e-3` the check's roundoff floor is `ε·|score|/H = 2.80e-2`, which is **1.40× larger than `TOL = 2e-2`** — the assertion demands more precision than the estimator can deliver, on any target. It fails in **4 of 6** configurations measured here, including **SSE2 baseline**, the level CI reports green. avx2+fma is not the cause; it is the coin flip that landed tails. Presubmit could not have caught it because presubmit's required ISA-matrix job is `--build-only` by design.
**Reproduction:** all local numbers from `pixelflow-search` at `ddb51a91`, host Apple M2 Max, x86-64 rows translated by Rosetta 2 (which does execute AVX2+FMA). Per-config commands in §2.

## 1. Which level failed, and what did not

Both workflow runs (33706957934, 33709399553) are bit-identical and deterministic — same test, same entry, same two float literals. The matrix log separates the levels cleanly:

| level | build | clippy | test |
|---|---|---|---|
| `sse2 (baseline)` | ok | passed | **passed** |
| `avx2+fma` | ok | passed | **FAILED** — `candidate_w1[5][7]` |
| `avx512f+dq` | ok | passed | **never executed** — `host lacks avx512f` |

Three facts the log settles before any hypothesis:

- The postsubmit `other-crates` suites passed **5/5 on both ubuntu-latest and macos-latest**. Those run at the default SSE2/NEON baseline. Only the matrix's avx2+fma row failed.
- `avx512f+dq` has *never* been executed anywhere. `ubuntu-latest` has no AVX-512, so `xtask isa-matrix` builds and lints that level and reports `BuiltNotRun`. The matrix's third level is build-only in practice, presubmit and postsubmit alike.
- The `isa-matrix` job does **not** retry. The 5-iteration loop applies only to the `test` job; any matrix failure fails the workflow once and arms `automatic-revert.yaml`.

**Neither PR touched the failing file.** `pixelflow-search/src/nnue/guide/scoring/backward.rs` has exactly one commit in its history — 09fd8758 (#1124), which introduced the file and all six gradient checks. #1128 changed only `docs/results/*` and two `pixelflow-pipeline` files; it inherited an already-red `main`. **Auto-revert #1130 is collateral damage — reverting that docs commit cannot fix this test**, and any subsequent commit to `main` will fail identically and open another revert PR.

## 2. The measurements

Instrumented `score_of` at `w ± h` for the probed weights, across four local builds, plus the two CI values from the log. `H = 1e-3`, `TOL = 2e-2`, and `assert_close` uses `denom = numeric.abs().max(1.0)`.

Base score at the fixture: **|score| = 234.84**, `ulp(score) = 2.80e-5`. The finite difference being extracted is `~1.9e-3` — a **cancellation ratio of 121,000:1**, i.e. 69 ulps of signal after subtracting two numbers that agree to 5 significant digits.

`candidate_w1[5][7]` — the entry CI panicked on (analytic ≈ 0.973, so `denom` clamps to 1.0):

| configuration | analytic | numeric @ H=1e-3 | abs err | numeric @ h=1e-1 | abs err |
|---|---|---|---|---|---|
| CI ubuntu x86 `avx2+fma` O0 | 0.97299886 | 1.0299683 | **5.70e-2** | — | — |
| local x86 `sse2` O0 (Rosetta) | 0.97299784 | 0.9078979 | **6.51e-2** | 0.97366333 | 6.7e-4 |
| local x86 `avx2+fma` O0 (Rosetta) | 0.97299886 | 0.98419183 | 1.12e-2 | 0.97351074 | 5.1e-4 |
| local aarch64 O3 (FMLA) | 0.97299784 | 0.9918212 | 1.88e-2 | 0.97229004 | 7.1e-4 |
| local aarch64 O0 (dev) | 0.97299755 | 0.96893305 | 4.06e-3 | 0.973053 | 5.5e-5 |

The analytic column spans **1.3e-6** (1.3 ppm). The numeric column spans **0.122** (12.5%) — five orders of magnitude more scatter. At `h = 1e-1` the numeric column collapses to a spread of **1.4e-3**, and every row agrees with analytic.

The local `avx2+fma` analytic value is **0.97299886 — bit-identical to CI's**. Rosetta reproduces the CI codegen's gradient exactly while producing a *different* numeric estimate (0.98419 vs 1.02997, 4.7% apart). The instability is entirely in the estimator.

`candidate_w1[0][0]` — where three of the four local builds actually panic (analytic ≈ −3.1508, `denom` ≈ 3.1):

| configuration | numeric @ H=1e-3 | abs err | rel err | vs TOL=2e-2 |
|---|---|---|---|---|
| local x86 `avx2+fma` O0 | −3.0746458 | 7.62e-2 | 2.48e-2 | FAIL |
| local x86 `sse2` O0 | −3.0822752 | 6.85e-2 | 2.22e-2 | FAIL |
| local aarch64 O3 | −3.2196043 | 6.88e-2 | 2.14e-2 | FAIL |
| local aarch64 O0 | −3.0899045 | 6.09e-2 | 1.97e-2 | pass — **by 1.5% of the budget** |

Running the whole six-test gradient-check set per configuration:

| configuration | result |
|---|---|
| CI ubuntu x86 `sse2` O0 | 6 pass |
| CI ubuntu x86 `avx2+fma` O0 | **FAIL** (tower, `[5][7]`) |
| local aarch64 O0 (dev) | 6 pass |
| local aarch64 O3 (release) | **FAIL** (tower, `[0][0]`) |
| local x86 `sse2` O0 | **FAIL** (tower, `[0][0]`) |
| local x86 `avx2+fma` O0 | **FAIL** (tower, `[0][0]`) |

**Four of six fail, and SSE2 baseline is among them.** The level CI calls green fails on this machine with a *larger* error (6.5%) than the level CI calls red (5.7%). The only configuration that reliably passes is aarch64 dev — the one the test was authored on.

The other five gradient checks pass everywhere. Only the tower check probes `candidate_w1`, the first layer: its gradient path traverses the entire six-layer chain (36→64→64→32→16→32, then a 32×32 bilinear), accumulating the most rounding, and its gradient magnitude is small enough that `denom` clamps to 1.0 and the relative tolerance becomes an absolute one.

Commands (`RUSTFLAGS` must repeat `-C llvm-args=-fp-contract=fast`, since the env var *replaces* `.cargo/config.toml`'s `[build] rustflags` rather than merging):

```
RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  cargo test -p pixelflow-search --lib --target x86_64-apple-darwin numerical_gradient_check
RUSTFLAGS="-C llvm-args=-fp-contract=fast -C target-feature=+avx2,+fma" \
  cargo test -p pixelflow-search --lib --target x86_64-apple-darwin numerical_gradient_check
cargo test --release -p pixelflow-search --lib numerical_gradient_check
```

## 3. The error model, and why the check cannot pass

A central difference `(f(w+h) − f(w−h)) / 2h` in `f32` carries two error terms that trade against each other:

```
err(h)  ≈  C·ε·|f| / h        (roundoff: each evaluation is accurate to ~C·ε·|f|,
        +  (h²/6)·|f'''|       the subtraction cancels the signal, the division amplifies)
                               (truncation: the Taylor remainder)
```

with `ε = f32::EPSILON = 1.1921e-7` and `C` the accumulation factor over the summation chain. The optimum is `h* = (3·C·ε·|f| / |f'''|)^(1/3)`.

Substituting the fixture's measured values at the current `H`:

```
roundoff floor  =  ε·|score| / H  =  1.1921e-7 × 234.84 / 1e-3  =  2.80e-2
effective budget (denom = 1.0)    =  TOL                        =  2.00e-2

floor / budget = 1.40
```

**The check's noise floor is 1.40× its tolerance at `C = 1`, before any ISA question is asked.** No target can satisfy it; the passing configurations pass because the roundoff term happened to land with a favourable sign, not because they are more accurate. That is exactly the observed behaviour — the five measured `[5][7]` errors (4.1e-3, 1.1e-2, 1.9e-2, 5.7e-2, 6.5e-2) scatter around the predicted 2.80e-2 floor with `C ∈ [0.15, 2.3]`, and the `[0][0]` errors (6.1e-2 … 7.6e-2 against a 6.2e-2 budget) imply `C ≈ 2.5`, the plausible accumulation factor for six chained dot products.

The h-sweep confirms the model's shape directly. The error should fall as `h` grows until truncation takes over; instead it falls monotonically to the largest `h` probed, and explodes below `1e-3`:

| h | 1e-1 | 1e-2 | **1e-3** | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|---|
| rel err (aarch64 O0) | 0.00006 | 0.00738 | **0.00406** | 0.58860 | 0.74493 | 0.97300 |
| rel err (x86 sse2) | 0.00067 | 0.00356 | **0.06510** | 0.56023 | 0.78745 | 0.97300 |
| rel err (x86 avx2+fma) | 0.00051 | 0.00356 | **0.01119** | 0.66782 | 0.93623 | 0.97300 |

At `h = 1e-6` the difference underflows to **exactly zero** and the estimate is 0. `H = 1e-3` is already deep on the roundoff side of the V — there is no truncation-limited regime visible at all.

There is a structural reason for that, and it is the key to the fix: **the network is piecewise linear in this weight.** A ReLU MLP followed by a bilinear form is piecewise polynomial, and as a function of a single first-layer weight the score is affine within one activation pattern — so `f''' = 0` away from ReLU kinks, and the truncation term the step size is normally chosen to balance simply is not there. The measured truncation at `h = 1e-1` is ≤ 7.1e-4 across every build, i.e. no kink is crossed. Raising `H` from `1e-3` to `1e-1` therefore costs nothing and divides the roundoff floor by 100, to **2.80e-4** — a 71× margin under the existing `TOL`, and consistent with the 1.4e-3 cross-build spread actually observed there.

## 4. Verdict on the three hypotheses

**(a) The `MulAdd` one-vs-two-rounding divergence — NO.** The mechanism is real and repo-wide (`.cargo/config.toml` sets `-C llvm-args=-fp-contract=fast`, and at `--release` on aarch64 that yields 287 `fmla` vs 0 with `-fp-contract=off`), but it is not what failed here. Two disproofs: SSE2 baseline — which emits no FMA at all — fails on this machine with a *larger* error than avx2+fma; and at `opt-level=0`, which is what CI's `cargo test` actually builds, the emitted x86 object contains **0 `vfmadd` at both levels** (61 `mul[sp]s` either way). The divergence CI saw is not contraction; it is ordinary rounding scatter in a quantity that has no error budget. The CLAUDE.md doctrine still applies — "portable code should not depend on a single row's answer" — but the dependence here is on rounding in general, not on the `MulAdd` row.

**(b) A gradient bug aarch64 masks — NO.** The analytic gradient reproduces to 1.3 ppm across four builds and two architectures, matches CI's avx2+fma value bit-for-bit under Rosetta, and is the value the numeric estimate *converges to* from every configuration as `h` grows into the well-conditioned regime (spread 1.4e-3 at `h = 1e-1`). The backward pass is correct. The five non-tower gradient checks — including the transpose trap on `interaction` that the module doc is most worried about — pass in all six configurations.

**(c) A step-size / cancellation problem independent of ISA — YES.** §3. The tolerance is 1.40× inside the noise floor; the test asserts a quantity it cannot measure, and the pass/fail outcome is decided by rounding luck.

This is a house-rules matter, not a tuning matter. Widening `TOL` to accommodate the observed 6.5% would set the tolerance *from the noise* and leave the check unable to detect a real 5% gradient error — a check that passes for the wrong reason is the silent failure the rule forbids. The correction is to move `H` into the regime where the estimator is trustworthy, and to make the derivation checkable rather than commented: assert the floor `ε·|score|/H` is some stated factor below `TOL`, so a future fixture whose score magnitude grows fails loudly at the *model* rather than intermittently at one entry on one ISA.

## 5. Why presubmit did not catch it

Not a stale head, and not an unrequired check. Concretely:

- `ISA matrix (SSE2/AVX2/AVX-512)` **is** a required status check on `main` (with `strict: true`), alongside Rustfmt, Clippy, the two Test jobs, Cross-target ABI, Behavior contracts and CL metadata.
- It **ran on both PRs' final merged heads** and reported success: 913bfb2e at 02:06:34Z, 1aa96476 at 02:46:15Z. No earlier-head green was reused.
- The required presubmit job and the failing postsubmit job **are different jobs with different names**. Presubmit's `rust.yaml` runs `cargo run -p xtask -- isa-matrix --clippy --build-only`; postsubmit's `postsubmit-flake-detection.yaml` runs the same command **without** `--build-only`, and is named `... -- build+lint+test`.

So the root cause of the escape is a deliberate, documented design choice: **presubmit's ISA matrix compiles and lints every level and executes none of them.** `IsaExecutionMode::BuildOnly` returns `"build-only mode: tests run in postsubmit"` for every level unconditionally. The job comment states the rationale plainly — build+lint catches cfg mistakes, missing ops and lints fast, and "actually *running* the tests for each level happens in postsubmit ... which is what a merged, reverted-on-failure change can afford to wait on."

That trade was sound and this is the first time it was charged. It also has a second edge worth recording: `avx512f+dq` is never executed in *either* phase, because `ubuntu-latest` lacks the feature. A genuine AVX-512-only miscompile would reach `main` and stay there.

The presubmit `test` jobs did run the gradient check — once each on ubuntu and macOS (`cargo nextest run --workspace --all-features --profile ci`) — and it passed both times, at the baseline ISA. Postsubmit ran it 5× more per OS, also green. Repetition was never going to help: this failure is perfectly deterministic within a build and varies only *across* builds, so N iterations on one machine measure the same rounding N times.

## 6. Shift-left, ranked by cost

CI minutes measured from run 33706957934's own timestamps: build 102s + clippy 52s per level × 3 levels = **7.6 min for the build-only matrix**; test execution adds 158s (sse2) + 148s (avx2) = **5.1 min**; the full build+lint+test matrix is **12.7 min** against a 20-minute timeout.

| # | option | added CI min | would have caught this | leaves uncovered |
|---|---|---|---|---|
| 1 | Derive `H`/`TOL` from the error model in the test, and assert the floor is below the tolerance | **0** | yes, at authoring time | a real ISA-specific miscompile |
| 2 | Run the `nnue` gradient checks under `-C target-feature=+avx2,+fma` in a presubmit step | ~2 (parallel; ~0 wall) | yes, as a presubmit red | AVX-512; non-FMA rounding scatter (it passes at avx2 on 1 of 2 machines) |
| 3 | Drop `--build-only` from presubmit's matrix | ~5.1 | yes | AVX-512 (still unrunnable on `ubuntu-latest`) |
| 4 | x86-emulated smoke of the numeric-gradient tests | ~3–6 | yes | correctness of the emulation itself; Rosetta ≠ native rounding, as §2 shows |

**Recommendation: option 1, at zero CI minutes.**

The reason to prefer it over the others is not just cost. Options 2–4 would have turned this into a presubmit red — but a red reading `analytic 0.973 vs numeric 1.030` **at avx2+fma only**, whose natural and wrong response is to widen `TOL` until the one observed level passes. That would have shipped a check permanently blind to a 5% gradient error, and it would still fail on the next machine, because SSE2 baseline fails here too. Buying CI minutes to observe a symptom more promptly, while leaving the estimator 1.40× outside its budget, is paying for a faster route to the wrong fix.

Option 1 is also the only one that generalizes. All six checks in this file share one `H` and one `TOL`; the tower check is merely the one whose gradient path is longest and whose magnitude clamps `denom` to 1.0. A shared helper that computes `ε·|score|/H` from the fixture's own measured score and refuses to run when that floor is not a stated factor below `TOL` makes the next out-of-budget gradient check **impossible to write**, rather than merely faster to observe. That is the shift-left: the failure moves from "postsubmit, one ISA, revert cascade" to "fails at authoring, with the reason printed."

Option 2 remains worth doing on its own merits, independently of this bug — it is the cheapest way to give presubmit *any* view of a non-baseline ISA, and §5's AVX-512 gap argues the matrix's coverage claim is currently weaker than its name suggests. It should be filed separately rather than justified by this failure.

## 7. Disposition of the open reverts

- **#1130 (reverting ddb51a91 / #1128) should be closed, not merged.** That commit changed `docs/results/*` and two `pixelflow-pipeline` files; it did not touch the test, the head, or anything the test reads. Reverting it leaves `main` red and the next commit will open a third revert PR.
- **#1126 (reverting 09fd8758 / #1124) would make CI green**, because #1124 introduced the file and the test. But it would do so by deleting the six gradient checks that pin the bilinear head's backward pass — five of which are correct, load-bearing, and pass everywhere. Per JP's instruction the fix goes forward: correct `H` and the tolerance derivation in place, keep all six checks.

The forward fix touches only `pixelflow-search/src/nnue/guide/scoring/backward.rs`'s test module — no public API, no product code.

## 8. What landed, and where §3 was wrong

The forward fix is in `pixelflow-search/src/nnue/guide/scoring/backward.rs`'s
test module only. No product code, no public API, all six gradient checks kept.

**§3's recommended `H = 1e-1` is wrong, and the guard is what proved it.** The
h-sweep above probed only `candidate_w1[5][7]` and `[0][0]`; it inferred
"truncation ≤ 7.1e-4 at h = 1e-1, so no kink is crossed" from two entries and
generalized. The first run with the per-probe activation-pattern assertion in
place failed immediately on a third:

```
candidate_w1[scalar row 35][63]: the ±H probe crossed a ReLU kink
  (open-unit masks h1/h2/g = [fffc000001ffffff, 3fffffff80000, 7fff] at w,
                             [fffc000001ffffff, 1fffffff80000, 7fff] at w+H)
```

Scalar row 35 is `ln(1 + expr_node_count) ≈ 3.18`, the fixture's largest input,
so it moves the tower furthest per unit of weight; a trunk unit closes at
`h ≈ 0.1`. Measured, the fixture's narrowest activation cell is **8.2e-2**. At
`H = 1e-1` the check would have shipped estimating a chord across two affine
pieces — a *different* wrong answer than the one it replaced, and one the
h-sweep's own methodology could not see, because a sweep reports error at the
entries it samples and this fixture's binding entry was not among them.

`H` is therefore squeezed between two bounds, both asserted in the test rather
than argued in a comment:

| bound | from | value |
|---|---|---|
| lower | roundoff: `FLOOR_MARGIN · ε·|score| / TOL` | `2.2e-2` |
| upper | narrowest activation cell (measured) | `8.2e-2` |
| **landed** | `H` | **`4e-2`** — 1.8× above, 2.1× below |

- `the_step_size_must_keep_the_roundoff_floor_under_the_tolerance` recomputes
  the lower bound from the fixture's own score, so a fixture that grows fails
  at the model.
- `check_gradient` asserts on **every** probe that the ReLU pattern is identical
  at `w`, `w+H` and `w−H`, so the upper bound cannot be violated silently by an
  entry nobody sampled. This is the guard §6 recommended, in its strong form.

**A second instance of the same defect, found by the verification sweep.**
`activations_score_should_equal_score_candidate` asserted
`|acts.score() − direct| < 1e-5` absolute against a score of 234.8, where one
f32 ulp is **1.37e-5** — a tolerance *below one ulp* demands bit identity from
two separately written summations. It survived only because CI builds tests at
`opt-level = 0`; at `opt-level = 3` on aarch64 the two paths land one ulp apart
and it fails. Replaced with the derived bound `HIDDEN_DIM · ε · |score|`, the
accumulated rounding of the chain's longest reduction.

**Verified green** on every configuration that can execute: aarch64 at
opt-level 0 and 3 and at `-fp-contract=off`; `x86_64-apple-darwin` at the SSE2
baseline and at `+avx2,+fma`, each at opt-level 0 and 3. `+avx512f,+avx512dq`
builds and lints but SIGILLs under Rosetta — the same gap §5 records for
`ubuntu-latest`, unchanged by this fix.

**No training result is invalidated.** The backward pass was never wrong: §4(b)
holds, the analytic gradient is what the numeric estimate converges to from
every configuration once the step is inside its budget, and the five non-tower
checks passed throughout. #1124's numbers stand as reported.
