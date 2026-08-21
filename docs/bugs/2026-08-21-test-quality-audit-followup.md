# Test quality control follow-up — 2026-08-21

Scope: scheduled continuation of
`docs/bugs/2026-08-15-test-quality-audit-followup.md`. Since that pass landed
(`ce2df0e`), `main` moved by six commits, including two mutation-testing/style
passes already covered on their own dates (`addb79d`, `99713d0`) and four
unrelated refactors (`f001bac`, `5436ed7`, `be4f98d`, `31b8f42`, `9576ee1`).
None of those touched test files newly, so no delta audit was needed this
pass.

## Picked up: recommended next step #5 from the 08-15 audit

> `pixelflow-core/src/backend/x86.rs`/`arm.rs`'s *required* `SimdOps` methods
> ... never independently mutation-tested as a whole-file sweep; a natural
> next target now that the provided-method half of the same file's contract
> is closed.

`arm.rs` is `#[cfg(target_arch = "aarch64")]`-gated and this environment is
x86_64, so it's out of reach here (nothing to compile). Took `x86.rs`.

## STYLE.md naming sweep

`pixelflow-core/tests/x86_backend_tests.rs` had eight tests named after their
*subject* rather than their *behavior* — `sse2_arithmetic`, `sse2_logic`,
`sse2_math`, etc. — none of which read as an "it should" sentence. Renamed
all eight (`add_sub_mul_div_compute_elementwise_results`,
`cmp_and_select_choose_the_true_or_false_branch_per_lane`, ...). Also renamed
`backend/x86.rs`'s in-file `avx512_log2` to
`log2_matches_the_scalar_base_2_logarithm` for the same reason (mechanical
rename only — that test is `#[cfg(target_feature = "avx512f")]`-gated and
doesn't run in this environment's SSE2-only build, so it wasn't part of the
mutation sweep below).

No public-interface violations found: both files reach the backend only
through `SimdOps`/`SimdU32Ops`/`MaskOps`/`Backend` (all `pub`) and operator
traits on the `pub` `F32x4`/`U32x4`/`Mask4` types.

## Mutation testing: `pixelflow-core/src/backend/x86.rs`

`cargo-mutants` v27.1.0 (freshly installed, as in every prior pass).

**Scoping trap, hit and corrected within this pass:** the first sweep used
`-- --lib`, which — as the 08-08 and 08-15 audits already flagged once each
for other files — restricts execution to the crate's unit-test binary and
silently skips `tests/x86_backend_tests.rs` (a separate integration-test
target). That run reported **172/229 mutants missed**, including three
`log2` polynomial-coefficient sign flips (`log2_poly::C4`/`C2`/`C0`) that are
trivially caught by the *existing* `simd_ln_matches_the_scalar_natural_log`
test — confirmed by hand-computing the mutated polynomial in Python (a C4
sign flip alone moves `log2(10.0)` from `3.32` to `4.88`, blowing through the
test's `1e-2` tolerance by two orders of magnitude). That result was a false
negative from the missing test target, not a real gap; re-running without
`--lib` caught all three immediately.

**Second problem, diagnosed rather than worked around:** even without
`--lib`, the un-scoped sweep still reported 172/229 missed. `x86.rs` defines
three lane-width tiers gated behind `target_feature`: `F32x4`/`Mask4`/`U32x4`
(SSE2, unconditional), `F32x8`/`Mask8`/`U32x8` (`avx2`), and
`F32x16`/`Mask16`/`U32x16` (`avx512f`). This container's ambient build
(`rustc --print cfg`) only has `sse2`/`sse`/`fxsr` — no `avx2`, no
`avx512f` — so the AVX2 and AVX-512 impls are `#[cfg]`'d out of the compiled
crate entirely. `cargo-mutants` still lists and "runs" mutants against that
dead source text; since the mutated function doesn't exist in the binary
either way, the build succeeds unchanged and every test still passes —
reported as MISSED, but not a real coverage gap, just cfg'd-out code the
tool can't see is absent. 149 of the 172 "misses" were in that dead code (81
avx2-gated, 68 avx512f-gated) — confirmed by cross-referencing each missed
line number against the file's `#[cfg(target_feature = ...)]` gates.

Rather than accept that as unknowable, checked `/proc/cpuinfo`: this host's
CPU does support `avx2`, `avx512f`, and `fma`. **Not done this pass** —
re-running the sweep with `RUSTFLAGS="-C target-feature=+avx512f,+avx2,+fma"`
to get genuine coverage of those two tiers is a natural next step (see
below); this pass stayed scoped to what the *default* build compiles, since
that's what every other crate in this workspace is tested against by
default too.

**Rescoped to the SSE2 tier that's actually compiled**
(`--re 'F32x4|Mask4|U32x4|log2_poly|exp2_poly'`, 77 mutants): **23 real
misses**, all genuine gaps —

- **`Debug` impls** (`Mask4`, `F32x4`, `U32x4`) — zero tests asserted on
  their output; a mutant replacing the whole `fmt` body with `Ok(())` (empty
  output) survived every existing test.
- **`gather`'s clamp boundary** — `(idx as isize).clamp(0, len as isize -
  1)`; mutating `-` to `+` or `/` (giving `len + 1` or `len` as the upper
  bound) survived because no test exercised an out-of-range index.
- **`add_masked`** — no test at all; a "replace whole function with
  `Default::default()`" mutant (zeroing every lane) survived.
- **The raw bit-cast primitives** — `from_u32_bits`, `shr_u32`, `i32_to_f32`
  — no direct tests; only reachable indirectly through `log2`/`exp2`, whose
  tolerance was too loose to pin them individually.
- **`F32x4`'s `BitOr`/`Not` operators** — only `BitAnd` had a test.
- **All of `U32x4`** (the packed-RGBA-pixel lane) — `Debug`, `splat`,
  `store`, `BitAnd`/`BitOr`/`Not`, `Shl`/`Shr`, and `pack_rgba` had zero
  tests between them; every mutant on the type, including a "replace
  `pack_rgba` with `Default::default()`" mutant, survived.

### Fixed: 18 new tests in `pixelflow-core/tests/x86_backend_tests.rs`

One test per gap, through the public `SimdOps`/`SimdU32Ops` traits and
operator impls on `F32x4`/`U32x4` — no private-field access. Two choices
worth calling out:

- **`gather_clamps_an_out_of_range_index_to_the_last_valid_slot`** uses an
  index exactly one past the slice's end (`4.0` for a 4-element slice). The
  correct clamp (`len - 1` = 3) resolves it to the last element; either
  mutant (`len + 1` = 5, or `len` via the no-op `/1`) leaves the index
  unclamped, so the test's 4-element slice panics on out-of-bounds access
  instead of returning a value — the mutation is caught by a crash, not a
  wrong-value comparison.
- **`f32x4_not_flips_every_bit`** compares via `f32::to_bits()` rather than
  `assert_eq!` on the `f32` lanes directly: inverting most bit patterns
  produces a NaN (all-ones exponent), and `NaN != NaN` under `PartialEq`
  would make the assertion vacuously pass regardless of the actual bits.

**Left unfixed, and why:** `SimdU32Ops::from_f32_scaled` for `U32x4`
(`x86.rs:527`) is already `fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
Self::default() }` — a documented placeholder ("actual packing is done via
`pack_rgba`"). The "replace with `Default::default()`" mutant is
byte-for-byte identical to the unmutated function; no test can distinguish
them because there is no behavioral difference to distinguish. This is an
equivalent mutant, not a coverage gap.

`cargo test -p pixelflow-core --test x86_backend_tests`: 35/35 passed (17
pre-existing + 18 new).

**Re-running the SSE2-scoped sweep**
(`--re 'F32x4|Mask4|U32x4|log2_poly|exp2_poly'`, 77 mutants): **76/77
caught**, the sole survivor being the documented equivalent mutant above.

## Verified

- `cargo test -p pixelflow-core --test x86_backend_tests`: 35 passed, 0
  failed.
- `cargo test -p pixelflow-core` (all targets incl. doctests): passed, 0
  failed.
- `cargo clippy -p pixelflow-core --tests`: clean.
- `cargo fmt -p pixelflow-core -- --check`: clean.
- `cargo mutants -p pixelflow-core --file pixelflow-core/src/backend/x86.rs
  --re 'F32x4|Mask4|U32x4|log2_poly|exp2_poly'`: 76/77 caught, 1 missed
  (documented equivalent mutant, see above).

## Recommended next steps (not done here)

1. Re-run `x86.rs`'s mutation sweep under
   `RUSTFLAGS="-C target-feature=+avx512f,+avx2,+fma"` (this host's CPU
   supports all three per `/proc/cpuinfo`) to get genuine coverage of the
   `F32x8`/`Mask8`/`U32x8` (AVX2) and `F32x16`/`Mask16`/`U32x16` (AVX-512)
   tiers — the 149 "missed" mutants this pass set aside as cfg'd-out dead
   code under the default build. Budget for a materially longer sweep: this
   pass's SSE2-only 77-mutant sweep took ~9 minutes, and the un-scoped
   229-mutant sweep took ~26 minutes even with most of that code absent from
   the binary.
2. `pixelflow-search/src/egraph/cost.rs` — still open per the 08-08 audit
   (repeated in every follow-up since; not independently re-checked this
   pass).
3. `pixelflow-codegen/src/emit/*` (~1,400 lines) — still flagged as never
   mutation-tested under its post-crate-split location (not independently
   re-checked this pass).
4. `pixelflow-graphics/src/spatial_bsp.rs` — still open: 19 tests reach into
   private `bsp.interiors[...]` fields with no public accessor. A design
   call (test-only introspection API vs. property tests over `eval()` vs.
   documented rule-break), not a mechanical fix.
5. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` — still open per the
   2026-07-20 audit; not independently re-verified since.
