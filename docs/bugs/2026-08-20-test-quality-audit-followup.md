# Test quality control follow-up — 2026-08-20

Scope: scheduled continuation of
`docs/bugs/2026-08-15-test-quality-audit-followup.md`. Since that pass landed
(`ce2df0e`), `main` moved by six commits, ending at `9576ee1`.

## Delta audit: intervening commits' new/changed tests

Reviewed the test-touching commits since the 08-15 baseline
(`5436ed7` op-numbering rewrite, `be4f98d` e-graph/NNUE pipeline hardening,
`31b8f42` `pixelflow-search` domain-modeled cleanup, `9576ee1` the four-P1
denotation refactor) against STYLE.md's two testing rules (public API only,
"it should" naming).

`31b8f42`'s own PR description already did the diligence a delta audit would
redo: it diffed the full `--list` test-name set against its pre-branch
baseline and reconciled every one of the 15 disappeared / 18 appeared names
(relocations, legacy-code deletions, genuinely new coverage). `be4f98d` and
`5436ed7` both closed with a full `cargo test --workspace` green run and
described their new tests' intent in-PR. None of the four introduced a
`test1`/`it_works`-style name or a test reaching past a public API into
private fields — spot-checked `pixelflow-ir/src/kind.rs`'s new op-numbering
tests (`5436ed7`) and `pixelflow-pipeline/src/journal.rs`'s
`append_record`/`ConfigHash` tests (`31b8f42`, `be4f98d`): both go through
`pub fn` constructors and public accessors only. No fixes needed here.

## Mutation testing: `pixelflow-core/src/backend/x86.rs` (SSE2/`F32x4`/`U32x4`)

Picked up backlog item #5 from the 08-15 audit: the *required* `SimdOps`
methods in the per-ISA backend files, deliberately left untouched by that
pass's sweep of `backend/mod.rs`'s *provided* methods.

This container is `x86_64`, so only `arm.rs` is out of reach (it doesn't
compile here) and only the SSE2 region of `x86.rs` is unconditionally
compiled — the `Avx2`/`Avx512` `F32x8`/`F32x16`/`U32x8`/`U32x16` impls sit
behind `#[cfg(target_feature = "avx2"/"avx512f")]`, which is off by default
(`.cargo/config.toml` sets no `target-cpu`/`target-feature` flags). Scoped
the sweep to the SSE2 region with
`--exclude-re "F32x8|F32x16|U32x8|U32x16|Mask8|Mask16|Avx2|Avx512"` —
without it, `cargo-mutants` still generates mutants inside the cfg'd-out
blocks, and since that code isn't part of the compiled binary at all, every
one of those mutants is trivially unkillable noise (same trap the 08-15
audit hit for `fold_is_platform_specific`'s per-target scoping).

**First sweep: 80 mutants, 23 missed, 57 caught.** The 23 gaps split into
three shapes:

1. **`Debug` impls with nothing checking their output** — `Mask4`, `F32x4`,
   `U32x4`'s `fmt` all survived "replace with `Ok(Default::default())`".
   `U32x4`'s private `to_array` (only reachable through its own `Debug`
   impl at this scope, since its other call site — `pack_rgba` — builds the
   packed value a different way) rode along: two more mutants there.
2. **Functions with zero call sites in any existing test** — `add_masked`,
   `from_u32_bits`, `shr_u32`, `i32_to_f32`, `F32x4`'s `BitOr`/`Not`, and the
   *entire* `U32x4` surface (`SimdU32Ops::splat`/`store`/`from_f32_scaled`,
   `BitAnd`/`BitOr`/`Not`/`Shl`/`Shr`, `pack_rgba`) — the existing
   `x86_backend_tests.rs` only ever exercised `F32x4`'s arithmetic, compare,
   select, and a handful of the transcendental-support primitives; `U32x4`
   (the packed-RGBA-pixel path) had no direct test at all.
3. **A real gap hidden in `gather`'s clamp arithmetic**: `(idx as isize).clamp(0, len as isize - 1)` had no test exercising an out-of-range
   index, so the `- 1` could become `+ 1` or `/ 1` (i.e., `len` unclamped)
   without any test noticing.

### Fixed: 25 new tests + 8 renames in `pixelflow-core/tests/x86_backend_tests.rs`

New tests, one purpose each, values chosen to distinguish the real
implementation from its listed mutants rather than merely from zero:

- **`gather_clamps_an_out_of_range_index_to_the_last_slice_element`** uses a
  5-element slice and an out-of-range index (99.0). The correct clamp lands
  on index 4 (in bounds); both the `+`-mutant (clamps to 6) and the
  `/`-mutant (clamps to 5) index past the slice and panic — so the mutant
  is caught by the panic itself, not a value mismatch.
- **`pack_rgba_packs_four_float_channels_into_one_u32_per_pixel`** picks
  `b = 0.5` specifically so the truncating `f32 → i32` conversion
  (`0.5 * 255 = 127.5 → 127`) is visible in the expected literal
  (`0xff7f00ff`), rather than an input that happens to be an exact integer
  after scaling.
- **`mask4_debug_format_shows_which_lanes_are_true`** and
  **`u32x4_debug_format_lists_all_four_lane_values`**/
  **`f32x4_debug_format_lists_all_four_lane_values`** assert on the actual
  `format!("{:?}", …)` output — the `U32x4` one incidentally also kills the
  `to_array` mutants, since `Debug::fmt` is `to_array`'s only caller at this
  scope.
- **`from_u32_bits`/`shr_u32`/`i32_to_f32`** read their result back via
  `F32x4::store` (public) + `f32::to_bits` (`core`), rather than any
  private accessor — these are bit-manipulation helpers that return `Self`
  as a reinterpreted float, so recovering the raw bits to assert on needs
  nothing beyond the public API and the standard library.
- **`U32x4`'s bitwise/shift/`pack_rgba` tests** build on `SimdU32Ops::splat`
  + `store`, mirroring the existing `F32x4` test style (`lanes` helper) with
  a `u32_lanes` counterpart.

One gap was **not** fixed: `SimdU32Ops::from_f32_scaled` for `U32x4`
(`x86.rs:527`) is a documented placeholder —
`// Placeholder - actual packing is done via pack_rgba` — whose body is
already `Self::default()`. The "replace with `Default::default()`" mutant
is therefore byte-identical to the unmutated code; no test can distinguish
a function from itself. This isn't a coverage gap, it's the correct
mutation-testing outcome for a stub that hasn't been implemented yet — left
as-is rather than writing an assertion against `Default::default()` that
would look like a real regression test but isn't one.

Also renamed 8 pre-existing tests in the same file that predated both this
audit and the "it should" convention (`sse2_arithmetic`, `sse2_sequential`,
`sse2_logic`, `sse2_bitwise`, `sse2_math`, `sse2_mask_any_all`,
`sse2_store_panic`, `sse2_reciprocal_math`) to sentence-form names — done
in-line with the mutation-gap fixes since the file was already open and the
newer tests in the same file (`simd_exp_matches_the_scalar_exponential`,
etc., added by the 08-15 audit) already used the sentence form, leaving the
file internally inconsistent.

**Re-run: 80 mutants, 1 missed (the documented placeholder above), 79
caught.**

## Verified

- `cargo test -p pixelflow-core --test x86_backend_tests`: 36/36 passed (11
  pre-existing + 25 new).
- `cargo test -p pixelflow-core` (all targets incl. doctests): passed, 0
  failed (123 lib + 102 integration test-file passes, 44 doctests ignored
  as before — `pixelflow-core`'s doctests are all `ignore`d, consistent with
  every prior pass's note).
- `cargo test --workspace --lib`: 122 passed, 1 ignored, 0 failed (126s,
  dominated by `pixelflow-search`'s NNUE tests as in every prior pass).
- `cargo clippy -p pixelflow-core --tests`: clean.
- `cargo fmt -p pixelflow-core -- --check`: clean.
- `cargo mutants -p pixelflow-core --file pixelflow-core/src/backend/x86.rs -E "F32x8|F32x16|U32x8|U32x16|Mask8|Mask16|Avx2|Avx512"`:
  79/80 caught, 1 missed (documented placeholder, not fixed — see above).

## Recommended next steps (not done here)

1. `pixelflow-search/src/egraph/cost.rs` — still open per the 08-08/08-15
   audits: needs either a narrower test filter or a longer time budget than
   its `--lib` baseline (~110s) allows within one sweep.
2. `pixelflow-codegen/src/emit/*` (~1,400 lines) — still flagged as never
   mutation-tested under its post-crate-split location.
3. `pixelflow-graphics/src/spatial_bsp.rs` — still open: 19 tests reach into
   private `bsp.interiors[...]` fields with no public accessor. A design
   call (test-only introspection API vs. property tests over `eval()` vs.
   documented rule-break), not a mechanical fix.
4. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` — still present, not
   re-verified since 2026-07-20.
5. `pixelflow-core/src/backend/x86.rs`'s `Avx2`/`Avx512` regions
   (`F32x8`/`F32x16`/`U32x8`/`U32x16`) and all of `arm.rs` — never
   mutation-tested; needs a container with the right `target-feature`/
   `target-cpu` flags (x86_64 host, `+avx2`/`+avx512f`) or a cross/emulated
   aarch64 target respectively. This pass's SSE2-only scoping (backlog #5's
   other half) is now closed; these are the two ISA tiers it deliberately
   didn't reach.
