# Test quality control follow-up — 2026-08-19

Scope: scheduled continuation of
`docs/bugs/2026-08-15-test-quality-audit-followup.md`. Since that pass
landed (`ce2df0e`), `main` moved by 5 commits to `9576ee1`:

- `addb79d` test(core-term): descriptive test names + mutation-testing gap
  fixes (#997)
- `5436ed7` fix(pixelflow-ir): write the op numbering down and let the
  compiler check it (#993)
- `be4f98d` fix(pixelflow-pipeline): harden the e-graph NNUE measurement
  pipeline (#984)
- `31b8f42` refactor(pixelflow-search): domain-modeled cleanup — encapsulate
  the guide behind its contract (#1015)
- `9576ee1` refactor: the four P1 defect classes die to denotations (J2-J4,
  J7-J9, J14, J15) (#1019)

## Delta audit: STYLE.md compliance for every new/changed test in the range

Reviewed all ~296 `#[test]` functions that are new or were modified across
`ce2df0e..9576ee1` (confirmed per-function via `git log -S`, to separate
genuinely new/changed tests from code that only moved during the two large
refactors), against STYLE.md's two testing rules: descriptive "it should"
names, and public-API-only test surfaces.

**8 violations found, 7 fixed:**

Descriptive-name fixes (all pre-existing behavior unchanged):

- `pixelflow-pipeline/src/jit_bench.rs`: `iqr_computation` →
  `iqr_should_be_order_independent_and_zero_for_a_flat_sample`
- `pixelflow-pipeline/src/schema.rs`: `hex_and_const_agree` →
  `fnv1a64_hex_and_fnv1a64_const_should_agree_on_the_same_bytes`
- `pixelflow-pipeline/src/training/split.rs`: `seed_range_helpers` (tested
  three unrelated behaviors under one non-descriptive name) →
  `seed_range_contains_count_and_overlap_should_match_their_definitions`
- `pixelflow-search/src/nnue/guide/scoring.rs`:
  `bilinear_score_computation` →
  `bilinear_score_should_match_the_manual_dot_product_and_stay_finite`;
  `verify_randomize_is_deterministic_and_finite` (redundant `verify_`
  prefix) → `randomize_should_be_deterministic_and_finite`;
  `encode_rule_from_arena_deterministic` (missing verb) →
  `encode_rule_from_arena_should_be_deterministic`

Public-API fix:

- `pixelflow-search/src/egraph/extract.rs` (4 sites): tests built a
  well-founded, acyclic `choices` vector by hand, then constructed
  `Extraction { egraph, root, choices }` via a private-field struct
  literal — directly contradicting the type's own doc comment, which names
  `from_dp`/`from_backfill` as the only ways to obtain one (this
  encapsulation is J2 from #1019, one of this delta's own commits).
  Replaced all 4 with `Extraction::from_backfill(&egraph, root, choices)`,
  behavior-identical here since the fixtures were already
  acyclic/complete. Two other struct-literal sites in the same file were
  left alone — they deliberately test `choices_to_arena`'s independent
  cycle guard (one explicitly comments that it's a deliberate bypass), a
  reasoned exception rather than a violation.

**1 reported, not fixed** (judgment call, no mechanical same-behavior fix
exists):

- `pixelflow-pipeline/src/training/structural.rs`, test
  `key_is_stable_under_dag_sharing`: reaches into `FenceKey`'s private
  tuple field and `QuotientNode`'s private `children` field to check two
  child indices are equal. `FenceKey` exposes no accessor for its node
  list (only `of()`, `PartialEq`/`Hash`/`Debug`), and the only public-API
  alternative — comparing `FenceKey::of` outputs across a shared vs.
  duplicated-subtree arena — would pin a different, weaker property than
  the one this test is actually checking. New file (this delta), no
  established precedent either way. Left for a maintainer call: add a
  `pub(crate)` accessor for the node-sharing structure, or keep this as a
  documented rule-break.

Verified: `cargo test -p pixelflow-search --lib` (122 passed, 1 ignored)
and `cargo test -p pixelflow-pipeline --lib` (133 passed), both 0 failed;
`cargo fmt`/`cargo clippy --tests` clean on both crates.

## Mutation testing: `pixelflow-core/src/backend/x86.rs`'s SSE2 impl

Picked up backlog item 5 from the 08-15 audit ("`x86.rs`/`arm.rs`'s
*required* `SimdOps` methods... never independently mutation-tested as a
whole-file sweep"). `arm.rs` isn't buildable on this x86_64 host, so this
pass scopes to `x86.rs`'s SSE2 lane types (`F32x4`, `Mask4`, `U32x4`) —
the ones the default `pixelflow-core` build actually compiles and the
existing `pixelflow-core/tests/x86_backend_tests.rs` already targets.

**First sweep** (`cargo mutants -p pixelflow-core --file
pixelflow-core/src/backend/x86.rs -F 'F32x4|Mask4|U32x4'`): **45/77
missed.** The file's existing tests only covered arithmetic operators,
`cmp_lt`/`cmp_gt`, `simd_sqrt`/`simd_abs`/`simd_min`, `float_to_mask`, and
the *provided*-method layer closed by the 08-15 audit. Every *other*
required primitive had zero direct coverage: `cmp_le`/`cmp_ge`/`cmp_eq`/
`cmp_ne`, `from_slice`, `gather`, `mul_add`, `add_masked`,
`mask_to_float`, `from_u32_bits`, `shr_u32`, `i32_to_f32`, the `F32x4`/
`Mask4`/`U32x4` bitwise operators (`BitAnd`/`BitOr`/`Not`), `U32x4`'s
`Shl`/`Shr`, `pack_rgba`, and every `Debug` impl in the file.

### Fixed: 17 new tests in `pixelflow-core/tests/x86_backend_tests.rs`

One test per gap (or per closely-related group, matching the file's
existing `sse2_bitwise`/`sse2_logic`-style grouping), against the public
`backend::x86` API — the same low-level testing surface the file already
uses, not a new exception to "no raw SIMD in public API": `F32x4`,
`Mask4`, `U32x4` and the `SimdOps`/`SimdU32Ops`/`MaskOps` traits are
already `pub` specifically so backend implementations can be tested this
way.

Two inputs were chosen deliberately to avoid a coincidental-zero trap
(a mutant that replaces a function body with `Default::default()` is
*undetectable* if the real answer at the chosen input also happens to be
all-zero):

- **Bitwise operator tests** use raw bit patterns via `from_u32_bits`
  (e.g. `0b1100 & 0b1010`) rather than float values — the file's original
  `sse2_bitwise` test computed `1.0_f32.to_bits() & 2.0_f32.to_bits() ==
  0`, which is exactly the same value `BitAnd::bitand`'s
  `Default::default()` mutant produces, so it never could have caught
  that mutant.
- **`float_to_mask`** was first tested only against an all-zero `F32x4`
  (`Mask4::default()` is also all-zero), so the mutant survived even
  after the new test file's first draft; fixed by reinterpreting an
  all-ones bit pattern instead, where the real result and the mutant
  diverge.

Other notable choices: `mul_add(2.0, 3.0, 4.0) = 10.0` distinguishes all
four arithmetic-operator mutants at that one call (`+`→`-`/`*` and
`*`→`+`/`/` all disagree with 10.0); `gather`'s out-of-bounds clamp is
tested with an index past the slice end, which also catches the
`len - 1` → `len + 1`/`len / 1` mutants at its clamp bound (both would
read past the slice and panic, which mutation testing counts as caught
same as a failed assertion).

`cargo test -p pixelflow-core --test x86_backend_tests`: 35/35 passed (18
pre-existing + 17 new). `cargo fmt -p pixelflow-core -- --check` and
`cargo clippy -p pixelflow-core --tests`: clean.

**Re-running the mutants sweep: 76/77 caught.**

### Not fixed: `U32x4::from_f32_scaled` — dead placeholder, not a test gap

The one remaining mutant (`from_f32_scaled -> Self` replaced with
`Default::default()`) is undetectable because the real implementation
already *is* `Self::default()`:

```rust
fn from_f32_scaled<F: SimdOps>(_f: F) -> Self {
    // Placeholder - actual packing is done via pack_rgba
    Self::default()
}
```

This is the same in all four backend implementations (`x86.rs`'s SSE2,
AVX2, AVX-512, and `arm.rs`'s NEON) — every one is an identical
placeholder, and `from_f32_scaled` has zero callers anywhere in the crate
(only its trait declaration and the four definition sites match). Writing
a test to "catch" this mutant would just be asserting that
`Default::default() == Default::default()`; it wouldn't pin any real
behavior. Flagging as a backlog item instead (see below) rather than
adding a test that can't mean anything.

## Verified

- `cargo test -p pixelflow-core --test x86_backend_tests`: 35 passed, 0
  failed.
- `cargo test -p pixelflow-core` (all targets incl. doctests): passed, 0
  failed.
- `cargo test -p pixelflow-search --lib`: 122 passed, 1 ignored, 0 failed.
- `cargo test -p pixelflow-pipeline --lib`: 133 passed, 0 failed.
- `cargo test --workspace --lib`: every crate's `test result: ok`, 0
  failed across the board (dominated by `pixelflow-search`'s NNUE
  training tests at ~131s, as in every prior pass).
- `cargo clippy -p pixelflow-core --tests`, `-p pixelflow-search --tests`,
  `-p pixelflow-pipeline --tests`: clean.
- `cargo fmt -p pixelflow-core`, `-p pixelflow-search`,
  `-p pixelflow-pipeline -- --check`: clean.
- `cargo mutants -p pixelflow-core --file
  pixelflow-core/src/backend/x86.rs -F 'F32x4|Mask4|U32x4'`: 76/77
  caught, 1 missed (dead placeholder, see above).

## Recommended next steps (not done here)

1. `SimdU32Ops::from_f32_scaled` — dead trait method, every implementation
   across every backend is an identical `Self::default()` placeholder
   with no live caller. Candidate for removal (frees `pack_rgba` as the
   trait's real f32→u32 packing story) or, if it's meant to be
   implemented eventually, an explicit `unimplemented!()` so a future
   caller fails loudly instead of silently getting zeros. Changing the
   trait's public surface needs the "Minimal public API" sign-off this
   file's header requires — not done as part of a test-quality pass.
2. `pixelflow-pipeline/src/training/structural.rs`'s `FenceKey`
   private-field access (this pass's one unfixed STYLE violation) — needs
   a maintainer call on adding an accessor vs. accepting the rule-break
   with a comment.
3. `pixelflow-core/src/backend/x86.rs`'s AVX2 (`F32x8`/`Mask8`) and
   AVX-512 (`F32x16`/`Mask16`) lane types — this pass only covered SSE2
   (`F32x4`/`Mask4`/`U32x4`), the tier the default host build actually
   compiles and tests. The AVX2/AVX-512 code in the same file is gated by
   `#[cfg(target_feature = "avx2"/"avx512f")]` and needs a host (or
   `RUSTFLAGS="-C target-feature=+avx2,+fma"` / `+avx512f,+avx512dq`) that
   has those features to build and mutation-test.
4. `arm.rs`'s NEON backend — still never independently mutation-tested;
   needs an aarch64 host, unavailable in this environment.
5. All prior audits' still-open items carry forward unchanged:
   `pixelflow-search/src/egraph/cost.rs` (needs a narrower test filter or
   longer time budget — 08-08 audit), `pixelflow-codegen/src/emit/*.rs`
   (never mutation-tested post-crate-split — 08-08 audit),
   `pixelflow-graphics/src/spatial_bsp.rs`'s private-field test access
   (design call — 08-08 audit), `actor-scheduler/src/lib.rs`'s
   `backoff_unit_tests` against private functions (not re-verified since
   2026-07-20).
