# Test quality control follow-up — 2026-09-01

Scope: scheduled continuation of
`docs/bugs/2026-08-26-test-quality-audit-followup.md`'s backlog item 4:
`pixelflow-core/src/backend/x86.rs`'s `F32x8`/`U32x8`/`Mask8` (AVX2) and
`F32x16`/`U32x16`/`Mask16` (AVX-512) `SimdOps`/`SimdU32Ops`/`MaskOps` impls
had never been tested at the unit level at all — a plain `cargo test` builds
the SSE2 baseline (`.cargo/config.toml` sets no `target-cpu`/`target-feature`),
so the `#[cfg(target_feature = "avx2"/"avx512f")]` code never even compiles
into the default test binary. Every prior pass flagged this and left it open
for lack of AVX-512 hardware; this sandbox has it
(`avx512f`/`bw`/`cd`/`dq`/`vl` present in `/proc/cpuinfo`).

Two open, unmerged, non-current draft PRs from earlier passes on this same
backlog series exist upstream (#1049, #1051, both `mergeable_state: behind`,
targeting `pixelflow-search/egraph`'s `cost.rs`/`graph.rs`, not this file) —
left untouched: they're on branches this session didn't create, and rebasing
someone else's stale branch is out of scope for a routine test-quality pass.
Flagging here so a future pass (or the branch owner) sees it; the repository
would benefit from someone deciding whether to land or close them.

## `pixelflow-core/src/backend/x86.rs`

Added two integration test files mirroring the existing SSE2 coverage in
`pixelflow-core/tests/x86_backend_tests.rs`:

- `pixelflow-core/tests/x86_avx2_backend_tests.rs` (45 tests, 8 lanes)
- `pixelflow-core/tests/x86_avx512_backend_tests.rs` (47 tests, 16 lanes)

Each file's whole module is gated on its own `target_feature`, so a normal
build compiles it down to nothing — consistent with the SSE2 file's existing
`target_arch` gate. Run with:

```
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p pixelflow-core
RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" cargo test -p pixelflow-core
```

or `cargo xtask isa-matrix` (which sets the same flags plus this repo's
`fp-contract=fast`) on capable hardware.

Also moved the sole pre-existing test in this file — `avx512_log2`, living in
`x86.rs`'s own `#[cfg(test)] mod tests` — into the new AVX-512 file (widened
to check every lane instead of just lane 0), removing the last internal test
module from `x86.rs` and putting all its coverage in the same
external-integration-test convention the SSE2 file already established.

### Gather semantics differ by tier — the SSE2 test doesn't port directly

`x86_backend_tests.rs`'s `simd_gather_clamps_an_out_of_range_index_to_the_last_valid_element`
test relies on SSE2's `gather` being a software fallback (no hardware gather
on SSE2) that clamps out-of-range indices to `len - 1`. AVX2's and AVX-512's
`gather` call real hardware gather instructions (`_mm256_i32gather_ps`,
`_mm512_i32gather_ps`) with **no clamping** — an out-of-range index is
undefined behavior, per the precondition comment already on `F32x16::gather`
("indices must be valid ... already ensured by `Texture::eval_raw`"). The new
tests use an in-bounds but scrambled/reversed index vector instead, strong
enough to catch a mutant that ignores per-lane indices (e.g. returns `self`
or an identity mapping) without touching a behavior that doesn't exist on
these tiers.

## Mutation testing: `cargo-mutants` v27.1.0

Not present in this environment (consistent with every prior pass) —
installed via `cargo install cargo-mutants --locked`.

### First sweep, AVX2, restricted to the new test binary

`cargo mutants -p pixelflow-core --file .../x86.rs -j 4 -- --test
x86_avx2_backend_tests`: **225 mutants, 151 missed, 74 caught.** Nearly all
the misses were spurious: restricting execution to only the new AVX2 test
binary meant the *existing* SSE2 tests never ran, so every SSE2-region mutant
was trivially "missed" despite being genuinely covered by
`x86_backend_tests.rs`.

**Methodology fix:** dropped the `-- --test` restriction and let `cargo
mutants` run the package's default test command, so every test binary that
actually compiles under the active `RUSTFLAGS` runs together. This also
surfaced a second cross-cfg subtlety worth recording: `-C
target-feature=+avx512f,+avx512dq` (this repo's AVX-512 ISA level, per
`xtask::ISA_LEVELS`) transitively implies `avx2` and `fma` in LLVM's feature
resolution — real AVX-512F silicon requires AVX2 — so an AVX-512-tier mutants
run *also* exercises the AVX2 section for real, and `#[cfg(not(target_feature
= "fma"))]` code (the SSE2 `mul_add` two-rounding fallback) is dead code
under **both** the `+avx2,+fma` and `+avx512f,+avx512dq` RUSTFLAGS this repo
uses — it only compiles under the true zero-flags baseline. A `cargo-mutants`
report on this file always includes some of these cross-cfg artifacts
(mutating source text without evaluating which `#[cfg(...)]` branch is
inactive under the *current* build); don't take a "missed" mutant on this
file at face value without checking whether its containing `#[cfg]` actually
compiles under the RUSTFLAGS the sweep used.

### Real gaps found and fixed (before the corrected re-run)

Two gaps were real and identical in shape across all three lane widths
(SSE2/AVX2/AVX-512), so fixed in all three test files together:

- **`Mask4`/`Mask8`/`Mask16`'s own `BitAnd`/`BitOr`/`Not`** had no direct
  test at all — every existing test only combined masks indirectly through
  `any()`/`all()`, which a wrong-but-nonzero mask still satisfies. Added one
  lanewise test per width pinning the exact bit pattern (via
  `mask_to_float` + `store` for Mask4/Mask8's movemask representation, the
  raw `__mmask16` value for Mask16).
- **`F32x4`/`F32x8`/`F32x16`'s `bitand` test** used `1.0 & 2.0 = 0.0`, which
  coincides with `Default::default()` (also all-zero) — the "replace bitand
  with `Default::default()`" mutant survived on all three widths. Added a
  second, non-coincidental pair (`3.0 & 2.0 = 2.0`) to each existing test.

Also documented `U32x4`/`U32x8`/`U32x16::from_f32_scaled`'s
`Default::default()` mutant in place as equivalent: the function body already
*is* `Self::default()` (an unimplemented packing placeholder), so there is no
other behavior for any test to distinguish it from.

### Re-run after the fixes, both tiers, unrestricted test scope

- AVX2 (`+avx2,+fma`): **225 mutants, 218 caught, 7 missed** — 3 are the
  documented `from_f32_scaled` equivalents, 4 are the `mul_add`
  `fma`-fallback cross-cfg artifact described above.
- AVX-512 (`+avx512f,+avx512dq`): **225 mutants, 218 caught, 7 missed** —
  same 7, same explanation (AVX2 region compiled-in via the implied `avx2`
  feature and fully caught this time; only the `fma`-gated fallback and the
  three equivalents remain).
- Confirmed the `mul_add` fallback claim directly: `cargo mutants -p
  pixelflow-core --file .../x86.rs --re 'F32x4.*mul_add' -j 4` under the
  **true zero-RUSTFLAGS baseline** (no target-feature override at all):
  **5 mutants, 5 caught, 0 missed.**

**Final state: 0 real gaps in the AVX2 or AVX-512 tiers.**

## Verified

- `cargo test -p pixelflow-core --test x86_backend_tests` (SSE2 baseline):
  45 passed, 0 failed.
- `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p pixelflow-core
  --test x86_avx2_backend_tests`: 45 passed, 0 failed.
- `RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" cargo test -p
  pixelflow-core --test x86_avx512_backend_tests`: 47 passed, 0 failed.
- `cargo test -p pixelflow-core --lib` (SSE2 baseline, unaffected by this
  change): 123 passed, 0 failed, 1 ignored.
- `cargo clippy -p pixelflow-core --tests -- -D warnings` under both the
  AVX2 and AVX-512 RUSTFLAGS: clean.
- `cargo fmt -p pixelflow-core -- --check`: clean.
- `cargo mutants -p pixelflow-core --file
  pixelflow-core/src/backend/x86.rs` under AVX2 and AVX-512 RUSTFLAGS: 0
  real gaps in either tier (see above for the accounting of every mutant
  still nominally "missed").

## Recommended next steps (not done here)

Backlog carried forward from 2026-08-26, minus the item closed above:

1. `pixelflow-search/src/egraph/cost.rs` — believed closed by the unmerged
   PR #1051 (2026-08-30 audit), but that branch is `mergeable_state: behind`
   and has sat as an open draft for two days. Worth checking whether it
   should be updated and landed, or superseded by a fresh pass, before
   assuming this item is done.
2. `pixelflow-search/src/egraph/graph.rs` — two STYLE.md-noncompliant test
   names identified by the unmerged PR #1049 (2026-08-28 audit), same
   `mergeable_state: behind` situation.
3. `pixelflow-codegen/src/emit/*` — `regalloc.rs` (#1050, `mergeable_state:
   dirty` — real conflict) and `x86_64.rs` (#1054, `mergeable_state:
   behind`) both have unmerged work from prior passes; `mod.rs` (4,378
   lines), `avx2.rs`, `avx512.rs`, `aarch64.rs`, `coverage.rs`, and
   `executable.rs` remain untouched by any pass.
4. `pixelflow-core/src/backend/arm.rs`'s NEON impls (the aarch64 half of
   this same backlog item) — still untestable from this session, which runs
   on x86_64. Needs an aarch64 host.
