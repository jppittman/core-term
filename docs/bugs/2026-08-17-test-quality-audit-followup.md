# Test quality control follow-up — 2026-08-17

Scope: scheduled continuation of
`docs/bugs/2026-08-15-test-quality-audit-followup.md`. Since that pass landed
(`ce2df0e8`), `main` moved by three commits: `addb79de` (`core-term`
descriptive-names + mutation pass, itself a prior run of this routine),
`5436ed73` (`pixelflow-ir`: rewrote `OpKind::index`/`from_index` to make the
op numbering a compile-time-checked table), and `be4f98df` (`pixelflow-pipeline`:
hardened the e-graph NNUE measurement pipeline — ~150 new tests across 26
files).

## Delta audit: tests added by `5436ed73` and `be4f98df`

Delegated to a sub-agent: diff each commit's test-function set, skim all
~230 new/renamed names for the "it should" sentence property, read the body
of anything that looked like a bare label, and confirm public-API-only
access for the two new cross-crate integration-test files.

**Public-API-only: clean.** Every new test lives in same-crate
`#[cfg(test)] mod tests` (the project's normal same-crate unit-testing
convention) or, for the two new integration-test files
(`pixelflow-codegen/tests/oracle_reference.rs`,
`pixelflow-search/tests/latency_prior_regression.rs`), goes exclusively
through `pub` items (`ExprArena`, `ExprId`, `OpKind`, `eval_scalar`,
`JitManifold`, `jit_cache::compile_cached`, `optimize_runtime_arena`). No
private-field or `pub(crate)` reach-through from outside a defining crate.

**Naming: 9 violations found and fixed**, all bare noun-phrase labels
(STYLE.md's `min_max`/`handles_edge_case` failure mode — no verb, doesn't
say what's being asserted):

| File | Old name | New name |
|---|---|---|
| `pixelflow-ir/src/eval.rs` | `mask_valued_root_detection` | `is_mask_valued_classifies_comparisons_bitwise_ops_select_and_trunc` |
| `pixelflow-ir/src/eval.rs` | `trunc_divergence_predicate_boundaries` | `trunc_input_is_divergent_beyond_the_i32_range_and_convergent_within_it` |
| `pixelflow-pipeline/src/training/split.rs` | `seed_range_helpers` | `seed_range_contains_count_and_overlaps_match_their_definitions` |
| `pixelflow-pipeline/src/jit_bench.rs` | `iqr_computation` | `iqr_is_order_independent_and_zero_for_constant_samples` |
| `pixelflow-pipeline/src/jit_bench.rs` | `sentinel_regime_change_detection` | `sentinel_drift_exceeded_trips_only_beyond_the_fifty_percent_band` |
| `pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` | `size_distribution_quartiles` | `size_distribution_of_reports_count_min_median_max_and_renders_them` |
| `pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` | `spearman_known_small_example` | `spearman_rho_matches_a_hand_computed_example` |
| `pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` | `average_ranks_no_ties` (borderline) | `average_ranks_assigns_distinct_ranks_when_there_are_no_ties` |
| `pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` | `average_ranks_with_ties` (borderline) | `average_ranks_splits_the_average_rank_across_tied_values` |

Renames only — no behavior change. Verified each renamed test still passes
under its new name, per-crate (`pixelflow-ir`'s via `--lib eval::`,
`pixelflow-pipeline`'s via `--lib training::split`/`--lib jit_bench::`, the
two `bin/bootstrap_extraction_head.rs` ones via `--bin
bootstrap_extraction_head`), and grepped the whole tree for the old names to
confirm nothing else referenced them. Landed in `037071da`.

All other ~215 new-test names (the bulk of `be4f98df`'s additions —
`oracle_reference.rs`, `bench_extraction_3way.rs`, `quarantine.rs`,
`mint.rs`, `episodes.rs`, `structural.rs`, `unified_backward.rs`,
`extract.rs`, etc.) already read as complete sentences.

## Mutation testing: `pixelflow-core/src/backend/x86.rs`

Picked up backlog item 5 from the 08-15 audit: the file's *required*
`SimdOps`/`SimdU32Ops`/`MaskOps` per-ISA primitives (F32x4/F32x8/F32x16,
U32x4/U32x8/U32x16, Mask4/Mask8/Mask16 — the SSE2/AVX2/AVX-512 tiers) had
never been swept as a whole file. The prior pass closed the *provided*
default-method half of the same trait family for `backend/mod.rs`; this is
the other half.

**First run made the exact mistake the 08-08 and 08-15 audits already
documented**: scoped to `-- --lib`, which hides
`pixelflow-core/tests/x86_backend_tests.rs` (an integration-test binary) from
the mutant test runner. 229 mutants, only 42 caught — the tool wasn't
running the tests that actually cover this file. **Re-run without `--lib`**:
229 mutants, **172 missed, 57 caught**. This host is `x86_64-unknown-linux-gnu`
with no AVX-512, so `arm.rs`'s NEON backend (recommended alongside x86.rs in
the 08-15 backlog) doesn't even compile here — dropped from scope, noted
below.

172 missed is too large to close in one pass (structurally, F32x8/AVX2 and
F32x16/AVX-512 duplicate F32x4/SSE2's shape three times over). Scoped this
pass to the **F32x4/U32x4/Mask4 tier** — the always-compiled SSE2 baseline,
independent of any `target-feature` flag, and the highest-value slice since
every other backend falls back to it.

### Fixed: 15 new tests in `pixelflow-core/tests/x86_backend_tests.rs`

Against the public `SimdOps`/`SimdU32Ops` traits on `F32x4`/`U32x4`, matching
the file's established convention. Each input was chosen to distinguish the
real formula from its mutants, not just from zero:

- **`gather`**: an out-of-range index (99) on a 4-element slice kills both
  the `len - 1` → `len + 1` and `len - 1` → `len / 1` mutants — either one
  clamps to an index at or past `slice.len()`, and the resulting
  out-of-bounds scalar load panics rather than silently returning the
  wrong-but-plausible value.
- **`from_u32_bits`/`shr_u32`/`i32_to_f32`**: chained off each other
  (`from_u32_bits(8).shr_u32(2)` should read back bit-pattern `2`) since bit
  reinterpretation is opaque to `f32` equality — asserted via `.to_bits()`,
  not `==`.
- **`add_masked`**: asserted both the true-mask (adds) and false-mask
  (no-op) cases — a `Default::default()` mutant would return `0.0` for both,
  the select-based miscompile pattern the codebase's Floating-point-at-the-
  edges section warns about generally.
- **`U32x4`'s Debug output** (`u32x4_debug_output_shows_each_lane_value`):
  `U32x4::to_array` is a private helper reachable only through `Debug::fmt`,
  so asserting the formatted string (`"U32x4([7, 7, 7, 7])"`) is the only
  public-API way to observe it — and incidentally kills the `Debug::fmt ->
  Ok(Default::default())` mutant too, since a mutated `fmt` writes nothing
  and the formatted string comes back empty rather than wrong-but-present.
- **`U32x4 Not`**: `_mm_set1_epi32(-1)` → `_mm_set1_epi32(1)` (the "delete
  `-`" mutant) would XOR with `1` instead of all-ones, leaving every lane at
  `1` instead of `u32::MAX` — asserted the full inverted bit pattern, not
  just non-equality to the input.
- **`pack_rgba`**: red-and-alpha-only input (`r=1.0, g=0, b=0, a=1.0`) packs
  to `0xFF0000FF`, which is non-zero in exactly the R and A byte lanes —
  distinguishes the real `r | (g<<8) | (b<<16) | (a<<24)` formula from a
  `Default::default()` collapse.

**Left as non-issues (3 remaining misses, confirmed by a targeted re-run
filtered to `F32x4|U32x4|Mask4`, 74/77 caught):**

1. `Debug for Mask4`/`Debug for F32x4` — formatters, not load-bearing,
   consistent with every prior audit's treatment of `Debug::fmt` mutants
   (the U32x4 case above was fixable cheaply because `to_array` had no
   other caller to hang a test on; these two don't have that same
   opportunity — `F32x4::to_array` is already covered indirectly via
   `gather`, and `Mask4` has no analogous private-helper angle).
2. `U32x4::from_f32_scaled -> Default::default()` — **genuinely
   equivalent**, not a coverage gap: the function's real body already *is*
   `Self::default()` (a documented placeholder — "actual packing is done via
   `pack_rgba`"), so the mutant produces byte-identical output to the
   original. No test can distinguish them because there is no behavioral
   difference to observe.

### Verified

- `cargo test -p pixelflow-core --test x86_backend_tests`: 33 passed (18
  pre-existing + 15 new), 0 failed.
- `cargo test -p pixelflow-core` (all targets incl. doctests): passed, 0
  failed.
- `cargo test --workspace --lib`: 1446 passed, 4 ignored, 0 failed (per-crate
  breakdown: 149/0/459/78/48/123/152/136/9/107/65/120, ~161s wall clock,
  dominated by `pixelflow-search`'s NNUE training tests as in every prior
  pass).
- `cargo clippy -p pixelflow-core --tests`: clean.
- `cargo fmt -p pixelflow-core -- --check`: clean.
- `cargo mutants -p pixelflow-core --file pixelflow-core/src/backend/x86.rs
  -F 'F32x4|U32x4|Mask4'` (post-fix, targeted re-run): 74/77 caught, 3
  missed (the non-issues above).

## Recommended next steps (not done here)

1. `pixelflow-core/src/backend/x86.rs`'s **F32x8 (AVX2)** and **F32x16
   (AVX-512)** tiers — same file, same structural gaps as F32x4 (roughly
   150 of the 172 original misses), just never reached this pass. AVX-512
   needs a host with that target-feature to even compile `F32x16`'s cfg'd-in
   code; AVX2 doesn't have that restriction and could go first.
2. `pixelflow-core/src/backend/arm.rs` (NEON) — still untested as a whole-file
   sweep; cannot be mutation-tested on this `x86_64` host at all (the file is
   `cfg(target_arch = "aarch64")`-gated and won't even build here). Needs an
   aarch64 runner.
3. `pixelflow-search/src/egraph/cost.rs` — still open per the 08-08 audit
   (partial run found and fixed one real gap before a slow `--lib` baseline
   timed out); not revisited this pass.
4. `pixelflow-codegen/src/emit/*` (~1,400 lines, mod.rs alone is 5,779) —
   still flagged as never mutation-tested under its post-crate-split
   location; large enough to warrant its own scoped pass rather than a
   whole-file sweep.
5. `pixelflow-graphics/src/spatial_bsp.rs` — confirmed still open per 08-15:
   19 tests reach into private `bsp.interiors[...]` fields with no public
   accessor. Still a design call (test-only introspection API vs. property
   tests over `eval()` vs. documented rule-break), not independently
   re-verified this pass.
6. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` — confirmed still
   present per every prior audit; mutation findings against the private
   `backoff_with_jitter`/`send_with_backoff` functions not re-verified this
   pass.
