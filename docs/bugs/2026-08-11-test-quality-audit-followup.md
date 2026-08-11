# Test quality control follow-up — 2026-08-11

Scope: scheduled continuation of
`docs/bugs/2026-08-08-test-quality-audit-followup.md`. 8 commits landed in
`f676b4b..HEAD` since that pass, but two of them (`876f5e0`, `7920159`) are
earlier audit passes that were authored before 2026-08-08 and merged late,
out of chronological order — already covered by
`docs/bugs/2026-07-29-test-quality-audit-followup.md` and
`docs/bugs/2026-07-31-test-quality-audit-followup.md` respectively, so not
re-reviewed here. The genuinely new material is two commits:

- `d049a07 feat(pixelflow): the terminal screen as ONE packed JIT kernel
  (3.05x steady-state) (#980)` — a large feature (41 files, ~5,564/-468
  lines): a new fixed-point `Dyadic` type (`pixelflow-ir/src/dyadic.rs`),
  buffer splicing/CSE in `ExprArena`, a packed-pixel `Gather`-based cell-grid
  kernel, and the e-graph/runtime plumbing to optimize it as one kernel.
- `507753a fix(fold): guard out-of-range shifts; correct the comment that
  hid them (#989)` — the out-of-range-shift-count fold/emitter fix analyzed
  in detail below.

## Static audit: two sweeps, both clean

**`d049a07`**: an `Explore` sub-agent swept every test file in the diff
against docs/STYLE.md's "Test Public API" rule and the descriptive-naming
convention. Result: clean. The ~1,373-line new `pixelflow-ir/tests/dyadic_spec.rs`
integration suite and the new/changed tests in `kernel.rs`, `arena.rs`,
`pixelflow-search/src/runtime.rs`, `pixelflow-graphics/src/render/{color,scene}.rs`,
and the two new `pixelflow-codegen`/`pixelflow-search` integration test files
all drive their crate's public API (`Dyadic::from_f32`/`checked_add`/`try_div`/...,
`Kernel::trunc_to_int`, `ExprArena::splice`, `optimize_runtime_arena`,
`Pixel::from_rgba`, `compile_cell_grid_for`) with no private-field or raw-lane
access anywhere. Names are specific `subject_predicate` sentences throughout
(`tie_at_top_of_binade_carries_into_the_next_exponent`,
`mul_declines_past_the_127_bit_mantissa_width`,
`distinct_buffer_identities_never_merge`). `pixelflow-ir/src/kind.rs`'s +20
lines in this commit are a single new `match` arm inside
`fold_is_platform_specific`, not a test change.

**`507753a`**: read directly (small diff). The three new tests in
`pixelflow-codegen/tests/transcendental_jit.rs`
(`out_of_range_shift_counts_are_platform_specific`,
`a_shift_count_that_aliases_to_zero_is_still_refused`,
`a_lane_crossing_shift_count_is_refused`, `a_zero_shift_count_compiles_as_the_identity`)
and the extended `invalid_trunc_to_int_is_platform_specific` construct kernels
through `ExprArena::push_var`/`push_const`/`push_binary` and
`pixelflow_codegen::emit::compile_arena_dag` — the sanctioned "construct the
AST, evaluate it" pattern, never a raw op. Names are specific about the exact
boundary each pins.

## Mutation testing

### `pixelflow-ir/src/kind.rs` — whole file, not just this delta

The 2026-08-07 and 2026-08-08 passes both flagged this file's pre-existing,
delta-untouched functions (`eval_binary`, `eval_ternary`, `from_name`,
`default_cost`, `is_commutative`, `is_associative`, `identity`, `annihilator`,
`is_idempotent`, `is_seed_op`, `is_bitwise_domain`) as a "large,
self-contained, deterministic-math candidate for a future pass" — this is
that pass. `kind.rs` had no `#[cfg(test)]` module of its own before
2026-08-02; everything exercising it lived in downstream crates.

`cargo-mutants` (`-p pixelflow-ir --file kind.rs --lib`): **101 → 20 missed**
(249 mutants total, 218 caught, 11 unviable). Added ~110 new tests across
eleven new `#[cfg(test)]` modules (`arity`, `op_names`, `default_cost`,
`algebraic_properties`, `domain_classification`, `eval_binary_arms`,
`eval_ternary_arms`, `round_platform_specific`, `min_max_platform_specific`),
each targeting a specific bucket of the function's contract rather than
re-deriving the production `match` (e.g. `eval_binary`'s `Lt`/`Le` test uses
`x == y` specifically because that's the one input where `<` and `<=`
disagree; `Min`/`Max`'s test uses `-0.0` vs `0.0` because bit pattern, not
`==`, is the only way to see x86's operand-order tie-break).

Two of the fixes are genuine gaps regardless of crate-scope, not just
invisible-to-this-run:

1. **`monoid_identity`'s `Min`/`BitOr`/`BitAnd` arms** had no test anywhere
   in the workspace — only `Add`/`Mul` (via the `dwrt`/`passes` reduction
   tests) were ever exercised as a `Reduce` combiner. Added
   `is_monoid_is_true_for_min_max_and_the_mask_monoids` and two
   value-pinning tests (`BitAnd`'s all-ones identity compared by bits, since
   it reads as NaN).
2. **`fold_is_platform_specific`'s `Min`/`Max` signed-zero branch**
   (`a == b && a.to_bits() != b.to_bits()`) was untested anywhere, including
   downstream. The one test that touches signed zero for `Min`/`Max`
   (`pixelflow-codegen/tests/transcendental_jit.rs`'s
   `min_max_nan_handling_agrees_between_tiers`) *gates* its own JIT-vs-oracle
   assertion on this function's result
   (`if op.fold_is_platform_specific(..) { continue }`) — that can only
   prove the function is self-consistent with whatever it returns, never
   that what it returns is correct. Added four direct tests in
   `min_max_platform_specific` covering the NaN case, the signed-zero case,
   and two false cases (equal-same-bits, ordinary distinct values).

The remaining 20 missed, checked individually against downstream tests:

- **Equivalent (2, unkillable by any input)**: `Round`'s `>` → `>=` at
  `212:84` — the only input where `x > -0.5` and `x >= -0.5` differ is
  `x == -0.5` exactly, and that's already forced true by the tie clause
  earlier in the same `||` chain, so the mutant produces identical output
  for every `f32`. `eval_ternary`'s `Select` `|` → `^` at `845:45` — `x.to_bits()
  & y.to_bits()` and `!x.to_bits() & z.to_bits()` are bitwise disjoint by
  construction (one is masked by `x`, the other by `!x`), so OR-of-disjoint
  equals XOR-of-disjoint for every input.
- **Covered, invisible to this crate-scoped run (18)**: the `TruncToInt`
  (`242`/`244`) and `Shl`/`Shr` (`265`) arms are directly asserted (not
  self-gated) by `transcendental_jit.rs`'s
  `invalid_trunc_to_int_is_platform_specific` and
  `out_of_range_shift_counts_are_platform_specific`. The whole-arm deletions
  for `Min|Max`/`Gt|Ge`/`Recip|Rsqrt` are caught by
  `platform_specific_ops_are_classified`'s direct NaN/estimate assertions
  (not the self-gated NaN-comparison test). `MulAdd`'s fold arm (`272`/`273`)
  is pinned by the same test's `fused_differs`/exact-agreement pair. Same
  methodology the 2026-08-08 pass used for `Round`/`MulAdd`: confirmed by
  reading the downstream assertions directly rather than re-running a
  cross-crate sweep.

`cargo test -p pixelflow-ir --lib`: 126/126. `cargo clippy -p pixelflow-ir
--lib --tests`: clean. `cargo fmt -p pixelflow-ir --check`: clean.

### `pixelflow-codegen/src/emit/{mod,aarch64}.rs` — scoped check of 507753a's new code

A whole-file `cargo-mutants` sweep of both files (5,776 + 2,168 lines —
the entire relocated JIT emitter, not just this delta's new code) ran for
38 minutes and found **1,040 missed of 1,489 mutants tested**. That is
overwhelmingly a pre-existing backlog unrelated to this delta — the
2026-08-08 pass's "next steps" already flagged `pixelflow-codegen/src/emit/*`
as "never been mutation-tested under their new crate location," and this
confirms it's a large one. Fixing it is out of scope for one pass; noted
below for future work, scoped tighter next time (this file pair is too big
to sweep whole — a future pass should target specific functions with `-F`).

**Process note**: this run and the `kind.rs` re-verification run were
launched concurrently and both wrote to the default `mutants.out/`
directory — `cargo-mutants` serializes them via `mutants.out/lock.json`
rather than erroring, but the second run's output overwrote the first's
result files (`missed.txt`/`caught.txt`) once it started, so the codegen
sweep's full missed-list could only be reconstructed from stdout, not the
result files. Future passes should always pass `-o <dir>` when running more
than one `cargo-mutants` invocation in the same session.

Scoped instead to exactly what `507753a` added, using `-F` to target the
new/changed functions directly rather than re-running the whole-file sweep:

- **`shift_immediate`** (`mod.rs:1241`, the arch-independent `Const` shift
  count narrowing this delta added) — both whole-function mutants
  (`replace with 0`, `replace with 1`) are caught. The assert's internal
  range/integrality condition
  (`(0.0..32.0).contains(&count) && (count as u32) as f32 == count`) has no
  mutants generated for it at all — `cargo-mutants` does not mutate
  expressions inside a macro invocation (`assert!`), a tool limitation, not
  a test gap. The three new `should_panic`/must-compile tests in
  `transcendental_jit.rs` (`a_shift_count_that_aliases_to_zero_is_still_refused`,
  `a_lane_crossing_shift_count_is_refused`,
  `a_zero_shift_count_compiles_as_the_identity`) directly exercise the 256.0
  alias case, the 32.0 boundary, and the 0.0 identity — everything the
  macro-mutation blind spot can't cover mechanically.
- **`emit_ushr`/`emit_shl`** (`aarch64.rs`) — both are gated
  `#[cfg(target_arch = "aarch64")]` at their call sites (verified via
  `grep target_arch pixelflow-codegen/src/emit/mod.rs`); this session runs
  on `x86_64`, so these functions' bodies — including the new range
  `assert!`s — are unreachable by any test executed here. The PR's own
  description says the aarch64 behavior was "measured on
  aarch64-apple-darwin by executing the emitted words," i.e. verified
  manually off-CI. This is the same category of gap `xtask isa-matrix`
  exists to manage for cross-target behavior (CLAUDE.md's "Debugging
  Pitfalls" section) — an environmental limitation of this machine, not
  something a new test written here could close.

## Verified

- `cargo test -p pixelflow-ir --lib`: 126/126, 0 failed.
- `cargo test --workspace --lib` unaffected by this pass's changes (only
  `pixelflow-ir/src/kind.rs` touched; no other crate's source changed).
- `cargo clippy -p pixelflow-ir --lib --tests`: clean.
- `cargo fmt -p pixelflow-ir --check`: clean.

## Recommended next steps (not done here)

1. `pixelflow-codegen/src/emit/mod.rs` + `emit/aarch64.rs`: 1,040 missed
   mutants, almost entirely pre-existing and unrelated to any single recent
   delta. Too large for a whole-file sweep in one pass — a future pass
   should pick specific functions (encoder helpers, schedule-building) with
   `cargo mutants -F <name>` rather than the whole file, and budget several
   passes to work through it incrementally.
2. `pixelflow-ir/src/kind.rs`'s remaining 20 missed mutants are fully
   accounted for (2 equivalent, 18 covered by direct downstream assertions)
   — no further local action needed, but if `pixelflow-codegen`'s
   `transcendental_jit.rs` ever loses those specific assertions, this file
   would silently lose coverage with no crate-local signal. Worth a comment
   pointing there if this file changes again.
3. No aarch64 execution environment is available in this session type, so
   `pixelflow-codegen/src/emit/aarch64.rs`'s target-gated functions
   (`emit_ushr`, `emit_shl`, and likely others) cannot be mutation-tested at
   all from here. Out of scope to fix, but worth knowing this file's
   mutation coverage numbers, whenever measured on an x86_64 host, will
   always undercount it.
