# Test quality control follow-up — 2026-08-08

Scope: scheduled continuation of
`docs/bugs/2026-08-01-test-quality-audit-followup.md`. Since that pass
(commit `eb03ee9`), the tree picked up 4 commits, the largest being
`refactor(pixelflow): split codegen out of the IR, and stop rendering when
nothing changed` (#974) — extracting a brand-new `pixelflow-codegen` crate
out of `pixelflow-ir/src/backend/*`, renumbering `OpKind`'s discriminants to
be dense, and introducing a new `OpMap<T>` total-map type to replace every
bare `[T; OpKind::COUNT]` in the workspace. 81 files changed, ~2,750/2,760
lines added/removed — mostly relocation, not new logic.

## Static audit: two `Explore` sub-agents swept `git diff eb03ee9..HEAD`

Split by area — core-term, and the `pixelflow-*` crates — against
docs/STYLE.md's "Test Public API" rule and the "it should ..." naming
convention, cross-checked against every prior audit's already-accepted
exceptions so those weren't re-flagged.

### core-term: nothing to fix

The four new tests in `core-term/src/term/tests.rs`
(`scrolled_back_idle_snapshot_is_clean`, `returning_to_live_screen_marks_dirty`,
`scrollback_growth_under_held_offset_marks_dirty`,
`cursor_only_movement_marks_its_rows_dirty`) prove new internal
dirty-tracking state entirely through the emulator's existing public
`interpret_input`/`scroll_viewport`/`get_render_snapshot` API and the public
`TerminalSnapshot`/`SnapshotLine` fields — never touching the new private
`last_snapshot_view`/`last_cursor_mark` state directly. Names are specific
and match the file's established `subject_predicate` convention (same shape
as pre-existing, previously-audited neighbors like
`mouse_press_starts_selection`). The one modified test
(`cuj_priority_control_before_management_before_data`) is a race-condition
fix (queue-then-drop instead of `sleep(100ms)`), no style concerns.

### pixelflow-*: 2 fixed

The crate-split itself was audited specifically for whether a test that
moved crates had its visibility assumptions broken (a `pub(crate)` item on
one side of the new crate boundary quietly needing to become `pub`, or a
`#[cfg(test)]`-only export appearing to paper over it). No case found: every
moved test module stayed in the same crate as the code it tests
(now `pixelflow-codegen`), so untouched `pub(crate)` items stayed reachable;
only `use crate::X` → `use pixelflow_ir::X` renames were needed for the
items that actually crossed the boundary, and all of those were already
`pub`.

1. **`pixelflow-pipeline/src/training/corpus.rs`** —
   `v1_corpus_is_refused_with_regeneration_hint` (new this delta) called the
   private `read_corpus_bytes(&data)` directly instead of the public
   `read_corpus(&Path)`, even though the same file's other five round-trip
   tests already write a temp file via `unique_tmp()` (also new this delta,
   added for per-process temp paths) and go through the public path.
   Rewrote to write the hand-built v1-header bytes to
   `unique_tmp("v1_refused")` and call `read_corpus(&tmp)`.
   `cargo test -p pixelflow-pipeline --lib training::corpus`: 9/9 after the
   fix.

2. **`pixelflow-pipeline/tests/judge_weights_load.rs`** — this delta bumped
   the NNUE weights-file magic from `TRID` to `TRIE` (the `OpKind`
   renumbering changed the format), and the runtime assertions/panic
   messages were updated to say TRIE — but the test function's own name
   (`judge_weights_round_trip_via_trid`), three module-doc references, and
   two more panic-message/comment references still said TRID. Misleading if
   this test ever fails and someone greps "TRID" to find why. Renamed to
   `judge_weights_round_trip_via_trie`, updated the module doc and remaining
   comments/messages, and renamed the temp-file prefix
   (`pf_judge_trid_` → `pf_judge_trie_`) to match.
   `cargo test -p pixelflow-pipeline --test judge_weights_load`: 1/1 after
   the fix.

Everything else in the diff — `pixelflow-search/src/egraph/cost.rs`'s three
new tests (all go through public `latency_prior_cycles()`/`CostModel`
methods/`OpKind::from_name`), `pixelflow-pipeline/src/training/factored.rs`'s
two new tests (public `parse_expr`), and the mechanical
import-path-only diffs across `pixelflow-codegen/tests/*`,
`pixelflow-core/tests/naked_scale.rs`, `pixelflow-compiler/tests/jit_parity.rs`,
`pixelflow-graphics/tests/kernel_glyph_*.rs` — drives its crate's genuine
public API. No `raw_mul`/`raw_select`/`SimdVec`/lane access anywhere in the
reviewed diff.

## Mutation testing: `OpMap<T>` and its consumers

`cargo-mutants` (v27.1.0, freshly installed — not present in this
environment, consistent with every prior pass) against the two files this
delta added genuinely new logic to, rather than the crate-split's relocated
code:

### `pixelflow-ir/src/kind.rs`

This file is mostly pre-existing (`eval_binary`, `eval_ternary`, `from_name`,
`is_commutative`, etc. — none touched by this delta), so a whole-file sweep
mixes in a large pre-existing backlog unrelated to this pass. Scoped the fix
to what the diff actually added or changed:

- **`OpMap<T>` (new type)** — `iter`, `as_slice`, `as_mut_slice`, `splat`,
  `Default` had no direct tests (only the struct's `Index`/`IndexMut` were
  exercised transitively via other crates). Added
  `iter_yields_every_op_paired_with_its_own_slot_in_index_order`,
  `as_slice_exposes_the_same_values_iter_yields_in_the_same_order`,
  `as_mut_slice_writes_are_visible_through_indexing_by_op`, and
  `default_matches_splat_of_the_value_type_default`.
- **`known_method_names` (new function)** — used only from
  `pixelflow-compiler::sema`, a different crate a `-p pixelflow-ir --lib`
  mutants run can't see, and no unit test exercised it directly. Added
  `every_returned_name_round_trips_through_from_name`,
  `excludes_every_op_whose_emit_style_is_special`, and
  `includes_an_ordinary_binary_op_by_its_method_name`.
- **`eval_unary`'s libm-swapped arms** (`Ceil`, `Round`, `IntToFloat`,
  `Recip` — this delta replaced `std` float methods with `libm` calls for
  `no_std` compatibility) had no direct assertions on their return values.
  Added `ceil_rounds_toward_positive_infinity`,
  `round_ties_to_even_matching_x86s_vroundps`,
  `int_to_float_reinterprets_bits_rather_than_converting_the_value`, and
  `recip_is_exact_reciprocal_not_an_estimate`.

Re-running mutants after the fix confirmed all of the above are now caught;
the file's remaining missed mutants are exclusively in functions this delta
did not touch (`eval_binary`, `eval_ternary`, `from_name`, `default_cost`,
`is_commutative`, `is_associative`, `identity`, `annihilator`,
`is_idempotent`, `is_seed_op`, `is_bitwise_domain`), plus
`fold_is_platform_specific` — which the delta *did* touch (swapped `Round`
and `MulAdd`'s bodies to `libm::truncf`/`libm::fmaf`) but which is already
exhaustively covered by `pixelflow-codegen/tests/transcendental_jit.rs`'s
`platform_specific_ops_are_classified` and
`round_agrees_between_tiers_away_from_ties` — invisible to a `-p pixelflow-ir
--lib`-scoped mutants run because that integration test lives in a
downstream crate. Confirmed by reading the test directly rather than
re-running the (much slower) cross-crate sweep.

`cargo test -p pixelflow-ir --lib kind::`: 12/12 (new tests only; full crate
suite below). `cargo clippy -p pixelflow-ir --lib --tests`: clean.

### `pixelflow-search/src/egraph/cost.rs`

322 changed lines (the `LATENCY_PRIOR_CYCLES` positional array became
`latency_prior_cycles() -> OpMap<usize>`, an exhaustive `match`). This
crate's `--lib` test suite is slow (110s baseline alone — NNUE training
tests dominate), so a full 240-mutant sweep at that rate would run for
hours; not attempted. Instead ran mutants until the first
timeout, which completed baseline + a handful of mutants and surfaced one
genuine gap before being cut off:

- **`CostModel::zero()`** (this delta's array→`OpMap::splat(0)` migration)
  had no test asserting its costs are actually all zero. The mutant
  `replace CostModel::zero -> Self with Default::default()` survived —
  `Default` calls `Self::new()` (the latency-prior table), a plausible
  real bug (`zero()` exists specifically so extraction tests can assume
  "cost is irrelevant here"; if it silently returned real cycle costs
  instead, every test built on that assumption would start depending on
  cycle counts without anyone noticing). Added
  `zero_prices_every_op_at_zero_cost`, asserting `model.cost(op) == 0` for
  every `OpKind::all()`.

  The same partial run also flagged `CostFunction::cost_by_kind`'s default
  trait-method body (`panic!("not implemented")`) as missed — but
  `CostModel` is the only implementor of `CostFunction` anywhere in the
  workspace and overrides `cost_by_kind`, so that default body is
  pre-existing, genuinely unreachable dead code, untouched by this delta.
  Left as-is (out of scope, same as `kind.rs`'s pre-existing gaps above).

`cargo test -p pixelflow-search --lib egraph::cost::`: 4/4 after the fix
(was 3/3). `cargo clippy -p pixelflow-search --lib --tests --all-features`:
clean.

## Verified

- `cargo test --workspace --lib`: all crates pass, 0 failures.
- `cargo test -p pixelflow-pipeline --test judge_weights_load`: 1/1.
- `cargo clippy -p pixelflow-ir --lib --tests`: clean.
- `cargo clippy -p pixelflow-pipeline --lib --tests --all-features`: clean.
- `cargo clippy -p pixelflow-search --lib --tests --all-features`: clean.
- `cargo fmt --check` on every touched crate: clean (after `cargo fmt`).

## Recommended next steps (not done here)

1. `pixelflow-search`'s test suite is slow enough (110s+ baseline for
   `--lib`) that mutation-testing any of its larger files end-to-end needs
   either a narrower test filter or a much longer time budget than a single
   scheduled pass affords. `cost.rs` itself likely has more gaps past the
   handful the partial run surfaced before timing out.
2. `pixelflow-ir/src/kind.rs`'s pre-existing, delta-untouched functions
   (`eval_binary`, `eval_ternary`, `from_name`, `default_cost`,
   `is_commutative`, `is_associative`, `identity`, `annihilator`,
   `is_idempotent`, `is_seed_op`, `is_bitwise_domain`) have never been
   mutation-tested as a whole-file sweep and are a large, self-contained,
   deterministic-math candidate for a future pass explicitly scoped to
   "pre-existing backlog" rather than "since last audit."
3. The other large surfaces from this delta's crate split —
   `pixelflow-codegen/src/emit/*` (the relocated JIT emitter, ~1,400 lines)
   and `pixelflow-core/src/backend/mod.rs` (331 new lines, though mostly
   trait definitions with few provided-method bodies to mutate) — have never
   been mutation-tested under their new crate location.
