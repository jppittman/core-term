# Test quality control follow-up — 2026-08-10

Scope: scheduled continuation of
`docs/bugs/2026-08-08-test-quality-audit-followup.md`. Since that pass
(commit `f676b4b`), the tree picked up 7 commits on `main`, dominated by
`d049a07 feat(pixelflow): the terminal screen as ONE packed JIT kernel
(3.05x steady-state)` (41 files, ~5,564/-468 lines — the largest delta
between audit passes to date, per that commit's own description). The other
six commits (`4c4c900`, `39d72e3`, `58885c5`, `7920159`, `c2cc426`, `876f5e0`)
are themselves prior scheduled test-quality/CI passes already covered by
their own write-ups; this pass's diff (`git diff f676b4b..d049a07`, then
`d049a07..HEAD` for the intervening audit commits) treats `d049a07` as the
only genuinely new surface.

## Static audit: two sub-agents split the diff, nothing to fix

Split the same way as prior passes: one pass on `pixelflow-ir/src/dyadic.rs`
+ `pixelflow-ir/tests/dyadic_spec.rs` (handled together with the mutation
pass below, since that pair was this delta's only wholly-new module), one
pass on everything else `d049a07` touched, against docs/STYLE.md's "Test
Public API" rule and this session's "reads as a complete 'it should ...'
sentence" naming convention, cross-checked against every prior pass's
accepted exceptions (`spatial_bsp.rs` interiors indexing,
`create_test_app()`/`new_registered`, `chunked_bake_matches_whole_stripe`,
actor-scheduler's two `try_send`-using backpressure tests, the
`pixelflow-codegen` module-private-encoder pattern) so those weren't
re-flagged.

**Verdict: no violations found.** Every file `d049a07` touched was reviewed
(full list in the pass's own report — `pixelflow-core/src/lattice/cell_grid.rs`,
`pixelflow-graphics/src/render/{color,pixel,scene}.rs`,
`pixelflow-ir/src/{arena,eval,kernel}.rs`, `pixelflow-search/src/runtime.rs`,
`pixelflow-search/tests/fold_exactness.rs`,
`pixelflow-codegen/tests/packed_pixel_shape_optimizes.rs`,
`core-term/src/terminal_app.rs`, and everything else in the 41-file diff).
One pre-existing carve-out reconfirmed rather than re-litigated:
`cell_grid.rs`'s tests call the module-private `channel_kernel`/
`packed_kernel`/`slot_roles`/`GridBuffers` — the same production functions
the real `compile()` path calls (verified by grep, not a test-only seam),
and there is no way to observe CSE/slot-sharing invariants through the
public surface without adding new introspection API, which CLAUDE.md
explicitly discourages ("Minimal public API"). Same category as the already
accepted `chunked_bake_matches_whole_stripe` precedent.

## Mutation testing: `dyadic.rs` (new module) and `extract.rs` (untested load-bearing logic)

Two targets, chosen for different reasons: `dyadic.rs` is `d049a07`'s only
wholly new module (388 lines, no prior mutation-testing baseline to speak
of); `extract.rs` was *not* touched by `d049a07` but turned up with
essentially zero coverage on its core method while auditing the surrounding
e-graph extraction machinery that commit leans on — worth closing now that
it's been found, rather than deferring another pass purely because it isn't
this delta's freshest code.

### `pixelflow-ir/src/dyadic.rs`

Exact dyadic-rational arithmetic (`mantissa × 2^exp`) backing the e-graph's
constant folder — see the module's own doc comment for why exactness matters
there. Tested by `pixelflow-ir/tests/dyadic_spec.rs` (1373 lines / 75 tests
going entirely through `Dyadic`'s public API, already following this file's
established idiom).

**Before: 178 mutants — 21 missed, 137 caught, 11 unviable, 9 timeouts.**

Hand-verified 15 of the 21 missed mutants as genuinely equivalent —
mutated code that produces identical externally-observable behavior given
`Dyadic`'s own invariants (private fields; mantissa always odd when nonzero;
`exp == 0` whenever `mantissa == 0`; magnitude always `< 2^127`). Each is
documented at its use site in `dyadic_spec.rs`'s new "Declining at the cap"
and Ord sections rather than chased with a contrived test. The equivalence
for `exact_isqrt`'s Newton-seed mutant (`333:31`, `-` → `+`) was additionally
checked empirically (2M random odd `u128`s plus an exhaustive sweep of
`3..200_000`, zero mismatches — the mutated seed is just a slower-converging
overestimate, not a wrong answer).

Found and fixed 3 genuine coverage gaps, all via new tests through the
existing public-API style:

1. **`checked_add`'s `> 127` mantissa-width cap** — i128 has 127 magnitude
   bits available (the function's own doc comment), but no test pinned the
   boundary exactly. Added `add_succeeds_when_the_aligned_sum_needs_exactly_127_magnitude_bits`
   (`pow2(0).checked_add(pow2(126))`, must be `Some`) and
   `add_declines_when_the_aligned_sum_needs_128_magnitude_bits`
   (`pow2(0).checked_add(pow2(127))`, must be `None`).
2. **`Ord::cmp`'s same-binade alignment-shift branch** — every existing
   ordering test only compared values across *different* binades (the fast
   path), never two values sharing one binade but different exponents (the
   branch that actually does the alignment shift and mantissa comparison).
   Added `ord_breaks_a_same_binade_tie_by_aligning_when_self_has_the_smaller_exponent`
   and `..._when_self_has_the_larger_exponent` using `d(2.25)` vs `d(3.0)`
   (same binade, distinct values), covering both branch arms.
3. **`to_f32`'s deep-underflow `shift >= 128` branch** — its tie-vs-zero
   check pins to exactly `shift == 128`; no existing underflow test landed
   precisely there. Added
   `underflow_pinned_at_the_exact_128_bit_shift_boundary_still_rounds_to_zero`
   using `pow2(-277)` (derived from `target_exp` pinning at
   `F32_MIN_SUBNORMAL_EXP = -149`: `-149 - (-277) == 128`).

**After: 178 mutants — 12 missed, 146 caught, 11 unviable, 9 timeouts.**

The remaining 12 are the equivalent/unreachable set (the 15 originally
identified minus the 3 the new tests turned out to also catch as a side
effect, plus 3 newly-confirmed-equivalent variants surfaced while pinning
gap 3 — `magnitude > 2^127` is unconditionally false given the type's own
mantissa-width invariant, so several comparison-operator mutants at that
site are permanently dead regardless of how the `shift == 128` clause is
computed).

No production code changed. `cargo test -p pixelflow-ir --test dyadic_spec`:
80/80 (was 75/75). `cargo clippy -p pixelflow-ir --lib --tests`: clean.
`cargo fmt -p pixelflow-ir -- --check`: clean.

### `pixelflow-search/src/egraph/extract.rs`

`IncrementalExtractor::extract_choices_only` is the crate's core NNUE-guided
incremental e-graph extraction method (`pub fn`, the sole production caller
is `extraction.rs:52-53`). `cargo-mutants` found it had **zero** test
coverage: all 12 mutants swapping its `(f32, Vec<Option<usize>>)` return
value for hardcoded garbage went unnoticed, and its private helpers
(`get_active_classes`, `backfill_reachable_defaults`, `choices_have_cycle_from`,
`break_choice_cycles`) were similarly untested — despite `break_choice_cycles`
guarding against a real, non-hypothetical failure mode its own doc comment
names explicitly ("class 68 picks neg(69), class 69 picks neg(68)").

**Before (partial run, hit the mutation harness's time budget): 42 missed,
3 caught, 1 timeout** for the region in scope.

Added 10 tests to the file's existing `#[cfg(test)] mod tests` block,
matching its established idiom (direct `EGraph`/`ENode` construction — these
are the module's own unit tests, not black-box integration tests):

- `extract_choices_only_backfills_every_reachable_class_when_equivalent_forms_are_unioned`
  and `..._backfills_a_class_reached_through_two_parents` — the backfill
  invariant holds under unioned equivalent forms (`x+x`/`x*2`, mirroring
  `extract_latency_prior_picks_cheaper_equivalent_form`'s shape but through
  the NNUE-guided extractor) and under a shared subexpression (`X*X`).
- `extract_choices_only_never_adopts_a_swap_that_would_create_a_mutual_reference_cycle`
  and `..._skips_a_directly_self_referential_candidate` — the refinement
  loop's cycle guard and self-reference filter, exercised across 4 NNUE
  seeds each (both are cost-independent rejections, so must hold regardless
  of the network's prediction).
- `get_active_classes_visits_a_shared_child_only_once` and
  `..._skips_a_recorded_choice_that_is_out_of_bounds_for_the_eclass` — direct
  calls, shared-child dedup and out-of-bounds-index safety.
- `backfill_reachable_defaults_fills_every_reachable_class_with_its_first_node`,
  `..._leaves_an_already_recorded_choice_untouched`, and
  `..._skips_a_class_id_outside_the_provided_choices_slice` — direct calls.
- `extract_breaks_a_mutual_reference_cycle_forced_by_an_inverted_cost_function`
  — a genuine DP-forced mutual cycle (a test-local `InvertedCost` where
  leaves cost more than ops, making the bottom-up DP prefer a
  self-referential-looking node over a real leaf), run through the public
  `extract()` entry point that calls `break_choice_cycles` internally;
  success is itself the strong signal (an unbroken cycle would make
  `choices_to_arena` recurse forever instead of producing the expected
  3-node acyclic arena).

**After (same-scope re-run): 4 missed, 30 caught, 8 timeout, 4 unviable.**
38 of the original 42 missed mutants resolved. The remaining 4 are
documented at the new test block's header rather than chased further:

- `143:34 < → <=` (best-improvement acceptance) — only observable on an
  exact float tie between two distinct subtrees' NNUE-predicted costs;
  forcing that would mean reverse-engineering MLP internals rather than
  testing extractor logic.
- `173:16 delete !` (fixpoint break) — only observable across two
  sequential, independently-NNUE-approved improvement passes; not
  controllable deterministically without hand-tuning network weights.
- `295:13 delete match arm 2` in `choices_have_cycle_from` (skip-already-black
  optimization) — correctness-preserving either way, only observable via
  exponential blowup on deep shared DAGs.
- `360:52 < → <=` in `break_choice_cycles` — a defensive bounds check whose
  divergence point (`node_idx == nodes.len()` exactly) the algorithm's own
  invariants never produce; its sibling `==`/`>` mutants at the same line
  ARE caught.

No production code changed. `cargo test -p pixelflow-search --lib
egraph::extract`: 24/24 (was 14/14). `cargo clippy -p pixelflow-search --lib
--tests --all-features`: clean. `cargo fmt -p pixelflow-search -- --check`:
clean.

## Verified

- `cargo test --workspace --lib`: all 12 crates pass, 0 failures, 1
  pre-existing ignored test (`pixelflow-search`, unrelated to this pass).
- `cargo clippy -p pixelflow-ir --lib --tests`: clean.
- `cargo clippy -p pixelflow-search --lib --tests --all-features`: clean.
- `cargo fmt --check` on both touched crates: clean.
- `cargo mutants -p pixelflow-ir -f pixelflow-ir/src/dyadic.rs`: re-run after
  the fix, 12 missed (all documented equivalent/unreachable above).
- `cargo mutants -p pixelflow-search -f pixelflow-search/src/egraph/extract.rs`:
  re-run after the fix (same time-budget scope as the initial partial run),
  4 missed (all documented above).

## Recommended next steps (not done here)

1. `pixelflow-search`'s test suite is still slow enough (~170s for
   `--lib` in this pass's full-workspace run) that a whole-file
   `cargo-mutants` sweep needs either a narrow test filter (as both targets
   in this pass used) or a much longer budget — carried over from
   2026-08-08.
2. `pixelflow-ir/src/kind.rs`'s pre-existing, delta-untouched functions
   (`eval_binary`, `eval_ternary`, `from_name`, `default_cost`,
   `is_commutative`, `is_associative`, `identity`, `annihilator`,
   `is_idempotent`, `is_seed_op`, `is_bitwise_domain`) — flagged 2026-08-08,
   still open, still a large self-contained deterministic-math candidate for
   a future pass explicitly scoped to "pre-existing backlog."
3. `pixelflow-search/src/egraph/cost.rs` — 2026-08-08 found one gap
   (`CostModel::zero()`) via a partial mutants run before timing out;
   likely has more past that handful, still open.
4. `pixelflow-search/src/egraph/extract.rs`'s `extract`/`extract_dag`
   entry points (the non-NNUE, static-cost-model extraction path) share
   `break_choice_cycles`/`choices_have_cycle_from` with the code this pass
   closed, but weren't independently re-swept beyond confirming the shared
   helper's mutants now catch — worth a quick confirmatory pass if `extract`/
   `extract_dag` grow independent logic later.
5. `pixelflow-core/src/lattice/cell_grid.rs` (+1012 lines this delta) and
   `pixelflow-search/src/runtime.rs` (+458 lines) are `d049a07`'s two largest
   genuinely-new logic surfaces after `dyadic.rs` and have never been
   mutation-tested under their current shape — good candidates for the next
   scheduled pass. `cell_grid.rs` in particular backs the terminal's
   production frame path.
