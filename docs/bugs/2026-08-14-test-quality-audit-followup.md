# Test quality control follow-up — 2026-08-14

Scope: scheduled continuation of
`docs/bugs/2026-08-08-test-quality-audit-followup.md`. `main` has not moved
since that pass landed as `e2c0c5f` (2026-08-11) — no new commits, so there is
no fresh delta to statically audit for STYLE.md/naming violations. Instead
this pass picked up the 08-08 audit's own recommended-next-steps item #2:

> `pixelflow-ir/src/kind.rs`'s pre-existing, delta-untouched functions
> (`eval_binary`, `eval_ternary`, `from_name`, `default_cost`,
> `is_commutative`, `is_associative`, `identity`, `annihilator`,
> `is_idempotent`, `is_seed_op`, `is_bitwise_domain`) have never been
> mutation-tested as a whole-file sweep and are a large, self-contained,
> deterministic-math candidate for a future pass explicitly scoped to
> "pre-existing backlog" rather than "since last audit."

## Also: codified the "it should" naming convention into docs/STYLE.md

`docs/STYLE.md`'s Testing section had one rule (test public API) and no
naming rule at all — the "name reads as a sentence once you prepend 'it
should'" convention has been applied by every one of these audits going back
to 2026-07-20, but only ever lived in the audits' own judgment, never in the
style guide itself. Added it as a second, explicit bullet with the two
examples this pass's own audit used to decide compliance (see below), so
future passes (and everyone else) have a written rule instead of tribal
knowledge.

## Mutation testing: `pixelflow-ir/src/kind.rs`, full-file sweep

`cargo-mutants` v27.1.0 (freshly installed — not present in this environment,
consistent with every prior pass), scoped to `pixelflow-ir/src/kind.rs`
against the crate's `--lib` test suite (107 tests, 0.12s — cheap enough for
repeated full sweeps rather than a single budget-limited run).

**Before: 249 mutants, 101 missed / 137 caught / 11 unviable (~3 min).** Every
one of the 08-08 audit's named pre-existing functions had gaps, plus three it
didn't name (`monoid_identity`, `is_monoid`, `arity`, `fold_is_platform_specific`
— the last already has cross-crate coverage via
`pixelflow-codegen/tests/transcendental_jit.rs`, invisible to a
`-p pixelflow-ir --lib`-scoped run, same situation the 08-08 audit already
documented for this exact function).

### Fixed: 11 functions, new tests added directly to `kind.rs`, `arena.rs`, `eval.rs`

All exhaustive over `OpKind::all()` where the predicate is a total, closed-set
function (so a single sweep catches both "flipped one op" and "always returns
the same answer" mistakes); spot-checked with representative/edge values
where it isn't (`default_cost`, `eval_binary`, `eval_ternary`,
`fold_is_platform_specific`).

- **`is_commutative`, `is_associative`, `identity`, `annihilator`,
  `is_idempotent`, `is_seed_op`, `is_bitwise_domain`** — each had zero direct
  coverage; a "replace the whole function with `true`/`false`/`None`" mutant
  survived for every one of them. Added one exhaustive test per predicate in
  a new `kind::algebraic_properties` module, each checking every `OpKind`
  against an independently-authored expected set (not copied from the
  `match` body — re-derived from what commutative/associative/idempotent/etc.
  actually mean for these ops).
- **`arity`** — added an exhaustive match mirroring the operand-count
  categories (0/1/2/3/4), which doubles as a compile-time check: it won't
  compile un-exhaustively if a new `OpKind` variant is added without deciding
  its arity.
- **`default_cost`** — not exhaustive (a price table is a tuning knob, not an
  algebraic law); spot-checked one op from each priced tier (leaf=0,
  unary=1, arithmetic=4/5, memory=10, transcendental=15, `Dwrt`=1,000,000)
  to catch a "replaced with a constant" mutant without freezing every literal
  against future re-tuning.
- **`from_name`** — the file's existing `every_returned_name_round_trips_through_from_name`
  only walks `known_method_names()`, which deliberately *excludes* every
  `EmitStyle::Special` op (`Var`/`Const`/`Tuple`/`Dwrt`/`Buffer`/`Gather`/
  `RawGather`/`Reduce`) — so those eight `from_name` match arms had zero
  coverage. Added a second round-trip test over `OpKind::all()` (unfiltered),
  plus direct tests for the `"powf"` alias and an unrecognized name.
- **`eval_binary`, `eval_ternary`** — direct value tests per arm (`kind::eval_binary_arms`,
  `kind::eval_ternary_arms`), including the NaN-unordered behavior of
  `Gt`/`Ge`, exact-vs-NaN `Eq`/`Ne`, `IAdd`'s wrapping bit-pattern add, and
  `Shl`/`Shr`'s `&31` count masking.
- **`fold_is_platform_specific`** — one test per declining case (`Min`/`Max`
  NaN and signed-zero, `Gt`/`Ge` NaN, `Round` ties and the negative-zero
  interval, `TruncToInt`'s two divergence points, `Shl`/`Shr`'s out-of-range/
  non-integral count, `MulAdd`'s fused-vs-split rounding, `Recip`/`Rsqrt`
  unconditional, and the `_ => false` default).
- **`monoid_identity` / `is_monoid`** (`pub(crate)`, no public wrapper) —
  rather than testing these directly (STYLE.md's public-API rule), tested
  through their one real entry point: `ExprArena::push_reduce`. Added
  `arena::tests::push_reduce_should_panic_when_the_combiner_is_not_a_monoid_op`
  (kills the `is_monoid -> true` mutant) and
  `eval::tests::fold_an_empty_reduce_domain_to_the_combiners_monoid_identity`
  (an extent-0 reduce skips `reduce()`'s loop entirely, so the result *is*
  `monoid_identity()` for every monoid combiner — kills the missing `BitAnd`
  arm, the only one none of the file's other reduce tests happened to
  exercise). Also added `push_reduce_should_panic_when_the_reduce_var_is_outside_4_to_8`,
  the same pattern for `push_reduce`'s other assertion, found to have the
  same zero-coverage shape while writing the first one.

**After: 249 mutants, 2 missed / 236 caught / 11 unviable.**

### The 2 remaining misses are equivalent mutants, not gaps

- **`kind.rs:212:84`, `Round`'s `*a > -0.5` → `*a >= -0.5`.** This clause is
  OR'd with a tie check (`(a - trunc(a)).abs() == 0.5`) earlier in the same
  expression. The only point where `>` and `>=` disagree is `a == -0.5`
  exactly — and at that value the tie check is already `true`, so the
  overall result is `true` under both the original and the mutant. No input
  exists that makes this comparator's direction observable.
- **`kind.rs:845:45`, `Select`'s `(x & y) | (!x & z)` → `(x & y) ^ (!x & z)`.**
  The two operands being combined are always bitwise-disjoint — one is
  masked by `x`, the other by `!x`, and a bit and its complement are never
  both set — so `|` and `^` compute the identical result for *every* `x`,
  `y`, `z`, not just canonical mask values. Provably equivalent; same
  category as the `07-20` audit's `Mask::all_false`/`Default` finding.

`cargo test -p pixelflow-ir --lib`: 107/107 (was 12/12 relevant subset
before this pass; 95 new tests added net). `cargo clippy -p pixelflow-ir
--lib --tests`: clean. `cargo fmt -p pixelflow-ir --check`: clean.

## Verified

- `cargo test -p pixelflow-ir --lib`: 107 passed, 0 failed.
- `cargo test --workspace --lib`: 117 passed, 1 ignored, 0 failed (105s,
  dominated by `pixelflow-search`'s NNUE training tests as in every prior
  pass).
- `cargo clippy -p pixelflow-ir --lib --tests`: clean.
- `cargo fmt -p pixelflow-ir -- --check`: clean.
- `cargo mutants -p pixelflow-ir --file pixelflow-ir/src/kind.rs -- --lib`:
  2/249 missed, both confirmed equivalent mutants (proofs above).

## Recommended next steps (not done here)

1. `pixelflow-search/src/egraph/cost.rs` — the 08-08 audit's partial mutants
   run (cut off by its own slow `--lib` baseline, ~110s) found one real gap
   (`CostModel::zero()`, already fixed) before timing out; likely has more.
   Still needs either a narrower test filter or a longer time budget than a
   single pass affords.
2. `pixelflow-codegen/src/emit/*` (~1,400 lines) and
   `pixelflow-core/src/backend/mod.rs` — flagged by 08-08 as never
   mutation-tested under their new post-crate-split location. Still true.
3. `pixelflow-graphics/src/spatial_bsp.rs` — confirmed still open: 19 tests
   (was ~20 at the 2026-07-20 audit) still reach into private
   `bsp.interiors[...]` fields with no public accessor. Still a design call
   (test-only introspection API vs. property tests over `eval()` vs.
   documented rule-break), not a mechanical fix.
4. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` (now ~line 1918) —
   confirmed still present; the 2026-07-20 audit's mutation findings against
   the private `backoff_with_jitter`/`send_with_backoff` functions it tests
   were not re-verified this pass.
5. `actor-scheduler/src/kubelet.rs`'s `ManagedPod` construction (flagged
   2026-07-20 as a public-API-rule violation) no longer appears in the file —
   looks resolved by an intervening commit, but not independently confirmed
   this pass; worth a one-line check before removing it from this list for
   good.
