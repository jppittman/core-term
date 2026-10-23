# A4 fixes, round 1 (TAG=a4-fix1)

Base of this round: e31861b (verify-r1.md). Every item was checked against the code first.

## Gate failures

- **5 isa-matrix --smoke** (did not run: `cargo test --workspace --no-run` filled the disk). Not a code
  defect; the xtask command is workspace-wide by design and the ground rules forbid it here. Substitute
  stands (r1: 1266 + 13 + 20 passed per tier). Round 1 reruns the codegen tests on both tiers
  (below); nothing it exercises moved except the changes listed here.
- **8 Reduce-site checklist** (UNCLASSIFIED 1, STALE 1): real, bookkeeping. `corpus_gaps.rs`'s
  `hash_cons_keeps_a_value_bound` is formatted over several lines, so the hit is `ExprNode::Reduce {` at
  :956 and the row's one-line `matches!` text matched nothing. Fixed: the row in
  `.wip/a4/gen/classes.json` now has `"text": "ExprNode::Reduce {"` and `"lines": [956]`. Rerun:
  A4 0, A5 0, UNCLASSIFIED 0, STALE 0 (also after every other edit below).
- **commit and push**: done by this round (the disk had room again: 5.9 GB free before the builds).

## Blocking

- **ownership.rs `FoldReads` doc says a `Reduce` def is a leaf to `operands`**: real. `operands` yields
  `lo` and `hi` of a value-bounded fold since A4 (`program/mod.rs`, pinned by the lower.rs unit test).
  Fixed: the doc says a `Reduce` def reads its value bounds (none for a literal fold), never its body;
  the def is where the loop opens; the paragraph is reflowed. Doc only.
- **`emit::compile` / `jit_cache::compile` changed contract and are unrecorded**: real.
  `emit::compile`'s doc gains a paragraph that a value-bounded fold compiles inside its guard and
  computes its fold or NaN on every tier, and its `# Panics` section names the `BoundSort` refusal of a
  bound that computes a mask. `jit_cache::compile`'s doc gains one sentence (a value-bounded fold
  compiles, every count shares one program). I did not add a `BoundSort` sentence to the cache's
  `# Panics`: a computed bound cannot reach it (`Kernel::over_values` refuses one, `BoundArithmetic`;
  only a hand-built arena reaches the mask refusal, and only through `emit::compile`).
  `public-surface.md` has a "Changed behavior or contract" table for both under pixelflow-codegen.
- **Reduce-site checklist red**: same as gate 8 above. Fixed.
- **The mask rule is written twice (E1 asked for one predicate both sides call)**: real. AVX-512's
  `lane` matched `(KOp::of(op), a, c)` against `Lane::Opmask`, and `is_mask` restated it. Fixed by
  extracting the rule: `BinaryOp::on_masks<M>(self, Option<M>, Option<M>) -> Option<(MaskOp, M, M)>`
  is the one definition of "a bitwise op on two masks leaves a mask". AVX-512's `lane` calls it with
  each operand's opmask (`Lane::opmask`, new, private) and takes the `KLogic` from the answer
  (`impl From<MaskOp> for KOp` replaces `KOp::of`, so the `(Some, Opmask, Opmask)` triple and its
  repeated test are gone); `ScheduledOp::is_mask` calls it with the schedule's closure. What stays two
  statements is "a comparison is a mask, nothing else is", a fact about which file a result lands in;
  the docs now say so (the four false texts: `lower.rs` the assert comment and the unit test's doc,
  `avx512.rs` `Lane`'s doc, the plan's section 3.2 bullet) and the test holds them equal.
  Bytes: the selected instructions are the same; byte_probe both tiers are identical.
- **`nest_of` / `count_guards` docs describe values they do not return**: real. Now "the scoped
  schedule of `kernel` at `TWO_BATCHES` and `FOUR_LANES`" and "how many `Compare` and `If` defs
  `schedule` holds, in that order".
- **The agreement test's doc says "every comparison" and selects one**: real, and the test also
  listed nine `BinaryOp`s by hand. Fixed: `legal_ops!` generates `#[cfg(test)] pub(crate) const ALL`
  on each op enum, so a new op is in the list by construction. The test now selects every
  `UnaryOp`, `ShiftOp`, `CompareOp`, `BinaryOp`, `MulAdd` and `Blend` over a mask and a number in
  every operand position and asks `is_mask` (value 0 the mask), plus three anchors so the agreement
  is not vacuous. Mutation check: making `Ne` not a mask in `is_mask` fails the test
  (`Compare(Ne, ValueId(0), ValueId(0))`); reverted.

## Non-blocking, applied (defects in A4's own diff)

- **N1 test measures the selects, not the arm tests** (semantics and conformance, one fix): real.
  `ScopeTraffic::instructions` counts scheduled defs, so the guard's two `If` selects alone make
  `value > literal`; the arm branches are not counted. The test now asserts
  `value == literal + 2` (the two selects, one per mask, run in every batch of the column fold) and
  its doc and message say that. Holds on avx2 and avx512 here, and on NEON as the CI note reads it
  (7 against 5). The reload and frame-growth pins stay AVX-512 only. The arm skip itself is pinned
  by `a_count_outside_the_domain_is_nan_and_never_hangs`, as before.
- **`emit/mod.rs` says the arm is skipped "outside `[0, 2^24]`"**: real, a fraction is skipped too.
  Now "a bound is not a count in `[0, 2^24]` (a fraction, NaN and an infinity are none)".
- **`Trips::nested_in` labels a saturated product `Known`**: real. `memory_ops * known()` could then
  overflow. A product past `u64` is now `Estimated(u64::MAX)`, via `checked_mul`; `weight()` is the
  same, so the allocator's carry pricing and every byte are unchanged. `nesting_saturates` became
  `a_product_past_u64_is_an_estimate` (also asserts `known()` is `None`); `Trips`' doc and the
  plan's bullet say so.
- **`a_value_bound_is_in_the_program_digest` could not fail for the reason it names** (conformance
  and craft): real, the arenas differed in a leaf's payload. Both uniforms are now leaves of both
  arenas and only the `Reduce`'s child edge differs. Mutation check: hashing only a `Reduce`'s body
  fails the test with "the digest reads the bound's child"; reverted.
- **`every_count_is_one_program` runs outside the watchdog**: real. The four compiles and the four
  calls go through `within`.
- **`a_count_at_the_exact_bound_runs_every_trip` rebuilds `both`**: real. It calls
  `both(LatticeShape::POINT, build)`.
- **Plan §3.2 says the selector calls `is_mask`**: real, same as E1. Reworded to what exists.

## Skipped

None. (The finding that the arm skip is "pinned only indirectly" is a description, not a defect.)

## Verification (TAG=a4-fix1; outputs in .wip/a4/run/r2/ and .wip/gates/out/*.a4-fix1.*)

- Gates 1 (both tiers, 127 lines each), 1b (29), 2 (29), 2x (29): all exit 0, IDENTICAL to base.
  `git status --short pixelflow-graphics`: only `test_support.rs`, the intended edit (the probe file
  `tests.rs` is clean).
- `cargo test -p pixelflow-codegen`, PIXELFLOW_ISA=avx2 and =avx512: 200 passed, 0 failed each.
  `-p pixelflow-graphics` 239 passed, `-p pixelflow-core` 136, `-p pixelflow-pipeline` 228, 0 failed.
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`: exit 0, no warning.
  `cargo fmt --all -- --check`: clean. `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen
  --document-private-items`: ok.
- Scripts: check-intent, check-doc-paths, check-cfg-encapsulation, check-emit-boundary,
  check-bin-declarations: exit 0. Every intent.md within 80 lines (program/intent.md 70). Added lines
  hold no `.wip` path, ruling label or model name.

# A4 fixes, round 2 (TAG=a4-fix2)

Base of this round: 3916d7d. Gate failures: none. Every item was checked against the code first;
the two that name a runtime failure were reproduced before the fix (the new test fails on the
unfixed `local.rs`, passes after, on both tiers).

## Blocking

- **`plan_carries` overflows on a deep value-bounded nest** (semantics): real. Reproduced with a new
  test, `a_nest_priced_past_u64_still_compiles` (`tests/value_bounds_jit.rs`): at `POINT`, depth 12,
  13 and 14 of `Kernel::over_values` nests with n = 1, and one `FRAME` (1920x1080) nest, `both` paths
  (as written, optimized). Unfixed it panics "attempt to add with overflow" at `local.rs:430`
  (`own[..].iter().map(..).sum()`) on avx2 and avx512; `reloads + latch_stores` (`:437`) is the second
  unchecked add, reached first on avx2 at 1920x1080 by the reviewer's probe (not separately reproduced
  here; the frame case in the test covers both sites).
  Fixed: the reads' weights add with `usize::saturating_add` and `reloads.saturating_add(latch_stores)`.
  Where nothing overflows the result is today's `+`, and every kernel that compiled in debug has no
  overflow, so no byte moves (gates 1, 1b, 2, 2x identical; total Ir +0.09 percent, measure.sh). The
  depth is `PAST_U64`, derived from `Fold::ESTIMATED_TRIPS`, not a literal 13. `Trips::nested_in` and
  `Trips::weight` docs now say a weight is a price to add up and the allocator's sums saturate; the
  plan's `Trips` bullet says the same.
- **DENOTATION says "outside `[0, 2²⁴]`", which gives 2.5 a non-NaN meaning** (semantics): real. The
  guard is `0 <= v <= 2^24 and floor(v) == v`; the interval wording was wrong in every place that
  copied it. Now: a bound is a count, an integer in `0..=2²⁴`, and a fold whose bound is none (a
  fraction, a negative, NaN, an infinity) is NaN. Fixed in `docs/DENOTATION.md` (§3, paragraph
  reflowed, and "the equation holds where every bound is a count"), `CLAUDE.md`, `pixelflow-ir/intent.md`
  (reflowed; still 80 lines), `passes.rs` (the table row and `legalize`'s doc), the ontology
  (`language.md`), `.claude/agents/algebraist.md` and `docs/BACKLOG.md` C8. Same defect, not on the
  list, found by a case-insensitive grep and fixed: `Kernel::over_values`'s rustdoc ("Outside
  `[0, 2²⁴]` ... a uniform set to `2.5`") and `pixelflow-core/README.md`. Left as they are, because
  they are right: the plan's L5 ("bounds denote integers in `[0, 2²⁴]`"), `emit/mod.rs`'s "not a
  count in `[0, 2²⁴]`" (round 1's wording), `select.rs`'s known range.
- **`jit_cache::compile`'s `# Panics` omits `BoundSort`; public-surface says it is unreachable**
  (emitter and conformance, one fix): real, and round 1's reason was wrong. `Kernel::from_parts`
  adopts any arena and `ExprArena::push_reduce` takes a computed comparison as a value bound, so a
  mask bound reaches `jit_cache::compile`. Fixed: the `# Panics` section names it, and says
  `Kernel::over_values` takes no such bound but `Kernel::from_parts` can carry one. New test
  `a_mask_bound_adopted_whole_is_refused_by_name_through_the_cache` (`should_panic(expected =
  "BoundSort")`) goes `Kernel::from_parts` then `jit_cache::compile`. The public-surface row and this
  file's round-1 reasoning ("a computed bound cannot reach it") are superseded by this entry.
- **Docs say every bound that computes a mask is refused** (conformance): real. The code refuses what
  the emitter holds as a mask (`ScheduledOp::is_mask`: a comparison, or a bitwise and/or of two such),
  which is ruling E1; a select between two masks is held in the vector file and its bits are read as a
  float. New test `a_select_between_masks_is_read_as_a_float_on_every_tier` pins it (u = 5 gives NaN,
  u = -1 the identity, both tiers). Reworded to the set the code refuses in `emit::compile`'s `#
  Panics` (plus a sentence in its body for the unrefused select), `arena_to_schedule`'s doc, the
  panic message ("the bound is a mask (a comparison, or a bitwise op on masks)"), the plan's two
  statements (the Done line and the lowering bullet, which also drops "AVX2 and NEON would read its
  bits as a float" as the reason for the refusal and states the select's behavior),
  `program/intent.md` and the public-surface row. The test
  `a_bound_is_refused_exactly_when_it_computes_a_mask` is renamed `..._when_the_emitter_holds_it_as_a_mask`
  (see the non-blocking `lowers_with_the_end` below); `program/intent.md` names it.
- **`glyph_phase_split.rs` quotes a CLAUDE.md sentence A4 deleted** (craft): real. The module doc now
  says what still holds (a literal bound is in the header bits, so in the key, so a different literal
  count is a different program; a production glyph's counts are literals until it moves them) and
  cites the plan for a count that is a value. "The language's premise" and the CLAUDE.md quote are gone.
- **`Trips::Estimated`'s variant doc is false for a nest of literals past `u64`** (craft): real, the
  enum's doc already said so. The variant doc now: "or the nest's product passes `u64` ... no count is
  claimed". The public-surface `Trips` row matches.
- **Three fields documented as an exact count** (craft): real. `Loop.trips` (`emit/mod.rs`),
  `Selector.loops` (`select.rs`) and the codegen ontology's `Loop::trips` now say "or an estimate's
  price where a fold's range is not known" (the ontology: "`Loop::trips` is a `Trips` ...").
- **`program/intent.md`'s Contract omits `MaskOp`** (craft): real. It is listed beside `BinaryOp` (`BinaryOp`
  and its `on_masks` and `MaskOp`); the file is 71 lines.
- **The avx512 agreement test's doc says it selects every op of every kind** (craft): real; `Gather`
  and `Broadcast` were never selected. Fixed by selecting them (over a mask and a number as the index,
  the base the entry's context pointer, `ScheduledOp::Gather`/`Broadcast` beside them), rather than by
  narrowing: both land in the vector file and `is_mask` says false, so the agreement holds. The doc
  now lists the kinds and says a constant, the lane iota and a uniform read no lane, so they have no
  operand to make a mask. The plan's sentence says "every op that reads a lane".

## Non-blocking, applied

- **`lowers_with_the_end` counts any panic as a refusal** (conformance): real, three of the four
  negative cases could not tell `BoundSort` from another panic. The helper now returns the panic's
  message (`refusal_of_the_end`), refused cases assert it names `BoundSort`
  (`is_refused_as_a_mask`), and the lowering cases assert `None`.
- **Ontology "Structural parameter" contradicts the rewritten entries** (craft): real. Scoped as the
  finding asks, and left to A5 otherwise: "every structural range is constant", and the "uniform
  iterations ... no where near the docket" quote now says they are a bound that is a value
  (`Kernel::over_values`), not a structural parameter.
- **`admits_a_value` restates `with_a_value`'s checks, and the doc analogy is inexact** (craft): the
  doc is fixed (the analogy to `KnownRange::admits` is gone; it names the test that holds the two
  equal). The restatement stays: that test, `new_panics_exactly_where_the_predicate_refuses`,
  enumerates every monoid, stride 1 and 2, and five bound shapes with the literal inside and past the
  domain, so a divergence fails it; and `KnownRange::new` spells its parts beside `admits` the same
  way. A refusal enum carrying three message shapes is more machinery than the copy, which "subtract
  before you add" argues against.
- **`every_backend_emits_a_fold_whose_bound_is_a_value` has two coupled magic numbers** (craft): real.
  `const WALL: u32 = 48` and `terms[WALL as usize - 1]`.
- **`over_values`' `# Panics` dropped the `ExprArena::close_over` link** (craft): real; restored,
  four lines rewrapped.
- **`assert_every_fold_was_read` has no summary line** (craft): real; added.

## Skipped

- **`features_of` rejects a product-past-`u64` kernel with `LITERAL_FOLDS`, whose text is wrong for
  it** (conformance, the second half of the `Trips::Estimated` finding): not changed. A collapse
  corpus kernel is a production glyph or a synthetic family at corpus extents; a product past `2^64`
  is not constructible there (the finding's own probe overflowed `plan_carries` first, now fixed), and
  the message still states the true precondition (every fold's range is known). The doc half is fixed
  above.

## Verification (TAG=a4-fix2; outputs in .wip/a4/run/r2/ and .wip/gates/out/*.a4-fix2.*)

- Gates 1 (both tiers, 127 lines each), 1b (29), 2 (29), 2x (29): all exit 0, IDENTICAL to base.
  `git status --short pixelflow-graphics`: empty. `measure.sh a4-fix2`: total Ir +0.09 percent,
  saturate -0.045, extraction -0.104 (limit +0.5).
- `cargo test -p pixelflow-codegen`, PIXELFLOW_ISA=avx2 and =avx512: 406 passed across both runs, 0 failed
  (`tests/value_bounds_jit.rs` 25 each). `-p pixelflow-ir` 396, `-p pixelflow-core` 136, `-p
  pixelflow-pipeline` 228 passed, 0 failed.
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`: exit 0. `cargo fmt --all
  -- --check`: clean. `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --document-private-items`:
  ok; rustdoc comparison against the a747770 bases: ir 38/38, search 28/28, codegen 0/0, NEW 0.
- Reduce-site checklist: UNCLASSIFIED 0, STALE 0. check-intent, check-doc-paths,
  check-cfg-encapsulation, check-emit-boundary, check-bin-declarations: exit 0. Every `intent.md`
  within 80 lines. Added lines hold no `.wip` path, ruling label or model name. LFS count 68.
