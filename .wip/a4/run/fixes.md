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

# A4 closing fixes, round 1 (TAG=a4-close1)

Base of this round: 4954553 (`.wip/a4/run/r3-open.json`: open_blocking 4, nonblocking 8). Every item was
checked against the code first; all twelve were real except the `.wip` spec item, which the task said
to ignore. Mutation evidence is in `.wip/a4/run/r5/mutation-run.log`.

## Blocking

- **1. A value `lo` outside the domain beside a literal `hi` is unpinned** (semantics): real. The arm
  `(Some(term), _) | (None, Some(term)) => in_domain(arena, term)` (`passes/guard.rs`) had no test with a
  lone value `lo`; every out-of-domain JIT test varied `hi`. New tests:
  - `value_bounds_jit.rs`: `a_value_lo_outside_the_domain_is_nan_and_never_hangs`
    (`Σ_{k∈[lo, 9)} (X + k)` through `both`/`within`, so as written and optimized, with the 60 s
    watchdog) and `a_pair_with_one_value_outside_the_domain_is_nan` (two values, `[out, 9)` with
    `lo = count(out)`, `hi = count(9)`, and the mirror `[3, out)`). Both run over one shared const,
    `OUTSIDE`: `2^24 + 2`, NaN, 2.5, -1, +inf, -inf, 1e30 (the finding's six plus the 1e30 the existing
    `hi` test already had; that test now reads the same const, so the three share one list).
  - `guard.rs`: `a_value_lo_beside_a_literal_hi_has_the_domain_test_of_lo`, beside
    `a_literal_bound_adds_no_domain_test`: `[u, Lit(8))` has a `dom` that reaches `u` (two `Le`, one
    `Floor`, one `Eq`, two `BitAnd`), `runs` is `Lt(u, Const 8)`.
  - Mutation (the reviewer's, `r4/mutation.diff.txt`, applied in the tree and reverted byte for byte,
    `cmp` against a copy taken before): a lone value `lo` gets `Lit::Mask(true)` as its domain.
    Under it, the guard unit test FAILS (`guard.rs:272`) and
    `a_value_lo_outside_the_domain_is_nan_and_never_hangs` FAILS on avx2 and avx512
    (`[16777218, 9) (as written) at (1, 2): got 0 (0x00000000), want NaN (0x7fc00000)`). At head all
    three pass (11 guard tests; 27 `value_bounds_jit` tests per tier). The pair test passes under this
    mutation, as it must: two value bounds take the first arm, which the mutation leaves alone, so it
    pins that arm (a mutation making the pair test only `hi`, or only `lo`, is what it catches), not the
    lone-`lo` arm.
- **2. `a_bound_on_a_lattice_binder_fails_loudly`'s reason is false for `row`** (semantics): real
  (`passes/lattice.rs:66-72`: `(x0 + col)` and `Add(Uniform(y0), Var(row))` are lane-uniform). The test's
  doc now says L3's reason: a bound is fixed before the lattice is traversed; only the lane binder would
  vary a guard's mask within a batch, `row` and `col` between batches, refused all the same.
  `assert_off_the_lattice`'s doc is aligned (one value for the call; the lane binder within a batch, the
  other two between batches).
- **3. The totality doc is false since A4** (conformance): real (`every_count_is_one_program` pins one
  `Arc`-equal program for four counts; `Kernel::over_values` is `pub`). `:110-112` now says every face is
  fixed before the lattice is traversed, a fold's count is a literal or a value known per call (linked to
  the plan), and the other three faces stay static until B1 makes extents values; the table's column is
  "Fixed before the traversal" and its reduction-binder row says `|D|` is a literal or a value known per
  call. The plan's section 5 row for that range now credits A4 for `:110-112` and the reduction-binder
  row, B1 for the extents row.
- **4. The mask-bound docs give the AVX-512 panic as the reason for both ends** (craft): real. Selection
  reads `lo` as the loop's seed, a head parameter (`select.rs` `seeds`, `param_lane` is
  `param.typed::<Vector>()`), and reads `hi` only as the latch compare's operand, which AVX-512 widens
  (`Compare` calls `.vector(b)` on both). So the law is the sort, a mask names no count (`BoundSort`), at
  either end; the AVX-512 opmask panic is the tier divergence that made it urgent, and it is `lo`'s alone.
  Reworded to that, in: the plan (the lowering bullet and the Done line), `lower.rs` (`arena_to_schedule`'s
  `# Panics`, the comment at the `is_mask` assert, the test doc), `pipeline.rs` (`emit::compile`'s
  `# Panics`), `jit_cache.rs` (`compile`'s `# Panics`), `emit/avx512.rs` (the agreement test's doc),
  `program/mod.rs` (`ScheduledOp::is_mask`'s doc: "where no head parameter can be"), and
  `tests/value_bounds_jit.rs` (`a_bound_that_is_a_mask_is_refused_by_name`). CLAUDE.md and the emit module
  docs carry no such justification (checked by grep for `opmask`, `seed`, `BoundSort`, `names no count`
  across every file A4 touched), so there was nothing to change there. The unit test
  `a_bound_is_refused_exactly_when_the_emitter_holds_it_as_a_mask` now runs every case at both ends
  (`for at in [Lo, Hi]`), so it covers the end the seed argument is about; the lowering of `lo` is
  refused by name and the number cases lower at `lo` too.

## Non-blocking, applied

- **`BoundRefusal::BoundSort`'s doc is stale** (craft): real. It said a computed mask bound is admitted
  and only T6 closes it. Now: a mask literal is refused where a fold is built; a computed bound that is a
  mask is admitted there and refused by lowering with this variant once the emitter holds it as one
  (`ScheduledOp::is_mask`), on every tier; a select between two masks is read as a float. The same
  sentence in `ExprArena::push_reduce`'s doc said "admitted"; it now says "admitted here", and that
  lowering refuses one the emitter holds as a mask.
- **CLAUDE.md:180's domain sentence reads backwards** (craft): real ("a bound that is no count, an
  integer in `0..=2²⁴`, makes the fold NaN" defines the negation). Now: "a count is an integer in
  `0..=2²⁴`, and a fold with a bound that is not one is NaN". The same construction in
  `pixelflow-core/README.md` and `.claude/agents/algebraist.md` is fixed alike; no other copy exists
  (grep).
- **`emit/mod.rs` loop-nest paragraph** (craft): real. Where the guarded fold reads a coordinate the
  guard's arm tests are the column fold's and run in every batch (an opmask is reloaded in each); where
  it reads none the guard and the loop are call-invariant and run once per call, in the body
  (`a_fold_that_reads_no_coordinate_is_guarded_once_per_call` pins it). "Batches of the fold they guard"
  is gone: the guarded fold has trips.
- **`passes.rs` module doc** (craft): real. The guard has an identity fast path (a kernel with all-literal
  bounds leaves it untouched, `a_kernel_with_no_value_bound_moves_back_untouched`); only idempotence is
  what it lacks. The sentence is now two: every pass but the guard is idempotent, and the guard wraps
  every value-bounded fold it meets, its own output's included, so `legalize` runs it once; every pass
  has an identity fast path.
- **`lower.rs`'s test fixture was built twice** (craft): real. `a_fold_bounded_by` and
  `a_guarded_fold_over_a_count` are now one helper, `a_guarded_fold(at: BoundEnd, end)`, returning
  `(arena, count, fold, root)`; the old helper's seven callers pass `(BoundEnd::Hi, the_count)`, and the
  refusal helpers (`refusal_of_the_end`, `is_refused_as_a_mask`) take the end too.
- **`cell_grid.rs`'s dump tests write a fixed temp name** (craft): real, and the integration twin in
  `pixelflow-graphics/tests/production_glyph_arena_dump.rs` did the same. Both paths now carry
  `std::process::id()`, as `training/corpus.rs`'s `unique_tmp` does.
- **`regalloc/local.rs`'s comment on the saturating sum does not parse** (craft): real. Now: the sum
  saturates; a read weighs its nest's trips, `u64::MAX` for a nest priced past `u64`, so adding two such
  reads would overflow; a value read that deep weighs the most a weight can and compiles.
- **`.wip/a4/spec.md` is stale** (conformance): ignored, as the task said; the rulings are authoritative.

## Verification (TAG=a4-close1; outputs in .wip/a4/run/r5/ and .wip/gates/out/*.a4-close1.*)

- Gates 1 (both tiers, 127 lines each), 1b (29), 2 (29), 2x (29): all exit 0, IDENTICAL to base.
  `git status --short pixelflow-graphics` shows only the two files edited above (never
  `program/tests.rs`).
- `cargo test -p pixelflow-ir` 397 passed (guard 11); `-p pixelflow-codegen`, PIXELFLOW_ISA=avx2 and
  =avx512: all green (`--lib` 102, `tests/value_bounds_jit.rs` 27 each); `-p pixelflow-graphics` 121 lib +
  integration tests, 0 failed (3 ignored lib, as before).
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`
  (`CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`): exit 0. `cargo fmt --all -- --check`: clean.
  `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --document-private-items`: ok; rustdoc
  comparison against the a747770 bases: ir 38/38, search 28/28, codegen 0/0, NEW 0.
- Reduce-site checklist: UNCLASSIFIED 0, STALE 0. check-intent, check-doc-paths,
  check-cfg-encapsulation, check-emit-boundary, check-bin-declarations: exit 0. Every `intent.md` within
  80 lines. Added lines hold no `.wip` path, ruling label or model name. LFS count 68.
- Not rerun: `check-feature-matrix.sh` (no feature, cfg or dependency changed in this round: tests and
  doc comments only) and `measure.sh` (no emitted code changed; gates 1, 1b, 2, 2x are identical).

# A4 closing fixes, round 2 (TAG=a4-close2)

Base of this round: 79c21ea. Gate failures: none. Every item was checked against the code first; the one
that names a wrong value was reproduced before the fix (the new test fails on the unfixed `select.rs`, on
avx2 and avx512: `MIN_{k in [uniform -0.0, 1)} 1/k (as written) at (1, 2): got -inf (0xff800000), want inf
(0x7f800000)`) and passes after.

## Blocking

- **1. A value `lo` of `-0.0` seeds the loop with `-0.0`, a literal seeds `+0.0`** (semantics): real.
  `Selector::fold` seeded the counter with the lane itself, so the first index was `-0.0` where a literal
  `0` (and a literal `-0.0`, which `Bound::discover` turns into `Lit(0)`) seeds `+0.0`: the same count gave
  different answers as a uniform and as a constant for a body that sees the sign of zero. Fixed in
  `emit/select.rs`: a `Bound::Value` `lo` is seeded `lo + 0.0` (`-0.0 + 0.0` is `+0.0` under round to
  nearest on AVX2, AVX-512 and NEON; no backend peephole folds an add of zero, grep); a `Bound::Lit` `lo`
  is untouched, so every literal kernel is byte-identical (gates below). `fold`'s doc says so.
  The cost is one add in the preheader of a fold whose `lo` is a value, which no production kernel has.
  Pin: `value_bounds_jit.rs` `a_value_lo_of_negative_zero_starts_the_fold_at_positive_zero`, through
  `both` (so `run_raw` and `run_linked`), for `lo` spelled as a uniform and as a constant, at `-0.0` and
  `0.0`: `MIN_{k in [lo, 1)} 1/k` is `+inf` and `MAX_{k in [lo, 1)} k` is `+0.0`, bit for bit. A
  uniform and a constant of the same count agree, which is the property. `a_value_lo_starts_the_fold`'s
  doc says the seed is the lane plus zero.
- **2 and 3. `emit/mod.rs`'s module doc says a guard's arm tests are always the column fold's, or run once
  per call in the body** (semantics, and literal-craft; one defect, one fix): real. A fold that reads `Y`
  alone sits in the row fold, and one bounded by an enclosing fold's index sits in that fold's scope,
  once per trip (the plan's section 1.2 table and `guard.rs`'s
  `a_bound_on_an_enclosing_index_is_guarded_inside_that_fold` already said so). The paragraph now says the
  arm tests sit where the fold does, in the innermost scope of the nest the fold reads, and lists the
  four cases (column fold per batch for `X`; row fold per row for `Y` alone; an enclosing fold's scope per
  trip for its index; the body per call for none), and that a mask the register file cannot keep across
  those trips (an AVX-512 opmask) is reloaded each time the scope runs. The `lo` seed's `plus zero` is in
  the same paragraph. The row-fold case had no pin, so one is added:
  `a_guard_around_a_fold_that_reads_only_y_is_the_row_folds` (`Y * Sum_{k<n}(Y + k)` against the literal
  twin: the row fold gains exactly the guard's two selects, the column fold gains none; the scopes are
  found by their trips, not by index). Mutation check: reading `X` in the value kernel's body moves the
  selects into the column fold and the test fails ("the row fold runs the guard's two selects ...");
  reverted, `cmp` clean.
- **4. `assert_off_the_lattice`'s doc says "a bound is one value for the call"** (literal-craft): real. An
  enclosing fold's index passes through this assert and takes one value per trip of that fold. The doc now
  says what L3 guarantees: the bound reads no lattice binder, so it is one value for the call or, for an
  enclosing fold's index, one value per trip of that fold, and either way one value in every lane of a
  batch.
- **5. The totality doc's count clause and B1 clause** (literal-craft): real. Both the paragraph and the
  reduction-binder row now read "a literal, a value known per call, or an enclosing fold's index"; the B1
  clause is scoped to extents (B1 makes the discrete-domain and typed-field extents values; bounded
  iteration's trip count stays static); the repeated "Each is fixed before the lattice is traversed" is
  gone. Same defect, not on the list: consequence 5 said "a literal or a per-call value"; it now says never
  a coordinate, a literal, and for a fold's count also a per-call value or an enclosing fold's index.
- **6. The plan's section 5 cell mixes two line numberings** (literal-craft): real (`:110-112` contains
  `:111`, and neither named what the cell meant). The row cites by content: "totality doc, 'One discipline,
  four faces' (the paragraph and its table) and consequence 5", CL "A1 the rule; the paragraph, the
  reduction-binder row and consequence 5 A4 as built; the discrete-domain and typed-field rows B1". Its
  Becomes cell now includes the enclosing index.

## Non-blocking, applied

- **`a_guarded_fold`'s doc calls its guard "the shape `passes::legalize` leaves"** (literal-craft): real, true
  only at `Hi`. Reworded to what it is: a stand-in `If`, not the `Lt(lo, hi)` legalize builds; lowering asks
  only that the fold has one reader, an `If`'s true arm. (Built from the real bounds instead? Not done: lowering
  checks only that the fold's one reader is an `If`'s true arm, so a stand-in test is enough.)
- **"`hi` is read only by the latch's compare"** (literal-craft), in `lower.rs`'s test doc and the plan's
  lowering bullet: real, the conclusion held and the reason did not. Both now say every op that reads `hi`
  (the latch's compare, the guard's tests, a body that reads it too) widens an opmask operand, so a mask
  `hi` would not panic, and only a mask `lo` (a head parameter) would. No other file carries the sentence
  (grep for "latch's compare", "read only by").

## Skipped

None. The reviewer's two notes on mutation coverage are a description of which tests pin which class, not
a defect.

## Verification (TAG=a4-close2; outputs in .wip/a4/run/c2/ and .wip/gates/out/*.a4-close2.*)

- Gates 1 (both tiers, 127 lines each), 1b (29), 2 (29), 2x (29): all exit 0, IDENTICAL to base.
  `git status --short pixelflow-graphics` is empty (the probe file `tests.rs` is clean). LFS count 68.
- `cargo test -p pixelflow-codegen`, PIXELFLOW_ISA=avx2 and =avx512: 207 passed, 0 failed each (`--lib` 102;
  `tests/value_bounds_jit.rs` 29 each, the 27 of round 1 and the two above). Only `pixelflow-codegen` has
  code changes this round; the other touched files are docs.
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`
  (`CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`): exit 0. (First pass flagged `type_complexity` on
  the new test's `[(&str, fn(f32) -> Kernel); 2]`; a local `type Spelling` fixed it.) `cargo fmt --all --
  --check`: clean. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p pixelflow-codegen
  --document-private-items`: ok.
- Reduce-site checklist: UNCLASSIFIED 0, STALE 0. check-intent, check-doc-paths, check-cfg-encapsulation,
  check-emit-boundary, check-bin-declarations: exit 0. Every `intent.md` within 80 lines. This round's added
  lines hold no `.wip` path, ruling label or model name.
- Not rerun: `check-feature-matrix.sh` (no feature, cfg or dependency changed) and `measure.sh` (gates 1, 1b,
  2, 2x are identical: the seed add exists only in a fold whose `lo` is a value).

# A4 closing fixes, round 3 (TAG=a4-close3)

Base of this round: ccb24ba. Gate failures: none. Every item was checked against the code first. Round 2's
seed (`lo + 0.0` for a value `lo`, `select.rs`) made two kinds of doc false; this round corrects the docs and
the one test that went vacuous, and adds the pins the findings named. No emitted code changed.

## Blocking

- **1. Two docs say the allocator copies a loop parameter** (semantics): real. `select.rs` seeds a
  `Bound::Value` `lo` with `lo + 0.0`, so the inner fold's entry argument is that Add's result: fresh, read
  by nothing after the jump, so `LinearScan::pass` -> `home` hands its lease over (`once && !live_after`)
  and `copy_of` is not reached. Checked by reading `pass`/`home` and the seed (the reviewer's instrumented
  run, which printed a copy only with the round-1 seed, was not repeated). Fixed:
  - `an_inner_fold_starting_at_the_outer_index`'s doc now says the inner counter is seeded `i + 0.0`, a fresh
    value, so the entry argument is not the outer counter itself, and each trip of the outer fold starts the
    inner one at that trip's index. The test stays: it pins a value `lo` that is an enclosing index, on both
    paths, which is semantics and not the allocator.
  - `every_backend_emits_a_fold_whose_bound_is_a_value`'s doc: the seed "reads another loop's counter
    (`i + 0`, added where the fold is entered)", and the loop's forms are the seed's add of zero, a compare of
    two registers and the arm skip ("a copy" is gone).
  - The copy path itself: no kernel reaches it now. It was not reachable at the base either (every seed was a
    constant), and the allocator has no unit tests to pin it with; `home`'s copy arm is unchanged and stays
    defensive. Not pinned, said plainly here.
- **2. Seven docs give a tier reason that round 2 made false** (literal-craft): real. `LaneOp::Binary` on
  AVX-512 calls `op.on_masks(..)`, which is `None` for `Add`, so `a.vector(b)` widens the opmask
  (`vpmovm2d`) and a mask `lo` no longer panics in selection: every tier reads its bits as a float (all ones is
  NaN, which fails the guard's domain test; all zeros is the count 0). The refusal is still right, because a
  mask names no count. The asymmetry rationale (lo panics on AVX-512, hi does not) is removed, not
  rewritten, at all seven sites, leaving the law: a mask names no count, refused at either end on every tier.
  Sites: `pipeline.rs` (`compile`'s doc), `lower.rs` (`arena_to_schedule`'s doc, the inline comment at the
  refusal, the doc of `a_bound_is_refused_exactly_when_the_emitter_holds_it_as_a_mask`), `emit/avx512.rs`
  (the doc of `a_value_is_an_opmask_exactly_when_is_mask_says_it_is_a_mask`), `tests/value_bounds_jit.rs`
  (`a_bound_that_is_a_mask_is_refused_by_name`), and the plan's lowering bullet. The plan keeps one true tier
  note (the `lo + 0` seed widens an opmask) and drops "every op that reads `hi` widens an opmask operand",
  which is not exact (a bitwise and/or of `hi` with another mask stays in the opmask file). An eighth site the
  grep found, not on the list: `ScheduledOp::is_mask`'s doc said "(AVX-512's opmasks, where no head parameter
  can be)"; now "(AVX-512's opmasks)".
- **3. The guard-placement list gives two answers for one fold** (literal-craft): real, reproduced on the
  reviewer's kernel `Σ_{i<8} Σ_{j<u}(X + j + i)`: the guard's two selects land in the `i` fold's scope, not the
  column fold's. `emit/mod.rs`'s list is now exclusive, innermost first, "the first of these that holds": an
  enclosing fold's scope once per trip (when the fold's body or bound reads that fold's index, the innermost
  such fold's); else the column fold, per batch (reads `X`); else the row fold, per row (reads `Y`); else the
  body, once per call. New pin, as the reviewer proposed:
  `a_guard_around_a_fold_that_reads_an_enclosing_index_is_that_folds` (`Σ_{i<8} Σ_{j<8}(X + j + i)`, the inner
  count a value, against its literal twin): the `i` fold's scope gains exactly the two selects, the column
  fold's gains none, and on AVX-512 the `i` fold's scope gains loads (the opmask reload per trip). Mutations,
  both tiers: asserting the two selects in the column fold's scope instead fails (`instructions: 3` against
  `1` in the `i` fold's scope, and the column fold has none); a body that no longer reads `i` moves the
  inner fold out of the `i` fold and fails the same-loop assertion first.
- **4. `a_guard_around_a_fold_that_reads_only_y_is_the_row_folds` finds the wrong "column fold" on AVX-512**
  (literal-craft): real. The fold over `k` runs 4 x 7 = 28 times against the column fold's 4 x (64/16) = 16, so
  "the most-run known scope" was the `k` fold there and the column assertion compared it. The scope is now
  found by its own trip count: new helpers `lanes()` (`jit_vector_bytes() / 4`), `column_trips()` (rows x
  `width.div_ceil(lanes)`, no remainder at `SHAPE`'s width) and `scope_running(traffic, trips)`, which panics
  unless exactly one scope runs that often, so a collision between two folds' counts is loud. The test and the
  new pin share them. The comment now says what is true. Mutation, both tiers: the value and literal kernels
  both read `X` in the `k` fold (so the guard moves to the column fold) with the row assertion removed: the
  test fails on "the column fold, run in every batch, runs none of the guard" (`instructions: 7` against
  `5`), AVX-512 included, where the old search compared the `k` fold and could not.

## Non-blocking, applied

- **`lower.rs`'s "every op that reads `hi` widens an opmask operand"** (semantics): real, removed with item 2.
- **`over_values`' "makes the fold NaN" has an exception it does not list** (semantics): real, reproduced by
  the new test: `Σ_{k<n} 0·(X + k)` at `n` = NaN and 2.5 is NaN as written and exactly `+0.0` through
  `jit_cache::compile`, on avx2 and avx512 (the factored `0` is annihilated with the fold, so the guard has
  nothing to wrap). In the domain the paths agree. The doc now lists both edges `FactorFold` accepts
  (decision 5): the factored form at a runtime-empty range, and a literal zero factor; `FactorFold`'s own doc
  gets the paragraph ("A literal zero `c` takes its fold with it"), and the plan's lowering bullet says so.
  Pin: `a_zero_factor_takes_the_fold_and_its_guard_with_it`, in domain both paths give `0`, outside as written
  NaN and optimized NaN or `+0.0` (the claim is the edge's existence, not a promise of which). `CLAUDE.md:180`
  is left: it states the denotation, and the optimizer's departures from it are what the "Floating point at
  the edges" section already says, which is also where the empty-range edge, never listed there, lives.
- **`jit_cache.rs`'s `# Panics` paragraph** (literal-craft): real. Reflowed to the paragraph's width, and the
  colon that read as introducing the `over_values`/`from_parts` clause is a sentence break ("since a mask
  names no count.").

## Skipped

None.

## Verification (TAG=a4-close3; outputs in .wip/a4/run/c3/ and .wip/gates/out/*.a4-close3.*)

- Gates 1 (both tiers, 127 lines each), 1b (29), 2 (29), 2x (29): all exit 0, IDENTICAL to base.
  `git status --short pixelflow-graphics` is empty (the probe file `tests.rs` is clean). LFS count 68.
- `cargo test -p pixelflow-codegen`, PIXELFLOW_ISA=avx2 and =avx512: 209 passed, 0 failed each (`--lib` 102;
  `tests/value_bounds_jit.rs` 31 each, the 29 of round 2 and the two above). `-p pixelflow-ir` 397 passed,
  `-p pixelflow-search` 434 passed (doc-only changes in both; the ir doctests ran).
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`
  (`CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`): exit 0. `cargo fmt --all -- --check`: clean.
  `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir|pixelflow-search --no-default-features`: ok.
  `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p pixelflow-codegen --document-private-items`: ok; rustdoc
  comparison against the a747770 bases: ir 38/38, search 28/28, codegen 0/0, NEW 0.
- Reduce-site checklist: UNCLASSIFIED 0, STALE 0. check-intent, check-doc-paths, check-cfg-encapsulation,
  check-emit-boundary, check-bin-declarations: exit 0. Every `intent.md` within 80 lines. Added lines hold no
  `.wip` path, ruling label or model name (`git diff a747770 -- . ':!.wip'`).
- Not rerun: `check-feature-matrix.sh` (no feature, cfg or dependency changed), `measure.sh` (no emitted code
  changed: tests and doc comments only, and gates 1, 1b, 2, 2x are identical).
