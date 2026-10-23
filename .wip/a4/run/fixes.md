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
