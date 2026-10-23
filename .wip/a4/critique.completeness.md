# A4 spec critique: completeness and byte identity

Read against `a747770` (HEAD `936d3aa` differs only under `.wip/`). Every claim below was rerun at
that tree. Ranked most severe first. "Blocking" means a named test or a gate in the spec cannot pass
as written.

## What holds

The plan's A4 bullet (plan:627-650) is covered site by site, and the line numbers are right:

- `VALUE_BOUND` has three readers and three expecting tests (`fold.rs:998`, `lower.rs:51`, and the
  tests at `fold.rs:2122`, `lower.rs:470`, `tests/value_bounds.rs:150`). All six are in the spec.
- Every serializer of a header is in §4.3's table: `key.rs:457`, `test_support.rs:37`,
  `cell_grid.rs:1173`, `production_glyph_arena_dump.rs:171`, `collapse_bench/corpus.rs:276` and
  `:501`, `training/corpus.rs:412` and `:605`, `corpus_gaps.rs:499`, and compiler `emit.rs:583`.
  The sites it leaves alone print `Display` plus every child, or copy every child.
- The training corpus's `SCHEMA` (`training/corpus.rs:94-113`) does not mention tag 9 today, so
  adding the clause is the version bump.
- The collapse corpus is v4 (`corpus.rs:113`), and no corpus fixture is tracked.
- `over_values` is `pub(crate)`, with the two `expect(dead_code)` attributes at
  `kernel.rs:812-815` and `:865-868`.
- The checklist reproduces: 175 sites, 15 marked `A4`, 0 unclassified, 0 stale.
- `check-intent.sh` passes at base. The budgets the spec states are the files' real line counts.

Byte identity on today's all-literal kernels holds at every changed site I traced:

- `to_bits` and `from_bits` take today's branch for two literals.
- The guard's fast path moves the arena back. Even an arena that holds only an unreachable
  value-bounded fold would rebuild to the same ids, because `copy_node` re-interns.
- `lo.map(map_child)` of a `Lit` is the `Lit`, and `operands` yields nothing for a literal.
- `bound(Lit(n))` makes the same `LaneOp::Const`, in today's order.
- The `Trips` `Display` prints a known count bare, so the `trips={trips}` text and `{trips:>8}`
  column are unchanged.
- The `AndOrGraph` bounds list is empty on a literal kernel. `shared_pass_bytes` is reach-set
  bytes (`extract.rs:2226`), not the `AndOrGraph`, so gate 2x is unaffected.

## Defects

**1. Blocking. Two key tests assert a `KernelKey` equality that cannot hold.**
- §8.1 `every_count_is_one_key` wants equal `canonical(..).key` *and* equal `KernelKey` for two
  uniform instances with different defaults.
- §8.1's flipped `a_value_bounded_kernel_has_a_key` wants `KernelKey::of` to equal the key of the
  fixture rebuilt with another default for `u`.

`KernelKey::of_canonical` hashes `form.uniforms` (`pixelflow-ir/src/key.rs:132-139`), and
`UniformDecl` hashes both its identity and its default's bits (`arena.rs:352-365`). So both
`KernelKey` assertions fail. One program per `n` is a property of the canonical *bytes*, and those
bytes are the JIT cache key (`jit_cache.rs:158-162`, `CacheKey.expanded`). Assert byte equality, and
either assert that the two `KernelKey`s differ or drop the `KernelKey` half.

**2. Blocking. §8.7's collapse-corpus round trip compares with `subtree_eq`, which fails for any
kernel with an `A` line.**
- `decode` declares each `A` line as `Uniform::new(default).decl()`, a freshly minted identity
  (`collapse_bench/corpus.rs:418-422`; `kernel.rs:108-115`).
- `subtree_eq` compares uniforms by slot *and* declaration (`arena.rs:2289-2293`), and
  `subtree_eq_distinguishes_uniform_declarations` (`arena.rs:3323`) pins exactly this.
- The module's own oracle is `encode` equality (`corpus.rs:716-731`, `:849-853`).

Use `assert_eq!(encode(&kernel), encode(&decoded))`.

**3. Blocking (gate). The §9 grep `'no corpus encoding before\|has no bits\|no code and no key'`
cannot come out empty.** `has no bits` matches two unrelated lines that A4 does not touch:
`pixelflow-codegen/src/program/lower.rs:74` (the `shift_immediate` message) and
`pixelflow-ir/src/kernel.rs:1283` (`shl`'s doc). Use the exact old text instead:
`'{self:?} has no bits'`, or `has no bits")`.

**4. Blocking (gate). The §9 grep for `known\(` in `pixelflow-codegen/src` cannot give "lower.rs's
lane fold only".** §3.3 adds `#[must_use] pub fn known(self) -> Option<u64>` to
`emit/traffic.rs`, and `dynamic_memory_ops`, built on `Option`'s `Sum`, calls `.known()`. Both
match. Anchor the pattern on the deleted helper (for example `fn known(bound`), or list
`traffic.rs` among the expected hits.

**5. Non-blocking. A4 makes two docs false, and neither §10.5 nor the `A4` grep reaches them.**
`pixelflow-search/src/egraph/fold_rules.rs:544` (`KnownBound`'s doc) and `:1812` (a test doc) both
say "by the guard codegen will emit". After A4 the guard is the last pass of
`pixelflow_ir::passes::legalize`, and it exists. Add both lines to §10.5.

**6. Non-blocking. §10.4's "the three `fails_loudly` tests → five" is off by one.** `lower.rs`
already has four `*_fails_loudly` tests at base (`:410`, `:423`, `:438`, `:455`), and §8.3 adds
two, which makes six. The intent line is already stale at base.

**7. Non-blocking (left to the implementer). §8.3 `a_bound_on_a_lattice_binder_fails_loudly`
cannot "hand-build" a `Write` the way it says.** The test sits in `pixelflow-codegen`, and
`ExprArena::push_write` is `pub(crate)` (`arena.rs:1044`). `collapse` also never picks a slot that a
reachable `Var` reads (`passes/lattice.rs:157-191`), so legalizing cannot produce the case either.
The public route is `pixelflow_ir::Ir::embed(&mut arena, Shape::Write { .. })` (`term.rs:144-149`,
`:184`, implemented in `term_arena.rs`). Name it.

**8. Non-blocking. §8.7's `hash_cons_keeps_a_value_bound` points at a test module that does not
exist, and half of it cannot fail.**
- The spec cites "`src/bin/corpus_gaps.rs` tests (`:240`)", but the bin has no
  `#[cfg(test)]` module; `:240` is the `buf` parser.
- "Two identical ones merge" is vacuous: `ExprArena` hash-conses on every push (`dag.rs:525-533`),
  so two identical folds are already one node before `hash_cons` runs.

Name the new module, and keep only the half that can fail: two folds that differ only in their
bound's child stay two nodes.

**9. Non-blocking (left to the implementer). §8.3's two `pipeline.rs` tests do not give
`schedule_for`'s shape or lane count (`pipeline.rs:154-162`).** If the width is not a multiple of
the lane count, `pack` makes a main column fold and a remainder fold that share one `Write`
(`lattice.rs` module doc; `lower.rs:97-101`). Then `R` has a `ScopeFold` under each column fold,
and "the `Reduce`'s `ScopeFold` has a column fold as `parent`" names two folds. Fix the shape, for
example `[8, 2]` at 4 lanes.

**10. Non-blocking. §8.2 `each_monoid_runs_under_its_guard` is wrong at `n = 0`, and its runner is
not named.** At `n = 0`, `MAX_{k<n}(X − k)` is `−∞`, not `X`, and `MIN_{k<n}(X − k)` is `+∞`, not
`X + 1`. §8.0's explicit-fold oracle would give the right values. State them, and say whether the
test runs through `run_linked` or `run_raw`.

**11. Non-blocking (input for the §6 review). The residual for (b) leaves out unchosen bounding
candidates.**
- §6's `AndOrGraph::of` code records a `BoundUse` for every member of every class
  (`and_or.rs:186-205`), chosen or not, and `relax` refuses per class (`extract.rs:1944-1964`).
- So a class read as a bound only by a candidate that loses can never settle when it has no lawful
  member, and its non-bound uses then fail too. Under (a) the same extraction succeeds.
- Example: the `i + X` / `i + Y` class of `fold_rules.rs:2201-2213`, also read as a body operand
  beside a cheaper sibling of the bounding fold.

Add this to (b)'s residual column, beside the two-levels-down case.

**12. Non-blocking. The §9 grep `'fold_header(id).to_bits()'` misses two of the sites it polices.**
`corpus_gaps.rs:495-499` and compiler `emit.rs:582-583` bind the header to a variable first, so the
grep does not see them. Grep `\.to_bits()` together with `fold_header` per file, or list the two by
name.

## Verdict

The scope is complete and the literal path is byte-identical. Four items need fixing before an
implementer can pass the spec's own gates:

- the two `KernelKey` equalities (1);
- the `subtree_eq` collapse round trip (2);
- the two §9 greps that cannot pass (3, 4).

The rest are doc and test-detail corrections.
