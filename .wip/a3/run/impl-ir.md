# A3 in pixelflow-ir: what was built (implementer 1 of 3)

The working branch, code base `3bfdaded`. Scope: `pixelflow-ir` only
(spec §11 steps 1 to 4, §10's ir tests, B1 for ir). Committed locally as
`chore(wip): A3 in pixelflow-ir (not for merge)`; not pushed.

## Status

| Gate | Result |
|---|---|
| `cargo test -p pixelflow-ir` | green: 292 lib tests (base 246, +46), 6 in `tests/value_bounds.rs` (new), the other integration tests as before |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features` | green (the `expect(dead_code)` pair is fulfilled) |
| `cargo fmt -p pixelflow-ir -- --check` | clean |
| `cargo clippy -p pixelflow-ir --all-targets --all-features -- -D warnings` | **red at base too, for 3 findings A3 did not write** (see below); with `-A clippy::must_use_candidate` it is clean; head's `must_use_candidate` set is identical to base's |
| rustdoc (ruling C4), `cargo doc -p pixelflow-ir --no-deps --document-private-items` | base 38 warnings, head 38, `NEW 0`, `GONE 0` |
| `bash scripts/check-intent.sh` | OK; `pixelflow-ir/intent.md` is 80 lines |
| B1 in ir (`grep -rnwE 'A3' --include=*.rs pixelflow-ir`, `grep -nw 'A3' pixelflow-ir/intent.md`, the phrase grep) | empty |
| Reduce-site checklist (`python3 -I .wip/a3/gen/checklist.py`) | `UNCLASSIFIED 0`, `STALE 0` (162 sites; the 11 new sites are tests) |
| `check-doc-paths.sh`, `check-cfg-encapsulation.sh` | OK |
| `cargo check -p pixelflow-search -p pixelflow-codegen --all-targets` against the new ir | compiles, no warnings |
| `cargo check -p pixelflow-compiler` | fails at exactly the two predicted sites, `lower.rs:515` and `emit.rs:838` (`close_fold` / `close_over` take `FoldTerms`): implementer 3's |

**The clippy finding.** On this toolchain (clippy 0.1.97) `-D warnings` fails at `3bfdaded` itself,
on `must_use_candidate` for `ExprArena::nodes` (`arena.rs`), `Dag::iter` and `Builder::entries`
(`dag.rs`), all `impl Iterator` returns. I extracted `3bfdaded` to a scratch directory and reproduced
it there, and diffed the warning set: the same three at head, only line numbers moved. A3 adds none
(every new `pub` fn has `#[must_use]`, or returns a `Result`, or is `&mut self`). I did not touch the
three: they are outside A3's diff. If you want the gate literally green, the fix is `#[must_use]` on
those three functions and nothing else; say so and it is a four-line change.

**Base-versus-head evidence for byte-identity at the ir level** (L1, §1.3). A scratch crate
(outside the repository) builds 16 kernels through the API both trees share and dumps, per
kernel, every arena node in order, the uniform count, `free_index`, the root's variance, the
`KernelKey` digest and the display text: `sum_over`, empty and zero ranges, `0..2^24`, `MIN`, `ANY`,
`ALL`, the mean `Σ/4` (a literal after its fold), a uniform in the body, nested folds with an outer
index in the inner body, six nested folds, `at`, `select`, `dwrt`, `by_ref`, sibling folds. Run
against `3bfdaded` extracted to scratch and against this tree, the two outputs (204 lines) are
identical with `diff`. This is not byte_probe or the 29 programs' keys (those are the gate's); it is
the ir-level share of them.

## What changed, by item

### `fold.rs` (spec §2)

- `BoundEnd`, `BoundRefusal` and their `Display` (§2.1, §2.2). Messages verbatim from the spec;
  each starts with the variant's name.
- `Fold::is_exact` (private, §2.3). Its callers are `KnownRange::admits`, `assert_stride`,
  `FoldHeader::from_bits`, `Fold::count`, and `FoldHeader::new`'s value arm. After the change the only
  `n <= EXACT_BOUND` comparison in the file is inside `is_exact`; the other `EXACT_BOUND` hits are the
  definition, docs, panic messages, the const assert, and tests.
- `Fold::count` (`pub(crate)`, §2.4). Departure from the spec text only in the sign test (C3): `real.is_nan() || real < 0.0`
  instead of `!(real >= 0.0)`, which trips `clippy::neg_cmp_op_on_partial_ord`. `-0.0` is still the count 0.
- `Bound::discover` and private `Bound::read` (§2.5), in an `impl<R: Copy> Bound<R>` block. The doc
  writes `Fold::count` in backticks without a link (C5). The doc and `BoundRefusal::BoundSort`'s doc state the
  S1 hole: the sort is read off the term as written, so a computed bound that is a mask is admitted
  and T6 closes it.
- `Fold::admits_bound` (§2.6), `Fold::ESTIMATED_TRIPS` (§2.7; placed beside `EXACT_BOUND`, the other
  associated const), `FoldHeader::priced_trips` (§2.8), `Monoid::takes_a_value_bound` (§2.9; N10's name).
- `FoldHeader::new` (§2.9): a literal pair takes today's `KnownRange::new` path first and unchanged;
  anything with a value goes to the private `with_a_value` (monoid sort, stride 1, a literal beside a
  value inside the domain), then `store`.
- Docs (§2.10): `store`, `to_bits`'s `# Panics`; `VALUE_BOUND` and `KNOWN_RANGE` texts (§2.11) with
  the doc paragraphs the spec gives. `VALUE_BOUND` is now expected only by the `to_bits` test and by
  T-IR-26 (C15's list).
- B1: every A3 mention in `fold.rs` is reworded; the five tests §2.12 names are rewritten
  (`a_value_bound_has_no_bits_until_a4`, `a_value_bound_is_not_the_literal_zero` with its new doc).

### `variance.rs` (§3.1, §3.2)

- `Variance::of_var`. Its doc carries J5's reason (a placeholder is a binder-to-be, never a
  coordinate, and L3 is checked again when `close_over` renames it into a slot and re-pushes the fold).
- `transfer`: `compute_arena_variance`'s match moved arm for arm into one private function over
  `child: impl Fn(ExprId) -> Variance`, with L4 in the `Reduce` arm and all the old comments kept.
  `compute_arena_variance` is its forward driver; `variance_from` (`pub(crate)`) is the rooted one
  with a `BTreeMap` memo (N1).
- Docs: S2's exception (the arena types a fold over a known empty range `CONST`; `ENode::variance`
  has no empty arm, so `EGraph::add`'s backstop can refuse a hand-built bound `push_reduce` admitted)
  is in `transfer`'s doc; C14's note (a `Ref` is resolved over the referent's whole arena per push;
  memoize per referent when B3's profile asks) is in `variance_from`'s doc.

### `arena.rs` (§3.3 to §3.6)

- `free_index`: the `Reduce` arm unions the value bounds unmasked; one sentence added to the doc.
- `push_reduce`: discovery, then `FoldHeader::new`, then L3 per value bound through `variance_from`;
  private `literal_at`. Doc rewritten (N4: a `Const` naming no count stays a value; S1's hole stated).
  Links to `BoundRefusal` use explicit `crate::fold::` paths (C5).
- `FoldTerms` (new, `pub`), `close_over` and `close_fold` take it. `close_over` discovers before the copy
  (N2), roots are `[body, lo?, hi?]` through `from_arena_roots` / `to_arena_roots`, and the value
  bounds' ids are paired in order with one iterator rather than through `attach`.
- `lowest_free_binder(roots)` marks binder `Var` reads as well as `Reduce` binders (J8).
- `substitute_vars_with`'s doc states the precondition (N5); no claim of L7 for it. `known_range`'s
  doc is the spec's.
- B1: the three A3 mentions are gone; the old `VALUE_BOUND` assert in `close_over` is gone.

### `kernel.rs` (§4)

- `Kernel::over` builds a `FoldTerms` for `close_over`; its doc paragraph is N9's replacement. Its
  `# Panics` is kept: a reversed or past-2^24 literal range is still refused by `KnownRange::new`'s
  message (J13), now inside `close_over`'s `push_reduce`.
- `Kernel::over_values` (`pub(crate)`) and `user_bound` (private), both with
  `#[cfg_attr(not(test), expect(dead_code, reason = "pub(crate) until A4 makes it public (decision 7)"))]` (C2). Visibility did not change.
  The two expects both compile clean in the lib and no_std builds; `user_bound`'s is in fact not
  needed (rustc treats the callee of an expected-dead item as live, and removing it also builds
  clean), and is kept because C2 names both.
- `Param`/`Buffer` roots get `BoundNotANumber` (N10), not `BoundArithmetic`.

### `passes.rs`, `intent.md`

- `an_extent_is_not_an_expression`: doc and comment reworded as T-IR-23 gives; sibling
  `a_value_bound_is_a_child_after_the_body` added.
- `intent.md:73`: "Codegen and the key take literal `Bound`s until A4." (80 lines, at budget).

### Tests (§10), all green

T-IR-1 to 8d in `fold.rs`; T-IR-9 to 11 in `variance.rs`; T-IR-12 to 22 and the `FoldTerms` updates in
`arena.rs`; T-IR-23 in `passes.rs`; T-IR-24 to 28 in the new `tests/value_bounds.rs` (public API only);
T-K-1 to 9d in `kernel.rs`; `an_argument_admitted_before_a_fold_is_applied_inside_it` updated.
Added beyond §10: `a_constant_bound_is_read_where_it_is_pushed` (value_bounds.rs) and
`over_values_reads_each_end_for_itself` (a uniform lower bound beside a constant upper one, and one
uniform at both ends, which pins the duplicate-root path in `close_over`).

### Reduce-site rows

`classes.json`: 4 rows edited in place (`free_index`, `lowest_free_binder`, and the two `variance.rs`
rows now keyed under `transfer`; all `changed-by-A3`), 10 rows added for 11 new sites (all `tests`: four in
`arena.rs`, two in `kernel.rs` of which one has `count: 2`, four in `tests/value_bounds.rs`). The checklist reports 0 and 0.
Rows are sorted by `(file, first line)` using the stale line numbers of untouched rows, so a few
rows moved within a file; no row changed class except by the edits above.

## Judgement calls (the rejected alternative in each)

1. **T-IR-24's fixture bound is `(j + u) + 2`, not the spec's `Add(Var(5), u)`.** The spec says `t` is
   "deeper than the body", but `Add(Var(5), u)` has depth 2, the same as the body `Var(4)·Y`, so
   `depth(inner) == depth(t) + 1` could not tell a walker that reads the body only from one that reads
   every child. `(j + u) + 2` has depth 3 and 5 nodes. Rejected: the spec's literal term.
2. **T-IR-24's display assertion is the full text `fmt_expr` writes** (C8), read from a run, not
   guessed: `reduce[add_4[0.._)](mul(Var(4), Var(1)), add(add(Var(5), Uniform(0)), Const(2)))`. The
   spec's `add_4[0.._)(` was wrong, and C8's corrected `add_4[0.._)](` is a fragment of this. Rejected: a
   prefix match, which would not pin that the bound follows the body.
3. **T-IR-5 is named `a_value_bound_takes_an_arithmetic_fold_at_stride_one`**, not the spec's
   `…_a_real_sorted_fold_…`: N10's grep `grep -rn 'real_sorted' --include=*.rs` must be empty, and the
   spec's own test name contains the substring.
4. **`BoundRefusal`'s variants carry no per-field docs**, as the spec writes them; the variant doc says what each
   field is. Rejected: field docs that repeat the variant doc.
5. **`ESTIMATED_TRIPS` sits beside `EXACT_BOUND`** at the top of `impl Fold`, not after `over`. Same
   item, grouped with the other associated const.
6. **Test helpers `b`, `sum`, `terms` are defined per test module** (fold, arena), as the spec's
   `b(s)`/`sum(s)` notation suggests; there is no shared test support in ir for them. Rejected: a new
   shared `#[cfg(test)]` module for three one-line functions.
7. **T-IR-11 builds its own arenas** with the same shapes as the tests it names
   (`nested_binders_consume_one_index_each`, `a_write_varies_with_its_value_and_its_binders`, T-IR-10's, and a `Ref`), because those tests build inline and
   expose no builder. Rejected: refactoring existing tests to share their arenas.
8. **`a_header_prints_a_value_bound_as_a_hole` and the `attach_*` tests keep `FoldHeader::store`.**
   They need stride 2 beside a value, which `new` now refuses, and `store` is the module-private writer
   that the doc for "the header prints" is about. Rejected: rewriting them to `new` for the cases that
   could, which would leave two spellings in one test.
9. **The `lowest_free_binder` comment** ("is why this is one line") no longer described the code once the arm
   became a `match` on `Reduce` and `Var`; reworded, history kept, and a sentence added for the `Var` arm (why a free binder is taken).
10. **`variance_from`'s doc cites `passes::differentiate`** (where the 86 s measurement is written) instead
    of the spec's `passes.rs:571-574`, which a line number would not survive.
11. **The module-level `use`s in `variance.rs`** (`ExprArena`, `ExprId`, `ExprNode`, `KnownRange`) replace
    the function-local ones `compute_arena_variance` had; the tests keep their own local `use`s, which shadow harmlessly.
12. **The "rest of the doc kept" items I left untouched**: `Fold`'s and `KnownRange::of`'s "when a bound
    becomes a value, the readers change and their callers do not" and `known_range`'s second paragraph are
    design rationale that stays true, and B1's grep does not flag them.

## Departures from the spec or the rulings

- Only the three above that change an observable string or fixture: judgement calls 1, 2, 3, and C3's
  sign test (a ruling, not a departure). No ruling is unmet in ir.
- Not done here, by scope (they are implementers 2 and 3's): S2's note at `EGraph::add`'s claim (the
  spec's §6.2 sentence, in `pixelflow-search`), S3, S6 and the rest of C15 (search docs), C16 and the
  plan text, `pixelflow-ir`-adjacent greps outside ir (`node.rs`, `insert.rs`, `program/lower.rs`,
  `training/corpus.rs` A3 mentions).

## Facts the other implementers need

- `Variance::of_var` is in; `ENode::variance` should call it and `var_variance` can go (search still has its own copy).
- `Fold::count` is `pub(crate)`: search must read a count through `Bound::discover` (as `insert` and `KnownBound` do).
- `Bound::discover(lo, hi, literal)` takes `impl Fn(R) -> Option<Lit>`; a literal pair never calls it.
- `FoldHeader::priced_trips()` and `Fold::ESTIMATED_TRIPS` are ready for `cost.rs` and the extractor; `Fold::admits_bound(end, reads)` is ready for `EGraph::add`.
- `ExprArena::close_over` and `close_fold` take `FoldTerms { monoid, stride, lo, hi }`; the compiler's `lower.rs` and `emit.rs` call sites no longer compile.
- `push_reduce` panics on a value bound under `ANY`/`ALL`/`SEQ` with a message starting `BoundSort`, so every search call that can build one (`ENode::reduce`, `insert`, plan replay, templates) panics the same way.
- Remaining A3 mentions to reword outside ir are in `node.rs`, `insert.rs`, `program/lower.rs` and `training/corpus.rs` (spec B1's list).
- `no_std` test target: `cargo test -p pixelflow-ir --no-default-features` does not compile at base either (`crate::store`, `format!` in existing tests); only `cargo check` is a gate, and it passes.
