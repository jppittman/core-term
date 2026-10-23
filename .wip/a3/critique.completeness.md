# A3 spec: critique, completeness and implementability lens

Read-only critique of `.wip/a3/spec.md` against the tree at `3bfdaded` (HEAD `2a5103b` differs only
under `.wip/`), the plan (`docs/language/2026-10-20-a-bound-is-a-value.md` §0–§4, §6 A3, §8), the
rulings (`.wip/a3/rulings.md`) and CLAUDE.md. The defects are ranked, the blocking ones first. Every
claim was checked against the code. Four of them were measured: a probe crate in the scratchpad
reproduced the clippy and rustdoc behaviour, and the gate-6 rustdoc command was run at base.

## What holds

- **§5 table (B2).** Re-running the spec's grep gives 155 lines and 151 unique `(file, line)` sites,
  and those sites are exactly the table's 151 rows (a `diff` of the two lists is empty). The class
  counts match the spec: 39 / 19 / 13 / 60 / 5 / 15. I read the doubtful rows. None is misclassified:
  `witness.rs:356`, `extraction_gap.rs:187`, `telemetry.rs:265`, the `lattice.rs` `collapse_shape`
  rows, the serializers marked A4, and the `runtime.rs` / `extract.rs` rows that sit inside test
  modules. No `.rs` file outside `pixelflow-*` has a hit.
- **Callers of the changed signatures (item 3).** Every caller is named:
  - `close_over` / `close_fold`: `kernel.rs:758` and `:1565`, `arena.rs:1723`, the four arena tests,
    compiler `lower.rs:515`, `emit.rs:838`.
  - `fold_cost`: only `cost.rs`.
  - `below`: `TreePricer`, `Selection::price` and `dense_reference_pass`.
  - `substitute_vars_with`: all five callers.
- **Rulings N2, N5, N7, N8, N9, N10** are carried as design. N2 is discovery in `close_over`, with
  J9. N10 is `is_exact`, `count` as `pub(crate)`, `takes_a_value_bound` and `BoundNotANumber`.
- **Docs and intent budgets (item 6).** Every plan §5 row marked A3 has an edit in §12:
  - arena-is-typed `:72` and `:216-217`;
  - BACKLOG `:32`, `:33` and E6 `:150`;
  - ontology "Unrolling" `:529-552`.

  `pixelflow-ir/intent.md` is 80 lines and stays 80. `pixelflow-search/src/egraph/intent.md` goes from
  74 to 75 lines. `scripts/check-intent.sh` passes at base.
- **`root_shapes.rs` needs T-S-15.** Its `no_rule_matches_a_node_outside_its_root_shapes` asserts that
  every rule fires (`silent.is_empty()`), so `KnownBound` must find a match there. The spec's T-S-15
  provides one.

## Blocking

1. **The B1 grep cannot come out empty as the spec writes it.** §7.3 gives verbatim doc text for
   `below` in `extract.rs`: "… charged once — before A3 `below` read only the body …" (spec.md:1679).
   §11 requires `grep -rnwE 'A3' --include=*.rs pixelflow-*` to be empty and lists `before A3` among
   the forbidden phrases (spec.md:2215-2222). Implemented verbatim, both checks fail. Fix: say "before
   value bounds" instead, or drop the clause.

2. **`Kernel::over_values` and `user_bound` are dead code in the library target.** `over_values` is
   `pub(crate)` (spec.md:1009-1010), and nothing outside the test module calls it or `user_bound`
   (spec.md:1059). rustc reports `dead_code` for an unused `pub(crate)` associated function; the probe
   crate showed "associated function `over_values` is never used". Two checks build the library
   target with `-D warnings` and so fail:
   - gate 6's `cargo clippy --workspace --all-targets --all-features -- -D warnings`, which is the
     same command as CI's (`.github/workflows/rust.yaml:421`);
   - the no_std presubmit `cargo check -p pixelflow-ir --no-default-features` (`lib.rs:13-14`).

   Fix: the spec must decide. One option is
   `#[cfg_attr(not(test), expect(dead_code, reason = "public in A4 (decision 7)"))]` on both
   functions. The other is a non-test caller.

3. **`Fold::count`'s body fails clippy.** The spec writes `if !(real >= 0.0) { return None; }`
   (spec.md:283). That trips `clippy::neg_cmp_op_on_partial_ord`, which is warn-by-default and so an
   error under `-D warnings`; the probe reproduced it. Nothing in the tree allows this lint. Fix:
   `if real.is_nan() || real < 0.0`. This keeps `-0.0` (since `-0.0 < 0.0` is false) and still
   rejects NaN.

4. **Gate 6's rustdoc command fails at base, before A3 changes anything.** The gate is
   `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-ir -p pixelflow-search -p pixelflow-codegen
   --no-deps --document-private-items` (spec.md:2204-2205). At `3bfdaded` it gives 39 errors in
   pixelflow-ir and 29 in pixelflow-search (public docs linking private items, unresolved links); the
   error lines were counted crate by crate. pixelflow-codegen gives 0. CI runs this check for codegen
   only. Fix: keep the codegen check as CI runs it, and for ir/search require no rustdoc warning beyond
   the base's set.

## Non-blocking, ranked

5. **New doc links that rustdoc rejects.** These would fail the corrected "no new warnings" gate from
   item 4:
   - `Bound::discover` is `pub`, and its doc links ``[`Fold::count`]``, which is `pub(crate)`
     (spec.md:301). That is a `private_intra_doc_links` error even with `--document-private-items`;
     the probe confirmed it.
   - Several docs link ``[`BoundRefusal::…`]`` from files that do not import `BoundRefusal`:
     `push_reduce` (spec.md:708, 716, 720), `close_over` (877) and `EGraph::add` (1355).
     `arena.rs:29` gains only `BoundEnd` (spec.md:754), and `graph.rs` imports nothing from `fold`.
   - `fold_cost`'s doc links ``[`FoldHeader::priced_trips`]`` (spec.md:1603), but `cost.rs` imports
     only `Fold, KnownRange, OpKind`.

   Each of these is a broken link. Fix: use explicit paths, and put `Fold::count` in backticks without
   a link.

6. **N4 is not carried at `Kernel::over`.** The ruling says `BoundOutOfDomain` is refused at the user
   surfaces, naming `Kernel::over`'s literal range (rulings.md:9). J13 keeps `KnownRange::new`'s "ends
   at most at…" message there instead (spec.md:972-974, 2429-2433). The plan text the spec writes
   (spec.md:2265-2269) says `BoundOutOfDomain` is refused at `Kernel::over`, which would be false as
   built. Either Kernel::over refuses by that name, or the ruling and the plan text say "refused, with
   `KnownRange::new`'s message".

7. **T-S-6 can pass without testing anything, against N6.** The test asserts that *every* slot-0
   `Reduce` reachable from `out_root` has an acceptable `hi` (spec.md:2093-2097). Nothing asserts
   that one exists, yet N6 says the test "asserts the arena holds the fold" (rulings.md:11).

   The case where none exists is the likely one:
   - the inner value-bounded fold is priced at `ESTIMATED_TRIPS` (32) per outer trip;
   - the peeled forms price at their real counts 0..3;
   - so extraction likely picks the unrolled outer fold.

   Fix: add an existence assertion and say which shape is expected.

8. **T-IR-24's display assertion is wrong.** It expects `add_4[0.._)(` (spec.md:1999). `fmt_expr`
   writes `reduce[{header}](` (`arena.rs:2075`), as the pin at `arena.rs:2616` shows, so the text is
   `add_4[0.._)](`.

9. **One B1 hit has no new text.** B1 lists `insert.rs:521` as reworded "as §2.10–§2.12, §3, §6, §8
   and §12 give" (spec.md:2218). No section gives text for it. T-S-11 keeps the `ValueBounded` term
   whose doc says "the arena cannot build one before A3" (spec.md:2062-2067). The implementer has to
   write that sentence, and if left as is it fails the B1 grep.

10. **§5's list of the rows A3 rewrites is incomplete, and the generator has no home.** Three hits
    whose trimmed text the spec's own code changes are missing from the list:
    - `arena.rs:1757`: `if let …` becomes a match arm (spec.md:909);
    - `extract.rs:1684`: `ENode::Reduce { .. }` becomes `{ header, .. }` (spec.md:1661);
    - `insert.rs:173`: the binding list changes (spec.md:1370).

    The generator will report each as both STALE and UNCLASSIFIED. Separately, the spec never gives
    the generator's or the checklist's path or format, and the table has no trimmed-text column for
    the generator to key on.

11. **Unused imports that the spec leaves in place.** `cost.rs:20`'s `KnownRange` becomes unused once
    `fold_cost` takes `u64` trips. The spec says the test's own `KNOWN_RANGE` import "may stay"
    (spec.md:1619-1620). After T-S-14, though, `range` at `cost.rs:760` is gone, so that import is
    unused too. Both are unused-import warnings, which fail clippy. They must be removed.

12. **A mask that the folder produces still panics at extraction.** `push_reduce` refuses a mask only
    when the bound is a mask *literal* (spec.md:729, 746-751). A mask-valued bound term passes
    construction; `user_bound` stops one only at `over_values`. Once the folder rewrites that term to
    `Const(Mask)`, `KnownBound` and `literal_twin` decline, but `embed`, through `push_reduce`, panics
    with `BoundSort` (spec.md:1737-1738). That is the extraction panic N4 meant to rule out. It is
    reachable only by building the arena by hand. The spec should state it or close it.

13. **N3's mechanism differs from the ruling's wording.** The ruling has the extractor pass the
    effective trips to `fold_cost` and to `below` (rulings.md:8). The spec instead hands a
    literal-twin `ENode` to `node_cost` and to `below`, and leaves `below`'s signature unchanged (J7).
    The prices are the same. This needs JP to acknowledge it.

14. **N1: one `Ref` costs a whole-arena pass on every push.** `variance_from` resolves each `Ref` it
    reaches through `referent_variance`, which runs `compute_arena_variance` over the referent's whole
    arena (`variance.rs:279-283`). That happens on every push and every re-push by a copier. Reachable
    only by building the arena by hand.

15. **Docs under changed code that the spec misses:**
    - the `ENode::variance` doc (`node.rs:509-512`, "a fold removes the index it binds");
    - `fold_rules.rs`'s module doc, which lists four laws and not discovery;
    - `fold_cost`'s kept paragraph (`cost.rs:415`) and the test doc at `cost.rs:736`, both of which
      still say `fold_cost(fold, range, 0)`;
    - §2.11, which says only the `to_bits` test and T-C-1 expect `VALUE_BOUND` and so omits T-IR-26.

16. **Plan text.** The A3 "Done" sentence cites J5, J7 and J10 (spec.md:2295-2300), which are labels
    in an uncommitted `.wip` file. The A3 bullet's list "the `push_reduce` refusals (`BoundVaries`,
    `BoundSort`, `BoundOutOfDomain`)" (plan:567-568) is not amended, although N4 makes it false.

## Items 4, 5 and 7

- **Item 4.** §10's pins cover every row of the plan's A3 gate, apart from items 7 and 8 above.
- **Item 5.** Of §11's greps, B1 (item 1) and the `KNOWN_RANGE` / `fold_cost(` lines (items 11, 15)
  need the fixes above. The rest can pass at head.
- **Item 7.** Nothing of A4 or A6 is built. What the spec leaves the implementer to decide is items
  2, 9, 10 and 11.
