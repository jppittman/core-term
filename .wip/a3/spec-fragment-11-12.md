## 11. Order of work and the grep-gated checklist

**Order** (each step leaves the crates it touches green, `cargo test -p <crate>`):

1. `fold.rs`: `BoundEnd`, `BoundRefusal`, `Fold::count`, `Bound::discover`, `ESTIMATED_TRIPS`,
   `priced_trips`, `FoldHeader::new`, the `VALUE_BOUND`/`KNOWN_RANGE` texts, docs, and T-IR-1..5,
   T-IR-8.
2. `variance.rs`: `of_var`, the `Var` arm, L4; T-IR-9 and T-IR-10.
3. `arena.rs`: `Indices`/`free_indices`/`free_index`, `push_reduce` (discover, header, L3),
   `FoldTerms`, `close_over`, `close_fold`, `lowest_free_binder(roots)`; the arena tests;
   `passes.rs`'s sibling; `tests/value_bounds.rs`.
4. `kernel.rs`: `over`'s call, `over_values`, T-K-*. (`pixelflow-ir` is now complete.)
5. `pixelflow-search`: `node.rs` (L4, `of_var`, docs), `graph.rs` (`add` backstop), `insert.rs`,
   `fold_rules.rs` (`KnownBound`, `FactorFold`, docs, `rebuild` comment), `cost.rs`,
   `extract.rs`; their tests; `tests/root_shapes.rs`.
6. `pixelflow-codegen`: the `lower.rs` comment and T-C-1.
7. **Mechanical tail**: `pixelflow-compiler` `lower.rs` and `emit.rs` (§9). This is last because
   #1427 rewrites these files (plan:483-487) and the diff is signature-only. Its own gate: the
   compiler's test suite (`cargo test -p pixelflow-compiler`, including `rustc_is_the_oracle.rs`,
   `fold_is_kernel_over.rs`, `kernel_typed_parameters.rs`) green, and byte_probe empty on the
   rows built through `kernel!`.
8. Docs (§12).
9. The gate (§10), then the greps below.

**Grep-gated checklist** (run at head; each must come out as stated):

- `grep -rn 'Reduce {' --include=*.rs pixelflow-*/src pixelflow-*/examples pixelflow-pipeline/src/bin`:
  every hit is one of §5's classified sites (already-correct, changed, or left with a reason). A
  new hit is classified before merge.
- `grep -rn 'VALUE_BOUND' --include=*.rs`: only `fold.rs` (definition, `to_bits`, and the
  `_until_a4` test), `program/lower.rs` (`known()` and T-C-1), and doc links.
- `grep -rn 'KNOWN_RANGE' pixelflow-search/src/egraph/cost.rs pixelflow-search/src/egraph/extract.rs`:
  only `cost.rs`'s test-local import.
- `grep -rnE 'until A3|before A3|A3 lifts|A3 gives|A3 carries|A3 of|\(A3\)|can be built until A3' --include=*.rs --include=*.md . | grep -v docs/language/2026-10-20-a-bound-is-a-value.md`:
  empty. Today's hits are fold.rs:556-557, 574, 687, 785-787, 1515, 1524; arena.rs:947, 1659,
  2592; passes.rs:1582, 1600; node.rs:392, 549, 709; insert.rs:521, 547;
  program/lower.rs:344; training/corpus.rs:408; `pixelflow-ir/intent.md:73`.
- `grep -rn 'fn close_over\|\.close_over(\|\.close_fold(' --include=*.rs`: every call takes
  `FoldTerms`. The call sites are kernel.rs:758 and `:1565`, arena.rs tests `:2321`, `:2349`,
  `:2367` and `:2814`, `close_fold`'s own `:1728`, compiler lower.rs:514, and emit.rs:838.
- `grep -rn 'fold_cost(' --include=*.rs`: `cost.rs` only, every call with a `u64` trip count.
- `grep -rn 'var_variance' pixelflow-search`: empty.
- `grep -n 'Variance::of_var\|Self::ALL\|Variance::ALL' pixelflow-ir/src/variance.rs`: the
  `Var` arm (`:322-328`) is `Variance::of_var(idx)`. `ALL` is spelled only in the `Param` arm
  (`:339-343`), in `of_var`'s final return, and in tests.
- `grep -n 'KnownBound' pixelflow-search/src/egraph/fold_rules.rs`: the definition, the last
  entry of `fold_rules()`, and the tests.

---

## 12. Docs A3 edits (plan §5 rows marked A3, and the code docs under changed code)

- **The plan** (`docs/language/2026-10-20-a-bound-is-a-value.md`):
  - A3's bullet gains **Done** with the facts: byte_probe empty under both ISAs; the judgement
    calls J5, J7 and J10 in one line each.
  - A4's bullet (plan:569) gains: "the key: a `Value` marker in `FoldHeader::to_bits`'s count
    field and its `from_bits` inverse; the serializers that write a header without its children
    (`cell_grid.rs`, `collapse_bench/corpus.rs`, `corpus_gaps.rs`, `test_support.rs`, the
    compiler's `arena_to_tokens`) refuse a value bound by name; the training corpus bumps its
    format version".
  - §3.1's `canonical` bullet (plan:281-283) gains "(A4)".
- `docs/language/2026-10-17-the-arena-is-typed.md`:
  - The §1.1 `Reduce` row (`:72`) becomes "`⊕ : S × S → S`, body `: S`, a value bound `: Real`
    (`Int` at T6), result `: S`; a known bound is node data".
  - T6 (`:216-217`) gains "and retypes a fold's value bounds with the binder
    (docs/language/2026-10-20-a-bound-is-a-value.md §5): it lifts `BoundArithmetic` and deletes
    the guard's integrality conjunct, measured. A known bound is node data, so T6 re-keys no
    bound literal."
- `docs/BACKLOG.md`:
  - Row `:32` (this plan): landed gains A3; next becomes A4.
  - The typed-arena row `:33` gains "T6 also retypes value bounds (a-bound-is-a-value §5)".
  - E6 `:150` gains "Since A3 the tie narrows to known ranges: a fold whose range is not known
    has no unrolling to tie with (L6)."
- `.claude/skills/ontology/language.md` "Unrolling" (`:529-552`) gains, after its first bullet:
  "Only over a known range (L6 of a-bound-is-a-value): an unknown trip count is not offered, as
  LLVM does not fully unroll one; `KnownBound` makes a range known when the e-graph discovers a
  constant bound (L8)." Its "Lives" line adds `KnownBound`.
- Code docs under changed code: everything §2–§7 names, plus `pixelflow-ir/intent.md:73` ("A
  fold's `Bound`s are all literal until A3" becomes "A fold's bound is a literal or a term;
  codegen and the key take a literal only until A4"), and `program/lower.rs:341-344`.
- **Not A3:** the ontology rows "Fold (Reduce)", "Range" and "Totality", the `DENOTATION.md`
  rows, and `CLAUDE.md:175`. These are A4 (plan:437-458).

---

## 13. Judgement calls, each with the alternative rejected

8:findings, both in the §11 grep gates, which cannot pass as written (B1, B2). Both are
28:  the entry from the `Reduce` to the body changes no node. The uniform and buffer tables
83:### B2. The §11 `Reduce {` checklist fails at head: §5 classifies about 17 of about 45 files
105:**Fix.** Paste this table into §5 so the checklist starts green. Otherwise narrow the
126:`passes.rs:571-574` records this exact pattern, an arena-sized table per site: "86 s, of
229:  be e-graph-level, with `fold_rules()` and no `ConstantFold`. That is acceptable, but
243:  "production glyphs before B2", or a message of its own.
Shell cwd was reset to /home/user/pixelflow