# A6 review 1: completeness

Against `git diff 29980bf HEAD -- . ':!.wip'`. HEAD is now `6c60dfa` ("feat(compiler): A6, kernel! bounds are typed
expressions"); the working tree is clean. The review began on the uncommitted tree, and the only tracked change
since is the plan's Done line, which now carries the gate clause. Read against `.wip/a6/spec.md`,
`.wip/a6/rulings.md` (which win; D-A6-1 is arm (b)), the plan, CLAUDE.md, `pixelflow-compiler/intent.md` and
`.wip/a6/gates-impl.md`. I did not edit the repo, run a build or run a gate script. The evidence is file:line at
HEAD and read-only greps. The three other review-1 files (`laws`, `stages`, `docs`) ran the experiments; where a
finding below repeats one of theirs, it says so.

**Verdict.** Every spec section and every ruling that touches code is applied, and I cite where below. The test
accounting is exact, every plan gate test exists and asserts, `pixelflow-core` gains no public item, and the diff
touches 20 files, all of them in the spec's list. Two blocking findings remain. Both are doc sentences the diff
writes that are false of the tree, and each was a ruling's wording copied verbatim (S5 and C3). Four
non-blocking gaps follow them.

## Spec, section by section

- **§0 / §1 (semantics, refusal table).**
  - Every row of §1.4's "where each refusal is" has a code site and a test (§6.6 below). The parser rows are
    unchanged. The sema rows are M1–M7. The lowering rows are `literal_range`/`literal_bound`, the name check
    and decision 8. The host rows are the two emitted asserts. The run-time row is the guard, pinned by
    `binding_times.rs` `a_count_written_raw_outside_the_domain_is_nan`.
  - No `-ir`, `-search` or `-codegen` code changed. `git diff 29980bf HEAD -- pixelflow-ir` is doc lines only
    (`fold.rs:439-445`).
- **§2.1.**
  - `Count`: `sema.rs:1804`.
  - `UsizeScope::Output` and `operand`: `:1842`, `:1852`.
  - `RangeScope` yields `Count` and stays consts-only: `:2001-2041`.
  - `ConstEvaluator` keeps `u64`. `range_bounds` returns `[Count; 2]`: `:1953`.
  - `bound_arithmetic`: `:1777`. `use std::ops::Range` is gone from sema.rs and emit.rs.
- **§2.2.**
  - `type_of_fold` judges every name of `lo` then `hi` before `range_bounds`, then decision 8: `:961-979`.
    It does this through the shared walk `bound_names` (`:1918`), which is impl.md departure 1 and serves
    ruling S3's mirror.
  - `bound_named` `:985`, `not_a_usize_bound` `:999`, `not_in_scope` `:1170`. `resolve_ident` is flat.
- **§2.3 (decision 8):** `refuse_a_value_bound_under_a_mask` `:1974`. `takes_a_value_bound` is still `pub(crate)`
  (`fold.rs:121`).
- **§2.4.**
  - `count_parameter` (M6, M7) `:688`.
  - `ParameterKind` `:258`, `Scalar` `:271`, `parameter_kind` `:321`.
  - `-> usize` and `let n: usize` still go through `of_a_value`.
- **§2.5.** M1 `:1762`, M2 `:999`, M3 `:1777`, M4 `:1974`, M5 `:2021`, M6/M7 `:688`. Every one cites
  `BOUND_PLAN`. The widened notes are in `of_a_value`, `a_count_is_not_a_value`, `type_of_cast`,
  `convert_cast` and `generic_refusal`. The turbofish help is unchanged, per S2.
- **§2.6:** every row is applied in sema.rs, symbol.rs, ast.rs and parser.rs, including the stray duplicated doc
  line at the old `sema.rs:1893`.
- **§3.**
  - `Binding::Count`: `lower.rs:208`.
  - `lower_entry` builds by kind with no `expect` (`:414-427`, S4).
  - `Site::open_fold` takes `[Bound<Term>; 2]` (`:183`), and `Expansion::open_fold` (`:362`) pushes `FoldTerms`.
  - `lower_fold` `:638` runs the name check `refuse_a_bound_that_is_no_count` `:681`/`count_term` `:692` (S3),
    then `fold_bounds` `:707`, `bound` `:727` and `literal_bound` `:256`.
  - `lower_cast` `:738`.
- **§4 (arm (b)).**
  - `argument` `emit.rs:170`.
  - The host function: `emit_entry` `:209`, assert before `#body`, `# Panics` (`:231-238`).
  - `count_parameters` `:254`, `COUNT_DOMAIN` `:267` (LAWS-2), `count_refusal` `:273`.
  - `emit_args` `:290` asserts before `set_declared` and carries a conditional `# Panics` (`:317-363`). The
    signature stays `Result<(), ArityMismatch>` (G8: `:360` only), and `# Errors` is reworded per S3.
  - `Staged::fold_terms` `:688` and `Staged::open_fold` `:788` take `[Bound<Ident>; 2]`.
  - Every §4.8 text edit is made.
- **§5:** no code, as specified. Gates 1, 1b, 2 and 2x are identical to base, and gate 5 moved −400 Ir
  (gates-impl.md).
- **§6:** every test the spec names exists, by name. Counts are below.
- **§7.**
  - §7.1, the plan: §1.4, §3.4, the §5 row, the §6 A6 bullet and its Done line, now with the measured gate
    clause.
  - §7.2, the-language-is-kernel: `:199`, plus C3's `:59`, `:113` and `:264-266`.
  - §7.3, README `:64`. CLAUDE.md and DENOTATION are not touched.
  - §7.4: every ontology row named, with LAWS-4's Range row verbatim.
  - §7.5, `pixelflow-compiler/intent.md`: Owns, Never, Enforced by, Behind, Decided in. `pixelflow-core/intent.md`
    is not touched under (b).
  - §7.6, lib.rs `# Folds`, `# The closure form`, `# Binding times` and `BOUND_PLAN`'s doc.
  - §7.7, BACKLOG `:32`, C8 and E9 (LAWS-1).
  - §7.8, `fold.rs:439-445`.
- **§8.**
  - My reruns of G1, G2 (with S1's filter), G3, G4, G6, G7 and G8 give the expected output.
  - G5, G9 and G10 match gates-impl.md's reading.
  - One step-7 deliverable is missing: finding N2.

## Rulings

- **D-A6-1 (b):** `emit.rs:317-363`, and nothing in `pixelflow-core/src`.
- **D-A6-2/3:** `sema.rs:688`. **D-A6-4:** `emit.rs:220-243`.
- **Decision 8:** sema `:1974` and lowering `:716-721`, both by `Reduction`.
- **Gates S1/C2, C4:** gates-impl.md G2 and G9.
- **C1:** the write_into asserts are in the emit test. See N1 for how much of them it pins.
- **C3:** applied at the-language-is-kernel `:59`, `:113` and `:264-266`. See B2.
- **S2:** `sema.rs:1001-1013`. The help is chosen by kind, type, role and spelling, and the helper row asserts
  no `help:` (`sema.rs` test).
- **S3:** `lower.rs:641-646`, `:681-700`. Pinned by `(0..X + 1)` and `|r: f32| (0..r - 1)` in
  `a_bound_that_is_no_count_is_refused_here_too`.
- **S4:** `lower.rs:417-427`. **S5:** `parser.rs:69-72`, `:91-93`. See B1.
- **C7:** `sema.rs:670-674` text. `has_args_record` and the `Ty::Usize` arms are unchanged (impl.md). The oracle
  comment is at `:595`.
- **LAWS-1:** lib.rs `:162-163`, the oracle doc `:811-812`, the plan's Done line, ontology "(E9 aside)" and
  BACKLOG E9. See N4 for two sentences it missed.
- **LAWS-2:** `COUNT_DOMAIN`. "Plausible pixels" appears nowhere new.
- **LAWS-3:** no code doc says a fold rule runs at expansion. A grep for `KnownBound|HalveFold|PeelFold|FactorFold`
  in the compiler finds only bake-time mentions.
- **LAWS-4:** the ontology Range row.
- **S3/S4 (surface):** `emit.rs:350-356`, `:232-233`.
- **C5:** gates-impl.md T5/C1. **C6:** "eight test files, 79".

## Test accounting

`grep -c '#\[test\]' pixelflow-compiler/src/*.rs pixelflow-compiler/tests/*.rs` gives 318. At `29980bf` it was
299. Per file, against base:

- sema 60→63
- lower 18→20
- emit 7→8
- `rustc_is_the_oracle` 30→34
- `binding_times` 7→12
- `fold_is_kernel_over` 14→16
- `kernel_typed_parameters` 16→17
- `optimized_agrees_with_raw` 10→11

Every other file is unchanged: ast 4, lib 4, parser 43, symbol 7, and eight test files with 79 between them.
That is spec §6.7 with C6's correction. gates-impl.md records 318 passed under both `avx2` and `avx512`.
`pixelflow-core` has 0 added, as arm (b) says. The old test names (`a_range_bound_that_is_not_constant_is_refused`,
`a_usize_is_not_a_declared_value_type`) appear nowhere.

**Plan gate tests.** Each exists and asserts:

- `rustc_is_the_oracle.rs`: `a_triangular_fold_folds_as_rustcs_does` and `a_per_call_count_folds_as_rustcs_does`
  (`rust_sum(n: usize, x: f32)`, `assert_eq!` per sample).
- The `Args` refusal of 2²⁴ + 1: `binding_times.rs:240-245`, `#[should_panic(expected = "past 16777216 (2^24)")]`
  on `SumToArgs { n: 16_777_217 }.write_into`.
- `fold_is_kernel_over.rs`: `a_per_call_count_is_over_values` (key and default bits) and
  `a_triangular_fold_is_over_values_inside_over` (`assert_same_program`, binders `[5, 4]`).

**Public surface.** `git diff --stat 29980bf HEAD -- pixelflow-core pixelflow-ir pixelflow-search pixelflow-codegen
pixelflow-graphics Cargo.toml Cargo.lock CLAUDE.md docs/DENOTATION.md` shows only `pixelflow-core/README.md` and
`pixelflow-ir/src/fold.rs`, and the second is doc comments only. `git diff --stat 29980bf HEAD -- . ':!.wip'` lists
20 files, every one named by spec §2–§7.

## Findings, most severe first

### B1. Blocking. The grammar the diff writes derives no parenthesized value bound

`parser.rs:69-72` reads `range ::= '(' bound '..' bound ')'` and `bound ::= iexpr | INDEX | COUNT`. `iexpr`'s
leaves are `CONST | INTEGER` (`:38-40`), so `(n)` is not a `bound`. Sema accepts it anyway:
`sema.rs:3002` `accepted(quote! { pub fn f(n: usize) -> f32 { (0..(n))… } })`. The stages reviewer ran it under
both macros. Ruling S5's text was copied without being checked against `range_bounds`, whose `Paren` arm reads
through to `usize_named`. Fix: `bound ::= iexpr | INDEX | COUNT | '(' bound ')'`. The docs and stages reviewers
found the same.

### B2. Blocking. "A6" in the-language-is-kernel names that document's own A6

`docs/language/2026-09-25-the-language-is-kernel.md:59` and `:113` say "A6 made a `kernel!` bound a value" and
"has been a value since A6". In that document A6 is "`Select` renamed `If`" (`:725`, also `:365` and `:1271`).
`:110-112` just before uses "its A4" and "its B2" for the other plan's steps. Ruling C3's wording was applied
verbatim. Fix: "a-bound-is-a-value's A6" at `:59`, and "its A6" at `:113`. The docs reviewer found the same.

### N1. Non-blocking. Two of the host refusals have no test

- **The test cannot tell the two write sites apart.** The emit unit test (`emit.rs:1043-1081`) checks
  `code.contains(…)` for `"# Panics"`, `"past 16777216 (2^24)"` and `"KnownRange :: admits (& (0 .. n as u64))"`.
  `sum_to`'s host function satisfies each of these on its own: `emit_kernel` extends `emit_entry` before
  `emit_args` (`emit.rs:121-122`), and both share the same `count_refusal` text.
- **`write_into`'s `# Panics` section is pinned nowhere.** Deleting `#panics` at `emit.rs:356` leaves every test
  green.
- **The staged entry's host refusal is pinned nowhere.** That is spec J10's "in `Staged` entries too", and the
  `columns` entry in the same test. The only past-the-domain host tests are `binding_times.rs:230-245`, both
  over `sum_to`. A grep for `16_777_217` and `should_panic` across `pixelflow-compiler/tests` finds nothing else.
- **C1 is met only in part.** Ruling C1 asks the emit test to assert "the panic's text in the generated tokens".
  The text it finds is the host function's.
- Both refusals hold today because the code is shared, and the laws reviewer saw both panic in a scratch crate.
  Nothing fails if the sharing changes.
- **Fix:**
  - In the emit test, take the slice from `impl SumToArgs` onward and assert `# Panics` and the message there.
  - Add a `#[should_panic(expected = "past 16777216 (2^24)")]` test in `kernel_typed_parameters.rs` on
    `columns_to(&k, 16_777_217)`.

### N2. Non-blocking. `.wip/a6/public-surface.md` was never written

Spec §8 step 7 asks for it: the host function's `n: usize` and its panic, the `Args` field, and `write_into`'s
contract. `ls .wip/a6/public-surface.md` gives "No such file or directory". The content is in impl.md's "Public
surface (arm b)" section, and gates-impl.md lists it as not done. Fix: move that section into the file. The PR
body's offer of arm (a) (D-A6-1) can cite it.

### N3. Non-blocking. `pixelflow-compiler/intent.md:15-16` gives a uniform a host function

The line reads "…or is an entry's `n: usize`, a uniform whose host function and `write_into` panic past 2²⁴".
The host function and the `Args` record belong to the entry, not to the uniform. A staged entry has a count and
no `Args` record (`emit.rs` test: no `ColumnsArgs`). Spec §7.5's wording was "a uniform the host function and
`write_into` refuse". Fix: "an entry's `n: usize`, a uniform, refused past 2²⁴ by the entry's host function and
by its `Args` record's `write_into`". The docs reviewer's note on "whose host function and `Args` record" is the
same defect at `sema.rs:683-684`, `:696-697` and plan §3.4.

### N4. Non-blocking. LAWS-1 was applied in two places and missed two others

The ruling covers every place a doc states L2 for `kernel!`. lib.rs `:162-163` and `rustc_is_the_oracle.rs:811-812`
now say the identity holds "up to the … optimizer edges". Two sentences still state it bare:

- `ast.rs:619-621`, the `FoldExpr` doc block the diff edits at `:623-628`: "or the monoid's identity when the
  range is empty".
- `parser.rs:101-103`: "and its monoid's identity when the range is empty".

After A6 a `kernel!` fold can be empty per call, and there the bake's `FactorFold` accepts the infinite-`c` edge
(plan `:321-323`, decision 5). Fix: per the docs discipline, delete rather than qualify. Leave the denotation as
it is, and point once to lib.rs's `# Folds`, which carries the qualification. The docs reviewer's finding 3
(each law takes one edge, not two) decides what that one qualification should say.

## Not findings

- The plan's Done line now ends with the measured gate clause. That matches gates-impl.md: gates 1, 1b, 2 and 2x
  identical on both tiers.
- The fourth unchanged text in §2.5, the `RangeScope` `F32` arm, is now unreachable from both stages (stages
  finding 3). No comment claims it is a backstop.
