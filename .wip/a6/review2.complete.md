# A6 review 2: completeness

Against HEAD `079c01d` (`git diff 29980bf HEAD -- . ':!.wip'`), read with `.wip/a6/rulings.md` (both sections),
the four `review1.*.md` files, `.wip/a6/spec.md`, `impl.md`, `gates-impl.md` and the gate outputs. Read-only on the
repo: nothing was edited, committed or stashed, and no gate script was run. `git status --short` shows only
untracked `.wip/a6/review2.*.md`. Commands I ran are quoted with their output.

**Verdict.** Every ruling in both sections is applied, or is a spec-only ruling the code agrees with. I cite where
each one is below. Every review-1 finding is applied, refused by a ruling, or left in a sentence A6 did not write.
The mechanical checks all pass:

- The test count is 319 by `grep` and 319 by `cargo test` on both tiers.
- `pixelflow-core/src` is unchanged.
- The 20-file diff is accounted for.
- fmt and the check scripts are clean.
- The `a6-fix1` gate outputs equal base in every file a gate compares.

Two doc sentences A6 wrote are false of the code. One is the round-1 D9 defect again, in the test that pins it. The
other is `bound`'s doc, which the docs lens found too. Five non-blocking gaps follow them.

## Mechanical checks

**Test count.**

- `grep -c '#\[test\]' pixelflow-compiler/src/*.rs pixelflow-compiler/tests/*.rs` gives 319:
  - `src/`: ast 4, emit 8, lib 4, lower 20, parser 43, sema 63, symbol 7;
  - `tests/`: binding_times 12, fold_is_kernel_over 16, kernel_typed_parameters 18, optimized_agrees_with_raw 11,
    rustc_is_the_oracle 34.
  - The eight unchanged test files hold 79: bake_vs_scalar_truth 5, derivative_under_warp 6, items_block 6,
    jit_parity 12, kernel_5params 5, kernel_composition 6, kernel_macro 30, the_library_is_the_builders 9.
- Against `29980bf` the per-file changes are: emit 7→8, lower 18→20, sema 60→63, binding_times 7→12,
  fold_is_kernel_over 14→16, kernel_typed_parameters 16→18 (17 at `6c60dfa`), optimized_agrees_with_raw 10→11,
  rustc_is_the_oracle 30→34. That is 299 → 318 → 319.
- `cargo test -p pixelflow-compiler` ran with no `Compiling` line, so the binaries were fresh. It reports 149 lib +
  5 + 12 + 6 + 16 + 6 + 12 + 5 + 6 + 30 + 18 + 11 + 34 + 9 = **319 passed**, 0 failed, and 6 doc-tests ignored. The
  host has `avx512f`/`avx512dq`, so that run is the AVX-512 tier.
- `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler`: `avx2 passed 319 failed 0`.
- Every `#[test]` in the crate ran: the grep count and the run count agree.

**No public item in `pixelflow-core`.**

- `git diff 29980bf -- pixelflow-core/src | wc -c` gives `0`.
- Outside the compiler, the diff touches only `pixelflow-core/README.md` and `pixelflow-ir/src/fold.rs`. The
  `fold.rs` change is doc lines only, at `:439-443`.
- No `Cargo.toml`, `Cargo.lock`, `CLAUDE.md`, `.github` or `scripts` file changes.

**`git diff --stat 29980bf -- . ':!.wip'`: 20 files, +1664/−397. Each is accounted for:**

| File | Spec section | Ruling |
|---|---|---|
| `.claude/skills/ontology/language.md` | §7.4 | LAWS-4 Range row, `:520-523` |
| `docs/BACKLOG.md` | §7.7: plan row `:32`, C8 `:166`, E9 `:153` | LAWS-1; C13 `:171` (L-2) |
| `docs/language/2026-09-25-the-language-is-kernel.md` | §7.2 `:199` | C3: `:59`, `:113`, `:265-267`; D1 |
| `docs/language/2026-10-20-a-bound-is-a-value.md` | §7.1: §1.4 `:133-138`, §3.4 `:409-423`, table row `:495`, A6 bullet and Done line `:712-731` | N4, D3 |
| `pixelflow-compiler/intent.md` | §7.5 | complete N3 |
| `src/ast.rs` | §2.6 | D3/complete N4 at `:619-622` |
| `src/emit.rs` | §4 | the emit test, complete N1; Polish at `:267` |
| `src/lib.rs` | §4.8/§7.6 | D3 `:161-162`, D4 `:170-171`, D5 `:75`, N3 `:275-276` |
| `src/lower.rs` | §3 | S3, S4, S2 `:681-683`, D2 `:707-709`, `:1285-1289` |
| `src/parser.rs` | §2.6 | S5, S1/D6/B1 `:69-71`, `:91-93` |
| `src/sema.rs` | §2 | S2, S3, C7, D7, D8, D9, N4, N5, N8 |
| `src/symbol.rs` | §2.6 | N6 `:181` |
| `tests/binding_times.rs` | §6.2 | |
| `tests/fold_is_kernel_over.rs` | §6.3 | N2 `:1-2`, `:8` |
| `tests/items_block.rs` | §6.4 | `counted`, `CountedArgs` |
| `tests/kernel_typed_parameters.rs` | §6.4 | complete N1, `:286-293` |
| `tests/optimized_agrees_with_raw.rs` | §6.4 | |
| `tests/rustc_is_the_oracle.rs` | §6.1 | C7 comment `:592-597`; D3 `:810-812` |
| `pixelflow-core/README.md` | §7.3 | N1 `:67-68` |
| `pixelflow-ir/src/fold.rs` | §7.8 | N7 `:439-443` |

**fmt and the check scripts.**

- `cargo fmt --all -- --check`: exit 0.
- `scripts/check-intent.sh`: `OK: 51 intent files present…`, exit 0.
- `check-doc-paths.sh`: `OK: 694 source paths … (1 accepted via scripts/doc-paths-baseline.txt)`, exit 0.
- `check-cfg-encapsulation.sh`: `OK: no new … (121 pre-existing, grandfathered)`, exit 0.
- `check-emit-boundary.sh`: `self-test OK: 66 cases`, `OK: … (25 files)`, exit 0.
- `check-bin-declarations.sh`: `OK: all 29 src/bin targets…`, exit 0.
- Beyond the five, all exit 0:
  - `check-provenance-journal-scope.sh`: OK for `pixelflow-compiler` and `core-term`.
  - `check-cl-metadata.sh "feat(compiler): A6, kernel! bounds are typed expressions" 29980bf HEAD`: every commit
    subject satisfies the pattern.
  - `check-lfs-append-only.sh 29980bf HEAD`: OK.
- `cargo +1.99.0 clippy -p pixelflow-compiler --all-targets --all-features -- -D warnings`: `Finished`, exit 0.

**Gates `a6-fix1` against base** (files read and compared under `.wip/gates/out/`):

- **Identical by `cmp`:**
  - `byteprobe.a6-fix1.{avx2,avx512}.{txt,err}`, 127 lines each, `.err` empty;
  - `gate1b.a6-fix1.{txt,lines,err}`, 29 lines;
  - `gate2.a6-fix1.{txt,lines,err}`;
  - `gate2x.a6-fix1.txt`.
- **`gate1b`/`gate2` `.raw`** differ from base on one line, libtest's summary: `123 filtered out` at base against
  `125` now, and the time. Every run since `a5-main` reads 125: `gate1b.{a5-main,a6-main,a6-impl,a6-fix1}.raw`. The
  two extra tests in that binary predate A6. The compared `.txt`/`.lines` are equal.
- **`gate2.*.jsonl`** are equal once `wall_clock_us` is removed (`sed 's/"wall_clock_us":[0-9]*//'`, then `diff`
  is silent). The gates README says this field moves run to run. `gate2x_fields.py` rerun over both `.jsonl` files
  gives identical output, and that output equals the stored `gate2x.a6-fix1.txt`.
- **The gates ran on the tree that was committed.** The outputs are stamped 11:18:46–11:20:31. The last tracked
  edit (`docs/BACKLOG.md`) is 11:17:44 and the commit is 11:20:41, so the gates ran after the last edit and no gate
  was running at the commit.

## Rulings, first section

| Ruling | Applied at (HEAD) |
|---|---|
| D-A6-1 (b) | `emit_args` asserts each count before `set_declared`: `emit.rs:318-327`, then `:361-362`. The signature stays `Result<(), ArityMismatch>` at `:360`. `pixelflow-core/src` is unchanged. The offer of arm (a) belongs to the PR body, which does not exist yet; spec §4.4-4.5 holds both arms (`spec.md:863-1020`). |
| D-A6-2, D-A6-3 | M6 (helper) at `sema.rs:702-711` and M7 (closure form) at `:691-701`, pinned at `sema.rs:3277-3290`. |
| D-A6-4 | `emit_entry`'s asserts at `emit.rs:221-229` run before `#body` (`:241-242`). |
| Decision 8 | Sema: `refuse_a_value_bound_under_a_mask`, `sema.rs:1976-1981`, by `Reduction::Any`/`All`. Lowering: `lower.rs:719`. `takes_a_value_bound` stays `pub(crate)` (`pixelflow-ir/src/fold.rs:121`). |
| S1/C2 (G2) | `gates-impl.md:187`. Rerun at HEAD with the filter: nothing. |
| C1 | The emit test asserts the panic text in the generated tokens (`emit.rs:1052-1064`) and the refusal before the write (`:1066-1074`). The refusal it finds is `write_into`'s own, `KnownRange :: admits (& (0 .. self . n as u64))`. |
| C4 (G9) | Rerun at HEAD: `lib.rs:76` (`:77` before D5 cut the doc), `lib.rs:289`, `parser.rs:112`, and test assertions only. |
| S1/C3 | the-language-is-kernel `:265-267` (arity and the count panic), `:59-60` and `:113-114`. Each is true of the code. |
| S2 | The help match is at `sema.rs:1002-1013`. The items-form pin is at `sema.rs:2888-2895`, the closure-form row at `:2897-2903`, and the helper's no-`help:` assertion at `:2971-2975`. The turbofish help, `parser.rs:677-688`, carries no appended clause. |
| S3 | The name loop is at `lower.rs:639-644`, ahead of `range_bounds` at `:648`. The check is `refuse_a_bound_that_is_no_count`/`count_term` at `:684-703`. `\|\| (0..X + 1)` and `\|r: f32\| (0..r - 1)` are pinned at `lower.rs:1301-1308`. |
| S4 | `lower_entry` builds each binding by kind at `lower.rs:417-430`. No `expect` is left in non-test lowering except the two IR invariants at `:366` and `:376`, which predate A6. |
| S5 | `parser.rs:69-71`, and the legend at `:91-93`. |
| C7 | `sema.rs:672` reads "an `f32`, a count `n: usize` or a record". `has_args_record` (`ast.rs:200-202`) is byte-identical to base. The three `Ty::Usize` arms are at `sema.rs:440`, `:577` and `:1591` (base `:411`, `:548`, `:1458`) and are unchanged. The oracle comment is widened at `rustc_is_the_oracle.rs:592-597`. |
| LAWS-1, as narrowed by round-1 D3 | L2 takes the factoring edge at `lib.rs:161-162` and `rustc_is_the_oracle.rs:810-812`. L5 takes the annihilation edge at plan `:725-727`, README `:67-68`, ontology `:520-523`, and BACKLOG E9 `:153` (widened). Spec §1.3 row 0 is not edited (`spec.md:100`); the ruling wins, and the code says "may differ from rustc's in sign" (`rustc_is_the_oracle.rs:787`). |
| LAWS-2 | `COUNT_DOMAIN` (`emit.rs:267-269`) appears in the host `# Panics` (`:232-234`), in `write_into`'s `# Panics` (`:329-332`) and in the message (`:273-282`). "Plausible pixels" is in no file A6 changed. impl.md records `N as f32` past 2²⁴ ("A `usize` const read as…"). |
| LAWS-3 | `grep -rn 'KnownBound\|HalveFold\|PeelFold\|FactorFold' pixelflow-compiler/` finds only bake-time mentions: `lower.rs:32`, `fold_is_kernel_over.rs:17` and `:296`. |
| LAWS-4 | `language.md:520-523`, verbatim. |
| surface S3 | `emit.rs:348-354`. |
| surface S4 | `emit.rs:232-234`: "If a count argument is past 2^24: …" under `# Panics`, which does not claim to be the whole list. |
| surface S6, C5, C6 | Spec-only. On C5: impl.md records `cargo test -p pixelflow-graphics --lib` (121 passed) and `cargo check --workspace --all-targets` at `6c60dfa`. Round 1 changed no expansion that reaches sema's name check. `pixelflow-graphics`' one fold-bearing `kernel!` (`program.rs`) was rebuilt by the `a6-fix1` gates, and `tests/optimization_fuzz.rs` writes no fold. C6: 79, as above. |
| Docs discipline | No sentence A6 adds says anything about `Param`, a declined term, or what `kernel!` offers the optimizer, beyond the reflowed A5 list (`plan:419-423`). |

## Round 1

| Ruling | Applied at (HEAD) |
|---|---|
| S1/D6/B1 | `parser.rs:71`, `\| '(' bound ')'`. Pinned by `sema.rs:2997`, `(0..(n))`. |
| D1/B2 | the-language-is-kernel `:59` "A-bound-is-a-value's A6" and `:113` "that plan's A6". |
| D2 | `lower.rs:707-709` and the test doc at `:1285-1289`. |
| D3 | As in the LAWS-1 row above. `ast.rs:619-622` points once to `# Folds`. |
| D4 | `lib.rs:170-171`. |
| D5 | `lib.rs:75`. |
| D7 | `sema.rs:1770`. |
| D8 | `sema.rs:1954`. |
| D9 | `not_a_usize_bound`'s doc, `sema.rs:996-999`. Its test's doc was not fixed: see B1. |
| N1 | README `:67-68`. |
| N2 | `fold_is_kernel_over.rs:1-2`, `:8`. |
| N3 | `lib.rs:275-276`. |
| N4 | `count_parameter`'s doc (`sema.rs:682-687`), plan `:134-136` and `:418-419`, and M7's note (`sema.rs:696-697`, no `Args`). Three other sentences still presuppose an `Args` record: see N1 below. |
| N5 | `sema.rs:657-658`. |
| N6 | `symbol.rs:181`. |
| N7 | `fold.rs:441-443`. |
| N8 | `sema.rs:1790-1791`. |
| S2 | `lower.rs:681-683`. This is true: sema refuses a parameter, a fold index or a `let` named after a const (`refuse_shadowing_an_item`, `sema.rs:783`, `:970`, `:1634`, const arm `:830-834`). |
| S3 (subtract) | `sema.rs:2009-2012`. The `Count::Name` doc is at `:1810`. The f32-const row is still pinned at `sema.rs:2915-2921`. |
| S4, refused | No change. See N5 below on the reason it gives. |
| complete N1 | The `write_into` slice check is at `emit.rs:1075-1083`. `a_staged_entry_refuses_a_count_past_the_domain` is at `kernel_typed_parameters.rs:286-293`. The count is 319 (above). "Its domain text" is met only as "past 2^24": see N2 below. |
| complete N2 | `git diff 29980bf -- pixelflow-core/src` is empty. `.wip/a6/public-surface.md` still does not exist: see N3 below. |
| complete N3 | `intent.md:15-17`. |
| L-2 | BACKLOG C13 `:171`. Its claims hold: the depth check is at `lower.rs:655` and `collapse`'s three binders and panic text are at `passes/lattice.rs:92-103`. |
| Polish | `emit.rs:267`, "`0..=2^24`; `2^24 + 1` …". |
| Checks after round 1 | Reproduced above for the compiler tests (both tiers), compiler clippy 1.99, fmt, every check script, and the gates. I did not rerun the IR lib tests or rustdoc. |

## Review-1 findings, by file

- **complete.**
  - B1 → S1/D6/B1.
  - B2 → D1/B2.
  - N1 → complete N1.
  - N2 → complete N2.
  - N3 → complete N3.
  - N4 → D3 for `ast.rs`. Its second sentence, `parser.rs:102-103` ("its monoid's identity when the range is
    empty"), is unchanged. A6 did not write it (N4 below).
- **stages.**
  - 1 → S1.
  - 2 → S2.
  - 3, first half → S3. Second half (`bound()`'s `count_term` error after the name loop): not ruled. It is one
    shared function (`lower.rs:694`, called at `:688` and `:733`), not a second guard, so nothing is left to
    subtract.
  - 4 → refused (S4).
- **laws.**
  - 1 (the plan's `GATES: pending`) → fixed in `6c60dfa`: `grep 'GATES: pending'` on the plan finds nothing, and
    `:729-731` carries the measured clause.
  - 2 → C13.
- **docs.**
  - 1–9 → D1–D9.
  - The non-blocking items → N1–N8.
  - The §1.3 IR column → applied in round 1 (the-language-is-kernel `:199`, "a `usize` const's `Const` (as a
    value) or literal bound"), though no ruling lists it.
  - `CLAUDE.md:180` → not applied. A6 does not touch `CLAUDE.md` (N4 below).

## Findings, most severe first

### B1. Blocking. A sema test's doc restates the help rule that D9 corrected, and its own rows contradict it

**The sentence.** `pixelflow-compiler/src/sema.rs:2870-2871` is the doc of
`a_range_bound_is_a_usize_and_reads_no_coordinate`, written by A6 (`git blame` gives `6c60dfad` for
`:2867-2871`): "The help of a name that is an `f32` is the entry's `n: usize`, where an entry of the items form can
take one."

**Its own rows contradict it.**

- `:2915-2921`: `const R: f32` read in an items-form entry asserts `"declare the count as `const R: usize`"`.
- `:2912-2914`: the `let n = 4.0` row gets no help.
- `:2971-2975`: a helper's `f32` parameter asserts `!err.contains("help:")`.

**The code is the match.** `sema.rs:1002-1013` gives an `f32` const the `const N: usize` help, an entry's `f32`
parameter the `n: usize` help in the items form or the items-form pointer in the closure form, and nothing to any
other name.

Round 1 ruled the same claim blocking in `not_a_usize_bound`'s doc (D9) and fixed only that copy.

**Fix.** Mirror `:996-999`: "The help is by what the name is: an `f32` const's is `const N: usize`; an entry's
`f32` parameter's is the per-call count `n: usize` (the items form's); any other has none." Or delete the sentence.
The docs lens found the same (`review2.docs.md` finding 2).

### B2. Blocking, minor. `bound`'s doc describes only some of its calls

**The doc.** `pixelflow-compiler/src/lower.rs:728-729`, new in A6: "One end of a fold whose other end is a value:
a literal, checked alone against the domain, or the term a count binding holds."

**The calls.** `fold_bounds` returns early only when both ends are known (`:715-718`). Otherwise it calls
`self.bound(lo)` and `self.bound(hi)` (`:725`). For `(0..n)` the second call is `bound(Count::Name(n))`, and the
other end of that fold is the literal `0`, not a value.

**Fix.** "One end of a fold with a bound that is a value: …". This is the docs lens's finding 3; I checked it
against `:715-735`.

### N1. Non-blocking. Three A6 sentences presuppose an `Args` record that a staged entry lacks

Round 1's N4 put "if it has one" into `count_parameter`'s doc and the plan, and took the `Args` record out of M7.
Three sentences A6 wrote still name the record unconditionally:

- `pixelflow-compiler/intent.md:15-17`: "refused past 2²⁴ by the entry's host function and its `Args` record's
  `write_into`". This is complete N3's own prescribed wording, so the ruling reintroduced the N4 defect.
- `.claude/skills/ontology/language.md:72-74`: "in the entry's host function and its `write_into`".
- `docs/BACKLOG.md:166` (C8): "the entry's host function panics past 2²⁴, and so does its `Args::write_into`".

**The code.** `has_args_record` is `!self.params.is_empty() && !self.takes_a_kernel()` (`ast.rs:200-202`), and the
emit test asserts `!code.contains("ColumnsArgs")` for `columns(k, n: usize)` (`emit.rs:1084-1087`).

**Grading.** Round 1 graded this class non-blocking (N4), so this review does too. The docs lens reports the same
(finding 4).

**Fix.** Add "if it has one" in each.

### N2. Non-blocking. `COUNT_DOMAIN`, LAWS-2's sentence, is pinned by no test

Round 1's complete N1 says the emit test asserts `write_into`'s `# Panics` "and its domain text". The test asserts
two things in the `write_into` slice: `"# Panics"` and `"a count in this record is past 2^24"`
(`emit.rs:1080-1081`). Its whole-expansion list (`:1052-1062`) has `"past 16777216 (2^24)"` and the §1.4 citation.
None of these is in `COUNT_DOMAIN`.

**Evidence.** I searched both `pixelflow-compiler/tests` and `src` for each part of the domain sentence: `trip
short`, `name none`, `written as`, `0..=2^24` and `integer in`. Outside comments, the only hits are the definition
at `emit.rs:267-269` and two unrelated messages, `ast.rs:513` and `parser.rs:1784`. The three `should_panic` tests
expect only `"past 16777216 (2^24)"` (`binding_times.rs:230`, `:240`; `kernel_typed_parameters.rs:289`).

**Effect.** Emptying `COUNT_DOMAIN` leaves every test green. The sentence is new in A6, so this is a missing pin
rather than a lost one.

**Fix.** Add `"a count is an integer in `0..=2^24`"` to the emit test's expected list. Assert it inside the
`write_into` slice too, if the slice is meant to carry it.

### N3. Non-blocking. `public-surface.md` is still unwritten, and the ruling's one-line surface is incomplete

`ls .wip/a6/public-surface.md` gives "No such file or directory". Round 1's complete N2 states the surface in
rulings.md instead: "the generated host function's and `write_into`'s `# Panics` sections are the only surface
change".

That omits two changes:

- **Every block's `write_into`.** Every generated `Args` record's `write_into` now carries a reworded `# Errors`.
  Base read "The count is all that is checked"; HEAD reads "Of the block, only the number of arguments is checked"
  (`emit.rs:348-354`).
- **Count spellings.** An `n: usize` host parameter and a `pub n: usize` field are newly generated wherever a
  count is written.

impl.md's "Public surface (arm b)" section lists the second. Neither file is in A6's diff, so this is not blocking.
The PR body's offer of arm (a) (D-A6-1) should cite the complete list.

### N4. Non-blocking. Review-1 sentences left where A6 did not write

- **`parser.rs:102-103`.** "its monoid's identity when the range is empty" stays bare. This is complete N4's
  second sentence; round 1 fixed only `ast.rs`.
- **`CLAUDE.md:180`.** "a fold with a bound that is not one is NaN" has no E9 exception (docs review-1, non-blocking).
- **the-language-is-kernel `:54`.** "A5 deleted `kernel!`'s `const N: usize`" names the other plan's A5 in a
  document whose own A5 is "64-bit fold ends". This is the same ambiguity D1 fixed for A6's two sentences, and the
  docs reviewer pointed at it.

All three predate A6. Each is a one-line follow-up.

### N5. Non-blocking. S4's refusal cites a BACKLOG row that does not cover it

Ruling S4 refuses stages finding 4 (a literal end past 2²⁴ beside a value is reported at the `kernel!` call) with
"BACKLOG C12 covers spans".

C12 (`docs/BACKLOG.md`, row C12) is about the spans of parser and sema refusals being unpinned. Lowering's refusals
have no span at all: `lib.rs:421` is `syn::Error::new(proc_macro2::Span::call_site(), e)`. No row records that.

The behaviour predates A6 for a known pair (`literal_range`), and it is loud. A clause added to C12 would make the
ruling's reason true.

## Not findings

- **The parser grammar's continuation comment** at `:70-72` sits beside the new `'(' bound ')'` alternative, so it
  reads as describing it. The grammar itself is right.
- **The `.wip` working records lag round 1.** impl.md's LAWS-1 row still quotes "decided per call" and "two
  optimizer edges", and gates-impl.md's G9 says `lib.rs:77`. Neither file is part of A6.
- **Out of this lens.** The docs lens's finding 1 (cancellation and inverse rules also remove a fold from an
  enclosing expression, so "up to the annihilation edge" is not the whole list) contests the content of D3's L5
  sites. The sites themselves are where the table above says. I did not rerun that probe.
