# A5 review 1: tests and docs fidelity

**Tree:** `7e9531e` plus the A5 commit `38703e0` (HEAD `2a3b04f`, `.wip/`-only on top). The diff against the base,
`.wip/` excluded, is 24 files, +537 −1261. Read-only on the repo. The one build was a scratch crate,
`/home/user/scratch/a5-review-fidelity`, which depends on `syn =2.0.101` alone (the repo's lockfile version).
Judged against `spec.md`, `rulings.md` (the rulings win), the plan
`docs/language/2026-10-20-a-bound-is-a-value.md`, CLAUDE.md and `pixelflow-compiler/intent.md`.

## Verdict

The tests are faithful. Every rewritten test pins what its predecessor pinned, or drops only what the spec or a
ruling drops, with the subject named. No assertion was weakened and no tolerance was widened. The count is 298,
the per-file deltas are exactly §5.4's, and no deleted or renamed test name survives anywhere in the tree.

The docs are mostly faithful. intent.md, the ontology, BACKLOG, the-language-is-kernel, the plan's §3.4, its §5
row and its A7 bullet are true of the tree after the change and follow R3, R10, R11, R13, R15 and R16. Three new
doc sentences are not true, and one spec requirement is undone:

- The plan's A5 **Done** line says the test templates "are saturated at expansion". Two of them are `Staged`.
- The Done line still lacks the measured gate clause spec §6.1 asks for, and the gates have now run.
- The parser's module doc still promises the count message for a generic on a `const` item. R7 ruled that out,
  and the semantics review found the same thing independently.
- `Staged::open`'s doc (R8's text) names the wrong variable.

The new refusal tests pin the header, the plan path (for the `const` case) and the help line. Nothing pins a span,
and nothing in the crate can. That gap has already hidden one wrong span, which the semantics review found.

## Findings, ranked

### 1. The Done line says the test templates are saturated at expansion; two are `Staged` (blocking: a doc made false)

`docs/language/2026-10-20-a-bound-is-a-value.md:700-702` reads: "The test templates are written over `usize`
consts and are saturated at expansion".

- `applied` (`pixelflow-compiler/tests/items_block.rs:141`) was `applied<const N: usize>` at the base. So was
  `mean_down` (`tests/kernel_typed_parameters.rs:101`). Both take `k: impl Fn(f32, f32) -> f32`.
- `emit_kernel` sends such an entry to `Staged::lower` and never to `entry_arena` (`src/emit.rs:107-108`). The
  module doc says "No optimizer runs at expansion on such an entry" (`emit.rs:31-33`).
- Spec §4.3 says the same: "`applied` and `mean_down` stay `Staged`, never optimized at expansion". So does R12's
  new `mean_down_one`.
- Four test templates are offered to the macro tier: `ramp`, `rings`, `swept`, `mean_index`. Two are not.
- No gate checks "saturated at expansion". Spec §6.1 says "do not keep a clause no gate checked". The unit pin,
  `every_entry_that_takes_no_kernel_is_offered_to_the_optimizer`, checks "offered", and its optimizer declines.
  The docs agent used "offered" in the ontology for this reason (departure 2) but not here.

The same line says "`FoldTerms` lost its stride, which only the template set". At the base every `FoldTerms`
constructor set `stride: 1` (`git show 7e9531e:pixelflow-ir/src/kernel.rs`, three sites, and the compiler's
`Expansion::close_fold`). Only the template set a stride other than 1.

**Fix:** "The test templates that take no kernel (`ramp`, `rings`, `swept`, `mean_index`) are written over
`usize` consts and are offered to the macro tier at expansion; the two that take one (`applied`, `mean_down`)
stay `Staged`", and "which only the template set past 1".

### 2. The Done line still lacks the measured gate clause (blocking: spec §6.1, and the reason has expired)

Spec §6.1 says to fill the Done line "with what the gates measured". A4's Done line (plan :675-677) is the
pattern. `impl-docs.md` ("Not done") held the clause back because no `a5-impl` output existed. Now:

- `.wip/gates/out/{byteprobe.a5-impl.avx2,byteprobe.a5-impl.avx512,gate1b.a5-impl,gate2.a5-impl,gate2x.a5-impl}.txt`
  exist (05:37-05:39).
- `gates-impl.md` records each as IDENTICAL to base.

The Done line at plan :698-702 still ends at "…the tests whose subject was the template are deleted". This
confirms the closure review's finding 2.

**Fix:** append "byte_probe empty in both columns under `PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font programs'
keys, digests and saturation telemetry, the extraction fields included, unchanged". Word it against the base the
gates compared to (`a5-main`); no `a5-pre` was run.

### 3. The parser's module doc promises the count message for a generic `const` item (blocking: R7, a doc made false)

`pixelflow-compiler/src/parser.rs:120-122`, under "Refused here, with a span, at the token", reads: "a generic or
a `where` clause on a `fn` or a `const` (a `const` generic by a message saying a count is a literal or a `usize`
const, §1.2 of the bound plan)".

For a `const` item, syn never builds an `ItemConst`. The probe shows it:

```
$ cargo run --offline -q   # /home/user/scratch/a5-review-fidelity, syn =2.0.101
const C<const N: usize>: f32 = 1.0;           -> Item::Verbatim
const C: f32 = 1.0 where f32: Copy;           -> Item::Verbatim
const C: f32 = 1.0;                           -> Item::Const
```

`parse_items` sends `Item::Verbatim` to `refuse_item` (`parser.rs:294`), whose message is "a `item` in a
`kernel!` block", spanning the whole item (`:301-321`). So neither the count message nor the `where` message is
ever seen for a `const`, and the span is not "at the token".

R7 rules: "the spec stops claiming the user sees the generic refusal for a `const`". `refuse_generics`' own doc
says it correctly ("though syn gives such an item the item refusal first"); the module-level list does not.

The new test's doc (`parser.rs:2700`) opens "No `fn` and no `const` takes generics", but the test has no `const`
case. R7 forbids pinning one through `refuse_generics`, not pinning that it is refused.

**Fix:** "a generic or a `where` clause on a `fn` (a `const` generic by a message saying …), and on a `const`,
which syn hands over as an unknown item and the item refusal names". Then either drop "and no `const`" from the
test's doc, or add a case asserting that a generic `const` is refused as "a `item` in a `kernel!` block".

### 4. No new refusal test pins a span, and two of the three new messages' plan paths are unpinned (non-blocking: a gap in CI)

- `generics_are_refused_and_a_const_generic_says_what_a_count_is` (`parser.rs:2707`) reads every refusal through
  `refusal()`, which returns `to_string()` (`:2008-2012`). That is the message alone.
- `pixelflow-compiler/Cargo.toml` takes `proc-macro2 = "1"` without `span-locations`, so no unit test in the crate
  can observe a span. There is no compile-fail harness either.
- What the test does pin:
  - The `const` case: the header, the plan path and the help line (R9 asks for these three).
  - The non-`const` generic case: the header and a phrase from its note. Its "§1.3 of
    docs/language/2026-09-25-the-language-is-kernel.md" is unpinned.
  - R6's call-site case: the header, the note phrase and the help line. Its plan path is unpinned too.
- This gap has already let one wrong span through. The semantics review (finding on the `where` clause) found
  that `where` with no predicates spans `Span::call_site()`, and no test fails.
- The span the code chooses also differs from the record:
  - R6 says the call is "refused at the segment", and `impl.md` §1 claims "at the segment".
  - The code spans `&segment.arguments` (`parser.rs:762-763`), which is the `::<4>` alone. That is arguably the
    better token, but it is neither what was ruled nor recorded as a departure.

**Fix:**
- Assert the two missing plan paths. This is one `contains` each.
- For spans, either enable `proc-macro2`'s `span-locations` in `[dev-dependencies]` and assert each refusal's
  `err.span().start()` column, or add a compile-fail target.
- Record in `impl.md` that R6's span is the arguments.

### 5. `Staged::open`'s doc names the variable that holds the `OpenFold` as holding the index (non-blocking: R8's text, verbatim)

`emit.rs:563-564`: "the variable holding each fold's index, and the `FoldTerms` it closes into".

- The `Ident` pushed into `open` is `fold`, bound to `__arena.open_fold(#depth)`, an `OpenFold`
  (`emit.rs:704-712`).
- The index is a different variable, `self.step(quote!(#fold.index()))` (`:713`), which `open_fold` returns.
- R8 transcribed critique a6fit 8's fix, "the variable holding each, and the `FoldTerms` it closes into", and
  added "fold's index".

**Fix:** amend R8 to "the variable holding each open fold, and the `FoldTerms` it closes into", then apply it.

## Tests, one by one

| Test | Predecessor pinned | Now | Judgement |
|---|---|---|---|
| parser `generics_are_refused_and_a_const_generic_says_what_a_count_is` (new) | the six generic cases of `a_fn_signature_is_plain` | all six, plus `<const N, const M>`, helper `<T>`, `where`, and R6's three call sites | holds; see finding 4 for what it does not pin |
| parser `a_fn_signature_is_plain` | 9 cases | the 3 non-generic cases, the doc pointing at the new test | holds (§5.1) |
| parser collection case 2 | `[Row; N]` refused as a collection | `[Row; 4]` | holds (§5.1) |
| sema `a_range_from_usize_consts_is_evaluated_at_expansion` | `matches!(…, Bounds::Known(3, 14))` | `assert_eq!(…, 3..14)` | stronger |
| sema, two structural tests (deleted) | `X * N`, shadowing, collision, helper scope, declared twice | `X * N` is case 9 of `a_usize_where_a_value_is_expected_is_a_type_error`; const shadowing is `an_item_is_not_shadowed`; the rest named only the structural symbol | holds (§5.1) |
| lower, three template tests (deleted) | holes, `Param(k)`, `range_of` | a known fold as one `Reduce` is `a_fold_lowers_to_one_reduce_and_leaves_nothing_behind` | holds |
| emit `a_record_and_an_args_record_expand_to_host_structs` | generic `fn`/`Args` headers, plus five asserts | non-generic headers, the five asserts unchanged | holds |
| emit `every_entry_that_takes_no_kernel_is_offered_to_the_optimizer` | `counted<const N>` not offered (1) | `counted` over a const offered (2) | the opposite pin, J12 |
| emit value-bound tests | `attach([…])` / `attach(empty)` | same asserts, `node(&arena, fold)` | holds |
| binding_times `every_call_of_an_entry_is_one_program` | key and code equal for `ramp::<4>`; `<4>` ≠ `<5>` | key and code equal for `ramp`; the `assert_ne!` deleted | per §5.1; A6 adds the opposite pin |
| binding_times rebinding and macro-doc tests | turbofish calls | plain calls; `RECIP_RELATIVE_TOLERANCE` unchanged | holds |
| binding_times `grid`, `span` tests (deleted) | holes, the `__lo`/`__hi` hygiene | gone with the subject; R14's reason in `impl.md` §3 and BACKLOG C11 | holds (R14) |
| items_block no-prelude test | a fold over a structural count, a staged fold | over `const N: usize = 3` | holds |
| kernel_typed_parameters | `mean_down::<4>` = 9.5, `::<1>` = 8.0 | `mean_down` = 9.5, `mean_down_one` = 8.0 | holds (R12) |
| rustc oracle | counts 1, 3, 4 at `FOLD_SAMPLES`; bake 7.5 | counts 4 and 1; bakes 7.5 and 3.0; messages added | count 3 dropped by spec §5.2's rewrite (the closure review agrees); 1 kept by R12 |

## Docs, one by one (every sentence the diff adds, and those kept beside an edit)

- **The plan:**
  - §3.4 is true and names every deleted item.
  - The §5 split row follows R15. The `:186` row stays A1's, consistent with A1's Done line (departure 6, a
    sound reason).
  - A1's Done line is spec §6.1 verbatim.
  - The A7 bullet follows R13 and is true. A nit: `inside` was rewritten without any const, so "rewrote over
    `usize` consts" fits nine of the ten.
  - The A5 Done line: findings 1 and 2.
  - The A5 bullet's own list of rewritten tests (:694-696) omits the parser and lower unit tests. It predates
    A5; noted, not reported.
- **The-language-is-kernel:**
  - The banner's enumeration matches the hits (§1.1 :150-151 and :163, §1.5 :298, §1.6 :329, §1.7 :499, "What
    the macro emits" :660-661). R16's §1.6 is in.
  - The §1.3 row matches plan §5's Becomes.
  - The §1.4 note ("Until B2 the table is what the tree does, but for that `const N: usize`") is true. Departure 4
    is sound.
- **intent.md** (61 lines, budget 80): R11 applied. Owns, Never, Enforced by, Behind and Decided in are each true
  after the change. `check-intent.sh` is OK.
- **Ontology:**
  - The retired entry, R10's Entry text, the `kernel!` Lives, Args and the Binding-time pointer (R3: no "structural
    parameter" phrase) are true.
  - In `compiler.md`, "its template is one kernel" (departure 3) is sound.
  - `terminal.md` and `SKILL.md` are as specified.
  - G10 hits exactly R3's set.
- **BACKLOG:**
  - The plan row is true.
  - C11's codes, cases and sites match `r14-probe.md` and `impl.md` §3's re-run on the A5 tree.
  - Its Where cell omits `Staged::lower`'s own `let mut __arena` (`emit.rs:580-583`). That is a pointer list, so
    not reported.
- **Code docs:** every edited doc in `ast.rs`, `parser.rs` (but finding 3), `symbol.rs`, `sema.rs`, `lower.rs`,
  `emit.rs` (but finding 5), `lib.rs`, `arena.rs`, `fold.rs` (`pack` and `HalveFold` are the only stride makers:
  `passes/lattice.rs:241`, `fold.rs:1324`), the codegen panic and `program.rs` is true.
  - `lower_fold`'s "its range known here" (`lower.rs:602-603`) and `fold_terms`' "its known range" qualify a case
    the change deleted. Both are spec §2.4/§2.5 text, and A6's value bound makes the contrast live again, so they
    are not reported.

## Departures in `impl.md` and `impl-docs.md`, judged

- **`impl.md`:**
  1. R6's pins inside the §5.3 test: allowed by R6.
  2. No `public-surface.md`: a ground-rule conflict. The content is in `impl.md` §5, and the coordinator should
     move it, because spec §2.7 asks for the file.
  3. The BACKLOG row delegated: done as C11.
  4. `CheckedOp`'s "two counts": sound.
  5. The one-arm `match`: the closure review's finding 3.
  6. "No other departures": not quite, see finding 4's span. `impl.md` §1's "at the segment" is not what the code
     does.
- **`impl-docs.md`:**
  - Departures 1-7 are each sound, and each makes a sentence true that the spec's text would have left false.
  - "Not done" (the gate clause) is finding 2.
  - R16's "impl.md records that disposition" is recorded in `impl-docs.md`, not `impl.md`. A nit.

## Looked at and not reported

- The greps G2, G3, G5, G9 and G10 were re-run and give what `gates-impl.md` records.
- `check-doc-paths.sh` is OK.
- The test count is 298, and the per-file deltas are §5.4's.
- `git grep` for the 13 deleted or renamed test names (archive excluded) finds nothing.
- `Param`'s ontology "Is" still calls it a builder template's slot. The code-side twin is the closure review's
  finding 1, and the ontology entry's Lives sentence, the only line edited, is true.
