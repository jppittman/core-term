# A5 closing review: completeness and the record

**Tree:** HEAD `20d8144` on `claude/integral-area-work-ojiej6`. A5 is `7e9531e..HEAD` outside `.wip/`. Outside `.wip/`,
`26b7d1c` and HEAD are the same tree (`git diff --quiet 26b7d1c HEAD -- . ':!.wip'` holds).

**Lens:** completeness and the record. The review checks the A5 work against four things:
- the plan's A5 bullet and its Done line;
- the spec's §2 inventory and §5.4 (as amended, 299 compiler `#[test]`s);
- every ruling of every round;
- the a5-fix3b gate outputs, the diff's file list, and CI.

**Method:** this file is the only one written in the repo. A scratch crate was built against `target.noindex`. The gate
scripts were not run.

**What was run:**
- **Formatting:** `cargo fmt --all -- --check` exits 0 with no output.
- **The five check scripts:** each exits 0.
  - `check-intent.sh`: 51 intent files OK.
  - `check-doc-paths.sh`: 690 paths across 170 documents resolve, one of them through the baseline.
  - `check-cfg-encapsulation.sh`: no new violations; 121 are grandfathered.
  - `check-emit-boundary.sh`: the self-test passes 66 cases, and the 25 files hold.
  - `check-bin-declarations.sh`: all 29 bins are declared.
- **CL metadata:** `check-cl-metadata.sh "refactor(compiler): placeholder" 7e9531e HEAD` exits 0, so every commit subject
  passes. The PR title is not known here.
- **Compiler tests:** `cargo test --locked -p pixelflow-compiler` sums to 299 passed and 0 failed.
  `grep -c '#\[test\]' pixelflow-compiler/src/*.rs pixelflow-compiler/tests/*.rs` also gives 299:
  - sema has 60, which is 61, less the two structural tests, plus R3C-1's;
  - parser 43, lower 18, emit 7, binding_times 7;
  - `kernel_typed_parameters` 16 and `rustc_is_the_oracle` 30. R12's two cases are asserts inside existing tests.
- **rustdoc:** `cargo doc -p pixelflow-compiler --no-deps --document-private-items | rustdoc_norm.py --compare` gives base
  4 and head 4, with no NEW warning. `pixelflow-search` has 28 warnings, the count gates-fix2 recorded, and none of them is
  in `insert.rs`.
- **Greps:** G1 to G10 give what the spec and R2/R3 say. G2, G5, G6 and G9 are empty.
  - **G3:** only `lib.rs:358`.
  - **G4:** only `emit.rs:440-441`.
  - **G8:** only the parser test's `quote!` cases.
  - **G10:** only the-language-is-kernel, the bound plan, and ontology `SKILL.md` and `language.md`.
- **A probe:** `/home/user/scratch/a5-review4-complete/tests/param_declined.rs`, for finding 1.
- **Not run:** clippy, which the task says to skip.

## Verdict

**Not ready: one blocking finding.** Every ruling of every round is applied, or departs from its text with the departure
recorded. The spec's §2 inventory is applied row by row. The a5-fix3b gate outputs equal the base. Every file in the diff
is accounted for by the spec or a ruling. No CI check run here fails. The blocker is a sentence round 3 wrote by
ruling (D3-9) that is false of the code. Four non-blocking items concern the record.

## Findings, ranked

### 1. Blocking: "Runtime declines a `Param`, so a slot with no value never reaches the emitter" is false

**Where:** three places, all written by round 3's D3-9.
- `pixelflow-search/src/egraph/insert.rs:54-55` (`Declined::Param`): "Only [`Vocabulary::Runtime`] declines it, so a slot
  with no value never reaches the emitter."
- `insert.rs:88-89` (`insert`'s doc): "Under [`Vocabulary::Runtime`] a `Param` declines, so a slot with no value never
  reaches the emitter."
- `pixelflow-search/tests/ir_insert.rs:149-150`: "`Runtime` declines it, so a slot with no value never reaches the
  emitter."

**What the code does:** declining is what sends the term, with its `Param`, to the emitter.
- `insert.rs:22-23`, in the same file: "a caller that meets one compiles the term unoptimized".
- `saturate_pass.rs:20-21`: "Declining is ordinary: the caller compiles the term unoptimized."
- `runtime.rs:150-151` lists `Param` among the declines.
- `runtime.rs:167`: "Callers compile the original arena unchanged in that case".
- `pixelflow-codegen/src/jit_cache.rs:137-142` names "a `Param`" among the declines that "still compile".
- `jit_cache.rs:151-157`: `emit_fn` falls back with `unwrap_or((linked_arena, linked_root))` and calls `emit::compile`.
- `pixelflow-codegen/src/program/lower.rs:373-375` is where it stops: it panics.

**Command:**
```
cd /home/user/scratch/a5-review4-complete
CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  RUST_BACKTRACE=1 cargo test --test param_declined
```

**What the probe does:** it builds `X + Param(0)` with `push_param` and `Kernel::from_parts`, then calls `Lattice::eval_at`.

**Output** (kept in `param_declined.out` beside the probe):
```
thread 'a_declined_param_reaches_the_jit_emitter' panicked at .../pixelflow-codegen/src/program/lower.rs:374:17:
ExprNode::Param(0) reached the JIT emitter: a `Param` has no value
   2: pixelflow_codegen::program::lower::arena_to_schedule
  14: pixelflow_codegen::jit_cache::compile::{closure#2}
  16: <pixelflow_core::lattice::manifold::Manifold>::compile
```

**Why D3-9 does not save it:** the ruling dictated this reason, and the reason is the ruling's error, not the fixer's. It
is still a sentence the A5 diff wrote, and it contradicts `insert.rs:22-23` three dozen lines above it.

**Fix:** state the consequence instead of a purpose. For example: "Only `Vocabulary::Runtime` declines it; the term is
then compiled as written, and the JIT emitter panics on the slot." Make the same change at `:88-89` and in
`ir_insert.rs`'s doc. The asserted values (`Err(Declined::Param(3))`) and the assert messages need no change.

### 2. Non-blocking (record): round 3 has no gate or test record in `.wip`

**What the other rounds have:** `gates-impl.md`, `gates-fix1.md` and `gates-fix2.md` record each of their rounds: test
counts, clippy, fmt, the scripts and rustdoc.

**What round 3 has:**
- The tracked a5-fix3b outputs.
- A four-line harness note in the gates README.

**What that leaves unrecorded:**
- `rulings.md:261` states R3C-1's mutation check ("the test fails with the guard removed").
- `rulings.md:262` states the 298 to 299 count.
- No file in `.wip/a5` holds the evidence for either, or round 3's clippy and fmt runs.

The facts hold: this review counted 299 and saw them pass, and the sibling `review4.code.md` mutant m1 reproduces the
mutation check. A reader of the record has to re-derive them, though.

**Fix:** a short `gates-fix3.md` with the a5-fix3b table, the 299, and the mutation.

### 3. Non-blocking (record): R-A6 never reached the A6 design

`rulings.md:3-5` says the findings for A6 are "carried to `../a6/design.md` (section R-A6 below)". `design.md` was last
changed in `81cedeb`, before the rulings commit `2bbfc99` (`git log -- .wip/a6/design.md`), and it still contradicts R-A6:
- **a6fit 7:** `design.md:140` still writes `Parameter { name, record, count: bool }`. The ruling says
  `Parameter { name, kind }` with no `bool`.
- **a6fit 1:** `design.md:222` still calls the closure form's `n: usize` "open (§11, D-A6-3)". The ruling refuses it and
  pins it in §7.4.
- **a6fit 2:** §7.4 (`design.md:281-290`) has no `(0..X + 1)` case.

R-A6 therefore lives only in `rulings.md`. A6's design round must start there, or edit `design.md` first.

### 4. Non-blocking (record): spec §2.7's `public-surface.md` was never written

The spec asks for "Record it in `.wip/a5/public-surface.md`" (spec:447): `FoldTerms` loses a `pub` field.
- `impl.md:104` says the file was not written because of a ground-rule conflict.
- `impl.md` §5 holds the content.
- review1 handed it to the coordinator.

It was never moved, and `ls .wip/a5` has no such file. A4 kept this record as its own file.

### 5. Non-blocking (craft): round 3's hand edits were not reflowed

Round 1's C4 ruled reflows of this kind. Round 3's lines were left ragged:
- **Plan `:696`:** 122 columns inside a paragraph wrapped at about 90. It is D3-13's insertion: "`sema.rs`, `lower.rs`
  and `emit.rs` unit tests) and docs rewritten. It comes before A6 because the obvious lowering of a".
- **Plan `:535-536`:** the Order paragraph's "…(§5's "Hunk" column). A2a goes" breaks short.
- **Ontology `compiler.md:71`:** "arguments when it is called. An" breaks short.
- **Ontology `language.md:80`:** "when called, and optimized at bake). That" breaks short.

No check reads Markdown line widths, so nothing fails. This is a craft item only.

## Every ruling, walked

Each line below names the ruling, then where the tree applies it.

**Open call**
- **J6** (`FoldTerms::stride` deleted):
  - `pixelflow-ir/src/arena.rs:177-184`: no field;
  - `arena.rs:1743-1754`: `FoldTerms { monoid, lo, hi }` and `Fold::new`;
  - `kernel.rs:765`, `:863`, `:1684`;
  - the compiler's `lower.rs:340`.
  G6 and G7 are clean.
- **J5** (A7 is empty): plan `:713-717`.

**Gates**
- **R1:** superseded by D1. The panic at `emit.rs:441` is D1's text.
- **R2:** G8 with `grep -v 'as_chunks::<'` hits only `parser.rs:2734-2738` and `:2767-2770`.
- **R3:** G10's files are as ruled. The Binding-time pointer at `language.md:129` no longer uses the phrase.

**Code**
- **R4:** `emit.rs:355`.
- **R5:**
  - `lower.rs:430` (`Frame<S: Site>`) and `:438`;
  - `literal_range` destructures first, at `lower.rs:229`, and formats from `lo`/`hi`.
- **R6, with F4's span amendment and S4's second arm:**
  - one builder, `parser.rs:663`, called at `:692` (path) and `:788` (call);
  - spanned on the arguments;
  - the pins are at `:2767-2771`: `h::<4>`, `DX::<f32>`, `k::<4>`, `N::<4>` and `X::<f32>`.
  - The header reads "generic arguments in a `kernel!` block", not R6's "… at a call in …". This departure is forced by S4
    and recorded at `impl.md:307-309`. The help line is D3-10's.
  - `leading_colon` on an expression path is untouched, as ruled.
- **R7:**
  - `convert_const` keeps `refuse_generics` (`parser.rs:399`);
  - the doc at `:621-624` says syn gives the item refusal first;
  - the module doc at `:120-123`;
  - there is no generic-`const` test.
- **R8:** superseded by F5 and then D3-14 (`emit.rs:559-560`).

**Wording**
- **R9 then D3-3:** the note at `parser.rs:647` keeps only "a count does not make a program of its own". The help line is
  unchanged at `:648-649`. The test pins the header, the plan path and the snippet (`:2741-2744`).
- **R10 then D3-1:** `language.md:87-90` ("as the macro's optimizer left it at expansion"). This is the substance of D3-1,
  if not its exact word, "offers". `language.md:77-80` and `intent.md:54-57` use "offers".
- **R11 then D4 then D3-8:** `intent.md:9-12`. The file is 63 lines, under 80.

**Tests**
- **R12:**
  - `kernel_typed_parameters.rs:66`, `:106`, `:278`;
  - `rustc_is_the_oracle.rs:654`, `:666`, `:680`, and the assert inside the existing test;
  - the count is unchanged.
- **R14:**
  - the test is gone;
  - its reason, with the probe's corrections, is at `impl.md` §3 (`:114-150`);
  - BACKLOG has row C11 (`docs/BACKLOG.md:169`), with the repro.

**Plan record**
- **R13 (with D3):** the A7 bullet is at plan `:713-717`. The Done line is at `:699-708` and includes "the tests whose
  subject was the template are deleted".
- **R15:** plan `:492`, which the-language-is-kernel `:197` matches word for word.
- **R16:** `one-pipeline.md` is not in the diff, and `impl-docs.md:48-49` records the disposition. The banner at
  the-language-is-kernel `:57` names §1.6.

**R-A6:** see finding 3.

**Round 1**
- **C1:** replaced by D1.
- **C2/F2 and F1:** the Done line at plan `:699-708`. Every name in it was checked against the diff:
  - `ramp`, `rings`, `swept`, `mean_index` and `counted` are over `usize` consts;
  - `inside` lost its generic;
  - `applied` and `mean_down` take a kernel.
- **S1/F3:** the module doc at `parser.rs:120-123`. The test doc no longer says "and no `const`".
- **S2:** superseded by D8 (`parser.rs:640-641`).
- **C3:** `sema.rs:759-763`.
- **C4:** `lower.rs:795-797`, `arena.rs:167-170` and `fold.rs:537-541`, each reflowed.
- **S3:** `parser.rs:786-789`.
- **S4:** see R6.
- **S5:** `parser.rs:633`, at the `where` keyword.
- **F4:** the plan path is asserted at `:2757` and `:2777`. BACKLOG has row C12 at `:170`. Its claims hold: the
  manifest's `proc-macro2 = "1"` has no `span-locations`, and both `refusal` helpers return strings.
- **F5:** see R8.
- **R14's corrections:** in `impl.md`.

**Round 2**
- **D1:** each named doc says what a `Param` is now:
  - `kind.rs:219-226`, `node.rs:248-253`, `insert.rs:52-56` and `:86-89`, `runtime.rs:150-151`, `expr.rs:32-33`,
    `term.rs:104-106` and `ir_insert.rs:145-170`;
  - both panics: `emit.rs:441` and codegen `lower.rs:374`;
  - plan §4 at `:434-436`;
  - the ontology rows.
  No test asserts either panic (grep).
- **D7:** applied at `variance.rs:374-376`, `cost.rs:105`, `saturate_pass.rs:19-20`, `kind.rs:799-801` and
  `nnue/factored.rs:604` and `:777`. `arena.rs:407` was left and flagged, since it is about `Var`.
  - Re-running D7's grep (`substitut|specializ|builder` within three lines of `Param`, over every `.rs`) finds only the
    identifier `self.builder` at `arena.rs:1278`.
- **D2:** `language.md:352-358`. The history quote is gone.
- **D3:** see the Done line.
- **D4:** `lib.rs:75-76`. `BOUND_PLAN` has one use, at `parser.rs:647`, so its doc is true.
- **D5:** superseded by D3-2.
- **D6:** superseded by D3-4.
- **D8:** `parser.rs:640-641`.
- **D9:** `compiler.md:86-88`.
- **R2C-1:** `parser.rs:343`.
- **R2C-2:** `parser.rs:659-661`.
- **R2C-3:** `turbofish_of` (`:678`) is used at all three method sites (`:755`, `:1151`, `:1248`). The parser test has
  cases at `:2667-2672`.

**Round 3**
- **D3-1:** see R10.
- **D3-2, D3-11, D3-12:**
  - `binding_times.rs:4`, `sema.rs:30-32` and `parser.rs:97-100`;
  - `lib.rs:256`, "An entry's parameters are bound at one of two times";
  - `lower.rs:33`.
  Sema's `signature` doc confirms that an entry's parameter is an `f32`, a record or a kernel, so the two-way list is
  complete.
- **D3-3:** `parser.rs:647`, the test doc at `:2724-2730`, the banner at the-language-is-kernel `:54-59`, and
  `binding_times.rs:99`.
- **D3-4:**
  - the-language-is-kernel `:203-205`;
  - plan `:493`, `:535-536` and `:558-560`;
  - BACKLOG `:32`'s next cell.
- **D3-5:** `language.md:129`.
- **D3-6:** `compiler.md:67-71`.
- **D3-7:** `variance.rs:374-376`.
- **D3-8:** `intent.md:9-12`.
- **D3-9:** applied as worded at `insert.rs:54-56`, `:86-89` and `ir_insert.rs:148-150`. The wording is false: finding 1.
- **D3-10:**
  - `lib.rs:278-281`;
  - `intent.md:13-15`, which reads "built from literals and `usize` consts at expansion". It drops the ruling's "with
    `+ - * /`" and stays true.
  - The parser's count sentence is gone from `:97-103`.
  - R6's help and its assert are at `:671` and `:2778`.
- **D3-13:** plan `:695-696`.
- **D3-14:** `emit.rs:559-560`, which matches `close_fold(#fold, #body, #terms)`.
- **R3C-1:** `sema.rs:131` and the test at `:2827`. The count is 299, as shown above.
- **R3C-2:** refused as ruled. `turbofish_of`'s doc at `parser.rs:676-677` says "on a name in a body".

## The spec's §2 inventory and §5.4

**§2:** every row is applied.
- **§2.1 ast.rs, §2.2 parser.rs, §2.3 symbol.rs and sema.rs (with §2.3.1's `u64` count), §2.4 lower.rs and §2.5
  emit.rs:** each row is applied, as M1, L1, E1 and `fold_terms` are written. The deleted names are gone by G1.
- **§2.6 lib.rs:** as written, with D3's wording.
- **§2.7 pixelflow-ir:** as written, except the `public-surface.md` record (finding 4).
- **§2.8:** codegen's panic is D1's text, `program.rs:15-17` is edited, and `intent.md` is edited.
- **§2.9:** the "kept" list is untouched (`Fold::strided`, `RangeScope`, `UsizeScope`, `literal_range`, the `Bound::Value`
  children path).

**§5.1:** every surviving neighbour the table names exists:
- `a_range_from_usize_consts_is_evaluated_at_expansion`
- `as_f32_converts_a_usize_and_nothing_else`
- `a_usize_where_a_value_is_expected_is_a_type_error`
- `an_item_is_not_shadowed`
- `a_fold_lowers_to_one_reduce_and_leaves_nothing_behind`
- `a_range_from_consts_folds_as_rustcs_does`

So no deleted pin leaves its subject behind. Lowering's 2²⁴ refusal is still pinned at `lower.rs:1169`.

**§5.4:** 299, as the count above shows.

## Gates: a5-fix3b against base

Read from `.wip/gates/out`, without running anything.
- **Identical to base (`cmp`):** both tiers of `byteprobe.*.avx2.txt` and `byteprobe.*.avx512.txt` (127 lines each), plus
  `gate1b.*.txt`, `gate1b.*.lines`, `gate2.*.txt`, `gate2.*.lines` and `gate2x.*.txt` (29 lines each). Every
  a5-fix3b `.err` file is empty.
- **`gate2.a5-fix3b.jsonl`** equals the base's record for record once `wall_clock_us` is dropped (29 records).
- **The `.raw` files** differ from base only in libtest's tail, "125 filtered out" against "123", and the time.
  `a5-main`, the unmodified `7e9531e`, reads 125 too, so the base was recorded on a tree with two fewer graphics tests.
  A5 changes no graphics test.
- **The tree the run measured:** the a5-fix3b files are stamped 07:44:14 to 07:44:30. That is after `26b7d1c` (07:44:06),
  whose tree is HEAD's. The harness refuses a modified `tests.rs`, so the run could not have been on `2121763`.
- **Gate 5:** last measured at `a5-impl`. Total Ir was -0.020 percent; `saturate` and `extract_dag_over` moved +0.019 and
  -0.144 percent, against spec §4.2's "identical". `gates-impl.md:87-91` records and explains this. The gate is
  report-only and is not repeated after the rounds.

## The diff's files, accounted for

`git diff --stat 7e9531e -- . ':!.wip'` lists 39 files. Each is accounted for:
- **Spec §2 and §6:**
  - the seven compiler sources, the four compiler test files, and `intent.md`;
  - `arena.rs`, `kernel.rs` and `fold.rs`'s stride doc;
  - codegen `program/lower.rs` and graphics `program.rs`;
  - the plan, the-language-is-kernel and BACKLOG;
  - ontology `SKILL.md`, `language.md`, `compiler.md` and `terminal.md`.
- **D1's list:** `kind.rs`, `node.rs`, `insert.rs`, `runtime.rs`, `expr.rs`, `term.rs` and `ir_insert.rs`.
- **D7's list:** `variance.rs`, `cost.rs`, `saturate_pass.rs` and `nnue/factored.rs`.
- **D7's grep clause:** each of the following has "substitut", "specializ" or "builder" next to `Param` at base.
  - `fold.rs`'s `BoundNotANumber`;
  - `kernel.rs`'s test doc;
  - `egraph/extract.rs:957`;
  - `egraph/graph.rs:1845`;
  - `pipeline/collapse_bench/corpus.rs`;
  - `pipeline/training/factored.rs`.
  `impl.md:264` and its "Departure 3" (`:391`) list them.

No file is unaccounted for.

## CI

Nothing run here fails:
- fmt;
- the five scripts;
- the commit subjects;
- the compiler tests;
- the compiler's rustdoc against its base.

Not run: clippy (skipped as the task says), the macOS NEON job and the feature matrix. A5 changes no manifest and no
feature. The one CI-visible defect is a doc sentence (finding 1), which no job reads.
