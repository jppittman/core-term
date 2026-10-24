# A5 implementation record

**Tree:** base `7e9531e` plus the `.wip/` commits (HEAD `2bbfc99`), branch `claude/integral-area-work-ojiej6`. Nothing committed,
stashed or reset. Scope: spec §2 (every row), §5 (tests), the code docs of §6.5 and §6.7, rulings R1, R2, R4, R5, R6, R7,
R8, R9, R12, R14. No `.md` file other than this one was edited by this pass (the docs agent owns the rest).

## 1. What changed, per spec row

### pixelflow-ir (§2.7)
- `arena.rs`: `FoldTerms::stride` and its doc deleted; `FoldTerms` doc says "algebra and its bounds"; `close_over` destructures
  `{ monoid, lo, hi }` and builds `Fold::new(monoid, binder)`; test helper `terms` drops `stride: 1` (doc "builds a fold").
- `kernel.rs`: `stride: 1,` deleted at the three `FoldTerms` constructors (`Kernel::over`, `Kernel::over_values`, the test).
- `fold.rs`: `Fold::stride` doc reads "only `pack` and `HalveFold` make one". `Fold::strided` and `FoldHeader::new`'s stride
  refusal kept.

### pixelflow-compiler/src/ast.rs (§2.1)
- Tree-doc `structural:` line, `FnItem::structural` and its doc, the three "structural parameter" doc phrases (`CastExpr`,
  `CastExpr::named`), and the test's `structural: Vec::new(),` deleted or edited as the table says.

### parser.rs (§2.2, §5.1, §5.3, R6, R7)
- Grammar: `VIS 'fn' IDENT '(' params ')'`, helper "-- a helper: private", `structural` production deleted, `iexpr` and cast
  comments, shadowing sentence, binding-times paragraph, refusal list and the "range over a structural parameter" sentence
  rewritten; `use crate::{BOUND_PLAN, PLAN};`.
- `convert_fn` calls `refuse_generics(&sig.generics)?` for every `fn`; `structural_parameters`, `refuse_a_helpers_generics`
  and `is_usize` deleted; closure sugar's and the `FnItem` literal's `structural` deleted.
- `refuse_generics` + new `generic_refusal` (spec §1.1, wording per R9): a `const` generic says "a count does not make a
  program of its own ... (§1.2 of {BOUND_PLAN})" and carries the unchanged help line; any other generic says "nothing in a
  kernel is generic" (§1.3 of {PLAN}); a `where` clause says "there are no generics to bound". `convert_const` keeps its
  `refuse_generics` call (R7); the doc says syn gives a generic `const` the item refusal first.
- R6: the `Expr::Call` arm refuses a one-segment callee whose segment carries arguments, at the arguments (`::<4>`, not the
  segment: amended in Round 1, F4): "generic arguments in a `kernel!` block / note: nothing in a kernel is generic, so no `fn`
  takes generic arguments (§1.3 of {PLAN}) / help: a count is a literal, or a `usize` const of the block" (the header said "at a
  call" until Round 1, S4). One message, no branch on the argument's kind. `leading_colon` is untouched.
- Tests: `an_entrys_const_generics_are_its_structural_parameters` deleted; collection case 2 rewritten to
  `pub fn f(pieces: [Row; 4])`; `a_fn_signature_is_plain` keeps 3 cases; new
  `generics_are_refused_and_a_const_generic_says_what_a_count_is` (spec §5.3 verbatim, plus R6's three call-site pins).

### symbol.rs / sema.rs (§2.3, §2.3.1)
- `SymbolKind::Structural`, `register_structural` (table and sema), the table row, the doc sentence, `FnAnalyzer::structural`,
  the `new` loop, the shadowing arm and its doc, every "structural parameter" phrase in docs and notes deleted or edited;
  resolution rules renumbered (6 become 5, 7 become 6); `use proc_macro2::Span; use std::ops::Range;`.
- `Count`, `Count::tokens`, `Bounds`, `StructuralRange`, `known_usize` deleted. `UsizeScope::{usize_named, eval_usize}`
  return `u64` (`symbol: &str`, same overflow and division texts); `range_bounds` returns `Range<u64>`; `RangeScope` is
  `{ consts }` (struct kept); `ConstEvaluator::usize_named` returns `u64`; `eval_f32`'s `Cast` arm is `Ok(self.usize_named(name)? as f32)`;
  `ConstValue::Usize(self.eval_usize(&item.init)?)`; `range_scope` is `RangeScope { consts: self.consts }`.
- Tests: `a_structural_parameter_is_a_count` and `a_structural_parameter_is_not_a_value_and_is_its_entrys` (and their section
  header) deleted; `a_range_from_usize_consts_is_evaluated_at_expansion` asserts `range_bounds(..) == 3..14`.

### lower.rs (§2.4, R5)
- `FoldRange`, `Site::count` and `Expansion::count`, `OPEN_RANGE`, `HOLE_STRIDE`, `Holes`, `Lowered`, `ExpansionFold` deleted.
- `Site::open_fold(&mut self, depth, monoid, range: Range<u64>) -> Self::Term` (infallible); `Expansion::lower` returns
  `Result<(ExprArena, ExprId), String>`; `Expansion { arena, open: Vec<(OpenFold, FoldTerms)> }`; `Expansion::open_fold`
  builds `FoldTerms { monoid, lo: Lit, hi: Lit }`; `close_fold` pops `(open, terms)`.
- R5: `Frame<S>` (no lifetime; `structural` field gone), `Lowering`'s field follows; `literal_range(range: Range<u64>)`
  destructures `let Range { start: lo, end: hi } = range;` before `Some(lo..hi).filter(KnownRange::admits)`.
- `lower_fold` is spec L1; `lower_cast` and `resolve` lose their structural branches; module, `lower_fold`, `lower_cast`,
  `inline` docs and the resolve comment edited; test imports `{Lit, Placeholder}`; three template tests deleted;
  `lower_unanalyzed` returns `Expansion::lower(..)` directly.

### emit.rs (§2.5, R4, R8)
- `instantiated_range`, `structural_generics`, `EntryContext`, `Staged::count` deleted; imports per spec (`Fold` moves to the
  test module, `use std::ops::Range;` added).
- `entry_arena` is M1 (the template exemption is deleted, so every entry that takes no kernel reaches the optimizer);
  `arena_to_tokens(arena, root, arguments: &[TokenStream])`; `node(arena, id)`; R4's `# Panics` reads "is not `arguments`
  one for one".
- `node`'s `Param` arm panics with R1's text ("a `Param` is a rewrite rule's metavariable, substituted before any arena
  reaches emission; lowering builds none"); the `Reduce` arm is the former value-or-literal branch alone.
- `Staged::fold_terms(monoid, range: Range<u64>)` returns the bare `FoldTerms { monoid, lo, hi }` tokens (no `__range`
  block, no `stride`); `Staged::open_fold` is infallible; R8: the `open` field doc reads "the variable holding each fold's
  index, and the `FoldTerms` it closes into".
- `emit_entry` and `emit_args` emit `#vis fn #name (..)`, `#vis struct #args {`, `impl #args {`.
- Tests: `a_templates_holes_are_filled_by_the_instantiation` deleted; `a_record_and_an_args_record_expand_to_host_structs`
  rewritten over `inside` (not generic); `a_template_is_offered_to_no_optimizer` becomes
  `every_entry_that_takes_no_kernel_is_offered_to_the_optimizer` (E1, asserts 2); the two value-bound tests call
  `node(&arena, fold)`.

### lib.rs (§2.6, §6.5)
- `BOUND_PLAN` added beside `PLAN`; `# Folds` sentence, `# Binding times` ("two times", the Structural bullet deleted, the
  count paragraph, the `rings` example over `const N: usize = 4;` with no turbofish), and pipeline step 4 rewritten.

### Integration tests (§5.1, §5.2, §6.7, R12)
- `binding_times.rs`: module doc per §6.7; `const N: usize = 4;` in the block; `ramp(scale)`, `rings(..)`; the instantiation
  test renamed `every_call_of_an_entry_is_one_program` (the `assert_ne!` deleted); the `kernel_raw!` block `grid`, the two
  structural-range tests and `mod lower_case` deleted; `use pixelflow_compiler::kernel;`; the two rebinding tests drop the
  turbofish.
- `items_block.rs`: `const N: usize = 3;` in the no-prelude block; `swept(..)`, `applied(..)`, docs, test calls.
- `kernel_typed_parameters.rs`: `DOWN` and `ONE` consts, `mean_down` (E2) and `mean_down_one` (R12); the test is
  `a_fold_over_a_usize_const_applies_its_argument` and asserts 9.5 and 8.0.
- `rustc_is_the_oracle.rs`: §5.2's block with `FIRST`, `N`, plus `ONE` and `mean_index_one` (R12), twins `rust_mean_index`
  and `rust_mean_index_one`; the test is `a_count_in_a_ranges_const_arithmetic_folds_as_rustcs_does` (7.5 and 3.0).

### Elsewhere (§2.8)
- `pixelflow-codegen/src/program/lower.rs`: the `Param` panic reads "ExprNode::Param({}) reached the JIT emitter -- a rewrite
  rule's metavariable is instantiated before a Kernel exists".
- `pixelflow-graphics/src/fonts/loop_blinn/program.rs`: "Choosing a glyph's program is choosing by its piece count: each
  count its own program."
- `pixelflow-compiler/intent.md` (§6.4): the docs agent's.

## 2. Departures from the spec and rulings, each with its reason

1. **R6's three pins live in §5.3's one new test**, as a fourth loop (the ruling allows "§5.3's new test, or a second parser
   test"). Reason: the accounting stays at +1 test, and the test's own doc now says a call's generic arguments are refused
   too. The test's name is the spec's.
2. **`.wip/a5/public-surface.md` was not written.** Reason: the ground rules forbid any `.md` edit but this file. Its content
   is section 5 below; the coordinator can move it.
3. **R14's BACKLOG row is not written here** (a `.md` edit; the docs agent's). The probe's proposed row is in
   `/home/user/pixelflow/.wip/a5/r14-probe.md`.
4. **`CheckedOp`'s doc says "two counts", not "two known counts"** (the spec marks the item K, not its words). Reason: after
   this change every count is known, so "known" carries nothing.
5. **`refuse_shadowing_an_item`'s `match` kept one arm** (`Some(SymbolKind::Const)` then `_ => None`) after the spec's deletion
   of the structural arm. Superseded in Round 1 (C3): it is now the `is_some_and` guard the `fn` check below it already was.
6. **No other departures.** Every row of §2.1 to §2.8 was applied as written, R1 and R4 to R9 verbatim.

## 3. R14: the hygiene test's reason

`a_structural_parameter_named_as_the_emissions_locals_is_its_own` is deleted with `mod lower_case`. Its subject, a `const`
generic captured as a constant pattern by `instantiated_range`'s `__lo`/`__hi`, went with the generic.

**Reason recorded (corrected by the probe in Round 1):** a block's `pub const` is emitted as a module item, so it is in scope in
every host `fn` the block emits, and a `let` of the same name in the emission is then a constant pattern, which rustc refuses
as **E0308** ("`__lo` is interpreted as a constant, not a new binding"). The names that collide:

- `__lo`, `__hi` and `__header`, from the `Reduce` arm's `let __header = ..; let (__lo, __hi) = __header.attach(..)`, so only
  beside an entry with a fold;
- `__arena` (E0530 + E0308) and `__root` (E0308), beside any entry, fold or not;
- the numbered locals (`__u<n>`, `__e<n>`, `__t<n>`, `__f<n>`, `__k<n>`) have the same shape in the code and were not probed.

Measured on the base for all but the numbered locals, and again on the A5 tree for every collision but `__root`. It predates A5, and
it is a compile error at the author's `kernel!` call (the span is the whole call, not the `const`), loud, so by CLAUDE.md's
priority it earns no type and no new refusal. BACKLOG C11 holds the repro.

**Pinned, on the base and again on this tree** (scratch crate `/home/user/scratch/a5-r14-probe`, one integration-test target
per case; the A5 run is after every edit above):

| case | base `7e9531e` | A5 tree |
|---|---|---|
| control (`pub const N`, a 3-term fold) | passes | passes |
| `pub const __lo: usize = 2;` + the fold entry | E0308 x2 ("`__lo` is interpreted as a constant, not a new binding") | identical |
| `pub const __hi` | E0308 x2 | identical |
| `pub const __header` | E0308 + 2x E0599 | identical |
| `pub const __arena` beside a fold-free entry | E0530 + 2x E0308 | identical |
| entry parameter `__lo` (also `__hi`, `__header`) | passes | passes |
| entry parameter `__arena` | E0308 at the parameter | identical |

The base was re-run here for `control` and `const_lo` before editing (E0308 reproduced); the other base cells are the
coordinator's `r14-probe.md`. The A5 column was measured here after the edits, not inferred, for `control`, `const_lo`,
`const_hi`, `const_header`, `const_arena_no_fold`, `param_lo` and `param_arena`; `param_hi` and `param_header` were not
re-run on the A5 tree (their base cells stand from `r14-probe.md`). The ruling's original wording said E0530/E0005; the probe
found E0308 (E0530 only where the emission's binding is `let mut`, i.e. `__arena`; E0005 appears nowhere), and a parameter named
`__lo`/`__hi`/`__header` is fine (only `__arena` collides with one). Read from the code, not measured: only a `pub const` is
emitted by `emit_kernel` (its `filter(is_pub)`), so a private `const __lo` reaches no emitted scope.

## 4. Test counts

Before: 306 `#[test]` in `pixelflow-compiler`. After: 298 (`grep -c '#\[test\]'` and the sum of `passed` agree: 298, on both
tiers, plus 6 ignored doc-tests). `pixelflow-ir` is unchanged (its tests edit a field initializer).

| File | Before | Removed | Added | After |
|---|---|---|---|---|
| src/parser.rs | 43 | `an_entrys_const_generics_are_its_structural_parameters` | `generics_are_refused_and_a_const_generic_says_what_a_count_is` (holds R6's pins) | 43 |
| src/sema.rs | 61 | `a_structural_parameter_is_a_count`, `a_structural_parameter_is_not_a_value_and_is_its_entrys` | | 59 |
| src/lower.rs | 21 | `a_structural_parameter_leaves_the_template_open`, `a_placeholder_no_hole_names_is_refused`, `a_known_fold_is_no_hole` | | 18 |
| src/emit.rs | 8 | `a_templates_holes_are_filled_by_the_instantiation` | (rename only) | 7 |
| tests/binding_times.rs | 9 | `each_structural_range_is_a_hole_of_its_own`, `a_structural_parameter_named_as_the_emissions_locals_is_its_own` | (rename only) | 7 |
| all others | 164 | | | 164 |
| **Total** | **306** | **9** | **1** | **298** |

R12 added no `#[test]`: `mean_down_one` is asserted inside `a_fold_over_a_usize_const_applies_its_argument`, and
`mean_index_one` inside `a_count_in_a_ranges_const_arithmetic_folds_as_rustcs_does` (both counted in "all others").
Renamed, not counted: emit `a_template_is_offered_to_no_optimizer` to `every_entry_that_takes_no_kernel_is_offered_to_the_optimizer`;
binding_times `every_call_of_an_entry_at_one_instantiation_is_one_program` to `every_call_of_an_entry_is_one_program`;
kernel_typed_parameters `a_structural_entry_applies_its_argument` to `a_fold_over_a_usize_const_applies_its_argument`; rustc
oracle `a_structural_count_folds_as_rustcs_const_generic_does` to `a_count_in_a_ranges_const_arithmetic_folds_as_rustcs_does`.
Per-file counts of the lib target: 142 (`cargo test -p pixelflow-compiler --lib`).

## 5. Public surface (what `public-surface.md` would say)

`pixelflow_ir::arena::FoldTerms` lost its `pub stride: u64` field. Its constructors outside `pixelflow-ir` are the compiler's
`Expansion::open_fold` and the tokens `Staged::fold_terms` emits (through `::pixelflow_core::__macro::ir::arena`), both
changed here; `grep -rn FoldTerms --include=*.rs .` shows no other builder (hits: `arena.rs`, `kernel.rs`, compiler
`emit.rs`/`lower.rs`). `pixelflow-compiler` loses `pub struct EntryContext`, `Holes`, `Lowered`, `FoldRange` (all inside
private modules; `arena_to_tokens` changes signature in the same private module). No change of visibility anywhere.
The `kernel!` language surface: `pub fn f<const N: usize>(..)` and every other `fn` generic is a parse error; a call with
generic arguments is a parse error.

## 6. Commands run and their result lines

| Command | Result |
|---|---|
| `cargo test -p pixelflow-ir` | 6 targets, all ok: 310, 1, 75 (1 ignored), 9, 1, 1 (1 ignored) passed; 0 failed |
| `cargo check -p pixelflow-ir --no-default-features` | `Finished dev profile [unoptimized + debuginfo] target(s) in 1.20s` |
| `cargo test -p pixelflow-compiler --lib` (twice: first build, and after the doc reflow and R8) | `test result: ok. 142 passed; 0 failed; 0 ignored` |
| `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` | every target ok; 142 + 5 + 7 + 6 + 14 + 6 + 12 + 5 + 6 + 30 + 16 + 10 + 30 + 9 = 298 passed, 0 failed; doc-tests 0 passed, 6 ignored |
| `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler` | identical counts, 298 passed, 0 failed |
| `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler --tests` (after the last edits) | every target ok (re-run after the doc reflow) |
| `cargo test -p pixelflow-core` | 10 targets ok: 68, 1, 7, 13, 16, 16, 3, 6, 6, 0 (1 ignored); 0 failed |
| `cargo test -p pixelflow-graphics --lib` | `test result: ok. 121 passed; 0 failed; 3 ignored` (99.7 s) |
| `cargo build -p pixelflow-codegen --examples` | `Finished dev profile [unoptimized + debuginfo] target(s) in 18.67s` |
| `cargo fmt --all` then `cargo fmt --all -- --check` | clean; touched no file outside the edited set |

Order, as §7 asks: the IR edits and their tests first; then every source edit of §2.1 to §2.6 and the unit tests of §5.1
and §5.3 before the compiler's first build (`cargo test -p pixelflow-compiler --lib`, which builds only the lib target);
then the integration tests. A `cargo check --lib --tests` between the two showed `binding_times.rs` and `items_block.rs`
refused at their `<const N: usize>` entries, which is the new refusal working; it needed no source fix.

`cargo clippy -p pixelflow-compiler -p pixelflow-ir --all-targets -- -D warnings` on the default toolchain (1.97) fails in
`pixelflow-ir/src/dag.rs` (`must_use_candidate`, three sites, untouched by this change) before it reaches the compiler. The
gate is `cargo +1.99.0 clippy --workspace ...` (spec §7 step 6, the coordinator's); this pass did not run it, to keep the
shared target directory's free space (5.6 GB at the end).

## 7. Greps (spec §7), at the end

- **G1** (deleted names): hits only in `docs/language/2026-10-20-a-bound-is-a-value.md` (:31, :32, :231, the rewritten §3.4
  sentence at :417-418, and the A2a bullet at :574 and :581, whose line numbers moved with the docs agent's edits). Nothing
  in code.
- **G2** (`structural` in the compiler crate): nothing.
- **G3**: only `src/lib.rs:356`, `macro_tier`'s "the template vocabulary".
- **G4**: only `src/emit.rs`'s `node` panic arm (:443, :445).
- **G5** (step label in `.rs` and `intent.md`): nothing.
- **G6** (`stride` in the compiler crate; `stride:` in `kernel.rs`): nothing from either.
- **G7** (`FoldTerms` builders): `Expansion::open_fold` (`monoid`, `lo`, `hi`), `Staged::fold_terms`' tokens (same three),
  `Kernel::over`, `Kernel::over_values`, the kernel.rs test, and the `terms` test helper: none has `stride`.
- **G8** with R2's `grep -v 'as_chunks::<'`: only the parser's new test (`<const N: usize ...` cases and R6's `h::<4>`,
  `k::<4>`).
- **G9**: nothing.
- **G10** (live docs): not mine; the docs agent's.
- Scratch probes only under `/home/user/scratch`; nothing under `/tmp/claude-0` or `.wip` deleted.

## Round 1 (rulings `rulings.md` "Round 1", reviews `review1.{closure,semantics,fidelity}.md`)

**Tree:** `38703e0` (the A5 commit) plus the working-tree edits below, 18 files, nothing committed, stashed or reset. Paths are
`/home/user/pixelflow/<path>`, line numbers are of the tree now.

### C1 (blocking): the four `Param` docs

Grep of who builds a `Param`: `kernel!` builds none (`push_param|ExprNode::Param` in `pixelflow-compiler` hits only `emit.rs`'s
panic arm); the producers left are `arena_pat!`'s `par N` leaf in the rule-template DSL (`pixelflow-search/src/egraph/rewrite.rs:42`,
used by no rule today), the corpus reader's `TAG_PARAM`, and copies (`term_arena.rs`, `nnue/mod.rs`, `extract.rs:958`). The consumers:
`insert.rs:129-130` (held under `Vocabulary::Templates`, `Declined::Param` under `Runtime`), `graph.rs:1847` (derivative zero),
`graph.rs:1720` (a rewrite's right-hand side panics on one), `codegen/src/program/lower.rs:373` and `compiler/src/emit.rs:440` (both
panic). Each sentence below says only that.

- `pixelflow-ir/src/kind.rs:219` `OpKind::Param`: "An unbound scalar slot of a rewrite-rule template", then `kernel!` builds none,
  the JIT emitter and the compiler's emission panic on one, the e-graph holds one only under `Vocabulary::Templates` and
  `Runtime` declines it. The "means a builder was never called" clause is gone.
- `pixelflow-search/src/egraph/node.rs:248` `ENode::Param`: same, and "A builder substitutes it away ... so only the macro tier's
  ..." is replaced by the panics and "Only a `Vocabulary::Templates` graph may hold one".
- `pixelflow-search/src/egraph/insert.rs:52` `Declined::Param`: "the macro tier holds params natively" is now
  "`Vocabulary::Templates` holds one as `ENode::Param`"; the `insert` fn doc beside it (`:89`) lost "a builder should have
  substituted it" for the same reason (it is in the file being edited and said the same thing).
- `pixelflow-search/src/runtime.rs:150`: "an unbound slot of a rewrite-rule template".
- `ir_insert.rs:174`'s message is not changed, as ruled.

**Departure 1 (words):** the docs say "an unbound slot of a rewrite-rule template", not "a rewrite rule's metavariable", and
do not say "substituted". Reason: a template's metavariables are `Var(N)` (`arena_pat!`'s doc, `collect_metavars_arena`, which skips
`Param`), `unify` never binds a `Param` (`template.rs:456`), and nothing in the tree substitutes one, so those two words would be
false of the code. The ruling's other phrase, "an unbound slot of a rule template", is the one used.

**Departure 2 (siblings, same defect, doc-only):** A5 made each of these false as well, so each is reworded; none changes behaviour.
`pixelflow-ir/src/expr.rs:32` and `term.rs:104` (`ExprData::Param`, `Shape::Param`: "Macro parameter placeholder"); `fold.rs:466`
(`BoundNotANumber`'s doc) and `fold.rs:502` (its message: "a builder's slot ... a `Param` is filled before a kernel exists" is
"an unbound slot ... a `Param` has no value"; the one test that reads the text, `fold.rs:2151`, checks the prefix
`BoundNotANumber`, the word `` `Param` `` and "a fold's hi bound", which the new text keeps); `kernel.rs:2010` (a test doc);
`pixelflow-search/src/egraph/node.rs:599`, `extract.rs:957`, `graph.rs:1845` (comments); `tests/ir_insert.rs:148` (the test's doc;
its two assertion messages, `:167` and `:174`, still say "builder" and are left as ruled).

**Not changed, for the coordinator:** (a) `compiler/src/emit.rs:442` (R1's text) and `codegen/src/program/lower.rs:374` (§2.8's text)
call a `Param` "a rewrite rule's metavariable", which departure 1 shows is not what the code calls it; both are ruled text, and G3
constrains `emit.rs`'s. (b) The ontology's `Param` entry (`.claude/skills/ontology/language.md:356-360`) still says "an unbound
scalar slot of a builder template ... exists only before the builder is called", and quotes 2026-09-10 text on `Declined::Param`.
(c) `docs/codegen/2026-09-10-a-surviving-reduce-is-a-loop.md:600` is dated history.

### C2/F2 and F1 (blocking): the A5 Done line

`docs/language/2026-10-20-a-bound-is-a-value.md:698-707`. "which only the template set" is "which only the template set past 1"
(at the base every constructor wrote `stride: 1` but `ExpansionFold::Hole`'s). The test-template sentence is now: the ones that take
no kernel (`ramp`, `rings`, `swept`, `mean_index`, `emit.rs`'s `counted`) are written over `usize` consts and are offered to the
macro tier at expansion; the two that take one (`applied`, `mean_down`) stay `Staged`. Checked against `git diff 7e9531e 38703e0`: the
other base templates were deleted with their tests (`grid`, `span`, `mean`) or rewritten without a const (`inside`), and
`rust_mean_index` is a host function. The gate clause is appended as ruled: byte_probe empty in both columns under
`PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font programs' keys, digests and saturation telemetry, the extraction fields included,
unchanged (gates compared to the base verified at main `7e9531e`). It rests on `gates-impl.md`'s `a5-impl` outputs, taken on
`38703e0`'s code; this round changes no path an accepted program takes, only refusal texts and spans, docs and one comment-only
reflow, and the gates were not re-run.

### S1/F3 and S2 (blocking): the parser's docs

- `pixelflow-compiler/src/parser.rs:120-125` (module doc): "a generic or a `where` clause on a `fn` (a `const` generic by a message
  saying a count is a literal or a `usize` const, §1.2 of the bound plan), and on a `const`, which syn hands over as an unknown
  item and the item refusal names; generic arguments on a name, at a call or not". The last clause is S4's.
- `parser.rs:640` `generic_refusal`'s doc: "a `const` one was a count, which a block now writes as a literal or a `usize` const".
- The new test's doc (`parser.rs:2711`) drops "and no `const`". No generic-`const` case (R7).

### C3, C4

- C3: `pixelflow-compiler/src/sema.rs:756-768`, `refuse_shadowing_an_item` is the `is_some_and(|symbol| symbol.kind ==
  SymbolKind::Const)` guard; the message is unchanged (`an_item_is_not_shadowed` holds).
- C4: reflowed `compiler/src/lower.rs:795-797`, `pixelflow-ir/src/arena.rs:167-170`, `pixelflow-ir/src/fold.rs:536-541`.

### S3, S4, S5, F4: refusal shapes and the test

- S3: the `Expr::Call` arm tests `!segment.arguments.is_empty()` (`parser.rs:781`), so `h::<>(X)` is `h(X)`, as the `Path` arm
  already read `N::<>`.
- S4: the `Path` arm refuses a one-segment unqualified path with non-empty arguments (`N::<4>`, `X::<f32>`) at `parser.rs:685-686`.
  `generic_arguments_refusal` (`parser.rs:665`) is the one function that builds the error; both arms call it. It is spanned on the
  arguments (`::<4>`).
- **Departure 3 (S4, R6's header):** the message's header is "generic arguments in a `kernel!` block", not "... at a call in ...".
  Reason: the one message now stands at a value position too (`N::<4> as f32`, `X::<f32>`), where "at a call" is false. The note
  and the help line are R6's, word for word.
- S5: `parser.rs:633`, a `where` clause is refused at `where_clause.where_token.span` with `syn::Error::new`. syn's `ToTokens` prints
  an empty clause as nothing (`generics.rs:1312`), which is why the span of the clause fell back to the call site.
- F4: the new test (`parser.rs:2719`) asserts the plan path in the non-`const` generic message (§1.3 of
  `docs/language/2026-09-25-the-language-is-kernel.md`) and in the generic-arguments message, and has the Path-arm cases
  (`N::<4> as f32` over a block `const`, `X::<f32>`) beside the three call cases. It also pins S3: `h::<>(X)`, `N::<>` and `X::<>`
  parse. No `#[test]` was added.
- **R6 amended, as ruled:** the refusal is spanned on the segment's arguments, not the segment. Spans stay unpinned in the crate's
  tests. Measured outside them, in a scratch crate (`/home/user/scratch/a5-round1-probe`, the repo's `Cargo.lock`, so syn 2.0.101):
  an empty `where` is underlined at `where`; `N::<4>` at `::<4>`; `X::<f32>` at `::<f32>`; `h::<4>(X)` at `::<4>`; `(0..N::<4>)`
  at `::<4>`; and `h::<>(X)`, `(0..N::<>)`, `N::<> as f32` and `X::<>` build and run.
- BACKLOG `docs/BACKLOG.md:170`, row **C12** (one row, in "Correctness and CI" after C11): refusal spans are unpinned, the two ways a
  span-observing check could be built, and the empty `where` as the wrong span it would have caught. "S5" is not named in the row.

### F5, R14

- F5: `pixelflow-compiler/src/emit.rs:563`, `Staged::open`'s doc: "the variable holding each open fold, and the `FoldTerms` it
  closes into". Checked: the `Ident` pushed is `fold`, bound to `__arena.open_fold(#depth)`, an `OpenFold` (`emit.rs:703-712`).
  R8 is amended to that text.
- R14: section 3 above now records the probe's corrections (E0308, the whole `__` family, `__arena`/`__root` beside any entry, a
  parameter named `__lo`/`__hi`/`__header` fine).

### Commands, in this order, after the last edit unless said

| Command | Result |
|---|---|
| `cargo fmt --all` (before the build) then `cargo fmt --all -- --check` | clean |
| `cargo test -p pixelflow-compiler --lib` | `142 passed; 0 failed` |
| `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` | 14 targets ok: 142 + 5 + 7 + 6 + 14 + 6 + 12 + 5 + 6 + 30 + 16 + 10 + 30 + 9 = **298** passed, 0 failed; doc-tests 6 ignored |
| `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler` | identical, **298** passed, 0 failed |
| `cargo test -p pixelflow-ir` | 310, 1, 75 (1 ignored), 9, 1, doc 1 (1 ignored): 397 passed, 0 failed |
| `cargo test -p pixelflow-search --features provenance-journal` | 397 (8 ignored), 5, 13, 2, 7, 1, 2, 7: 434 passed, 0 failed; doc-tests 6 + 1 ignored |
| `cargo clippy -p pixelflow-compiler --no-deps --all-targets -- -D warnings` (default toolchain) | clean |
| `cargo doc --no-deps --document-private-items` on `pixelflow-ir`, `-search`, `-compiler` | no warning at any line this round touched (the existing ones are elsewhere) |
| `scripts/check-doc-paths.sh`, `scripts/check-intent.sh` | OK |

Test count: 298, unchanged (no `#[test]` added or removed; the new assertions are inside the existing parser test). The edit to
`pixelflow-ir` (docs and one message) changed the build of everything above it; `cargo test -p pixelflow-core` and `-graphics` were
not re-run because nothing they compile changed in meaning. `df -h /` before each build: 5.3 GB free, 5.0 GB after the scratch probe.
