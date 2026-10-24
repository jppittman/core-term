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
- R6: the `Expr::Call` arm refuses a one-segment callee whose segment carries arguments, at the segment: "generic arguments at
  a call in a `kernel!` block / note: nothing in a kernel is generic, so no `fn` takes generic arguments (§1.3 of {PLAN}) /
  help: a count is a literal, or a `usize` const of the block". One message, no branch on the argument's kind.
  `leading_colon` is untouched.
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
5. **`refuse_shadowing_an_item`'s `match` keeps one arm** (`Some(SymbolKind::Const)` then `_ => None`) after the spec's deletion
   of the structural arm. Not collapsed further: the spec says delete the arm, and the shape is one `match` away from a
   future kind. Clippy was not run (see section 4).
6. **No other departures.** Every row of §2.1 to §2.8 was applied as written, R1 and R4 to R9 verbatim.

## 3. R14: the hygiene test's reason

`a_structural_parameter_named_as_the_emissions_locals_is_its_own` is deleted with `mod lower_case`. Its subject, a `const`
generic captured as a constant pattern by `instantiated_range`'s `__lo`/`__hi`, went with the generic.

**Reason recorded (the ruling's):** the emission's own `__`-prefixed locals can still meet a block's `pub const __lo`, which
rustc reads as a constant pattern, so the failure survives A5 in `node`'s `Reduce` arm (`let __header = ..; let (__lo, __hi) =
__header.attach(..)`). It predates A5 and is a compile error at the author's `kernel!` call, loud, so by CLAUDE.md's priority
it earns no type and no new refusal.

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
re-run on the A5 tree (their base cells stand from `r14-probe.md`). Corrections to the
ruling's wording, all from the probe: the code is **E0308** (E0530 only where the emission's binding is `let mut`, i.e.
`__arena`; E0005 appears nowhere); the hazard is the whole `__`-family and `__arena`/`__root` need no fold; a parameter
named `__lo`/`__hi`/`__header` is fine. Read from the code, not measured: only a `pub const` is emitted by `emit_kernel`
(its `filter(is_pub)`), so a private `const __lo` reaches no emitted scope.

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
