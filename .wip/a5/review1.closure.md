# A5 review 1: deletion closure and dead code

**Tree:** `7e9531e` plus the uncommitted A5 working tree (24 files, `git diff 7e9531e --stat -- . ':!.wip'`:
537 insertions, 1261 deletions). Read-only; nothing built (the post-edit gate record in `gates-impl.md` is
newer than every edited file: last source edit `emit.rs` 05:35:31, `gate2x.a5-impl.txt` 05:39, `gates-impl.md`
05:53). Against `spec.md`, `rulings.md` (rulings win), plan `docs/language/2026-10-20-a-bound-is-a-value.md`,
`CLAUDE.md`, `pixelflow-compiler/intent.md` and `../a6/design.md` §10.

## Verdict

The deletion is closed in code. Every name in spec §2 and G1–G10 is gone from code and live docs, or appears
only where the gates allow it. No import, field, variant, lifetime or helper is left dead. Nothing that §2.9 or
A6 design §10 says to keep was deleted. Every removed `#[test]` is one that §5.1/§5.4 lists. No surviving pin
was lost.

The closure was not complete one level out. A5 removed the last `kernel!` producer of `Param`, and the
spec's §2.8 fixed only the codegen panic's wording. The type's own doc and three doc siblings in
`pixelflow-search` still describe `Param` as a slot of a `kernel!` builder or a `pixelflow-compiler` macro
parameter (finding 1). The plan's A5 Done line still lacks the measured gate clause spec §6.1 asks for. The
docs agent held it back until the gates ran, and they have now run (finding 2). The other two findings are
non-blocking.

## The greps, re-run

- **G1** (deleted names, `*.rs *.md *.py *.sh`, archive and `.wip` excluded): hits only in the bound plan:
  :31–32 (§0), :231 (§2's row), :417–418 (the rewritten §3.4 sentence), and :574 and :581 (the A2a bullet,
  moved from :578/:585 by the docs edits). A `git grep` over every tracked file type outside the archive and
  the bound plan finds nothing.
- **G2** `structural` in `pixelflow-compiler/`: nothing.
- **G3** `template|instantiat|\bholes?\b` in the compiler: only `src/lib.rs:356` (`macro_tier`, "the template
  vocabulary"). I widened it to `templat|instantiat|hole|stride|struct?ural|placeholder`. The extra hits are
  `whole` (several) and `placeholder` in the fold-binder sense (lower.rs, tests). None belongs to the deleted
  path.
- **G4** `push_param|ExprNode::Param` in the compiler: only `src/emit.rs:440,442`, `node`'s panic arm (R1's
  text, "substituted"). `push_param` has no hit anywhere in the compiler crate.
- **G5** `A5` as a word in `.rs` and `intent.md`: nothing.
- **G6** `stride` in the compiler and `stride:` in `kernel.rs`: nothing.
- **G7** `FoldTerms`: every builder is `{ monoid, lo, hi }` (arena.rs:1743 destructure, :2449 test helper;
  kernel.rs:765, :863, :1684; lower.rs:340; emit.rs:616's tokens).
- **G8** (R2's pipe): only `parser.rs:2709–2713, 2741, 2743`, the new test's `quote!` cases. Workspace-wide,
  `<const X: usize` appears only in ordinary Rust (`value_bounds_jit.rs:166`, `scoring.rs`, `backward.rs`,
  `pixelflow-ml/src/graphics.rs`). No `kernel!` block anywhere has a generic.
- **G9**: nothing.
- **G10** (R3): the bound plan, the-language-is-kernel, ontology `SKILL.md:59` and `language.md:149`. That is
  exactly R3's set.
- **`Param` (word) in the compiler**: the AST's `Param` struct (a `fn` parameter, unrelated), emit.rs's panic
  arm, and `tests/derivative_under_warp.rs:129`. That last one is history of a deleted test and predates
  A5.
- **Template-era phrasing** (`monomorph|const generic|generic over|per instantiation|own program|const
  block|rustc evaluates|E0158|__range|open fold|…`) in the compiler: none of the deleted path's. `rustc
  evaluates` at sema.rs:1782 and the oracle's comments are about `f32` consts.
- **Deleted and renamed test names** (`git grep` over every tracked file, archive excluded): no reference
  survives.

## What was kept, checked

- **Spec §2.9:** `Param(u8)`, `push_param` (arena.rs:839), `ExprNode::Param`, `ENode::Param`,
  `Declined::Param`, `Ty::Usize`, `SymbolKind::Index`, `register_index`, `Binding::Index`,
  `a_count_is_not_a_value`, `names_a_count`, `LiteralExpr::usize_value`, `ConstValue::Usize`, `RangeScope`
  (one field), `UsizeScope`, `range_bounds` (`Range<u64>`), `literal_range` (`Range<u64>` in), `CheckedOp`,
  `FoldHeader::to_bits`/`from_bits` in emission and `Staged`, and `node`'s `Bound::Value` children. Every one
  is present.
- **A6 design §10:** all of the above, plus `type_of_fold`'s bounds-before-`push_scope` order (sema.rs:885–888),
  `relink` in `entry_arena` (emit.rs:149–151), `argument(scalar)`, `Site::open_fold(depth, monoid, Range<u64>)`
  (infallible), `Expansion.open: Vec<(OpenFold, FoldTerms)>`, `Staged::fold_terms` without `__range`,
  `BOUND_PLAN`, and the two unit tests A6 rewrites, which are untouched by the diff. `lower_cast` still reads
  `Binding::Index` first (lower.rs:647–649), the hook A6's `Binding::Count` needs.
- **`Fold::strided`:** still used by `pack` (passes/lattice.rs:241) and `from_bits` (fold.rs:1036).
  `KnownRange::stride()` still has a production reader (fold_rules.rs:646), so deleting `Holes::range_of`
  orphaned nothing in the IR.
- **Dead code:** clippy 1.99 with `-D warnings --all-targets --all-features` passed after the last edit
  (gates-impl.md). In a proc-macro crate, rustc's dead-code lint sees through `pub` in private modules, so
  an orphaned helper would have failed it. I also read every import the diff touched. Each is used:
  `BOUND_PLAN`/`PLAN` (parser), `Span`/`Range` (sema), `Range`/`KnownRange`/`Fold`/`Binder`/`IndexSpaceFull`
  (lower), `Bound`/`Binder`/`Range`, and the test module's `Fold` (emit). `refuse_attributes` keeps two
  callers (parser.rs:397, :411). Lifetimes: `Frame<S>` lost its `'a` (R5), and `FnAnalyzer<'a>`,
  `Lowering<'a, S>` and `RangeScope<'a>` still use theirs.
- **Rustdoc:** no NEW and no GONE warnings in the three crates (gates-impl.md). The deleted intra-doc targets
  (`[`Holes`]`, `[`instantiated_range`]`, `[`Site::count`]`) had resolved, so they left no warning behind,
  and nothing links to them now (G1).

## Tests

**Removed `#[test]` functions**, computed per file from `7e9531e` against the tree: 9. They are exactly
§5.4's 9.

- parser `an_entrys_const_generics_are_its_structural_parameters`
- sema `a_structural_parameter_is_a_count` and `a_structural_parameter_is_not_a_value_and_is_its_entrys`
- lower `a_structural_parameter_leaves_the_template_open`, `a_placeholder_no_hole_names_is_refused` and
  `a_known_fold_is_no_hole`
- emit `a_templates_holes_are_filled_by_the_instantiation`
- binding_times `each_structural_range_is_a_hole_of_its_own` and
  `a_structural_parameter_named_as_the_emissions_locals_is_its_own` (R14's reason is in `impl.md` §3, the
  C11 row in BACKLOG)

The four renames are §5.4's. Count: 298, against 306 at the base.

**Pins checked for loss:**

- **`X * N` refused** survives as case 9 of `a_usize_where_a_value_is_expected_is_a_type_error`.
- **Shadowing a const by parameter, and a `fn` by `let`**, survives in `an_item_is_not_shadowed`.
- **A helper's `(0..N)` over a non-const** survives: "is not a `const`" is pinned by
  `a_range_bound_that_is_not_constant_is_refused` (5 of its 9 cases).
- **`*` and `+` in count arithmetic** survives in `a_range_from_usize_consts_is_evaluated_at_expansion`
  (`LO * 4 - 1`, `LO..LO + N`).
- **A known fold is one `Reduce`** survives in `a_fold_lowers_to_one_reduce_and_leaves_nothing_behind`.
- **`kernel_raw!` in the items form** survives in `fold_is_kernel_over.rs:144, :164`,
  `kernel_typed_parameters.rs:138` and `the_library_is_the_builders.rs:122`.
- **Count 1** stays per R12 (`mean_down_one`, `mean_index_one`).
- **The macro doc's `rings`** is token-identical in lib.rs and binding_times.rs.
- **Every call of an entry with a fold is one program** is now pinned on a saturated `ramp`.
- **Removed with their subject:** the oracle's count-3 case (the spec's choice) and the
  `ramp::<4> ≠ ramp::<5>` inequality (A6 design §7.2 adds the opposite).

## Findings

### 1. `Param`'s own doc still names a `kernel!` builder slot, and A5 removed the last one (blocking: a doc made false)

`pixelflow-ir/src/kind.rs:219`, `OpKind::Param`: "An unbound scalar slot in a `kernel!` builder, before the
builder is called." At `7e9531e` that was true of exactly one thing: the template hole. `lower.rs:475` built
`self.arena.push_param(hole)` for a structural `N as f32`, and the host function filled it
(`emit.rs:529`, `__arena.push_const(#count as f32)`). A5 deletes that hole, and G4 shows that `kernel!` now
builds no `Param`. Every remaining producer is a rewrite-rule template (`pixelflow-search/src/egraph/rewrite.rs:42`'s
`par`, `nnue/mod.rs`, `term_arena.rs`).

Spec §2.9 and the plan (:434–436) say `Param` stays as "a rewrite rule's metavariable". §2.8 and J16
rewrote the codegen panic for this reason ("After A5 the term names nothing in the language"), but the
type's own doc was missed. Three siblings say the same:

- `pixelflow-search/src/egraph/node.rs:248`, `ENode::Param`: "Unbound scalar slot of a `kernel!` builder"
- `pixelflow-search/src/egraph/insert.rs:52–57`, `Declined::Param`: "A macro-parameter slot … The macro tier
  holds params natively … an unbound slot is what a builder *is*"
- `pixelflow-search/src/runtime.rs:150`: "a `pixelflow-compiler` macro-parameter slot"

The three e-graph docs were already loose at the base, because the template exemption kept the hole away
from the macro tier. After A5 they describe a producer that does not exist. Under CLAUDE.md's "denote before
you build", the type's doc is where its meaning lives.

**Fix:** reword all four to the denotation §2.9 states: a rewrite rule's metavariable, an unbound slot of a
rule template, substituted before any arena reaches emission or a `Kernel` exists, which `kernel!` does not
build. `insert.rs`'s "the macro tier holds params natively" becomes "`Vocabulary::Templates` holds a rule
template's metavariables". `pixelflow-search/tests/ir_insert.rs:174` asserts only "a Param at bake time
means a builder was never called", which stays true of a rule template.

### 2. The plan's A5 Done line still lacks the measured gate clause (blocking: spec §6.1, and its reason has expired)

`docs/language/2026-10-20-a-bound-is-a-value.md:698–702` ends at "the tests whose subject was the template
are deleted". Spec §6.1 asks for the Done line "filled with what the gates measured": byte_probe empty in
both columns under `PIXELFLOW_ISA=avx512` and `=avx2`, and the 29 font programs' keys, digests and saturation
telemetry, the extraction fields included, unchanged. A4's Done line (:675) carries that clause.

`impl-docs.md` withheld it because "No `a5-impl` output exists". `gates-impl.md` now records gates 1, 1b,
2 and 2x as IDENTICAL to the base on both tiers (`byteprobe.a5-impl.*`, `gate1b`, `gate2`, `gate2x`
`.a5-impl.txt`).

**Fix:** append the clause from those outputs. If the coordinator wants an `a5-pre` run first (spec §4.2
step 0; gates-impl.md says none exists in `out/`), word the clause against the base it was compared to.

### 3. `refuse_shadowing_an_item` keeps a one-arm dispatch whose second arm was the deleted one (non-blocking)

`pixelflow-compiler/src/sema.rs:756–768` builds `let shadowed = match … { Some(SymbolKind::Const) =>
Some(format!(…)), _ => None }` and then `if let Some(shadowed) = shadowed { return Err(…) }`. The
`Option<String>` indirection existed so that `Const` and `Structural` could share one error (base
:785–789). With one kind left, it is a dispatch over a single case. The `fn` check right below it
(:769) is already written as the direct guard. `impl.md` departure 5 keeps it as "one `match` away from a
future kind", which is the speculative structure "subtract before you add" and "fold before you dispatch"
argue against.

**Fix:** use `if self.symbols.lookup(&text).is_some_and(|s| s.kind == SymbolKind::Const) { return
Err(…"shadows the `const {text}` of this block"…) }`. The message is unchanged, so `an_item_is_not_shadowed`
still holds.

### 4. Three ragged reflows where a deletion shortened a comment (non-blocking, cosmetic)

- `pixelflow-compiler/src/lower.rs:795–798`: "…a const, then the / coordinates. Sema refuses a / parameter
  named X or Y…"
- `pixelflow-ir/src/arena.rs:167–169`: `FoldTerms`' doc, "the / fold's algebra and its bounds, each a literal
  or a term in the / arena…"
- `pixelflow-ir/src/fold.rs:539–541`: "…only `pack` / and `HalveFold` make one. At most / [`Fold::EXACT_BOUND`]…"

`cargo fmt` does not rewrap comments. **Fix:** reflow each paragraph to the file's width.

## Looked at and not reported

- **`Staged::open`'s doc** ("the variable holding each fold's index"). The variable holds the `OpenFold`,
  whose `index` field is the index. That is R8's text verbatim, and defensible read that way.
- **`node`'s `let decoded`** is now single-use. It is a macro-side local, not emitted tokens (unlike J13's
  `__range`), so no expansion carries it.
- **The codegen panic** says "instantiated" while emit's says "substituted". Spec §2.8 and R1 prescribe each,
  and R1 exists only for G3, which scopes to the compiler crate.
- **`convert_const`'s `refuse_generics`** is unreachable in practice, because syn yields `Item::Verbatim`.
  R7 keeps it deliberately.
- **`public-surface.md`** was not written. `impl.md` §5 holds its content for the coordinator to move.
