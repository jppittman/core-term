# A5 rulings

These rulings are on `spec.md` and the three critiques (`critique.{closure,semantics,a6fit}.md`). Where a
ruling and the spec differ, the ruling wins. Every finding is ruled on: accepted, accepted with a change, or
refused with a reason. Findings for A6 are carried to `../a6/design.md` (section R-A6 below) and are not built
in A5.

## The spec's open call

- **J6, `FoldTerms::stride`: accepted, deleted in A5.** A5 is the CL that makes it dead, and a field
  documented "always 1" is a convention in a comment.
- **J5, A7: accepted, A7 is empty and folded into A5.** The plan's A7 bullet is rewritten (§6.1), worded as
  in R13.

## Gates (the spec's own gates must pass as written)

- **R1 (closure 1, semantics 1): the `Param` panic says "substituted", not "instantiated".** The text is:
  "kernel! produced ExprNode::Param({k}): a `Param` is a rewrite rule's metavariable, substituted before any
  arena reaches emission; lowering builds none". G3 then keeps its stated expectation.
- **R2 (closure 2, semantics 2): G8 excludes ordinary Rust's turbofish.** The pattern is
  `grep -rnE '<const [A-Za-z_]+: *usize|::<[0-9]+>|Args::<' pixelflow-compiler/ pixelflow-graphics/ pixelflow-core/ | grep -v 'as_chunks::<'`.
  The expectation is unchanged: only the parser's new test hits.
- **R3 (closure 6): G10's expected files** are the-language-is-kernel, the bound plan, ontology `SKILL.md`
  and `language.md`. `compiler.md` drops out. The criterion also admits "a pointer to the ontology's retired
  entry". Better still, word the Binding-time pointer at `language.md:125-126` without the phrase.

## Code

- **R4 (closure 3): `arena_to_tokens`' `# Panics` (`emit.rs:436`)** reads "is not `arguments` one for one".
- **R5 (closure 4, semantics 7): `Frame<'a, S>` becomes `Frame<S>`.** `Lowering`'s field follows.
  `literal_range` destructures before it filters: `let Range { start: lo, end: hi } = range;`, then
  `Some(lo..hi).filter(KnownRange::admits).ok_or_else(..)`, formatting the message from `lo`/`hi`.
- **R6 (semantics 3): a turbofish at a call inside a block is refused.** This is the call-site half of J1/J2
  ("no `fn` takes generics"), so it is A5's subject.
  - **Today:** the parser's `Expr::Call` arm (`parser.rs:820-833`) drops `segments[0].arguments`, so
    `h::<4>(X)` reads as `h(X)`.
  - **After A5:**
    - **Where:** in that arm, a callee whose one segment carries arguments is refused at the segment.
    - **Message:** one, with no branch on the argument's kind: "generic arguments at a call in a `kernel!`
      block\n\nnote: nothing in a kernel is generic, so no `fn` takes generic arguments (§1.3 of
      {PLAN})\nhelp: a count is a literal, or a `usize` const of the block".
  - **Pinned** in §5.3's new test, or in a second parser test beside it: `h::<4>(X)` (a helper),
    `DX::<f32>(X * X)` (a builtin) and `k::<4>(X, Y)` (a kernel parameter).
  - **`leading_colon` (`::h`, `::X`) is refused nowhere in A5.** It names nothing of generics, it gives no
    wrong value (`::h` is the block's `h`), and it is outside the bound plan.
- **R7 (semantics 5): a generic `const` item.**
  - **Behaviour:** syn parses it as `Item::Verbatim`, so `refuse_item`'s catch-all refuses it.
    `convert_const` keeps its `refuse_generics` call. It is correct, it costs nothing, and it holds if a
    future syn parses one into `ItemConst`.
  - **Spec:** the spec stops claiming the user sees the generic refusal for a `const`. J2 and the parser doc
    say "a `fn` takes no generics; a `const`'s are refused by the same function, though syn gives the item
    refusal first".
  - **No test** of a generic `const` through `refuse_generics`.
- **R8 (a6fit 8): `Staged::open`'s doc (`emit.rs:674-675`)** reads "the variable holding each fold's index,
  and the `FoldTerms` it closes into".

## Wording

- **R9 (semantics 4, a6fit 3): the `const` generic note uses the front end's vocabulary.** In the front end
  a value is an `f32` or a `bool`, and a `usize` is a count:
  "note: a count does not make a program of its own: an entry is one program, whatever its counts (§1.2 of
  {BOUND_PLAN})". The help line is unchanged. The test pins the header, the plan path and the help line, as
  §5.3 says.
- **R10 (semantics 6): the ontology's Entry text** reads "rebuilds its lowered arena, saturated at expansion
  unless it carries a `Dwrt` or the macro is `kernel_raw!`; an entry that takes a kernel runs lowering's steps
  when it is called (`Staged`)".
- **R11 (closure 8): intent.md's "each citing its plan section (`PLAN`)"** becomes "(`PLAN`, `BOUND_PLAN`)".
  The 80-line budget still holds.

## Tests

- **R12 (closure 5): keep the count-1 pins.**
  - **`kernel_typed_parameters.rs`:** add `const ONE: usize = 1;` beside `DOWN`, and an entry
    `mean_down_one` over it, asserting `bake(&mean_down_one(&k)) == 8.0`. This is the `mean_down::<1>` case:
    a one-term fold around an application in a `Staged` entry.
  - **The rustc oracle:** a second entry over `FIRST..FIRST + ONE` against a one-term `rust_mean_index`
    twin. This is the `mean_index::<1>` case.
  - **Accounting:** §5.4 is updated. Both are assertions inside existing `#[test]`s if they fit there;
    otherwise they are new `#[test]`s, counted.
- **R14 (closure 9): the hygiene test's reason is corrected.**
  - **The test:** `a_structural_parameter_named_as_the_emissions_locals_is_its_own` is still deleted. Its
    subject, a `const` generic captured as a pattern, is gone.
  - **The reason, recorded in `impl.md`:** the emission's `__lo`/`__hi` locals can still meet a block's
    `pub const __lo`, which rustc reads as a constant pattern.
  - **Why no new refusal:** that failure is a compile error (E0530/E0005), loud and at the author's own line,
    and it predates A5. By CLAUDE.md's priority, loud failure does not earn a type.
  - **Pin first:** the implementer confirms it on the base with a scratch probe. If it does not reproduce,
    the reason says so instead.
  - **BACKLOG:** a one-line row in the compiler section, with the repro.

## Plan record

- **R13 (a6fit 5): the A7 bullet and A5's Done line.**
  - **The A7 bullet:** "…exempted only test templates, which A5 rewrote over `usize` consts or deleted with
    the tests whose subject they were".
  - **The Done line:** adds "the tests whose subject was the template are deleted".
- **R15 (a6fit 4): plan §5's row for the-language-is-kernel §1.3** (plan:496). Its CL cell reads "A5 (a
  literal or a `usize` const), A6 (a per-call count)", and its Becomes text is the one A5 writes.
- **R16 (closure 7): no rewrite of `one-pipeline.md:667`.** It is history under its "Superseded in part"
  banner, and impl.md records that disposition. The-language-is-kernel's banner sentence adds §1.6 to its
  enumeration.

## R-A6: carried to A6's design, not built here

- **a6fit 1:** `n: usize` is accepted for an entry of `Spelling::Items` only. The closure form's `usize` is
  refused, and §7.4 pins it.
- **a6fit 2:**
  - **Type first:** every name in a bound is typed before arithmetic is judged. `X`/`Y` are refused by L3,
    and any other non-`usize` by type.
  - **Arithmetic:** `BoundArithmetic` is refused only when every name is a `usize` index or count parameter.
  - **Pin:** `(0..X + 1)` is added to §7.4.
- **a6fit 6:** `UsizeScope` gets an associated output type. `ConstEvaluator` keeps A5's `u64`, and
  `RangeScope` yields `Count`.
- **a6fit 7:** `Parameter { name, kind }`, with `kind` one of `Value`, `Count` or `Record(..)`, and no `bool`
  beside `record`.
- **D-A6-1..4** (the `write_into` refusal, which needs a `pixelflow-core` public API decision, plus the
  helper, closure and host-panic questions) are A6's design round. D-A6-1 touches public API and goes to JP
  with a recommendation.

## Round 1 (reviews `review1.{closure,semantics,fidelity}.md`, on 38703e0)

Every finding accepted except where said. Code and docs both.

- **C1 (blocking): the four `Param` docs** (`pixelflow-ir/src/kind.rs:219` `OpKind::Param`,
  `pixelflow-search/src/egraph/node.rs:248` `ENode::Param`, `pixelflow-search/src/egraph/insert.rs:52-57`
  `Declined::Param`, `pixelflow-search/src/runtime.rs:150`) say what a `Param` is now: a rewrite rule's
  metavariable, an unbound slot of a rule template, substituted before any arena reaches emission or a
  `Kernel` exists; `kernel!` builds none. In insert.rs, "the macro tier holds params natively" becomes
  "`Vocabulary::Templates` holds a rule template's metavariables". `ir_insert.rs:174`'s asserted substring
  is not changed. Check each sentence against the code: grep who builds a `Param` today
  (`push_param`, `ExprNode::Param`, `ENode::Param`) and describe exactly that.
- **C2/F2 (blocking): the A5 Done line's gate clause**, worded against the base actually compared:
  "byte_probe empty in both columns under `PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font programs' keys,
  digests and saturation telemetry, the extraction fields included, unchanged" (gates compared to the base
  verified at main 7e9531e).
- **F1 (blocking): the A5 Done line.** The test templates that take no kernel (`ramp`, `rings`, `swept`,
  `mean_index`, and emit.rs's `counted`) are offered to the macro tier at expansion; the two that take one
  (`applied`, `mean_down`) stay `Staged`. "which only the template set" → "which only the template set past
  1". Verify the list against the diff.
- **S1/F3 (blocking): parser.rs's module doc** (:120): "a generic or a `where` clause on a `fn` (a `const`
  generic by a message saying a count is a literal or a `usize` const, §1.2 of the bound plan), and on a
  `const`, which syn hands over as an unknown item and the item refusal names". The new test's doc (:2700)
  drops "and no `const`". No generic-`const` case is added (R7).
- **S2 (blocking): `generic_refusal`'s doc** (:637): "a `const` one was a count, which a block now writes as
  a literal or a `usize` const, and any other names a type the language does not have".
- **C3: `refuse_shadowing_an_item`** loses the one-arm `match` into `Option`: `if self.symbols.lookup(&text)
  .is_some_and(|s| s.kind == SymbolKind::Const) { return Err(..) }`, message unchanged.
- **C4:** reflow `lower.rs:795-798`, `arena.rs:167-169`, `fold.rs:539-541` to the file's width.
- **S3:** the `Expr::Call` arm tests `!segment.arguments.is_empty()`, as the `Path` arm does, so `h::<>(X)`
  is `h(X)` in both.
- **S4:** the `Path` arm refuses a one-segment unqualified path whose segment carries non-empty arguments
  (`N::<4>`, `X::<f32>`) with R6's message, not the outside-the-block one. One function builds R6's error;
  both arms call it. A case of each arm in the new parser test.
- **S5:** a `where` clause is refused at `where_clause.where_token.span` (`syn::Error::new`), so an empty one
  points at `where`.
- **F4:** the new test asserts the plan path in the non-`const` generic message (`§1.3 of
  docs/language/2026-09-25-the-language-is-kernel.md`) and in R6's. **Spans stay unpinned in A5**: the crate
  cannot observe a span (no compile-fail harness, no `span-locations`); that is a CI gap for the whole front
  end, not A5's, and gets a BACKLOG row (C12: refusal spans are unpinned; how a span-observing check could
  be built — a compile-fail target or proc-macro2's `span-locations` in tests — and the one wrong span it
  would have caught, S5). **R6 is amended**: the refusal is spanned on the segment's arguments (`::<4>`), not
  the segment; impl.md records it.
- **F5: R8 is amended** to "the variable holding each open fold, and the `FoldTerms` it closes into", and
  `Staged::open`'s doc says that. Check it against the code.
- **R14's reason in impl.md** takes the probe's corrections (E0308, not E0530/E0005; the family is every
  `__` local, `__arena`/`__root` beside any entry). BACKLOG C11 already says so.
