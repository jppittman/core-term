# A5 review, round 2: docs truth

Scope: every doc sentence the round-1 diff (`git diff 38703e0 -- . ':!.wip'`) adds or edits, the sentences
beside those edits, and the rest of the A5 diff (`7e9531e..HEAD` plus the working tree) for anything false.
Read-only on the repo; nothing was built. Each finding was checked against the code, not against the ruling's
wording. Ranked most serious first.

## 1. `Param` is not a rewrite rule's metavariable (blocking)

C1 asked for the four `Param` docs to say "a rewrite rule's metavariable, an unbound slot of a rule template",
and to check it against who builds a `Param` today. The check fails: the code says the opposite.

What a rewrite template's metavariable is:
- `pixelflow-search/src/egraph/template.rs:19-20`: templates "read metavariables (`ExprNode::Var`) the same way
  every existing template does: `Var(0)` = A, `Var(1)` = B".
- `pixelflow-search/src/egraph/rewrite.rs:37` (`arena_pat!`'s doc): "Metavariables are encoded as `var N`".
- `pixelflow-ir/src/expr.rs:28-29`, in the same enum as the new `Param` doc: `Var` is a "coordinate …, reduction
  binder …, or rewrite rule metavariable". `pixelflow-ir/src/kernel.rs:327-328` says the same.

What a `Param` in a rule template does:
- On a left-hand side it matches nothing: `match_root` and `match_class` return `false` for `ExprData::Param`
  (`template.rs:73`, `:109`).
- On a right-hand side it panics: `instantiate_template`'s doc says "a rewrite RHS template must never contain"
  one (`pixelflow-search/src/egraph/graph.rs:1695-1696`), and its arm is
  `panic!("instantiate_template: Param({p}) in a rewrite RHS template")` (`:1720-1722`).
- `arena_pat!`'s `par N` leaf (`rewrite.rs:42`) has no call site in the workspace.
- The NNUE matcher treats a template `Param` as a literal leaf that matches only the same `Param`
  (`pixelflow-search/src/nnue/mod.rs:103-107`), not a binder.

Who builds a `Param` today: tests (`pixelflow-search/tests/ir_insert.rs:160`, `pixelflow-ir/src/kernel.rs:2015`,
`pixelflow-ir/src/arena.rs:2543`); the training corpus's reader on a `TAG_PARAM` record
(`pixelflow-pipeline/src/training/corpus.rs:575-577`); and copy-throughs that rebuild one they were handed
(`arena.rs`, `passes.rs:240`, `term_arena.rs`, `expr.rs`, `nnue/mod.rs:227,881`, `corpus_gaps.rs:484`,
`extract.rs:958`, `witness.rs:376`). Nothing makes one from nothing on a production path. The premise came from
the plan's §4, `docs/language/2026-10-20-a-bound-is-a-value.md:434-435` ("`Param(u8)` stays: it is also a rewrite
rule's metavariable"), which is false too, and from the closure review's list of "remaining producers", which
named `par` (never called), `nnue/mod.rs` and `term_arena.rs` (copy-throughs).

Sentences that are false for this reason:
- Round 1: `pixelflow-ir/src/kind.rs:219` (`OpKind::Param`), `pixelflow-search/src/egraph/node.rs:248`
  (`ENode::Param`), `pixelflow-search/src/egraph/insert.rs:52` (`Declined::Param`),
  `pixelflow-search/src/runtime.rs:150`, `pixelflow-ir/src/expr.rs:32` (`ExprData::Param`),
  `pixelflow-ir/src/term.rs:104` (`Shape::Param`), `pixelflow-search/tests/ir_insert.rs:148`.
- The A5 commit's two panics: `pixelflow-compiler/src/emit.rs:439-445` ("a `Param` is a rewrite rule's
  metavariable, substituted before any arena reaches emission", and the comment above it) and
  `pixelflow-codegen/src/program/lower.rs:373-376` ("a rewrite rule's metavariable is instantiated before a
  Kernel exists"). Nothing substitutes or instantiates a `Param`.
- Beside the round-1 doc edits: "one tier earlier" (`kind.rs:221`, `node.rs:250`) names a tier that builds no
  `Param` any more. `ir_insert.rs:165` ("the macro tier must hold an unbound builder slot") and `:172` ("a Param
  at bake time means a builder was never called") name a builder that no longer exists. C1 kept `:172` on the
  ground that it "stays true of a rule template", which the code above contradicts.
- A smaller tension: `node.rs:599` now says a `Param` "is one number for the whole lattice", while
  `fold.rs:466` (also round 1) says it "has no value".

Fix: say only what is true, and delete the rest. For example, for `OpKind::Param`: "An unbound scalar slot: a
leaf with no value, never folded, matched by no rewrite rule, derivative zero. Nothing builds one but tests and
the training corpus's reader: `kernel!` builds none, and a rewrite template's metavariable is a `Var`." Keep the
`Vocabulary` sentence. The panics become "kernel! produced ExprNode::Param({k}): lowering builds none" and
"ExprNode::Param({}) reached the JIT emitter: a `Param` has no value". The `ir_insert.rs` messages lose the
builder. Also correct plan §4's sentence, or rule that it is left as history.

## 2. The ontology's `Param` "Is" still describes a builder (blocking: beside an edit, and it contradicts that edit)

`.claude/skills/ontology/language.md:356-360`: "an unbound scalar slot of a builder template. `Param(i)` exists
only before the builder is called … `Declined::Param` still means a builder was never called." A5 edited this
entry's Lives sentence (`:374-376`) to "`kernel!` builds none since a-bound-is-a-value A5 deleted the template
hole it was for". The two now disagree inside one entry, and the "Is" is false (finding 1 lists who builds a
`Param`). The fidelity review noted it under "Looked at and not reported", and no ruling covers it. The ontology
is what an agent reads before designing, so this is where a false denotation spreads from. Fix: rewrite the "Is"
to what finding 1 establishes, keeping the quotation marked as history, or delete the quotation.

## 3. The A5 Done line and the A7 bullet leave out `emit.rs`'s `inside` (blocking: the record is false for one member)

At `7e9531e`, `emit.rs:963` was `pub fn inside<const N: usize>(b: Bounds, r: f32)`. `N` was unread, but the
entry was structural, so the exemption at `emit.rs:154-156` applied to it. Now it is `pub fn inside(b: Bounds,
r: f32)` (`emit.rs:835`): there is no count, and its test, `a_record_and_an_args_record_expand_to_host_structs`,
is kept. Two sentences are false for it:
- The Done line (`docs/language/2026-10-20-a-bound-is-a-value.md:700-702`): "The test templates that take no kernel
  (`ramp`, `rings`, `swept`, `mean_index`, and `emit.rs`'s `counted`) are written over `usize` consts". F1
  asked for the list to be checked against the diff.
- The A7 bullet (`:713-714`): "It exempted only test templates, which A5 rewrote over `usize` consts or deleted
  with the tests whose subject they were".

The fidelity review called the A7 half a nit. Fix: name `inside` in both places, for example "…; `emit.rs`'s
`inside`, whose count nothing read, lost its generic".

## 4. `BOUND_PLAN`'s doc states a rule the refusals do not follow

`pixelflow-compiler/src/lib.rs:75-76`: "The plan that made a fold's bound a value: a refusal that turns on what a
count is names it." Only one refusal names it, the `const` generic one (`parser.rs:649`). Two others explain
what a count is and do not name it:
- Round 1's `generic_arguments_refusal` (`parser.rs:668-673`). Its help is "a count is a literal, or a `usize`
  const of the block", and it cites §1.3 of `PLAN`.
- sema's `usize` refusal (`sema.rs:152-155`). Its note is "a `usize` is a count, not a value: a fold's index or
  a `usize` const", and it cites no plan.

Relatedly, intent.md's "refusals, each citing its plan section (`PLAN`, `BOUND_PLAN`)" (`intent.md:10`, R11)
is not true of every refusal: the `where` refusal (`parser.rs:633-636`) and sema's note above cite none. That
was already the case at base. Fix: "The plan that made a fold's bound a value, named by the `const` generic
refusal." Leave intent.md as ruled, or soften "each".

## 5. "Every parameter is a uniform" and "one entry is one program" leave out kernel-typed parameters

A kernel-typed parameter (`k: impl Fn(f32, f32) -> f32`) is not a uniform. The ontology A5 edited says "two
binding times: uniform and kernel-typed" (`language.md:129-130`), and `lib.rs`'s binding-time section lists
kernel-typed separately. An entry that takes one builds a different program for each argument kernel (`Staged`,
`emit.rs:24-33`). The sentences:
- A5 wrote these: `pixelflow-compiler/tests/binding_times.rs:4-7` ("Every parameter is a **uniform** … So every
  call of an entry is one program") and `pixelflow-compiler/intent.md:30` ("one entry is one program", which was
  "one entry at one instantiation is one program").
- Beside A5's edits: `parser.rs:97`, `sema.rs:28`, `emit.rs:37-38`.

Each was already loose at base under other wording; A5 dropped its qualifier and left the universal. Fix: scope
each to an entry that takes no kernel, or say "every `f32` and record parameter".

## 6. The plan's Order paragraph still assigns `:179` to A1, after #1431

`docs/language/2026-10-20-a-bound-is-a-value.md:535-536`: "A1 goes now, except its the-language-is-kernel
`:179`/`:186` rows, which follow #1431". After R15, §5's row gives `:179` to "A5 …, A6 …" (`:492`), and A1's Done
line (`:558-560`) no longer excepts it. #1431 has landed (`9c33002`, an ancestor of `7e9531e`), and the `:186`
row's Hunk cell is now "none". BACKLOG's plan row still says "A1's the-language-is-kernel rows after #1323"
(`docs/BACKLOG.md:32`), but the `:186` row now names no collision. Fix: say `:186` in the Order sentence and drop
"which follow #1431", or mark the paragraph as history.

## 7. `Param` mechanism comments outside the diff that A5 made false (not blocking)

`kernel!`'s template hole was the last thing that substituted a `Param` away. These still describe that mechanism:
- `pixelflow-ir/src/variance.rs:374`: "Parameters are substituted before JIT compilation."
- `pixelflow-search/src/egraph/cost.rs:105`: "substituted away before codegen".
- `pixelflow-search/src/saturate_pass.rs:19-20`: "a `Param` slot that should have been specialized".
- `pixelflow-ir/src/kind.rs:802-803`: "a value arrives per call or from a builder".

Each should go with finding 1's wording.

## 8. Nits

- `generic_refusal`'s doc (`parser.rs:640-642`, beside round 1's S2 edit): "a `const` one was a count … and any
  other names a type the language does not have". `<const N: u32>` was never a count; base refused it as "a
  structural parameter is a `usize`". `<'a>`, which the new test pins in that arm (`parser.rs:2737`), names a
  lifetime, not a type.
- `.claude/skills/ontology/compiler.md:85-86`: "Templates are `Staged`". The same entry's "Is" (`:67-70`) says
  lowering produces a template for every entry. An entry that takes no kernel is emitted by `arena_to_tokens`,
  not `Staged`.

## Checked and true

- `Staged::open` (F5, `emit.rs:563-565`): `#fold` holds what `__arena.open_fold(depth)` returns (`emit.rs:703-713`).
- The parser's module-doc refusal list (`parser.rs:117-125`):
  - syn 2.0.101 parses a generic `const` (or one with a `where`) as `Item::Verbatim` (`syn/src/item.rs:1029-1042`),
    which `refuse_item`'s catch-all names "a `item`".
  - Both the `Path` and the `Call` arms refuse non-empty generic arguments on a one-segment name.
  - Range bounds and call arguments go through `convert_expr`, so `(0..N::<4>)` is refused too.
  - `PathArguments::is_empty` treats `h::<>` as no arguments.
- BACKLOG C12:
  - `refusal()` returns `to_string()` (`parser.rs:2019-2023`, `sema.rs:2073-2076`).
  - The tests build their inputs with `quote!`.
  - No crate in the workspace or the registry enables `span-locations`.
  - The compiler crate has no `trybuild` or `compile_fail` target.
  - syn's `WhereClause` prints no tokens when it has no predicates, so `new_spanned` falls back to `call_site`.
  - The `where` refusal is now at `where_token.span`.
- The Done line's gate clause:
  - Gate 1b's 29 lines are `nodes= key= digest=`, gate 2 is the saturation telemetry, and gate 2x the extraction
    fields.
  - The bases were recorded at `3bfdaded` and verified at `7e9531e` (tag `a5-main`, `.wip/gates/README.md`).
  - Both `gates-impl.md` and `gates-fix1.md` record every one IDENTICAL.
- The Done line's other names:
  - `FoldTerms`' stride was set past 1 only by `ExpansionFold::Hole` at base.
  - `pack` and `HalveFold` are the only remaining strided constructors (`passes/lattice.rs:241`,
    `fold_rules.rs`), as `fold.rs:537-540` says.
  - Every name §3.4 lists as deleted is gone outside the plan.
  - `applied` and `mean_down` take a kernel and are `Staged`.
  - `ramp`, `rings`, `swept`, `mean_index` and `counted` are over `usize` consts under `kernel!` (or the
    `Offered` optimizer).
- The round-1 reflows (`lower.rs:795-797`, `arena.rs:167-169`, `fold.rs:537-540`) and their sentences, the
  `extract.rs`/`graph.rs` comments, the new parser test's doc, and the-language-is-kernel's banner and §1.3 row.
