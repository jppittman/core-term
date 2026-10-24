# A5 review, round 3: docs truth across the whole diff

Scope: `git diff 7e9531e -- . ':!.wip'` as it stands in the working tree (round 1 committed as 0c3b52a, round 2
uncommitted on top), read against `rulings.md` (every round), the bound plan and CLAUDE.md. Every sentence,
comment and message the diff adds or edits, and every sentence beside an edit, was checked against the code.
Read-only on the repo; the only file written is this one.

## What holds

**The `Param` denotation, by grep.** No pipeline stage builds a `Param`. The callers of `push_param` are tests
(`arena.rs:2543`, `kernel.rs:2015`, `tests/ir_insert.rs:158`), the training corpus reader
(`pixelflow-pipeline/src/training/corpus.rs:577`), copies of a `Param` that already exists (`arena.rs:1617`, `:1951` and
`:2116`, `passes.rs:240`, `expr.rs:324` and `:390`, `term_arena.rs:61`, `nnue/mod.rs:227` and `:881`, `corpus.rs:257`,
`corpus_gaps.rs:484`), and `arena_pat!`'s `par` arm (`rewrite.rs:42`), which no rule uses. Nothing folds one. No
rule template has one: a `Param` on a left-hand side matches nothing (`template.rs:456`), and one on a right-hand side
panics (`graph.rs:1720`). Its derivative is zero in both the IR (`passes.rs:732`) and the e-graph (`graph.rs:1847`).
These panic on one: the JIT emitter (`program/lower.rs:373`), the compiler's emission (`emit.rs:440`), the edge
walker, `arena_to_kernel_code` and the collapse corpus. `Vocabulary::Runtime` declines it (`insert.rs:128`). The
docs at `kind.rs`, `node.rs`, `insert.rs`, `runtime.rs`, `expr.rs`, `term.rs`, `fold.rs`, `cost.rs`, `extract.rs`,
`saturate_pass.rs`, the pipeline messages, `ir_insert.rs` and the ontology's `Param` entry describe exactly that. The
one exception is variance (finding 7). The proximity grep for "metavariable", "substitut" and "specializ" near
`Param` is clean outside history.

**The plan record.**
- **The A5 Done line is true entry by entry.**
  - At the base the templates were `ramp`, `rings`, `grid` and `span` (binding_times), `swept` and `applied`
    (items_block), `mean_down`, `mean_index`, emit's `inside`, `mean` and `counted`, lower's `f`, parser's `f`, and
    sema's test inputs.
  - Those listed as rewritten over `usize` consts are, and are offered at expansion.
  - `inside` lost a generic that nothing read.
  - `applied` and `mean_down` are `Staged`.
  - The rest went with their tests.
  - `FoldTerms`' stride was set past 1 only by the `Hole` arm.
  - The gate clause matches `gates-fix2.md`: byte_probe is identical to the a5-main base on both tiers, and gates
    1b, 2 and 2x are identical.
- **The A7 bullet is true.** At `7e9531e`, `emit.rs:154-156` is the `if !entry.structural.is_empty()` early return,
  and no production entry had a generic.
- **The §4 "Deleted" sentence and the §3.4 list are true.**
- **The §5 `:179` row is true.**

## Findings, most severe first

1. **"Saturated at expansion unless it carries a `Dwrt`" is false (blocking).**
   - **Where:** the ontology's Entry (`language.md:87-90`, R10's text), `kernel!` Lives (`language.md:76-79`) and
     intent.md Behind (`:53-54`, "saturates any entry with no kernel-typed parameter and no `Dwrt`").
   - **The mechanism:**
     - `&` and `|` lower to `OpKind::BitAnd`/`BitOr` (`lower.rs:477-478`).
     - `template_may_name` refuses both, and `Vocabulary::admits` adds them only for `Runtime` (`ops.rs`).
     - `Saturate::macro_tier` inserts under `Templates`, so `insert` returns `Declined::Op` and the arena is emitted
       unsaturated.
   - **Examples:**
     - `rings` in `binding_times.rs` (`(X <= b.x1) & (Y <= b.y1)`) and the macro doc's own example are never
       saturated at expansion.
     - CLAUDE.md says the same: "a `&` mask made the e-graph decline their arena".
   - **What is true:** "offered to the optimizer", the wording the "Structural parameter (retired)" entry and the Done
     line already use.

2. **"Every `f32` and record parameter is a uniform" is false for a helper (blocking).**
   - **Where:** `binding_times.rs:4`, `sema.rs:30-31`, `parser.rs:97` and `lib.rs:258`.
   - **Why it is false:** a helper's parameters are its arguments, inlined (`lower.rs` `inline`;
     `AnalyzedKernel::parameters` is an entry's).
   - **Two contradictions inside the diff:**
     - `sema.rs`'s own resolution rule 2 (`:47-48`) says "a helper's is the argument at the call".
     - `binding_times.rs` declares `fn inside(b: Bounds, x: f32, y: f32)` 25 lines below its module doc.
   - **Cause:** D5 scoped the sentence by type and not by role.

3. **"An entry is one program" is still unscoped in three places (blocking).** It is false for an entry that takes
   a kernel, whose program is the composed term's.
   - **The places:**
     - The-language-is-kernel banner (`:58-59`, "`kernel!` has none: an entry is one program").
     - The `const`-generic refusal message (`parser.rs:647-648`, "an entry is one program, whatever its counts"),
       with its test doc (`:2727`).
     - `binding_times.rs:101-104`, whose doc and name read "Every call of an entry is one program".
   - **Elsewhere the CL already scopes it:** lib.rs, emit.rs, lower.rs and intent.md all say "an entry that takes no
     kernel".

4. **The plan makes `:186` wait on #1323, and §5 does not (blocking).**
   - **Where:** the Order paragraph (`:535-536`), A1's Done line (`:558-560`) and BACKLOG's "next" cell.
   - **What §5 and the collision list say:**
     - The `:186` row (`:493`) is CL "A1", Hunk "none".
     - The collision paragraph gives #1323's the-language-is-kernel hunks as `:40-45`, `:164-172` and `:644-650`
       (`:524-527`).
     - `:186`'s only blocker was #1431, which landed (9c33002, an ancestor of 7e9531e; §8 item 3).
   - **The contradiction:** the sentence cites "§5's 'Hunk' column", and that column contradicts it.

5. **The ontology's Binding time contradicts itself (blocking).** At `language.md:129-131` it says: "A count is a
   literal or a `usize` const, known at expansion (a-bound-is-a-value §1.2)". At `:142` the same entry says that "a
   uniform bound is a per-call count", and the §1.2 it cites has a uniform row. True only of `kernel!` until A6.

6. **The ontology's Program entry contradicts itself (non-blocking).**
   - **"Its template is one kernel" (`compiler.md:68-70`) is false.**
     - The template mints new uniform identities and defaults on each call (`UniformIdentity::mint()` in
       `arena_to_tokens`).
     - So every call builds a different kernel, one instance each. The entry's next sentence says so.
     - What the template is, is one program.
   - **"Lowering produces a template" contradicts the entry's Lives line.** Lives says "Templates are
     `arena_to_tokens`' output", and that is emission after the optimizer, not lowering.

7. **Two answers for a `Param`'s variance (non-blocking).**
   - **The comments disagree.**
     - `variance.rs:374-375` says "nothing is known of what it varies with" and makes it all-varying.
     - `graph.rs:1845-1846` says "a slot does not vary along the axis".
     - `node.rs:590-592` classes it with a uniform (`CONST`).
     - `kind.rs` and the ontology give its derivative as zero.
     - Every one of these is this CL's.
   - **The code disagrees too:** the IR says `ALL` and the e-graph says `CONST`. That predates the CL, but the
     comment's reason contradicts the denotation this CL writes.

8. **"Refusals, each citing its plan section (`PLAN`, `BOUND_PLAN`)" is false (`intent.md:9-10`; non-blocking).**
   - **Counterexamples:**
     - The two `where` refusals the CL re-spanned (`parser.rs:344` and `:634`) cite no section.
     - Neither do `refuse_item` and the variadic refusal.
   - **So the ruling's premise is false:** D4 left the sentence on the premise that each refusal cites one of the
     two.

9. **The reason given for "only Runtime declines a `Param`" does not tell the vocabularies apart (non-blocking).**
   - **The reasons given:**
     - `insert.rs:54-55` and `:88-89`: "the runtime tier builds kernels".
     - `ir_insert.rs:149-150`: "has no meaning in a kernel that is built".
   - **Why they do not distinguish:** a `Templates` graph is the macro tier's, and its arenas also become kernels
     (`entry_arena`, then `arena_to_tokens`, then `Kernel::from_parts`).

10. **"A count is a literal or a `usize` const" leaves out the arithmetic the code accepts (non-blocking).**
    - **Where:** `lib.rs:278-280`, `parser.rs:102`, intent.md `:13-14` and the refusal help text.
    - **The contradictions:**
      - `lib.rs:163` says a bound has "`+ - * /` between them".
      - `UsizeScope::eval_usize` evaluates exactly that.
      - The CL's own oracle case is `FIRST..FIRST + N`.

11. **`lib.rs:256` "Every value a program reads is bound at one of two times (§1.4)" is false (non-blocking).**
    - §1.4's table has three rows, and its note keeps the tile extent structural.
    - The program also reads X and Y, and `const` items. The doc's own Uniform bullet says "A constant is a `const`
      item".

12. **`lower.rs:33` "An entry's parameters are its uniforms" is false beside the scoped edit at `:38`
    (non-blocking).** A kernel-typed parameter is not a uniform, and `AnalyzedKernel::parameters` filters it out.

13. **The A5 bullet's list of users is incomplete (`:694-696`, beside its edit; non-blocking).** It omits the
    `lower.rs` and `parser.rs` unit tests that used `<const N>` and were deleted.

14. **`Staged::open`'s doc says "the `FoldTerms` it closes into" (`emit.rs:559-560`; non-blocking).**
    - A fold closes *over* its `FoldTerms`, as `fold_terms`' doc, `FoldTerms`' own doc and `Expansion::open`'s doc
      all say.
    - What the field holds is the token stream that builds them.
