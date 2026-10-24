# A5 docs: what changed

Markdown only. Spec §6.1-§6.4, §6.6, §6.8, with R3, R10, R11, R13, R15, R16, and R14's BACKLOG row. The `.rs`
docs (§6.5 `lib.rs`, §6.7 `binding_times.rs`, every §2 table row) are the code agent's. Checks at the end:
`bash scripts/check-intent.sh` exit 0 (51 intent files), `bash scripts/check-doc-paths.sh` exit 0 (it prints
`head: cannot open '../AGENTS.md'` noise from its own relative paths; the verdict line is OK).

## Files

| File | Change |
|---|---|
| `docs/language/2026-10-20-a-bound-is-a-value.md` | §3.4: "The structural path goes with …" through "… work unchanged through A2-A4." replaced by the spec's "went with `<const N: usize>` … `kernel!` refuses a generic on any `fn`" (names kept in one sentence so G1 and `check-doc-paths` rule 2 hold). §5: the-language-is-kernel §1.3 row split in two (below). §6 A1 Done line: the `usize` row leaves the exceptions; only the "how many instances the host composed" sentence (`:186`, `:199` after A1) and the D-table rows, after #1323. §6 A5: **Done:** line added. §6 A7: rewritten "Folded into A5". |
| `docs/language/2026-09-25-the-language-is-kernel.md` | Banner (`:48-54`): one sentence appended, 2026-10, enumerating §1.1 diagram and binding-time bullet, §1.5, **§1.6 and §1.7**, "What the macro emits". §1.3 `usize` row: "a fold binder or a count (a literal or a `usize` const)". §1.4 note: "(its A5)" to "(A5, done)", and its last sentence reworded (below). |
| `pixelflow-compiler/intent.md` | §6.4 table: Owns line 12-13, line 29, Enforced-by unit tests (new emit and parser test names), Behind line 49-52, Behind line 53-54 deleted, Decided in gains the bound plan (§1.2, §3.4). R11 applied to line 10. 60 lines (budget 80). No step label, no "structural", no "template" left in it. |
| `.claude/skills/ontology/language.md` | "Structural parameter" becomes "(retired)" with the spec's four bullets; `kernel!` Lives; Entry; Args; Binding time; Param Lives. |
| `.claude/skills/ontology/SKILL.md` | `:59` "- Structural parameter (retired)". |
| `.claude/skills/ontology/compiler.md` | "Program, template, instance": Is, Follows, Lives. |
| `.claude/skills/ontology/terminal.md` | Font programs Follows: "choosing by the piece count, a host-side choice". |
| `docs/BACKLOG.md` | Plan-table row: A5 in "landed", "next" is A1's rows after #1323, **A6**, B1-B5. New row **C11** (R14), after C10, in "Correctness and CI". |

**No edit, with evidence:** `CLAUDE.md`, `docs/DENOTATION.md`, `pixelflow-core/README.md` (§6.3's grep reproduced:
only CLAUDE.md:299, :329, :362 and README:106, all "template" in other senses). `docs/language/2026-09-24-one-pipeline.md:667`
(R16, below). Plan §0, §2's row, the A2a bullet and the A6 bullet (spec §6.1 "Not edited"). The-language-is-kernel's
§1.4 `structural` table row, D1/D3/D9, B3 (spec §6.2 "Not edited"). Ontology `language.md:178` (JP's quote), `:135`,
`terminal.md:205` (spec §6.6 "Not edited").

## Rulings

- **R3.** The Binding-time pointer (`language.md:125-126`) is worded without the phrase: "An author sees two binding
  times: **uniform** and **kernel-typed**. A count is a literal or a `usize` const, known at expansion
  (a-bound-is-a-value §1.2)." G10 now hits exactly: the bound plan, the-language-is-kernel, ontology `SKILL.md`,
  ontology `language.md` (its retired heading). `compiler.md` and `terminal.md` no longer hit.
- **R10.** Entry text: "The macro emits a host function for it that returns the `Kernel`. It rebuilds its lowered arena,
  saturated at expansion unless it carries a `Dwrt` or the macro is `kernel_raw!`; an entry that takes a kernel runs
  lowering's steps when it is called (`Staged`)." (R10's sentence verbatim; the "returns the `Kernel`" clause moved
  ahead of it so R10's clause stands whole.)
- **R11.** intent.md: "each citing its plan section (`PLAN`, `BOUND_PLAN`)". Still under the 80-line budget.
- **R13.** A7 bullet: "It exempted only test templates, which A5 rewrote over `usize` consts or deleted with the tests
  whose subject they were, and no production entry". Done line adds "the tests whose subject was the template are deleted".
- **R14.** BACKLOG **C11** is the probe's row, with two additions that are true of the probe: "Present at `7e9531e`, and
  not a-bound-is-a-value's" (C10's phrasing), and the Where cell names `emit_kernel` and `emit_closure`, whose
  `let (__arena, __root)` is the x6/x8 collision site. For `impl.md`, the probe's three corrections to R14's text stand:
  the code is E0308 (E0530 only where the emission's binding is `let mut`, i.e. `__arena`; E0005 nowhere); the hazard
  is the whole `__`-prefixed family and `__arena`/`__root` need no fold; a parameter named `__lo`/`__hi`/`__header`
  is fine. The row says "Present at `7e9531e`": after A5 it is inferred from spec §2.5, not measured.
- **R15.** Plan §5's the-language-is-kernel §1.3 row: CL cell "A5 (a literal or a `usize` const), A6 (a per-call count)",
  Becomes "a fold binder or a count (a literal or a `usize` const)", the text A5 writes at `:192`.
- **R16.** No rewrite of `one-pipeline.md:667` (its F4 row, "delete the `Param` family"): it is history under that
  document's "Superseded in part" banner, as R16 rules, and `impl.md` records that disposition. The-language-is-kernel's
  banner sentence enumerates §1.6 as well as §1.7.

## Where I departed from the spec's text, each for a sentence that would otherwise be false after A5

1. **BACKLOG plan row** says "`<const N>` deleted: `kernel!` refuses a generic on any `fn`, and the template exemption went
   with it, so A7 is empty", not "has no structural parameters". BACKLOG is not in G10's expected files (R3's list), so the
   spec's phrase would be an unexpected G10 hit.
2. **Retired entry's Follows** says every entry that takes no kernel is "offered to the optimizer at expansion", not
   "saturated": `macro_tier` declines a `Dwrt`-carrying arena and `kernel_raw!` is `Identity` (R10's own qualification).
   The same wording is the new emit test's name.
3. **`compiler.md`**: "its template is one kernel", not "its program". The lattice's extent is still structural, so one
   template compiles to a program per extent and tier; a kernel is exactly what the host function returns.
4. **The-language-is-kernel §1.4 note** ended "Until then the table is what the tree does", false for `const N: usize`
   once A5 lands. It now reads "Until B2 the table is what the tree does, but for that `const N: usize`."
5. **Banner subject**: "`kernel!` has none", not "the tree has none". The host still chooses a glyph's program by piece
   count until B2, which §1.6 and §1.7 call a structural parameter; `kernel!` has no such parameter.
6. **Plan §5 row split** into `:179` (CL per R15) and `:186` (CL "A1"; Becomes "a count is a value"), because A1's Done
   line still points at the `:186` sentence. Both Hunk cells are "none": #1431 has landed, and "Hunk" names open PRs.
7. **Retired entry, Is**: "decision 2's `n: usize` entry parameter, which A6 builds" (a plan statement, not a claim
   about the tree).

## Not done, and why

- **The A5 Done line carries no gate clause.** No `a5-impl` output exists in `.wip/gates/out` (only `a5-main`), so
  "byte_probe empty in both columns …; the 29 font programs' keys, digests and saturation telemetry … unchanged" would be
  a claim no gate has checked. Append it from the measured outputs after gates 1, 1b, 2, 2x run.
- Statements that depend on the code agent's work and that I confirmed present in the tree at the time of writing:
  `refuse_generics` in `parser.rs`, `BOUND_PLAN` in `lib.rs`, both test names cited in intent.md, `FoldTerms` without a
  stride. Re-run `scripts/check-intent.sh` after the last code edit (it resolves the paths intent.md names).
