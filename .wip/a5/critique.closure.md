# A5 spec critique: deletion closure

Read-only review of `.wip/a5/spec.md` (and `.wip/a6/design.md` §9-§10 where they constrain
A5) against the tree at `7e9531e` (`git diff --quiet 7e9531e HEAD -- . ':!.wip'` holds),
the plan `docs/language/2026-10-20-a-bound-is-a-value.md`, `CLAUDE.md` and
`pixelflow-compiler/intent.md`. Lens: deletion closure.

## What I re-ran

Every grep in the brief, over the whole workspace (all crates, tests, examples, benches,
`docs/` outside `archive/` and `results/`, `.claude/`, every `intent.md`, root agent docs,
`scripts/`, `.github/`):

- each structural-path name (`StructuralRange`, `instantiated_range`, `FoldRange`, `Holes`,
  `OPEN_RANGE`, `HOLE_STRIDE`, `ExpansionFold`, `structural_generics`,
  `structural_parameters`, `refuse_a_helpers_generics`, `register_structural`,
  `EntryContext`, `is_usize`, `known_usize`, `Lowered`, `Count::Structural`,
  `Bounds::Structural`, `SymbolKind::Structural`);
- `structural` (194 hits inside `pixelflow-compiler/`, every one mapped to a row of
  §2/§5/§6.4; outside it, every concept-level hit triaged);
- `<const`, `::<N>`, `Args::<`, `Param(` / `push_param` / `ExprNode::Param` in the compiler;
- `HOLE`, `hole(s)`, `instantiat`, `template`;
- every `FoldTerms` constructor and every `stride` in the compiler and `kernel.rs`;
- the step labels in `*.rs` and `intent.md`.

All five `fn … <const` hits outside the compiler are ordinary Rust, as §3 says. The
production `kernel!` block has no generic and no fold. `FoldTerms` is built only in
`pixelflow-ir` and the compiler. `#[test]` counts per file match §5.4 (306). The
intent.md budget holds (59 lines now, about 60 after §6.4). Nothing of A6 is built:
`BOUND_PLAN` has a user in A5 (the `const` generic refusal), and everything §10 of the
design asks A5 to keep is kept as existing code, not new machinery.

The inventory is close to complete. What follows are the places where it is not, ranked.

## Findings

### 1. G3 cannot pass as written: the spec's own panic text trips it (blocking)

§2.5 rewrites `node`'s `Param` arm (`emit.rs:520-530`) to
`panic!("kernel! produced ExprNode::Param({k}) — a `Param` is a rewrite rule's metavariable,
instantiated before any arena reaches emission; lowering builds none")`. G3 is
`grep -rniE 'template|instantiat|\bholes?\b' pixelflow-compiler/` with "expected: only
`src/lib.rs`'s `macro_tier` doc". The mandated text contains `instantiated`, so G3 finds
`src/emit.rs` too. Every other G3 hit at `7e9531e` is deleted or rewritten by a §2/§5/§6.4
row, so this is the only surplus hit, and the spec put it there.

Fix: reword the panic (for example "a rewrite rule's metavariable, bound before any arena
reaches emission"), or list the `emit.rs` panic arm in G3's expected output.

### 2. G8 cannot pass as written: `.as_chunks::<3>()` in pixelflow-graphics (blocking)

G8 is `grep -rnE '<const [A-Za-z_]+: *usize|::<[0-9]+>|Args::<' pixelflow-compiler/
pixelflow-graphics/ pixelflow-core/`, "expected: only the parser's new test, in its
`quote!` cases". `::<[0-9]+>` also matches ordinary Rust at
`pixelflow-graphics/tests/common/mod.rs:145`, `:148` and `:193` (`.as_chunks::<3>()`),
which A5 does not and must not touch. The gate's stated result cannot be met.

Fix: narrow the turbofish alternative to a `kernel!` host spelling (for example
`[a-z_]+::<[0-9]+>\(` with `as_chunks` excluded), or list the three `as_chunks` hits as
expected.

### 3. `arena_to_tokens`' `# Panics` still names the deleted `context` (blocking, trivial)

The §2.5 row for `emit.rs:425-432` rewrites the first paragraph ("one per argument of
`context`" becomes "one per entry of `arguments`") but not the `# Panics` section at
`emit.rs:436`: "If the arena's uniform table is not `context`'s arguments one for one".
After M1 the parameter is `arguments: &[TokenStream]` and `context` (the `EntryContext`
the spec deletes) names nothing. A doc of the deleted path is left behind.

Fix: add `emit.rs:436` to the row: "is not `arguments` one for one".

### 4. `Frame<'a, S>`'s lifetime goes dead with its one field (non-blocking)

`lower.rs:592-597`: `struct Frame<'a, S: Site> { role, structural: &'a [Ident], locals }`.
`'a` is used only by `structural`. The §2.4 row for `:587-598` deletes the field and says
nothing of the parameter, so the spec as written does not compile (E0392, lifetime
parameter never used). The closure is `Frame<S: Site>`, and `Lowering::frame`
(`lower.rs:602`) becomes `Frame<S>`. Forced by rustc, so it costs a build cycle rather
than a decision, but the spec claims every such edit is made in it.

### 5. Two single-term fold pins are dropped though their subject survives (non-blocking)

`kernel_typed_parameters.rs:267` (`mean_down::<1>`: a one-term fold around an application
in a `Staged` entry, `N as f32 = 1`) and `rustc_is_the_oracle.rs:679-680`
(`mean_index::<1>`: a one-term range in const arithmetic, through a helper, against
rustc) are deleted by E2 and §5.2 as "a second instantiation". A fold over one index is
not a template subject: it is a literal range after A5, as before, and nothing else in
`pixelflow-compiler/tests/` folds over a one-index range (every range there is `0..2`,
`0..3`, `0..4`, `1..4`, `3..3`, or a `const` range of 3 or 4 terms). The `<3>` case is
covered by `a_range_from_consts_folds_as_rustcs_does` (`/ TERMS`, three terms); `<1>`
is not.

Fix: keep a count-1 case over a const, for example `const ONE: usize = 1;` beside `DOWN`
with `assert_eq!(bake(&mean_down_one(&k)), 8.0)`, and in the oracle a second entry over
`FIRST..FIRST + ONE` compared against `rust_mean_index`'s one-term twin.

### 6. G10's expected files omit `SKILL.md`, and §6.6's own text fails its criterion (non-blocking)

G10 matches `structural parameter` case-insensitively. §6.6 rewrites
`.claude/skills/ontology/SKILL.md:59` to "- Structural parameter (retired)", which still
matches, but G10's expected files are "the-language-is-kernel, the bound plan, ontology
`language.md` and `compiler.md`". §6.6 also rewrites `language.md:125-126` (Binding time)
to "…(structural parameters retired, a-bound-is-a-value A5)", a hit outside the retired
entry, which G10's criterion ("history under a banner, the bound plan's own text, or the
ontology's retired entry") does not admit. Conversely, after §6.6, `compiler.md` has no G10
hit at all ("structural values" does not match).

Fix: list `SKILL.md` in the expected files, drop `compiler.md`, and admit "a pointer to
the retired entry" in the criterion (or word the Binding-time pointer without the phrase).

### 7. Two live-doc hits have no disposition (non-blocking)

- `docs/language/2026-09-24-one-pipeline.md:667`, the F4 row: "parameters are uniforms or
  structural (the-language-is-kernel §1.4)". The spec never mentions this document. It
  carries a "Superseded in part" banner, so it is defensible to leave, but the inventory
  should say so.
- `docs/language/2026-09-25-the-language-is-kernel.md:324` (§1.6: "The host chooses a
  glyph's program by a structural parameter, its piece count"). §6.2's banner sentence
  enumerates §1.1, §1.5, §1.7 and "What the macro emits", but not §1.6. The banner's
  general clause covers it; the enumeration reads as complete and is not.

### 8. intent.md's "each citing its plan section (`PLAN`)" goes inexact (non-blocking)

`pixelflow-compiler/intent.md:9-10`: "refusals, each citing its plan section (`PLAN`)".
After A5 the `const` generic refusal cites `BOUND_PLAN` (§1.1), and A6 adds more. §6.4
does not touch this line. Fix: "(`PLAN`, `BOUND_PLAN`)".

### 9. The hygiene pin's stated reason is wrong; its hazard survives for `pub const` (non-blocking)

§5.1 deletes `a_structural_parameter_named_as_the_emissions_locals_is_its_own` with "no
emitted local is spelled from a user name after A5". That was never the hazard: the
emission's own `__lo`/`__hi` (`emit.rs:610`, kept) were captured as a pattern by a
same-named const generic in scope (E0158, `emit.rs:397-399`). A `pub const` of the block
is emitted at the host fn's own module scope (`emit_const`, `emit.rs:351-370`), so
`pub const __lo: usize = 2;` beside an entry with a fold puts a const named `__lo` in
scope of `let (__lo, __hi) = __header.attach(…)`, which rustc reads as a constant pattern
and refuses. This is not an A5 regression: it holds at `7e9531e` and nothing pins it.
But A5 deletes the only test of the class with a reason that says the class is gone.
Fix: correct the reason in §5.1. A pin, or a `sema` refusal of a `__`-prefixed item name,
is optional and belongs to whoever owns emission hygiene.

## Checked and found sound

- The template exemption's deletion (§3): its condition reads only `entry.structural`
  (`emit.rs:154`), and no production entry reaches `entry_arena` with a generic.
- Every non-test user of `TokenStream` and `quote!` in `sema.rs` goes with `Count`,
  `StructuralRange` and `eval_usize`'s token building. `FoldHeader` in the `lower.rs`
  tests and `Fold` in the `emit.rs` imports are each left with only the users §2.4 and
  §2.5 name.
- `KnownRange::stride`/`range` and `ExprArena::known_range` lose their compiler callers
  but keep IR, search and codegen users. `Fold::strided` keeps `pack` and `HalveFold`.
  `push_param` keeps the search crate's users.
- No intra-doc link to a deleted item survives the §2 rows (`[`Holes`]`,
  `[`instantiated_range`]`, `[`Site::count`]`, `[`OPEN_RANGE`]`, `[`HOLE_STRIDE`]` are
  each inside a deleted or rewritten doc).
- No script, CI step or doc outside the compiler names a deleted or renamed test.
- G1, G2, G4, G5, G6, G7 and G9 give the stated results once §2 is applied. Every hit at
  `7e9531e` is deleted or rewritten by a row, and no new text adds one.
