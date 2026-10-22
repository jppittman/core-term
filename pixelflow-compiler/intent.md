# pixelflow-compiler

**Is.** The language's front end: a function from a `kernel!` block's tokens to
the code that rebuilds, at load time, the kernels it denotes. An entry is a term
of arrow type with uniform parameters (DENOTATION §2); the optimizer it hands
that term to is an `Optimize` value, never a branch (§8).

## Owns
- The surface: grammar, types (`Ty`: `f32`, `bool`, `usize`, record, kernel)
  and refusals, each citing its plan section (`PLAN`). The IR keeps one lane
  for a number and a mask; the difference lives in `Ty` (Behind).
- Names as the language compares them (`Name`: an identifier without its `r#`),
  and a method or projection resolved once, in the parser, to the `Method` or
  `Projection` the AST stores.
- Binding times as the host sees them (structural, uniform, kernel-typed), the
  host `fn`, the `Args` record, and `const` evaluation.
- Which `Optimize` each macro hands `expand` (`kernel!` saturates, `kernel_raw!`
  is `Identity`), and which entries no optimizer sees.

## Never
- No construct is defined here. Lowering calls `pixelflow_ir::library`
  (`fract`, `hypot`, `clamp`, the derivative), `ExprArena::close_over`,
  `Fold::admits` and `ExprArena::apply`; a copy is a future divergence.
- No `Dwrt` is resolved at expansion: `macro_tier` declines an arena that
  carries one, so a warp reaches its operand and the chain rule holds.
- No kernel is compiled or evaluated at expansion (`[dependencies]` holds no
  `pixelflow-codegen`; only `const` items are evaluated), and no fold is
  unrolled: it lowers to one `Reduce`.
- No stage accepts what a later one refuses: every advertised method name has a
  path through sema, lowering and emission, under both macros.
- No collection type, recursion, `X` or `Y` in a helper, mask read as a number,
  or whole-record arithmetic: each is refused where it is written.
- No call's value becomes a constant: one entry at one instantiation is one program.

## Contract
`src/lib.rs` exports exactly `kernel!` and `kernel_raw!`, both `expand(input,
optimizer)` over `front_end`, the one pipeline (`parser::parse`, `sema::analyze`,
`emit::emit_kernel`), in which every stage refuses with a `syn::Error` spanned
at the construct (one exception: a block with no entry has no construct to
span, and is refused at the invocation). Behind them: `src/parser/` (tokens to
the vocabulary, `src/ast/`, whose `KernelDef` is a sum of the closure and items
spellings), `src/sema/` (types, consts, call graph, binding times; scopes are the
vocabulary's `Scopes`), `src/lower/` (AST to a term at a `Site`; `Expansion`
is the site at expansion) — each has its contract in its `mod.rs`, its intent in
its `intent.md` — and `emit.rs` (arena to `TokenStream`; the private `Staged`
site for a kernel-taking entry). Only `pixelflow-graphics` uses it.

## Enforced by
- Types: `Ty` keeps `Bool` and `F32` apart; `Name` is the key of every map of names;
  `Method`, `Projection`: a name is resolved once; "do not optimize" is `Identity`;
  desloppify `mask-is-not-a-number` covers `pixelflow-compiler/**`.
- `tests/derivative_under_warp.rs`, `tests/the_library_is_the_builders.rs`,
  `tests/fold_is_kernel_over.rs`, `tests/binding_times.rs`,
  `tests/rustc_is_the_oracle.rs`, `tests/optimized_agrees_with_raw.rs`,
  `tests/bake_vs_scalar_truth.rs`; `src/tests/front_end.rs` `every_advertised_method_compiles`.
- Unit tests: `emit.rs` `an_optimizer_is_offered_only_entries_it_can_read`; `src/tests/lower.rs`, lowering's one
  through `Expansion::lower`; `src/tests/sema.rs`, sema's fifty-seven through `parse` then `analyze`, `recursion_is_refused` among them;
  `src/tests/parser.rs`, the parser's fifty-four through `parse`,
  `a_collection_spelled_any_way_is_refused_naming_its_sections` among them; `src/tests/ast.rs`, the vocabulary's.

## Behind
- Sema's mask/scalar typing (`Ty::Bool` against `Ty::F32`) is the arena's job:
  the typed-arena plan has `kernel!` emit typed pushes and deletes sema's
  duplicate enforcement, keeping `Count` and `Record` (typed-arena §2, T4).
- `macro_tier`, `DwrtFree` and `kernel_raw!` are in `lib.rs`, and `kernel!`
  saturates any entry with no structural or kernel-typed parameter and no
  `Dwrt`; the plan optimizes nothing at expansion unless the instance is
  declared, and deletes all three (the-language-is-kernel §1.8, D2, D14).
- A fold's ends are `u32` (`range_bounds`) and a structural count read as a
  value rides `Param(u8)`, at most 256, against the 64-bit control plane (A5, D2).
- `src/lower/site.rs` `Holes` encodes a structural entry's open fold as a placeholder
  stride over `OPEN_RANGE` in the fold's own bits: a convention the fold type
  does not state. Template instantiation (the-language-is-kernel §1.4, §1.8,
  D2) replaces it.

## Decided in
`docs/archive/plans/2026-09-08-macro-tier-is-arena-native.md`,
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.1–§1.5, §1.8, D2, D14),
`docs/language/2026-09-24-one-pipeline.md` (§1.2, M1–M3),
`docs/language/2026-10-17-the-arena-is-typed.md` (T4), `docs/DENOTATION.md` §2, §8.
