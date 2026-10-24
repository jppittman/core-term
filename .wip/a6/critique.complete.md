# A6 spec critique: completeness and byte identity

I checked `.wip/a6/spec.md` against main `29980bf`. The tree is that commit plus `.wip/`, and
`git diff --stat 29980bf HEAD -- . ':!.wip'` is empty. I also read the plan
(`docs/language/2026-10-20-a-bound-is-a-value.md`), `.wip/a6/design.md` (its R-A6 section wins),
CLAUDE.md and `pixelflow-compiler/intent.md`. The review was read-only: I ran greps and
`bash scripts/check-intent.sh`, and built nothing.

## Verdict

The code side is complete:
- Every grep hit for the lens terms is either rewritten by the spec or unchanged and still correct.
- No production byte can move.
- Every gate in the plan's A6 bullet has a named test with an assertion.
- The intent budgets hold.

Three defects block as written:
1. The order of work runs a unit test before the code it pins exists.
2. One grep gate cannot pass.
3. One live design doc becomes false and the spec leaves it out.

Four smaller defects are listed after them.

## What I checked and found sound

- **Code sites.** I ran each lens term against `pixelflow-compiler/` and accounted for every hit:
  - **Rewritten by the spec:** `UsizeScope` (4 hits), `RangeScope` (8), `range_bounds` (6),
    `literal_range` (3), `eval_usize` (7), `open_fold` (10), `close_fold` (11), `FoldTerms` (6),
    `Parameter`/`Scalar` (`sema.rs:243-300`, `emit.rs:141/163/247/681`, `lower.rs:63/165/319/394`),
    `write_into` (16), `set_declared` (7), `names_a_count`, `a_count_is_not_a_value`.
  - **Unchanged and still correct:** the remaining `Ty::Usize` hits (`sema.rs:143/153/179/411/548/909/926/1035/1458/1924`,
    `symbol.rs:197`, `lower.rs:898`).
  - **`Bounds`:** only the test records.
  - **G1 at the base:** finds exactly the sites §3.1, §3.2 and §4.1 rename (`lower.rs:195/395/632/649/692/701/781`,
    `sema.rs:267/272`, `emit.rs:164-165`).
  - **The trait impls:** there are two `Site` impls and two `UsizeScope` impls, and `emit_entry` has one
    caller (`emit.rs:118`). The spec covers all of them.
- **Byte identity.**
  - The one production block (`pixelflow-graphics/src/fonts/loop_blinn/program.rs:31-141`) has no
    `usize`, `.map(`, `.any(` or `.all(` (G6 prints `0`).
  - `one_piece(p: Row)` and `glyph`'s `bounds: Bounds` classify as `ParameterKind::Record` in the same
    order (`record_named` first) that `parameters()` uses today.
  - `argument(Scalar::Field)` is today's `quote!(#param.#field)`.
  - `emit_entry` adds no tokens when there is no count: `Option` and the empty iterator interpolate as
    nothing, and the test at `emit.rs:793` keeps passing.
  - `Staged::fold_terms` interpolates a `u64` for a literal end exactly as `range.start` did. Production
    has no fold anyway.
  - `byte_probe` has no compiler dependency. Only `pixelflow-graphics` depends on `pixelflow-compiler`.
  - Under arm (a), `OnePieceArgs::write_into` changes, but that is host code nothing calls.
- **The plan's gates as named tests.**
  - `a_triangular_fold_folds_as_rustcs_does` (26 at `X = 3`) and `a_per_call_count_folds_as_rustcs_does`
    (with the plan's `rust_sum(n: usize, x: f32)`).
  - `the_args_write_refuses_a_count_past_the_domain`: `Err` plus "nothing written" under (a), and
    `should_panic` under (b).
  - `fold_is_kernel_over.rs` gains two cases.
  - byte_probe is §8 step 7.
  - Every helper these tests call exists with the signature used (`FOLD_SAMPLES`, `bake`, `plane`,
    `program_key`, `code_at_frame`, `canonical(..).key/.uniforms`, `binders`, `holds_a_placeholder`,
    `check`, `four_ulps`).
  - No new item name collides in its file.
- **Intent budgets.** `check-intent.sh` passes at the base. `pixelflow-compiler/intent.md` is 63 lines and
  the spec adds about 5. `pixelflow-core/intent.md` is 71 lines and adds 1 under (a). The budget is 80.

## Findings, ranked

### 1. Steps 2 and 3 run a unit test whose `write_into` half lands in step 4 (blocking)

- **The ordering.** Step 2 (spec:1922-1924) edits `emit.rs` with "`write_into` as today plus nothing
  yet", then runs `cargo test -p pixelflow-compiler --lib`. Step 3 runs the whole crate. §4.3
  (spec:861) leaves the `write_into` refusal to D-A6-1, which is step 4 (spec:1927).
- **The test.** §6.5's emit unit test is part of step 2 ("unit tests §6.5"). It ends with:
  ```rust
  let refused = code.find("KnownRange :: admits (& (0 .. self . n as u64))").expect("write_into refuses");
  ```
  (spec:1714). That code is emitted only by an arm of D-A6-1 (spec:970 or 1010).
- **The result.** The test panics in step 2 and again in step 3, so neither step's check can pass as
  ordered.
- **Fix.** Write the emit test without its last block (spec:1714-1716) in step 2, and add that block in
  step 4 with the arm. Or move the whole test to step 4.

### 2. G2 expects nothing, but an unrelated line matches at the base and after (blocking)

- **The claim.** Spec:1953 says G2 "expected: nothing; at the base it finds `pixelflow-ir/src/fold.rs:441`".
- **The base.** At `29980bf`, G2 prints two lines. The second is another plan's step label in a file A6
  does not touch:
  ```
  ./pixelflow-ir/src/fold.rs:441:    /// (`Kernel::over_values`, and `kernel!` from A6); every other constructor
  ./pixelflow-pipeline/src/bin/guide_scope_saturation_delta.rs:11://! chess-engine ~98% (`docs/archive/plans/2026-08-17-cost-model-domain.md`, A6).
  ```
- **The result.** After §7.8, G2 still prints the second line.
- **Fix.** Add `| grep -v guide_scope_saturation_delta.rs` to the pipe, or expect that one line.

### 3. §7.2 edits one line of the-language-is-kernel, and A6 makes others false (blocking: a missed doc site)

Spec §7.2 (spec:1805-1811) edits only `:197`. A6 also falsifies the lines below.

- **`:262-264`.** It reads: "`write_into` streams the values by position into any `UniformBlock`
  (`set_declared`), and the one check is the count (`ArityMismatch`)." Under either arm of D-A6-1,
  `write_into` also refuses a count parameter past 2²⁴, so it no longer has one check. The spec rewrites
  the ontology's parallel sentence to "checks only the arity and each count's domain" (§7.4 →
  `.claude/skills/ontology/language.md:107-109`), but not this one.
- **`:52-59`.** This is the "Reversed in part" note. It says that until the plan's steps land, §1.5's
  "Ranges are constant" (`:299`: "`a` and `b` are expressions over literals and structural parameters")
  describes what the tree does. A5 added a sentence there for its own deletion. After A6, a `kernel!`
  range may be an enclosing index or an entry's `n: usize`, so the note needs an A6 clause too.
- **`:111`.** "Until then the quote's rule, constant ranges, is what the tree does" is false at the
  language surface once A6 lands.
- **Fix.** Add these three edits to §7.2, worded as the ontology edit is for `:262-264`.

### 4. G9's stated expectation contradicts text the spec keeps (non-blocking)

- **The claim.** Spec:2008 says the second G9 grep finds "only `lib.rs:77`'s definition and test
  assertions".
- **The base.** It also finds `lib.rs:281` (the `# Binding times` count paragraph) and `parser.rs:103`
  (module doc).
- **After A6.** §7.6 keeps `lib.rs`'s citation ("(docs/language/2026-10-20-a-bound-is-a-value.md §1.2,
  §3.4)"), and §2.6's rewrite of `parser.rs:97-103` keeps its own. Both lines are still there.
- **Fix.** Name them in the expectation.

### 5. Step 5 says `pixelflow-core`'s tests expand `kernel!`, and they do not (non-blocking)

- **The claim.** Spec:1929 runs "`cargo test -p pixelflow-core` (its tests expand `kernel!`)".
- **The tree.** `pixelflow-core/Cargo.toml` has no `pixelflow-compiler` dependency, dev or otherwise.
  `kernel!` appears in `pixelflow-core/tests/` only in comments (`bind_allocates_nothing.rs:116`,
  `reduce_binder_reads_bound_buffer.rs:28`).
- **What does expand it.** Outside the compiler, only `pixelflow-graphics` (and `cargo check --workspace`)
  expands `kernel!`.
- **Effect.** Under arm (a), the emitted `::pixelflow_core::WriteRefused` path is exercised only by
  the compiler's own tests. That is enough, but the step's reason is wrong.
- **Fix.** Drop the parenthesis.

### 6. §6.7's description of the unchanged files is wrong; its totals are right (non-blocking)

- **The claim.** Spec:1755 says "ast 4, lib 4, parser 43, symbol 7, and nine test files, 63 in all |
  137".
- **The count.** The unchanged test files are eight: `bake_vs_scalar_truth` 5, `derivative_under_warp` 6,
  `items_block` 6, `jit_parity` 12, `kernel_5params` 5, `kernel_composition` 6, `kernel_macro` 30,
  `the_library_is_the_builders` 9. Together they hold 79 tests, and 58 + 79 = 137.
- **The totals.** The 299 → 318 figures check out.
- **Fix.** "eight test files, 79 in all".

### 7. Three grep families have no disposition, and two texts stay narrow (non-blocking)

- **`has_args_record`** (`ast.rs:200`, `sema.rs:468`, `emit.rs:224-226`) appears nowhere in the spec. It
  is unchanged and still right: a count is a uniform, so `f(n: usize)` gets `FArgs`. Say so in one
  line, since `refuse_a_taken_args_record` now covers count-only entries.
- **The unchanged `Ty::Usize` arms** (`sema.rs:411`, `:548`, `:1458`) have no disposition either. They
  are correct as they are.
- **`param_type`'s entry refusal** (`sema.rs:638-645`, "an entry's parameter is an `f32` or a record …
  an `f32` is one, a record is one per field") is the refusal of `pub fn f(m: bool)`. After A6 it
  omits the count parameter, while §2.5 widens every sibling note (`of_a_value`,
  `a_count_is_not_a_value`, `type_of_cast`, `convert_cast`).
- **`tests/rustc_is_the_oracle.rs:595`** says "A count is a `usize` const". It sits in a section A6
  leaves in place, beside the new "values as bounds" section.
- **Fix.** Add these to §2.5, §2.6 and §6.1, or record them as unchanged.
