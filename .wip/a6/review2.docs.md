# A6 review 2: docs truth

Against HEAD `079c01d`, read-only on the repo. First every sentence in `git diff 6c60dfa -- . ':!.wip'` (round 1,
applied by hand). Then every sentence the whole A6 diff (`git diff 29980bf -- . ':!.wip'`) adds or edits that
`review1.{docs,complete,laws,stages}.md` did not already clear. Each is checked against the code, not against a
ruling's wording.

The lens covers:

- generated doc text: both `# Panics` sections, `write_into`'s doc, `COUNT_DOMAIN`, the panic message;
- M1–M7 and their notes and help;
- the plan's §1.4, §3.4, §5 row, the A6 bullet and its Done line;
- the-language-is-kernel's three edits;
- the ontology rows;
- both `intent.md`;
- BACKLOG C8, C13, E9 and the plan row;
- the core README.

## How it was checked

- **Reading.** Sources and docs at HEAD, with `grep` for every cross-reference a sentence makes.
- **One experiment,** for the claim the round-1 edits rest on: annihilation is the one optimizer edge that takes
  a domain NaN.
  - The scratch crate is `/home/user/scratch/a6-review2-docs`, built against the repo's `target.noindex` with
    the workspace `RUSTFLAGS`. Its test is `tests/edges.rs`.
  - Each entry is defined under both `kernel!` and `kernel_raw!`. The test prints each expansion's arena, then
    evaluates a program compiled at `n = 0` with its block written raw (`set_declared([v])`) at `(3, 7)`.

```sh
cd /home/user/scratch/a6-review2-docs
CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  cargo test --offline --test edges --no-run
PIXELFLOW_ISA=avx2   /home/user/pixelflow/target.noindex/debug/deps/edges-031362fa2b859e9d --nocapture
PIXELFLOW_ISA=avx512 /home/user/pixelflow/target.noindex/debug/deps/edges-031362fa2b859e9d --nocapture
```

Both tiers print the same output.

## Round 1's sentences (`6c60dfa..HEAD`)

Most are true of the code. These were checked and hold:

- **BACKLOG C13.**
  - Lowering refuses only `open_folds >= Binder::COUNT` (`lower.rs:655`), so it admits `Binder::COUNT` deep.
  - `collapse` takes three free binders and panics "collapse: no free binder slot …" (`passes/lattice.rs:91-106`).
  - Both pieces date from `057f7fe` (2026-10-14). The plan is `298d770` (2026-10-22), so the gap is older than
    the plan, as the row says.
- **the-language-is-kernel.**
  - `:59` and `:113` now name a-bound-is-a-value's A6.
  - `:199` says "a `usize` const's `Const` (as a value) or literal bound". That is `lower_cast`'s
    `site.constant` for `N as f32`, and `Bound::Lit` in `fold_bounds` for a bound.
- **The plan.**
  - §1.4 and §3.4 say "and, if it has one, by its `Args` record". An entry that takes a kernel has none
    (`emit.rs:1085`).
  - The Done line is covered by finding 1.
- **`lower.rs`.**
  - **The const-first comment (`:680-682`).** Sema refuses a binding that shadows a const at every binder:
    a parameter (`sema.rs:783`), a `let` and each name of a tuple `let` (`:1634`, through `let_type`), and a
    fold's index (`:970`), all through `refuse_shadowing_an_item` (`:830-841`).
  - **The `fold_bounds` doc and the lowering test's doc.** The domain checks are `literal_range` and
    `literal_bound`, and only lowering has them.
- **The parser.** The grammar's `'(' bound ')'` matches `eval_usize`'s `Paren` arm and `bound_names`' `Paren` arm.
- **Sema's texts.**
  - The `param_type` refusal.
  - `count_parameter`'s doc, and M7's note without the `Args` record.
  - `not_a_usize_bound`'s doc. It matches the match at `sema.rs:1002-1014` arm for arm: an `f32` const, an
    entry's `f32` by spelling, none otherwise.
  - M1's note and M3's help.
  - `Count::Name`'s doc and `RangeScope`'s comment. The `F32` arm is unreachable: both callers judge every name
    first (`sema.rs:962-967`, `lower.rs:639-644`), and the only other call is a unit test over known bounds.
- **The rest:**
  - `register_parameter`'s doc;
  - `BOUND_PLAN`'s doc;
  - lib.rs `# Folds` ("or `Kernel::over_values` for a bound that is a value"), N3's "number of arguments", and
    the factoring edge;
  - `COUNT_DOMAIN`'s semicolon;
  - the emit test's `write_into` slice;
  - `fold_is_kernel_over.rs`'s lead (`ExprNode::Reduce` holds a `fold: Fold`);
  - the staged-refusal test's doc (its assert precedes the body: `emit.rs:220-242`);
  - the oracle doc's factoring edge;
  - `BoundOutOfDomain`'s doc.

Two of round 1's sentences are not: the Done line's edge (finding 1) and `intent.md` (finding 4).

## What else A6 writes that holds

- **The generated text:**
  - **The host function's `# Panics`.** Emitted after the entry's own attributes. It reads "If a count
    argument is past 2^24: …". It does not claim to be the whole list (ruling S4).
  - **`write_into`'s `# Errors` and `# Panics`.** "Of the block, only the number of arguments is checked", and
    "Nothing is written".
  - **The panic message.** One `{}`. Neither `COUNT_DOMAIN` nor `BOUND_PLAN` has a brace, so the message
    is a valid format string for `assert!`.
  - **`2^24 + 1` is written as `2^24`.** Rust's `usize as f32` rounds a tie to even. Every larger count lands
    above the domain.
- **The refusals:**
  - M2's note, and M2's help by kind, type, role and spelling.
  - M4 (decision 8, `refuse_a_value_bound_under_a_mask` and lowering's twin).
  - M5's `not_a_count` note.
  - M6 ("a helper's parameter is an `f32`, a `bool`, a record or a kernel"): helpers do take kernel-typed
    parameters, `sema.rs:633`.
  - The widened `usize` notes in `of_a_value`, `a_count_is_not_a_value`, `type_of_cast` and `convert_cast`,
    and the generic refusal's help.
- **The module and type docs:**
  - sema's module doc items 3, 5 and 6 and its resolution list;
  - `Ty`, `ParameterKind`, `Scalar`, `parameters`, `parameter_kind`, `signature`;
  - `bound_named`, `bound_names`, `range_bounds`, `UsizeScope` and its `Output` and `operand`;
  - lowering's module doc, `Binding::Count`, `Site::open_fold`, `lower_fold`, `count_term`, `lower_cast`;
  - emit's module doc, `emit_kernel`, `argument`, `emit_closure`'s comment, `emit_entry`, `emit_args`,
    `count_refusal`, `Staged::fold_terms`;
  - parser's module doc: the `type` and `iexpr` rows, the legend, the binding-times paragraph, and the
    refused-by-sema and refused-by-lowering lists;
  - ast's `Param`, `FoldExpr`, `CastExpr`;
  - symbol's table row and parameter paragraph.
- **Test docs:**
  - **`binding_times.rs`.** The module doc, every new test's doc, `HANG` and `within`. "The loop not entered"
    is the guard's `If` arm skip, `passes/guard.rs:19-27`.
  - **`kernel_typed_parameters.rs`.** "declared first" matches the defaults `[3.0, 2.0]`.
  - **The other test files:**
    - `items_block.rs`;
    - the oracle's new section, whose worked sums are 26 and 63;
    - `fold_is_kernel_over.rs`'s two new tests, binders `[5, 4]` with `REDUCE_BINDER_BASE` 4;
    - `optimized_agrees_with_raw.rs`;
    - the sema tests' docs other than finding 2.
- **The plan and the docs around it:**
  - The A6 bullet's gate. `rust_sum(n: usize, x: f32)` and the `Args` refusal test both exist.
  - The §5 row's third column, which matches the-language-is-kernel `:199`.
  - the-language-is-kernel `:263-265`.
  - The BACKLOG plan row.
  - The ontology rows other than the edge. Args "checks only the arity and each count's domain" and Lives
    "panics on a count past 2²⁴" both hold.
  - `pixelflow-compiler/intent.md`'s Never, Enforced-by and Behind lines. `a_range_bound_is_a_usize_and_reads_no_coordinate`
    exists.
  - `pixelflow-core/intent.md` is untouched, and nothing in it is false after A6. It names `UniformBlock` as
    "its scalar arguments", and claims nothing about counts.

## Findings, most severe first

### 1. Blocking. The docs name annihilation as the one way an optimizer takes a domain NaN, and it is not

**What the docs say.** A6's round-1 edits treat annihilation as L5's only edge:

- **The plan's Done line** (`docs/language/2026-10-20-a-bound-is-a-value.md:725-727`): "a raw block write that
  names no count makes the fold NaN by the guard, up to the annihilation edge `Kernel::over_values` documents
  (BACKLOG E9)".
- **The core README** (`pixelflow-core/README.md:67-68`): "(up to BACKLOG E9's annihilation)".
- **The ontology Range row** (`.claude/skills/ontology/language.md:522-523`): "(E9 aside)".

**Why round 1 wrote them.** Round 1 based these on `review1.docs.md:66`: "Only the annihilator, on a factor
proven zero, removes it." A6 also widened BACKLOG E9 (`docs/BACKLOG.md:153`) to count an enclosing zero factor
as taking the fold: "a zero factor written beside a value-bounded fold (`0.0 * (0..n)…`) takes the fold with
it".

**The other rules that remove a fold.** On that reading, the production rule set holds three more rules that
take an enclosing fold the same way, at expansion under `kernel!` and at bake through `Manifold::compile` for
any kernel:

- `Cancellation::<AddNeg>`, `(x + a) − a → x`;
- `InverseAnnihilation::<AddNeg>`, `x + neg(x) → 0`;
- `InverseAnnihilation::<MulRecip>`, `x · recip(x) → 1`.

They are listed at `pixelflow-search/src/math/algebra.rs:1080-1089`.

**What the probe measured.** All values are at `(3, 7)` with the raw count written as each of `2.5`, `−1`, NaN
and `16777218`. The same output appeared under `PIXELFLOW_ISA=avx2` and `=avx512`.

| entry | `kernel!` expansion | `kernel!` | `kernel_raw!` | as written |
|---|---|---|---|---|
| `(Y + Σ) − Σ` | `Var(1)` | 7 | 7 | NaN |
| `Σ − Σ + Y` | `Var(1)` | 7 | 7 | NaN |
| `(Σ / Σ) · Y` | `Var(1)` | 7 | 7 | NaN |
| `0.0 · Σ + Y` (E9's own example) | `Var(1)` | 7 | 7 | NaN |
| `Σ_{i<n} (X·i − X·i) + Y` | `add(reduce[add_4[0.._)](Const(0), Uniform(0)), Var(1))` | NaN | NaN | NaN |

**What the probe shows.**

- No zero factor appears in the first three rows. The fold is gone at expansion: the `kernel_raw!` arena
  still holds both `reduce` nodes, and the `kernel!` arena is `Var(1)`.
- The output is `Y` where the program as written is NaN.
- The last row shows the edge is the enclosing expression's. A body that cancels keeps its fold, and the guard
  still answers NaN.

**Why it is blocking.** The Done line's "up to the annihilation edge" is presented as the whole list of
exceptions. For `cancel(n)` the raw write gives no NaN, and nothing annihilates. Under the reading E9's
widening takes, which is the kernel's output, the list is incomplete. Under the other reading, which is the
fold's own value, the enclosing `0.0 *` that A6 added to E9 is not an L5 edge at all. Either way, a sentence A6
wrote is false of the code.

**Fix.** Choose the reading and say it once.

- **The fold's own value.** "A raw write that names no count makes the fold NaN by the guard. The one rule that
  changes the fold's own value is E9's (`FactorFold`, then annihilation)." Then move E9's `kernel!` sentence
  out of the L5 accounting. An expression around the fold is subject to every algebraic rule, as any NaN is
  (CLAUDE.md, "Floating point at the edges").
- **The kernel's output.** Name the family wherever the edge is cited, in the README, the ontology row and the
  Done line: "an algebraic rule unsound at NaN (annihilation, cancellation) can drop a fold from an expression
  around it." Widen E9's title the same way.

The probe also shows `(Y + min_∅) − min_∅` is `Y` at `n = 0`, where rustc gives NaN. That case is the
enclosing expression's IEEE edge, not the fold's identity. So L2's "one edge" (lib.rs `# Folds`, `ast.rs:621-622`,
the oracle's doc) is true on the fold-value reading, and becomes false only if the kernel-output reading is
chosen. If that reading wins, L2 needs the same widening.

### 2. Blocking. A sema test's doc misstates the help it pins

`pixelflow-compiler/src/sema.rs:2870-2871`, the doc of `a_range_bound_is_a_usize_and_reads_no_coordinate`, says:
"The help of a name that is an `f32` is the entry's `n: usize`, where an entry of the items form can take
one."

This is false in two ways:

- **The test's own row disagrees.** `const R: f32` in an items-form entry (`:2916-2922`) asserts the help
  "declare the count as `const R: usize`". The arm is `sema.rs:1003-1005`: an `f32` const's help is the const,
  whatever the form.
- **Two other `f32` names get no help.** A `let` (`:2912-2914`) gets none, and so does a helper's `f32`
  parameter: `:2971-2975` asserts `!err.contains("help:")`.

This is round 1's D9 defect again, in the test that pins it. D9 corrected `not_a_usize_bound`'s doc and
left this one.

**Fix.** "The help is by what the name is: an `f32` const's is a `usize` const; an entry's `f32` parameter's is
`n: usize` (the items form's); any other has none."

### 3. Blocking, minor. `bound`'s doc describes only half its calls

`pixelflow-compiler/src/lower.rs:728`: "One end of a fold whose other end is a value: a literal, checked alone
against the domain, or the term a count binding holds."

`fold_bounds` calls `bound` on both ends whenever the pair is not both known (`:725`). For `(0..n)`, the call
`bound(Count::Name(n))` is for an end whose other end is the literal `0`.

**Fix.** "One end of a fold with a bound that is a value".

### 4. Non-blocking. Three sentences presuppose an `Args` record that a staged entry lacks

`pixelflow-compiler/intent.md:15-17` was rewritten in round 1 (complete N3). It says "refused past 2²⁴ by the
entry's host function and its `Args` record's `write_into`". Two older sentences say the same, and round 1
cleared both:

- the ontology's `kernel!` Follows, `.claude/skills/ontology/language.md:72-74`: "in the entry's host function
  and its `write_into`";
- BACKLOG C8, `docs/BACKLOG.md:166`: "and so does its `Args::write_into`".

An entry that takes a kernel and a count (`columns_to`, `kernel_typed_parameters.rs:111-113`) has no `Args`
record. `emit.rs:1085` asserts `!code.contains("ColumnsArgs")`. Round 1's N4 added "if it has one" in four
places for exactly this reason. These three are the same presupposition. Round 1 classed it non-blocking, and
so does this review.

**Fix.** "and, if it has one, its `Args` record's `write_into`" in each of the three.

## Seen, not findings

- **M7's note and `count_parameter`'s doc** give the closure form's reason as "rebound by position, with
  `f32`s, which cannot refuse one". The sentence is true. A staged items-form entry is rebound the same way
  (it has no `Args` record), and it takes a count. So the stated reason is not what separates the two forms:
  spec D-A6-3 rejected an assert in the closure as a design choice. This is design rationale, not a false
  sentence.
- **`binding_times.rs` `the_host_function_refuses_a_count_past_the_domain`** gives `2²⁴ + 1`'s reason ("its
  uniform would hold `2²⁴`"). That is the test's own value. `COUNT_DOMAIN` carries the general case.
- **`parser.rs:101-103`** still states an empty range's identity bare (round-1 complete N4). A6 did not edit
  it.
- **`literal_range`** reports a reversed known pair as "reaches past 2^24" when lowering runs without sema (the
  unit tests' `lower_unanalyzed`). This predates A6, and sema refuses the pair first.
