# A6 spec critique: front-end stage agreement

Spec: `.wip/a6/spec.md`. Checked against main `29980bf` (the tree is that plus `.wip/`), the plan
`docs/language/2026-10-20-a-bound-is-a-value.md`, `.wip/a6/design.md` (its R-A6 section wins),
CLAUDE.md and `pixelflow-compiler/intent.md`. Read-only. The one build was a scratch crate,
`/home/user/scratch/a6-stages-probe`.

## Verdict

The front end agrees with itself. I traced every new spelling (`0..i`, `i..4`, `0..n`, `n..8`,
`lo..hi`, `(n)`, `LO + N..n`) and every new refusal (M1 to M7, lowering's own refusals, the host
refusals) through the parser, sema, lowering at both `Site`s, emission, the macro tier and the IR's
`push_reduce`/`close_over`. I did this for `kernel!` and `kernel_raw!`, the items form and the closure
form, helpers, records and kernel-typed entries. I found no stage that accepts what a later stage
refuses.

R-A6's typing order holds in sema. `type_of_bound` walks the same shapes `eval_usize` walks (all 12
`Expr` variants are covered). It runs over both bounds before `range_bounds`. `eval_usize` evaluates
both operands and checks the operator before `operand` judges arithmetic. So `X`/`Y` are refused by
L3 and any other non-`usize` name by type, both before `BoundArithmetic`. `0..i + X.floor()` and
`0..(i < 2)` are M5, not M3.

The `Count` / `UsizeScope` / `ParameterKind` / `Scalar` types do not guard unrepresentable states,
with one exception (finding 4). Five findings below, ranked. Only the first blocks, and only because
the spec's own grep gate cannot pass as written.

## What I checked and found sound

- **Scope.** Sema types a bound before `push_scope`, and lowering resolves one in `fold_bounds`
  before `push_scope`/`bind`. So the fold's own index is never its bound, and an inner `|i|` may
  shadow the outer `i` its bound reads. Both stages use `Scopes` with the same order. A helper's
  frame holds only its own parameters and indices at both stages (`inline`, `lower.rs:734-764`;
  sema analyses each `fn` separately). Neither stage lets a caller's count or index into a helper.
- **Classification.** `param_type` (record, then `FunctionType`, then `from_syn == Usize`) and
  `parameter_kind` (record, then `from_syn`) decide which parameters are counts in the same order.
  `parameters()` drops kernels first. Uniform order is the same at lowering, in
  `arena_to_tokens`' `arguments`, in `Staged::uniform`, and in `emit_args`' fields and values.
- **Staged.** A count's term is `__t` from `push_uniform`, and an index's is `__t` holding
  `__f.index()`. Both are bound before the inner fold's `open_fold` statement and valid in the copy
  it makes (`arena.rs:1798-1838`). `fold_terms` emits `Bound::Lit(#u64)` with today's tokens for a
  literal.
- **IR.** `close_over` roots the value bounds and renames only this fold's placeholder.
  `push_reduce` admits an enclosing placeholder (variance `CONST` until its own close). Decision 8
  is refused in sema (M4), in lowering, and by `push_reduce`'s `BoundSort`, in that order.
  `takes_a_value_bound` is `Add | Mul | Min | Max` (`fold.rs:121-126`), and sema refuses exactly
  `Reduction::Any | All`.
- **The macro tier on every accepted shape.** The spec's evidence covers two shapes, through
  builders, with real uniform defaults. I extended it in
  `/home/user/scratch/a6-stages-probe/tests/shapes.rs`. It builds each shape with
  `Uniform::new(f32::NAN)`, as `Expansion` declares a uniform, runs
  `Saturate::macro_tier()`, relinks as `entry_arena` does, rebinds through `set_declared`, and
  compares the result to Rust's iterator. It covers:
  - a value `lo`, two values, and `n..n`;
  - MIN with a value `hi`, MAX with a value `lo`, PRODUCT;
  - a constant body;
  - a count read both as a bound and as `n as f32`;
  - a count bounding a triangle;
  - a bound that skips a nesting level;
  - an inner MIN bounded by an index;
  - a value `lo` beside an index `hi`.

  All 13 tests pass and report `changed=true` under `PIXELFLOW_ISA=avx2` and `=avx512`. So no shape
  sema accepts makes the macro tier panic at expansion, and none loses its meaning. Step 0 of the
  spec could re-run this probe instead of the two-shape one.
- **Host side.** For an entry with no count, `emit_entry` and `argument` produce today's tokens, so
  §5.3's byte identity holds. Every emitted path is absolute, so the `items_block.rs` no-prelude case
  holds for both refusals.

## Findings

### 1. G2's expected output is wrong at the base, so the gate cannot pass (blocking, checklist)

Spec:1953 says G2 expects nothing, and that at the base it finds only `pixelflow-ir/src/fold.rs:441`.
Run at `29980bf`, it also finds:

```
./pixelflow-pipeline/src/bin/guide_scope_saturation_delta.rs:11://! chess-engine ~98% (`docs/archive/plans/2026-08-17-cost-model-domain.md`, A6).
```

That is another plan's "A6", in a file A6 does not touch. After §7.8, G2 still prints that line.
Fix: add `| grep -v guide_scope_saturation_delta.rs` to the pipe, or expect that one line.

Two related inaccuracies do not gate anything:
- The second grep of G9 (spec:2008) also finds `lib.rs:281` and `parser.rs:103`, which are docs, and
  §7.6 keeps `lib.rs`'s.
- §6.7's "nine test files, 63 in all" (spec:1755) is eight files with 79 tests. The 137 total is
  right.

### 2. M2's help sends the closure form and helpers to a spelling sema itself refuses (non-blocking)

M2's help for a `Parameter` is unconditional (spec:522): "a count that changes per call is an entry
parameter `{name}: usize`".
- **The closure form.** `|n: f32| (0..n)` gets this help, and the test pins it (spec:1569). An author
  who follows it writes `|n: usize|`, which M7 refuses (spec:426, 531-532).
- **Helpers.** The same help on `fn h(r: f32) -> f32 { (0..r)… }` leads to `r: usize`, which M6
  refuses.
- **The parser.** The new `generic_arguments_refusal` help (spec:547) fires on a turbofish at a
  helper call (`h::<4>(x)`). It says "or take it as an entry parameter `n: usize`", but A6 gives a
  helper no way to take that count (D-A6-2).

Fix:
- Choose M2's help by `(self.role, self.items.def.spelling)`. Use today's text for an items-form
  entry. Name the items form for the closure (or point at M7's help). Give a helper no count help.
- Move the pinned help substring to an items-form row, for example
  `pub fn f(r: f32) -> f32 { (0..r)… }`. Keep the closure row for the header and §3.4 only.
- Drop the parser's appended clause, or limit it to an entry.

### 3. Lowering's backstop judges arithmetic before it resolves names, against R-A6's order (non-blocking)

`lower_fold` calls the shared `range_bounds` first (spec:670), and `RangeScope::operand` refuses
arithmetic on any `Count::Name` (spec:247-250). Names are resolved only after that, in `bound()`
(spec:671, 700-711). So lowering on its own (`lower_unanalyzed`) refuses `0..X + 1` and
`|r: f32| (0..r - 1)` with M3. M3's note calls `X` and `r` "a bound that is a value — an enclosing
fold's index, or an entry's count parameter".

The refusal is right, but its words are wrong. It is also the reverse of the order R-A6 a6fit 2
fixes. `a_bound_that_is_no_count_is_refused_here_too` covers neither case.

Fix, either of:
- In `lower_fold`, check every name of both bounds against `self.frame.locals` before
  `range_bounds`. This mirrors `type_of_bound`: only a `Binding::Count` or a `usize` const passes,
  and anything else is refused as "`{name}` as a range's bound". Add those two inputs to the lowering
  test.
- Accept the backstop's wording and say so in J12.

### 4. `lower_entry` defends with `expect` a shape its own `match` already proves (non-blocking, "Denote before you build")

Spec:604-616 builds an iterator from `parameter.scalars()` and then matches on `parameter.kind`. For
`Value` and `Count` it pulls one item with `.expect("… is one uniform")`. The spec adds the second
`expect`; `lower.rs:397` has the first today. The `match` arm already knows these kinds have exactly
one scalar, so the `expect` guards a state the arm makes unrepresentable.

Fix: build from the kind.
- `ParameterKind::Value` → `Binding::Value(site.uniform(Scalar::Value(name)))`.
- `ParameterKind::Count` → `Binding::Count(site.uniform(Scalar::Count(name)))`.
- `ParameterKind::Record(record, _)` → `Binding::Record(record, parameter.scalars().map(|s| site.uniform(s)).collect())`.

This drops both `expect`s and keeps the same declaration order.

### 5. The new grammar production says every stage accepts `i + 1` (non-blocking, doc)

The grammar is "what every stage accepts" (`parser.rs:9-10`). Spec:575 adds
`bound ::= iexpr | IDENT`, but `iexpr` already contains `IDENT` (`parser.rs:36-38`, meaning a
`usize` const) and its `+ - * /`. So `0..i + 1` derives, and M3 refuses it.

Fix: in `iexpr`, write the leaf as `CONST` (a `usize` const). Make `bound ::= iexpr | INDEX | COUNT`.
Add `CONST`, `INDEX` and `COUNT` to the legend at `parser.rs:75-80`.
