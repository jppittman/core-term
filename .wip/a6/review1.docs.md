# A6 review 1: docs truth

Working tree against `29980bf` (`git diff 29980bf -- . ':!.wip'`), read against `.wip/a6/spec.md`,
`.wip/a6/rulings.md` (which win), the plan, CLAUDE.md and `pixelflow-compiler/intent.md`. The lens is every
doc sentence, comment, generated `#[doc]` text and message the diff adds or edits, and every sentence beside
an edit, read against the code rather than against a ruling's wording. Read-only on the repo: no build, no
gate script. Evidence is file:line in the working tree, a grep, or the stages reviewer's scratch output
(`/home/user/scratch/a6-review-stages/report1.txt`), cited where used. The plan's `GATES: pending.` placeholder
is the coordinator's and is not a finding.

## What holds

Most of what the diff writes is true of the code, and several places were checked against the code in more
than one way.

- **The host refusal docs.** The `# Panics` text, `write_into`'s `# Errors` and `# Panics`, `COUNT_DOMAIN`
  and the panic message all match `emit_entry` and `emit_args`. `KnownRange::admits(&(0..n as u64))` is
  `start <= end && end <= 2^24` (`fold.rs:1236-1238`). The assert precedes the body and precedes
  `set_declared`. `self.n as f32` parses as `(self.n) as f32`. "`2^24 + 1` would be written as `2^24`" is
  ties-to-even. "A larger count would name none" holds because every `f32` above `2^24` is above the
  domain.
- **The other docs on the host write.** The ontology rows (`kernel!` Follows, Args Is/Follows/Lives,
  Binding time, Fold, Range with "(E9 aside)") match. So do BACKLOG C8 and the plan-table row,
  the-language-is-kernel `:262-266`, the plan's §1.4 and §3.4 paragraphs, `pixelflow-compiler/intent.md`
  (every test it names exists), and the `pixelflow-core` README's new parenthetical.
- **BACKLOG E9's widening.** The macro tier's set is `RuleSet::production()`, which is `all_rules()`. It
  holds `Annihilator::new(OpKind::Mul)` (`math/algebra.rs:1108`), and that rule fires when either operand's
  class `contains_const(0)`. `FactorFold` is runtime-only (`rules.rs:196-211`, `fold_rules.rs:532-534`).
- **Test docs.** Each test doc that cites another test cites one that exists
  (`a_fold_over_an_empty_range_is_the_identity_as_in_rustc`,
  `the_f32_nearest_two_to_the_24_plus_one_is_two_to_the_24`). "Every value here is exact" in
  `optimized_agrees_with_raw.rs` is true: the frame is 64 by 8 at integer coordinates, with `n <= 5`.
- **`Site::open_fold`'s doc.** "Names the same node in the copy" is `ExprArena::open_fold`/`close_fold`'s
  stated contract (`arena.rs:1778-1838`).
- **`literal_bound`'s doc.** "Before `FoldHeader::new` would panic" holds: `FoldHeader::new` panics on a
  literal past `EXACT_BOUND` beside a value (`fold.rs:825-830`, `:856`).

## Findings, most severe first

1. **the-language-is-kernel's "A6" names that document's own A6** (`:59`, `:113`). That document has its
   own Phase A list, and its A6 is "`Select` renamed `If`" (`:725`, and `:365`, `:1271`). At `:109-113` the
   diff's clause follows a sentence that points at the other plan's steps as "its A4" and "its B2". The new
   clause then says "a `kernel!` bound has been a value since A6", which read in this file is the rename.
   `:59` has the same defect beside the older "A5 deleted", and this document's A5 is "64-bit fold ends",
   at `:736`. Fix: "since a-bound-is-a-value's A6" or "since its A6" at `:113`, and name the plan at `:59`.
2. **`fold_bounds`' doc says `sema` refuses what lowering refuses, and it does not** (`lower.rs:702-706`:
   "All of it without relying on `sema`, which refuses the same first").
   - The domain checks in `fold_bounds` (`literal_range`, `literal_bound`) are lowering's alone.
     `grep -n 'EXACT_BOUND\|admits' pixelflow-compiler/src/sema.rs` finds no bound check, and `parser.rs:186`
     lists "a range bound past 2²⁴, alone or in a pair" under "Refused by lowering".
   - The stages reviewer's run shows `(16777217..n)` passing sema and refused by lowering at the
     `kernel! {` call site (`report1.txt`, `low_literal_past_beside_count`).
   - The test doc at `lower.rs:1282`, "Lowering refuses a bound that is no count by itself, as `sema`
     does", is false for that test's own `(16777217..n)` row.
   - Fix: say that `sema` refuses the names and decision 8 first, and that the domain is lowering's.
3. **"The two optimizer edges" are given to each law, but each law has one edge.**
   - Where: `lib.rs:162-163` ("An empty range gives the monoid's identity, up to two optimizer edges") and
     `rustc_is_the_oracle.rs:810-812` ("L2, which holds up to the two optimizer edges") give both edges to
     L2. The plan's Done line (`:726-727`: "makes the fold NaN by the guard, up to the two optimizer edges
     … (BACKLOG E9's annihilation, decision 5's factoring)") gives both to L5.
   - What `Kernel::over_values` documents (`kernel.rs:791-797`): one edge per law. The factoring edge is the
     backwards pair's (L2): "the infinite-`c` edge at a runtime `n = 0`" (plan `:321-323`, decision 5 at
     `:795`). The annihilation edge is the NaN's (L5).
   - Why annihilation never touches an empty range: it replaces `0·y` by `0`. Under `SUM` that is the
     identity, `+0.0`, either way. It has no rule for `Add`, the factor `MIN`/`MAX` take out.
   - Why factoring never removes a NaN: `c·NaN` and `c + NaN` are NaN. Only the annihilator, on a factor
     proven zero, removes it.
   - The ruling (LAWS-1) put the pair on L2 and L5 together. Read against the code, each law takes one.
     Fix: L2 "up to the factoring edge"; L5 "up to the annihilation edge (BACKLOG E9)". The ontology's
     "(E9 aside)" is already right.
4. **`lib.rs:170-171`, `# Folds`: "A fold lowers to one `Reduce`, the node `Kernel::over` builds".** The
   paragraph now introduces `0..n` and `0..i` bounds. `Kernel::over` takes a `Range<u32>`
   (`kernel.rs:754`) and cannot build either. `fold_is_kernel_over.rs`'s `a_per_call_count_is_over_values`
   pins a count-bounded fold as `Kernel::over_values`'s. `lower.rs`'s module doc was corrected to "built as
   `Kernel::over` builds it — or `Kernel::over_values`, for a bound that is a value"; the public `kernel!`
   rustdoc was not. Fix: the same wording.
5. **`BOUND_PLAN`'s doc** (`lib.rs:75-76`, "a refusal about a count or a bound names it"). Many bound and
   count refusals name no plan:
   - the reversed known pair (`sema.rs:1960`);
   - `literal_range` (`lower.rs:241-251`) and `literal_bound` (`lower.rs:256-266`);
   - `count_term` ("`n` as a range's bound", `lower.rs:692-699`);
   - `resolve`'s "`n` is a `usize`" (`lower.rs:876`);
   - the unary-minus and overflow refusals.

   `grep -n BOUND_PLAN pixelflow-compiler/src/*.rs` lists the sites that do name it. Fix: "the plan that
   made a fold's bound a value, which the refusals of A6's spellings cite", or delete the clause.
6. **The grammar derives no parenthesized value bound** (`parser.rs:69-72`), yet `(0..(n))` is accepted
   (`sema.rs` `a_range_bound_may_be_an_enclosing_index_or_a_count_parameter`). This is the stages reviewer's
   finding 1, repeated here because it is a doc sentence the diff writes. Fix:
   `bound ::= iexpr | INDEX | COUNT | '(' bound ')'`.
7. **M1's reason is false for the triangular fold A6 adds** (`sema.rs:1768-1770`: "every sample runs the
   same loop, its trip count fixed before the lattice is traversed"). An inner `0..i` fold's trip count is
   the outer index. That index is computed on each outer trip, inside the traversal. What L3 buys is that
   every sample shares the trip count, not that the count is fixed beforehand. Fix: "its trip count the same
   at every sample".
8. **"Decided per call" is false for an index bound** (`sema.rs:1951-1952`, `range_bounds`' doc: "A pair
   with a value is decided per call (L2)"). `(i..2)` inside an outer fold is decided on each outer trip.
   The test doc at `sema.rs:3079` is about `5..n`, where "per call" is right. Fix: "decided at run time".
9. **`not_a_usize_bound`'s doc** (`sema.rs:996-998`) says "The help is the fix an `f32` has: a count that
   changes per call is an entry's `n: usize`". The code's match (`sema.rs:1001-1013`) disagrees:
   - an `f32` const gets "declare the count as `const R: usize`";
   - a `let` `f32` gets no help;
   - a helper's `f32` parameter gets no help.

   Fix: describe the match: a const's help, an entry parameter's help by spelling, none otherwise.

## Non-blocking

- **The `pixelflow-core` README, `:64-67`.** The reflowed paragraph now names `kernel!`. Its next sentence
  says "a reduction with a bound that is not one is NaN" without the annihilation exception that
  `Kernel::over_values` (`kernel.rs:793-797`) and the ontology ("E9 aside") carry. The sentence is A4's, and
  `impl-docs.md` leaves it to the coordinator. The same is true of `CLAUDE.md:180`, which the diff does not
  touch.
- **`fold_is_kernel_over.rs:1-7`.** The lead sentence, "A fold written in `kernel!` is the fold
  `Kernel::over` builds", is unqualified until the sentence the diff appends. `:7` names `Fold::Range`,
  which does not exist (`fold.rs:535`, `pub struct Fold`). Both are in the paragraph the diff edits, and both
  predate A6.
- **"Count" means two things in one paragraph** (`lib.rs:275-276`). "Beside other arguments the count
  differs, and `write_into` refuses" uses "count" for the arity, and the next sentence uses it for a count
  parameter. `write_into`'s own doc was reworded for this reason (ruling S3: "only the number of arguments
  is checked"). Fix: "the number of arguments differs".
- **"Whose host function and `Args` record".** Three places say this: `count_parameter`'s doc
  (`sema.rs:683-684`), M7's note (`sema.rs:696-697`) and the plan's §3.4. An entry that takes a kernel and
  a count has no `Args` record (`ast.rs:200`; `emit.rs`'s unit test asserts no `ColumnsArgs`). Fix: "its
  `Args` record, if it has one".
- **`param_type`'s unknown-type refusal** (`sema.rs:656-658`) still says "a kernel parameter is an `f32` or
  a `bool`, or one of this block's records". An items-form entry now also takes `usize`. Ruling C7 updated
  the `bool` refusal beside it, and this one was left as it was.
- **`register_parameter`'s doc** (`symbol.rs:181`) reads as a complete list: `f32`, a record, a count.
  Every parameter is registered through it (`sema.rs:760-761`), including a helper's `bool` and a
  kernel-typed one.
- **`BoundOutOfDomain`'s doc** (`fold.rs:439-442`) is about "a real literal that names no count". It lists
  "where the host writes a count" as a place `kernel!` refuses one, but a host-written count is a
  uniform's value, not a literal.
- **M3's help for an index.** "One that changes per call is computed by the host and passed as a parameter
  of its own" does not fit `0..i + 1`, which changes per trip. The help is not false, but it does not
  apply.
- **The-language-is-kernel §1.3's IR column** (`:199`) says "a `usize` const's `Const`". That is true of
  `N as f32`. A `usize` const as a bound is node data (`Bound::Lit`), not a `Const`. The column is
  incomplete, not wrong.

## Not findings

- **The rulings' wording (S1/C3) of the-language-is-kernel's clause.** It is checked above only for the
  step name it points at.
- **E9's claim that `kernel!` takes the fold at expansion.** It holds by the rule set. It is unpinned, and
  `impl-docs.md` says so.
- **The unreachable `RangeScope` `F32` arm.** It is the stages reviewer's finding 3, and it is code, not
  a doc.
