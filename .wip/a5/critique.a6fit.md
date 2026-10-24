# A5 spec critique: A6/A7 fit and plan fidelity

Read-only review of `.wip/a5/spec.md` and `.wip/a6/design.md` against the tree at `7e9531e`
(`git diff --quiet 7e9531e HEAD -- . ':!.wip'` holds), the plan
`docs/language/2026-10-20-a-bound-is-a-value.md`, `CLAUDE.md` and `pixelflow-compiler/intent.md`.
The lens is A6/A7 fit and plan fidelity. Deletion closure is `critique.closure.md`'s, and I do not
repeat its findings.

## Verdict

A5 deletes nothing A6 needs. The A7 decision is right by the code, and the spec corrects the
plan's A7 bullet. No finding blocks A5. Two small edits to the plan record and one diagnostic
wording are A5's to fix. Four findings belong to A6's design, which A6's spec would otherwise
inherit.

## What I checked and found sound

- **Everything design §10 asks A5 to keep survives the spec.** `Ty::Usize`, `SymbolKind::Index`,
  `register_index`, `Binding::Index`, `a_count_is_not_a_value` and `type_of_cast`'s
  `names_a_count` are all kept (spec §2.9). `type_of_fold` still calls `range_bounds` before
  `push_scope` (`sema.rs:916-918`); A5 changes only `range_scope()`'s field list, so the fold's own
  index stays out of scope for its bounds, which is how L3 holds by scope. `usize` consts and
  `ConstValue::Usize` stay. `RangeScope` keeps one field. `UsizeScope`, `CheckedOp`, `range_bounds`
  and `literal_range` stay, and `literal_range` still asks `KnownRange::admits`
  (`fold.rs:1235`, `fn admits(range: &Range<u64>) -> bool`, so `Some(range).filter(KnownRange::admits)`
  type-checks). `node`'s value-bound path stays: the `None` arm's
  `__header.attach(children)` (`emit.rs:596-606`) becomes the whole `Reduce` arm, and
  `FoldHeader::to_bits`, which carries A4's marker, is untouched. `reached` still walks
  `arena.children`, which yields value bounds. `argument(scalar)` (`emit.rs:176-182`) and `relink`
  (`emit.rs:165-167`) are kept as they are. The `Args` record's emission loses only its generics:
  fields keep their declared types, so A6's `n: usize` field gets through as written, and
  `write_into`'s `set_declared([...])` is unchanged.
- **`range_of`**: `Holes::range_of` is the only compiler `range_of`. A6 does not use it, and once
  A5 lands nothing reads `known_range` in emission.
- **The IR supports A6's lowering.** `ExprArena::open_fold` clones, and an id bound before the
  fold opened means the same node in the copy (`arena.rs:1797-1815`). `close_over` renames only
  this fold's placeholder (`arena.rs:1768-1774`). So a triangular bound or a count's leaf, lowered
  before `open_fold`, is valid in `FoldTerms` under both `Expansion` and `Staged`.
- **A7 is empty.** The exemption's condition is `!entry.structural.is_empty()` (`emit.rs:154`). That
  reads only the field A5 deletes, and nothing else in the macro tier exempts a template:
  `DwrtFree` tests for `Dwrt` (`lib.rs:384-393`), and kernel-taking entries are `Staged` and never
  reach `entry_arena`. Every `fn …<const` in a `kernel!` block is in `pixelflow-compiler/`, and
  the spec's §3 table lists them all. The production block
  (`pixelflow-graphics/src/fonts/loop_blinn/program.rs`) has no generic and no fold. The emit.rs
  doc's second reason (`:136`, "no optimization runs at expansion unless the instance is declared")
  holds after A5 without the exemption, because a non-generic entry is its own declared instance.
  The plan's A7 bullet is rewritten to say so (spec §6.1).
- **A6's design is consistent with A4's guard and L9.** `Saturate::macro_tier()`
  (`saturate_pass.rs:61-68`) inserts, saturates and extracts, and runs no legalize. So the guard is
  added at bake, as the design says. A uniform bound keys by dense slot. The scratch probe
  `a6_macro_tier.rs` exercises `macro_tier` on a uniform `hi` and on a triangular bound.
- **Decision 2 and L5 on the main path.** For an items-form entry, design §3's `n: usize`
  is one uniform. The host function asserts `n <= 2^24` before `n as f32` can round 2^24 + 1 into
  the domain, and `write_into` checks it too. That matches plan §1.4's "refused where it is
  written". The raw `set_declared` path is untyped and the guard's NaN covers it.
- **The deferred messages.** The "ranges are constant (§1.5 of PLAN)" note that A5 keeps at
  `sema.rs:1861-1862` is A6's to correct: plan:509 assigns code docs to "the CL that changes the
  code under them", which for that refusal is A6. The design records this (§1).

## Findings, ranked

### 1. A6 design: the `n: usize` rule admits the closure form it says it refuses (A6, non-blocking for A5)

Design §1 (`design.md:93-97`) says A6 "accepts `Ty::Usize` for `Role::Entry` only; a helper's
`usize` parameter and the closure form's stay refused (§11)". The rule as written does not
produce that outcome. The closure sugar's one entry is `vis: Public`
(`parser.rs:265-266`), and `FnItem::role` maps every non-inherited visibility to `Role::Entry`
(`ast.rs:188-193`). `Items::signature` passes `f.role()` to `param_type` (`sema.rs:557-561`). So
`kernel!(|n: usize| …)` would be accepted.

`emit_closure` then hardcodes `f32` parameters (`emit.rs:202-207`). The declared `usize` becomes an
`f32` closure parameter, with no 2^24 check and no `Args` record. That is the untyped L5 path that
D-A6-3 (`design.md:369-372`) and §7.4's refused list (`design.md:289-290`) both mean to close.

Fix: state the rule as "accepted for an entry of `Spelling::Items`" (`def.spelling`, already read
at `sema.rs:336`). §7.4's closure case is then the pin.

### 2. A6 design: arithmetic is refused before its names are typed, so a coordinate gets the wrong law (A6, non-blocking for A5)

`eval_usize` refuses any `Binary` with a `Name` on either side, with `BoundArithmetic`'s text
(`design.md:62-67`). `RangeScope::usize_named` returns `Name` for every non-const name
(`design.md:69-70`). Names are typed afterwards, and only when an end is a bare `Name`
(`design.md:78-87`).

So `(0..X + 1)` is refused as "a bound that is a value is a single name until bounds are typed
integers". That tells the author T6 will lift it. In fact L3 refuses it permanently (plan:743-744,
"a count varying with X, Y or a lattice binder is refused, permanently"; plan:413, "sema refuses
L3 by type"). The same happens to an `f32` parameter in arithmetic (`(0..r * 2)`) and to a
misspelled name (`(0..nn + 1)`, which should get "cannot find"). None of the tests in §7.4 has a
coordinate inside arithmetic.

Fix: type every name in a bound first, refusing `X`/`Y` by L3 and any other non-`usize` by type.
Refuse `BoundArithmetic` only when every name is a `usize` index or count parameter. Add
`(0..X + 1)` to §7.4.

### 3. A5's new `const` generic note says "a count is a value"; the type system says the opposite (A5, non-blocking)

The spec's `generic_refusal` note (`spec.md:108`) reads "a count is a value, not a program of its
own". A5 also rewrites `of_a_value`'s note to "a `usize` is a count, not a value" (`spec.md:211`,
and its test asserts "a `usize` is a count"). `a_count_is_not_a_value` (`sema.rs:1642-1655`)
already says "`{name}` is a `usize`, where a value is expected".

An author who follows the first message to `n: usize` meets the second, which contradicts it. The
plan's sense of "value" (an operand, as in "a bound is a value") is not the front end's sense (an
`f32` or a `bool`, intent.md:9-11). A6 does not resolve this: after A6 the closure form, helpers
and `let` still refuse `usize` with "a count, not a value".

Fix: word the note in the front end's vocabulary, for example "a count does not make a program of
its own: an entry is one program, whatever its counts (§1.2 of …)". The test pins only the header,
the plan path and the help line (`spec.md:731-733`), so this rewording is free.

### 4. The plan's §5 row for the-language-is-kernel §1.3 is left naming A1 and a text A5 does not write (plan fidelity, non-blocking)

Plan:496 assigns the row (`:179`, now `:192`) to "A1, after #1431", with Becomes "a fold binder or
a per-call count". A5 edits that line to "a fold binder or a count (a literal or a `usize` const)"
(`spec.md:819-821`), and it edits only A1's Done line to drop the row from A1's exceptions
(`spec.md:802-806`). Plan:479 says "each row is edited by the CL named". After A5 the §5 row
still says A1 owns the line and names text that is not in the tree. "Per-call count" stays false
for `kernel!` until A6, which design §8 (`design.md:302-303`) says will add it.

Fix: in §6.1, also edit plan:496's CL cell to "A5 (literal or `usize` const), A6 (per-call
count)", or split the row. Optionally note that the row's IR column, `Var(REDUCE_BINDER_BASE +
slot)`, is not what a `usize` const lowers to. A const lowers to a `Const`.

### 5. The plan record A5 writes slightly misstates what was exempted and what the tests became (plan fidelity, non-blocking)

The new A7 bullet (`spec.md:797-801`) says the exemption "exempted only the test templates A5
rewrote over `usize` consts". The spec's own §3 table (`spec.md:493-503`) shows three exempted
entries that A5 deletes rather than rewrites: `lower_case::span`, `grid` and emit's `mean`. The
draft Done line (`spec.md:790-796`) says "The test templates are written over `usize` consts". It
does not say that the tests whose subject was the template are deleted (J11), although the plan's
A5 bullet says the tests are "rewritten".

Fix: "…exempted only test templates, which A5 rewrote over `usize` consts or deleted with the
tests whose subject they were"; in the Done line, add "the tests whose subject was the template
are deleted".

### 6. A6 design: `ConstEvaluator` would return a `Count` it can never fill with `Name` (A6, non-blocking for A5)

Design §1 makes `UsizeScope::eval_usize` return `Count` for both scopes. Its callers in
`ConstEvaluator` then handle "an unreachable arm refused by name" (`design.md:71-73`). That
brings back the guard A5 deletes in `known_usize` (`sema.rs:2044-2053`) and `eval_f32`'s
`Count::Structural` arm (`sema.rs:1959-1962`), only for a different variant.
CLAUDE.md's "Denote before you build" names this shape: "the fix arrives as a runtime guard
defending what a type should have made unrepresentable". A5's `u64` is the right type for a
`const`'s count.

Fix (in A6): give `UsizeScope` an associated output type. `ConstEvaluator` keeps A5's `u64`, and
`RangeScope` yields `Count`. Nothing in A5 changes. J7's reasoning still stands.

### 7. A6 design: `Parameter { count: bool }` beside `record: Option<…>` (A6, non-blocking for A5)

`design.md:140-142` adds a boolean field beside `record` in `Parameter` (`sema.rs:245-251`). That
makes a parameter that is both a record and a count representable, and every reader then has to
check both fields. CLAUDE.md says to extend the type when its meaning is extended, and to avoid
boolean mode switches.

Fix (in A6): `Parameter { name, kind }` with `kind` one of `Value`, `Count` or
`Record(RecordId, &[RecordField])`, and `Scalar` carrying whether it is a count.

### 8. `Staged::open`'s doc stays stale through the spec's rewrite of `fold_terms` (A5, non-blocking; outside the lens)

`emit.rs:674-675` documents `open` as "the closure that makes its `FoldHeader` once its binder is
chosen". Since A3 the stored `TokenStream` has been a `FoldTerms` expression (plan:423). Spec §2.5
rewrites `fold_terms` and its doc (`spec.md:383-386`) but has no row for the field. A6 will emit
`Bound::Value(#ident)` into this same field.

Fix: "the variable holding each, and the `FoldTerms` it closes into".
