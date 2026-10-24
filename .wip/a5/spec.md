# A5 spec: `kernel!` has no structural parameters

**Base:** `7e9531e` (main after A4, #1448). HEAD may carry `.wip/`-only commits on top
(`git diff --quiet 7e9531e HEAD -- . ':!.wip'` holds at writing). Every file:line below is at
`7e9531e`. Re-grep before trusting a line number; §7's greps are the authority.
**Plan:** `docs/language/2026-10-20-a-bound-is-a-value.md` ("plan"): §1.2 (plan:65-80, "Structural
is not a binding time"), §3.4 (plan:407-424), §4 "Deleted" (plan:437-439), §6 A5/A6/A7
(plan:697-708), §8 decision 2 (plan:760-762, decided: an `n: usize` entry parameter, which is A6's).
**A4's record** (the standard): `/home/user/scratch/a4-wip-archive/.wip/a4/{spec,rulings}.md`.
**Design of A6/A7** (what A5 keeps for them): `/home/user/pixelflow/.wip/a6/design.md`, cited as
"design".
**Evidence run for this spec:** `/home/user/scratch/a5-probe` (a crate outside the repo; built
against the repo's `target.noindex` with the workspace's `RUSTFLAGS`). `tests/probe.rs` holds the
five integration-test kernels §5 rewrites, written over `usize` consts exactly as §5 prescribes, and
run at `7e9531e`, where such entries are already saturated at expansion: 5/5 pass under
`PIXELFLOW_ISA=avx2` and `=avx512` (§4.3).

Every decision an implementer would otherwise make is made here; §8 lists them with the rejected
alternative. Nothing here waits on a ruling.

---

## 0. What A5 is, and what it is not

**A5 is** (plan:697-701, plan:416-424): `kernel!` loses its one syntax for a count bound per
instantiation, `pub fn f<const N: usize>(…)`, and with it everything that existed only to carry
such a count from rustc's monomorphization into a program:

1. the parser stops producing `FnItem::structural` and refuses a generic on every `fn`, a `const`
   generic by a message that says what a count is now (§1);
2. sema loses the `structural` binding time: `SymbolKind::Structural`, the `Count::Structural`
   token expression, `Bounds::Structural`/`StructuralRange`, and the `structural` fields that fed
   them (§2.3);
3. lowering loses the template: `FoldRange`, `Holes`, `OPEN_RANGE`, `HOLE_STRIDE`,
   `ExpansionFold` (whole enum), `Site::count` and the `Param(u8)` hole it built, `Lowered`'s
   `holes` (and `Lowered` itself) (§2.4);
4. emission loses instantiation: `instantiated_range`, `structural_generics`, `EntryContext`
   (whole struct), the `Param` arm's structural read, the `Reduce` arm's hole branch, and the
   **template exemption** at `emit.rs:154-156`, whose condition names the deleted field (§2.5, §3);
5. `pixelflow_ir::arena::FoldTerms` loses `stride`, which every remaining constructor sets to 1
   and whose own doc says it lives "until A5 deletes them" (`arena.rs:180-181`) (§2.7);
6. the test kernels that were templates are rewritten over `usize` consts and are saturated at
   expansion from now on; the tests whose subject was the template are deleted (§5);
7. docs (§6).

**A5 is not:**
- **A6.** No bound that is a value in `kernel!`: sema still evaluates both bounds to `u64` at
  expansion and still refuses any bound naming a fold's index, a parameter, a `let` or a
  coordinate (`a_range_bound_that_is_not_constant_is_refused` keeps every case, including the
  triangular `(0..i)`). `n: usize` is still refused as a parameter type
  (`a_usize_is_not_a_declared_value_type` keeps its first case). No `Args` count check, no
  `BoundArithmetic`/`BoundSort` refusal in sema. The line: **after A5 every `kernel!` fold has two
  `Bound::Lit` ends, exactly as every non-template fold had at `7e9531e`.**
- **A7** as a separate CL: A5 deletes the exemption because its condition cannot be written once
  `FnItem::structural` is gone, and A7 is empty (§3).
- **B1-B5, T6, decision 8**: nothing about discrete-kernel arguments, `Extent`, `FontPrograms`,
  bound typing or `ANY`/`ALL`.
- **`Param(u8)` in the IR**: it stays (plan:438-440): a rewrite rule's metavariable, deleted with
  `Vocabulary::Templates` on the-language-is-kernel D2's schedule. A5 only stops the compiler from
  building one.
- **Not a byte change on any production kernel** (§4).

---

## 1. The surface after A5

### 1.1 What is refused, where, and by what message

**Stage: the parser** (`convert_fn`, `parser.rs:419-491`). Reasons: the parser already refuses
every generic but an entry's `const N: usize` (`refuse_generics`, `parser.rs:700-719`, for a
`const`; `refuse_a_helpers_generics`, `:537-555`; `structural_parameters`, `:493-535`), and its
contract is "Nothing is passed through for a later stage to refuse: the AST holds only what the
language means" (`parser.rs:155-156`). Its precedent for a construct the language does not have is
`a_collection_spelled_any_way_is_refused_naming_its_sections` (`parser.rs:2540-2588`): refused at
the token, naming the plan section. With the refusal at parse, `FnItem` has no field for generics,
so no later stage can accept one (intent.md:25-26, "No stage accepts what a later one refuses").

**One function for every item**: `refuse_generics` is called by `convert_const` (as today) and by
`convert_fn` for entries and helpers alike, at the point `structural_parameters` /
`refuse_a_helpers_generics` are called today (`parser.rs:453-459`, after the variadic check,
before the parameters). Its first-parameter refusal is chosen by the parameter's kind:

```rust
/// A `fn` and a `const` take no generics: a value is an `f32` or a `bool`, a
/// record is one of the block's, a kernel is `impl Fn(f32, f32) -> f32`, and a
/// count is a literal or a `usize` const, evaluated at expansion.
fn refuse_generics(generics: &syn::Generics) -> syn::Result<()> {
    if let Some(param) = generics.params.first() {
        return Err(syn::Error::new_spanned(param, generic_refusal(param)));
    }
    if let Some(where_clause) = &generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "a `where` clause in a `kernel!` block: there are no generics to bound",
        ));
    }
    Ok(())
}

/// The refusal of a generic, by what it would have meant: a `const` one was a
/// count, which is a value now, and any other names a type the language does not
/// have.
fn generic_refusal(param: &syn::GenericParam) -> String {
    match param {
        syn::GenericParam::Const(_) => format!(
            "a `const` generic in a `kernel!` block\n\
             \n\
             note: a count is a value, not a program of its own: an entry is one program, \
             whatever its counts (§1.2 of {BOUND_PLAN})\n\
             help: write the count as a literal, or as a `usize` const of the block: \
             `const N: usize = 4;`"
        ),
        _ => format!(
            "generics in a `kernel!` block\n\
             \n\
             note: nothing in a kernel is generic: a value is an `f32` or a `bool`, a record \
             is one of the block's, and a kernel is `impl Fn(f32, f32) -> f32` (§1.3 of {PLAN})"
        ),
    }
}
```

`BOUND_PLAN` is a new `pub(crate) const` in `lib.rs`, beside `PLAN` (`lib.rs:71-73`):

```rust
/// The plan that made a fold's bound a value: a refusal that turns on what a
/// count is names it.
pub(crate) const BOUND_PLAN: &str = "docs/language/2026-10-20-a-bound-is-a-value.md";
```

The message avoids the word "structural" on purpose: §7's gate G2 holds the compiler crate to no
occurrence of it. The span is the generic parameter (`const N: usize`), as the parser's other
refusals point at the token.

### 1.2 What each spelling does after A5

| Written | At `7e9531e` | After A5 |
|---|---|---|
| `pub fn f<const N: usize>(…) -> f32 { … }` | an entry with a structural parameter, a template | parse error at `const N: usize`: "a `const` generic in a `kernel!` block … (§1.2 of docs/language/2026-10-20-a-bound-is-a-value.md) … `const N: usize = 4;`" |
| `pub fn f<const N: u32>()`, `<const N: usize = 4>`, `<const N: usize, const M: usize>` | three distinct parser refusals / accepted | the same `const` generic refusal, at the first parameter |
| `fn h<const N: usize>(…)` (a helper) | "generics on a helper … pass `n as f32`" | the same `const` generic refusal |
| `pub fn f<T>()`, `pub fn f<'a>()`, `fn h<T>()` | "a type or lifetime parameter on an entry … B3" / "generics on a helper" | "generics in a `kernel!` block … nothing in a kernel is generic" |
| `pub fn f() -> f32 where … { … }` | "a `where` clause in a `kernel!` block: a structural parameter is a `usize` …" | "a `where` clause in a `kernel!` block: there are no generics to bound" |
| `rings::<4>(b, 1.0, 0.0)` in host code | calls the instantiation | rustc's E0107 at the call ("function takes 0 generic arguments but 1 generic argument was supplied"): the host function has no generics. Nothing of the macro's; not tested (rustc's own error, and the crate has no compile-fail harness). |
| `RingsArgs::<4> { … }` | the instantiation's record | rustc's E0107 likewise |

**Still works, unchanged:** literal bounds (`(0..4)`, `(2..6)`, `(3..3)`); `usize` consts and
checked `+ - * /` over them in a bound (`(LO..LO + N)`) and as a value (`N as f32`); a fold's index
read as `i as f32` in its body and in any fold nested in it (nested folds capture nothing); every
refusal of a bound that is not constant (A6's to lift); records and `Args`; kernel-typed parameters
(`Staged`); `kernel_raw!`. An enclosing index as a **bound** (`(0..i)`) is refused before and after
A5 alike; it is A6's.

---

## 2. The deletion inventory

Legend: **D** delete; **S** simplify to (the text says to what); **E** edit text only;
**K** keep, with the reason. Every row names `file:line` at `7e9531e`.

### 2.1 `pixelflow-compiler/src/ast.rs`

| Line | Item | Act |
|---|---|---|
| 25 | tree doc `├── structural: [name, ...] // an entry's const N: usize parameters` | D (the line) |
| 163-167 | `FnItem::structural` and its doc | D |
| 671-675 | `CastExpr` doc: "a fold's index, a `usize` const or an entry's structural parameter" | E: "a fold's index or a `usize` const" |
| 683-686 | `CastExpr::named` doc: "a fold's index, a `usize` const or a structural parameter" | E: "a fold's index or a `usize` const" |
| 821 | test `an_entrys_args_record_is_its_name_in_upper_camel_case`: `structural: Vec::new(),` | D (the field init) |

### 2.2 `pixelflow-compiler/src/parser.rs`

| Line | Item | Act |
|---|---|---|
| 19 | grammar: `VIS 'fn' IDENT structural? '(' …` | S: `VIS 'fn' IDENT '(' params ')' '->' type block` |
| 21-22 | `'fn' IDENT …` "-- a helper: private, no generics" | S: "-- a helper: private" |
| 24-25 | `structural ::= …` production | D |
| 38-45 | `iexpr` comment "… at expansion, or when the host fn is instantiated if it names a structural parameter" | S: "-- a `usize`: a range's bounds or a `usize` const's initializer; each operation checked as rustc checks it, at expansion" |
| 57-59 | cast comment "(a fold's index, a `usize` const or a structural parameter)" | E: "(a fold's index or a `usize` const)" |
| 96-97 | "Nothing shadows `X`, `Y`, a `const`, a structural parameter or a `fn`" | E: drop "a structural parameter" |
| 104-110 | "Binding times (plan §1.4). An entry's `const N: usize` generics are **structural** …" | S: "Binding times (plan §1.4). Every parameter is a **uniform**: … [rest of the paragraph as is] … A count is a literal or a `usize` const; no `fn` takes generics (§1.2 of docs/language/2026-10-20-a-bound-is-a-value.md)." |
| 127-129 | refusal list: "generics on a helper and on a `const`, and any generic of an entry but a `const N: usize` without a default;" | S: "a generic or a `where` clause on a `fn` or a `const` (a `const` generic by a message saying a count is a literal or a `usize` const, §1.2 of the bound plan);" |
| 183-184 | "A range over a structural parameter is refused the same ways when its host function is instantiated, by rustc, in a `const` block." | D |
| 192 | `use crate::PLAN;` | S: `use crate::{BOUND_PLAN, PLAN};` |
| 270 | closure sugar's `structural: Vec::new(),` | D |
| 453-459 | `let structural = match item.vis { … }` | S: `refuse_generics(&sig.generics)?;` |
| 486 | `structural,` in the `FnItem` literal | D |
| 493-535 | `fn structural_parameters` | D |
| 537-555 | `fn refuse_a_helpers_generics` | D |
| 700-719 | `fn refuse_generics` (const items only; its note names structural parameters) | S: §1.1's `refuse_generics` + `generic_refusal` |
| 1216-1219 | `fn is_usize` (only caller: `structural_parameters`, `:519`) | D (dead once its caller goes) |
| 2381-2397 | test `an_entrys_const_generics_are_its_structural_parameters` | D (§5.1) |
| 2545-2546 | case 2 of `a_collection_spelled_any_way_is_refused_naming_its_sections` | S (§5.1) |
| 2757-2790 | test `a_fn_signature_is_plain` | S (§5.1) |

### 2.3 `pixelflow-compiler/src/symbol.rs` and `sema.rs`

| File:line | Item | Act |
|---|---|---|
| symbol.rs:12 | table row `\| Structural \| Instantiation \| …` | D |
| symbol.rs:42-45 | "An entry's structural parameters (`const N: usize`) are counts, fixed per instantiation of its host function." | D (the sentence) |
| symbol.rs:79-81 | `SymbolKind::Structural` and doc | D |
| symbol.rs:196-199 | `SymbolTable::register_structural` | D |
| sema.rs:19-21 | "…a `usize` const or an entry's structural parameter. It is never…" | E: "a fold's index or a `usize` const. It is never…" |
| sema.rs:31-33 | responsibility 6: "an entry's structural parameters are counts, and every parameter is a uniform —" | S: "every parameter is a uniform —" |
| sema.rs:53 | resolution rule 5 "an entry's structural parameter → a count, fixed per instantiation" | D; renumber 6→5, 7→6 |
| sema.rs:61-62 | "Nothing shadows X, Y, a `const`, a structural parameter or a `fn`" | E: drop "a structural parameter" |
| sema.rs:80-81 | `use proc_macro2::{Span, TokenStream}; use quote::quote;` | S: `use proc_macro2::Span;` (no non-test user of `TokenStream` or `quote!` remains once `Count`, `StructuralRange` and the token-building in `eval_usize` go; the test module imports its own, `sema.rs:2190-2191`) |
| sema.rs:93-95 | `Ty` doc: "A fold's index is one, and so are a `usize` const and an entry's structural parameter." | E: "A fold's index is one, and so is a `usize` const." |
| sema.rs:118 | `Ty::Usize` doc "a fold's index, a `usize` const or a structural parameter" | E: "a fold's index or a `usize` const" |
| sema.rs:157-161 | `of_a_value`'s note "…a `usize` const or an entry's structural parameter (`pub fn f<const N: usize>`), which a body converts by `i as f32`" | E: "a `usize` is a count, not a value: a fold's index or a `usize` const, which a body converts by `i as f32`" (the test asserts "a `usize` is a count", kept) |
| sema.rs:658-659 | `FnAnalyzer::structural` | D |
| sema.rs:675-676 | `new`'s doc "the `fn`'s own structural and scalar parameters" | E: "the `fn`'s own parameters" |
| sema.rs:685, 694-696 | `structural: &f.structural,` and the `register_structural` loop | D |
| sema.rs:736-750 | `fn register_structural` | D |
| sema.rs:752-755 | doc "…a `const`, a structural parameter or a `fn` — rustc refuses a binding that shadows a const generic, too." | E: "…a projection, a `const` or a `fn`. A `let` may shadow…" |
| sema.rs:787 | `Some(SymbolKind::Structural) => …` arm | D |
| sema.rs:902-909 | `range_scope`: doc "…and this `fn`'s structural parameters", field init | S: doc "The bounds a fold's range may name: the block's `usize` consts."; `RangeScope { consts: self.consts }` |
| sema.rs:962 | `type_of_cast` note "a fold's index, a `usize` const or a structural parameter" | E: "a fold's index or a `usize` const" |
| sema.rs:1638-1641, 1648 | `a_count_is_not_a_value` doc and note "(a fold's index, a `usize` const or a structural parameter)" | E: "(a fold's index or a `usize` const)" |
| sema.rs:1657-1683 | `enum Count { Known, Structural }`, `Count::tokens` | D: a count is a `u64` |
| sema.rs:1685-1687 | `type CheckedOp` | K |
| sema.rs:1689-1699 | `UsizeScope` doc: "…one that names a structural parameter is checked by rustc itself…", "…or a structural parameter." | E: drop both clauses (§2.3.1) |
| sema.rs:1700-1763 | `UsizeScope::{usize_named, eval_usize}` returning `Count` | S (§2.3.1) |
| sema.rs:1765-1773 | `enum Bounds { Known, Structural }` | D |
| sema.rs:1775-1792 | `struct StructuralRange`, `StructuralRange::text` | D |
| sema.rs:1794-1830 | `range_bounds` returning `Bounds` | S (§2.3.1) |
| sema.rs:1832-1838 | `RangeScope { consts, structural }` | S: `RangeScope { consts }`; doc "What a fold's bound may name: a `usize` const, already evaluated." **K the struct** (A6's `usize_named` hook, design §10) |
| sema.rs:1840-1866 | `RangeScope::usize_named`: the `None if self.structural.contains(name)` arm; note "integer literals, `usize` consts, the entry's structural parameters (`const N: usize`), `+ - * /`…" | S: delete the arm; note "integer literals, `usize` consts, `+ - * /` and parentheses" |
| sema.rs:1868-1880 | `not_a_count`'s note "…the entry's structural parameters, `+ - * /`…" | E: drop the clause |
| sema.rs:1922-1935 | `ConstEvaluator::usize_named` returning `Count::Known` | S: returns `u64` |
| sema.rs:1952-1964 | `eval_f32`'s `Cast` arm: comment "A `const` cannot name a structural parameter …", `match … Count::Structural(_) => Err` | S: comment "Rust's own `as`, which rounds to the nearest `f32` as rustc's does: exact up to 2²⁴."; body `Ok(self.usize_named(name)? as f32)` |
| sema.rs:2030 | `ConstValue::Usize(self.known_usize(&item.init)?)` | S: `ConstValue::Usize(self.eval_usize(&item.init)?)` |
| sema.rs:2044-2053 | `fn known_usize` | D |
| sema.rs:2160 | `collect_calls`: "A bound is constant, so it calls nothing" | K (true after A5, and after A6: a bound is a leaf) |
| sema.rs:2787-2820 | test `a_range_from_usize_consts_is_evaluated_at_expansion`: `structural: &[]`, `Bounds::Known(3, 14)` | S (§5.1) |
| sema.rs:3165-3232 | section and tests `a_structural_parameter_is_a_count`, `a_structural_parameter_is_not_a_value_and_is_its_entrys` | D (§5.1) |

#### 2.3.1 The count after A5 (sema.rs:1685-1830)

```rust
trait UsizeScope {
    /// The value of `name` in a `usize` expression, or why it has none.
    fn usize_named(&mut self, name: &Ident) -> syn::Result<u64>;
    /// The refusal of something a `usize` expression is not built from.
    fn not_a_count(&self, span: Span) -> syn::Error;
    /// An integer literal, a name, `+ - * /` and parentheses.
    fn eval_usize(&mut self, expr: &Expr) -> syn::Result<u64> {
        let binary = match expr { /* Literal → literal.usize_value(), Ident → usize_named,
                                     Paren → recurse, Unary Neg → refusal (unchanged),
                                     Binary → binary, other → not_a_count */ };
        let lhs = self.eval_usize(&binary.lhs)?;
        let rhs = self.eval_usize(&binary.rhs)?;
        let (checked, symbol): (CheckedOp, &str) = match binary.op {
            BinaryOp::Add => (u64::checked_add, "+"),
            BinaryOp::Sub => (u64::checked_sub, "-"),
            BinaryOp::Mul => (u64::checked_mul, "*"),
            BinaryOp::Div => (u64::checked_div, "/"),
            /* comparisons and bitwise → not_a_count, unchanged */
        };
        // division by zero and overflow refusals: the same texts, `lhs`/`rhs` now `u64`
    }
}

/// A fold's bounds, each a `usize` built from integer literals, `usize` consts,
/// `+ - * /` and parentheses, running forwards. [rest of today's doc, minus
/// "and a structural parameter is part of the program"]
pub(crate) fn range_bounds(range: &RangeExpr, scope: RangeScope<'_>) -> syn::Result<Range<u64>>
```

`range_bounds` returns `lo..hi` after the unchanged "runs backwards" refusal. The overflow text
renders identically (`format!("… `{lhs}_usize {symbol} {rhs}_usize` …")` with `symbol: &str`
prints `3_usize - 4_usize`, the text `a_usize_const_is_checked_as_rustc_checks_it` asserts). Add
`use std::ops::Range;` to sema.rs.

### 2.4 `pixelflow-compiler/src/lower.rs`

| Line | Item | Act |
|---|---|---|
| 43-48 | module doc, `Expansion` bullet: "…an entry with structural parameters lowers to a *template* whose folds … ([`Holes`]) and filled when its host function is instantiated." | S: end the bullet at "emission declares each with the call's value." |
| 66-68 | `use crate::sema::{AnalyzedKernel, Bounds, ConstValue, RangeScope, Scalar, StructuralRange, range_bounds};` | S: `{AnalyzedKernel, ConstValue, RangeScope, Scalar, range_bounds}` |
| 138-145 | `enum FoldRange` | D |
| 187-192 | `Site::count` | D |
| 193-205 | `Site::open_fold(&mut self, depth, monoid, range: FoldRange) -> Result<Self::Term, String>` and its "# Errors" | S: `fn open_fold(&mut self, depth: usize, monoid: Monoid, range: Range<u64>) -> Self::Term;` (infallible: its one error was `Holes::stride`'s); doc "Open a fold over `range` under `monoid`, `depth` folds being open already: its index ([`ExprArena::open_fold`]). Lowering has refused a depth past the binders, and a range past the index ([`literal_range`])." |
| 251-266 | `literal_range(lo, hi)` | S: `fn literal_range(range: Range<u64>) -> Result<Range<u64>, String>` (`Some(range).filter(KnownRange::admits)`; message unchanged, its `{lo}..{hi}` from `range.start`/`range.end`). **K** (A6 keeps it for a known pair, design §10) |
| 274-277 | `OPEN_RANGE` | D |
| 279-283 | `HOLE_STRIDE` | D |
| 285-351 | `struct Holes`, `Holes::stride`, `Holes::range_of` | D |
| 353-359 | `struct Lowered { arena, root, holes }` | D: `Expansion::lower` returns `Result<(ExprArena, ExprId), String>` |
| 361-370 | `struct Expansion { arena, holes, open: Vec<(OpenFold, ExpansionFold)> }` and doc "…with the folds open in it and the template's holes" | S: `{ arena: ExprArena, open: Vec<(OpenFold, FoldTerms)> }`; doc "…at macro expansion, with the folds open in it, each with the terms it closes over." Keep `#[derive(Default)]`. |
| 372-378 | `enum ExpansionFold { Known, Hole }` | D: the stored value is the `FoldTerms` itself |
| 380-401 | `Expansion::lower`: "# Errors" text, `Lowered { … holes }` | S: return `Ok((site.arena, root))` |
| 461-476 | `Expansion::count` (the `Param` hole, `push_param`, the `u8` comment) | D |
| 478-495 | `Expansion::open_fold` | S: build `FoldTerms { monoid, lo: Bound::Lit(range.start), hi: Bound::Lit(range.end) }`, push `(open, terms)`, return the index |
| 497-525 | `Expansion::close_fold`'s `match fold { Known…, Hole… }` | S: `let (open, terms) = …pop()…; self.arena.close_fold(open, body, terms).map_err(…)` (message unchanged) |
| 567-576 | `lower_entry`'s `Frame { role, structural: &entry.structural, locals }` | S: `Frame { role: entry.role(), locals }` |
| 587-598 | `Frame` doc "its structural parameters, and…", field `structural` | S: doc "The function being lowered: the bindings in scope —…"; delete the field |
| 767-783 | `lower_fold` doc "a known range, or one naming the entry's structural parameters"; `RangeScope { consts, structural }`; `match bounds { Known…, Structural… }` | S: doc "`⊕_{i ∈ [lo, hi)} body` as one `Reduce` over its [`Fold`], its range known here."; body L1 (below the table) |
| 796-798 | `self.site.open_fold(…)?` | S: no `?` |
| 810-813 | `lower_cast` doc "…a structural parameter is the site's [`Site::count`]…" | S: end at "rounded as Rust's `as` rounds it." |
| 829-831 | `lower_cast`'s structural branch | D |
| 905-908 | `inline` doc "no local, no entry parameter, no structural parameter" | E: "no local and no entry parameter" |
| 927-931 | `Frame { role: Role::Helper, structural: &[], locals }` | S: drop the field |
| 971-974 | comment "a binding, then a const, then a structural parameter, then the coordinates" | E: "a binding, then a const, then the coordinates" |
| 984-989 | `resolve`'s structural branch | D |
| 1061-1063 | tests' `use pixelflow_ir::{FoldHeader, Lit, Placeholder};` | S: `{Lit, Placeholder}` (`FoldHeader` was the deleted tests' alone) |
| 1094 | `let Lowered { arena, root, .. } = Expansion::lower(…)?;` | S: `let (arena, root) = Expansion::lower(…)?;` |
| 1160-1240 | tests `a_structural_parameter_leaves_the_template_open`, `a_placeholder_no_hole_names_is_refused`, `a_known_fold_is_no_hole` | D (§5.1) |

L1, `lower_fold`'s range (replaces `:775-783`):

```rust
let scope = RangeScope {
    consts: &self.program.analyzed.consts,
};
let range = literal_range(range_bounds(&fold.range, scope).map_err(|e| e.to_string())?)?;
```

Imports that stay used: `KnownRange` (`literal_range`), `Fold` (`Fold::EXACT_BOUND` in
`literal_range`), `Range`, `Bound`, `FoldTerms`, `Monoid`.

### 2.5 `pixelflow-compiler/src/emit.rs`

| Line | Item | Act |
|---|---|---|
| 31-33 | "No optimizer runs at expansion on such an entry, as on a template: the composed program…" | E: "No optimizer runs at expansion on such an entry: the composed program…" |
| 44-46 | "An entry's structural parameters (`const N: usize`) are the host function's const generics, and each value is its own program." | D |
| 52 | `use pixelflow_ir::{Bound, Fold, Lit, OpKind};` | S: `{Bound, Lit, OpKind}`; add `use pixelflow_ir::Fold;` to the test module (its `sum_to` uses `Fold::new`). `Fold`'s only non-test user was `instantiated_range` (`:387`). |
| 58 | `use crate::lower::{Expansion, FoldRange, Holes, Lowered, Site, lower_entry};` | S: `{Expansion, Site, lower_entry}`; add `use std::ops::Range;` |
| 75-80 | `emit_kernel` doc "…its host `fn`, generic over its structural parameters and taking its parameters…" | E: "…its host `fn`, taking its parameters…" |
| 130-137 | `entry_arena` doc's template paragraph | D (keep the first sentence) |
| 143-156 | `Lowered { arena, root, holes }`, `EntryContext { arguments, structural, holes }`, **the template exemption** `if !entry.structural.is_empty() { return … }` | S: M1 (below the table). **The exemption is deleted** (§3). |
| 184-191 | `pub struct EntryContext` | D: its one remaining field becomes `arena_to_tokens`'s `arguments: &[TokenStream]` |
| 210-213 | `emit_entry` doc "…generic over its structural parameters and taking…" | E: drop the clause |
| 226, 237 | `let (generics, _) = structural_generics(…)`, `#generics` | D both (an empty `TokenStream` interpolates to no tokens, so the expansion of a non-generic entry is unchanged token for token) |
| 243-249 | `emit_args` doc "…by their declared types, generic over its structural parameters; and…" | E: drop the clause |
| 256, 281, 285 | `structural_generics` call, `#generics`, `impl #generics #args #arguments` | S: `#vis struct #args {`, `impl #args {` |
| 310-321 | `fn structural_generics` | D |
| 373-409 | `fn instantiated_range` | D |
| 425-432 | `arena_to_tokens` doc: "…in order: …positional binding cannot shift. A template's holes are filled from…([`instantiated_range`])." | S: end at "…so a positional binding cannot shift." and "one per argument of `context`" → "one per entry of `arguments`" |
| 440-447 | `pub fn arena_to_tokens(arena, root, context: &EntryContext)`; `context.arguments` | S: `pub fn arena_to_tokens(arena: &ExprArena, root: ExprId, arguments: &[TokenStream]) -> TokenStream`; `arguments.len()`, `arguments.iter()`; panic text unchanged |
| 459 | `let expr = node(arena, id, context);` | S: `node(arena, id)` |
| 503 | `fn node(arena, id, context: &EntryContext)` | S: `fn node(arena: &ExprArena, id: ExprId) -> TokenStream` |
| 520-530 | `ExprNode::Param(k)` arm reading `context.structural` | S: `ExprNode::Param(k) => panic!("kernel! produced ExprNode::Param({k}) — a `Param` is a rewrite rule's metavariable, instantiated before any arena reaches emission; lowering builds none")` |
| 576-631 | `Reduce` arm: comment `:588-591`, `let hole = …range_of…`, `match hole { None => …, Some(range) => … }` | S: the `None` arm's body is the whole arm (its comment "A bound that is a value is a child after the body, `lo`'s first, as the header's bits say." kept). **K** the `Bound::Value` children path (A4's; A6 emits through it, design §10). |
| 713-744 | `Staged::fold_terms(monoid, range: FoldRange)`: doc "…known, or this instantiation's ([`instantiated_range`])"; `match range`; `let __range = #range;`; `stride: 1,` | S (below) |
| 822-825 | `Staged::count` | D |
| 827-843 | `Staged::open_fold(…, range: FoldRange) -> Result<Ident, String>` | S: `(…, range: Range<u64>) -> Ident`; body unchanged but `self.step(…)` returned without `Ok` |
| 958-1001 | test `a_record_and_an_args_record_expand_to_host_structs` | S (§5.1) |
| 1003-1023 | test `a_templates_holes_are_filled_by_the_instantiation` | D (§5.1) |
| 1035-1054 | test `a_template_is_offered_to_no_optimizer` | S, renamed (§5.1) |
| 1125-1154 | tests `a_value_bounded_fold_emits_its_children`, `a_literal_fold_emits_no_children` (build an `EntryContext`) | S: `node(&arena, fold)` |

M1, `entry_arena` after A5 (replaces `:138-172`; the optimizer `match` at `:158-170` is kept as is):

```rust
/// Lower an entry, run the optimizer over it, and emit the code that
/// rebuilds the arena: an expression evaluating to `(arena, root)`.
fn entry_arena(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    optimizer: &mut dyn Optimize,
) -> Result<TokenStream, String> {
    let (arena, root) = Expansion::lower(entry, analyzed)?;
    let arguments: Vec<TokenStream> = analyzed
        .parameters(entry)
        .into_iter()
        .flat_map(|parameter| parameter.scalars())
        .map(argument)
        .collect();

    // [`:158-170`, unchanged: declining is ordinary; relink restores the
    //  declaration order]
    let (arena, root) = match optimizer.optimize(&arena, root).into_changed() { /* … */ };
    Ok(arena_to_tokens(&arena, root, &arguments))
}
```

`Staged::fold_terms` after A5 (the `let __range` local only existed so one quote served the known
and the instantiated arm):

```rust
/// The [`FoldTerms`](pixelflow_ir::arena::FoldTerms) an open fold closes
/// over: its monoid, carried as a header's bits (as [`node`] carries a
/// `FoldHeader`), and its known range. The binder is the one `close_fold`
/// chooses.
fn fold_terms(monoid: Monoid, range: Range<u64>) -> TokenStream {
    let first = Binder::from_slot(0).expect("the IR has a binder");
    let bits = pixelflow_ir::FoldHeader::new(
        pixelflow_ir::Fold::new(monoid, first),
        pixelflow_ir::Bound::Lit(0),
        pixelflow_ir::Bound::Lit(0),
    )
    .to_bits();
    let (lo, hi) = (range.start, range.end);
    quote! {
        ::pixelflow_core::__macro::ir::arena::FoldTerms {
            monoid: ::pixelflow_core::__macro::ir::fold::FoldHeader::from_bits(#bits)
                .expect("kernel! emitted a well-formed fold")
                .fold()
                .monoid(),
            lo: ::pixelflow_core::__macro::ir::fold::Bound::Lit(#lo),
            hi: ::pixelflow_core::__macro::ir::fold::Bound::Lit(#hi),
        }
    }
}
```

This changes the tokens of a fold inside a `Staged` entry (no `{ let __range … }` block, no
`stride`). No production `Staged` entry has a fold (`program.rs`'s `sum2` and `glyph`,
`pixelflow-graphics/src/fonts/loop_blinn/program.rs:125-137`), so no production byte moves (§4).

### 2.6 `pixelflow-compiler/src/lib.rs`

| Line | Item | Act |
|---|---|---|
| 71-73 | `PLAN` | K; add `BOUND_PLAN` after it (§1.1) |
| 153-164 | `# Folds`: "The bounds are constant — integers, `usize` consts and an entry's structural parameters — and the index…" | E: "The bounds are constant — integers and `usize` consts, with `+ - * /` between them — and the index…" |
| 250-304 | `# Binding times`: "one of three times", the **Structural** bullet, the `rings<const N: usize>` example with `rings::<4>` and `RingsArgs::<4>` | S (§6.5) |
| 313-317 | pipeline step 4: "…so composition still gets the chain rule; so is an entry with structural parameters, a template until it is instantiated." | E: end at "…still gets the chain rule." |
| 351 | `macro_tier` doc "equality saturation over the template vocabulary" | K (`Vocabulary::Templates`, the rewrite rules' vocabulary: unrelated) |

### 2.7 `pixelflow-ir` (the transitive closure: `FoldTerms::stride`)

After §2.4 and §2.5 no constructor of `FoldTerms` passes a stride other than 1: the only one that
did was `ExpansionFold::Hole` (`lower.rs:509-514`). The other constructors are
`Kernel::over` (`kernel.rs:765-770`), `Kernel::over_values` (`:864-869`), a kernel.rs test
(`:1686-1691`), `arena.rs`'s test helper `terms` (`:2455-2462`), `Expansion::close_fold`
(`lower.rs:503-508`) and `Staged::fold_terms`' tokens (`emit.rs:734-742`): all `stride: 1`.

| File:line | Item | Act |
|---|---|---|
| arena.rs:166-169 | `FoldTerms` doc "the fold's algebra and step, and its bounds" | E: "the fold's algebra and its bounds" |
| arena.rs:180-182 | `pub stride: u64` and its doc "1, but for `kernel!`'s open folds (`HOLE_STRIDE`) until A5 deletes them" | D |
| arena.rs:1746-1751 | `close_over`'s destructure `FoldTerms { monoid, stride, lo, hi }` | S: `{ monoid, lo, hi }` |
| arena.rs:1762 | `let fold = Fold::strided(monoid, binder, stride);` | S: `Fold::new(monoid, binder)` (`Fold::new` is `Fold::strided` with 1, `fold.rs:1768`'s pin) |
| arena.rs:2455-2462 | test helper `terms`, doc "What a builder closes a stride-1 fold over." | S: drop `stride: 1`; doc "What a builder closes a fold over." |
| kernel.rs:767, 866, 1688 | `stride: 1,` | D |
| fold.rs:538-541 | `Fold::stride` doc "only `pack`, `kernel!`'s open folds and `HalveFold` make one" | E: "only `pack` and `HalveFold` make one" |

`Fold::strided` (`fold.rs:595`) **K**: `pack` (`passes/lattice.rs:241`) and `HalveFold` make
strides. `FoldHeader::new`'s refusal of a stride other than 1 beside a value bound **K**.

Public surface: `pixelflow_ir::arena::FoldTerms` loses a `pub` field. Its constructors outside
`pixelflow-ir` are the compiler's and the code the compiler emits (through
`::pixelflow_core::__macro::ir`), both changed here; nothing else in the tree builds one
(`grep -rn FoldTerms --include=*.rs .`). Record it in `.wip/a5/public-surface.md`.

### 2.8 Elsewhere

| File:line | Item | Act |
|---|---|---|
| `pixelflow-codegen/src/program/lower.rs:373-377` | `Param` panic: "a template's slot (a `kernel!` template's structural hole, a rewrite rule's metavariable) is filled before a Kernel exists" | E: "ExprNode::Param({}) reached the JIT emitter -- a rewrite rule's metavariable is instantiated before a Kernel exists" (no test pins the text: `grep -rn "reached the JIT emitter" --include=*.rs .` has this one hit) |
| `pixelflow-graphics/src/fonts/loop_blinn/program.rs:15-16` | "Choosing a glyph's program is choosing by a structural parameter, the piece count, which is what structural means: each value its own program." | E: "Choosing a glyph's program is choosing by its piece count: each count its own program." (doc only; B4 deletes the file) |
| `pixelflow-compiler/intent.md` | §6.4 | E |

### 2.9 What is not deleted, and why (the closure stops here)

- `Param(u8)`, `push_param`, `ExprNode::Param`, `ENode::Param`, `Declined::Param`: the rewrite
  rules' metavariable (plan:438-440, the-language-is-kernel D2). After A5 `kernel!` builds none
  (G4).
- `Ty::Usize`, `SymbolKind::Index`, `register_index`, `Binding::Index`, `a_count_is_not_a_value`,
  `type_of_cast`'s `names_a_count`, `LiteralExpr::usize_value`, `ConstValue::Usize`: the fold's
  index and the `usize` const, which stay, and on which A6 builds (design §10).
- `RangeScope` (one field), `UsizeScope`, `range_bounds`, `literal_range`, `CheckedOp`: the
  evaluation of a known bound, which stays and which A6 widens (design §10).
- `FoldHeader::to_bits`/`from_bits` in emission and `Staged`: how a monoid crosses into tokens.
- `node`'s `Bound::Value` children (A4): no `kernel!` entry writes a value bound until A6, whose
  emission goes through exactly this path (and an arena the optimizer hands back is emitted through
  it whatever its bounds are).

---

## 3. The template exemption, and A7

**The code** (`emit.rs:154-156`):

```rust
if !entry.structural.is_empty() {
    return Ok(arena_to_tokens(&arena, root, &context));
}
```

**Its condition after A5 is not expressible**: it reads `FnItem::structural`, which §2.1 deletes,
and nothing else in an entry distinguishes a template: an entry's folds are all `Bound::Lit` pairs
over `Fold::new`, the shape every non-template entry has always had. Its reason, the `0..0`
placeholder a rewrite would read as an empty range (`EmptyFold`), leaves with `Holes`. So A5
deletes the three lines; keeping any exemption would mean inventing a condition with no reason.

**What it exempted at `7e9531e`** — every entry with a `const` generic that reaches `entry_arena`
(a kernel-taking entry is `Staged` and never reaches it, with or without generics):

| Entry | File | Optimizer it was hiding from | After A5 |
|---|---|---|---|
| `ramp<const N>` | `tests/binding_times.rs:39` | `kernel!`'s `macro_tier` | `ramp` over `const N: usize = 4`, saturated |
| `rings<const N>` | `tests/binding_times.rs:46` | `macro_tier` | `rings` over the same const, saturated |
| `lower_case::span<const lo, const hi>` | `tests/binding_times.rs:178` | `macro_tier` | deleted with its test |
| `grid<const M, const N>` | `tests/binding_times.rs:134` | `kernel_raw!`'s `Identity` (no difference) | deleted with its test |
| `swept<const N>` | `tests/items_block.rs:131` | `macro_tier` | `swept` over `const N: usize = 3`, saturated |
| `mean_index<const N>` | `tests/rustc_is_the_oracle.rs:659` | `macro_tier` | `mean_index` over `const N: usize = 4`, saturated |
| `inside<const N>` | `src/emit.rs:963` (unit test) | `Identity` | `inside`, not generic |
| `mean<const N>` | `src/emit.rs:1009` (unit test) | `Identity` | deleted with its test |
| `counted<const N>` | `src/emit.rs:1043` (unit test) | `Offered` (the exemption's own pin) | `counted` over `const N: usize = 4`, now offered |
| `rings<const N>` | `src/lib.rs:286` (```` ```ignore ```` doc) | never compiled | `rings` over a const (§6.5) |

`applied<const N>` (`items_block.rs:138`) and `mean_down<const N>`
(`kernel_typed_parameters.rs:95`) take a kernel, so they are `Staged`; the exemption never applied
to them. **No production entry is in the table**: the one production `kernel!` block
(`pixelflow-graphics/src/fonts/loop_blinn/program.rs:31-141`; entries `one_piece`, `sum2`,
`glyph`, `blank`) has no generic, and no other crate, example or bench declares a `const` generic
in a `kernel!` block (`grep -rnE 'fn [a-z_0-9]+ ?<\s*const' --include=*.rs .` outside
`pixelflow-compiler/` finds none in a `kernel!` block; its five hits are ordinary Rust:
`pixelflow-search/src/nnue/guide/scoring.rs:1145`, `:1161`, `scoring/backward.rs:395`, `:409`, and
`pixelflow-codegen/tests/value_bounds_jit.rs:166`).

**So A7 is empty.** The plan wanted A7 in its own CL "so the byte change is measured, not a side
effect" (plan:706-708). The byte change is the four test kernels above going through `macro_tier`;
§4 shows no production byte moves; the gates of §7 measure that. The plan's A7 bullet is rewritten
to say A7 was folded into A5, and why (§6.1).

**The docs that state the exemption:** `lib.rs:313-317` (§2.6), `emit.rs:130-137` (§2.5),
`intent.md:49-52` ("`kernel!` saturates any entry with no structural or kernel-typed parameter and
no `Dwrt`" → "no kernel-typed parameter and no `Dwrt`", §6.4), the ontology's `kernel!` row
(`language.md:76-80`, §6.6), and the plan's A7 bullet.

**The pin that replaces the exemption's**: `a_template_is_offered_to_no_optimizer` becomes
`every_entry_that_takes_no_kernel_is_offered_to_the_optimizer` (§5.1): an entry with a fold is
offered, so an exemption by shape cannot come back unnoticed.

---

## 4. Byte identity

### 4.1 Production kernels pass through no deleted branch

- **The 29 font programs** (gate 1b's and gate 2's workload) are built from `program.rs`'s
  `kernel!` entries. `one_piece(p: Row)` and `blank()` go through `entry_arena`; `sum2` and
  `glyph` take kernels and are `Staged`. None has a generic and none has a fold
  (`grep -n '\.map(' pixelflow-graphics/src/fonts/loop_blinn/program.rs` hits only host code at
  `:161`, `:245`). Branch by branch:
  - parser: `structural_parameters(&empty generics)` returned `vec![]`; `refuse_generics` returns
    `Ok(())`; the AST is the same less a field that was empty.
  - sema: `register_structural` never ran; `RangeScope::usize_named`'s structural arm unreachable
    (no fold at all here).
  - lowering: no fold, so no `FoldRange`, `Holes` or `ExpansionFold`; no `as f32` of a structural
    name, so no `count`/`Param`.
  - emission: the exemption's condition was false; `structural_generics(&[])` was two empty token
    streams, which `quote!` interpolates as nothing, so `#vis fn #name ( … )` and
    `#vis struct #args { … }` / `impl #args` are token-identical; `node` met no `Param` and no
    fold. **The expansion of every production entry is token-identical**, so the `Kernel`s it
    rebuilds are node-identical.
  - `Staged` (`sum2`, `glyph`): no fold, so `fold_terms` is not reached.
- **The runtime tier and `byte_probe`** build folds through `Kernel::over`/`over_values` and
  `push_reduce`; `close_over`'s `Fold::strided(monoid, binder, 1)` becomes `Fold::new(monoid,
  binder)`, equal by `fold.rs:1768`'s pin. `byte_probe` (`pixelflow-codegen/examples/byte_probe.rs`)
  does not use `kernel!` (`pixelflow-codegen`'s manifest has no `pixelflow-compiler`).
- **Production's `Glyph::bake`** builds with builders, not `kernel!`.

### 4.2 Gates (`.wip/gates/README.md`; A5 runs them with `TAG=a5-impl`, after `TAG=a5-pre` on the unmodified tree)

| Gate | Expected |
|---|---|
| 1 `byteprobe.sh` | identical to base, both tiers (exit 0) |
| 1b `gate1b.sh` | identical: 29 `ZZGATE` lines, keys and digests |
| 2 `gate2.sh` | identical saturation telemetry |
| 2x `gate2x.sh` | identical extraction fields |
| 5 `measure.sh` | `a5-impl` against `a5-pre`: total Ir within ±0.01% (run-to-run noise is 3e-5%); `saturate` and `extract_dag_over` inclusive identical. Report, never assert (README). |

`a5-pre` runs first because the recorded bases are `a747770`'s (README "Bases"): if any of gates 1,
1b, 2, 2x differs from base at the unmodified `7e9531e`, stop and report (A4's Done line says they
did not move).

### 4.3 Test kernels that change, and that this is expected

`ramp`, `rings` (binding_times), `swept` (items_block) and `mean_index` (rustc oracle), rewritten
over `usize` consts, are **saturated by `macro_tier` at expansion** where as templates they were
emitted as lowered and optimized only at bake. Their emitted arenas change; that is the exemption's
deletion working, not a regression. Their values do not: `/home/user/scratch/a5-probe/tests/probe.rs`
runs exactly §5's rewrites at `7e9531e` (where a non-generic entry is already saturated) and all
five tests pass under `PIXELFLOW_ISA=avx2` and `=avx512`. `rings`' corner reads `0.7498169` on AVX2
(its `/ (N as f32)` divides through `Recip`, `rcpps`) and `0.75` on AVX-512, inside the test's
`RECIP_RELATIVE_TOLERANCE`; `ramp`, `swept` and `mean_index` compare exactly and pass on both
tiers. NEON runs in CI's macOS job only: every comparison there is either against the same program
(rebinding), toleranced (`rings`), or of sums of small integers times quarters (exact on every
tier). `applied` and `mean_down` stay `Staged`, never optimized at expansion; their folds' tokens
change (§2.5) and their arenas do not.

---

## 5. Tests

### 5.1 Each test that uses or names the structural path

| Test (file:line) | What it pinned | A5 |
|---|---|---|
| parser `an_entrys_const_generics_are_its_structural_parameters` (`parser.rs:2381-2397`) | `<const N, const M>` parse to `structural` in order; a helper has none | **Delete.** Subject (the field) gone; the refusal that replaces it is the new test below. |
| parser `a_collection_spelled_any_way_is_refused_naming_its_sections`, case 2 (`:2545`) | an entry's array-of-records parameter `[Row; N]` is refused as a collection | **Rewrite the case** to `quote! { pub struct Row { pub x0: f32 } pub fn f(pieces: [Row; 4]) -> f32 { X } }`: with a `const` generic the generic refusal now fires first, and the pin is the collection. |
| parser `a_fn_signature_is_plain` (`:2757-2790`) | 9 cases: 6 generic spellings, `fn f() { X }`, `mut x`, `const fn` | **Keep 3 cases**: `pub fn f() { X }` → "declares no return type", `pub fn f(mut x: f32) -> f32 { x }` → "plain name", `const fn f() -> f32 { X }` → "`const fn`". The 6 generic cases move to the new test. Doc: "A `fn` signature carries nothing the language cannot honor. (Generics are refused by `generics_are_refused_and_a_const_generic_says_what_a_count_is`.)" |
| sema `a_structural_parameter_is_a_count` (`sema.rs:3167-3195`) | `LO..LO + N * 2` over a structural `N` evaluates to a `StructuralRange` with text `2..(2 + (N * 2))` | **Delete.** Subject gone; const arithmetic in a range is `a_range_from_usize_consts_is_evaluated_at_expansion`'s, `N as f32` of a const is `as_f32_converts_a_usize_and_nothing_else`'s. |
| sema `a_structural_parameter_is_not_a_value_and_is_its_entrys` (`:3197-3232`) | `X * N` refused; `let N`/param `N` shadow it; a `const N` collides; a helper does not see it; declared twice | **Delete.** Every subject is the structural symbol. Surviving neighbours: `X * N` for a `usize` const is case 9 of `a_usize_where_a_value_is_expected_is_a_type_error`; const shadowing is `an_item_is_not_shadowed`. |
| sema `a_range_from_usize_consts_is_evaluated_at_expansion` (`:2787-2820`) | `LO..LO + N` over consts is `Bounds::Known(3, 14)` | **Edit:** `RangeScope { consts: &analyzed.consts }`; `assert_eq!(range_bounds(&fold.range, scope).expect("constant"), 3..14, "`LO..LO + N` is `3..14`");` |
| sema `a_range_bound_that_is_not_constant_is_refused` (`:2725-2773`), `a_usize_is_not_a_declared_value_type` (`:2946-2972`) | non-constant bounds refused; `usize` is no parameter type | **Keep unchanged** (their asserted substrings survive §2.3's message edits). A6 rewrites both (design §7.4). |
| lower `a_structural_parameter_leaves_the_template_open` (`lower.rs:1160-1203`) | `N as f32` is `Param(k)`; each structural range is its own hole; a known fold is not one | **Delete.** Subject gone. A known fold lowering to one `Reduce` is `a_fold_lowers_to_one_reduce_and_leaves_nothing_behind`'s. |
| lower `a_placeholder_no_hole_names_is_refused` (`:1205-1217`), `a_known_fold_is_no_hole` (`:1219-1240`) | `Holes::range_of`'s panic and its `None` | **Delete.** Subject gone. |
| emit `a_record_and_an_args_record_expand_to_host_structs` (`emit.rs:958-1001`) | a record's host struct; an entry's `Args`, generic over `<const N: usize>`; `set_declared` order; no `::std` | **Rewrite** the entry to `pub fn inside(b: Bounds, r: f32) -> f32 { … }`; assert `code.contains("pub fn inside (b : Bounds , r : f32)")` and `code.contains("pub struct InsideArgs {")`; doc drops "generic over its structural parameters". Other asserts unchanged. |
| emit `a_templates_holes_are_filled_by_the_instantiation` (`:1003-1023`) | the `const { … }` range block and `push_const (N as f32)` | **Delete.** Subject gone. |
| emit `a_template_is_offered_to_no_optimizer` (`:1035-1054`) | the exemption: `counted<const N>` not offered, `plain` offered | **Rewrite and rename** `every_entry_that_takes_no_kernel_is_offered_to_the_optimizer` (code below the table, E1). |
| emit `a_value_bounded_fold_emits_its_children`, `a_literal_fold_emits_no_children` (`:1125-1154`) | A4's `attach([…])` / `attach(::core::iter::empty())` | **Edit:** `node(&arena, fold).to_string()`; asserts unchanged. |
| ast `an_entrys_args_record_is_its_name_in_upper_camel_case` (`ast.rs:807-831`) | `Args` naming | **Edit:** drop `structural: Vec::new(),` |
| binding_times `ramp<const N>`, `rings<const N>` (`:38-53`) | the entries | **Rewrite**: add `const N: usize = 4;` to the block after `Bounds`; `pub fn ramp(scale: f32)`, `pub fn rings(b: Bounds, fg: f32, bg: f32)`, bodies unchanged (they read `N`). Docs: `ramp` "The mean of `X·i` over `i ∈ [0, N)`, scaled."; `rings` keeps "the macro doc's example, compiled, so the doc cannot drift from the language". |
| binding_times `every_call_of_an_entry_at_one_instantiation_is_one_program` (`:97-126`) | every call one program (key + code), `boxed` and `ramp::<4>`; `ramp::<4>` ≠ `ramp::<5>` | **Rename** `every_call_of_an_entry_is_one_program`; `ramp(1.0)`/`ramp(2.5)` in the two equalities; **delete** the `assert_ne!` (its subject, "a structural value is its own program", is gone; A6 adds the opposite pin for a per-call count, design §7.2). Doc ends at "…share a canonical key and a compiled region." |
| binding_times `kernel_raw!` block `grid`, `grid_two_by_five` (`:128-148`) and `each_structural_range_is_a_hole_of_its_own` (`:150-167`) | two holes are two programs, filled = written | **Delete** (block and test). Subject gone. `use pixelflow_compiler::{kernel, kernel_raw};` (`:17`) becomes `use pixelflow_compiler::kernel;` (no other `kernel_raw!` in the file). |
| binding_times `mod lower_case` (`:169-182`) and `a_structural_parameter_named_as_the_emissions_locals_is_its_own` (`:184-194`) | `instantiated_range`'s `__lo`/`__hi` hygiene | **Delete.** Subject gone; no emitted local is spelled from a user name after A5. |
| binding_times `a_program_compiled_once_is_rebound_from_args` (`:229-235`) | rebinding `ramp`'s program from `RampArgs` | **Edit**: `ramp(0.0)`, `RampArgs { scale: 2.0 }`, `ramp(2.0)` |
| binding_times `the_macro_docs_example_bakes_and_rebinds` (`:377-408`) | the lib.rs example bakes and rebinds | **Edit**: `rings(b, …)`, `RingsArgs { … }` |
| binding_times module doc (`:1-15`) | | **Rewrite** (§6.7) |
| items_block `swept<const N>`, `applied<const N>` (`:129-140`); test `a_block_expands_where_no_prelude_is_in_scope` (`:150-183`) | a block expands in `#[no_implicit_prelude]`, a fold over a structural count and a staged fold among its parts | **Rewrite**: add `const N: usize = 3;` to the no-prelude block; `pub fn swept(m: Mass, v: f32, r: f32)`, `pub fn applied(k: impl Fn(f32, f32) -> f32, m: Mass)`, bodies unchanged; in the test `swept(m, v, r)`, `SweptArgs { … }`, `swept(args.m, args.v, args.r)`, `without_a_prelude::applied(&k, m)`. Doc `:111` "a fold over a structural count" → "a fold over a `usize` const"; `:137` "A kernel applied in a fold over a structural count, weighed." → "…over a `usize` const, weighed." |
| kernel_typed_parameters `mean_down<const N>` (`:94-97`), `a_structural_entry_applies_its_argument` (`:260-268`) | a staged fold around an application, `N` as a value; two counts | **Rewrite** (code below the table, E2); the `<1>` case goes (its subject, a second instantiation, is gone). |
| rustc oracle section header and comment (`:590-596`), `mean_index<const N>`, `rust_mean_index<const N>`, `a_structural_count_folds_as_rustcs_const_generic_does` (`:651-693`) | a range in const arithmetic over a count, the count as a value through a helper, against rustc, at three counts | **Rewrite** (§5.2) |

E1, emit.rs (replaces `:1035-1054`):

```rust
/// Every entry that takes no kernel is offered to the optimizer at
/// expansion, one with a fold among them: nothing is exempt by its shape.
#[test]
fn every_entry_that_takes_no_kernel_is_offered_to_the_optimizer() {
    let analyzed = analyze(
        parse(quote! {
            const N: usize = 4;
            pub fn counted(r: f32) -> f32 {
                r + (0..N).map(|i| (i as f32) * X).sum::<f32>()
            }
            pub fn plain(r: f32) -> f32 { r + X }
        })
        .expect("parses"),
    )
    .expect("analyzes");
    let mut offered = Offered(0);
    emit_kernel(&analyzed, &mut offered).expect("emits");
    assert_eq!(offered.0, 2, "`counted` and `plain`");
}
```

E2, kernel_typed_parameters.rs: in the `kernel!` block (`:61-122`) add `const DOWN: usize = 4;`
before the entries and replace `:94-97` by

```rust
    /// An argument applied under a fold over a `usize` const.
    pub fn mean_down(k: impl Fn(f32, f32) -> f32) -> f32 {
        (0..DOWN).map(|i| k(X, Y + (i as f32))).sum::<f32>() / (DOWN as f32)
    }
```

and `:260-268` by

```rust
/// A fold over a `usize` const around an application, the count read as a
/// value too.
#[test]
fn a_fold_over_a_usize_const_applies_its_argument() {
    let k = kernel!(|| X + Y);
    // (8 + 9 + 10 + 11) / 4.
    assert_eq!(bake(&mean_down(&k)), 9.5);
}
```

### 5.2 The rustc oracle rewrite (`rustc_is_the_oracle.rs:590-693`)

Header `// ─── B3: records, and counts ───`; the second comment paragraph becomes: "A count is a `usize`
const, and the oracle for a range over one, and for the count as a value, is the same tokens over
Rust `const`s." Then:

```rust
kernel! {
    const FIRST: usize = 1;
    const N: usize = 4;

    /// A helper reads a count as an argument.
    fn per(n: f32, total: f32) -> f32 { total / n }

    /// The mean of `X·i` over `i ∈ [FIRST, FIRST + N)`: `N` in a range's const
    /// arithmetic, and as a value.
    pub fn mean_index() -> f32 {
        per(N as f32, (FIRST..FIRST + N).map(|i| X * (i as f32)).sum::<f32>())
    }
}

fn rust_mean_index(x: f32) -> f32 {
    const RUST_FIRST: usize = 1;
    const RUST_N: usize = 4;
    (RUST_FIRST..RUST_FIRST + RUST_N)
        .map(|i| x * (i as f32))
        .sum::<f32>()
        / (RUST_N as f32)
}

/// A range in a count's const arithmetic, and the count as a value through a
/// helper, mean what the same tokens over Rust `const`s mean; each sum is exact.
#[test]
fn a_count_in_a_ranges_const_arithmetic_folds_as_rustcs_does() {
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&mean_index(), x, y), rust_mean_index(x), "at ({x}, {y})");
    }
    // X = 3: (1 + 2 + 3 + 4) · 3 / 4.
    assert_eq!(bake(&mean_index()), 7.5);
}
```

(The probe's `mean_index_oracle` is this test with `COUNT` for `N`.) A6 adds the per-call count
case beside it (design §7.1).

### 5.3 The new test (parser.rs, after `a_fn_signature_is_plain`)

```rust
/// No `fn` and no `const` takes generics. A `const` generic is refused by
/// saying what a count is — a literal or a `usize` const, an entry being one
/// program whatever its counts — naming the plan that made a bound a value;
/// any other generic names what the language's values are.
#[test]
fn generics_are_refused_and_a_const_generic_says_what_a_count_is() {
    for input in [
        quote! { pub fn f<const N: usize>() -> f32 { X } },
        quote! { pub fn f<const N: usize, const M: usize>(r: f32) -> f32 { r } },
        quote! { pub fn f<const N: u32>() -> f32 { X } },
        quote! { pub fn f<const N: usize = 4>() -> f32 { X } },
        quote! { fn h<const N: usize>(x: f32) -> f32 { x } pub fn f() -> f32 { h(X) } },
    ] {
        let err = refusal(input);
        assert!(
            err.contains("a `const` generic in a `kernel!` block")
                && err.contains("§1.2 of docs/language/2026-10-20-a-bound-is-a-value.md")
                && err.contains("`const N: usize = 4;`"),
            "got: {err}"
        );
    }
    for input in [
        quote! { pub fn f<T>() -> f32 { X } },
        quote! { pub fn f<'a>() -> f32 { X } },
        quote! { fn h<T>(x: f32) -> f32 { x } pub fn f() -> f32 { X } },
    ] {
        let err = refusal(input);
        assert!(
            err.contains("generics in a `kernel!` block")
                && err.contains("nothing in a kernel is generic"),
            "got: {err}"
        );
    }
    let err = refusal(quote! { pub fn f() -> f32 where f32: Copy { X } });
    assert!(err.contains("a `where` clause in a `kernel!` block"), "got: {err}");
}
```

### 5.4 Test-count accounting (`#[test]` in `pixelflow-compiler`, 306 at `7e9531e`)

| File | At `7e9531e` | Removed | Added | After |
|---|---|---|---|---|
| src/parser.rs | 43 | `an_entrys_const_generics_are_its_structural_parameters` | `generics_are_refused_and_a_const_generic_says_what_a_count_is` | 43 |
| src/sema.rs | 61 | `a_structural_parameter_is_a_count`, `a_structural_parameter_is_not_a_value_and_is_its_entrys` | | 59 |
| src/lower.rs | 21 | `a_structural_parameter_leaves_the_template_open`, `a_placeholder_no_hole_names_is_refused`, `a_known_fold_is_no_hole` | | 18 |
| src/emit.rs | 8 | `a_templates_holes_are_filled_by_the_instantiation` | (rename only) | 7 |
| tests/binding_times.rs | 9 | `each_structural_range_is_a_hole_of_its_own`, `a_structural_parameter_named_as_the_emissions_locals_is_its_own` | (rename only) | 7 |
| all others | 164 | | | 164 |
| **Total** | **306** | **9** | **1** | **298** |

Renamed, not counted: emit `a_template_is_offered_to_no_optimizer` →
`every_entry_that_takes_no_kernel_is_offered_to_the_optimizer`; binding_times
`every_call_of_an_entry_at_one_instantiation_is_one_program` → `every_call_of_an_entry_is_one_program`;
kernel_typed_parameters `a_structural_entry_applies_its_argument` →
`a_fold_over_a_usize_const_applies_its_argument`; rustc oracle
`a_structural_count_folds_as_rustcs_const_generic_does` →
`a_count_in_a_ranges_const_arithmetic_folds_as_rustcs_does`. `pixelflow-ir`'s count is unchanged
(its tests edit a field initializer). Verify with
`grep -c '#\[test\]' pixelflow-compiler/src/*.rs pixelflow-compiler/tests/*.rs` and `cargo test
-p pixelflow-compiler 2>&1 | grep 'test result'` (sum of `passed`).

---

## 6. Docs

### 6.1 The plan (`docs/language/2026-10-20-a-bound-is-a-value.md`)

- **§3.4, plan:416-424**: replace from "The structural path goes with…" to the end of the
  paragraph with: "The structural path went with `<const N: usize>`, its only producer, in A5:
  `Count::Structural`, `StructuralRange`, `instantiated_range`, `FoldRange`, `Holes`,
  `OPEN_RANGE`, `HOLE_STRIDE`, `ExpansionFold`, the lowering's `Param` hole and the macro tier's
  template exemption are deleted, and `kernel!` refuses a generic on any `fn`." (The deletion is
  stated in the paragraph, so `check-doc-paths.sh`'s rule 2 holds for any name it might read.)
- **§6 A5, plan:697-701**: append a **Done:** line, filled with what the gates measured:
  "**Done:** `kernel!` refuses a generic on any `fn` at parse, a `const` one naming §1.2; the
  structural path is deleted (§3.4), and `FoldTerms` lost its stride, which only the template set;
  the template exemption went with it, so A7 is empty (below). The test templates are written over
  `usize` consts and are saturated at expansion. byte_probe empty in both columns under
  `PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font programs' keys, digests and saturation telemetry,
  the extraction fields included, unchanged." (Edit to the measured facts; do not keep a clause no
  gate checked.)
- **§6 A7, plan:706-708**: replace with "- **A7. Folded into A5.** The exemption (`emit.rs:154-156`
  at `7e9531e`) was `if !entry.structural.is_empty()`, a condition on the field A5 deletes, so A5
  deleted it. It exempted only the test templates A5 rewrote over `usize` consts, which are
  saturated at expansion now, and no production entry, so there was no production byte change to
  measure."
- **§6 A1's Done line, plan:560-564**: the-language-is-kernel §1.3's `usize` row is done in A5
  (§6.2), so the line's exceptions become "…except the-language-is-kernel §1.3's "how many
  instances the host composed" sentence (§5's `:186`, `:199` after A1) and its D-table rows, after
  #1323." (#1431 has landed, `9c33002`; the sentence is A1's to finish, not A5's: A5 does not make
  it false.)
- **Not edited:** §0 (plan:30-33) and §2's row (plan:231) describe the tree the plan was verified
  against; §4 "Deleted" (plan:437-439) already lists A5; the A2a bullet (plan:578, :585) records
  A2a; the A6 bullet (plan:702-705, "Waits on decision 2") is A6's to update (design §8).

### 6.2 `docs/language/2026-09-25-the-language-is-kernel.md`

- **Metadata banner, :48-54** ("Reversed in part by …"): append one sentence: "A5 deleted
  `kernel!`'s `const N: usize` (2026-10): where this document has an entry's structural
  parameters, a template instantiated with structural values or a range over them (§1.1's diagram
  and binding-time bullet, §1.5's "Ranges are constant", §1.7's choice by a structural parameter,
  "What the macro emits"), the tree has none: a count is a literal or a `usize` const." (Use the
  month the CL lands.)
- **§1.3, :192**, the `usize` row: "a fold binder or a structural count" → "a fold binder or a
  count (a literal or a `usize` const)". (§5's row for `:179`, marked "A1, after #1431"; #1431 has
  landed and A5 makes "structural count" name nothing.)
- **§1.4, :211-214**, the reversal note: "`const N: usize` is deleted (its A5)" → "`const N:
  usize` is deleted (A5, done)".
- **Not edited:** the §1.4 table's `structural` row (:208) stays under its note (the tile extent is
  still structural, its §7); D1 (:673) waits on #1323 (plan §5); B3 (:739-745) is a record of what
  was built.

### 6.3 `CLAUDE.md`, `docs/DENOTATION.md`, `pixelflow-core/README.md`

No edit. Evidence: `grep -n -i "structural\|const N\|template\|instantiat" CLAUDE.md
docs/DENOTATION.md` finds only CLAUDE.md:299 ("rewrite rule templates", unrelated) and
CLAUDE.md:329/362 (CI matrix template, expression-template types); `pixelflow-core/README.md` has
no hit. CLAUDE.md:180 ("a per-call count is a uniform") is about the IR, true since A4.

### 6.4 `pixelflow-compiler/intent.md` (59 lines; budget 80; `bash scripts/check-intent.sh`)

| Line | Now | After |
|---|---|---|
| 12-13 | "Binding times as the host sees them (structural, uniform, kernel-typed), the host `fn`, the `Args` record, and `const` evaluation." | "Binding times as the host sees them (uniform, kernel-typed), the host `fn`, the `Args` record, and `const` evaluation: a count is a literal or a `usize` const." |
| 29 | "No call's value becomes a constant: one entry at one instantiation is one program." | "No call's value becomes a constant: one entry is one program." |
| 45-46 | "Unit tests: `emit.rs` `a_template_is_offered_to_no_optimizer`; `sema.rs` `recursion_is_refused`; `parser.rs` `a_collection_spelled_any_way_is_refused_naming_its_sections`." | "Unit tests: `emit.rs` `every_entry_that_takes_no_kernel_is_offered_to_the_optimizer`; `sema.rs` `recursion_is_refused`; `parser.rs` `a_collection_spelled_any_way_is_refused_naming_its_sections`, `generics_are_refused_and_a_const_generic_says_what_a_count_is`." |
| 49-52 | "…and `kernel!` saturates any entry with no structural or kernel-typed parameter and no `Dwrt`; …" | "…and `kernel!` saturates any entry with no kernel-typed parameter and no `Dwrt`; …" (rest unchanged) |
| 53-54 | "A structural count read as a value rides `Param(u8)`, at most 256, against the 64-bit control plane (A5, D2)." | delete (discharged) |
| 57-59 | Decided in | add "`docs/language/2026-10-20-a-bound-is-a-value.md` (§1.2, §3.4)," |

No tracked file may cite a step label in code or intent (A4's rule: `grep -rnwE 'A5' --include=*.rs
--include=intent.md .` finds nothing after A5, G5).

### 6.5 `lib.rs`'s `# Binding times` (`:250-304`)

```
/// # Binding times
///
/// Every value a program reads is bound at one of two times (§1.4):
///
/// - **Uniform**: every parameter. [the bullet at :258-269, unchanged]
/// - **Kernel-typed**: [the bullet at :270-275, unchanged]
///
/// A count — a fold's bound, or `N as f32` — is a literal or a `usize` const,
/// evaluated at expansion: an entry is one program, whatever its counts, and
/// no `fn` takes generics (docs/language/2026-10-20-a-bound-is-a-value.md §1.2).
///
/// ```ignore
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{Lattice, Manifold};
///
/// kernel! {
///     /// An axis-aligned box: a record.
///     pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }
///
///     /// How many rings.
///     const N: usize = 4;
///
///     /// `N` rings about the box's corner, `fg` on `bg`.
///     pub fn rings(b: Bounds, fg: f32, bg: f32) -> f32 {
///         [body at :287-292, unchanged]
///     }
/// }
///
/// let lattice = Lattice::frame(64, 64);
/// let b = Bounds { x0: 8.0, y0: 8.0, x1: 40.0, y1: 40.0 };
/// let once = lattice.bake(&rings(b, 1.0, 0.0));            // one call, baked
///
/// let program = Manifold::compile(&rings(b, 1.0, 0.0), lattice.extent);
/// let mut block = program.block();
/// RingsArgs { b, fg: 0.5, bg: 0.25 }.write_into(&mut block)?;
/// let again = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
/// ```
```

`binding_times.rs`'s `rings` (§5.1) is this block's entry, compiled.

### 6.6 The ontology (`.claude/skills/ontology/`)

"Latest decision wins; earlier meanings go under **Is not**, with the date that retired them"
(SKILL.md:16-19).

- **language.md:145-163, "Structural parameter"** → heading "Structural parameter (retired)":
  - **Is:** nothing in `kernel!` since a-bound-is-a-value A5 deleted it. A count is a literal or a
    `usize` const, known at expansion; a count that changes per call is a value bound
    (`Kernel::over_values`; in `kernel!`, decision 2's `n: usize` entry parameter, A6). What stays
    structural is the lattice's own extent, in the program and the key (a-bound-is-a-value §4,
    "Deferred").
  - **Is not:** a binding time. Retired 2026-10-20 (a-bound-is-a-value §1.2: "a structural count
    was a per-call value spelled so that rustc evaluated it"), deleted in A5: an entry's `const N:
    usize` generic, each value its own program, its ranges evaluated by rustc per instantiation,
    its `N as f32` a `Param(k)` hole.
  - **Follows:** `kernel!` refuses a generic on any `fn` at parse. The template holes and the macro
    tier's template exemption went with it, so every entry that takes no kernel is saturated at
    expansion.
  - **Lives:** `refuse_generics` (`pixelflow-compiler/src/parser.rs`); a-bound-is-a-value §1.2,
    §3.4, §6 A5.
- **SKILL.md:59**: "- Structural parameter" → "- Structural parameter (retired)".
- **language.md:76-80** (`kernel!` Lives): "it saturates at expansion unless the term carries a
  `Dwrt` or structural parameters." → "it saturates at expansion unless the term carries a `Dwrt`
  (an entry that takes a kernel is lowered when called, and optimized at bake)."
- **language.md:85-87** (Entry): "…a host function for it that instantiates the lowered template
  with its structural values and returns the `Kernel`." → "…a host function for it that rebuilds
  its lowered, optimized arena and returns the `Kernel`; an entry that takes a kernel runs
  lowering's steps when it is called (`Staged`)."
- **language.md:92-93** (Args): "each entry's record of its non-structural arguments" → "each
  entry's record of its uniform arguments".
- **language.md:125-126** (Binding time): "An author sees three binding times: **structural**,
  **uniform** and **kernel-typed**." → "An author sees two: **uniform** and **kernel-typed**; a
  count is a literal or a `usize` const (structural parameters retired, a-bound-is-a-value A5)."
- **language.md:369-375** (Param Lives): "…and it has a new job: the template hole for a
  structural parameter read as a value (`count` in `pixelflow-compiler/src/lower.rs`), which
  refuses past 256 rather than wrapping." → "…; `kernel!` builds none since a-bound-is-a-value A5
  deleted the template hole it was for."
- **compiler.md:64-85** ("Program, template, instance"): **Is** "…and instantiating it with
  structural values gives a program." → "…; an entry has no structural values since
  a-bound-is-a-value A5, so its template is its program, and an entry that takes a kernel is
  instantiated with its arguments when it is called."; **Follows** "structural parameters and the
  shape key the program, … The host chooses a program by its structural parameters, for the font
  the glyph's piece count, and then…" → "the program's terms and the shape key it, … The host
  chooses a program (for the font, by the glyph's piece count, until a-bound-is-a-value B2) and
  then…"; **Lives** "Templates are `Staged` and the `Param` holes of
  `pixelflow-compiler/src/{emit,lower}.rs`." → "Templates are `Staged`
  (`pixelflow-compiler/src/emit.rs`)."
- **terminal.md:62-63** (Font programs Follows): "choosing the program is choosing by a structural
  parameter, the piece count, which is what structural means." → "choosing the program is
  choosing by the piece count, a host-side choice."
- **Not edited:** language.md:178 ("Everything that is not structural is a uniform" is JP's quote);
  language.md:135 ("A structural change recompiles…", true of the lattice's extent);
  terminal.md:205 (zoom level: the tile extent stays structural).

### 6.7 Code docs

All in §2's tables. `binding_times.rs`'s module doc (`:1-15`) becomes:

```
//! Records, binding times and `Args`
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.3, §1.4; B3).
//!
//! Every parameter is a **uniform**: an `f32` parameter is one, a record
//! parameter one per field, and a call's values are the kernel's arguments,
//! never its constants. So every call of an entry is one program — one
//! canonical key, one compiled region — and a program compiled once is rebound
//! per call from the entry's `Args` record, into a block the caller keeps; so
//! is a program the entry's kernel was composed into, beside kernels with no
//! arguments of their own. A count is a literal or a `usize` const of the
//! block (docs/language/2026-10-20-a-bound-is-a-value.md §1.2).
//!
//! What a record and a count *mean* is pinned against rustc in
//! `rustc_is_the_oracle.rs`; the refusals are `sema`'s and the parser's unit
//! tests. This file pins the binding: keys, positions and rebinding.
```

### 6.8 BACKLOG (`docs/BACKLOG.md:32`)

The plan-table row: **landed** gains ", A5 (`kernel!` has no structural parameters; the template
exemption with them, so A7 is empty)"; **next** becomes "A1's the-language-is-kernel rows after
#1323; **A6** (bounds as typed expressions; `n: usize`); B1–B5 after typed-arena T3 and item 2".
No other BACKLOG row is discharged: `grep -n -i "template\|Param\|structural\|const N"
docs/BACKLOG.md` finds only this row (and H3's "structurally hash-conses", unrelated). BACKLOG C8's
host-write half is A6's.

---

## 7. Order of work, and the grep-gated checklist

Never `cargo test --workspace`; `df -h /` before every build, stop under 2 GB. Never build from a
worktree. The `pixelflow-compiler` crate does not build between its first and last source edit, so
§2.1-§2.6 and §2.7 land before its first build.

0. **Bases** (unmodified tree): gates 1, 1b, 2, 2x and 5 with `TAG=a5-pre` (§4.2); rustdoc bases
   `cargo doc -p <crate> --no-deps --document-private-items 2>&1 | python3 -I
   /home/user/pixelflow/.wip/a5/rustdoc_norm.py > /home/user/pixelflow/.wip/a5/rustdoc-base.<crate>.txt`
   for `pixelflow-compiler`, `pixelflow-ir`, `pixelflow-codegen`; record `cargo test -p
   pixelflow-compiler`'s passed total (306).
1. **IR** (§2.7): `arena.rs`, `kernel.rs`, `fold.rs`. `cargo test -p pixelflow-ir`; `cargo check -p
   pixelflow-ir --no-default-features`.
2. **Compiler sources** (§2.1-§2.6, unit tests §5.1, §5.3): `ast.rs`, `parser.rs`, `symbol.rs`,
   `sema.rs`, `lower.rs`, `emit.rs`, `lib.rs`. `cargo test -p pixelflow-compiler --lib`.
3. **Compiler integration tests** (§5.1, §5.2). `PIXELFLOW_ISA=avx2 cargo test -p
   pixelflow-compiler` and `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler`.
4. **Other crates** (§2.8): `cargo test -p pixelflow-core` (its tests expand `kernel!`), `cargo test
   -p pixelflow-graphics --lib`, `cargo build -p pixelflow-codegen --examples`.
5. **Docs** (§6).
6. **Gates**: gates 1, 1b, 2, 2x, 5 with `TAG=a5-impl` (§4.2); `cargo +1.99.0 clippy --workspace
   --all-targets --all-features -- -D warnings`; `cargo fmt --all -- --check`; `bash
   scripts/check-intent.sh`, `check-doc-paths.sh`, `check-cfg-encapsulation.sh`,
   `check-emit-boundary.sh`, `check-bin-declarations.sh`; `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH
   bash scripts/check-feature-matrix.sh` (whole: it takes no crate filter); rustdoc for the three crates with no warning outside the
   bases (`… | python3 -I /home/user/pixelflow/.wip/a5/rustdoc_norm.py --compare
   /home/user/pixelflow/.wip/a5/rustdoc-base.<crate>.txt`; GONE lines are expected for the deleted
   intra-doc links); the test count (§5.4); the greps below.

**Greps** (from `/home/user/pixelflow`; each gives exactly what is stated at A5's head). They are
code blocks, not table cells, because a table cell's `\|` is a literal pipe to `grep -E` and the
gate would pass by matching nothing. Before trusting a grep that should find nothing, run it once on
the unmodified tree and see it find the names (it does at `7e9531e`).

G1, every deleted name (expected: only `docs/language/2026-10-20-a-bound-is-a-value.md`, at §0
:31-32, §2's row :231, the rewritten §3.4 sentence, and the A2a bullet :578, :585):

```sh
grep -rnE '\b(StructuralRange|instantiated_range|FoldRange|Holes|OPEN_RANGE|HOLE_STRIDE|ExpansionFold|structural_generics|structural_parameters|refuse_a_helpers_generics|register_structural|EntryContext|is_usize|known_usize)\b|Count::Structural|Bounds::Structural|SymbolKind::Structural|lower::Lowered|Lowered \{' \
  --include=*.rs --include=*.md --include=*.py --include=*.sh . \
  | grep -v '^./target' | grep -v '^./docs/archive/' | grep -v '^./.wip/'
```

G2, the word in the compiler crate (expected: nothing):

```sh
grep -rni 'structural' pixelflow-compiler/
```

G3, templates and instantiation in the compiler crate (expected: only `src/lib.rs`'s `macro_tier`
doc, "the template vocabulary", which is `Vocabulary::Templates`):

```sh
grep -rniE 'template|instantiat|\bholes?\b' pixelflow-compiler/
```

G4, the `Param` hole (expected: only `src/emit.rs`'s `node` panic arm):

```sh
grep -rnE 'push_param|ExprNode::Param' pixelflow-compiler/
```

G5, the step label in code and intents (expected: nothing):

```sh
grep -rnwE 'A5' --include=*.rs --include=intent.md . | grep -v '^./target' | grep -v '^./.wip/' | grep -v 0xA5A5
```

G6, strides set by the compiler or by `Kernel` (expected: nothing from either):

```sh
grep -rn 'stride' pixelflow-compiler/
grep -rn 'stride:' pixelflow-ir/src/kernel.rs
```

G7, every `FoldTerms` built (read each hit: `monoid`, `lo`, `hi`, and no `stride`):

```sh
grep -rn 'FoldTerms' --include=*.rs . | grep -v '^./target'
```

G8, a `const` generic or an instantiation anywhere a `kernel!` is written (expected: only the
parser's new test, in its `quote!` cases):

```sh
grep -rnE '<const [A-Za-z_]+: *usize|::<[0-9]+>|Args::<' pixelflow-compiler/ pixelflow-graphics/ pixelflow-core/
```

G9, the concept in code and intents (expected: nothing):

```sh
grep -rniE 'structural param|structural hole|structural count' --include=*.rs --include=intent.md . | grep -v '^./target' | grep -v '^./.wip/'
```

G10, the concept in live docs (read each hit; every one must be history under a banner, the bound
plan's own text, or the ontology's retired entry; expected files: the-language-is-kernel, the bound
plan, ontology `language.md` and `compiler.md`):

```sh
grep -rniE 'structural parameter|const N: usize>|rings::<' docs .claude CLAUDE.md | grep -v 'docs/archive/' | grep -v 'docs/results/'
```

---

## 8. Judgement calls, each with the alternative rejected

- **J1. The refusal is the parser's.** Every other generic is refused there, and the AST then has
  no field for what the language does not mean. Rejected: parsing generics into `FnItem` and
  refusing in sema (keeps a field whose only use is a refusal, and lets a stage hold what the
  language has no meaning for).
- **J2. One `refuse_generics` for every `fn` and `const`, choosing its message by the parameter's
  kind.** The entry/helper split existed because entries had structural parameters. Rejected:
  keeping `refuse_a_helpers_generics` and a third entry-only function (three spellings of one
  refusal).
- **J3. A `const` generic gets its own message naming §1.2 and the spelling a count has now**, as
  the parser names a collection's sections. Rejected: the bare "generics in a `kernel!` block",
  which tells someone with an old `<const N>` nothing about where counts went.
- **J4. The message cites a new `BOUND_PLAN` const.** A6's refusals (bound arithmetic, a coordinate
  as a bound, `ANY`/`ALL` with a value bound) cite the same plan. Rejected: spelling the path in the
  string (three copies by A6).
- **J5. A5 deletes the template exemption, and A7 is empty.** Its condition names a deleted field,
  and nothing else marks a template (§3). Rejected: keeping A7 by inventing a condition (an entry
  with a fold, say) so the exemption could be deleted later in its own CL: an exemption with no
  reason, added to be removed.
- **J6. `FoldTerms::stride` goes in A5.** Every remaining constructor sets 1, and its doc names A5
  as its end. Rejected: keeping the field documented "always 1" (a convention in a comment, which
  CLAUDE.md says something else will eventually break) or leaving it for a later IR CL (A5 is the
  CL that makes it dead). If review wants A5 compiler-only, this is the one step to drop; the
  fallback is the doc edit alone.
- **J7. `Count` collapses to `u64` and `Bounds` to `Range<u64>`.** Rejected: keeping `Count` with
  one variant for A6 to reuse: A6's second variant is a leaf name, not a token expression rustc
  evaluates, a different denotation (design §10).
- **J8. `EntryContext` and `Lowered` are deleted; `node` takes no context.** After A5 each holds one
  thing (the arguments; the arena and root). Rejected: keeping the wrappers for a future field
  (A6 needs none, design §10).
- **J9. `Site::open_fold` takes `Range<u64>` and is infallible.** Its only error was `Holes::stride`.
  Rejected: `[Bound<Self::Term>; 2]` now, which would need `Staged` code for a `Value` arm nothing
  produces until A6.
- **J10. Rewritten tests use `usize` consts, not literals.** A const keeps the pins on a count read
  as a value (`N as f32`) and in a range's const arithmetic, at expansion and in `Staged`, and in a
  `#[no_implicit_prelude]` module. Rejected: literals, which lose those paths.
- **J11. Tests whose subject is the template are deleted, not rewritten** (holes, instantiation,
  `__lo`/`__hi` hygiene, `<1>` vs `<4>`, `ramp::<4>` ≠ `ramp::<5>`). Rejected: const-spelled
  copies, which would duplicate `a_range_from_usize_consts…` and `a_range_from_consts_folds…`.
- **J12. The exemption's pin is kept as its opposite** (`every_entry_that_takes_no_kernel_is_offered…`).
  Rejected: deleting it (an exemption by shape could return with no test failing).
- **J13. `Staged::fold_terms` drops its `let __range` block.** It served the instantiated arm only.
  Rejected: keeping the block (tokens with no reason). No production `Staged` entry has a fold.
- **J14. `node`'s `Param` arm panics by name**, as its `Buffer`, `Ref` and `Write` arms do.
  Rejected: an `unreachable!` (says less) or emitting a `push_param` (an unbound slot at bake is a
  bug).
- **J15. The-language-is-kernel gets one banner sentence, the §1.3 row and the §1.4 note**, not a
  rewrite of every passage. The document is a plan of record under a reversal banner. Rejected:
  rewriting §1.1, §1.5, §1.7 and "What the macro emits" (history rewritten as if decided
  otherwise).
- **J16. `program.rs:15-16`, the codegen `Param` panic and ontology `terminal.md:62-63` lose the
  words "structural parameter" now.** After A5 the term names nothing in the language. Rejected:
  leaving them until B2/B4 (prose naming a concept the language no longer has).
