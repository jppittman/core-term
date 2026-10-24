# A6 spec: `kernel!` bounds are typed expressions

**Base:** `29980bf` (main after A5, #1449). HEAD is `bc1c011`, which adds only `.wip/`
(`git diff --quiet 29980bf HEAD -- . ':!.wip'` holds at writing). Every file:line below is at
`29980bf`. Re-grep before trusting a line number; §8's greps are the authority.
**Plan:** `docs/language/2026-10-20-a-bound-is-a-value.md` ("plan"): §1.1 (plan:43-63), §1.2
(plan:65-80), §1.4 (plan:118-154, the domain and the leaf rule), L2 (plan:174-179), L3 (plan:180-190),
L5 (plan:192-193), L9 (plan:214-216), §3.1's `push_reduce` refusals (plan:263-276), §3.4 (plan:407-420),
§6 A6 (plan:709-712), §8 decision 2 (plan:769-771, decided: an `n: usize` entry parameter), decision 7
(plan:783-785, approved: only the items it lists), decision 8 (plan:786-788, `ANY`/`ALL` keep refusing a
value bound).
**Design:** `/home/user/pixelflow/.wip/a6/design.md` ("design"). Its last section, "Carried from A5's
rulings (R-A6)" (design:378-398), overrides the design where they differ; this spec follows R-A6
(a6fit 1, 2, 6, 7) throughout.
**Standard:** A5's record, `/home/user/scratch/a5-wip-archive/.wip/a5/{spec,rulings}.md`.
**Evidence:** `/home/user/scratch/a5-probe/tests/a6_macro_tier.rs` (design:10-15): at `7e9531e` the
macro tier's optimizer (`Saturate::macro_tier()`, what `kernel!` runs) took a fold with a uniform `hi`
(n ∈ {0, 1, 3, 7}) and a triangular fold, answered `Changed`, and the baked values were rustc's. A5
changed nothing in `pixelflow-search`; step 0 (§8) re-runs the probe at `29980bf` before any edit.

Every decision an implementer would otherwise make is made here, and §9 lists each with the
alternative rejected, except **D-A6-1**, which adds a public item to `pixelflow-core` and so goes to JP
(§4.4). The spec writes both of its arms in full behind one switch (§4.5) and recommends (a). Nothing
else waits on a ruling.

---

## 0. What A6 is, and what it is not

**A6 is** (plan:409-416, plan:709-712, decision 2):

1. **A bound is a `usize` typed in the enclosing scope.** sema types every name in a range's bound
   before the fold's own scope opens: a `usize` const, an enclosing fold's index, or an entry's count
   parameter. `X`/`Y` are refused by L3, any other name that is not a `usize` by type (R-A6 a6fit 2).
   Only when every name is a `usize` is arithmetic judged, and arithmetic on a name the program binds
   is refused (`BoundArithmetic`, plan §1.4, until T6). A value bound under `any`/`all` is refused
   (decision 8). A known pair keeps its expansion-time checks (reversed, past 2²⁴) (§2).
2. **`n: usize` is an entry parameter of the items form** (decision 2; R-A6 a6fit 1): one uniform,
   declared in parameter order with `n as f32` as its default, read as a fold's bound or as `n as f32`.
   A helper's `usize` parameter (D-A6-2) and the closure form's (D-A6-3) are refused (§2.4).
3. **Lowering resolves both bounds before `open_fold`** and hands each to the IR as `Bound::Lit(k)` or
   `Bound::Value(term)` through `FoldTerms` (plan:414-415), at both `Site`s (§3).
4. **The host refuses a count past 2²⁴ where it writes one** (plan:133-138): the entry's host function
   panics (D-A6-4), and `Args::write_into` refuses before writing anything (D-A6-1) (§4).
5. Tests (§6) and docs (§7).

**A6 is not:**
- **B1–B5.** No discrete-kernel argument, `Extent` leaf, `pieces.map(…)` surface (decision 1),
  `FontPrograms` deletion or glyph-as-a-call. Production's glyph path is untouched.
- **T6.** Bounds stay `Real`-sorted: `BoundArithmetic` stays, and so does the guard's integrality
  conjunct.
- **Decision 8.** `any`/`all` keep refusing a value bound, at every stage.
- **A count in a helper's or the closure form's parameters** (D-A6-2, D-A6-3, §2.4).
- **The lattice's own extent** (plan §4, "Deferred").
- **A change to `pixelflow-ir`, `-search` or `-codegen` code.** Value bounds have been the IR's since
  A3 and codegen's since A4. A6's one IR edit is a doc (`fold.rs:440-443`, §7.8).
- **A production byte change** (§5.3).
- **Specialization.** Nothing compiles a program per count (plan §7).

---

## 1. Semantics

### 1.1 The laws A6 is held to

- **Denotation** (plan §1.1): `⟦Reduce{⊕, b, lo, hi, body}⟧ = ⊕_{i ∈ [⟦lo⟧, ⟦hi⟧)} ⟦body⟧[b := i]`, and
  `kernel!`'s `(lo..hi).map(|i| e).sum()` denotes exactly that for every spelling below. Rust's
  iterator over the same tokens is the oracle (`rustc_is_the_oracle.rs`).
- **L2 (empty).** A known pair is refused if it runs backwards (the literal spelling, as today). A pair
  with a value that runs backwards at run time is the monoid's identity, by the guard's `run` test,
  as Rust's `5..3` is empty.
- **L3 (legality).** No bound reads `X`, `Y` or its own fold's index. In `kernel!` this holds by type
  and by scope: `X`/`Y` are `f32` and nothing converts an `f32` to a `usize` (`parser.rs:1188-1198`
  refuses `as usize`), and a bound is typed before its fold's index is in scope. `push_reduce`'s
  `BoundVaries` (`arena.rs:1013-1016`) is behind both.
- **L5 (domain).** A count is an integer in `0..=2²⁴`. Outside it a `Real`-sorted fold is NaN and never
  hangs: the guard `If(dom, If(Lt(lo, hi), R, identity), NaN)` (`passes/guard.rs:1-36`), added at bake
  by `legalize`'s last pass (`passes.rs:129-131`).
- **L9 (identity).** A uniform bound contributes a marker and a dense slot to the key, not a value, so
  one program serves every `n`.
- **The leaf rule** (plan:140-154). A value bound in `kernel!` is a single name: a `Uniform` leaf or a
  binder `Var`. Arithmetic on one is refused at expansion until T6.

### 1.2 Per spelling

| Spelling | sema | Lowering → IR | Variance, placement (plan §1.2) | Key | Refused, where |
|---|---|---|---|---|---|
| literal `2..6` | `Count::Known` pair | `Bound::Lit` pair (`literal_range`) | node data, a pool constant | header | reversed: expansion (sema, `range_bounds`); past 2²⁴: expansion (lowering, `literal_range`) |
| `usize` const expression `LO..LO + N` | evaluated, each operation checked, `Count::Known` | `Bound::Lit` | node data | header | overflow, ÷0, `-`, a float literal: expansion (sema); then as a literal |
| enclosing index `0..i`, `i..4` | `i` typed `Index`/`Usize` in the enclosing scope; `Count::Name` | `Bound::Value(index)`: the enclosing fold's placeholder `Var`, renamed to its binder when that fold closes (`arena.rs:1758-1765`) | the enclosing binder's bit (L4): the inner fold runs once per outer trip, in the outer fold's scope | marker + child key | `0..i + 1`: expansion (`BoundArithmetic`); under `any`/`all`: expansion (decision 8); a literal end past 2²⁴: expansion (lowering) |
| count parameter `0..n`, `n..8`, `lo..hi` | `n` typed `Parameter`/`Usize`; `Count::Name` | `Bound::Value(uniform)`: the `Uniform` leaf `lower_entry` declared | `CONST`: once per call, `Scope::Body` | marker + dense slot: one program for every `n` (L9) | as for an index, plus §1.3 for its value |

A pair may mix a literal and a value (`5..n`, `n..8`); the literal is checked alone against 2²⁴ at
lowering (§3.4), and whether the pair runs backwards is decided per call (L2).

### 1.3 Per value class of a count parameter `n`

| `n` (the `usize` the host holds) | Host function `f(n)` | `FArgs { n }.write_into(&mut block)` | A raw write `block.set_declared([v])` | What the program computes |
|---|---|---|---|---|
| `0` | builds the kernel, default `0.0` | writes `0.0` | `0.0` | `0..0` is empty: the monoid's identity (L2), `+0.0` for a sum where rustc's is `-0.0`, which `==` cannot tell apart |
| `1..=2²⁴` | builds it, default `n as f32`, exact | writes `n as f32`, exact | the same `f32` | `⊕` over `[lo, n)`; a pair running backwards per call is the identity (L2) |
| `2²⁴ + 1` | **panics** (D-A6-4) | **refuses, writes nothing** (D-A6-1) | not writable: `16777217_f32 == 16777216.0` (`value_bounds_jit.rs:454-457`), so the slot holds `2²⁴`, a count | the fold would run `2²⁴` trips, one short, with no NaN: why the host is the only place this value can be refused (plan:133-138) |
| `≥ 2²⁴ + 2` (e.g. `16777218`, `usize::MAX`) | panics | refuses, writes nothing | `16777218.0`, `1.8e19`: names no count | NaN; the loop is not entered (the guard's arm), so nothing hangs (L5) |
| — (only a raw write reaches these) | — | — | NaN, ±∞, `-1.0`, `2.5`, `1e30` | NaN, no hang (L5) |

A count parameter is a count by its type, read or not, a bound or not: `n as f32` past 2²⁴ would be
another count too (§9, J10).

### 1.4 Where each refusal is

| When | Stage | What |
|---|---|---|
| expansion | parser (unchanged) | a generic, `as usize` (no syntax converts to a count) |
| expansion | sema (§2) | a bound name that is a coordinate (L3) or not a `usize` (type); an unknown name; a shape no bound is built from; arithmetic on a value bound; a value bound under `any`/`all`; a known pair running backwards; a const evaluation error; a `usize` parameter of a helper or of the closure form; `-> usize`, `let n: usize` (unchanged) |
| expansion | lowering (§3), on its own, without relying on sema (`lower.rs:770-776`'s rule) | a known end past 2²⁴, alone or in a pair; a bound name that is not a count binding; a value bound under `any`/`all`; arithmetic on a value bound (through the shared `range_bounds`) |
| expansion | the IR, behind both | `push_reduce`'s `BoundSort` and `BoundVaries`, a proc-macro panic if ever reached |
| host write | emitted host function | a count past 2²⁴: panic |
| host write | emitted `write_into` | a count past 2²⁴: refused before anything is written (D-A6-1) |
| run time | the guard | a raw write naming no count: NaN; a pair running backwards: the identity |

---

## 2. Sema (`pixelflow-compiler/src/sema.rs`, `symbol.rs`)

### 2.1 The count: `Count`, `UsizeScope`, `RangeScope`, `range_bounds`

Add beside `CheckedOp` (`sema.rs:1631-1633`):

```rust
/// What a range's bound is (plan §3.4): a count evaluated here, or a name the
/// program binds — an enclosing fold's index or an entry's count parameter —
/// whose value is a leaf of the program, read per trip or per call (§1.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Count {
    /// Evaluated at expansion, each operation checked: an integer literal, a
    /// `usize` const, or `+ - * /` over them.
    Known(u64),
    /// A name that is no `const`. What it names is the scope's to say: `sema`
    /// has typed every name in the bound before it is evaluated, and lowering
    /// resolves it to the term it is bound to.
    Name(Ident),
}

impl Count {
    /// The name, when this bound is one.
    pub(crate) fn name(&self) -> Option<&Ident> {
        match self {
            Count::Known(_) => None,
            Count::Name(name) => Some(name),
        }
    }
}

impl From<u64> for Count {
    fn from(count: u64) -> Self {
        Count::Known(count)
    }
}
```

`UsizeScope` (`sema.rs:1635-1702`) gets an associated output type (R-A6 a6fit 6) and one method:

```rust
trait UsizeScope {
    /// What a `usize` expression is here: a number in a `const`'s
    /// initializer, a [`Count`] in a range's bound.
    type Output: From<u64>;

    /// The value of `name` in a `usize` expression, or why it has none.
    fn usize_named(&mut self, name: &Ident) -> syn::Result<Self::Output>;

    /// The refusal of something a `usize` expression is not built from.
    fn not_a_count(&self, span: Span) -> syn::Error;

    /// `count` as an operand of `binary`, one of `+ - * /`: its number, or
    /// the refusal of arithmetic on it.
    fn operand(&self, count: Self::Output, binary: &BinaryExpr) -> syn::Result<u64>;

    /// An integer literal, a name, `+ - * /` and parentheses.
    fn eval_usize(&mut self, expr: &Expr) -> syn::Result<Self::Output> {
        let binary = match expr {
            Expr::Literal(literal) => return literal.usize_value().map(Self::Output::from),
            Expr::Ident(ident) => return self.usize_named(&ident.name),
            Expr::Paren(inner) => return self.eval_usize(inner),
            // [the `Unary` arm, unchanged: :1656-1667]
            Expr::Binary(binary) => binary,
            other => return Err(self.not_a_count(other.span())),
        };
        let lhs = self.eval_usize(&binary.lhs)?;
        let rhs = self.eval_usize(&binary.rhs)?;
        let (checked, symbol): (CheckedOp, &str) = match binary.op {
            // [unchanged: :1673-1686, comparisons and bitwise → not_a_count]
        };
        // Arithmetic is judged after both sides are evaluated and the operator
        // is one a count takes, so a comparison is `not_a_count`'s, and a
        // `const`'s errors come in the order they always did.
        let lhs = self.operand(lhs, binary)?;
        let rhs = self.operand(rhs, binary)?;
        // [division by zero and overflow, unchanged: :1687-1700]
        checked(lhs, rhs).map(Self::Output::from).ok_or_else(|| /* unchanged */)
    }
}
```

Its doc (`:1635-1642`): replace the last two sentences with "A trait because its scopes differ in what a
name means: a `const` being evaluated may name another, evaluated on demand; a fold's bound names a
`usize` const, already evaluated, or a name the program binds, which stands alone ([`Count`])."

`ConstEvaluator` (`:1821-1839`) keeps A5's numbers:

```rust
impl UsizeScope for ConstEvaluator<'_> {
    type Output = u64;
    // `usize_named`, `not_a_count`: unchanged.

    /// A `const`'s operands are numbers already: every name in one is a
    /// `const`, evaluated on demand.
    fn operand(&self, count: u64, _binary: &BinaryExpr) -> syn::Result<u64> {
        Ok(count)
    }
}
```

`eval_f32`'s `Cast` arm (`:1853-1858`) and `value_of` (`:1924`) are unchanged: their `usize_named` /
`eval_usize` already return `u64`.

`RangeScope` (`:1734-1780`) stays consts-only (design §10; §9 J4) and yields `Count`:

```rust
impl UsizeScope for RangeScope<'_> {
    type Output = Count;

    fn usize_named(&mut self, name: &Ident) -> syn::Result<Count> {
        match self.consts.get(&name.to_string()) {
            Some(ConstValue::Usize(value)) => Ok(Count::Known(*value)),
            Some(ConstValue::F32(_)) => Err(/* unchanged: :1744-1751 */),
            // No `const` of that name: one the program binds. Whether it is a
            // count, and what it is bound to, is the caller's scope's to say.
            None => Ok(Count::Name(name.clone())),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        syn::Error::new(span, /* M5, §2.5 */)
    }

    fn operand(&self, count: Count, binary: &BinaryExpr) -> syn::Result<u64> {
        match count {
            Count::Known(value) => Ok(value),
            Count::Name(name) => Err(bound_arithmetic(&name, binary)),
        }
    }
}
```

`RangeScope`'s doc (`:1734`): "What a fold's bound may name besides the program's own bindings: a
`usize` const, already evaluated." `FnAnalyzer::range_scope`'s doc (`:876`): "The `const`s a fold's
bound is evaluated over; any other name in it is the program's (`Count::Name`)."

`range_bounds` (`:1704-1732`) returns `[Count; 2]`:

```rust
/// A fold's bounds: each a [`Count`], known at expansion — integer literals,
/// `usize` consts, `+ - * /` and parentheses — or a name the program binds,
/// which stands alone (§1.4 of the bound plan: until bounds are typed
/// integers, arithmetic on a value is `f32` arithmetic and can round into the
/// domain).
///
/// The one evaluation of them: `sema` checks a fold with it, and lowering
/// reads the bounds it returns, so the two cannot disagree on a range.
///
/// A known pair that runs backwards is refused rather than read as the empty
/// fold it would be in Rust: an empty range is written `a..a`, and `b..a` is
/// almost always a slip (clippy's `reversed_empty_ranges` is deny-by-default
/// for the same reason), and the IR's own `KnownRange::new` refuses one. A
/// pair with a value is decided per call: one that runs backwards then is the
/// identity, as Rust's `5..3` is (L2).
pub(crate) fn range_bounds(range: &RangeExpr, scope: RangeScope<'_>) -> syn::Result<[Count; 2]> {
    let mut scope = scope;
    let bounds = [scope.eval_usize(&range.lo)?, scope.eval_usize(&range.hi)?];
    match &bounds {
        [Count::Known(lo), Count::Known(hi)] if lo > hi => Err(/* unchanged: :1720-1729 */),
        _ => Ok(bounds),
    }
}
```

(The crate's `rust-version` is 1.85, so no let-chains; any form that compiles there is fine.) Drop
`use std::ops::Range;` (`:79`) if nothing else in sema.rs reads it (it is `range_bounds`' alone).

`bound_arithmetic`, a free function after `a_count_is_not_a_value` (`:1614`), returns M3 (§2.5) spanned at
`binary.span`, the operator.

### 2.2 Typing a bound (R-A6 a6fit 2)

`type_of_fold` (`:883-899`) becomes:

```rust
/// `(a..b).map(|i| e).sum()` and its siblings: each bound a `usize` typed in
/// the scope the fold is written in (plan §3.4), and the body a term of the
/// fold's monoid, typed in a scope of its own where the closure's parameter
/// is the index. The bounds are typed before that scope opens, so a fold's
/// own index is no name its bound can read (L3), and an enclosing fold's is.
/// The body sees every enclosing binding, as a Rust closure does.
fn type_of_fold(&mut self, fold: &FoldExpr) -> syn::Result<Ty> {
    self.type_of_bound(&fold.range.lo)?;
    self.type_of_bound(&fold.range.hi)?;
    let bounds = range_bounds(&fold.range, self.range_scope())?;
    refuse_a_value_bound_under_a_mask(fold, &bounds)?;
    self.refuse_shadowing_an_item(&fold.binder, "a fold's index")?;
    // [unchanged: :891-898]
}
```

New, in `impl FnAnalyzer` after `type_of_fold`:

```rust
/// A range's bound typed where the fold is written: every name in it is a
/// `usize` — a `usize` const, an enclosing fold's index, an entry's count
/// parameter — whatever arithmetic joins them, which [`range_bounds`] judges
/// after. So `0..X + 1` breaks L3, not the leaf rule. Only the shapes a bound
/// is built from are walked; any other expression names nothing here and is
/// [`range_bounds`]' to refuse.
fn type_of_bound(&self, bound: &Expr) -> syn::Result<()> {
    match bound {
        Expr::Ident(ident) => self.bound_named(&ident.name),
        Expr::Paren(inner) => self.type_of_bound(inner),
        Expr::Unary(unary) => self.type_of_bound(&unary.operand),
        Expr::Binary(binary) => {
            self.type_of_bound(&binary.lhs)?;
            self.type_of_bound(&binary.rhs)
        }
        Expr::Literal(_)
        | Expr::MethodCall(_)
        | Expr::Call(_)
        | Expr::If(_)
        | Expr::Fold(_)
        | Expr::Cast(_)
        | Expr::Field(_)
        | Expr::Block(_) => Ok(()),
    }
}

/// A name in a range's bound: a `usize`, or the refusal saying what it is.
fn bound_named(&self, name: &Ident) -> syn::Result<()> {
    let Some(symbol) = self.symbols.lookup(&name.to_string()) else {
        return Err(self.not_in_scope(name));
    };
    match (symbol.kind, symbol.ty) {
        (_, Ty::Usize) => Ok(()),
        (SymbolKind::Intrinsic, _) => Err(a_coordinate_bounds_no_fold(name)),
        (kind, ty) => Err(self.not_a_usize_bound(name, kind, ty)),
    }
}
```

`not_in_scope(&self, ident: &Ident) -> syn::Error` is `resolve_ident`'s tail (`:1040-1073`) moved
verbatim into a method; `resolve_ident` (`:1019-1074`) becomes a guard clause
`let Some(symbol) = self.symbols.lookup(&name) else { return Err(self.not_in_scope(ident)); };` then
its two checks and `Ok(symbol.ty)`, flat. No message changes.

`a_coordinate_bounds_no_fold(name)` (free function, M1) and `not_a_usize_bound(&self, name, kind, ty)`
(method, M2, `ty` named by `self.items.name_of(ty)`) are in §2.5.

What each name in a bound gets:

| Name | Symbol | Result |
|---|---|---|
| `X`, `Y` | `Intrinsic`, `F32` | M1, L3 |
| `N`, a `usize` const | `Const`, `Usize` | accepted; `range_bounds` reads `Count::Known` |
| `R`, an `f32` const | `Const`, `F32` | M2, help "declare the count as `const R: usize`" |
| `i`, an enclosing index | `Index`, `Usize` | accepted; `Count::Name` |
| the fold's own index | not in scope yet | "cannot find `i` in this kernel body" (`not_in_scope`) |
| `n: usize`, an entry's | `Parameter`, `Usize` | accepted; `Count::Name` |
| `r: f32`, a parameter | `Parameter`, `F32` | M2, help "a count that changes per call is an entry parameter `r: usize`" |
| `let n = 4.0` | `Local`, `F32` | M2 |
| a record, a kernel | `Parameter`/`Local`, `Record`/`Kernel` | M2 (found the record's name, `impl Fn(f32, f32) -> f32`) |
| unknown, a `fn`, `Z` | none | `not_in_scope`'s messages, unchanged |

### 2.3 Decision 8

A free function after `range_bounds`:

```rust
/// `any` and `all` take bounds known at expansion (decision 8 of the bound
/// plan): `push_reduce` refuses a value bound under `ANY` or `ALL`
/// (`BoundSort`), and no stage accepts what a later one refuses.
fn refuse_a_value_bound_under_a_mask(fold: &FoldExpr, bounds: &[Count; 2]) -> syn::Result<()> {
    let spelling = match fold.reduction {
        Reduction::Any => "any",
        Reduction::All => "all",
        Reduction::Sum | Reduction::Product | Reduction::Min | Reduction::Max => return Ok(()),
    };
    let Some(name) = bounds.iter().find_map(Count::name) else {
        return Ok(());
    };
    Err(syn::Error::new(name.span(), /* M4 with `spelling`, `name` */))
}
```

It reads the surface's own `Reduction`; the IR's `Monoid::takes_a_value_bound` is `pub(crate)`
(`fold.rs:121`), and making it public is a visibility change (§9 J5). Lowering refuses the same way on
its own (§3.4), and `push_reduce` is behind both.

### 2.4 `n: usize` as a parameter (decision 2; D-A6-2, D-A6-3)

`Items::param_type` (`:599-648`): before `let ty = Ty::of_a_value(…)` (`:636`), insert

```rust
if Ty::from_syn(&param.ty) == Some(Ty::Usize) {
    return self.count_parameter(param, role);
}
```

and add after it:

```rust
/// `n: usize`: a count that changes per call, an entry's in the items form
/// (decision 2 of the bound plan), whose host function and `Args` record
/// refuse a count past 2²⁴ where the host writes it (§1.4). A helper is
/// inlined and takes a count's value as an argument, and the closure form's
/// program is rebound by position with `f32`s, so neither takes one.
fn count_parameter(&self, param: &Param, role: Role) -> syn::Result<Ty> {
    match (role, self.def.spelling) {
        (Role::Entry, Spelling::Items) => Ok(Ty::Usize),
        (Role::Entry, Spelling::Closure) => Err(syn::Error::new_spanned(&param.ty, /* M7 */)),
        (Role::Helper, _) => Err(syn::Error::new_spanned(&param.ty, /* M6 */)),
    }
}
```

`Spelling` is already imported (`:72`). The closure sugar's one entry is `Role::Entry`
(`parser.rs:261`, `ast.rs:182-187`), which is why the spelling is read (R-A6 a6fit 1). `FnAnalyzer::new`
(`:686-692`) registers the parameter as `Symbol { Parameter, Usize }` through the unchanged
`register_parameter`; `resolve_ident` refuses it as a value (`a_count_is_not_a_value`, `:1035-1037`), and
`type_of_cast` accepts `n as f32` (`names_a_count`, `:906-913`). No other sema code changes for it.
`-> usize` and `let n: usize` stay refused by `of_a_value` (`:150-171`).

**`Parameter`, `ParameterKind`, `Scalar`** (R-A6 a6fit 7; replaces `:243-277`):

```rust
/// An entry's parameter as its program sees it (plan §1.4).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Parameter<'a> {
    pub name: &'a Ident,
    pub kind: ParameterKind<'a>,
}

/// What an entry's parameter is to its program: each is uniform scalars,
/// declared in parameter order.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ParameterKind<'a> {
    /// An `f32`: one uniform, the argument.
    Value,
    /// `n: usize`: one uniform holding `n as f32`, read as a fold's bound or
    /// by `n as f32`; the host refuses a count past 2²⁴ before writing it.
    Count,
    /// One of the block's records: one uniform per field, in field order.
    Record(RecordId, &'a [RecordField]),
}

/// One uniform scalar of an entry, as host code spells this call's value of
/// it: the default its declaration takes, and what `write_into` writes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Scalar<'a> {
    /// An `f32` parameter: `param`.
    Value(&'a Ident),
    /// A count parameter: `param as f32`.
    Count(&'a Ident),
    /// One field of a record parameter: `param.field`.
    Field(&'a Ident, &'a Ident),
}

impl<'a> Parameter<'a> {
    /// Its uniform scalars, in the order its program declares them: the
    /// parameter itself, or its record's fields in field order.
    pub fn scalars(self) -> impl Iterator<Item = Scalar<'a>> {
        let param = self.name;
        let (whole, fields): (Option<Scalar<'a>>, &'a [RecordField]) = match self.kind {
            ParameterKind::Value => (Some(Scalar::Value(param)), &[]),
            ParameterKind::Count => (Some(Scalar::Count(param)), &[]),
            ParameterKind::Record(_, fields) => (None, fields),
        };
        whole
            .into_iter()
            .chain(fields.iter().map(move |field| Scalar::Field(param, &field.name)))
    }
}
```

`AnalyzedKernel::parameters` (`:287-300`) maps each non-kernel parameter to
`Parameter { name: &param.name, kind: self.parameter_kind(param) }`, with

```rust
/// What `param`, one of an entry's, is to its program: by the block's
/// records first, then by its declared type (sema refused any other).
fn parameter_kind<'a>(&'a self, param: &Param) -> ParameterKind<'a> {
    if let Some(id) = self.def.record_named(&param.ty) {
        return ParameterKind::Record(id, self.def.record(id).fields.as_slice());
    }
    match Ty::from_syn(&param.ty) {
        Some(Ty::Usize) => ParameterKind::Count,
        _ => ParameterKind::Value,
    }
}
```

Its doc gains "a count one, `n as f32`". `Ty::from_syn` is the one classification of a declared type
(`:126-146`), so sema, lowering and emission cannot disagree about which parameters are counts; lowering's
unanalyzed tests (`lower.rs:888-912`) build an `AnalyzedKernel` without sema and still classify one.

### 2.5 Messages (every new refusal cites `BOUND_PLAN`)

`use crate::{BOUND_PLAN, PLAN};` replaces `use crate::PLAN;` (`sema.rs:68`). Texts, as `format!`
strings in the crate's style (header, blank line, notes):

- **M1, L3** (`a_coordinate_bounds_no_fold`, spanned at the name):
  `"mismatched types: expected `usize`, found `f32`\n\nnote: `{name}` is a coordinate, and a range's bound reads no coordinate: every sample runs the same loop, its trip count fixed before the lattice is traversed (L3 of {BOUND_PLAN})"`
- **M2, type** (`not_a_usize_bound`, at the name):
  `"mismatched types: expected `usize`, found `{found}`\n\nnote: `{name}` is a range's bound, and a bound is an integer literal, a `usize` const, an enclosing fold's index or an entry's count parameter `n: usize` (§3.4 of {BOUND_PLAN}){help}"`,
  `help` by kind: `Const` → `"\nhelp: declare the count as `const {name}: usize`"`; `Parameter` →
  `"\nhelp: a count that changes per call is an entry parameter `{name}: usize`"`; otherwise empty.
- **M3, `BoundArithmetic`** (`bound_arithmetic`, at the operator):
  `"arithmetic on `{name}` in a range's bound\n\nnote: a bound that is a value — an enclosing fold's index, or an entry's count parameter — stands alone: the program computes a bound in an `f32` lane, where a sum of two counts can round to another count and the fold run the wrong number of trips with no NaN to say so, until bounds are typed integers (§1.4 of {BOUND_PLAN})\nhelp: a count known at expansion is a `usize` const, which takes `+ - * /`; one that changes per call is computed by the host and passed as a parameter of its own"`
- **M4, decision 8** (at the name):
  `"`{spelling}` over a range bounded by `{name}`, a value\n\nnote: `any` and `all` take bounds known at expansion: outside a count's domain a fold is NaN, and a mask has no NaN to be (decision 8 of {BOUND_PLAN})"`
- **M5, not a count** (`RangeScope::not_a_count`, replaces `:1767-1779`):
  `"a range's bound is a count, and this is not one\n\nnote: a bound is an integer literal, a `usize` const, an enclosing fold's index or an entry's count parameter `n: usize`; literals and `usize` consts combine by `+ - * /` and parentheses (§3.4 of {BOUND_PLAN})"`
- **M6, a helper's count** (at the type):
  `"a helper's parameter is an `f32`, a `bool`, a record or a kernel\n\nnote: a count parameter `n: usize` is an entry's: the host writes its value per call and refuses one past 2^24 there (§1.4 of {BOUND_PLAN}); a helper is inlined, and takes a count's value as an argument, `n as f32`"`
- **M7, the closure's count** (at the type):
  `"the closure form's parameter is an `f32`\n\nnote: a count parameter `n: usize` is an entry's in the items form, whose host function and `Args` record refuse a count past 2^24 where the host writes it (§1.4 of {BOUND_PLAN}); the closure form's program is rebound by position, with `f32`s, which cannot\nhelp: write an entry: `kernel! {{ pub fn f(n: usize) -> f32 {{ … }} }}`"`
- **Unchanged texts:** the reversed known pair (`:1723-1727`), the `f32` const in `RangeScope`
  (`:1747-1749`; lowering's backstop now, since sema refuses one first with M2), unary `-` (`:1661-1664`),
  a float literal (`ast.rs:511-513`), overflow and ÷0 (`:1690`, `:1697`), `not_in_scope`'s.
- **Notes widened** to name the count parameter (each test substring they carry survives):
  - `of_a_value` (`:156-159`): "note: a `usize` is a count, not a value: a fold's index, a `usize`
    const or an entry's count parameter, which a body converts by `i as f32`".
  - `a_count_is_not_a_value` (`:1620`): "note: a `usize` (a fold's index, a `usize` const or an entry's
    count parameter) becomes a value…".
  - `type_of_cast` (`:934-935`): "note: a `usize` is a fold's index, a `usize` const or an entry's count
    parameter, converted by name: `i as f32`".
  - `parser.rs` `convert_cast` (`:1194-1195`): "a `usize` (a fold's index, a `usize` const or an entry's
    count parameter) becomes one only as `i as f32`".
  - `parser.rs` `generic_refusal` help (`:649-650`): append "; a count that changes per call is an entry
    parameter `n: usize`" (the test's substring "`const N: usize = 4;`" stays, `parser.rs:2744`).
  - `parser.rs` `generic_arguments_refusal` help (`:672`): append ", or take it as an entry parameter
    `n: usize`" (the substring at `parser.rs:2780` stays).

### 2.6 Text edits in sema.rs, symbol.rs, ast.rs, parser.rs

| File:line | Now | After |
|---|---|---|
| sema.rs:19-22 | "A third type, `usize`, is a count: a fold's index or a `usize` const." | "…a fold's index, a `usize` const or an entry's count parameter `n: usize`." |
| sema.rs:27-29 | "5. **Folds**: a fold's bounds are constant and run forwards ([`range_bounds`]), …" | "5. **Folds**: a fold's bounds are `usize`s typed where the fold is written — known at expansion, or a name the program binds, alone, reading no coordinate — and a known pair runs forwards ([`range_bounds`]); …" |
| sema.rs:30-32 | "every `f32` and record parameter of an entry is a uniform" | "every `f32`, count and record parameter of an entry is a uniform" |
| sema.rs:47-48 | rule 2 "an entry's is a uniform, one per field of a record" | "…a uniform, one per field of a record; a count parameter `n: usize` is one too" |
| sema.rs:90-94 | "A fold's index is one, and so is a `usize` const." | "A fold's index is one, and so are a `usize` const and an entry's count parameter." |
| sema.rs:114 | `/// A count: a fold's index or a `usize` const.` | `/// A count: a fold's index, a `usize` const or an entry's count parameter.` |
| sema.rs:552-554 | "An entry's parameters are its uniforms: `f32`s, and records of them." | "An entry's parameters are its uniforms: `f32`s, counts, and records of `f32`s." |
| sema.rs:1610-1613 | `a_count_is_not_a_value`'s doc | "A `usize` — a fold's index, a `usize` const or an entry's count parameter — is a value only as `i as f32`: …" |
| sema.rs:1893 | a duplicated doc line with stray text (`…first demand.impl ConstEvaluator<'_> {`) | deleted (the line below it is the doc); pre-existing, `057f7fe` |
| sema.rs:2043 | "// A bound is constant, so it calls nothing; a body may." | "// A bound calls nothing — it is literals and names — and a body may." |
| symbol.rs:11 | `\| Parameter \| Per call (uniform) \| `Uniform(slot)`, one per record field \| cx, p \|` | `\| Parameter \| Per call (uniform) \| `Uniform(slot)`, one per record field; a count's holds `n as f32` \| cx, p, n \|` |
| symbol.rs:37-42 | "An entry's `f32` and record parameters are its uniforms…" | append: "A count parameter `n: usize` is one uniform too, declared as `n as f32` and read as a fold's bound or by `n as f32`." |
| symbol.rs:179 | `/// Register a parameter symbol (e.g., `r: f32`).` | `/// Register a parameter symbol: `r: f32`, a record, or an entry's count `n: usize`.` |
| ast.rs:25 | `params: [(name, type), ...] // scalar and record parameters` | `… // scalar, count and record parameters` |
| ast.rs:232-233 | "(`f32`, a record or `impl Fn(f32, f32) -> f32`; `bool` in a helper)" | "(`f32`, a record, `usize` for an entry's count, or `impl Fn(f32, f32) -> f32`; `bool` in a helper)" |
| ast.rs:379 | "A fold over a constant range" | "A fold over a range" |
| ast.rs:624-626 | "The bounds are constant, evaluated by `sema` at expansion, and the binder is a `usize`, which a body makes a value of only by `i as f32`." | "Each bound is a `usize`, known at expansion or a name the program binds (an enclosing fold's index, an entry's count parameter), typed by `sema` where the fold is written; the binder is a `usize`, which a body makes a value of only by `i as f32`." |
| ast.rs:667, 678-679 | "a fold's index or a `usize` const" (twice) | "a fold's index, a `usize` const or an entry's count parameter" |
| parser.rs:26-27 | `type ::= 'f32' \| 'bool' \| RECORD -- an entry's `f32`s and records are its uniforms` | add a line `\| 'usize' -- an entry's count, in the items form: a uniform` |
| parser.rs:36-40 | `iexpr` comment "a `usize`: a range's bounds or a `usize` const's initializer…" | "a `usize` known at expansion: a `usize` const's initializer, or a range's bound…" (rest unchanged) |
| parser.rs:52-53 | "a `usize` (a fold's index or a `usize` const) as a value" | "a `usize` (a fold's index, a `usize` const or an entry's count) as a value" |
| parser.rs:67 | `range ::= '(' iexpr '..' iexpr ')' -- half-open, constant, forwards` | `range ::= '(' bound '..' bound ')' -- half-open` and a new production `bound ::= iexpr \| IDENT -- known at expansion; or an enclosing fold's index or an entry's count, alone (§1.4 of the bound plan)` |
| parser.rs:97-103 | "An entry's `f32` parameter is a **uniform**, a record one per field…" | "An entry's `f32` and count (`n: usize`) parameters are **uniforms**, a record one per field…" |
| parser.rs:163-164 | sema's list: "a range whose bounds are not constant or that runs backwards;" | "a range's bound that names a coordinate (L3) or anything but a `usize`, that is arithmetic on a value, or that is a value under `any` or `all`; a known range that runs backwards; a count parameter of a helper or of the closure form;" |
| parser.rs:175-177 | lowering's list: "a range bound past 2²⁴, …" | "a range bound past 2²⁴, alone or in a pair, …" |
| parser.rs:883-893 | "A bounded reduction is a fold over a constant range." / "a bounded reduction over a constant range is a fold" | drop "over a constant range" in both |
| parser.rs:1070 | "a fold iterates a constant range" | "a fold iterates a range" |
| parser.rs:1321-1322 | "Whether they are constant, and run forwards, is `sema`'s question" | "What they are, and whether a known pair runs forwards, is `sema`'s question" |

---

## 3. Lowering (`pixelflow-compiler/src/lower.rs`)

### 3.1 `Binding::Count`

`Binding::Index(T)` (`:193-195`) is renamed `Binding::Count(T)`:

```rust
/// A `usize`: a fold's index — its placeholder `Var` while the fold's body is
/// built — or an entry's count parameter — its `Uniform` leaf. A body reads
/// one only as `i as f32`, and a fold reads one as a bound.
Count(T),
```

One variant for one behaviour: every use matches the two alike (§9 J6). Rename at `:632`, `:649`, `:692`,
`:701`, `:781`. `resolve`'s arm (`:781-784`) says `"`{name}` is a `usize`, where a value is expected: `{name} as f32`"`.

### 3.2 `lower_entry` (`:393-400`)

```rust
for parameter in analyzed.parameters(entry) {
    let mut uniforms = parameter.scalars().map(|scalar| site.uniform(scalar));
    let binding = match parameter.kind {
        ParameterKind::Value => {
            Binding::Value(uniforms.next().expect("an `f32` parameter is one uniform"))
        }
        ParameterKind::Count => {
            Binding::Count(uniforms.next().expect("a count parameter is one uniform"))
        }
        ParameterKind::Record(record, _) => Binding::Record(record, uniforms.collect()),
    };
    locals.bind(parameter.name.to_string(), binding);
}
```

Import `ParameterKind` and `Count` from `crate::sema` (`:63`); `use crate::{BOUND_PLAN, PLAN};` replaces
`use crate::PLAN;` (`:58`).

### 3.3 `Site::open_fold` takes both bounds

`Site::open_fold` (`:173-176`):

```rust
/// Open a fold under `monoid` between `bounds`, `depth` folds being open
/// already: its index ([`ExprArena::open_fold`]). A value bound's term was
/// built before the fold opened, so it names the same node in the copy the
/// body is built in ([`ExprArena::close_fold`]). Lowering has refused a depth
/// past the binders and a literal past the index ([`literal_range`],
/// [`literal_bound`]).
fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: [Bound<Self::Term>; 2]) -> Self::Term;
```

`Expansion::open_fold` (`:334-347`):

```rust
fn open_fold(&mut self, depth: usize, monoid: Monoid, [lo, hi]: [Bound<ExprId>; 2]) -> ExprId {
    let open = self
        .arena
        .open_fold(depth)
        .expect("lowering refuses a fold nested past the binders before it opens one");
    let index = open.index();
    self.open.push((open, FoldTerms { monoid, lo, hi }));
    index
}
```

`Staged::open_fold` (`emit.rs:700-711`) passes `bounds` to `fold_terms` (§4.7). `close_fold` is unchanged
at both sites: `ExprArena::close_fold` (`arena.rs:1824-1838`) already takes a value bound's term built
before the fold opened ("so a front end can lower a bound before it opens its fold", `arena.rs:1815-1818`),
`close_over` discovers a literal and roots the values (`arena.rs:1742-1775`), renames only this fold's
placeholder, and `push_reduce` checks L3 and the sort.

### 3.4 `lower_fold` (`:602-638`)

```rust
/// `⊕_{i ∈ [lo, hi)} body` as one `Reduce` over its [`Fold`], built as
/// `Kernel::over` builds it, or `Kernel::over_values` for a bound that is a
/// value: the bounds are resolved first, in the scope the fold is written in,
/// so its own index is none of them (L3); the body is built against the
/// fold's index, a placeholder, and the fold is closed after it, by the IR's
/// one definition of how a binder is chosen and a placeholder renamed
/// ([`ExprArena::close_over`]).
fn lower_fold(&mut self, fold: &FoldExpr) -> Result<S::Term, String> {
    let scope = RangeScope {
        consts: &self.program.analyzed.consts,
    };
    let counts = range_bounds(&fold.range, scope).map_err(|e| e.to_string())?;
    let bounds = self.fold_bounds(fold.reduction, counts)?;
    // [the depth check, unchanged: :614-625]
    let index = self
        .site
        .open_fold(self.open_folds, monoid(fold.reduction), bounds);
    // [unchanged from :629, with `Binding::Count(index)` at :632]
}

/// A fold's bounds as the IR takes them: a known pair if
/// [`KnownRange::admits`] it; otherwise each end a literal checked alone, or
/// the term the name is bound to — and only a count binding is a bound. A
/// value bound under `any` or `all` is refused (decision 8). All of it
/// without relying on `sema`, which refuses the same first.
fn fold_bounds(&self, reduction: Reduction, [lo, hi]: [Count; 2]) -> Result<[Bound<S::Term>; 2], String> {
    if let (Count::Known(lo), Count::Known(hi)) = (&lo, &hi) {
        let range = literal_range(*lo..*hi)?;
        return Ok([Bound::Lit(range.start), Bound::Lit(range.end)]);
    }
    if matches!(reduction, Reduction::Any | Reduction::All) {
        return Err(format!(
            "`any` or `all` over a range with a bound that is a value: a mask has no NaN for a \
             count outside the domain (decision 8 of {BOUND_PLAN})"
        ));
    }
    Ok([self.bound(lo)?, self.bound(hi)?])
}

/// One end of a fold whose other end is a value: a literal, checked alone
/// against the domain, or the term a count binding holds.
fn bound(&self, count: Count) -> Result<Bound<S::Term>, String> {
    let name = match count {
        Count::Known(literal) => return literal_bound(literal).map(Bound::Lit),
        Count::Name(name) => name,
    };
    match self.frame.locals.lookup(&name.to_string()) {
        Some(Binding::Count(term)) => Ok(Bound::Value(term.clone())),
        _ => Err(format!(
            "`{name}` as a range's bound: a bound that is a value is an enclosing fold's index or \
             an entry's count parameter, a `usize`"
        )),
    }
}
```

and beside `literal_range` (`:222-238`, kept):

```rust
/// A known bound beside a value, if [`KnownRange::admits`] it alone: the
/// plan's predicate on one count (§1.4), asked so the refusal names the
/// bound before `FoldHeader::new` would panic on it.
fn literal_bound(bound: u64) -> Result<u64, String> {
    Some(bound)
        .filter(|&end| KnownRange::admits(&(0..end)))
        .ok_or_else(|| {
            format!(
                "the bound `{bound}` reaches past {} (2^24): a fold's index is an `f32` lane, which \
                 names every integer only that far (`Fold::EXACT_BOUND`)",
                Fold::EXACT_BOUND
            )
        })
}
```

The lookup reads the current frame only: an inlined helper's frame holds its own parameters and indices
(`inline`, `:734-764`), so a caller's count or index is no bound inside a helper, as sema has it.

### 3.5 Other lowering edits

- `lower_cast` (`:640-662`): the arm is `Binding::Count(term) => Ok(term.clone())`; its doc: "A fold's
  index is an `f32` lane already, and a count parameter's uniform holds `n as f32`, so either converts
  to its own term; a `usize` const is its value's `f32`, rounded as Rust's `as` rounds it."
- `record_named`, `kernel_named` (`:692`, `:701`): `Binding::Count(_)` in the `None` arms.
- Module doc (`:28-31`): "A fold lowers to one `Reduce` node over its [`Fold`], built as
  `Kernel::over` builds it — or `Kernel::over_values`, for a bound that is a value, resolved before the
  fold opens: the body against a placeholder, then closed over it." (`:33-38`): "An entry's `f32`, count
  and record parameters are its uniforms…; a count is declared as `n as f32`."

---

## 4. Emission and the host surface (`pixelflow-compiler/src/emit.rs`, `pixelflow-core`)

### 4.1 A scalar's value in host code

`argument` (`:161-169`):

```rust
/// What a scalar argument's value is in host code: `param`, `param.field`
/// for a record parameter's field, or `param as f32` for a count — the `f32`
/// its uniform holds, which the host has refused past 2²⁴ before this
/// conversion could make it another count. `write_into` reads each through
/// `self.`: `as` binds looser than a field access, so `self.n as f32` is
/// `(self.n) as f32`.
fn argument(scalar: Scalar<'_>) -> TokenStream {
    match scalar {
        Scalar::Value(param) => quote!(#param),
        Scalar::Count(param) => quote!(#param as f32),
        Scalar::Field(param, field) => quote!(#param.#field),
    }
}
```

Its three readers are unchanged: `entry_arena`'s `arguments` (`:138-143`, the defaults `arena_to_tokens`
declares), `Staged::uniform` (`:681-691`), `emit_args`' values (`:244-251`). For an `f32` or a field the
tokens are today's, so every existing expansion is unchanged token for token (§5.3).

### 4.2 The host function refuses a count past 2²⁴ (D-A6-4: a panic)

`emit_entry` (`:188-217`) takes the analysis, and the call at `:118` becomes
`items.extend(emit_entry(entry, analyzed, &body));`:

```rust
fn emit_entry(entry: &FnItem, analyzed: &AnalyzedKernel, body: &TokenStream) -> TokenStream {
    let attrs = &entry.attrs;
    let vis = &entry.vis;
    let name = &entry.name;
    let params = /* unchanged: :203-209 */;
    let counts = count_parameters(entry, analyzed);
    let refusals = counts.iter().map(|count| {
        let message = count_refusal(entry, count);
        quote! {
            ::core::assert!(
                ::pixelflow_core::__macro::ir::KnownRange::admits(&(0..#count as u64)),
                #message,
                #count
            );
        }
    });
    let panics = (!counts.is_empty()).then(|| {
        let text = format!(
            " A count parameter past {} (2^24), which its `f32` uniform would hold as another \
             count (§1.4 of {BOUND_PLAN}).",
            pixelflow_ir::Fold::EXACT_BOUND
        );
        quote! { #[doc = ""] #[doc = " # Panics"] #[doc = ""] #[doc = #text] }
    });
    quote! {
        #(#attrs)*
        #panics
        #[must_use]
        #vis fn #name ( #(#params),* ) -> ::pixelflow_core::Kernel {
            #(#refusals)*
            #body
        }
    }
}

/// An entry's count parameters, in declaration order.
fn count_parameters<'a>(entry: &'a FnItem, analyzed: &'a AnalyzedKernel) -> Vec<&'a Ident> {
    analyzed
        .parameters(entry)
        .into_iter()
        .filter(|parameter| matches!(parameter.kind, ParameterKind::Count))
        .map(|parameter| parameter.name)
        .collect()
}

/// The refusal of a count past 2²⁴ where the host writes it, as a format
/// string with one `{}`, the count: written as the `f32` its uniform holds,
/// `2²⁴ + 1` is `2²⁴`, a count, and the fold would run one trip short with no
/// NaN (plan §1.4), so it can be refused only here.
fn count_refusal(entry: &FnItem, param: &Ident) -> String {
    let bound = pixelflow_ir::Fold::EXACT_BOUND;
    format!(
        "kernel!: `{entry}`'s count `{param}` is {{}}, past {bound} (2^24): a count is an \
         integer in 0..={bound}, the integers a fold's `f32` index names exactly (§1.4 of \
         {BOUND_PLAN})",
        entry = entry.name.unraw(),
        param = param.unraw(),
    )
}
```

Imports: `use crate::BOUND_PLAN;`, `ParameterKind` beside `Scalar` from `crate::sema` (`:58`),
`use syn::ext::IdentExt;` (for `unraw`). `Fold` is spelled `pixelflow_ir::Fold::EXACT_BOUND` in both
functions, as `fold_terms` spells `pixelflow_ir::Fold::new` (`:606`); the test module keeps its own
`use pixelflow_ir::Fold;` (`:752`). Drop `use std::ops::Range;` (`:61`): after §4.7 nothing in emit.rs
reads it.

The check is the plan's "`KnownRange::admits` turned into a predicate on one count value" (plan:133-134),
public at the IR's root and reached through `__macro::ir`, so the predicate is not restated (§9 J9). Every
path is absolute (`::core::assert!`, `::pixelflow_core::…`), so a block expands where no prelude is
(`items_block.rs`). A `Staged` entry (one that takes a kernel) gets the same refusals: `emit_entry` is
shared. Entries without a count get no tokens from either addition (`Option<TokenStream>` interpolates
as nothing; an empty iterator likewise).

### 4.3 The `Args` record

The field is the declared type, unchanged code (`emit_args`' `fields`, `:236-243`): `pub n: usize`. The
values (`:244-251`) are `self.n as f32` for a count, by §4.1. `write_into` refuses a count past 2²⁴
**before** `set_declared`, so nothing is written, as an arity mismatch writes nothing
(`manifold.rs:251-256`). How it refuses is D-A6-1.

### 4.4 D-A6-1: how `write_into` refuses (needs JP: a public API decision)

| Option | `pixelflow-core` public API | Emitted `write_into` | Other effects |
|---|---|---|---|
| **(a) An error enum** | **adds one public item**, `WriteRefused { Arity(ArityMismatch), Count { parameter: &'static str, count: u64 } }` with `Display` and `Error`, re-exported at the crate root | every `Args` record: `-> Result<(), WriteRefused>`; a count past 2²⁴ is `Err(WriteRefused::Count { .. })`, an arity mismatch `Err(WriteRefused::Arity(..))`; nothing written either way | `binding_times.rs`'s arity pins (`:202-208`, `:225-232`) wrap their `ArityMismatch` in `WriteRefused::Arity`; `lib.rs`'s doc example (`:311`, `?`) is unchanged; ontology Args row; core `intent.md` Contract |
| **(b) A panic** | **adds nothing** | `-> Result<(), ArityMismatch>` unchanged; an entry with a count asserts before `set_declared`, and its `write_into` gains a `# Panics` doc | none outside the compiler |
| (c) A typed count | adds a `Count` newtype (core or IR) whose constructor refuses past 2²⁴ | `Args { n: Count }` | the host function's declared `n: usize` and its `Args` field differ, or the host function takes `Count` too, which is not decision 2's `n: usize` at the host, nor the plan's `rust_sum(n: usize, x: f32)` parity |

**`write_into` has no caller outside the compiler's tests and `lib.rs`'s doc**
(`grep -rn write_into --include=*.rs .`: `binding_times.rs`, `items_block.rs`, `lib.rs:311`). The one
production `Args` record, `OnePieceArgs` (`program.rs:118`), is never written through. So no option
breaks production; (a) changes a generated signature every `kernel!` block emits.

**Recommendation: (a).** `write_into`'s contract is already "a refusal is an `Err`, and nothing is
written" (`emit.rs:266-273`): an arity mismatch is refused that way because the pixels would otherwise be
plausible (`manifold.rs:152-163`), and a count of `2²⁴ + 1` written as `2²⁴` is the same kind of failure
(one trip short, plausible pixels). A per-call count is the caller's data, so on the per-call path it is a
value the caller can handle, and every write's refusals are one type. The host function still panics
(D-A6-4): it is the builder, as `Kernel::over` panics on a literal past 2²⁴. Rejected: (b), which has no
public cost but two failure modes on one call (an `Err` for arity, a panic for a count); (c), for the
reasons in its row. **(a) adds a public item to `pixelflow-core` beyond decision 7's list, so it needs
JP's approval. (b) is the arm to build if JP declines (a).**

### 4.5 The switch: both arms in full

Everything outside this table is the same under both arms.

| Site | (a) `WriteRefused` | (b) panic |
|---|---|---|
| `pixelflow-core/src/lattice/manifold.rs`, after `ArityMismatch` (`:164-182`) | add `WriteRefused` (code below) | nothing |
| `pixelflow-core/src/lib.rs:99-101` | add `WriteRefused` to the `lattice::manifold` re-export list (the same `use`, no new `cfg`) | nothing |
| `pixelflow-core/intent.md` Contract | add `WriteRefused` after `ArityMismatch`, reflowed (71 lines, budget 80) | nothing |
| `pixelflow-core` tests | `manifold.rs`'s test module (`:815`): `a_refused_write_says_why` (+1) | nothing |
| `emit_args`' `write_into` | code (a) below | code (b) below |
| `binding_times.rs` | §6.2 arm (a) | §6.2 arm (b) |
| docs | `lib.rs` `# Binding times` and the ontology Args row name `WriteRefused` | they say `write_into` panics |
| plan's Done line (§7.1) | "refuses one with `WriteRefused::Count`" | "panics on one" |

**(a), `pixelflow-core`:**

```rust
/// Why an entry's `Args` record wrote nothing into a block: `kernel!`'s
/// `write_into` (docs/language/2026-10-20-a-bound-is-a-value.md §1.4).
///
/// Each is a refusal of plausible pixels. Values bound by position to a
/// program of another arity shift every argument after the first difference;
/// a count past 2²⁴, written as the `f32` its uniform holds, is another count
/// — `2²⁴ + 1` is `2²⁴` — and the fold runs the wrong number of trips with no
/// NaN to say so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteRefused {
    /// The block's program declares another number of arguments.
    Arity(ArityMismatch),
    /// A count parameter's value is past `Fold::EXACT_BOUND` (2²⁴), the last
    /// integer a fold's `f32` index names exactly.
    Count {
        /// The parameter, as the entry declares it.
        parameter: &'static str,
        /// Its value.
        count: u64,
    },
}

impl core::fmt::Display for WriteRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Arity(mismatch) => mismatch.fmt(f),
            Self::Count { parameter, count } => write!(
                f,
                "the count `{parameter}` is {count}, past {} (2^24): a count is an integer a \
                 fold's `f32` index names exactly (docs/language/2026-10-20-a-bound-is-a-value.md \
                 §1.4)",
                pixelflow_ir::Fold::EXACT_BOUND
            ),
        }
    }
}

impl core::error::Error for WriteRefused {}
```

No `From<ArityMismatch>`: nothing needs it (the emitted code maps with `WriteRefused::Arity`).

The test, in `manifold.rs`'s test module:

```rust
/// A refused write says which refusal: an arity mismatch in its own words, a
/// count with its value and the domain it left.
#[test]
fn a_refused_write_says_why() {
    let arity = ArityMismatch { declared: 2, supplied: 3 };
    assert_eq!(WriteRefused::Arity(arity).to_string(), arity.to_string());
    let count = WriteRefused::Count { parameter: "n", count: 16_777_217 }.to_string();
    assert!(
        count.contains("`n` is 16777217, past 16777216 (2^24)")
            && count.contains("docs/language/2026-10-20-a-bound-is-a-value.md"),
        "{count}"
    );
}
```

**(a), `emit_args`' `write_into`** (replaces `:259-280`):

```rust
let refusals = count_parameters(entry, analyzed).into_iter().map(|count| {
    let parameter = count.unraw().to_string();
    quote! {
        if !::pixelflow_core::__macro::ir::KnownRange::admits(&(0..self.#count as u64)) {
            return ::core::result::Result::Err(::pixelflow_core::WriteRefused::Count {
                parameter: #parameter,
                count: self.#count as u64,
            });
        }
    }
});
quote! {
    // [the struct, unchanged: :253-257]
    impl #args {
        /// This call's values, written into `block` — [unchanged first paragraph, :260-264]
        ///
        /// # Errors
        ///
        /// `WriteRefused::Count` when a count parameter is past 2²⁴, where the
        /// `f32` its uniform holds would be another count; `WriteRefused::Arity`
        /// when `block`'s program declares a different number of arguments —
        /// [the rest of :268-273, unchanged]. Nothing is written either way.
        #vis fn write_into(
            &self,
            block: &mut ::pixelflow_core::UniformBlock,
        ) -> ::core::result::Result<(), ::pixelflow_core::WriteRefused> {
            #(#refusals)*
            block
                .set_declared([ #(#values),* ])
                .map_err(::pixelflow_core::WriteRefused::Arity)
        }
    }
}
```

**(b), `emit_args`' `write_into`:** the signature and `# Errors` stay (`:266-279`); before
`block.set_declared(…)`:

```rust
let refusals = count_parameters(entry, analyzed).into_iter().map(|count| {
    let message = count_refusal(entry, count);
    quote! {
        ::core::assert!(
            ::pixelflow_core::__macro::ir::KnownRange::admits(&(0..self.#count as u64)),
            #message,
            self.#count
        );
    }
});
```

and, when the entry has a count, `/// # Panics` / `/// When a count parameter is past 2²⁴: nothing is
written.` after `# Errors` (emitted as a conditional `#[doc]` block, as §4.2's).

### 4.6 D-A6-4: the host function panics (decided)

A panic, as §4.2 emits. The host function is the program's builder, and builders panic on a count past the
domain: `Kernel::over` on a literal range (`KnownRange::new`, `fold.rs:1195-1200`), `over_values` on a
constant naming no count (`kernel.rs:840-844`). Rejected: `-> Result<Kernel, _>` for an entry with a count,
which gives count-taking entries a different signature from every other entry and makes composition
(`sum2(&a, &sum_to(n))`) unwrap at each call. No public API changes: the generated signature is the
declared one, `-> Kernel`, with a `# Panics` doc.

### 4.7 `Staged`: a fold whose bound is a value

`fold_terms` (`:599-622`):

```rust
/// The [`FoldTerms`](pixelflow_ir::arena::FoldTerms) an open fold closes
/// over: its monoid, carried as a header's bits (as [`node`] carries a
/// `FoldHeader`), and its bounds: a literal, or the variable holding a value
/// bound's term, built before the fold opened. The binder is the one
/// `close_fold` chooses.
fn fold_terms(monoid: Monoid, bounds: [Bound<Ident>; 2]) -> TokenStream {
    // [`first`, `bits`: unchanged, :604-610]
    let [lo, hi] = bounds.map(|bound| match bound {
        Bound::Lit(count) => quote!(::pixelflow_core::__macro::ir::fold::Bound::Lit(#count)),
        Bound::Value(term) => quote!(::pixelflow_core::__macro::ir::fold::Bound::Value(#term)),
    });
    quote! {
        ::pixelflow_core::__macro::ir::arena::FoldTerms {
            monoid: /* unchanged :614-617 */,
            lo: #lo,
            hi: #hi,
        }
    }
}
```

A literal end interpolates a `u64` exactly as `range.start` did (`0u64`), so a `Staged` fold over a
literal range emits today's tokens. A value end names a `__t…` variable — the count's
`__arena.push_uniform(__u…)` or an enclosing fold's `__f….index()` — bound before the fold's
`__arena.open_fold(…)` statement, and valid in the copy that statement makes (`arena.rs:1788-1791`).
`Staged::open_fold` (`:700-711`) takes `bounds: [Bound<Ident>; 2]` and pushes
`(fold.clone(), Self::fold_terms(monoid, bounds))`.

### 4.8 Text edits in emit.rs and lib.rs

| File:line | After |
|---|---|
| emit.rs:37-45 | "A host function takes each `f32`, count and record parameter by its declared type, and a kernel-typed one as a `&Kernel`; every `f32`, count and record one is a uniform: … with the call's value as its default (a count's as `n as f32`, refused past 2²⁴ first). …" |
| emit.rs:75-79 | append to the items-form sentence: "refusing a count parameter past 2²⁴ before anything is built" |
| emit.rs:180-181 | "Every parameter is an `f32`: the closure sugar has no records, `sema` refuses a `bool` parameter of an entry, and a count is an entry's of the items form." |
| emit.rs:188-198 | `emit_entry`'s doc: append "An entry with a count parameter refuses one past 2²⁴ before building anything, and says so in a `# Panics` section." |
| emit.rs:219-224 | `emit_args`' doc: "…its parameters as fields, by their declared types (a count's is `usize`), and `write_into`, which refuses a count past 2²⁴ and then writes them…" |
| lib.rs:75-76 | `BOUND_PLAN`'s doc: "The plan that made a fold's bound a value: a refusal about a count or a bound names it." |
| lib.rs:157-168 | `# Folds` (§7.6) |
| lib.rs:187-199 | `# The closure form`: after "every parameter a uniform, as an entry's are": "A count parameter, `n: usize`, is an entry's of the items form: the closure form takes `f32`s." |
| lib.rs:254-281 | `# Binding times` (§7.6) |

---

## 5. The macro tier

### 5.1 Saturation at expansion takes value bounds

An entry that takes no kernel is offered to `macro_tier()` (`lib.rs:368-370`) at expansion, whatever its
folds (A5 removed the only exemption). `Saturate::optimize` (`saturate_pass.rs:71-…`) inserts, saturates
and extracts; the probe (design:10-15) shows it takes a uniform `hi` and a triangular bound and keeps
their meaning. What A6 relies on, by code:

- A count's `Uniform` leaf inserts as `ENode::Uniform(decl)` (`egraph/insert.rs:138`), keyed by the
  declaration's bits (`arena.rs:338-362`), never folded: the expansion's NaN default (`UNBOUND`,
  `lower.rs:240-244`) is read by nothing.
- A value bound is unknown to the rules (L6): no peel, halve or empty-fold fires on it; `KnownBound`
  (L8) may make a triangular bound known after the outer fold is peeled, which only removes a value
  bound. `HalveFold` on an outer known fold can make an inner `0..(i + s)`: rule-made, exact, accepted by
  `push_reduce` when the emitted code rebuilds it at load time (plan:146-152).
- `relink` (`emit.rs:148-154`) restores the declaration order; a count read only as a bound is reached
  through its fold's children, so it is kept.
- **The loud hole** (A4, BACKLOG E8): extraction picks one member per class, and a bound class whose
  chosen member reads a coordinate is refused at materialization with `BoundVaries`. At expansion that is
  a proc-macro panic naming the law, at the `kernel!` call. No rule today makes a member reading `X` in a
  `Uniform`'s class or a binder `Var`'s; if a test hits one, report it: it is E8, not A6's to constrain.

### 5.2 The guard is added at bake, not at expansion

`macro_tier` runs no `legalize`: `Saturate::optimize` is insert, run, `to_arena` (`saturate_pass.rs:71-…`).
An expanded entry therefore carries its value-bounded folds unguarded, as `Kernel::over_values` builds
them, and the guard is added when the kernel is compiled: `legalize`'s last pass (`passes.rs:129-131`),
which `jit_cache::compile` runs for `Lattice::bake`, `Lattice::eval_at` and `Manifold::compile` alike.
`kernel_raw!` emits the lowered arena; its kernels are guarded at the same point.

### 5.3 Byte identity for production kernels

No production `kernel!` entry has a count or a fold, and nothing else changes an entry's tokens:

- **The one production block** is `pixelflow-graphics/src/fonts/loop_blinn/program.rs:31-141`: `one_piece(p:
  Row)` and `blank()` through `entry_arena`, `sum2` and `glyph` `Staged`. `sed -n 31,141p … | grep -cE
  'usize|\.map\(|\.any\(|\.all\('` is 0 (G6). No other crate, example or bench declares a `kernel!`
  entry with a `usize` (it was refused until A6), so none has a value bound.
- **Expansion entries without a count:** `analyzed.parameters` classifies their parameters as `Value` or
  `Record` exactly as `record: None`/`Some` did; `argument` gives today's tokens; `emit_entry` adds no
  refusal and no doc; a literal pair lowers to the same `FoldTerms { monoid, Lit, Lit }`. So the arena
  `macro_tier` is offered is node-identical, and so is what it returns and what `arena_to_tokens` emits.
- **`Staged` entries:** none in production has a fold, and a literal fold's tokens are today's (§4.7).
- **Host code:** under (a), every `Args::write_into`'s signature and body change (`OnePieceArgs` among
  them). That is host code, not a kernel: no arena, key, digest or saturation moves.
- **`byte_probe`** builds with the builders (`pixelflow-codegen` has no `pixelflow-compiler` dependency).

So gates 1, 1b, 2 and 2x are identical to base (`.wip/gates/README.md`, tag `a6-main` verified at
`29980bf`). Gate 5's workload (`zz_prof_all_cold`, the 29 font programs) runs no expansion: expected
within noise; report, never assert.

---

## 6. Tests

Every new value comparison is exact on every tier: sums of small integers times quarters
(`FOLD_SAMPLES`, `rustc_is_the_oracle.rs:335`), or the same program rebound. NEON runs in CI's macOS job.
Some new entries' `Args` records go unread in their file, as `AffineLessArgs` does today
(`rustc_is_the_oracle.rs:595-610`, no reader in `tests/`), which clippy `-D warnings` already passes.

### 6.1 `rustc_is_the_oracle.rs` (the plan's gate)

After `a_count_in_a_ranges_const_arithmetic_folds_as_rustcs_does` (`:692-710`), a section:

```rust
// ─────────────── values as bounds ───────────────
//
// A fold's bound may be a value (docs/language/2026-10-20-a-bound-is-a-value.md
// §3.4): an enclosing fold's index — a triangular `0..i` — or an entry's count
// parameter `n: usize`, bound per call. Rust's iterator over the same tokens is
// the oracle, the count a `usize` argument of the host `fn`.

kernel! {
    /// Σ_{i<4} Σ_{j<i} (X·j + i): an inner fold bounded by the outer index.
    pub fn triangle() -> f32 {
        (0..4).map(|i| (0..i).map(|j| X * (j as f32) + (i as f32)).sum::<f32>()).sum()
    }

    /// Σ_{i<n} X·i, the count a per-call value.
    pub fn sum_to(n: usize) -> f32 {
        (0..n).map(|i| X * (i as f32)).sum()
    }

    /// Σ_{i<n} Σ_{j<i} (X·j + i): a count bounding the fold an index bounds.
    pub fn triangle_to(n: usize) -> f32 {
        (0..n).map(|i| (0..i).map(|j| X * (j as f32) + (i as f32)).sum::<f32>()).sum()
    }

    /// Σ_{n≤i<8} X·i: a count at the first end.
    pub fn sum_from(n: usize) -> f32 {
        (n..8).map(|i| X * (i as f32)).sum()
    }

    /// Σ_{lo≤i<hi} (X·i + Y): two counts, a pair that may run backwards.
    pub fn sum_between(lo: usize, hi: usize) -> f32 {
        (lo..hi).map(|i| X * (i as f32) + Y).sum()
    }
}

fn rust_triangle(x: f32) -> f32 {
    (0..4).map(|i| (0..i).map(|j| x * (j as f32) + (i as f32)).sum::<f32>()).sum()
}

fn rust_sum(n: usize, x: f32) -> f32 {
    (0..n).map(|i| x * (i as f32)).sum()
}

fn rust_triangle_to(n: usize, x: f32) -> f32 {
    (0..n).map(|i| (0..i).map(|j| x * (j as f32) + (i as f32)).sum::<f32>()).sum()
}

fn rust_sum_from(n: usize, x: f32) -> f32 {
    (n..8).map(|i| x * (i as f32)).sum()
}

fn rust_sum_between(lo: usize, hi: usize, x: f32, y: f32) -> f32 {
    (lo..hi).map(|i| x * (i as f32) + y).sum()
}

/// A fold bounded by the index of the fold around it, `Σ_i Σ_{j<i}`, means
/// what the same iterator means; every sum is exact.
#[test]
fn a_triangular_fold_folds_as_rustcs_does() {
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&triangle(), x, y), rust_triangle(x), "at ({x}, {y})");
    }
    // X = 3: 0 + 1 + (2 + 5) + (3 + 6 + 9).
    assert_eq!(bake(&triangle()), 26.0);
}

/// A count passed per call bounds a fold as Rust's `0..n` does, at every
/// count — the empty one too, whose `+0.0` against rustc's `-0.0` `==` cannot
/// see (`a_fold_over_an_empty_range_is_the_identity_as_in_rustc`) — and bounds
/// the outer fold of a triangle.
#[test]
fn a_per_call_count_folds_as_rustcs_does() {
    for n in [0, 1, 3, 7] {
        for (x, y) in FOLD_SAMPLES {
            assert_eq!(Lattice::eval_at(&sum_to(n), x, y), rust_sum(n, x), "n = {n}, at ({x}, {y})");
            assert_eq!(
                Lattice::eval_at(&triangle_to(n), x, y),
                rust_triangle_to(n, x),
                "n = {n}, at ({x}, {y})"
            );
        }
    }
    // X = 3: 3 · (0 + 1 + … + 6).
    assert_eq!(bake(&sum_to(7)), 63.0);
}

/// Counts at either end, and a pair that runs backwards per call: Rust's
/// `5..2` is empty, and so is the fold, its monoid's identity (L2).
#[test]
fn a_range_between_counts_folds_as_rustcs_does() {
    for n in [0, 3, 8, 10] {
        for (x, y) in FOLD_SAMPLES {
            assert_eq!(Lattice::eval_at(&sum_from(n), x, y), rust_sum_from(n, x), "n = {n}, at ({x}, {y})");
        }
    }
    for (lo, hi) in [(0, 0), (2, 5), (5, 2), (3, 3), (0, 7)] {
        for (x, y) in FOLD_SAMPLES {
            assert_eq!(
                Lattice::eval_at(&sum_between(lo, hi), x, y),
                rust_sum_between(lo, hi, x, y),
                "{lo}..{hi}, at ({x}, {y})"
            );
        }
    }
}
```

In the D-a block (`:751-783`), after `over_columns`:

```rust
    /// D4 under a count: a kernel applied in a fold whose range the host passes.
    pub fn over_n_columns(k: impl Fn(f32, f32) -> f32, n: usize) -> f32 {
        (0..n).map(|i| k(X + (i as f32), Y)).sum()
    }
```

```rust
fn rust_over_n_columns(k: impl Fn(f32, f32) -> f32, n: usize, x: f32, y: f32) -> f32 {
    (0..n).map(|i| k(x + (i as f32), y)).sum()
}

/// A kernel applied in a fold over a per-call count means the same closure
/// called over Rust's `0..n`: the entry is staged, and its count a uniform of
/// the program its host function builds.
#[test]
fn a_per_call_count_around_an_application_is_rusts() {
    let (k, rust_k) = plane();
    for n in [0, 1, 4] {
        let written = over_n_columns(&k, n);
        for (x, y) in FOLD_SAMPLES {
            assert_eq!(
                Lattice::eval_at(&written, x, y),
                rust_over_n_columns(rust_k, n, x, y),
                "n = {n}, at ({x}, {y})"
            );
        }
    }
}
```

`rust_sum(n: usize, x: f32)` is the plan's gate spelling (plan:711).

### 6.2 `binding_times.rs`

Add to its `kernel!` block (`:20-56`):

```rust
    /// Σ_{i<n} X·i, its count per call.
    pub fn sum_to(n: usize) -> f32 {
        (0..n).map(|i| X * (i as f32)).sum()
    }

    /// A count between a record and a scalar: declared in parameter order.
    pub fn scaled_sum(b: Bounds, n: usize, s: f32) -> f32 {
        (0..n).map(|i| (X - b.x0) * (i as f32)).sum::<f32>() * s
    }
```

Tests (both arms unless marked):

```rust
/// One program for every count (L9): a count is a uniform, keyed by its slot,
/// so two calls with different counts share a canonical key and a compiled
/// region.
#[test]
fn every_count_is_one_program() {
    assert_eq!(program_key(&sum_to(3)), program_key(&sum_to(7)));
    assert_eq!(code_at_frame(&sum_to(3)), code_at_frame(&sum_to(7)), "one compiled region");
}

/// A count is one uniform, declared in parameter order as the `f32` it names,
/// and a program compiled at one count is rebound to another from `Args`: the
/// pixels of baking that call. `2²⁴` itself is a count, and is written.
#[test]
fn a_program_compiled_at_one_count_is_rebound_to_another() {
    let defaults: Vec<f32> = scaled_sum(UNIT, 5, 0.5).uniforms().iter().map(|u| u.default).collect();
    assert_eq!(defaults, [2.0, 1.0, 6.0, 4.0, 5.0, 0.5]);

    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let program = Manifold::compile(&sum_to(0), lattice.extent);
    let bound = program.bind(&[]);
    let mut block = program.block();
    for n in [7, 0, 3] {
        SumToArgs { n }.write_into(&mut block).expect("sum_to's own count");
        let rebound = lattice.collapse(&bound.clone().with_uniforms(&block));
        assert_eq!(rebound.buffer(), lattice.bake(&sum_to(n)).buffer(), "n = {n}");
    }
    let at_the_top = sum_to(16_777_216);
    assert_eq!(at_the_top.uniforms()[0].default, 16_777_216.0);
    SumToArgs { n: 16_777_216 }.write_into(&mut block).expect("2^24 is a count");
    assert_eq!(block.values(), [16_777_216.0]);
}

/// The host function refuses a count past 2²⁴, as `Kernel::over` refuses a
/// literal range past it: its uniform would hold `2²⁴`, a count, and the fold
/// would run one trip short with no NaN to say so (§1.4).
#[test]
#[should_panic(expected = "past 16777216 (2^24)")]
fn the_host_function_refuses_a_count_past_the_domain() {
    let _refused = sum_to(16_777_217);
}

/// A block written by position is untyped: any `f32` reaches the count's
/// slot, and one that names no count makes the fold NaN (L5), the loop not
/// entered, so nothing hangs. (`2²⁴ + 1` cannot be written this way at all:
/// no `f32` names it, `value_bounds_jit.rs`'s
/// `the_f32_nearest_two_to_the_24_plus_one_is_two_to_the_24`.)
#[test]
fn a_count_written_raw_outside_the_domain_is_nan() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let program = Manifold::compile(&sum_to(0), [1, 1]);
        let bound = program.bind(&[]);
        let mut block = program.block();
        let values: Vec<f32> = [16_777_218.0, f32::NAN, -1.0, 2.5, f32::INFINITY, 1e30]
            .into_iter()
            .map(|raw| {
                block.set_declared([raw]).expect("one argument");
                bound.clone().with_uniforms(&block).eval_at(3.0, 0.0)
            })
            .collect();
        tx.send(values).expect("the test is listening");
    });
    let values = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("a count outside the domain never hangs: no answer in a minute");
    assert!(values.iter().all(|v| v.is_nan()), "{values:?}");
}
```

(If a value in the worker panics, `recv_timeout` sees `Disconnected`: join the worker and resume its panic,
as `value_bounds_jit.rs:84-103`'s `within` does. Put those few lines in a local `within` if more than this
test needs it; nothing else in this file does.)

**The `Args` write, arm (a):**

```rust
/// The `Args` write refuses a count past 2²⁴ and writes nothing (§1.4): as an
/// `f32`, `2²⁴ + 1` is `2²⁴`, a count, so the program could not tell.
#[test]
fn the_args_write_refuses_a_count_past_the_domain() {
    let program = Manifold::compile(&sum_to(0), [1, 1]);
    let mut block = program.block();
    assert_eq!(
        SumToArgs { n: 16_777_217 }.write_into(&mut block),
        Err(WriteRefused::Count { parameter: "n", count: 16_777_217 })
    );
    assert_eq!(block.values(), program.block().values(), "nothing written");
}
```

and in `an_arity_mismatch_is_an_error` (`:193-233`) the two `write_into` assertions expect
`Err(WriteRefused::Arity(ArityMismatch { .. }))` (the `set_declared` one, `:209-215`, keeps
`ArityMismatch`); import `WriteRefused` (`:17`).

**Arm (b):** the same test is `#[should_panic(expected = "past 16777216 (2^24)")]` with the one call
`let _refused = SumToArgs { n: 16_777_217 }.write_into(&mut block);`. "Nothing written" is pinned by the
emit unit test's order (§6.5: the refusal precedes `set_declared` in the tokens) and by `set_declared`
being the only writer. `an_arity_mismatch_is_an_error` is unchanged.

The module doc (`:1-14`) gains: "An entry's count parameter, `n: usize`, is a uniform too: one program for
every count, rebound per call, and refused past 2²⁴ where the host writes it."

### 6.3 `fold_is_kernel_over.rs` (green, and two cases)

```rust
kernel_raw! {
    /// Σ_{i<n} (X·i + Y), its count per call.
    pub fn sum_to(n: usize) -> f32 {
        (0..n).map(|i| X * (i as f32) + Y).sum()
    }
}

/// A fold over a per-call count is `Kernel::over_values` over a uniform: one
/// key, and the same default. (`Canonical` holds each uniform's identity too,
/// which two mints never share, so the key and the defaults are compared.)
#[test]
fn a_per_call_count_is_over_values() {
    let written = sum_to(3);
    let built = Kernel::over_values(
        Monoid::SUM,
        &Kernel::constant(0.0),
        &Uniform::new(3.0).kernel(),
        |i| x().mul(i).add(&y()),
    );
    let (written_arena, written_root) = written.parts();
    let (built_arena, built_root) = built.parts();
    let (written_key, built_key) = (
        canonical(written_arena, written_root),
        canonical(built_arena, built_root),
    );
    assert_eq!(
        written_key.key,
        built_key.key,
        "written: {}\nbuilt:   {}",
        written_arena.display(written_root),
        built_arena.display(built_root),
    );
    let defaults = |uniforms: &[pixelflow_ir::arena::UniformDecl]| -> Vec<u32> {
        uniforms.iter().map(|u| u.default.to_bits()).collect()
    };
    assert_eq!(defaults(&written_key.uniforms), defaults(&built_key.uniforms));
}

/// A triangular fold — the inner bounded by the outer index — is `Kernel::over`
/// with `Kernel::over_values` inside it, the outer index its `hi`: the inner at
/// slot 0, the outer at slot 1.
#[test]
fn a_triangular_fold_is_over_values_inside_over() {
    let written = kernel_raw!(|| (0..4)
        .map(|i| (0..i).map(|j| X * (j as f32) + Y).sum::<f32>())
        .sum());
    let built = Kernel::over(Monoid::SUM, 0..4, |i| {
        Kernel::over_values(Monoid::SUM, &Kernel::constant(0.0), i, |j| x().mul(j).add(&y()))
    });
    assert_same_program(&written, &built);
    let (arena, root) = written.parts();
    assert_eq!(binders(arena, root), [5, 4], "outer slot 1, inner slot 0");
}
```

Imports: `pixelflow_core::Uniform` (`:20`). The module doc (`:1-17`) gains: "A bound that is a value — an
entry's count, an enclosing index — is `Kernel::over_values`'s." If either key differs, print both
displays and report: do not loosen the comparison without a ruling (§9, the riskiest part).

### 6.4 `kernel_typed_parameters.rs`, `items_block.rs`, `optimized_agrees_with_raw.rs`

**`kernel_typed_parameters.rs`** (+1). In the block (`:61-140`):

```rust
    /// A count beside a kernel: a fold's range per call, around an application.
    pub fn columns_to(k: impl Fn(f32, f32) -> f32, n: usize) -> f32 {
        (0..n).map(|i| k(X + (i as f32), Y)).sum()
    }
```

```rust
/// An entry that takes a kernel and a count is built when its host function is
/// called, and is one program for every count, as one that takes no kernel is:
/// the count is the composed program's own uniform, declared first.
#[test]
fn a_count_beside_a_kernel_is_one_program_for_every_count() {
    let k = kernel!(|s: f32| X * s)(2.0);
    let (three, five) = (columns_to(&k, 3), columns_to(&k, 5));
    let ((three_arena, three_root), (five_arena, five_root)) = (three.parts(), five.parts());
    assert_eq!(canonical(three_arena, three_root).key, canonical(five_arena, five_root).key);
    let defaults: Vec<f32> = three.uniforms().iter().map(|u| u.default).collect();
    assert_eq!(defaults, [3.0, 2.0], "the entry's count first, then the argument's");
    // Σ_{i<3} 2·(X + i) at X = 3: 2 · (3 + 4 + 5).
    assert_eq!(bake(&three), 24.0);
}
```

**`items_block.rs`** (0 new). In `without_a_prelude`'s block (`:120-140`):

```rust
        /// The weighed point summed over a count the host passes.
        pub fn counted(m: Mass, n: usize) -> f32 {
            (0..n).map(|i| weighed(m, i as f32)).sum::<f32>()
        }
```

In `a_block_expands_where_no_prelude_is_in_scope` (`:154-183`), with `CountedArgs` and `counted`
imported:

```rust
    // Σ_{i<4} (i·m.x + m.w), rebound from its `Args` at another count.
    let by_rust: f32 = (0..4).map(|i| (i as f32) * m.x + m.w).sum();
    assert_eq!(Lattice::eval_at(&counted(m, 4), x, 0.0), by_rust);
    let program = Manifold::compile(&counted(m, 0), lattice.extent);
    let mut block = program.block();
    CountedArgs { m, n: 4 }.write_into(&mut block).expect("counted's own arguments");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    assert_eq!(rebound.buffer(), lattice.bake(&counted(m, 4)).buffer());
```

This pins that the host refusal and the `write_into` refusal expand without a prelude. The block doc
(`:110-117`) adds "a count parameter".

**`optimized_agrees_with_raw.rs`** (+1, `kernel_raw!` parity):

```rust
kernel! {
    /// Σ_{i<n} (X + i)·Y, the count per call.
    pub fn counted(n: usize) -> f32 {
        (0..n).map(|i| (X + i as f32) * Y).sum::<f32>()
    }
}

kernel_raw! {
    /// The same, as written.
    pub fn counted_raw(n: usize) -> f32 {
        (0..n).map(|i| (X + i as f32) * Y).sum::<f32>()
    }
}

/// A per-call count: `Σ_{i<n} (X + i)·Y = Y·(nX + n(n−1)/2)`, the program
/// saturated at expansion and the one as written. The lattice's coordinates
/// are integers, so every value here is exact.
#[test]
fn a_per_call_count_agrees() {
    for n in [0_usize, 1, 5] {
        check(&format!("counted({n})"), &counted(n), &counted_raw(n), |x, y| {
            let n = n as f64;
            let want = y * (n * x + n * (n - 1.0) / 2.0);
            (want, four_ulps(want))
        });
    }
}
```

### 6.5 Unit tests

**sema.rs** (+3; the others rewritten in place):

1. `a_range_bound_that_is_not_constant_is_refused` (`:2605-2654`) is renamed
   `a_range_bound_is_a_usize_and_reads_no_coordinate`, doc "A range's bound is a `usize` typed where the
   fold is written (§3.4 of the bound plan): a coordinate is refused by L3, any other name that is not a
   `usize` by type, and a shape no count is built from as such.", cases (input → substrings, all):

   | Input | Substrings |
   |---|---|
   | `\|\| (0..X).map(\|i\| i as f32).sum()` | "`X` is a coordinate", "L3 of docs/language/2026-10-20-a-bound-is-a-value.md" |
   | `\|\| (0..X + 1).map(\|i\| i as f32).sum()` (R-A6) | the same |
   | `\|\| (Y..4).map(\|i\| i as f32).sum()` | "`Y` is a coordinate" |
   | `\|n: f32\| (0..n).map(\|i\| i as f32).sum()` | "expected `usize`, found `f32`", "§3.4 of docs/language/2026-10-20-a-bound-is-a-value.md", "an entry parameter `n: usize`" |
   | `\|\| { let n = 4.0; (0..n).map(\|i\| i as f32).sum() }` | "expected `usize`, found `f32`" |
   | `const R: f32 = 4.0; pub fn f() -> f32 { (0..R).map(\|i\| i as f32).sum() }` | "found `f32`", "declare the count as `const R: usize`" |
   | `\|\| (0..nn).map(\|i\| i as f32).sum()` | "cannot find `nn`" |
   | `\|\| (0..4.0).map(\|i\| i as f32).sum()` | "expected `usize`, found a float literal" |
   | `\|\| (0..X.floor()).map(\|i\| i as f32).sum()` | "a range's bound is a count, and this is not one" |
   | `\|\| (0..(1 < 2)).map(\|i\| i as f32).sum()` | the same |
   | `\|\| (0..-1).map(\|i\| i as f32).sum()` | "cannot apply unary operator `-` to type `usize`" |

   (`(0..X.floor())` names `X` inside a method call, which the bound walk does not enter: it is no
   count's shape, refused as such; §9 J1.) The trailing `ranges are constant` assertion (`:2652-2653`) goes.

2. New `a_range_bound_may_be_an_enclosing_index_or_a_count_parameter`: `accepted` for
   `|| (0..4).map(|i| (0..i).map(|j| (j as f32) * X).sum::<f32>()).sum()`;
   `|| (0..4).map(|i| (i..4).map(|j| j as f32).sum::<f32>()).sum()`;
   `pub fn f(n: usize) -> f32 { (0..n).map(|i| X * (i as f32)).sum() }`;
   `pub fn f(n: usize) -> f32 { (n..8).map(|i| i as f32).sum::<f32>() + (n as f32) }`;
   `pub fn f(lo: usize, hi: usize) -> f32 { (lo..hi).map(|i| i as f32).sum() }`;
   `pub fn f(n: usize) -> f32 { (0..(n)).map(|n| n as f32).sum() }` (a fold's index may shadow the count
   that bounds it, inside its body only, as in Rust);
   `fn tri(x: f32) -> f32 { (0..4).map(|i| (0..i).map(|j| x * (j as f32)).sum::<f32>()).sum() } pub fn f() -> f32 { tri(X) }`;
   `pub fn f(k: impl Fn(f32, f32) -> f32, n: usize) -> f32 { (0..n).map(|i| k(X + (i as f32), Y)).sum() }`.

3. New `arithmetic_on_a_value_bound_is_refused_until_bounds_are_integers`: each refused with "arithmetic on
   `{name}` in a range's bound" and "§1.4 of docs/language/2026-10-20-a-bound-is-a-value.md":
   `|| (0..4).map(|i| (0..i + 1).map(|j| j as f32).sum::<f32>()).sum()` (`i`);
   `pub fn f(n: usize) -> f32 { (0..n - 1).map(|i| i as f32).sum() }` (`n`);
   `const LO: usize = 2; pub fn f(n: usize) -> f32 { (LO + n..n).map(|i| i as f32).sum() }` (`n`);
   `pub fn f(n: usize) -> f32 { (0..2 * n).map(|i| i as f32).sum() }` (`n`). And
   `accepted(const LO: usize = 2; const N: usize = 3; pub fn f(n: usize) -> f32 { (LO + N..n)… })`: arithmetic
   over consts beside a value is not refused.

4. New `any_and_all_take_bounds_known_at_expansion`: refused with "decision 8 of
   docs/language/2026-10-20-a-bound-is-a-value.md": `pub fn f(n: usize) -> bool { (0..n).any(|i| X < i as f32) }`;
   `pub fn f(n: usize) -> bool { (n..4).all(|i| X < i as f32) }`;
   `|| (0..4).map(|i| if (0..i).any(|j| X < j as f32) { 1.0 } else { 0.0 }).sum()`.

5. `a_usize_is_not_a_declared_value_type` (`:2835-2860`) is renamed `a_usize_parameter_is_an_items_entrys_count`,
   doc "A `usize` parameter is a count, an entry's in the items form (decision 2 of the bound plan); a
   helper's and the closure form's are refused, and no `let` or `fn` return is a `usize`.":
   `accepted(pub fn f(n: usize) -> f32 { X })`;
   `pub fn f() -> usize { 4 }` → "a kernel `fn` returns an `f32` or a `bool`" and "a `usize` is a count";
   `|| { let n: usize = 4; X }` → "a `let` in a kernel body is an `f32` or a `bool`" and "a `usize` is a count";
   `fn h(n: usize) -> f32 { n as f32 } pub fn f() -> f32 { X }` → "a helper's parameter is" and "§1.4 of
   docs/language/2026-10-20-a-bound-is-a-value.md";
   `|n: usize| X` (R-A6 a6fit 1) → "the closure form's parameter is an `f32`" and the same plan path.

6. In place: `a_usize_where_a_value_is_expected_is_a_type_error` (`:2765-2798`) gains
   `(quote! { pub fn f(n: usize) -> f32 { X * n } }, "`n`")` (its array grows to 10);
   `as_f32_converts_a_usize_and_nothing_else` (`:2803`) gains
   `accepted(quote! { pub fn f(n: usize) -> f32 { X * (n as f32) } })`;
   `a_range_from_usize_consts_is_evaluated_at_expansion` (`:2693-2697`) asserts
   `[Count::Known(3), Count::Known(14)]`;
   `a_reversed_range_is_refused_and_an_empty_one_is_not` (`:2659-2666`) gains
   `accepted(quote! { pub fn f(n: usize) -> f32 { (5..n).map(|i| i as f32).sum() } })` (a pair with a value
   is decided per call, L2), its doc saying so;
   `a_fold_index_is_scoped_to_the_folds_body` (`:2866-2881`) gains the fold's own index in its own bound:
   `refusal(|| (0..i).map(|i| i as f32).sum())` contains "cannot find `i`", and
   `accepted(|| (0..4).map(|i| (0..i).map(|i| i as f32).sum::<f32>()).sum())` (the inner bound's `i` is the
   outer index, as Rust reads it).

**lower.rs** (+2): `an_index_where_a_value_is_expected_is_refused_here_too` (`:1129-1144`) asserts
"`i` is a `usize`" and gains `pub fn f(n: usize) -> f32 { X * n }` → "`n` is a `usize`". New:

```rust
/// A bound that is a value is a child of its fold: an entry's count its
/// `Uniform` leaf, an enclosing fold's index that fold's binder; and the
/// fold's range is not known.
#[test]
fn a_value_bound_is_a_child_of_its_fold() {
    let (arena, root) = lowered(quote! {
        pub fn f(n: usize) -> f32 { (0..n).map(|i| X * (i as f32)).sum() }
    });
    let ExprNode::Reduce { lo, hi, .. } = arena.node(root) else {
        panic!("expected a fold, got {}", arena.display(root));
    };
    assert_eq!(lo, Bound::Lit(0));
    let Bound::Value(count) = hi else {
        panic!("`n` is a value bound");
    };
    assert!(matches!(arena.node(count), ExprNode::Uniform(UniformId(0))));
    assert!(arena.known_range(root).is_none());

    let (arena, root) = lowered(quote! {
        || (0..4).map(|i| (0..i).map(|j| j as f32).sum::<f32>()).sum()
    });
    let ExprNode::Reduce { fold: outer, body, .. } = arena.node(root) else {
        panic!("expected the outer fold, got {}", arena.display(root));
    };
    let ExprNode::Reduce { hi: Bound::Value(index), .. } = arena.node(body) else {
        panic!("the inner fold's `hi` is the outer index, got {}", arena.display(body));
    };
    assert!(matches!(arena.node(index), ExprNode::Var(v) if v == outer.binder().var()));
    assert!(!holds_a_placeholder(&arena));
}

/// Lowering refuses a bound that is no count by itself, as `sema` does,
/// without relying on it.
#[test]
fn a_bound_that_is_no_count_is_refused_here_too() {
    for (input, expected) in [
        (quote! { |n: f32| (0..n).map(|i| i as f32).sum() }, "`n` as a range's bound"),
        (quote! { || (0..X).map(|i| i as f32).sum() }, "`X` as a range's bound"),
        (quote! { pub fn f(n: usize) -> bool { (0..n).any(|i| X < i as f32) } }, "decision 8"),
        (quote! { pub fn f(n: usize) -> f32 { (0..n - 1).map(|i| i as f32).sum() } }, "arithmetic on `n`"),
        (quote! { pub fn f(n: usize) -> f32 { (16777217..n).map(|i| i as f32).sum() } }, "past 16777216 (2^24)"),
    ] {
        let Err(err) = lower_unanalyzed(input) else {
            panic!("lowering builds no fold over a bound that is no count");
        };
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}
```

`use pixelflow_ir::{Bound, …}` is already in scope through `super::*` (`lower.rs:69`).

**emit.rs** (+1):

```rust
/// A count parameter is the host function's declared `usize`, refused past
/// 2²⁴ before anything is built; its uniform's default and its `Args` value
/// are `n as f32`, and `write_into` refuses before it writes. A staged fold
/// closes over its count's term.
#[test]
fn a_count_parameter_is_refused_on_the_host_and_declared_as_an_f32() {
    let code = expansion(quote! {
        pub fn sum_to(n: usize, s: f32) -> f32 { (0..n).map(|i| X * (i as f32)).sum::<f32>() * s }
        pub fn columns(k: impl Fn(f32, f32) -> f32, n: usize) -> f32 {
            (0..n).map(|i| k(X + (i as f32), Y)).sum()
        }
    });
    for expected in [
        "pub fn sum_to (n : usize , s : f32)",
        "KnownRange :: admits (& (0 .. n as u64))",
        "past 16777216 (2^24)",
        "§1.4 of docs/language/2026-10-20-a-bound-is-a-value.md",
        "# Panics",
        "default : n as f32",
        "pub n : usize",
        "self . n as f32 , self . s",
        "Bound :: Value (__t",
    ] {
        assert!(code.contains(expected), "expected `{expected}`: {code}");
    }
    let refused = code.find("KnownRange :: admits (& (0 .. self . n as u64))").expect("write_into refuses");
    let written = code.find("block . set_declared").expect("write_into writes");
    assert!(refused < written, "refused before anything is written: {code}");
    assert!(!code.contains("ColumnsArgs"), "an entry that takes a kernel has no `Args`: {code}");
    assert!(!code.contains(":: std ::"), "{code}");
}
```

(`to_string` spacing is `proc_macro2`'s: if a substring differs only in spacing, pin what the first run
prints. The non-ASCII `§` is inside a string literal and survives `to_string` as written.)

### 6.6 Every new refusal, pinned

| Refusal | Where | Test | Plan path pinned |
|---|---|---|---|
| a coordinate as a bound (L3), incl. `(0..X + 1)` | sema M1 | `a_range_bound_is_a_usize_and_reads_no_coordinate` | L3 of BOUND_PLAN |
| a name that is not a `usize` | sema M2 | the same | §3.4 |
| a shape no count is built from | sema M5 | the same | §3.4 (in the text; the test pins the header) |
| arithmetic on a value bound | sema M3 (and lowering, shared) | `arithmetic_on_a_value_bound_is_refused_until_bounds_are_integers`; lower `a_bound_that_is_no_count_is_refused_here_too` | §1.4 |
| a value bound under `any`/`all` | sema M4; lowering | `any_and_all_take_bounds_known_at_expansion`; the lower test | decision 8 |
| a helper's `usize` (D-A6-2) | sema M6 | `a_usize_parameter_is_an_items_entrys_count` | §1.4 |
| the closure form's `usize` (D-A6-3, R-A6) | sema M7 | the same | §1.4 |
| the fold's own index in its own bound | sema (`not_in_scope`) | `a_fold_index_is_scoped_to_the_folds_body` | — (scope) |
| a name that is no count binding | lowering | the lower test | — |
| a literal past 2²⁴ beside a value | lowering (`literal_bound`) | the lower test | — (`Fold::EXACT_BOUND`, as `literal_range`) |
| the host function past 2²⁴ (D-A6-4) | emitted assert | `the_host_function_refuses_a_count_past_the_domain` | §1.4 (emit unit test) |
| the `Args` write past 2²⁴ (D-A6-1) | emitted check | `the_args_write_refuses_a_count_past_the_domain`; (a) core's `a_refused_write_says_why` | §1.4 |
| a raw write naming no count | the guard (run time) | `a_count_written_raw_outside_the_domain_is_nan` | — |

### 6.7 Test-count accounting (`#[test]` in `pixelflow-compiler`, 299 at `29980bf`)

| File | At `29980bf` | Added | After |
|---|---|---|---|
| src/sema.rs | 60 | `a_range_bound_may_be_an_enclosing_index_or_a_count_parameter`, `arithmetic_on_a_value_bound_is_refused_until_bounds_are_integers`, `any_and_all_take_bounds_known_at_expansion` | 63 |
| src/lower.rs | 18 | `a_value_bound_is_a_child_of_its_fold`, `a_bound_that_is_no_count_is_refused_here_too` | 20 |
| src/emit.rs | 7 | `a_count_parameter_is_refused_on_the_host_and_declared_as_an_f32` | 8 |
| tests/rustc_is_the_oracle.rs | 30 | `a_triangular_fold_folds_as_rustcs_does`, `a_per_call_count_folds_as_rustcs_does`, `a_range_between_counts_folds_as_rustcs_does`, `a_per_call_count_around_an_application_is_rusts` | 34 |
| tests/binding_times.rs | 7 | `every_count_is_one_program`, `a_program_compiled_at_one_count_is_rebound_to_another`, `the_host_function_refuses_a_count_past_the_domain`, `a_count_written_raw_outside_the_domain_is_nan`, `the_args_write_refuses_a_count_past_the_domain` | 12 |
| tests/fold_is_kernel_over.rs | 14 | `a_per_call_count_is_over_values`, `a_triangular_fold_is_over_values_inside_over` | 16 |
| tests/kernel_typed_parameters.rs | 16 | `a_count_beside_a_kernel_is_one_program_for_every_count` | 17 |
| tests/optimized_agrees_with_raw.rs | 10 | `a_per_call_count_agrees` | 11 |
| all others (ast 4, lib 4, parser 43, symbol 7, and nine test files, 63 in all) | 137 | | 137 |
| **Total** | **299** | **19** | **318** |

Renamed, not counted: sema `a_range_bound_that_is_not_constant_is_refused` →
`a_range_bound_is_a_usize_and_reads_no_coordinate`; `a_usize_is_not_a_declared_value_type` →
`a_usize_parameter_is_an_items_entrys_count`. Removed: none. The same total under both arms of D-A6-1.
`pixelflow-core`: +1 under (a) (`a_refused_write_says_why`), 0 under (b). Verify with
`grep -c '#\[test\]' pixelflow-compiler/src/*.rs pixelflow-compiler/tests/*.rs` (sum 318) and the passed
totals of `cargo test -p pixelflow-compiler`.

---

## 7. Docs

### 7.1 The plan (`docs/language/2026-10-20-a-bound-is-a-value.md`)

- **§1.4, plan:133-136:** "…at `kernel!`'s `Args` write for a per-call count, in whichever spelling
  decision 2 picks (A6), and at bind…" → "…at `kernel!`'s host write of a per-call count, an entry's
  `n: usize` (decision 2): its host function and its `Args` record (A6); and at bind…".
- **§3.4, plan:409-416** (to "expansion-time checks it has (reversed, past 2²⁴)."): "A bound is a
  `Ty::Usize` expression typed in the enclosing scope (`type_of_fold` types its names before it pushes
  the fold's scope): a literal, a `usize` const, an enclosing index, or an entry parameter `n: usize` of
  the items form (decision 2), a uniform. An argument's extent arrives with B5's surface. X and Y are
  `Ty::F32` with no `as usize`, so sema refuses L3 by type, the IR's `BoundVaries` behind it; every name
  in a bound is typed before its arithmetic is judged, and a non-leaf value bound (`0..i + 1`) is refused
  until T6 (§1.4). `lower_fold` resolves `lo`/`hi` before `open_fold`, which takes them, and
  `Site::close_fold` hands them to the IR. A known bound keeps the expansion-time checks it has
  (reversed, past 2²⁴), and a known end beside a value is checked alone against 2²⁴. A count the host
  writes is refused past 2²⁴ by the entry's host function and its `Args` record (§1.4); a helper and the
  closure form take no count." The structural-path sentence (plan:416-420) is unchanged.
- **§5's row for the-language-is-kernel §1.3 (plan:492):** Becomes → "a fold binder or a count (built
  from literals and `usize` consts with `+ - * /`, or an entry's per-call `n: usize`)".
- **§6 A6, plan:709-712:** "Waits on decision 2, the spelling; decision 1 is B5's." → "Decision 2's
  spelling, an `n: usize` entry parameter; decision 1 is B5's." Then append, edited to what the gates
  measured (keep no clause no gate checked):
  "**Done:** a bound's names are typed where the fold is written: a literal or a `usize` const (`+ - * /`
  over them), an enclosing fold's index (a triangular `0..i`), or an entry parameter `n: usize` of the
  items form, a uniform declared in parameter order as `n as f32` and read as a bound or by `n as f32`;
  one program for every `n`. Refused at expansion: `X` or `Y` (L3), any other name that is not a
  `usize` (by type), arithmetic on a bound that is a value (`BoundArithmetic`, until T6), a value bound
  under `any` or `all` (decision 8), a known end past 2²⁴, and a `usize` parameter of a helper or of the
  closure form. The host function panics on a count past 2²⁴, and `Args::write_into` [(a) refuses one
  with `WriteRefused::Count` | (b) panics on one] before writing anything; a raw block write that names
  no count makes the fold NaN by the guard. `rustc_is_the_oracle.rs` has the triangular and per-call
  cases, and `fold_is_kernel_over.rs` the same two against `over_values`. byte_probe empty in both columns
  under `PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font programs' keys, digests and saturation telemetry,
  the extraction fields included, unchanged."
- **Not edited:** §0, §2's row (plan:230, "`kernel!` refuses `(0..4).map(|i| (0..i)…)`") — the tree the
  plan was verified against; §8 decision 2's record (plan:769-771).

### 7.2 `docs/language/2026-09-25-the-language-is-kernel.md`

`:197`, the §1.3 `usize` row: "a fold binder or a count (built from literals and `usize` consts with
`+ - * /`)" → "a fold binder or a count (built from literals and `usize` consts with `+ - * /`, or an
entry's per-call parameter `n: usize`)"; its IR column `Var(REDUCE_BINDER_BASE + slot)` → "a binder's
`Var(REDUCE_BINDER_BASE + slot)`, a `usize` const's `Const`, a count parameter's `Uniform`" (A5's
critique a6fit 4 noted a const lowers to a `Const`), "converted by an explicit `i as f32`" kept.

### 7.3 `CLAUDE.md`, `docs/DENOTATION.md`, `pixelflow-core/README.md`

- **CLAUDE.md: no edit.** `:180` already says "a per-call count is a uniform, computed once per call, so
  one program serves every count; a count is an integer in `0..=2²⁴`", true since A4; the `kernel!`
  pattern section (`:505-517`) shows the closure form only. `grep -n 'usize\|per-call' CLAUDE.md` finds
  `:180` alone.
- **DENOTATION: no edit.** `:129-131` ("A uniform may bound a fold, as a per-call count") is the meaning;
  the surface spelling is the compiler's (`lib.rs`) and the ontology's. Plan §5 assigns DENOTATION no A6
  row.
- **`pixelflow-core/README.md:64`:** "a per-call count (`Kernel::over_values` with a `Uniform`)" → "a
  per-call count (`Kernel::over_values` with a `Uniform`; in `kernel!`, an entry's `n: usize`)".

### 7.4 The ontology (`.claude/skills/ontology/language.md`)

- **`kernel!`, Follows (`:70-72`):** append "A count the host writes is refused past 2²⁴ by
  `KnownRange::admits`, in the entry's host function and its `write_into`."
- **Entry, helper, record, Args — Is (`:95-97`):** "**Args** is an entry's record of its uniform
  arguments — `f32`s, counts (`n: usize`) and records — when it has some and takes no kernel." Follows
  (`:107-109`): "`write_into` streams values by position and checks only the arity and each count's
  domain, so…". Lives (`:113-114`): "(`write_into` returns `Result<(), WriteRefused>`)" under (a), or
  "(`write_into` returns `Result<(), ArityMismatch>` and panics on a count past 2²⁴)" under (b); add
  `WriteRefused` beside `ArityMismatch` in the next line under (a).
- **Binding time, Is (`:130`):** "An author sees two binding times: **uniform** and **kernel-typed**." →
  "… **uniform** and **kernel-typed**; a count parameter `n: usize` is a uniform, read as a fold's bound,
  so one program serves every count."
- **Structural parameter (retired), Is (`:154`):** "which A6 builds" → "which A6 built".
- **Fold (Reduce), Is (`:427-429`):** after the spellings: "Its bounds are `usize`s: a literal or a
  `usize` const, an enclosing fold's index (`0..i`), or an entry's count parameter `n: usize`."
- **Range (trip count), Is (`:509-510`):** "…a value (a child): a per-call count — in `kernel!` an
  entry's `n: usize` — or an enclosing fold's index." Is not (`:514-515`): "Past 2²⁴: a literal is refused
  at construction, a count the host writes is refused at the write, and a value makes the fold NaN."
- `SKILL.md`: nothing (no entry is added or renamed).

### 7.5 `intent.md`

**`pixelflow-compiler/intent.md`** (63 lines, budget 80; `bash scripts/check-intent.sh`):

| Line | After |
|---|---|
| 13-15 (Owns) | "Binding times as the host sees them (uniform, kernel-typed), the host `fn`, the `Args` record, and `const` evaluation: a count is built from literals and `usize` consts at expansion, or is an entry's `n: usize`, a uniform the host function and `write_into` refuse past 2²⁴." |
| 31-32 (Never) | "No call's value becomes a constant: an entry that takes no kernel is one program, whatever its counts." |
| 48-51 (Enforced by) | add "`sema.rs` `a_range_bound_is_a_usize_and_reads_no_coordinate`" to the unit tests |
| Behind (after 57) | "- A bound that is a value is a single name until bounds are typed integers: `0..i + 1` is refused (`BOUND_PLAN` §1.4, T6)." |
| 62 (Decided in) | "`docs/language/2026-10-20-a-bound-is-a-value.md` (§1.2, §1.4, §3.4)," |

**`pixelflow-core/intent.md`** under (a) only: the Contract's re-export list (`:39-42`) adds
`WriteRefused` after `ArityMismatch`, reflowed.

### 7.6 `lib.rs`'s macro docs

`# Folds` (`:157-168`), from "The bounds are constant" to "nothing to index.":

```
/// Each bound is a `usize`: an integer, a `usize` const or `+ - * /` over
/// them, known at expansion; an enclosing fold's index, for a triangular
/// `(0..i)`; or an entry's count parameter, `n: usize`, bound per call. A
/// bound that is a name the program binds stands alone — `0..i + 1` waits
/// for bounds typed as integers — and no bound reads `X` or `Y`. The index
/// `i` is a `usize`, which a body reads only as `i as f32`: there is no
/// arithmetic on an index and nothing to index.
```

`# Binding times` (`:254-281`): the Uniform bullet begins "every `f32`, count and record parameter. An
`f32` is one uniform, a count (`n: usize`) one holding `n as f32`, and a record one per field; …"; after
"…`write_into` refuses." add "A count past 2²⁴ is refused where the host writes it: the host function
panics, and `write_into` [(a) returns `WriteRefused::Count` | (b) panics], writing nothing." The count
paragraph (`:278-281`) becomes: "A count known at expansion — a fold's bound, or `N as f32` — is built
from literals and `usize` consts with `+ - * /`; one that changes per call is an entry parameter
`n: usize`. Neither makes a program of its own, and no `fn` takes generics
(docs/language/2026-10-20-a-bound-is-a-value.md §1.2, §3.4)." The `rings` example is unchanged
(`binding_times.rs`' `rings` compiles it).

### 7.7 BACKLOG (`docs/BACKLOG.md`)

- **`:32`, the plan-table row:** landed gains ", A6 (`kernel!` bounds are typed: an enclosing index, and
  an entry's `n: usize`, refused past 2²⁴ where the host writes it)"; next becomes "A1's
  the-language-is-kernel D-table rows after #1323; B1–B5 after typed-arena T3 and item 2".
- **`:166`, C8's last sentence:** "…and a host-written count is refused at its write in A6." → "…and a
  count the host writes through `kernel!` is refused at the write (a-bound-is-a-value A6): the entry's host
  function panics past 2²⁴ and its `Args::write_into` [(a) returns `WriteRefused::Count` | (b) panics]
  before writing anything."
- No new row: the refusals A6 adds are the design's, recorded in the plan's Done line.

### 7.8 The IR's one doc (`pixelflow-ir/src/fold.rs:440-443`)

`BoundRefusal::BoundOutOfDomain`'s doc carries a step label in code ("`kernel!` from A6", which G2
forbids after A6): "Refused at the user surfaces only: `Kernel::over_values`, and `kernel!` in its own
words — at expansion for a literal, and where the host writes a count; every other constructor keeps such
a literal as a value bound…" (rest unchanged). No code changes; `pixelflow-ir`'s rustdoc is compared
(§8, step 7).

---

## 8. Order of work, and the grep-gated checklist

Never `cargo test --workspace`; `df -h /` before every build, stop under 2 GB. Never build from a
worktree; never commit, push, stash or reset; never run a gate script while another build of
`pixelflow-graphics` is going (the scripts append and restore their probe; `.wip/gates/README.md`). The
compiler crate does not build between its first and last source edit of step 2.

0. **Bases and the probe.** `git diff --quiet 29980bf HEAD -- . ':!.wip'`; the gate bases for `a6-main`
   are recorded (`.wip/gates/out/*.a6-main.*`, README "Verified again at `29980bf`") and the rustdoc bases
   are `.wip/a6/rustdoc-base.{pixelflow-compiler,pixelflow-core,pixelflow-ir}.txt` (normalizer
   `.wip/a6/rustdoc_norm.py`); record `cargo test -p pixelflow-compiler`'s passed total (299). Re-run
   `/home/user/scratch/a5-probe/tests/a6_macro_tier.rs` at HEAD as A5 ran its probe (A5 spec header: the
   repo's target dir and the workspace's `RUSTFLAGS`), under `PIXELFLOW_ISA=avx2` and `=avx512`. If it
   fails, stop and report. Run G2, G3, G4 and G6 once on the unmodified tree and see them find what they
   should (§8 greps).
1. **IR doc** (§7.8).
2. **Compiler sources**, all arms but D-A6-1's (§2, §3, §4.1-§4.3, §4.6-§4.8; unit tests §6.5): ast.rs,
   parser.rs, symbol.rs, sema.rs, lower.rs, emit.rs (`write_into` as today plus nothing yet), lib.rs.
   `cargo test -p pixelflow-compiler --lib`.
3. **Compiler integration tests** (§6.1, §6.3, §6.4, §6.2 except the `Args`-write test): `PIXELFLOW_ISA=avx2
   cargo test -p pixelflow-compiler` and `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler`.
4. **D-A6-1's arm** (§4.5), as JP ruled. If JP has not ruled, stop here and report: do not pick an arm.
   Then `cargo test -p pixelflow-core` (under (a)) and step 3's two runs again.
5. **Other crates:** `cargo test -p pixelflow-core` (its tests expand `kernel!`),
   `cargo test -p pixelflow-graphics --lib`, `cargo check --workspace --all-targets` (compiles every
   `kernel!` user; not a test run).
6. **Docs** (§7).
7. **Gates and checks:** gates 1, 1b, 2, 2x with `TAG=a6-impl` (identical to base, §5.3); gate 5 with
   `TAG=a6-impl` (report); `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`;
   `cargo fmt --all -- --check`; `bash scripts/check-intent.sh`, `check-doc-paths.sh`,
   `check-cfg-encapsulation.sh`, `check-emit-boundary.sh`, `check-bin-declarations.sh`;
   `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH bash scripts/check-feature-matrix.sh`; rustdoc for
   `pixelflow-compiler`, `pixelflow-ir` and (under (a)) `pixelflow-core` with no warning outside the bases
   (`cargo doc -p <crate> --no-deps --document-private-items 2>&1 | python3 -I .wip/a6/rustdoc_norm.py
   --compare .wip/a6/rustdoc-base.<crate>.txt`); the test count (§6.7); the greps below. Record
   deviations in `.wip/a6/impl.md`, and the public surface (the host function's `n: usize` and panic, the
   `Args` field, `write_into`'s contract, and under (a) `WriteRefused`) in `.wip/a6/public-surface.md`.

**Greps** (from `/home/user/pixelflow`; code blocks, since a table cell's `\|` is a literal pipe to
`grep -E`):

G1, the renamed and replaced (expected: nothing):

```sh
grep -rnE 'Binding::Index|\bBinding<.*Index|parameter\.record|scalar\.(param|field)|Scalar \{' pixelflow-compiler/
```

G2, step labels in code and intents (expected: nothing; at the base it finds `pixelflow-ir/src/fold.rs:441`):

```sh
grep -rnwE 'A6' --include=*.rs --include=intent.md . | grep -v '^./target' | grep -v '^./.wip/' | grep -v 0xA6
```

G3, "constant" bound wording in the compiler (expected: only `tests/rustc_is_the_oracle.rs:324`, the B2
section header, whose folds are over constant ranges; at the base it also finds ast.rs:379, sema.rs:27,
883, 1710, 1755, 1760, 1771, 1775, 2605, 2637, 2641, 2653, parser.rs:885, 893, 1070):

```sh
grep -rniE "ranges are constant|bounds are constant|over a constant range|iterates a constant range|over constant ranges" pixelflow-compiler/
```

G4, the two-item `usize` lists (expected: nothing; at the base ast.rs:667, 679, sema.rs:19, 114, 158, 934,
1620, parser.rs:52, 1195):

```sh
grep -rn "index or a \`usize\`" pixelflow-compiler/
```

G5, the domain predicate (expected, read each: `lower.rs` `literal_range` and `literal_bound`; `emit.rs`
the host refusal and the `write_into` refusal):

```sh
grep -rn 'KnownRange::admits\|KnownRange :: admits' pixelflow-compiler/src
```

G6, production has no count and no fold (expected: `0` before and after):

```sh
sed -n 31,141p pixelflow-graphics/src/fonts/loop_blinn/program.rs | grep -cE 'usize|\.map\(|\.any\(|\.all\('
```

G7, decision-2 placeholders in live docs (expected: nothing):

```sh
grep -rnE "decision 2 picks|Waits on decision 2|which A6 builds|refused at its write in A6" docs .claude CLAUDE.md | grep -v 'docs/archive/'
```

G8, `write_into`'s error type in emission (expected: (a) nothing; (b) the one `ArityMismatch` in the
signature):

```sh
grep -n 'ArityMismatch' pixelflow-compiler/src/emit.rs
```

G9, refusals cite the plan by const (read each: lib.rs's `BOUND_PLAN`; parser.rs `generic_refusal`;
sema.rs M1-M7; lower.rs decision 8; emit.rs `count_refusal` and the `# Panics` text):

```sh
grep -rn 'BOUND_PLAN' pixelflow-compiler/src
grep -rn 'docs/language/2026-10-20' pixelflow-compiler/src
```

(The second is expected to find only `lib.rs:77`'s definition and test assertions.)

G10, every `open_fold` (read each: the trait's, `Expansion`'s and `Staged`'s, `lower_fold`'s call, and
the arena calls each site makes):

```sh
grep -rn 'fn open_fold\|\.open_fold(' pixelflow-compiler/src
```

---

## 9. Judgement calls, each with the alternative rejected

- **J1. A bound's names are typed by a walk over the shapes a bound is built from** (`type_of_bound`:
  names, parentheses, unary, binary), before `range_bounds` judges its arithmetic (R-A6 a6fit 2). A shape
  outside it (`X.floor()`, `i as f32`, a call) names nothing to the walk and is M5's. Rejected: typing
  the bound with `type_of` (which refuses every `usize` name as a value and types binary operands as
  `f32`), or a walk that types every subexpression to say "found `f32`" of `X.floor()` (a second refusal
  path for shapes `range_bounds` already refuses).
- **J2. `Count::Name` owns its `Ident`.** Rejected: `Name(&'a Ident)` (design §1), which needs a generic
  associated type on `UsizeScope` to tie the output to the expression's lifetime; an `Ident` clone at
  expansion costs nothing that matters.
- **J3. `UsizeScope::Output: From<u64>` and `operand`** (R-A6 a6fit 6); arithmetic is judged after both
  sides are evaluated and after the operator is checked, so a `const`'s errors keep their order and
  `(i < 2)` is M5's, not M3's. Rejected: refusing at the first name met (`(i < 2)` would read as
  arithmetic), or one `Count` output for both scopes (the guard A5 deleted, for a variant a `const` can
  never hold).
- **J4. `RangeScope` stays consts-only**; which names are leaves is each stage's scope's question: sema's
  symbols, lowering's locals (design §10). Rejected: a scope carrying sema's symbol table (lowering has
  none, and one type would mean two scopes).
- **J5. Decision 8 by `Reduction::Any | Reduction::All`** in sema and in lowering. Rejected: making
  `Monoid::takes_a_value_bound` (`fold.rs:121`, `pub(crate)`) public, a visibility change CLAUDE.md
  reserves to JP. `push_reduce`'s `BoundSort` stays behind both, and the sema and lowering tests pin
  that `kernel!` refuses what it refuses.
- **J6. `Binding::Index` becomes `Binding::Count`**, covering an index and a count parameter. Rejected: a
  second variant every use matches identically.
- **J7. `Parameter { name, kind }` and `Scalar` an enum** (R-A6 a6fit 7). Rejected: `count: bool` beside
  `record` (a parameter both a record and a count would be representable, and every reader would check
  both).
- **J8. `argument(Scalar::Count(n))` is `n as f32`, and `write_into` reads it as `self.n as f32`** by
  precedence (a field access binds tighter than `as`), pinned by the emit unit test's
  `self . n as f32`. Rejected: a receiver argument to `argument` (an extra parameter at every call for one
  case), or destructuring `*self` in `write_into` (a parameter named `block` would shadow its own
  argument).
- **J9. The host's check is `KnownRange::admits(&(0..n as u64))`**, the plan's predicate on one count
  (plan:133-134), public at the IR's root. Rejected: `n <= 16777216` baked at expansion (a copy of the
  predicate), or a new IR function (a public addition for what one call says).
- **J10. Every count parameter is checked, read or not, a bound or not, in `Staged` entries too.** A
  count's domain is its type's: `n as f32` past 2²⁴ is another count whether a fold or an `as` reads it.
  Rejected: checking only counts that bound a fold.
- **J11. The host function of an entry with a count carries a generated `# Panics` section.** Rejected:
  an undocumented panic on a generated public function.
- **J12. Lowering refuses by itself** a name that is no count binding, a value under `any`/`all`, and a
  literal past 2²⁴ beside a value (`lower.rs:770-776`'s rule). Rejected: relying on sema.
- **J13. A known end beside a value is checked alone at lowering** (`literal_bound`), in the front end's
  words. Rejected: letting `FoldHeader::new` (`fold.rs:855-862`) panic at expansion.
- **J14. The oracle gains four tests, not two:** the plan's triangular and per-call cases, value ends and
  per-call reversed pairs (L2 at the surface), and a staged entry with a count. A4 runs a count bounding a
  triangle in codegen (`value_bounds_jit.rs`' `a_triangular_sum_under_a_count`), so `triangle_to` is
  expected to pass. Rejected: the two the gate names alone, which leave the value-`lo` and staged paths
  unrun at the surface.
- **J15. A helper's and the closure form's `usize` are refused in sema's `param_type`**, where types are
  classified. Rejected: the parser, which has no `Ty`.
- **J16. Messages cite `BOUND_PLAN` only;** T6 is named through §1.4, which names it (intent.md: "a
  refusal that cites a plan names it by `PLAN` or `BOUND_PLAN`"). Rejected: a third plan path in a message.
- **J17. `sema.rs:1893`'s duplicated doc line is deleted** in passing: it sits in the const evaluation A6
  rewrites, and is a defect a reader of the rewrite will meet. Rejected: leaving it.
- **J18. The raw-write test runs in a worker with a one-minute wait**, so a missing guard is a failed
  assertion, not a CI timeout (A4's `within`). Rejected: an unbounded call.

### Decisions

- **D-A6-1, needs JP (public API).** How `Args::write_into` refuses a count past 2²⁴ (§4.4).
  **Recommended: (a)**, `pixelflow_core::WriteRefused { Arity(ArityMismatch), Count { parameter, count } }`.
  It **adds one public item to `pixelflow-core`** beyond decision 7's list, and changes every generated
  `write_into`'s error type (no production caller). Fallback **(b)**, a panic, adds no public item.
  Rejected **(c)**, a `Count` newtype, which breaks decision 2's `n: usize` at the host.
- **D-A6-2, decided: a helper takes no count** (M6). Rejected: binding a helper's `n: usize` to the
  caller's count binding at inlining, so a helper can fold over a caller's per-call count. Nothing needs
  it before B5; it is then a lowering-only extension (`inline` binds a `usize` argument's name to its
  `Binding::Count`) plus sema typing a `usize` argument at a call.
- **D-A6-3, decided (R-A6 a6fit 1): the closure form takes no count** (M7). Rejected: accepting it with
  an assert in the closure, `emit_closure` spelling declared types, and an untyped positional rebind.
- **D-A6-4, decided: the host function panics** (§4.6). No public API change: the signature stays
  `-> Kernel`. Rejected: `-> Result<Kernel, _>` for count-taking entries.

### The riskiest part

What has not been run: the canonical-key equalities of §6.3 (`kernel_raw!` against `over_values` for a
per-call count and a triangle) and the macro tier saturating a value-bounded entry from the real lowering
rather than the builders. The probe covers the second for two shapes, through builders, at `7e9531e`;
step 0 re-runs it. If a key differs, the arenas' displays say where (node order of a value bound's term
before the fold opens, against `over_values`' body-then-bounds splice, is the likely place); report it
rather than loosen the test. If saturation at expansion panics (E8's `BoundVaries` at materialization),
that is a proc-macro panic at the user's `kernel!`; report it with the input.
