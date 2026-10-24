# A6 and A7: design, as far as it constrains A5 and as far as A6's spec will need

**Base:** `7e9531e` (main after A4, #1448); file:line at that commit. A5's spec:
`/home/user/pixelflow/.wip/a5/spec.md` ("A5 spec").
**Plan:** `docs/language/2026-10-20-a-bound-is-a-value.md` ("plan"): §1.1 (the fold), §1.2 (binding
times, plan:65-80), §1.4 (the domain and the leaf rule, plan:118-154), §1.6 L2, L3, L5, L9
(plan:169-216), §3.1 (`push_reduce`'s refusals, plan:263-276), §3.4 (`kernel!`, plan:407-424), §6
A6/A7 (plan:702-708), §8 decision 2 (plan:760-762: **decided, an `n: usize` entry parameter**) and
decision 8 (plan:777-779: `ANY`/`ALL` keep refusing a value bound).
**Evidence run:** `/home/user/scratch/a5-probe/tests/a6_macro_tier.rs`, at `7e9531e`: the macro
tier's optimizer (`pixelflow_search::Saturate::macro_tier()`, what `kernel!` runs) takes a fold
with a uniform `hi` (`n` ∈ {0, 1, 3, 7}) and a triangular fold (`Σ_{i<4} Σ_{j<i}`), answers
`Changed` for both, relinks, and the baked values are rustc's (63 at `n = 7`, `X = 3`; 26 for the
triangle). So nothing in the optimizer stands between A6 and saturating a value-bounded entry at
expansion.

---

## 0. What A6 is, and what it is not

**A6 is** (plan:409-416, plan:702-705): a `kernel!` fold's bound is a `usize` expression typed in
the enclosing scope, and each bound is one of

| Spelling | Its value | IR bound | Variance |
|---|---|---|---|
| an integer literal | itself | `Bound::Lit` | node data |
| a `usize` const, or `+ - * /` over literals and consts (`LO + N`) | evaluated at expansion, checked | `Bound::Lit` | node data |
| an enclosing fold's index `i` (a triangular `0..i`) | that index | `Bound::Value(placeholder Var)` | the enclosing binder's bit |
| an entry parameter `n: usize` (decision 2) | the call's `n` | `Bound::Value(Uniform leaf)` | `CONST` |

and refused: `X`, `Y` or any other `f32` (by type: L3); arithmetic with a value bound in it
(`0..i + 1`, `n - 1..n`: `BoundArithmetic`, plan §1.4, until T6); a value bound under `any`/`all`
(`BoundSort`, decision 8); a known end past 2²⁴ (as today); a reversed pair of known ends (L2's
literal spelling, as today). The fold's own index cannot be named by its bound (it is not in scope
when the bound is typed). `n: usize`'s value is refused past 2²⁴ where the host writes it (plan
§1.4): in the host function and in `Args::write_into`.

**A6 is not:** a discrete-kernel argument or an `Extent` leaf (B1, B5; decision 1); bound
arithmetic (T6); a value bound under `any`/`all` (decision 8); a count in a helper's or the
closure form's parameters (§11, open, recommended out); the lattice's own extent (plan §4,
"Deferred").

---

## 1. Sema (`pixelflow-compiler/src/sema.rs`)

**The count, again an enum, with a different second variant.** A5 collapses
`Count::{Known(u64), Structural(TokenStream)}` to `u64` (A5 spec §2.3.1). A6 makes
`UsizeScope::eval_usize` return

```rust
pub(crate) enum Count<'a> {
    /// Evaluated here, each operation checked: a literal, a `usize` const, or
    /// `+ - * /` over them.
    Known(u64),
    /// A name that is no const: a fold's index or a count parameter, whose
    /// value is the program's, bound per trip or per call. A leaf (plan §1.4).
    Name(&'a Ident),
}
```

`Structural` held a Rust expression that rustc evaluated per instantiation, combinable by `+`; a
`Name` is a leaf and **combines with nothing**: `eval_usize` of a `Binary` with a `Name` on either
side refuses with `BoundArithmetic`'s reason:

> "`i + 1` as a range's bound: a bound that is a value is a single name until bounds are typed
> integers, since `f32` arithmetic on a count can round into the domain (§1.4 of {BOUND_PLAN})"

`RangeScope::usize_named` (A5 keeps it with one field, `consts`) returns `Known` for a `usize`
const and `Name(name)` for anything else (A5 refuses there with "`{name}` is not a `const`").
`ConstEvaluator::usize_named` never yields `Name` (an unknown name is "cannot find a `const`"), so
its callers keep treating the result as a number (`Count::Known` or an unreachable arm refused by
name).

**`range_bounds` returns `[Count; 2]`** (A5: `Range<u64>`). The reversed-range refusal applies
when both are `Known`. A `Known` end beside a `Name` is checked alone against 2²⁴ in lowering (§2).

**`type_of_fold` types each `Name` end in the enclosing scope** (it already calls `range_bounds`
before `push_scope`, `sema.rs:916-918`, so the fold's own index is not in scope and L3's "not its
own binder" holds by scope). The name's symbol must be `Ty::Usize` and of kind `Index` or
`Parameter`:

| Symbol | Refusal |
|---|---|
| `Intrinsic` (`X`, `Y`) | "mismatched types: a range's bound is a `usize`, and `X` is the coordinate: a bound reads no coordinate (L3 of {BOUND_PLAN})" |
| any other `Ty::F32` / `Ty::Bool` / record / kernel | "mismatched types: a range's bound is a `usize`, and `{name}` is a `{ty}`" |
| none | the existing "cannot find `{name}`" |

**The reduction**: `Reduction::Any`/`All` with a `Name` end refuses ("an `any` or `all` takes
constant bounds: a mask has no out-of-domain value yet, decision 8 of {BOUND_PLAN}"), because
`push_reduce` refuses it (`BoundSort`) and no stage may accept what a later one refuses.

**`n: usize` as a parameter.** `Items::param_type` (`sema.rs:601-650`) calls `Ty::of_a_value`,
which refuses `usize` (`:151-173`). A6 accepts `Ty::Usize` for `Role::Entry` only; a helper's
`usize` parameter and the closure form's stay refused (§11). `register_parameter` binds it as
`Symbol { Parameter, Usize }`; `resolve_ident` already refuses it as a value
(`a_count_is_not_a_value`) and `type_of_cast` already accepts `n as f32` (`names_a_count`). A
record field stays `f32`.

**Messages to update** (A5 has already dropped "structural" from them): `RangeScope::usize_named`'s
"a range's bounds are constant… ranges are constant (§1.5 of PLAN)" and `not_a_count`'s; the
parser's `const` generic help (A5 spec §1.1) gains "or, for a count that changes per call, an
entry parameter `n: usize`".

## 2. Lowering (`pixelflow-compiler/src/lower.rs`)

**Bounds before the fold opens** (plan:414-415). `lower_fold` (A5's shape: `literal_range(
range_bounds(…)?)?` then `open_fold(depth, monoid, range)`) becomes:

1. `range_bounds` → `[Count; 2]`;
2. each end to a `Bound<S::Term>`: `Known(k)` → `Bound::Lit(k)`; `Name(n)` → `Bound::Value(term)`
   where `term` is `n`'s binding in `self.frame.locals`: a fold's index (`Binding::Index(t)`, its
   placeholder `Var` in the enclosing fold's arena copy) or a count parameter (its `Uniform` leaf;
   §3 for the binding). Lowering refuses any other binding by name, without relying on sema (the
   crate's rule, `lower.rs:946-952`);
3. both `Lit` → `literal_range(lo..hi)` (A5 keeps it); one `Lit` → that end alone against
   `KnownRange::admits(&(0..k))`, so the front end refuses past 2²⁴ in its own words before
   `FoldHeader::new` panics (`fold.rs:658`, "a literal beside a value bound");
4. the depth check (unchanged), then `self.site.open_fold(depth, monoid, [lo, hi])`.

The IR already supports exactly this: `ExprArena::close_fold`'s doc (`arena.rs:1824-1828`) says "a
value bound's term is built in this arena (the copy) before the fold closes, and an id bound before
the fold opened means the same node in the copy … so a front end can lower a bound before it opens
its fold". `close_over` discovers a literal `Value` term into `Lit` (`Bound::discover`), roots the
value bounds beside the body, renames only this fold's placeholder (an enclosing fold's stays, and
its variance is `CONST` until its own `close_over`, A3), and `push_reduce` checks L3
(`BoundVaries`) and the sort (`BoundSort`) behind sema.

**`Site::open_fold(&mut self, depth: usize, monoid: Monoid, bounds: [Bound<Self::Term>; 2]) ->
Self::Term`** (A5: `range: Range<u64>`). `Expansion` stores `FoldTerms { monoid, lo, hi }` (A5
already stores a `FoldTerms`, so only how `lo`/`hi` are built changes). `Staged::fold_terms` emits
`Bound::Lit(#k)` or `Bound::Value(#ident)`, `#ident` the `__t…` variable that holds the term
(lowered before `open_fold`'s statement, so in scope). `close_fold` is unchanged.

## 3. `n: usize`: a uniform, a host argument, an `Args` field

**The program side.** An `n: usize` entry parameter is one uniform scalar, declared in parameter
order with the others (`AnalyzedKernel::parameters`, `sema.rs:289-302`), read as a bound or as
`n as f32` (the `Uniform` leaf itself, as an index's `as f32` is its `Var`). `Parameter` (and so
`Scalar`) needs to say it is a count: `Parameter { name, record, count: bool }`, set from
`Ty::from_syn(&param.ty) == Some(Ty::Usize)`. `lower_entry` binds a count parameter as
`Binding::Index(leaf)`, renamed `Binding::Count` with its doc widened to "a `usize`: a fold's index
or a count parameter; read only as `x as f32` or as a bound" (one variant for one behaviour; a
second variant would be matched identically at every use).

**The host side** (`emit.rs`). `argument(scalar)` (`emit.rs:176-182`) emits `n as f32` for a count:
the uniform's default in `arena_to_tokens` and `Staged::uniform`, and the value `write_into`
writes. The host function `pub fn f(n: usize, …) -> Kernel` first refuses an `n` past 2²⁴:

```rust
::core::assert!(
    ::pixelflow_core::__macro::ir::KnownRange::admits(&(0..(n as u64))),
    "kernel!: `n` is {}, and a count is at most 16777216 (2^24): a fold's index is an \
     `f32` lane, which names every integer only that far",
    n,
);
```

(`KnownRange::admits(&(0..n))` is plan §1.4's "`KnownRange::admits` turned into a predicate on one
count value"; it is public, `fold.rs:1235`, and re-exported at the IR's root; no new IR API.) A
panic, as `Kernel::over` panics on a literal past 2²⁴ (`fold.rs:1617-1620`): the host function is a
builder. Without it `16777217 as f32` is 2²⁴, in the domain, and the fold silently runs one trip
short (plan §1.4: "2²⁴ + 1 can be refused only there").

**`Args::write_into`** checks every count field before writing anything (as an arity mismatch
writes nothing, `binding_times.rs:264-308`), then writes `self.n as f32`. Its error type is open
(§11, D-A6-1); the recommendation is (a):

- (a) `write_into` returns `Result<(), WriteRefused>`, a new `pixelflow-core` enum
  `WriteRefused { Arity(ArityMismatch), Count(CountOutOfDomain) }`, `CountOutOfDomain { parameter:
  &'static str, count: u64 }` with a `Display` naming 2²⁴. Errors stay values on the per-call path.
- (b) keep `Result<(), ArityMismatch>` and panic on a count (a builder's refusal); one error type,
  two failure modes on one call.
- (c) type the field: `n: Count`, a `pixelflow-core` newtype whose constructor refuses past 2²⁴, so
  `Args` cannot hold one ("when you extend a type's meaning, extend its type"); the host function's
  parameter then differs from the declared `usize`.

All three change `pixelflow-core`'s public surface (CLAUDE.md asks for that to be said and
approved). `write_into` has no caller outside the compiler's tests
(`grep -rn write_into --include=*.rs .`), so none breaks production.

**One program for every `n`** (L9): the bound is `Bound::Value(Uniform)`, keyed by its dense slot
(A4's marker), so `f(3)` and `f(7)` share a key and a compiled region.

## 4. Emission and the macro tier

Nothing new in `arena_to_tokens`: A4 taught `node`'s `Reduce` arm to emit a value bound's children
(`attach([…])`, `emit.rs:596-612`), the branch A5 keeps as the whole arm. `reached` walks
`arena.children`, which yields value bounds (A2b), so a bound's term is emitted before its fold.

An `n: usize` entry that takes no kernel is **saturated at expansion** like any other (A5 removed
the only exemption): the probe shows `macro_tier` takes value-bounded folds at `7e9531e`.
Risks for A6's spec to pin, not to design around:

- **A4's loud hole** (plan:661-665): extraction picks one member per class, so a bound class whose
  chosen member reads a coordinate is refused at materialization (`BoundVaries`), which at
  expansion is a proc-macro panic, i.e. a compile error naming the law. A uniform's class has no
  such member unless a rule spells `n` as `n + X − X`; no rule does today.
- `relink` (`emit.rs:165-167`) restores the declaration order after extraction drops an unread
  uniform: a count read only as a bound is read, so it is kept.
- The guard is legalize's at bake, not the macro tier's: an expanded entry carries no guard.

## 5. What a bound reads, by law

- L3 by type: a `usize` leaf is a fold's index or a count parameter; neither is a coordinate, and
  nothing converts an `f32` to a `usize` (the parser refuses `as usize`, `parser.rs:1224-1234`), so
  no `kernel!` bound reads `X` or `Y`. `push_reduce`'s `BoundVaries` is behind it.
- L2: a known reversed pair is refused at expansion; a pair with a value is decided per call by the
  guard's `run` test (A4).
- L5: an `n` in `[0, 2²⁴]` is written; nothing else reaches the program from `kernel!`'s typed
  surface. A raw `set_declared` can still write any `f32` into the slot, and the guard makes the
  fold NaN for one that names no count (A4): that path is untyped by design.
- §1.4's leaf rule: every `kernel!` value bound is a leaf by construction (§1).

## 6. Helpers, the closure form, kernel-typed entries

- A **kernel-typed entry** (`Staged`) may take `n: usize` beside its kernels: `Staged::uniform`
  declares it with `n as f32`, and its folds emit `Bound::Value(#ident)` (§2). It has no `Args`
  record; the host function's assert refuses past 2²⁴.
- A **helper** with an index bound works as today plus A6 (a helper's fold over its own enclosing
  fold's index). A helper parameter `n: usize` is open (§11, D-A6-2).
- The **closure form** `kernel!(|n: usize| …)` is open (§11, D-A6-3): it has no `Args` record, its
  rebinding is by position through `set_declared` with `f32`s (`lib.rs:193-195`), so a per-call
  count written there is untyped; and `emit_closure` hardcodes `f32` parameters
  (`emit.rs:202-207`).

## 7. Tests A6's spec will need

### 7.1 `rustc_is_the_oracle.rs` (plan:703-704)

Beside A5's `a_count_in_a_ranges_const_arithmetic_folds_as_rustcs_does`:

```rust
kernel! {
    /// Σ_{i<4} Σ_{j<i} (X·j + i): an inner fold bounded by the outer index.
    pub fn triangle() -> f32 {
        (0..4).map(|i| (0..i).map(|j| X * (j as f32) + (i as f32)).sum::<f32>()).sum()
    }

    /// Σ_{i<n} X·i, the count a per-call value.
    pub fn sum_to(n: usize) -> f32 {
        (0..n).map(|i| X * (i as f32)).sum()
    }
}

fn rust_triangle(x: f32) -> f32 {
    (0..4).map(|i| (0..i).map(|j| x * (j as f32) + (i as f32)).sum::<f32>()).sum()
}

fn rust_sum(n: usize, x: f32) -> f32 {
    (0..n).map(|i| x * (i as f32)).sum()
}
```

`triangle` against `rust_triangle` at `FOLD_SAMPLES`, and 26 at `X = 3`; `sum_to(n)` against
`rust_sum(n, x)` for `n` ∈ {0, 1, 3, 7} at `FOLD_SAMPLES` (`n = 0`: `+0.0` against rustc's `-0.0`,
which `==` does not see, as `a_fold_over_an_empty_range_is_the_identity_as_in_rustc` says). Every
sum is of small integers times quarters, exact on every tier. The probe's `a6_macro_tier.rs` is the
same two kernels through the builders and `macro_tier`.

### 7.2 `binding_times.rs`

- **One program for every count**: `sum_to(3)` and `sum_to(7)` share `program_key` and
  `code_at_frame` (the opposite of the pin A5 deletes, `ramp::<4>` ≠ `ramp::<5>`).
- **Rebinding**: a program compiled from `sum_to(0)`, rebound with `SumToArgs { n: 7 }`, gives the
  pixels of baking `sum_to(7)`.
- **The write refuses 2²⁴ + 1 and writes nothing**: `SumToArgs { n: 16_777_217 }.write_into(&mut
  block)` is the count error and `block.values()` is unchanged; `n: 16_777_216` is accepted (plan
  A6 gate).
- **The host function refuses 2²⁴ + 1**: `#[should_panic(expected = "at most 16777216 (2^24)")]` on
  `sum_to(16_777_217)`.

### 7.3 `fold_is_kernel_over.rs`

A `kernel_raw!` items-form entry `pub fn sum_to(n: usize) -> f32 { (0..n).map(|i| X * (i as f32) +
Y).sum() }` is `Kernel::over_values(Monoid::SUM, &Kernel::constant(0.0), &Uniform::new(_).kernel(),
|i| x().mul(i).add(&y()))` by `canonical` (one uniform each, the same `close_over`): the syntax and
the builder are one program for a per-call count too. A triangular one against `Kernel::over` with
`over_values` inside, likewise.

### 7.4 Unit tests that A5 keeps and A6 rewrites

- sema `a_range_bound_that_is_not_constant_is_refused` (`sema.rs:2725-2773`): the triangular case
  (`(0..i)`, case 4) moves to an accepted list; `|n: f32| (0..n)` now refuses as an `f32` bound;
  `(0..X)` as L3; `let n = 4.0` as an `f32`; the float literal, `X.floor()`, `(1 < 2)` and `-1`
  cases keep their refusals.
- sema `a_usize_is_not_a_declared_value_type` (`:2946-2972`): `pub fn f(n: usize)` becomes
  accepted; the `-> usize` and `let n: usize` cases stay refused; a helper's `n: usize` (and the
  closure's, per §11) join the refused list.
- New: `BoundArithmetic` (`(0..i + 1)`, `(0..n - 1)`, `(LO + n..n)`), `any`/`all` over `(0..n)`,
  a known end past 2²⁴ beside a value, lowering's own refusal of a non-count name.

### 7.5 Gates

byte_probe empty, both tiers (no production kernel has a value bound); gates 1b, 2, 2x identical
(the font programs have no fold in `kernel!`); `tests/fold_is_kernel_over.rs` green (plan:705).

## 8. Docs A6's spec will need

The plan's A6 bullet (plan:702-705) still says "Waits on decision 2, the spelling"; decision 2 is
decided (plan:760), so A6 rewrites the bullet and adds its Done line. The-language-is-kernel §1.3's
`usize` row (A5 leaves "a fold binder or a count (a literal or a `usize` const)") gains "or a
per-call count, an entry's `n: usize`". `lib.rs`'s `# Folds` and `# Binding times` (A5 spec §6.5)
gain the per-call count and its `Args` refusal. BACKLOG C8's host-write half (plan §5, "A6 (host
write)"). Ontology: `Structural parameter (retired)` already points at A6 (A5 spec §6.6); `Binding
time`, `Range (trip count)`, `Fold (Reduce)` gain the `kernel!` spelling. `pixelflow-compiler`'s
intent.md: Owns (a count is a literal, a `usize` const or an entry's `n: usize`) and Enforced by.

---

## 9. A7: empty, folded into A5

The plan's A7 (plan:706-708) deletes the macro-tier template exemption (`emit.rs:154-156`) in its
own CL "so the byte change is measured, not a side effect". A5 must delete it:

- **Its condition is not expressible after A5.** It is `if !entry.structural.is_empty()`, and A5
  deletes `FnItem::structural`. Nothing else marks a template: after A5 every entry's folds are
  `Lit` pairs over `Fold::new`, the shape every non-template entry always had.
- **Its reason leaves with A5.** The `0..0` placeholder an `EmptyFold` would read as an empty
  range lives in `Holes`, which A5 deletes.
- **Nothing else keeps it alive.** A6's value-bounded entries need no exemption (the probe:
  `macro_tier` takes them and keeps their meaning); kernel-taking entries are `Staged`, outside
  `entry_arena`, by a different mechanism; `DwrtFree` declines a `Dwrt`-carrying arena by its own
  test.
- **The byte change it wanted measured is test-only.** At `7e9531e` it exempted `ramp`, `rings`,
  `lower_case::span`, `swept`, `mean_index` (from `macro_tier`), `grid` (`kernel_raw!`, no
  difference) and three emit unit-test entries (A5 spec §3's table). No production entry has a
  generic: the one production `kernel!` block, `pixelflow-graphics/src/fonts/loop_blinn/program.rs:31-141`,
  has none, and no other crate, example or bench declares one in a `kernel!` block. A5's gates
  (byte_probe, the 29 font programs' keys, digests and telemetry) measure that no production byte
  moves; the four surviving test kernels are now saturated, verified at `7e9531e` on AVX2 and
  AVX-512 by `/home/user/scratch/a5-probe/tests/probe.rs`.

So **A7 is empty**, and A5 rewrites the plan's A7 bullet to "Folded into A5" with this reason (A5
spec §6.1). The A-series ends at A6.

---

## 10. What A5 must keep, delete or reshape for A6

| A5's act (A5 spec) | Why it suits A6 |
|---|---|
| **Keep** `RangeScope` with one field, `consts` (§2.3) | A6's `usize_named` returns `Name` for a non-const instead of refusing; the scope stays consts-only, because which names are leaves is each stage's own scope's question (sema's symbols, lowering's locals) |
| **Keep** `UsizeScope`, `CheckedOp`, `range_bounds` (returning `Range<u64>`) | A6 changes their return type to `Count` / `[Count; 2]`; the evaluation of known bounds and its refusals are unchanged |
| **Delete** `Count`, collapse to `u64` (§2.3.1) | A6's `Count::Name` is a leaf, not a token expression rustc evaluates (`Structural`); keeping the old enum would carry A5's meaning into A6's variant |
| **Keep** `literal_range` (`Range<u64>` in, §2.4) | A6 calls it for a known pair, and its single-end check beside a value uses the same `KnownRange::admits` |
| **Reshape** `Site::open_fold` to `Range<u64>`, infallible (§2.4) | A6 reshapes the argument to `[Bound<Self::Term>; 2]`; infallible either way (bounds are lowered before it) |
| **Reshape** `Expansion.open` to `Vec<(OpenFold, FoldTerms)>`; delete `ExpansionFold` (§2.4) | `FoldTerms` already holds `Bound<ExprId>` ends: A6 only builds `lo`/`hi` differently |
| **Reshape** `Staged::fold_terms(monoid, Range<u64>)` without `let __range` (§2.5) | A6 emits `Bound::Lit(#k)` / `Bound::Value(#ident)` per end, which the simpler shape takes directly |
| **Keep** `node`'s `Bound::Value` children branch, deleting only the hole branch (§2.5) | A6's emission path (A4's) |
| **Delete** `EntryContext`, `Lowered`; `node(arena, id)` (§2.5) | A6 needs no context in `node`; a count's `as f32` is in `argument(scalar)`, which builds `arguments` |
| **Delete** `FoldTerms::stride` (§2.7) | a value bound requires stride 1 (`FoldHeader::new` refuses otherwise); A6 never sets one |
| **Keep** `Ty::Usize`, `SymbolKind::{Index, Parameter}`, `register_index`, `Binding::Index`, `a_count_is_not_a_value`, `type_of_cast`'s `names_a_count` (§2.9) | A6 accepts `n: usize` as a `Parameter` of `Ty::Usize`; the value refusal and `as f32` already handle it; `Binding::Index` becomes `Binding::Count` |
| **Keep** `type_of_fold`'s order: bounds typed before `push_scope` (`sema.rs:916-922`) | the fold's own index cannot be a bound by scope (L3) |
| **Keep** `a_range_bound_that_is_not_constant_is_refused` and `a_usize_is_not_a_declared_value_type` unchanged (§5.1) | A6 rewrites them case by case (§7.4); A5 must not pre-empt the decisions |
| **Add** `BOUND_PLAN` (§1.1) | A6's refusals (L3, `BoundArithmetic`, `BoundSort`, 2²⁴) cite it |
| **Write** the parser's `const` generic help as "a literal, or … a `usize` const" (§1.1) | A6 appends "or, for a count that changes per call, an entry parameter `n: usize`"; A5 must not name a spelling that does not exist yet |
| **Delete** the template exemption (§3) | A6's entries are saturated at expansion; no exemption is wanted |

## 11. Decisions A6 cannot make alone (for JP or review)

- **D-A6-1. `write_into`'s refusal of a count past 2²⁴**: (a) a `WriteRefused` enum in
  `pixelflow-core` (recommended: errors stay values on the per-call path); (b) a panic beside the
  arity `Result`; (c) a `Count` newtype field. Each changes `pixelflow-core`'s public surface.
- **D-A6-2. A helper's `n: usize`.** Recommended: refused in A6 (no current need; a helper reads a
  count's value as `n as f32` today). The alternative lets a helper fold over a caller's per-call
  count, binding the parameter to the caller's `Binding::Count`; it is the natural home of B5's
  per-piece helper if B5 needs one.
- **D-A6-3. The closure form's `n: usize`.** Recommended: refused ("a per-call count is an entry
  parameter of the items form, whose `Args` record refuses an out-of-domain count"). The
  alternative accepts it with a host-side assert, `emit_closure` using declared types, and an
  untyped positional rebind.
- **D-A6-4. The host function's refusal**: a panic (recommended, as `Kernel::over` does) or a
  `Result<Kernel, …>` return, which would change every count-taking entry's signature.
