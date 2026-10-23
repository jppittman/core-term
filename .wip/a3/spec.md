# A3 spec: value bounds in the IR and the e-graph

Step A3 of `docs/language/2026-10-20-a-bound-is-a-value.md` (the plan). Ground truth is
`3bfdaded` (main: #1446, the fold header's counts stored in 32 bits, on #1443, A2b). Every
`file:line` below is at that commit unless it says "plan:" (the plan at the same commit). Plan
sections read: §0–§4 in full, §5's rows marked A3, §6's A2b, A3 and A4 bullets (plan:537-586),
and §8 in full (decision 2 = `n: usize` in `kernel!`, which is A6; decision 5 = `FactorFold`
factors under a value bound and declines only a known empty range; decision 7 = public API
approved; decision 8 = mask-sorted folds keep refusing a value bound).

The rulings on the previous critique (B1, B2, N1–N10) are this spec's own text: each is stated
where it applies, marked with its name so a reader can find it.

What #1446 changed underneath the previous spec, and what this one is written against:
`FoldHeader` stores `lo`, `hi` and `stride` as `u32` through `field` (`fold.rs:757-760`) beside
`lo_held`/`hi_held: Held` (`fold.rs:485-514`, `:562-570`); every reader returns `u64`; a value
bound is stored `Held::Value` with count 0 (`Held::hold`, `fold.rs:500-505`); `FoldHeader::store`
is the one writer (`fold.rs:596-609`); and `FoldHeader::new` panics with `VALUE_BOUND` on any
value bound (`fold.rs:582-587`).

Contents: §0 what A3 is and is not · §1 the laws as code (§1.1 the table, §1.2 refusal order,
§1.3 the literal path is unchanged) · §2 new items in `fold.rs` · §3 the arena and variance ·
§4 `Kernel::over_values` and `Kernel::over` · §5 L7 and every walker, with the classified
`Reduce {` table · §6 the e-graph · §7 pricing · §8 codegen · §9 the compiler · §10 tests ·
§11 order of work and the grep-gated checklist · §12 docs · §13 judgement calls.

## 0. What A3 is, and what it is not

**Is.** A fold's bound may be a term (`Bound::Value`) in the arena (`ExprNode::Reduce`) and in
the e-graph (`ENode::Reduce`). Constness is discovered where a node is built, never declared
(`Bound::discover`). The laws L2, L3, L4, L6, L7 and L8 are enforced, and so are §1.4's domain,
sort and arithmetic refusals. L3 is computed by the arena's one per-node variance transfer
function, run from the bound and memoized (N1). `Kernel::over_values` is the builder
(`pub(crate)`); `ExprArena::close_over` discovers a literal bound before it copies anything (N2).
`push_reduce` keeps a literal naming no count as a value bound (N4). The rules are gated on known
ranges, `KnownBound` makes a range known, `FactorFold` factors under a value bound, and the price
of a fold whose range is not known is the count its bound's chosen literal names, else
`Fold::ESTIMATED_TRIPS` (N3).

**Is not.**
- **Codegen (A4).** `arena_to_schedule` keeps refusing a value bound by name
  (`pixelflow-codegen/src/program/lower.rs:47-52`, `known`; the `Reduce` arm `:334-348`). There
  is no operand or carve change, no guard, and no `Trips`.
- **The key (A4).** `FoldHeader::to_bits` keeps refusing a value bound (`fold.rs:712-715`), so
  `canonical` (`key.rs:454-462`), `KernelKey::of`, `Kernel::by_ref`, the JIT cache and the
  runtime tier's optimizer cache refuse a value-bounded kernel by the same name. A2b's plan text
  had `canonical` writing a marker; that is A4's now (J10).
- **`kernel!` value bounds (A6).** `kernel!` keeps literal bounds, and its two `close_fold`
  call sites (`pixelflow-compiler/src/lower.rs:497-522`; `emit.rs:701-732` with `:815-844`)
  change only mechanically (§9).

**Visibility, stated plainly.** `ExprArena::push_reduce` is `pub` (`arena.rs:944`) and accepts a
value bound after A3, and `Kernel::from_parts` is `pub` (`kernel.rs:331`). So a value-bounded
kernel is publicly constructible in A3 through the arena. `pub(crate)` on `over_values` hides the
spelling, not the capability. Decision 7 (plan:661-663) makes `over_values` public in A4; A4
does not open the door, because A3 already has. The new public items are listed in §2 and §3
(`BoundEnd`, `BoundRefusal`, `Bound::discover`, `Fold::admits_bound`, `Fold::ESTIMATED_TRIPS`,
`FoldHeader::priced_trips`, `Variance::of_var`, `FoldTerms`, and the changed signatures of
`ExprArena::close_over`, `ExprArena::close_fold` and `CostModel::fold_cost`). Nothing new is
re-exported from `pixelflow-ir/src/lib.rs:35`: the items are reached through `pixelflow_ir::fold`
and `pixelflow_ir::arena`, so the crate's intent Contract line (`pixelflow-ir/intent.md:41-47`)
stays true.

**Gate** (plan:566-575): byte_probe is empty, and there are pins for L2, L3, ANY/ALL/SEQ, a
non-leaf bound at `over_values`, L4, L7, L8, and a triangular fold under `HalveFold`. The
workflow adds: the 29 font programs' keys and digests and their saturation telemetry, the
extraction fields included, unchanged; total instructions on the cold compile of the 29 programs
within +0.5% of base. §10 lists the pins; §11 the order and the greps.

---
## 1. The laws, as the code that enforces them

### 1.1 The table

| Law (plan) | Where | Check | Refusal | Panic or Result |
|---|---|---|---|---|
| §1.4 domain (plan:120-138; L5 plan:190-191) | `Fold::count` (new, `pub(crate)`), read by `Bound::discover` (new, `pub`) | a literal names a count iff it is `Lit::Real(v)` with `v` a non-negative integer `n` and `Fold::is_exact(n)`, the one `n ≤ EXACT_BOUND` function (N10) | at the **user surfaces only** (N4): `Kernel::over_values` → `BoundRefusal::BoundOutOfDomain { end, literal }`; `Kernel::over`'s literal `Range<u32>` → `KnownRange::new`'s existing "ends at most at 16777216 (2^24)" (J13); `kernel!` at A6 | panic at the surfaces. Everywhere else the literal **stays a value bound** — `push_reduce`, `close_over`, `insert`, extraction's `embed` and every copier — so a literal a rule or the folder made never panics, and A4's guard makes the fold NaN (L5). `KnownBound` declines; the extractor prices it at the estimate. |
| §1.4 mask bound (N4) | `Bound::discover` | a value bound whose term is `Lit::Mask(_)`: a mask names no count | `BoundRefusal::BoundSort { end }`, everywhere | panic in `push_reduce`, `close_over`, `insert`; `over_values` refuses it in its own check with the same refusal; `KnownBound` declines (`.ok()?`); the extractor prices the fold at the estimate (§7) |
| §1.4 sort (plan:125-129, plan:257-258) | `FoldHeader::new` (`fold.rs:582`), its value arm | any `Bound::Value` and `!fold.monoid().takes_a_value_bound()` (ANY, ALL, SEQ) | `BoundRefusal::BoundSort { end }`, `end` the first value bound | panic. Reached from `push_reduce`, `ENode::reduce`, `insert`, plan replay (`graph.rs:1626`), templates (`graph.rs:1580-1588`) and `witness.rs:421`. |
| stride (plan:266-268) | `FoldHeader::new`'s value arm | any `Bound::Value` and `fold.stride() != 1` | not a `BoundRefusal`: a malformed node, "a fold's stride is not 1 only over a known range it divides" | panic |
| a literal beside a value | `FoldHeader::new`'s value arm | the literal end of a mixed pair must satisfy `Fold::is_exact` | "a fold's literal bound is at most 16777216 (2^24)" (no discovery or rule makes one: a hand-built node) | panic |
| §1.4 arithmetic (plan:140-154) | `Kernel::over_values` only (`kernel!`'s sema is A6) | the bound kernel's root `ExprData` is `Var`, `Const` or `Uniform` | `BoundRefusal::BoundArithmetic { end }` for `Op`, `Reduce`, `Ref`; `BoundRefusal::BoundNotANumber { end }` for `Param` and `Buffer` (N10: not `BoundArithmetic`) | panic |
| L2 (plan:174-179) | `Bound::discover` normalizes; `KnownRange::new` (`fold.rs:858-879`) keeps refusing a declared reversed pair | a pair made known **by this read** with `lo > hi` becomes `[lo, lo)`. A pair the caller passed as two literals is returned untouched | "runs forwards" (existing, `KnownRange::new`) | panic for a declared pair; none for a discovered one |
| L3 (plan:180-188) | `ExprArena::push_reduce` (every arena construction) and `EGraph::add` (backstop) | `Fold::admits_bound(end, reads)`: `reads ∩ (COORDS ∪ {b}) = ∅`. In the arena `reads` is `variance_from(arena, term)`, the per-node transfer function run from the bound and memoized (N1); in the e-graph, the bound class's fact | `BoundRefusal::BoundVaries { end, reads }` | panic in both |
| L4 (plan:189) | `variance.rs`'s `transfer` (both drivers) and `ENode::variance` (`node.rs:528-556`) | `(var(body) \ b) ∪ var(lo) ∪ var(hi)`; a literal bound contributes nothing; a known empty range is `CONST` (arena only, as today) | — | — |
| L6 (plan:192-195) | `PeelFold` (`fold_rules.rs:393`), `HalveFold` (`:435`), `EmptyFold` (`:480-481`): already gated at `3bfdaded`. `FactorFold` (`:499`) changes to decline a known empty range only (decision 5) | the rule reads `node.known_range()` | — | decline |
| L7 (plan:196-198) | `PlanBuilder::substitute` (`fold_rules.rs`) through L4's class fact; arena side, `close_over`'s rename. `ExprArena::substitute_vars_with` is **not** L7 (N5, §3.4) | substitutes into a nested fold's bounds always, and into its body unless the fold rebinds the slot. A class reached as a nested fold's value bound (a `Site::Bound`) is rebuilt through its first member that is no fold and whose own variance, over its children's facts, `Fold::bound_may_read` accepts, and so is every class under it; the plan declines when a class under a bound has none (round 4, F2, §6.8; polish round 1). Bodies keep the first member. The rebuilt bound is judged again where it lands: `plan_shape` asks the sink whether it `refuses` a fold before making it, and a bound that hash-conses into a class whose fact fails L3 declines the plan (closing round 2, J27). | — | the rule declines (`None`) |
| L8 (plan:199-202) | `KnownBound` (new rule, `fold_rules.rs`, last in `fold_rules()`) | the value bounds' classes hold literals naming counts, so the fold over those counts is added to the class | — | decline otherwise |

### 1.2 Refusal order

- **`push_reduce`**: discovery (`BoundSort` for a mask literal), then `FoldHeader::new`
  (`BoundSort` for the monoid; the stride; a literal beside a value; a literal pair's
  `KnownRange::new` checks), then L3 (`BoundVaries`, `lo` before `hi`). The cheap checks run
  first and the one walk last (J15).
- **`over_values`**: the user checks of `lo`, then of `hi` (`BoundArithmetic`, `BoundNotANumber`,
  `BoundOutOfDomain`, `BoundSort` for a mask), before the body closure runs; then `close_over`
  (discovery, the binder, the copy) and its `push_reduce` (`BoundSort` for the monoid when a
  value bound survived discovery; `BoundVaries`). The monoid is not checked up front: a pair of
  literals is legal under any monoid (J18).
- **`insert`**: discovery at the fold's visit (`BoundSort` for a mask literal), `FoldHeader::new`
  through `ENode::reduce` at its completion, then `EGraph::add`'s L3.
- **`EGraph::add`**: `ENode::reduce` already ran `FoldHeader::new`; `add` checks L3, `lo` before
  `hi`, before the graph changes.

### 1.3 The literal path is unchanged

Today every bound is a literal. Per site A3 changes on that path, why no byte, key, price or
saturation result moves (the model is the old critique's section (a)):

| Site | Why it is identical on an all-literal fold |
|---|---|
| `FoldHeader::new` (§2.9) | A literal pair takes today's `KnownRange::new` → `known` → `store` path, unchanged; the value arm is not reached. Same header, same `to_bits`, same messages. |
| `ExprArena::push_reduce` (§3.4) | `Bound::discover` returns a literal pair at once without calling its reader (T-IR-3 pins "not called"); `FoldHeader::new` as above; the L3 loop's two `let Bound::Value(..) else { continue }` find nothing, so no walk and no allocation; `intern` gets the same `children[..1]`. Node for node. |
| `ExprArena::close_over` / `close_fold` (§3.5) | With no value bound `roots == [body]`, and `from_arena_roots(self, &[body])` is `from_arena(self, body)` (`expr.rs:274-279`). The closed builder pushes `Var(binder)` first and the renamed body as today; `finish(&[body])` records entries without reordering (`dag.rs:545-558`); `to_arena_roots` pushes every DAG node in push order (`expr.rs:381`); then the same `push_reduce(fold, Lit, Lit, body)` interns the fold last, where today's DAG `Reduce` was pushed last. The tables are the same `Environment`. Same arena. Pinned by `a_fold_is_laid_out_as_the_builder_always_laid_it_out` (`kernel.rs:1278-1303`). |
| `lowest_free_binder(roots)` (§3.5, J8) | Neutral for every builder: a placeholder (64 and up) gives `Binder::from_var(..) == None`, and a slot `Var` in a well-formed body sits under the `Reduce` that binds it, whose slot is already marked. |
| `Kernel::over` (§4) | `FoldTerms { monoid, stride: 1, lo: Lit, hi: Lit }` into the `close_over` above. A reversed or past-2²⁴ range now panics at `push_reduce` inside `close_over`, not in the `fold_at` closure, with the same message. |
| `Variance::of_var`, `transfer`, `compute_arena_variance` (§3.1-3.2) | An index below 64 is `from_var` as today; only placeholders (64..128) change, `ALL` → `CONST`, and no term a production path builds holds one. A placeholder can still escape its builder (a closure that keeps a clone of the index), and variance cannot say so, so no check of closedness or placement may read it (round 4, F1, §3.1): the unit check (`runtime.rs`, `Program::of`) refuses by `free_index`; `arena_to_schedule` refuses a `Var` that is no fold's binder by name (`program/lower.rs`); `lower_dwrt`'s tabulation (`passes.rs`) and `nnue::factored`'s histogram ask coordinate dependence and a statistic, which a placeholder answers exactly. A literal fold's arm is `(var(body) \ b)` with nothing unioned, today's. |
| `free_index` (§3.3) | The fold arm unions the value bounds' free indices; a literal fold has none. |
| `ENode::variance` (§6.1) | Folds over `children[1..]`, empty for a literal fold. `var_variance` → `of_var`: equal below 64, and no e-graph holds a `Var` at 64 or above (`insert` reads closed arenas; template metavariables shifted by `B_METAVAR_OFFSET`, `template.rs:335`, are renamed by `canonicalize_vars` before a rule exists and are matched, never inserted). |
| `EGraph::add` (§6.2) | The backstop is skipped outright when `header.arity() == 1`: no fact is read, no class id moves. |
| `insert` / `inserted_class_count` (§6.3) | Discovery returns a literal pair at once, without projecting; the visit and the measure are today's. |
| `KnownBound` (§6.4) | Appended last to `fold_rules()`, which `RuleSet::runtime` appends after `all_rules` (`rules.rs:207-211`), so every earlier `RuleIdx` is kept. On a literal fold it returns `None` after reading the header's arity and nothing else. The budget counts commits only (`graph/runner.rs:1157`, inside `commit`, which a `None` never reaches), so applications, unions and iterations are unchanged, and stop reasons are unchanged except in the corner of the next row. The changed fingerprint reaches only `SaturationKey` (`optimizer.rs:638`), the key of the in-process `CACHE` (`runtime.rs:655`, filled through `runtime_optimizer`, `:627-631`); research binaries use `RuleSet::production`, which holds no fold rule. |
| `KnownBound`'s extra check point (§6.4; round-1 review) | `Saturation::sweep` checks `num_classes() > minted` before every rule's scan (`graph/runner.rs:797-801`), so the rule appended after `FactorFold` adds one check, right after `FactorFold`'s commit and partial rebuild. Base had none there. The scan holds `num_classes + Σ predicted` under the cap, and `predicted_growth`'s sum can under-count (`graph.rs:946-953`), so `FactorFold` can complete and still leave the graph past `minted`. On any earlier iteration both sides report `MintedCap` with the same iteration count (base from the loop-top check, `runner.rs:726`). On the last allowed iteration (`limits.iterations`) base falls out of the loop at its default `IterationCeiling`, and head reports `MintedCap` from the new check. Only the telemetry `stop_reason` label moves (`iteration_ceiling` becomes `class_cap`): the graph, applications, iterations and extraction are identical. The application allowance has no such corner: `FactorFold`'s own scan stops at `applications + queued >= cap` (`runner.rs`, `allowance_spent`), so it reports the budget itself. No gate kernel reaches the corner (28 of the 29 gate2 records stop at `class_cap` in 2 iterations, and `n=0` quiesces in 1). Putting `KnownBound` first would shift every `RuleIdx` (§6.4). |
| `FactorFold` (§6.6) | `is_some_and(KnownRange::is_empty)` agrees with today's `is_none_or(KnownRange::is_empty)` whenever the range is known. |
| `CostModel::node_op_cost`, `fold_cost` (§7) | `header.priced_trips()` is `range.len()` for a known range; `usize::try_from(trips)` and today's `range.len() as usize` agree on a 64-bit host. |
| the extractor's pricers and `below` (§7) | §7.5 proves the visit order and every price unchanged: `literal_twin` returns `None` on a literal fold after one match, `below` multiplies by the same length and its fold over the remaining operands meets none. The `chosen` table written at each settle is read only when a value-bounded fold is priced. |
| the `KNOWN_RANGE` and `VALUE_BOUND` texts (§2.11) | Panic messages only. |
| `kernel!`'s tail (§9) | `arena_to_tokens`' fold arm (`emit.rs:577-614`) is unchanged; the `close_fold` tokens change shape and build the same arena; no test snapshots them (no `__binder`, `fold_at` or `__range` outside `src/`). |

The arbiter is the gate set (§11): byte_probe on both tiers, the 29 font programs' keys and
digests, their saturation telemetry including the extraction fields, and the instruction count.

---
## 2. New items in `pixelflow-ir/src/fold.rs`

Everything here lives in `fold.rs`. `fold.rs` already reaches `crate::variance::Variance`
(`PLACEHOLDER_BASE`, `fold.rs:226`); `Variance` is imported for §2.6. Items are placed beside
what they extend: `BoundEnd` and `BoundRefusal` after `Bound`'s impl (`fold.rs:336`), the `Fold`
items in `impl Fold` (`fold.rs:373-470`), the `FoldHeader` items in `impl FoldHeader`
(`fold.rs:572-752`).

### 2.1 `BoundEnd` (new, `pub`)

```rust
/// Which end of a fold's range a bound is: what a refusal names, so the
/// message says which of the two the caller wrote wrong.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BoundEnd {
    /// The first bound: the index the fold starts at.
    Lo,
    /// The last bound: the index the fold stops before.
    Hi,
}

impl core::fmt::Display for BoundEnd {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Lo => "lo",
            Self::Hi => "hi",
        })
    }
}
```

### 2.2 `BoundRefusal` (new, `pub`)

```rust
/// Why a fold refuses a bound: each the law of
/// docs/language/2026-10-20-a-bound-is-a-value.md it breaks, and the end it
/// breaks it at.
///
/// Every constructor that refuses a bound refuses with one of these, and
/// panics with its `Display`: a refusal is a builder's error, and the message
/// names the law. Where each is refused is the plan's §1.4 and L3, and this
/// type's variants say no more than which.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BoundRefusal {
    /// A real literal that names no count — not an integer, or outside
    /// `0..=Fold::EXACT_BOUND` (L5). Refused at the user surfaces only
    /// (`Kernel::over_values`, and `kernel!` from A6); every other
    /// constructor keeps such a literal as a value bound, whose guard makes
    /// the fold NaN (A4), so a literal a rule made never panics.
    BoundOutOfDomain { end: BoundEnd, literal: f32 },
    /// A bound of a sort the fold cannot take (§1.4): a mask literal, which
    /// names no count, under any fold; or any value bound under `ANY`, `ALL`
    /// or `SEQ`, whose sorts have no out-of-domain answer (decision 8) or
    /// whose ranges stay constant (the lattice's). The fold is in the panic's
    /// context, not here: the mask case has no algebra to blame.
    BoundSort { end: BoundEnd },
    /// A bound that varies with a coordinate or with its own fold's binder
    /// (L3): `reads` is what it varies with.
    BoundVaries { end: BoundEnd, reads: Variance },
    /// A bound that is arithmetic, at a user surface (§1.4): until bounds are
    /// `Int` (T6), a computed bound is `f32` arithmetic that can round into
    /// the domain, so a user writes a leaf.
    BoundArithmetic { end: BoundEnd },
    /// A bound that is not a number at all: a builder's `Param` slot, filled
    /// before a kernel exists, or a `Buffer`'s address, read through a gather.
    BoundNotANumber { end: BoundEnd },
}
```

`Display`, verbatim (each message starts with the variant's name, which is what the tests'
`should_panic(expected = ..)` match):

```rust
impl core::fmt::Display for BoundRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const PLAN: &str = "docs/language/2026-10-20-a-bound-is-a-value.md";
        match *self {
            Self::BoundOutOfDomain { end, literal } => write!(
                f,
                "BoundOutOfDomain: a fold's {end} bound is {literal}, which names no count: a \
                 count is an integer in 0..={} (2^24), where an `f32` counter is exact (L5, §1.4 \
                 of {PLAN})",
                Fold::EXACT_BOUND
            ),
            Self::BoundSort { end } => write!(
                f,
                "BoundSort: a fold's {end} bound has a sort the fold cannot take: a mask names \
                 no count, and an ANY, ALL or SEQ fold takes literal bounds only (§1.4 and \
                 decision 8 of {PLAN})"
            ),
            Self::BoundVaries { end, reads } => write!(
                f,
                "BoundVaries: a fold's {end} bound varies with {reads:?}: a bound reads no \
                 coordinate and not its own fold's binder (L3 of {PLAN})"
            ),
            Self::BoundArithmetic { end } => write!(
                f,
                "BoundArithmetic: a fold's {end} bound is arithmetic: until bounds are `Int` (T6 \
                 of docs/language/2026-10-17-the-arena-is-typed.md) a bound is a leaf — a \
                 constant, a uniform or an enclosing fold's index — whose value is the count \
                 exactly (§1.4 of {PLAN})"
            ),
            Self::BoundNotANumber { end } => write!(
                f,
                "BoundNotANumber: a fold's {end} bound is a builder's slot or a table's address, \
                 not a number: a `Param` is filled before a kernel exists and a `Buffer` is read \
                 through a gather (§1.4 of {PLAN})"
            ),
        }
    }
}
```

No `core::error::Error` impl: nothing returns one through `?` across a crate boundary in A3.

### 2.3 `Fold::is_exact` (new, private): the one `n ≤ EXACT_BOUND` function (N10)

```rust
/// Whether an `f32` counter holds `n` exactly: `n ≤ EXACT_BOUND`. The one
/// statement of the domain's ceiling — an end (`KnownRange::admits`), a
/// literal (`Fold::count`), a stride (`assert_stride`,
/// `FoldHeader::from_bits`) and a literal beside a value bound
/// (`FoldHeader::new`) all ask it.
const fn is_exact(n: u64) -> bool {
    n <= Self::EXACT_BOUND
}
```

Callers, each replacing its own comparison: `KnownRange::admits` (`fold.rs:903-905`) becomes
`range.start <= range.end && Fold::is_exact(range.end)`; `assert_stride` (`:478-482`) asserts
`Fold::is_exact(stride)` with its message unchanged; `FoldHeader::from_bits` (`:744`) tests
`stride == 0 || !Fold::is_exact(stride)`; `Fold::count` (§2.4); `FoldHeader::new`'s value arm
(§2.9). After A3 `grep -n 'EXACT_BOUND' pixelflow-ir/src/fold.rs` shows a comparison only inside
`is_exact`; the other hits are the definition, docs, messages, the const assert (`:772`) and tests.

### 2.4 `Fold::count` (new, `pub(crate)`, N10)

```rust
/// The count `literal` names: a real that is a non-negative integer an `f32`
/// counter holds exactly ([`Fold::is_exact`]), or `None` — a mask, a fraction,
/// a negative, a NaN, an infinity, or an integer past 2²⁴.
///
/// `-0.0` is the integer `0`, as it is to a shift count (`ShiftCount::from_literal`).
/// Crate-private: another crate reads a count off a bound through
/// [`Bound::discover`], which is where a count is a bound.
#[must_use]
pub(crate) fn count(literal: Lit) -> Option<u64> {
    let real = literal.real()?;
    // NaN and the negatives fail this; `-0.0 >= 0.0`.
    if !(real >= 0.0) {
        return None;
    }
    // `as` truncates a fraction and saturates past `u64::MAX`, so the round
    // trip is exact for an integer and only for one.
    let count = real as u64;
    (count as f32 == real && Self::is_exact(count)).then_some(count)
}
```

### 2.5 `Bound::discover` (new, `pub`) and `Bound::read` (new, private)

In a new `impl<R: Copy> Bound<R>` block after `fold.rs:336`:

```rust
impl<R: Copy> Bound<R> {
    /// `lo` and `hi` with their constness discovered (plan §3.1: "constness is
    /// discovered, never declared"): a value bound whose term is a literal
    /// naming a count ([`Fold::count`]) is that count, and every other bound
    /// is returned as it was — a term that is not a literal, or a real naming
    /// no count, which stays a value for its guard to make NaN (L5).
    ///
    /// A pair this read makes known runs forwards: `[lo, hi)` with `lo > hi`
    /// is the empty range at `lo` (L2), as Rust's `5..3` is empty. A pair
    /// passed as two literals is returned untouched and `literal` is not
    /// called, so a reversed one stays `KnownRange::new`'s to refuse.
    ///
    /// `literal` reads a value bound's term: its literal, or `None` when the
    /// term is not one. The one reader of constness for every representation
    /// that builds a fold — the arena's `push_reduce` and `close_over` read a
    /// `Const` node, the e-graph's `insert` a `Shape::Const`, `KnownBound` a
    /// class's members, the extractor a class's chosen form.
    ///
    /// # Errors
    ///
    /// [`BoundRefusal::BoundSort`] when a value bound's term is a mask
    /// literal: a mask names no count.
    pub fn discover(
        lo: Self,
        hi: Self,
        literal: impl Fn(R) -> Option<Lit>,
    ) -> Result<(Self, Self), BoundRefusal> {
        if let (Self::Lit(_), Self::Lit(_)) = (lo, hi) {
            return Ok((lo, hi));
        }
        let lo = lo.read(BoundEnd::Lo, &literal)?;
        let hi = hi.read(BoundEnd::Hi, &literal)?;
        // A value was read, so if the pair is known now, this read made it so.
        Ok(match (lo, hi) {
            (Self::Lit(l), Self::Lit(h)) if l > h => (lo, Self::Lit(l)),
            pair => pair,
        })
    }

    /// This bound with its term read: a value whose term is a literal naming
    /// a count is that count.
    fn read(self, end: BoundEnd, literal: &impl Fn(R) -> Option<Lit>) -> Result<Self, BoundRefusal> {
        let Self::Value(term) = self else {
            return Ok(self);
        };
        match literal(term) {
            Some(Lit::Mask(_)) => Err(BoundRefusal::BoundSort { end }),
            Some(real) => Ok(Fold::count(real).map_or(self, Self::Lit)),
            None => Ok(self),
        }
    }
}
```

`Bound<R>` derives `Copy` when `R` is; every representation's `R` (`ExprId`, `EClassId`,
`HeadRef`, `u8` in tests) is `Copy`.

### 2.6 `Fold::admits_bound` (new, `pub`): L3 stated once

*Polish round 1:* the law is `Fold::bound_may_read(self, reads) -> bool` (new, `pub`, `#[must_use]`),
the same at both ends of a range, and `admits_bound` is `match self.bound_may_read(reads) { true =>
Ok(()), false => Err(BoundVaries { end, reads }) }`. A reader that only chooses among terms
(`Site::member`) asks the predicate and fabricates no `BoundEnd`; a reader that names a refusal
asks `admits_bound`. The code below is the first spelling.

```rust
/// L3: whether this fold may take a bound at `end` that varies with
/// `reads` — one sharing no bit with the coordinates, which would make the
/// trip count a function of the pixel, or with this fold's own binder,
/// which would make it a function of itself. An enclosing fold's binder is a
/// bound's to read (a triangular `0..i`), and a placeholder names no bit
/// ([`Variance::of_var`]).
///
/// The one statement of the law: the arena asks it of a bound's term
/// (`ExprArena::push_reduce`), the e-graph of a bound's class
/// (`EGraph::add`).
///
/// # Errors
///
/// [`BoundRefusal::BoundVaries`], with what the bound varies with.
pub fn admits_bound(self, end: BoundEnd, reads: Variance) -> Result<(), BoundRefusal> {
    let refused = Variance::COORDS.union(Variance::from_var(self.binder.var()));
    if reads.intersection(refused).is_const() {
        return Ok(());
    }
    Err(BoundRefusal::BoundVaries { end, reads })
}
```

### 2.7 `Fold::ESTIMATED_TRIPS` (new, `pub`)

```rust
/// The trips a fold is priced at when its range is not known: 32.
///
/// LLVM's `BranchProbabilityInfo` weights a loop's back edge 124 against its
/// exit's 4 (`LBH_TAKEN_WEIGHT`, `LBH_NONTAKEN_WEIGHT`), a back-edge
/// probability of 31/32, so a loop it knows nothing about runs
/// `1 / (1 − 31/32) = 32` times. Policy, not a measurement of these kernels:
/// the static model prices a fold before its bound has a value (plan §1.5),
/// and a ranking at 32 may differ from one at the real count (plan §4).
/// Nested folds whose ranges are not known multiply, as their trips do.
pub const ESTIMATED_TRIPS: u64 = 32;
```

### 2.8 `FoldHeader::priced_trips` (new, `pub`)

```rust
/// The trips this fold is priced at by a reader holding only its header:
/// its known range's length, or [`Fold::ESTIMATED_TRIPS`] when a bound is a
/// value. The extractor prices a fold whose value bounds were chosen as
/// literals through the fold over those literals, whose header this is
/// (`pixelflow-search`'s `ENode::literal_twin`).
#[must_use]
pub fn priced_trips(self) -> u64 {
    self.known_range()
        .map_or(Fold::ESTIMATED_TRIPS, KnownRange::len)
}
```

### 2.9 `FoldHeader::new` for a value bound, and `Monoid::takes_a_value_bound`

`Monoid` gains (in `impl Monoid`, `fold.rs:47-114`):

```rust
/// Whether a fold under this algebra may take a bound that is a value: one
/// whose sort has an answer for a bound outside the domain, NaN (plan §1.4).
/// `ANY` and `ALL` fold masks, which have none (decision 8, open); `SEQ` is
/// the lattice's, whose ranges stay constant (plan §4, "Deferred").
pub(crate) const fn takes_a_value_bound(self) -> bool {
    matches!(self.0, OpKind::Add | OpKind::Mul | OpKind::Min | OpKind::Max)
}
```

`FoldHeader::new` (`fold.rs:573-587`) becomes:

```rust
/// `fold` between `lo` and `hi`: the one place a fold is checked against its
/// bounds.
///
/// # Panics
///
/// On a literal pair `KnownRange::new` refuses (backwards, ending past
/// [`Fold::EXACT_BOUND`], a stride that does not divide the span), with its
/// messages. On a value bound the fold cannot take: under `ANY`, `ALL` or
/// `SEQ` ([`BoundRefusal::BoundSort`]); at a stride other than 1, which only
/// a known range it divides may have; or beside a literal past
/// [`Fold::EXACT_BOUND`].
#[must_use]
pub fn new(fold: Fold, lo: Bound<()>, hi: Bound<()>) -> Self {
    let (Bound::Lit(l), Bound::Lit(h)) = (lo, hi) else {
        return Self::with_a_value(fold, lo, hi);
    };
    Self::known(fold, KnownRange::new(l..h, fold.stride))
}

/// `fold` between `lo` and `hi`, at least one of them a value: what a value
/// bound asks of its fold, checked, and the header stored.
fn with_a_value(fold: Fold, lo: Bound<()>, hi: Bound<()>) -> Self {
    let end = match lo {
        Bound::Value(()) => BoundEnd::Lo,
        Bound::Lit(_) => BoundEnd::Hi,
    };
    assert!(
        fold.monoid.takes_a_value_bound(),
        "{}: {fold:?}",
        BoundRefusal::BoundSort { end }
    );
    assert!(
        fold.stride == 1,
        "a fold's stride is not 1 only over a known range it divides: {fold:?} from {lo:?} to {hi:?}"
    );
    for literal in [lo.lit(), hi.lit()].into_iter().flatten() {
        assert!(
            Fold::is_exact(literal),
            "a fold's literal bound is at most {} (2^24): {literal} — its index is an `f32` \
             lane, which names every integer only that far",
            Fold::EXACT_BOUND
        );
    }
    Self::store(fold, lo, hi)
}
```

The `stride` doc on `Fold` (`fold.rs:365-369`) already says what this checks.

### 2.10 The other `FoldHeader` docs (B1)

- `FoldHeader::store` (`fold.rs:596-597`): "The one writer of a header's fields: it checks
  nothing, and its callers ([`FoldHeader::known`], and `FoldHeader::new`'s value arm) have."
- `FoldHeader::to_bits`'s `# Panics` (`fold.rs:707-710`): "On a value bound, with
  [`VALUE_BOUND`]: the layout has no field for a child until A4 gives it a marker."

### 2.11 `VALUE_BOUND` and `KNOWN_RANGE` (texts change, names do not)

`VALUE_BOUND` keeps its prefix "a fold bound that is a value", which every `should_panic` that
names it matches (`fold.rs:1543`, `:1577`, `node.rs:713`, `insert.rs:549`, `arena.rs:2590`; after
A3 only the `to_bits` test and T-C-1 expect it):

```rust
/// The message the schedule and the bits refuse a value bound with, until
/// A4: the IR and the e-graph hold one, `arena_to_schedule` takes a literal
/// only, and so does [`FoldHeader::to_bits`] — and with it the canonical
/// key, `kernel!`'s tokens, the dumps and the corpora. One grep target for
/// the step that lifts it, as [`KNOWN_RANGE`] is for the steps that make an
/// unknown range reachable where one is read.
pub const VALUE_BOUND: &str = "a fold bound that is a value has no code and no key before A4 of \
     docs/language/2026-10-20-a-bound-is-a-value.md: the IR and the e-graph hold one, the \
     schedule and the bits take a literal";

/// The message of an `expect` on a range a site's folds always have. A site
/// that reads a range and has no answer of its own for an unknown one reads
/// it where every fold is literal: a lattice's fold (`SEQ`, which takes
/// literal bounds only), a `kernel!` fold before A6, a production glyph's
/// before B2, a schedule's (codegen refuses a value bound first, with
/// [`VALUE_BOUND`]), or a test's own literal fold. One grep target for the
/// steps that make the `None` reachable there.
pub const KNOWN_RANGE: &str = "this fold's bounds are literals: a lattice's fold, a kernel! fold \
     before A6, a production glyph's before B2, a schedule's, or a test's \
     (docs/language/2026-10-20-a-bound-is-a-value.md)";
```

Every `.expect(KNOWN_RANGE)` that survives A3 is one of those (N9): codegen `select.rs:172`
(a schedule's) and `program/lower.rs:329` (the lane fold, a lattice's); `passes/lattice.rs:222`,
`:227` (a lattice's) and its tests; the compiler's `emit.rs:591` (`kernel!`) and its tests;
`pixelflow-pipeline/examples/glyph_phase_split.rs:79` (a production glyph's, before B2); the
tests in `fold.rs`, `passes.rs`, `fold_rules.rs` (`:671`, `:1173`) and `cost.rs`'s test. The
pricing sites `cost.rs:402` and `extract.rs:1686` stop expecting one (§7).

### 2.12 The rest of `fold.rs` that A3 rewords (B1)

- The module doc (`fold.rs:13-16`) already says a `Reduce`'s children are its body and any value
  bound: unchanged.
- The test `a_value_bound_is_refused_by_name_until_a3` (`:1540-1547`) is replaced by T-IR-5–7;
  `a_value_bound_has_no_bits_until_a3` (`:1574-1582`) is renamed `a_value_bound_has_no_bits_until_a4`
  and builds its header with `FoldHeader::new`; `a_value_bound_is_not_the_literal_zero`
  (`:1549-1572`) builds with `FoldHeader::new` and its doc becomes "A value bound's count field
  is 0, and the header still tells it from the literal 0: how a bound is held is a field `Eq`,
  `Hash` and `Ord` read." (§10).

---
## 3. The arena (`pixelflow-ir/src/arena.rs`, `variance.rs`)

### 3.1 `Variance::of_var` (variance.rs, new, `pub`): one answer for a `Var` leaf

In `impl Variance` (`variance.rs:56-229`), after `from_var` (`:83-111`):

```rust
/// What a `Var` leaf whose index is `index` varies with: its own bit for a
/// coordinate, a retired axis or a binder slot (`0..VARIABLES`,
/// [`Self::from_var`]); nothing for a placeholder, an enclosing fold's index
/// before `ExprArena::close_over` chooses its slot (the
/// [`Placeholder::COUNT`](crate::fold::Placeholder::COUNT) indices from
/// `VARIABLES`); and every variable for an index past them, which this
/// analysis does not know.
///
/// A placeholder names no bit because the bit it will have is chosen when
/// its fold closes, and is none the term reading it binds or reads
/// (`close_over` takes the lowest slot no fold reachable from the body or
/// its bounds binds and no `Var` there reads). So the one question a
/// placeholder's variance is asked — L3's, inside `close_over`: does this
/// bound vary with a coordinate, or with its own fold's binder? — has the
/// answer no, which is exact, and a triangular fold built through the
/// builder is accepted.
///
/// That answer is *not* "closed". A placeholder can escape its builder (a
/// closure may keep a clone of the index it was given), and the term that
/// holds it is open while its variance is empty. So no check of closedness or
/// placement may read a variance: they ask [`ExprArena::free_index`], which
/// counts a placeholder free (`Kernel::by_ref`, the runtime tier's unit walk),
/// or refuse a `Var` that is no fold's binder (codegen's lowering).
#[must_use]
pub const fn of_var(index: u8) -> Self {
    if index < Self::VARIABLES {
        return Self::from_var(index);
    }
    if (index as usize) < crate::fold::PLACEHOLDER_BASE as usize + crate::fold::Placeholder::COUNT {
        return Self::CONST;
    }
    Self::ALL
}
```

Callers: `transfer`'s `Var` arm (§3.2), replacing `variance.rs:322-328`; and `ENode::variance`'s
`Var` arm (`node.rs:534`), replacing `var_variance` (`node.rs:558-571`), which is deleted (§6.1).
`ALL` is spelled afterwards only in the `Param` arm (`variance.rs:339-343`), in
`referent_variance` (`:280`, `:291`), in `of_var`'s last line, and in tests. The choice of
`CONST` over today's `ALL` (and over `BINDERS`) is J5.

### 3.2 L4, and the one per-node transfer function with two drivers (N1)

`compute_arena_variance`'s match (`variance.rs:319-387`) moves, arm for arm, into a private
function; the function becomes its forward driver; a second driver walks from one root. One rule,
two traversal orders, as `ENode::variance` (`node.rs:509-556`) already is for the e-graph's
class fact and the extractor.

```rust
/// What the node at `id` varies with, given what each of its children
/// varies with: the arena's forward transfer function of variance, and the
/// one definition its two drivers share — `compute_arena_variance`, forward
/// over the whole arena, and `variance_from`, from one root over what it
/// reaches. (`pixelflow-search`'s `ENode::variance` is the e-graph's.)
///
/// A leaf names its own variable ([`Variance::of_var`]) or none; an
/// operation unions its children; a fold (L4) unions its bounds with its
/// body less the index it binds — `(var(body) \ b) ∪ var(lo) ∪ var(hi)`, a
/// literal bound contributing nothing — and a fold over a known empty range
/// is its monoid's identity, a constant, whatever its body reads.
fn transfer(arena: &ExprArena, id: ExprId, child: impl Fn(ExprId) -> Variance) -> Variance {
    match arena.node(id) {
        ExprNode::Var(idx) => Variance::of_var(idx),
        // ... `Const`, `Buffer`, `Uniform`, `Ref`, `Param`, `Unary`, `Binary`,
        // `Ternary`, `Write`, `Nary`: today's arms (`variance.rs:329-348`,
        // `:367-386`) with `result[c.0 as usize]` read as `child(c)`, and
        // their comments kept.
        ExprNode::Reduce { .. } if arena.known_range(id).is_some_and(KnownRange::is_empty) => {
            Variance::CONST
        }
        // The one node that shrinks the set — it binds its index — and the
        // one whose bounds are read outside that binding: a bound is fixed
        // before the fold runs (L3), so what it reads is free in the fold,
        // unmasked.
        ExprNode::Reduce { fold, lo, hi, body } => [lo.value(), hi.value()]
            .into_iter()
            .flatten()
            .fold(
                child(body).without(Variance::from_var(fold.binder().var())),
                |v, bound| v.union(child(bound)),
            ),
    }
}

/// [doc: today's `compute_arena_variance` doc, `variance.rs:294-308`, with
/// "Cost: O(n) … No allocations beyond the result vec." kept and one
/// sentence added: "The forward driver of `transfer`; `variance_from` is
/// the rooted one."]
#[must_use]
pub fn compute_arena_variance(arena: &crate::arena::ExprArena) -> Vec<Variance> {
    let mut result = Vec::with_capacity(arena.len());
    for i in 0..arena.len() {
        let v = transfer(arena, ExprId(i as u32), |c| result[c.0 as usize]);
        result.push(v);
    }
    result
}

/// What the term at `root` varies with: [`transfer`] run from `root` over
/// what it reaches, each node once. `compute_arena_variance(arena)[root]`,
/// for the price of the term rather than of the arena.
///
/// The rooted driver exists for L3 (`ExprArena::push_reduce`): a fold's
/// bound is a leaf at the user surfaces and a few nodes after a rule, the
/// arena holding it is the whole kernel, and every rebuild re-pushes every
/// value-bounded fold — `splice_with`, `substitute_vars_with`, `relink`,
/// `to_arena_roots`, `passes::copy_node`, extraction's `embed`. An
/// arena-sized table per push is the shape `passes.rs:571-574` measured at
/// 86 s on a 613-piece run. The memo is keyed, not arena-sized, and a `Ref`
/// the term does not reach is not resolved.
#[must_use]
pub(crate) fn variance_from(arena: &crate::arena::ExprArena, root: ExprId) -> Variance {
    let mut memo: BTreeMap<ExprId, Variance> = BTreeMap::new();
    let mut stack = alloc::vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        if memo.contains_key(&id) {
            continue;
        }
        if expanded {
            let v = transfer(arena, id, |c| memo[&c]);
            memo.insert(id, v);
            continue;
        }
        stack.push((id, true));
        stack.extend(arena.children(id).map(|c| (c, false)));
    }
    memo[&root]
}
```

`alloc::collections::BTreeMap` is imported beside `alloc::vec::Vec` (`variance.rs:269`). The
`Reduce` arm's `fold` and the known-empty test are today's; only the union is new. Codegen sees
L4 on a value-bounded fold only after A4: it refuses one first.

### 3.3 `free_index` (arena.rs:734-797)

Only the fold arm (`:781-785`) changes. No `free_indices` is added (J17): `free_index` is the one
reader of the set, and `lowest_free_binder` asks a different question (§3.5).

```rust
ExprNode::Reduce { fold, lo, hi, body } => {
    let bound = bit(fold.binder().var());
    let [low, high] = free[body.0 as usize];
    // L4's shape: a bound is outside its fold's binding, so what it reads
    // is free in the fold, unmasked.
    [lo.value(), hi.value()].into_iter().flatten().fold(
        [low & !bound[0], high & !bound[1]],
        |[low, high], value| {
            let [value_low, value_high] = free[value.0 as usize];
            [low | value_low, high | value_high]
        },
    )
}
```

The doc (`:734-757`) gains one sentence after "Scoped, not a scan: …": "A fold's bound is read
outside the fold's binding, so a binder its bound reads is free in the fold (L4's shape)."

### 3.4 `push_reduce` (arena.rs:929-959): discovery, then the header, then L3; the copiers; `substitute_vars_with`'s precondition

**`push_reduce` (N4).**

```rust
/// Push the fold `fold` performs over `body`, from `lo` to `hi` — a `⊕`
/// over a range. A known bound is the node's data; a bound that is a value
/// is a child after the body (`lo`'s, then `hi`'s).
///
/// **Constness is discovered here, never declared** (plan §3.1): a value
/// bound whose term is a `Const` naming a count is that count
/// ([`Bound::discover`]), so every constructor — a builder, a copier,
/// extraction's `embed` — holds a known bound as data. A `Const` naming no
/// count (not an integer, or outside `0..=Fold::EXACT_BOUND`) stays a value
/// bound, whose guard makes the fold NaN (L5, A4): a literal a rule or the
/// folder made never panics here. The user surfaces refuse one by name
/// ([`BoundRefusal::BoundOutOfDomain`]).
///
/// [`Fold`] is the metadata: which algebra, which index, which step. [today's
/// paragraph, `:933-938`, kept.]
///
/// # Panics
///
/// In this order, cheapest first: a value bound whose term is a mask literal
/// ([`BoundRefusal::BoundSort`]); what [`FoldHeader::new`] refuses (a literal
/// pair `KnownRange::new` refuses, a value bound under `ANY`, `ALL` or `SEQ`,
/// a stride other than 1 beside a value bound, a literal past the domain
/// beside one); and a value bound that varies with a coordinate or with
/// this fold's own binder ([`BoundRefusal::BoundVaries`], L3), computed from
/// the bound's term alone.
pub fn push_reduce(
    &mut self,
    fold: Fold,
    lo: Bound<ExprId>,
    hi: Bound<ExprId>,
    body: ExprId,
) -> ExprId {
    let (lo, hi) = Bound::discover(lo, hi, |id| self.literal_at(id))
        .unwrap_or_else(|refusal| panic!("{refusal}: {fold:?}"));
    let header = FoldHeader::new(fold, lo.erased(), hi.erased());
    for (end, bound) in [(BoundEnd::Lo, lo), (BoundEnd::Hi, hi)] {
        let Bound::Value(term) = bound else {
            continue;
        };
        if let Err(refusal) = fold.admits_bound(end, crate::variance::variance_from(self, term)) {
            panic!("{refusal}: {fold:?}");
        }
    }
    let mut children = [body; 3];
    // ... today's lines `:952-958`, unchanged.
}

/// The literal at `id`, when the node there is a `Const`: what discovery
/// reads a bound's term as.
fn literal_at(&self, id: ExprId) -> Option<Lit> {
    match *self.builder.dag().get(id.0) {
        NodeData::Const(lit) => Some(lit),
        _ => None,
    }
}
```

`arena.rs:29` imports `BoundEnd` beside `Bound`.

**The copiers re-push through it.** `splice_with` (`:1594-1597`), `substitute_vars_with`
(`:1834-1837`), `relink` (`:1990-1993`), `passes::copy_node` (`passes.rs:238-240`),
`expr::to_arena_roots` (`expr.rs:393-396`), `term_arena.rs`'s `embed` (`:85`, which
`rebuild_into` and extraction's `Selection::to_arena` reach), `nnue/mod.rs`'s junkify (`:888-895`)
and `pixelflow-pipeline`'s `reachable_subtree` (`training/corpus.rs:275-277`) map both bounds and
call `push_reduce`, so each copy rediscovers (a no-op on a copy: a `Const` in domain was read the
first time) and re-checks L3 on the bound's term. That is the price N1 accepts, the term's and
not the arena's. None of them changes.

**`substitute_vars_with`'s precondition (N5).** Its doc (`arena.rs:1765-1773`) gains, after the
first paragraph:

```rust
/// **Precondition: `subs` names no slot a fold inside `root` rebinds.** The
/// rebuild enters every fold's body and bounds alike and replaces every
/// `Var(i)` it meets, so substituting a slot a nested fold binds would
/// replace that fold's own index — capture, plausible wrong pixels. Its
/// callers substitute coordinates (`ExprArena::apply`,
/// `passes::lattice::collapse`), retired axes (`pixelflow-pipeline`'s
/// `bind_retired_axes`), or a rewrite template's metavariables
/// (`template.rs`'s `shift_vars` and `canonicalize_vars`, over templates that
/// hold no fold), none of which a fold binds. This is not the plan's L7
/// substitution: that is the e-graph's `PlanBuilder::substitute`, which
/// names a fold that rebinds the slot instead of entering it.
```

The callers are `kernel.rs:1144` (`apply`), `passes/lattice.rs:123` (`collapse`),
`pixelflow-search/src/egraph/template.rs:393`, `:526-527`, and
`pixelflow-pipeline/src/training/factored.rs:712`. No caller changes, and A3 does not claim L7
for this function.

### 3.5 `FoldTerms`, `close_over`, `close_fold`, `lowest_free_binder` (arena.rs:1616-1763)

**`FoldTerms` (new, `pub`, in `arena.rs` beside `OpenFold`, `:144-165`).** Data replaces the
`fold_at: impl FnOnce(Binder) -> FoldHeader` callback (J14): the binder is the one thing
`close_over` computes, and everything else about the fold is the builder's.

```rust
/// What a builder closes a fold's body over ([`ExprArena::close_over`]):
/// the fold's algebra and step, and its bounds, each a literal or a term in
/// the arena the body is built in. The binder is `close_over`'s to choose,
/// so it is not here.
///
/// A value bound is a term like any other, built where the body is before
/// the fold closes (`Kernel::over_values` splices it beside the body). One
/// that is a literal naming a count is read into the fold when it closes,
/// and is not copied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FoldTerms {
    /// The algebra the fold combines under.
    pub monoid: Monoid,
    /// The step between visited indices: 1, but for `kernel!`'s open folds
    /// (`HOLE_STRIDE`) until A5 deletes them.
    pub stride: u64,
    /// The first bound.
    pub lo: Bound<ExprId>,
    /// The last bound.
    pub hi: Bound<ExprId>,
}
```

**`close_over` (N2: discovery before the copy).** Signature
`pub fn close_over(&self, body: ExprId, placeholder: Placeholder, terms: FoldTerms) ->
Result<(ExprArena, ExprId), IndexSpaceFull>`.

```rust
pub fn close_over(
    &self,
    body: ExprId,
    placeholder: Placeholder,
    terms: FoldTerms,
) -> Result<(ExprArena, ExprId), IndexSpaceFull> {
    let FoldTerms { monoid, stride, lo, hi } = terms;
    // Discovered before anything is copied: a bound whose term is a literal
    // naming a count is the fold's data and not a root, so nothing is copied
    // for it, and a fold over two such bounds is the one over literals,
    // node for node (L1).
    let (lo, hi) = Bound::discover(lo, hi, |id| self.literal_at(id))
        .unwrap_or_else(|refusal| panic!("{refusal}: a {monoid:?} fold"));
    let mut roots = alloc::vec![body];
    roots.extend(lo.value());
    roots.extend(hi.value());
    let binder = self.lowest_free_binder(&roots).ok_or(IndexSpaceFull)?;
    let fold = Fold::strided(monoid, binder, stride);

    use crate::expr::{ExprBuilderExt, from_arena_roots, substitute_vars, to_arena_roots};
    let (open, tables) = from_arena_roots(self, &roots);
    let mut closed = Builder::new();
    let index = closed.push_var(binder.var());
    // Every root renamed, the body first: a bound that reads this fold's own
    // placeholder reads its binder after the rename, and `push_reduce`
    // refuses it (L3). An enclosing fold's placeholder is left as it is.
    let renamed: Vec<Id> = open
        .entries()
        .map(|root| substitute_vars(&mut closed, root, &[(placeholder.var(), index)]))
        .collect();
    let closed = closed.finish(&renamed);
    let entries: Vec<_> = closed.entries().collect();
    let (mut arena, ids) = to_arena_roots(&closed, &entries, &tables);
    // The value bounds' ids follow the body's, `lo`'s before `hi`'s, as the
    // roots did.
    let mut values = ids[1..].iter().copied();
    let mut next = |_| values.next().expect("one closed root per value bound");
    let (lo, hi) = (lo.map(&mut next), hi.map(&mut next));
    let root = arena.push_reduce(fold, lo, hi, ids[0]);
    Ok((arena, root))
}
```

Notes for the implementer:
- The ids are paired with the bounds in root order by one iterator, not through
  `FoldHeader::attach`: `attach` would need a header, and `push_reduce` builds the one header.
- The doc (`:1616-1645`) is rewritten: the first paragraph becomes "This arena's `body`, built
  against `placeholder`, closed into the fold `terms` names: `Reduce(fold(b), body[placeholder :=
  Var(b)], lo, hi)`, as an arena of its own." The layout sentence becomes "… then the binder's
  `Var`, then the body, then each value bound's term, then the fold, and nothing else". A
  paragraph is added: "**Constness is discovered first** ([`Bound::discover`]): a bound whose
  term is a literal naming a count is the fold's data, and its term is not copied — so a fold
  closed over two such terms is the fold over the literals, node for node, and a pair that
  discovery makes known runs forwards (L2). A literal naming no count stays a value bound, as
  `push_reduce` keeps one." `# Panics` is added: "A value bound whose term is a mask literal
  ([`BoundRefusal::BoundSort`]), and what [`ExprArena::push_reduce`] refuses of the closed fold."
- The comment and `assert_eq!(header.arity(), 1, …VALUE_BOUND)` at `:1654-1658` go.
- `entries()` returns `Node`s in entry order (`dag.rs:579-584`); `to_arena_roots` takes them as a
  slice (`expr.rs:367-371`). `Rooted` derefs to its `Dag` (`kernel.rs`'s `dag()` returns it so).

**`close_fold`** (`:1702-1725`): signature `pub fn close_fold(&mut self, open: OpenFold, body:
ExprId, terms: FoldTerms) -> Result<ExprId, IndexSpaceFull>`; its body passes `terms` to
`close_over` where it passed `fold_at`, and its doc says "closed by [`close_over`] into the fold
`terms` names". A value bound's term is built in this arena (the copy) before `close_fold`, and
an id bound before the fold opened means the same node in the copy (`open_fold`'s doc), so
`kernel!` can lower a bound before `open_fold` (A6).

**`lowest_free_binder(roots)`** (`:1743-1763`), J8:

```rust
/// The lowest binder no `Reduce` reachable from `roots` binds and no `Var`
/// reachable from them reads, or `None` when every one is taken:
/// [`ExprArena::close_over`]'s choice. `roots` are a fold's body and its
/// value bounds: a bound is read where the fold is, so a slot it reads is
/// not one the fold may take (L3 forbids a fold rebinding a slot its bounds
/// read), and a body that reads a slot it does not bind — a term open in a
/// binder — is not captured either.
fn lowest_free_binder(&self, roots: &[ExprId]) -> Option<Binder> {
    let mut taken = [false; Binder::COUNT];
    let mut seen = vec![false; self.len()];
    let mut stack = roots.to_vec();
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        // [today's comment, `:1753-1756`, kept]
        match self.node(id) {
            ExprNode::Reduce { fold, .. } => taken[usize::from(fold.binder().slot())] = true,
            ExprNode::Var(index) => {
                if let Some(binder) = Binder::from_var(index) {
                    taken[usize::from(binder.slot())] = true;
                }
            }
            _ => {}
        }
        stack.extend(self.children(id));
    }
    Binder::all().find(|binder| !taken[usize::from(binder.slot())])
}
```

`seen` stays arena-sized, as today: one walk per fold closed, not per push. This computes the set
`passes::lattice::reachable_taken_binders` (`passes/lattice.rs:157-190`) computes; they are not
merged (J19).

### 3.6 The `known_range` docs

- `ExprArena::known_range` (`arena.rs:1148-1162`): "The range of the `Reduce` at `id`, when it is
  known: `Some` when both bounds are literals, `None` when either is a value — the case a
  reader's own answer for an unknown range is for (a rule declines, an emptiness test is
  false)." The rest of the doc is kept.
- `ENode::known_range` (`node.rs:446-457`): the same first sentence, "every bound is a constant
  today, so this is `Some` for every fold" replaced likewise.
- `NodeData`'s doc (`arena.rs:56-57`) is already true.

---
## 4. `Kernel::over_values` (`pixelflow-ir/src/kernel.rs`, `pub(crate)`) and `Kernel::over`

### 4.1 `Kernel::over`'s call and its doc (N9)

The call (`kernel.rs:758-764`) becomes:

```rust
let terms = FoldTerms {
    monoid,
    stride: 1,
    lo: Bound::Lit(u64::from(range.start)),
    hi: Bound::Lit(u64::from(range.end)),
};
let closed = arena.close_over(root, scope.placeholder, terms);
```

`kernel.rs:21-24` imports `FoldTerms` from `crate::arena`; `FoldHeader` and `Fold` leave
`kernel.rs`'s imports if nothing else there names them (the test module imports its own).

The doc's third paragraph (`kernel.rs:711-717`, "`range` is static, which is what keeps the
language total and its cost closed-form (`|D| × cost(body)`); the backend unrolls, so the domain
is bounded in practice as well as in principle.") is false after A3 and is replaced with:

```rust
/// `range` is the literal spelling of a known range; a bound that is a value
/// is `Kernel::over_values`'. Either way every bound is known before the
/// lattice is traversed, which is what keeps the language total and its
/// cost closed-form — `|D|·cost(body) + (|D|−1)·cost(⊕)`
/// (docs/language/2026-10-20-a-bound-is-a-value.md §1.5). A fold that
/// survives extraction is a loop; extraction unrolls a known one where
/// that is cheaper.
```

`over_values` is named in backticks, not linked: it is `pub(crate)`, and a public item's doc may
not link a private one under `-D warnings`. The `# Panics` section (`:733-738`) is kept:
`Kernel::over`'s out-of-domain literal is refused by `KnownRange::new`'s message, which is the
user surface's refusal for a declared literal (N4; J13).

### 4.2 `Kernel::over_values` (new, `pub(crate)`; public in A4, decision 7)

Placed after `Kernel::over` (`kernel.rs:745-774`). Four arguments, the fold's own: monoid, the two
bounds, the body — `push_reduce`'s precedent and the plan's spelling (plan:271-274; J16).

```rust
/// `⊕_{i ∈ [lo, hi)} body(i)`: [`Kernel::over`] with bounds that are
/// values — a per-call count (a [`Uniform`]), an enclosing fold's index
/// (a triangular `Σ_i Σ_{j<i}`), or a constant
/// (docs/language/2026-10-20-a-bound-is-a-value.md §1.2).
///
/// Constness is discovered, not declared: a bound that is a constant
/// naming a count is the fold's data, so `over_values(⊕, 0, 4, f)` is
/// `over(⊕, 0..4, f)` node for node, and a pair of constants running
/// backwards is the empty range (`over_values(⊕, 5, 3, f)` is the
/// identity, L2), where `over`'s literal `5..3` is refused. A bound that is
/// not a constant is a child of the fold, after the body, and the fold's
/// range is not known: it is not unrolled, and is priced at an estimate.
///
/// The arena declares the body's uniforms, then `lo`'s, then `hi`'s, each in
/// its own order, as every composition does ([`Kernel::uniforms`]).
///
/// # Panics
///
/// On a bound that is not a leaf ([`BoundRefusal::BoundArithmetic`]: until
/// bounds are `Int`, a computed bound can round into the domain, §1.4); on
/// one that is not a number ([`BoundRefusal::BoundNotANumber`]: a `Param`, a
/// `Buffer`); on a constant naming no count
/// ([`BoundRefusal::BoundOutOfDomain`]) or a mask ([`BoundRefusal::BoundSort`]);
/// and, from the fold [`ExprArena::close_over`] builds, on a bound reading a
/// coordinate ([`BoundRefusal::BoundVaries`], L3), on a value bound under
/// `ANY`, `ALL` or `SEQ` ([`BoundRefusal::BoundSort`]), and when the body
/// binds every slot.
#[must_use]
pub(crate) fn over_values(
    monoid: Monoid,
    lo: &Kernel,
    hi: &Kernel,
    body: impl FnOnce(&Kernel) -> Kernel,
) -> Self {
    for (end, bound) in [(BoundEnd::Lo, lo), (BoundEnd::Hi, hi)] {
        if let Err(refusal) = Self::user_bound(end, bound) {
            panic!("{refusal}");
        }
    }
    let scope = BinderScope::enter();
    let index = {
        let mut b = Builder::new();
        let r = b.push_var(scope.placeholder.var());
        Self::wrap(b.finish(&[r]), Environment::new(), BTreeMap::new())
    };
    let body = body(&index);
    // One arena, the body's terms first and each bound's after, declared in
    // that order. A leaf bound carries no tabulation (`user_bound`), so the
    // body's table is the fold's.
    let mut b = Builder::new();
    let mut env = body.inner.env.clone();
    let body_root = copy_subgraph(&mut b, body.root());
    let lo_root = splice(&mut b, &mut env, lo.root(), &lo.inner.env);
    let hi_root = splice(&mut b, &mut env, hi.root(), &hi.inner.env);
    let rooted = b.finish(&[body_root, lo_root, hi_root]);
    let entries: Vec<_> = rooted.entries().collect();
    let (arena, ids) = to_arena_roots(&rooted, &entries, &env);
    let terms = FoldTerms {
        monoid,
        stride: 1,
        lo: Bound::Value(ids[1]),
        hi: Bound::Value(ids[2]),
    };
    let (arena, root) = arena
        .close_over(ids[0], scope.placeholder, terms)
        .unwrap_or_else(|IndexSpaceFull| {
            panic!(
                "more than {} live nested reductions: the index space is full",
                Binder::COUNT
            )
        });
    Self::adopt(arena, root, body.inner.buffers.clone())
}

/// Whether `bound`, the fold's `end` bound as a user wrote it, is one
/// `over_values` takes: a leaf naming a number — a constant naming a count,
/// a uniform, an index — and nothing computed (§1.4, until T6).
fn user_bound(end: BoundEnd, bound: &Kernel) -> Result<(), BoundRefusal> {
    match *bound.root() {
        ExprData::Var(_) | ExprData::Uniform(_) => Ok(()),
        ExprData::Const(Lit::Mask(_)) => Err(BoundRefusal::BoundSort { end }),
        ExprData::Const(literal @ Lit::Real(real)) => Fold::count(literal)
            .map(|_| ())
            .ok_or(BoundRefusal::BoundOutOfDomain { end, literal: real }),
        ExprData::Param(_) | ExprData::Buffer(_) => Err(BoundRefusal::BoundNotANumber { end }),
        ExprData::Op(_) | ExprData::Reduce(_) | ExprData::Ref(_) => {
            Err(BoundRefusal::BoundArithmetic { end })
        }
    }
}
```

Notes:
- `kernel.rs` imports `to_arena_roots` from `crate::expr` (it imports `to_arena` today,
  `kernel.rs:25-28`), `BoundEnd`/`BoundRefusal` from `crate::fold`, and `FoldTerms` from
  `crate::arena`.
- A `Var` that is a coordinate passes `user_bound` (it is a leaf) and is refused by L3 inside
  `close_over`'s `push_reduce`; that is the one statement of L3, not a second one here.
- A `Ref` is `BoundArithmetic`: a named kernel evaluated here is computed, whatever it names.
- The refusal of the monoid is left to `FoldHeader::new` (J18): `over_values(ANY, 0, 3, f)` over
  two constants is the literal `ANY` fold, which is legal.
- The `BinderScope` is entered after the user checks, so a refused bound claims no placeholder.

---
## 5. L7 and the every-child walkers

**L7, both sides.** The e-graph's `PlanBuilder::substitute` (`fold_rules.rs:191-275`) is the L7
substitution: it walks one representative per class, names a class whose variance fact lacks the
substituted slot, and otherwise enters every child of the node — `children_slice`, the body and
then each value bound (`node.rs:478-488`). L4 (§6.1) makes a rebinding fold's fact lack the slot
(its body's reads less the slot, and no bound reads its own fold's slot, L3), so such a fold is
named and not entered; a nested fold over another slot is entered, and the substitution reaches
its bounds and its body alike. No code changes in `substitute`; `rebuild`'s comment does (§6.8),
and T-S-4, T-S-4b and T-S-6 pin it. The arena's one binding-aware rename is `close_over`'s
(§3.5): it renames a placeholder unique among binders under construction, which no fold
rebinds, so entering every body and bound is exact. `ExprArena::substitute_vars_with` is not L7
(N5, §3.4).

**The walkers.** A2b gave every walker that must see a fold's value bounds the children it needs
(`children()` yields the body then each value bound, `arena.rs:1261-1278`; `children_slice` the
same, `node.rs:478-488`), so A3's work here is to classify every site and change the few whose
answer A3 changes. The generator (B2) runs

```
grep -rn 'Reduce {' --include=*.rs pixelflow-*/src pixelflow-*/examples pixelflow-*/tests pixelflow-pipeline/src/bin
```

which lists 155 lines at `3bfdaded`; the four `pixelflow-pipeline/src/bin` files are matched by
both `pixelflow-*/src` and `pixelflow-pipeline/src/bin`, so it dedupes by `(file, line)` to **151
sites**. It keys each site by `(file, enclosing item, trimmed hit text)` with a per-key count, so
line drift does not unclassify a site, prints `UNCLASSIFIED` (a hit with no row) and `STALE` (a
row with no hit), and exits non-zero on either; at `3bfdaded` it must report 0 and 0 with the
table below as its rows (B2). A site A3 adds or changes gets a row in the same CL.

Classes: **already-correct** (takes every child, or answers rightly for a value bound without
change); **changed-by-A3** (A3 edits the arm or its doc); **left** (reason: it refuses or never
meets a value bound, and stays so); **tests**; **n/a** (a definition or a doc); **A4** (the
schedule, the key and the serializers, which refuse a value bound by name until A4). No site is
A5: the compiler's `lower.rs` hits are all in its test module (from `:1054`).

Counts at `3bfdaded`: **already-correct 39, changed-by-A3 19, left 13, tests 60, n/a 5, A4 15;
151 sites.**

**Rows A3 itself moves or adds** (the generator's key is the enclosing item and the trimmed hit
text, so each of these is a row to rewrite in the CL, not a new class): `variance.rs:355`, `:364`
move into `transfer` (changed-by-A3, `transfer`); `node.rs:552`'s arm text changes (changed-by-A3);
new production hits — `graph.rs`'s `add` (`if let ENode::Reduce { header, children } = &node`,
changed-by-A3: the backstop), `fold_rules.rs`'s `KnownBound::apply` (changed-by-A3: L8),
`ENode::literal_twin` (new in `node.rs`: N3, F3) — and every new test of §10 that matches
(tests). The fold arm of `free_index` keeps its hit with new text (changed-by-A3).

| Site (`3bfdaded`) | In | Class | Why |
|---|---|---|---|
| `pixelflow-codegen/src/emit/select.rs:147` | definition | A4 | the schedule: lowering refuses a value bound (`known`, VALUE_BOUND) before selection sees one |
| `pixelflow-codegen/src/emit/select.rs:171` | definition | A4 | the schedule: lowering refuses a value bound (`known`, VALUE_BOUND) before selection sees one |
| `pixelflow-codegen/src/program/lower.rs:58` | is_empty_fold | already-correct | `is_empty_fold`: a range that is not known is not known to be empty (`is_some_and`) |
| `pixelflow-codegen/src/program/lower.rs:312` | arena_to_schedule | already-correct | the identity arm fires on a known empty range only |
| `pixelflow-codegen/src/program/lower.rs:317` | arena_to_schedule | left | the lane fold is `collapse`'s SEQ fold, which `BoundSort` keeps literal |
| `pixelflow-codegen/src/program/lower.rs:340` | arena_to_schedule | changed-by-A3 | comment only ("none can be built until A3"); `known` keeps refusing by name (A4) |
| `pixelflow-codegen/src/program/mod.rs:148` | ScheduledOp | A4 | `ScheduledOp::Reduce`, `operands`, `structural_children`: A4 yields the value bounds |
| `pixelflow-codegen/src/program/mod.rs:398` | operands | A4 | `ScheduledOp::Reduce`, `operands`, `structural_children`: A4 yields the value bounds |
| `pixelflow-codegen/src/program/mod.rs:438` | structural_children | A4 | `ScheduledOp::Reduce`, `operands`, `structural_children`: A4 yields the value bounds |
| `pixelflow-codegen/src/program/scopes.rs:213` | extract_folds_bound_by | A4 | the carve: A4 adds the value bounds to placement (plan §3.3) |
| `pixelflow-codegen/src/program/scopes.rs:250` | extract_folds_bound_by | A4 | the carve: A4 adds the value bounds to placement (plan §3.3) |
| `pixelflow-codegen/src/program/scopes.rs:268` | extract_folds_bound_by | A4 | the carve: A4 adds the value bounds to placement (plan §3.3) |
| `pixelflow-codegen/tests/halve_fold_jit.rs:59` | node_cost | tests | integration test |
| `pixelflow-compiler/src/emit.rs:581` | node | A4 | `arena_to_tokens` writes header bits; `to_bits` refuses a value bound by name, and `kernel!` builds none before A6 |
| `pixelflow-compiler/src/lower.rs:1182` | a_structural_parameter_leaves_the_template_open | tests | lowering unit tests (module from :1054) |
| `pixelflow-compiler/src/lower.rs:1350` | a_fold_lowers_to_one_reduce_and_leaves_nothing_behind | tests | lowering unit tests (module from :1054) |
| `pixelflow-compiler/src/lower.rs:1376` | nested_folds_take_distinct_slots_inside_out | tests | lowering unit tests (module from :1054) |
| `pixelflow-compiler/src/lower.rs:1382` | nested_folds_take_distinct_slots_inside_out | tests | lowering unit tests (module from :1054) |
| `pixelflow-compiler/src/lower.rs:1436` | a_bound_past_the_exact_index_is_refused | tests | lowering unit tests (module from :1054) |
| `pixelflow-compiler/src/lower.rs:1454` | folds_nest_as_deep_as_the_ir_has_binders | tests | lowering unit tests (module from :1054) |
| `pixelflow-compiler/tests/fold_is_kernel_over.rs:208` | binders | tests | integration test |
| `pixelflow-compiler/tests/kernel_typed_parameters.rs:202` | the_fold_around_an_application_binds_past_the_arguments | tests | integration test |
| `pixelflow-compiler/tests/kernel_typed_parameters.rs:206` | the_fold_around_an_application_binds_past_the_arguments | tests | integration test |
| `pixelflow-core/tests/fold_factoring.rs:148` | assert_extracts_factored | tests | integration test |
| `pixelflow-core/tests/kernel_by_reference.rs:191` | a_fold_around_a_unit_that_rebinds_its_slot | tests | integration test |
| `pixelflow-core/tests/reduce_binder_reads_bound_buffer.rs:240` | legalize_keeps_a_bare_reduce_and_its_size_does_not_track_the_extent | tests | integration test |
| `pixelflow-graphics/examples/glyph_saturation_cost.rs:49` | has_fold | already-correct | `has_fold` flags a fold and walks `children()` |
| `pixelflow-graphics/src/render/cell_grid.rs:1169` | dump_arena (tests) | A4 | test-module `dump_arena`: header bits, `to_bits` refuses by name |
| `pixelflow-graphics/src/test_support.rs:37` | program_digest | A4 | `program_digest`: header bits, `to_bits` refuses by name |
| `pixelflow-graphics/tests/production_glyph_arena_dump.rs:167` | dump_arena | A4 | `dump_arena` (:171 writes `to_bits`), on N8's serializer list |
| `pixelflow-ir/src/arena.rs:432` | ExprNode | n/a | the `ExprNode::Reduce` definition |
| `pixelflow-ir/src/arena.rs:781` | free_index | changed-by-A3 | `free_index`: `(free(body) \ b) ∪ free(lo) ∪ free(hi)` (§3.3) |
| `pixelflow-ir/src/arena.rs:1128` | node | already-correct | `node` attaches the value children (`FoldHeader::attach`) |
| `pixelflow-ir/src/arena.rs:1330` | depth | already-correct | `depth` pushes `children()` |
| `pixelflow-ir/src/arena.rs:1372` | has_var | already-correct | `has_var`, `has_degenerate`, `node_count_subtree` extend by `children()` |
| `pixelflow-ir/src/arena.rs:1424` | has_degenerate | already-correct | `has_var`, `has_degenerate`, `node_count_subtree` extend by `children()` |
| `pixelflow-ir/src/arena.rs:1468` | node_count_subtree | already-correct | `has_var`, `has_degenerate`, `node_count_subtree` extend by `children()` |
| `pixelflow-ir/src/arena.rs:1594` | splice_with | already-correct | `splice_with` maps both bounds into `push_reduce` |
| `pixelflow-ir/src/arena.rs:1757` | lowest_free_binder | changed-by-A3 | `lowest_free_binder(roots)`: scans the value bounds and marks binder `Var` reads (§3.5, J8) |
| `pixelflow-ir/src/arena.rs:1834` | substitute_vars_with | changed-by-A3 | `substitute_vars_with` maps both bounds; A3 states its precondition (N5, §3.4) |
| `pixelflow-ir/src/arena.rs:1990` | relink | already-correct | `relink` maps both bounds |
| `pixelflow-ir/src/arena.rs:2069` | fmt_expr | already-correct | `fmt_expr` prints each value bound after the body |
| `pixelflow-ir/src/arena.rs:2186` | subtree_eq | already-correct | `subtree_eq` compares the erased bounds and pushes the value pairs |
| `pixelflow-ir/src/arena.rs:2192` | subtree_eq | already-correct | `subtree_eq` compares the erased bounds and pushes the value pairs |
| `pixelflow-ir/src/arena.rs:2321` | close_over_binds_the_lowest_free_binder_and_renames_the_placeholder | tests | arena unit tests |
| `pixelflow-ir/src/arena.rs:2552` | a_fold_carries_its_own_metadata | tests | arena unit tests |
| `pixelflow-ir/src/arena.rs:2573` | a_literal_bound_is_the_nodes_data | tests | arena unit tests |
| `pixelflow-ir/src/arena.rs:2813` | a_fold_opened_and_closed_is_close_overs_and_leaves_nothing_behind | tests | arena unit tests |
| `pixelflow-ir/src/expr.rs:327` | from_arena_roots | already-correct | `from_arena_roots` pushes the header with every child |
| `pixelflow-ir/src/fold.rs:343` | `Fold` doc | n/a | the denotation in `Fold`'s doc |
| `pixelflow-ir/src/kernel.rs:1292` | a_fold_is_laid_out_as_the_builder_always_laid_it_out | tests | kernel unit tests |
| `pixelflow-ir/src/kernel.rs:1325` | a_literal_equal_to_a_bound_lands_after_its_fold | tests | kernel unit tests |
| `pixelflow-ir/src/kernel.rs:1570` | an_argument_admitted_before_a_fold_is_applied_inside_it | tests | kernel unit tests |
| `pixelflow-ir/src/key.rs:454` | encode | A4 | `canonical`: keys the value children already; `to_bits` refuses a value bound by name until A4's marker |
| `pixelflow-ir/src/key.rs:657` | folds | tests | key test helper |
| `pixelflow-ir/src/kind.rs:204` | `OpKind::Reduce` doc | n/a | doc of `OpKind::Reduce`'s variant name |
| `pixelflow-ir/src/passes.rs:238` | copy_node | already-correct | `copy_node` maps both bounds into `push_reduce` |
| `pixelflow-ir/src/passes.rs:705` | push_deriv_children | left | `lower_dwrt` has no derivative of a fold: it neither walks nor differentiates one (`Err`) |
| `pixelflow-ir/src/passes.rs:949` | diff_node | left | `lower_dwrt` has no derivative of a fold: it neither walks nor differentiates one (`Err`) |
| `pixelflow-ir/src/passes.rs:1595` | an_extent_is_not_an_expression | tests | `an_extent_is_not_an_expression`; its doc is reworded (B1) and a sibling added (T-IR-14) |
| `pixelflow-ir/src/passes.rs:1933` | reachable_reduces | tests | legalize tests |
| `pixelflow-ir/src/passes.rs:1994` | legalize_leaves_a_nested_reduce_standing_inside_the_lattices_folds | tests | legalize tests |
| `pixelflow-ir/src/passes.rs:2005` | legalize_leaves_a_nested_reduce_standing_inside_the_lattices_folds | tests | legalize tests |
| `pixelflow-ir/src/passes.rs:2012` | legalize_leaves_a_nested_reduce_standing_inside_the_lattices_folds | tests | legalize tests |
| `pixelflow-ir/src/passes/lattice.rs:184` | reachable_taken_binders | already-correct | `reachable_taken_binders` walks `children()` and marks binder `Var` reads |
| `pixelflow-ir/src/passes/lattice.rs:296` | collapse_shape | left | `collapse_shape`: the lattice's SEQ folds, which keep literal ranges (`BoundSort`, plan §4 "Deferred") |
| `pixelflow-ir/src/passes/lattice.rs:305` | collapse_shape | left | `collapse_shape`: the lattice's SEQ folds, which keep literal ranges (`BoundSort`, plan §4 "Deferred") |
| `pixelflow-ir/src/passes/lattice.rs:314` | collapse_shape | left | `collapse_shape`: the lattice's SEQ folds, which keep literal ranges (`BoundSort`, plan §4 "Deferred") |
| `pixelflow-ir/src/passes/lattice.rs:452` | collapse_wraps_a_kernel_in_three_seq_folds_around_one_write | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:463` | collapse_wraps_a_kernel_in_three_seq_folds_around_one_write | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:474` | collapse_wraps_a_kernel_in_three_seq_folds_around_one_write | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:542` | collapse_takes_the_slots_the_kernel_leaves_free | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:695` | pack_strip_mines_an_exact_multiple_with_no_remainder | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:706` | pack_strip_mines_an_exact_multiple_with_no_remainder | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:714` | pack_strip_mines_an_exact_multiple_with_no_remainder | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:742` | pack_sequences_the_remainder_after_the_main_fold | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:751` | pack_sequences_the_remainder_after_the_main_fold | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:759` | pack_sequences_the_remainder_after_the_main_fold | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:767` | pack_sequences_the_remainder_after_the_main_fold | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:775` | pack_sequences_the_remainder_after_the_main_fold | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:808` | pack_of_a_row_narrower_than_a_batch_is_all_remainder | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:812` | pack_of_a_row_narrower_than_a_batch_is_all_remainder | tests | lattice pass tests |
| `pixelflow-ir/src/passes/lattice.rs:819` | pack_of_a_row_narrower_than_a_batch_is_all_remainder | tests | lattice pass tests |
| `pixelflow-ir/src/term.rs:134` | Shape< | n/a | the `Shape::Reduce` definition |
| `pixelflow-ir/src/term_arena.rs:42` | project | already-correct | `project` carries both bounds |
| `pixelflow-ir/src/term_arena.rs:85` | embed | already-correct | `embed` is `push_reduce`, which discovers and checks |
| `pixelflow-ir/src/term_arena.rs:148` | rebuild_into | already-correct | `rebuild_into` visits and maps body, `lo`, `hi` |
| `pixelflow-ir/src/term_arena.rs:180` | rebuild_into | already-correct | `rebuild_into` visits and maps body, `lo`, `hi` |
| `pixelflow-ir/src/term_arena.rs:184` | rebuild_into | already-correct | `rebuild_into` visits and maps body, `lo`, `hi` |
| `pixelflow-ir/src/variance.rs:355` | compute_arena_variance | changed-by-A3 | L4 in the per-node transfer function (§3.2) |
| `pixelflow-ir/src/variance.rs:364` | compute_arena_variance | changed-by-A3 | L4 in the per-node transfer function (§3.2) |
| `pixelflow-ir/src/variance.rs:689` | binder_index_is_a_variable_like_any_other | tests | variance unit tests |
| `pixelflow-ir/src/variance.rs:781` | six_nested_binders_each_take_a_slot | tests | variance unit tests |
| `pixelflow-ir/src/variance.rs:798` | six_nested_binders_each_take_a_slot | tests | variance unit tests |
| `pixelflow-pipeline/examples/glyph_phase_split.rs:78` | trip_counts | left | production glyph folds are literal until B2: `KNOWN_RANGE`'s reason (N9) |
| `pixelflow-pipeline/src/bin/corpus_gaps.rs:493` | hash_cons | A4 | `hash_cons` keys on header bits; `to_bits` refuses by name |
| `pixelflow-pipeline/src/bin/extraction_witnesses.rs:442` | node_label | already-correct | prints every child class |
| `pixelflow-pipeline/src/bin/phase3_at_budget_eval.rs:443` | non_leaf_op | already-correct | an op-kind classifier; reads no child |
| `pixelflow-pipeline/src/bin/strict_label_output_class_recount.rs:169` | render_chosen | already-correct | renders every child class |
| `pixelflow-pipeline/src/collapse_bench/corpus.rs:272` | encode | A4 | serializer of header bits; `to_bits` refuses by name |
| `pixelflow-pipeline/src/training/corpus.rs:275` | reachable_subtree | already-correct | `reachable_subtree` maps both bounds |
| `pixelflow-pipeline/src/training/corpus.rs:404` | write_node | changed-by-A3 | message reworded to "before A4's format version" (B1); the refusal is A4's |
| `pixelflow-pipeline/src/training/factored.rs:623` | arena_to_kernel_code | left | `arena_to_kernel_code`: kernel code has no binder, refuses every fold |
| `pixelflow-search/src/egraph/cost.rs:399` | node_op_cost | changed-by-A3 | `node_op_cost`: `fold_cost(fold, header.priced_trips(), 0)` (§7) |
| `pixelflow-search/src/egraph/extract.rs:971` | to_arena | already-correct | `Selection::to_arena` attaches the value operands; `embed` discovers a chosen literal |
| `pixelflow-search/src/egraph/extract.rs:973` | to_arena | already-correct | `Selection::to_arena` attaches the value operands; `embed` discovers a chosen literal |
| `pixelflow-search/src/egraph/extract.rs:1684` | below | changed-by-A3 | `below` and `TreePricer::price`: the priced form (§7) |
| `pixelflow-search/src/egraph/extract.rs:2101` | price | changed-by-A3 | `below` and `TreePricer::price`: the priced form (§7) |
| `pixelflow-search/src/egraph/extract.rs:2758` | dense_reference_pass | tests | extraction tests (`dense_reference_pass` changes mechanically) |
| `pixelflow-search/src/egraph/extract.rs:3503` | node_cost | tests | extraction tests (`dense_reference_pass` changes mechanically) |
| `pixelflow-search/src/egraph/extract.rs:4116` | node_cost | tests | extraction tests (`dense_reference_pass` changes mechanically) |
| `pixelflow-search/src/egraph/fold_rules.rs:99` | HeadNode | n/a | the `HeadNode::Reduce` definition |
| `pixelflow-search/src/egraph/fold_rules.rs:165` | PlanBuilder::reduce | already-correct | `PlanBuilder::reduce` carries both bounds |
| `pixelflow-search/src/egraph/fold_rules.rs:477` | EmptyFold::apply | already-correct | `EmptyFold` fires on a known empty range only (L6) |
| `pixelflow-search/src/egraph/fold_rules.rs:627` | rebuild | changed-by-A3 | `rebuild`: comment rewritten on L3 and L4 (§6.8); the code already attaches the value bounds |
| `pixelflow-search/src/egraph/fold_rules.rs:632` | rebuild | changed-by-A3 | `rebuild`: comment rewritten on L3 and L4 (§6.8); the code already attaches the value bounds |
| `pixelflow-search/src/egraph/fold_rules.rs:671` | reduce_range | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:699` | a_peel_shares_the_body_it_folds | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:728` | a_peel_shares_the_body_it_folds | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:799` | halve_doubles_the_body_sharing_it_when_the_binder_is_unused | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:815` | halve_doubles_the_body_sharing_it_when_the_binder_is_unused | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:1021` | decode | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:1283` | a_peel_stops_at_a_fold_that_rebinds_its_slot | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:1373` | node_cost | tests | fold rule tests |
| `pixelflow-search/src/egraph/fold_rules.rs:1382` | has_fold | tests | fold rule tests |
| `pixelflow-search/src/egraph/graph.rs:1626` | plan_shape | already-correct | `plan_shape` resolves both bounds into `ENode::reduce` |
| `pixelflow-search/src/egraph/graph.rs:1695` | derivative_shape | left | `derivative_shape`: no derivative rule for a fold; the `Dwrt` is kept |
| `pixelflow-search/src/egraph/insert.rs:173` | insert (Visit) | changed-by-A3 | `insert` discovers a literal bound; `inserted_class_count` mirrors it (§6.3) |
| `pixelflow-search/src/egraph/insert.rs:211` | insert (Complete) | changed-by-A3 | `insert` discovers a literal bound; `inserted_class_count` mirrors it (§6.3) |
| `pixelflow-search/src/egraph/insert.rs:264` | inserted_class_count | changed-by-A3 | `insert` discovers a literal bound; `inserted_class_count` mirrors it (§6.3) |
| `pixelflow-search/src/egraph/insert.rs:532` | project | tests | the `ValueBounded` test term |
| `pixelflow-search/src/egraph/node.rs:311` | ENode | changed-by-A3 | doc: a rule reaching a value bound is how a range becomes known (L8) |
| `pixelflow-search/src/egraph/node.rs:399` | reduce | changed-by-A3 | `ENode::reduce`: doc drops "until A3"; panics as `FoldHeader::new` |
| `pixelflow-search/src/egraph/node.rs:410` | as_reduce | already-correct | `as_reduce`, `shape`, `children_slice(_mut)` carry every child |
| `pixelflow-search/src/egraph/node.rs:440` | shape | already-correct | `as_reduce`, `shape`, `children_slice(_mut)` carry every child |
| `pixelflow-search/src/egraph/node.rs:460` | known_range | changed-by-A3 | `known_range`: doc ("every bound is a constant today") rewritten |
| `pixelflow-search/src/egraph/node.rs:485` | children_slice | already-correct | `as_reduce`, `shape`, `children_slice(_mut)` carry every child |
| `pixelflow-search/src/egraph/node.rs:500` | children_slice_mut | already-correct | `as_reduce`, `shape`, `children_slice(_mut)` carry every child |
| `pixelflow-search/src/egraph/node.rs:552` | ENode::variance | changed-by-A3 | L4 in `ENode::variance` (§6.1) |
| `pixelflow-search/src/egraph/witness.rs:356` | induce | left | research: a fold keyed as stored, for the measured miss counts' continuity |
| `pixelflow-search/src/egraph/witness.rs:421` | induce | already-correct | `induce` builds and visits body, `lo`, `hi` |
| `pixelflow-search/src/egraph/witness.rs:505` | post_order_term | already-correct | `induce` builds and visits body, `lo`, `hi` |
| `pixelflow-search/src/extraction_gap.rs:187` | build | left | research: the harness prices a fold's row as empty (its own documented model) |
| `pixelflow-search/src/math/pict_rewrite_tests.rs:120` | expr_to_egraph | tests | a test file's converter that refuses folds |
| `pixelflow-search/src/nnue/mod.rs:166` | pattern_match_arena | left | templates hold no fold: no match, and a refusal |
| `pixelflow-search/src/nnue/mod.rs:239` | substitute_template_arena | left | templates hold no fold: no match, and a refusal |
| `pixelflow-search/src/nnue/mod.rs:888` | junkify_arena_pass | already-correct | `junkify` maps both bounds |
| `pixelflow-search/src/nnue/mod.rs:890` | junkify_arena_pass | already-correct | `junkify` maps both bounds |
| `pixelflow-search/src/runtime.rs:1137` | a_fold_reads | tests | runtime tier tests |
| `pixelflow-search/src/runtime.rs:2335` | arena_cost | tests | runtime tier tests |
| `pixelflow-search/src/telemetry.rs:265` | latency_prior_cost | already-correct | `latency_prior_cost` walks `children()` |

---
## 6. The e-graph (`pixelflow-search`)

### 6.1 `ENode` (node.rs)

- **L4 in `ENode::variance`** (`node.rs:549-554`):

  ```rust
  // The one node that *shrinks* the set: its index is bound, so it is not
  // free in the result. Its value bounds are read outside that binding
  // (L3), so what they read is free in the fold, unmasked (L4).
  ENode::Reduce { header, children } => children[1..].iter().fold(
      child(children[0]).without(Variance::from_var(header.fold().binder().var())),
      |acc, &bound| acc.union(child(bound)),
  ),
  ```

  The `// Body only: a value bound's variance is a law of its own (A3).` comment goes (B1).
- **The `Var` arm** (`node.rs:534`) is `ENode::Var(v) => Variance::of_var(*v)`; `var_variance`
  (`node.rs:558-571`) is deleted, its doc's content now `of_var`'s (§3.1).
- **`ENode::reduce`'s doc** (`node.rs:389-397`): `# Panics` becomes "As [`FoldHeader::new`]: on a
  literal pair it refuses, and on a value bound the fold cannot take (`ANY`, `ALL` or `SEQ`; a
  stride other than 1)." The `VALUE_BOUND` link goes.
- **`ENode::Reduce`'s doc** (`node.rs:286-310`) gains, after "A known bound is the header's, for
  the reason a shift's count is the node's (above).": "For a bound that is a value the reason is
  reversed: a rule reaching it is the point — `PeelFold`'s substitution turning an inner `0..i`
  into `0..k` is how a range becomes known (L8, `KnownBound`)." (plan §3.2; §5's row for
  `egraph/node.rs:226-239`, now these lines.)
- **`ENode::known_range`'s doc**: §3.6.
- `as_reduce`, `shape`, `children_slice(_mut)` and `Operands::fold` (`node.rs:152-163`) are
  already right (§5).

### 6.2 `EGraph::add`: the L3 backstop (graph.rs:359-391)

After the memo lookup (`:361-363`) and before the variance is read (`:368`):

```rust
// L3's backstop (plan §3.2): a bound read off the classes it is made of,
// as `push_reduce` reads one off its term, for a node a rule or a caller
// built. A fold whose bounds are both literals has none and is not looked
// at. Read before the graph changes: a refusal leaves it as it was.
if let ENode::Reduce { header, children } = &node
    && header.arity() > 1
{
    self.refuse_a_varying_bound(header, children);
}
```

```rust
/// Panic unless each value bound of the fold `header` and `children` name
/// varies with neither a coordinate nor the fold's own binder (L3, through
/// `Fold::admits_bound`), read off the bound classes' facts.
fn refuse_a_varying_bound(&self, header: &FoldHeader, children: &Operands) {
    let fold = header.fold();
    let (lo, hi) = header.attach(children[1..].iter().copied());
    for (end, bound) in [(BoundEnd::Lo, lo), (BoundEnd::Hi, hi)] {
        let Bound::Value(class) = bound else {
            continue;
        };
        if let Err(refusal) = fold.admits_bound(end, self.variance(class)) {
            panic!("{refusal}: {fold:?}");
        }
    }
}
```

`node` is canonical here (`:360`), so each child is its class's root and `self.variance(class)`
(`:407-410`) reads its fact. `add`'s `# Panics` (`:354-358`) gains: "On a fold whose value bound
varies with a coordinate or with the fold's own binder ([`BoundRefusal::BoundVaries`], L3): the
backstop behind `ExprArena::push_reduce`, for a node a rule or a caller built." It does not refuse
bound arithmetic (plan §1.4): `HalveFold`'s `binder + s` in an inner bound is the rules' own, and
exact. A class fact is the meet of its members' variances, and a member's is computed from its
children's facts, so a bound class inserted from an arena that passed `push_reduce`'s L3 passes
here; a merge only shrinks a fact, so a node the graph holds never becomes illegal.

### 6.3 `insert` (insert.rs): discovery, in `children()` order

`insert` reads a literal bound into the node as `push_reduce` does (plan §3.2), so the e-graph's
node for a term is the arena's.

- **Visit** (`insert.rs:170-181`):

  ```rust
  Shape::Reduce { fold, body, lo, hi } => {
      // Constness discovered as `push_reduce` discovers it: a value bound
      // whose term is a literal naming a count is the node's data, neither
      // visited nor made a class. Refused before anything is built.
      let (lo, hi) = Bound::discover(lo, hi, |r| literal(term, r))
          .unwrap_or_else(|refusal| panic!("{refusal}: {fold:?}"));
      tasks.push(Task::Complete(r));
      // Reverse, so children pop in `children()` order: the body, then
      // `lo`'s, then `hi`'s.
      for child in [hi.value(), lo.value()].into_iter().flatten() {
          tasks.push(Task::Visit(child));
      }
      tasks.push(Task::Visit(body));
      continue;
  }
  ```

- **Complete** (`:211-220`):

  ```rust
  Shape::Reduce { fold, lo, hi, .. } => {
      let (lo, hi) = Bound::discover(lo, hi, |r| literal(term, r))
          .expect("a fold's bounds were discovered when it was visited");
      // The header refuses a value bound its fold cannot take before
      // `built` is touched.
      let header = FoldHeader::new(fold, lo.erased(), hi.erased());
      // ... today's `:214-219`
  }
  ```

- **`inserted_class_count`** (`:264-268`):

  ```rust
  // A value bound's class counts, as `insert` makes it; one read as a
  // literal makes none. A mask, which `insert` refuses, counts as the
  // class it would be: this is a measure, not a check.
  Shape::Reduce { body, lo, hi, .. } => {
      stack.push(body);
      let (lo, hi) = Bound::discover(lo, hi, |r| literal(term, r)).unwrap_or((lo, hi));
      stack.extend(lo.value());
      stack.extend(hi.value());
  }
  ```

- A private helper beside `insert`: `fn literal<I: Ir>(term: &I, r: I::Ref) -> Option<Lit> {
  match term.project(r) { Shape::Const(lit) => Some(lit), _ => None } }`.

For an arena this discovery finds nothing: `push_reduce` already read every in-domain `Const`
bound. It is for any other `Ir` and keeps the node equal to the arena's (J11). The mask case
panics rather than declining: `BoundSort` everywhere (N4; J21).

### 6.4 `KnownBound` (fold_rules.rs, new; L8)

```rust
/// `⊕_{[lo,hi)} f = ⊕_{[n_lo, n_hi)} f` where the fold's value bounds' classes
/// hold literals naming counts: **L8, discovery**, the one normalizing rule
/// (docs/language/2026-10-20-a-bound-is-a-value.md §3.2). A value bound
/// whose class gains a `Const` member — `PeelFold` substituting `i := k`
/// into an inner `0..i` is how one does — is that count, and the fold over
/// it joins the fold's class. The known form is the one `PeelFold`,
/// `HalveFold` and `EmptyFold` read (L6), so the inner fold becomes
/// unrollable; nothing declared it.
///
/// It fires only when the read makes the whole range known (J12): a fold
/// with one bound known and one a value is no more unrollable or priceable
/// than before, and would spend a minted id and an application on a node no
/// rule reads. A literal naming no count — a fraction, a negative, past 2²⁴
/// — is no count, and a mask is no count of any sort: the rule declines on
/// both (L5: such a fold is NaN, by A4's guard). A pair it makes known runs
/// forwards (L2: `[k, j)` with `k > j` is the empty range at `k`), which
/// `EmptyFold` then closes.
///
/// The member is found by scan, as `math::algebra`'s `constant_of` finds
/// one, not through the class's constant fact (plan §3.2).
///
/// **Match depth 1**, inside `DIRTY_TRACKING_MAX_DEPTH`: it reads the
/// members of the fold's bound classes. A `Const` joins a bound class by a
/// union, a content change to that class, which the dirty tracker watches.
///
/// Last in [`fold_rules`], so every rule before it keeps its `RuleIdx`, and
/// inert on a fold over literals: it declines before reading the graph.
pub struct KnownBound;

impl Rewrite for KnownBound {
    fn name(&self) -> &str {
        "known-bound"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Reduce)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // A fold over literals has nothing to discover: declined on its
        // header alone.
        let ENode::Reduce { header, .. } = node else {
            return None;
        };
        if header.arity() == 1 {
            return None;
        }
        let (fold, lo, hi, body) = node.as_reduce()?;
        let literal = |class: EClassId| egraph.nodes(class).find_map(ENode::literal);
        let (lo, hi) = Bound::discover(lo, hi, literal).ok()?;
        let (Bound::Lit(_), Bound::Lit(_)) = (lo, hi) else {
            return None;
        };
        Some(RewriteAction::Create(ENode::reduce(fold, lo, hi, body)))
    }
}
```

`fold_rules()` (`fold_rules.rs:574-581`) appends `Box::new(KnownBound)` after `FactorFold`; its doc
gains "and [`KnownBound`] to make a range known when the graph discovers a bound's constant
(L8)". The member scan is written here, not imported: `constant_of` is private to
`math/algebra.rs` (`:1060-1063`) and is this one expression: a one-line iterator chain is the
definition, not a mechanism to share, and widening a private function for it is not worth the
cross-module edge. `egraph.nodes(class)` resolves the class through
`find` (`graph.rs:604-605`).

### 6.5 Rules gated on known ranges (L6)

Already true at `3bfdaded`: `PeelFold` and `HalveFold` decline on `node.known_range()?`
(`fold_rules.rs:393`, `:435`), `EmptyFold` fires on `is_some_and(KnownRange::is_empty)`
(`:479-482`), and none fires on class equality of the bounds (plan L6). A3 adds no code here; T-S-7
pins the declines on a uniform-bounded fold.

### 6.6 `FactorFold` under a value bound (decision 5)

- `fold_rules.rs:497-501`:

  ```rust
  // A range known to be empty is `EmptyFold`'s. One that is not known
  // factors: at a runtime empty range the factored form is `c ⊗ identity`,
  // the edge decision 5 accepts (doc above).
  if node.known_range().is_some_and(KnownRange::is_empty) {
      return None;
  }
  ```

- The doc's "**An empty fold declines.**" paragraph (`:355-358`) becomes: "**A fold known to be
  empty declines.** `Σ_∅ (c·f) = 0` but `c·Σ_∅ f = c·0`, which is NaN for an infinite `c`, and
  `min_∅ (c+f) = +∞ = c + ∞` for every `c` but `−∞`; [`EmptyFold`] closes a known empty fold
  outright. A fold whose range is not known factors, accepting that edge at a runtime empty range
  under "edge-case IEEE conformance is not on offer" (CLAUDE.md; decision 5 of
  docs/language/2026-10-20-a-bound-is-a-value.md, JP 2026-10-22): declining would lose `c·Σ f` for
  every per-call count."
- The factored fold keeps the node's own bounds (`:507-514` already maps each value to its class),
  so the plan replays through `ENode::reduce` and `add`'s L3 over the same bound classes.
  Soundness at an unknown trip count `n ≥ 1` is the law itself; at an out-of-domain bound both
  sides are NaN (A4's guard).

### 6.7 `HalveFold`'s rule-made bound arithmetic

`HalveFold`'s code is unchanged; its doc is amended (round 4, F2). `HalveFold` substitutes
`binder := binder + s` through `substitute`, which enters an inner fold's value bounds (L7), so an inner `0..i` becomes `0..(i + s)`: a plan
`HeadNode::Op { op: Add, children: [Class(i), Plan(s)] }` as the inner `HeadNode::Reduce`'s `hi`.
`plan_shape` (`graph.rs:1626-1631`) builds it through `ENode::reduce` (a value bound under the
inner fold's own monoid, stride 1: accepted) and `add` (a new node's fact is `{i}`, the outer binder:
accepted; a node that hash-conses into an existing class is judged on that class's fact, J27). It is bound arithmetic, which only the user surfaces refuse (plan §1.4): it stays
inside the outer fold's known range, which ends at or before 2²⁴, so it is exact. `PeelFold` plans
`hi − s` as a literal (`KnownRange::peel_back`), on a known range only. T-S-6 pins it.

### 6.8 `substitute` and `rebuild`

`substitute` (`fold_rules.rs`) is **changed in round 4 (F2)**; the version §5 described walked
every class through `representative`, its first member, and that is unlawful for a class reached
as a nested fold's value bound. A class's variance fact is the meet over its members, so its first
member can read a coordinate the fact lacks (`((i + X) + 1) − X` beside `i + 1`: older class,
older member first), and the rebuilt bound reads `X`, which `EGraph::add`'s backstop refuses by
panicking inside saturation (`.wip/a3/run/scratch-r4-run1.txt`: `Σ_{i<8} ( Σ_{j<i} X·j + ((i + X) +
1) − X )` under `RuleSet::runtime()` at `compatibility(12)`). The walk now carries a `Site`:

- `Site::Body` (every site of a fold over literals, and everything under a fold's body): the first
  member, `egraph.nodes(class).next()` (`representative` was inlined in polish round 1: it had one
  caller). Its memo is the old `BTreeMap<EClassId, Done>`, so the literal path builds no wider key
  (`Memo.body`).
- `Site::Bound { fold }` (a nested fold's value bound, and every class under one): the first member
  that is no `ENode::Reduce` and whose own variance, `ENode::variance` over its children's class
  facts, `fold.bound_may_read` accepts. `None` when no member does, and the plan declines. A fold
  under a bound is skipped as a member, not declined on: the walk does not enter one (its body would
  need the law lifted by its own binder), and saturation can put a fold into a bound's class
  (`RuleSet::runtime()` derives `Σ_{k<1} i = i`; `PeelFold` alone does not), so "no builder makes
  one" was wrong. `Site::of_child` is
  total because of that skip.

The substitution's own effect needs no accounting in the check: a leaf becomes a literal or the
substituted binder plus its stride, and that binder is neither a coordinate nor the nested fold's
own (a fold that rebinds it is named before it is entered, L4), so the refused set never contains it.
By induction over the rebuilt bound's classes, every class the plan *mints* under the bound has a
fact within what the nested fold's L3 admits. A node that hash-conses into a class the graph already
holds is judged on that class's fact, which is seeded wider than the term when a union has since
shown a coordinate does not matter and is never repaired upward; the walk cannot know which classes
are stale, and the graph answers instead (J27). The version of this paragraph that claimed "a
member's variance before the substitution is the law's answer after it" was false for that reason. Pins: `a_nested_bound_is_rebuilt_through_the_member_that_keeps_l3` (HalveFold at
n=4 and PeelFold at n=1, each alone, over a hand-made union; both **plan** through the lawful member
`i + 1` rather than decline, because the class has one), `a_nested_bound_with_no_member_that_keeps_l3_declines`
(`i + X` beside `i + Y`: the class reads neither, no member keeps L3, both rules decline), and
`the_runtime_rules_saturate_a_triangular_fold_with_a_varying_spelling` (the repro and three variants
under `RuleSet::runtime()`, extracting a closed term whose bounds read the outer index alone). Each
fails without the change: the third with the `BoundVaries` panic. The extraction-side hole is J24.

`rebuild`'s comment
(`fold_rules.rs:620-626`) becomes:

```rust
// A nested fold reached here does not rebind the slot being substituted,
// so the substitution passes through its body and its value bounds alike
// (L7). One that rebinds the slot is never reached: its class's variance is
// `(var(body) \ slot) ∪ var(lo) ∪ var(hi)` (L4), and no bound reads its own
// fold's slot (L3), so the fact lacks the slot and `substitute` names the
// class before descending. "`lowest_free_binder` never reissues a live
// slot" is not what makes that safe — it is false across a `Ref`, which
// `Kernel::over` does not see through.
```

The `rebuild` code (`:627-639`) already attaches every planned value bound.

---
## 7. Pricing (N3)

### 7.1 What a fold is priced at

A fold over a known range is priced at its length, as today. A fold with a value bound is priced
at the count its bounds' **chosen forms** name when each is an in-domain integer literal — the
price of its literal twin, which `KnownBound` puts in the same class — and otherwise at
`Fold::ESTIMATED_TRIPS`. Pricing therefore reads the bound children's extracted picks, and the DP
already holds them: `Settlement::run` (`extract.rs:1948-2034`) relaxes a candidate only when
every child class it names has settled (`waiting` reaches 0, `:1958-1963` and `:1990-2003`), and a
fold's children are its body and its value bounds (`children_slice`). So when a fold is priced,
each bound class's pick is final.

`FoldHeader::priced_trips` (§2.8) stays the header-side reader (literal or estimate). The
extractor computes the effective trips from the picks by discovery and hands them to the fold's
two price halves through the fold's **priced form**, the literal twin: `CostFunction::node_cost`
of the twin is `CostModel::node_op_cost`'s `fold_cost(fold, twin.priced_trips(), 0)`, and
`below` multiplies by the form's `priced_trips()`. This is how "the extractor passes the effective
trips to `fold_cost` and `below`" is met for an extractor generic over `CostFunction` (J7):
`fold_cost` is a `CostModel` method, and a wrapping cost function (`LoopsCostMore`,
`fold_rules.rs:1369-1378`, `halve_fold_jit.rs:55-63`) prices the twin as it prices any fold.

### 7.2 The header side: `CostModel` (cost.rs)

- `node_op_cost`'s fold arm (`cost.rs:399-405`):

  ```rust
  ENode::Reduce { header, .. } => {
      let fold = header.fold();
      match super::fold_rules::combiner_op(fold.monoid()) {
          Some(_) => self.fold_cost(fold, header.priced_trips(), 0),
          None => usize::MAX / 4,
      }
  }
  ```

  Its comment gains one sentence after "So the node's own share is `fold_cost` with the body
  free.": "Its trips are the header's (`FoldHeader::priced_trips`): the range's length, or the
  estimate when a bound is a value; the extractor prices a fold whose bounds it chose as literals
  through the fold over them (`ENode::literal_twin`)."
- `fold_cost` (`cost.rs:408-431`) takes the trips:

  ```rust
  /// **What one evaluation of `fold` costs**, when it runs `trips` times
  /// and one evaluation of its body costs `body`: the body once per trip,
  /// and the combiner between consecutive trips —
  /// `trips · body + (trips − 1) · combine`.
  ///
  /// The one formula for a fold's price. [today's second paragraph, kept.]
  ///
  /// `trips` is read off the node that holds the fold
  /// ([`FoldHeader::priced_trips`]): its range's length when the range is
  /// known, else [`Fold::ESTIMATED_TRIPS`](pixelflow_ir::Fold::ESTIMATED_TRIPS);
  /// the extractor reads it off a fold's literal twin when it chose the
  /// fold's bounds as literals.
  ///
  /// Saturating: [today's, kept.]
  #[must_use]
  pub fn fold_cost(&self, fold: Fold, trips: u64, body: usize) -> usize {
      let trips = usize::try_from(trips).unwrap_or(usize::MAX);
      let combines = trips.saturating_sub(1);
      trips
          .saturating_mul(body)
          .saturating_add(combines.saturating_mul(self.cost(fold.combine_op())))
  }
  ```

  `cost.rs:18`'s `use pixelflow_ir::fold::KNOWN_RANGE;` goes; the test's own import
  (`cost.rs:742`) may stay.

### 7.3 The extractor (extract.rs): the DP order, `literal_twin`, `below`

**`ENode::literal_twin` (new, `pub(crate)`, in `node.rs`; round 4, F3).** It was a private function
in `extract.rs` beside a copy of its body in `KnownBound::apply`; now there is one, and both call
it, so the docs no longer assert that the two agree. The text below is the function as it was
first specified in `extract.rs`; it is now `ENode::literal_twin(&self, literal: impl Fn(EClassId)
-> Option<Lit>) -> Option<ENode>`, `node.literal_twin(chosen)` at the three price sites and
`node.literal_twin(|class| egraph.nodes(class).find_map(ENode::literal))` in `KnownBound::apply`.

**`literal_twin` (as first specified):**

```rust
/// The fold `node` is priced as when it is not priced as itself: a fold
/// whose value bounds were chosen as literals naming counts is priced as the
/// fold over those counts — the literal twin `KnownBound` puts in its class,
/// so the two price alike (docs/language/2026-10-20-a-bound-is-a-value.md
/// §3.2). `chosen` is the literal a class was chosen as, `None` when its
/// chosen form is no literal. `None` for every other node: one that is no
/// fold, a fold over literals, and a fold over a value whose chosen form
/// names no count (a mask among them), each priced as it stands — the last
/// at `Fold::ESTIMATED_TRIPS`.
fn literal_twin(node: &ENode, chosen: impl Fn(EClassId) -> Option<Lit>) -> Option<ENode> {
    let ENode::Reduce { header, .. } = node else {
        return None;
    };
    if header.arity() == 1 {
        return None;
    }
    let (fold, lo, hi, body) = node.as_reduce()?;
    let (lo, hi) = Bound::discover(lo, hi, chosen).ok()?;
    let (Bound::Lit(_), Bound::Lit(_)) = (lo, hi) else {
        return None;
    };
    Some(ENode::reduce(fold, lo, hi, body))
}
```

The twin is built only to be priced: it is never added to the graph. It is known by construction,
so `ENode::reduce` takes `FoldHeader::new`'s literal path (discovery normalized it, L2).

**`below`** (`extract.rs:1651-1704`) keeps its signature and reads the trips off the node it is
handed, which is now the priced form:

```rust
fn below(node: &ENode, mut operand: impl Iterator<Item = usize>) -> usize {
    match node {
        ENode::Reduce { header, .. } => {
            let body = operand.next().expect("a fold has a body");
            let body = usize::try_from((body as u64).saturating_mul(header.priced_trips()))
                .unwrap_or(usize::MAX);
            // Each value bound is evaluated once per evaluation of the fold:
            // plan §1.5's `+ cost(lo) + cost(hi)`.
            operand.fold(body, usize::saturating_add)
        }
        // ... the `If` arm and the default arm, unchanged.
    }
}
```

Its doc: "**What `node`'s children cost per evaluation of `node`**, from each operand's cost in the
order of the children of the node as written" (unchanged otherwise), and the fold paragraph gains:
"The trips are the node's (`FoldHeader::priced_trips`), and a price site hands `below` a fold's
priced form (`literal_twin`), so a fold whose value bounds were chosen as literals is multiplied
by their count. Its operands are still those of the fold as written: the body, then each value
bound, charged once — before A3 `below` read only the body, so bound arithmetic would have gone
unpriced." `extract.rs:15`'s `use pixelflow_ir::fold::KNOWN_RANGE;` goes.

**The three price sites.**

- `TreePricer` (`extract.rs:2075-2142`) gains a field
  `chosen: Vec<Option<Lit>>` ("By e-class id, the literal each settled class was chosen as: what a
  fold whose bound is that class is priced by."), initialized in `tree_dp_pass` (`:2059-2069`) to
  `alloc::vec![None; problem.egraph().num_classes()]` beside `var`. `settle` (`:2130-2135`) adds,
  first:

  ```rust
  let chosen = self.problem.egraph().node(self.problem.graph().node_id(candidate));
  self.chosen[candidate.class().index()] = chosen.literal();
  ```

  `price` (`:2087-2122`) computes the form before `own`:

  ```rust
  let var = node_variance(self.problem.egraph(), node, &self.var, candidate.class());
  let egraph = self.problem.egraph();
  let twin = literal_twin(node, |class| self.chosen[egraph.find(class).index()]);
  let form = twin.as_ref().unwrap_or(node);
  let own = weighted_own(self.dp.costs, form, self.dp.shape.evals(var));
  ```

  and its compound arm ends `own.saturating_add(below(form, settled))`. `settled` still iterates
  `node.children_slice()`, the node as written. The variance is the node's as written: L4 reads
  its bounds.
- `SharedPricer` (`:2321-2380`) gains the same `chosen` field (initialized in
  `shared_dag_dp_pass`, `:2174-2190`), the same two lines first in `settle` (`:2351`), and in
  `price` (`:2332-2344`) the same `twin`/`form` lines with `own = weighted_own(self.dp.costs,
  form, …)`. `cost_below` is unchanged: a bound class is a member of the fold's reach like any
  child, its own cost paid once.
- `Selection::price` (`:996-1031`) reads a bound's pick off the selection itself:

  ```rust
  let chosen = |class: EClassId| {
      let class = egraph.find(class);
      let at = operands
          .iter()
          .find(|&&operand| self.picks[operand].class == class)
          .expect("a chosen node's operands are picks");
      self.node(egraph, *at).literal()
  };
  let twin = literal_twin(node, chosen);
  let form = twin.as_ref().unwrap_or(node);
  let own = weighted_own(costs, form, shape.evals(node_variance));
  // ...
  let below = below(form, operands.iter().map(|&operand| tree[operand]));
  ```

  The selection was made from the settlement's picks, so its operand nodes are the classes'
  settled nodes: the claim and the price read the same literals, and `costed`'s claim/price audit
  (`:1563-1618`) holds. The other `price` callers (`optimizer.rs:811`, `witness.rs:199`, `:279`,
  `:288`, `:543`) go through this one function.

`dense_reference_pass` (a test, `:2720-2800`) calls `below(node, …)` on literal graphs and needs
no change. `Selection::to_arena` (`:935-984`) is unchanged: `embed` is `push_reduce`, which reads
a chosen `Const` bound into the fold, so the arena holds the twin the price was read off.

### 7.4 What A3 adds to a price, stated

For a fold `F` with value bound `v` and body `b`, under the tree objective:
`cost(F) = own(F') + n(F')·cost(b) + cost(v)`, where `F'` is `literal_twin(F)` when `v`'s pick is a
literal naming a count and `F` otherwise, `own(F') = fold_cost(fold, n(F'), 0)` weighted by
`evals(var(F))`, and `n(F') = F'.header.priced_trips()` — the count, or 32. Under the shared
objective `v`'s class is one more member of the reach, paid once.

### 7.5 Proof: an all-literal fold's visit order and price are unchanged

Take any graph in which every `ENode::Reduce` has `header.arity() == 1` (every graph today).

1. **Rows and children.** `children_slice` and the AND-OR rows are unchanged (§6.1 changes no
   node), so `waiting`, `parents` and the first relax sweep (`:1956-1963`) are unchanged.
2. **A non-fold candidate.** `literal_twin` returns `None` at its first `let … else`; `form` is
   `node`; `own`, `below` and the leaf arm are today's expressions on today's arguments.
3. **A fold candidate.** `literal_twin` returns `None` at `header.arity() == 1`; `form` is
   `node`. `own` is `weighted_own(costs, node, w)`, and `node_op_cost` gives
   `fold_cost(fold, header.priced_trips(), 0)`, where `priced_trips() = known_range().len()` and
   `usize::try_from` equals today's `range.len() as usize` on a 64-bit host. `below(node, …)`
   multiplies the body by the same `len` with the same saturating expression, then folds the
   remaining operands, of which there are none (a literal fold's `children_slice` is `[body]`).
   So `Priced { cost, own, var }` is bit-identical.
4. **Order.** The heap key is `(cost, class id, live)` (`:1907-1911`) and `prefer` is unchanged,
   so every relax takes or declines as today, every pop settles the same class on the same
   candidate, in the same sequence; `Settling::settle`'s new write to `chosen` is read only by
   `literal_twin`, which on these graphs returns before reading it.
5. **Outcome.** The choices, `root_cost`, the selection, `Selection::price` (by 2 and 3 over the
   same picks), `total_cost`, `dag_cost` and the shared pass's reach bytes (`sets.bytes` counts
   reach sets only) are today's; so are the arena `to_arena` builds and telemetry's
   `extracted_latency_prior_cost`, its own walk over that arena (`telemetry.rs:235-280`).

What A3 adds on this path is one `Vec<Option<Lit>>` of one entry per class per pass and one node
lookup and store per settle; the instruction gate (§11) holds it to the noise.

### 7.6 The twins pin

T-S-13 (§10): a class holding `Σ_{j<c} |X − j|` with `c`'s class `{Const(40)}` and `Σ_{j<40}
|X − j|`, unioned, prices both candidates at exactly `fold_cost(fold, 40, cost(Sub) + cost(Abs))`;
`40 > ESTIMATED_TRIPS`, so a header-only price (32) would make the value twin the cheaper and
the pin would fail.

---
## 8. Codegen (A3 keeps the refusal)

Nothing in `pixelflow-codegen` changes behaviour. A value-bounded kernel is refused before
codegen sees it on every compile path — `KernelKey::of`/`canonical` call `to_bits`, which panics
with `VALUE_BOUND` — and `arena_to_schedule`'s `known` (`program/lower.rs:47-52`) refuses one by
the same name for a caller that schedules an arena directly. `mark_reachable` (`:23-45`) and
`is_empty_fold` (`:55-59`) already read a value bound's range as "not known to be empty".

- The comment at `program/lower.rs:341-344` becomes: "Codegen takes no value bound until A4
  places its operand (docs/language/2026-10-20-a-bound-is-a-value.md §3.3); the IR and the
  e-graph hold one, and `known` refuses it here by name." (B1.)
- `known`'s doc (`:47-48`) is already true ("refused by name until A4 places its operand").
- `pixelflow-codegen/src/program/intent.md:20-21` ("literal until A4: lowering refuses a value
  bound") is already true; no edit.
- T-C-1 (§10) pins the refusal at `arena_to_schedule`.

## 9. The compiler (mechanical: signature only, no arena change)

`kernel!` builds no value bound before A6. Its two `close_fold` sites take `FoldTerms` where they
took a `fold_at` closure; the arena each builds is unchanged, node for node. This is last in the
order (§11) because #1427 rewrites these files (plan:483-487) and the diff is signature-only.

- **`Expansion::close_fold`** (`pixelflow-compiler/src/lower.rs:497-522`):

  ```rust
  let terms = match fold {
      ExpansionFold::Known(monoid, range) => FoldTerms {
          monoid,
          stride: 1,
          lo: Bound::Lit(range.start),
          hi: Bound::Lit(range.end),
      },
      ExpansionFold::Hole(monoid, stride) => FoldTerms {
          monoid,
          stride,
          lo: Bound::Lit(OPEN_RANGE.start),
          hi: Bound::Lit(OPEN_RANGE.end),
      },
  };
  self.arena
      .close_fold(open, body, terms)
      .map_err(/* today's, `:516-521` */)
  ```

  `Fold::strided(monoid, binder, stride)` for a hole is now built inside `close_over`
  (`Fold::strided` from `FoldTerms::stride`), with the same panics. The imports of `Fold` and
  `FoldHeader` in `lower.rs` go if nothing else there names them; `FoldTerms` is imported from
  `pixelflow_ir::arena`.
- **`Staged::fold_at`** (`pixelflow-compiler/src/emit.rs:701-732`) is renamed `fold_terms` and
  emits a `FoldTerms` value instead of a closure. The monoid still travels as header bits decoded
  on the far side (its opcode is crate-private), exactly as today:

  ```rust
  fn fold_terms(monoid: Monoid, range: FoldRange) -> TokenStream {
      let first = Binder::from_slot(0).expect("the IR has a binder");
      let bits = pixelflow_ir::FoldHeader::new(
          pixelflow_ir::Fold::new(monoid, first),
          pixelflow_ir::Bound::Lit(0),
          pixelflow_ir::Bound::Lit(0),
      )
      .to_bits();
      let range = /* today's `:709-715` */;
      quote! {{
          let __range = #range;
          ::pixelflow_core::__macro::ir::arena::FoldTerms {
              monoid: ::pixelflow_core::__macro::ir::fold::FoldHeader::from_bits(#bits)
                  .expect("kernel! emitted a well-formed fold")
                  .fold()
                  .monoid(),
              stride: 1,
              lo: ::pixelflow_core::__macro::ir::fold::Bound::Lit(__range.start),
              hi: ::pixelflow_core::__macro::ir::fold::Bound::Lit(__range.end),
          }
      }}
  }
  ```

  `Staged::open_fold` (`:815-830`) stores `Self::fold_terms(monoid, range)`; `Staged::close_fold`
  (`:832-844`) emits `__arena.close_fold(#fold, #body, #terms)`. The `__range` expression is
  evaluated where the closure body evaluated it, at the call, so the instantiation it reads is the
  same. `Staged` builds a structural fold at stride 1 over `instantiated_range`'s range, as today's
  closure does (the hole's stride is the expansion arena's, not the staged code's).
- `arena_to_tokens`' fold arm (`emit.rs:577-614`) is unchanged; it is A4's (§5).

Its own gate: `cargo test -p pixelflow-compiler` green — `rustc_is_the_oracle.rs`,
`fold_is_kernel_over.rs`, `kernel_typed_parameters.rs`, `binding_times.rs` among them — and
byte_probe empty on the rows built through `kernel!`.

---
## 10. Tests (the gate's pins, and the rewrites A3 forces)

Each test is named as it is written; its doc states the law it pins, in the file's register.
`b(s)` below is `Binder::from_slot(s).expect(..)`, `sum(s)` is `Fold::new(Monoid::SUM, b(s))`.

**The L8 and `HalveFold` pins are e-graph-level (N8).** They saturate a `Saturation` over
`fold_rules()` alone — no `ConstantFold`, no production `Optimizer` — because the key refuses a
value bound until A4 (`to_bits`, §0), so no production path can carry one, and because without
`ConstantFold` a rule-made `0 + 1` stays the arithmetic it is, which T-S-6 must allow.

### pixelflow-ir unit tests, `src/fold.rs`

- **T-IR-1** `a_count_is_a_non_negative_integer_an_f32_counter_holds_exactly`: `Fold::count` is
  `Some(0)` for `Real(0.0)` and `Real(-0.0)`, `Some(7)` for `7.0`, `Some(Fold::EXACT_BOUND)` for
  `16777216.0`; `None` for `16777218.0`, `2.5`, `-1.0`, NaN, `+∞`, `Mask(true)`, `Mask(false)`.
- **T-IR-2** `the_exact_bound_is_stated_once` (N10): for `n` in `[0, 7, EXACT_BOUND,
  EXACT_BOUND + 2]`, `KnownRange::admits(&(0..n))`, `Fold::count(Lit::Real(n as f32)).is_some()`
  and `Fold::is_exact(n)` agree.
- **T-IR-3** `discovery_reads_a_literal_term_and_keeps_every_other`, over `u8` terms `0 → 2.0`,
  `1 → 9.0`, `2 → None`, `3 → 2.5`, `4 → 16777218.0`, `5 → Mask(true)`:
  `discover(Value(0), Value(1)) == Ok((Lit(2), Lit(9)))`;
  `discover(Value(2), Value(1)) == Ok((Value(2), Lit(9)))`;
  `discover(Lit(0), Value(3)) == Ok((Lit(0), Value(3)))` and the same for `Value(4)` (N4: kept);
  `discover(Lit(0), Value(5)) == Err(BoundSort { end: Hi })`;
  `discover(Value(5), Lit(3)) == Err(BoundSort { end: Lo })`;
  `discover(Lit(7), Lit(3), |_| panic!("a literal pair reads no term")) == Ok((Lit(7), Lit(3)))`.
- **T-IR-4** `a_pair_discovery_makes_known_runs_forwards` (L2), terms `0 → 5.0`, `1 → 3.0`:
  `discover(Value(0), Value(1)) == Ok((Lit(5), Lit(5)))`, `discover(Lit(9), Value(1)) ==
  Ok((Lit(9), Lit(9)))`, and `FoldHeader::new(sum(0), Lit(5), Lit(5)).known_range()` is empty.
- **T-IR-5** `a_value_bound_takes_a_real_sorted_fold_at_stride_one` (replaces
  `a_value_bound_is_refused_by_name_until_a3`): for `SUM`, `PRODUCT`, `MIN`, `MAX`,
  `FoldHeader::new(Fold::new(m, b(0)), Lit(0), Value(()))` has `arity() == 2`,
  `known_range() == None`, `hi() == Value(())`, `priced_trips() == Fold::ESTIMATED_TRIPS`; and
  `Monoid::{ANY, ALL, SEQ}.takes_a_value_bound()` is false, the other four true.
- **T-IR-6a/6b/6c** `a_value_bound_on_an_any_fold_is_refused`, `…_all_…`, `…_seq_…`:
  `#[should_panic(expected = "BoundSort")]`.
- **T-IR-7** `a_value_bound_needs_stride_one`: `FoldHeader::new(Fold::strided(SUM, b(0), 2),
  Lit(0), Value(()))`, `#[should_panic(expected = "only over a known range it divides")]`.
- **T-IR-7b** `a_literal_beside_a_value_is_in_the_domain`: `FoldHeader::new(sum(0),
  Lit(EXACT_BOUND + 1), Value(()))`, `#[should_panic(expected = "a fold's literal bound is at most
  16777216 (2^24)")]`.
- **T-IR-8** `a_value_bound_has_no_bits_until_a4` (renamed; header from `FoldHeader::new`):
  `#[should_panic(expected = "a fold bound that is a value")]`.
- **T-C-2** (round 4, F1) `a_surviving_placeholder_fails_loudly`: a `Var` of `Placeholder::nth(0)`
  times a constant: `arena_to_schedule` panics with "escaped its builder" (the `Var` arm's refusal;
  variance types a placeholder `CONST` and would let placement see a constant).
- **T-IR-8b** `priced_trips_is_the_known_length_or_the_estimate`: `0..40 → 40`, `3..3 → 0`, a
  value bound → `32`; `Fold::ESTIMATED_TRIPS == 32`.
- **T-IR-8c** `a_fold_admits_a_bound_reading_neither_a_coordinate_nor_its_own_binder` (L3's one
  statement): over `sum(0)` (binder `Var(4)`), `admits_bound(Lo, CONST)` and `(Hi,
  from_var(5))` are `Ok`; `(Hi, X)`, `(Lo, Y)` and `(Hi, from_var(4))` are `Err(BoundVaries {
  end, reads })` with the given `end` and `reads`.
- **T-IR-8d** `a_bound_refusal_names_its_law`: each variant's `Display` starts with its name and
  holds its law: `BoundVaries` "L3", `BoundOutOfDomain` "L5" and the literal, `BoundSort`
  "decision 8", `BoundArithmetic` "T6", `BoundNotANumber` "`Param`"; `BoundEnd`'s are `lo`, `hi`.
- `a_value_bound_is_not_the_literal_zero`: headers from `FoldHeader::new`; doc per §2.12.

### `src/variance.rs`

- **T-IR-9** `of_var_names_a_coordinate_and_a_binder_and_no_placeholder`: `of_var(0) == X`,
  `of_var(1) == Y`, `of_var(2) == from_var(2)`, `of_var(4) == from_var(4)`, `of_var(63) ==
  from_var(63)`, the first and last placeholders' `var()` give `CONST`, `of_var(128)` and
  `of_var(255)` give `ALL`.
- **T-IR-10** `a_fold_varies_with_its_bounds_and_its_body_less_its_index` (L4): body `Var(4)·X`;
  over `sum(0)` with `hi` a uniform the fold is `X`; with `hi` `Var(5)` it is `X ∪ {i5}`; over
  `Lit(2)..Lit(2)` it is `CONST`.
- **T-IR-11** `the_rooted_walk_is_the_forward_pass_at_every_node` (N1): for every node of the
  arenas of `nested_binders_consume_one_index_each`, `a_write_varies_with_its_value_and_its_binders`,
  T-IR-10's, and (under `feature = "std"`) one holding a `Ref` to `Kernel::x().sqrt()`,
  `variance_from(&arena, id) == compute_arena_variance(&arena)[id]`.

### `src/arena.rs`

- **T-IR-12** `push_reduce_reads_a_literal_bound_into_the_node` (discovery, L1): with `c =
  push_const(7.0)`, `push_reduce(sum(0), Lit(0), Value(c), body)` returns the id
  `push_reduce(sum(0), Lit(0), Lit(7), body)` returns; its `children` are `[body]`, its range
  `0..7`.
- **T-IR-13** `push_reduce_keeps_a_literal_naming_no_count_as_a_value` (N4): for `2.5`, `-1.0`,
  `16777218.0`, NaN, `+∞`, the fold's `hi` is `Value(c)`, its children `[body, c]`, its range not
  known.
- **T-IR-14** `push_reduce_refuses_a_mask_bound`: `hi = push_lit(Lit::Mask(true))`,
  `#[should_panic(expected = "BoundSort")]`.
- **T-IR-15a/b/c/d** (L3) `push_reduce_refuses_a_bound_reading_x`, `…_reading_y`,
  `…_reading_its_own_binder` (`hi = Var(b(0).var())` on `sum(0)`), and, under
  `#[cfg(feature = "std")]`, `…_reading_an_x_varying_ref` (`hi = push_ref(KernelStore::intern(
  &Kernel::x().add(&Kernel::constant(1.0))))`): each `#[should_panic(expected = "BoundVaries")]`.
  Replaces `push_reduce_refuses_a_value_bound_by_name` (`:2586-2601`).
- **T-IR-16** `push_reduce_takes_a_bound_reading_an_enclosing_binder_a_uniform_or_a_placeholder`:
  on `sum(0)`, `hi` = `Var(5)`, a uniform, and `Var(Placeholder::nth(0).var())` each build a
  fold whose `hi` is that `Value`.
- **T-IR-17** `the_sort_is_refused_before_the_variance`: an `ANY` fold with `hi = Var(0)`,
  `#[should_panic(expected = "BoundSort")]` (§1.2: not `BoundVaries`).
- **T-IR-18** `free_index_reads_a_value_bound_unmasked`: `inner = push_reduce(sum(0), Lit(0),
  Value(Var(5)), Var(4))` has `free_index == Some(5)`; `push_reduce(sum(1), Lit(0), Lit(3),
  inner)` has `None`.
- **T-IR-19** `close_over_copies_a_value_bound_after_the_body`: `open_body(0)` with a declared
  uniform `u` as `hi`; the closed arena's nodes are `Var(b)`, the body's, the `Uniform`, then the
  `Reduce` whose `hi` is `Value` of the uniform's id; its uniform table is the open arena's.
- **T-IR-20** `close_over_reads_a_literal_bound_and_copies_nothing_for_it` (N2, L1):
  `open_body(0)`, `c = push_const(4.0)`: the arena closed with `hi: Value(c)` and the one closed
  with `hi: Lit(4)` have equal `nodes()` vectors, and the first holds no `Const`.
- **T-IR-21** `close_over_binds_past_a_slot_its_bound_reads` (J8): `open_body(0)`, `hi =
  Var(b(0).var())`: the fold binds slot 1, not 0, and the closed term's `free_index` is
  `Some(b(0).var())`.
- **T-IR-22** `close_over_refuses_a_bound_reading_its_own_placeholder`: `hi =
  Var(placeholder.var())`, `#[should_panic(expected = "BoundVaries")]`.
- The `close_over_*` tests (`:2301-2368`) and
  `a_fold_opened_and_closed_is_close_overs_and_leaves_nothing_behind` (`:2799-2832`) pass
  `FoldTerms { monoid, stride: 1, lo: Lit(..), hi: Lit(..) }`, asserting what they assert today.

### `src/passes.rs`

- **T-IR-23** `a_value_bound_is_a_child_after_the_body`, beside `an_extent_is_not_an_expression`
  (`:1575-1602`): a uniform `hi` makes `children(red) == [body, u]` and `known_range(red) ==
  None`. `an_extent_is_not_an_expression`'s doc (`:1581-1584`) becomes "… and never an `ExprId`,
  so there is no child to put a uniform in. A `Bound::Value` is a child after the body, which the
  next test pins; the property below — the other side of the same rule — is the half that was
  always about semantics." and its comment (`:1599-1601`) "The trip count is not reachable from
  the node's children, so no rewrite can substitute anything for it." (B1.)

### pixelflow-ir integration tests, new file `tests/value_bounds.rs` (public API only)

One fixture: an arena with a uniform `u`, an outer `sum(1)` over `0..3` around an inner `sum(0)`
whose `hi` is `Value(t)`, `t` a term deeper than the body (`Add(Var(5), u)`), body `Var(4)·Y`.

- **T-IR-24** `every_walker_takes_a_value_bound`: `depth(inner)` is one more than `t`'s depth;
  `node_count_subtree(inner)` counts `t`'s nodes; `has_var` is true of a fold over `Const(1.0)`
  whose only `Var` is in `hi`; `has_degenerate` is true of one whose `hi` is `Div(u, 0.0)`;
  `splice` into a fresh arena, `compact`, `rebuild_into` a fresh `ExprArena`, and
  `Kernel::from_parts(..).parts()` each give a term `subtree_eq` to the fixture whose inner `hi`
  is still a `Value`; `display` prints `add_4[0.._)(` and `t` after the body.
- **T-IR-25** `substituting_a_coordinate_leaves_a_bound_alone`: `substitute_vars_with(root,
  &[(Axis::Y.var(), c)])` rewrites the body's `Y` and leaves `t` the same node (N5's caller shape).
- **T-IR-26** `a_value_bounded_kernel_has_no_key_until_a4`: `KernelKey::of` on the closed
  fixture, `#[should_panic(expected = "a fold bound that is a value")]` (A4).
- **T-IR-27** `a_bound_reading_a_free_binder_makes_the_term_open`: the inner fold alone has
  `free_index == Some(5)`, the fixture `None`.
- **T-IR-28** `variance_follows_l4_at_the_public_api`: `compute_arena_variance` gives the inner
  fold `Y ∪ {i5}` and the fixture `Y`.

### `src/kernel.rs` unit tests (`over_values` is `pub(crate)`)

`f` below is `|i| Kernel::x().mul(i).add(&Kernel::y())`.

- **T-K-1** `over_values_over_literals_is_over` (L1, N2): `over_values(SUM, &constant(0.0),
  &constant(4.0), f)` and `Kernel::over(SUM, 0..4, f)` have equal `nodes()` vectors, and the
  first's arena holds no `Const(0.0)` or `Const(4.0)`.
- **T-K-2** `over_values_from_five_to_three_is_the_empty_range` (L2): its root's `known_range` is
  `5..5`, empty, and its arena equals `Kernel::over(SUM, 5..5, f)`'s node for node.
- **T-K-2b** `over_from_five_to_three_is_still_refused` (L2): `Kernel::over(SUM, lo..hi, f)` with
  `(lo, hi) = (5, 3)` through variables, `#[should_panic(expected = "runs forwards")]`.
- **T-K-3** `over_values_takes_a_uniform_count`: `n = Uniform::new(8.0)`, `m = Uniform::new(2.0)`
  read in the body: the root's `hi` is a `Value` whose node is a `Uniform`, its range is not known,
  and `uniforms() == [m.decl(), n.decl()]` (body first, then `hi`).
- **T-K-4** `over_values_takes_an_enclosing_index` (L3: an enclosing binder accepted, through the
  builder): `Kernel::over(SUM, 0..4, |i| Kernel::over_values(SUM, &constant(0.0), i, |j|
  j.mul(&Kernel::x())))`: the outer fold binds slot 1, the inner slot 0; the inner fold's `hi` is
  `Value(v)` with `node(v) == Var(b(1).var())`; `free_index(root) == None`;
  `compute_arena_variance(arena)[root] == X`.
- **T-K-5** `over_values_refuses_arithmetic`: `hi = n.kernel().add(&constant(1.0))`,
  `expected = "BoundArithmetic"`.
- **T-K-6** `over_values_refuses_a_coordinate`: `hi = Kernel::x()`, `expected = "BoundVaries"`.
- **T-K-7a/b/c** `over_values_refuses_a_fraction`, `…_a_negative`, `…_a_count_past_the_exact_bound`
  (`2.5`, `-1.0`, `16777218.0`): `expected = "BoundOutOfDomain"`.
- **T-K-8a/b** `over_values_refuses_a_param`, `…_a_buffer` (a `Kernel::from_parts` of an arena
  whose root is `push_param(0)`, resp. a declared `push_buffer`): `expected = "BoundNotANumber"`.
- **T-K-9a/b/c** `over_values_refuses_a_value_bound_on_any`, `…_all`, `…_seq` (a uniform `hi`):
  `expected = "BoundSort"`.
- **T-K-9d** `over_values_takes_literal_bounds_on_any`: `over_values(ANY, &constant(0.0),
  &constant(3.0), |i| i.lt(&Kernel::x()))` builds a fold whose range is `0..3` (J18).
- `an_argument_admitted_before_a_fold_is_applied_inside_it` (`:1555-1572`) passes `FoldTerms`.

### pixelflow-search, `src/egraph/node.rs`

- **T-S-1** `a_fold_node_holds_a_value_bound_after_its_body` (replaces
  `a_fold_node_refuses_a_value_bound_by_name`, `:707-721`): `ENode::reduce(sum(0), Lit(0),
  Value(EClassId(2)), EClassId(1))` has `children_slice() == [1, 2]`, `as_reduce()` gives the
  parts back, `known_range() == None`.
- **T-S-2** `a_folds_variance_is_its_bodys_less_its_index_and_its_bounds` (L4): that node's
  `variance(no_unit, child)` with `child(1) = X ∪ {i4}`, `child(2) = {i5}` is `X ∪ {i5}`.
- `a_var_leaf_names_its_own_bit` (`:686-703`): `leaf(Variance::VARIABLES)` is now `CONST` (a
  placeholder), and the test adds `leaf(127) == CONST` and `leaf(128) == ALL`.

### `src/egraph/graph.rs`

- **T-S-3a/b** `add_refuses_a_bound_varying_with_x` and `add_refuses_a_bound_reading_its_own_binder`
  (L3's backstop): `EGraph::add(ENode::reduce(sum(0), Lit(0), Value(x_class), body))`, resp. with
  `Value(class of Var(4))`, `#[should_panic(expected = "BoundVaries")]`.
- **T-S-3c** `add_takes_a_bound_reading_an_enclosing_binder`: `Value(class of Var(5))` on `sum(0)`
  is added, and the class's fact is `body's \ {i4} ∪ {i5}`.

### `src/egraph/insert.rs`

- **T-S-11** `insert_reads_a_literal_bound_into_the_node`: the `ValueBounded` test term
  (`:519-542`) is kept with `lo: Value(2)` over `Const(1.0)` and gains a sibling term whose `lo` is
  a `Shape::Uniform`. Inserting the first gives a fold node with `lo == Lit(1)` and
  `inserted_class_count == 2` (fold, body); the second keeps `lo` a `Value` and counts 3.
  `insert_refuses_a_value_bound_by_name` (`:544-551`) and `a_value_bounds_class_is_counted`
  (`:553-558`) are replaced by it; their docs' "before A3" and "until A3" go (B1).

### `src/egraph/fold_rules.rs` (tests module)

`tri(n)` is the arena `Σ_{i<n} Σ_{j<i} X·j` built with `push_reduce`: inner `sum(0)` over
`Lit(0)..Value(Var(5))`, outer `sum(1)` over `Lit(0)..Lit(n)`.

- **T-S-4** `a_peel_substitutes_into_a_nested_folds_bound` (L7): `PeelFold.apply` on `tri(3)`'s
  outer node plans a `HeadNode::Reduce { fold: sum(0), lo: Lit(0), hi: Value(Plan(k)), body:
  Class(..) }` with `nodes[k] == HeadNode::Const(Real(2.0))`: the bound substituted, the body
  (which never reads `i`) named.
- **T-S-4b** `a_peel_names_a_rebinding_fold_with_a_value_bound` (N7): `Σ_{i<3} (max_{i<u} i·X +
  i)`, the inner `MAX` fold rebinding slot 0 with `hi` a uniform `u`: `PeelFold.apply` on the
  outer plans no `HeadNode::Reduce` whose fold is the inner one, and its head reads the inner
  fold's class (`HeadRef::Class`) — modelled on `a_peel_stops_at_a_fold_that_rebinds_its_slot`
  (`:1257-1295`).
- **T-S-5** `a_peel_makes_an_inner_bound_known` (L8): `tri(3)` inserted into
  `Saturation::over(RuleSet::new(fold_rules()))`, `SaturationConfig::compatibility(2)`: the class
  of the peeled head holds a `Reduce` whose `known_range` is `0..2`; and `KnownBound.apply` on the
  value-bounded inner node over a class `{Const(2.0)}` is `Create` of `ENode::reduce(sum(0),
  Lit(0), Lit(2), body)`.
- **T-S-6** `a_triangular_fold_halves_with_a_rule_made_bound` (the gate's `HalveFold` pin; N6):
  on `tri(4)`, `HalveFold.apply` on the outer node plans an inner `HeadNode::Reduce` whose `hi`
  is `Value(Plan(k))` with `nodes[k] == HeadNode::Op { op: Add, children: [Class(i), Plan(s)] }`
  and `nodes[s] == Const(Real(1.0))`. Then `SaturationConfig::compatibility(50).run(&mut eg)`
  completes (the backstop accepts every rule-made bound), and the test ends by extracting:
  `let (out, out_root, _) = extract(&eg, class, &CostModel::latency_prior());`. It asserts that
  every `Reduce` reachable from `out_root` whose binder is slot 0 has `hi` a `Bound::Lit`, or a
  `Bound::Value(v)` with `compute_arena_variance(&out)[v]` within `{i5}` — the outer binder, or
  nothing for a rule-made `0 + 1` no `ConstantFold` folds here — and that `out.free_index(out_root)
  == None`.
- **T-S-7** `an_unknown_range_is_not_unrolled` (L6): a `sum(0)` with `hi` a uniform:
  `PeelFold`, `HalveFold` and `EmptyFold` `apply` to `None`.
- **T-S-8a** `factor_fold_factors_under_a_value_bound` (decision 5): `Σ_{i<u} (Y·wave(i))` gives
  the plan `factoring` decodes (`:991-1030`) with `factor == Y`'s class and the factored fold's
  `hi == Value(HeadRef::Class(u))`. **T-S-8b** `factor_fold_declines_an_empty_fold` (`:1163-1177`)
  is kept, its doc "A fold known to be empty declines".
- **T-S-9a/b/c/d** `known_bound_declines_a_literal_fold`, `known_bound_declines_a_count_out_of_domain`
  (bound class `{Const(2.5)}`, then `{Const(16777218.0)}`), `known_bound_declines_a_mask`
  (`{Const(Mask(true))}`, added directly: `insert` would refuse it), and
  `known_bound_runs_a_discovered_pair_forwards` (`lo` class `{Const(5.0)}`, `hi` `{Const(3.0)}`
  gives `Create` of the fold over `Lit(5)..Lit(5)`, which `EmptyFold` then closes).
- **T-S-10** `known_bound_is_the_last_fold_rule`: `fold_rules()`' last rule is named
  `"known-bound"`; for every index of `RuleSet::new(all_rules() ++ [HalveFold, PeelFold,
  EmptyFold, FactorFold])`, `RuleSet::runtime()` holds the same `RuleId` at the same index, and
  its one more index names `KnownBound`.
- `reduce_range` (`:669-672`) keeps `expect(KNOWN_RANGE)`: no new test hands it a value-bounded
  node.
- **T-S-11** (round 4, F2) `a_nested_bound_is_rebuilt_through_the_member_that_keeps_l3`,
  **T-S-12** `a_nested_bound_with_no_member_that_keeps_l3_declines`, **T-S-13**
  `the_runtime_rules_saturate_a_triangular_fold_with_a_varying_spelling`, **T-S-14**
  `extracting_a_bound_class_with_no_member_that_keeps_l3_is_refused_by_name` (`#[should_panic(expected
  = "BoundVaries")]`: the hole of J24, loud, which A4 may change). Shared helpers
  `over_a_joined_bound`, `a_varying_spelling_beside_a_lawful_one`,
  `two_spellings_that_each_read_a_coordinate`, `a_triangular_sum_beside` and `nested_bounds`
  (which T-S-6 now uses too).

### pixelflow-search, `src/runtime.rs` tests (round 4, F1)

- **T-R-1** `a_unit_that_reads_an_escaped_fold_index_is_refused`: an index kept from
  `Kernel::sum_over`'s closure, interned and named by `push_ref` beside `X`;
  `compute_arena_variance` of it is `CONST` (asserted: the premise), and
  `optimize_runtime_arena(.., LatticeShape::POINT)` panics with "an index no fold inside it binds"
  (`Program::of`'s `free_index` check).

### `src/egraph/cost.rs`

- **T-S-12** `a_fold_whose_range_is_not_known_is_priced_at_the_estimate`: `node_op_cost` of
  `ENode::reduce(fold, Lit(0), Value(u), body)` is `fold_cost(fold, ESTIMATED_TRIPS, 0)`; and
  `extract_dag` on `Σ_{j<u} |X − j|` totals `fold_cost(fold, 32, cost(Sub) + cost(Abs))` (the
  uniform leaf costs 0).
- **T-S-14** `the_extractor_prices_a_fold_at_fold_cost` (`:733-767`): `fold_cost(fold, 40,
  body_cost)` and `fold_cost(fold, 40, 0)` with a `u64` trip count; its doc gains nothing.

### `src/egraph/extract.rs`

- **T-S-13** `both_twins_of_a_fold_price_at_the_count_their_bound_names` (N3): `forty =
  add(Const(40.0))`; `value = add(ENode::reduce(sum(0), Lit(0), Value(forty), body))`, `literal =
  add(ENode::reduce(sum(0), Lit(0), Lit(40), body))`, `union`, `rebuild`. A tree pass with a
  recording `StageRecorder`, as `the_fixpoint_prices_a_class_the_post_order_dp_left_at_the_sentinel`
  records one (`:2919-2955`), prices both candidates of the class at exactly
  `fold_cost(fold, 40, cost(Sub) + cost(Abs))`; `extract_dag`'s `total_cost` is the same; on a
  fresh graph holding only `value`, `total_cost` is the same too (the pick of `forty` is
  `Const(40)`).

### `pixelflow-search/tests/root_shapes.rs`

- **T-S-15**: `terms()` (`:47-…`) gains `tri(3)`'s shape (inner `sum(0)` over
  `Lit(0)..Value(Var(5))`, outer `sum(1)` over `0..3`), so `every_rule()` (`:28-35`, which
  already includes `fold_rules()`) wakes `KnownBound` and the check covers its root shapes.

### pixelflow-codegen, `src/program/lower.rs` tests

- **T-C-1** `a_value_bound_is_refused_by_name_before_it_is_scheduled`: an arena declaring one
  uniform (slot 0), a `sum(0)` over `Lit(0)..Value(Uniform(0))` whose body is the binder's `Var`
  (no coordinate, which would be refused first): `arena_to_schedule(&a, root, RAW_ORIGIN)`,
  `#[should_panic(expected = "a fold bound that is a value")]`.

### Gate (plan:566-575, and the workflow's)

| Gate row | Pin |
|---|---|
| byte_probe empty, `PIXELFLOW_ISA=avx512` and `=avx2` | §11 gate 1 |
| the 29 font programs' keys and digests; saturation telemetry, extraction fields included | §11 gates 2, 3 |
| instructions within +0.5% | §11 gate 4 |
| L2 | T-IR-4, T-K-2, T-K-2b, T-S-9d |
| L3 (X, Y, own binder, an X-varying `Ref` refused; enclosing binder accepted) | T-IR-15a–d, T-IR-16, T-IR-22, T-K-4, T-K-6, T-S-3a–c |
| `ANY`/`ALL`/`SEQ` under a value bound refused | T-IR-6a–c, T-K-9a–c |
| a non-leaf bound refused at `over_values` | T-K-5, T-K-8a/b |
| L4 | T-IR-10, T-IR-28, T-S-2 |
| L7 | T-S-4, T-S-4b, T-IR-25 (N5's non-claim) |
| L8 | T-S-5 |
| triangular under `HalveFold`, rule-made `0..(i + s)` accepted | T-S-6 |
| N3 pricing | T-S-12, T-S-13 |

---
## 11. Order of work and the grep-gated checklist

**Order** (each step leaves the crates it touches green, `cargo test -p <crate>`; `df -h /`
before every build):

1. `fold.rs`: `BoundEnd`, `BoundRefusal`, `Fold::is_exact` and its callers, `Fold::count`,
   `Bound::discover`, `Fold::admits_bound`, `Fold::ESTIMATED_TRIPS`, `FoldHeader::priced_trips`,
   `Monoid::takes_a_value_bound`, `FoldHeader::new`'s value arm, the `VALUE_BOUND`/`KNOWN_RANGE`
   texts, the docs; T-IR-1–8d.
2. `variance.rs`: `of_var`, `transfer`, the two drivers, L4; T-IR-9–11.
3. `arena.rs`: `free_index`, `push_reduce` (discovery, header, L3) and `literal_at`, `FoldTerms`,
   `close_over`, `close_fold`, `lowest_free_binder(roots)`, the `substitute_vars_with` and
   `known_range` docs; T-IR-12–22 and the mechanical test updates; `passes.rs`'s T-IR-23 and its
   docs; `tests/value_bounds.rs` (T-IR-24–28).
4. `kernel.rs`: `over`'s call and doc, `over_values`, `user_bound`; T-K-*. `pixelflow-ir` is now
   complete: `cargo test -p pixelflow-ir`, `cargo clippy -p pixelflow-ir --all-targets
   --all-features -- -D warnings`, and the `no_std` build CI runs.
5. `pixelflow-search`: `node.rs` (L4, `of_var`, `var_variance` gone, docs), `graph.rs` (`add`'s
   backstop), `insert.rs`, `fold_rules.rs` (`KnownBound`, `FactorFold`, docs, `rebuild`'s
   comment), `cost.rs`, `extract.rs`; their tests; `tests/root_shapes.rs`. `cargo test -p
   pixelflow-search` default, `--all-features`, and `--features saturation-telemetry`.
6. `pixelflow-codegen`: the `program/lower.rs` comment and T-C-1.
7. **Mechanical tail**: `pixelflow-compiler` `lower.rs` and `emit.rs` (§9), and
   `pixelflow-pipeline/src/training/corpus.rs:408`'s message. Its own gate: §9's.
8. Docs (§12).
9. The gates below, then the greps.

**Gates** (the workflow's, each against its recorded base): (1) byte_probe under
`PIXELFLOW_ISA=avx512` and `=avx2`, `cmp`-equal; (2) the 29 programs' node counts, keys and
digests (`gate1b`); (3) their saturation telemetry (`gate2`) and its extraction fields
(`gate2x`); after (2) and (3) `git status` shows the probe file restored; (4) total callgrind
instructions on the cold compile of the 29 programs within +0.5% of base (a larger rise on
all-literal kernels is literal-path work A3 must not do: name the function from the
`callgrind_annotate --inclusive=yes` diff); (5) `cargo test -p` for pixelflow-ir,
pixelflow-search (three feature sets), pixelflow-codegen, pixelflow-compiler,
pixelflow-graphics, pixelflow-pipeline, pixelflow-core; (6) `cargo clippy --workspace
--all-targets --all-features -- -D warnings`, `cargo fmt --all -- --check`,
`RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-ir -p pixelflow-search -p pixelflow-codegen
--no-deps --document-private-items`; (7) `scripts/check-intent.sh`, `scripts/check-doc-paths.sh`,
`scripts/check-cfg-encapsulation.sh`, and the `Reduce {` generator: 0 unclassified, 0 stale.

**Grep-gated checklist** (run at head; each must come out as stated):

- **B2.** `grep -rn 'Reduce {' --include=*.rs pixelflow-*/src pixelflow-*/examples
  pixelflow-*/tests pixelflow-pipeline/src/bin`, deduplicated by `(file, line)`: every site is a
  row of §5's table, keyed by `(file, enclosing item, trimmed text)`; the generator reports 0
  unclassified and 0 stale. A site A3 adds (T-* tests, `KnownBound`, `literal_twin`,
  `refuse_a_varying_bound`, `transfer`) is classified in the same CL.
- **B1.** `grep -rnwE 'A3' --include=*.rs pixelflow-*` and `grep -nw 'A3' pixelflow-ir/intent.md`:
  **empty**. Today's hits, each reworded as §2.10–§2.12, §3, §6, §8 and §12 give: `fold.rs:579`,
  `:580`, `:597`, `:710`, `:811`, `:813`, `:819`, `:1541`, `:1550`; `arena.rs:942`, `:1654`,
  `:2587`; `passes.rs:1582`, `:1600`; `node.rs:394`, `:551`, `:711`; `insert.rs:521`, `:547`;
  `program/lower.rs:344`; `training/corpus.rs:408`; `pixelflow-ir/intent.md:73`. New text names
  what is still to come (A4, A6, B2) or nothing. The phrase check
  `grep -rnE 'until A3|before A3|A3 lifts|A3 gives|A3 carries|A3 of|A3.s|can be built until A3'
  --include=*.rs pixelflow-*` is empty too.
- `grep -rn 'VALUE_BOUND' --include=*.rs`: only `fold.rs` (the definition, `to_bits`'s panic and
  its doc link, `KNOWN_RANGE`'s doc link) and `program/lower.rs` (its import and `known`). The
  tests that pin the refusal (`a_value_bound_has_no_bits_until_a4`, T-C-1, T-IR-26) match the
  message's prefix, not the name.
- `grep -rn 'KNOWN_RANGE' pixelflow-search/src/egraph/cost.rs pixelflow-search/src/egraph/extract.rs`:
  at most `cost.rs`'s test-local import.
- `grep -rn 'fn close_over\|\.close_over(\|\.close_fold(' --include=*.rs`: every call takes a
  `FoldTerms`. The call sites at `3bfdaded`: `kernel.rs:758` (`over`) and `:1565` (a test);
  `arena.rs:1723` (`close_fold`'s own) and the tests `:2317`, `:2344`, `:2364`, `:2808`; the
  compiler's `lower.rs:515` and `emit.rs:838`; plus `over_values`' new one.
- `grep -rn 'fold_cost(' --include=*.rs`: `cost.rs` only — `node_op_cost` and the tests — every
  call with a `u64` trip count. The extractor reaches it through `node_cost` of a fold's priced
  form (§7.1, N3, J7).
- `grep -rn 'var_variance' pixelflow-search`: empty. `grep -n 'Variance::of_var\|Self::ALL\|Variance::ALL'
  pixelflow-ir/src/variance.rs`: `transfer`'s `Var` arm is `Variance::of_var(idx)`; `ALL` is spelled
  only in the `Param` arm, `referent_variance`, `of_var`'s last line, and tests.
- `grep -n 'fn transfer\|fn variance_from\|fn compute_arena_variance' pixelflow-ir/src/variance.rs`:
  one each (N1). `grep -n 'compute_arena_variance' pixelflow-ir/src/arena.rs`: empty (L3 walks the
  bound, never the arena).
- `grep -n 'KnownBound' pixelflow-search/src/egraph/fold_rules.rs`: the definition and its impl,
  the last entry of `fold_rules()` and its doc, and tests.
- **N10.** `grep -rn 'real_sorted' --include=*.rs`: empty (the predicate is
  `takes_a_value_bound`: `grep -rn 'takes_a_value_bound' --include=*.rs` shows its definition,
  `FoldHeader::new`'s value arm, and T-IR-5). `grep -n 'EXACT_BOUND' pixelflow-ir/src/fold.rs`: a
  comparison only inside `Fold::is_exact`. `grep -rn 'Fold::count\|fn count(' pixelflow-ir/src/fold.rs`:
  `pub(crate) fn count`.
- `grep -rn 'fn over_values' pixelflow-ir/src/kernel.rs`: `pub(crate) fn over_values`.

---
## 12. Docs A3 edits (plan §5 rows marked A3, the rulings' plan amendments, and the code docs under changed code)

### 12.1 The plan (`docs/language/2026-10-20-a-bound-is-a-value.md`)

- **L3** (plan:180-188, N1): "computed by `compute_arena_variance` over the bound roots, `Ref`
  resolved through the store (…), the same function L4 extends" becomes "computed by the arena's
  per-node variance transfer function run from each bound and memoized (`variance_from`,
  `pixelflow-ir/src/variance.rs`) — the one `compute_arena_variance` runs forward, so a `Ref` the
  bound reaches is resolved through the store and one it does not reach is not — the same
  function L4 extends". The rest of the bullet stands.
- **§3.1's `push_reduce` bullet** (plan:252-260, N4): "It refuses, by name: a non-integer or
  out-of-domain `Const` (`BoundOutOfDomain`, today's `admits` as a predicate on one count); a
  `Lit::Mask` bound (a mask is not a count); L3 (`BoundVaries`); a value bound on `ANY`, `ALL` or
  `SEQ` (`BoundSort`, §1.4)." becomes "It keeps a non-integer or out-of-domain `Const` as a
  `Value` bound, whose guard makes the fold NaN (L5), so a literal a rule or the folder made never
  panics at extraction; `BoundOutOfDomain` (today's `admits` as a predicate on one count) is
  refused at the user surfaces only: `over_values`, `Kernel::over`'s literal range and `kernel!`
  (A6). It refuses, by name: a `Lit::Mask` bound (`BoundSort`: a mask is not a count); L3
  (`BoundVaries`); a value bound on `ANY`, `ALL` or `SEQ` (`BoundSort`, §1.4)." The sentence after
  it (`BoundArithmetic` at the user surfaces) stands.
- **§3.1's walkers bullet** (plan:275-280, N5) gains at its end: "`ExprArena::substitute_vars_with`
  is not an L7 substitution: it enters every fold's body and bounds, so its callers substitute
  only slots no nested fold rebinds (coordinates, retired axes, a template's metavariables), a
  precondition its doc states."
- **§3.1's `close_over` bullet** (plan:269-270, N2): "`close_over` takes the `Value` bounds as
  extra roots (`from_arena_roots`). A known bound pushes nothing." becomes "`close_over` discovers
  constness before it copies (`Bound::discover`): a bound whose term is a `Const` naming a count
  is the fold's data and not a root, and the remaining `Value` bounds are extra roots
  (`from_arena_roots`). A known bound pushes nothing, so `over_values` over two constants is
  `over` over the literals, node for node."
- **§3.1's `canonical` bullet** (plan:281-283) gains "(A4)" after "`canonical`:".
- **§3.2's pricing bullet** (plan:308-312, N3) becomes: "**Pricing.** A known fold's `len` is read
  off the node. A value-bounded fold whose bound's chosen form is an in-domain integer literal is
  priced at that count — the price of the literal twin `KnownBound` puts in its class — and any
  other at `ESTIMATED_TRIPS` (§4). So pricing reads the bound children's extracted picks: the DP
  prices a fold only after every child class has settled (`extract.rs`'s `Settlement`), and hands
  the count to `fold_cost` and `below` through the fold's literal twin. `below` adds its `Value`
  bound operands once: it read only the body operand before A3, so bound arithmetic would have
  gone unpriced with no audit failure."
- **§4's sentence** (plan:409-410, N3): "A fold's own node says whether its range is known (§3.2),
  so no price site needs a graph." becomes "A fold's own node says whether its range is known; a
  value bound is priced by its class's chosen form (§3.2), which the extractor holds before it
  prices the fold, so no price site needs more of the graph than the picks."
- **The A3 bullet** (plan:566-575) gains, when it lands: "**Done:** byte_probe empty in both
  columns under `PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font programs' keys, digests and
  saturation telemetry, the extraction fields included, unchanged. A placeholder's variance is
  `CONST`, so a triangular fold built through the builder passes L3 (J5); a value bound is priced
  by its class's chosen literal through the fold's literal twin (J7); the key refuses a value
  bound until A4 (J10)." (Its figures are filled in only from gates that passed.)
- **The A4 bullet** (plan:576-586, N4, N8) gains, after "`over_values` public.": "The key: a
  `Value` marker in `FoldHeader::to_bits`'s count field and its `from_bits` inverse. The
  serializers that write a header without its children refuse a value bound by name today and
  learn the marker: `pixelflow-graphics/src/render/cell_grid.rs`'s and
  `pixelflow-graphics/tests/production_glyph_arena_dump.rs:171`'s `dump_arena`,
  `pixelflow-graphics/src/test_support.rs`'s `program_digest`,
  `pixelflow-pipeline/src/collapse_bench/corpus.rs`, `pixelflow-pipeline/src/bin/corpus_gaps.rs`,
  and the compiler's `arena_to_tokens`; the training corpus bumps its format version. The guard
  handles a rule-made out-of-domain literal: `push_reduce` keeps one as a value bound (A3), so the
  guard, not a panic, decides it."

### 12.2 `docs/language/2026-10-17-the-arena-is-typed.md` (plan §5 row, plan:449)

- The §1.1 `Reduce` row (`:72`) becomes: "| `Reduce` with `⊕` | `⊕ : S × S → S`, body `: S`, a
  value bound `: Real` (`Int` at T6), result `: S`; a known bound is node data |".
- T6 (`:216-217`) gains: "T6 also retypes a fold's value bounds with the binder
  (docs/language/2026-10-20-a-bound-is-a-value.md §5): it lifts `BoundArithmetic` and deletes the
  guard's integrality conjunct, measured. A known bound is node data, so T6 re-keys no bound
  literal."

### 12.3 `docs/BACKLOG.md` (plan:451-452)

- The plan's row (`:32`): "landed" gains "A3 (value bounds in the IR and the e-graph)"; "next"
  becomes "**A4** (value bounds in codegen, and the key); A1's the-language-is-kernel rows after
  #1323; A5–A7; B1–B5 after typed-arena T3 and item 2".
- The typed-arena row (`:33`) gains, at the end of its notes: "T6 also retypes value bounds
  (a-bound-is-a-value §5)."
- E6 (`:150`) gains, at the end: "Since a-bound-is-a-value A3 the tie narrows to known ranges: a
  fold whose range is not known has no unrolling to tie with (L6)."

### 12.4 `.claude/skills/ontology/language.md`, "Unrolling" (`:529-552`; plan:457)

After the **Is:** bullet, a new bullet:

> - **Only over a known range** (L6 of a-bound-is-a-value): a fold whose bound is a value is not
>   unrolled, as LLVM does not fully unroll an unknown trip count; `KnownBound` makes a range known
>   when the e-graph discovers a bound's constant — a peel substituting `i := k` into an inner
>   `0..i` (L8).

The **Lives:** line's list `(`HalveFold`, `PeelFold`, `EmptyFold`, `FactorFold`)` gains
`KnownBound`.

### 12.5 `intent.md` files (each within 80 lines, `scripts/check_intent.py`'s `MAX_LINES`)

| File | Lines at `3bfdaded` | Edit | Lines after |
|---|---|---|---|
| `pixelflow-ir/intent.md` | **80 (at budget)** | `:72-73` in place: "- `Var(u8)` means an axis or a binder by range; `Binder` a `u8`, against / the 64-bit control-plane rule. Codegen and the key take literal `Bound`s until A4." | 80 |
| `pixelflow-search/src/egraph/intent.md` | 74 | `:18-21`: "… a fold costs `n·body + (n−1)·combine`, plus its value bounds once; `n` is read off its node when its range is known (`ENode::known_range`, per node: after `HalveFold` a class holds two ranges), else off its bounds' chosen literals, else `Fold::ESTIMATED_TRIPS`; a surviving `Dwrt` costs `LEGALIZATION_PRICE`. …" (one line longer). `:58-59` in place (N9): "a fold ties its unrolling" → "a fold over a known range ties its unrolling". | 75 |

No other `intent.md` changes: `pixelflow-codegen/src/program/intent.md:20-21` (64 lines) and
`pixelflow-ir/intent.md`'s Contract (`:41-47`) are already true (§0, §8).

### 12.6 Code docs under changed code (plan §5's last row, plan:460, A3's share)

Everything §2–§9 names, and in particular: `Kernel::over`'s doc (`kernel.rs:711-717`, N9, §4.1);
`ENode::Reduce`'s doc (`node.rs:286-310`, the plan's `egraph/node.rs:226-239`, §6.1);
`fold_rules.rs:620-626` (§6.8); `FactorFold`'s doc (§6.6); `KNOWN_RANGE`'s true reason at
`glyph_phase_split.rs:79` (§2.11, N9); `pixelflow-ir/intent.md:73` (§12.5);
`program/lower.rs:341-344` (§8); `training/corpus.rs:408`'s message: "a value bound has no corpus
encoding before A4's format version".

**Not A3:** the ontology rows "Fold (Reduce)", "Range (trip count)" and "Totality", the
`DENOTATION.md` rows, `CLAUDE.md:175` and `pixelflow-core/README.md:63-65`. These are A4's, as
built (plan:437-458).

---
## 13. Judgement calls, each with the alternative rejected

- **J1. `Bound::discover` reads the pair, not one bound.** L2's normalization needs both ends
  and needs to know whether this read made the pair known. Rejected: a per-bound `discover` plus a
  separate normalizing step every caller must remember, which is a second place to forget it.
- **J2. A mask literal bound is `BoundSort` everywhere (N4).** A mask is a sort error, not a value
  outside a domain, and L5's NaN cannot answer for it (decision 8 is open). Rejected: the previous
  spec's `BoundOutOfDomain` for a mask, which would also have let `push_reduce` keep it as a value
  bound under N4.
- **J3. `BoundSort` carries only its `end`.** The mask case has no algebra to blame, and the
  monoid case names its fold in the panic's context (`"{refusal}: {fold:?}"`). Rejected: a
  `monoid` field the mask case would have to fill with the fold's own and so mislead.
- **J4. L3 by one per-node transfer function with two drivers (N1).** `transfer` is
  `compute_arena_variance`'s match, run forward over the arena and from one root with a keyed memo
  (`variance_from`); `push_reduce` runs the rooted one per value bound, the bound's term and not
  the arena. Rejected: a whole-arena `compute_arena_variance` per value-bounded push, which every
  rebuild repeats for every such fold and which resolves every `Ref` in the arena each time — the
  arena-sized-table-per-site shape `passes.rs:571-574` measured at 86 s on a 1.5 M-node arena. The
  previous spec's objection, that a second definition of the rule would be needed, does not hold:
  there is one definition and two traversal orders, as `ENode::variance` already is.
- **J5. A placeholder's variance is `CONST`.** When an inner fold closes, an enclosing fold's
  index is still a placeholder; L3 asks whether the inner bound reads a coordinate or the inner
  fold's own binder, and for a placeholder the exact answer is no (its slot, chosen later, is none
  the inner term binds or reads). Rejected: today's `ALL`, which intersects `COORDS`, and the
  previous spec's `BINDERS`, which contains the inner binder's bit — either refuses every
  triangular fold built through `Kernel::over` and `over_values`, the plan's "enclosing binder
  accepted" pin. Index 128
  and up stays `ALL`. **Round 4 (F1) corrects this entry's claim that no arena a placer reads holds a
  placeholder:** a placeholder can escape its builder (a closure that keeps a clone of the index),
  and then the term that holds it is open with an empty variance. `CONST` is exact for the one
  question it is asked (L3's, inside `close_over`), and it is not "closed", so no check of
  closedness or placement may read it. The audit of every reader of a variance (`grep -rn
  'compute_arena_variance\|of_var\|variance_from\|admits_bound'`): `Program::of` (`runtime.rs`) judged
  a unit closed by `variance.without(COORDS).is_const()`, which at base refused an escaped index
  (`ALL`) and at head admitted it, so it now also refuses `free_index(body_root).is_some()`, naming
  the index (T-R-1). `EGraph::admit_unit` sees a variance and nothing else; its only caller is
  `Program::of`, and its doc says so. `arena_to_schedule` (`program/lower.rs`) refuses any `Var` that
  is no fold's binder before the variance reaches a placement, by name (T-C-2; the message now says a
  placeholder is one of the two things it refuses). `lower_dwrt`'s tabulation (`passes.rs`) asks
  whether an index moves with the variable being differentiated, which a placeholder answers
  exactly (it is none of them; `diff_node`'s `Var` arm differentiates every other `Var` to 0, a
  binder's included): not a closedness or placement check, unchanged. `nnue::factored`'s histogram and
  the extractor's `best_var` price and classify; they judge nothing closed. `EGraph::add`'s backstop
  and `push_reduce` are L3 itself. Rejected: restoring `ALL` for a placeholder, which refuses every
  triangular fold built through the builder (this entry's first reason), and a `free_index` in each
  reader, where one check in the unit walk and the lowering's own refusal cover the two places a
  closed term is assumed.
- **J6. `Fold::count` is `pub(crate)` (N10).** Another crate reads a count off a bound through
  `Bound::discover`, which is where a count is a bound. Rejected: a public `count` that invites a
  second reading of constness.
- **J7. The extractor prices a value-bounded fold through its literal twin (N3).** The pick
  becomes a count by discovery, and the fold is priced as the fold over that count: `node_cost` of
  the twin (for `CostModel`, `fold_cost(fold, n, 0)`) and `below` multiplying by the twin's
  `priced_trips`. The extractor is generic over `CostFunction` and `fold_cost` is `CostModel`'s, so
  this is how the effective trips reach `fold_cost`, and a wrapping cost function prices the twin
  as it prices any fold. Rejected: adding a `fold_cost` (or a `trips` argument) to the
  `CostFunction` trait, which every wrapper (`LoopsCostMore` twice) would have to forward, and a
  direct `CostModel::fold_cost` call, which the generic extractor cannot make. The twins pin holds
  by construction.
- **J8. `lowest_free_binder(roots)` marks binder `Var` reads, not only `Reduce` binders.** A fold
  may not take a slot its bound reads (L3), nor capture one an open body reads. Neutral for every
  builder (§1.3). Rejected: scanning only `Reduce`s, under which a bound reading a free slot `0`
  makes `close_over` choose slot 0 and `push_reduce` then refuse its own fold.
- **J9. Discovery in `close_over`, before the copy (N2).** A bound whose term is a literal naming
  a count is never a root, so nothing is copied for it, and `over_values` needs no discovery of its
  own. Rejected: `over_values` discovering for itself and `close_over` copying whatever it was
  given, which pushes a `Const` into the closed arena that `push_reduce` then reads into the
  header and leaves behind — an arena that is not `Kernel::over`'s node for node (L1).
- **J10. The key refuses a value bound until A4.** `to_bits` has no field for a child; A4 writes a
  marker there, with `from_bits`' inverse, the serializers that write a header without its
  children, and the corpus's format version, in one CL. This deviates from A2b's plan text, which
  had `canonical` writing a marker. Rejected: a marker in A3 with no reader that can compile the
  kernel it names.
- **J11. `insert` discovers, and `inserted_class_count` mirrors it.** The e-graph's node for a
  term is then the arena's, and the measure counts exactly the classes `insert` makes. Rejected:
  leaving discovery to `KnownBound`, which costs an application and a minted id per such fold and
  makes the node differ from the arena's for the same term.
- **J12. `KnownBound` fires only when the read makes the whole range known.** L6 and the price
  need both ends. Rejected: firing per bound (the plan's per-bound sentence read literally), which
  adds a half-known node no rule reads and spends a minted id and an application on it.
- **J13. `Kernel::over`'s out-of-domain literal keeps `KnownRange::new`'s message.** It is a
  declared `Range<u32>`, refused where every declared literal pair is, with the message
  `a_fold_past_the_exact_bound_is_refused_by_the_builder_too` (`kernel.rs:1265-1270`) and
  `kernel!`'s expansion-time check already pin; N4's "refused at the user surfaces" holds.
  Rejected: re-spelling it as `BoundOutOfDomain`, a second message for one refusal.
- **J14. `FoldTerms` is data** (`monoid`, `stride`, `lo`, `hi`), replacing the `fold_at` closure.
  The binder is the only thing `close_over` computes. Rejected: keeping the closure and passing
  the bounds beside it, five arguments, or a closure that builds a `Fold` and must agree with
  bounds passed elsewhere.
- **J15. Refusal order is cheapest first** (§1.2): discovery, the header, then L3's walk.
  Rejected: L3 first, which walks a bound only to then refuse its fold's monoid.
- **J16. `over_values` keeps four arguments.** They are the fold's own (monoid, two bounds, body),
  `push_reduce`'s precedent, and decision 7 approved the plan's spelling (plan:271-274). Rejected:
  `Range<&Kernel>`, three arguments for a type that reads as an iterator and is not one.
- **J17. No `free_indices`.** `free_index` is the set's one reader; `lowest_free_binder` asks
  which slots are taken, a different question. Rejected: extracting a set-valued `free_indices`
  with no second caller.
- **J18. `over_values` checks its bounds before the body runs and leaves the monoid to
  `FoldHeader::new`.** A user error is reported before the closure does work; a pair of constants
  is a literal fold, legal under `ANY` (T-K-9d). Rejected: refusing `ANY`, `ALL`, `SEQ` up front,
  which would refuse `over_values(ANY, 0, 3, f)`.
- **J19. `lowest_free_binder` is not merged with `passes::lattice`'s `reachable_taken_binders`**
  (`passes/lattice.rs:157-190`), which computes the same set. The lattice walk also asserts
  `collapse`'s preconditions (a `Ref`, a `Dwrt`) in the same pass and moves out of `pixelflow-ir`
  with the CPU schedule (DENOTATION §7). Rejected: one shared function, which would carry
  `collapse`'s panics into `close_over` or split the lattice walk in two.
- **J20. `KNOWN_RANGE` is one text listing its true reasons** (§2.11). Rejected: one message per
  site, a dozen strings for one fact ("every fold this site reads is literal, until a named step").
- **J21. `insert` panics on a mask literal bound** rather than declining. `BoundSort` everywhere
  (N4), and no arena can hand one over (`push_reduce` refused it). Rejected: a new public
  `Declined` variant for a term only a hand-written `Ir` can present.
- **J22. `EGraph::add`'s backstop panics**, as `add` already does on a `Ref` no walk admitted. A
  refused node is a constructor's or a rule's bug. Rejected: declining inside `add`, which is
  total by law (L1) and has no decline to return.
- **J23. The literal twin is built to be priced and never added.** Rejected: pricing a
  value-bounded fold only when `KnownBound` has already put the twin in its class, which makes
  the price depend on whether a rule fired in the budget, and leaves a lone value twin over
  `Const(40)` at 32.
- **J24. The extraction-side hole in L7 is documented, not closed (round 4, F2).** Extraction picks
  one member per class, by cost, and a bound's pick is not constrained to keep L3. A bound class
  whose chosen member reads a coordinate the class does not vary with is materialized through it,
  and `ExprArena::push_reduce` (`arena.rs`, L3 on the term) refuses the fold with `BoundVaries`:
  loud, never silent. The hole is reached whenever the cheapest member of a bound class reads a
  coordinate, which includes a class whose lawful member costs more as well as one with no lawful
  member at all. With no lawful member: a fact that is the meet of members that each read a
  different coordinate (`i + X` beside `i + Y`, T-S-14). With a dearer lawful one: `Pow(i, 2)` beside
  `((i·i) + X) − X`, unioned as the true equation `i² = (i·i + X) − X`, has the fact `{i}`, so its
  fold is admitted, and extraction (no rules) picks the spelling that reads `X`, the cheaper, and is
  refused (pinned by `extracting_a_bound_class_whose_cheapest_member_reads_a_coordinate_is_refused_by_name`).
  The earlier wording, that the hole is reached only where a class holds no lawful member, held for
  the cases tried first (the repro extracts `i + 1`) and set the boundary in the wrong place; whether
  a saturated graph still holds the cheap spelling is the rules' business (`RuleSet::runtime()` at
  `compatibility(6)` rewrites that graph's bound away and extracts `6X`, correct). A4 decides whether a bound's pick is
  constrained (the settlement, the `Reranker` seam, or a refusal at the settle). The F2 pin does
  **not** show that extraction never picks such a member, so the plan's A4 bullet carries one
  sentence saying so. Rejected: constraining the pick in A3, which is the extractor's DP (a
  per-site member filter inside `Settling`), with no value-bounded fold in any production pick and
  a price site per extractor to change.
- **J25. `PlanBuilder::substitute` keeps a site, not a mode flag (round 4, F2).** `Site::Body` and
  `Site::Bound { fold }` are one enum, the memo is two maps so the literal path's key stays a class,
  and `Site::of_child` is the one place a child's site is decided. A fold nested inside a value
  bound is not entered: its body would need the refused set lifted by its own binder. Round 4 had
  the walk decline on one (`None` from `of_child`), justified by "no builder makes one", which was
  wrong: saturation can put a fold into a bound's class. Polish round 1: `Site::member` skips a
  `Reduce` member at a bound and reads the next lawful one, so a fold under a bound is never chosen,
  the justification is gone, and `of_child` is total. Declining on the fold had lost a rewrite
  whenever a later member of the class kept L3. Rejected: a `refused: Variance` set carried down and
  lifted at a nested fold, which is the same decision with a bitset and a case no builder reaches.
- **J27. The graph declines a plan whose bound lands in a class that fails L3 (closing round 2).**
  A rebuilt bound is a term, `plan_shape` adds it, and the e-graph is hash-consed: the node can
  already be in the memo, in a class whose fact was seeded wider than the node and is never repaired
  upward (`var_fact`'s law; `rebuild` re-keys a node it processes and leaves its parents' facts
  alone). `EGraph::add`'s backstop reads exactly that fact, so a lawful term panicked with
  `BoundVaries`. Three hand-made-union cases, each an equation the algebra derives
  (`a_rebuilt_bound_that_lands_in_a_stale_class_is_declined`, the runtime-set pin): `P = ((i + X) − X)
  + 1` with `(i + X) − X = i` and `P = ((i + X) + 1) − X` puts the halved copy's `i + 1` into `P`
  (`HalveFold` alone at n=4); `P = (X − X) + 1` with `X − X = 0` puts the peeled copy's `0 + 1` into
  `P` (`PeelFold` alone at n=1); and `P` holding both spellings of its sum first leaves no fresh `i +
  1` or `1 + i` for the full runtime set (`commutativity`, `ConstantFold`) to mint as a lawful twin,
  which panicked at n = 2, 4 and 8. No arena built a trigger (616 kernels under `RuleSet::runtime()`,
  `compatibility(12)`, spellings of the index through X, Y and j, constants 1, 2, 4, junk in the outer
  or nested body, bound `i` or `i + 1`, n from 1 to 16): the hand-made union is the same standard as
  the round-4 pins.
  The decision: `NodeSink::refuses(&node)` asks, before `plan_shape` makes a `Reduce`, whether `make`
  would panic on it (`EGraph::add`'s backstop: the same `varying_bound`, asked after the same memo
  lookup); `plan_shape` returns `None` on a refusal, whence `apply_action` makes no union (0). The
  growth predictor declines at the same node, reading the fact of a class it minted off the node it
  minted it for (`ENode::variance` over its children's facts, exactly what `add` seeds), so a declined
  plan's prediction is the classes built before the refusal, as the applier builds them. `add` and
  `make` keep their shape and their return type. The first draft returned a `Result` from `add`
  (`try_add`, `NodeSink::try_make`): one definition of `add`, but the measured cost was +0.26 percent
  of total Ir on the 29 programs (`add` is the hot call), over the +0.5 limit with the round-1 +0.25,
  and the hot path is not where a refusal is wanted. The check is a function of the node and the facts
  alone, asked after the memo lookup as `add` asks it, so a run that did not panic does no different
  thing: byte-identical on every pinned kernel, and a fold over literals returns on its arity before
  any class is read (`has_a_value_bound`).
  Rejected: (a) a planner-side memo replay (`substitute` resolving the bound's nodes, declining at
  scan time) which mints nothing on a decline but is a prediction of where `add` will land, and a
  commit earlier in the same batch can change the answer (a union that merges a child changes what
  the node canonicalizes to), so `add` could still panic; (b) refreshing a class's fact on a memo hit
  (sound by the law, but it touches every `add` of a literal kernel); (c) loosening the backstop to
  "some member keeps L3", which is wrong one level down (a bound `W + 1` whose child `W` is stale has
  one member and no lawful one). A declined plan makes no union, but it is not a plan that built
  nothing: `plan_shape` refuses a fold only when it reaches it, so every node made before that point
  stays in the graph, a lawful nested fold earlier in the same plan among them (reviewed with
  `PeelFold` alone over `Σ_{i<1}(Σ_{j<i} X·j + Σ_{j<i+1} X·j)` after `P = (X − X) + 1`, `X − X = 0`:
  value-bounded inner folds 2 → 3, one class more, the matched class gains no member, and the
  prediction counts the orphan exactly). Those orphans cost classes, never soundness, and only on a
  plan that panicked before. The refusal reads a bound's class *fact*, so it covers both ways a
  bound's class is seeded wider than its term: the bound landed in a stale class, or the plan minted
  a class for it over a child that did (`(i + 1)·2` with `i + 1` in `P`; pinned by the third case of
  `a_rebuilt_bound_that_lands_in_a_stale_class_is_declined`).
- **J26. One `ENode::literal_twin` (round 4, F3).** `KnownBound::apply` and the three price sites
  call the same `pub(crate)` method on `ENode`, so a fold's literal form is one definition and
  neither the rule's doc nor the extractor's asserts that two agree. Rejected: keeping the two
  copies and a test that compares them, which is a check that the duplication has not drifted.
- **J28. Polish round 1: the closing pass's non-blocking findings (TAG=a3-polish1).** Twenty findings,
  two not applied (the blocking one, fixed in d4bcd11, and the one that says "no defect"). Decisions:
  (a) L3 has an end-free predicate, `Fold::bound_may_read(self, reads) -> bool` (`pub`, because the
  e-graph is another crate); `admits_bound` calls it and `Site::member` calls it, so no `BoundEnd` is
  fabricated for a question that has none. Rejected: leaving `member` on `admits_bound(BoundEnd::Hi,
  ..)` with a comment saying the end is ignored, which is a convention held by prose.
  (b) `Site::member` skips an `ENode::Reduce` at a bound, so `of_child` is total and the round-4
  justification "no builder makes one" is gone with the case (J25). The skip is more complete than the
  decline it replaces (a class whose first lawful member is a fold reads the next one) and a literal
  fold has no site that is a bound.
  (c) `representative` is inlined (`egraph.nodes(class).next()`); `Site` derives `Clone, Copy` only (the
  memo keys on `(EClassId, Fold)`).
  (d) The literal twin's fast path. The finding measured the closing pass at +5.6 M Ir over r3 (close1)
  and asked for an inlined guard over a non-inlined body. Measured on the same workload (total Ir,
  extraction inclusive Ir): r3 6,305,898,607 / 1,136,559,328; close2 6,308,131,681 / 1,137,870,806;
  guard + `#[inline(never)]` body 6,310,530,002 / 1,140,855,973 (worse than close2 by 2.4 M);
  guard + `#[inline]` body 6,307,997,174 / 1,138,113,466 (equal to close2 within 0.14 M, kept);
  guard + one `&dyn Fn` body 6,350,414,320 / 1,143,588,336 (an inlining cliff in saturation, +0.865
  percent, rejected). The guard is inlined in every form, so what moves is placement in the pricing
  loops (`extract_dag_over` self +1.36 M, `BinaryHeap::pop` +0.88 M, `memset` -0.65 M against r3), and
  the guard split alone is worth nothing measurable. Recorded, not removed: +2.10 M Ir (+0.033
  percent) over r3.
  (e) The test assertions that could not fail are replaced by ones that can: the plan was declined
  (`total_unions == 0` for a rule that is alone in its set), `P`'s fact still reads `X`, the stale
  graph's value-bounded inner fold count is below its control's, and the rebuilt bound holds no `Sub`
  member. Each was checked by mutation: removing the decline makes the stale tests panic with
  `BoundVaries`; removing the decline and the backstop makes the new assertions themselves fail.
  (f) A second extraction pin (J24) and the `(i + 1)·2` case (J27) are tests, because a gap that is
  only written down is a gap nothing will tell A4 about.
  (g) `Kernel::over`'s doc said an index cannot escape its fold. It can (a clone), and the next fold
  that reclaims the placeholder captures it with no refusal anywhere: `Kernel::sum_over(3, |j|
  j.add(&escaped))` is `Σ_j (j + j)`, closed. Pre-existing at `3bfdaded` and not A3's, so the doc is
  corrected and `docs/BACKLOG.md` C10 tracks the hole.

