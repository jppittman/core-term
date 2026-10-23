# A3 spec critique: soundness and identity

Reviewed: `.wip/a3/spec.md` against the live code (HEAD's code is `3bfdaded`; HEAD itself
only adds `.wip` files), the plan `docs/language/2026-10-20-a-bound-is-a-value.md` (§0-§4,
§6 A3, §8), `.wip/a3/rulings.md` and CLAUDE.md. Read-only, nothing built.

## Verdict

No blocking defect. Every value-bound path the spec defines is sound against L2-L8 and
§1.4. Implementing it as written moves no byte, key, node id, rule-application count, price
or extraction choice on today's all-literal kernels. Six non-blocking findings follow,
ranked. The first two show that two of the spec's own claims are false at edges the plan
leaves open. Both are reachable today only through the public `push_reduce` with a
hand-built bound. Neither is reachable through `over_values`, `Kernel::over`, `kernel!` or
any rule.

## What I checked and cleared

- **Identity on all-literal kernels.**
  - `close_over`: with `roots == [body]`, `from_arena_roots(self, &[body])` is
    `from_arena` (`expr.rs:274-279`). `substitute_vars` runs once on the one entry.
    `to_arena_roots` walks every DAG node in push order (`expr.rs:381`). Then
    `push_reduce` interns the fold last, where `to_arena`'s loop interned the DAG's
    `Reduce` today (`expr.rs:393-396`). The arena pushes come in the same order with the
    same tables.
  - `Fold::strided(monoid, b, 1)` equals `Fold::new`. The hole stride is carried as
    `FoldTerms::stride`.
  - `push_reduce`: `Bound::discover` returns a literal pair on its first match.
    `FoldHeader::new` takes today's literal path. The two `let Bound::Value(..) else
    { continue }` make no walk and no allocation.
  - `EGraph::add`: one discriminant test plus `header.arity()`.
  - `lowest_free_binder(roots)`, which now also marks `Var` reads (J8), is neutral. Every
    production body is closed: `Kernel::over`'s, and `kernel!`'s lowering through
    `open_fold`/`close_fold` (the only callers, `lower.rs:490/515`, `emit.rs:825/838`,
    `kernel.rs:758`).
  - `of_var` changes only indices 64..128. No arena that reaches codegen, the unit
    check, `lower_dwrt`, `nnue::factored` or the e-graph holds a placeholder.
  - `free_index`, both variance transfers, and the `FactorFold` guard
    (`is_some_and(is_empty)` against `is_none_or(is_empty)` on a known range) are
    unchanged on literals.
- **Extractor DP order.** `Settlement::relax` prices a candidate only once `waiting`
  reaches 0 over its distinct child classes (`extract.rs:1958-1963`, `:1990-2003`). A
  fold's children are its body and its value bounds. `settle` writes `chosen` before the
  parent's `relax`, and a class settles once, so each bound's pick is final when the fold
  is priced.
  - On a literal fold, `literal_twin` returns at `arity() == 1`.
    `below(form, settled)` multiplies by `priced_trips() == len` and folds no further
    operands.
  - The heap key, `prefer`, `root_cost`, `Selection::price`, the claim/price audit and
    `sets.bytes` are unchanged.
  - Telemetry's `extracted_latency_prior_cost` (`telemetry.rs:235-280`) never calls
    `fold_cost`.
- **The laws.**
  - L2: normalization happens in `discover`. A declared literal pair still panics with
    "runs forwards".
  - L3: `variance_from` is the same per-node transfer that the forward pass runs, keyed
    by `ExprId`. It resolves only the `Ref`s the bound reaches.
  - L4: holds in both `transfer` and `ENode::variance`, in the same CL, which `FactorFold`
    needs. A bound that reads the outer binder must make the inner fold's fact carry it.
  - L6: already gated.
  - L7: `substitute` names a rebinding fold because its fact lacks the slot. L3 at `add`
    keeps the slot out of every bound fact, and facts only shrink (`columns.rs:284-303`).
    N5 is not claimed as L7.
  - L8: `KnownBound` is a true equality. It keeps monoid, binder and stride (stride 1, as
    `with_a_value` enforces). A reversed discovered pair becomes `[k, k)` for
    `EmptyFold`. It declines on a non-count and on a mask.
- **`HalveFold`'s `0..(i + s)`.** It is exact: `i + s < hi <= 2^24`. `add`'s backstop
  accepts it (fact `{outer binder}`).
- **`FactorFold` (decision 5).** It is exact for `n >= 1`. At an out-of-domain bound both
  sides are NaN under A4's guard. At a runtime `n = 0` the difference is the accepted edge.
- **N3.** The twin pricing is the same in the tree pass, the shared pass and
  `Selection::price`, so `costed`'s audit holds. T-S-13's equality needs leaves to cost 0,
  and `cost.rs:346-349` makes them free.

## Findings (non-blocking, ranked)

### 1. A fold-made mask literal in a bound panics at materialization, against N4's "never panics at extraction"

`Bound::discover` refuses a mask only when the bound is a `Const(Lit::Mask)`. That covers
the spec's §1.1 row 2 (spec:78) and `push_reduce` (spec:729-730). A mask-sorted *computed*
bound is accepted. Take `push_reduce(sum(0), Lit(0), Value(Lt(Var(5), Const(3))), body)`
under an outer `sum(1)`:
- The bound is not a literal, its L3 variance is `{i5}`, and SUM takes a value bound, so
  `push_reduce` builds the node.
- `PeelFold` on the outer fold plans `Lt(Const(2), Const(3))`. `ConstantFold`'s comparison
  arm yields `Lit::Mask(true)` (CLAUDE.md:67), which joins the bound class.
- `KnownBound` and `literal_twin` decline (`.ok()?`), as spec:78 says. The extractor then
  picks the free `Const(Mask(true))`, and `Selection::to_arena` embeds it before the fold
  (`extract.rs:935-984`). `embed` is `push_reduce` (`term_arena.rs:85`), whose discovery
  panics with `BoundSort` during extraction.

N4 asks for two things that conflict here: "a rule- or fold-made literal never panics at
extraction" and "a `Lit::Mask` bound is still `BoundSort` everywhere". The spec carries
both without noting that they collide. Separately, from A4 on, a false-mask bound reads as
the count 0 and silently runs no trips. That is the domain confusion CLAUDE.md says
deserves a type.

**Fix:** state it in §1.1 and §13 as a hole that T6 closes. Cheaply, `push_reduce` can also
refuse with `BoundSort` a value bound whose root is a comparison (`Lt`/`Le`/`Gt`/`Ge`/`Eq`/`Ne`).
Such a bound then never reaches the e-graph. No rule builds one.

### 2. The two L3 checkers disagree on a bound that contains an empty fold; spec:1358-1360's claim is false

The arena's transfer types a fold over a known empty range as `CONST`
(`variance.rs:355`). The e-graph's `ENode::variance` gives it `var(body) \ b`
(`node.rs:549-554`), with no empty arm. Take a bound `Reduce{[0,0), body: X}`:
- `push_reduce` accepts it, because `variance_from` gives `CONST`.
- `insert` then seeds the class fact `{X}`, and `add`'s backstop panics with
  `BoundVaries` on an arena that `push_reduce` admitted.

So "a bound class inserted from an arena that passed `push_reduce`'s L3 passes here" does
not hold for that bound. Only a hand-built bound can contain a fold: `over_values` refuses
one with `BoundArithmetic`, and no rule puts a fold into a bound.

**Fix:** make the sentence exact, naming the empty-fold exception, and pin it. Do not give
`ENode::variance` an empty arm in A3: that changes e-graph facts for literal folds, which
the identity argument would then have to re-prove.

### 3. `Saturation::total_evals` moves on every kernel that holds a fold; §1.3's identity table omits it

`KnownBound` has root shape `Reduce`. The runner calls `apply` on every `Reduce` node of
every dirty class, and `total_evals` counts those calls, matched or not
(`runner.rs:338-350`). So the counter rises on every production glyph kernel. The
anytime curves read it as `evals_actual` (`anytime.rs:245`), and so do the Guide research
bins (`phase3_round2_unguided_curves.rs:144`, `guide_scope_saturation_delta.rs:681`).
- Applications, unions, iterations and stop reasons do not move (spec:123 is right about
  those).
- `match_counts` does not move (no match is made).
- No gate reads `total_evals`.

**Fix:** add a row to §1.3 saying that this counter moves and no gate reads it, so a
shifted research curve is not taken for a regression.

### 4. T-S-6's extraction assertion can be vacuous; N6 asks for the fold to be present

Spec:2094 asserts that *every* slot-0 `Reduce` reachable from `out_root` has `hi` a literal
or a value bound within `{i5}`. That passes when no such fold survives. N6
(rulings.md:12) asks that the arena "holds the fold with `hi` a `Value` reading the outer
binder (or a literal after unrolling)".

**Fix:** also assert that at least one slot-0 `Reduce` is reachable. Its `hi` must be either
`Value(v)` with `v`'s variance `{i5}`, or a `Lit`. The rule-made `0 + 1` stays allowed
beside it, as the spec argues.

### 5. J13 spells `Kernel::over`'s out-of-domain refusal differently from N4

N4 (rulings.md:10) lists `Kernel::over`'s literal range among the places that refuse with
`BoundOutOfDomain`. The spec keeps `KnownRange::new`'s "ends at most at" message there
(spec:77, J13 at spec:2429). The refusal itself, and so soundness, is unchanged. Either
have the ruling amended to match J13, or spell the refusal `BoundOutOfDomain` as N4 names
it.

### 6. `FactorFold`'s new doc understates the runtime `n = 0` edge

The rewritten doc (spec:1510-1516) names only the infinite `c`. Under a value bound at a
runtime `n = 0`, the factored SUM also differs in two more cases:
- A NaN `c`: `NaN·0` is NaN, where the fold gives 0.
- A negative finite `c`: `c·(+0)` is `-0.0`, which `1/x` turns into `-∞` (CLAUDE.md's
  signed-zero note).

MIN and MAX differ at NaN `c`. Decision 5's reasoning accepts all of these. The doc is the
one place the edge is written down, so it should list them.
