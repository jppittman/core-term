1:# A3 spec: value bounds in the IR and the e-graph
9:## 0. What A3 is, and what it is not
41:## 1. The laws, as the code that enforces them
63:## 2. New items in `pixelflow-ir/src/fold.rs`
207:## 3. The arena (`pixelflow-ir/src/arena.rs`, `variance.rs`)
209:### 3.1 `Variance::of_var` (variance.rs, new, `pub`): one answer for a `Var` leaf
236:### 3.2 L4 in `compute_arena_variance` (variance.rs:349-366)
255:### 3.3 `free_index` and `free_indices` (arena.rs:739-802)
263:### 3.4 `push_reduce` (arena.rs:934-964): discovery, then the header, then L3
322:### 3.5 `FoldTerms`, `close_over`, `close_fold`, `lowest_free_binder` (arena.rs:1621-1768)
401:### 3.6 `known_range` docs
409:## 4. `Kernel::over_values` (`pixelflow-ir/src/kernel.rs`, `pub(crate)`)
480:## 5. L7 and the every-child walkers
534:## 6. The e-graph (`pixelflow-search`)
536:### 6.1 `ENode` (node.rs)
561:### 6.2 `EGraph::add`: the L3 backstop (graph.rs:359-389)
582:### 6.3 `insert` (insert.rs)
606:### 6.4 `KnownBound` (fold_rules.rs, new; L8)
648:### 6.5 Rules gated on known ranges (L6)
658:### 6.6 `FactorFold` under a value bound (decision 5)
679:### 6.7 `HalveFold`'s rule-made bound arithmetic
689:### 6.8 `substitute` and `rebuild`
696:## 7. Pricing
740:## 8. Codegen (A3 keeps the refusal)
761:## 9. The compiler (mechanical: signature only, no arena change)
790:## 10. Tests (the gate's pins, and the rewrites A3 forces)
795:### pixelflow-ir unit tests, `src/fold.rs`
817:### `src/variance.rs`
826:### `src/passes.rs`
833:### pixelflow-ir integration tests, new file `tests/value_bounds.rs`
867:### `src/kernel.rs` unit tests (`over_values` is `pub(crate)`)
901:### `src/arena.rs` unit tests
923:### pixelflow-search, `src/egraph/node.rs`
936:### `src/egraph/graph.rs`
942:### `src/egraph/insert.rs`
956:### `src/egraph/fold_rules.rs` (tests module)
1006:### `src/egraph/cost.rs`
1014:### `pixelflow-search/tests/root_shapes.rs`
1020:### pixelflow-codegen, `src/program/lower.rs` tests
1029:### Gate (plan:563)
1044:## 11. Order of work and the grep-gated checklist
1094:## 12. Docs A3 edits (plan §5 rows marked A3, and the code docs under changed code)
1129:## 13. Judgement calls, each with the alternative rejected
# A3 spec: value bounds in the IR and the e-graph

Step A3 of `docs/language/2026-10-20-a-bound-is-a-value.md` (the plan). Ground truth is
`f772852c` (the A2b squash, #1443); every `file:line` below is at that commit unless it says
"plan". Plan sections read: §0–§4 in full, §6's A3 bullet (plan:559-568), §8 as recorded
(decision 2 = `n: usize` in `kernel!`, which is A6; decision 5 = `FactorFold` factors under a value
bound and declines only a known empty range; decision 7 = public API approved).

## 0. What A3 is, and what it is not

**Is.** A fold's bound may be a term (`Bound::Value`) in the arena (`ExprNode::Reduce`) and in
the e-graph (`ENode::Reduce`). Constness is discovered where a node is built, never declared. The
laws L2, L3, L4, L6, L7 and L8 are enforced, and so are §1.4's domain, sort and arithmetic
refusals. `Kernel::over_values` is the builder (`pub(crate)`). The rules are gated on known ranges,
`KnownBound` makes a range known, `FactorFold` factors under a value bound, and the price is
`ESTIMATED_TRIPS`.

**Is not.**
- **Codegen (A4).** `arena_to_schedule` keeps refusing a value bound by name
  (`pixelflow-codegen/src/program/lower.rs:48-52`, `:340-348`). There is no operand or carve
  change, no guard, and no `Trips`.
- **The key (A4).** `FoldHeader::to_bits` keeps refusing a value bound (`fold.rs:689-692`), so
  `canonical`, `KernelKey::of`, `Kernel::by_ref`, the JIT cache and the runtime tier's optimizer
  cache refuse a value-bounded kernel by the same name. This deviates from what A2b wrote down;
  see J10.
- **`kernel!` value bounds (A6).** `kernel!` keeps literal bounds, and its two `close_fold` call
  sites change only mechanically (§9).

**Visibility, stated plainly.** `ExprArena::push_reduce` is `pub` (`arena.rs:949`) and accepts a
value bound after A3, and `Kernel::from_parts` is `pub` (`kernel.rs:331`). So a value-bounded kernel
is publicly constructible in A3 through the arena. `pub(crate)` on `over_values` hides the
spelling, not the capability. Decision 7 (plan:654-656) makes `over_values` public in A4; A4 does
not open the door, because A3 already has.

**Gate** (plan:563-568): byte_probe is empty, and there are pins for L2, L3, ANY/ALL/SEQ, a
non-leaf bound at `over_values`, L4, L7, L8, and a triangular fold under `HalveFold`. §10 lists
them.

---

## 1. The laws, as the code that enforces them

| Law (plan) | Where | Check | Refusal | Panic or Result |
|---|---|---|---|---|
| §1.4 domain (plan:118-138, L5 plan:190) | `Fold::count` (new, fold.rs) via `Bound::discover` (new), called by `ExprArena::push_reduce`, `Kernel::over_values`, `insert`, `KnownBound` | a literal value bound must be `Lit::Real(v)` with `v` an integer in `0..=2^24` | `BoundRefusal::BoundOutOfDomain { end, literal }` | `discover` returns `Result`. `push_reduce`, `over_values` and `insert` panic. `KnownBound` declines (`.ok()?`). |
| §1.4 mask bound (plan:256-257) | same | `Lit::Mask(_)` names no count | `BoundOutOfDomain` (J2) | same |
| §1.4 sort (plan:125-129, 258) | `FoldHeader::new` (fold.rs:559) | any `Bound::Value` and a monoid not in {SUM, PRODUCT, MIN, MAX} | `BoundRefusal::BoundSort { monoid }` | panic. Reached from `push_reduce`, `ENode::reduce`, `insert`, plan replay and templates. |
| stride (plan:266-268) | `FoldHeader::new` | any `Bound::Value` and `fold.stride() != 1` | not a `BoundRefusal`: a malformed node, with assert message "a fold's stride is not 1 only over a known range it divides" | panic |
| §1.4 arithmetic (plan:140-154, 258-260) | `Kernel::over_values` only (kernel! sema is A6) | the bound kernel's root `ExprData` is not `Var`, `Const` or `Uniform` | `BoundRefusal::BoundArithmetic { end }` | panic |
| L2 (plan:174-179) | `Bound::discover` normalizes. `KnownRange::new` (fold.rs:832-853) keeps refusing a declared reversed pair. | a pair made known *by this read* with `lo > hi` becomes `[lo, lo)`. A pair the caller passed as literals is not touched. | "runs forwards" (existing message, `KnownRange::new`) | panic |
| L3 (plan:180-188) | `ExprArena::push_reduce` (arena). `EGraph::add` (backstop). | `(var(lo) ∪ var(hi)) ∩ (COORDS ∪ {b}) = ∅`; §3 below gives the exact computation | `BoundRefusal::BoundVaries { end, reads }` | panic in both |
| L4 (plan:189) | `compute_arena_variance` (variance.rs:310) and `ENode::variance` (node.rs:526) | `(var(body) \ b) ∪ var(lo) ∪ var(hi)` | — | — |
| L6 (plan:192-195) | `PeelFold`, `HalveFold` and `EmptyFold` (fold_rules.rs:390-484) | each fires only on `node.known_range()` being `Some` (already true at f772852c: `:393`, `:435`, `:480-481`) | — | decline |
| L7 (plan:196-198) | `PlanBuilder::substitute` (fold_rules.rs:191-278) through L4's class fact. Arena side: `close_over`'s rename. | substitutes into a nested fold's bounds always, and into its body unless the fold rebinds the slot | — | — |
| L8 (plan:199-202) | `KnownBound` (new rule, fold_rules.rs) | a value bound's class has a `Const` member that `Fold::count` reads, so the `Lit` form of the fold is created in the class | — | decline otherwise |

Where both could apply, the refusals in `push_reduce` come in this order: discover
(`BoundOutOfDomain`), then `FoldHeader::new` (`BoundSort`, stride, literal-pair checks), then L3
(`BoundVaries`). The cheap checks run first and the O(n) variance pass runs last (J15).
Shell cwd was reset to /home/user/pixelflow