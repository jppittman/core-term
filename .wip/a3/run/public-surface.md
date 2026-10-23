# A3 public surface: every new or changed `pub` item

One section per crate; each implementer appends or edits its own. "Visibility" is the item's
visibility after A3. "Re-export" says whether `src/lib.rs` re-exports it (nothing new is: the items
are reached through `pixelflow_ir::fold`, `pixelflow_ir::arena` and `pixelflow_ir::variance`, so
`pixelflow-ir/intent.md`'s Contract line stays true).

## pixelflow-ir

### New

| Item | Kind | Visibility | Where | What it is |
|---|---|---|---|---|
| `fold::BoundEnd` (`Lo`, `Hi`) and its `Display` | enum | `pub` | `fold.rs` | which end of a range a refusal names (`lo` / `hi`) |
| `fold::BoundRefusal` (`BoundOutOfDomain { end, literal }`, `BoundSort { end }`, `BoundVaries { end, reads }`, `BoundArithmetic { end }`, `BoundNotANumber { end }`) and its `Display` | enum | `pub` | `fold.rs` | why a fold refuses a bound; every refusal panics with its `Display`, which starts with the variant's name |
| `fold::Bound::<R: Copy>::discover(lo, hi, literal) -> Result<(Bound<R>, Bound<R>), BoundRefusal>` | assoc fn | `pub` | `fold.rs` | constness discovered, never declared: the one reader every representation calls |
| `fold::Fold::admits_bound(self, end, reads) -> Result<(), BoundRefusal>` | method | `pub` | `fold.rs` | L3 asked of the bound at `end`; calls `bound_may_read` |
| `fold::Fold::bound_may_read(self, reads) -> bool` | method, `#[must_use]` | `pub` | `fold.rs` | polish round 1: L3 stated once, end-free; `admits_bound` and the e-graph's `Site::member` (another crate, so `pub`) both call it |
| `fold::Fold::ESTIMATED_TRIPS: u64 = 32` | assoc const | `pub` | `fold.rs` | the trips a fold is priced at when its range is not known |
| `fold::FoldHeader::priced_trips(self) -> u64` | method | `pub` | `fold.rs` | known length, else `ESTIMATED_TRIPS` |
| `variance::Variance::of_var(index: u8) -> Variance` | const fn | `pub` | `variance.rs` | one answer for a `Var` leaf: its bit, `CONST` for a placeholder, `ALL` past them |
| `arena::FoldTerms { monoid, stride, lo, hi }` | struct, `pub` fields | `pub` | `arena.rs` | what a builder closes a fold's body over; replaces the `fold_at` closure |

### Changed (signature or accepted input)

| Item | Change |
|---|---|
| `arena::ExprArena::close_over(&self, body, placeholder, terms: FoldTerms)` | the third argument was `impl FnOnce(Binder) -> FoldHeader`; value bounds are extra roots, discovered before the copy |
| `arena::ExprArena::close_fold(&mut self, open, body, terms: FoldTerms)` | same |
| `arena::ExprArena::push_reduce(fold, lo, hi, body)` | signature unchanged; now discovers a `Const` bound, accepts a value bound, and refuses `BoundSort` / `BoundVaries` |
| `fold::FoldHeader::new(fold, lo, hi)` | signature unchanged; now accepts a value bound on `SUM`/`PRODUCT`/`MIN`/`MAX` at stride 1 |
| `variance::compute_arena_variance(arena)` | signature unchanged; L4 (a fold unions its value bounds) and a placeholder `Var` is `CONST` |
| `arena::ExprArena::free_index(root)` | signature unchanged; a fold's value bounds are free in it, unmasked |
| `fold::VALUE_BOUND`, `fold::KNOWN_RANGE` | texts only |
| `kernel::Kernel::over(monoid, range, body)` | signature unchanged; the closure became a `FoldTerms` value handed to `close_over`; the reversed-range and past-2^24 refusals are raised from `push_reduce` / `FoldHeader::new` after the body is copied, with the messages they had (`KnownRange::new`'s, not `BoundOutOfDomain`); the doc's static-range paragraph is replaced |
| `arena::ExprArena::known_range(id)` | doc only, as `ENode::known_range`'s: `None` for a fold with a value bound |
| `arena::ExprArena::substitute_vars_with(root, subs)` | doc states its precondition (`subs` names no slot a fold inside `root` rebinds); behavior unchanged; not the plan's L7 |
| `fold::FoldHeader::to_bits`, `fold::FoldHeader::from_bits` | `to_bits`: doc text only (a value bound has no bits until A4 gives it a marker), still panics with `VALUE_BOUND`; `from_bits`: the stride test is written through the private `Fold::is_exact`, the same predicate |
| `fold::KnownRange::admits` | the same predicate, written through `Fold::is_exact`, the one function `Fold::count` shares |

### New, not public (listed so a reviewer can see none leaked)

`Fold::count` (`pub(crate)`), `Fold::is_exact` (private), `Monoid::takes_a_value_bound` (`pub(crate)`),
`Bound::read` (private), `FoldHeader::with_a_value` (private: what a value bound asks of its fold, checked; `FoldHeader::new`'s value arm), `ExprArena::literal_at` (private), `variance::transfer` (private),
`variance::variance_from` (`pub(crate)`), `Kernel::over_values` (`pub(crate)`, `expect(dead_code)`
until A4), `Kernel::user_bound` (private, same).

### Not re-exported

Nothing new in `lib.rs`.

## pixelflow-search

Nothing is added to `pixelflow-search/src/lib.rs` or re-exported at `egraph::` (`egraph/mod.rs:83`
is untouched, so `KnownBound` is reached as `egraph::fold_rules::KnownBound`, the path every
`fold_rules` item has beside its re-export).

### New

| Item | Kind | Visibility | Where | What it is |
|---|---|---|---|---|
| `egraph::fold_rules::KnownBound` | unit struct, `impl Rewrite` (name `"known-bound"`, root shape `Reduce`) | `pub` | `fold_rules.rs` | L8, discovery: the fold over the literals its value bounds' classes hold, in the fold's class; last in `fold_rules()` |

### Changed (signature or accepted input)

| Item | Change |
|---|---|
| `egraph::cost::CostModel::fold_cost(&self, fold: Fold, trips: u64, body: usize)` | the second argument was `range: KnownRange`; the only callers are `node_op_cost` and the tests, all in `cost.rs` |
| `egraph::cost::CostModel::node_op_cost` / `CostFunction::node_cost` | a fold whose range is not known prices at `FoldHeader::priced_trips()` (the estimate) instead of panicking on `KNOWN_RANGE`; a literal fold is priced exactly as before |
| `egraph::FactorFold` | declines only a fold whose range is known to be empty (it declined any range not known to be non-empty), so it factors under a value bound and accepts decision 5's runtime `n = 0` edges, listed in its doc; a literal fold's range is always known, so its answer is the one it had |
| `egraph::HalveFold` | doc only: a nested fold's value bound is shifted with the binder, bound arithmetic only the rules make |
| `egraph::ENode` | the `Reduce` variant's doc only (for a value bound the shift's reason is reversed) |
| `egraph::fold_rules::fold_rules()` | returns five rules, `KnownBound` last; `RuleSet::runtime()` gains one index at the end and a new fingerprint (every earlier `RuleIdx` is kept; `SaturationKey` and the in-process cache change with it) |
| `egraph::ENode::reduce` | accepts a value bound on `SUM`/`PRODUCT`/`MIN`/`MAX` at stride 1; panics as `FoldHeader::new` |
| `egraph::ENode::known_range` | `None` for a fold with a value bound (doc only; signature unchanged) |
| `egraph::extract`, `egraph::extract_dag` (re-exported at `egraph/mod.rs:79-82`), (`extract_dag_scoped` is `pub` in the crate-private `extract` module, and reaches a caller through `Optimized::to_arena`) | a fold with a value bound is priced as its literal twin when its bounds' chosen forms are literals naming counts (`ENode::literal_twin`, through `node_cost` and `below`), else at `Fold::ESTIMATED_TRIPS`; `below` adds the value bounds' operands, and no longer expects `KNOWN_RANGE`. A fold over literals is priced as it was. |
| `egraph::HalveFold`, `egraph::PeelFold` (`Rewrite::apply`) | round 4: a class reached as a nested fold's value bound is rebuilt through its first member that keeps L3 (`Fold::admits_bound` on the member's variance over its children's facts), and the rule declines (`None`) when a class under a value bound has none, or when a fold is nested inside one. Bodies keep the first member. A fold over literals walks as it did. Polish round 1: the member test is `Fold::bound_may_read`, and a fold is skipped as a member at a bound (the walk does not enter one), so the rule declines only when a class in or under a bound has no member that keeps L3 and is no fold; a fold nested inside a bound is no longer a reason to decline. |
| `runtime::optimize_runtime_arena` | round 4: a unit (a kernel named by a `Ref`) that reads an index no fold inside it binds panics, naming the index, beside the existing refusal of a unit that varies with a binder or a retired axis. Terms without a `Ref`, and units of every term the repository builds, are unchanged. |
| `egraph::EGraph::add` | panics with `BoundVaries` on a fold whose value bound's class varies with a coordinate or with the fold's own binder; a fold over literals is not looked at. Closing round 2: the body is the base's with the backstop read through `refusal` (private), the doc says a rule's plan does not reach the panic through a lawful term that lands in a stale class (`plan_shape` declines first); behavior for every other caller is the same |
| `egraph::RewriteAction::Plan` | closing round 2, doc and behavior: applying a plan whose fold's value bound lands in a class failing L3 unions nothing (it used to panic in `add`); a plan every bound of which is lawful builds exactly what it built |
| `egraph::insert`, `egraph::inserted_class_count` | read a value bound whose term is a literal naming a count into the node (no class made for it); `insert` panics with `BoundSort` on a mask-literal bound |

### New, not public (listed so a reviewer can see none leaked)

`extract.rs`: the `chosen` field of `TreePricer` and `SharedPricer` (private structs). `node.rs`:
`ENode::literal_twin` (`pub(crate)`, round 4: the one function that makes a fold's literal form,
called by `KnownBound::apply` and the three price sites; it replaces the private `literal_twin` in
`extract.rs` and the copy of its body in `KnownBound::apply`; polish round 1: an `#[inline(always)]`
guard that answers `None` for everything but a fold with a value bound, over `ENode::value_bounded_twin`
(private)). `fold_rules.rs`: `Site` (derives `Clone, Copy` only), `Site::member`,
`Site::of_child` (total since polish round 1), `Memo` (private, round 4: where
`PlanBuilder::substitute` is rebuilding a class and what it has finished); `representative` is gone
(inlined into `Site::member`'s body arm). `graph.rs` (closing round 2; all private to `graph.rs`): `EGraph::refuse_a_varying_bound` (the backstop for a canonical node outside the memo, out of line), `VaryingBound` (the refusal, `Display` is the panic text), `has_a_value_bound` and `varying_bound` (L3 over classes, stated once, read by `refuse_a_varying_bound` and the growth predictor), `NodeSink::refuses` (whether `make` would refuse a node; EGraph's and the predictor's), `GrowthPredictor::fact` (the fact `add` would seed for a class the prediction minted), and `plan_shape` returns `Option<EClassId>`. `insert.rs`: `literal`
(private fn). Removed: `node.rs`'s private `var_variance` (`Variance::of_var` is the one answer).
`ENode::variance` stays `pub(crate)`.

## pixelflow-codegen, pixelflow-compiler, pixelflow-pipeline (the tail)

No `pub` item is added, removed or re-signed in these three crates, and nothing is re-exported.

| Crate | Item | Visibility | What changed |
|---|---|---|---|
| `pixelflow-compiler` | `Staged::fold_at` renamed `Staged::fold_terms` (`emit.rs`) | private associated fn | emits a `FoldTerms` value where it emitted a closure; the arena it builds is the same, node for node |
| `pixelflow-compiler` | `Expansion::close_fold` (`lower.rs`, the `Site` impl's method) | private | builds a `FoldTerms` for `ExprArena::close_fold`; `Fold` stays imported (`Fold::EXACT_BOUND`), `FoldHeader` moved into the test module's imports |
| `pixelflow-codegen` | `program/lower.rs`: a comment, tests `a_value_bound_is_refused_by_name_before_it_is_scheduled` and `a_surviving_placeholder_fails_loudly`, and the `Var` arm's panic text (round 4: names a placeholder beside a coordinate; polish round 1: two messages, a coordinate (`Var(i)`, `i` below `Variance::VARIABLES`: `collapse` was skipped) and an index past the binder slots (an open kernel)) | text and tests only | `arena_to_schedule` and `known` are unchanged and still refuse a value bound with `VALUE_BOUND`; a `Var` that is no fold's binder was refused and is still |
| `pixelflow-pipeline` | `training/corpus.rs` `write_node`'s error text; a comment in `examples/glyph_phase_split.rs` | text only | "before A4's format version" |

What the tail's code reaches that was already public: the `kernel!` expansion now names
`::pixelflow_core::__macro::ir::arena::FoldTerms` (the `pub struct` from `pixelflow-ir`, listed above)
and `::pixelflow_core::__macro::ir::fold::Bound::Lit`. `__macro` is `pixelflow-core`'s existing
macro-support module; nothing is added to it.
