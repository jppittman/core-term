# A3 in pixelflow-search: what was built (implementer 2 of 3)

The working branch, code base `3bfdaded`, on top of implementer 1's `pixelflow-ir`. Scope: `pixelflow-search`
only (spec §6, §7 and §11 step 5, §10's search tests, B1 for search, C15's search share). Committed locally as
`chore(wip): A3 in pixelflow-search (not for merge)`; not pushed. `pixelflow-ir` needed no change.

## Status

| Gate | Result |
|---|---|
| `cargo test -p pixelflow-search`, default, `--all-features`, `--features saturation-telemetry` | green in all three: 387 lib tests passed, 8 ignored (base 367 passed, 8 ignored: +20), and every integration binary (`root_shapes` 1, `fold_exactness` 5, `ir_insert` 13, `production_budget_determinism` 7, `saturation_stop` 7, `saturation_ceiling_env` 2, `latency_prior_regression` 2; the two growth binaries under the feature) |
| `cargo test -p pixelflow-codegen` against the new ir and search | green (all binaries, `halve_fold_jit` among them) |
| `cargo clippy -p pixelflow-search --all-targets --all-features -- -D warnings` | **red at base too, through `pixelflow-ir`** (the three `must_use_candidate` findings implementer 1 reported: `ExprArena::nodes`, `Dag::iter`, `Builder::entries`); with `--no-deps` the search crate itself is clean, and with `-A clippy::must_use_candidate` the whole chain is clean |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features` | green |
| `cargo fmt -p pixelflow-search -- --check` | clean |
| rustdoc (ruling C4), `cargo doc -p pixelflow-search --no-deps --document-private-items` | base 28 warnings, head 28, `NEW 0`, `GONE 0` |
| `bash scripts/check-intent.sh` | OK; `pixelflow-search/src/egraph/intent.md` is 75 lines (was 74) |
| `check-doc-paths.sh`, `check-cfg-encapsulation.sh` | OK |
| B1 in search (`grep -rnwE 'A3' --include=*.rs pixelflow-search`, the phrase grep, `*.md` under `pixelflow-search`) | empty. The two remaining workspace hits are not search's: `pixelflow-codegen/src/program/lower.rs:344` and `pixelflow-pipeline/src/training/corpus.rs:408` (implementer 3) |
| Reduce-site checklist (`python3 -I .wip/a3/gen/checklist.py`) | `UNCLASSIFIED 0`, `STALE 0` (169 sites) |
| Other §11 greps | `grep -rn 'var_variance' pixelflow-search`: empty. `grep -rn 'fold_cost(' --include=*.rs`: `egraph/cost.rs` only. `grep -rn 'KNOWN_RANGE' cost.rs extract.rs`: empty. `grep -rn 'VALUE_BOUND'`: `fold.rs` and `program/lower.rs` only. `KnownBound` in `fold_rules.rs`: the struct, its impl, the last entry of `fold_rules()` and its doc, and tests |

**Byte identity, what ran.**

- **Gate 1, byte_probe** (`.wip/gates/byteprobe.sh a3-search`, with implementer 1's ir): `IDENTICAL` to base under
  `PIXELFLOW_ISA=avx512` and `=avx2` (127 lines each). `git status` shows nothing outside `.wip/` and
  `pixelflow-search/`.
- **A differential harness** (a scratch crate outside the repository, in the session scratchpad
  `.../scratchpad/a3-search-diff`, built once against `3bfdaded` extracted with `git archive` and once against this
  tree, `saturation-telemetry` on). 25 literal-fold kernels: `Σ|X−j|` over 0, 1, 2, 3, 4, 5, 7, 8, 16, 33 and 40 terms,
  every monoid, `FactorFold`'s three shapes, three table folds (4, 9, 40), a nested pair, a fold with a uniform and an
  `If`, sibling folds. Per kernel: `inserted_class_count`; `optimize_runtime_arena` at `[16,16]` and `[64,4]` (every
  arena node in order and the `KernelKey` bits); and the runtime rule set at 3 and 40 iterations (`SaturationStats`,
  `num_classes`, `application_count`, `extract_dag`'s `total_cost`, `dag_cost` and `report`, and the extracted arena).
  The two transcripts (1,122 lines) are `diff`-identical, and the 25 telemetry records are equal field for field once
  `wall_clock_us` is dropped, `extraction_objective`, `extracted_latency_prior_cost` and `shared_pass_bytes` included.
- **Not run: gates 1b, 2, 2x and 5.** They build `pixelflow-graphics`, which needs `pixelflow-compiler`, which does not
  compile until implementer 3's `lower.rs` and `emit.rs` change. They remain to be run once it does; the harness
  above is the search-level share of them.
- **S3, measured.** `Saturation::total_evals` rises on every kernel with a fold, as the soundness critique said:
  by 1 to 55 on the 3-iteration runs of this set (at most 5% of a 3,000-call run) and by 1 to 326 on the 40-iteration
  ones (at most 2.3% of a 44-call run; for the larger kernels a few hundredths of a percent: `sum_abs_40`,
  141,243 to 141,291; `min`, 815,462 to 815,788). Nothing else moves. No gate reads it; the anytime curves and the
  Guide research bins do.

## What changed, by item

### `node.rs` (spec §6.1)

- `ENode::variance`: L4. A fold is its body's variance less its own index, unioned with each value bound's; a literal
  fold has no bound children, so the `children[1..]` fold meets nothing. The `Var` arm is `Variance::of_var`;
  `var_variance` is deleted.
- Docs: `ENode::Reduce` (L8: a rule reaching a value bound is the point; `PeelFold`'s `0..i` to `0..k`), `ENode::reduce`'s
  `# Panics` (as `FoldHeader::new`: `ANY`, `ALL`, `SEQ`, a stride other than 1; the `VALUE_BOUND` link is gone),
  `ENode::known_range` (§3.6), and `ENode::variance` (C15), which also says the arena types a fold over a known empty
  range `CONST` and this has no such arm (S2).
- Tests: T-S-1 (`a_fold_node_holds_a_value_bound_after_its_body`), T-S-2 (`a_folds_variance_is_its_bodys_less_its_index_and_its_bounds`), and
  `a_var_leaf_names_its_own_bit` now pins `leaf(VARIABLES) == CONST`, `leaf(127) == CONST`, `leaf(128) == ALL`.

### `graph.rs` (§6.2)

- `EGraph::add`: after the memo lookup and before the variance is read, `if let ENode::Reduce { header, children } = &node
  && header.arity() > 1 { self.refuse_a_varying_bound(header, children) }`. The private
  `refuse_a_varying_bound` reads each value bound's class fact through `Fold::admits_bound`, `lo` before `hi`, and
  panics with the refusal's `Display` and the fold. A literal fold costs one discriminant test and one arity read.
- `add`'s doc: the new panic, "does not refuse bound arithmetic" (`HalveFold`'s `binder + s`), and the sentence on why a
  bound class inserted from an arena that passed `push_reduce` passes here, with S2's exception stated exactly there
  (the arena's transfer types a fold over a known empty range `CONST`, `ENode::variance` has no such arm, so
  a hand-built bound holding such a fold may be refused). `ENode::variance` got no empty arm.
- Tests: T-S-3a/b/c in `variance_fact_tests`, beside `a_fold_removes_its_binder`.

### `insert.rs` (§6.3)

- `insert` discovers at the fold's visit (`Bound::discover` through a private `literal(term, r)`; `BoundSort` for a mask
  panics before anything is built) and again at its completion (`expect`: it was discovered when visited), then builds
  the header from the discovered pair. `inserted_class_count` mirrors it with `unwrap_or((lo, hi))`: a measure, not a
  check.
- Tests: T-S-11 (`insert_reads_a_literal_bound_into_the_node`) replaces `insert_refuses_a_value_bound_by_name` and
  `a_value_bounds_class_is_counted`; `ValueBounded` is now an enum (`Constant`, `Uniform`, `Mask`). C9: its doc no longer
  says "before A3". Added: `insert_refuses_a_mask_bound`.

### `fold_rules.rs` (§6.4 to §6.8)

- `KnownBound` (`pub struct`, `impl Rewrite`, name `known-bound`, root shape `Reduce`), appended last to `fold_rules()`.
  `apply`: declines unless the node is a fold with a value bound (`header.arity() > 1`); otherwise scans each bound
  class for a literal member (`egraph.nodes(class).find_map(ENode::literal)`), reads the pair through
  `Bound::discover`, declines on a refusal (a mask) or on anything but a pair of literals (J12), and answers
  `Create(ENode::reduce(fold, lo, hi, body))`. A count past the domain stays a value in `discover`, so it declines too.
- `FactorFold`: `node.known_range().is_some_and(KnownRange::is_empty)` instead of `is_none_or(...)` (decision 5). Its doc is
  S6's: "A fold known to be empty declines" and "A fold whose range is not known factors", with the four runtime `n = 0`
  edges listed (infinite `c`; NaN `c`; negative `c`, `c·(+0)` is `-0.0`; for `MIN` and `MAX`, NaN `c`).
- `rebuild`'s comment is §6.8's. The module doc gains the discovery law and a paragraph (C15); `HalveFold`'s doc gains
  the shifted-bound paragraph (§6.7, no code change); `fold_rules()`' doc names `KnownBound`.
- Tests: T-S-4 (`a_peel_substitutes_into_a_nested_folds_bound`), T-S-4b (`a_peel_names_a_rebinding_fold_with_a_value_bound`),
  T-S-5 (`a_peel_makes_an_inner_bound_known`, over two saturation rounds, plus
  `known_bound_makes_a_range_known_from_a_class_holding_a_count` for the direct `Create`), T-S-6
  (`a_triangular_fold_halves_with_a_rule_made_bound`, below), T-S-7 (`an_unknown_range_is_not_unrolled`), T-S-8a
  (`factor_fold_factors_under_a_value_bound`), T-S-8b (`factor_fold_declines_an_empty_fold`, doc reworded), T-S-9a to d,
  T-S-10 (`known_bound_is_the_last_fold_rule`). Added: `a_literal_joining_a_bound_class_makes_the_range_known` (a union
  one level below the fold wakes the rule: the "match depth 1" claim, through the runner).
- **T-S-6, shape (C7).** With no `ConstantFold` only the peel's `i := 0` gives an inner bound that is a literal (the empty
  fold `0..0`; I listed every `Reduce` in the saturated graph to check), and a rule-made `0 + 1` stays arithmetic, so the
  shifted copies' inner folds stay loops at the estimate and the unrolling is no cheaper. Extraction therefore returns
  `reduce[add_5[0..4)](reduce[add_4[0.._)](mul(Var(0), Var(4)), Var(5)))`, the triangular fold itself. The test asserts
  the HalveFold plan (`hi` is `Value(Plan(k))`, `nodes[k] = Add [Class(i), Plan(s)]`, `nodes[s] = Const 1`),
  saturation at 50 iterations completes, the extracted arena reaches at least one slot-0 `Reduce` (non-vacuous), each has
  `hi` a `Lit` or a `Value` whose variance is within the outer binder's alone, and `free_index == None`.

### `cost.rs` and `extract.rs` (§7, ruling J7)

- `CostModel::fold_cost(fold, trips: u64, body)`; `node_op_cost`'s fold arm is `fold_cost(fold, header.priced_trips(), 0)`, so
  `usize::try_from(trips)` replaces `range.len() as usize` (equal on a 64-bit host). The imports of `KNOWN_RANGE` and
  `KnownRange` are gone (C11), here and in `extract.rs`; the cost test's `KNOWN_RANGE` import went with its use.
  `fold_cost` stays in `cost.rs` alone.
- `extract.rs`: the private `literal_twin(node, chosen)` (spec text, with the module doc's path to the plan in place of a
  bare citation); `below` reads `header.priced_trips()` and then folds the remaining operands (the value bounds, charged
  once) with `usize::saturating_add`; its doc says "before value bounds `below` read only the body" (C1).
  `TreePricer` and `SharedPricer` each gain `chosen: Vec<Option<Lit>>`, written first in `settle` and read by
  `literal_twin` in `price`, which hands the form to `weighted_own` and (tree pass) to `below`. `Selection::price` reads a
  bound's pick off the selection (`pick_of`, the lookup its variance closure already did, now shared) and prices the
  form. `settled` still iterates the node as written.
- **Identity on a literal fold** (§7.5): `literal_twin` returns at the first `let ... else` for a non-fold and at
  `header.arity() == 1` for a literal fold; `below` folds an empty remainder. The harness above covers it.
- Tests: T-S-12 (`a_fold_whose_range_is_not_known_is_priced_at_the_estimate`, in `cost.rs`), T-S-13
  (`both_twins_of_a_fold_price_at_the_count_their_bound_names`), T-S-14 (`the_extractor_prices_a_fold_at_fold_cost`, now with
  `fold_cost(fold, 40, ..)`). T-S-13 prices both candidates under the tree and the shared pass through a recording
  `StageRecorder`, checks `extract_dag`'s total and the claim/price audit (`signed_error == 0`), a graph holding only the
  value twin, and that the extracted arena holds the twin (`known_range == 0..40`); I broke `literal_twin` on purpose to
  see it fail (124 against 156), then restored it. Added: `a_bound_that_picks_no_count_is_priced_at_the_estimate`
  (`2.5`, `16777218.0` and `Mask(true)` each price at 32 trips).

### `tests/root_shapes.rs` (T-S-15) and `intent.md`

- `terms()` gains the triangular fold (inner `sum(0)` over `0..Var(5)`, outer `sum(1)` over `0..3`); `KnownBound` fires in
  `no_rule_matches_a_node_outside_its_root_shapes`, so the silent-rule check passes.
- `pixelflow-search/src/egraph/intent.md` (N9): the cost line (`n·body + (n−1)·combine`, plus value bounds once; `n` from
  the node, else the bounds' chosen literals, else `Fold::ESTIMATED_TRIPS`) and "a fold over a known range ties its
  unrolling". 75 lines.

### Reduce-site rows

`classes.json` (rows keyed by `(file, fn, text)`): 4 rows edited in place (`below`, `insert` Visit and `variance` for their
new text, `price` for its reason) and 7 added (`literal_twin`, `EGraph::add`, `KnownBound::apply`, and four new test
sites in `fold_rules.rs`). The checklist reports 0 and 0. The informational `lines` of the search rows were refreshed and
the file re-sorted, so some search rows moved within the file.

## Judgement calls (the rejected alternative in each)

1. **`KnownBound::apply`'s guard is one `!matches!(node, ENode::Reduce { header, .. } if header.arity() > 1)`**, not the
   spec's `let ... else` followed by `if header.arity() == 1`. Same behaviour. The checklist keys by
   `(file, fn, text)`, and `EmptyFold::apply` has the spec's first line verbatim in a method of the same name, which
   would have put two sites of different class under one row. Rejected: a shared row of mixed class.
2. **T-S-13 states its expected price as `node_op_cost` of the literal fold plus `40 · body`**, not `fold_cost(..)`. Ruling J7
   keeps `fold_cost` in `cost.rs` alone, and §11's grep reads that way; the spec's own T-S-13 text calls it. The
   test's identity is `own + n·body`, and `the_extractor_prices_a_fold_at_fold_cost` pins `fold_cost` against the
   extractor in `cost.rs`. Rejected: relaxing the grep.
3. **T-S-13's recorder is local to the test**, as the three existing `Prices` recorders are. Rejected: hoisting one.
4. **T-S-3 sits in `variance_fact_tests` (`graph.rs`)**, with `a_fold_removes_its_binder`. Rejected: the first test module,
   which holds no fold.
5. **`KnownBound` is not re-exported at `egraph::`.** The spec lists no re-export; it is `pub` in `pub mod fold_rules`,
   reached as `egraph::fold_rules::KnownBound`, and every consumer takes `fold_rules()`. Rejected: widening
   `egraph/mod.rs:83`.
6. **`ENode::Reduce`'s L8 paragraph follows the "child class, held inline" sentences** instead of sitting between the
   "(above)" sentence and "A bound that is a value is a child class", and says "the shift's reason is reversed" to
   keep the antecedent. Same content as §6.1.
7. **`insert`'s Complete arm re-discovers and `expect`s** (the spec's code) rather than carrying the discovered pair from
   Visit in a side table. Rejected: a second per-fold table for a read that costs nothing on a literal fold.
8. **`ValueBounded` is one enum** with `Constant`, `Uniform` and `Mask`, not a sibling struct per term. Rejected: three `Ir`
   impls for one shape.
9. **Added beyond §10:** `insert_refuses_a_mask_bound` (J21's `BoundSort`),
   `known_bound_makes_a_range_known_from_a_class_holding_a_count`, `a_literal_joining_a_bound_class_makes_the_range_known`,
   `a_bound_that_picks_no_count_is_priced_at_the_estimate`, and T-S-13's audit and arena assertions. Each pins a claim a
   doc makes (J12's decline, match depth 1, J7's "otherwise the estimate", "the arena holds the twin").
10. **S2 is stated, not pinned.** Ruling S2 says to state the exception where the spec claims the two checkers agree and
    to add no empty arm; a test that pins the refusal would encode a hole. It is stated in `add`'s doc and in
    `ENode::variance`'s. Rejected: the critique's "and pin it".
11. **S3 is not pinned** (the ruling makes it a doc row only). The measurement above is its evidence.
12. **Plan citations in new docs are paths** (`docs/language/2026-10-20-a-bound-is-a-value.md §n`), not "plan §n", and no
    ruling label (N3, J7, C-numbers) appears in a tracked file; laws (L2 to L8) and "decision 5" are the plan's own names.

## Departures from the spec or the rulings

- Judgement calls 1 and 2 are the only places code or a test is not the spec's text. Neither changes behaviour.
- T-S-6's assertion is the C7 form (existence plus the bound's variance within the outer binder's), not the spec's
  original universal form; T-S-6's doc states the shape.
- The spec's `fold_cost` doc links `FoldHeader::priced_trips` bare; it is an explicit `pixelflow_ir::FoldHeader::priced_trips`
  path (C5).
- Not done here, by scope: B1's two non-search hits, the compiler tail, `program/lower.rs`, docs (§12), and gates 1b, 2, 2x
  and 5.

## Facts the other implementer needs

- `CostModel::fold_cost` takes `trips: u64`; nothing outside `cost.rs` calls it.
- `RuleSet::runtime()` has one more rule at the end. The in-process optimizer cache key changes; nothing persisted keys on
  the rule set's fingerprint except `SaturationKey`.
- `pixelflow-codegen` and `pixelflow-core` need no source change for search; `cargo test -p pixelflow-codegen` is green.
- Gates 1b, 2, 2x and 5 need `pixelflow-compiler` to build (its `lower.rs:515` and `emit.rs:838`). Run them once step 7 lands:
  the harness above says the search crate itself moves nothing on literal folds, but only those gates read the 29 programs.
- The `chosen` table costs one `Vec<Option<Lit>>` per DP pass and one node lookup and store per settle. Gate 4's
  instruction count is the arbiter; I expect it well inside +0.5% (about 2 MB of writes and a few million instructions
  against 6.3e9) but did not measure it, for the reason above.
