# The extractor is a trait, and beam search is its second implementation

2026-09-08. Denotation first; the code below is obliged to it.

## Why now

The saturation side has a seam. `ApplicationFilter` (#1228/#1231) is a
`rules × nodes ⇒ rules × nodes` function applied *before* any rule fires,
with a naive impl that is a no-op and byte-identical to production.
Extraction has no such seam:

- `Optimizer` (`egraph/optimizer.rs:275`) holds `cost: CostModel` as a
  concrete struct and calls `super::extract::extract_dag_scoped(..)` **by
  name** at `optimizer.rs:685`.
- `cost.rs:177` defines `CostFunction` — a node-cost trait — and
  `CostModel` implements it at `:337`. Every extraction function is already
  generic in it (`extract_dag_scoped<C: CostFunction>`). Nothing *above*
  those functions is: `Optimizer` fixes `CostModel` at its field.
- `extract.rs:268` has `pub trait Reranker`, an inner seam with no shipped
  impl, reached through a second, hand-rolled arm of the same `match`.

And the extractor is now the measured bottleneck. The class-cap sweep
(`docs/results/2026-09-08-class-cap-sweep.md`) shows dag_cost getting
**worse** as the graph grows — `shader:chrome` 1,668 at a 5,000-class cap,
~2,373 at 100,000 — in the extractor's own objective. More budget cannot
remove a term from the e-graph; monotone saturation only adds. So every one
of those cases leaves a **witness**: the cheaper term is provably still in
the bigger graph and the extractor did not find it. Beam search is the
first *non-learned* test of whether that is a search-**width** failure.

## Denotation

An **extractor** is a function

```text
(EGraph, root: EClassId, costs: C, shape: LatticeShape) -> ExtractedDAG
```

Its **objective** is `ExtractedDAG::dag_cost` — each distinct chosen
e-class priced once under `costs`, weighted by `shape` — because that is
what the emitted kernel pays. `dag_cost` is a property of the term, so
every number here is exact, not sampled.

The value it returns is a **choice map** `Vec<Option<usize>>` (one node
index per reachable e-class) plus the derived schedule/share/cost data
`ExtractedDAG` already carries. Two extractors are equal iff their choice
maps are equal — a comparison this repo can make byte-for-byte, which is
why the identity claim below is a test rather than a hope.

An extractor must report **which objective it actually optimized**
(`ExtractionReport { objective, shared_pass }`). `ExtractionObjective` is
already a four-state enum precisely so a number quoted from an
`ExtractedDAG` cannot be mistaken for a number from the production
objective — `TreeOnly` (the sharing-aware pass hit
`SHARED_DAG_PASS_BYTE_BUDGET` and was abandoned) and `External` (choices
came from outside the DP) are loud variants, not silently identical maps.
A new extractor inherits that obligation: it does not get to invent a
number and label it `Shared`.

## The trait boundary

```rust
pub trait Extractor<C: CostFunction + ?Sized> {
    fn extract(&self, egraph: &EGraph, root: EClassId, costs: &C,
               shape: LatticeShape) -> ExtractedDAG;
}
```

**One method.** Everything else — `compute_ref_counts`,
`build_extracted_dag_from_choices`, `choices_to_arena`, `cost_of_choices`,
`repair_choices_well_founded` — is shared machinery below the seam, not
per-impl policy, and stays exactly where it is.

### Why generic in `C` rather than `&dyn CostFunction`

`CostFunction::node_cost` is called once per *(class, node)* in each DP
pass — millions of times on a 100k-class graph. A `&dyn CostFunction`
parameter would pay a virtual call there, which is precisely the shape
CLAUDE.md's fold rule is against: dispatch repeated at every use.

Making the **trait** generic in `C` recovers both properties at once.
`Optimizer` stores `Box<dyn Extractor<CostModel>>`: the extractor is chosen
once, dispatched once per extraction, and inside the impl `C` is the
concrete `CostModel`, so every `node_cost` is a direct call. A caller with
its own cost model uses the generic impls directly, unboxed. The one
dynamic dispatch per extraction is the "once, at construction" case the
fold rule explicitly permits.

### `CostFunction` is threaded, not duplicated

The existing node-cost trait is the right trait: it is exactly "price one
`ENode` in a parent context", which is what both DP passes ask of it, and
it is already what every extraction function is generic over. It is
threaded through `Extractor` unchanged. **No second cost trait is added.**
(Its `cost_by_kind` default that `panic!`s is a wart, but it is out of
scope here and removing it is a separate subtraction.)

### `Reranker` does not survive as a parameter of `Greedy`

It becomes a *different extractor*. Today `Optimizer::run` has a two-arm
`match self.rerank` where the `Some` arm builds an `IncrementalExtractor`,
costs its choices, and stamps `ExtractionReport::external()`. That arm is
already an extractor — an unnamed one, inlined into the caller.

So: `Reranked<R>` is an `Extractor` impl wrapping a `Reranker` and a
`top_k`, and `Optimizer::rerank(Some(r))` becomes
`Optimizer::extractor(Reranked::new(r, RERANK_TOP_K))`. The `rerank` field
and the `match` in `run` are **deleted**. This is subtraction: one dispatch
point where there were two, and the swap-refinement search keeps the seam
it was kept for.

`Reranker` is not a parameter of `Greedy` because it optimizes a different
objective — a `f64` score on its own scale, accepted only on strict
improvement — and `Greedy` minimizes `dag_cost`. Folding a second objective
into `Greedy` as an `Option` would keep both cases live in every call.

## `Greedy` — today's extractor, byte-identical

`Greedy` is the existing chain, **moved**, not copied:

```text
tree_dp_pass → repair → cost
shared_dag_dp_pass → repair → cost      (or abandon at the byte budget)
min by ChoiceCost::dag, ties to tree
assemble
```

`impl<C: CostFunction> Extractor<C> for Greedy` is a one-line delegation to
`extract_dag_scoped`, which stays as the free function every existing
caller (`runtime.rs`, `extraction_gap.rs`, the tests) already names. That
keeps the diff a seam rather than a rewrite, and makes the identity claim
structural: `Greedy` *is* `extract_dag_scoped`.

The identity claim that must be **tested**, not asserted:

1. In-crate: on a corpus of saturated graphs, `Greedy::extract(..)` and
   `extract_dag_scoped(..)` return equal `choices`, `dag_cost`,
   `total_cost` and `report`.
2. End-to-end: `egraph_off_on run --no-clock --no-probe` with
   `--beam 0` (the `Greedy` default) reproduces production's `code_fnv`
   on every kernel — the same machine-code column #1231's filter identity
   pin uses.

## `Beam` — width where the DP has one seat

The sharing-aware DP (`shared_dag_dp_pass`) walks classes in post-order and
keeps, per class, **one** choice: the node minimizing
`own + Σ own over the union of its children's reach sets`. The union is
what makes it exact — a shared subterm entering from two children is
counted once — and the single seat is what makes it greedy. A class settles
before any parent that might have preferred a different settlement.

`Beam` keeps **k** seats.

**State.** A beam state at class `c` is a *partial choice map*: the set of
`(class, node_idx, own)` triples for exactly the classes in the chosen
sub-DAG at `c`, held sorted by class. Its cost is the sum of the `own`
column — i.e. exactly `dag_cost` of that sub-DAG. Nothing is estimated;
the set-carrying representation is what makes it exact, and it is the same
representation `shared_dag_dp_pass` already uses (`Reach`), with the node
index and own-cost carried alongside the member instead of read out of a
side table.

**Expansion.** For class `c` and each of its nodes `n`:

- leaf → one state `{(c, n, own)}`.
- op with children `c1..cm` → merge one state from each child's beam.
  Combinations are built **incrementally**: merge `beam(c1)`'s states with
  `beam(c2)`'s (k² pairs), prune to k, merge with `beam(c3)`, and so on —
  `O(m · k²)` merges per node rather than `k^m`.
- A merge of two sorted maps is linear. A **conflict** — the two maps
  assign different nodes to the same class — makes the pair inadmissible
  and it is dropped. This is the price of exactness: a state must name one
  term, and two disagreeing halves do not.

**Non-emptiness is structural, not lucky.** Rank 0 of every class's beam is
seeded with the DP's own choice, and the DP assigns exactly one node per
class globally, so merging the rank-0 states of any set of children never
conflicts. Every node therefore has at least one admissible state. This is
also what makes `k = 1` reproduce the DP.

**Pruning.** Keep the k lowest-cost states per class, tie-broken
canonically: `(cost, node_idx, then the state's (class, node) pairs
lexicographically)`. Not insertion order — insertion order would make the
result depend on `egraph.nodes()`'s layout, and a kernel would build
differently on two machines, which is the thing the budget-determinism
plan exists to forbid.

**The result.** `Beam` returns `min(tree arm, DP shared arm, beam shared
arm)` by `ChoiceCost::dag`, ties to the earlier arm — `Greedy`'s fold with
one more candidate. Three arms rather than two, because the beam's own
comparison happens *before* `repair_choices_well_founded` and the repair can
rewrite picks: a state the beam preferred pre-repair can repair to a term
dearer than the DP's. That is not hypothetical — at `k = 64` on the 95
DejaVu glyphs, six came back dearer than `Greedy` before this arm existed.
Folding the DP's own repaired term back in makes "`Beam` is never dearer than
`Greedy`" structural, and keeps `Beam { k: 1 }` byte-identical to `Greedy`, so
width is the only variable between them.

Ties go to the anchor, so a width only ever changes the returned term by being
**strictly cheaper** — which is what makes a `code_fnv` difference between an
arm and `Greedy` readable as a win rather than a coin flip.

**Objective reporting.** `Beam` reports `Shared` / `TreeCheaper` /
`TreeOnly` exactly as `Greedy` does. It does **not** report `External`:
it minimizes the same `dag_cost` over the same choice-map space, it just
searches more of it.

**Cost.** Time `O(Σ_c |nodes(c)| · m · k² · |state|)`. Memory
`O(k · Σ_c |reach(c)| · 12 bytes)` — the same sum `shared_dag_dp_pass`
bounds with `SHARED_DAG_PASS_BYTE_BUDGET`, times k, times 3 (class, node,
own) rather than 1. `Beam` carries the same byte budget and, on crossing
it, returns the tree arm as `TreeOnly` — the identical loud variant, never
a silent partial answer.

## What production does

Nothing changes. `Optimizer::production()` uses `Greedy`. `Beam` ships as a
second `impl` reachable through `Optimizer::extractor(..)` and through
`egraph_off_on run --beam K`. The default flips only if the measurement
says it should, and only on JP's word.

## The question the measurement answers

Not "is beam better on average". The sweep already produced a specific
falsifiable claim:

> Does `Beam` at a 50,000-class cap reach a term at or below `Greedy`'s
> cost at a 5,000-class cap?

If yes, the cap regression is a search-width failure and more budget stops
hurting once the extractor can use it. If no, width is not the axis and
the next move is the witness table's per-class evidence
(`origin/claude/extraction-witnesses`) about *which* class, stage and rule
lost the term.
