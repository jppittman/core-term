# The tier is the class count

Production picks a saturation tier (blitz, rapid, classical) from an input's size and sizes the
classical cap from the number of e-classes the input inserts to. Until now the tier was keyed on
`reachable_count(arena, root)`, the number of distinct arena nodes reachable from the root, and the
cap on the inserted class count. This note records why one number now does both, and the
measurement that says doing so moved no kernel.

## The problem with the arena count

The saturation cache is keyed on the canonical, hash-consed serialization of the term
(`pixelflow_ir::key::Canonical`) plus the optimizer configuration. The tier was chosen from the
*saturating caller's arena*, which is not a function of that key: two arenas that serialize to the
same bytes, but are spelled with different amounts of sharing, would share a cache slot and could
have resolved different tiers. The first caller to reach the slot decided the budget for everyone.
The class count after insertion is a function of what was inserted, so a budget read from it is a
function of the key (plus the vocabulary and the admitted units).

One caveat, stated once: the canonical key is order-free but the e-graph memo is not, so
`(X + Y) * (Y + X)` keys one subterm and inserts two classes. The class count is therefore a
function of the term as inserted, not strictly of the key. It is still strictly better than the
arena count of whichever arena arrived first.

## The measurement

`node_count` (`reachable_count`) and `inserted_classes` are both on every
`saturation-telemetry` record. Taken at `54b07f5a`, before any change:

| source | records |
|---|---|
| macro tier: `cargo check --workspace --all-targets --features pixelflow-compiler/saturation-telemetry` (every `kernel!` in the workspace expands) | 143 |
| runtime tier: `byte_probe` (21 lines; one is a decline record with no sizes) | 20 |
| runtime tier: `cargo test --release -p pixelflow-graphics --features pixelflow-core/saturation-telemetry` (the font programs, up to 27,018 nodes, and the scene kernels) | 645 |
| **total** | **808** |

Band mismatches (bands are at most 10, 11 to 50, 51 and up): **0**. Records where the two counts
differ at all: **0**. `ExprArena::push_*` interns, so an arena built through the public API is
already hash-consed and its reachable count is its canonical node count; the telemetry shows the
e-graph's insertion agreeing with it on every shipped kernel.

So keying the tier on the class count is byte-identical on everything shipped, by measurement, and
needs no accessor on `Canonical`.

Not measured: the research corpus the `phase3_at_budget_eval` and `corpus_gaps` bins read
(`production_probe` inserts under `Vocabulary::Templates`; `corpus_gaps` inserts lowered research
kernels). Those bins formerly chose the tier from `arena.len()` or `reachable_count`; they now
choose it from the class count after insertion, so their tier columns may differ from runs made
before this change wherever an arena is not interned or insertion folds.

Also not done: asserting equality with `Canonical`'s own node count. `Canonical` exposes none (the
count is `Walk::starts.len()`, private), and adding an accessor is a public `pixelflow-ir` API
addition, which is JP's call. The comparison made is against the quantity the budget now uses.

## What changed

- `InputSize` is one field, `classes`. `Optimizer::run` and `saturate_term` no longer take a node
  count; they read the class count off the graph they are handed.
- `config_for_node_count` keeps its meaning for offline callers that have only an arena size: the
  tier for that size, with the classical cap at its floor.
- Telemetry drops the `node_count` field; `inserted_classes` is the number that picks the tier.
  The `max_classes` key and the `class_cap` stop string are unchanged, because they are an external
  schema (`pixelflow-core/tests/units_past_the_class_limit.rs` reads the former).
- `production_limits_key_on_both_sizes` became `production_limits_key_on_the_inserted_class_count`.
  Every assertion survives with the single size; the one that pinned "40 nodes, 10,000 classes is
  rapid" pinned the old key, and its replacement says a 10,000-class input is classical.
  `the_tier_is_read_off_the_graph` pins the new key against a graph built directly.

## Reproducing

The telemetry record no longer carries `node_count`, so the probe above needs a build from before
this change (`54b07f5a`), where both fields are on every `saturation-telemetry` record:

```text
PIXELFLOW_SATURATION_TELEMETRY=out.jsonl \
  cargo check --workspace --all-targets --features pixelflow-compiler/saturation-telemetry
PIXELFLOW_SATURATION_TELEMETRY=out.jsonl \
  cargo run --release -p pixelflow-codegen --features saturation-telemetry --example byte_probe
PIXELFLOW_SATURATION_TELEMETRY=out.jsonl \
  cargo test --release -p pixelflow-graphics --features pixelflow-core/saturation-telemetry
```

then compare `node_count` with `inserted_classes` per record. What stays checkable after the change
is the premise itself: `a_pushed_arena_inserts_to_as_many_classes_as_it_reaches`
(`pixelflow-search/tests/ir_insert.rs`) fails if insertion stops hash-consing exactly as
`ExprArena::push_*` does.
