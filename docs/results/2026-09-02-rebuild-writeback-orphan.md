# `rebuild_budgeted` orphaned e-nodes when the worklist item lost its own merge

A bug, its fix, and its blast radius. Companion to
`docs/results/2026-09-02-missing-congruence.md`, which measured a *different*
under-merging gap (no e-node parent list) on the same corpus; this one is a
plain write-back-to-the-wrong-slot defect.

## The bug

`EGraph::rebuild_budgeted` drains a worklist class's e-nodes with
`mem::take`, canonicalizes each one, and when the memo already maps the
canonical form to a different class `existing`, calls `self.union(id,
existing)`. `union` keeps `min(a, b)` as the surviving parent. The write-back
at the bottom of the loop was unconditionally:

```rust
self.classes[id.index()].nodes.extend(new_nodes);
self.classes[id.index()].tags.extend(new_tags);
```

When `existing.0 < id.0`, `existing` survives and `id` is merged away — so
that write lands in a slot `find` no longer routes to. `EGraph::nodes(id)`
and `EGraph::tags(id)` canonicalize before indexing, so from that moment the
nodes are unreachable from every public entry point: invisible to rule
matching, to extraction, to `node_for_tag`. They still occupy memory.

The function's comment had noticed only the mirror case — `id` survives, so
`union`'s `extend()` has already appended `existing`'s nodes onto the vector
`mem::take` emptied, and the write-back must extend rather than assign (the
fix in `rebuild_budgeted_does_not_drop_nodes_when_current_class_survives_union`).
The `existing`-survives direction was unhandled.

**Severity: quality, not soundness.** Every class that remains still holds
only provably-equal terms, so nothing miscompiles; what is lost is extraction
alternatives — under-merging, the same category the constant-union refusal
valve in `union` accepts deliberately. But it is lost *silently*, with no
counter and no assertion, which is what this project forbids.

## The fix

```rust
let dest = self.find(id);
self.classes[dest.index()].nodes.extend(new_nodes);
self.classes[dest.index()].tags.extend(new_tags);
```

`find(id)` handles both directions with one write-back (`find(id) == id`
whenever `id` survived), and `extend` rather than assign is still required,
because in that direction `union` has already moved `existing`'s nodes onto
the destination.

Tests:
- `rebuild_budgeted_does_not_orphan_nodes_when_current_class_is_merged_away`
  (`pixelflow-search/src/egraph/graph.rs`) — the mechanism, on a six-class
  hand-built graph. It is the existing `..._when_current_class_survives_union`
  test with the two `Neg` classes created in the opposite order, which is all
  it takes to flip which side `min(a, b)` keeps. It fails on the pre-fix tree
  with `Got: [Op { op: Op(neg), children: [EClassId(1)] }]` — the unique
  marker node `Var(9)` is gone from `nodes(find(nc))`.
- `saturation_strands_no_enodes` (`pixelflow-search/src/runtime.rs`,
  `mod congruence_gap_probe`, `#[ignore]`d) — the invariant on the real
  corpus: after production saturation, no e-node sits in a non-canonical
  class. `union` empties merged-away classes with `mem::take`, so the only
  way to strand one is to write to a slot after it stopped being canonical;
  the count is therefore 0 iff the write-back is right.

## Blast radius

**Method.** `repair_writeback_blast_radius` (`#[ignore]`d, same module) runs
the production regime — `Optimizer::production()`, i.e. the production rule
set, `Budget::Production`, `CostModel::latency_prior`, `Dwrt` lowered and
`Reduce` unrolled first, exactly as `optimize_runtime_arena_uncached` calls
it — over every kernel in the corpus and writes one CSV row per kernel. The
A/B is a **diff of two runs of that same test**, one on the fixed tree and
one on a tree whose write-back is reverted to `id`. The two behaviours
deliberately cannot coexist in one binary: a runtime switch would mean
keeping the bug alive in production code in order to measure it.

**Corpus.** The same 206 real kernels the missing-congruence probe uses — 12
`shader_bench` ShaderToy ports, 1 hand-transcribed psychedelic shader, 3
packed cell-grid geometries at the sizes core-term compiles, 190 glyph arenas
(95 printable ASCII × 2 display densities) — regenerated from the three
`#[ignore]`d dumpers. No synthetics: this is a question about production, and
the real corpus already answers it. 191 of the 206 stop on `ClassCap`.

Cost is `CostModel::latency_prior()` summed over the extracted arena's
reachable op nodes.

### The bug fires on half the corpus

| | pre-fix | fixed |
|---|---|---|
| orphaned e-nodes (in non-canonical classes) | **760** | **0** |
| kernels with ≥1 orphaned e-node | **105 / 206** (51%) | 0 |
| reachable e-nodes, pooled | 839,682 | 840,218 |

By category: cellgrid 3/3, psychedelic 1/1, shader 5/12, glyph 96/190. Worst
single kernel: `shader:julia_set`, 50 orphaned nodes.

Splitting the 760 by whether the surviving class already held a structurally
identical node (canonicalized through `find`): **666 were duplicates, 94 were
alternatives the graph proved and then lost.** That ratio is what the loop's
shape predicts — the node that *triggers* a merge is by construction the memo
key naming `existing`, so it is always already there; only the other nodes
drained alongside it can be genuinely new.

### It changes no emitted kernel

| observable | kernels differing (of 206) | net change |
|---|---|---|
| **extracted cost** (`latency_prior`) | **0** | 0 (pooled 1,036,301 both arms) |
| **extracted node count** | **0** | 0 (pooled 235,982 both arms) |
| saturation stop reason | 0 | — |
| live classes (`find(i) == i`) | 29 | −49 |
| raw `classes.len()` | 64 | −224 |
| reachable e-nodes | 103 | +536 |

The graph shape does move — recovered nodes participate in later congruence,
which merges more (live classes net −49) and allocates less (raw −224) — but
the extraction is bit-identical on every kernel, including all 105 affected
ones. The 94 genuinely-lost alternatives were not on any extracted path under
the latency prior.

Per-kernel rows: `2026-09-02-rebuild-writeback-orphan-fixed.csv` and
`2026-09-02-rebuild-writeback-orphan-prefix.csv` (same schema, keyed by
`name`).

| kernel | orphaned | reachable nodes | live classes | cost |
|---|---|---|---|---|
| `shader:julia_set` | 50 | 4483 → 4533 | 819 → 819 | 716 → 716 |
| `shader:smooth_min_scene` | 21 | 1198 → 1215 | 160 → 159 | 136 → 136 |
| `shader:torus_slice` | 21 | 1270 → 1283 | 166 → 165 | 141 → 141 |
| `shader:domain_warp_fbm` | 20 | 4557 → 4591 | 1244 → 1248 | 444 → 444 |
| `shader:mandelbrot_distance` | 16 | 3662 → 3675 | 415 → 415 | 576 → 576 |
| `psychedelic` | 15 | 4832 → 4845 | 946 → 940 | 766 → 766 |
| `glyph16:U+004C` | 12 | 3347 → 3353 | 420 → 420 | 185 → 185 |
| `glyph32:U+004C` | 12 | 3347 → 3353 | 420 → 420 | 185 → 185 |

## Verdict

Fix it because a silent loss is not allowed to stand, not because it buys
speed. Nothing downstream needs re-tuning: no kernel's emitted code moves, so
the latency-prior table, the saturation budgets, and every measurement keyed
to extracted cost are unaffected.

One published number does shift, and this is the notice so it is not silently
stale: rerunning `missing_congruence_measurement`
(`docs/results/2026-09-02-missing-congruence.md`) on the fixed tree finds
**807** additional closure unions where it found **922** — the fix recovers
some of what that probe's offline closure was picking up. Its five headline
medians are unchanged (0.00% class reduction, 0.000% cost change). That
document's committed `.csv`/`.json` are left as the record of the run they
describe.

## Reproducing

```bash
# 1. Corpus (206 .arena dumps). Needs git-lfs for the production font.
export CORPUS=/tmp/arena-corpus && mkdir -p "$CORPUS"
PIXELFLOW_TELEMETRY_DIR=$CORPUS cargo test -p pixelflow-core --release -- \
  --ignored dump_production_cell_grid_arenas
PIXELFLOW_TELEMETRY_DIR=$CORPUS cargo test -p pixelflow-graphics --release \
  --test production_glyph_arena_dump -- --ignored
PIXELFLOW_TELEMETRY_DIR=$CORPUS cargo test -p pixelflow-pipeline --release \
  --test shader_and_psychedelic_arena_dump -- --ignored

# 2. The invariant, and the per-kernel rows.
PIXELFLOW_CONGRUENCE_ARENA_DIR=$CORPUS cargo test -p pixelflow-search --release \
  --lib -- --ignored saturation_strands_no_enodes
PIXELFLOW_CONGRUENCE_ARENA_DIR=$CORPUS PIXELFLOW_REPAIR_WRITEBACK_OUT=/tmp/fixed.csv \
  cargo test -p pixelflow-search --release --lib -- \
  --ignored repair_writeback_blast_radius --nocapture

# 3. The other arm: change `let dest = self.find(id);` to `let dest = id;` in
#    rebuild_budgeted, rerun step 2's second command to /tmp/prefix.csv, and
#    restore. Diff the two CSVs by `name`.
```
