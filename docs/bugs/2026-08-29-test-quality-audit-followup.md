# Test quality control follow-up — 2026-08-29

Scope: scheduled continuation of
`docs/bugs/2026-08-26-test-quality-audit-followup.md`. `main` had not moved
on any of that pass's backlog since `b4cc51f`. This pass picks up backlog
item 2: `pixelflow-codegen/src/emit/*` (~1,400 lines total across
`mod.rs`/`x86_64.rs`/`aarch64.rs`/`avx2.rs`/`avx512.rs`/`regalloc.rs`/
`executable.rs`/`coverage.rs`), flagged since 08-08 as never mutation-tested
under its post-crate-split location. Of those, `regalloc.rs` (730 lines,
graph-coloring + linear-scan register allocators) was the smallest
self-contained file with real algorithmic logic and its own existing test
module, so this pass scoped to it alone; the rest of `emit/` remains open
(see Recommended next steps).

## `pixelflow-codegen/src/emit/regalloc.rs`

The file's 6 pre-existing tests were bare noun phrases (`empty_graph`,
`no_interference`, `chain_interference`, `clique_needs_more_colors`,
`precolored`, `spilling`), violating STYLE.md's "it should" rule, and only
exercised `InterferenceGraph`/`color_graph` — `build_interference_graph` and
`linear_scan` (the allocator actually wired into the compiler; `color_graph`
is reachable only from this file's own tests) had zero coverage.

### Renamed the 6 pre-existing tests to STYLE.md's "it should" form

E.g. `empty_graph` → `color_graph_should_assign_and_spill_nothing_for_an_empty_graph`,
`spilling` → `color_graph_should_spill_the_excess_values_of_a_clique_too_large_for_the_register_budget`.
All test only the public API (`InterferenceGraph`, `color_graph`,
`build_interference_graph`, `linear_scan` — `simplicial_elimination_order`
is private and is exercised indirectly through `color_graph`, per STYLE.md's
guidance that a child `mod tests` may read a private field/call a private fn
directly without that counting as a public-API violation, but here it's not
even needed since `color_graph` covers it).

### Found and fixed two real bugs while writing tests for the untested functions

1. **`build_interference_graph` never killed a value's liveness.** Once any
   value became live (via being used, or via its own "just defined" step),
   it stayed in `live_list` for the rest of the backward walk and interfered
   with *every* value scheduled earlier — regardless of whether their live
   ranges ever actually overlapped. For any 2+-value schedule this made
   every pair of values interfere unconditionally, which would have made
   `color_graph`'s "optimal for chordal graphs" contract vacuous (every
   schedule looks like a clique). Fixed by killing a value's liveness right
   after using it to build edges for its own definition, matching standard
   backward liveness analysis (kill def, gen uses).
2. **`color_graph`'s elimination order was reversed.** MCS's visitation
   order numbers vertices n, n-1, ..., 1 as visited; for a chordal graph
   this is a valid ordering only when a vertex's *higher*-numbered (i.e.
   *earlier*-visited) neighbors form a clique — coloring must therefore
   proceed in that same visitation order, not its reverse. The existing
   `order.reverse()` colored each vertex *before* its constraining neighbors
   instead of after them. Confirmed with a brute-force-found counterexample
   (two triangles joined by a single bridge edge, specific `ValueId`
   labeling): the reversed order needed a 4th color where 3 always suffice
   for this chordal graph. Since neither `color_graph` nor
   `build_interference_graph` has any caller yet (`linear_scan` is what's
   actually wired into the compiler), this had no runtime effect, but the
   public contract was broken for any future caller.

### Added coverage for the two previously-untested public functions

`build_interference_graph`: connects two operands simultaneously live at a
shared use, leaves unrelated single-use chains unconnected (the direct
regression test for bug 1 above), sizes its live-set correctly against the
highest `ValueId` (ids need not match schedule position), and doesn't panic
when caller-supplied `uses_of` names a `ValueId` outside the schedule.

`linear_scan`: empty schedule, sufficient-register assignment, rematerializing
a constant instead of spilling it, keeping a precolored value pinned, freeing
a dead value's register for immediate reuse, and five tests pinning the
eviction-priority logic (constant-over-non-constant preference, tie-breaking
toward the first-seen candidate on both the constant and Belady paths, and
the tied-with-the-incoming-value case) — this last group is what mutation
testing showed the pre-existing tests couldn't distinguish.

## Mutation testing

`cargo-mutants` v27.1.0 (freshly installed, consistent with every prior
pass — not present in this environment beforehand).

**First sweep** (post-STYLE-cleanup, pre-new-tests, i.e. just the two bug
fixes applied): **102 mutants, 42 missed, 54 caught, 6 unviable.** Nearly
all of `InterferenceGraph`'s direct accessors (`degree`, `dedup_edges`,
`neighbors`/`is_precolored`/`precolor_of` bounds, `is_empty`) had zero direct
tests; `build_interference_graph` and `linear_scan`'s eviction-priority
branches were similarly wide open.

**Round 2** (adding accessor tests, boundary tests, and the 5 eviction-priority
tests): **9 missed** (down from 42). Two boundary tests initially used a
far-out-of-range `ValueId` (e.g. `ValueId(5)` against a capacity-1 graph)
which distinguishes a guard's presence but not `<` from `<=` — fixed by using
the value exactly *at* the capacity boundary instead, which is the only input
that actually differs between the two operators.

**Round 3** (after the boundary fixes): **6 missed**, all in
`simplicial_elimination_order` (2, one of them the real weight-tracking gap
below), `build_interference_graph`'s kill-guard (2), `ensure_capacity` (1),
and `linear_scan`'s running-max update (1).

**Round 4**: closed the real gap — a search-found chordal graph (built from
a random valid elimination ordering, so chordality is guaranteed by
construction) where the real cardinality-weight MCS order needs 3 colors but
the order MCS degenerates to if a remaining vertex's weight is never
incremented (every vertex permanently tied at weight 0, ties broken by
descending `ValueId` alone) needs 5. The bridge-graph regression test added
for the reversed-order bug happens to produce the *same* order whether or
not weights update for its specific graph, so it didn't cover this — a
second, independent counterexample was needed.

**Final: 102 mutants, 91 caught, 6 unviable, 5 missed — all 5 documented as
equivalent mutants**, each verified by manually re-applying the exact
mutation with `sed` and re-running the affected test to confirm it doesn't
(and provably can't) observe a difference, with a proof sketch left as a
comment at each site (matching the precedent in
`pixelflow-compiler/src/codegen/util.rs` and the 08-26 audit's
`spatial_bsp.rs` finding):

- `ensure_capacity`'s `>`/`>=` boundary — at `idx == capacity`, the "grow"
  body just resizes every `Vec` to its current length and reassigns
  `capacity` to the value it already holds; both are no-ops.
- `simplicial_elimination_order`'s stale-heap-entry check `&&`/`||` —
  `in_remaining[vi]` and `w == weight[vi]` are provably always equal at that
  point: a vertex's pushed weights strictly increase, so the max-heap always
  surfaces a remaining vertex's single freshest entry (matching its current
  weight) before any staler one, and once removed no entry can ever again
  equal the now-frozen weight.
- `build_interference_graph`'s kill-guard `vi < live_capacity` (both the
  `&&`/`||` and `<`/`<=` variants) — `vi` is `v.0` for a `v` drawn directly
  from `schedule`, the exact same source `live_capacity`'s max is computed
  from, so `vi <= max_vid < live_capacity` always holds; no schedule can
  violate it. (The mirroring guard on caller-supplied `uses_of`'s `ui`, which
  has no such invariant, is real and is covered.)
- `linear_scan`'s running-max update `>`/`>=` — assigning a value to itself
  on a tie is a no-op, so both operators produce an identical final
  `max_reg_used` for every input.

## Verified

- `cargo test -p pixelflow-codegen --lib`: 103 passed, 0 failed (31 in
  `regalloc::tests`, up from 6).
- `cargo test --workspace --lib`: 1,699 passed, 0 failed, 5 ignored, across
  all 12 crates (final re-run against the fully finished state of this pass).
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/regalloc.rs`:
  102 mutants, 91 caught, 6 unviable, 5 documented equivalents, 0 real gaps.

## Recommended next steps (not done here)

Backlog carried forward from 2026-08-26, minus the item closed above:

1. `pixelflow-search/src/egraph/cost.rs` — per 08-26, likely already closed
   by `#1027`; an in-flight PR (#1049, unmerged as of this pass) found and
   corrected the same stale-backlog situation for a different file
   (`graph.rs`) in the same crate. Worth confirming once that PR lands.
2. `pixelflow-codegen/src/emit/*` — `regalloc.rs` (730 lines) is now closed
   above; `mod.rs` (5,779 lines), `x86_64.rs` (863), `aarch64.rs` (2,168),
   `avx2.rs` (710), `avx512.rs` (866), and `executable.rs` (550) remain
   never mutation-tested under this location. `mod.rs` in particular is
   large enough to need its own scoped pass (likely per-function-group, not
   whole-file).
3. `pixelflow-core/src/backend/x86.rs`'s `F32x8`/`F32x16`/`U32x8`/
   `U32x16`/`Mask8`/`Mask16` (AVX2/AVX-512) impls, and `arm.rs`'s NEON
   impls — never tested at the unit level under a build that actually
   activates those ISA levels (`xtask isa-matrix`). Open since 08-08,
   untouched by every intervening pass; likely needs the ISA-matrix build
   infrastructure investigated rather than more test-writing per se.
