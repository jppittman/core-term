# Test quality control follow-up — 2026-08-29

Scope: scheduled continuation of
`docs/bugs/2026-08-26-test-quality-audit-followup.md`, backlog item 2 —
`pixelflow-codegen/src/emit/*`, flagged since 08-08 as never mutation-tested
under its post-crate-split location. This pass scoped to `regalloc.rs` (then
730 lines, a graph-coloring allocator and a linear-scan one) as the smallest
self-contained file with real algorithmic logic and its own test module.

**Most of what this pass produced no longer applies, and the section below
records why rather than deleting it.** While the branch was in flight, `main`
merged #1055 ("give register allocation a trait and one platform description")
and #1068 ("declare the registers backends take for themselves"), which
rewrote `regalloc.rs` end to end and **deleted the graph-coloring allocator
outright** — `InterferenceGraph`, `build_interference_graph`, `color_graph`
and `simplicial_elimination_order` do not exist anywhere in the workspace any
more. Roughly two thirds of this pass's tests, and both of the bugs it found,
were against those functions. That work is superseded; it is not salvageable,
because there is no longer any code for it to be about.

## What survived, and is in this branch

Four tests against `LinearScan`'s eviction rule — the one part of the old
file's logic that #1055 carried forward intact, and the part `main`'s own
rewritten test module still does not separate:

- `a_constant_is_evicted_before_a_non_constant_needed_farther_out` — the
  constant-beats-Belady preference. `main`'s
  `belady_evicts_the_value_used_farthest_out` has no constant in play and
  `constants_are_rematerialized_rather_than_spilled` has nothing *but*
  constants, so neither scenario can tell the two rules apart.
- `tied_constant_eviction_candidates_break_toward_the_first_seen` and
  `tied_non_constant_eviction_candidates_break_toward_the_first_seen` — the
  strict `>` in each candidate scan. Relaxed to `>=`, the allocator evicts the
  later occupant instead: the same spill count, a different program, and both
  runs of the collapse driver still agree with each other, so nothing
  downstream notices.
- `on_a_tied_last_use_the_arriving_value_is_evicted_not_the_resident` — the
  `>=` in `evict_new`. Tightened to `>`, the resident is evicted for a value
  with exactly the same next use, paying a store for nothing.

Each was verified to kill its mutant by re-applying the exact mutation by hand
and confirming the test fails, which is the check that matters here: the four
sit on branches a value-comparison test can otherwise reach without
distinguishing.

`cargo test -p pixelflow-codegen --lib emit::regalloc`: 24 passed, 0 failed.

## What the pass found in the deleted allocator (historical)

Kept because it is the record of two genuine algorithmic breaks, and because
the second is the sort of thing a future graph-coloring attempt would
reintroduce:

1. **`build_interference_graph` never killed a value's liveness.** Once a
   value became live it stayed in `live_list` for the rest of the backward
   walk and interfered with every value scheduled earlier, whether or not
   their live ranges overlapped — so every schedule looked like a clique and
   `color_graph`'s "optimal for chordal graphs" contract was vacuous.
2. **`color_graph`'s elimination order was reversed.** MCS numbers vertices
   n..1 as visited, and coloring must proceed in that visitation order, not
   its reverse; the `order.reverse()` colored each vertex *before* its
   constraining neighbors. A brute-force search found a chordal graph (two
   triangles joined by a bridge) needing a 4th color where 3 suffice.

Neither had a caller — `LinearScan` was always what the compiler used — which
is also why #1055 was able to delete the allocator rather than fix it. Its
doc comment on `main` now records the reason: the graph-coloring allocator
never learned to rematerialize constants, and on a 4–10 register pool that is
the difference that decides the generated code.

A third finding from this pass is still open and does *not* apply to the
deleted code: greedy coloring in MCS order is optimal only for
*unconstrained* chordal coloring, and precolored vertices break the
no-backtracking guarantee (raised by review on this PR, confirmed by
hand-simulation on a `1—0—2` path with one pinned register). It is recorded
here only so a future graph-coloring attempt does not rediscover it.

## Recommended next steps (not done here)

Backlog carried forward from 2026-08-26:

1. `pixelflow-codegen/src/emit/*` — `x86_64.rs` is being closed separately in
   #1054. `mod.rs`, `aarch64.rs`, `avx2.rs`, `avx512.rs` and `executable.rs`
   remain never mutation-tested under this location, and all of them were
   reshaped by #1055/#1068, so a pass should be scoped against the current
   file rather than a report from before that refactor.
2. `regalloc.rs` itself is worth a fresh sweep against `main`'s rewritten
   version: the four tests here were ported from a mutation report on code
   that no longer exists, so they close the eviction branches and nothing
   else. `RegSet`, `RegisterFile::checked` and `Allocation` are all new
   surface.
3. `pixelflow-core/src/backend/x86.rs`'s AVX2/AVX-512 impls — closed in
   #1073, which found this sandbox now has the hardware.

## Methodology note

This branch is the second in this series (after #1049) to be invalidated by
`main` moving underneath it, and the first to be invalidated by the code under
test being *deleted*. A mutation report names functions; when the target file
is under active refactor, the report expires with the function names. Cheap
guard for the next pass: re-check `git log origin/main -- <file>` immediately
before writing tests against a report, not only before pushing.
