# Test quality control follow-up — 2026-08-26

Scope: scheduled continuation of
`docs/bugs/2026-08-15-test-quality-audit-followup.md`. `main` had not moved
on any of that pass's "recommended next steps" targets since `ce2df0e8`
(the intervening `9576ee1f` was an unrelated refactor), so this pass picks
up backlog item 3: `pixelflow-graphics/src/spatial_bsp.rs`, open since the
2026-07-20 audit first flagged it and re-confirmed still-open by every
intervening pass.

## `pixelflow-graphics/src/spatial_bsp.rs` — STYLE.md "test public API" violation

~22 tests reached directly into the private `SpatialBSP::interiors` field
(`bsp.interiors[0]`, `bsp.interiors.iter()`). `InteriorNode`'s own fields
(`axis`, `threshold`, `left`, `right`) were already public — only the
containing field was private, with no accessor.

### Fixed

- Added `SpatialBSP::interiors(&self) -> &[InteriorNode]`, a minimal public
  accessor returning the interior-node slice (mirrors the existing
  `interior_count()`/`leaf_count()`). Rewrote every test that read
  `bsp.interiors[...]`/`.iter()` to go through it instead. No behavioral
  change; same assertions, same access pattern otherwise.
- Found and fixed a real bug while doing this mechanical rewrite: one test
  had `matches!(root.left, NodeRef::Leaf(_));` as a bare statement — the
  `bool` it produces was discarded instead of asserted, so the check was a
  silent no-op. Wrapped both occurrences in `assert!()`.
- Renamed 9 test names that were bare noun phrases (violating STYLE.md's
  "it should" rule): `single_leaf`, `two_leaves`, `empty_bsp`,
  `four_leaves_grid`, `many_items_stress_test`, `stack_of_wide_strips`,
  `node_ref_types_are_distinct`, `binary_tree_property_interior_count`,
  `binary_tree_exact_interior_count`.

`cargo test -p pixelflow-graphics --lib spatial_bsp::`: 56/56 passed
(unchanged count at this point — the bug fix above and the accessor
rewrite didn't add tests, just corrected two of them).

## Mutation testing: the same file, now that it's STYLE-clean

`cargo-mutants` v27.1.0 (freshly installed — not present in this
environment, consistent with every prior pass).

**First sweep** (`cargo mutants -p pixelflow-graphics --file
.../spatial_bsp.rs -j 4`): **74 mutants, 15 missed, 55 caught, 4
unviable.** All 15 were in the split-axis heuristic (`build_tree`) and the
SIMD blend path (`traverse`). The existing tests checked loose bounds
("threshold is in `[25, 75]`", "axis is `X`") that happened to hold under
several different wrong formulas — sufficient to pin *approximate*
behavior, not exact arithmetic.

### Fixed: 8 new tests, each built to disagree with a specific mutant

- **cx/cy center formula** (`* 0.5` → `/`/`+`, `+` → `*`,
  `pixelflow-graphics/src/spatial_bsp.rs:149`): two items whose Y
  center-spread (15) exceeds X (10) only under the correct formula — each
  mutant inflates the X spread (4× for `/`, 2× for `+`) past 15 and flips
  the axis to X.
- **`extent_x > extent_y` → `==`/`>=`** (line 164): two scenarios — one
  with a strict, non-tied inequality where the bbox-width fallback would
  independently disagree (catches `==`), one with an exact tie (bounds in
  eighths, exactly representable in `f32`, for a bit-exact tie rather than
  a merely-close one) where `>=` fires before the fallback ever runs
  (catches `>=`).
- **bbox width/height tie-break** (`-` → `+`/`/`, lines 161–162): tied
  centers with a bbox min close to zero (exposes the `/` mutant via a
  blown-up ratio) and, separately, shifted far from zero (exposes the `+`
  mutant via a blown-up sum) — each flips the width-vs-height comparison
  outcome that a plain subtraction wouldn't.
- **median-threshold sort comparator** (`+` → `*`/`-`, `/` → `%`, lines
  175/185/186): three items whose sum-order and product-order differ,
  fed to `from_positioned` in *descending* sum-order rather than
  already-sorted. This mattered more than expected — see methodology note
  below.
- **`mid_idx - 1` → `mid_idx / 1`** (line 190): the same three-item
  fixture; the mutant reads `items[mid_idx]` for both sides of the
  average instead of `items[mid_idx - 1]` and `items[mid_idx]`, which this
  asymmetric set turns into a different number (0.5 vs. the correct -5.5).
- **`!mask.any()` → `mask.any()`** (line 290, `traverse`): SIMD lanes
  straddling the split threshold via `Field::sequential`, asserting the
  per-lane color on both sides of the boundary rather than "no panic" —
  the mutant makes any mixed mask take the right-only early return,
  silently dropping the left-child result for the lanes that need it.

**Re-run after the first 8 tests: 74 mutants, 5 missed** (down from 15).
All 5 were sort-comparator mutants that survived despite direct targeting.

### Methodology finding: an already-sorted fixture under-exercises `sort_by`

The sort-comparator mutants target `ca`/`cb` inside the closure passed to
`items.sort_by`, e.g. `let ca = (a.bounds.0 + a.bounds.2) / 2.0;`. My first
attempt fed the three items in ascending sum-order — already correctly
sorted. Rust's small-slice insertion sort then needs zero swaps, and the
specific sequence of comparator calls it makes never puts the item whose
`a`-role formula is mutated into a comparison where the corruption changes
the outcome (an item that's never re-examined after its initial "no swap
needed" comparison never has its own `a`-role value meaningfully used
again). Re-running the fixture with items given in **descending** sum-order
forces every element through a real comparison, and closed 4 of the 5
remaining gaps — same expected threshold value, since the algorithm sorts
internally and shouldn't depend on input order (confirmed empirically:
tests still pass with the same `-5.5`).

**Second re-run: 74 mutants, 1 missed** (`spatial_bsp.rs:185:52`, `/ 2.0`
→ `* 2.0` inside the Y sort comparator). Diagnosed as a genuine equivalent
mutant: dividing and multiplying by a positive constant are both monotonic
scalings, so they can never change a `partial_cmp`-based comparator's
ordering for any input — no test can distinguish them. Documented with a
comment at the site (matching the precedent in
`pixelflow-compiler/src/codegen/util.rs`) rather than chased further.

**Final state: 74 mutants, 73 caught, 1 documented equivalent, 0 real
gaps.**

## Verified

- `cargo test -p pixelflow-graphics --lib spatial_bsp::`: 64 passed, 0
  failed.
- `cargo test -p pixelflow-graphics` (all targets incl. doctests): 160 lib
  tests + all integration/doctest targets passed, 0 failed.
- `cargo test --workspace --lib`: passed (exit 0), 0 failed.
- `cargo clippy -p pixelflow-graphics --lib --tests`: clean.
- `cargo fmt -p pixelflow-graphics -- --check`: clean.
- `cargo mutants -p pixelflow-graphics --file
  pixelflow-graphics/src/spatial_bsp.rs`: 74 mutants, 73 caught, 1
  documented equivalent, 0 missed.

## Recommended next steps (not done here)

Backlog carried forward from 2026-08-15, minus the item closed above:

1. `pixelflow-search/src/egraph/cost.rs` — still open per the 08-08
   audit: a partial mutants run found one real gap (`CostModel::zero()`,
   already fixed) before its own slow `--lib` baseline (~110s) timed out
   the pass. Still needs either a narrower test filter or a longer time
   budget.
2. `pixelflow-codegen/src/emit/*` (~1,400 lines) — flagged since 08-08 as
   never mutation-tested under its post-crate-split location. Still true.
3. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` — re-investigated
   this pass while scoping work (not independently re-verified with a
   fresh mutants run, just re-read against the audit trail): both halves
   of the original 2026-07-20 finding are already resolved — the
   mutation-weakness half by `9deb9852` ("Add mutation-targeted tests for
   actor-scheduler", exact-value jitter/timeout assertions), and the
   private-API-violation half by an explicit, documented 2026-07-24
   Flexibility-clause judgment call (the precise Ok/Timeout boundary cases
   require observing an *unstarted* backoff attempt, unreachable through
   the public surface without racing a real sleep against a timeout).
   Recommend removing this from the backlog; nothing left to do without
   re-litigating a decision already made and documented.
4. `pixelflow-core/src/backend/x86.rs`'s `F32x8`/`F32x16`/`U32x8`/
   `U32x16`/`Mask8`/`Mask16` (AVX2/AVX-512) impls, and `arm.rs`'s NEON
   impls — never tested at the unit level at all under a build that
   actually activates those ISA levels (`xtask isa-matrix`).
