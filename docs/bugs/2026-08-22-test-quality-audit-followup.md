# Test quality control follow-up — 2026-08-22

Scope: scheduled continuation of
`docs/bugs/2026-08-15-test-quality-audit-followup.md`. `main` had not moved
on any of that pass's "recommended next steps" targets since `ce2df0e8`, so
this pass picks up three of the five open items directly:

1. `pixelflow-search/src/egraph/cost.rs` — flagged since 08-08 as never
   mutation-tested to completion (a partial run had timed out on the
   crate's slow `--lib` baseline).
2. `pixelflow-graphics/src/spatial_bsp.rs` — flagged in three consecutive
   passes (07-26 through 08-15) as a STYLE.md "test public API" violation:
   19 tests reached into the private `bsp.interiors` field. Each prior pass
   called it "a design call" and left it open; this pass makes the call.
3. `pixelflow-core/src/backend/x86.rs`'s *required* `SimdOps` methods —
   flagged as the natural next target once `backend/mod.rs`'s *provided*
   methods closed in 08-15.

`cargo-mutants` v27.1.0 (freshly installed — not present in this
environment, consistent with every prior pass).

## `pixelflow-search/src/egraph/cost.rs`

The file had exactly 4 tests, all pinning `latency_prior_cycles()`'s prices
(the `every_op_is_priceable` module). Everything else — `CostModel::shallow`,
`cost`/`set_cost`/`costs`/`costs_mut`, `depth_cost`, `node_op_cost`,
`save_toml`/`load_toml`, `load_or_default`, `from_map`/`to_map`, and both
`CostFunction` trait methods — had zero direct coverage. Nothing outside the
file exercises them either (`graph.rs` calls `node_op_cost` internally, but
nothing asserts on its result).

Scoped the mutants run to this file with the test filter `--lib cost::` (not
`--lib` alone) to sidestep the crate's ~105s baseline, dominated by
`math::pict_rewrite_tests::pict_rewrite_rules_preserve_semantics`, which has
nothing to do with this file. First sweep: **73 mutants, 11 missed.**

### Fixed: 19 new tests, two new test modules

Added `cost_model_accessors` (accessor/pricing methods, matching the
existing `every_op_is_priceable` module's per-concern-module convention) and
`persistence` (`save_toml`/`load_toml`/`load_or_default`/`from_map`/`to_map`).
All go through `CostModel`'s and `CostFunction`'s public methods; the two
tests that need an `ENode::Op` node go through the crate's existing
`EGraph`/`ENode`/`op_from_kind` public re-exports (`crate::egraph::{EGraph,
ENode}`, `crate::egraph::ops::op_from_kind`) rather than a private
constructor — matching the pattern already used by
`math::pict_rewrite_tests`.

Notable misses caught on the first pass and fixed by choosing better inputs
rather than more tests:

- `node_cost_trait_method_delegates_to_node_op_cost` initially used a
  `Const` leaf, whose cost is 0 — so "replace `node_cost` with the constant
  `0`" coincidentally passed. Switched to an `Add` op node (cost 4).
- `load_toml`'s zero-start semantics (starts from `CostModel::zero()`, not
  the latency prior, so unmentioned ops read 0) is what distinguishes it
  from a mutant that returns `Ok(Default::default())` — a naive round-trip
  test that only checks the keys it wrote wouldn't catch that; added
  `load_toml_leaves_unmentioned_ops_at_zero_rather_than_the_latency_prior`
  to check an omitted key directly.
- `load_or_default`'s `PIXELFLOW_COST_MODEL` env-var override is the one
  path with a `return` on success, so it's the only part of that function
  whose *return value* is testable. The test spawns a child process with
  the variable set rather than mutating this one's environment: a
  file-local `Mutex` cannot satisfy `set_var`'s requirement that no other
  thread touch the environment concurrently, since the harness runs tests
  on parallel threads and `load_or_default` itself reads `HOME`. See the
  note on the parent/child pair further down.
- Added a minimal second `CostFunction` implementor to exercise the trait's
  own default `cost_by_kind` (`panic!("not implemented")`) — `CostModel`
  overrides it, so nothing else in the crate ever calls the default body.

Re-ran the same scoped sweep: **73 mutants, 7 missed, 0 build errors.**

### Remaining 7 misses — not fixed, and why

All seven are outside what a black-box test of `CostModel`'s public
contract can distinguish:

- `depth_cost`'s `>` → `>=` (line 277): at `depth == threshold` the penalty
  multiplier `(depth - threshold)` is `0` either way, so the two branches
  are observationally identical at the only input where they diverge. A
  genuinely equivalent mutant.
- Six mutants in `load_or_default`'s `$HOME`-config and workspace-data
  fallback branches (lines 423, 440): these guards (`e.kind() !=
  NotFound`) gate only whether a diagnostic `eprintln!` fires on a
  *parse* failure (as opposed to a missing file, which is expected and
  silent) — they don't affect the function's return value on any path.
  Killing them would mean asserting on captured stderr, which nothing else
  in this crate's test suite does. Left as a known gap.

### Verified

- `cargo test -p pixelflow-search --lib egraph::cost::`: 22 passed, 0 failed,
  1 ignored. The count breaks down as the 4 pre-existing
  `every_op_is_priceable` tests plus 19 new ones — 12 in
  `cost_model_accessors` and 7 in `persistence`. One of the 19 is the
  `#[ignore]`d `env_var_override_child`, which does run, but in a child
  process spawned by its parent rather than in the ordinary sweep.
- `cargo test -p pixelflow-search` (all targets incl. doctests): passed, 0
  failed.
- `cargo clippy -p pixelflow-search --tests`: clean.
- `cargo fmt -p pixelflow-search -- --check`: clean.
- `cargo mutants -p pixelflow-search --file pixelflow-search/src/egraph/cost.rs -- --lib cost::`:
  73 mutants, 7 missed (all documented above), 0 unviable-classification
  surprises.

## `pixelflow-graphics/src/spatial_bsp.rs`

19 of the file's ~40 tests read `bsp.interiors[idx]` directly — a private
field on `SpatialBSP<L>` — to get at an `InteriorNode`'s `axis`/`threshold`/
`left`/`right`. `InteriorNode`'s fields are already all `pub`; the only
thing standing between these tests and the public API was a way to index
into the array. `interior_count()`/`leaf_count()` already expose the two
array *lengths* this same way, so an indexed accessor is the same idea
applied to element access:

```rust
/// Read-only access to interior node `idx`.
///
/// # Panics
/// Panics if `idx >= self.interior_count()`.
pub fn interior(&self, idx: usize) -> &InteriorNode {
    &self.interiors[idx]
}
```

Rewrote all 19 call sites (`&bsp.interiors[N]` → `bsp.interior(N)`,
`bsp.interiors[N].field` → `bsp.interior(N).field`, and three
`bsp.interiors.iter()`/`.iter().enumerate()` loops → `0..bsp.interior_count()`
with `bsp.interior(i)` inside) — including inside two recursive test-local
helper functions (`collect_leaves`, `verify_partition`/`collect_centers`)
that themselves take `bsp: &SpatialBSP<L>`. `self.interiors` inside the
type's own `impl` block (traversal, `interior_count`) is untouched — that's
the implementation, not a test.

### Descriptive names

While in the file for the above, did a full pass against STYLE.md's "it
should" test: 29 of ~56 test names were either bare noun phrases with no
verb (`two_leaves`, `binary_tree_property_interior_count`,
`many_items_stress_test`, `stack_of_wide_strips`), had a subject/verb
number mismatch (`identical_bounds_creates_valid_tree`), or — one real find
— had a name that overclaimed what the test actually checked:
`square_bounds_splits_on_either_axis` asserts the split axis is
*specifically* `Axis::X` for a square, never varies input to get the other
axis, and its own inline comment says "square splits on X (width >=
height)" — the name promised more than the test verifies. Renamed to
`square_bounds_split_on_the_x_axis`. Same issue in
`linear_chain_creates_unbalanced_tree` and
`alternating_dimensions_creates_balanced_tree`: both only assert leaf/
interior *counts*, which hold for any binary tree regardless of balance —
renamed to `sorted_items_still_produce_the_correct_leaf_and_interior_counts`
and `grid_layout_produces_the_correct_leaf_and_interior_counts` to match
what's actually checked, rather than implying an unverified balance
property.

All 29 renames are name-only; no assertions changed except where noted
above (which changed a name to match existing assertions, not the reverse).

### Verified

- `cargo test -p pixelflow-graphics --lib spatial_bsp::`: 56 passed, 0
  failed (same 56 as before — no tests added, removed, or behaviorally
  changed).
- `cargo test -p pixelflow-graphics` (all targets): passed, 0 failed.
- `cargo clippy -p pixelflow-graphics --tests`: clean.
- `cargo fmt -p pixelflow-graphics -- --check`: clean.

Mutation-testing this file was not attempted this pass — the fix here was
the STYLE.md violation specifically, not a coverage gap (the 19 rewritten
tests already covered the same assertions before and after; only the
access path changed). A mutants sweep of `spatial_bsp.rs` remains open for
a future pass.

## `pixelflow-core/src/backend/x86.rs` — backlog item 5 from 08-15

Picked up the last open item from the 08-15 audit's backlog: the *required*
`SimdOps`/`SimdU32Ops` per-ISA primitives in `x86.rs` (as opposed to the
*provided* expansions the 08-15 pass already closed in `backend/mod.rs`)
had never been swept as a whole file.

### Methodology finding: `cargo mutants` is blind to `#[cfg(target_feature)]` reachability

First sweep (`cargo mutants -p pixelflow-core --file .../x86.rs -j 4`, no
other flags): **229 mutants, 172 missed.** That number is misleading. This
crate's default build (`.cargo/config.toml` sets no `target-cpu`/
`target-feature`; confirmed with `cargo rustc -- --print cfg`, which shows
only `sse2`) never compiles the `#[cfg(target_feature = "avx2")]`- and
`#[cfg(target_feature = "avx512f")]`-gated `F32x8`/`U32x8`/`Mask8` and
`F32x16`/`U32x16`/`Mask16` impls at all — `xtask isa-matrix` is how those
ISA levels actually get built and tested, with matching `-C target-feature`
flags, per CLAUDE.md's "SIMD Backend Selection" section.

`cargo mutants`' `--list`/mutate step parses the source with `syn` and does
not evaluate `#[cfg(...)]` at all, so it happily generates and "tests"
mutants inside code that a plain build strips out entirely. For such a
mutant, the mutated line never makes it into the compiled artifact — the
baseline and mutant binaries are identical — so it trivially "survives" no
matter how well-tested the reachable code is. Of the 229 mutants, **149**
(65%) were inside `F32x8`/`F32x16`/`U32x8`/`U32x16`/`Mask8`/`Mask16` or
their `Avx2`/backend-selector impls — confirmed by grepping the mutant
names and cross-checking against `--print cfg`. These are not real
findings for this build configuration; treating them as "missed coverage"
and writing tests to silence them would just be chasing a tooling
artifact. (Separately, real coverage for those ISA levels is worth
checking under a build that actually activates them — see "not done here"
below — but that's a different, not-yet-attempted piece of work.)

Re-ran scoped to the reachable code only
(`-E "F32x8|F32x16|U32x8|U32x16|Mask8|Mask16|Avx2|Avx512"` to exclude the
unreachable-at-this-build-config mutants): **80 mutants, 1 missed.** All 80
are real: `Mask4`, `F32x4`, and `U32x4` are unconditionally compiled at the
SSE2 baseline, so every mutant here reflects actual reachable,
actually-tested-or-not code.

### Fixed: 19 new tests in `pixelflow-core/tests/x86_backend_tests.rs`

Extended the file's existing `SimdOps`-required-method coverage (the
08-15 audit closed the *provided*-method gap in the same file) with the
required primitives, `Debug` impls, and `U32x4` operators that had none:

- **`Debug` for `Mask4`/`F32x4`/`U32x4`** — each had a
  `replace fmt with Ok(Default::default())` mutant survive, meaning
  nothing checks the formatted string has any content. Added one test per
  type asserting the exact `{:?}` output against a value chosen to be
  unambiguous (e.g. `Mask4`'s bit order, `U32x4`'s `to_array` — the latter
  had two more mutants of its own, `[0;4]`/`[1;4]`, killed by the same
  assertion).
- **`gather`'s index clamp** — `(idx[i] as isize).clamp(0, len as isize -
  1)`. A `- with +` or `- with /` mutant on `len - 1` only diverges from
  correct behavior when an index is clamped past the end of a short slice;
  picked a 3-element slice with an index of 100 so a wrong clamp bound
  indexes out of range and panics instead of returning the last element.
- **`add_masked`, `from_u32_bits`, `shr_u32`, `i32_to_f32`** — each had
  zero coverage; added one direct test per method against a value where a
  `Default::default()` (`0.0`) mutant is obviously wrong.
- **`BitOr`/`Not` for `F32x4`** — `sse2_bitwise` already covered `BitAnd`
  but not its siblings; added matching tests using `from_u32_bits` so the
  expected bit pattern is exact rather than approximate.
- **`U32x4`**: `splat`+`store` round trip, `BitAnd`, `BitOr`, `Not`, `Shl`,
  `Shr`, and `pack_rgba` (clamp+scale+pack of four `F32x4` channels into
  one `u32` per lane) — none had any coverage at all.

Not fixed: `SimdU32Ops::from_f32_scaled`'s mutant
(`replace ... with Default::default()`) is equivalent — the function's own
body is already `Self::default()` verbatim (it's a documented
placeholder; "actual packing is done via `pack_rgba`"), so the mutant and
the real code are byte-identical. Nothing to test. Re-verified after the
fixes: **80 mutants, 1 missed** (that same equivalent mutant, and nothing
else).

### Verified

- `cargo test -p pixelflow-core --test x86_backend_tests`: 37 passed, 0
  failed (8 pre-existing SSE2-required + 10 pre-existing SSE2-provided
  [08-15] + 19 new).
- `cargo test -p pixelflow-core` (all targets incl. doctests): passed, 0
  failed.
- `cargo clippy -p pixelflow-core --tests`: clean.
- `cargo fmt -p pixelflow-core -- --check`: clean.
- `cargo mutants -p pixelflow-core --file pixelflow-core/src/backend/x86.rs -E "F32x8|F32x16|U32x8|U32x16|Mask8|Mask16|Avx2|Avx512"`:
  80 mutants, 1 missed (the equivalent mutant above), 79 caught.

## Recommended next steps (not done here)

1. `pixelflow-codegen/src/emit/*` (~1,400 lines) — still open per 08-08/
   08-15, never mutation-tested under its post-crate-split location.
2. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` — the 2026-07-20
   audit's mutation findings against the private `backoff_with_jitter`/
   `send_with_backoff` functions it tests were never re-verified.
3. `pixelflow-graphics/src/spatial_bsp.rs` — now STYLE.md-clean, but never
   mutation-tested; a natural pairing with item 1's un-tested surface.
4. `pixelflow-core/src/backend/x86.rs`'s `F32x8`/`F32x16`/`U32x8`/`U32x16`/
   `Mask8`/`Mask16` (AVX2/AVX-512) impls, and `arm.rs`'s NEON impls — never
   tested at the unit level at all (`pixelflow-core/tests/` has zero
   references to any of those types; they're only exercised indirectly,
   through generic `Field`-level integration tests, when the crate happens
   to be built at a matching ISA level via `xtask isa-matrix`). A real
   `x86_backend_tests.rs`-style direct test file for each ISA level, run
   under `xtask isa-matrix`'s `-C target-feature` flags, is the natural
   fix — distinct from, and larger than, this pass's SSE2-only sweep.
5. If `cargo mutants` becomes a routine part of this audit series, its
   cfg-blindness (this pass's methodology finding, above) is worth
   automating around — e.g. a repo-level `.cargo/mutants.toml` with an
   `exclude_regex` for the higher-ISA type names, so a plain
   `cargo mutants -p pixelflow-core` doesn't need the `-E` flag
   reconstructed by hand every time.
