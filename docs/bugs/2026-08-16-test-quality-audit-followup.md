# Test quality control follow-up — 2026-08-16

Scope: scheduled continuation of
`docs/bugs/2026-08-14-test-quality-audit-followup.md`. `main` had one new
commit since that pass (`084d3ef`, "give sin a range reduction that survives
large arguments") — audited it for STYLE.md compliance below — then this pass
worked the 08-14 audit's own recommended-next-steps backlog: items #2
(`pixelflow-core/src/backend/mod.rs`, never mutation-tested under its
crate-split location), a continuation of the 08-08 audit's cut-off
`pixelflow-search/src/egraph/cost.rs` run, and the two long-open design-call
items (`pixelflow-graphics/src/spatial_bsp.rs`'s private-field tests,
`actor-scheduler/src/lib.rs`'s `backoff_unit_tests`).

## Static audit: `084d3ef`'s new test file

`pixelflow-ir/tests/trig_range.rs` (421 new lines, the sin-range-reduction
fix's regression suite) is an integration test — by construction it can only
reach `pixelflow_ir`'s public surface (`ExprArena`, `OpKind`, `eval_scalar`,
`passes::TRIG_DOMAIN`), so the "Test Public API" rule holds automatically.
Names (`sin_and_cos_never_leave_unit_range`, `sin_is_nan_outside_the_domain`,
`sin_of_negative_zero_is_positive_zero`, etc.) each name the property that
breaks if the test fails. No violations found.

## Fixed: `pixelflow-graphics/src/spatial_bsp.rs` private-field tests

Open since the very first audit (2026-07-20), re-flagged every pass since as
a "design call, not fixed" — ~20 tests read `bsp.interiors[i]` directly, where
`interiors` is `SpatialBSP`'s private field. Resolving it needed one
observation the earlier passes' framing missed: `InteriorNode` itself (and
all four of its fields — `axis`, `threshold`, `left`, `right`) was already
`pub`. The only thing missing was a way to *reach* one. Added

```rust
pub fn interior(&self, idx: usize) -> &InteriorNode
```

next to the existing `interior_count()`/`leaf_count()` accessors, and
mechanically rewrote all 18 test call sites (`&bsp.interiors[i]` →
`bsp.interior(i)`; the file's one production call site, `traverse`'s
`&self.interiors[idx]` hot path, is untouched). No test-only introspection
API, no property-test rewrite — the minimal public accessor this type was
already one method short of. `cargo test -p pixelflow-graphics --lib
spatial_bsp::`: 56/56. `cargo clippy -p pixelflow-graphics --lib --tests`:
clean.

## Investigated, judged not to need a fix: `actor-scheduler`'s `backoff_unit_tests`

Also open since 2026-07-20. `backoff_with_jitter` is a private free function;
`backoff_unit_tests` (in the same file) calls it directly, with an existing
`// These functions are private so tests live in the same module.` comment —
already the documented-exception form STYLE.md's Flexibility clause asks for,
not a silent rule-break. Checked whether the same coverage is reachable
through the public surface instead: `send_with_backoff` (the one caller of
`backoff_with_jitter`) is used only by `ActorHandle::send` for the
control/management lanes, and reaching its timeout/jitter behavior from there
would mean filling a real channel and depending on wall-clock backoff timing
instead of calling the pure function with tuned `SchedulerParams` directly —
slower and less deterministic for no gain in what's actually verified. Each
test's header comment already names the specific mutant it kills
(`> with ==`, `> with >=`, the jitter arithmetic, etc.), so this reads as a
judicious, already-justified exception rather than a backlog item. Not
changed. (`actor-scheduler/src/kubelet.rs`, the other 2026-07-20 finding in
this crate, no longer exists in the tree — confirmed resolved, dropping it
from the backlog for good.)

## Mutation testing: `pixelflow-core/src/backend/mod.rs`, full-file sweep

`cargo-mutants` v27.1.0 (freshly installed — not present in this environment,
consistent with every prior pass), scoped to this file against
`pixelflow-core`'s `x86_backend_tests` integration test.

**Before: 32 mutants, 32 missed (100%).** Every one of `SimdOps`'s ten
*provided* (default trait-method) functions — `exp`, `ln`, `log10`, `pow`,
`hypot`, `mul_rsqrt`, `ceil`, `round`, `fract`, `clamp` — had zero direct
coverage. These are shared formulas inherited by every backend (SSE2, AVX2,
AVX-512, NEON all use the same default; none override them), so the gap was
real for all four ISAs at once, not just x86.

### Fixed: 10 methods, tests added to `pixelflow-core/tests/x86_backend_tests.rs`

Following the file's existing pattern of exercising `SimdOps` directly
through the public `x86::F32x4` type (already how `sse2_math`,
`sse2_reciprocal_math`, etc. are written — `F32x4` and the `SimdOps`/
`MaskOps` traits are genuinely `pub`, so this is public-API testing, not a
rule-break). Each formula's test value is chosen so every listed mutant
produces a *different* wrong answer rather than one that happens to
coincide with the correct one (e.g. `hypot(3, 4) == 5` — the 3-4-5 triangle —
kills all seven of `self*self + y*y`'s arithmetic mutants because swapping
any operator lands on 17, 22, or 144, never 5; `pow(4, 3) == 64` was picked
over `pow(2, 3)` specifically because at base 2 the `*`↔`/` mutant on
`exp * self.log2()` is degenerate, `log2(2) == 1`, so `exp/1 == exp*1`).

**One test pins a known, documented non-fix, not a verified contract:**
`round`'s formula is `(self + 0.5).simd_floor()`, which is exactly the
"combinator tier's `Round`" CLAUDE.md calls a "genuine bug... not any IEEE
rounding mode" that "should be replaced with the target's rounding
instruction." No backend overrides it (checked: no `roundps`/`vroundps`/NEON
`FRINTA` call exists anywhere in `x86.rs`/`arm.rs` for `round` or `ceil`),
so the bug is live on every ISA today. `round_currently_implements_x_plus_
half_floor_not_ieee_rounding` pins the current output (including CLAUDE.md's
own `round(-1.5) == -1` example) so a future accidental change doesn't slip
through silently — its name and comment say explicitly that this documents
current behavior, not a certified-correct one. **Fixing the underlying bug
is out of scope for this pass**: it needs per-ISA instruction selection
(SSE2 baseline has no hardware round instruction; SSE4.1/AVX2/AVX-512/NEON
do) verified across all four backends via `cargo xtask isa-matrix`, which is
a numeric-semantics change affecting every `.round()`/`.ceil()` caller in the
workspace, not a mutation-testing fix. Recommended as the top follow-up item
below.

**After: 32 mutants, 0 missed / 32 caught.** Complete closure — every
provided method's mutants are now killed.

`cargo test -p pixelflow-core --test x86_backend_tests`: 18/18 (10 new).
`cargo clippy -p pixelflow-core --lib --tests`: clean (checked as part of
workspace-wide clippy below). `cargo fmt -p pixelflow-core -- --check`: clean.

## Mutation testing: `pixelflow-search/src/egraph/cost.rs`, continued

The 08-08 audit's run timed out after finding one gap (`CostModel::zero()`,
already fixed then); 08-14 listed "likely has more gaps" as a named
follow-up. Re-ran with the same `-- --lib egraph::cost::` filter that keeps
the baseline to ~5s instead of pixelflow-search's 110s+ full `--lib` (NNUE
training tests dominate that).

**Before: 73 mutants, 46 missed / 5 caught / 22 unviable.** Essentially the
entire `CostModel` API past pricing lookups had no direct coverage:
`shallow()`, `cost()`/`set_cost()`, `depth_cost()`, `node_op_cost()`'s
leaf/`Dwrt` special cases, `save_toml`/`load_toml`/`load_or_default`
persistence, `from_map`/`to_map` interop, and the `CostFunction` trait impl
(`node_cost`, `cost_by_kind`) on `CostModel` itself.

### Fixed: 6 new test modules in `cost.rs`

- **`construction_and_accessors`** — `shallow()` (depth_threshold=16,
  depth_penalty=500, latency-prior costs retained via `..Self::new()`),
  `cost`/`set_cost` round-trip, `depth_cost` at/below/above the threshold
  (values chosen so `-`↔`+`/`/` and `*`↔`+`/`/` each land on a distinct wrong
  number).
- **`node_op_cost_tests`** — leaves (`Var`/`Const`/`Buffer`, the last built
  via the public `BufferIdentity::mint()`) cost 0, `Dwrt` prices at
  `usize::MAX / 4` regardless of the table, ordinary ops read through
  `cost()`.
- **`cost_function_trait_impl`** — `CostFunction::node_cost`/`cost_by_kind`
  on `CostModel` delegate correctly (the trait-object interop path, distinct
  from calling `.cost()`/`.node_op_cost()` directly).
- **`map_interop`** — `to_map`/`from_map` round-trip (`OpKind::COUNT + 2`
  entries, spot-checked value), unrecognized keys ignored.
- **`toml_persistence`** — `save_toml`/`load_toml` round-trip (starts from
  `zero()`, not `latency_prior()` — an unset op stays 0, matching the
  documented contract), comment/blank-line skipping, malformed-line and
  non-numeric-value rejection, unrecognized-key tolerance, `depth_threshold`/
  `depth_penalty` key parsing.
- **`load_or_default_tests`** — an `EnvVarGuard` RAII helper plus a
  module-local `Mutex` serializes the three tests that touch
  `PIXELFLOW_COST_MODEL`/`HOME` (process-global state, otherwise racy under
  parallel `cargo test`): explicit env-var file wins, falls back to
  `latency_prior()` when nothing is configured (`HOME` pointed at an empty
  temp dir so a real developer's own `~/.config/pixelflow/cost_model.toml`
  can't leak into the result), falls back past a malformed `HOME` config file
  rather than surfacing a half-built model.

**After: 73 mutants, 9 missed / 42 caught / 22 unviable.** All 9 remaining
misses confirmed, not left unexamined:

- **`cost.rs:146`, `CostFunction::cost_by_kind`'s default trait-method
  body** (`panic!(...)`) — pre-existing, out of scope; `CostModel` is the
  only implementor anywhere in the workspace and always overrides it, same
  finding the 08-08 audit already made for this exact line.
- **`cost.rs:248:18`, `depth_cost`'s `depth > threshold` → `depth >= threshold`.**
  Equivalent mutant: the two operators disagree only at `depth == threshold`,
  and there `(depth - threshold) * penalty` evaluates to `0`, identical to
  the `else` branch's constant `0`. No input observes the difference — same
  category as the 08-14 audit's `Round` `-0.5` tie finding in `kind.rs`.
- **`cost.rs:394`/`411` (×3 each), the `e.kind() != NotFound` guards in
  `load_or_default`.** Genuinely different behavior under mutation, but not
  one the function's contract exposes: both the guarded arm and its `Err(_)
  => {}` sibling fall through to the *same* next fallback candidate and
  return the *same* eventual model — the only difference is whether a
  warning reaches `stderr`. Killing these would require capturing process
  stderr from within the test, a different (and more invasive) technique
  than the rest of this file uses; left as a documented, deliberately
  unobserved gap rather than attempted.

`cargo test -p pixelflow-search --lib egraph::cost::`: 24/24 (20 new).
`cargo test -p pixelflow-search --lib`: 137/137, 1 ignored. `cargo clippy -p
pixelflow-search --lib --tests`: clean. `cargo fmt -p pixelflow-search --
--check`: clean.

## Verified

- `cargo test -p pixelflow-graphics --lib spatial_bsp::`: 56/56.
- `cargo test -p pixelflow-core --lib`: 123 passed, 1 ignored, 0 failed.
- `cargo test -p pixelflow-core --test x86_backend_tests`: 18/18.
- `cargo test -p pixelflow-search --lib`: 137/137, 1 ignored.
- `cargo clippy -p pixelflow-core --lib --tests`, `-p pixelflow-graphics
  --lib --tests`, `-p pixelflow-search --lib --tests`: clean.
- `cargo fmt --check` on every touched crate: clean.
- `cargo mutants -p pixelflow-search --file
  pixelflow-search/src/egraph/cost.rs -- --lib egraph::cost::`: 9/73 missed,
  all 9 confirmed equivalent/unobservable/pre-existing (proofs above).
- `cargo mutants -p pixelflow-core --file pixelflow-core/src/backend/mod.rs
  -- --test x86_backend_tests`: 0/32 missed.

## Flagged for human attention (not a test-quality-control fix)

`SimdOps::round`'s default implementation, `(self + 0.5).simd_floor()`, is a
known, CLAUDE.md-documented correctness bug — not any IEEE rounding mode,
and live on every backend (SSE2, AVX2, AVX-512, NEON), since none override
it with the hardware instruction the doc says it "should" use. This pass
added a test that pins the current behavior (so it doesn't silently drift
further) but deliberately did not attempt the fix: it needs per-ISA
instruction selection verified across all four backends via `cargo xtask
isa-matrix`, which is a numeric-semantics change with real behavioral
consequences for every `.round()`/`.ceil()` caller in the workspace, not a
mechanical mutation-testing gap. Recommending this as the top item for a
dedicated follow-up (or human) pass rather than folding it into this one.

## Recommended next steps (not done here)

1. **`SimdOps::round`/`ceil`'s hardware-instruction fix** (see above) — the
   highest-value remaining item, and the only one this pass found that's a
   production bug rather than a coverage gap.
2. `pixelflow-codegen/src/emit/*` (~5,800 lines across `mod.rs` plus the
   per-ISA `aarch64.rs`/`avx2.rs`/`avx512.rs`/`x86_64.rs`/`regalloc.rs`/
   `executable.rs`) — still flagged since 08-08, still never mutation-tested
   under its crate-split location. Too large for a single pass; needs either
   a per-file split across several audits or a much longer time budget.
3. `pixelflow-ir/src/kind.rs`'s 2 confirmed-equivalent misses from 08-14 and
   this pass's 9 confirmed misses in `cost.rs` don't need revisiting unless
   the surrounding code changes shape (an equivalence proof can be
   invalidated by a refactor that makes the two branches newly
   distinguishable).
