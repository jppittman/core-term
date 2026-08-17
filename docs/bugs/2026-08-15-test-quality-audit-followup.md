# Test quality control follow-up — 2026-08-15

Scope: scheduled continuation of
`docs/bugs/2026-08-14-test-quality-audit-followup.md`. Since that pass landed
(`99713d0`), `main` moved by one commit: `084d3ef`, "give sin a range
reduction that survives large arguments" (#992).

## Delta audit: `084d3ef`'s new/changed tests

Reviewed every test the sin fix added or touched for STYLE.md compliance:

- `pixelflow-ir/tests/trig_range.rs` (new, 421 lines) — builds kernels
  through `ExprArena::push_var`/`push_unary`/`push_binary` and evaluates with
  `eval_scalar`, all `pub fn`. No private-field or `pub(crate)` access.
- `pixelflow-codegen/tests/transcendental_jit.rs` (+55 lines,
  `sin_cos_jit_never_leave_unit_range`) — goes through `Kernel::x()`,
  `jit_cache::compile_cached`, and the compiled `JitManifold`'s call
  interface. Public API only.
- `pixelflow-core/src/ops/trig.rs`'s new
  `asin_acos_are_nan_outside_their_domain` and
  `sin_cos_respect_the_domain_limit` — go through `Field::from(x)` and its
  public trig methods.

All three read as descriptive sentences once "it should" is prepended and
none reach into implementation details. No fixes needed here — this delta
was clean.

## Confirmed resolved: backlog item #5 from the 08-14 audit

> `actor-scheduler/src/kubelet.rs`'s `ManagedPod` construction... no longer
> appears in the file — looks resolved by an intervening commit, but not
> independently confirmed.

`actor-scheduler/src/kubelet.rs` does not exist in the current tree, and
`ManagedPod` has zero matches anywhere under `actor-scheduler/`. Confirmed
resolved; removing from the backlog for good.

## Mutation testing: `pixelflow-core/src/backend/mod.rs`

Picked up backlog item #2 from the same audit: this file (the `SimdOps`
trait — required per-ISA primitives plus provided/default expansions built
on them) has never been mutation-tested since the 2026-08-02 crate split
that moved it out of `pixelflow-ir`.

`cargo-mutants` v27.1.0 (freshly installed — not present in this
environment, consistent with every prior pass).

**First sweep, scoped to `-p pixelflow-core --file .../backend/mod.rs --
--lib`: 32/32 mutants missed.** Every mutant targeted one of the ten
*provided* methods — `exp`, `ln`, `log10`, `pow`, `hypot`, `mul_rsqrt`,
`ceil`, `round`, `fract`, `clamp` — and every one of them, including the
trivial "replace the whole function with `Default::default()`" mutant,
survived. These functions had **zero** direct test coverage anywhere in the
crate: the crate's SSE2 backend tests
(`pixelflow-core/tests/x86_backend_tests.rs`) only ever exercised the
*required* trait methods (arithmetic, `sqrt`, `recip`, comparisons), never
the default expansions layered on top.

(The required methods aren't part of this gap — `simd_sqrt`, `simd_min` etc.
already have direct coverage in the same file; the mutants tool wasn't asked
to touch backend-specific `.rs` files where those are implemented.)

### Fixed: 10 new tests in `pixelflow-core/tests/x86_backend_tests.rs`

One test per provided method, against the public `SimdOps` trait on the
existing file's `F32x4` (SSE2) backend — matching the file's established
convention rather than reaching for a private accessor. Each input was
chosen to distinguish the real formula from its likely mutants, not just
from zero:

- **`pow`** uses `self=4.0, exp=0.5`, not `self=2.0, exp=3.0` — at
  `self=2.0`, `log2(self) == 1`, so `exp * log2(self)` and
  `exp / log2(self)` coincide and a "replace `*` with `/`" mutant would
  survive undetected. At `self=4.0`, `log2(self) == 2`, so `*`, `+`, and `/`
  all disagree.
- **`hypot(3.0, 4.0) == 5.0`** — the 3-4-5 triple was picked because it
  simultaneously kills all seven mutants in `(self*self + y*y).simd_sqrt()`
  (both `*`→`+`/`/` substitutions, the `+`→`-`/`*` substitution, and the
  `Default::default()` fallback): `-` even produces a negative radicand,
  which `simd_sqrt` (backed by `sqrt_fast`) resolves to `0`, nowhere near
  `5.0`.
- **`ceil`, `round`, `fract`** each got a non-tie, non-edge input
  (`1.2`/`2.7`/`2.75`) so the assertion isn't entangled with the
  known-and-documented (CLAUDE.md, "Floating point at the edges")
  combinator-tier rounding quirks at exact ties or negative-zero boundaries
  — this pass pins current behavior at ordinary values, not the disputed
  edge cases.

`cargo test -p pixelflow-core --test x86_backend_tests`: 18/18 passed (8
pre-existing + 10 new).

**Re-running the mutants sweep without `--lib`** (the new coverage lives in
an integration-test binary, invisible to a `--lib`-scoped run — the same
per-target-scoping trap the 08-08 audit already hit once for
`fold_is_platform_specific`): **32/32 caught.**

## Verified

- `cargo test -p pixelflow-core --test x86_backend_tests`: 18 passed, 0
  failed.
- `cargo test -p pixelflow-core` (all targets incl. doctests): passed, 0
  failed.
- `cargo test --workspace --lib`: 117 passed, 1 ignored, 0 failed (114s,
  dominated by `pixelflow-search`'s NNUE training tests, as in every prior
  pass).
- `cargo clippy -p pixelflow-core --tests`: clean.
- `cargo fmt -p pixelflow-core -- --check`: clean.
- `cargo mutants -p pixelflow-core --file pixelflow-core/src/backend/mod.rs`
  (no `--lib` restriction): 32/32 caught, 0 missed.

## Recommended next steps (not done here)

1. `pixelflow-search/src/egraph/cost.rs` — still open per the 08-08 audit:
   a partial mutants run found one real gap (`CostModel::zero()`, already
   fixed) before its own slow `--lib` baseline (~110s) timed out the pass.
   Still needs either a narrower test filter or a longer time budget.
2. `pixelflow-codegen/src/emit/*` (~1,400 lines) — flagged by 08-08 as never
   mutation-tested under its post-crate-split location. Still true;
   `backend/mod.rs` (this pass) was the other half of that finding and is
   now closed.
3. `pixelflow-graphics/src/spatial_bsp.rs` — confirmed still open: 19 tests
   reach into private `bsp.interiors[...]` fields with no public accessor.
   Still a design call (test-only introspection API vs. property tests over
   `eval()` vs. documented rule-break), not a mechanical fix; not
   independently re-verified this pass.
4. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` (now ~line 1918) —
   confirmed still present; the 2026-07-20 audit's mutation findings against
   the private `backoff_with_jitter`/`send_with_backoff` functions it tests
   were not re-verified this pass.
5. `pixelflow-core/src/backend/x86.rs`/`arm.rs`'s *required* `SimdOps`
   methods (the per-ISA primitives this pass's provided-method sweep
   deliberately didn't touch) — never independently mutation-tested as a
   whole-file sweep; a natural next target now that the provided-method half
   of the same file's contract is closed.
