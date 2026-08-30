# Test quality control follow-up — 2026-08-30

Scope: scheduled continuation of
`docs/bugs/2026-08-26-test-quality-audit-followup.md`. `main` had not moved
since `7b4ab731`/`b4cc51f0` (the intervening `9068865b` was an unrelated
docs commit), so this pass picks up backlog item 1: the `cargo-mutants`
sweep of `pixelflow-search/src/egraph/cost.rs` that the 08-08 and 08-26
passes both left blocked on the crate's slow whole-`--lib` baseline.

## Unblocking the sweep

`cargo-mutants` v27.1.0 (freshly installed — not present in this
environment, consistent with every prior pass). `cargo test -p
pixelflow-search --lib` takes ~130s, almost entirely
`math::pict_rewrite_tests::pict_rewrite_rules_preserve_semantics` (a
property test, unrelated to `cost.rs`), and cargo-mutants reruns that full
suite once per mutant. A 60s per-mutant timeout — reasonable everywhere
else in this crate — made every mutant against `cost.rs` time out before
its own baseline even finished.

Two changes made the sweep tractable: `--timeout 180`, and passing `--
--skip pict_rewrite_rules_preserve_semantics` through to the test binary
(cargo-mutants does not insert its own `--` before forwarded args, so this
needs `-- -- --skip ...` on the `cargo mutants` command line — the first
`--` is cargo-mutants' own separator, the second becomes `cargo test`'s).
Skipping that test is safe here: it exercises `pict` rewrite-rule
semantics, nothing in `cost.rs`'s reachable set. Baseline dropped from a
timeout to 6s build + 16s test.

## `pixelflow-search/src/egraph/cost.rs`

`cargo mutants -p pixelflow-search --file .../cost.rs -j 4 --timeout 180`:
**73 mutants, 22 unviable, 44 caught, 7 missed.**

### Fixed: 4 new tests, closing 6 of the 7 gaps

All 6 real gaps were the same shape, at two call sites in
`load_or_default`: the `HOME`-config branch (line 423) and the
workspace-relative fallback loop (line 440) each guard their
warn-and-continue arm with `Err(e) if e.kind() != NotFound`. Nothing
exercised either branch — no test ever pointed `HOME` at a directory with
a malformed `.config/pixelflow/cost_model.toml`, and no test ever put a
malformed file at a workspace-relative path — so the guard's `!=`/`==`
swap and both `true`/`false` replacements all survived.

Both warn arms are pure side effects (an `eprintln!`) that don't change
`load_or_default`'s return value — the function falls through to the next
candidate location regardless of which arm runs — so the only way to
observe the mutation is on stderr. Added a `run_child` helper, following
the file's existing `env_var_override_child` pattern (a real parse race on
process-global env vars means these have to run in a spawned child, not by
mutating this process's environment), that re-invokes the test binary
under a controlled `HOME`/cwd and returns its captured stdout/stderr.
`--nocapture` on the child invocation turned out to be load-bearing, not
cosmetic: libtest buffers a *passing* test's output and only releases it
on failure, so without it the child's `eprintln!` warnings never reached
the real process stderr this helper reads — the first version of these
tests failed with an empty `stderr` before that flag was added.

Four new tests, two pairs:

- **HOME-config branch**: `load_or_default_should_warn_on_stderr_when_the_home_config_exists_but_fails_to_parse`
  (malformed file present → guard true → must warn — kills the `false` and
  `==` mutants) and
  `load_or_default_should_stay_silent_on_stderr_when_the_home_config_is_absent`
  (no file at all → guard false → must stay silent — kills the `true`
  mutant).
- **Workspace-path branch**: the same pair,
  `load_or_default_should_warn_on_stderr_when_a_workspace_config_exists_but_fails_to_parse`
  and
  `load_or_default_should_stay_silent_on_stderr_when_no_workspace_config_is_present`,
  with `HOME` pointed at a config-free directory so the earlier `HOME`
  branch can't be the source of the observed warning.

### Left open: 1 documented equivalent mutant

`depth_cost`'s `depth > self.depth_threshold` → `>=` mutant survives and
is genuinely equivalent: at `depth == self.depth_threshold` the
multiplier `(depth - self.depth_threshold)` is `0` under either operator,
so both branches return the same `0` for every input — no test can
distinguish them. Documented with a comment at the site, matching the
precedent in `pixelflow-graphics/src/spatial_bsp.rs` (a `/` vs `*`
equivalent in a sort comparator) and `pixelflow-compiler/src/codegen/util.rs`.

**Final state: 73 mutants, 72 caught, 1 documented equivalent, 0 real gaps.**

## Verified

- `cargo test -p pixelflow-search --lib egraph::cost::persistence::`: 10
  passed, 0 failed, 5 ignored (the child-process halves of the 5
  spawn-based tests in this module).
- `cargo test -p pixelflow-search --lib`: 187 passed, 0 failed, 6 ignored.
- `cargo clippy -p pixelflow-search --lib --tests`: clean.
- `cargo fmt -p pixelflow-search -- --check`: clean.
- `cargo mutants -p pixelflow-search --file
  pixelflow-search/src/egraph/cost.rs -j 4 --timeout 180 -- -- --skip
  pict_rewrite_rules_preserve_semantics`: 73 mutants, 72 caught, 1
  documented equivalent, 0 missed.

## Recommended next steps (not done here)

Backlog carried forward from 2026-08-26, minus the item closed above and
item 3 (`actor-scheduler`'s `backoff_unit_tests`), which that pass
recommended dropping outright — re-checked this pass (by re-reading, not a
fresh mutants run) and still nothing actionable there; see the 08-26
writeup for the full reasoning.

1. `pixelflow-codegen/src/emit/*` (~11,700 lines across `mod.rs`,
   `x86_64.rs`, `avx2.rs`, `avx512.rs`, `aarch64.rs`, `regalloc.rs`,
   `executable.rs`, `coverage.rs`) — flagged since 08-08 as never
   mutation-tested under its post-crate-split location. Still true, and
   large enough to need its own multi-file-scoped passes rather than one
   sweep; `coverage.rs` (77 lines, pure `const` data, no branches) can
   likely be skipped entirely.
2. `pixelflow-core/src/backend/x86.rs`'s `F32x8`/`F32x16`/`U32x8`/
   `U32x16`/`Mask8`/`Mask16` (AVX2/AVX-512) impls, and `arm.rs`'s NEON
   impls — never tested at the unit level under a build that actually
   activates those ISA levels. Newly worth reattempting: this session's
   sandbox CPU reports `avx512f`/`avx512dq`/`avx512cd`/`avx512bw`/
   `avx512vl`/`avx512_vnni` (`/proc/cpuinfo`), so the AVX2/AVX-512 half is
   locally testable here for the first time in this audit's history —
   just not attempted this pass, which stayed scoped to the cost.rs
   backlog item. `arm.rs`/NEON still needs aarch64 hardware or a cross
   build this sandbox isn't set up for.
