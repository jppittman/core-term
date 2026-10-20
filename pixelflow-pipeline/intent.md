# pixelflow-pipeline

**Is.** The observer of the optimizer `P : Kernel × LatticeShape × Isa → code`
(DENOTATION §8): it times the code `P` emits and records what was measured, with
provenance. Performance is an outcome against a baseline, never a mechanism
asserted (the as-if contract, §8).

## Owns
- The clock: `BenchSession` (pin, burn-in, sentinel calibration, identity-kernel
  overhead, median of `TIMED_RUNS` samples) and the clocks a number can be in,
  `LocalNs`, `SessionNs`, `DriftFactor` (`jit_bench.rs`); `collapse_bench/` does
  the same for collapse bodies at their bake shapes.
- Corpus tooling (`training/`), Guide research bins (`src/bin/`), benchmark
  kernels (`shader_bench.rs`, `poly.rs`).
- Provenance: `ConfigHash`, `JournalEntry`, the one writer `append_record`, and
  `environment_fingerprint()`, whose `isa=` is the tier the JIT selected
  (`journal.rs`); a format's identity is the hash of its `SCHEMA` (`schema.rs`).

## Never
- No other workspace crate depends on it.
- A cost is never a bare `f64`: `LocalNs` has no `Sub` and becomes a
  `SessionNs` only through `normalize(drift)`; a sessionless measurement is
  never a label (`corrected_session_ns` panics).
- A held-out tier is never read by a decision: `Fence<T>` takes `T:
  HoldoutSide`, sealed to the DEV and FINAL markers.
- No journal write fails quietly: `append_record` panics, and refuses an LFS stub.
- A static feature of emitted code is a column beside the wall clock, never a
  verdict; predictors are scored against the clock (`collapse_bench/predict.rs`).
- No scalar interpreter or oracle here to accept a label by agreeing with the
  expansion that produced it; the lane count is `jit_vector_bytes()`, not a flag.

## Contract
`src/lib.rs` declares `alloc_probe`, `collapse_bench`, `jit_bench`, `journal`,
`poly`, `schema`, `shader_bench`, and (feature `training`, default) `training`.
Every `src/bin/*.rs` has a `[[bin]]`; all but `collapse_cost` and
`extraction_witnesses` require `training`.

## Enforced by
- Types: `LocalNs`/`SessionNs` (`jit_bench.rs`); `Fence<T: HoldoutSide>` and
  its `compile_fail` doctest (`training/split.rs`).
- Tests: `sentinel_normalization_is_a_clock_correction` and
  `plausibility_floor_is_per_lane_not_per_call` (`jit_bench.rs`);
  `environment_fingerprint_records_the_tier_the_jit_selected` (`journal.rs`).
- `scripts/check-bin-declarations.sh`, `scripts/check-provenance-journal-scope.sh`;
  `xtask isa-matrix --smoke` runs this crate per ISA tier (`xtask/src/main.rs`).

## Behind
- `gen_bench_corpus` and `quarantine` went with the scalar interpreter
  (CLAUDE.md, "Cost Model and the Guide"), yet 21 references to it remain in
  `src/` (`training/corpus.rs`, `training/split.rs` tell users to run it),
  `lib.rs` and `training/mod.rs` still list `quarantine`, and
  `training/bezier_family.rs` documents a `Quarantine` against `eval_scalar`.
  `NNUE_TRAINING.md` describes the deleted extraction-head pipeline. No
  printed number is checked to be a ratio against a baseline.

## Decided in
`docs/process/2026-10-13-tests-check-the-as-if-contract.md`,
`docs/compiler/2026-09-07-benchmark-correction.md`, `docs/archive/plans/2026-08-17-cost-model-domain.md`,
`docs/compiler/2026-09-01-schedule-cost-model-denotation.md`, `docs/results/2026-08-05-bench-harness-integrity-audit.md`.
