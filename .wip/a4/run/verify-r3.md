# A4 verification, round 3 (TAG=a4-r3)

Tree under test: HEAD 6bc1b5e (A4 fixes, round 2), tracked tree clean before and after (`git diff --quiet`,
`git status --short` shows only `.wip/`). Code base a747770. Every log and table below is in
`.wip/a4/run/v3/`; gate outputs are `.wip/gates/out/*.a4-r3.*`.
Verdict: every gate passes. One gate (`isa-matrix --smoke`) is not run as written, with its per-tier
replica passing instead (same as rounds 1 and 2; reason below).

## 1. Byte and key gates: all exit 0, identical to base
| gate | result |
|---|---|
| 1 byteprobe | avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b | 29 lines IDENTICAL; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 | 29 lines IDENTICAL |
| 2x | 29 records IDENTICAL (`extraction_objective`, `extracted_latency_prior_cost`, `shared_pass_bytes`) |
| `git status --short pixelflow-graphics` | empty after 1b, after 2/2x and after measure |

No `.build.lock` trouble: the sccache server was restarted first from a shell with no fd 9.

## 4. measure.sh: total Ir +0.09% (limit +0.5%)
total_ir 6,315,915,671 against base 6,310,238,494. vs_base: `Saturation::saturate` inclusive -0.045%,
`extract_dag_over` inclusive -0.104%, median peak RSS +0.68%, median wall -3.731% (VM drift, never decisive).

## 5. Tests, crate by crate (all 0 failed)
| crate / command | passed | ignored |
|---|---|---|
| pixelflow-ir | 396 | 2 |
| pixelflow-search (default) | 434 | 15 |
| pixelflow-search --all-features | 437 | 15 |
| pixelflow-search --features saturation-telemetry | 437 | 15 |
| pixelflow-codegen, PIXELFLOW_ISA=avx2 | 203 | 0 |
| pixelflow-codegen, PIXELFLOW_ISA=avx512 | 203 | 0 |
| pixelflow-compiler | 306 | 6 |
| pixelflow-graphics | 239 | 15 |
| pixelflow-pipeline | 228 | 2 |
| pixelflow-core | 136 | 1 |
| pixelflow-runtime | 152 | 6 |

Round 2's verification counted 200 for codegen; the fix round after it added three tests (203, the figure
that round's own notes record). Every other count equals round 2's.

### The new JIT tests, per tier
Computed as the `#[test]` functions in pixelflow-codegen present at HEAD and absent at a747770
(`new_codegen_tests.json`, 43), then matched against each tier's log: **43 on avx2, 43 on avx512, all `ok`, the
same 43 names, 8 of them `should_panic`** (6 unit, 2 integration). Lists with status:
`new_codegen_tests.avx2.txt`, `new_codegen_tests.avx512.txt` (columns: status, kind, test, file).
- 18 unit: `emit::avx512::tests` 1 (`a_value_is_an_opmask_exactly_when_is_mask_says_it_is_a_mask`),
  `emit::tests` 1 (`every_backend_emits_a_fold_whose_bound_is_a_value`), `emit::traffic::tests` 5,
  `pipeline::tests` 2 (`a_guard_on_a_uniform_count_is_computed_once_per_call`,
  `a_fold_whose_body_reads_no_coordinate_is_carved_in_the_body`), `program::lower::tests` 9 (6 `should_panic`:
  `a_bound_on_a_lattice_binder_fails_loudly`, `a_bound_that_is_a_mask_fails_by_name`,
  `a_value_bounded_fold_in_the_false_arm_fails_loudly`, `a_value_bounded_fold_nothing_reads_fails_loudly`,
  `a_value_bounded_fold_with_two_guards_fails_loudly`, `an_unguarded_value_bounded_fold_fails_loudly`).
- 25 in `tests/value_bounds_jit.rs`: the integration binary reports `25 passed` on each tier, 2 of them
  `should_panic` (`a_bound_that_is_a_mask_is_refused_by_name`,
  `a_mask_bound_adopted_whole_is_refused_by_name_through_the_cache`).
- The AVX-512-only half of `a_guard_computed_once_per_call_is_read_again_in_every_batch` (reload and frame
  growth) is gated on `isa::detect() == Avx512` in the test; the arm-test half runs on every tier. The host has
  AVX-512, so the `avx512` leg ran the real AVX-512 emitter; the `avx2` leg ran under `PIXELFLOW_ISA=avx2`.

### isa-matrix --smoke: not run as written
`xtask isa-matrix --smoke` first runs `cargo test --workspace --no-run` (xtask/src/main.rs:827), which the ground
rules forbid. Replica of its three per-level commands, built with its exact `-p` set in one invocation (a tmpfs
target, `RUSTC_WRAPPER` empty, disk untouched), once per tier:

| command | avx2 | avx512 |
|---|---|---|
| `test -p pixelflow-codegen -p pixelflow-ir -p pixelflow-core -p pixelflow-compiler -p pixelflow-pipeline --no-fail-fast` | 1269 passed, 11 ignored | 1269 passed, 11 ignored |
| `test -p pixelflow-graphics --test run_is_a_glyph --test font_rasterization_regression --no-fail-fast` | 13 passed | 13 passed |
| `test -p pixelflow-graphics --lib --no-fail-fast -- fonts::loop_blinn::` | 20 passed | 20 passed |

(Round 2: 1266 for the first command; +3 is the codegen fix-round tests.)

## 6. Build gates
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`: exit 0, no warning (target
  /dev/shm/a4-t199; every workspace package cleaned first, and the log shows all 14 `Checking`; the one `warning:`
  line is core-term's build-script note "Building with X11 display driver").
- `cargo fmt --all -- --check`: exit 0, empty.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features`: ok. Same for pixelflow-search: ok.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`: ok.
- rustdoc compare (docs cleaned first): pixelflow-ir base 38 head 38, NEW 0, GONE 0; pixelflow-search base 28
  head 28, NEW 0, GONE 0 (compare exit 0 for both).

## 7. Scripts (all exit 0)
check-intent (51 intent files; every intent.md within 80 lines, ir at 80, search egraph at 77), check-doc-paths
(682 paths; prints `grep: ../GEMINI.md: No such file or directory`, noise, exit 0), check-cfg-encapsulation (121
grandfathered, none new), check-bin-declarations (29), check-emit-boundary (self-test 66 cases; 25 files),
check-feature-matrix (cargo-hack on PATH; "every crate's every feature builds in isolation").

## 8. Greps (full output: v3/greps.txt)
- `VALUE_BOUND` in *.rs: 0 hits. Verdict: pass.
- `KNOWN_RANGE` outside tests, classified by file offset against each file's first `#[cfg(test)]`: non-test uses
  are `fold.rs:1150` (the definition), `passes/lattice.rs:31,222,227,342,347,352` (lattice folds), codegen
  `program/lower.rs:12,500` (import and the lane fold), `glyph_phase_split.rs:45,80,81` (the production glyph
  example, its comment states the reason). Every other hit (compiler `lower.rs` 6, `fold.rs` 4, `passes.rs` 7,
  `lattice.rs:399`, `fold_rules.rs` 3) is inside a `#[cfg(test)]` module. None in `emit/select.rs` or compiler
  `emit.rs` (grep: 0). Verdict: matches the spec's list.
- The known-helper grep, as amended by C4: `fn known(bound` has 0 hits (the helper is gone); the expect forms
  (`known_range().expect`, `KnownRange::of(..).expect`, `.expect(KNOWN_RANGE)`) hit `lower.rs:500` only. The
  unanchored `known(` also hits `emit/traffic.rs:126,238,350` (`Trips::known`), the other expected file. The spec's
  literal `-E` line with `\|` matches nothing as typed (0), which is why C4 re-anchors it. Verdict: pass.
- `grep -rnwE 'A4' --include=*.rs pixelflow-*`: 0 hits. `grep -rnw A4 --include=intent.md .`: 0 hits. Nothing
  says "until A4" in a tracked source or doc except `docs/language/2026-10-20-a-bound-is-a-value.md:629`, "the key
  refused a value bound until A4 gave it a marker", which is past tense and true after A4. The plan's other A4
  mentions are its work plan, its table of doc rewrites, and the cut-over sentence at :537 ("when A4 lands, A5
  goes against main"), a statement of merge order that stays true. Verdict: pass.
- `expect(dead_code, reason = "pub(crate) until A4` in pixelflow-ir: 0 hits.
- `iter::empty()` in pipeline, graphics, compiler/src: `pixelflow-compiler/src/emit.rs:605` only, the literal
  branch. Verdict: pass.
- `fold_header(id).to_bits()` (6 hits) plus the sites that grep misses (C12: compiler `emit.rs:582`,
  `corpus_gaps.rs:509`), each read in context: test_support:37 (the digest feeds `arena.children(id)` after),
  cell_grid:1219 and production_glyph_arena_dump:178 (print the body, then each value bound's child),
  key.rs:461 (pushes the body, then each value bound), training corpus:416 (body, then `lo`'s and `hi`'s children),
  collapse corpus:291 (as cell_grid), compiler emit.rs:582 (builds `[#(#values),*]` and `attach`es it),
  corpus_gaps:509 (`ids` is body then values; `header.attach(values)`). Verdict: pass.
- `FoldHeader::from_bits` decoders: training corpus:612 reads `header.arity() - 1` children after the body;
  collapse corpus:516 reads the line's values and asserts their count equals `arity() - 1`; compiler emit.rs:585
  attaches `#children`; emit.rs:735 reads only the monoid of a structural (literal-range) fold. Verdict: pass.
- `trips` in `pixelflow-codegen/src/emit`: every per-loop count is a `Trips` (`traffic.rs:156,192`, `select.rs:305-322`,
  `build.rs:323`, `mod.rs:529`); the one `u64` is `LoopFacts.trips` in `regalloc/local.rs:134`, set from
  `l.trips.weight()` at :202 and read at :389,:437. The two other hits are a doc/test constant (`mod.rs:1362`,
  `const TRIPS: u64 = 5` inside a test) and "round-trips" in an aarch64 comment. Verdict: pass.
- `dynamic_memory_ops`: callers `constant_traffic.rs:50` (maps `None` to "an estimate"), `collapse_bench/mod.rs:341`
  (`expect(LITERAL_FOLDS)`, its corpus has literal folds), the tests assert `Some` and `None`. Verdict: every
  caller handles the `Option`.
- Old texts: `has no bits")` (the exact old text, C3): 0 hits. The spec's looser alternation hits
  `program/lower.rs:204` and `pixelflow-ir/src/kernel.rs:1294`, both the shift-count sentence "a 32-bit lane has no
  bits there", unrelated. `no corpus encoding before`, `no code and no key`: 0 hits.
- `grep -rln '\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'`: empty.
- Added lines of the A4 diff (3,765 `+` lines, v3/added-lines.txt) in tracked non-.wip files: no `.wip` path; no
  model name (the one case-insensitive hit is `CLAUDE.md`, the repository file name, in a doc comment); the label
  hits are the plan's step names (B2) and BACKLOG row ids (E8, C8, and a rewrapped pre-existing "C1"), none a
  ruling label. "Ruling 6" in the BACKLOG row and the plan is not new (both hold it at a747770).
- Reduce-site checklist (`.wip/a4/gen/checklist.py`): classified 190, A4 0, A5 0, UNCLASSIFIED 0, STALE 0.
- `git lfs ls-files | grep -c .`: 68.

## 9. Diffstat (a747770 against HEAD, excluding .wip)
46 files changed, 3719 insertions(+), 405 deletions(-) (v3/diffstat.txt).

## Environment notes
- Disk stayed at 6.2 GB free on `/` throughout (the guard script would have stopped at 2.3 GB). The clippy,
  doc, check and smoke-replica builds used `/dev/shm` targets with `RUSTC_WRAPPER` empty; the smoke replica's
  8.8 GiB target (`/dev/shm/a4-r3-smoke`, created this round) was cleaned afterwards to leave RAM free, and
  nothing under /tmp/claude-0, /home/user/scratch or .wip was deleted.
- Tracked files were not edited. The gate scripts restored their own probe file each time.
