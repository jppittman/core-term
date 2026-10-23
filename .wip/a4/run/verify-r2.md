# A4 verification, round 2 (TAG=a4-r2)

Tree under test: HEAD 5652c63 (A4 fixes, round 1), clean tracked tree before and after
(`git diff --quiet`, no modified tracked file; head.txt in v2/). Code base a747770.
Every log and table below is in `.wip/a4/run/v2/`; gate outputs are `.wip/gates/out/*.a4-r2.*`.
Verdict: every gate passes. One gate (`isa-matrix --smoke`) is not run as written, with its
per-tier replica passing instead (same as round 1; reason below).

## 1. Byte and key gates: all exit 0, identical to base
| gate | result |
|---|---|
| 1 byteprobe | avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b | 29 lines IDENTICAL; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal |
| 2 | 29 lines IDENTICAL |
| 2x | 29 records IDENTICAL (`extraction_objective`, `extracted_latency_prior_cost`, `shared_pass_bytes`) |
| `git status --short pixelflow-graphics` | empty after 1b, 2, 2x and measure |

No .build.lock trouble (the sccache server held fd 9 as its own socket, not the lock).

## 4. measure.sh: total Ir +0.09% (limit +0.5%)
total_ir 6,315,889,257 against base 6,310,238,494. vs_base: saturate inclusive -0.045%,
extract_dag_over inclusive -0.104%, median peak RSS +0.083%, median wall -1.026% (drift, never decisive).

## 5. Tests, crate by crate (all 0 failed)
| crate / command | passed | ignored |
|---|---|---|
| pixelflow-ir | 396 | 2 |
| pixelflow-search (default) | 434 | 15 |
| pixelflow-search --all-features | 437 | 15 |
| pixelflow-search --features saturation-telemetry | 437 | 15 |
| pixelflow-codegen, PIXELFLOW_ISA=avx2 | 200 | 0 |
| pixelflow-codegen, PIXELFLOW_ISA=avx512 | 200 | 0 |
| pixelflow-compiler | 306 | 6 |
| pixelflow-graphics | 239 | 15 |
| pixelflow-pipeline | 228 | 2 |
| pixelflow-core | 136 | 1 |
| pixelflow-runtime | 152 | 6 |

These equal round 1's counts exactly (the fix round changed tests in place and added none).

### The new JIT tests, per tier
Computed as the `#[test]` functions in pixelflow-codegen present at HEAD and absent at a747770
(`new_codegen_tests.json`), then matched against each tier's log: **40 on avx2, 40 on avx512, all
`ok`, same 40 names, 7 of them `should_panic`** (6 unit, 1 integration). Lists with status:
`new_codegen_tests.avx2.txt`, `new_codegen_tests.avx512.txt`.
- 18 unit: `emit::avx512::tests` 1 (`a_value_is_an_opmask_exactly_when_is_mask_says_it_is_a_mask`),
  `emit::tests` 1 (`every_backend_emits_a_fold_whose_bound_is_a_value`), `emit::traffic::tests` 5
  (`dynamic_memory_ops_is_no_count_over_an_estimate`, `a_known_trip_count_prints_as_its_number`,
  `an_estimate_prints_marked`, `nesting_an_estimate_is_an_estimate`, `a_product_past_u64_is_an_estimate`),
  `pipeline::tests` 2 (`a_guard_on_a_uniform_count_is_computed_once_per_call`,
  `a_fold_whose_body_reads_no_coordinate_is_carved_in_the_body`), `program::lower::tests` 9.
- 22 in `tests/value_bounds_jit.rs`: the integration binary reports `22 passed` on each tier.
- The AVX-512-only half of `a_guard_computed_once_per_call_is_read_again_in_every_batch` (reload and
  frame growth) is gated on `isa::detect() == Avx512` in the test; the arm-test half runs on every tier.

### isa-matrix --smoke: not run as written
`xtask isa-matrix --smoke` first runs `cargo test --workspace --no-run` (xtask/src/main.rs:827), which the
ground rules forbid and which filled the disk in round 1. Replica of its three per-level commands, built
with its exact `-p` set in one invocation (a tmpfs target, `RUSTC_WRAPPER` empty, so the disk was not touched),
once per tier:

| command | avx2 | avx512 |
|---|---|---|
| `test -p pixelflow-codegen -p pixelflow-ir -p pixelflow-core -p pixelflow-compiler -p pixelflow-pipeline --no-fail-fast` | 1266 passed, 11 ignored | 1266 passed, 11 ignored |
| `test -p pixelflow-graphics --test run_is_a_glyph --test font_rasterization_regression --no-fail-fast` | 13 passed | 13 passed |
| `test -p pixelflow-graphics --lib --no-fail-fast -- fonts::loop_blinn::` | 20 passed | 20 passed |

## 6. Build gates
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`: exit 0, no warning
  (target /dev/shm/a4-t199; every workspace package cleaned first so all 14 were re-checked; the one
  `warning:` line is core-term's build-script note).
- `cargo fmt --all -- --check`: exit 0, empty.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features`: ok. Same for pixelflow-search: ok
  (both re-checked from cleaned artifacts).
- `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`: ok.
- rustdoc compare: pixelflow-ir base 38 head 38, NEW 0, GONE 0; pixelflow-search base 28 head 28, NEW 0, GONE 0
  (compare exit 0 for both).

## 7. Scripts (all exit 0)
check-intent (51 intent files), check-doc-paths (682 paths; prints `grep: ../GEMINI.md: No such file or
directory`, noise, exit 0), check-cfg-encapsulation (121 grandfathered, none new), check-bin-declarations (29),
check-emit-boundary (self-test 66 cases; 25 files), check-feature-matrix (cargo-hack on PATH; "every crate's every
feature builds in isolation").

## 8. Greps (full output: v2/greps.txt)
- `VALUE_BOUND` in *.rs: 0 hits. Verdict: pass.
- `KNOWN_RANGE` outside tests: `fold.rs:1149` (the definition), `passes/lattice.rs:31,222,227,342,347,352`
  (lattice folds), `codegen program/lower.rs:12,497` (import and the lane fold), `glyph_phase_split.rs:45,80,81`
  (the production glyph example, with the reason now stating it). The other hits (compiler `lower.rs:1062-1442`,
  `fold.rs:1676-1912`, `passes.rs:1531-2114`, `lattice.rs:399`, `fold_rules.rs:885-1450`) are inside
  `#[cfg(test)] mod tests`, checked by file offset. None in `emit/select.rs` or compiler `emit.rs`. Verdict: matches.
- Known-helper grep, as amended by C4: `fn known(bound` has 0 hits (the helper is gone); the expect forms
  (`known_range().expect`, `KnownRange::of(..).expect`, `.expect(KNOWN_RANGE)`) hit `lower.rs:497` only. The
  unanchored `known(` also hits `emit/traffic.rs:123,235,347` (`Trips::known`), the other expected file. The spec's
  literal `-E` line with `\|` matches nothing as typed, which is why C4 re-anchors it. Verdict: pass.
- `grep -rnwE 'A4' --include=*.rs pixelflow-*`: 0 hits. `grep -rnw A4 --include=intent.md .`: 0 hits. Nothing
  says "until A4" in any tracked source or doc except `docs/language/2026-10-20-a-bound-is-a-value.md:626`,
  "the key refused a value bound until A4 gave it a marker", which is past tense and true after A4. The plan's
  other A4 mentions are its work-plan and its table of doc rewrites. Verdict: pass.
- `expect(dead_code, reason = "pub(crate) until A4` in pixelflow-ir: 0 hits.
- `iter::empty()` in pipeline, graphics, compiler/src: `pixelflow-compiler/src/emit.rs:605` only, the literal
  branch. Verdict: pass.
- `fold_header(id).to_bits()` (6 hits) plus the two sites that grep misses (C12: compiler `emit.rs:582`,
  `corpus_gaps.rs:509`): each read in context and each also writes or attaches the value children. test_support:37
  (the digest feeds `arena.children(id)` after), cell_grid:1219 and production_glyph_arena_dump:178 (print the `lo`
  and `hi` values after the body), key.rs:461 (pushes body then each bound value), training corpus:416 (body, then
  `lo`'s child, then `hi`'s), collapse corpus:291 (as cell_grid), compiler emit.rs:582 (builds `[#(#values),*]`
  and `attach`es it), corpus_gaps:509 (`ids` is body then values; `header.attach(values)`). Verdict: pass.
- `FoldHeader::from_bits` decoders: training corpus:612 reads `header.arity() - 1` children after the body;
  collapse corpus:516 reads the line's values and asserts their count equals `arity() - 1`; compiler emit.rs:585
  attaches `#children`; emit.rs:735 reads only the monoid of a structural (literal-range) fold. Verdict: pass.
- `trips` in `pixelflow-codegen/src/emit`: every per-loop count is a `Trips` (`traffic.rs:153`, `select.rs:308`,
  `build.rs:323`, `mod.rs:529`); the one `u64` is `LoopFacts.trips` in `regalloc/local.rs:134`, set from
  `l.trips.weight()` at :202. Verdict: pass.
- `dynamic_memory_ops`: callers `constant_traffic.rs:50` (maps `None` to "an estimate"), `collapse_bench/mod.rs:341`
  (`expect(LITERAL_FOLDS)`, its corpus has literal folds), tests assert `Some` and `None`. Verdict: every caller
  handles the `Option`.
- Old texts: `has no bits")` (the exact old text, C3): 0 hits. The spec's looser alternation hits
  `program/lower.rs:202` and `pixelflow-ir/src/kernel.rs:1292`, both the shift-count sentence "a 32-bit lane has no
  bits there", unrelated. `no corpus encoding before`, `no code and no key`: 0 hits.
- `grep -rln '\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'`: empty.
- Added lines of the A4 diff (3,536) in tracked non-.wip files: no `.wip` path; no model name (the one
  case-insensitive hit is `CLAUDE.md`, the repository file name, in a doc comment); labels E8 and C8 in
  `docs/BACKLOG.md` are BACKLOG row ids (the table has E1 to E8); "Ruling 6" in that BACKLOG row and in the plan
  is not new (both hold it at a747770).
- Reduce-site checklist (`.wip/a4/gen/checklist.py`): classified 190, A4 0, A5 0, UNCLASSIFIED 0, STALE 0.
- `git lfs ls-files | grep -c .`: 68.

## 9. Diffstat (a747770 against HEAD, excluding .wip)
44 files changed, 3536 insertions(+), 381 deletions(-) (v2/diffstat.txt).

## Environment notes
- RAM, not disk, was the limit: /dev/shm held about 14 GB of earlier rounds' build caches, so the container's
  memory cgroup was full and cargo, rustc, ld and a test binary were OOM-killed (first attempt at pipeline, core,
  runtime). I ran `cargo clean` against `/dev/shm/a4-tdbg` (a cargo target dir of the previous round, 13.9 GiB),
  which freed it; nothing under /tmp/claude-0, /home/user/scratch or .wip was touched. `/dev/shm/a4-t199` and
  `/dev/shm/a4-tchk` were kept and reused. Crate tests for pipeline, core and runtime then ran on the default
  target dir; ir, search, codegen, compiler and graphics ran on a4-tdbg before it was cleaned (their logs are here).
- One unified `-p` test invocation on the disk target consumed about 3 GB before a guard script stopped it at
  2.1 GB free; the replica was rerun on a tmpfs target with `RUSTC_WRAPPER` empty. Disk ended at 9.5 GB free.
- Tracked files were not edited. The gate scripts restored their own probe file each time.
