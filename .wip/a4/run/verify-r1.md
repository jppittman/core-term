# A4 verification, round 1 (TAG=a4-r1)

Gates ran at 1c8c8c7 (clean tree). HEAD then moved to e31861b (fa31236 changed only
pixelflow-codegen/tests/value_bounds_jit.rs; e31861b only rulings.md); clippy, fmt, the codegen
reruns and the per-tier smoke replica below ran at e31861b. No src/ file differs between the two.

## Byte/key gates: all exit 0, cmp-equal to base
1 byteprobe (avx512 + avx2, 127 lines each); 1b (29 lines); 2 (29 lines); 2x (29 records);
`git status --short pixelflow-graphics` empty after each.
Gate 4/measure: total Ir 6,317,448,245 vs base 6,310,238,494 = +0.114% (limit +0.5%);
saturate -0.001%, extract_dag_over -0.013%, RSS -0.008%, wall -5.8% (drift, not decisive).

## Crate tests (all pass, 0 failed)
ir 396 (+2 ign); search default 434, --all-features 437, saturation-telemetry 437 (+15 ign each);
compiler 306; graphics 239 (+15 ign); pipeline 228; core 136; runtime 152.
codegen, avx2 and avx512 alike, at 1c8c8c7 and again at e31861b: 200 passed, 0 failed.
New codegen tests, all ok on both tiers (list: run/r1/new_codegen_tests.json): 40 = 18 unit
(avx512 1, emit/mod 1, traffic 5, pipeline 2, program/lower 9) + 22 in tests/value_bounds_jit.rs.
Of those, 6 unit + 1 integration are should_panic (all ok).

## isa-matrix --smoke: does NOT run locally
It runs `cargo test --workspace --no-run` first. That wrote ~13 GB into target.noindex and filled
the disk (ENOSPC, linker Bus error; `test build FAILED`). Run on my side without reading the
code first; the ground rule against workspace tests applies to it. Replica of its three per-level
commands (codegen+ir+core+compiler+pipeline; graphics run_is_a_glyph + font_rasterization_regression;
graphics --lib fonts::loop_blinn::) on a tmpfs target dir, PIXELFLOW_ISA=avx2 and =avx512, at
e31861b: 1266 + 13 + 20 passed, 0 failed, on each tier.

## Build gates (tmpfs targets; the disk was full)
clippy 1.99 --workspace --all-targets --all-features -D warnings: exit 0. fmt --check: clean.
RUSTFLAGS=-D warnings check --no-default-features: ir ok, search ok.
RUSTDOCFLAGS=-D warnings doc codegen --document-private-items: ok.
rustdoc compare: ir base 38 head 38 NEW 0 GONE 0; search base 28 head 28 NEW 0 GONE 0.
Scripts exit 0: check-intent (51 files), check-doc-paths, check-cfg-encapsulation, check-bin-declarations,
check-emit-boundary, check-feature-matrix (cargo-hack on PATH). check-doc-paths prints
`head: cannot open '../GEMINI.md'` and exits 0 (noise, GEMINI.md is at the root).

## Greps (full output: run/r1/greps.txt)
VALUE_BOUND: none. KNOWN_RANGE non-test: fold.rs (def), passes/lattice.rs 222-352, codegen
program/lower.rs:497 (lane fold), glyph_phase_split.rs:81; none in emit/select.rs or compiler emit.rs
(the other hits sit in mod tests). Known-helper grep (C4, `fn known(bound|...expect`): lower.rs:497 only;
the unanchored `known(` also hits traffic.rs:122,234 (`Trips::known`). The spec's -E line with `\|` matches
nothing as typed. `A4` word: 0 hits in *.rs and intent.md. dead_code-until-A4: none. iter::empty():
compiler emit.rs:605 only. Old texts (C3 `has no bits")`): none (loose `has no bits` hits lower.rs:202 and
kernel.rs:1292 are other sentences). `.wip` cited outside .wip: none; added lines hold no model name.
to_bits: 6 `fold_header(id).to_bits()` hits all write the value children (test_support via children(),
cell_grid, dump test, key.rs, training corpus, collapse corpus); the two the grep misses (C12) are
compiler emit.rs:582 and corpus_gaps.rs:509, both attach the children. from_bits decoders: training
corpus.rs:612 and collapse corpus.rs:516 read arity-1 children; compiler emit.rs:585 attaches `#children`;
:735 reads only the monoid. trips: every per-loop count is a Trips; LoopFacts.trips is u64 from
`l.trips.weight()`. dynamic_memory_ops: constant_traffic maps None to "an estimate", collapse_bench
expects with LITERAL_FOLDS, tests assert None. git lfs ls-files: 68.
Added-line labels E8/C8 in docs/BACKLOG.md are BACKLOG row ids, not ruling labels.

## Findings
1. Reduce-site checklist (.wip/a4/gen/checklist.py, run at e31861b): UNCLASSIFIED 1, STALE 1, A4 0.
   Same site: pixelflow-pipeline/src/bin/corpus_gaps.rs:956 in hash_cons_keeps_a_value_bound; the test
   text changed after classes.json was written. Bookkeeping only, in .wip.
2. isa-matrix --smoke cannot run here (above). Side effect: target.noindex/debug holds ~13 GB of
   artifacts from the failed run (files newer than 19:55:02, including truncated ones). A deletion
   of them was refused by the permission layer and not retried; `/` has ~0.2 MB free.
