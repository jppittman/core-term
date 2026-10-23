# A3 gates, polish round 2 (TAG=a3-p2)

Code base `3bfdaded`; the A3 diff is `git diff 3bfdaded -- . ':!.wip'`. HEAD was `d4161cd` for the whole run (`origin/claude/integral-area-work-ojiej6`
was the same commit before and after). The tree under test is polish round 2's: relative to the tree the p1 gates ran on, the only source file that
moved is `pixelflow-search/src/egraph/graph.rs` (16 lines), so the three byte gates and gate 5 rebuilt the probe binary (`Compiling pixelflow-graphics`
in `/home/user/scratch/gates-bin/build.*.log`). Host: 4 cores, AVX-512, `cargo 1.97.0` (default stable); clippy with `cargo +1.99.0` (CI's). No
tracked file was edited. `df -h /` showed 11 GB free before every build. `git lfs ls-files | grep -c .` is 68; `git status --short pixelflow-graphics`
is empty after the probes.

One slip, corrected: the grep gate's first output went to `greps-p2.txt`, a name polish round 2 had already committed. The committed content was
written back byte for byte from `git show HEAD:` (`git status` shows the file unmodified) and this round's grep output lives in `greps-gates-p2.txt`
and `greps-added-p2.txt`.

## Byte/key gates (outputs `../../gates/out/*.a3-p2.*`)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-p2` | exit 0; avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b `gate1b.sh a3-p2` | exit 0; 29 lines IDENTICAL to `gate1b.base.txt`; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-p2` | exit 0; IDENTICAL to `gate2.base.txt` (29 records) |
| 2x `gate2x.sh a3-p2` | exit 0; IDENTICAL to `gate2x.base.txt` (extraction fields) |

## Compile cost (`measure.sh a3-p2`; `../../gates/out/measure.a3-p2.txt`)

| metric | base | a3-p2 | change |
|---|---|---|---|
| total Ir | 6,295,969,957 | 6,307,995,948 | +0.191 percent (+12.03 M Ir; limit +0.5): PASS |
| `Saturation::saturate` inclusive | 2,079,429,510 | 2,076,970,165 | -0.118 percent |
| `extract_dag_over::<CostModel>` inclusive | 1,123,555,965 | 1,138,113,466 | +1.296 percent (+14.56 M Ir; reported, not part of the limit) |
| median peak RSS KiB (five runs 96,016 to 98,332) | 96,476 | 96,760 | +0.294 percent (reported only) |
| median wall ms (five runs 1116 to 1414; reported only, the VM drifts) | 956 | 1278 | +33.7 percent |

Total Ir differs from p1's measurement (6,307,995,698) by 250 Ir, which is callgrind's own run-to-run noise (about 4e-6 percent of the total); the
`saturate` and `extract_dag_over` inclusive figures are identical to p1's, digit for digit. Round 2's `graph.rs` edit did not move the literal path.

## Tests per crate (`cargo test -p <crate>`; `tests-p2-summary.txt`, logs `tests-p2.<crate>.txt`)

| crate | passed | failed | ignored |
|---|---|---|---|
| pixelflow-ir | 375 | 0 | 2 |
| pixelflow-search (default) | 434 | 0 | 15 |
| pixelflow-search `--all-features` | 437 | 0 | 15 |
| pixelflow-search `--features saturation-telemetry` | 437 | 0 | 15 |
| pixelflow-codegen | 161 | 0 | 0 |
| pixelflow-compiler | 304 | 0 | 6 |
| pixelflow-graphics | 236 | 0 | 15 |
| pixelflow-pipeline | 221 | 0 | 2 |
| pixelflow-core | 136 | 0 | 1 |
| pixelflow-runtime | 152 | 0 | 6 |

Every `cargo test` exited 0 with no compiler warning in any log; the counts equal p1's. No `cargo test --workspace`.

## Lints and docs

- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` (`CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`): exit 0, no
  warning (`clippy-p2-199-workspace.txt`; the only `warning:` line is core-term's `build.rs` note "Building with X11 display driver"). Crates whose
  sources and flags did not change since the previous run were replayed from cargo's fingerprint cache, which replays their diagnostics; the rest
  were checked in this run.
- `cargo fmt --all -- --check`: exit 0, empty output.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features`: exit 0. Same for `pixelflow-search`: exit 0.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`: exit 0.
- Rustdoc subset compare against `rustdoc-base.md`: pixelflow-ir `warnings: base 38, head 38`, `NEW 0`, `GONE 0`; pixelflow-search
  `warnings: base 28, head 28`, `NEW 0`, `GONE 0`. PASS.

## Scripts

The script gates ran twice: once while gate 5's build held the probe appended to `tests.rs`, and again, whose output is recorded, on the clean tree.

- `scripts/check-intent.sh`: exit 0 (51 intent files).
- `scripts/check-doc-paths.sh`: exit 0 (674 source paths across 170 docs; 1 accepted via baseline); the `cannot open ../<file>.md` lines are the
  script's own relative reads, as in every earlier round.
- `scripts/check-cfg-encapsulation.sh`: exit 0, no new violations (121 grandfathered).
- `scripts/check-bin-declarations.sh`: exit 0 (all 29 `src/bin` targets declared).
- `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH scripts/check-feature-matrix.sh`: exit 0, "every crate's every feature builds in isolation".
  Checksums of `Cargo.lock` and every `*/Cargo.toml` equal before and after; no tracked file modified.
- `python3 -I .wip/a3/gen/checklist.py`: 175 sites, UNCLASSIFIED 0, STALE 0 (already-correct 38, changed-by-A3 26, left 13, tests 78, n/a 5, A4 15,
  A5 0).

## Greps (`greps-gates-p2.txt`, `greps-added-p2.txt`; `target.noindex`, `.git` and `.wip` excluded, and also run literally)

| grep | hits | verdict |
|---|---|---|
| `grep -rnwE 'A3' --include=*.rs pixelflow-*` | none | PASS (empty) |
| `grep -nw 'A3' pixelflow-ir/intent.md` | none | PASS (empty) |
| `grep -rn 'fold_cost(' --include=*.rs` | 10 lines, all in `pixelflow-search/src/egraph/cost.rs` | PASS (only cost.rs) |
| `grep -rln '\.wip' --include=*.rs --include=*.md . \| grep -v '^./.wip/'` | none (also none with `target.noindex` included) | PASS (empty) |
| added lines of the A3 diff naming a `.wip` path | none | PASS |
| added lines of the A3 diff naming a model | one `///` line in `fold_rules.rs` citing the repository's instruction file by name | PASS (a file name, as in earlier rounds) |
| added lines naming a ruling label | none; the `A6`, `B2`, `B3` hits are the plan's own steps (`docs/language/2026-10-20-a-bound-is-a-value.md`) and `C10` is the new BACKLOG row, all in tracked docs; the `finding` hits are the noun in prose | PASS |

## Notes

- The A3 diff lists `Cargo.lock` (27 files, +4538 -484) because of the other agent's `f124eae` (tokio bump); it is not A3's and changes no source.
- Nothing was edited: `git status --short` before the commit shows only untracked files under `.wip/a3/run/` and `.wip/gates/out/`.

## Verdict

Every byte/key gate is identical to base; compile cost is +0.191 percent (limit +0.5; extraction inclusive +1.296 percent, reported); every test,
clippy (1.99.0), fmt, no-default-features, doc, rustdoc, script, checklist and grep gate passes.
