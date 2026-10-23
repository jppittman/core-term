# A3 gates, polish round 1 (TAG=a3-p1)

Code base `3bfdaded`; the A3 diff is `git diff 3bfdaded -- . ':!.wip'`. HEAD was `dd77200` when the run started and `f124eae` from 15:16:26 on, when
another agent committed and pushed a `Cargo.lock` bump (tokio 1.53.2 to 1.53.3, which reaches the workspace only through `desloppify`'s HTTP
dependencies, plus six crates' `windows-sys` pick). `git diff dd77200 f124eae --stat` is that one file, 9 lines each way, so every source file
the gates read is the polish-round-1 tree. The lockfile moved while this run was in progress (mtime 15:16:09), so gates 1, 1b and 2 were run a
second time after every other gate had finished, at `f124eae`, and the table below is that second run (the first run was identical too). Gate 5 and
every cargo-driven gate other than those three ran after the bump. Host: 4 cores, AVX-512, `cargo 1.97.0` (default stable); clippy with
`cargo +1.99.0` (CI's). No tracked file was edited. `df -h /` showed 11 GB free before every build. `git lfs ls-files | grep -c .` is 68;
`git status --short pixelflow-graphics` is empty after the probes.

## Byte/key gates (outputs `../../gates/out/*.a3-p1.*`)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-p1` | exit 0; avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b `gate1b.sh a3-p1` | exit 0; 29 lines IDENTICAL to `gate1b.base.txt`; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-p1` | exit 0; IDENTICAL to `gate2.base.txt` (29 records) |
| 2x `gate2x.sh a3-p1` | exit 0; IDENTICAL to `gate2x.base.txt` (extraction fields) |

## Compile cost (`measure.sh a3-p1`; `../../gates/out/measure.a3-p1.txt`)

| metric | base | a3-p1 | change |
|---|---|---|---|
| total Ir | 6,295,969,957 | 6,307,995,698 | +0.191 percent (limit +0.5): PASS |
| `Saturation::saturate` inclusive | 2,079,429,510 | 2,076,970,165 | -0.118 percent |
| `extract_dag_over::<CostModel>` inclusive | 1,123,555,965 | 1,138,113,466 | +1.296 percent (+14.6 M Ir; reported, not part of the limit) |
| median peak RSS KiB (five runs 96,264 to 98,400) | 96,476 | 97,796 | +1.368 percent (reported only) |
| median wall ms (five runs 1104 to 1329; reported only, the VM drifts) | 956 | 1249 | +30.6 percent |

Total Ir differs from polish1's measurement (6,307,995,455) by 243 Ir, which is callgrind's own run-to-run noise (about 3e-5 percent); the
`saturate` and `extract_dag_over` inclusive figures are identical to polish1's, digit for digit. The literal path did not move since polish1.

## Tests per crate (`cargo test -p <crate>`; `tests-p1-summary.txt`, logs `tests-p1.<crate>.txt`)

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

Every `cargo test` exited 0 with no compiler warning in any log; the counts equal polish1's for ir, search, codegen and compiler, and the
rest equal round 3's. No `cargo test --workspace`.

## Lints and docs

- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` (`CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`): exit 0, no
  warning (`clippy-p1-199-workspace.txt`; the only `warning:` line is core-term's `build.rs` note "Building with X11 display driver"). The member
  crates whose sources did not change since the previous run were replayed from cargo's fingerprint cache (same sources, same flags), the rest
  was checked.
- `cargo fmt --all -- --check`: exit 0, empty output.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features`: exit 0. Same for `pixelflow-search`: exit 0.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`: exit 0.
- Rustdoc subset compare against `rustdoc-base.md`: pixelflow-ir `warnings: base 38, head 38`, `NEW 0`, `GONE 0`; pixelflow-search
  `warnings: base 28, head 28`, `NEW 0`, `GONE 0`. PASS.

## Scripts

- `scripts/check-intent.sh`: exit 0 (51 intent files).
- `scripts/check-doc-paths.sh`: exit 0 (674 source paths across 170 docs; 1 accepted via baseline); the `cannot open ../<file>.md` lines are the
  script's own relative reads, as in every earlier round.
- `scripts/check-cfg-encapsulation.sh`: exit 0, no new violations (121 grandfathered).
- `scripts/check-bin-declarations.sh`: exit 0 (all 29 `src/bin` targets declared).
- `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH scripts/check-feature-matrix.sh`: exit 0, "every crate's every feature builds in isolation"
  (43 s). Checksums of `Cargo.lock` and every `*/Cargo.toml` equal before and after; no tracked file modified.
- `python3 -I .wip/a3/gen/checklist.py`: 175 sites, UNCLASSIFIED 0, STALE 0 (already-correct 38, changed-by-A3 26, left 13, tests 78, n/a 5, A4 15,
  A5 0).

## Greps (`greps-p1.txt`, `greps-added-p1.txt`; `target.noindex`, `.git` and `.wip` excluded, and also run literally)

| grep | hits | verdict |
|---|---|---|
| `grep -rnwE 'A3' --include=*.rs pixelflow-*` | none | PASS (empty) |
| `grep -nw 'A3' pixelflow-ir/intent.md` | none | PASS (empty) |
| `grep -rn 'fold_cost(' --include=*.rs` | 10 lines, all in `pixelflow-search/src/egraph/cost.rs` | PASS (only cost.rs) |
| `grep -rln '\.wip' --include=*.rs --include=*.md . \| grep -v '^./.wip/'` | none | PASS (empty) |
| added lines of the A3 diff naming a `.wip` path | none | PASS |
| added lines of the A3 diff naming a model | one `///` line in `fold_rules.rs` citing the repository's instruction file by name | PASS (a file name, as in earlier rounds) |
| added lines naming a ruling label | none; the `A6`, `B2`, `B3`, `C10` hits are the plan's own steps and the new BACKLOG row, all in tracked docs | PASS |

## Notes

- The A3 diff now lists `Cargo.lock` (27 files, +4535 -481) because of the other agent's `f124eae`; it is not A3's, and it changes no source.
- Nothing was edited: `git status --short` before the commit shows only untracked files under `.wip/a3/run/` and `.wip/gates/out/`.

## Verdict

Every byte/key gate is identical to base; compile cost is +0.191 percent (limit +0.5); every test, clippy (1.99.0), fmt, no-default-features, doc,
rustdoc, script, checklist and grep gate passes.
