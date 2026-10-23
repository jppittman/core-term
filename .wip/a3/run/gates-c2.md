# A3 closing gates, round 2 (TAG=a3-c2)

Code base `3bfdaded`; the A3 diff is `git diff 3bfdaded -- . ':!.wip'`: 26 files, +4281 -464 (`diffstat-c2.txt`). HEAD was `d4bcd11` when
the run started. No tracked file was edited; `git status --short pixelflow-graphics` is empty after the probes and `git status --short`
shows only `.wip/`. `git lfs ls-files | grep -c .` is 68. `df -h /` showed 13 GB free at every build (limit 2 GB).

## Byte/key gates (outputs in `../../gates/out/*.a3-c2.*`)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-c2` | exit 0; avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b `gate1b.sh a3-c2` | exit 0; 29 lines IDENTICAL to `gate1b.base.txt`; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-c2` | exit 0; IDENTICAL to `gate2.base.txt` (29 records) |
| 2x `gate2x.sh a3-c2` | exit 0; IDENTICAL to `gate2x.base.txt` |

## Compile cost (`measure.sh a3-c2`, `../../gates/out/measure.a3-c2.txt`)

| metric | base | head | change |
|---|---|---|---|
| total Ir | 6,295,969,957 | 6,308,132,943 | +0.193 percent (limit +0.5): PASS |
| `Saturation::saturate` inclusive | 2,079,429,510 | 2,076,709,606 | -0.131 percent |
| `extract_dag_over::<CostModel>` inclusive | 1,123,555,965 | 1,137,870,806 | +1.274 percent |
| median peak RSS KiB | 96,476 | 97,168 | +0.717 percent |
| median wall ms (reported only) | 956 | 1128 | +18.0 percent (VM drift) |

## Tests per crate (`cargo test -p <crate>`; `tests-c2-summary.txt`, logs `tests-c2.<crate>.txt`; every run exit 0, zero warnings)

| crate | passed | failed | ignored |
|---|---|---|---|
| pixelflow-ir | 375 | 0 | 2 |
| pixelflow-search (default) | 432 | 0 | 15 |
| pixelflow-search (--all-features) | 435 | 0 | 15 |
| pixelflow-search (--features saturation-telemetry) | 435 | 0 | 15 |
| pixelflow-codegen | 161 | 0 | 0 |
| pixelflow-compiler | 304 | 0 | 6 |
| pixelflow-graphics | 236 | 0 | 15 |
| pixelflow-pipeline | 221 | 0 | 2 |
| pixelflow-core | 136 | 0 | 1 |
| pixelflow-runtime | 152 | 0 | 6 |

## Lints and docs

| check | result |
|---|---|
| `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` | exit 0 (`clippy-c2-199.txt`; the one `warning:` line is core-term's build.rs note "Building with X11 display driver") |
| `cargo fmt --all -- --check` | exit 0, empty output |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features` | exit 0 |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features` | exit 0 |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` | exit 0 |
| rustdoc pixelflow-ir vs `rustdoc-base.pixelflow-ir.txt` | base 38, head 38, NEW 0, GONE 0 |
| rustdoc pixelflow-search vs `rustdoc-base.pixelflow-search.txt` | base 28, head 28, NEW 0, GONE 0 |

## Repository scripts and greps

| check | result |
|---|---|
| `scripts/check-intent.sh` | exit 0, 51 intent files |
| `scripts/check-doc-paths.sh` | exit 0, 672 source paths resolve |
| `scripts/check-cfg-encapsulation.sh` | exit 0, no new violations (121 grandfathered) |
| `scripts/check-bin-declarations.sh` | exit 0, 29 targets declared |
| `scripts/check-feature-matrix.sh` | exit 0. The first run (`feature-matrix-c2-nopath.txt`) failed with `no such command: hack` because `cargo-hack` is installed at `/home/user/scratch/cargo-hack-root/bin`, which this shell does not have on PATH; the rerun with that directory prepended to PATH for the one invocation passed (`feature-matrix-c2.txt`), as in round 1 |
| `python3 -I .wip/a3/gen/checklist.py` | UNCLASSIFIED 0, STALE 0 (`checklist-c2.txt`) |
| `grep -rnwE 'A3' --include=*.rs pixelflow-*` | empty |
| `grep -nw 'A3' pixelflow-ir/intent.md` | empty |
| `grep -rn 'fold_cost(' --include=*.rs .` | only `pixelflow-search/src/egraph/cost.rs` |
| `grep -rln '\.wip' --include=*.rs --include=*.md . \| grep -v '^./.wip/'` | empty (`greps-c2.txt`) |

All gates pass: A3 is byte-identical to base on today's kernels.
