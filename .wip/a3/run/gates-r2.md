# A3 gates, round 2 (TAG=a3-r2)

Code base `3bfdaded`; HEAD `cc913d5` for every result below (the A3 diff is `git diff 3bfdaded -- . ':!.wip'`:
24 files, +3217 -419; `diffstat-r2.txt`). Host: 4 cores, AVX-512, rustc/cargo/clippy 1.97.0 (stable). No tracked file was
edited by this run. `git status --short` shows nothing outside `.wip/` after every step, and `git status --short pixelflow-graphics`
is empty after the probes. `git lfs ls-files | grep -c .` is 68 before and after.

HEAD moved once during the run: `cc913d5` (09:30:32) removed the three `#[must_use]` attributes of `ExprArena::nodes`,
`Dag::iter` and `Rooted::entries` (files saved 09:30:20). The first pass of gates 1, 1b, 2, 2x and 4 had finished by 09:30:11, so
all five were run again on `cc913d5` and the files in `gates/out/*.a3-r2.*` are from that second pass. The per-crate tests were
compiled after the edit (a re-run of `cargo test -p pixelflow-ir` recompiled nothing).

## Byte/key gates (outputs in `../../gates/out/*.a3-r2.*`)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-r2` | exit 0; avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b `gate1b.sh a3-r2` | exit 0; 29 lines IDENTICAL to `gate1b.base.txt`; self-check n=2 `22ad8819798461a6` and n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-r2` | exit 0; IDENTICAL to `gate2.base.txt` (29 records) |
| 2x `gate2x.sh a3-r2` | exit 0; IDENTICAL to `gate2x.base.txt` |

## Gate 4, compile cost (`measure.sh a3-r2`, callgrind on `zz_prof_all_cold`)

| metric | base | head | change |
|---|---|---|---|
| total Ir | 6,295,969,957 | 6,305,897,048 | +0.158 percent (limit +0.5): PASS |
| `Saturation::saturate` inclusive | 2,079,429,510 | 2,076,320,833 | -0.149 percent |
| `extract_dag_over::<CostModel>` inclusive | 1,123,555,965 | 1,136,559,328 | +1.157 percent (+13.0 M Ir) |
| median peak RSS KiB | 96,476 | 97,156 | +0.705 percent |
| median wall ms (reported only) | 956 | 1097 | +14.7 percent (VM drift; the first pass of this round read 1013, +6.0 percent) |

`cg-diff.a3-r2.txt` is the inclusive-profile diff (`cg-annotate.{base,a3-r2}.txt`). It matches round 1's to within 400 Ir per line:
the same renames (`drop_glue::<(ExprArena, ExprId)>` to `drop_glue::<KernelData>`, `drop_glue::<ReachSets>` to
`drop_glue::<SharedPricer<..>>`, the `TreePricer` `Map<Iter<EClassId>..>` and the two `ENode::variance::<Selection::price::..>`
closures changing monomorphization name) and `CostModel::node_op_cost` -17.9 M (its `Reduce` arm now calls `fold_cost`). No function
that should be absent on the all-literal path appears as new work. The +13.0 M inside `extract_dag_over` is unchanged from round 1.

## Gate 5, tests per crate (`cargo test -p <crate>`; `tests-r2-summary.txt`)

| crate | passed | failed | ignored |
|---|---|---|---|
| pixelflow-ir | 375 | 0 | 2 |
| pixelflow-search (default) | 424 | 0 | 15 |
| pixelflow-search `--all-features` | 427 | 0 | 15 |
| pixelflow-search `--features saturation-telemetry` | 427 | 0 | 15 |
| pixelflow-codegen | 160 | 0 | 0 |
| pixelflow-compiler | 304 | 0 | 6 |
| pixelflow-graphics | 236 | 0 | 15 |
| pixelflow-pipeline | 221 | 0 | 2 |
| pixelflow-core | 136 | 0 | 1 |
| pixelflow-runtime | 152 | 0 | 6 |

Every `cargo test` exited 0 with no compiler warning. The counts equal round 1's. No `cargo test --workspace`.

## Gate 6, lints and docs

- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: **EXIT 101 on this host**, 3 errors, all
  `clippy::must_use_candidate`: `pixelflow-ir/src/arena.rs:1144` (`ExprArena::nodes`), `pixelflow-ir/src/dag.rs:226` (`Dag::iter`) and
  `:579` (`Rooted::entries`). These are the three `cc913d5` removed on purpose (its message: a newer clippy reports
  `double_must_use` on a `#[must_use]` over a function returning an `impl Iterator`). They are **not A3 lines and not A3 findings**:
  measured, not inferred, by extracting `git archive 3bfdaded` to `/home/user/scratch/a3-r2/base-src` (a plain directory, not a
  worktree) and running `cargo clippy -p pixelflow-ir --all-targets --all-features -- -D warnings` there with its own target dir:
  exit 101, the same three errors at `arena.rs:1067`, `dag.rs:226`, `dag.rs:579` (`clippy-base-ir-r2.txt`). So this host's clippy 1.97.0
  fails the literal command at base too, and the failure stops the build of `pixelflow-ir`, so the literal command lints nothing
  downstream. Supplementary runs: `-- -D warnings -A clippy::must_use_candidate` exits 0 for the whole workspace, and a plain
  `cargo clippy --workspace --all-targets --all-features` exits 0 with exactly those 3 warnings and no other. First errors:
  `clippy-r2-first-errors.txt`.
- `cargo fmt --all -- --check`: exit 0.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features`: exit 0.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features`: exit 0.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`: exit 0.
- Rustdoc by the C4 method (`rustdoc-base.md`): pixelflow-ir `warnings: base 38, head 38`, `NEW 0`, `GONE 0`; pixelflow-search
  `warnings: base 28, head 28`, `NEW 0`, `GONE 0` (`rustdoc-r2.pixelflow-{ir,search}.txt`). PASS.

## Gate 7, scripts

- `scripts/check-intent.sh`: exit 0 (51 intent files).
- `scripts/check-doc-paths.sh`: exit 0 (672 source paths across 170 docs; 1 accepted via baseline); the six `cannot open ../AGENTS.md|CLAUDE.md|GEMINI.md`
  lines on stderr are the script's own relative reads, as in round 1.
- `scripts/check-cfg-encapsulation.sh`: exit 0, no new violations (121 grandfathered).
- `scripts/check-feature-matrix.sh`: RAN, exit 0, 1 min 28 s ("every crate's every feature builds in isolation"), with
  `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH` (cargo-hack 0.6.45, installed outside the repo in round 1). No manifest or lock
  change left behind.
- `python3 -I .wip/a3/gen/checklist.py`: 169 sites, 0 UNCLASSIFIED, 0 STALE (already-correct 39, changed-by-A3 22, left 13, tests 75, n/a 5, A4 15, A5 0).

## Gate 8, spec section 11 greps (`greps-r2.txt`; `target.noindex`, `.git` and `.wip` excluded)

| grep | hits | verdict |
|---|---|---|
| `grep -rnwE 'A3' --include=*.rs pixelflow-*` | none | PASS (empty) |
| `grep -nw 'A3' pixelflow-ir/intent.md` | none | PASS (empty) |
| `grep -rn 'fold_cost(' --include=*.rs .` | 10 lines, all in `pixelflow-search/src/egraph/cost.rs` (405, 419, 432, 743, 745, 767, 770, 777, 816, 822) | PASS (only cost.rs) |
| `grep -rn 'real_sorted\|var_variance' --include=*.rs .` | none | PASS (empty) |
| `grep -rn 'VALUE_BOUND' --include=*.rs .` | `pixelflow-ir/src/fold.rs` (981 doc link, 986 panic, 1088 definition, 1096 doc link) and `pixelflow-codegen/src/program/lower.rs` (12 import, 51 panic) | PASS (fold.rs, program/lower.rs and doc links only) |
| `grep -rln '\.wip' --include=*.rs --include=*.md . \| grep -v '^./.wip/'` | none | PASS (empty) |
| added lines of the A3 diff matching `.wip` or a model name | one: a `///` line citing the repository's instruction file by name, `CLAUDE.md`, in `pixelflow-search/src/egraph/fold_rules.rs`; 3bfdaded already cites that file in tracked `.rs` files (`pixelflow-codegen/tests/edges_x86_64.rs` and others) | PASS (a file name, not a model identifier; no `.wip` path) |

## Gate 9

`git diff --stat 3bfdaded -- . ':!.wip'`: `diffstat-r2.txt` (24 files, 3217 insertions, 419 deletions). Round 1 had 24 files, 3189/417; the
difference is round 1's fixes (committed in `31d4b58`) and the `cc913d5` removal.
