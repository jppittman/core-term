# A3 gates, round 1 (TAG=a3-r1)

Code base `3bfdaded`; HEAD `29a00e5` (A3 diff = `git diff 3bfdaded -- . ':!.wip'`, 24 files, +3189 -417; stat in
`diffstat-r1.txt`). Host: 4 cores, AVX-512, rustc/cargo/clippy 1.97.0 (stable). No tracked file was edited, and
`git status --short pixelflow-graphics` is empty after the probes.

## Byte/key gates (outputs in `../../gates/out/*.a3-r1.*`)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-r1` | exit 0; avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b `gate1b.sh a3-r1` | exit 0; 29 lines IDENTICAL to `gate1b.base.txt`; self-check n=2 `22ad8819798461a6` and n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-r1` | exit 0; IDENTICAL to `gate2.base.txt` (29 records) |
| 2x `gate2x.sh a3-r1` | exit 0; IDENTICAL to `gate2x.base.txt` |

## Gate 4, compile cost (`measure.sh a3-r1`, callgrind on `zz_prof_all_cold`)

| metric | base | head | change |
|---|---|---|---|
| total Ir | 6,295,969,957 | 6,305,896,638 | +0.158 percent (limit +0.5): PASS |
| `Saturation::saturate` inclusive | 2,079,429,510 | 2,076,320,833 | -0.149 percent |
| `extract_dag_over::<CostModel>` inclusive | 1,123,555,965 | 1,136,559,328 | +1.157 percent (+13.0 M Ir) |
| median peak RSS KiB | 96,476 | 97,184 | +0.734 percent |
| median wall ms (reported only) | 956 | 1014 | +6.1 percent (VM drift) |

The total is inside the budget. The whole +9.93 M Ir of the total sits in `extract_dag_over` (+13.0 M) less saturation
(-3.1 M). `cg-diff.a3-r1.txt` is the inclusive-profile diff (`cg-annotate.{base,a3-r1}.txt`). What moved, by name:

- Symbol renames with the same work: `drop_glue::<(ExprArena, ExprId)>` -62.8 M becomes `drop_glue::<KernelData>` +66.5 M
  (`drop_glue::<ExprArena>` +62.6 M inclusive reads as the same drop under the type that now owns the arena; inferred from names, not from source);
  `drop_glue::<ReachSets>` -29.8 M becomes `drop_glue::<SharedPricer<..>>` +29.8 M; the `TreePricer` `Map<Iter<EClassId>..>`
  entry (45.9 M) moves under a new monomorphization name; the two `ENode::variance::<Selection::price::{closure#0|#1}>`
  entries (10.7 M and 10.8 M) swap places.
- Not a rename: `CostModel::node_op_cost` -17.9 M (no longer a separate frame at head; its `Reduce` arm now calls
  `fold_cost(fold, header.priced_trips(), 0)`) and `Settlement<TreePricer<..>>` +4.7 M inclusive. The +13.0 M inside
  `extract_dag_over` is the sum of these shifts plus the renamed monomorphizations; it is not attributed to one function,
  and no function that should be absent from the all-literal path shows up as new work. The total is inside the +0.5 percent
  budget, so this is reported, not a failure; the +1.157 percent on extraction alone is the figure to watch if a later change
  adds to it.

## Gate 5, tests per crate (`cargo test -p <crate>`, logs in scratch, summary `tests-r1-summary.txt`)

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

Every `cargo test` exited 0. No `cargo test --workspace`.

## Gate 6, lints and docs

- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: **EXIT 101, 3 errors, all
  `clippy::must_use_candidate`, none on a line A3 touched**: `pixelflow-ir/src/arena.rs:1142` (`ExprArena::nodes`, present at
  base as line 1067, blame 057f7fe which is an ancestor of the base) and `pixelflow-ir/src/dag.rs:226` and `:579`
  (`dag.rs` is not in the A3 diff). Clippy 1.97.0 flags a `pub fn` returning `impl Iterator`; the workspace sets
  `must_use_candidate = "warn"`, so `-D warnings` makes it an error. The failure stops the build of `pixelflow-ir`, so the
  literal command lints nothing downstream. Two supplementary runs (not the gate command) cover the rest:
  `-- -D warnings -A clippy::must_use_candidate` exits 0 for the whole workspace, and a plain
  `cargo clippy --workspace --all-targets --all-features` (warnings not denied) reports exactly those 3 warnings and no other.
  So A3 adds no clippy finding; the 3 are pre-existing under this toolchain, derived from unchanged lines (a clippy run on a
  base build was not possible: no worktree, no checkout). First errors: `clippy-r1-first-errors.txt`.
- `cargo fmt --all -- --check`: exit 0.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features`: exit 0.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features`: exit 0.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`: exit 0.
- Rustdoc by the C4 method (`rustdoc-base.md`): pixelflow-ir `warnings: base 38, head 38`, `NEW 0`, `GONE 0`;
  pixelflow-search `warnings: base 28, head 28`, `NEW 0`, `GONE 0`. PASS (no new warning).

## Gate 7, scripts

- `scripts/check-intent.sh`: OK (51 intent files).
- `scripts/check-doc-paths.sh`: OK (672 source paths across 170 docs; 1 accepted via baseline). It prints six `head`/`grep: cannot open ../AGENTS.md|CLAUDE.md|GEMINI.md`
  lines on stderr (the script's own relative reads of root instruction files) and still exits 0; not an A3 matter.
- `scripts/check-cfg-encapsulation.sh`: OK, no new violations (121 grandfathered).
- `scripts/check-feature-matrix.sh`: NOT RUN. It needs `cargo hack`, which is not installed on this host
  (`error: no such command: hack`), and installing it is outside this task. The crate-level `--no-default-features` checks above
  and the three `pixelflow-search` test configurations cover the two touched crates' feature axes only in part.
- `python3 -I .wip/a3/gen/checklist.py`: 169 sites, 0 UNCLASSIFIED, 0 STALE (already-correct 39, changed-by-A3 22, left 13,
  tests 75, n/a 5, A4 15, A5 0).

## Gate 8, spec section 11 greps (run from the repository root; `target.noindex` and `.wip` excluded)

| grep | hits | verdict |
|---|---|---|
| `grep -rnwE 'A3' --include=*.rs pixelflow-*` | none | PASS (empty) |
| `grep -nw 'A3' pixelflow-ir/intent.md` | none | PASS (empty) |
| `grep -rn 'fold_cost(' --include=*.rs .` | 10 lines, all in `pixelflow-search/src/egraph/cost.rs` (405, 419, 432, 743, 745, 767, 770, 777, 816, 822) | PASS (only cost.rs) |
| `grep -rn 'real_sorted\|var_variance' --include=*.rs .` | none | PASS (empty) |
| `grep -rn 'VALUE_BOUND' --include=*.rs .` | `pixelflow-ir/src/fold.rs` (981 doc link, 986 panic, 1088 definition, 1096 doc link) and `pixelflow-codegen/src/program/lower.rs` (12 import, 51 panic) | PASS (fold.rs, program/lower.rs and doc links only) |
| `grep -rln '\.wip' --include=*.rs --include=*.md . \| grep -v '^./.wip/'` | none | PASS (empty) |
| added lines of the A3 diff matching `.wip`, a model name | none (the only match is the `.claude/skills/...` path in a diff header) | PASS |

## Gate 9

`git diff --stat 3bfdaded -- . ':!.wip'`: see `diffstat-r1.txt` (24 files, 3189 insertions, 417 deletions).
`git lfs ls-files | grep -c .` is 68 before and after; `*.jsonl` and `cg.*.out` stay ignored under `.wip/`.
