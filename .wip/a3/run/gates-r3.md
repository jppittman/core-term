# A3 gates, round 3 (TAG=a3-r3)

Code base `3bfdaded`; the A3 diff is `git diff 3bfdaded -- . ':!.wip'`: 24 files, +3217 -419 (`diffstat-r3.txt`), byte-identical in
size to round 2's. HEAD was `c1cbe26` when the run started and `0f730eb` after the scratch-crate ignore commit; the two differ only
under `.wip/` (checked with `git diff --quiet c1cbe26 HEAD -- . ':!.wip'`, exit 0), so every result below is for the same code.
Host: 4 cores, AVX-512, rustc/cargo/clippy 1.97.0 (stable). No tracked file was edited. `git status --short pixelflow-graphics` is
empty after the probes; `git lfs ls-files | grep -c .` is 68 before and after. `df -h /` showed 15 GB free at every build.

## Byte/key gates (outputs in `../../gates/out/*.a3-r3.*`)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-r3` | exit 0; avx512 and avx2 (127 lines each) IDENTICAL to base |
| 1b `gate1b.sh a3-r3` | exit 0; 29 lines IDENTICAL to `gate1b.base.txt`; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-r3` | exit 0; IDENTICAL to `gate2.base.txt` (29 records) |
| 2x `gate2x.sh a3-r3` | exit 0; IDENTICAL to `gate2x.base.txt` |

## Compile cost (`measure.sh a3-r3`; `cg-annotate.a3-r3.txt`, `cg-diff.a3-r3.txt`)

| metric | base | head | change |
|---|---|---|---|
| total Ir | 6,295,969,957 | 6,305,898,607 | +0.158 percent (limit +0.5): PASS |
| `Saturation::saturate` inclusive | 2,079,429,510 | 2,076,320,833 | -0.149 percent |
| `extract_dag_over::<CostModel>` inclusive | 1,123,555,965 | 1,136,559,328 | +1.157 percent (+13.0 M Ir) |
| median peak RSS KiB | 96,476 | 97,496 | +1.057 percent |
| median wall ms (reported only) | 956 | 1116 | +16.7 percent (VM drift; same binary class as r2's 1097) |

The inclusive-profile diff equals round 2's to within ~2,000 Ir per line (total +9,928,650 here, +9,927,091 there). The movers are the
same: `drop_glue::<(ExprArena, ExprId)>` -62.8 M to `drop_glue::<KernelData>` +66.5 M (and `drop_glue::<ExprArena>` +62.6 M under it),
`drop_glue::<ReachSets>` -29.8 M to `drop_glue::<SharedPricer<..>>` +29.8 M, the `TreePricer` `Map<Iter<EClassId>..>` and the two
`ENode::variance::<Selection::price::..>` closures changing monomorphization name at equal cost, `CostModel::node_op_cost` -17.9 M (its
`Reduce` arm now calls `fold_cost`, so the symbol is gone and the cost sits in the caller), and `Settlement<TreePricer<..>>::relax` +4.7 M.
No function that should be absent on the all-literal path appears as new work; the +13.0 M inside `extract_dag_over` is the sum of those
renames and the `relax` growth, unchanged from rounds 1 and 2, and the total is inside the limit.

## Tests per crate (`cargo test -p <crate>`; `tests-r3-summary.txt`, logs `tests-r3.<crate>.txt`)

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

Every `cargo test` exited 0 with no compiler warning; counts equal round 2's. No `cargo test --workspace`.

## Lints and docs

- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: **EXIT 101 on this host**, 3 errors, all
  `clippy::must_use_candidate`, at `pixelflow-ir/src/arena.rs:1144` (`ExprArena::nodes`), `dag.rs:226` (`Dag::iter`), `dag.rs:579`
  (`Rooted::entries`) (`clippy-r3.txt`). Not A3 findings: re-measured this round on the pristine base source
  (`git archive 3bfdaded` extracted to a plain directory, own target dir): `cargo clippy -p pixelflow-ir --all-targets --all-features
  -- -D warnings` exits 101 with the same three errors at `arena.rs:1067`, `dag.rs:226`, `dag.rs:579` (`clippy-base-ir-r3.txt`). The
  failure stops the build of `pixelflow-ir`, so the literal command lints nothing downstream. Supplementary: `-- -D warnings -A
  clippy::must_use_candidate` exits 0 for the whole workspace (`clippy-r3-allow-must-use-candidate.txt`); a plain `cargo clippy
  --workspace --all-targets --all-features` exits 0 with exactly those three warnings and no other (`clippy-r3-plain.txt`).
  The attributes are the ones `cc913d5` removed on purpose (a CI clippy reports `double_must_use` on them); this host's clippy asks for
  them, so the literal command cannot be green here at base or head.
- `cargo fmt --all -- --check`: exit 0.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features`: exit 0.
- `RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features`: exit 0.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`: exit 0.
- Rustdoc by the C4 method (`rustdoc-base.md`): pixelflow-ir `warnings: base 38, head 38`, `NEW 0`, `GONE 0`; pixelflow-search
  `warnings: base 28, head 28`, `NEW 0`, `GONE 0` (`rustdoc-r3.pixelflow-{ir,search}.txt`). PASS.

## Scripts

- `scripts/check-intent.sh`: exit 0 (51 intent files).
- `scripts/check-doc-paths.sh`: exit 0 (672 source paths across 170 docs; 1 accepted via baseline); the `cannot open ../<file>.md` lines on
  stderr are the script's own relative reads, as in rounds 1 and 2.
- `scripts/check-cfg-encapsulation.sh`: exit 0, no new violations (121 grandfathered).
- `scripts/check-feature-matrix.sh`: RAN, exit 0, 10 s on warm build artifacts ("every crate's every feature builds in isolation"), with
  `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH` (cargo-hack installed outside the repo in round 1). No manifest or lock change
  (checksum of `Cargo.lock` and every `*/Cargo.toml` equal before and after; `git status` shows no tracked change).
- `python3 -I .wip/a3/gen/checklist.py`: 169 sites, 0 UNCLASSIFIED, 0 STALE (already-correct 39, changed-by-A3 22, left 13, tests 75, n/a 5,
  A4 15, A5 0).

## Spec section 11 greps (`greps-r3.txt`; `target.noindex`, `.git`, `.wip` excluded; also run literally)

| grep | hits | verdict |
|---|---|---|
| `grep -rnwE 'A3' --include=*.rs pixelflow-*` | none | PASS (empty) |
| `grep -nw 'A3' pixelflow-ir/intent.md` | none | PASS (empty) |
| `grep -rn 'fold_cost(' --include=*.rs .` | 10 lines, all in `pixelflow-search/src/egraph/cost.rs` | PASS (only cost.rs) |
| `grep -rn 'real_sorted\|var_variance' --include=*.rs .` | none | PASS (empty) |
| `grep -rn 'VALUE_BOUND' --include=*.rs .` | `pixelflow-ir/src/fold.rs` (981, 986, 1088, 1096; two are doc links) and `pixelflow-codegen/src/program/lower.rs` (12 import, 51 panic) | PASS |
| `grep -rln '\.wip' --include=*.rs --include=*.md . \| grep -v '^./.wip/'` | none | PASS (empty) |
| added lines of the A3 diff naming a `.wip` path or a model identifier | one `///` line in `fold_rules.rs` citing the repository's instruction file by name | PASS (a file name) |

## Diffstat

`git diff --stat 3bfdaded -- . ':!.wip'`: `diffstat-r3.txt`, 24 files changed, 3217 insertions, 419 deletions.

## Verdict

Every byte/key gate is identical to base; compile cost is +0.158 percent (limit +0.5); every test, fmt, no-default-features, doc, rustdoc
(C4), script, checklist and grep gate passes. The one literal command that does not exit 0 is the workspace clippy with `-D warnings`,
and it fails identically on the pristine base source on this host (three `must_use_candidate` errors in pixelflow-ir that predate A3 and
whose removal `cc913d5` made for CI's clippy); with that single lint allowed the whole workspace is clean.
