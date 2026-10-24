# Gates for "The e-graph is an arena"

Rebuilt from `README.recovered.md` after the container was re-provisioned. Everything lives in
`/home/user/pixelflow/.wip/gates/`; text outputs are in `out/`; test binaries and build logs are in
`/home/user/scratch/gates-bin/` (never in the repo). Scripts `cd /home/user/pixelflow` themselves and run from
anywhere. Nothing here is committed: `git status --short` shows only `.wip/` after any run.

Plan (as named by the recovered README; `docs/plans/` does not exist at this HEAD):
`/home/user/pixelflow/docs/plans/2026-10-15-the-egraph-is-an-arena.md`. Every CL passes gates 1, 1b, 2 and 4 and adds
a ledger row from gate 5 (`ledger.md` was not recreated; the first CL that needs a row creates it).

**Bases** were recorded at tree `3bfdaded` (main). HEAD was then `c3889c4`, which differs from `3bfdaded` only by
adding `.wip/` notes (no non-`.wip` file differs; checked with `git diff --quiet 3bfdade HEAD -- . ':!.wip'`).
Verified at `a747770` (main after A3, #1447): gates 1, 1b, 2 and 2x (tag `a4-main`) are identical to those bases, so
the byte, key and saturation bases stand and A3 is byte-identical on today's kernels. The gate 5 (`measure`) base was
re-recorded there with `GATES_REBASE=1` (A3 moved total Ir by +0.227 percent; figures under "Gate 5 details"). The
tree at the time was `72db468`, which differs from `a747770` only by `.wip/` (checked with
`git diff --quiet a747770 HEAD -- . ':!.wip'`). The rustdoc bases for A4 are `/home/user/pixelflow/.wip/a4/rustdoc-base.<crate>.txt`
(`pixelflow-ir` 38 warnings, `pixelflow-search` 28, `pixelflow-codegen` 0), normalized by `.wip/a4/rustdoc_norm.py`.
Verified again at `7e9531e` (main after A4, #1448; tag `a5-main`): gates 1, 1b, 2 and 2x each exit 0, identical to the
same bases, so the byte, key and saturation bases stand and A4 is byte-identical on today's kernels. The gate 5
(`measure`) base was re-recorded there with `GATES_REBASE=1` (A4 moved total Ir by +0.089 percent; figures under "Gate 5
details"). The tree at the time was `2f61e7c`, which differs from `7e9531e` only by `.wip/` (checked with
`git diff --quiet 7e9531e HEAD -- . ':!.wip'`). The rustdoc base for A5 is
`/home/user/pixelflow/.wip/a5/rustdoc-base.pixelflow-compiler.txt` (`pixelflow-compiler` 4 warnings, 3 distinct),
normalized by `.wip/a5/rustdoc_norm.py` (that directory is archived in `/home/user/scratch/a5-wip-archive/.wip/a5/`).
Verified again at `29980bf` (main after A5, #1449; tag `a6-main`): gates 1, 1b, 2 and 2x each exit 0, identical to the
same bases, so the byte, key and saturation bases stand and A5 is byte-identical on today's kernels. The gate 5
(`measure`) base was re-recorded there with `GATES_REBASE=1` (A5 moved total Ir by +0.020 percent; figures under "Gate 5
details"). The tree at the time was `bc1c011`, which differs from `29980bf` only by `.wip/` (checked with
`git diff --quiet 29980bf HEAD -- . ':!.wip'`). The rustdoc bases for A6 are
`/home/user/pixelflow/.wip/a6/rustdoc-base.<crate>.txt` (`pixelflow-compiler` 4 warnings, 3 distinct; `pixelflow-core`
5, 4 distinct; `pixelflow-ir` 38, 35 distinct), normalized by `.wip/a6/rustdoc_norm.py`. The `pixelflow-compiler` and
`pixelflow-ir` bases are identical to A5's and A4's, and the normalizer is unchanged.
Host: 4 cores, 15 GB, AVX-512 capable (`avx512f/dq/bw/cd/vl`), valgrind 3.22, Python 3.11, no `/usr/bin/time`.

## Using the scripts

`TAG` is any new name per run (`cl3b`, `cl3b-retry`). A tag overwrites its own outputs and nothing else. `base` is
reserved: a `TAG=base` run refuses (exit 3) if its base file already exists, unless `GATES_REBASE=1` is set, so a stray
run cannot replace what every comparison reads. Never delete files here; use a new tag.

Exit codes: `0` recorded (base) or identical to base; `1` differs from base; `2` the harness itself failed (build
error, wrong probe output, failed self-check); `3` refused to start (under 2 GB free, `tests.rs` already modified,
another gate build holds the lock, base already recorded). `measure.sh` never asserts: it exits 0 whenever it ran.

| gate | command | pass = | measured cost on this host |
|---|---|---|---|
| 1 | `gates/byteprobe.sh TAG` | `out/byteprobe.TAG.avx512.txt` and `.avx2.txt` (127 lines each) `cmp`-equal to `byteprobe.base.{avx512,avx2}.txt`. Runs `PIXELFLOW_ISA=<tier> cargo run --release -q -p pixelflow-codegen --example byte_probe`, stderr to `.err` | 35 s first (builds the example), 1.2 s warm |
| 1b | `gates/gate1b.sh TAG` | `out/gate1b.TAG.txt` equals `gate1b.base.txt`: 29 lines `ZZGATE n= nodes= key= digest=` sorted by `n`. Also self-checks the probe against the suite's pinned digests (exit 2 if not) | 62 s first (nothing built before), 6.5 s warm |
| 2 | `gates/gate2.sh TAG` | `out/gate2.TAG.txt` equals `gate2.base.txt`: per program `n applications unions classes_at_stop live iterations stop` (29 lines). Probe built with `--features pixelflow-core/saturation-telemetry`, run single-threaded with `PIXELFLOW_SATURATION_TELEMETRY=out/gate2.TAG.jsonl` (truncated first, never deleted) | 24 s first (feature set rebuilds the core/codegen/search chain), 6.5 s warm |
| 2x | `gates/gate2x.sh TAG` (after `gate2.sh TAG`) | `gate2x_fields.py` over `gate2.TAG.jsonl` equals the same over the base: `extraction_objective`, `extracted_latency_prior_cost`, `shared_pass_bytes`, 29 records. Catches a `TreeOnly` flip, a root-cost drift or a reach-byte change that gate 1b would only see as an output change | under 0.1 s |
| 3 | not scripted | lock-step shadow, development only, deleted before submit | |
| 4 | not scripted | `cargo test -p <crate>` per touched crate (`--all-features` where a feature gates changed code), `cargo clippy --workspace --all-targets --all-features -- -D warnings`. Never `cargo test --workspace` | |
| 5 | `gates/measure.sh TAG` | measured, never asserted: JSON on stdout and in `out/measure.TAG.txt` (with `vs_base` percent changes when a base exists); add a ledger row | 52 s (callgrind 41 s, five native runs 1 s each, plus build) |

Compare raw `gate2.TAG.jsonl` files only through gate 2 / 2x: each record carries `wall_clock_us`, which moves run to
run. With it removed the recheck's file equals the base's.

## What the probes are

`probes-gate1b.rs` is appended to `pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs` for the length of one
build (`lib.sh: build_probe`) and restored with `git checkout --` by a trap and again right after the build. A script
refuses to start if `tests.rs` is already modified; after a run `git status` shows nothing outside `.wip/`. The built
test binary is copied to `/home/user/scratch/gates-bin/bin.NAME` (`gate1b`, `gate2`, `measure`) so a later build
cannot replace what a result was measured on. One gate build at a time (`flock` on `.build.lock`).

- `zz_gate_font_kernels` (`#[ignore]`): one line per program, `ZZGATE n=.. nodes=.. key=.. digest=..`, for `n` in
  {0, 2..8, 10..13, 16..21, 23..25, 27..29, 32, 34, 36, 38, 40} at 16 px. `n` is the program's piece count (`n=0` is the
  blank program); the rest is read off what `optimize_runtime_arena` returns for `program(n).linked_parts()` at
  `LatticeShape [16, 16]`, exactly as `the_optimizer_selects_what_it_selected` builds it. `nodes` is the optimized arena's
  length, `key` is `KernelKey::of(out, root).bits()`, `digest` is `program_digest` from `crate::test_support` (FNV-1a over
  the nodes in push order). libtest glues the first line onto its `test ... ` line, so scripts match `.*ZZGATE `.
- `zz_prof_all_cold` (`#[ignore]`): gate 5's workload. `Manifold::compile(&program(n), [16, 16])` for the same 29
  counts in one process: build the program, optimize it through the runtime tier, emit it. That is the first-draw path of
  `FontPrograms::draw`; nothing is drawn.

**Self-check.** The probe's `n=2` and `n=40` digests at 16 px equal the values `the_optimizer_selects_what_it_selected`
pins (`SELECTED_PROGRAM_DIGESTS`, rows `(2, 16, ..)` and `(40, 16, ..)`): `22ad8819798461a6` and `e95137a9d464391e`.
`gate1b_selfcheck.py` reads the pins from the working tree's `tests.rs` on every `gate1b.sh` run and exits 2 on a
mismatch, so a probe that drifts from the suite cannot produce a base. (If a CL deliberately moves those pins, the check
follows them; gate 1b then reports the move as a difference from base.)

Gate 2 pairs the k-th telemetry record with the k-th program (the sink appends in call order). `gate2_pair.py` refuses
(exit 2) unless there is exactly one runtime-tier saturation record per program: a decline, a macro-tier record or a
different count means the pairing would be wrong. Telemetry fields used: `application_count`, `union_count`,
`classes_at_stop`, `live_classes`, `iterations`, `stop_reason`; gate 2x: `extraction_objective`,
`extracted_latency_prior_cost`, `shared_pass_bytes` (`pixelflow-search/src/telemetry.rs`).

## Gate 5 details

`measure.sh` builds the probe, runs `valgrind --tool=callgrind` on `zz_prof_all_cold` (`out/cg.TAG.out`, kept), then
five native runs for median peak RSS (`ru_maxrss`, KiB) and median wall ms (reported, never decisive; the VM drifts).

- Peak RSS comes from `runstat.py`, which reads the child's `ru_maxrss` from `wait4`: the number
  `/usr/bin/time -f %M` prints, for a host that has no `/usr/bin/time`.
- Reported from `callgrind_annotate --inclusive=yes --threshold=100`: total Ir (`PROGRAM TOTALS`);
  `<pixelflow_search::egraph::graph::runner::Saturation>::saturate`; and
  `pixelflow_search::egraph::extract::extract_dag_over::<CostModel>`. `measure_report.py` lists every matching line under
  `"matched"`, so a renamed function shows up as an empty match, not a wrong number. If a CL renames either function,
  edit the two patterns at the top of `measure_report.py`.
- Valgrind's emulated CPU has no AVX-512, so the callgrind run emits the AVX2 tier (the annotate output holds
  `emit::avx2::Avx2`, no `avx512`); the native runs use the host's AVX-512 tier. Compare tags on this host only.
- Total Ir is not bit-stable: two runs of the same binary differed by 2,038 Ir of 6.296e9 (3e-5 percent). The
  `Saturation::saturate` and `extract_dag_over` inclusive figures were identical across the two runs.
- Under callgrind the test runs on a spawned thread (libtest does that even with `--test-threads=1`); callgrind sums
  threads into the one file.

Base (tree `29980bf`, re-recorded): total Ir 6,317,091,756; `Saturation::saturate` inclusive 2,076,022,575 (32.9 percent);
`extract_dag_over` inclusive 1,139,268,582 (18.0 percent); median peak RSS 96,824 KiB; median wall 1,203 ms.

Previous base (tree `7e9531e`; the old `measure.base.txt` and `cg.base.*` are copied to
`/home/user/scratch/gates-old-base-7e9531e/`): total Ir 6,315,838,727; `Saturation::saturate` inclusive 2,075,982,919
(32.9 percent); `extract_dag_over` inclusive 1,139,023,257 (18.0 percent); median peak RSS 97,592 KiB; median wall
1,000 ms. A5 moved total Ir by +0.020 percent, `saturate` by +0.002 percent and `extract_dag_over` by +0.022 percent;
peak RSS moved -0.8 percent and the wall figure +20 percent between sessions (VM drift, never decisive).

Earlier base (tree `a747770`; the old `measure.base.txt` and `cg.base.*` are copied to
`/home/user/scratch/gates-old-base-a747770/`): total Ir 6,310,238,494; `Saturation::saturate` inclusive 2,076,952,032
(32.9 percent); `extract_dag_over` inclusive 1,139,515,928 (18.1 percent); median peak RSS 96,536 KiB; median wall
1,072 ms. A4 moved total Ir by +0.089 percent, `saturate` by -0.047 percent and `extract_dag_over` by -0.043 percent;
peak RSS moved +1.1 percent and the wall figure -6.7 percent between sessions (VM drift, never decisive).

Earlier still (tree `3bfdaded`, kept for the record; the old `measure.base.txt` and `cg.base.*` are copied to
`/home/user/scratch/gates-old-base-3bfdaded/`): total Ir 6,295,969,957; `Saturation::saturate` inclusive 2,079,429,510
(33.0 percent); `extract_dag_over` inclusive 1,123,555,965 (17.8 percent); median peak RSS 96,476 KiB; median wall 956 ms.
A3 moved total Ir by +0.227 percent, `saturate` by -0.119 percent and `extract_dag_over` by +1.420 percent; the wall
figure moved +12 percent between sessions and is VM drift, never decisive.

**Lock note.** `build_probe` runs `cargo` inside `flock` on fd 9. If `sccache` (the `RUSTC_WRAPPER`) has no server running,
cargo starts one that inherits fd 9, daemonizes, and keeps the lock held after the gate exits; the next gate then exits 3
("another gate build holds ... .build.lock"). Fix without editing a script: `sccache --stop-server; sccache --start-server`
from a shell that has no fd 9 open, so the server lives outside the lock.

## Files

| file | role |
|---|---|
| `lib.sh` | `build_probe NAME [cargo args]`, `df_guard`, `guard_base`, `compare_to_base`, paths and exit codes |
| `probes-gate1b.rs` | the two probes above |
| `gate1b.sh`, `gate1b_selfcheck.py` | gate 1b and its self-check |
| `gate2.sh`, `gate2_pair.py` | gate 2 and the record-to-program pairing |
| `gate2x.sh`, `gate2x_fields.py` | gate 2x and the extraction-field printer (`gate2x_fields.py JSONL`) |
| `byteprobe.sh` | gate 1 |
| `measure.sh`, `measure_report.py`, `runstat.py` | gate 5 |
| `out/` | outputs: `gate1b.TAG.{txt,lines,raw,err}`, `gate2.TAG.{txt,jsonl,lines,raw,err}`, `gate2x.TAG.txt`, `byteprobe.TAG.{avx512,avx2}.{txt,err}`, `measure.TAG.txt`, `cg.TAG.{out,stdout,err}` |

`df -h /` is printed before every build, and a build stops (exit 3) with under 2 GB free.

## Recorded

Bases and rechecks (tags `base`, `recheck`) are in `out/`. The `recheck` run of gate 1, 1b, 2 and 2x at the same clean
tree was identical to base on every file.

## Not in the recovered README

The recovered text ended mid-sentence in the Gate 5 section ("extraction inclusive Ir (`extract_dag_over`, or"), began
with the stray output of an `ls` of the lost `gates/` directory, and described `cl7/probes-append.rs` and `prof2/`
(both lost). The probes, function names, and every number above were re-derived on this host from the sources named
here; the old bases (`94aeef7a`, `7cd96a32`) could not be reproduced and are not claimed.

**Never commit while a gate runs.** `build_probe` appends the probes to `tests.rs` and restores it with
`git checkout -- tests.rs`, so a commit taken mid-build records the probes, and every later restore puts them
back. A5's round 3 did this once (2121763, reverted in the next commit).
