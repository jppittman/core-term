total 4476
drwxr-xr-x 2 root root    4096 Oct 23 00:26 .
drwxr-xr-x 4 root root    4096 Oct 23 01:33 ..
-rwxr-xr-x 1 root root 4567720 Oct 23 00:26 count
-rw-r--r-- 1 root root     510 Oct 23 00:26 count.rs
5.1M	/home/user/scratch/egraph/arena/gates/bin.gate3.D-pre
5.1M	/home/user/scratch/egraph/arena/gates/bin.gate3.mut1
5.1M	/home/user/scratch/egraph/arena/gates/bin.gate3.perturb
5.1M	/home/user/scratch/egraph/arena/gates/bin.gate3.post
5.1M	/home/user/scratch/egraph/arena/gates/bin.gate3.pre
559M	total
-rwxr-xr-x 1 root root 5300944 Oct 23 00:00 /home/user/scratch/egraph/arena/gates/bin.measure.narrow-exp
-rwxr-xr-x 1 root root 5303792 Oct 22 23:56 /home/user/scratch/egraph/arena/gates/bin.measure.narrow-wide
# Gates for "The e-graph is an arena" (plan §3.2)

Plan: `/home/user/pixelflow/docs/plans/2026-10-15-the-egraph-is-an-arena.md`. Every CL passes gates 1, 1b, 2 and
4, and adds a ledger row from gate 5. Bases were recorded at HEAD `94aeef7a` (docs commit on `2a99d933`; no
e-graph code differs). Run everything from anywhere; the scripts `cd` to `/home/user/pixelflow`.

TAG is any new name per run (`cl3b`, `cl3b-retry`); a tag overwrites its own outputs and nothing else, and
`base` is reserved for the recorded bases. Never delete files here; use a new tag.

| gate | command | pass = | cost |
|---|---|---|---|
| 1 | `gates/gate1.sh TAG` | `byte_probe` output `cmp`-equal to `byteprobe-base.txt` (sha256 `fab7d3b7...3181885`, 107 lines; also equal to `speed/byteprobe-base.txt`) | ~1 min (cold), seconds warm |
| 1b | `gates/gate1b.sh TAG` | `gate1b.TAG.txt` equals `gate1b.base.txt`: per program `n`, node count, `KernelKey::bits()` and `program_digest` (FNV-1a over nodes in push order) of what `optimize_runtime_arena` returns, 29 lines | ~30 s |
| 2 | `gates/gate2.sh TAG` | `gate2.TAG.txt` equals `gate2.base.txt`: per program applications, unions, classes_at_stop, live, iterations, stop reason (29 lines; probe built with `--features pixelflow-core/saturation-telemetry`, single-threaded, `PIXELFLOW_SATURATION_TELEMETRY=gate2.TAG.jsonl`) | ~30 s |
| 2x | `gates/gate2x.sh TAG` (after `gate2.sh TAG`) | the extraction half of the same telemetry records (`extraction_objective`, `extracted_latency_prior_cost`, `shared_pass_bytes`) equals `gate2.base.jsonl`'s, 29 records. Added for group B (extraction CLs): gate 1b catches an output change, this catches a `TreeOnly` flip, a root-cost drift or a reach-byte change directly | seconds |
| 3 | not scripted | lock-step shadow, development only, deleted before submit | |
| 4 | not scripted | `cargo test -p <crate>` per touched crate (`--all-features` where a feature gates changed code), `cargo clippy --workspace --all-targets --all-features -- -D warnings`. Never `cargo test --workspace` | |
| 5 | `gates/measure.sh [--cache-sim] TAG` | measured, never asserted: JSON on stdout and in `measure.TAG.txt`; add a row to `ledger.md` | ~1 min; `--cache-sim` adds ~2.5 min |

Gates 1b and 2 and gate 5 build `pixelflow-graphics`' release test binary with `probes-gate1b.rs` appended to
`pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs` (`lib.sh: build_probe`). The file is restored with
`git checkout --` by a trap and again right after the build; a script refuses to start if `tests.rs` is already
modified. After a run `git status` must be clean. The test binary is copied to `gates/bin.<script>.TAG` so a later
build cannot replace what a result was measured on.

`probes-gate1b.rs` is `cl7/probes-append.rs` (the `zz_prof_*` probes; `zz_prof_all_cold` is gate 5's workload)
plus `zz_gate_font_kernels`, the probe gates 1b and 2 read. It prints `ZZGATE n=.. nodes=.. key=.. digest=..`
per program (libtest glues the first one onto its `test ... ` line, so scripts match `.*ZZGATE `).
The 29 programs are `n` in {0, 2..8, 10..13, 16..21, 23..25, 27..29, 32, 34, 36, 38, 40} at 16 px; `n=0` is the
blank program (space). `digest` is `program_digest` from `crate::test_support`, the one the suite's
`the_optimizer_selects_what_it_selected` already pins: `n=2` and `n=40` here equal its recorded values.

Gate 2 pairs the k-th telemetry record with the k-th program (the sink appends in call order; the script
truncates it first). Its base equals the tuples in `prof2/tel.all_cold.jsonl` (taken at `7cd96a32`) on all 29.

## Gate 5 details

`measure.sh` builds the probe, runs `valgrind --tool=callgrind` on `zz_prof_all_cold` (`cg.TAG.out`), then three
native runs for median peak RSS (`ru_maxrss`, KiB) and median wall ms (reported, never decisive; the VM drifts
~30%). Reported: total Ir, `Saturation::saturate` inclusive Ir, extraction inclusive Ir (`extract_dag_over`, or