# Open-PR triage — 2026-09-01

Sweep of all 13 open pull requests against three conditions: up to date with
`main`, no unresolved review threads, no CI failures. Plus the requested
judgement call on which branches are superseded, obsolete, non-salvageable, or
worth closing.

`main` at time of sweep: `cc4f0a7` (#1082).

## What this pass changed

Ten branches were behind `main` and have been brought up to date via GitHub's
merge-from-base ("Update branch"), which also re-triggers Presubmit Tests
against the current base:

| PR | branch | was behind |
|---|---|---|
| #1083 | `claude/saturation-telemetry-flag` | 1 |
| #1079 | `claude/integration-audit` | 3 |
| #1044 | `claude/round2b-contrastive` | 5 |
| #1053 | `fix/no-std-extraction-from-bytes` | 6 |
| #1072 | `claude/workshop-writeup` | 9 |
| #1054 | `claude/zen-babbage-wjmnit` | 9 |
| #1051 | `claude/zen-babbage-ccjkhv` | 9 |
| #1050 | `claude/zen-babbage-0rq98y` | 9 |
| #1049 | `claude/zen-babbage-6a9p2k` | 9 |
| #994 | `claude/macos-release-signing-pipeline` | 9 |

`#1085`, `#1084` and `#1081` were already current. **All 13 open PRs now merge
cleanly against `main` with no conflicts** — verified with `git merge-tree
--write-tree` per branch before the updates, and re-verified after.

Nothing else was pushed. Code fixes belonging to other people's branches are
listed below as recommendations rather than applied, since this session is
scoped to its own development branch.

## Status board

CI column is the last *completed* Presubmit Tests run at the head that was
current when the sweep started; every updated branch has a fresh run in flight.

| PR | Title (short) | Draft | CI | Unresolved threads | Merge-ready? |
|---|---|---|---|---|---|
| #1081 | codegen `CompileError` type | yes | in progress (prior head green) | **0** | closest to ready |
| #1051 | cost.rs mutation gaps | yes | green | 0 | ready |
| #1049 | graph.rs test renames | yes | green | 0 | ready (near-empty) |
| #1044 | VariantSet contrastive machinery | no | green | 0 | ready, but see below |
| #994 | macOS signed DMG release | yes | green | 0 | ready, but untestable |
| #1054 | x86_64.rs mutation gaps | no | green | 0 (2 resolved) | one open caveat |
| #1050 | regalloc.rs mutation gaps | no | green | **2** (P2) | mostly superseded |
| #1053 | gate NNUE opt-in behind `std` | no | green | **2** (P2) | **merge first** |
| #1079 | integration-audit doc | no | green | **6** (P2) | going stale |
| #1085 | one optimizer policy for Dwrt tier | no | queued (prior head **red**) | **3** (2×P1) | active |
| #1083 | saturation telemetry flag | no | **red** | **5** (2×P1) | own CI break |
| #1084 | Phase 3 at-budget evaluation | no | in progress | **17** (5×P1) | active, large |
| #1072 | workshop paper draft | no | green | **15** (4×P1) | biggest blocker |

Totals: **60 unresolved review threads across 8 PRs**, 13 of them P1. Every one
was filed by `chatgpt-codex-connector`; there are no unaddressed human reviews.

## Complications

### 1. Three PRs are colliding on the same saturation code

This is the structural problem in the set. `#1083`, `#1084` and `#1085` all
rewrite `pixelflow-search/src/egraph/saturate.rs` and the
`SaturationStopReason` / `SaturationStats` / `SaturationResult` triple, from
three directions:

- `#1083` says it *completes a non-compiling partial commit* that introduced
  `SaturationStopReason` (two struct literals were missing the new field), and
  builds telemetry on top of it.
- `#1084` makes `SaturationStats` / `SaturationResult::stop` read from the
  saturation loop instead of being inferred, and adds `GuidedSaturation`
  candidate dedup in the same file.
- `#1085` deletes `EGraph::saturate()` / `saturate_with_limit()` outright and
  routes all three tiers through one `saturate_for_extraction`.

They merge cleanly against `main` today only because none of them has landed.
Whichever goes first forces real rework on the other two — this is not a
textual conflict that `git` will resolve. **Recommendation: pick a landing
order deliberately (#1085 → #1083 → #1084 reads best: policy unification
first, then the instrument, then the experiment that consumes it) and tell the
other two branches to rebase onto it rather than racing.**

### 2. #1083 has a CI failure that is genuinely its own

`Feature matrix` fails on three `pixelflow-search` std-off combinations. Two
(`--no-default-features`, `--features extraction-profile`) are the pre-existing
`ExprNnue::from_bytes` defect and are listed in `KNOWN_BROKEN`. The third,
`--no-default-features --features saturation-telemetry`, is new and is this
PR's: adding a std-only feature to `pixelflow-search` adds a std-off
combination that fails identically. `scripts/check-feature-matrix.sh` predicts
exactly this in its own header comment.

Fix, one line, and the *subtractive* one rather than another exception row:
declare the feature's real dependency in `pixelflow-search/Cargo.toml` —

```toml
saturation-telemetry = ["std"]
```

The telemetry module writes JSONL to a file or stderr, so it genuinely requires
`std`; with the dependency declared, `--no-default-features --features
saturation-telemetry` turns `std` back on and compiles. Adding a third
`KNOWN_BROKEN` row would also make CI green and would be wrong — it suppresses
a combination instead of describing it.

### 3. #1053 is the unblocker and should merge first

`#1053` gates `load_opt_in_weights` / `env_extraction_policy` behind `std` and
empties `KNOWN_BROKEN`. Both `#1083` and `#1084` name it in their own test
plans as the fix for the failure they are carrying. It is green with two P2
threads outstanding, one of which is trivial:

- **Trivial:** the header comment at `scripts/check-feature-matrix.sh:11–18`
  still points at the `std-off-status` job that the same commit deletes.
  Delete or rewrite those lines.
- **Substantive but out of scope:** the reviewer notes the new blocking
  `no-std` job does not prove no-std — `pixelflow-search/src/lib.rs` never
  applies `#![no_std]`, `egraph/graph.rs` still uses `std` unconditionally, and
  the manifest forces `pixelflow-ir/std`. So the job proves "builds with the
  feature off on a std host," not "builds without libstd." That is a fair
  objection to the *claim* in the job's description, not to the change. Narrow
  the job's wording and land it.

### 4. #1072's headline numbers are contested at P1

Four unresolved P1s go at the paper's central results, not its prose:

- Reported ratios never subtract the measured 4.272 ns call overhead
  (`bench_extraction_3way.rs:2589` aggregates `bench.ns * normalization`;
  `adjusted_ns` is only serialized). Adding a constant to both arms pulls
  `(nnue+c)/(static+c)` toward 1 — toward the paper's parity finding.
- The learned arm and the static arm differ in *search initialization* as well
  as scoring (`IncrementalExtractor` starts from `Extraction::from_backfill`,
  static DP runs independently), which confounds attributing the result to the
  cost model.
- Round 3 vs Round 2a is never tested directly; the two CIs overlap
  substantially and were timed in separate sessions.
- The bootstrap resamples individual kernels, but the corpus defines
  `(band, seed)` families as the split unit (56 families × 14). Treating ~716
  kernels as independent understates the interval that the "confirmed
  regression" verdict rests on.

Two further findings the author already conceded cannot be fixed from this
tree: the per-kernel `D2a`/`D3` JSONL artifacts were written to a gitignored
path on a machine whose worktree is gone, and the Round-3 checkpoints
`inspect_flip` requires are not in the repository. Both were handled by
qualifying the text rather than restoring the data.

**This is not a rebase-and-merge PR.** Either the four P1s get answered with
re-analysis (the overhead subtraction and the family-clustered bootstrap are
both re-aggregations of data that exists, so they are cheap), or the paper
lands explicitly marked as a draft whose intervals are provisional.

### 5. #1079's audit doc is going stale while it sits

Six P2 threads, all "the document says X, the code does Y" — the `oracle`
feature is harness code not test-only, `CellGridProgram` is library-only not
dead, production *does* already count rule applications, the timeout ratio
range is wrong, the recompile-per-resize count is wrong in both directions.
Worse, the audit is the document that *spawned* `#1083` and `#1085`, and both
have since changed the code it describes — `#1085` unifies the Dwrt tier the
audit flags, and `#1083` adds the stop-reason instrumentation that thread 2
says is missing. Land it with the six corrections now, or it will need a
rewrite against post-`#1083`/`#1085` `main`.

## Superseded / obsolete / recommended for closure

### Recommend closing: #1050 — regalloc mutation tests

Its own merge commit is the case for closure: "#1055 and #1068 rewrote
`regalloc.rs` end to end and deleted the graph-coloring allocator outright —
`InterferenceGraph`, `build_interference_graph`, `color_graph` and
`simplicial_elimination_order` are gone from the workspace. Two thirds of this
branch's tests, and **both of the bugs it found**, were about those functions."

What survives is four `LinearScan` eviction tests — and both open threads argue
those four are themselves wrong: they assert *which* `ValueId` wins an
arbitrary last-use tie (killing equivalent mutants and pinning allocator policy
that is free to change), and their names violate `docs/STYLE.md`'s "it should"
rule. A PR whose findings are deleted code and whose residue is contested is
better closed with the two bug write-ups moved to `docs/bugs/` than merged.

### Recommend closing or folding: #1049 — graph.rs test renames

The PR's own description records that its substantive work (cost.rs mutation
gaps) was already closed by `#1027`, fifteen commits ahead of this branch's
base, and that those additions were dropped rather than merged as a worse
duplicate. What is left is **two test renames and one stale backlog-note
correction**, open since 28 Aug. Land it in the next passing PR that touches
`pixelflow-search`, or close it.

### Check for duplication: #1051 — cost.rs mutation gaps

`#1051` is the 2026-08-30 cost.rs audit; `#1049` states that `#1027`
(2026-08-22 audit) already closed the cost.rs backlog item and specifically
avoided an unsafe-env-var-under-parallel-tests flaw that a re-attempt
reproduced. `#1051` is green and thread-free, so it is cheap to land, but
someone should confirm it is not re-adding what `#1027` already covers before
it goes in.

### Superseded by the Phase-3 program: #1044 — Round 2b contrastive

The experiment this PR ran **failed**, and the PR says so honestly: geomean
1.0153 with CI [1.0097, 1.0213], entirely above 1.0, a confirmed regression
against both the static prior and Round 2a. It has also been ported twice
against `main` (`#1063`'s `EdgeTrace` rework invalidated the original). The
research direction has since moved to guided saturation (`#1084`), which does
not use this machinery. Merge it as the durable record of a negative result —
`#1072` cites Round 2b and currently calls it the paper's weakest-traced
number, which merging would partly fix — or close it and keep the finding in
the journal. Do not leave it open indefinitely; it will need a third port.

### Salvageable with one caveat: #1054 — x86_64 mutation tests

Invalidated mid-flight by `#1055`–`#1062`'s Vex-builder refactor, then rewritten
against the new API; both review threads are resolved and CI is green. One
honesty gap remains in the author's own reply: the audit's "0 real gaps"
conclusion was not re-verified for `emit_vpextrd_to_gpr` and
`emit_vmovss_load_scaled` after the refactor. Either rerun
`cargo mutants -p pixelflow-codegen --file .../x86_64.rs -- --lib --test collapse_loop`
or soften the claim in the doc, then merge.

### Stalled on external dependency: #994 — macOS signed DMG

Open since 11 Aug, green, no review threads, and **untested end to end** — it
needs five repository secrets (`MACOS_CERTIFICATE_P12_BASE64`,
`MACOS_CERTIFICATE_PASSWORD`, `AC_API_KEY_P8_BASE64`, `AC_API_KEY_ID`,
`AC_API_ISSUER_ID`) that do not exist, and the codesign/notarytool/stapler path
has never run. The workflow is tag-triggered, so it is inert until someone
pushes a `v*.*.*` tag. It is not obsolete and not wrong; it is blocked on an
Apple Developer account. Merge it as dormant infrastructure, or close it until
the credentials exist — but three weeks of drift on a branch that only CI can't
validate is a poor use of an open slot.

## Suggested landing order

1. **#1053** — unblocks the std-off feature-matrix noise that `#1083` and
   `#1084` are both carrying. Fix the stale header comment, narrow the no-std
   job's claim, merge.
2. **#1081** — zero threads, mechanical type change, green on its prior head.
3. **#1051**, **#1049**, **#1054** — small, green, thread-free (modulo the
   `#1054` mutants rerun and the `#1051` duplication check).
4. **#1079** — six factual corrections, then merge before it goes stale.
5. **#1085** — resolve its 3 threads; it is the policy change the other two
   saturation PRs should rebase onto.
6. **#1083** — declare `saturation-telemetry = ["std"]`, then the two P1
   threads (the `Converged`-on-budget-exit misreport is the one that matters:
   the PR exists to measure stop reasons and would record the wrong one).
7. **#1084** — 17 threads, 5 P1, 27k additions. Its P1s are about experiment
   validity, not style; budget real time.
8. **#1044**, **#994** — decide merge-or-close; both are records rather than
   work in progress.
9. **#1072** — last, and only after the four P1 re-analyses or an explicit
   provisional framing.

## Residual gap against the stated goal

"Rebased" is met for all 13. "No CI failures" and "no unresolved comments" are
not, and cannot be closed from this session without pushing commits to eight
other people's branches, which is outside this session's branch scope. The
per-PR fixes above are written to be actionable by whoever owns each branch.
