# Open-PR sweep — execution follow-up, 2026-09-01 21:15–22:00 UTC

Second pass over the open pull requests, continuing
`docs/results/2026-09-01-open-pr-triage.md` (#1086). That document did the
rebase sweep and wrote recommendations; this one **executed** what could be
executed and re-derived the parts of its status board that have since gone
stale.

`main` at time of writing: `38ea6eaa`.

## What changed

| Action | PR | Why |
|---|---|---|
| **Merged** | #1051 — `cost.rs` mutation gaps | test + docs only, 0 unresolved threads, all blocking checks green |
| **Merged** | #1049 — `graph.rs` test renames | test + docs only, 0 unresolved threads, all blocking checks green |
| **Closed** | #1050 — `regalloc.rs` mutation gaps | superseded; see below |
| Reviewed | #1044 | `shepherd` label; had never been reviewed |
| Retitled | #1087 | `CL metadata` lint wanted a conventional-commit prefix |

Both merges are squash commits (`83015dcd`, `38ea6eaa`), matching the
repository's existing convention. After merging, every remaining open PR was
re-checked with `git merge-tree --write-tree` against the new `main`: **all
11 still merge cleanly.** The merges introduced no conflicts.

## Corrections to #1086's status board

#1086 was accurate when written and is now one step behind. Four rows have
changed materially:

- **#1049, #1051 — merged.** Listed there as "ready".
- **#1050 — closed.** Listed there as open with 2 unresolved threads.
- **#1054 — was "green, one open caveat"; it is now RED.** See below. This is
  the most consequential drift: it moved in the opposite direction from
  everything else.
- **#1087 did not exist** when #1086 was written. It is a fifth participant in
  the saturation collision and a direct duplicate of #1083's purpose.

Additionally, #1086's three-way collision analysis (#1083 / #1084 / #1085) is
correct but **undercounts**: #1044 and #1087 are also on that surface.

## Complications

### 1. #1054 is chasing a moving target and is now red

`pixelflow-codegen/src/emit/x86_64.rs` has been refactored three times in
about two days. #1054 adds byte-exact encoder tests, and each refactor has
invalidated them:

- The first invalidation is already in the branch's own history — commit
  `4435869f`, "re-close x86_64.rs mutation gaps against the Vex-builder
  refactor," written after #1055–#1062 landed mid-PR.
- The second is now: #1082 ("one kernel ABI, one compile entry") removed
  `X86Backend::prologue` and `X86Backend::epilogue` entirely, and #1081
  replaced `&'static str` errors with `CompileError`. The tests no longer
  compile — `error[E0599]: no method named 'prologue'` at `x86_64.rs:1390`
  and `:1402`, `epilogue` at `:1413` and `:1426`, plus an `E0308` from the
  error-type change. Four jobs are red: Clippy, both test jobs, ISA matrix.

The tests themselves are not wrong; the surface they pin keeps moving.
Funding a third re-close pass only pays off if that file is now stable.
**Recommendation: hold #1054 until the encoder API settles, then re-close
once** — or close it and re-derive the coverage afterwards. Its audit
document is independently useful and could be split out.

### 2. #1083 and #1087 are two answers to the same question

Both branches are named for saturation telemetry (`saturation-telemetry-flag`
and `saturation-telemetry`), which makes this easy to miss:

- **#1083** adds a real `SaturationStopReason` field to `SaturationResult` —
  ground truth, at the cost of touching a production type.
- **#1087** infers the stop reason from outside, in `#[ignore]`d tests, with
  an explicit no-public-API-change rule. Its own summary names #1083 as the
  legitimate alternative "if you want to adopt it later."

They answer the same question (*does production saturation bind, and on
what?*) by opposite means. #1083 went green during this sweep — the std-off
`Feature matrix` failure it was carrying has been fixed on its branch.
**Recommendation: pick one before either lands.** They are not complementary;
landing both leaves two mechanisms for one fact.

### 3. Five PRs now share the saturation surface

#1083, #1084, #1085, #1087, and #1044. They pairwise merge cleanly today only
because none has landed. Two are *semantic* collisions that git cannot see:

- **#1044 ↔ #1085**: `pixelflow-search/src/egraph/variants.rs` calls
  `eg.saturate_with_limit(64)` at lines 229 and 262. #1085 deletes
  `EGraph::saturate_with_limit` along with `EGraph::saturate()`, leaving only
  the three-argument `saturate_with_limits`. The files are disjoint, so the
  merge is clean and the **build** breaks on whichever lands second. Fix:

  ```rust
  eg.saturate_with_limits(64, 10_000, std::time::Duration::from_millis(500));
  ```

- **#1083 / #1084 / #1085** all rewrite `saturate.rs` and the
  `SaturationStopReason` / `SaturationStats` / `SaturationResult` triple from
  three directions, as #1086 describes.

**Suggested order, unchanged from #1086 and still correct:** #1053 first (it
is green, and both #1083 and #1084 name it in their own test plans as the
fix for the std-off failure they carry), then #1085 → #1083 → #1084, with the
#1083-vs-#1087 decision taken before #1083 goes in.

### 4. #1044's clean review status is an artifact

#1044 shows zero unresolved review threads. That is not a pass. Its only bot
comment, from 2026-08-28, is `chatgpt-codex-connector` reporting that it hit
its Codex usage limit — **the automated review never ran.** Every other open
PR carries between 2 and 17 Codex threads.

This matters beyond #1044: any triage that ranks PRs by unresolved-thread
count will rank an unreviewed 3060-line diff as the cleanest thing in the
set. #1086's board has exactly that shape.

Its scope claims were checked and hold: `git diff main...HEAD --
pixelflow-search/src/egraph/extract.rs
pixelflow-search/src/egraph/extraction.rs` is empty, and `egraph/mod.rs` is
the two claimed lines.

### 5. #1072's headline numbers are contested at P1 — one verified here

Of the four P1 threads against the paper's central results, the
call-overhead one was checked directly and **is correct**:
`bench_extraction_3way.rs:2607` pushes `bench.ns * normalization` into the
sample vector, while `adjusted_ns` is computed and serialized at `:2591` but
never aggregated. The measured 4.272 ns call overhead is therefore in every
reported ratio.

Worth noting the direction, since it is not the one that flatters the
reviewer: adding a constant to both arms pulls `(nnue+c)/(static+c)` toward
1, so the **true regression is larger** than the reported 1.0153, not
smaller. The paper's qualitative verdict survives; its intervals do not.

## Recommended for closure

- **#1050 — closed during this sweep.** Verified first: the graph-coloring
  allocator its tests and both its "two real bugs" targeted
  (`simplicial_elimination_order`, `InterferenceGraph`, `color_graph`) is
  absent from `main` *and* from the branch itself, deleted by #1055/#1068.
  What survived the rebase was four eviction tests, and its two open threads
  argued — correctly — that those assert arbitrary tie-breaks, i.e. they kill
  equivalent mutants and will fail on harmless allocator refactors.

- **#1054 — hold or close**, per complication 1.

- **#994** (macOS signed DMG) is not obsolete but is **unverifiable in CI**:
  it cannot be exercised without signing secrets, so its green checks say
  nothing about whether the pipeline works. It should be landed on a
  deliberate decision to test it live, not swept in as "green".

## What this pass could not do

The remaining gap to "no unresolved comments, no CI failures" is entirely
commits that belong on other people's branches. This session is scoped to
its own development branch, so the following are written down rather than
pushed:

| PR | Fix | Size |
|---|---|---|
| #1083 | `saturation-telemetry = ["std"]` in `pixelflow-search/Cargo.toml` | 1 line |
| #1053 | Delete the stale `std-off-status` reference in `scripts/check-feature-matrix.sh:11-18` | ~6 lines |
| #1044 | `saturate_with_limit(64)` → `saturate_with_limits(64, 10_000, …)` ×2 | 2 lines |
| #1087 | `cargo fmt` on the anomaly filter in the telemetry test | 1 hunk |

Note on #1087: its `CL metadata` failure was a missing conventional-commit
prefix in the **PR title**, which was fixed here. The check will nonetheless
stay red until its next push — `check-cl-metadata.sh` reads `CL_TITLE` from
the workflow's stored event payload, and a re-run replays the *original*
payload, so a title edit can never be validated by a re-run. The `cargo fmt`
push resolves both at once.

Roughly 60 unresolved review threads remain across 8 PRs, 13 of them P1. All
were filed by `chatgpt-codex-connector`; no human review is unaddressed.
