# Open-PR triage — 2026-09-01, refreshed 2026-09-02

Two passes over the open pull requests against three conditions: up to date
with `main`, no unresolved review threads, no CI failures. Plus the requested
judgement on which branches are superseded, obsolete, non-salvageable, or worth
closing.

The first pass ran 2026-09-01 ~21:00–22:00 UTC against `main` at `cc4f0a7`.
This document has been rewritten for the state at **2026-09-02 10:10 UTC**,
`main` at `44c9fa3f` — seventeen commits later. A parallel session's follow-up
(#1089, merged as `2e82cdc2`) covers the intervening window and is not repeated
here.

## The headline: the collision landed, and it took the rest of the board with it

The first pass flagged that #1083, #1084 and #1085 all rewrote
`pixelflow-search/src/egraph/saturate.rs` and the `SaturationStopReason` /
`SaturationStats` / `SaturationResult` triple from three directions, that they
merged cleanly only because none had landed, and that whichever went first
would force real rework on the others.

That is now what happened, at a larger scale than predicted. `#1083`
(`82961fe3`), `#1085` (`c1afd4b9`), `#1107` and the `#1108` optimizer-entry-point
refactor all landed overnight. **Ten of the fifteen open PRs now conflict with
`main`; yesterday all thirteen merged cleanly.**

Nine of the ten share one epicentre — every conflict is in some subset of:

```
pixelflow-search/src/egraph/graph.rs
pixelflow-search/src/egraph/saturate.rs
pixelflow-search/src/egraph/mod.rs
pixelflow-search/src/runtime.rs
```

The tenth, #1072, is a different and worse shape (below).

This is the cost of landing four PRs that touch one seam without first rebasing
the branches queued behind them. It was foreseeable — it was, in fact,
foreseen — and the cheap mitigation was to rebase the queue after the first of
the four landed rather than after all four.

## Board at 2026-09-02 10:10 UTC

| PR | Branch | Ahead | Behind | Merges? |
|---|---|---|---|---|
| #1114 | `claude/class-cap-live` | 1 | 0 | clean |
| #1113 | `claude/upward-congruence` | 3 | 0 | clean |
| #1109 | `claude/cap-break-ab` | 1 | 3 | **conflict** — `runtime.rs` |
| #1103 | `claude/all-rules-numeric-first` | 10 | 9 | **conflict** — `graph.rs` + 5 |
| #1101 | `claude/rule-order-numeric-first` | 9 | 6 | **conflict** |
| #1096 | `claude/phase3-r2g` | 38 | 9 | **conflict** |
| #1095 | `claude/phase3-label-constfold` | 32 | 9 | **conflict** |
| #1091 | `claude/phase3-domain-shift` | 32 | 5 | **conflict** |
| #1088 | `claude/phase3-round2` | 38 | 9 | **conflict** |
| #1087 | `claude/saturation-telemetry` | 6 | 10 | **conflict** — + `cell_grid.rs` |
| #1086 | `claude/brave-faraday-tw3054` | — | 0 | clean (this doc) |
| #1084 | `claude/phase3-guide` | 26 | 9 | **conflict** — `graph.rs` + 4 |
| #1072 | `claude/workshop-writeup` | 12 | 17 | **conflict** — modify/delete |
| #1054 | `claude/zen-babbage-wjmnit` | 9 | 17 | clean, but **CI red** |
| #994 | `claude/macos-release-signing-pipeline` | 3 | 17 | clean |

## What the first pass recommended, and what happened

Every recommendation was acted on within the following hours, mostly by other
sessions. Recording the outcome rather than the advice:

| PR | Recommendation | Outcome |
|---|---|---|
| #1053 | merge first — it unblocks #1083/#1084 | **merged** (`436d3af8`) |
| #1081 | ready now, zero threads | **merged** (`c7e65096`) |
| #1051 | land after a duplication check vs #1027 | **merged** (`83015dcd`) |
| #1049 | land or close; near-empty | **merged** (`38ea6eaa`) |
| #1079 | correct 6 findings, land before it goes stale | **merged** (`aa27cf1c`) |
| #1083 | declare `saturation-telemetry = ["std"]`, fix the P1 | **merged** (`82961fe3`) |
| #1085 | resolve 3 threads; land first of the saturation three | **merged** (`c1afd4b9`) |
| #1050 | close — superseded by #1055/#1068 | **closed** unmerged |
| #1044 | merge as a record or close | **closed** unmerged |
| #1054 | land after a mutants rerun | **still open, now red** |
| #1072 | last, after the P1 re-analyses | **still open, now conflicted** |
| #994 | merge dormant or close | **still open, untouched** |

Nine of twelve resolved. The three that did not are the three this pass still
recommends closing or holding.

### Two corrections to the first pass

**Thread count is not a quality signal.** The first pass ranked #1044 as
merge-ready partly on "zero unresolved threads." That was an artifact: the
review bot hit its usage limit on 2026-08-28 and never reviewed it, so an
unreviewed 3,060-line diff scored as the cleanest thing on the board. Zero
threads means either "clean" or "never looked at," and the board could not tell
them apart. Any future sweep should read thread count alongside whether a
review actually ran.

**The saturation collision had five participants, not three.** #1044's
`variants.rs:229,262` called `eg.saturate_with_limit(64)`, which #1085 deletes —
disjoint files, so `git` merged clean and the *build* would have broken on
whichever landed second. Moot now that #1044 is closed, but the three-way
framing was too narrow, and the general lesson is the one the conflict table
above makes concrete: **a clean `git merge-tree` is not evidence that a branch
still builds.**

## Complications

### 1. #1054 — red, and now twice-superseded. Recommend closing.

Green at `136cc63`, red the moment it was brought up to date. Nine compile
errors in `pixelflow-codegen/src/emit/x86_64.rs`: `X86Backend::epilogue` no
longer exists (#1082 removed `prologue`/`epilogue`, methods moved into a
`driver` module) plus `E0308` mismatches from #1081's `&'static str` →
`CompileError` change. Failed on ubuntu, macOS, Clippy, and all three ISA
levels. **Git merged it cleanly and it does not compile** — the clearest
instance on this board of a semantic conflict that no merge check catches.

This is the second encoder refactor to invalidate the branch; the first
(#1055–#1062's Vex-builder rewrite) is already in its own history at
`4435869f`, which discarded and rewrote the original tests. Its tests keep
being deleted out from under it by the file it targets. Funding a third
re-close pass against a file still in motion is not a good trade — close it,
and re-run the mutation audit once `x86_64.rs` settles.

### 2. #1072 — the harness it documents has been deleted. Recommend closing.

Its conflict is not a content conflict but **modify/delete**:
`pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` was deleted from
`main` by #1093 ("delete the extraction head's shape, keep its denotation"),
and #1072 modifies it. The paper documents a training program whose harness has
since been deliberately removed.

That compounds problems the first pass already recorded and the author already
conceded: four unresolved P1s at the headline intervals (unsubtracted 4.272 ns
call overhead in every reported ratio — #1089 verified this at
`bench_extraction_3way.rs:2607` and noted the true regression is therefore
*larger* than the reported 1.0153; confounded search initialization between the
learned and static arms; an untested Round-3-vs-Round-2a comparison; a
bootstrap that resamples kernels where the corpus defines `(band, seed)`
families as the split unit), plus per-kernel artifacts and checkpoints that
cannot be recovered because the run machine's worktree is gone.

The measurements retain historical value. The branch does not: it cannot be
rebased without reinstating a binary `main` deliberately deleted. Recommend
extracting the paper and `NUMBERS.md` onto a fresh branch off current `main`,
with the intervals marked provisional and the four P1s either re-analysed or
disclosed, and closing this one.

### 3. #994 — still blocked on credentials that do not exist.

Open since 11 Aug, green, zero threads, still untested end to end. Needs five
repository secrets that have never been created, and the
codesign/notarytool/stapler path has never run. Tag-triggered, so it is inert
until someone pushes `v*.*.*`. Three weeks of drift on something CI cannot
validate. Merge it as dormant infrastructure or close it — but decide, because
leaving it open costs a rebase every time `main` moves.

### 4. The nine saturation-family conflicts

#1084, #1087, #1088, #1091, #1095, #1096, #1101, #1103, #1109 all need a merge
from `main` and real reconciliation against the landed
`SaturationStopReason` / optimizer-entry-point work. Several are large (#1088
and #1096 are 38 commits ahead) and several have active sessions. This is
rework that has to happen branch by branch by whoever owns each; it is not
mechanical, because the conflicts are in the type that four separate PRs each
redefined.

The one piece of good news: #1083's landed version deliberately converged its
stop-reason type onto the names #1084 independently chose
(`SaturationStop { Quiesced, ClassCap, IterationCeiling, Timeout }`, field
`stop`), so for #1084 at least the reconciliation should be closer to a
duplicate-delete than a rename.

## Recommended order

1. **Close #1054 and #1072**; decide #994. These three have not moved in
   ~13 hours and each is blocked on something no rebase fixes.
2. **#1114, #1113** — both already current and clean; land or review them
   before they join the conflicted set.
3. **#1109** (3 behind, one conflicted file) — cheapest of the nine to
   reconcile.
4. The remaining saturation family, owner by owner, smallest first: #1101,
   #1103, #1087, #1091, #1095, #1084, #1088, #1096.

## Method note

Branch state is from `git merge-tree --write-tree` and `git rev-list --count`
per branch against `origin/main` at `44c9fa3f`; conflicting paths from
`git merge-tree --name-only`. CI conclusions are from the check-run API at each
branch's current head, read directly rather than relayed. The #1072 file
deletion was confirmed with `git ls-tree -r origin/main` and the deleting commit
identified with `git log -- <path>`.

Two limits worth stating. Thread counts below are as of the first pass and were
not re-derived for this refresh, so treat them as indicative. And this session
is scoped to its own branch, so nothing here was pushed to another PR — the
recommendations are for the branch owners.
