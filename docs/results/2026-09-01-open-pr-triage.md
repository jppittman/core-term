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

## Board at 2026-09-02 10:40 UTC

Six of fifteen merge cleanly; the one red CI is fixed. Rows this pass changed
are marked.

| PR | Branch | Behind | Merges? |
|---|---|---|---|
| #1114 | `claude/class-cap-live` | 0 | clean |
| #1113 | `claude/upward-congruence` | 0 | clean |
| #1109 | `claude/cap-break-ab` | 0 | **clean — reconciled this pass** |
| #1103 | `claude/all-rules-numeric-first` | 9 | conflict (6 files) |
| #1101 | `claude/rule-order-numeric-first` | 6 | conflict (5 files) |
| #1096 | `claude/phase3-r2g` | 9 | conflict (4 files) |
| #1095 | `claude/phase3-label-constfold` | 9 | conflict (4 files) |
| #1091 | `claude/phase3-domain-shift` | 5 | conflict (4 files, 12 hunks) |
| #1088 | `claude/phase3-round2` | 9 | conflict (4 files) |
| #1087 | `claude/saturation-telemetry` | 10 | conflict (6 files, 23 hunks) |
| #1086 | `claude/brave-faraday-tw3054` | 0 | clean (this doc) |
| #1084 | `claude/phase3-guide` | 9 | conflict (5 files) |
| #1072 | `claude/workshop-writeup` | 17 | conflict — modify/delete |
| #1054 | `claude/zen-babbage-wjmnit` | 0 | **clean, CI fixed this pass** |
| #994 | `claude/macos-release-signing-pipeline` | 0 | **clean, brought current** |

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

### 1. #1054 — was red, now fixed. Recommend a mutants rerun, then merge.

Green at `136cc63`, red the moment it was brought up to date. Nine compile
errors in `pixelflow-codegen/src/emit/x86_64.rs`: `X86Backend::epilogue` no
longer exists (#1082 removed `prologue`/`epilogue`, methods moved into a
`driver` module) plus `E0308` mismatches from #1081's `&'static str` →
`CompileError` change. Failed on ubuntu, macOS, Clippy, and all three ISA
levels. **Git merged it cleanly and it does not compile** — the clearest
instance on this board of a semantic conflict that no merge check catches.

**Fixed** in `a42a16d5`: `emit_movups_store_base_*` now drive
`emit_movups_store` through a `NoDisp` address (the encoding the deleted
function produced), the redzone overflow test compares against
`CompileError::Internal(..)`, and the four `prologue_*`/`epilogue_*` tests are
removed. That removal was checked rather than assumed: `prologue`/`epilogue`
appear nowhere in the workspace, and `frame_alloc`/`frame_free` are
unconditional one-line delegations to `emit_sub_rsp`/`emit_add_rsp`
(`x86_64.rs:1298-1304`) with no `frame_bytes > 0` branch left to cover. Deleting
tests for deleted code, not dropping tests for green. Gates: `-p
pixelflow-codegen --lib` 125/0, full workspace green, clippy `-D warnings`
clean, `fmt` clean.

Still open: this is the second encoder refactor to invalidate the branch (the
first, #1055–#1062's Vex-builder rewrite, is in its own history at `4435869f`),
and the audit's "0 real gaps" conclusion has never been re-verified since
either — `emit_vpextrd_to_gpr` and `emit_vmovss_load_scaled` carry no coverage
claim at all. Rerun `cargo mutants -p pixelflow-codegen --file
pixelflow-codegen/src/emit/x86_64.rs -- --lib --test collapse_loop` and record
real numbers, or soften the conclusion, before merging.

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

### 4. The saturation-family conflicts, and what kind of work they actually are

#1084, #1087, #1088, #1091, #1095, #1096, #1101, #1103 need a merge from `main`
and reconciliation against the landed stop-reason and optimizer-entry-point
work. #1109 was in this set and is now resolved (below), which is what makes
the rest classifiable rather than merely daunting.

Each conflicted branch is **two separable jobs**, and conflating them is what
makes the pile look worse than it is.

**(a) The core delta is superseded, and resolving it is mechanical.** Every one
of these branches carries its own answer to "why did saturation stop," written
before #1083 landed one. #1087 is the clearest case: its `graph.rs` conflict is
ten hunks of `bool truncated` against `main`'s `ScanStop { Completed, ClassCap,
Deadline }`. That is the same fact at strictly more resolution — and it is this
codebase's own rule, *extend the type, not the convention*, already applied on
the `main` side. There is nothing to weigh: take `main`'s. The same shape holds
for the `saturate.rs`, `mod.rs`, and test-import hunks.

**(b) The harness collision is the real work.** These branches also add
`#[ignore]`d measurement modules, and `main` has since added its own in the same
file regions. #1087's `runtime.rs` conflict includes a single 456-vs-668-line
hunk where its telemetry harness meets `main`'s #1106 congruence probe, each
with near-duplicate helpers under different names (`load_arena` vs
`load_arena_dump`, `arena_cost` vs `arena_static_cost`). Both must survive, and
the helpers want deduplicating rather than both being kept. That is ~1,100 lines
of careful test-only reconciliation per branch. It cannot break production — it
is all `#[cfg(test)]` — but it decides whether a published measurement is
reproducible, so it should be done by someone who can say which helper is
authoritative.

The branches also do **not** share a resolvable base. #1084, #1088, #1091,
#1095 and #1096 all fork from the same merge-base with `main`, but none is an
ancestor of another: five independent lines, five registered experiments, no
single resolution that templates them.

The one shortcut worth knowing: #1083's landed version deliberately converged
its stop-reason type onto the names #1084 independently chose
(`SaturationStop { Quiesced, ClassCap, IterationCeiling, Timeout }`, field
`stop`), so #1084's job (a) should be closer to a duplicate-delete than a
rename.

### 5. #1109 — resolved, and the template for job (a)

Worth recording because its raw conflict was the most alarming on the board and
the least real: an ~890-line region in `runtime.rs`, which turned out to be
nothing but both sides appending an independent module at end-of-file
(`main`'s #1106 probe and this branch's `cap_break_ab`). Both kept; neither
touches the other. Its actual subject — classify a `ClassCap` sweep without
ending the run, and re-arm `stop` so it names the *last* sweep — is two lines in
`graph.rs` and merged clean.

The genuine work was porting its harness: it drove `saturate_with_full_budget`
and extracted through `env_extraction_policy()`, which #1108 deleted. It now
runs through `Optimizer` — production's own entry point — with
`Budget::Explicit` plus `hard_ceiling`, which is what keeps the caps as
parameters, and this A/B requires that: both arms must meet identical caps for
the control flow to be the only difference between them. Stats come off
`OptimizerStats`, extraction off `Optimized::to_arena`.

One consequence flagged rather than buried: the old path honoured
`PIXELFLOW_NNUE_WEIGHTS` through `env_extraction_policy`; the new one takes
production's default static latency prior. That matches what production does
today, but a previously recorded NNUE arm would not reproduce through this
harness.

Verified before pushing: all 7 of `saturation_stop.rs` — including
`class_cap_reports_class_cap_not_quiesced` and
`productive_but_class_capped_final_sweep_is_class_cap`, the two that would catch
this exact change going wrong — and all 7 of `optimizer_laws.rs`, plus
`cargo test -p pixelflow-search` 197/4/2/7/7 passing, clippy `-D warnings`
clean, `fmt --check` clean.

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
