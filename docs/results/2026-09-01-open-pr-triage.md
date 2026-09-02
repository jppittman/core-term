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

## Review threads: the headline number was wrong

Both earlier passes carried "~60 unresolved threads across 8 PRs, 13 of them
P1." That figure was minted on 2026-09-01 and never re-derived. Re-derived now,
across all fifteen open PRs:

**There is exactly one unresolved review thread.**

| PR | threads | resolved | unresolved |
|---|---|---|---|
| #1084 | 28 | 28 | 0 |
| #1072 | 24 | 23 | **1** (P2) |
| #1091 | 9 | 9 | 0 |
| #1054 | 2 | 2 | 0 |
| #994, #1086, #1087, #1088, #1095, #1096, #1101, #1103, #1109, #1113, #1114 | 0 | — | 0 |

Two things caused the drift. Seven of the PRs carrying threads merged overnight
(#1049, #1051, #1053, #1079, #1081, #1083, #1085) and took their threads with
them. And the branch owners worked through the rest: #1084 closed all 28,
#1091 all 9, each with a substantive reply rather than a dismissal — the #1091
responses in particular confirm the finding against source, name the fix, and
disclose which committed numbers the fix invalidates.

The remaining one is on #1072 (`Recompute baselines on the Round-3 DEV corpus`,
P2) — a branch this document already recommends closing, so it will most likely
be resolved by that decision rather than by a fix.

The lesson is the same one the #1044 correction taught, pointing the other way:
a thread count is a snapshot of a moving target, and quoting a stale one
overstates the work as badly as misreading a zero understates it. Any future
sweep should re-derive it rather than carry it forward.

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

### 4. The conflicted branches, triaged by what actually blocks each

Working through them empirically rather than by inspection changed the picture:
two were mechanical and are done, and the rest split into two clearly different
kinds of work. The blocker is **never** the stop-reason redesign by itself — in
every case `main`'s landed version strictly supersedes the branch's, including
the stop reason for an application budget (`SaturationStop::ApplicationBudget`)
and the application-capped run itself (`Budget::Applications(u64)`, whose own
doc calls it "the budget the research arms compare under"). What blocks each
branch is whatever *else* it added to the same files.

**Kind 1 — superseded core, mechanical. Both done.**

*#1109* and *#1087* carried only the stop-reason work plus an `#[ignore]`d
measurement module. Recipe: take `main`'s `graph.rs`/`saturate.rs`/`mod.rs`
wholesale, keep the branch's harness intact as its own module, and port that
harness off the deleted `env_extraction_policy`/`saturate_with_full_budget`
onto `Optimizer` + `Budget::Explicit` + `hard_ceiling`, which is what keeps the
caps as parameters a measurement varies. Both pushed and now merge clean.

Two things worth stealing from those merges. First, `git` interleaved #1087's
`production_telemetry` module with `main`'s `congruence_gap_probe` *inside each
other's bodies* and produced an unclosed delimiter; the fix is to take `main`'s
file whole and re-append the branch's module from its own side, not to resolve
hunk by hunk. Second, each branch's `saturation_stop.rs` tests turned out to be
a strict subset of `main`'s, name for name — independent confirmation that
nothing is lost by taking `main`'s.

**Kind 2 — an additive provenance subsystem to re-thread. #1101, #1103.**

Core delta is only 237 lines, which makes these look like Kind 1. They are not.
Alongside the superseded stop-reason work each carries `ApplicationId` threaded
through `UnionEvent`, `ApplicationRecord`, `active_application` and
`derivation_ancestors_tight`, across `graph.rs`, `provenance.rs` and
`labeler.rs`. `main` has none of it, so it must survive — but `main` rewrote the
functions it threads through, so it has to be re-applied by hand.

Two traps, both hit and recorded: `git checkout --theirs graph.rs` silently
destroys the additive work (it takes the whole file, including the parts that
auto-merged cleanly), and resolving hunk-by-hunk instead leaves fragments of the
branch's `saturate_until_applications` orphaned *inside* `main`'s rewritten
function — dangling `'outer`, `max_total_applications`, `mid_sweep_stop`. The
workable route is `main`'s file whole, then re-thread `ApplicationId`
deliberately.

**Kind 3 — a large additive subsystem. #1084, #1088, #1091, #1095, #1096.**

800–1,100 lines added to `graph.rs`/`saturate.rs` alone: the GuidedSaturation
machinery each experiment runs on. "Take `main`'s" would delete the experiment.
These need the subsystem re-applied onto `main`'s restructured `saturate.rs` by
someone who can say which parts the registered claim depends on. They also do
not share a resolvable base — all five fork from the same merge-base with `main`
but none is an ancestor of another, so there are five of these, not one.

Sizes, for planning: #1084 1,011 core / 28.5k total; #1088 1,091 / 308.9k;
#1091 937 / 41k; #1095 804 / 42.5k; #1096 961 / 60k.

**One collision to expect.** #1087 and #1101 both add a module named
`production_telemetry` to `runtime.rs`. They do not conflict today because
neither has landed; whichever goes second will need a rename.

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
