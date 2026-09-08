# Open-PR triage — 2026-09-08

Scheduled sweep of all open PRs against three conditions — up to date with
`main`, no unresolved review threads, no CI failures — plus the requested
judgement on which branches are superseded, obsolete, non-salvageable, or
worth closing.

Predecessor: [2026-09-01-open-pr-triage.md](2026-09-01-open-pr-triage.md)
(#1086), landed in this same sweep. Every PR it discusses is now closed
except #1054 and #994, and its recommendation for #1054 is superseded by
the finding below.

## The headline: `main` was red, and every PR inherited it

`Feature matrix` had been failing on `main` since #1212 merged. `corpus_gaps`
landed without a `[[bin]]` entry in `pixelflow-pipeline/Cargo.toml`, so Cargo
auto-discovered it — and an auto-discovered target carries no
`required-features`. The bin reads `training::{bezier_family, sh_family}`,
gated behind the `training` feature:

```
error[E0432]: unresolved import `pixelflow_pipeline::training`
  --> pixelflow-pipeline/src/bin/corpus_gaps.rs:40:25
note: found an item that was configured out
  --> pixelflow-pipeline/src/lib.rs:24:9
   |   #[cfg(feature = "training")]
```

So `cargo check -p pixelflow-pipeline --no-default-features` and
`--features profiling` both failed. #1206 and #1209 were red for this reason
and no other; their own diffs were fine.

Fixed and merged as #1217. All 20 sibling training bins already carry
`required-features = ["training"]`; this one just never got written down.

### The gate, not the author

`Feature matrix` did catch this — correctly, and ~90 seconds and 40 workspace
checks in, long after the author saw green locally with default features on.
Per CLAUDE.md, a gap in CI is a check to write rather than a caveat to
attach, and the gap here is *when* the signal arrives.

What was actually missing is a `[[bin]]` entry, which is a property of two
adjacent files rather than of a build. `scripts/check-bin-declarations.sh`
greps for it and runs in the metadata jobs — no toolchain, no build, ~6
seconds. It deliberately does **not** require `required-features` on every
bin (`collapse_cost` correctly has none); it requires that somebody wrote the
entry down, which is the moment the question gets asked. Reverting only the
`Cargo.toml` half of #1217 reproduces the original failure through it.

CLAUDE.md's "three metadata jobs" is now four.

## Disposition

| PR | Rebased | Threads | CI | Call |
|---|---|---|---|---|
| #1217 | — | — | green | **merged** — the `main` fix above |
| #1188 | yes | 0 | green | **merged** — docs-only |
| #1199 | yes | 0 | green | **merged** — test-only, target files untouched since its base |
| #1086 | yes | 0 | green | **merged** — the 2026-09-01 triage record |
| #1154 | yes | 0 | green after repair | **merged** — repaired, see below |
| #1206 | yes | 0 (3 resolved) | green | **ready — author's call** |
| #1209 | yes | 0 (1 resolved) | green | **meets all three** — author's call |
| #994 | yes | 0 | green | blocked on credentials, correctly draft |
| #1213 | **no** | 13 open (9 P1) | never ran | active WIP |
| #1207 | **no** | 18 open | docs-skip | conflict unresolvable here |
| #1215 | **no** | 7 open (5 P1) | docs-skip | blocked on #1207 |
| #1054 | yes | 0 (2 resolved) | **red** | **recommend closure** |

## Salvageable vs not: the same series, opposite outcomes

#1154 and #1054 are consecutive passes of the same test-quality-audit series,
both invalidated by `emit/` refactors. Only one survived, and the difference
is worth naming: **whether the subject of the tests still exists.**

### #1154 — repaired

`analyze_select_guards` was rewritten by #1177 (S3b) and #1183 (uniforms).
The part that reaches these tests is a cost gate: `SelectArms::range` now
refuses any arm costing `<= MISPREDICT_PENALTY_CYCLES` (16), because a branch
can save at most what it costs when mispredicted. All three fixtures put a
single `Neg` — 3 cycles in `latency_prior` — in the guarded arm, so they
formed no guard at all and every assertion read 0 where it expected 1.

The finding was not stale: `guards.rs` still has **no `#[test]` of its own**
on `main`, re-confirmed today. Only the fixtures were. They now use `Rsqrt`
(21 cycles), which clears the gate and leaves every schedule index — and so
every asserted range — unchanged.

Added a fourth test for the gate itself. `Recip` is *exactly*
`MISPREDICT_PENALTY_CYCLES`, so it pins the boundary as `<=` rather than a
value safely past it, and asserts that coincidence rather than assuming it.

Hand mutation check after the repair:

| mutant | 3 original tests | + boundary test |
|---|---|---|
| `end = last + 1` → `last` | killed | killed |
| gate `<=` → `<` | **survives** | killed |
| `MISPREDICT_PENALTY_CYCLES` 16 → 15 | **survives** | killed |

The boundary test is what catches the latter two, which is why it is there.
Four targeted mutants is not a sweep, so the audit doc's `cargo mutants`
tallies carry a dated supersession banner: a full re-run against the current
file is still owed before "0 real gaps" is restated.

### #1054 — recommend closing

Five jobs fail on the same five compile errors. Neither symbol survives:

- **`emit_binary_safe` was deleted.** The only occurrence left on `main` is a
  comment at `x86_64.rs:948` describing it in the past tense. Two of the PR's
  tests exist solely to characterize it.
- **`X86_SCRATCH` was deleted.** It was `const X86_SCRATCH: Reg = Reg(10)` —
  one fixed scratch register. `main` replaced that with an allocator-managed
  pool, `scratch: regalloc::RegSet::range(4, 12)`, reached as
  `plan.scratch.temp(n)`.

That second one is why this is not a rebase. The tests do not name renamed
things; they encode the **fixed-scratch-register model**, and the model is
gone. There is no mechanical translation into a world where the allocator
hands out scratch from a range.

And it is the second time. The PR body records the first: #1055–#1062 landed
mid-flight, `x86_64.rs` went 864 → 1647 lines, "the first draft's ~35 tests
targeted functions that refactor deleted or reshaped," and the pass restarted.
Since its base, six more commits have touched `emit/` — #1082 deleted
`prologue`/`epilogue`, then #1177 and #1183 reshaped it again.

The `388/334/39/0` tally cannot be salvaged either, independently of the
compile errors: it describes a 1647-line file, and `main`'s is 1874 and
structurally different. The two review threads were closed on the promise of
a `cargo mutants` re-run that never happened, so "0 real gaps" has never been
true of any tree that also had these tests.

Deleting the two dead tests would compile, but it trades the PR's claimed
contribution for a green tick and leaves a stale mutation report attached.
The underlying finding — the x86-64 encoder has never had a mutation pass
survive to `main` — is real and worth redoing against today's file. The
artifact is not recoverable.

## #1213 — a semantic conflict underneath a one-hunk textual one

The textual conflict is one paragraph in
`docs/plans/2026-09-06-lattice-is-the-index.md`; both sides amend it
independently and both edits can stand.

Underneath it, the merged tree **does not compile**. This branch changes
`Font::glyph_kernel_scaled` from `-> Option<Kernel>` to `-> Option<Glyph>`.
`egraph_off_on.rs` landed on `main` separately as #1210, after the branch was
cut, and is a new consumer of the old signature — three `E0308`s.

The obvious fix is the trap. Taking `.kernel` at those sites compiles, and
`Glyph::binding`'s own doc on that branch says why it is wrong: the binding is
*required*, and "a bare `Lattice::bake` panics on the declared buffer".
`egraph_off_on` does exactly a bare bake. So `.kernel` alone converts a
compile error into a runtime panic — the same defect, in a third consumer,
that the PR already has two open P1 threads about (`collapse_cost` discarding
`glyph.binding`; the corpus replaying zeros).

Three consumers hitting the same edge suggests the shape is the finding: if a
`Glyph`'s kernel is unusable without its binding, handing out the kernel alone
is the thing to make unsayable. That is this repo's own "when you extend a
type's meaning, extend its type."

Not pushed — landing a red tree or papering over an open P1 to get a green
tick is not a trade worth making. Reported on the PR.

Independently: `cargo fmt --all -- --check` already fails on the branch tip
(`81264a4`) before any merge — five files. With 13 open threads (9 P1, several
against the newest revisions) and the PR's own "Not done yet" list, this reads
as active WIP; marking it draft would say so.

## #1207 and #1215 — blocked by the environment, not by the work

Both conflict on **`docs/results/journal.jsonl` only**, and it is a textbook
append-vs-append: the merge base has 38 lines and is a strict prefix of both
sides, `main` appended #1210's entry, each branch appended its own. The
resolution is 38 + both, and it is unambiguous.

It could not be applied from this session. `*.jsonl` is Git LFS
(`.gitattributes:31`), so resolving creates a new LFS object, and this
environment's network policy denies `lfs.github.com:443` — LFS *downloads*
succeed, uploads return 403 at CONNECT. GitHub's server-side "Update branch"
refuses too, since it hits the same textual conflict on the pointer file.

So this is an environment limit, not a judgement call. Anyone with LFS push
access resolves both in about a minute. Nothing else about either branch
conflicts.

Substantively both still need work regardless:

- **#1207** (claims ledger): 18 open threads, all P2, all specific factual
  corrections to the ledger's own arithmetic — per-group verdict counts that
  disagree with the companion CSV, a real-shader HELD count of 21 where the
  CSV has 16, rows classified FAILED whose corpus is synthetic. A document
  whose thesis is that this program's numbers were not checked needs its own
  numbers checked; that is content work, not a rebase.
- **#1215** (benchmark correction): 7 open, 5 P1 — including a held-out grid
  that already appears in five committed result artifacts, and a 5% decision
  threshold sitting below the plan's own stated 10% noise floor. One thread
  correctly observes it cannot be audited until #1207 lands. **#1207 first.**

## #1209 and #994

**#1209** is green, rebased, and now thread-free. Its last open thread — a
P2 marked outdated, saying `Loc::Remat` satisfies the public `StoreTarget`
bound while `target_storage` panics on it — was already fixed and simply
never resolved. Verified against head `87aa0dd` rather than trusting the
outdated marker: `Loc` has no `StoreTarget` impl at all (only `Reg`, `Slot`
and `Storage`, none of which can be `Remat`), no `target_storage` impl
panics, and the read side is the fallible one (`impl SourceOperand for Loc`
returning `None` for `Remat`). The branch's own history carries the commit
that did it, `fix(codegen): exclude Loc::Remat from StoreTarget`. Replied
and resolved.

That makes #1209 the second PR after #1206 to satisfy all three conditions;
both are now waiting only on a merge decision.

**#994** is green, rebased, thread-free, and correctly a draft: it is blocked
on five Apple credentials that do not exist as repository secrets, and its
signing and notarization steps have never executed. Not stale — nothing in the
tree competes with it — just parked. Leave it.

## The review bot is out of quota

Mid-sweep, `chatgpt-codex-connector[bot]` posted on #1217:

> You have reached your Codex usage limits for code reviews.

Every one of the 60-odd review threads across these PRs — and all 38 still
open — was filed by that bot; no human review is unaddressed anywhere in the
set. So while it is out of quota, **"no unresolved review comments" is
satisfied for the wrong reason** on anything pushed from here: #1154's repair
and #1218 both have zero threads, and neither has actually been reviewed.

Worth reading the two conditions separately from now on. A green,
thread-free PR pushed today has been checked by CI and by nothing else, which
is a weaker statement than the same PR made yesterday. This does not change
any disposition above — CI is the gate, per CLAUDE.md — but it is the
difference between "reviewed and clean" and "not reviewed".

## Stagger the branch updates next time

Bringing eight branches up to date in one burst queued roughly sixteen
macOS jobs against a scarce runner pool. Every `ubuntu-latest` job drained
normally; `Test on macos-latest` and the macOS launch check sat unassigned
for the better part of an hour on the last branch in the queue, which is why
#1154 finished 15/17 green with two jobs never started rather than merged
outright.

Nothing failed and nothing was learned by waiting — but the sweep was its own
bottleneck, and the fix is free: update in small batches, or update the ones
you actually intend to merge first. Worth doing on the next run of this
routine.

## Addendum — final state

Written mid-sweep; two rows moved afterwards and the table above is corrected
to match. #1154's macOS jobs eventually got runners (~70 minutes after
queueing), the run went green on all 17, and it merged. #1209's last thread
was verified stale and resolved.

Six PRs landed this run: #1217, #1188, #1199, #1086, #1218 (this document)
and #1154. Of the seven still open, #1206, #1209 and #994 satisfy all three
conditions — the first two await a merge decision, the third is correctly
parked. The four that do not are #1054 (red, closure recommended), #1213
(active WIP), and #1207/#1215 (LFS-blocked conflicts plus content work).

`/root/.ccr/README.md` is explicit that a 403 from the egress proxy is an
organization policy denial to be reported rather than worked around, so the
LFS block on #1207/#1215 is a hard stop for any session under this policy —
not a gap in this one. It is the single environment change that would let a
scheduled sweep close the "rebased" condition unaided.

## Method note

No journal entry accompanies this document: `docs/results/journal.jsonl` is
LFS and this session cannot upload LFS objects (see #1207/#1215 above).
Appending one is the first thing to do alongside landing this.
