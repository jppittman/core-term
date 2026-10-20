# Open PR sweep — 2026-10-20

Point-in-time evidence, filed per `docs/README.md`'s classification of
`results/`. PR states, CI runs and `main` positions below are as of
2026-10-20. Where a finding stays live, it is also posted as a comment on the
PR it concerns.

Six open pull requests at the start of the sweep, all drafts, all authored by
the repo owner. **No PR in the queue had an unresolved review thread** — zero
review comments exist on any of the six, so the "no unresolved comments" half
of the goal was already satisfied and needed no work.

`main` was at `29773114` (#1428, `fix(graphics): saturate cached text
coverage at one`) when the sweep began, and advanced three times during it:
`16588848` (#1375, merged by this sweep), `4e8014b2` (#1424, merged by this
sweep) and `5ac5d053` (#1429, **merged by JP** while the sweep was waiting on
CI). The three PRs left open were re-updated against `5ac5d053` as the
sweep's final action, so each one's green run was produced against the base
it will actually have — see "On the cost of 'rebased'" below.

## What changed

| PR | Title | Was behind | Action | Outcome |
|---|---|---|---|---|
| #1375 | `docs(results)`: the 2026-10-13 sweep | 51 | brought current, **merged** | merged `16588848` |
| #1424 | `test(core-term)`: two screen.rs mutation gaps | 3 | brought current, **merged** | merged `4e8014b2` (23/23 green) |
| #994 | `ci(release)`: macOS signed DMG | 51 | brought current | green; **4th sweep, still credential-blocked** |
| #1323 | `feat(compiler)`: `.at` is contramap | 51 | merge pushed manually | green; **2 owner decisions + new collision** |
| #1427 | `refactor(compiler)`: desloppify (in progress) | 2 | brought current | green; **description badly understates the diff** |
| #1429 | `perf(desloppify)`: fan out, one judge | 2 | brought current, reported ready | **merged by JP** `5ac5d053` |

Two were merged by this sweep under the trivial rule (documentation; tests).
A third, #1429, JP merged during the sweep. The remaining three are
not trivial and each is stated below with what it is actually waiting on.

## The headline complication: #1323 and #1427 collide

This is new this sweep and is the one finding that gets worse with time.

`git merge-tree` of #1323 against #1427 **conflicts in
`pixelflow-compiler/src/lower.rs`**, and auto-merges `sema.rs`, `parser.rs`
and `tests/items_block.rs` — four of the five files they share.

The asymmetry is what matters. #1323 *adds* `.at` lowering: +142 in
`sema.rs`, +23 in `lower.rs`, +5 in `parser.rs`, +4 in `emit.rs`, plus
`ExprArena::warp` in `pixelflow-ir/src/arena.rs`. #1427 *removes* at scale:
−318 in `lower.rs`, −259 in `sema.rs`, −201 in `emit.rs` across a 24-file,
−1,966-line diff. A careless resolution in that direction drops `.at`
lowering silently, and the three auto-merged files mean the conflict markers
will appear in only one of the four places where the two changes actually
interact.

**Recommendation: decide #1323 first.** It is the smaller change, it is the
one with a written denotation and a pinned contract
(`derivative_under_warp.rs`), and it has been waiting on a yes/no since
2026-10-06. #1427 is explicitly still growing ("further commits will land on
this branch"), so every day it grows the resolution gets larger and #1323's
cost of landing rises. Resolving in the other order means re-deriving `.at`
lowering against a rewritten `lower.rs`.

## #1323 — `.at` is first-class contramap

**Verified not superseded, on today's `main`:** `ExprArena::warp` exists
nowhere in `pixelflow-ir/src/`, and D5 is still the live entry in the
decision table — `| D5 | .at | application is contramap (§1.2) |` at
`docs/language/2026-09-25-the-language-is-kernel.md:647`. The reversal this
PR makes is still an open question.

**A near-miss the clean merge hid, checked rather than assumed.** `main`
moved the plan document this PR edits: `docs/plans/2026-09-25-…` is now
`docs/language/2026-09-25-…`. The PR still edits the old path. Git's rename
detection resolved it correctly — the merged tree has the file only at
`docs/language/`, carries the struck-through D5 at line 668, and leaves no
duplicate behind at `docs/plans/` — but a rename plus an edit is exactly the
shape that a textual "merges cleanly" cannot speak to, and the Oct-13 report
flagged this same class (#1339's `section_1_7.rs` deletion). Confirmed by
reading the merged tree, not by trusting the exit code.

**GitHub's mergeability flag was wrong.** The PR read `mergeable_state:
"dirty"` and GitHub's Update-branch API refused with "merge conflict between
base and head", while `git merge` of `origin/main` into the branch produced
**zero conflicts** locally. The stale flag did not clear on re-query. The
merge was therefore built and verified locally before being pushed:
`cargo test -p pixelflow-compiler -p pixelflow-ir` exits 0,
`at_is_contramap` 11/11, `derivative_under_warp` 6/6, `items_block` 6/6,
`cargo fmt --all --check` clean. Worth knowing for future sweeps: a `dirty`
flag on this repo is a claim to check, not a fact — and the remedy is a
locally verified merge commit, never a force-push.

**Still the owner's to decide** (unchanged from the PR's own description and
the Oct-13 comment, neither answered):

1. `ExprArena::warp` is new **public** API in `pixelflow-ir`. It has to be —
   the staged site emits `__arena.warp(...)` into the caller's crate, the
   same reason `apply`/`admit`/`close_fold` are public — and CLAUDE.md
   requires explicit permission to widen public surface.
2. `Kernel::at` still runs its own substitution over the DAG. A test pins it
   to the same program as `warp`, but they are two implementations of one
   idea, and CLAUDE.md calls a copy "a future divergence." Deferring is
   defensible; it should become a `docs/BACKLOG.md` entry so it is
   rediscovered as known debt rather than as a bug.

## #1427 — desloppify pixelflow-compiler

Green, and the work itself is careful. The problem is that **the PR
description describes about 4% of the diff.**

The body lists exactly one item — 82 `pub` items narrowed to `pub(crate)`,
"same-line token swaps" — and says "Crates touched: `pixelflow-compiler`
only." The actual diff is 24 files, +748/−1,966, across six commits, and
includes:

- **A crash fix.** `326d369e` — DSL-012: an entry named `_1`, `_2d` or `__3`
  produced an `Args` record whose camel-cased name starts with a digit, and
  `Ident::new` **panicked inside the proc macro**. Reproduced first, then
  fixed. A compiler-crash fix is the most reviewable thing in this PR and the
  description does not mention it.
- **A net −56 tests**: integration tests 158 → 128, `src/` unit tests
  146 → 120. `tests/jit_parity.rs` (12 tests) and `tests/kernel_5params.rs`
  (5 tests) are deleted outright.
- `Cargo.lock` and `Cargo.toml` changes (a `[dev-dependencies]` entry
  dropped), i.e. not "`pixelflow-compiler` only".

To its credit, the commit messages are the opposite of the description: every
deleted test is listed with the test that now covers its property, which is
exactly the right standard. **Spot-checked rather than taken on trust**, and
the mapping holds: `bake_vs_scalar_truth.rs` really does carry
`arithmetic_matches_scalar_truth`, `params_match_scalar_truth`,
`piecewise_matches_scalar_truth` and `abs_recip_matches_scalar_truth`;
`optimized_agrees_with_raw.rs` carries `add_sub_mul_agrees`, `mul_add_agrees`,
`div_agrees`, `sqrt_agrees`, `min_max_agrees`, `abs_and_neg_agree`,
`sin_and_cos_agree_within_domain` and `exp_agrees_within_domain`; and
`kernel_parameters.rs` carries the eight-parameter declaration-order test
that replaces five `kernel_N_params_compiles` cases. The `jit_parity.rs`
deletion is a consolidation, not a loss. **Two are listed as covered by nothing**, and
those are the ones worth a second look:

- `a_kernel_used_multiple_times_stays_one_dag` → *none* ("a value cannot tell
  a DAG from a tree").
- `a_composed_kernel_holds_only_its_program` → *none* ("a garbage-size
  property no value reveals").

**On inspection the author is mostly right, and the residual gap is narrower
than the deletion count suggests.** Checked rather than assumed: sharing is
still asserted one level down, by a unit test in `pixelflow-ir/src/arena.rs`
(line 2723, `assert_eq!(copies, by_hand, "hash-consed onto the same nodes")`).
Because `ExprArena` hash-conses, a `kernel!` body that lowered a reused name
*twice* would have its two identical subtrees merged back into one node
anyway — so `a_kernel_used_multiple_times_stays_one_dag` really could not
observe the thing its name claimed, exactly as the commit says. Deleting it
loses less than it looks like.

What is left uncovered is narrower and more specific: nothing now asserts
that a **reused name stays its own optimization unit** — one node reached
twice, as against two structurally-identical nodes that hash-consing happened
to merge. Those are indistinguishable by value *and* by node count, which is
why no surviving test sees the difference. It matters because #1323's `warp`
depends on precisely that property ("a name stays a name and stays its own
optimization unit"), so the two PRs want it pinned for opposite reasons.

Per CLAUDE.md, "a gap in CI is a check to write, not a caveat to attach" —
but the check has to observe something, and this one needs an observable the
crate does not currently expose. Worth a `docs/BACKLOG.md` entry rather than
a blocker on this PR.

**Recommendation:** not mergeable as it stands, and not because of the code —
the code is good and the commit messages are exemplary. Rewrite the
description to match the diff, and split the DSL-012 crash fix out so it can
be reviewed and landed on its own merit; a proc-macro panic fix should not
need a 24-file refactor to ship. The author's own note that this is "in
progress" with more commits coming means it is not up for merge this sweep
regardless.

## #1429 — desloppify fans out

Green, and the cleanest of the four non-trivial PRs. Deterministic ordering
of findings, two new tests that fail if the parallelism regresses
(`every_rule_about_a_unit_is_asked_at_once`,
`every_converging_module_is_named_at_once`, at `desloppify/tests/review.rs`
1179 and 1199), net +1 test, and a "What the output loses" section that
states the capability reduction instead of hiding it. It does not conflict
with #1427 — #1427 is `pixelflow-compiler/`, this is `desloppify/`.

Confirmed it is actually gated: `desloppify` appears in no workflow file by
name, which initially looked like a coverage hole, but it **is** a workspace
member in `Cargo.toml`, so `cargo test --workspace` covers it.

**Outcome: merged** as `5ac5d053`, by JP, about four minutes after the sweep
posted the above on the PR. Recorded because it is the one PR in this queue
where the sweep's job was simply to establish that nothing was blocking and
say so. (Noted below: CLAUDE.md's crate table omitted this crate; fixed in
this change.)

## #994 — macOS signed DMG: the fourth identical sweep

**Re-verified on today's `main`, not assumed**, because this is the one that
would ship a crash-on-first-glyph binary if it had rotted:

- `MAP_JIT` and `pthread_jit_write_protect_np` appear in **no** `.rs` file in
  the workspace, so the single
  `com.apple.security.cs.allow-unsigned-executable-memory` entitlement is
  still the correct minimal set and withholding `allow-jit` is still right.
- `cargo run -p xtask --release -- bundle` still resolves
  (`xtask/src/main.rs:46`), so the workflow's first step still works.

Still +249/−0 across three files, still merges cleanly, still blocked on the
same five admin-only App Store Connect secrets.

**The finding is the sweep, not the code — and this sweep paid the cost
again.** Comments of 2026-09-15, 2026-09-15, 2026-10-13 and now 2026-10-20
reach the identical conclusion. The Oct-13 comment already observed that the
rebase "was not worth its cost here" and asked for a decision either way; no
decision was made, so the sweep spent a fourth full presubmit — including a
macOS runner slot — to learn nothing new.

Three ways to stop it, in the order I would pick them:

1. **Merge it inert.** The workflow triggers on `v*.*.*` tags and nothing
   else, so on `main` with no tag pushed it never runs. It leaves the queue,
   the pipeline is simply there when the secrets appear, and the honest risk
   is a failed *release* run on the first real tag — not a broken `main`.
2. **Close it.** Additive-only across three files; it will keep merging
   cleanly, so re-applying later loses nothing. The right call if signed
   macOS releases are not on the roadmap this quarter.
3. **Stop rebasing it, and leave it open.** The rebase is what costs the
   presubmit, not the PR's existence. A sweep can verify a credential-blocked
   draft still merges cleanly with `git merge-tree` locally, for free, and
   skip the push. This is the cheapest option but it is also the one that
   keeps the PR in the queue forever.

A fourth option exists and is deliberately *not* recommended from a sweep:
extending `scripts/ci-change-scope.sh` with a narrow "release-pipeline-only"
class (`.github/workflows/release.yaml`, `assets/entitlements.plist`,
`scripts/make-dmg.sh` — all consumed only on a tag push) so that this diff
skips the build-and-test jobs the way a docs-only diff does. It would work,
but it *weakens* the merge gate for the whole repository, and that script's
own stated default is that any path outside the pattern runs everything.
Adding to the gate is a sweep's business; loosening it is the owner's.

## Nothing here is obsolete or non-salvageable

Checked explicitly, since it was asked:

- **None of the six is superseded.** #1323's `warp` is absent from `main` and
  its decision is live; #994's entitlement claim re-verified today; #1427 and
  #1429 are both days old and based within 2 commits of `main`.
- **None is orphaned.** All six share a real merge base with `main` and all
  six merge cleanly against it.
- **One recommended for closure, conditionally:** #994, and only as the
  second-choice alternative to merging it inert. It is not
  *non-salvageable* — it is correct, green, and cheap to re-apply. Closing it
  would be a roadmap decision, not a quality one.

## On the cost of "rebased"

Worth recording, because it is the structural problem behind half of this
report and the Oct-13 sweep got it wrong in both directions before settling.

A sweep that both merges and rebases cannot reach a fixed point in one pass.
Merging #1375 put the other five PRs 1 behind; merging #1424 put them 2
behind; JP merging #1429 mid-sweep put the rest 3 behind. Each of those
invalidated the "0 behind" the previous step had just bought, and the third
one was not even the sweep's own doing — which is the point. In a repository
that is being worked on, "every open PR is 0 behind `main`" is not a state a
sweep can leave behind; it is a state that decays as soon as anything lands.

The resolution used here is ordering, not argument: **merge everything the
sweep is going to merge first, then do exactly one update pass as the final
action.** `main` is settled by then, so each remaining PR gets one green run
against its real base and the sweep terminates. That costs one presubmit per
open PR per sweep, which is the honest price of the "rebased" criterion.

I considered and rejected the cheaper argument — that both intervening
commits were provably incapable of affecting the four remaining PRs. It is
true: #1375 adds one file under `docs/results/` that no build reads, and
#1424 adds two `#[test]` functions to `core-term/src/term/screen.rs`, and
`core-term` is a leaf (nothing in the workspace declares a dependency on it,
checked). But this is the same reasoning the Oct-13 sweep advanced and then
retracted, and the retraction was right for a reason that still applies: a
disjoint diff makes a semantic break unlikely, not impossible, and "unlikely"
is not what a green check is for. The argument is recorded here so the next
sweep does not have to re-derive it — and so it is visible that it was
rejected on purpose, not overlooked.

The one place this cost is not worth paying is #994, where the blocker is a
credential rather than code; that case is handled above.

## Trap for future sweeps (recurring)

This session's clone was **shallow** again (`depth ~50`). Unshallowed before
any history reasoning. The Oct-13 report recorded the same trap and it is
worth repeating, because the symptom is indistinguishable from a real
disaster: before unshallowing, `git merge-base` can report *no common
ancestor* and ahead-counts in the thousands, which is exactly the evidence
that would justify writing a branch off as orphaned by a history rewrite.
`git rev-parse --is-shallow-repository` is the check, and it belongs before
any merge-base or ahead/behind claim.

New this sweep: **`mergeable_state` is not evidence either.** See #1323
above — GitHub reported `dirty` and refused its own Update-branch API on a
branch that merges with zero conflicts. Check with git before believing it.

## Doc drift noticed in passing

CLAUDE.md's Workspace Structure says "Cargo workspace with **13** member
crates" and its table omits `desloppify`, which `Cargo.toml` has listed as a
14th member for some time. Corrected in the same change as this report.
