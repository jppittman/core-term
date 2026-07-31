# Test quality control follow-up — 2026-07-31

Scope (as originally run, 2026-07-31): scheduled continuation of
`docs/bugs/2026-07-28-test-quality-audit-followup.md`. Since that pass, 32
commits had landed, including a substantial actor-scheduler refactor
(`refactor(actor-scheduler): delete supervision and shedding; the crate owns
send policy` #968, `feat(actor-scheduler): one runtime primitive — step_os on
Transducer, DedicatedThread driver` #967) and unrelated graphics/IR/compiler
work. Two things this pass: (1) a static audit of every test file changed
since `d7d33b8` (at that point in history) against docs/STYLE.md's "Test
Public API" rule, via a background `Explore` sub-agent; (2) `cargo-mutants`
(installed fresh, v27.1.0, consistent with every prior pass's note it
doesn't persist between sessions) scoped to actor-scheduler's most-refactored
files: `host.rs`, `lifecycle.rs`, `sharded.rs`, `spsc.rs`, `error.rs`,
`mealy.rs`, `lib.rs` (318 mutants).

**This branch has since been rebased through several later merges** (most
recently onto `main` past #980/#987/#981/#964/#986/#983/#985/#979/#976). The
"since `d7d33b8`" static-audit scope above describes what was actually
reviewed on 2026-07-31 and does not extend to cover commits from the
2026-08-01/02/04/05/06/07/08 audit passes that landed afterward — each of
those is its own dated doc in this directory, and any test file they touch
that this document doesn't mention was outside this pass's scope, not
silently found clean.

## Static audit: one new violation, fixed

**`actor-scheduler/src/mealy.rs`** — three new tests added alongside the
`step_os`/`DedicatedThread` refactor read the private `Node.continuation`
field directly instead of relying on public return values:
`step_os_continuation_lands_in_the_slot`,
`an_endless_self_yielder_does_not_starve_step_os`,
`a_pending_continuation_survives_step_os`. In every case the same fact was
already, or could cheaply be, proven through `poll`/`poll_os`'s public
`(Step, ActorStatus)` return values and the test fixture's own `last_resumed`/
`resumed` fields — e.g. "the slot is empty" is provable by asserting the next
`poll()` returns `Step::Idle` rather than reading `node.continuation.is_none()`
directly.

**Not fixed in this pass's diff.** By the time this fix was ready, `main` had
moved: `docs/bugs/2026-08-01-test-quality-audit-followup.md` independently
found and fixed the same three violations one day later, before this branch
merged. That fix is what's on `main`; rebasing this branch onto it drops the
would-be-duplicate rewrite rather than reapplying it on top. Recorded here so
the finding isn't lost, not as a claim that this pass did the work.

Two other diffs were reviewed and judged not violations, both matching
already-established precedent: `pixelflow-runtime/src/coordinator_node.rs`
(new file) tests the private `CoordinatorCore` directly, but mirrors the
unmodified, never-flagged pattern already used for `VsyncCore`/`RasterCore`
("pure core, thin actor shell" types documented as intentionally
table-testable); `pixelflow-graphics/src/render/scene.rs`'s
`chunked_bake_matches_whole_stripe` calls a private chunking function to force
boundaries a production-sized frame wouldn't hit, with an inline comment
explaining why — same shape as the accepted `window.rs`/`mouse.rs`
exceptions. Everything else in the diff (all of `tests/*.rs`, `host.rs`'s
mechanical parts, `cell_grid.rs`, `atlas.rs`, `rasterizer/actor.rs`) is
compliant.

## Mutation testing: actor-scheduler's refactored code (318 mutants)

**Before: 48 missed, 154 caught, 89 unviable, 27 timeouts.**

### `sharded.rs` — 18 of the 18 missed mutants investigated; 13 real gaps fixed

This file's `ShardedInbox::drain`/`take_one` round-robin arithmetic has been
flagged as a coverage gap since **2026-07-20** and never fixed in any of the
four follow-up passes since. Investigated all 18 by hand before writing
anything, to separate real gaps from equivalent mutants rather than write
tests that would only add noise:

- **Real gaps (13, now fixed):**
  - `drain`'s status-selection guard (`total >= limit || !all_empty`,
    `||`→`&&` and `delete !`) was never independently exercised: every
    existing test uses a `limit` evenly divisible by the shard count, so
    `total` reaches `limit` exactly when a per-shard cap also fires,
    making the two guard terms indistinguishable. Added
    `drain_reports_more_when_per_shard_caps_leave_total_below_limit`
    (`limit=5`, 2 shards, `per_shard=2` — caps stop both shards at a
    combined 4, strictly under the 5 limit, so `!all_empty` alone must
    carry the guard).
  - `drain`'s round-robin rotation (`self.round_robin = (self.round_robin +
    1) % n`, `%`→`/` and `+`→`*`) leaves `round_robin` stuck at 0 forever
    (it starts at 0), invisible to any count-based assertion — only item
    *order* across successive calls can see it. Added
    `drain_round_robin_start_rotates_across_calls`.
  - `take_one` (reached via the public `Inbox::take` trait method, not the
    private fn directly) had **zero test coverage at all** before this pass
    — every arithmetic op in its search-index computation
    (`(round_robin + i) % n`) and both its rotation updates survived. Added
    `take_skips_empty_shards_via_wrapping_search`,
    `take_round_robins_across_shards_and_wraps`, and
    `take_rotates_the_start_index_even_on_a_total_miss`. Review found
    `take_round_robins_across_shards_and_wraps`'s first version still didn't
    kill the post-hit `%`→`/` mutant: refilling only some shards afterward
    let the search's own wraparound (which tries every index regardless of
    where `round_robin` starts) mask an error in its exact value. Rewrote to
    refill every shard, so the very first index tried is guaranteed to hit
    and the returned value's identity reveals the rotated index directly;
    reproduced the original version passing against the mutant, confirmed
    the rewrite doesn't.
  - `drain`'s `total += 1` → `*=` was misclassified equivalent below in this
    pass's first draft — wrong, per review. The claim ("total's accumulated
    value never independently changes which branch is taken") only checked
    `total`'s use in the post-loop status computation; it missed `total`'s
    *first* use, the in-loop early-exit guard (`total >= limit ||
    shard_count >= per_shard`, line 116). When shard count exceeds `limit`,
    `per_shard = (limit / n).max(1)` floors to 1, so with the mutant's
    `total` stuck at 0 that guard's `total >= limit` disjunct never fires —
    every shard independently delivers its one `per_shard`-capped message,
    `n` messages instead of the requested `limit`. Reproduced by hand (4
    shards, `limit=2`: real code delivers 2, the mutant delivers 4). Added
    `drain_total_stops_the_scan_once_the_limit_is_hit_even_with_more_shards_than_limit`.
- **Equivalent mutants (5, documented, not chased)** — every remaining miss
  matches a mutant predicted equivalent by hand:
  - `% → +` on both rotation lines (147, 190, 199): `round_robin` is only
    ever read through `% n` at its use sites, and `+n` is `≡ 0 (mod n)`, so
    the mutant produces an identical index sequence, just with the stored
    field growing unboundedly instead of wrapping.
  - `total >= limit` → `<` at both the outer guard (151) and the inner
    branch (153): the outer guard's `total>=limit` disjunct is dead in
    practice — reaching it always coincides with `!all_empty` already being
    true (per the point above) — and the inner branch's two arms (`if
    total>=limit` and the final `else`) both return the identical
    `Ok(DrainStatus::More)`, so no return value distinguishes them. (Worth a
    human look as a code-simplification candidate — `else if all_empty {
    Empty }` at line 155 is unreachable dead code by the same argument — but
    that's a production-logic change outside this pass's scope of
    test-only fixes.)

### `host.rs` — `GreenSender`'s wiring-flush delivery path had no coverage

**Correction (per review): public `send`'s wake path was already covered.**
`a_green_send_wakes_a_wired_host_too` and
`a_green_send_wakes_a_host_asleep_on_its_doorbell` (both pre-existing,
real-thread tests, unrelated to this pass) already call the actual public
`GreenSender::send` and would starve past their 5-second deadline if its
body were replaced with `Ok(())`. Whether cargo-mutants' original run for
this file genuinely exercised those two slow, real-thread tests isn't
independently confirmed here — but under normal `cargo test`, a `send`-body
mutant does not survive. `green_sender_send_delivers_the_message` (kept from
this pass) isn't closing an unconditional gap; it's faster, synchronous,
delivery-only coverage of the same method, useful on its own merits.

The confirmed-uncovered surface, both then and now, is `PortTarget::
try_deliver` — how `Wiring::flush` reaches a green actor from *another
node's output*, a different call path from `send` with its own separate
`waker.wake()` call. `green_sender_delivers_through_wiring_as_a_port_target`
(this pass) closes the coarse case (body replaced with `Ok(())`, losing both
delivery and wake). Review surfaced a finer one it doesn't close: patching
just the wake away — `try_deliver`'s delegation from `self.try_send(msg)` to
a bare `self.tx.try_send(msg)` (delivery intact, wake dropped) — survives it,
since that test reads the destination's inbox directly and can't distinguish
a wake call from a dropped one. Confirmed by hand (patched the delegation,
watched the existing test still pass, then added
`a_wiring_flush_wakes_a_host_asleep_on_its_doorbell` — a real destination
host asleep on a real thread, woken only by a directly-polled source node's
flush, the `try_deliver` counterpart to the pre-existing `send`-based
sleeping-host tests — and confirmed *that* fails loudly against the same
patch).

**After (workspace-relevant subset).** `cargo test -p actor-scheduler --lib`:
149/149 (was 140 on `main` immediately before this branch's diff; 9 new
tests here — 6 in `sharded.rs`, 3 in `host.rs`). `cargo test -p actor-scheduler` (incl.
integration and doctests): all passing. `cargo check --workspace` and
`cargo clippy -p actor-scheduler --lib --tests`: clean. Not restating a
gaps-closed total for `host.rs`: the original "2 missed" cargo-mutants count
predates this correction and no longer maps cleanly onto what's actually
gapped versus already-covered.

## Investigated, deferred to a future pass

- **`actor-scheduler/src/lib.rs`'s `DedicatedThread::sweep`** (brand new in
  this refactor, never mutation-tested before): 4 missed mutants on
  `sweep_burst` bookkeeping (`polls += 1` → `*=`, `polls >= sweep_burst` →
  `<`, and two `||`/`==` conditions gating the busy/idle report). `sweep` is
  private, reached only through the real-OS-thread `DedicatedThread::run` —
  the existing integration test
  (`a_flooded_lane_does_not_starve_shutdown`, `actor-scheduler/tests/
  dedicated_thread.rs`) proves the burst bound exists at all (an unbounded
  drain would hang it), but a deterministic test for the *exact* boundary
  needs either test-only introspection (against CLAUDE.md's minimal-API
  rule) or a timing-sensitive real-thread test whose reliability is itself
  a design question — a judgment call worth more time than this pass had.
- **`actor-scheduler/src/mealy.rs::Topology::find_cycle`** (982): 2 missed
  mutants on the DFS visited-state guard (`mark[to] == Mark::Unseen`,
  guard→`true` and `==`→`!=`). Hand-traced several candidate graphs
  (including the existing `a_diamond_is_acyclic` shape) looking for one
  where a spuriously-revisited `Done` node changes `validate()`'s answer;
  the diamond case turns out *not* to distinguish the mutant (redundant
  re-exploration of a leaf is harmless), and constructing a graph that does
  needs more careful adversarial-case work than warranted rushing here,
  given this is a correctness-critical path (a missed cycle here would let
  a real deadlock-prone actor wiring through undetected). Flagging for a
  dedicated look next pass rather than shipping an unverified test.
- **`actor-scheduler/src/lib.rs`'s `backoff_with_jitter`/`send_with_backoff`**
  and **`handle_wake`'s 2-mutant `||`-grouping gap**: both re-confirmed
  present in this run, already investigated and left as-is in 2026-07-24 (the
  precise cases need either a racy real-timeout test or multi-producer
  sharding disproportionate to the gap's severity) — not re-litigated.
- **`spsc.rs`'s ring-buffer arithmetic** (1 missed, several timeouts): flagged
  as concerning since 2026-07-20 ("SPSC correctness is concurrency-critical,"
  timeouts meaning a wrong mutant hangs rather than fails cleanly) — still
  open, not touched this pass.
- **`pixelflow-graphics/src/spatial_bsp.rs`**: unchanged, still the oldest
  open item (since 2026-07-20), still needs a human design call.

## Recommended next steps

1. `DedicatedThread::sweep`'s burst-boundary mutants and `Topology::
   find_cycle`'s guard mutants are this pass's two open, real (not
   equivalent) findings — both need dedicated design/construction time
   rather than a rushed fix.
2. Line 155 of `sharded.rs::drain` (`else if all_empty { Ok(DrainStatus::
   Empty) }`) is unreachable dead code by the equivalence argument above;
   worth a human sanity-check and possible removal under CLAUDE.md's
   "subtract before you add," but that's a production-logic change, not a
   test fix.
3. `spatial_bsp.rs` and `spsc.rs` remain open from prior passes.
