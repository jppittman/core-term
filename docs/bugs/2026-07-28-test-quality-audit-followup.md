# Test quality control follow-up — 2026-07-28

Scope: scheduled continuation of `docs/bugs/2026-07-26-test-quality-audit-followup.md`.
Since that pass (commit `94b42f3`), the actor-scheduler crate went through a
refactor renaming `park` → `handle_os` across the workspace and introducing a
new `Host`/`Transducer` sweep model for green (co-located) actors
(`actor-scheduler/src/host.rs`, commits `78cad80`, `acb815c`, `3eaba85`). That
file is new ground: no prior audit pass had mutation-tested it.

## Static audit: no new violations

Two `Explore` sub-agents independently swept every file touched since
`94b42f3` (`git diff 94b42f3..HEAD`) against docs/STYLE.md's "Test Public API"
rule:

- **actor-scheduler**: the only substantive new/modified tests are the five
  new ones in `host.rs`'s own test module covering the `Host`/`Transducer`
  sweep refactor. All of them build exclusively from the crate's public
  surface (`Host::new`/`adopt`, `Node::new`/`poll`, public fields on
  `Supervision`/`HostOut`) — none touch a private field or call a private
  method directly. `lib.rs`, `mealy.rs`, and the `tests/*.rs` files in this
  crate are pure `park`→`handle_os` renames with no logic change.
- **core-term / pixelflow-runtime**: every changed line across the 20+ files
  in this diff (event_monitor_actor, terminal_app, display/platform code, and
  their test files) is the same mechanical rename, plus one cosmetic local
  struct rename (`ParkTrackingActor` → `HandleOsTrackingActor`) in
  `pixelflow-runtime/tests/actor_model_tests.rs`. No new violations.

`pixelflow-graphics/src/spatial_bsp.rs` (the one item left open by every prior
pass — 19 tests still index the private `bsp.interiors[...]` array) is
unchanged and still open; still a design call (test-only accessor vs.
property-test rewrite vs. documented exception), not attempted here.

## Mutation testing: `actor-scheduler/src/host.rs`

Installed `cargo-mutants` fresh (v27.1.0, not present in this environment —
consistent with every prior pass's note that it doesn't persist between
sessions). Ran it scoped to `host.rs`, the new sweep code:

**Before: 59 mutants — 3 missed, 23 caught, 29 unviable, 4 timeouts.**

1. **`Host::is_empty()` → `true`** (host.rs:238) — missed. Every existing
   test that calls `is_empty()` only ever expects `true` (checking a host is
   empty after removing its only actor), so a mutant that hard-codes `true`
   passes unnoticed. Fixed by asserting `!host.is_empty()` right after
   adopting an actor, in the new test below.

2. **`i += 1` → `i *= 1` in `Host::sweep`'s `Step::Ran` arm** (host.rs:269) —
   missed, and the more interesting one. `sweep`'s doc comment promises to
   "advance every green actor by at most one step" per call. With `i *= 1`,
   `i` never leaves index 0 once an actor there produces `Step::Ran`, so — if
   that actor has more than one message queued — the sweep loop keeps
   re-polling the *same* actor until its queue is empty, only then moving on.
   That silently breaks the documented single-step contract for any actor
   with backlog, and every existing test happened to queue exactly one
   message per actor before sweeping, so nothing distinguished "step once" from
   "drain the queue." Added `a_sweep_advances_at_most_one_step_per_actor`:
   queues two messages for one actor, takes one `sweep()`, and asserts only
   the first was forwarded — the second must wait for the next `sweep()`
   call. Bounded, deterministic, no timing dependency.

3. **`GreenSender`'s `Debug` impl → `Ok(Default::default())`** (host.rs:349)
   — missed; the impl was never exercised by any test. Added
   `green_sender_debug_names_its_type`, asserting `format!("{:?}", sender)`
   contains `"GreenSender"`.

All three fixes reach the code exclusively through `actor-scheduler`'s public
surface (`Host::new`/`adopt`/`sweep`/`is_empty`, `spsc_channel`,
`green_channel`, `ActorScheduler::new`/`waker`) — no private-field or
private-method access, consistent with docs/STYLE.md's testing rule.

**After: 59 mutants — 0 missed, 26 caught, 29 unviable, 4 timeouts.**

### The 4 timeouts are not coverage gaps

`i += 1` → `i *= 1` on `sweep`'s `Blocked`/`Idle` and `Disconnected` arms
(host.rs:271, 282), and `<` → `==`/`<=` on `step_data`'s
`self.reported < self.stuck.len()` (host.rs:503), all time out rather than
get caught by an assertion. In each case the mutant makes the loop that
contains it never terminate (an actor stuck at `Idle`/`Disconnected` is
re-polled forever within one `sweep()` call; a `step_data` continuation loop
that never sees `more_findings` go false never stops re-stepping). Any test
that exercised the buggy branch would have to hang forever to observe the
difference — cargo-mutants catches this as a 20s timeout, which is the
correct signal (real code running this mutant would also hang), but writing
a unit test to "kill" it the normal way would mean shipping a test that only
passes by never returning, which is worse than the gap it would close. Same
judgment call the 2026-07-24 pass made for `actor-scheduler`'s timing-internal
backoff arithmetic under the Flexibility clause. Left as-is.

## Verified

- `cargo test -p actor-scheduler --lib`: 133/133 (was 130/130 before the 3
  new tests).
- `cargo mutants -p actor-scheduler --file actor-scheduler/src/host.rs`:
  0 missed (re-run after the fix, confirmed above).
- `cargo check --workspace`: clean.

## Recommended next steps (not done here)

1. `spatial_bsp.rs`'s private `interiors[...]` indexing (open since
   2026-07-20) still needs a human design call.
2. `core-term/src/terminal_app.rs`'s `create_test_app()` calling
   `new_registered` instead of `spawn_terminal_app()` (open since 2026-07-24)
   — still judged an intentional seam (avoiding real thread/window/PTY I/O
   in a fast unit test), not re-litigated here.
3. If future refactors add more sweep/host-adjacent code, mutation-test it
   the same way before it accumulates the way `host.rs` had — this pass
   found real, non-cosmetic gaps in code that had been in the tree with a
   green test suite the whole time.
