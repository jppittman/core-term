# Test quality control follow-up — 2026-08-07

Scope: scheduled continuation of
`docs/bugs/2026-08-01-test-quality-audit-followup.md`. Since that pass
(commit `eb03ee9`), the tree picked up 4 commits and ~2,750/-2,760 changed
lines, concentrated in a codegen-split refactor
(`eeec0ce refactor(pixelflow): split codegen out of the IR, and stop
rendering when nothing changed` — moves `pixelflow-ir`'s emit backends into
a new `pixelflow-codegen` crate) plus a training-feature build fix and a
`pixelflow-pipeline` test-isolation fix — the smallest delta between audit
passes to date.

## Static audit: one violation, fixed independently by #985

An `Explore` sub-agent swept `git diff eb03ee9..HEAD` against docs/STYLE.md's
"Test Public API" rule, cross-checked against every prior pass's accepted
exceptions (`spatial_bsp.rs` interiors indexing, `create_test_app()`/
`new_registered`, `chunked_bake_matches_whole_stripe`, actor-scheduler's two
`try_send`-using backpressure tests) so those weren't re-flagged. Almost
every touched test in this diff is a mechanical import-path rewrite
(`pixelflow_ir::backend::emit::…` → `pixelflow_codegen::emit::…`) with no
privacy change; every renamed symbol the tests call
(`compile_arena_dag`, `ExecutableCode`, `JitManifold`, `OpKind::from_index`/
`index`/`from_name`, ...) stayed public across the move.

One real, newly-added violation:

1. **`pixelflow-pipeline/src/training/corpus.rs`** —
   `v1_corpus_is_refused_with_regeneration_hint` (new test, part of this
   pass's version-rejection feature) called the private
   `read_corpus_bytes(&data)` directly with a hand-built byte buffer instead
   of the crate's actual entry point, `pub fn read_corpus(path: &Path)`.
   `read_corpus` is a two-line wrapper (`std::fs::read(path)?;
   read_corpus_bytes(&data)`), so the private call wasn't reaching anything
   the public path couldn't. Rewrote to write the hand-built bytes to a temp
   path via the same commit's own `unique_tmp` helper and call
   `read_corpus(&tmp)`, asserting on the returned `io::Error` exactly as
   before — no loss of test intent.

   `cargo test -p pixelflow-pipeline --features training --lib
   training::corpus::`: 9/9 after the fix.

   **This fix is not in this pass's diff.** The next scheduled pass
   (`docs/bugs/2026-08-08-test-quality-audit-followup.md`, #985) swept an
   overlapping window, found the same violation, and landed a functionally
   identical fix first — same temp-file-through-`read_corpus` shape, differing
   only in the `unique_tmp` label. This pass's branch was rebased onto that,
   dropping the duplicate hunk and keeping `main`'s version. Recorded here
   because the finding was genuine and independently reproduced; the
   duplicated *work*, not the finding, is the thing worth avoiding — see
   "Recommended next steps" for the overlap-window issue that caused it.

Worth noting, not flagged as a violation: the new `pixelflow-codegen` crate
inherits a large body of `mod tests` blocks (`emit/mod.rs`, `emit/avx2.rs`,
`emit/avx512.rs`, `emit/aarch64.rs`) calling module-private encoder functions
(`emit_unary`, `emit_binary`, ...) directly. Unchanged in substance from
before the move (only `crate::` → `pixelflow_ir::` path fixes) and arguably
load-bearing for this crate specifically — its job is emitting exact machine
instructions, and there is no public surface that lets a test assert on raw
opcode bytes short of calling the encoder. Same category of carve-out as the
actor-scheduler backoff-internals exception from 2026-07-24; flagging for
awareness only, since it predates this audit window.

## Mutation testing: the three targets 2026-08-01 left open

That pass's "Recommended next steps" named three large, never-mutation-tested
surfaces from the prior delta, in priority order:
`pixelflow-runtime/src/coordinator_node.rs` (871 lines),
`pixelflow-runtime/src/engine_troupe.rs` (rewritten), and
`actor-scheduler/src/mealy.rs`'s own 969-line `DedicatedThread`/`step_os`
rewrite. All three fit comfortably in this pass (mutant counts were much
smaller than the file sizes suggest, since most of the code is trait
plumbing and doc comments).

### `actor-scheduler/src/mealy.rs`

**Before: 77 mutants — 6 missed, 55 caught, 15 unviable, 1 timeout.**

1. **`Node::poll`'s `slot_progress += 1` on the Management and Data lane
   arms** (mealy.rs:538, :554; `+=` → `*=`) — missed on both. The identical
   mutation on the Control lane arm (line 522) was already caught by
   `control_backlog_does_not_starve_data_forever`, but nothing exercised the
   same starvation property for the other two lanes. Since `slot_progress`
   starts at 0, `*= 1` leaves it at 0 forever, so `advance_if_exhausted`
   never fires and a backlogged Management or Data lane would monopolise the
   node past its configured burst limit — the exact bug class the 2026-07-28
   pass found in `Host::sweep`'s `i += 1` → `i *= 1`. Added
   `management_backlog_does_not_starve_data_forever` and
   `data_backlog_does_not_starve_control_forever`, mirroring the existing
   control-lane test's shape (small burst limits, an oversized backlog on
   one lane, a competing message on another, asserting the exact `seen`
   sequence across several `poll()` calls).

**After: 77 mutants — 4 missed, 57 caught, 15 unviable, 1 timeout.**

The remaining 4 are equivalent mutants, not coverage gaps, traced by hand
rather than by adding a test:

- **`Transducer::step_control`/`step_management`'s default bodies**
  (mealy.rs:99, :104) — both already return `Ok(Self::Out::default())`
  verbatim; the mutant rewrites it to `Ok(Default::default())`, which
  resolves to the identical value under the same type inference. Same
  category as the `Host::is_empty()`-adjacent equivalent mutants noted in
  earlier passes.
- **`Topology::find_cycle`'s match guard at line 982**
  (`mark[to] == Mark::Unseen` → `true`, and → `== ` replaced with `!=`) —
  traced by hand rather than guessed at. `find_cycle`'s only externally
  observable signal is whether *any* iteration hits the sibling arm
  (`mark[to] == Mark::OnPath`, at line 977, unmutated in both surviving
  mutants) — that arm alone determines `Some(cycle)` vs `None`, and it
  always sees the true, current OnPath status of whatever node it's
  checking, regardless of how the guard at 982 chooses to expand the
  frontier. Because the traversal is a single explicit stack (iterative DFS,
  not concurrent), a node can only be "reactivated" via the mutated guard
  after its entire subtree from the first visit has already unwound back to
  `Done` — so reactivating it just repeats already-finished, side-effect-free
  work before landing back on `Done` again; it can never make the sibling
  arm see a stale `OnPath` it shouldn't, in either direction. Confirmed by
  hand-tracing both the accepted `a_diamond_is_acyclic` fixture (extra
  revisits of the shared `sink` node, but `find_cycle` still returns `None`)
  and a two-cycle case (the reactivation coincidentally *helps* rediscover
  the real cycle via the unmutated arm). `Topology::validate`'s public
  contract — `Ok`/`Err` and which actors are named — cannot distinguish
  these mutants from correct code short of asserting on a specific DFS
  visitation order the doc comment never promises, or authoring a
  worst-case exponential-blowup graph purely to force a timeout (the same
  category the next bullet already covers, and the 2026-07-28 pass declined
  for the same reason: shipping a test whose only failure mode is hanging is
  worse than the gap it closes).
- **The same function's `*next += 1` → `*next *= 1`** (mealy.rs:974) —
  already reported as `TIMEOUT`, not `MISSED`: without the index advancing,
  `nth(*next)` re-fetches the same successor forever, so any test that
  exercised the buggy branch would have to hang to observe it.
  `cargo-mutants`' 20s timeout is exactly the correct signal here (real code
  running this mutant would also hang) — same judgment call the 2026-07-28
  pass made for `Host::sweep`'s analogous timeout-class mutants.

`cargo test -p actor-scheduler --lib mealy::`: 40/40 (was 38/38 before the 2
new tests). `cargo clippy -p actor-scheduler --lib --tests`: clean.

### `pixelflow-runtime/src/coordinator_node.rs`

**Before: 8 mutants — 2 missed, 5 caught, 1 unviable.**

1. **`CoordinatorCore::present_cooked_frame`'s `self.frame_number += 1`**
   (`+=` → `*=`) — missed. The existing test asserted only
   `out.rendered.is_some()`, never the actual `frame_number` value, so a
   mutant that leaves the counter stuck at its zero default (frame_number
   starts at 0, so `0 *= 1` stays 0 forever) passed unnoticed — meaning the
   FPS telemetry sent to vsync could silently report frame 0 forever.
   Tightened the existing test to assert `rendered.frame_number == 1`, and
   added `the_frame_counter_advances_once_per_presented_frame`, which drives
   3 full submit→grant→complete cycles and asserts the counter is exactly
   1, 2, 3.
2. **`CoordinatorData`'s hand-written `Debug` impl** — missed; nothing ever
   formatted a `CoordinatorData` value. Added
   `coordinator_data_debug_names_its_variant`, checking all three variants.

**After: 8 mutants — 0 missed, 7 caught, 1 unviable.**

`cargo test -p pixelflow-runtime --lib coordinator_node::`: 13/13 (was
11/11). `cargo clippy -p pixelflow-runtime --lib --tests`: clean.

### `pixelflow-runtime/src/engine_troupe.rs`

**Before: 19 mutants — 4 missed, 2 caught, 12 unviable, 1 timeout.**

1. **`EngineHandler::send_vsync_control`** and **`EngineHandler::shut_down`**
   whole-body-replaced-with-`()` — both missed. The `Rig` test fixture (the
   file's own doc comment: "no green host, thread, or rasterizer in the
   loop") always left `vsync_control`, `vsync_host`, `self_handle`, and
   `rasterizer_forwarder` at `None`, so both methods' bodies were
   unconditionally no-ops in every existing test regardless of whether the
   real logic ran. Added `skipped_frame_returns_the_vsync_token` (wires a
   real `vsync_control` `GreenSender` and checks a `Skipped` frame's
   `ReturnToken` actually arrives) and
   `quit_shuts_down_the_driver_and_every_configured_handle` (wires real
   `ActorScheduler`s for the driver plus every optional handle, sends
   `EngineControl::Quit`, and polls each scheduler via `poll_once` for the
   `Shutdown` message, using small no-op `Actor` stand-ins that exist only
   to observe delivery).
2. **`RasterizerForwarder::shut_down`** (line 228) — reported as `TIMEOUT`,
   not `MISSED`: `forwarder_relays_responses_and_exits_when_the_rasterizer_disconnects`
   already spawns a real scheduler thread and `.join()`s it, which hangs
   under this mutation rather than failing an assertion. Same accepted
   timeout-class signal as `mealy.rs`'s line 974 above; left as-is.
3. **`Troupe::with_config`'s `1.0 / refresh_rate`** (`/` → `%`, `/` → `*`) —
   both missed, and the one gap left open rather than fixed. No test calls
   `with_config` at all: it is the top-level bootstrap that spawns a real
   `Timer` clock thread and a real green-host OS thread, and requires a live
   platform waker (`CocoaWaker`/`X11Waker`) that only exists with an actual
   display connection — not reachable from a unit test in this or any prior
   pass's environment. This is the same class of gap the reviewer
   checklist's "platform-specific code (may break on untested platforms)"
   already calls out, not a cheaply-closable coverage hole; documenting here
   rather than inventing an integration harness this pass wasn't scoped for.

**After: 19 mutants — 2 missed (both `Troupe::with_config`, documented
above), 4 caught, 12 unviable, 1 timeout.**

`cargo test -p pixelflow-runtime --lib engine_troupe::`: 10/10 (was 8/8).
`cargo clippy -p pixelflow-runtime --lib --tests`: clean.

## Verified

- `cargo test --workspace --lib`: all 12 crates pass (112/112 in
  `pixelflow-search` alone, the largest suite touched by this pass's diff;
  every other crate's suite unaffected by these changes), 0 failures, 1
  pre-existing ignored test.
- `cargo clippy --workspace --lib --tests`: clean.
- `cargo mutants -p actor-scheduler --file actor-scheduler/src/mealy.rs`:
  4 missed (all documented equivalent/timeout mutants above), re-run after
  the fix.
- `cargo mutants -p pixelflow-runtime --file
  pixelflow-runtime/src/coordinator_node.rs`: 0 missed, re-run after the fix.
- `cargo mutants -p pixelflow-runtime --file
  pixelflow-runtime/src/engine_troupe.rs`: 2 missed (both documented
  `Troupe::with_config` bootstrap-code exceptions above), re-run after the
  fix.

## Recommended next steps (not done here)

1. `spatial_bsp.rs`'s private `interiors[...]` indexing (open since
   2026-07-20) still needs a human design call — unchanged this pass.
2. `core-term/src/terminal_app.rs`'s `create_test_app()` calling
   `new_registered` instead of `spawn_terminal_app()` (open since
   2026-07-24) — still judged an intentional seam, not re-litigated here.
3. All three surfaces 2026-08-01 flagged as never-mutation-tested
   (`mealy.rs`, `coordinator_node.rs`, `engine_troupe.rs`) are now
   mutation-clean modulo the documented equivalent/timeout/
   integration-bootstrap exceptions above — no more open mutation-testing
   debt from that pass.
4. The new `pixelflow-codegen` crate (split out of `pixelflow-ir` this pass)
   has never been mutation-tested under its new name/location. Its
   module-private-encoder test pattern (noted above, not flagged as a
   violation) makes it a different shape of target than the actor/green-node
   code this pass covered — worth a dedicated pass rather than folding into
   the next scheduled one, since a first look should also judge whether that
   test pattern is the right long-term shape or a `#[cfg(test)] pub(crate)`
   seam worth introducing.
5. If `Troupe::with_config` ever grows a testable pure-computation seam
   (e.g. a small `fn tick_interval(target_fps: u32) -> Duration` extracted
   from the bootstrap), the two remaining `engine_troupe.rs` mutants become
   closable without an integration harness — not attempted here since it
   would be a production-code change beyond this pass's remit.
6. **The audit window needs to key off merged history, not the last audit
   doc.** This pass and #985 both computed their window as "since `eb03ee9`"
   — the 2026-08-01 pass's commit — because that is what the previous doc
   named. Two passes therefore swept the same delta and independently found
   and fixed the same `corpus.rs` violation, and the only reason it surfaced
   as a conflict rather than a silent double-fix is that they touched the
   same lines. A pass should derive its base from what is actually on `main`
   at the moment it starts (and skip work already landed by a
   later-numbered doc), or successive passes will keep re-auditing the same
   window whenever one is still in review when the next fires. The failure
   is cheap here; it would not be if two passes "fixed" the same test in
   incompatible ways.
