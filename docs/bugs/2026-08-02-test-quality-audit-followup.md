# Test quality control follow-up — 2026-08-02

Scope: scheduled continuation of
`docs/bugs/2026-08-01-test-quality-audit-followup.md`. No commits landed
between that pass (`eb03ee9`) and this one, so there was nothing new to
static-audit against `docs/STYLE.md`'s "Test Public API" rule. Instead this
pass picked up the mutation-testing backlog that pass's "Recommended next
steps" left open, in the order given: `pixelflow-runtime/src/coordinator_node.rs`
(871 lines) and `pixelflow-runtime/src/engine_troupe.rs` (984 lines,
rewritten), both never mutation-tested before. `actor-scheduler/src/mealy.rs`'s
969-line `DedicatedThread`/`step_os` rewrite — third on that list — is still
open for a future pass.

Installed `cargo-mutants` fresh (v27.1.0, not present in this environment,
consistent with every prior pass).

## Mutation testing: `pixelflow-runtime/src/coordinator_node.rs`

**Before: 8 mutants — 2 missed, 5 caught, 1 unviable.**

1. **`<impl Debug for CoordinatorData>::fmt` replaced wholesale
   (`Ok(Default::default())`)** — nothing ever called `format!("{:?}", ...)`
   on the type; the impl exists for `log::debug!` call sites nobody's test
   exercises. Added `debug_format_reflects_the_variant`, checking the
   `Advance` and `Submit` arms' literal output.
2. **`present_cooked_frame`'s `frame_number += 1` → `*=`** — every existing
   test that reaches `step_management`'s presented arm only checks
   `out.rendered.is_some()`, never the counter's actual value. Since
   `frame_number` starts at 0, `+= 1` and `*= 1` are indistinguishable after
   any number of frames *except* by checking the value is `1` and not `0`
   after the first one — `*=` leaves it at `0` forever, `-=` would underflow
   a `u64` and panic (already caught). Added
   `the_first_presented_frame_is_numbered_one_not_zero`.

`cargo test -p pixelflow-runtime --lib coordinator_node::`: 13/13 (was 11/11).

**After: 8 mutants — 7 caught, 1 unviable. 0 missed.**

## Mutation testing: `pixelflow-runtime/src/engine_troupe.rs`

**Before: 19 mutants — 4 missed, 2 caught, 12 unviable, 1 timeout.**

The 12 unviable are mostly `Default::default()` substitutions on functions
returning types with no sensible default (handles, `RuntimeError`) — not real
gaps. The timeout (`RasterizerForwarder::shut_down` → `()`) is a genuine
catch reported oddly: without the self-shutdown send, the forwarder's
scheduler thread parks on its doorbell forever once the rasterizer
disconnects, so `forwarder_relays_responses_and_exits_when_the_rasterizer_disconnects`'s
`thread.join()` hangs rather than failing outright — the same "timeout-class
mutant" `cargo-mutants` produced against actor-scheduler's sweep loops in the
2026-07-28 pass. A hang is the correct signal for a wedged shutdown path (CI
would time out on it too), so left as-is.

Of the 4 missed, 2 were real coverage gaps:

1. **`EngineHandler::send_vsync_control`, wholesale** — no test ever wired an
   `EngineHandler` with `vsync_control: Some(...)`, so the relay
   (`AppData::RenderSurface`/`Skipped` → `VsyncCommand::ReturnToken`) had
   zero coverage; the two existing tests that trigger it
   (`render_surface_relays_a_submit_to_the_coordinator`,
   `skipped_frame_does_not_touch_the_coordinator`) only ever checked the
   coordinator port. Added `render_surface_returns_the_vsync_token`, a
   standalone fixture (not the shared `Rig`, to avoid changing what every
   other `Rig`-based test observes) wiring a real `vsync_control` channel and
   asserting the `ReturnToken` lands, driven through the public
   `Actor::handle_data` entry point exactly as the existing coordinator-port
   tests are.
2. **`EngineHandler::shut_down`, wholesale** — no test ever drove
   `EngineControl::Quit` through `EngineHandler` at all (only
   `engine_core.rs`'s pure-core `quit_control_sets_quit` covered the `Quit →
   out.quit = true` half). Added `quit_sends_shutdown_to_the_driver`: a
   second standalone fixture keeping a real `ActorScheduler` alive (the
   shared `Rig` discards its driver scheduler immediately, since no test
   needed to observe driver messages before), driving `Quit` through the
   public `Actor::handle_control` entry point and asserting
   `ActorScheduler::poll_once` reports `Message::Shutdown` received — the
   driver is the one shutdown-cascade handle never gated behind an `Option`,
   so it is the cascade's simplest observable proof of life.

The other 2 missed (`553:57`, `1.0 / refresh_rate` → `%`/`*` in
`Troupe::with_config`) were judged an acceptable exception, documented rather
than changed: `with_config` is the real bootstrap — it spawns the platform
driver, the green-host thread, and the vsync clock, none of which a unit test
should stand up just to pin one arithmetic op. Extracting a pure
`fn tick_interval(fps: f64) -> Duration` helper would make it testable, but
that is new `pub(crate)` surface on a function that does not otherwise need
one, against CLAUDE.md's "do not change visibility of internal APIs without
explicit permission" — a call outside this pass's authority. Added a comment
at the call site documenting the exception, same judgment call as
`pixelflow-graphics/src/render/scene.rs`'s `chunked_bake_matches_whole_stripe`
(2026-08-01 pass).

`cargo test -p pixelflow-runtime --lib engine_troupe::`: 10/10 (was 8/8).

**After: 19 mutants — 4 caught, 12 unviable, 1 timeout. 0 missed (2 accepted exceptions).**

## Verified

- `cargo test --workspace --lib`: all 11 crates pass, 850 tests, 0 failures
  (1 ignored, pre-existing).
- `cargo clippy -p pixelflow-runtime --lib --tests`: clean.
- `cargo fmt -p pixelflow-runtime --check`: clean.
- `cargo mutants -p pixelflow-runtime --file pixelflow-runtime/src/coordinator_node.rs`:
  0 missed (re-run after the fix, confirmed above).
- `cargo mutants -p pixelflow-runtime --file pixelflow-runtime/src/engine_troupe.rs`:
  0 missed, 2 accepted exceptions (re-run after the fix, confirmed above).

## Recommended next steps (not done here)

1. `spatial_bsp.rs`'s private `interiors[...]` indexing (open since
   2026-07-20) still needs a human design call — unchanged this pass.
2. `core-term/src/terminal_app.rs`'s `create_test_app()` calling
   `new_registered` instead of `spawn_terminal_app()` (open since
   2026-07-24) — still judged an intentional seam, not re-litigated here.
3. `actor-scheduler/src/mealy.rs`'s 969-line `DedicatedThread`/`step_os`
   rewrite has never been mutation-tested — the last item on the
   2026-08-01 pass's list, now the only one left. Likely to produce more
   timeout-class mutants given its sweep-loop shape (per the 2026-07-28
   pass's experience with similar actor-scheduler code); budget accordingly.
