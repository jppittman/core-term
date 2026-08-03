# Test quality control follow-up — 2026-08-03

Scope: scheduled continuation of
`docs/bugs/2026-08-01-test-quality-audit-followup.md`. `git diff
eb03ee9..HEAD` (that pass's merge commit) is empty — `main` and this
branch both start at exactly that commit, so no new work has landed to
statically audit against `docs/STYLE.md`'s "Test Public API" rule since
the last pass. This pass instead goes straight to that doc's "Recommended
next steps": mutation-testing the two largest untested surfaces from the
2026-07-20 engine-mesh migration, in the priority order it gave
(`coordinator_node.rs`, then `engine_troupe.rs`).

Installed `cargo-mutants` fresh (v27.1.0, as every prior pass — not
present in this environment).

## Mutation testing: `pixelflow-runtime/src/coordinator_node.rs`

871 lines, the newest and least-exercised surface per the last pass's own
assessment. Small mutation surface for its size (8 mutants — most of the
file is enum/struct declarations, trait glue, and doc comments rather than
branching logic), but 2 of the 8 were real gaps.

**Before: 8 mutants — 2 missed, 5 caught, 1 unviable.**

1. **`CoordinatorCore::present_cooked_frame`'s `frame_number += 1` → `*=`
   1** — every existing test called `present_cooked_frame` (via
   `step_management`) at most once per `CoordinatorCore`, starting from
   `frame_number: 0`, and only asserted `out.rendered.is_some()`. `0 *= 1`
   is `0`, same as `0 += 1` would need one more step to distinguish, and
   nothing checked the actual counter value at all. Added
   `frame_number_increments_by_one_per_presented_frame`, driving two full
   submit→grant→complete cycles and asserting the telemetry frame number
   is `1` then `2`.
2. **`CoordinatorData`'s manual `Debug` impl → `Ok(Default::default())`
   (a no-op)** — this impl exists only to satisfy `Message<D, C,
   M>`'s `#[derive(Debug)]` bound (`Scene` itself isn't `Debug`), and
   nothing in the reviewed code path ever formats a `CoordinatorData`, so
   a silently blank impl was indistinguishable from the real one to every
   existing test. Added `coordinator_data_debug_names_each_variant`,
   asserting each variant's formatted text directly.

**After: 8 mutants — 0 missed, 7 caught, 1 unviable.**

`cargo test -p pixelflow-runtime --lib coordinator_node::`: 13/13 (was
11/11). `cargo clippy -p pixelflow-runtime --lib --tests`: clean. `cargo
fmt --check`: clean.

## Mutation testing: `pixelflow-runtime/src/engine_troupe.rs`

984 lines, the second-priority target from the last pass. 19 mutants: 4
missed, 2 caught, 12 unviable, 1 timeout.

**Fixed — real coverage gaps:**

1. **`EngineHandler::send_vsync_control` → `()`** — the file's shared test
   fixture (`Rig`) leaves `vsync_control: None`, so every existing call to
   `send_vsync_control` (via `flush`, e.g. from `RenderSurface`) took the
   same early-return-on-`None` path a no-op replacement would also take —
   indistinguishable regardless of what the function's body actually does.
   Added `render_surface_returns_a_vsync_token`, a standalone fixture (not
   `Rig`, which nothing else needs `vsync_control` wired for) with a real
   channel, asserting `RenderSurface` actually returns a token.
2. **`EngineHandler::shut_down` → `()`** — no test in the file fed
   `EngineControl::Quit` (or any quit path) at all; `engine_core.rs`'s own
   tests cover that `EngineCore` *decides* to quit, but nothing exercised
   the shell's cascade of real sends once it does. Added
   `quit_cascades_a_shutdown_to_the_driver`, asserting the one cascade
   step that's unconditional regardless of which other handles are wired
   (every `Rig`-style fixture leaves `vsync_host`/`self_handle`/
   `rasterizer_forwarder` as `None`): the driver always gets
   `Message::Shutdown` on `Quit`.

**Left as documented exceptions — not coverage gaps:**

3. **`RasterizerForwarder::shut_down` → `()`** reported as a `TIMEOUT`
   (14s build + 30s test), not `MISSED`.
   `forwarder_relays_responses_and_exits_when_the_rasterizer_disconnects`
   already depends on this running for real: it `thread.join()`s the
   forwarder after dropping the rasterizer's sender, and a no-op
   `shut_down` would leave that thread parked on its channel forever
   instead of exiting — the mutant *is* detected, cargo-mutants just can't
   tell a correctly-triggered hang from a genuinely missed mutant and
   reports both as needing attention. Same class of result the 2026-07-28
   pass documented for actor-scheduler's own sweep loops.
4. **`Troupe::with_config`'s `1.0 / refresh_rate` → `%`/`*`** (tick-interval
   arithmetic, line 553) — `with_config` spawns real OS threads, a
   green-host scheduler, and (indirectly, via the driver) the platform
   display backend; it is explicitly outside this file's own stated test
   scope (`mod tests`'s doc: "`EngineHandler`, driven directly... not
   about full runtime bootstrap"). The formula itself is an unambiguous
   one-liner (`seconds_per_frame = 1 / fps`) sitting inside otherwise-
   untested bootstrap glue — the same class of exception the 2026-08-01
   pass documented for `pixelflow-graphics/src/render/scene.rs`'s
   `bake_and_pack_chunked`, and for the same reason: the alternative is
   restructuring production wiring code to manufacture a test seam for one
   line no plausible mutation would actually miscompile silently.

**Also found and fixed — a flaky test, not a mutation gap:**

5. While re-running `cargo mutants -p pixelflow-runtime --file
   pixelflow-runtime/src/engine_troupe.rs -j4` to confirm the two fixes
   above, the *unmutated baseline* itself failed once:
   `forwarder_relays_responses_and_exits_when_the_rasterizer_disconnects`
   polled its response channel for a fixed `0..10_000`
   `std::thread::yield_now()` iterations before giving up — a budget of
   scheduler turns, not wall-clock time. Under `-j4`'s parallel builds
   oversaturating this environment's 4 cores, the forwarder thread
   apparently didn't get scheduled within that many turns before the poll
   gave up and the assertion failed. Not reproducible with a simple 4-way
   busy-loop stress harness, but the mechanism is fragile by construction
   independent of how easy it is to trigger: an iteration count is not a
   time bound under variable scheduling load. Replaced it with a 5-second
   wall-clock deadline (`Instant::elapsed()` against a `Duration`),
   keeping the same yield-don't-sleep polling style so the common case is
   unaffected. Re-ran the full mutation suite twice after this fix: clean
   baseline both times.

**After: 19 mutants — 2 missed (both documented exceptions above), 4
caught, 12 unviable, 1 timeout (documented exception above).**

`cargo test -p pixelflow-runtime --lib engine_troupe::`: 10/10 (was
8/8). `cargo clippy -p pixelflow-runtime --lib --tests`: clean. `cargo
fmt --check`: clean.

## Verified

- `cargo test --workspace --lib`: all 11 crates pass, 137+0+434+48+114+148+131+9+28+65+109 = 1223 tests, 0 failures, 1 ignored (pre-existing, `pixelflow-search`'s `provenance_overhead_timing`).
- `cargo clippy --workspace --lib --tests`: clean.
- `cargo fmt --check` on every touched crate: clean.
- `cargo mutants -p pixelflow-runtime --file pixelflow-runtime/src/coordinator_node.rs`: 0 missed (re-confirmed).
- `cargo mutants -p pixelflow-runtime --file pixelflow-runtime/src/engine_troupe.rs`: 0 missed beyond the two documented exceptions (re-confirmed twice, including once after the flaky-test fix).

## Recommended next steps (not done here)

1. `spatial_bsp.rs`'s private `interiors[...]` indexing (open since
   2026-07-20) still needs a human design call — unchanged this pass.
2. `core-term/src/terminal_app.rs`'s `create_test_app()` calling
   `new_registered` instead of `spawn_terminal_app()` (open since
   2026-07-24) — still judged an intentional seam, not re-litigated here.
3. `actor-scheduler/src/mealy.rs`'s 969-line `DedicatedThread`/`step_os`
   rewrite (flagged by the 2026-08-01 pass, third priority after
   `coordinator_node.rs` and `engine_troupe.rs`, both now done) has never
   been mutation-tested and is the next highest-value target.
4. This pass's iteration-count-vs-wall-clock-deadline flakiness class
   (item 5 above) is worth a quick grep across the rest of the workspace
   for the same `for _ in 0..N { ...; yield_now() }` shape — this was the
   first time it surfaced, found incidentally rather than through a
   deliberate sweep.
