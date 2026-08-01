# Test quality control follow-up — 2026-08-01

Scope: scheduled continuation of
`docs/bugs/2026-07-28-test-quality-audit-followup.md`. Since that pass
(commit `d7d33b8`), the tree picked up 33 commits and ~8,500 changed lines
across actor-scheduler, core-term, and every `pixelflow-*` crate — the
largest delta between audit passes to date, including a green-actor
send-policy refactor (`actor-scheduler`'s `try_send` demoted to
`pub(crate)` in favor of `send`/`send_port`), a brand-new JIT cell-grid
lattice (`pixelflow-core/src/lattice/cell_grid.rs`) backing core-term's new
cell-grid terminal scene, and a rewritten `pixelflow-runtime` engine/vsync
troupe (`coordinator_node.rs`, `engine_troupe.rs`).

## Static audit: two `Explore` sub-agents swept `git diff d7d33b8..HEAD`

Split by area — actor-scheduler/core-term, and the `pixelflow-*` crates —
against docs/STYLE.md's "Test Public API" rule, cross-checked against every
prior audit's already-accepted exceptions (`pixelflow-graphics/src/spatial_bsp.rs`'s
private `interiors[...]` indexing; core-term's `create_test_app()` using
`new_registered`) so those weren't re-flagged.

### Fixed this pass

1. **`actor-scheduler/src/mealy.rs`** — the `Node` test module read the
   private `continuation` field directly at ten call sites instead of
   relying on `poll()`/`poll_os()`/`actor()`'s already-public signal. Every
   one of these was redundant with an adjacent or easily-added
   public-surface assertion:
   - `step_os_continuation_lands_in_the_slot`: the field check was fully
     subsumed by the following `poll()` call already asserting
     `actor().last_resumed == Some(42)`.
   - `an_endless_self_yielder_does_not_starve_step_os`: each round's
     `continuation.is_some()` check is exactly what the *next* round's
     `poll() == Step::Ran` already proves (the lane was fed only one
     message up front, so a lost continuation would starve it). Added one
     more `poll()` after the loop to cover round 3, which previously had no
     following round to prove its own survival.
   - `a_pending_continuation_survives_step_os`: both intermediate field
     reads were subsumed by the test's own final
     `actor().last_resumed == Some(2)` check.
   - `the_continuation_slot_holds_at_most_one_message`: the loop body's
     assertion was a literal tautology
     (`is_none() || is_some()`, always true for any `Option`) — dead weight
     regardless of privacy. Replaced with capturing the loop's final `Step`
     and asserting it's `Step::Idle`, which is the actual public-observable
     form of "the slot is empty once the machine finishes."
   - `a_continuation_resumes_before_new_inbox_work`: the two intermediate
     `continuation == Some(n)` checks were redundant with the final
     `actor().finished == Some(0)` after exactly three `poll()` calls — a
     scheduler that touched the queued message instead of the pending
     continuation could not reach that state in three steps.

   `cargo test -p actor-scheduler --lib mealy::`: 36/36 after the fix.

2. **`actor-scheduler/src/host.rs`** — this pass's refactor demoted
   `GreenSender::try_send` from `pub` to `pub(crate)` ("the scheduler owns
   send policy... a handler pushing into another green actor's inbox from
   outside a flush should use `Self::send` instead"), adding public `send`
   as the replacement. Two tests
   (`a_green_send_wakes_a_wired_host_too`,
   `a_green_send_wakes_a_host_asleep_on_its_doorbell`) still called the
   now-private `try_send` for what is, in both cases, a single message into
   an otherwise-empty channel — `send`'s bounded backoff behaves
   identically there. Swapped both to `.send(41)`.
   `a_failed_send_reports_backpressure` and
   `many_green_sends_coalesce_into_one_pending_wake` genuinely need the
   non-blocking `Full` signal `send` doesn't expose; left as-is (same
   Flexibility-clause carve-out as the actor-scheduler backoff internals in
   2026-07-24's pass).

   `cargo test -p actor-scheduler --lib host::`: 15/15 after the fix.

3. **`actor-scheduler/src/lib.rs`** — the same refactor demoted
   `ActorHandle::try_send` to `pub(crate)`. Three tests in
   `try_send_tests` (`try_send_succeeds_and_is_received`,
   `try_send_on_a_full_data_ring_returns_full_with_the_message_recoverable`,
   `try_send_after_receiver_drop_returns_disconnected`) still called it
   directly. Unlike `GreenSender`, there's an exact public replacement that
   exercises the same code: `mealy::send_port(&mut Option<T>, &target)`,
   returning `Flush::{Done,Blocked,Disconnected}`. Rewrote all three
   through it (payload recovery on `Blocked`/`Disconnected` checked via
   `match` — `Message` only derives `Debug`, not `PartialEq`).

   `cargo test -p actor-scheduler --lib try_send_tests::`: 3/3 after the
   fix; `cargo clippy -p actor-scheduler --lib --tests`: clean.

4. **`core-term/src/terminal_app.rs`** —
   `scene_paints_default_background_and_recompiles_on_resize` called the
   private `app.build_scene()` directly and read the private `program`
   field's geometry before/after a resize to prove the scene program
   recompiles. Rewrote to drive the actor's real entry point
   (`app.handle_data(TerminalData::Engine(EngineEventData::RequestFrame {
   .. }))`) and observe the rendered output on a new `EngineProbe` test
   double (mirroring the file's existing `WriterProbe`/`drain_writer`
   pattern, and `pixelflow-runtime`'s own `testing::mock_engine`), draining
   the engine scheduler's `poll_once`. The resize-recompiles claim no longer
   needs to peek at `program`'s geometry at all: `CellGridProgram::frame`
   asserts the cell buffer's length against its compiled geometry, and the
   cell buffer is always rebuilt fresh from the *current* terminal
   snapshot's dimensions — so a stale, un-recompiled program would panic on
   the very next frame after a resize. Rendering successfully post-resize
   is therefore itself the proof; kept the background-color pixel check
   before and after for good measure.

   `cargo test -p core-term --lib terminal_app::`: 4/4 after the fix;
   `cargo clippy -p core-term --lib --tests`: clean.

### Judged an acceptable exception, documented rather than changed

5. **`pixelflow-graphics/src/render/scene.rs`** —
   `chunked_bake_matches_whole_stripe` calls the private free function
   `bake_and_pack_chunked` directly with an explicit `chunk_rows` no public
   API exposes. Unlike the `#[cfg(test)] pub(crate) fn holds_buffer()`
   test-only-window pattern already established elsewhere in this same
   diff (`coordinator_node.rs`, `render_coordinator.rs`), this function
   isn't a test seam — it's shared production logic
   (`bake_and_pack_stripe` calls it on the real render path too), just with
   a parameter (`chunk_rows`) that production always derives from
   `PLANE_SCRATCH_BYTES`, which only forces a chunk boundary on frames far
   larger than a unit test should render. `Scene::render`'s public surface
   (`frame`, `num_threads`) has no way to reach that boundary
   deterministically. Rather than invent a `pub(crate)` parameter on a
   production function against CLAUDE.md's "do not change visibility of
   internal APIs without explicit permission," added a comment on the test
   documenting the Flexibility-clause exception — same judgment call the
   2026-07-24 pass made for actor-scheduler's timing-internal backoff
   arithmetic.

Everything else in the diff — `dedicated_thread.rs`'s 598 new lines,
`coordinator_node.rs`'s 871 new lines (`Transducer`-based, mirroring the
already-accepted `VsyncCore`/`RasterCore` pattern), `engine_troupe.rs`'s
rewrite, `fonts/atlas.rs`'s new tests, `tests/common/mod.rs`'s golden-image
harness, `pixelflow-ir/tests/collapse_loop.rs` — drives its crate's genuine
public API. No `raw_mul`/`raw_select`/`SimdVec`/lane access anywhere in the
reviewed diff.

## Mutation testing: `pixelflow-core/src/lattice/cell_grid.rs`

Brand new this pass (637 lines, no prior audit had touched it) and the
JIT-compiled backbone of core-term's new cell-grid terminal scene —
self-contained, deterministic math with no actor/timing surface, so a good
candidate to mutation-test cleanly (no timeout-class mutants the way
actor-scheduler's sweep loops produced in 2026-07-28). Installed
`cargo-mutants` fresh (v27.1.0, not present in this environment, consistent
with every prior pass).

**Before: 47 mutants — 9 missed, 34 caught, 4 unviable.**

All nine were real coverage gaps, not equivalent mutants:

1. **`CellGridGeometry::cells_len` (`*` → `/`)** — never called directly by
   any test, only indirectly through a mismatch assertion that doesn't
   care about the exact value. Added
   `cells_len_and_atlas_len_match_declared_geometry` asserting both
   accessors against hand-computed values.
2. **`channel_arena`'s `bg_idx` offset (`6 + channel` → `6 - channel`)** —
   every existing test only checked channels 0 (R) and 2 (B), where the
   wrong offset happened to land on a field with a coincidentally
   identical value. Added
   `green_channel_reads_its_own_bg_field_not_a_neighboring_one`, which
   checks channel 1 (G) specifically because its wrong-offset field (fg_a)
   and correct field (bg_g) differ in `tiny_scene`'s fixture.
3. **The vertical apron clamp (`tile_h as f32 + 0.5` → `-`/`*`)** — every
   existing "cell bigger than its tile" test only varied the cell
   horizontally; nothing exercised the row-direction apron clamp. Added
   `cells_taller_than_their_tile_fade_to_background`, the vertical mirror
   of the existing `cells_wider_than_their_tile_fade_to_background`.
4. **`grid_w`/`grid_h`'s boundary math (`cols * cell_w` → `cols + cell_w`,
   same for rows)** — the existing "outside the grid" test only sampled
   points that were outside both the correct AND the wrong (`+`-computed)
   boundary, so it couldn't tell them apart. Added
   `grid_extent_is_cell_count_times_cell_size_not_their_sum`, sampling a
   point inside the correct boundary but outside the smaller wrong one.
5. **`CellGridFrame::padded_width`'s lane count (`/` → `*`)** — nothing
   called `padded_width` directly and asserted its value; downstream code
   recomputes the same lane count independently, so an inflated stride was
   silently absorbed as wasted memory rather than a wrong answer. Added
   `padded_width_rounds_up_to_whole_simd_batches`, asserting exact values
   at and around a lane-count boundary.
6. **`bake_channel_rows`'s pixel-center offset (`y0 as f32 + 0.5` → `-`/`*`)**
   — every existing test baked from `y0 = 0`, and this class of mutation
   shifts every row's absolute Y by the same constant, so a
   self-consistency check (bake offset region, compare against a full bake
   sliced at the same relative rows) can't see it — the shift cancels in
   the comparison. First attempt at
   `baking_a_row_offset_region_matches_the_same_rows_from_a_full_bake`
   confirmed this empirically (it caught the `*` mutant but missed the `-`
   one). Replaced it with
   `baking_a_row_offset_region_samples_the_correct_absolute_row`, which
   asserts a concrete expected pixel value tied to a specific `y0 > 0`
   instead of comparing two bakes to each other.

**After: 47 mutants — 0 missed, 43 caught, 4 unviable.**

`cargo test -p pixelflow-core --lib lattice::cell_grid::`: 13/13 (was 7/7
before this pass's additions). `cargo clippy -p pixelflow-core --lib
--tests`: clean (one `identity_op` lint from an early draft of the vertical
apron test, fixed before landing).

## Verified

- `cargo test --workspace --lib`: all 11 crates pass, 0 failures.
- `cargo clippy --workspace --lib --tests`: clean.
- `cargo fmt --check` on every touched crate: clean.
- `cargo mutants -p pixelflow-core --file pixelflow-core/src/lattice/cell_grid.rs`:
  0 missed (re-run after the fix, confirmed above).

## Recommended next steps (not done here)

1. `spatial_bsp.rs`'s private `interiors[...]` indexing (open since
   2026-07-20) still needs a human design call — unchanged this pass.
2. `core-term/src/terminal_app.rs`'s `create_test_app()` calling
   `new_registered` instead of `spawn_terminal_app()` (open since
   2026-07-24) — still judged an intentional seam, not re-litigated here.
3. This pass only mutation-tested `cell_grid.rs`. The other large new
   surfaces from this delta —
   `pixelflow-runtime/src/coordinator_node.rs` (871 lines),
   `engine_troupe.rs` (rewritten), and `actor-scheduler/src/mealy.rs`'s own
   969-line rewrite around the `DedicatedThread`/`step_os` primitive — have
   never been mutation-tested and are the highest-value next targets,
   in roughly that order (coordinator_node.rs is the newest and least
   exercised by any prior pass).
