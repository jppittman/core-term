# Test quality control follow-up — 2026-07-29

Scope: scheduled continuation of `docs/bugs/2026-07-28-test-quality-audit-followup.md`.
28 commits landed since that pass (`d7d33b8..HEAD`), including a new `pixelflow-core`
tile-grid sampler (`lattice/cell_grid.rs`), a new `pixelflow-runtime` actor
(`coordinator_node.rs`, step 5c of the engine-mesh migration), an `actor-scheduler`
backpressure refactor (`GreenSender::send`), and unrelated JIT/e-graph work
(`pixelflow-ir` collapse loop, `pixelflow-search`).

## Static audit: no new violations

An `Explore` sub-agent swept all 19 test files that changed in this diff range against
docs/STYLE.md's "Test Public API" rule and the Comments section's ban on historical/change-
summary narration. Result: every change in that set is either a pure `rustfmt` reflow, an
import reorder, or a new test built exclusively from its crate's public surface
(`NixPty::spawn_with_config`, `TerminalEmulator::interpret_input`, `compile_collapse`/
`ExprArena`, etc.). No violations found; nothing to fix here this pass.

`spatial_bsp.rs`'s private `interiors[...]` indexing (open since 2026-07-20) and
`terminal_app.rs`'s `create_test_app()` seam (open since 2026-07-24) are unchanged and still
open — both still judged design calls, not attempted here.

## Mutation testing

Installed `cargo-mutants` fresh (v27.1.0 — not present in this environment, consistent with
every prior pass). Picked the three pieces of genuinely new logic in this diff range that no
prior pass had mutation-tested, mirroring the "new ground" criterion the 2026-07-28 pass used
for `host.rs`.

### `pixelflow-core/src/lattice/cell_grid.rs` (new file, 565 lines)

**Not in this pass's final diff.** By the time this branch merged, `main` had
moved: `docs/bugs/2026-08-01-test-quality-audit-followup.md` (#971)
independently found and fixed the same four gaps below, three days after this
pass ran. That fix is what's on `main`; rebasing this branch onto it drops
the would-be-duplicate tests rather than reapplying them. Kept below as a
record of the investigation, not a claim that this pass's fix shipped.

**Before: 35 mutants — 6 missed, 24 caught, 5 unviable.**

1. **`CellGridGeometry::cells_len`'s `*` → `/`** — undetected because every existing test uses
   a geometry with `rows == 1`, where `cols / 1 == cols * 1`. Added
   `cells_len_and_atlas_len_multiply_their_dimensions`, a direct unit test with `cols: 3, rows:
   5` where multiply and integer-divide diverge.
2. **`channel_arena`'s `bg_idx = field(cx, 6 + channel)` → `6 - channel`** — undetected because
   the existing fixture (`tiny_scene`) happened to give every cell equal fg/bg values on the
   channels it checked (R, B), so reading the wrong buffer slot produced the same number by
   coincidence. Added `each_channel_reads_its_own_buffer_slot`: a half-coverage cell with
   distinct fg/bg values on all four channels, checked against the exact expected blend.
3. **`tile_h_edge`'s `+ 0.5` → `- 0.5`/`* 0.5`** (2 mutants) — the existing apron-clamp test
   (`misaligned_samples_fade_through_the_apron_not_into_neighbors`) only stresses the x-axis;
   the y-axis clamp is a separate constant and was untested. Added
   `cells_taller_than_their_tile_fade_to_background`, the height-axis mirror of the existing
   `cells_wider_than_their_tile_fade_to_background`.
4. **`grid_w`/`grid_h`'s `cols * cell_w` → `cols + cell_w`** (2 mutants) — no existing test
   samples a point between the (wrong) sum-boundary and the (correct) product-boundary. Added
   `grid_extent_multiplies_dimension_by_cell_size`: a 3×3 grid at 2pt/cell, sampled at 5.5pt
   (outside the wrong 3+2=5 boundary, inside the correct 3·2=6 one) on both axes.

**After (per #971's independent fix, not this pass's): 35 mutants — 0
missed, 30 caught, 5 unviable.**

### `actor-scheduler/src/host.rs` (96 more lines since 07-28's pass closed it to 0 missed)

The new lines are `GreenSender::send` — a backoff-based blocking delivery path added
alongside the existing `try_send`, per the module doc's "delivery policy belongs to the
scheduler" — which had zero direct test coverage in this crate (its only exerciser is
`pixelflow-runtime`'s `engine_troupe.rs`, a different crate).

**Before: 61 mutants — 1 missed, 26 caught, 30 unviable, 4 timeouts.**

- **`GreenSender::send`'s whole body → `Ok(())`** — missed, since nothing called `send` (only
  `try_send`). Added a happy-path delivery test and
  `green_sender_send_times_out_on_permanently_full_inbox` (the same permanently-full-channel-
  plus-minimal-backoff technique already established in `lib.rs`'s
  `send_with_backoff_returns_timeout_on_permanently_full_channel`), both through
  `GreenSender::new_with_params`/`send` — no private access.
  **Update:** the happy-path test is dropped from this branch's final diff —
  `docs/bugs/2026-07-31-test-quality-audit-followup.md` (#969) independently
  added an equivalent (`green_sender_send_delivers_the_message`), which
  merged first and is what's on `main`. Only the timeout test, unique to this
  pass, survives the rebase.

**After: 61 mutants — 0 missed, 27 caught, 30 unviable, 4 timeouts** (the same 4 timeouts as
2026-07-28's pass: `Host::sweep`'s `+=`→`*=` on the `Blocked`/`Idle`/`Disconnected` arms and
`step_data`'s `<`→`==`/`<=`, all of which hang the mutant rather than get caught — the correct
signal for a loop whose only observable difference is never terminating. Not re-litigated.)

### `pixelflow-runtime/src/coordinator_node.rs` (new file, 907 lines)

**Not in this pass's final diff.** `docs/bugs/2026-08-07-test-quality-audit-followup.md`
(#983) independently found and fixed both gaps below, over a week later —
`coordinator_data_debug_names_its_variant` and the frame-counter assertion
are what's on `main` today, word-for-word equivalent to this pass's version.
Rebasing this branch onto that fix drops the would-be-duplicate rather than
reapplying it. Kept below as a record of the investigation.

Step 5c of the engine-mesh migration: the render coordinator left `EngineCore` and became its
own green node. Small mutant count (8) because most of the file is wiring/plumbing that
`cargo-mutants` marks unviable; the two viable gaps were real.

**Before: 8 mutants — 2 missed, 5 caught, 1 unviable.**

1. **`CoordinatorData`'s `Debug` impl → `Ok(Default::default())`** — missed; no test ever
   called `format!` on it. Added `coordinator_data_debug_names_its_variant`.
2. **`present_cooked_frame`'s `frame_number += 1` → `*= 1`** — missed because the existing
   completion test (`render_complete_with_a_presentable_frame_emits_present_and_vsync_
   telemetry`) checked `out.rendered.is_some()` but never the frame number's value. Strengthened
   that assertion to `frame_number == 1`, and added
   `presenting_two_frames_advances_the_frame_number_each_time` (checks 1, then 2, across two
   full submit→grant→complete cycles) so the mutant can't survive by coincidence at frame 1.

**After: 8 mutants — 7 caught, 1 unviable.**

## Verified

Numbers below are for this pass as originally run, against all three files.
Two of the three (`cell_grid.rs`, `coordinator_node.rs`) are no longer part
of this branch's diff — see the "not in this pass's final diff" notes above —
so the current, rebased state is just `actor-scheduler --lib`: 140/140 (139
on `main` plus this branch's one surviving test).

- `cargo test -p pixelflow-core --lib`: 112/112 (as originally run).
- `cargo test -p actor-scheduler --lib`: 138/138 (as originally run).
- `cargo test -p pixelflow-runtime --lib`: 64/64 (as originally run).
- `cargo check --workspace`: clean.
- `cargo fmt --check` on all three touched crates: clean.
- Re-ran `cargo-mutants` on all three files after the fixes (numbers above) to confirm the
  gaps are closed, not just that the new tests pass.

## Recommended next steps (not done here)

1. `pixelflow-runtime/src/engine_troupe.rs` (440/682 lines changed) and `engine_core.rs`
   (82/185 changed) are the other large pieces of this diff range — `engine_core.rs` shed most
   of its logic to `coordinator_node.rs` (now mutation-tested above), but `engine_troupe.rs`'s
   own wiring/relay code hasn't been mutation-tested directly. Worth a pass if it keeps
   growing.
2. `spatial_bsp.rs` and `terminal_app.rs`'s known seams (2026-07-20/07-24) are still open,
   still judged design calls rather than mechanical fixes.
3. Same standing note as every prior pass: keep mutation-testing new "green node"/actor code
   as it lands, before it accumulates the way `host.rs` and now `coordinator_node.rs` did.
