# Test quality control follow-up — 2026-08-06

Scope: scheduled continuation of
`docs/bugs/2026-08-01-test-quality-audit-followup.md`. Since that pass
(commit `eb03ee9f`), the tree picked up 4 commits (`ab17bab1`..`a2cf8f8a`) —
much smaller than the prior pass's 33-commit, ~8,500-line delta, but the
largest *structural* change: `pixelflow-ir`'s backend/emit code (JIT codegen,
executable buffers, ISA-specific emitters) was extracted into a brand-new
13th workspace crate, `pixelflow-codegen`, and `pixelflow-core/src/backend/mod.rs`
grew ~325 lines of trait definitions (`Backend`, `MaskOps`, `SimdOps`,
`SimdU32Ops`) moved back from `pixelflow-ir`. The rest of the delta is a new
scrollback/cursor damage-tracking feature in `core-term`
(`last_snapshot_view`, `last_cursor_mark`, `scrollback_generation`) and a
`pixelflow-search`/`pixelflow-pipeline` cost-model refactor.

## Static audit: STYLE.md "Test Public API" rule

Delegated to an `Explore` sub-agent scoped to `git diff eb03ee9f..a2cf8f8a`
(this pass's actual cutoff, pinned explicitly rather than as `HEAD` — this
branch has since been rebased past several later commits, and `HEAD` in
prose drifts with every rebase instead of describing what was reviewed),
cross-checked against every prior pass's accepted exceptions so those
weren't re-flagged.

**Result: incomplete — missed a real violation, found here.**
`pixelflow-pipeline/src/training/corpus.rs`'s new
`v1_corpus_is_refused_with_regeneration_hint` test called the private
`read_corpus_bytes` directly instead of the public `read_corpus(path)`, at
this pass's own cutoff — the sub-agent's static pass should have caught this
and didn't. The 2026-08-08 audit (`f676b4b8`, #985) independently found and
fixed the same violation two passes later, before this branch was rebased
onto it. Recorded here for the record; not re-fixed in this diff since it's
already fixed on `main`.

A second violation at the same cutoff, missed by both this pass and #985 and
recorded here for the first time:
`pixelflow-pipeline/src/training/factored.rs`'s new
`parse_expr_fract_builds_sub_floor_compound` and
`parse_expr_hypot_builds_sqrt_mul_add_compound` exercise `parse_expr`, which
is itself `#[cfg(test)]` (`factored.rs:21-22`), and all of the parsing logic
they cover lives in the likewise test-only private `parse_expr_into`. These
tests pin a test harness, not the crate's public API. Accepted as an
exception rather than fixed: `parse_expr` exists only to spell corpus
expressions compactly in *other* tests, so its own misparse is a defect in
the fixtures every downstream test depends on, and there is no public entry
point that reaches it. Left as-is here — fixing it means either promoting the
parser to public API (it is not wanted there) or deleting the coverage — and
carried forward so the next pass sees a recorded decision rather than
rediscovering it a fourth time.

The rest of the apparent size of the crate-split diff was
mostly a `git diff --stat` artifact — without rename detection, moved files
(e.g. `pixelflow-ir/tests/collapse_loop.rs` → `pixelflow-codegen/tests/collapse_loop.rs`,
`pixelflow-ir/src/backend/emit/lowering.rs` → `pixelflow-ir/src/passes.rs`)
look like wholesale new files. Using `git diff -M` recovered the true
mappings: the actual changes at each destination were a handful of
import-path updates (`pixelflow_ir::backend::emit::` →
`pixelflow_codegen::emit::`), not new test logic. Everything genuinely new —
`core-term/src/term/tests.rs`'s damage-tracking tests, `terminal_app.rs`,
`message_cuj_tests.rs`, `pixelflow-search/src/egraph/cost.rs`'s
`every_op_is_priceable` module, `pixelflow-codegen/tests/prod_kernel_jit.rs` —
drives its crate's public API only, the two `factored.rs` parser tests above
excepted. No `raw_mul`/`raw_select`/`raw_add`/SIMD lane exposure found in any
added test code.

## Mutation testing: `core-term`'s new damage-tracking logic

Targeted the *producer* side of this delta's new logic —
`core-term/src/term/emulator/mod.rs` (the `last_snapshot_view`/
`last_cursor_mark` fields and `get_render_snapshot`'s use of them) and
`core-term/src/term/screen.rs` (`scrollback_generation`, bumped at its three
sites there) — not all of it. Two files carrying this delta's new
production lines were left out of the run and remain **unaudited**, noted
here explicitly rather than implied covered:

- `terminal_app.rs`, which gained ~68 new production lines at this pass's
  cutoff (`build_scene`'s `nothing_drawn_changed`/geometry damage gate,
  synchronized-output handling, `has_presented` state) — the *consumer* side
  of the same feature, deciding whether a dirty snapshot actually gets
  rendered.
- `core-term/src/term/emulator/screen_ops.rs:60`, a *fourth*
  `scrollback_generation += 1` site on the producer side (`ED` mode 3, "erase
  saved lines") added by this same delta. Its behavior is reasoned about
  below by hand, but reasoning is not mutation testing: no mutant was ever
  generated for that line, so the "1 real gap" count covers the three
  `screen.rs` sites only. `cargo-mutants` doesn't scope mutation to a diff — `--file`
mutates the whole file — so this run necessarily also covered a large amount
of pre-existing, previously-unaudited code in both files (`resize`,
`set_scrolling_region`, `set_glyph`, `is_selected`, `get_selected_text*`,
`scroll_viewport`, mouse-tracking predicates, etc.): **324 mutants, 132
missed**. Per this series' standing practice, only mutants landing on lines
actually touched by this delta (confirmed against `git diff eb03ee9f..a2cf8f8a`
hunk-by-hunk, not just line number) are this pass's concern; the large
pre-existing-code backlog is noted below for a future pass rather than
addressed here.

**Filtered to the new logic: 1 real gap, found and fixed.**

- `core-term/src/term/emulator/mod.rs:230` — `get_render_snapshot`'s
  active-grid branch: `let is_dirty = view_changed || self.screen.dirty.get(grid_idx)...`.
  Replacing `||` with `&&` survived. Tracing why: `view_changed` becomes true
  whenever `scrollback_generation` bumps, and all four bump sites
  (`screen.rs`'s scroll-push and its two resize-eviction bumps, plus
  `screen_ops.rs`'s `ED` mode 3 / "erase saved lines") each
  *also* touch every affected row's own per-line dirty bit through a
  different path in the same operation — a plain full-screen scroll or erase
  coincidentally dirties every row it could possibly matter for, so `||` and
  `&&` agreed on every scenario the existing four damage-tracking tests
  covered.
  The distinguishing case is a **scroll region that doesn't span the whole
  screen** (`DECSTBM`/`CSI r`): when `scroll_top == 0` a scroll saves history
  and bumps the generation, but only rows *inside* the region are touched —
  rows below the region's bottom keep their independent, untouched per-line
  dirty state. At that point `view_changed` is true globally while those
  rows' own dirty bit is false, and only `||` redraws them.
  Added `scroll_region_push_dirties_rows_outside_the_region`: sets a 3-row
  scroll region on a 6-row screen, drives a scroll via three linefeeds
  (verified this hits the `scroll_top == 0 && history_mode == Save` history
  path in `screen.rs`'s scroll-and-push logic), and asserts rows 3-5
  (outside the region, never touched) come back dirty on that snapshot and
  clean on the next idle one.

  **Correction (per review), with a wrong first fix caught before landing.**
  The assertion above pins per-row dirty state that no production consumer
  reads — `TerminalApp::build_scene`'s `nothing_drawn_changed` is a
  whole-snapshot `any(is_dirty)`, and every row past that gate is rebuilt
  unconditionally regardless of its own flag — so the review comment is
  right that a black-box test should assert what `build_scene` actually
  reads. First attempt did exactly that: replaced the row-3-5 assertion with
  `snapshot.lines.iter().any(is_dirty)`, matching every sibling
  damage-tracking test in this file. **That version does not catch the
  mutant this test exists for**, confirmed by hand: applying the same
  `||`→`&&` mutation at `emulator/mod.rs:230` and re-running still passed.
  Root cause — `is_dirty = view_changed || own_dirty_bit`, and every
  realistic trigger for `view_changed` (scroll-push, resize-eviction,
  `ED 3`) also independently dirties at least one row through its own bit in
  the same operation (see the two ruled-out scenarios below); rows 0-2
  (inside the region) stay dirty from their own bit regardless of the
  mutation, so a whole-snapshot `any()` is satisfied either way and can't
  tell `||` from `&&` apart. Only rows 3-5, which have no independent dirty
  reason, isolate `view_changed`'s own term.
  Reverted to the row-specific assertion, now justified explicitly as a
  white-box test of `get_render_snapshot`'s documented contract (its own doc
  comment on `last_snapshot_view`: the visible surface is "a pure function
  of the grids, the scrollback, and this pair") — the same established
  exception this series already applies to `CoordinatorCore` and other
  "pure core" types, not a claim that any consumer observes per-row state.
  Re-ran `cargo mutants -p core-term --file core-term/src/term/emulator/mod.rs --re "230:42"`
  against the final version: caught (was missed).

The other two candidate distinguishing scenarios were checked and ruled out
as not needed:
- `EraseMode::Scrollback` (`CSI 3J`) bumps the generation but also clears
  every visible row via `clear_line_segment` in the same branch
  (`screen_ops.rs:58-64`) — already dirties everything through the normal
  per-line path regardless of `view_changed`.
- `Screen::resize`'s scrollback-eviction bump (`screen.rs:522`, `:540`)
  coincides with a full grid rebuild in the same call — same reasoning.

## Verified

- `cargo test -p core-term --lib`: 439/439 passing (was 438 before the new
  test).
- `cargo clippy -p core-term --lib --tests`: clean.
- `cargo fmt -p core-term --check`: clean.
- `cargo check --workspace`: clean.
- `cargo mutants -p core-term --file core-term/src/term/emulator/mod.rs --re "230:42"`:
  0 missed (re-run after the fix, confirmed above).

## Recommended next steps (not done here)

1. This pass's `cargo-mutants` run surfaced 131 other missed mutants across
   `screen.rs` (`resize`, `mark_all_dirty`, `mark_line_dirty`,
   `enter_alt_screen`, `set_scrolling_region`, `set_glyph`, `set_tabstop`,
   `clear_tabstops`, `mark_dirty_for_selection`, `is_selected`,
   `get_selected_text`/`get_selected_text_cell`/`get_selected_text_block`,
   `trim_trailing_whitespace_if_needed`) and `emulator/mod.rs`
   (`scroll_viewport`, `reset_viewport`, `viewport_offset`,
   `scrollback_len`, `encode_mouse_event`, `is_mouse_tracking_active`,
   `reports_all_motion`, `reports_button_motion`) — all pre-existing code,
   untouched by this delta and never mutation-tested by any prior pass in
   this series (which has so far focused on `pixelflow-core`,
   `actor-scheduler`, and `pixelflow-core/src/lattice/`). Given the volume
   (selection/tabstop/mouse-tracking logic with essentially no mutation
   coverage), this is the single highest-value target for the next pass —
   likely needs its own dedicated pass rather than a shared one, given 131
   candidate gaps.
2. `spatial_bsp.rs`'s private `interiors[...]` indexing (open since
   2026-07-20) and `core-term/src/terminal_app.rs`'s `create_test_app()`
   using `new_registered` (open since 2026-07-24) — both still unchanged,
   still judged intentional/deferred design calls, not re-litigated here.
3. ~~`pixelflow-runtime/src/coordinator_node.rs`, `engine_troupe.rs`, and
   `actor-scheduler/src/mealy.rs`'s `DedicatedThread`/`step_os` rewrite~~ —
   done: the 2026-08-07 audit (`4c4c900c`, #983), already an ancestor of this
   branch, specifically mutation-tested all three files and added regression
   coverage for the real gaps it found (`slot_progress` lane-starvation in
   `mealy.rs`, the FPS-telemetry frame counter and `Debug` impl in
   `coordinator_node.rs`, `send_vsync_control`/`shut_down` no-ops in
   `engine_troupe.rs`). Removed as an outstanding recommendation so a future
   pass doesn't re-audit completed work; #1 above (`screen.rs`/
   `emulator/mod.rs`'s pre-existing backlog) is this pass's actual
   highest-value outstanding target.
