# Test quality control follow-up — 2026-08-12

Scope: scheduled continuation of
`docs/bugs/2026-08-06-test-quality-audit-followup.md`. That pass's own
"Recommended next steps" named the single highest-value outstanding
target for the next pass explicitly: `core-term/src/term/screen.rs` and
`core-term/src/term/emulator/mod.rs` carried a 131-missed-mutant backlog
(mostly pre-existing scroll/resize/selection logic that predates this
audit series and had never been mutation-tested end to end). This pass
took that target directly rather than diffing since the last commit — the
branch was already even with `main` at `e2c0c5f`, so there was no new
delta to static-audit.

## Static audit: STYLE.md compliance of the two target files' existing tests

Before adding coverage, read every existing test in
`screen.rs`'s inline `mod tests` (35 tests) and the `selection_logic_tests`/
`get_selected_text_tests` submodules of `core-term/src/term/tests.rs` (13
tests) against STYLE.md's "Test Public API" rule and this series'
established "reads as a complete sentence" naming convention.

**Public-API compliance: no violations.** `screen.rs`'s own tests
construct a `Screen` and call its own methods directly — that's a
`pub(super)` type's unit tests living in its own module, the same
established "pure core" exception this series already applies to
`CoordinatorCore` and similar internal types (most recently in the
2026-08-06 pass), not a violation of testing "internal implementation
details" a public caller can't reach. `tests.rs`'s tests drive
`TerminalEmulator`'s actual public API (`interpret_input`,
`get_render_snapshot`, `clear_selection`, `get_selected_text`) exclusively.
No `raw_mul`/`raw_select`/direct-field-poking-past-a-public-accessor
anywhere in either file.

**Naming: 48 non-descriptive names, all fixed.** A large fraction of both
files' pre-existing tests predate this audit series and don't read as a
sentence when "it should" is prepended — `verify_start_selection`,
`verify_clear_selection` (in *two* different files, ambiguous which one a
stack trace means), bare `start_selection`, `is_selected_normal_no_selection`
("normal" is dead jargon from before `SelectionMode` had that name),
`resize_larger`. Renamed all 48 to state the behavior under test as a
complete sentence — e.g. `verify_clear_selection` →
`clear_selection_resets_selection_and_marks_its_lines_dirty`,
`is_selected_block_reverse_points` →
`is_selected_normalizes_reversed_points_in_block_mode`. Full list is the
diff; not reproduced here. `cargo test -p core-term --lib -- term::screen::tests
term::tests::` before and after the rename: same pass count, confirming
these were pure renames.

## Mutation testing: `screen.rs` and `emulator/mod.rs`

`cargo-mutants` v27.1.0 (freshly installed, not present in this
environment — consistent with every prior pass in this series) against
both files together (`--file screen.rs --file emulator/mod.rs`), 324
mutants total.

**Baseline (before this pass's new tests): 130 missed, 188 caught.**
Concentrated entirely in code with *zero* direct unit coverage —
`scroll_up`, `scroll_down`, `insert_blank_chars_in_line`,
`delete_chars_in_line`, `resize`, `mark_all_dirty`/`mark_all_clean`,
`enter_alt_screen`/`exit_alt_screen`, `set_scrolling_region`, `set_glyph`,
the tab-stop methods, `TabClearMode::from`, and (in `emulator/mod.rs`)
`scroll_viewport`/`reset_viewport`/`scrollback_len`/`viewport_offset` and
the mouse-tracking-mode accessors. All of this is only reachable
indirectly through the file's higher-level ANSI/PS1 integration tests,
which exercise the code paths but don't assert precisely enough to catch
an off-by-one or a swapped comparison operator.

**Round 1 — direct unit tests for the previously-uncovered methods.**
Added ~70 tests exercising each of the above methods directly through
their existing public/`pub(super)` API (no visibility changes), asserting
concrete before/after grid content, dirty-flag state, and return values —
e.g. `scroll_up_shifts_region_rows_up_and_fills_the_bottom_with_blank`,
`resize_growing_preserves_top_left_content_and_fills_new_area_with_blanks`,
`insert_blank_chars_discards_content_pushed_past_the_line_end`. Re-running
mutants: **60 missed, 258 caught** — the bulk of the backlog closed, but a
fresh run (mutants aren't scoped to a diff; the first run had started
before these tests existed) surfaced that several of the *first* round's
own boundary cases weren't tight enough to distinguish `<` from `<=` or
isolate one arm of an `||`/`&&` — e.g. the "invalid region" test used
`scroll_top=3, scroll_bot=1`, which also happens to make the region-height
computation and the dirty-marking range both naturally empty regardless of
which comparison operator is live, so it couldn't tell `scroll_top > scroll_bot`
from `scroll_top >= scroll_bot`.

**Round 2 — precise boundary tests for what round 1 missed.** Added 16
more tests, each constructed to isolate exactly one operand/operator:

- `scroll_up`/`scroll_down`: `top == bot` is a *valid* one-row region (not
  the invalid-region case) — added
  `scroll_up_with_a_single_row_region_is_valid_not_an_error` and its
  `scroll_down` counterpart; `scroll_bot == height` (not just "way past
  it") is still out of range —
  `scroll_up_with_scroll_bot_past_the_screen_height_is_a_no_op` and its
  `scroll_down` counterpart.
- `scroll_up_with_zero_scrollback_limit_does_not_grow_scrollback`
  strengthened to also assert `scrollback_generation` is unchanged: with
  `scrollback_limit == 0`, a mutant that weakens `scrollback_limit > 0` to
  `>= 0` lets a push through, but the very next line
  (`if len() > limit { pop_front() }`) immediately trims it back to
  length 0 — so a length-only assertion can't tell the two apart, but the
  push always bumps the generation whether or not it's immediately
  trimmed away.
- `resize`: `resize_does_not_bump_scrollback_generation_when_exactly_at_the_limit`
  (`len == limit`, not over it, must not bump — the existing
  over-the-limit test alone couldn't distinguish `>` from `>=`) and
  `resize_widens_existing_scrollback_lines_to_the_new_width` (existing
  scrollback rows are actually resized to the new column count, not just
  counted).
- `set_scrolling_region_falls_back_to_full_screen_when_bottom_equals_height`,
  `set_tabstop_at_the_line_width_is_a_no_op`,
  `clear_tabstops_current_column_at_the_line_width_is_a_no_op`: each is
  the "one past the last valid index" boundary, which the existing
  "way out of bounds" tests didn't isolate from "just past the edge".
- `is_selected_returns_false_for_points_above_or_below_a_multi_line_selection`:
  a point strictly above the selection's top row (or strictly below its
  bottom row) — the existing multi-line tests only checked points inside
  and at the exact edges of the selection's column range, never outside
  its row range, so `point.y < sel_start_y || point.y > sel_end_y` and a
  mutant `&&` version agreed on every case they covered.
- `get_selected_text_returns_none_for_a_single_point_selection_entirely_past_the_grid_width`
  and `get_selected_text_pads_columns_beyond_the_grid_width_in_block_mode`:
  a selection point can have `x` past the grid's width (nothing clamps
  `Point` coordinates before they reach `Screen`) — `Cell` mode's
  single-line skip-and-continue branch and `Block` mode's per-column
  padding both had this reachable but untested; the block-mode case in
  particular panics under the `<`→`<=` mutant since the "beyond width"
  branch is what prevents an out-of-bounds row index.
- `TerminalEmulator::scroll_viewport(negative)`: added
  `scroll_viewport_negative_decrements_the_offset_by_the_given_amount`,
  starting the offset well above 1 before scrolling back by 1. A mutant
  that deletes the negation in `let abs_lines = (-lines) as usize` casts
  the *negative* `i32` straight to `usize`, wrapping to a huge number;
  `saturating_sub` on that floors to 0 regardless of the real magnitude —
  indistinguishable from a correct decrement-by-1 only when the offset
  started at 1, which is exactly the case the existing round-1 test used.
- `is_mouse_tracking_active_is_true_when_only_the_any_event_mode_is_set`:
  the existing test only set the VT200 mode (the *second* of four `||`
  operands); the *last* operand (`mouse_any_event_mode`) was untested in
  isolation.
- `scrolled_back_view_shows_scrollback_lines_in_chronological_order`: the
  existing scrollback/viewport tests (this pass's and the 2026-08-06
  pass's) all assert offsets and dirty flags but none had ever read a
  scrolled-back snapshot's actual cell *content* — so
  `get_render_snapshot`'s scrollback-index arithmetic
  (`scrollback_len - effective_offset + y_idx`) had no test that would
  notice if it addressed the wrong line. Fills six distinct lines,
  scrolls fully back, and checks each visible row shows the right
  scrollback entry in the right order.

Final re-run: **41 missed, 277 caught, 5 unviable, 1 timeout** (of 324).
`cargo test -p core-term --lib`: 519/519 (was 439 at the start of this
pass). `cargo clippy -p core-term --lib --tests`: clean. `cargo fmt
--check`: clean (after `cargo fmt`).

### The 41 remaining misses, verified equivalent or deferred by hand

Mutation testing on a whole file (not a diff) always surfaces some
mutants a test genuinely cannot kill because no input makes the mutated
and unmutated code disagree — an "equivalent mutant" — plus some that
are real but out of this pass's reach. Chasing every one with a
contrived test would make the suite more fragile, not more useful, so
each is recorded here with why, rather than forced:

1. **`fill_region_with_glyph:278`, `set_glyph:695` (both `<`→`<=`
   pairs)** — in both cases the branch's *only* effect when the operands
   are equal is either an empty slice iteration or (in `set_glyph`) a
   branch whose invariant (`y < self.height` and row lengths equal
   `self.width`) is already established by an earlier guard in the same
   function — so entering vs. not entering the branch is unobservable.
2. **`insert_blank_chars_in_line:425`, `delete_chars_in_line:466`
   (`||`→`&&`)** — the `x >= width || n == 0` early-return is redundant
   with the `count = n.min(width.saturating_sub(x)); if count == 0 { return }`
   check a few lines later: `saturating_sub` already forces `count` to 0
   whenever either disjunct holds, so the first guard never changes the
   final state, only whether a defensive early-return fires first.
3. **`resize:520,522` (the `scrollback_limit == 0` branch)** —
   `CONFIG.behavior.scrollback_lines` defaults to 1000 and this
   environment's config loader is a placeholder that always returns
   `Config::default()` (`core-term/src/config.rs`), so this branch is
   unreachable through any public path in this test binary. Setting the
   shared global `CONFIG` to 0 to force it would leak into every other
   test running in the same process. Left unreachable, as the prior
   passes in this series have also left CONFIG-gated branches.
4. **`resize:599` (tab-stop loop, `<`→`<=`)** — `i` is drawn from
   `(spacing..nw).step_by(spacing)`, whose upper bound is already
   exclusive of `nw == self.tabs.len()`; the loop can never produce an
   `i` where `<` and `<=` disagree.
5. **`mark_dirty_for_selection:770` (`<`→`<=`)** — redundant with
   `mark_line_dirty`'s own `y < self.dirty.len()` guard, called
   immediately after; the outer check can never observably differ from
   relying on the inner one.
6. **`is_selected:916,926` (`<`→`<=`)** — both lines are only reached
   after the function has already special-cased and returned on
   `raw_start.y == raw_end.y`, so `raw_start.y < raw_end.y` can never
   see an equal pair at that point; `<=` behaves identically to `<`.
7. **`get_selected_text:974` (point-swap `>`→`>=`)** — the boundary case
   (`start.x == end.x` on the same row) makes the "swap" a no-op either
   way (swapping an equal pair produces the same pair).
8. **`get_selected_text:982,987,990` (`+`/`-`/`*` in the capacity
   estimate)** — `est_rows`/`est_cols`/`capacity` feed only
   `String::with_capacity(capacity)`, a performance hint with no effect
   on correctness or observable content.
9. **`get_selected_text_cell:1045` (the `!(width == 0 ...)` cluster, 6
   mutants)** — guards a screen-width-zero special case; `Screen::new`
   and `Screen::resize` both clamp to `MIN_GRID_DIMENSION = 1`
   (`core-term/src/term/mod.rs:36`), so `self.width == 0` is unreachable
   through any public constructor.
10. **`get_selected_text_cell:1049` (`<`→`>`)** — `y_abs` is drawn from
    `norm_start_point.y..=norm_end_point.y`, whose own upper bound
    guarantees `y_abs <= norm_end_point.y` always; `y_abs > norm_end_point.y`
    is unreachable. (The `==` and `<=` variants at this same site *are*
    real and are what round 2's
    `get_selected_text_returns_none_for_a_single_point_selection_entirely_past_the_grid_width`
    test now catches.)
11. **`get_selected_text_cell:1059` (`<`→`<=`)** — `x_abs` is drawn from
    `line_col_start..=effective_line_col_end`, and `effective_line_col_end`
    is explicitly clamped to `self.width.saturating_sub(1)`, so `x_abs`
    can never reach `current_row_glyphs.len()`.
12. **`trim_trailing_whitespace_if_needed:1141,1142`** — this function is
    only ever called after the per-line character loop has pushed at
    least one character (the zero-characters-pushed case takes the
    earlier `continue` instead), so `current_line_len` is always `>= 1`
    when reached; `-`→`+` and `>`→`>=` both preserve the same `> 0`/`>= 0`
    truth value on every reachable input.
13. **`get_render_snapshot:228` (`<`→`<=`)** — `grid_idx = y_idx - effective_offset`
    where `y_idx < height` and this branch only runs when
    `y_idx >= effective_offset`, so `grid_idx < height == active_grid.len()`
    always.
14. **`get_render_snapshot:255` (cursor-underneath bounds, 7 mutants)** —
    `cursor_y`/`cursor_x` come from `CursorController::physical_screen_pos`,
    which clamps to the screen via `move_to_logical` on every resize and
    movement; reaching this branch with an out-of-bounds cursor would
    need either a `CursorController` bug or a transient inconsistency
    between a screen resize and the next cursor clamp. Plausible but not
    demonstrated reachable through the public API in the time this pass
    had — **flagged for a dedicated follow-up**, not claimed equivalent.
15. **`scroll_viewport:325,328` (`lines > 0` → `>=`, `lines < 0` → `<=`)**
    — at the exact boundary (`lines == 0`), both the "scroll up" and
    "scroll down" branches compute `offset + 0` or `offset - 0`: adding
    or subtracting zero changes nothing, so which branch fires (or
    neither) is unobservable.
16. **`resize:542` (`>`→`<`) — `TIMEOUT`, not `MISSED`.** Turns
    `while len() > limit { pop_front() }` into `while len() < limit { pop_front() }`,
    which — for any scrollback shorter than the configured limit (true of
    every existing resize test) — pops down to empty and then spins
    forever (`pop_front` on an empty deque is a no-op that doesn't change
    `len()`, so `0 < limit` stays true). `cargo-mutants` classifies a hang
    as `TIMEOUT` rather than `MISSED`; it isn't a coverage gap the test
    suite failed to catch, it's the mutant itself being non-terminating.

## Verified

- `cargo test -p core-term --lib`: 519/519 passing (439 before this pass).
- `cargo clippy -p core-term --lib --tests`: clean.
- `cargo fmt -p core-term -- --check`: clean.
- `cargo mutants -p core-term --file core-term/src/term/screen.rs --file
  core-term/src/term/emulator/mod.rs`: 41 missed / 277 caught / 5 unviable
  / 1 timeout (of 324), down from a 130-missed baseline; every remaining
  miss is itemized above with the reasoning for leaving it as-is.

## Recommended next steps (not done here)

1. `get_render_snapshot`'s cursor-underneath-bounds branch
   (`emulator/mod.rs:255`, item 14 above) is the one cluster in this
   pass's residual list not verified equivalent — worth a dedicated look
   at whether `CursorController` can transiently disagree with the
   screen's current dimensions (e.g. mid-resize) rather than assuming
   the invariant holds everywhere it's used.
2. This pass covered `screen.rs` and `emulator/mod.rs` specifically,
   per the 2026-08-06 pass's recommendation. The rest of `core-term`
   (`ansi_handler.rs`, `char_processor.rs`, `cursor_handler.rs`,
   `mode_handler.rs`, `osc_handler.rs`, `screen_ops.rs` under
   `term/emulator/`, plus `layout.rs`, `key_translator.rs`) has not been
   mutation-tested by any pass in this series and is a reasonable next
   target.
3. As in every prior pass: this covered two files in `core-term`. The
   `pixelflow-*` crates' own test-quality and mutation-coverage backlog
   (noted across several earlier passes, e.g.
   `pixelflow-search/src/egraph/cost.rs`'s slow test suite from the
   2026-08-08 pass) remains untouched by this pass.
