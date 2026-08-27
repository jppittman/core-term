# Test quality control follow-up — 2026-08-27

Scope: scheduled continuation of the recurring test-quality audit (most
recent prior pass: `docs/bugs/2026-08-15-test-quality-audit-followup.md`,
`ce2df0e`). This pass covers two crates in depth — `core-term` and
`pixelflow-core` — via full manual audits against STYLE.md's "Testing"
section (public-API-only scope, "it should" descriptive names) plus
`cargo-mutants` (v27.1.0, freshly installed — not present in this
environment, consistent with every prior pass) mutation testing.

## Mutation testing: `core-term/src/ansi/parser.rs`

Never independently mutation-tested before (the file wasn't touched by any
prior audit's fix commits; `key_translator.rs` was mutation-tested in the
same sweep as a spot check and came back 0/0 missed — already closed by
`addb79d`).

**First sweep: 134 mutants, 10 missed.** Triage of each:

- **Real gap — `clear_string_buffer` no-op survives.** `dispatch_osc`/`dcs`/
  `pm`/`apc` all call `mem::take` on the string buffer before dispatch, which
  already empties it regardless of whether `clear_string_buffer()` runs — so
  those call sites can't distinguish the mutant. But the CAN/SUB
  cancel-string path (`process_token`, `State::OscString | ...`) calls
  `clear_string_buffer()` directly *without* an accompanying `mem::take`. If
  that call were a no-op, an aborted OSC/DCS/PM/APC string's bytes would
  leak into the *next* string sequence's payload. **Fixed**: added
  `osc_buffer_does_not_leak_across_a_cancelled_sequence` to
  `ansi/tests.rs`'s `mutation_tests` module, driving the abort-then-restart
  sequence through `process_bytes` (public API) and asserting on the
  resulting `AnsiCommand::Osc` payload.
- **Real gap — `add_string_byte`'s `<` vs `<=` at `MAX_OSC_LEN`.** The
  string buffer's length cap flows straight into the `Osc`/`Dcs`/`Pm`/`Apc`
  command's byte vector (unlike CSI intermediates — see below — nothing
  remaps or discards it), so it's directly observable. **Fixed**: added
  `osc_string_length_is_capped_not_grown_past_max_osc_len`, feeding
  `MAX_OSC_LEN + 1` bytes and asserting the payload is truncated to the cap.
- **Dead code — the four `ConsumeSt` mutants** (`dispatch_osc`/`dcs`/`pm`/
  `apc`'s `if consume_st == ConsumeSt::Keep { /* ST handled separately */ }`).
  The block is empty — both branches of every one of these `if` statements
  do *exactly nothing*, so no test, however designed, could ever tell them
  apart. Confirmed `ConsumeSt` had no other reader anywhere in the file.
  **Fixed by deletion**, not by testing: removed the `ConsumeSt` enum, its
  parameter on all four `dispatch_*` functions, and both call-site variants
  (`ConsumeSt::Keep`/`::Consume`) collapsed to no-arg calls. This is a
  behavior-preserving simplification (confirmed by the before/after mutant
  count: 134 mutants → 70 once the dead branches no longer existed to
  mutate) and directly matches CLAUDE.md's "subtract before you add."
- **Reviewed and accepted as unreachable — `clear_esc_state` no-op,
  `add_intermediate`'s `<`/`<=`, `to_byte_lossy` → `0`/`1` (4 mutants,
  3 call sites).** All four trace back to the same invariant: `state ==
  State::EscIntermediate` is reachable *only* via the one call site
  (`Escape`'s `AnsiToken::Print(inter @ ('('|')'|'*'|'+'))` arm) that sets
  `esc_intermediate = Some(inter)` in the same statement as the state
  transition — so `self.esc_intermediate` is always freshly `Some` at every
  point it's read, and:
  - `clear_esc_state`'s effect (`esc_intermediate = None`) is never observed
    by a subsequent read through any input sequence the public
    `process_token`/`process_bytes` API can produce.
  - The `else` branch at `EscIntermediate`'s handler (`error!("Invalid
    EscIntermediate state"); dispatch_error(token.to_byte_lossy())`) — the
    *only* call site of `to_byte_lossy` — is consequently unreachable from
    any public input; it is a defensive fallback guarding an "impossible"
    state, not a live code path.
  - Separately, `add_intermediate`'s boundary (`MAX_INTERMEDIATES = 2`) has
    no observable effect on any producible `AnsiCommand`: unlike
    `MAX_OSC_LEN`, no recognized CSI command in `commands.rs::from_csi`
    matches on an intermediates slice of length 2 (only `b""`, `b" "`, or a
    private-CSI wildcard that ignores intermediates entirely), and *every*
    path that falls through to `CsiCommand::Unsupported` — capped at 2
    intermediates or not — gets remapped to the same generic
    `AnsiCommand::Error(final_byte)` by `dispatch_csi`, discarding the
    intermediates vector. A capped-at-2 sequence and an uncapped-at-3
    sequence currently produce bit-identical output for every reachable
    final byte.

  Per STYLE.md's "Flexibility: Break Rules Sensibly," these four are left
  undercovered rather than forced: the alternative would be reaching into
  private state to construct an otherwise-unreachable parser configuration,
  which trades one rule (public-API-only testing) for another (kill every
  mutant). Not re-verified as a request for future work — this is a
  considered call, not an open item — but flagged here for visibility if
  the state machine's invariants ever change (e.g. a future refactor that
  lets `EscIntermediate` be entered without freshly setting
  `esc_intermediate` would make `clear_esc_state` load-bearing again).

**Final sweep after fixes: 70 mutants (down from 134 — the `ConsumeSt`
deletion removed 64 now-nonexistent mutation sites), 4 missed (the reviewed
set above), 65 caught, 1 unviable.**

## STYLE.md naming audit: `core-term` (514 tests across 21 files) and `pixelflow-core` (31 files)

Full manual audits of every `#[test]` function in both crates (methodology:
enumerate via `grep -rl '#\[test\]'`, verify naming against the "it should"
rule and scope against public-API-only per file). Full per-file tables were
generated but are not reproduced here — see the session transcript; the
following is the actioned subset plus the backlog.

### Fixed: 40 test renames (core-term)

All verified individually against the actual test body before renaming
(not applied blindly from the audit pass), then the full `core-term`
lib test suite (461 tests) re-run to confirm no breakage:

- `ansi/tests.rs` (1): `csi_s_is_save_cursor` → `csi_lowercase_s_is_save_cursor`
  (disambiguates from the neighboring `csi_s_uppercase_is_scroll_up`).
- `keys.rs` (1): `map_key_found` → `it_should_return_the_bound_action_when_key_and_modifiers_match_a_binding`.
- `io/pty_tests.rs` (4): `pty_spawn_successful`, `pty_read_write_interaction`,
  `pty_resize_successful`, `pty_spawn_invalid_command` → outcome-describing
  `it_should_*` names.
- `term/layout.rs` (3): `pixels_to_cells_basic`/`_with_padding`/`_out_of_bounds`
  → `it_should_*` names.
- `term/unicode.rs` (5): `ascii_char_width`, `box_drawing_char_widths`,
  `cjk_wide_char_widths`, `control_char_widths`, `locale_initializer_called`
  → `it_should_*` names.
- `term/emulator/key_translator.rs` (2): `arrow_keys_normal_mode`/`_app_mode`
  → names stating which escape family each mode produces.
- `term/emulator/input_handler.rs` (3): `paste_text_action_bracketed_on`/
  `_off`, `control_event_resize_minimum_dimensions` → `it_should_*` names.
- `term/screen.rs` (1): `update_selection_when_not_active` →
  `it_should_leave_selection_unchanged_when_updating_while_inactive`
  (previously stated only the condition, not the outcome).
- `term/core_tests.rs` (1): the "`..._or_similar_logic`" hedge name →
  `it_should_wrap_a_wide_char_to_the_next_line_when_it_does_not_fit_in_the_last_column`.
- `terminal_app.rs` (3): `handle_control_resize`, `ctrl_c_interrupts_yes_flood`,
  `handle_management_keydown` → `it_should_*` names.
- `term/tests.rs` (16): `carriage_return_input`, `csi_cursor_forward_cuf`,
  `csi_ed_clear_below_csi_j`, `csi_sgr_fg_color`, `initiate_copy_no_selection`,
  `initiate_copy_with_selection`, `initiate_copy_block_selection`,
  `resize_larger`, `osc_set_window_title`, `key_event_printable_char`,
  `key_event_arrow_up`, `mode_show_cursor_dectcem`,
  `primary_device_attributes_response`, `extend_selection_active_and_inactive`,
  `apply_selection_clear_click_and_drag`, `verify_clear_selection` →
  `it_should_*` names.
- `tests/ansi_to_grid_integration.rs` (2): `cursor_position_command` →
  `it_should_move_the_cursor_to_the_position_specified_by_cup`; and
  `bug_grid_changes_without_render_trigger` — its own comment claimed to
  reproduce a `TerminalApp.handle_os()`/`send_frame()` bug, but the test
  body never touches `TerminalApp` or `send_frame` at all and is otherwise
  identical to `it_should_change_the_grid_checksum_after_each_character_printed`
  one test above it. Renamed to
  `it_should_change_the_grid_checksum_when_ansi_print_commands_are_processed`
  and rewrote the stale "BUG CONFIRMED"/"THE FIX" comments (which described
  a bug this test doesn't actually exercise) into one accurate note. Left
  in place rather than deleted, since it's still a valid (if redundant)
  checksum assertion — deleting a test is a bigger call than a rename/doc
  fix for an unattended pass.

### Fixed: CLAUDE.md-forbidden `raw_add`/`raw_mul` in test fixtures (pixelflow-core)

`src/lattice/tests.rs`'s two test-only `Manifold` fixtures (`XPlusY`,
`ZTimes100`, used by `point_collapse_single_eval`,
`discrete_manifold_round_trip`, `scanline_collapse_round_trip`, and
`box_collapse_4d_layout`) called `Field::raw_add`/`raw_mul` directly instead
of `+`/`*` — exactly the pattern CLAUDE.md's Critical Constraints singles
out ("NO PUBLIC raw_mul, raw_select, raw_add ETC USAGE... ALWAYS construct
the AST, then use the nested contramap pattern to evaluate it"). Replaced
with `+`/`*` operators; since those operators lower to lazy `Manifold`
combinator nodes rather than eager `Field`s (`x + y : Add<Field, Field>`,
not `Field`), each fixture's `eval` now reads `(expr).eval(p)` to reduce the
combinator against the incoming domain point, matching how every other
`Manifold` impl in the codebase is written. Removed the now-unused
`use crate::numeric::Numeric;` import. `cargo test -p pixelflow-core --lib
lattice::tests::`: 27/27 passed, including `box_collapse_4d_layout` (the
`ZTimes100` consumer).

## Backlog — not done this pass

In rough priority order, largest/highest-value first:

1. **`core-term/tests/message_cuj_tests.rs` (17 tests) and 13 of 19 tests in
   `core-term/tests/actor_roundtrip_tests.rs`** never touch the `core_term`
   crate at all — they hand-roll `MockParserActor`/`TestParserActor`/
   `MockTerminalAppActor`/`TestTerminalAppActor` etc. that reimplement
   fragments of ANSI-parsing/terminal-app logic and test *those*, using
   only `actor_scheduler`. Names like `cuj_pty02_command_batch_delivery`,
   `parser_roundtrip_simple_text`, `terminal_app_roundtrip_key_input`
   strongly imply real coverage of core-term's parser/app but provide none.
   The sibling `ansi_parser_message_tests.rs` does this correctly (its
   `RealParserActor` wraps the actual `AnsiProcessor`) and is the pattern to
   copy. This is the single largest finding of the pass and the most
   consequential — it's a false sense of coverage, not just a style
   nit — but rewriting ~30 tests to exercise real core-term types (or
   relocating the actor-plumbing-only ones to `actor-scheduler`'s own test
   suite) is a genuine design decision, not a mechanical fix, and too large
   a blast radius for an unattended pass. Left for a dedicated follow-up.
2. **`Field::store` (`pixelflow-core/src/lib.rs:529`) is `pub(crate)`, not
   public, and its own doc comment says "If you're reading this, you're
   trying to use the library wrong... The function you're looking for is
   `materialize`."** Despite that, the large majority of unit tests across
   `pixelflow-core/src/*.rs` (roughly 90+ of the crate's ~160 tests) call
   `.store()` directly as their test oracle — a systemic TESTS_PRIVATE
   pattern, not isolated incidents. The reference pattern already exists
   in the same crate and should be the template for fixing this backlog
   item: `tests/unit_tests.rs`'s `assert_field_approx_eq` helper (and
   `src/lattice/tests.rs`'s `bilinear_sampler` submodule) compare `Field`s
   via `.lt(...).eval(coords).all()` — an all-public-API boolean-mask
   comparison — never extracting a raw scalar at all. Converting the
   `.store()`-based tests to this pattern crate-wide is a large, mechanical
   but non-trivial sweep (each call site needs its assertion restructured,
   not just renamed) — a natural next multi-pass target, ideally one module
   at a time the way `backend/mod.rs` was closed on 2026-08-15.
3. **Two files bypass privacy even more directly via unsafe pointer casts
   on `Field`** (`unsafe { *(&f as *const Field as *const f32) }`):
   `pixelflow-core/src/mask.rs:201` (`lane0` helper, 3 call sites) and
   `pixelflow-core/tests/test_log2.rs` (`eval_to_f32`, used throughout that
   file, including a raw `.offset()` lane walk in
   `log2_simd_consistency`). Worse than `.store()`: it reads through
   `Field`'s private layout with raw pointers rather than calling any
   accessor. `tests/test_jet2.rs` has 4 tests `#[ignore]`d with the comment
   *"Needs internal Field access for lane extraction"* — direct evidence
   the missing safe accessor is actively blocking legitimate black-box
   tests from running at all. Investigated during this pass: `mask.rs`'s 3
   affected tests (`select_picks_if_true_when_mask_all_true`,
   `select_picks_if_false_when_mask_all_false`,
   `select_opt_matches_select_on_uniform_masks`) could plausibly move to
   the `.lt(...).eval(coords).all()` pattern from item 2, but `Mask::select`
   returns a concrete `Field` (eager), not a lazy combinator, and getting
   the coordinate-domain plumbing right without introducing a subtly wrong
   assertion in SIMD-sensitive code needs more care than this pass's
   remaining budget allowed. Left open rather than risk a wrong fix.
4. **`pixelflow-search/src/egraph/cost.rs`** — still open per the 08-08
   audit and re-flagged 08-15: a partial mutants run previously found one
   real gap (`CostModel::zero()`, already fixed) before its own slow
   `--lib` baseline (~110s) timed out the pass. Not attempted this pass
   (out of scope — this pass's budget went to `core-term`/`pixelflow-core`
   as scoped above). Still needs either a narrower test filter or a longer
   time budget.
5. **`pixelflow-codegen/src/emit/*`** (~1,400 lines) — flagged 08-08, still
   true as of 08-15, not attempted this pass: never mutation-tested under
   its post-crate-split location.
6. **`pixelflow-graphics/src/spatial_bsp.rs`** — confirmed still open as of
   08-15: 19 tests reach into private `bsp.interiors[...]` fields with no
   public accessor. Still a design call (test-only introspection API vs.
   property tests over `eval()` vs. documented rule-break), not attempted
   this pass.
7. **`actor-scheduler/src/lib.rs`'s `backoff_unit_tests`** (~line 1918) —
   confirmed still present; tests `backoff_with_jitter` (private) and
   `send_with_backoff` (`pub(crate)`) directly rather than through a public
   entry point. The 2026-07-20 audit's mutation findings against these were
   not re-verified this pass (checked the code, not re-run through
   `cargo-mutants` — out of this pass's two-crate scope).
8. **`pixelflow-core/src/backend/x86.rs`/`arm.rs`'s *required* `SimdOps`
   methods** (the per-ISA primitives the 08-15 pass's provided-method sweep
   deliberately didn't touch) — never independently mutation-tested as a
   whole-file sweep. Not attempted this pass.
9. **Remaining `pixelflow-core` NEEDS_RENAME/TESTS_PRIVATE findings not
   covered by item 2 or 3** — roughly 60 more renames (largest clusters:
   the compile-only `src/combinators/context.rs` tests, which assert
   nothing and only type-check; the assertion-free `tests/test_jet2h.rs`
   tests, a direct consequence of item 2) and a handful of narrower scope
   issues (`src/lattice/cell_grid.rs`'s two private-IR-arena-internals
   tests, `src/backend/x86.rs`'s `avx512_log2` testing the raw `F32x16`
   type directly, `tests/x86_backend_tests.rs` testing raw `F32x4`
   throughout). Left for a future pass — see item 2's note about doing
   this crate module-by-module rather than in one sweep.
10. **Remaining `core-term` TESTS_PRIVATE findings not covered by item 1**
    — `term/screen.rs` (35/35 tests reach `Screen`'s `pub(super)` fields/
    methods directly rather than through `TerminalEmulator`),
    `term/emulator/mouse.rs` (24/24 call the `pub(crate)` `encode_mouse_event`
    directly), `term/emulator/key_translator.rs` (11/11 call the
    `pub(super)` `translate_key_input` directly), `terminal_app.rs` (4/4
    call private `handle_control`/`handle_management`/`handle_data`/
    `new_registered`). Renamed in this pass where flagged (see above) but
    scope left as-is: this is an established, repo-wide convention for
    testing internal state machines directly (multiple prior audit passes,
    e.g. `addb79d`, added *more* tests to these same modules rather than
    refactoring them toward public-only access), not a regression, so
    treated as accepted rather than backlog — noted here only for
    completeness of the audit record.

## Verified

- `cargo test -p core-term --lib`: 461 passed, 0 failed (up from the
  pre-existing count by the 2 new mutation-gap tests).
- `cargo test -p pixelflow-core`: 123 passed, 1 ignored, 0 failed (lib) +
  all doctests/integration targets green.
- `cargo clippy -p core-term --tests`: clean (fixed one `manual_repeat_n`
  lint on the new OSC-cap test along the way).
- `cargo clippy -p pixelflow-core --tests`: clean.
- `cargo fmt -p core-term -- --check` / `-p pixelflow-core -- --check`: clean.
- `cargo mutants -p core-term --file core-term/src/ansi/parser.rs`: 70
  mutants, 65 caught, 4 reviewed-and-accepted (see above), 1 unviable.
- `cargo mutants -p core-term --file core-term/src/term/emulator/key_translator.rs`
  (spot check, no fixes needed): 0 missed — confirms `addb79d` closed this
  file's gaps.
