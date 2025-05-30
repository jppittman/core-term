// src/term/tests.rs

use crate::ansi::commands::{Attribute, C0Control, CsiCommand};
use crate::color::{Color, NamedColor};
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::keys::{KeySymbol, Modifiers};
use crate::term::action::{MouseButton, MouseEventType};
use crate::term::{
    AnsiCommand,
    ControlEvent,
    CursorRenderState,
    CursorShape,
    DecModeConstant, // For DECTCEM test
    EmulatorAction,
    EmulatorInput,
    // Imports for new tests:
    Point, // From snapshot
    RenderSnapshot,
    SelectionMode,
    Selection, // Added missing import
    // SelectionRenderState, // This was the old name, replaced by snapshot::Selection
    SnapshotLine,
    TerminalEmulator,
    UserInputAction,
}; // For mouse input

// Default scrollback for tests, can be adjusted.
const TEST_SCROLLBACK_LIMIT: usize = 100;

fn create_test_emulator(cols: usize, rows: usize) -> TerminalEmulator {
    TerminalEmulator::new(cols, rows, TEST_SCROLLBACK_LIMIT)
}

// Helper to get a Glyph from the snapshot.
fn get_glyph_from_snapshot(snapshot: &RenderSnapshot, row: usize, col: usize) -> Option<Glyph> {
    if row < snapshot.dimensions.1 && col < snapshot.dimensions.0 {
        snapshot
            .lines
            .get(row)
            .and_then(|line| line.cells.get(col).cloned())
    } else {
        None
    }
}

// asserts screen content and cursor position
#[allow(clippy::panic_in_result_fn)] // Allow panic in this test helper
fn assert_screen_state(
    snapshot: &RenderSnapshot,
    expected_screen: &[&str],
    expected_cursor_pos: Option<(usize, usize)>, // (row, col) for physical cursor
) {
    assert_eq!(
        snapshot.dimensions.1, // rows
        expected_screen.len(),
        "Snapshot row count mismatch. Expected {}, got {}. Snapshot lines: {:?}",
        expected_screen.len(),
        snapshot.dimensions.1,
        snapshot.lines
    );
    if !expected_screen.is_empty() {
        // Ensure the first expected line (if any) isn't wider than the snapshot's column count.
        // This helps catch issues where the test itself might define an impossible expected screen.
        assert!(
            snapshot.dimensions.0 >= expected_screen[0].chars().map(|c| crate::term::unicode::get_char_display_width(c).max(1)).sum::<usize>(),
            "Snapshot col count ({}) is less than the character-width-aware width of the first expected row ({}). Expected screen: {:?}",
            snapshot.dimensions.0,
            expected_screen[0].chars().map(|c| crate::term::unicode::get_char_display_width(c).max(1)).sum::<usize>(),
            expected_screen[0]
        );
    }

    for r in 0..snapshot.dimensions.1 {
        let expected_row_str = expected_screen.get(r).unwrap_or_else(|| {
            panic!("Expected screen data missing for row {}", r);
        });
        let mut s_col = 0; // current column in the snapshot being checked
        let mut expected_chars_iter = expected_row_str.chars().peekable();

        while let Some(expected_char) = expected_chars_iter.next() {
            if s_col >= snapshot.dimensions.0 {
                // This condition means we've run out of snapshot columns to check,
                // but there are still expected characters. This indicates a mismatch
                // if the expected string (considering char widths) is wider than the terminal.
                let remaining_expected: String = expected_chars_iter.collect();
                panic!(
                    "Snapshot row {} (len {}) is shorter than expected string '{}'. Expected char '{}' (and potentially '{}') at snapshot col {} would exceed width.",
                    r, snapshot.dimensions.0, expected_row_str, expected_char, remaining_expected, s_col
                );
            }

            let glyph = get_glyph_from_snapshot(snapshot, r, s_col).unwrap_or_else(|| {
                panic!(
                    "Glyph ({}, {}) not found in snapshot. Expected char: '{}'",
                    r, s_col, expected_char
                )
            });

            assert_eq!(
                glyph.c, expected_char,
                "Char mismatch at (row {}, snapshot_col {}). Expected '{}', got '{}'. Full expected row: '{}', Full actual row: '{:?}'",
                r, s_col, expected_char, glyph.c, expected_row_str, snapshot.lines.get(r).map(|l| &l.cells)
            );

            let char_width = crate::term::unicode::get_char_display_width(expected_char).max(1);

            // If it's a wide char, check the spacer cell
            if char_width == 2 {
                if s_col + 1 < snapshot.dimensions.0 {
                    let spacer_glyph = get_glyph_from_snapshot(snapshot, r, s_col + 1)
                        .unwrap_or_else(|| {
                            panic!(
                                "Wide char spacer glyph ({}, {}) not found. Primary char: '{}'",
                                r,
                                s_col + 1,
                                expected_char
                            )
                        });
                    assert_eq!(
                        spacer_glyph.c,
                        crate::glyph::WIDE_CHAR_PLACEHOLDER,
                        "Expected wide char placeholder at ({}, {}) for char '{}'",
                        r,
                        s_col + 1,
                        expected_char
                    );
                    assert!(
                        spacer_glyph
                            .attr
                            .flags
                            .contains(AttrFlags::WIDE_CHAR_SPACER),
                        "Spacer glyph at ({},{}) should have WIDE_CHAR_SPACER flag",
                        r,
                        s_col + 1
                    );
                } else {
                    // This means a wide character was expected at the very last column, which is valid if it's the primary part.
                }
            }
            s_col += char_width;
        }

        // Check that remaining cells in the row are spaces (default fill)
        for c_fill in s_col..snapshot.dimensions.0 {
            let glyph = get_glyph_from_snapshot(snapshot, r, c_fill)
                .unwrap_or_else(|| panic!("Glyph ({}, {}) not found for fill check", r, c_fill));
            assert_eq!(
                glyph.c, ' ',
                "Expected empty char ' ' for fill at ({}, {}), got '{}'",
                r, c_fill, glyph.c
            );
        }
    }

    if let Some((r_expected, c_expected)) = expected_cursor_pos {
        let cursor_state = snapshot.cursor_state.as_ref().unwrap_or_else(|| {
            panic!(
                "Expected cursor to be visible, but cursor_state is None. Expected pos: ({},{})",
                r_expected, c_expected
            );
        });
        assert_eq!(cursor_state.y, r_expected, "Cursor row mismatch");
        assert_eq!(cursor_state.x, c_expected, "Cursor col mismatch");
    } else {
        assert!(
            snapshot.cursor_state.is_none(),
            "Expected cursor to be hidden, but cursor_state is Some"
        );
    }
}

#[test]
fn test_simple_char_input() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    let snapshot = term.get_render_snapshot();
    // Cursor is at (0,1) *after* printing 'A' at (0,0)
    assert_screen_state(&snapshot, &["A         "], Some((0, 1)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot_b = term.get_render_snapshot();
    assert_screen_state(&snapshot_b, &["AB        "], Some((0, 2)));
}

#[test]
fn test_newline_input() {
    let mut term = create_test_emulator(10, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot = term.get_render_snapshot();
    // LF moves to next line, performs carriage return. 'B' prints at (1,0), cursor moves to (1,1)
    assert_screen_state(&snapshot, &["A         ", "B         "], Some((1, 1)));
}

#[test]
fn test_carriage_return_input() {
    let mut term = create_test_emulator(10, 1);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D')));
    let snapshot = term.get_render_snapshot();
    // "ABC", CR -> (0,0), "D" prints at (0,0) over 'A', cursor moves to (0,1)
    assert_screen_state(&snapshot, &["DBC       "], Some((0, 1)));
}

#[test]
fn test_csi_cursor_forward_cuf() {
    let mut term = create_test_emulator(10, 1); // Cursor at (0,0)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(1),
    )));
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["          "], Some((0, 1))); // Cursor physical (0,1)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorForward(2),
    )));
    let snapshot2 = term.get_render_snapshot();
    assert_screen_state(&snapshot2, &["          "], Some((0, 3))); // Cursor physical (0,3)
}

#[test]
fn test_csi_ed_clear_below_csi_j() {
    let mut term = create_test_emulator(3, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // Cursor to (1,0) after LF
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('F'))); // Screen: ABC, DEF. Cursor at (1,3)

    // Move cursor to (1,0) (second row, first col) physical for the snapshot
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 1),
    )));
    let snapshot_before = term.get_render_snapshot();
    assert_screen_state(&snapshot_before, &["ABC", "DEF"], Some((1, 0)));

    // CSI J (EraseInDisplay(0) - Erase Below)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::EraseInDisplay(0),
    )));
    let snapshot_after = term.get_render_snapshot();
    // Clears from cursor (1,0) to end of screen. Line 1 from (1,0) becomes "   "
    assert_screen_state(&snapshot_after, &["ABC", "   "], Some((1, 0)));
}

#[test]
fn test_csi_sgr_fg_color() {
    let mut term = create_test_emulator(5, 1);
    let red_attr = vec![Attribute::Foreground(Color::Named(NamedColor::Red))];
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(red_attr),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));

    let snapshot = term.get_render_snapshot();
    let glyph_a = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();

    assert_eq!(glyph_a.c, 'A');
    assert_eq!(
        glyph_a.attr.fg,
        Color::Named(NamedColor::Red),
        "Foreground color should be Red"
    );

    // Reset SGR
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot_b = term.get_render_snapshot();
    assert_screen_state(&snapshot_b, &["AB   "], Some((0, 2))); // A is red, B is default
    let glyph_b = get_glyph_from_snapshot(&snapshot_b, 0, 1).unwrap();
    assert_eq!(
        glyph_b.attr.fg,
        Attributes::default().fg,
        "Foreground color should have reset to default"
    );
}

// --- Helpers for Selection Integration Tests ---

fn send_mouse_input(
    emu: &mut TerminalEmulator,
    col: usize,
    row: usize,
    event_type: MouseEventType,
    button: MouseButton,
) -> Option<EmulatorAction> {
    emu.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
        col,
        row,
        event_type,
        button,
        modifiers: Modifiers::empty(), // Default: no modifiers
    }))
}

fn fill_emulator_screen(emu: &mut TerminalEmulator, text_lines: Vec<String>) {
    for (r, line) in text_lines.iter().enumerate() {
        if r > 0 {
            // Simulate CRLF for new lines after the first.
            // CR might not be strictly necessary if LF alone moves to col 0 in this emulator,
            // but it's safer for typical terminal behavior.
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
        }
        for char_val in line.chars() {
            // Using AnsiCommand::Print for individual characters.
            // Emulator's char_processor will handle them.
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(char_val)));
        }
    }
}

// --- End of Helpers for Selection Integration Tests ---

// --- Basic Selection Flow Tests ---
#[test]
fn test_mouse_press_starts_selection() {
    let mut emu = create_test_emulator(10, 5);
    let action = send_mouse_input(&mut emu, 1, 1, MouseEventType::Press, MouseButton::Left);

    assert!(
        emu.screen.selection.is_active,
        "Selection should be active after left press."
    );
    assert_eq!(
        emu.screen.selection.start,
        Some(Point { x: 1, y: 1 }),
        "Selection start point mismatch."
    );
    assert_eq!(
        emu.screen.selection.end,
        Some(Point { x: 1, y: 1 }),
        "Selection end point should be same as start initially."
    );
    assert_eq!(
        emu.screen.selection.mode,
        SelectionMode::Normal,
        "Default selection mode should be Normal."
    );
    assert_eq!(
        action,
        Some(EmulatorAction::RequestRedraw),
        "Action should be RequestRedraw."
    );
}

#[test]
fn test_mouse_drag_updates_selection() {
    let mut emu = create_test_emulator(10, 5);
    // Initial press
    send_mouse_input(&mut emu, 1, 1, MouseEventType::Press, MouseButton::Left);

    // Drag
    let action = send_mouse_input(&mut emu, 5, 2, MouseEventType::Move, MouseButton::Left); // Button for Move is often ignored if already dragging

    assert!(
        emu.screen.selection.is_active,
        "Selection should remain active during drag."
    );
    assert_eq!(
        emu.screen.selection.start,
        Some(Point { x: 1, y: 1 }),
        "Selection start point should not change during drag."
    );
    assert_eq!(
        emu.screen.selection.end,
        Some(Point { x: 5, y: 2 }),
        "Selection end point should update to drag position."
    );
    assert_eq!(
        action,
        Some(EmulatorAction::RequestRedraw),
        "Action should be RequestRedraw."
    );
}

#[test]
fn test_mouse_release_ends_selection_activity() {
    let mut emu = create_test_emulator(10, 5);
    // Press and Drag
    send_mouse_input(&mut emu, 1, 1, MouseEventType::Press, MouseButton::Left);
    send_mouse_input(&mut emu, 5, 2, MouseEventType::Move, MouseButton::Left);

    // Release
    let action = send_mouse_input(&mut emu, 5, 2, MouseEventType::Release, MouseButton::Left);

    assert!(
        !emu.screen.selection.is_active,
        "Selection should be inactive after release."
    );
    assert_eq!(
        emu.screen.selection.start,
        Some(Point { x: 1, y: 1 }),
        "Selection start point should be retained."
    );
    assert_eq!(
        emu.screen.selection.end,
        Some(Point { x: 5, y: 2 }),
        "Selection end point should be retained."
    );
    assert_eq!(
        action,
        Some(EmulatorAction::RequestRedraw),
        "Action should be RequestRedraw."
    );
}

// --- End of Basic Selection Flow Tests ---

// --- Copy Action Tests ---
#[test]
fn test_initiate_copy_no_selection() {
    let mut emu = create_test_emulator(10, 5);
    let action = emu.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
    assert_eq!(action, None, "Should return None if no selection exists.");
}

#[test]
fn test_initiate_copy_with_selection() {
    let mut emu = create_test_emulator(10, 2);
    fill_emulator_screen(&mut emu, vec!["Hello".to_string(), "World".to_string()]);

    // Select "Hello" which is from (0,0) to (4,0)
    send_mouse_input(&mut emu, 0, 0, MouseEventType::Press, MouseButton::Left);
    send_mouse_input(&mut emu, 4, 0, MouseEventType::Move, MouseButton::Left); // End of "Hello" is col 4
    send_mouse_input(&mut emu, 4, 0, MouseEventType::Release, MouseButton::Left);

    let action = emu.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
    assert_eq!(
        action,
        Some(EmulatorAction::CopyToClipboard("Hello".to_string())),
        "Selected text mismatch."
    );
}

#[test]
fn test_initiate_copy_block_selection() {
    let mut emu = create_test_emulator(3, 3);
    fill_emulator_screen(
        &mut emu,
        vec!["ABC".to_string(), "DEF".to_string(), "GHI".to_string()],
    );

    // Manually set up a block selection from (0,0) to (1,1)
    // This creates a 2x2 block: A B
    //                           D E
    emu.screen.clear_selection(); // Ensure no prior state
    emu.screen
        .start_selection(Point { x: 0, y: 0 }, SelectionMode::Block);
    emu.screen.update_selection(Point { x: 1, y: 1 });
    emu.screen.end_selection(); // Mark as not active, but coordinates remain

    let action = emu.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
    // Expected text for block (0,0)-(1,1) from "ABC\nDEF\nGHI":
    // Line 0, cols 0-1: "AB"
    // Line 1, cols 0-1: "DE"
    assert_eq!(
        action,
        Some(EmulatorAction::CopyToClipboard("AB\nDE".to_string())),
        "Block selected text mismatch."
    );
}
// --- End of Copy Action Tests ---

// --- Selection Clearing Tests ---
#[test]
fn test_new_mouse_press_clears_old_selection() {
    let mut emu = create_test_emulator(10, 5);

    // First selection
    send_mouse_input(&mut emu, 0, 0, MouseEventType::Press, MouseButton::Left);
    send_mouse_input(&mut emu, 2, 0, MouseEventType::Move, MouseButton::Left);
    send_mouse_input(&mut emu, 2, 0, MouseEventType::Release, MouseButton::Left);

    let old_selection_end = emu.screen.selection.end;
    assert_eq!(
        old_selection_end,
        Some(Point { x: 2, y: 0 }),
        "Pre-condition: First selection should be (0,0) to (2,0)"
    );
    assert!(!emu.screen.selection.is_active);

    // New mouse press
    let action = send_mouse_input(&mut emu, 1, 1, MouseEventType::Press, MouseButton::Left);

    assert!(
        emu.screen.selection.is_active,
        "New selection should be active."
    );
    assert_eq!(
        emu.screen.selection.start,
        Some(Point { x: 1, y: 1 }),
        "New selection start point mismatch."
    );
    assert_eq!(
        emu.screen.selection.end,
        Some(Point { x: 1, y: 1 }),
        "New selection end point should be same as new start."
    );
    assert_ne!(
        emu.screen.selection.end, old_selection_end,
        "New selection should differ from old one."
    );
    assert_eq!(action, Some(EmulatorAction::RequestRedraw));
}
// --- End of Selection Clearing Tests ---

// --- Selection Interaction with Scrolling Test ---
#[test]
fn test_selection_coordinates_adjust_on_scroll() {
    let mut emu = create_test_emulator(10, 3); // 3 rows high
    fill_emulator_screen(
        &mut emu,
        vec![
            "Line0".to_string(),
            "Line1".to_string(),
            "Line2".to_string(),
        ],
    );
    // Screen:
    // Line0
    // Line1
    // Line2
    // Cursor is at end of Line2.

    // Select "Line1" which is from (0,1) to (4,1)
    send_mouse_input(&mut emu, 0, 1, MouseEventType::Press, MouseButton::Left);
    send_mouse_input(&mut emu, 4, 1, MouseEventType::Move, MouseButton::Left);
    send_mouse_input(&mut emu, 4, 1, MouseEventType::Release, MouseButton::Left); // Selection is (0,1)-(4,1), inactive

    assert_eq!(emu.screen.selection.start, Some(Point { x: 0, y: 1 }));
    assert_eq!(emu.screen.selection.end, Some(Point { x: 4, y: 1 }));
    assert!(!emu.screen.selection.is_active);

    // Cause a scroll by adding a new line.
    // Move cursor to end of Line2 first if not already there (fill_emulator_screen might leave it there)
    // Then print a newline.
    // To be sure, explicitly move cursor to last line, then print LF.
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 1),
    ))); // Row 3, Col 1
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // This LF at the last line (Line2) should scroll up the content.
    // Line0 is lost.
    // Line1 moves to row 0.
    // Line2 moves to row 1.
    // New blank line at row 2.

    // The selection was for "Line1". Its original coordinates were y=1.
    // After scroll, "Line1" is now on y=0.
    // The selection coordinates should have been adjusted by screen.scroll_up_serial().
    // Screen::scroll_up_serial does not currently adjust selection coordinates.
    // This test will demonstrate that.
    // If selection *should* adjust, this test will fail and Screen::scroll_up_serial needs an update.
    // For now, let's test current behavior: selection coordinates are NOT adjusted by scroll.
    // This means the selection might now point to different text or invalid rows.

    // Current expectation based on Screen::scroll_up_serial not touching selection:
    // assert_eq!(emu.screen.selection.start, Some(Point { x: 0, y: 1 }), "Selection start Y should NOT change if scroll doesn't update it.");
    // assert_eq!(emu.screen.selection.end, Some(Point { x: 4, y: 1 }), "Selection end Y should NOT change if scroll doesn't update it.");

    // Revised expectation: If lines are added to scrollback (which scroll_up_serial does for top==0),
    // the grid lines are rotated. So (0,1) becomes (0,0) effectively.
    // This implies that the selection *should* track the content.
    // The `Screen::scroll_up_serial` method does:
    //   `active_grid[top..=bot].rotate_left(n_val);`
    // This means the underlying `Row` objects in the `grid` Vec are shifted.
    // If `Selection` stores absolute row indices relative to the `grid` Vec,
    // then selection *should* automatically follow the scrolled content.

    // Let's re-verify: scroll_up_serial when top == 0 (which is the case here after cursor move and LF):
    // 1. Lines are moved to scrollback.
    // 2. The grid itself is rotated (`active_grid[top..=bot].rotate_left(n_val)`).
    // This means `grid[0]` becomes the old `grid[1]`, `grid[1]` becomes old `grid[2]`, etc.
    // So, if selection was `Point {y:1, ...}`, it should now refer to the content that was previously at `y=1`.
    // No, this is incorrect. The Point {y:1} still means the *new* grid[1].
    // The content at grid[1] has changed.
    // So, selection does NOT automatically follow content with the current `scroll_up_serial`.

    // Therefore, the test should assert that selection coordinates *do not change* with scroll,
    // and thus the selected text would change.
    // OR, if the requirement is that selection *should* try to follow content, then this test
    // will highlight that `scroll_up_serial` needs to adjust selection coordinates.

    // Sticking to testing the *current* implementation of `Screen::scroll_up_serial` which does not adjust selection:
    assert_eq!(
        emu.screen.selection.start,
        Some(Point { x: 0, y: 1 }),
        "Selection start Y should not change due to scroll."
    );
    assert_eq!(
        emu.screen.selection.end,
        Some(Point { x: 4, y: 1 }),
        "Selection end Y should not change due to scroll."
    );

    // Now, if we get selected text, it should be from the *new* line 1, which is old Line2.
    let selected_text_after_scroll = emu.screen.get_selected_text();
    assert_eq!(
        selected_text_after_scroll,
        Some("Line2".to_string()),
        "Selected text should be from the new line 1 (old Line2)."
    );
}
// --- End of Selection Interaction with Scrolling Test ---

// --- Selection with Alternate Screen Test ---
#[test]
fn test_selection_on_alt_screen_then_exit() {
    let mut emu = create_test_emulator(10, 3);
    fill_emulator_screen(
        &mut emu,
        vec!["Primary1".to_string(), "Primary2".to_string()],
    );

    // Enter alt screen
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));
    // Alt screen should be clear. Let's fill it.
    fill_emulator_screen(&mut emu, vec!["Alt1".to_string(), "Alt2".to_string()]);

    // Select "Alt1" on the alt screen (0,0) to (3,0)
    send_mouse_input(&mut emu, 0, 0, MouseEventType::Press, MouseButton::Left);
    send_mouse_input(&mut emu, 3, 0, MouseEventType::Move, MouseButton::Left); // "Alt1" is 4 chars
    send_mouse_input(&mut emu, 3, 0, MouseEventType::Release, MouseButton::Left);

    assert!(emu.screen.alt_screen_active, "Should be on alt screen.");
    assert_eq!(
        emu.screen.get_selected_text(),
        Some("Alt1".to_string()),
        "Selection on alt screen incorrect."
    );

    // Exit alt screen
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));

    assert!(
        !emu.screen.alt_screen_active,
        "Should be back on primary screen."
    );

    // Behavior of selection when switching screens can vary.
    // Common choices:
    // 1. Selection is cleared.
    // 2. Selection from alt screen is discarded, primary screen might have an old selection or none.
    // The current `exit_alt_screen` in `screen.rs` calls `mark_all_dirty` but does not touch `self.selection`.
    // The `enter_alt_screen` also does not touch `self.selection`.
    // This means a selection made on one screen *could* persist visually on the other if not cleared,
    // which is usually undesirable.
    // Let's assume the desired behavior is that selection is cleared when switching.
    // This test will fail if `enter_alt_screen` or `exit_alt_screen` don't clear it.
    // This is a good test for that behavior.
    // The most robust behavior is usually that selection is specific to a screen buffer.
    // So, exiting alt screen should clear any selection that was on the alt screen.
    // And entering alt screen might clear any primary screen selection.

    // For this test, we'll assert that selection is cleared upon exiting alt screen.
    assert_eq!(
        emu.screen.selection,
        Selection::default(),
        "Selection should be cleared after exiting alt screen."
    );
    assert_eq!(
        emu.screen.get_selected_text(),
        None,
        "No selection should be active/present on primary screen after exiting alt."
    );

    // Verify primary screen content is restored (sanity check, not main focus of test)
    let snapshot = emu.get_render_snapshot();
    assert_eq!(snapshot.lines[0].cells[0].c, 'P'); // "Primary1"
}
// --- End of Selection with Alternate Screen Test ---

#[test]
fn test_resize_larger() {
    let mut term = create_test_emulator(5, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('3')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('4')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('5')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E'))); // Cursor at (1,5) logical

    term.interpret_input(EmulatorInput::Control(ControlEvent::Resize {
        cols: 10,
        rows: 4,
    }));
    let snapshot = term.get_render_snapshot();
    // Content should be preserved, cursor position might be clamped or adjusted.
    // After resize, cursor is at its old logical position (1,5), which is (row 1, col 5) physical.
    assert_screen_state(
        &snapshot,
        &["12345     ", "ABCDE     ", "          ", "          "],
        Some((1, 5)),
    );
}

#[test]
fn test_resize_smaller_content_truncation() {
    let mut term = create_test_emulator(5, 2);
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('H')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('e')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('o')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('W'))); // Cursor at logical (1,1)

    term.interpret_input(EmulatorInput::Control(ControlEvent::Resize {
        cols: 3,
        rows: 1,
    }));
    let snapshot = term.get_render_snapshot();
    // Content "Hello" becomes "Hel". "World" is lost. Cursor (1,1) logical becomes (0,1) physical (clamped).
    assert_screen_state(&snapshot, &["Hel"], Some((0, 1)));
}

#[test]
fn test_osc_set_window_title() {
    let mut term = create_test_emulator(10, 1);

    let action = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(
        "2;New Title".as_bytes().to_vec(),
    )));
    assert_eq!(
        action,
        Some(EmulatorAction::SetTitle("New Title".to_string()))
    );

    let action2 = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(
        "0;Another Title".as_bytes().to_vec(),
    )));
    assert_eq!(
        action2,
        Some(EmulatorAction::SetTitle("Another Title".to_string()))
    );
}

#[test]
fn test_key_event_printable_char() {
    let mut term = create_test_emulator(5, 1);
    let key_input = UserInputAction::KeyInput {
        symbol: KeySymbol::Char('x'),
        modifiers: Modifiers::SHIFT,
        text: Some("X".to_string()),
    };
    let action = term.interpret_input(EmulatorInput::User(key_input));
    assert_eq!(
        action,
        Some(EmulatorAction::WritePty("X".to_string().into_bytes()))
    );

    // Simulate PTY echoing 'X' back
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["X    "], Some((0, 1)));
}

#[test]
fn test_key_event_arrow_up() {
    let mut term = create_test_emulator(5, 1);
    let key_input = UserInputAction::KeyInput {
        symbol: KeySymbol::Up,
        modifiers: Modifiers::empty(),
        text: None,
    };
    let action = term.interpret_input(EmulatorInput::User(key_input));

    // Expected sequence depends on cursor key mode (normal vs application)
    // Normal mode: CSI A (\x1b[A)
    // Application mode: SS3 A (\x1bOA)
    // For this test, assume normal mode.
    let expected_pty_output = "\x1b[A".to_string().into_bytes();
    assert_eq!(action, Some(EmulatorAction::WritePty(expected_pty_output)));

    // Arrow key itself doesn't change screen content or cursor without PTY echo
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["     "], Some((0, 0)));
}

#[test]
fn test_snapshot_with_selection() {
    let num_cols = 10;
    let num_rows = 2;
    let default_glyph = Glyph {
        c: ' ',
        attr: Attributes::default(),
    };

    let lines = vec![
        SnapshotLine {
            is_dirty: true,
            cells: vec![default_glyph; num_cols]
        };
        num_rows
    ];

    let selection_state = Some(Selection {
        start: Some(Point { x: 1, y: 0 }), // col 1, row 0
        end: Some(Point { x: 3, y: 1 }),   // col 3, row 1
        mode: SelectionMode::Normal,
        is_active: false, // For a static snapshot, assume selection is not actively changing
    });

    let snapshot_with_selection = RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines,
        cursor_state: Some(CursorRenderState {
            x: 0,
            y: 0,
            shape: CursorShape::Block,
            cell_char_underneath: ' ',
            cell_attributes_underneath: Attributes::default(),
        }),
        selection_state,
    };

    assert!(snapshot_with_selection.selection_state.is_some());
    let sel = snapshot_with_selection.selection_state.unwrap();
    assert_eq!(sel.start, Some(Point { x: 1, y: 0 }));
    assert_eq!(sel.end, Some(Point { x: 3, y: 1 }));
    assert_eq!(sel.mode, SelectionMode::Normal);

    let snapshot_cleared = RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines: snapshot_with_selection.lines.clone(),
        cursor_state: snapshot_with_selection.cursor_state.clone(),
        selection_state: None,
    };
    assert!(snapshot_cleared.selection_state.is_none());
}

#[test]
fn test_mode_show_cursor_dectcem() {
    let mut term = create_test_emulator(5, 1);

    let snap_default = term.get_render_snapshot();
    assert!(
        snap_default.cursor_state.is_some(),
        "Cursor should be visible by default"
    );
    let initial_shape = snap_default.cursor_state.as_ref().unwrap().shape;

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::TextCursorEnable as u16),
    )));
    let snap_hidden = term.get_render_snapshot();
    assert!(
        snap_hidden.cursor_state.is_none(),
        "Cursor should be hidden after DECRST ?25l"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::TextCursorEnable as u16),
    )));
    let snap_shown = term.get_render_snapshot();
    assert!(
        snap_shown.cursor_state.is_some(),
        "Cursor should be visible again after DECSET ?25h"
    );
    assert_eq!(
        snap_shown.cursor_state.as_ref().unwrap().shape,
        initial_shape,
        "Cursor should revert to its initial non-hidden shape"
    );
}

// --- PS1 Multi-line Prompt Tests ---

#[test]
fn test_ps1_multiline_prompt_at_bottom_causes_scroll() {
    let mut term = create_test_emulator(5, 3); // 5 columns, 3 rows

    // Fill first two lines
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: AAAAA\nBBBBB\n     , cursor at (phys 2,0)

    // PS1 Line 1: "P1> " (4 chars)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('>')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));
    // Screen: AAAAA\nBBBBB\nP1>  , cursor at (phys 2,4)

    // PS1 Newline (should cause scroll)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Expected scroll: AAAAA is lost. Screen becomes: BBBBB\nP1>  \n
    // Cursor moves to (phys 2,0) of the new screen state.

    // PS1 Line 2: "$ " (2 chars)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));
    // Screen: BBBBB\nP1>  \n$    , cursor at (phys 2,2)

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["BBBBB", "P1>  ", "$    "], Some((2, 2)));
}

#[test]
fn test_ps1_multiline_prompt_ends_on_last_line_no_scroll_by_prompt() {
    let mut term = create_test_emulator(5, 3); // 5 columns, 3 rows

    // Fill first line
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: AAAAA\n     \n     , cursor at (phys 1,0)

    // PS1 Line 1: "L1"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    // Screen: AAAAA\nL1   \n     , cursor at (phys 1,2)

    // PS1 Newline
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: AAAAA\nL1   \n     , cursor at (phys 2,0)

    // PS1 Line 2: "$ "
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));
    // Screen: AAAAA\nL1   \n$    , cursor at (phys 2,2)

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["AAAAA", "L1   ", "$    "], Some((2, 2)));
}

#[test]
fn test_ps1_multiline_prompt_last_line_fills_screen_then_input() {
    let mut term = create_test_emulator(3, 2); // 3 columns, 2 rows

    // Fill first line
    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: AAA\n   , cursor at (phys 1,0)

    // PS1 Line 1: "B"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    // Screen: AAA\nB  , cursor at (phys 1,1)

    // PS1 Newline (should scroll)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen becomes: B  \n   , cursor at (phys 1,0) of new screen state

    // PS1 Line 2: "CDE" (fills the new last line)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C'))); // phys (1,0) -> (1,1)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D'))); // phys (1,1) -> (1,2)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E'))); // phys (1,2) -> (1,3) logical, wrap_next=true
                                                                        // Screen: B  \nCDE, cursor logical (1,3) (phys (1,2)), term.cursor_wrap_next should be true.

    let snapshot_after_prompt = term.get_render_snapshot();
    assert_screen_state(&snapshot_after_prompt, &["B  ", "CDE"], Some((1, 2)));
    assert!(
        term.cursor_wrap_next,
        "cursor_wrap_next should be true after prompt fills line"
    );

    // Simulate user typing 'X'
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    // print_char('X') should:
    // 1. See cursor_wrap_next = true.
    // 2. Perform CR, then LF (which scrolls). Screen: CDE\n
    // 3. Cursor moves to (phys 1,0) of new screen.
    // 4. Print 'X'. Screen: CDE\nX
    // 5. Cursor moves to (phys 1,1).

    let snapshot_after_input = term.get_render_snapshot();
    assert_screen_state(&snapshot_after_input, &["CDE", "X  "], Some((1, 1)));
    assert!(
        !term.cursor_wrap_next,
        "cursor_wrap_next should be false after printing 'X'"
    );
}

#[test]
fn test_ps1_prompt_causes_multiple_scrolls() {
    let mut term = create_test_emulator(3, 2); // 3 cols, 2 rows

    // Line 0: "AAA"
    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: AAA\n   , cursor at (phys 1,0)

    // Prompt: "L1\nL2\n$ "
    // P1. "L1"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    // Screen: AAA\nL1 , cursor (phys 1,2)

    // P2. LF (scrolls)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: L1 \n   , cursor (phys 1,0)

    // P3. "L2"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    // Screen: L1 \nL2 , cursor (phys 1,2)

    // P4. LF (scrolls again)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: L2 \n   , cursor (phys 1,0)

    // P5. "$ "
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));
    // Screen: L2 \n$  , cursor (phys 1,2)

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["L2 ", "$  "], Some((1, 2)));
}

#[test]
fn test_ps1_prompt_with_internal_wrapping_and_scrolling() {
    let mut term = create_test_emulator(3, 2); // 3 cols, 2 rows
                                               // Fill line 0
    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: AAA\n   , cursor at (phys 1,0)

    // Prompt: "L1long\nL2"
    // P1. "L1l" (fills line)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));
    // Screen: AAA\nL1l, cursor (phys 1,2), term.cursor_wrap_next = true

    // P2. "o" (wraps and scrolls)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('o')));
    // print_char('o') with wrap_next=true:
    // CR -> LF (scrolls). Screen: L1l\n
    // 'o' printed. Screen: L1l\no
    // Cursor (phys 1,1)

    // P3. "n"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('n')));
    // Screen: L1l\non, cursor (phys 1,2)

    // P4. "g" (fills line)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('g')));
    // Screen: L1l\nong, cursor (phys 1,2), term.cursor_wrap_next = true

    // P5. LF (scrolls)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: ong\n   , cursor (phys 1,0)

    // P6. "L2"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    // Screen: ong\nL2 , cursor (phys 1,2)

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["ong", "L2 "], Some((1, 2)));
}

#[test]
fn test_ps1_multiline_exact_fill_then_scroll_on_final_lf() {
    let mut term = create_test_emulator(3, 2); // 3 cols, 2 rows
                                               // Prompt: "P1\nP2\n" (P1, newline, P2, newline)
                                               // This prompt is 3 lines effectively if P1 and P2 are short enough.
                                               // For a 2-row terminal, this will involve scrolling.

    // P1. "P1"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    // Screen: P1 \n   , cursor (phys 0,2)

    // P2. LF
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: P1 \n   , cursor (phys 1,0) (no scroll yet)

    // P3. "P2"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));
    // Screen: P1 \nP2 , cursor (phys 1,2)

    // P4. LF (final LF of the prompt, should scroll)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: P2 \n   , cursor (phys 1,0) (P1 scrolled off)
    // The cursor should be at the start of the new blank line.

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["P2 ", "   "], Some((1, 0)));
}

#[test]
fn test_ps1_multiline_with_sgr_at_bottom_scrolls() {
    let mut term = create_test_emulator(5, 2); // 5 cols, 2 rows

    // Line 0: "AAAAA"
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: AAAAA\n     , cursor (phys 1,0)

    // Prompt: \e[31mP1\e[0m\n\e[32m$ \e[0m
    // Set Red
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(
            NamedColor::Red,
        ))]),
    )));
    // "P1"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    // Reset
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));
    // Screen: AAAAA\nP1   (P1 in red), cursor (phys 1,2)

    // LF (scrolls)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    // Screen: P1   \n     (P1 in red), cursor (phys 1,0)

    // Set Green
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(
            NamedColor::Green,
        ))]),
    )));
    // "$ "
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));
    // Reset
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));
    // Screen: P1   \n$    ($ and space in green), cursor (phys 1,2)

    let snapshot = term.get_render_snapshot();
    // Check content
    assert_screen_state(&snapshot, &["P1   ", "$    "], Some((1, 2)));

    // Check attributes
    let glyph_p = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap(); // P from P1
    let glyph_1 = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap(); // 1 from P1
    let glyph_dollar = get_glyph_from_snapshot(&snapshot, 1, 0).unwrap(); // $
    let glyph_space_after_dollar = get_glyph_from_snapshot(&snapshot, 1, 1).unwrap(); // space after $
    let glyph_final_cursor_cell = get_glyph_from_snapshot(&snapshot, 1, 2).unwrap(); // cell where cursor is

    assert_eq!(glyph_p.attr.fg, Color::Named(NamedColor::Red));
    assert_eq!(glyph_1.attr.fg, Color::Named(NamedColor::Red));
    assert_eq!(glyph_dollar.attr.fg, Color::Named(NamedColor::Green));
    assert_eq!(
        glyph_space_after_dollar.attr.fg,
        Color::Named(NamedColor::Green)
    );
    assert_eq!(
        glyph_final_cursor_cell.attr.fg,
        Attributes::default().fg,
        "Cursor cell attributes should be reset"
    );
}

#[test]
fn test_ps1_multiline_at_bottom_of_scrolling_region() {
    let mut term = create_test_emulator(5, 4); // 5 cols, 4 rows

    // Set scrolling region: top=2, bottom=4 (1-based), so physical rows 1 to 3.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetScrollingRegion { top: 2, bottom: 4 },
    )));
    // Cursor moves to (0,0) logical, which is (phys 0,0) since origin mode is off by default.
    let snap_after_stbm = term.get_render_snapshot();
    assert_screen_state(
        &snap_after_stbm,
        &["     ", "     ", "     ", "     "],
        Some((0, 0)),
    );

    // Line 0: "XXXXX" (outside scroll region)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(1, 1),
    ))); // Phys (0,0)
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    }
    // Screen: XXXXX\n     \n     \n     , cursor (phys 0,4) after wrap_next logic (logical (0,5))

    // Line 1 (top of scroll region, phys_row=1): "AAAAA"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 1),
    ))); // Phys (1,0)
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // LF within region
                                                                                      // Screen: XXXXX\nAAAAA\n     \n     , cursor (phys 2,0)

    // Line 2 (middle of scroll region, phys_row=2): "BBBBB"
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // LF within region
                                                                                      // Screen: XXXXX\nAAAAA\nBBBBB\n     , cursor (phys 3,0) - this is bottom of scroll region

    // Prompt: "P1\nP2"
    // P1. "P1" (on physical row 3, which is term.scroll_bot)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    // Screen: XXXXX\nAAAAA\nBBBBB\nP1   , cursor (phys 3,2)

    // P2. LF (should scroll *within region* rows 1-3)
    // "AAAAA" (phys row 1) scrolls out.
    // "BBBBB" (phys row 2) moves to phys row 1.
    // "P1   " (phys row 3) moves to phys row 2.
    // New blank line at phys row 3.
    // Cursor moves to start of phys row 3.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    // P3. "P2" (on new physical row 3)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    let snapshot = term.get_render_snapshot();
    assert_screen_state(
        &snapshot,
        &["XXXXX", "BBBBB", "P1   ", "P2   "],
        Some((3, 2)),
    );
}

#[test]
fn test_ps1_multiline_prompt_overflows_tiny_screen_repeatedly() {
    let mut term = create_test_emulator(2, 2); // 2x2 terminal

    // Prompt: "A\nB\nC\n> "
    // P1. "A"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A'))); // A , cur(0,1)
                                                                        // P2. LF
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // A \n  , cur(1,0)
                                                                                      // P3. "B"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B'))); // A \nB , cur(1,1)
                                                                        // P4. LF (Scrolls. A is lost)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // B \n  , cur(1,0)
                                                                                      // P5. "C"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C'))); // B \nC , cur(1,1)
                                                                        // P6. LF (Scrolls. B is lost)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // C \n  , cur(1,0)
                                                                                      // P7. ">"
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('>'))); // C \n> , cur(1,1)
                                                                        // P8. " "
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' '))); // C \n> , cur(1,2) -> wrap_next

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["C ", "> "], Some((1, 1)));
    assert!(
        term.cursor_wrap_next,
        "cursor_wrap_next should be true after prompt fills line"
    );
}

// Add to src/term/tests.rs
#[test]
fn test_lf_at_bottom_of_partial_scrolling_region_no_origin_mode() {
    let cols = 10;
    let rows = 5; // e.g., phys rows 0-4
    let mut term = create_test_emulator(cols, rows);

    // Ensure origin mode is OFF (default, but explicit for clarity)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::Origin as u16),
    )));

    // Set scrolling region from phys row 1 to phys row 3 (1-based: top=2, bottom=4)
    // So, scroll_top = 1, scroll_bot = 3. Physical screen bottom is row 4.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetScrollingRegion { top: 2, bottom: 4 },
    )));

    // Sanity check the region (optional, but good for test setup)
    let screen_ctx_after_stbm = term.current_screen_context();
    assert_eq!(
        screen_ctx_after_stbm.scroll_top, 1,
        "STBM scroll_top mismatch"
    );
    assert_eq!(
        screen_ctx_after_stbm.scroll_bot, 3,
        "STBM scroll_bot mismatch"
    );
    assert!(
        !screen_ctx_after_stbm.origin_mode_active,
        "Origin mode should be off"
    );

    // After STBM, cursor moves to (0,0) physical if origin mode off
    // Move cursor to bottom of scrolling region: phys row 3, col 0
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(4, 1), // 1-based: row 4 (phys 3), col 1 (phys 0)
    )));

    // Print some text to fill a few lines in the scroll region
    // and ensure content is distinct.
    // Line 1 (phys): "XXXXX"
    // Line 2 (phys): "YYYYY"
    // Line 3 (phys): "ZZ" -> cursor after ZZ, on phys row 3, col 2
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(2, 1),
    ))); // phys row 1
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(3, 1),
    ))); // phys row 2
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Y')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::CursorPosition(4, 1),
    ))); // phys row 3
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z')));

    let snapshot_before_lf = term.get_render_snapshot();
    assert_screen_state(
        &snapshot_before_lf,
        &[
            "          ", // Row 0 (outside scroll region)
            "XXXXX     ", // Row 1 (scroll_top)
            "YYYYY     ", // Row 2
            "ZZ        ", // Row 3 (scroll_bot, cursor after ZZ)
            "          ", // Row 4 (outside scroll region)
        ],
        Some((3, 2)), // Cursor at physical (row 3, col 2)
    );

    // Issue an LF. Cursor is at physical row 3 (scroll_bot), origin mode is OFF.
    // Expected: Region [1,3] should scroll up.
    // "XXXXX" should be lost (scrolled out of region).
    // "YYYYY" should move to physical row 1.
    // "ZZ   " should move to physical row 2.
    // Physical row 3 should become blank.
    // Cursor should be at physical row 3, col 0.
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    let snapshot_after_lf = term.get_render_snapshot();
    assert_screen_state(
        &snapshot_after_lf,
        &[
            "          ", // Row 0 (outside scroll region, untouched)
            "YYYYY     ", // Row 1 (old line 2 scrolled up)
            "ZZ        ", // Row 2 (old line 3 scrolled up)
            "          ", // Row 3 (new blank line in scroll region)
            "          ", // Row 4 (outside scroll region, untouched)
        ],
        Some((3, 0)), // Cursor at physical (row 3, col 0)
    );
}
#[test]
fn test_primary_device_attributes_response() {
    let mut term = create_test_emulator(80, 24); // Assuming this helper exists

    // Simulate PTY sending CSI 0 c (which is often treated as CSI c)
    // Option 1: If CsiCommand has a dedicated variant (preferred)
    let input_da = EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::PrimaryDeviceAttributes));

    // Option 2: If from_csi is directly used and robust for default params
    // let csi_da_cmd = AnsiCommand::from_csi(vec![0], vec![], false, b'c').unwrap();
    // let input_da = EmulatorInput::Ansi(csi_da_cmd);

    let action = term.interpret_input(input_da);

    let expected_response_bytes = b"\x1b[?6c".to_vec();
    assert_eq!(
        action,
        Some(EmulatorAction::WritePty(expected_response_bytes)),
        "Terminal should respond to Primary Device Attributes query (CSI c)"
    );
}
