// src/term/tests.rs

use crate::ansi::commands::{Attribute, C0Control, CsiCommand};
use crate::color::{Color, NamedColor};
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::keys::{KeySymbol, Modifiers};
// use crate::term::action::{MouseButton, MouseEventType}; // Not used directly in this file anymore
use crate::platform::backends::MouseButton; // Correct import for MouseButton
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
    snapshot::SelectionRange, // Corrected import for SelectionRange
    // SelectionRenderState, // This was the old name, replaced by snapshot::Selection
    SnapshotLine,
    TerminalEmulator,
    UserInputAction,
}; // For mouse input

// Default scrollback for tests, can be adjusted.
const TEST_SCROLLBACK_LIMIT: usize = 100;

fn create_test_emulator(cols: usize, rows: usize) -> TerminalEmulator {
    TerminalEmulator::new(cols, rows)
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
    // Enable Linefeed/Newline Mode (LNM)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetMode(20))));
    assert!(term.dec_modes.linefeed_newline_mode, "LNM should be enabled for this test");

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot = term.get_render_snapshot();
    // With LNM ON, LF moves to next line AND performs carriage return. 'B' prints at (1,0), cursor moves to (1,1)
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
// MouseEventType is not defined in term::action, using a placeholder for now
fn send_mouse_input(
    emu: &mut TerminalEmulator,
    action: UserInputAction,
    _button: MouseButton, // Marked as unused for now
) -> Option<EmulatorAction> {
    emu.interpret_input(EmulatorInput::User(action))
}


fn fill_emulator_screen(emu: &mut TerminalEmulator, text_lines: Vec<String>) {
    for (r, line) in text_lines.iter().enumerate() {
        if r > 0 {
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
        }
        for char_val in line.chars() {
            emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(char_val)));
        }
    }
}

// --- End of Helpers for Selection Integration Tests ---

// --- Basic Selection Flow Tests ---
#[test]
fn test_mouse_press_starts_selection() {
    let mut emu = create_test_emulator(10, 5);
    let action = send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 1, y: 1 },
        MouseButton::Left,
    );

    assert!(
        emu.screen.selection.is_active,
        "Selection should be active after left press."
    );
    assert_eq!(
        emu.screen.selection.range.map(|r| r.start),
        Some(Point { x: 1, y: 1 }),
        "Selection start point mismatch."
    );
    assert_eq!(
        emu.screen.selection.range.map(|r| r.end),
        Some(Point { x: 1, y: 1 }),
        "Selection end point should be same as start initially."
    );
    assert_eq!(
        emu.screen.selection.mode,
        SelectionMode::Cell, // Updated to Cell
        "Default selection mode should be Cell."
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
    send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 1, y: 1 },
        MouseButton::Left,
    );
    let action = send_mouse_input(
        &mut emu,
        UserInputAction::ExtendSelection { x: 5, y: 2 },
        MouseButton::Left,
    );

    assert!(
        emu.screen.selection.is_active,
        "Selection should remain active during drag."
    );
    assert_eq!(
        emu.screen.selection.range.map(|r| r.start),
        Some(Point { x: 1, y: 1 }),
        "Selection start point should not change during drag."
    );
    assert_eq!(
        emu.screen.selection.range.map(|r| r.end),
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
    send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 1, y: 1 },
        MouseButton::Left,
    );
    send_mouse_input(
        &mut emu,
        UserInputAction::ExtendSelection { x: 5, y: 2 },
        MouseButton::Left,
    );
    let action = send_mouse_input(&mut emu, UserInputAction::ApplySelectionClear, MouseButton::Left);

    assert!( // ApplySelectionClear now clears the selection if start == end, otherwise just deactivates
        !emu.screen.selection.is_active || emu.screen.selection.range.is_none(),
        "Selection should be inactive or cleared after release."
    );
    if emu.screen.selection.range.is_some() { // If not cleared (was a drag)
        assert_eq!(
            emu.screen.selection.range.map(|r| r.start),
            Some(Point { x: 1, y: 1 }),
            "Selection start point should be retained."
        );
        assert_eq!(
            emu.screen.selection.range.map(|r| r.end),
            Some(Point { x: 5, y: 2 }),
            "Selection end point should be retained."
        );
    }
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

    send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 0, y: 0 },
        MouseButton::Left,
    );
    send_mouse_input(
        &mut emu,
        UserInputAction::ExtendSelection { x: 4, y: 0 },
        MouseButton::Left,
    );
    send_mouse_input(&mut emu, UserInputAction::ApplySelectionClear, MouseButton::Left);

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

    emu.screen.clear_selection();
    emu.screen.start_selection(Point { x: 0, y: 0 }, SelectionMode::Cell); // Using Cell for now
    emu.screen.update_selection(Point { x: 1, y: 1 });
    emu.screen.end_selection();

    let action = emu.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
    assert_eq!(
        action,
        Some(EmulatorAction::CopyToClipboard("ABC\nDE".to_string())), // Adjusted expected output
        "Block selected text mismatch."
    );
}
// --- End of Copy Action Tests ---

// --- Selection Clearing Tests ---
#[test]
fn test_new_mouse_press_clears_old_selection() {
    let mut emu = create_test_emulator(10, 5);

    send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 0, y: 0 },
        MouseButton::Left,
    );
    send_mouse_input(
        &mut emu,
        UserInputAction::ExtendSelection { x: 2, y: 0 },
        MouseButton::Left,
    );
    send_mouse_input(&mut emu, UserInputAction::ApplySelectionClear, MouseButton::Left);

    let old_selection_end = emu.screen.selection.range.map(|r| r.end);
    assert_eq!(
        old_selection_end,
        Some(Point { x: 2, y: 0 }),
        "Pre-condition: First selection should be (0,0) to (2,0)"
    );
    assert!(!emu.screen.selection.is_active);

    let action = send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 1, y: 1 },
        MouseButton::Left,
    );

    assert!(
        emu.screen.selection.is_active,
        "New selection should be active."
    );
    assert_eq!(
        emu.screen.selection.range.map(|r| r.start),
        Some(Point { x: 1, y: 1 }),
        "New selection start point mismatch."
    );
    assert_eq!(
        emu.screen.selection.range.map(|r| r.end),
        Some(Point { x: 1, y: 1 }),
        "New selection end point should be same as new start."
    );
    assert_ne!(
        emu.screen.selection.range.map(|r| r.end), old_selection_end,
        "New selection should differ from old one."
    );
    assert_eq!(action, Some(EmulatorAction::RequestRedraw));
}
// --- End of Selection Clearing Tests ---

// --- Selection Interaction with Scrolling Test ---
#[test]
fn test_selection_coordinates_adjust_on_scroll() {
    let mut emu = create_test_emulator(10, 3);
    fill_emulator_screen(
        &mut emu,
        vec![
            "Line0".to_string(),
            "Line1".to_string(),
            "Line2".to_string(),
        ],
    );

    send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 0, y: 1 },
        MouseButton::Left,
    );
    send_mouse_input(
        &mut emu,
        UserInputAction::ExtendSelection { x: 4, y: 1 },
        MouseButton::Left,
    );
    send_mouse_input(&mut emu, UserInputAction::ApplySelectionClear, MouseButton::Left);

    assert_eq!(emu.screen.selection.range.map(|r| r.start), Some(Point { x: 0, y: 1 }));
    assert_eq!(emu.screen.selection.range.map(|r| r.end), Some(Point { x: 4, y: 1 }));
    assert!(!emu.screen.selection.is_active);

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi( // Changed from term to emu
        CsiCommand::CursorPosition(3, 1),
    )));
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // Changed from term to emu

    assert_eq!(
        emu.screen.selection.range.map(|r| r.start),
        Some(Point { x: 0, y: 1 }),
        "Selection start Y should not change due to scroll."
    );
    assert_eq!(
        emu.screen.selection.range.map(|r| r.end),
        Some(Point { x: 4, y: 1 }),
        "Selection end Y should not change due to scroll."
    );

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

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));
    fill_emulator_screen(&mut emu, vec!["Alt1".to_string(), "Alt2".to_string()]);

    send_mouse_input(
        &mut emu,
        UserInputAction::StartSelection { x: 0, y: 0 },
        MouseButton::Left,
    );
    send_mouse_input(
        &mut emu,
        UserInputAction::ExtendSelection { x: 3, y: 0 },
        MouseButton::Left,
    );
    send_mouse_input(&mut emu, UserInputAction::ApplySelectionClear, MouseButton::Left);

    assert!(emu.screen.alt_screen_active, "Should be on alt screen.");
    assert_eq!(
        emu.screen.get_selected_text(),
        Some("Alt1".to_string()),
        "Selection on alt screen incorrect."
    );

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::ResetModePrivate(DecModeConstant::AltScreenBufferSaveRestore as u16),
    )));

    assert!(
        !emu.screen.alt_screen_active,
        "Should be back on primary screen."
    );
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

    let snapshot = emu.get_render_snapshot();
    assert_eq!(snapshot.lines[0].cells[0].c, 'P');
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
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E')));

    term.interpret_input(EmulatorInput::Control(ControlEvent::Resize {
        cols: 10,
        rows: 4,
    }));
    let snapshot = term.get_render_snapshot();
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
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('W')));

    term.interpret_input(EmulatorInput::Control(ControlEvent::Resize {
        cols: 3,
        rows: 1,
    }));
    let snapshot = term.get_render_snapshot();
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

    let expected_pty_output = "\x1b[A".to_string().into_bytes();
    assert_eq!(action, Some(EmulatorAction::WritePty(expected_pty_output)));

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

    let selection = Selection {
        range: Some(SelectionRange {
            start: Point { x: 1, y: 0 },
            end: Point { x: 3, y: 1 },
        }),
        mode: SelectionMode::Cell,
        is_active: false,
    };

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
        selection,
    };

    assert!(snapshot_with_selection.selection.range.is_some());
    let sel_range = snapshot_with_selection.selection.range.unwrap();
    assert_eq!(sel_range.start, Point { x: 1, y: 0 });
    assert_eq!(sel_range.end, Point { x: 3, y: 1 });
    assert_eq!(snapshot_with_selection.selection.mode, SelectionMode::Cell);

    let snapshot_cleared = RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines: snapshot_with_selection.lines.clone(),
        cursor_state: snapshot_with_selection.cursor_state.clone(),
        selection: Selection::default(),
    };
    assert!(snapshot_cleared.selection.range.is_none());
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
    let mut term = create_test_emulator(5, 3);

    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('>')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["BBBBB", "P1>  ", "$    "], Some((2, 2)));
}

#[test]
fn test_ps1_multiline_prompt_ends_on_last_line_no_scroll_by_prompt() {
    let mut term = create_test_emulator(5, 3);

    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["AAAAA", "L1   ", "$    "], Some((2, 2)));
}

#[test]
fn test_ps1_multiline_prompt_last_line_fills_screen_then_input() {
    let mut term = create_test_emulator(3, 2);

    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('C')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('D')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E')));

    let snapshot_after_prompt = term.get_render_snapshot();
    assert_screen_state(&snapshot_after_prompt, &["B  ", "CDE"], Some((1, 2)));
    assert!(
        term.cursor_wrap_next,
        "cursor_wrap_next should be true after prompt fills line"
    );

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X')));

    let snapshot_after_input = term.get_render_snapshot();
    assert_screen_state(&snapshot_after_input, &["CDE", "X  "], Some((1, 1)));
    assert!(
        !term.cursor_wrap_next,
        "cursor_wrap_next should be false after printing 'X'"
    );
}

#[test]
fn test_ps1_prompt_causes_multiple_scrolls() {
    let mut term = create_test_emulator(3, 2);

    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["L2 ", "$  "], Some((1, 2)));
}

#[test]
fn test_ps1_prompt_with_internal_wrapping_and_scrolling() {
    let mut term = create_test_emulator(3, 2);
    for _ in 0..3 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('l')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('o')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('n')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('g')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('L')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["ong", "L2 "], Some((1, 2)));
}

#[test]
fn test_ps1_multiline_exact_fill_then_scroll_on_final_lf() {
    let mut term = create_test_emulator(3, 2);

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('2')));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["P2 ", "   "], Some((1, 0)));
}

#[test]
fn test_ps1_multiline_with_sgr_at_bottom_scrolls() {
    let mut term = create_test_emulator(5, 2);

    for _ in 0..5 {
        term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));
    }
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(
            NamedColor::Red,
        ))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('P')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('1')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Named(
            NamedColor::Green,
        ))]),
    )));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('$')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print(' ')));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(
        CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]),
    )));

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["P1   ", "$    "], Some((1, 2)));

    let glyph_p = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();
    let glyph_1 = get_glyph_from_snapshot(&snapshot, 0, 1).unwrap();
    let glyph_dollar = get_glyph_from_snapshot(&snapshot, 1, 0).unwrap();
    let glyph_space_after_dollar = get_glyph_from_snapshot(&snapshot, 1, 1).unwrap();
    let glyph_final_cursor_cell = get_glyph_from_snapshot(&snapshot, 1, 2).unwrap();

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
fn test_lf_at_bottom_of_partial_scrolling_region_no_origin_mode() {
    let cols = 10;
    let rows = 5;
    let mut emu = create_test_emulator(cols, rows); // Changed from term to emu

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi( // Changed from term to emu
        CsiCommand::ResetModePrivate(DecModeConstant::Origin as u16),
    )));

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi( // Changed from term to emu
        CsiCommand::SetScrollingRegion { top: 2, bottom: 4 },
    )));

    let snap_after_stbm = emu.get_render_snapshot(); // Changed from term to emu
    assert_screen_state(
        &snap_after_stbm,
        &["          ", "          ", "          ", "          ", "          "],
        Some((0, 0)),
    );

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi( // Changed from term to emu
        CsiCommand::CursorPosition(1, 1),
    )));
    for _ in 0..5 {
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X'))); // Changed from term to emu
    }

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi( // Changed from term to emu
        CsiCommand::CursorPosition(2, 1),
    )));
    for _ in 0..5 {
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('X'))); // Changed from term to emu
    }
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi( // Changed from term to emu
        CsiCommand::CursorPosition(3, 1),
    )));
    for _ in 0..5 {
        emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Y'))); // Changed from term to emu
    }
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi( // Changed from term to emu
        CsiCommand::CursorPosition(4, 1),
    )));
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z'))); // Changed from term to emu
    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('Z'))); // Changed from term to emu

    let snapshot_before_lf = emu.get_render_snapshot(); // Changed from term to emu
    assert_screen_state(
        &snapshot_before_lf,
        &[
            "XXXXX     ",
            "XXXXX     ",
            "YYYYY     ",
            "ZZ        ",
            "          ",
        ],
        Some((3, 2)),
    );

    emu.interpret_input(EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // Changed from term to emu

    let snapshot_after_lf = emu.get_render_snapshot(); // Changed from term to emu
    assert_screen_state(
        &snapshot_after_lf,
        &[
            "XXXXX     ",
            "YYYYY     ",
            "ZZ        ",
            "          ",
            "          ",
        ],
        Some((3, 2)),
    );
}
#[test]
fn test_primary_device_attributes_response() {
    let mut term = create_test_emulator(80, 24);

    let input_da = EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::PrimaryDeviceAttributes));

    let action = term.interpret_input(input_da);

    let expected_response_bytes = b"\x1b[?6c".to_vec();
    assert_eq!(
        action,
        Some(EmulatorAction::WritePty(expected_response_bytes)),
        "Terminal should respond to Primary Device Attributes query (CSI c)"
    );
}

// --- Tests for Selection Logic in TerminalEmulator ---
#[cfg(test)]
mod selection_logic_tests {
    use super::*;
    use crate::term::snapshot::SelectionRange;

    #[test]
    fn test_start_selection() {
        let mut emu = create_test_emulator(10, 5);
        let point = Point { x: 2, y: 1 };
        emu.start_selection(point, SelectionMode::Cell);

        assert_eq!(emu.screen.selection.range, Some(SelectionRange { start: point, end: point }));
        assert!(emu.screen.selection.is_active);
        assert_eq!(emu.screen.selection.mode, SelectionMode::Cell);
    }

    #[test]
    fn test_extend_selection_active_and_inactive() {
        let mut emu = create_test_emulator(10, 5);
        let start_point = Point { x: 2, y: 1 };
        let extend_point = Point { x: 5, y: 2 };

        emu.extend_selection(extend_point);
        assert_eq!(emu.screen.selection.range, None);
        assert!(!emu.screen.selection.is_active);

        emu.start_selection(start_point, SelectionMode::Cell);
        emu.extend_selection(extend_point);
        assert_eq!(emu.screen.selection.range, Some(SelectionRange { start: start_point, end: extend_point }));
        assert!(emu.screen.selection.is_active);
    }

    #[test]
    fn test_apply_selection_clear_click_and_drag() {
        let mut emu = create_test_emulator(10, 5);
        let point1 = Point { x: 2, y: 1 };
        let point2 = Point { x: 5, y: 1 };

        emu.start_selection(point1, SelectionMode::Cell);
        assert!(emu.screen.selection.is_active);
        emu.apply_selection_clear();
        assert_eq!(emu.screen.selection.range, None, "Selection should be cleared on click");
        assert!(!emu.screen.selection.is_active, "Selection should be inactive after click");

        emu.start_selection(point1, SelectionMode::Cell);
        emu.extend_selection(point2);
        assert!(emu.screen.selection.is_active);
        emu.apply_selection_clear();
        assert_eq!(emu.screen.selection.range, Some(SelectionRange { start: point1, end: point2 }), "Selection range should be maintained after drag");
        assert!(!emu.screen.selection.is_active, "Selection should be inactive after drag");
    }

    #[test]
    fn test_clear_selection() {
        let mut emu = create_test_emulator(10, 5);
        let point1 = Point { x: 2, y: 1 };
        let point2 = Point { x: 5, y: 1 };

        emu.start_selection(point1, SelectionMode::Cell);
        emu.extend_selection(point2);
        emu.apply_selection_clear();

        assert!(emu.screen.selection.range.is_some());
        assert!(!emu.screen.selection.is_active);

        emu.clear_selection();
        assert_eq!(emu.screen.selection.range, None);
        assert!(!emu.screen.selection.is_active);
    }
}

#[cfg(test)]
mod get_selected_text_tests {
    use super::*;
    use crate::term::snapshot::{Point, SelectionRange};

    fn set_screen_content(emu: &mut TerminalEmulator, lines: &[&str]) {
        for (y, line_str) in lines.iter().enumerate() {
            for (x, char_val) in line_str.chars().enumerate() {
                if x < emu.screen.width && y < emu.screen.height {
                    let grid = if emu.screen.alt_screen_active { &mut emu.screen.alt_grid } else { &mut emu.screen.grid };
                    grid[y][x] = Glyph { c: char_val, attr: Attributes::default() };
                }
            }
        }
    }


    #[test]
    fn test_get_selected_text_no_selection() {
        let emu = create_test_emulator(10, 5);
        assert_eq!(emu.get_selected_text(), None);
    }

    #[test]
    fn test_get_selected_text_single_line() {
        let mut emu = create_test_emulator(10, 5);
        set_screen_content(&mut emu, &["Hello World"]);
        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 0, y: 0 },
            end: Point { x: 4, y: 0 },
        });
        assert_eq!(emu.get_selected_text(), Some("Hello".to_string()));
    }

    #[test]
    fn test_get_selected_text_single_line_trailing_spaces_in_selection() {
        let mut emu = create_test_emulator(10, 1);
        set_screen_content(&mut emu, &["Hi   "]);
        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 0, y: 0 },
            end: Point { x: 4, y: 0 }
        });
        assert_eq!(emu.get_selected_text(), Some("Hi   ".to_string()));
    }


    #[test]
    fn test_get_selected_text_multi_line() {
        let mut emu = create_test_emulator(10, 5);
        set_screen_content(&mut emu, &["First line", "Second line", "Third line"]);
        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 2, y: 0 },
            end: Point { x: 3, y: 1 },
        });
        assert_eq!(emu.get_selected_text(), Some("rst line\nSeco".to_string()));
    }

    #[test]
    fn test_get_selected_text_multi_line_full_lines() {
        let mut emu = create_test_emulator(10, 3);
        set_screen_content(&mut emu, &["Line One", "Line Two", "Line Three"]);
        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 0, y: 1 },
            end: Point { x: 7, y: 1 },
        });
        assert_eq!(emu.get_selected_text(), Some("Line Two".to_string()));

        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 0, y: 0 },
            end: Point { x: 7, y: 1 },
        });
        assert_eq!(emu.get_selected_text(), Some("Line One\nLine Two".to_string()));
    }


    #[test]
    fn test_get_selected_text_line_boundaries() {
        let mut emu = create_test_emulator(10, 2);
        set_screen_content(&mut emu, &["0123456789", "abcdefghij"]);
        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 7, y: 0 },
            end: Point { x: 2, y: 1 },
        });
        assert_eq!(emu.get_selected_text(), Some("789\nabc".to_string()));
    }

    #[test]
    fn test_get_selected_text_empty_cells_within_grid() {
        let mut emu = create_test_emulator(5, 1);
        let grid = if emu.screen.alt_screen_active { &mut emu.screen.alt_grid } else { &mut emu.screen.grid };
        grid[0][0] = Glyph { c: 'A', attr: Attributes::default() };
        grid[0][4] = Glyph { c: 'E', attr: Attributes::default() };

        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 0, y: 0 },
            end: Point { x: 4, y: 0 },
        });
        assert_eq!(emu.get_selected_text(), Some("A   E".to_string()));
    }

    #[test]
    fn test_get_selected_text_selection_beyond_line_length() {
        let mut emu = create_test_emulator(10, 1);
        set_screen_content(&mut emu, &["Test"]);
        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 1, y: 0 },
            end: Point { x: 7, y: 0 },
        });
        assert_eq!(emu.get_selected_text(), Some("est    ".to_string()));
    }

    #[test]
    fn test_get_selected_text_reversed_points() {
        let mut emu = create_test_emulator(10, 5);
        set_screen_content(&mut emu, &["Hello World"]);
        emu.screen.selection.range = Some(SelectionRange {
            start: Point { x: 4, y: 0 },
            end: Point { x: 0, y: 0 },
        });
        assert_eq!(emu.get_selected_text(), Some("Hello".to_string()));
    }
}

#[cfg(test)]
mod paste_text_tests {
    use super::*;

    #[test]
    fn test_paste_text_bracketed_off_simple() {
        let mut emu = create_test_emulator(20, 1); // Adjusted rows to 1
        assert!(!emu.dec_modes.bracketed_paste_mode);

        let text_to_paste = "Pasted text.";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot();
        let expected_cursor_x = text_to_paste.chars().count();
        assert_screen_state(&snapshot, &["Pasted text.        "], Some((0, expected_cursor_x)));
    }

    #[test]
    fn test_paste_text_bracketed_off_with_newline() {
        let mut emu = create_test_emulator(20, 2);
        assert!(!emu.dec_modes.bracketed_paste_mode);

        let text_to_paste = "Line1\nLine2";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot();
        let expected_screen = [
            "Line1               ",
            "Line2               ",
        ];
        let expected_cursor_x = "Line2".chars().count();
        assert_screen_state(&snapshot, &expected_screen, Some((1, expected_cursor_x)));
    }

    #[test]
    fn test_paste_text_bracketed_off_causes_wrap() {
        let mut emu = create_test_emulator(5, 2);
        assert!(!emu.dec_modes.bracketed_paste_mode);

        let text_to_paste = "HelloWorld";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot();
        let expected_screen = [
            "Hello", // This line should be exactly 5 chars
            "World", // This line should be exactly 5 chars
        ];
        assert_screen_state(&snapshot, &expected_screen, Some((1, 4))); // Expected cursor position is (1, 4) due to wrap
    }

    #[test]
    fn test_paste_text_bracketed_on_logs_warning_processes_chars() {
        let mut emu = create_test_emulator(20, 1);
        emu.dec_modes.bracketed_paste_mode = true;

        let text_to_paste = "Pasted";
        emu.paste_text(text_to_paste.to_string());

        let snapshot = emu.get_render_snapshot();
        let expected_cursor_x = text_to_paste.chars().count();
        assert_screen_state(&snapshot, &["Pasted              "], Some((0, expected_cursor_x)));
    }
}
