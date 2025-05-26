// src/term/tests.rs

use crate::term::{
    AnsiCommand,
    EmulatorInput, TerminalEmulator, UserInputAction, ControlEvent, EmulatorAction,
    RenderSnapshot, SelectionRenderState, SelectionMode, CursorRenderState, CursorShape, SnapshotLine,
    DecModeConstant, // For DECTCEM test
};
use crate::ansi::commands::{C0Control, CsiCommand, Attribute};
use crate::glyph::{Glyph, Attributes};
use crate::keys::{KeySymbol, Modifiers};
use crate::color::{Color, NamedColor}; // For color assertions

// Default scrollback for tests, can be adjusted.
const TEST_SCROLLBACK_LIMIT: usize = 100;

fn create_test_emulator(cols: usize, rows: usize) -> TerminalEmulator {
    TerminalEmulator::new(cols, rows, TEST_SCROLLBACK_LIMIT)
}

// Helper to get a Glyph from the snapshot.
// Note: RenderSnapshot now has get_glyph method, but this can be kept if specific behavior is needed for tests.
fn get_glyph_from_snapshot(snapshot: &RenderSnapshot, row: usize, col: usize) -> Option<Glyph> {
    if row < snapshot.dimensions.1 && col < snapshot.dimensions.0 {
        snapshot.lines.get(row).and_then(|line| line.cells.get(col).cloned())
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
        "Snapshot row count mismatch. Expected {}, got {}. Snapshot lines: {}",
        expected_screen.len(),
        snapshot.dimensions.1,
        snapshot.lines.len()
    );
    if !expected_screen.is_empty() {
        assert!(
            snapshot.dimensions.0 >= expected_screen[0].chars().count(), // cols
            "Snapshot col count ({}) is less than expected screen width ({}) for the first expected row.",
            snapshot.dimensions.0,
            expected_screen[0].chars().count()
        );
    }

    for r in 0..snapshot.dimensions.1 {
        let expected_row_str = expected_screen.get(r).unwrap_or_else(|| {
            panic!("Expected screen data missing for row {}", r);
        });
        let mut s_col = 0;
        for (char_idx, expected_char) in expected_row_str.chars().enumerate() {
            if s_col >= snapshot.dimensions.0 {
                panic!(
                    "While processing expected char '{}' (char_idx {}) for row {}, snapshot column index {} exceeded num_cols {}",
                    expected_char, char_idx, r, s_col, snapshot.dimensions.0
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
                "Char mismatch at (row {}, snapshot_col {}, char_idx {}). Expected '{}', got '{}'",
                r, s_col, char_idx, expected_char, glyph.c
            );
            // Assuming single-width characters for this assertion helper.
            // Wide char display width would need to be handled by get_char_display_width.
            s_col += 1;
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
            panic!("Expected cursor to be visible, but cursor_state is None");
        });
        assert_eq!(cursor_state.y, r_expected, "Cursor row mismatch");
        assert_eq!(cursor_state.x, c_expected, "Cursor col mismatch");
    } else {
        assert!(snapshot.cursor_state.is_none(), "Expected cursor to be hidden, but cursor_state is Some");
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
    // LF moves to next line, same column. 'B' prints at (1,0), cursor moves to (1,1)
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
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorForward(1))));
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["          "], Some((0, 1))); // Cursor physical (0,1)

    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorForward(2))));
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

    // Move cursor to (1,0) (second row, first col)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2, 1))));
    let snapshot_before = term.get_render_snapshot();
    assert_screen_state(&snapshot_before, &["ABC", "DEF"], Some((1, 0)));

    // CSI J (EraseInDisplay(0) - Erase Below)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(0))));
    let snapshot_after = term.get_render_snapshot();
    // Clears from cursor (1,0) to end of screen. Line 1 from (1,0) becomes "   "
    assert_screen_state(&snapshot_after, &["ABC", "   "], Some((1, 0)));
}

#[test]
fn test_csi_sgr_fg_color() {
    let mut term = create_test_emulator(5, 1);
    let red_attr = vec![Attribute::Foreground(Color::Named(NamedColor::Red))];
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(red_attr))));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('A')));

    let snapshot = term.get_render_snapshot();
    let glyph_a = get_glyph_from_snapshot(&snapshot, 0, 0).unwrap();

    assert_eq!(glyph_a.c, 'A');
    assert_eq!(glyph_a.attr.fg, Color::Named(NamedColor::Red), "Foreground color should be Red");

    // Reset SGR
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]))));
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('B')));
    let snapshot_b = term.get_render_snapshot();
    assert_screen_state(&snapshot_b, &["AB   "], Some((0, 2))); // A is red, B is default
    let glyph_b = get_glyph_from_snapshot(&snapshot_b, 0, 1).unwrap();
    assert_eq!(glyph_b.attr.fg, Attributes::default().fg, "Foreground color should have reset to default");
}

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
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('E'))); // Cursor at (1,5)

    term.interpret_input(EmulatorInput::Control(ControlEvent::Resize { cols: 10, rows: 4 }));
    let snapshot = term.get_render_snapshot();
    // Content should be preserved, cursor position might be clamped or adjusted.
    // After resize, cursor is usually at its old logical position, clamped. (1,5) -> (1,5)
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
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Print('W'))); // Cursor at (1,1)

    term.interpret_input(EmulatorInput::Control(ControlEvent::Resize { cols: 3, rows: 1 }));
    let snapshot = term.get_render_snapshot();
    // Content "Hello" becomes "Hel". "World" is lost. Cursor (1,1) becomes (0,1) (clamped)
    assert_screen_state(&snapshot, &["Hel"], Some((0, 1)));
}

#[test]
fn test_osc_set_window_title() {
    let mut term = create_test_emulator(10, 1);

    let action = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(
        "2;New Title".as_bytes().to_vec()
    )));
    assert_eq!(action, Some(EmulatorAction::SetTitle("New Title".to_string())));

    let action2 = term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Osc(
        "2;Another Title".as_bytes().to_vec() // ST as \x1b\\
    )));
    assert_eq!(action2, Some(EmulatorAction::SetTitle("Another Title".to_string())));
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
    assert_eq!(action, Some(EmulatorAction::WritePty("X".to_string().into_bytes())));

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
    // This test now focuses on constructing a snapshot with selection,
    // as TerminalEmulator doesn't have direct set_selection methods.
    let num_cols = 10;
    let num_rows = 2;
    let default_glyph = Glyph { c: ' ', attr: Attributes::default() };

    let lines = vec![
        SnapshotLine { is_dirty: true, cells: vec![default_glyph; num_cols] };
        num_rows
    ];

    let selection_state = Some(SelectionRenderState {
        start_coords: (1, 0), // col 1, row 0
        end_coords: (3, 1),   // col 3, row 1
        mode: SelectionMode::Normal,
    });

    let snapshot_with_selection = RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines,
        cursor_state: Some(CursorRenderState { x:0, y:0, shape: CursorShape::Block, cell_char_underneath: ' ', cell_attributes_underneath: Attributes::default()}),
        selection_state,
    };

    assert!(snapshot_with_selection.selection_state.is_some());
    let sel = snapshot_with_selection.selection_state.unwrap();
    assert_eq!(sel.start_coords, (1,0));
    assert_eq!(sel.end_coords, (3,1));
    assert_eq!(sel.mode, SelectionMode::Normal);

    // To test clearing, we'd create another snapshot without selection_state
    let snapshot_cleared = RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines: snapshot_with_selection.lines.clone(), // reuse lines
        cursor_state: snapshot_with_selection.cursor_state.clone(),
        selection_state: None,
    };
    assert!(snapshot_cleared.selection_state.is_none());
}

#[test]
fn test_mode_show_cursor_dectcem() { // DECTCEM is ?25
    let mut term = create_test_emulator(5, 1);

    let snap_default = term.get_render_snapshot();
    assert!(snap_default.cursor_state.is_some(), "Cursor should be visible by default");
    let initial_shape = snap_default.cursor_state.as_ref().unwrap().shape;

    // Hide cursor: DECRST ?25 (CSI ?25l)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(
        DecModeConstant::TextCursorEnable as u16
    ))));
    let snap_hidden = term.get_render_snapshot();
    assert!(snap_hidden.cursor_state.is_none(), "Cursor should be hidden after DECRST ?25l");

    // Show cursor: DECSET ?25 (CSI ?25h)
    term.interpret_input(EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(
        DecModeConstant::TextCursorEnable as u16
    ))));
    let snap_shown = term.get_render_snapshot();
    assert!(snap_shown.cursor_state.is_some(), "Cursor should be visible again after DECSET ?25h");
    assert_eq!(
        snap_shown.cursor_state.as_ref().unwrap().shape, initial_shape,
        "Cursor should revert to its initial non-hidden shape"
    );
}

