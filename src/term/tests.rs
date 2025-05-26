// src/term/tests.rs
#![cfg(test)]
use super::*; // Imports Term
use crate::config::Config;
use crate::glyph::{Cell, CellAttrs, Fill, Flags}; // Assuming Fill is part of public API for cell comparison
use crate::keys::{KeyEvent, KeyMods, NamedKey};
use crate::term::cursor::{Cursor, CursorShape, CursorStyle};
use crate::term::snapshot::{RenderSnapshot, SelectionRange}; // Key for assertions
use std::sync::mpsc;

// Helper to create a Term for testing with specified dimensions.
fn create_test_term(cols: usize, rows: usize) -> Term {
    let mut config = Config::default(); // Use a default config
    config.set_dimensions(cols, rows);
    // The TermEvent sender is needed by Term::new.
    // For many tests, we don't need to listen to these events, so we can ignore the receiver.
    let (term_event_tx, _term_event_rx) = mpsc::channel();
    Term::new(&config, term_event_tx)
}

// Helper to get a specific cell from a snapshot.
fn get_cell_from_snapshot(snapshot: &RenderSnapshot, row: usize, col: usize) -> Option<Cell> {
    if row < snapshot.num_rows && col < snapshot.num_cols {
        // Cells are stored in a flat Vec, row-major order.
        snapshot.cells.get(row * snapshot.num_cols + col).cloned()
    } else {
        None
    }
}

// Helper to assert screen content based on a snapshot.
// `expected_screen` is a slice of strings, each string representing a row.
// `expected_cursor` is Option<(row, col)>.
fn assert_screen_state(
    snapshot: &RenderSnapshot,
    expected_screen: &[&str],
    expected_cursor_pos: Option<(usize, usize)>, // (row, col)
) {
    assert_eq!(
        snapshot.num_rows,
        expected_screen.len(),
        "Snapshot row count mismatch"
    );
    if !expected_screen.is_empty() {
        assert_eq!(
            snapshot.num_cols,
            expected_screen[0].chars().count(),
            "Snapshot col count mismatch with expected screen width"
        );
    }

    for r in 0..snapshot.num_rows {
        let expected_row_str = expected_screen[r];
        let mut s_col = 0; // current column in snapshot.cells, accounts for multi-width chars
        for (char_idx, expected_char) in expected_row_str.chars().enumerate() {
            if s_col >= snapshot.num_cols {
                panic!(
                    "Expected row string '{}' is wider than snapshot num_cols {}",
                    expected_row_str, snapshot.num_cols
                );
            }
            let cell = get_cell_from_snapshot(snapshot, r, s_col)
                .unwrap_or_else(|| panic!("Cell ({}, {}) not found in snapshot", r, s_col));

            assert_eq!(
                cell.char, expected_char,
                "Char mismatch at (row {}, col_char_idx {})",
                r, char_idx
            );

            // Increment s_col by cell width, default to 1 if not available or always 1
            s_col += cell.width.max(1) as usize;
        }
        // Check if remaining cells in the row are empty if expected_row_str was shorter
        for c_fill in s_col..snapshot.num_cols {
            let cell = get_cell_from_snapshot(snapshot, r, c_fill)
                .unwrap_or_else(|| panic!("Cell ({}, {}) not found for fill check", r, c_fill));
            assert_eq!(
                cell.char, ' ',
                "Expected empty char for fill at ({}, {})",
                r, c_fill
            );
        }
    }

    if let Some((r, c)) = expected_cursor_pos {
        assert_eq!(snapshot.cursor.row, r, "Cursor row mismatch");
        assert_eq!(snapshot.cursor.col, c, "Cursor col mismatch");
    }
}

#[test]
fn test_simple_char_input() {
    let mut term = create_test_term(10, 1);
    term.process_input(b"A");
    let snapshot = term.get_render_snapshot();

    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 0).unwrap().char, 'A');
    assert_eq!(snapshot.cursor.col, 1);
    assert_eq!(snapshot.cursor.row, 0);

    term.process_input(b"B");
    let snapshot_b = term.get_render_snapshot();
    assert_eq!(get_cell_from_snapshot(&snapshot_b, 0, 1).unwrap().char, 'B');
    assert_eq!(snapshot_b.cursor.col, 2);
}

#[test]
fn test_newline_input() {
    let mut term = create_test_term(10, 2);
    term.process_input(b"A\nB"); // \n is Line Feed (LF)
    let snapshot = term.get_render_snapshot();

    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 0).unwrap().char, 'A');
    assert_eq!(get_cell_from_snapshot(&snapshot, 1, 0).unwrap().char, 'B');
    assert_eq!(snapshot.cursor.row, 1);
    assert_eq!(snapshot.cursor.col, 1); // Cursor should be after 'B'
}

#[test]
fn test_carriage_return_input() {
    let mut term = create_test_term(10, 1);
    term.process_input(b"ABC\rD"); // \r is Carriage Return (CR)
    let snapshot = term.get_render_snapshot();

    // 'D' should overwrite 'A'
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 0).unwrap().char, 'D');
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 1).unwrap().char, 'B');
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 2).unwrap().char, 'C');
    assert_eq!(snapshot.cursor.col, 1); // Cursor after 'D'
}

#[test]
fn test_csi_cursor_forward_cuf() {
    let mut term = create_test_term(10, 1);
    term.process_input(b"\x1b[C"); // CSI C (Cursor Forward 1)
    let snapshot = term.get_render_snapshot();
    assert_eq!(snapshot.cursor.col, 1);

    term.process_input(b"\x1b[2C"); // CSI 2C (Cursor Forward 2)
    let snapshot2 = term.get_render_snapshot();
    assert_eq!(snapshot2.cursor.col, 3);
}

#[test]
fn test_csi_ed_clear_below_csi_j() {
    let mut term = create_test_term(3, 2);
    term.process_input(b"ABC\nDEF");
    // Place cursor at D (row 1, col 0)
    term.process_input(b"\x1b[2;1H"); // CUP to row 2 (index 1), col 1 (index 0)

    let snapshot_before = term.get_render_snapshot();
    assert_eq!(
        get_cell_from_snapshot(&snapshot_before, 0, 0).unwrap().char,
        'A'
    );
    assert_eq!(
        get_cell_from_snapshot(&snapshot_before, 1, 0).unwrap().char,
        'D'
    );
    assert_eq!(snapshot_before.cursor.row, 1);
    assert_eq!(snapshot_before.cursor.col, 0);

    term.process_input(b"\x1b[J"); // CSI J (Erase Down: clears from cursor to end of screen)
    let snapshot_after = term.get_render_snapshot();

    // Line 0 should be untouched
    assert_eq!(
        get_cell_from_snapshot(&snapshot_after, 0, 0).unwrap().char,
        'A'
    );
    assert_eq!(
        get_cell_from_snapshot(&snapshot_after, 0, 1).unwrap().char,
        'B'
    );
    assert_eq!(
        get_cell_from_snapshot(&snapshot_after, 0, 2).unwrap().char,
        'C'
    );

    // Line 1 from cursor onwards should be cleared
    // Cell (1,0) 'D' should be cleared because cursor is on it.
    assert_eq!(
        get_cell_from_snapshot(&snapshot_after, 1, 0).unwrap().char,
        ' '
    );
    assert_eq!(
        get_cell_from_snapshot(&snapshot_after, 1, 1).unwrap().char,
        ' '
    );
    assert_eq!(
        get_cell_from_snapshot(&snapshot_after, 1, 2).unwrap().char,
        ' '
    );

    // Cursor position should not change by ED
    assert_eq!(snapshot_after.cursor.row, 1);
    assert_eq!(snapshot_after.cursor.col, 0);
}

#[test]
fn test_csi_sgr_fg_color() {
    let mut term = create_test_term(5, 1);
    // Set foreground to ANSI color red (31)
    term.process_input(b"\x1b[31mA");
    let snapshot = term.get_render_snapshot();
    let cell_a = get_cell_from_snapshot(&snapshot, 0, 0).unwrap();

    assert_eq!(cell_a.char, 'A');
    // Asserting specific color requires knowledge of how CellAttrs.fg is set
    // and how it resolves against the default palette.
    // For now, we check that an attribute was applied. A more detailed check
    // would involve inspecting `cell_a.attrs.fg` if it's a public, comparable type.
    // This test assumes `CellAttrs::default()` would have a different fg.
    assert_ne!(
        cell_a.attrs.fg,
        CellAttrs::default().fg,
        "Foreground color should have changed"
    );

    // Reset SGR attributes
    term.process_input(b"\x1b[0mB");
    let snapshot_b = term.get_render_snapshot();
    let cell_b = get_cell_from_snapshot(&snapshot_b, 0, 1).unwrap();
    assert_eq!(cell_b.char, 'B');
    assert_eq!(
        cell_b.attrs.fg,
        CellAttrs::default().fg,
        "Foreground color should have reset"
    );
}

#[test]
fn test_resize_larger() {
    let mut term = create_test_term(5, 2);
    term.process_input(b"12345\nABCDE");

    term.resize(10, 4); // Resize to larger dimensions
    let snapshot = term.get_render_snapshot();

    assert_eq!(snapshot.num_cols, 10);
    assert_eq!(snapshot.num_rows, 4);

    // Content should be preserved in top-left
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 0).unwrap().char, '1');
    assert_eq!(get_cell_from_snapshot(&snapshot, 1, 0).unwrap().char, 'A');

    // New areas should be filled with default cells (spaces)
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 5).unwrap().char, ' '); // New col
    assert_eq!(get_cell_from_snapshot(&snapshot, 2, 0).unwrap().char, ' '); // New row

    // Cursor position might be adjusted or reset depending on implementation,
    // typically stays if within new bounds, or clamps.
    // Let's assume it stays at (1,5) after "ABCDE" then newline.
    // After "ABCDE", cursor is (1,5). If resize clamps, it might be (1, min(5, new_cols-1)).
    // The default cursor behavior on resize needs to be defined by Term's spec.
    // Here, let's check it's within new bounds.
    assert!(snapshot.cursor.row < 4);
    assert!(snapshot.cursor.col < 10);
}

#[test]
fn test_resize_smaller_content_truncation() {
    let mut term = create_test_term(5, 2);
    term.process_input(b"Hello\nWorld"); // Cursor at (1,5) after World

    term.resize(3, 1); // Resize smaller, truncating content
    let snapshot = term.get_render_snapshot();

    assert_eq!(snapshot.num_cols, 3);
    assert_eq!(snapshot.num_rows, 1);

    // Content should be truncated
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 0).unwrap().char, 'H');
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 1).unwrap().char, 'e');
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 2).unwrap().char, 'l');

    // Cursor should be clamped to new bounds.
    // Original cursor (1,5). New bounds (0,0) to (0,2).
    // Expected: (min(1, 0), min(5, 2)) = (0, 2)
    assert_eq!(snapshot.cursor.row, 0);
    assert_eq!(snapshot.cursor.col, 2);
}

#[test]
fn test_osc_set_window_title() {
    let mut term = create_test_term(10, 1);
    let initial_snapshot = term.get_render_snapshot();
    // Default title might be empty or some default from Config.
    // For this test, let's assume it's not "New Title".
    assert_ne!(initial_snapshot.window_title, "New Title");

    // OSC 0: Set icon name and window title
    // OSC 2: Set window title only
    // Using OSC 2 for specificity.
    term.process_input(b"\x1b]2;New Title\x07"); // \x07 is BEL, common terminator
    // Alternative terminator: \x1b\\ (ESC \) String Terminator (ST)
    // term.process_input(b"\x1b]2;New Title\x1b\\");

    let snapshot_after = term.get_render_snapshot();
    assert_eq!(snapshot_after.window_title, "New Title");
}

#[test]
fn test_key_event_printable_char() {
    let mut term = create_test_term(5, 1);
    let key_event = KeyEvent {
        key: NamedKey::Character('X'),
        mods: KeyMods::NONE,
        // ... other fields if any, like raw_key, etc.
    };
    let written_to_pty = term.process_key_event(key_event);

    // Expect 'X' to be written to PTY (if Term translates it)
    assert_eq!(written_to_pty, Some("X".to_string()));

    // And also processed as input if it's an echo-like terminal
    // This depends on Term's internal logic for key events.
    // If process_key_event *only* generates PTY output, then snapshot won't change here.
    // If it also processes it (like local echo), then snapshot would change.
    // Assuming it also processes for this test:
    term.process_input(b"X"); // Simulate echo or direct processing
    let snapshot = term.get_render_snapshot();
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 0).unwrap().char, 'X');
}

#[test]
fn test_key_event_arrow_up() {
    let mut term = create_test_term(5, 1);
    let key_event = KeyEvent {
        key: NamedKey::ArrowUp,
        mods: KeyMods::NONE,
    };
    let written_to_pty = term.process_key_event(key_event);

    // Expect ANSI escape sequence for Arrow Up to be written to PTY
    // Common sequence is ESC [ A or ESC O A (application mode)
    // This depends on Term's key mapping logic.
    // Let's assume ESC [ A for now.
    assert!(
        written_to_pty == Some("\x1b[A".to_string())
            || written_to_pty == Some("\x1bOA".to_string())
    );

    // Snapshot should not change as this is PTY output, not direct screen manipulation by this call.
    let snapshot = term.get_render_snapshot();
    assert_eq!(get_cell_from_snapshot(&snapshot, 0, 0).unwrap().char, ' '); // Screen remains empty
}

#[test]
fn test_set_selection_and_clear() {
    let mut term = create_test_term(10, 2);
    term.process_input(b"0123456789\nABCDEFGHIJ");

    let selection = SelectionRange {
        start_row: 0,
        start_col: 1, // From '1'
        end_row: 1,
        end_col: 3, // To 'C' (inclusive end for range definition)
        is_block: false,
    };
    term.set_selection(selection.clone());

    let snapshot_with_selection = term.get_render_snapshot();
    assert_eq!(snapshot_with_selection.selection, Some(selection));

    term.clear_selection();
    let snapshot_cleared = term.get_render_snapshot();
    assert_eq!(snapshot_cleared.selection, None);
}

#[test]
fn test_mode_show_cursor_deccm() {
    let mut term = create_test_term(5, 1);

    // Default: cursor should be visible (e.g., Block style)
    let snap_default = term.get_render_snapshot();
    assert_ne!(
        snap_default.cursor.style.shape,
        CursorShape::Hidden,
        "Cursor should be visible by default"
    );

    // Hide cursor: DECRST ?25l
    term.process_input(b"\x1b[?25l");
    let snap_hidden = term.get_render_snapshot();
    // This assertion depends on how Term reflects DECTCEM ?25l in the CursorStyle.
    // If it changes shape to 'Hidden' or a similar indicator.
    // If `CursorShape::Hidden` is not a variant, this test needs adjustment based on
    // how cursor visibility is actually represented in `CursorStyle` or `RenderSnapshot`.
    // For this example, let's assume `CursorShape::Hidden` exists and is used.
    // If not, one might check if renderer receives different commands, but that's a renderer test.
    // This is a known challenge: if a mode's state isn't in RenderSnapshot, Term tests are limited.
    // Assuming Term's Cursor reflects this:
    assert_eq!(
        snap_hidden.cursor.style.shape,
        CursorShape::Hidden,
        "Cursor should be hidden after DECRST ?25l"
    );

    // Show cursor: DECSET ?25h
    term.process_input(b"\x1b[?25h");
    let snap_shown = term.get_render_snapshot();
    assert_ne!(
        snap_shown.cursor.style.shape,
        CursorShape::Hidden,
        "Cursor should be visible again after DECSET ?25h"
    );
    // It should revert to its previous non-hidden shape, e.g. Block
    assert_eq!(
        snap_shown.cursor.style.shape,
        CursorShape::Block,
        "Cursor should revert to Block shape"
    );
}
