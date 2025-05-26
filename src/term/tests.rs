// src/term/tests.rs
#![cfg(test)]
// Assuming Term is pub in src/term/mod.rs and term module is declared in lib.rs/main.rs
use crate::config::Config;
use crate::term::Term;
// Assuming these types are pub in src/glyph.rs and glyph module is declared
use crate::glyph::{Cell, CellAttrs, Fill, Flags};
// Assuming these types are pub in src/keys.rs and keys module is declared
use crate::keys::{KeyEvent, KeyMods, NamedKey};
// Assuming these types are pub in src/term/cursor.rs and cursor module is declared within term module
use crate::term::cursor::{Cursor, CursorShape, CursorStyle};
// Assuming these types are pub in src/term/snapshot.rs and snapshot module is declared within term module
use crate::term::snapshot::{RenderSnapshot, SelectionRange};
use std::sync::mpsc;

fn create_test_term(cols: usize, rows: usize) -> Term {
    let mut config = Config::default();
    config.set_dimensions(cols, rows);
    let (term_event_tx, _term_event_rx) = mpsc::channel();
    Term::new(&config, term_event_tx)
}

fn get_cell_from_snapshot(snapshot: &RenderSnapshot, row: usize, col: usize) -> Option<Cell> {
    if row < snapshot.num_rows && col < snapshot.num_cols {
        snapshot.cells.get(row * snapshot.num_cols + col).cloned()
    } else {
        None
    }
}

fn assert_screen_state(
    snapshot: &RenderSnapshot,
    expected_screen: &[&str],
    expected_cursor_pos: Option<(usize, usize)>,
) {
    assert_eq!(
        snapshot.num_rows,
        expected_screen.len(),
        "Snapshot row count mismatch. Expected {}, got {}. Snapshot cells len: {}",
        expected_screen.len(),
        snapshot.num_rows,
        snapshot.cells.len()
    );
    if !expected_screen.is_empty() {
        assert!(
            snapshot.num_cols >= expected_screen[0].chars().count(),
            "Snapshot col count ({}) is less than expected screen width ({}) for the first expected row.",
            snapshot.num_cols,
            expected_screen[0].chars().count()
        );
    }

    for r in 0..snapshot.num_rows {
        let expected_row_str = expected_screen[r];
        let mut s_col = 0;
        for (char_idx, expected_char) in expected_row_str.chars().enumerate() {
            if s_col >= snapshot.num_cols {
                panic!(
                    "While processing expected char '{}' (char_idx {}) for row {}, snapshot column index {} exceeded num_cols {}",
                    expected_char, char_idx, r, s_col, snapshot.num_cols
                );
            }
            let cell = get_cell_from_snapshot(snapshot, r, s_col).unwrap_or_else(|| {
                panic!(
                    "Cell ({}, {}) not found in snapshot. Expected char: '{}'",
                    r, s_col, expected_char
                )
            });

            assert_eq!(
                cell.char, expected_char,
                "Char mismatch at (row {}, snapshot_col {}, char_idx {}). Expected '{}', got '{}'",
                r, s_col, char_idx, expected_char, cell.char
            );
            s_col += cell.width.max(1) as usize;
        }
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
    assert_screen_state(&snapshot, &["A         "], Some((0, 1)));

    term.process_input(b"B");
    let snapshot_b = term.get_render_snapshot();
    assert_screen_state(&snapshot_b, &["AB        "], Some((0, 2)));
}

#[test]
fn test_newline_input() {
    let mut term = create_test_term(10, 2);
    term.process_input(b"A\nB");
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["A         ", "B         "], Some((1, 1)));
}

#[test]
fn test_carriage_return_input() {
    let mut term = create_test_term(10, 1);
    term.process_input(b"ABC\rD");
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["DBC       "], Some((0, 1)));
}

#[test]
fn test_csi_cursor_forward_cuf() {
    let mut term = create_test_term(10, 1);
    term.process_input(b"\x1b[C");
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["          "], Some((0, 1)));

    term.process_input(b"\x1b[2C");
    let snapshot2 = term.get_render_snapshot();
    assert_screen_state(&snapshot2, &["          "], Some((0, 3)));
}

#[test]
fn test_csi_ed_clear_below_csi_j() {
    let mut term = create_test_term(3, 2);
    term.process_input(b"ABC\nDEF");
    term.process_input(b"\x1b[2;1H");

    let snapshot_before = term.get_render_snapshot();
    assert_screen_state(&snapshot_before, &["ABC", "DEF"], Some((1, 0)));

    term.process_input(b"\x1b[J");
    let snapshot_after = term.get_render_snapshot();
    assert_screen_state(&snapshot_after, &["ABC", "   "], Some((1, 0)));
}

#[test]
fn test_csi_sgr_fg_color() {
    let mut term = create_test_term(5, 1);
    term.process_input(b"\x1b[31mA");
    let snapshot = term.get_render_snapshot();
    let cell_a = get_cell_from_snapshot(&snapshot, 0, 0).unwrap();

    assert_eq!(cell_a.char, 'A');
    assert_ne!(
        cell_a.attrs.fg,
        CellAttrs::default().fg,
        "Foreground color should have changed from default"
    );

    term.process_input(b"\x1b[0mB");
    let snapshot_b = term.get_render_snapshot();
    assert_screen_state(&snapshot_b, &["AB   "], Some((0, 2)));
    let cell_b = get_cell_from_snapshot(&snapshot_b, 0, 1).unwrap();
    assert_eq!(
        cell_b.attrs.fg,
        CellAttrs::default().fg,
        "Foreground color should have reset to default"
    );
}

#[test]
fn test_resize_larger() {
    let mut term = create_test_term(5, 2);
    term.process_input(b"12345\nABCDE");
    term.resize(10, 4);
    let snapshot = term.get_render_snapshot();
    assert_screen_state(
        &snapshot,
        &["12345     ", "ABCDE     ", "          ", "          "],
        Some((1, 5)),
    );
}

#[test]
fn test_resize_smaller_content_truncation() {
    let mut term = create_test_term(5, 2);
    term.process_input(b"Hello\nWorld");
    term.resize(3, 1);
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["Hel"], Some((0, 2)));
}

#[test]
fn test_osc_set_window_title() {
    let mut term = create_test_term(10, 1);
    let initial_snapshot = term.get_render_snapshot();
    assert_ne!(
        initial_snapshot.window_title, "New Title",
        "Initial title should not be 'New Title'"
    );

    term.process_input(b"\x1b]2;New Title\x07");
    let snapshot_after = term.get_render_snapshot();
    assert_eq!(snapshot_after.window_title, "New Title");

    term.process_input(b"\x1b]2;Another Title\x1b\\");
    let snapshot_st = term.get_render_snapshot();
    assert_eq!(snapshot_st.window_title, "Another Title");
}

#[test]
fn test_key_event_printable_char() {
    let mut term = create_test_term(5, 1);
    let key_event = KeyEvent {
        key: NamedKey::Character('X'),
        mods: KeyMods::NONE,
    };
    let written_to_pty = term.process_key_event(key_event);
    assert_eq!(written_to_pty, Some("X".to_string()));

    term.process_input(b"X");
    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["X    "], Some((0, 1)));
}

#[test]
fn test_key_event_arrow_up() {
    let mut term = create_test_term(5, 1);
    let key_event = KeyEvent {
        key: NamedKey::ArrowUp,
        mods: KeyMods::NONE,
    };
    let written_to_pty = term.process_key_event(key_event);
    let is_csi_a = written_to_pty == Some("\x1b[A".to_string());
    let is_ss3_a = written_to_pty == Some("\x1bOA".to_string());
    assert!(
        is_csi_a || is_ss3_a,
        "Expected Arrow Up escape sequence, got {:?}",
        written_to_pty
    );

    let snapshot = term.get_render_snapshot();
    assert_screen_state(&snapshot, &["     "], Some((0, 0)));
}

#[test]
fn test_set_selection_and_clear() {
    let mut term = create_test_term(10, 2);
    term.process_input(b"0123456789\nABCDEFGHIJ");

    let selection = SelectionRange {
        start_row: 0,
        start_col: 1,
        end_row: 1,
        end_col: 3,
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

    let snap_default = term.get_render_snapshot();
    assert_ne!(
        snap_default.cursor.style.shape,
        CursorShape::Hidden,
        "Cursor should be visible by default"
    );
    let initial_shape = snap_default.cursor.style.shape;

    term.process_input(b"\x1b[?25l");
    let snap_hidden = term.get_render_snapshot();
    assert_eq!(
        snap_hidden.cursor.style.shape,
        CursorShape::Hidden,
        "Cursor should be hidden after DECRST ?25l"
    );

    term.process_input(b"\x1b[?25h");
    let snap_shown = term.get_render_snapshot();
    assert_ne!(
        snap_shown.cursor.style.shape,
        CursorShape::Hidden,
        "Cursor should be visible again after DECSET ?25h"
    );
    assert_eq!(
        snap_shown.cursor.style.shape, initial_shape,
        "Cursor should revert to its initial non-hidden shape"
    );
}
