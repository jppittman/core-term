// src/term/screen.rs

//! Handles screen buffer manipulation, cursor movement, erasing, scrolling, etc.

use super::{Term, Cursor}; // Import Term and Cursor from parent module (mod.rs)
// FIX: Remove unused imports
use crate::glyph::Glyph;
use std::cmp::min;
use std::mem;

// --- Screen Access ---

/// Returns a mutable reference to the currently active screen buffer.
pub(super) fn current_screen_mut(term: &mut Term) -> &mut Vec<Vec<Glyph>> {
    if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen }
}

// --- Resizing ---

/// Resizes the terminal emulator state.
pub(super) fn resize(term: &mut Term, new_width: usize, new_height: usize) {
    if new_width == term.width && new_height == term.height {
        return;
    }

    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };

    // Helper closure to resize a screen buffer
    let resize_buffer = |buffer: &mut Vec<Vec<Glyph>>, old_height: usize| {
        buffer.resize_with(new_height, || vec![default_glyph; new_width]);
        for y in 0..min(old_height, new_height) {
            buffer[y].resize(new_width, default_glyph);
        }
    };

    let old_height = term.height;
    resize_buffer(&mut term.screen, old_height);
    resize_buffer(&mut term.alt_screen, old_height);

    term.width = new_width;
    term.height = new_height;
    // term.top = 0; // TODO: Reset scrolling region on resize?
    // term.bot = term.height - 1; // TODO: Reset scrolling region on resize?

    // Clamp cursor and saved cursor positions to new bounds
    term.cursor.x = min(term.cursor.x, term.width.saturating_sub(1));
    term.cursor.y = min(term.cursor.y, term.height.saturating_sub(1));
    term.saved_cursor.x = min(term.saved_cursor.x, term.width.saturating_sub(1));
    term.saved_cursor.y = min(term.saved_cursor.y, term.height.saturating_sub(1));
    term.saved_cursor_alt.x = min(term.saved_cursor_alt.x, term.width.saturating_sub(1));
    term.saved_cursor_alt.y = min(term.saved_cursor_alt.y, term.height.saturating_sub(1));

    // Recalculate tab stops
    term.tabs = (0..term.width).map(|i| i % super::DEFAULT_TAB_INTERVAL == 0).collect();
}

// --- Scrolling ---

/// Scrolls the region between `top` and `bot` up by `n` lines.
/// Lines shifted off the top are lost. Blank lines are inserted at the bottom.
pub(super) fn scroll_up(term: &mut Term, top: usize, bot: usize, n: usize) {
    let height = term.height;
    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };
    // Clamp region and scroll amount
    let top = min(top, height.saturating_sub(1));
    let bot = min(bot, height.saturating_sub(1));
    if top > bot { return; } // Invalid region

    let num_lines_in_region = bot - top + 1;
    let lines_to_scroll = min(n, num_lines_in_region);

    if lines_to_scroll == 0 {
        return;
    }

    let screen = current_screen_mut(term);

    // Rotate lines upwards within the slice screen[top..=bot]
    screen[top..=bot].rotate_left(lines_to_scroll);

    // Clear the newly exposed lines at the bottom of the region
    let clear_start = bot.saturating_sub(lines_to_scroll) + 1;
    for y in clear_start..=bot {
        // Ensure we don't index out of bounds if bot was already height-1
        if y < height {
            screen[y].fill(default_glyph);
        }
    }
}

/// Scrolls the region between `top` and `bot` down by `n` lines.
/// Lines shifted off the bottom are lost. Blank lines are inserted at the top.
pub(super) fn scroll_down(term: &mut Term, top: usize, bot: usize, n: usize) {
    let height = term.height;
    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };
    // Clamp region and scroll amount
    let top = min(top, height.saturating_sub(1));
    let bot = min(bot, height.saturating_sub(1));
    if top > bot { return; } // Invalid region

    let num_lines_in_region = bot - top + 1;
    let lines_to_scroll = min(n, num_lines_in_region);

    if lines_to_scroll == 0 {
        return;
    }

    let screen = current_screen_mut(term);

    // Rotate lines downwards within the slice screen[top..=bot]
    screen[top..=bot].rotate_right(lines_to_scroll);

    // Clear the newly exposed lines at the top of the region
    let clear_end = min(top + lines_to_scroll, height); // Avoid going past buffer end
    for y in top..clear_end {
         if y <= bot { // Ensure we only clear within the original region bounds
             screen[y].fill(default_glyph);
         }
    }
}

// --- Cursor Movement ---

/// Moves the cursor relative to its current position, clamping to bounds.
pub(super) fn move_cursor(term: &mut Term, dx: isize, dy: isize) {
    let width = term.width;
    let height = term.height;
    let new_x = (term.cursor.x as isize + dx).max(0) as usize;
    let new_y = (term.cursor.y as isize + dy).max(0) as usize;
    // TODO: Respect scrolling region origin mode (DECOM) if implemented
    term.cursor.x = min(new_x, width.saturating_sub(1));
    term.cursor.y = min(new_y, height.saturating_sub(1));
}

/// Sets the cursor to an absolute position (0-based), clamping to bounds.
pub(super) fn set_cursor_pos(term: &mut Term, x: usize, y: usize) {
    let width = term.width;
    let height = term.height;
    // TODO: Respect scrolling region origin mode (DECOM) if implemented
    term.cursor.x = min(x, width.saturating_sub(1));
    term.cursor.y = min(y, height.saturating_sub(1));
}

/// Handles Line Feed (LF). Moves cursor down, scrolls if at bottom.
pub(super) fn newline(term: &mut Term) {
    let y = term.cursor.y;
    let bot = term.height - 1; // TODO: Use term.bot when scrolling region is implemented
    if y == bot {
        scroll_up(term, 0, bot, 1); // TODO: Use term.top when scrolling region is implemented
    } else {
        term.cursor.y += 1;
    }
    term.cursor.x = 0;
}

/// Handles Carriage Return (CR). Moves cursor to the beginning of the line.
pub(super) fn carriage_return(term: &mut Term) {
    term.cursor.x = 0;
}

/// Handles Backspace (BS). Moves cursor left, stopping at the first column.
pub(super) fn backspace(term: &mut Term) {
    if term.cursor.x > 0 {
        term.cursor.x -= 1;
    }
}

/// Handles Horizontal Tab (HT). Moves cursor to the next tab stop.
pub(super) fn tab(term: &mut Term) {
    let current_x = term.cursor.x;
    let width = term.width;
    let tabs = &term.tabs;

    let next_tab_pos = tabs.iter()
        .skip(current_x + 1)
        .position(|&is_stop| is_stop);

    match next_tab_pos {
        Some(relative_pos) => {
            let absolute_pos = current_x + 1 + relative_pos;
            term.cursor.x = min(absolute_pos, width.saturating_sub(1));
        }
        None => {
            term.cursor.x = width.saturating_sub(1);
        }
    }
}

/// Saves the current cursor position (DECSC / SCOSC).
pub(super) fn save_cursor(term: &mut Term) {
    if term.using_alt_screen {
        term.saved_cursor_alt = term.cursor;
    } else {
        term.saved_cursor = term.cursor;
    }
}

/// Restores the saved cursor position (DECRC / SCORC).
pub(super) fn restore_cursor(term: &mut Term) {
    term.cursor = if term.using_alt_screen {
        term.saved_cursor_alt
    } else {
        term.saved_cursor
    };
    // Ensure restored cursor is within bounds (important after resize)
    term.cursor.x = min(term.cursor.x, term.width.saturating_sub(1));
    term.cursor.y = min(term.cursor.y, term.height.saturating_sub(1));
}

// --- Erasing Functions ---

/// Fills cells with spaces using current attributes.
fn fill_range(term: &mut Term, y: usize, x_start: usize, x_end: usize) {
    let height = term.height;
    let width = term.width;
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };

    if y >= height || x_start >= x_end || x_start >= width {
        return;
    }

    let screen = current_screen_mut(term);
    let clamped_end = min(x_end, width);
    for x in x_start..clamped_end {
        screen[y][x] = fill_glyph;
    }
}

/// Erases from cursor to end of line (EL 0).
pub(super) fn erase_line_to_end(term: &mut Term) {
    let y = term.cursor.y;
    let x = term.cursor.x;
    let width = term.width;
    fill_range(term, y, x, width);
}

/// Erases from start of line to cursor (inclusive) (EL 1).
pub(super) fn erase_line_to_start(term: &mut Term) {
    let y = term.cursor.y;
    let x = term.cursor.x;
    fill_range(term, y, 0, x + 1);
}

/// Erases the entire current line (EL 2).
pub(super) fn erase_whole_line(term: &mut Term) {
    let y = term.cursor.y;
    let width = term.width;
    fill_range(term, y, 0, width);
}

/// Erases from cursor to end of display (ED 0).
pub(super) fn erase_display_to_end(term: &mut Term) {
    let cursor_y = term.cursor.y;
    let height = term.height;
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };

    erase_line_to_end(term); // Erase rest of current line

    let screen = current_screen_mut(term);
    for y in (cursor_y + 1)..height { // Fill lines below
        screen[y].fill(fill_glyph);
    }
}

/// Erases from start of display to cursor (inclusive) (ED 1).
pub(super) fn erase_display_to_start(term: &mut Term) {
    let cursor_y = term.cursor.y;
    let height = term.height;
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };

    erase_line_to_start(term); // Erase start of current line

    let screen = current_screen_mut(term);
    for y in 0..cursor_y { // Fill lines above
         if y < height {
             screen[y].fill(fill_glyph);
         }
    }
}

/// Erases the entire display (ED 2).
pub(super) fn erase_whole_display(term: &mut Term) {
    let height = term.height;
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };

    let screen = current_screen_mut(term);
    for y in 0..height {
        screen[y].fill(fill_glyph);
    }
}

// --- Insert/Delete ---

/// Inserts `n` blank characters at the cursor position. (ICH)
pub(super) fn insert_blank_chars(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let x = term.cursor.x;
     let width = term.width;
     let num_to_insert = min(n, width.saturating_sub(x));

     if num_to_insert == 0 || y >= term.height {
         return;
     }

     let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };
     let screen = current_screen_mut(term);

     screen[y].copy_within(x..width - num_to_insert, x + num_to_insert);
     for i in 0..num_to_insert {
         screen[y][x + i] = fill_glyph;
     }
 }

/// Deletes `n` characters at the cursor position. (DCH)
pub(super) fn delete_chars(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let x = term.cursor.x;
     let width = term.width;
     let num_to_delete = min(n, width.saturating_sub(x));

     if num_to_delete == 0 || y >= term.height {
         return;
     }

     let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };
     let screen = current_screen_mut(term);

     screen[y].copy_within(x + num_to_delete.., x);
     let fill_start = width - num_to_delete;
     for i in fill_start..width {
         screen[y][i] = fill_glyph;
     }
 }

/// Inserts `n` blank lines at the cursor position. (IL)
pub(super) fn insert_blank_lines(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let top = 0; // TODO: Respect scrolling region
     let bot = term.height - 1; // TODO: Respect scrolling region

     if y < top || y > bot { return; }

     let num_to_insert = min(n, bot - y + 1);
     if num_to_insert > 0 {
         scroll_down(term, y, bot, num_to_insert);
     }
 }

/// Deletes `n` lines starting from the cursor position. (DL)
pub(super) fn delete_lines(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let top = 0; // TODO: Respect scrolling region
     let bot = term.height - 1; // TODO: Respect scrolling region

     if y < top || y > bot { return; }

     let num_to_delete = min(n, bot - y + 1);
     if num_to_delete > 0 {
         scroll_up(term, y, bot, num_to_delete);
     }
 }

// --- Character Handling ---

/// Handles a printable character decoded from the input stream.
pub(super) fn handle_printable(term: &mut Term, c: char) {
    // TODO: Implement proper Unicode width calculation
    let char_width = 1;
    let width = term.width;
    let height = term.height;
    let current_attributes = term.current_attributes;

    // TODO: Handle wrap mode properly (ATTR_WRAP flag?)
    if term.cursor.x + char_width > width {
         if term.cursor.x < width {
             newline(term);
         } else {
             newline(term);
         }
    }

    if term.cursor.y < height && term.cursor.x < width {
        let y = term.cursor.y;
        let x = term.cursor.x;
        let screen = current_screen_mut(term);
        screen[y][x] = Glyph { c, attr: current_attributes };

        if char_width > 0 {
            term.cursor.x += char_width;
        }
    }
}

// --- Alt Screen ---

/// Switches to the alternate screen buffer.
pub(super) fn enter_alt_screen(term: &mut Term) {
    if !term.using_alt_screen {
        save_cursor(term); // Save cursor on main screen
        mem::swap(&mut term.screen, &mut term.alt_screen);
        term.using_alt_screen = true;
        erase_whole_display(term); // Clear the alt screen
        set_cursor_pos(term, 0, 0); // Move cursor home
    }
}

/// Switches back to the main screen buffer.
pub(super) fn exit_alt_screen(term: &mut Term) {
    if term.using_alt_screen {
        mem::swap(&mut term.screen, &mut term.alt_screen);
        term.using_alt_screen = false;
        restore_cursor(term); // Restore cursor on main screen
    }
}

// --- Reset ---

/// Resets terminal state (called by RIS).
pub(super) fn reset(term: &mut Term) {
    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };
    // Clear the current screen
    let screen = current_screen_mut(term);
    for row in screen.iter_mut() { row.fill(default_glyph); }
    // Reset cursor, attributes, modes
    term.cursor = Cursor { x: 0, y: 0 };
    term.current_attributes = term.default_attributes;
    term.dec_modes = super::DecModes::default(); // Use super::DecModes
    // TODO: Reset other state like scrolling region, tabs?
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*; // Import functions from screen module
    use crate::term::Term; // Import Term struct
    // FIX: Remove unused imports
    // use crate::glyph::{Color, AttrFlags}; // Import glyph types

    // FIX: Move screen_to_string_line into an impl block for Term within the test module
    impl Term {
        /// Helper to get screen content as string for a specific line (for testing).
        fn screen_to_string_line(&self, y: usize) -> String {
            let current_screen = if self.using_alt_screen { &self.alt_screen } else { &self.screen };
            current_screen.get(y)
                .map(|row| row.iter().map(|g| g.c).collect())
                .unwrap_or_default()
        }
    }


    /// Helper to create a Term instance and process initial bytes.
    fn term_with_bytes(w: usize, h: usize, bytes: &[u8]) -> Term {
        let mut term = Term::new(w, h);
        term.process_bytes(bytes); // Use the public method on Term
        term
    }

    // --- Basic Functionality Tests ---
    #[test]
    fn test_printable_chars() {
        let term = term_with_bytes(5, 1, b"abc");
        assert_eq!(term.cursor.x, 3);
        assert_eq!(term.cursor.y, 0);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'b');
        assert_eq!(term.get_glyph(2, 0).unwrap().c, 'c');
        assert_eq!(term.get_glyph(3, 0).unwrap().c, ' '); // Default char
    }

    #[test]
    fn test_line_wrap() {
        let term = term_with_bytes(3, 2, b"abcde");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
        assert_eq!(term.screen_to_string_line(0), "abc");
        assert_eq!(term.screen_to_string_line(1), "de "); // Space from default glyph
    }

    #[test]
    fn test_newline() {
        let term = term_with_bytes(5, 2, b"ab\ncd");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
        assert_eq!(term.screen_to_string_line(0), "ab   ");
        assert_eq!(term.screen_to_string_line(1), "cd   ");
    }

    #[test]
    fn test_carriage_return() {
        let term = term_with_bytes(5, 1, b"abc\rde");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 0);
        assert_eq!(term.screen_to_string_line(0), "dec  "); // d overwrites a, e overwrites b
    }

    #[test]
    fn test_backspace() {
        let term = term_with_bytes(5, 1, b"abc\x08d");
        // Cursor moves from 3 to 2 after BS, then 'd' overwrites 'c' at pos 2, cursor moves to 3
        assert_eq!(term.cursor.x, 3);
        assert_eq!(term.screen_to_string_line(0), "abd  ");
    }

    // --- CSI Cursor Movement Tests ---
    #[test]
    fn test_csi_cursor_up() {
        let mut term = Term::new(10, 5);
        set_cursor_pos(&mut term, 3, 3);
        term.process_bytes(b"\x1b[A"); // CUU 1
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
    }

    #[test]
    fn test_csi_cursor_up_param() {
        let mut term = Term::new(10, 5);
        set_cursor_pos(&mut term, 3, 3);
        term.process_bytes(b"\x1b[2A"); // CUU 2
        assert_eq!(term.cursor, Cursor { x: 3, y: 1 });
    }

    #[test]
    fn test_csi_cursor_down() {
        let mut term = Term::new(10, 5);
        set_cursor_pos(&mut term, 3, 1);
        term.process_bytes(b"\x1b[B"); // CUD 1
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
        term.process_bytes(b"\x1b[0B"); // CUD 1 (0 defaults to 1)
        assert_eq!(term.cursor, Cursor { x: 3, y: 3 });
        term.process_bytes(b"\x1b[1B"); // CUD 1
        assert_eq!(term.cursor, Cursor { x: 3, y: 4 });
    }

    #[test]
    fn test_csi_cursor_forward() {
        let mut term = Term::new(10, 5);
        set_cursor_pos(&mut term, 1, 1);
        term.process_bytes(b"\x1b[C"); // CUF 1
        assert_eq!(term.cursor, Cursor { x: 2, y: 1 });
        term.process_bytes(b"\x1b[2C"); // CUF 2
        assert_eq!(term.cursor, Cursor { x: 4, y: 1 });
    }

    #[test]
    fn test_csi_cursor_backward() {
        let mut term = Term::new(10, 5);
        set_cursor_pos(&mut term, 3, 1);
        term.process_bytes(b"\x1b[D"); // CUB 1
        assert_eq!(term.cursor, Cursor { x: 2, y: 1 });
        term.process_bytes(b"\x1b[2D"); // CUB 2
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 });
        term.process_bytes(b"\x1b[D"); // CUB 1 (already at 0)
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 }); // Stays at 0
    }

    #[test]
    fn test_csi_cursor_position() {
        let term = term_with_bytes(10, 5, b"\x1b[3;4H"); // CUP row 3, col 4 -> (y=2, x=3)
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
        let term2 = term_with_bytes(10, 5, b"\x1b[H"); // CUP default (1;1) -> (y=0, x=0)
        assert_eq!(term2.cursor, Cursor { x: 0, y: 0 });
        let term3 = term_with_bytes(10, 5, b"\x1b[;5H"); // CUP default row 1, col 5 -> (y=0, x=4)
        assert_eq!(term3.cursor, Cursor { x: 4, y: 0 });
        let term4 = term_with_bytes(10, 5, b"\x1b[5;H"); // CUP row 5, default col 1 -> (y=4, x=0)
        assert_eq!(term4.cursor, Cursor { x: 0, y: 4 });
    }

    // --- CSI Erase Tests ---
    #[test]
    fn test_csi_erase_line_to_end() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        set_cursor_pos(&mut term, 2, 0); // Cursor at 'c'
        term.process_bytes(b"\x1b[K"); // EL 0
        assert_eq!(term.screen_to_string_line(0), "ab   ");
        assert_eq!(term.cursor.x, 2); // Cursor doesn't move
    }

    #[test]
    fn test_csi_erase_line_to_start() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        set_cursor_pos(&mut term, 2, 0); // Cursor at 'c'
        term.process_bytes(b"\x1b[1K"); // EL 1
        assert_eq!(term.screen_to_string_line(0), "   de"); // a, b, c erased
        assert_eq!(term.cursor.x, 2); // Cursor doesn't move
    }

    #[test]
    fn test_csi_erase_whole_line() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        set_cursor_pos(&mut term, 2, 0); // Cursor at 'c'
        term.process_bytes(b"\x1b[2K"); // EL 2
        assert_eq!(term.screen_to_string_line(0), "     "); // Whole line erased
        assert_eq!(term.cursor.x, 2); // Cursor doesn't move
    }

    // --- CSI Insert/Delete Tests ---
    #[test]
    fn test_csi_insert_blank_chars() {
        let mut term = term_with_bytes(5, 1, b"abcde");
        set_cursor_pos(&mut term, 1, 0); // Cursor at 'b'
        term.process_bytes(b"\x1b[2@"); // ICH 2
        assert_eq!(term.screen_to_string_line(0), "a  bc");
        assert_eq!(term.cursor.x, 1); // Cursor doesn't move
    }

    #[test]
    fn test_csi_delete_chars() {
        let mut term = term_with_bytes(5, 1, b"abcde");
        set_cursor_pos(&mut term, 1, 0); // Cursor at 'b'
        term.process_bytes(b"\x1b[2P"); // DCH 2
        assert_eq!(term.screen_to_string_line(0), "ade  ");
        assert_eq!(term.cursor.x, 1); // Cursor doesn't move
    }

    #[test]
    fn test_csi_insert_blank_lines() {
        let mut term = Term::new(5, 3);
        term.process_bytes(b"11111\n22222\n33333");
        set_cursor_pos(&mut term, 0, 1); // Cursor on line '22222'
        term.process_bytes(b"\x1b[L"); // IL 1
        assert_eq!(term.screen_to_string_line(0), "11111");
        assert_eq!(term.screen_to_string_line(1), "     ", "Line 1 should be blank after insert");
        assert_eq!(term.screen_to_string_line(2), "22222", "Line 2 should contain original line 1 content");
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 1);
    }

    #[test]
    fn test_csi_delete_lines() {
        let mut term = Term::new(5, 3);
        term.process_bytes(b"11111\n22222\n33333");
        set_cursor_pos(&mut term, 0, 1); // Cursor on line '22222'
        term.process_bytes(b"\x1b[M"); // DL 1
        assert_eq!(term.screen_to_string_line(0), "11111");
        assert_eq!(term.screen_to_string_line(1), "33333", "Line 1 should contain original line 2 content after delete");
        assert_eq!(term.screen_to_string_line(2), "     ", "Line 2 should be blank after delete");
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 1);
    }

    // --- Resize Test ---
    #[test]
    fn test_resize() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"123\n456"); // Cursor at (3, 1) after '6'
        set_cursor_pos(&mut term, 2, 1); // Manually set cursor to (2, 1) ('6')

        // Shrink
        resize(&mut term, 5, 3);
        assert_eq!(term.width, 5);
        assert_eq!(term.height, 3);
        assert_eq!(term.screen.len(), 3);
        assert_eq!(term.screen[0].len(), 5);
        assert_eq!(term.screen[1].len(), 5);
        assert_eq!(term.screen[2].len(), 5);
        assert_eq!(term.screen_to_string_line(0), "123  ");
        assert_eq!(term.screen_to_string_line(1), "456  "); // Content preserved
        assert_eq!(term.screen_to_string_line(2), "     "); // New line
        // Cursor clamped
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);

        // Grow
        resize(&mut term, 15, 7);
        assert_eq!(term.width, 15);
        assert_eq!(term.height, 7);
        assert_eq!(term.screen.len(), 7);
        assert_eq!(term.screen[0].len(), 15);
        assert_eq!(term.screen_to_string_line(0), "123            ");
        assert_eq!(term.screen_to_string_line(1), "456            ");
        assert_eq!(term.screen_to_string_line(2), "               "); // Old line 2 preserved (was blank)
        assert_eq!(term.screen_to_string_line(3), "               "); // New line 3
        // Cursor position unchanged
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
    }

    // Add more tests specific to screen manipulation here...
}
