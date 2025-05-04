// src/term/screen/tests.rs

// --- Unit Tests ---
use super::*; // Import functions from screen module
use crate::term::{Term, Cursor, DecModes, ParserState}; // Import Term struct and others
use crate::glyph::{Glyph, AttrFlags, Color}; // Import glyph types

// --- Test Helpers ---
impl Term {
    /// Helper to get screen content as string for a specific line (for testing).
    fn screen_to_string_line(&self, y: usize) -> String {
        let current_screen = if self.using_alt_screen { &self.alt_screen } else { &self.screen };
        current_screen.get(y)
            .map(|row| row.iter().map(|g| g.c).collect())
            .unwrap_or_default()
    }

    /// Helper to get screen content as a Vec of Strings (for testing).
    fn screen_to_string_vec(&self) -> Vec<String> {
         let current_screen = if self.using_alt_screen { &self.alt_screen } else { &self.screen };
         current_screen.iter()
             .map(|row| row.iter().map(|g| g.c).collect())
             .collect()
    }

    /// Helper to get a specific glyph (for testing attribute checks).
    fn get_glyph_test(&self, x: usize, y: usize) -> Glyph {
        self.get_glyph(x, y).cloned().unwrap_or_default()
    }
}

/// Helper to create a Term instance and process initial bytes.
fn term_with_bytes(w: usize, h: usize, bytes: &[u8]) -> Term {
    let mut term = Term::new(w, h);
    term.process_bytes(bytes); // Use the public method on Term
    term
}

/// Helper to check common post-conditions after processing a sequence.
fn assert_clean_state(term: &Term, expected_cursor: Cursor, message: &str) {
     assert_eq!(term.cursor, expected_cursor, "{}: Unexpected cursor position", message);
     assert_eq!(term.parser_state, ParserState::Ground, "{}: Parser should return to Ground state", message);
     // Check that the cell AT the final cursor position hasn't been altered unexpectedly
     // unless the test explicitly wrote there. This assumes tests start with a blank screen
     // or known state. For cursor movement tests, the destination cell should usually be a default space.
     // Note: This might need adjustment depending on the specific test's actions.
     if term.get_glyph_test(expected_cursor.x, expected_cursor.y).c == ' ' {
         // Ensure the default glyph check also considers attributes.
         assert_eq!(term.get_glyph_test(expected_cursor.x, expected_cursor.y), Glyph::default(),
             "{}: Cell at final cursor pos should be default glyph (char and attrs)", message);
     }
}


// --- Original Tests (From initial file upload) ---

#[cfg(test)]
mod original_tests {
    use super::*;

    // --- Scrolling Region and DECOM Tests ---

    #[test]
    fn test_set_scrolling_region_valid() {
        let mut term = Term::new(10, 5); // Height 5 -> rows 0..4
        set_scrolling_region(&mut term, 2, 4); // Set region rows 1-3 (0-based indices)
        assert_eq!(term.scroll_top, 1, "Scroll top should be 1");
        assert_eq!(term.scroll_bot, 3, "Scroll bottom should be 3");
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should move to absolute home");
    }

     #[test]
    fn test_set_scrolling_region_invalid_reset() {
        let mut term = Term::new(10, 5);
        // Case 1: top >= bottom
        set_scrolling_region(&mut term, 4, 2);
        assert_eq!(term.scroll_top, 0, "Invalid region (t>=b) should reset top");
        assert_eq!(term.scroll_bot, 4, "Invalid region (t>=b) should reset bottom");
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should move home after reset");

        // Case 2: bottom >= height
        set_scrolling_region(&mut term, 2, 6); // bottom=6 > height=5
        assert_eq!(term.scroll_top, 0, "Invalid region (b>=h) should reset top");
        assert_eq!(term.scroll_bot, 4, "Invalid region (b>=h) should reset bottom");

        // Case 3: Omitted/zero params (interpreted as 1 by parser, but 0 here resets)
        set_scrolling_region(&mut term, 0, 0); // Explicitly test 0
        assert_eq!(term.scroll_top, 0, "Zero params should reset top");
        assert_eq!(term.scroll_bot, 4, "Zero params should reset bottom");
         set_scrolling_region(&mut term, 1, 0); // Explicitly test 0
        assert_eq!(term.scroll_top, 0, "Zero bottom param should reset top");
        assert_eq!(term.scroll_bot, 4, "Zero bottom param should reset bottom");
    }

    #[test]
    fn test_scroll_up_in_region() {
        let mut term = Term::new(5, 5);
        term.process_bytes(b"11111\n22222\n33333\n44444\n55555"); // Populate screen
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3 (content: 222, 333, 444)
        set_cursor_pos_absolute(&mut term, 0, 3); // Place cursor inside region for context

        scroll_up(&mut term, 1); // Scroll region up by 1 line

        let expected = vec![
            "11111", // Outside region, unaffected
            "33333", // Row 1 (was 222) scrolled up, now contains row 2 content
            "44444", // Row 2 (was 333) scrolled up, now contains row 3 content
            "     ", // Row 3 (was 444) scrolled up, now blank
            "55555", // Outside region, unaffected
        ];
        assert_eq!(term.screen_to_string_vec(), expected);
    }

     #[test]
    fn test_scroll_down_in_region() {
        let mut term = Term::new(5, 5);
        term.process_bytes(b"11111\n22222\n33333\n44444\n55555");
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3 (content: 222, 333, 444)
        set_cursor_pos_absolute(&mut term, 0, 1); // Place cursor inside region for context

        scroll_down(&mut term, 1); // Scroll region down by 1 line

        let expected = vec![
            "11111", // Outside region, unaffected
            "     ", // Row 1 (was 222) scrolled down, now blank
            "22222", // Row 2 (was 333) scrolled down, now contains row 1 content
            "33333", // Row 3 (was 444) scrolled down, now contains row 2 content
            "55555", // Outside region, unaffected
        ];
        assert_eq!(term.screen_to_string_vec(), expected);
    }

    #[test]
    fn test_index_scrolls_at_bottom_margin() {
        let mut term = Term::new(5, 4);
        term.process_bytes(b"111\n222\n333\n444");
        set_scrolling_region(&mut term, 1, 3); // Region rows 0-2 (content: 111, 222, 333)
        set_cursor_pos_absolute(&mut term, 0, 2); // Cursor on line 2 (333), which is scroll_bot

        index(&mut term); // Should scroll region up

        let expected = vec![
            "222  ", // Row 0 scrolled up
            "333  ", // Row 1 scrolled up
            "     ", // Row 2 is new blank line
            "444  ", // Outside region, unaffected
        ];
        assert_eq!(term.screen_to_string_vec(), expected);
        assert_eq!(term.cursor.y, 2, "Cursor should remain on the bottom margin line after scroll");
    }

    #[test]
    fn test_reverse_index_scrolls_at_top_margin() {
        let mut term = Term::new(5, 4);
        term.process_bytes(b"111\n222\n333\n444");
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3 (content: 222, 333, 444)
        set_cursor_pos_absolute(&mut term, 0, 1); // Cursor on line 1 (222), which is scroll_top

        reverse_index(&mut term); // Should scroll region down

        let expected = vec![
            "111  ", // Outside region, unaffected
            "     ", // Row 1 is new blank line
            "222  ", // Row 2 scrolled down
            "333  ", // Row 3 scrolled down
        ];
         assert_eq!(term.screen_to_string_vec(), expected);
        assert_eq!(term.cursor.y, 1, "Cursor should remain on the top margin line after scroll");
    }

    #[test]
    fn test_origin_mode_enable_disable() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3
        set_cursor_pos_absolute(&mut term, 5, 2); // Place cursor somewhere

        // Enable origin mode
        enable_origin_mode(&mut term);
        assert!(term.dec_modes.origin_mode, "Origin mode should be enabled");
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 }, "Cursor should move to top-left of region (0, scroll_top)");

        // Disable origin mode
        set_cursor_pos_absolute(&mut term, 3, 3); // Move cursor within region again
        disable_origin_mode(&mut term);
        assert!(!term.dec_modes.origin_mode, "Origin mode should be disabled");
        assert_eq!(term.cursor, Cursor { x: 3, y: 3 }, "Cursor position should not change when disabling origin mode");
    }

    #[test]
    #[allow(deprecated)] // Allow use of deprecated set_origin_mode for testing
    fn test_deprecated_set_origin_mode() {
         let mut term = Term::new(10, 5);
         set_scrolling_region(&mut term, 2, 4); // Region 1-3

         set_origin_mode(&mut term, true);
         assert!(term.dec_modes.origin_mode, "Deprecated true failed");
         assert_eq!(term.cursor, Cursor { x: 0, y: 1 }, "Deprecated true cursor move failed");

         set_origin_mode(&mut term, false);
         assert!(!term.dec_modes.origin_mode, "Deprecated false failed");
         assert_eq!(term.cursor, Cursor { x: 0, y: 1 }, "Deprecated false cursor pos failed"); // Pos doesn't change on disable
    }


    #[test]
    fn test_origin_mode_cursor_addressing_cup() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3
        enable_origin_mode(&mut term); // Origin mode on, cursor at (0, 1)

        // CUP 2;3 -> 1-based relative y=2, x=3 -> absolute y = 1+(2-1)=2, x=(3-1)=2
        set_cursor_pos(&mut term, 3, 2);
        assert_eq!(term.cursor, Cursor { x: 2, y: 2 }, "CUP relative failed");

        // CUP 1;1 -> 1-based relative y=1, x=1 -> absolute y = 1+(1-1)=1, x=(1-1)=0
        set_cursor_pos(&mut term, 1, 1);
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 }, "CUP relative home failed");

        // Try CUP outside region (relative) - should clamp
        // CUP 5;1 -> 1-based relative y=5 -> absolute y=1+(5-1)=5 -> clamped to scroll_bot=3
        set_cursor_pos(&mut term, 1, 5);
        assert_eq!(term.cursor, Cursor { x: 0, y: 3 }, "CUP relative clamp bottom failed");
         // CUP 1;15 -> 1-based relative x=15 -> clamped to width-1 = 9
        set_cursor_pos(&mut term, 15, 1);
        assert_eq!(term.cursor, Cursor { x: 9, y: 1 }, "CUP relative clamp right failed");


        // Disable origin mode
        disable_origin_mode(&mut term);
        assert!(!term.dec_modes.origin_mode);
        // Cursor position remains where it was clamped
        assert_eq!(term.cursor, Cursor { x: 9, y: 1 }, "Cursor pos after disable");

        // CUP 2;3 -> 1-based absolute y=2, x=3 -> absolute y=(2-1)=1, x=(3-1)=2
        set_cursor_pos(&mut term, 3, 2);
        assert_eq!(term.cursor, Cursor { x: 2, y: 1 }, "CUP absolute failed");
    }

     #[test]
    fn test_origin_mode_cursor_movement_relative() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3
        enable_origin_mode(&mut term); // Origin mode on, cursor at (0, 1)

        // Move down within region (CUD)
        move_cursor(&mut term, 0, 1); // dy=1
        assert_eq!(term.cursor, Cursor { x: 0, y: 2 }, "Move down failed");

        // Move down to bottom margin
        move_cursor(&mut term, 0, 1); // dy=1
        assert_eq!(term.cursor, Cursor { x: 0, y: 3 }, "Move to bottom margin failed");

        // Try move past bottom margin - should clamp
        move_cursor(&mut term, 0, 1); // dy=1
        assert_eq!(term.cursor, Cursor { x: 0, y: 3 }, "Move past bottom clamp failed");

        // Move up within region (CUU)
        move_cursor(&mut term, 0, -1); // dy=-1
        assert_eq!(term.cursor, Cursor { x: 0, y: 2 }, "Move up failed");

        // Move up to top margin
        move_cursor(&mut term, 0, -1); // dy=-1
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 }, "Move to top margin failed");

        // Try move past top margin - should clamp
        move_cursor(&mut term, 0, -1); // dy=-1
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 }, "Move past top clamp failed");

         // Move right (CUF)
         move_cursor(&mut term, 2, 0); // dx=2
         assert_eq!(term.cursor, Cursor { x: 2, y: 1 }, "Move right failed");

         // Move left (CUB)
         move_cursor(&mut term, -1, 0); // dx=-1
         assert_eq!(term.cursor, Cursor { x: 1, y: 1 }, "Move left failed");
    }

    #[test]
    fn test_insert_delete_lines_respect_region() {
        let mut term = Term::new(5, 5);
        term.process_bytes(b"111\n222\n333\n444\n555");
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3 (content: 222, 333, 444)
        set_cursor_pos_absolute(&mut term, 0, 2); // Cursor on line 2 (content: 333)

        // Insert Line (IL)
        insert_blank_lines(&mut term, 1);
        let expected_il = vec![
            "111  ", // Unaffected
            "222  ", // Unaffected
            "     ", // Inserted blank line at cursor row 2
            "333  ", // Original row 2 shifted down to row 3
            "555  ", // Unaffected (original row 3 '444' scrolled off bottom)
        ];
        assert_eq!(term.screen_to_string_vec(), expected_il, "IL failed");
        assert_eq!(term.cursor, Cursor { x: 0, y: 2 }, "Cursor pos after IL"); // Cursor doesn't move

        // Reset state for DL test
        term = Term::new(5, 5);
        term.process_bytes(b"111\n222\n333\n444\n555");
        set_scrolling_region(&mut term, 2, 4); // Region rows 1-3 (content: 222, 333, 444)
        set_cursor_pos_absolute(&mut term, 0, 2); // Cursor on line 2 (content: 333)

        // Delete Line (DL)
        delete_lines(&mut term, 1);
         let expected_dl = vec![
            "111  ", // Unaffected
            "222  ", // Unaffected
            "444  ", // Original row 3 shifted up to row 2 (deleted row 2)
            "     ", // Blank line scrolled in at bottom margin (row 3)
            "555  ", // Unaffected
        ];
        assert_eq!(term.screen_to_string_vec(), expected_dl, "DL failed");
        assert_eq!(term.cursor, Cursor { x: 0, y: 2 }, "Cursor pos after DL"); // Cursor doesn't move
    }


    // --- Basic Functionality Tests (Copied & Adjusted) ---

    #[test]
    fn test_printable_chars_basic() {
        let term = term_with_bytes(5, 1, b"abc");
        assert_eq!(term.cursor.x, 3);
        assert_eq!(term.cursor.y, 0);
        assert_eq!(term.screen_to_string_line(0), "abc  ");
    }

    #[test]
    fn test_line_wrap_basic() {
        // Assumes autowrap is implicitly on (needs DECAWM check later)
        let term = term_with_bytes(3, 2, b"abcde");
        // After 'abc', cursor is at x=3 (end of line). Wrap occurs.
        // 'd' goes to y=1, x=0. Cursor moves to x=1.
        // 'e' goes to y=1, x=1. Cursor moves to x=2.
        assert_eq!(term.cursor.x, 2, "Cursor X after wrap");
        assert_eq!(term.cursor.y, 1, "Cursor Y after wrap");
        assert_eq!(term.screen_to_string_line(0), "abc");
        assert_eq!(term.screen_to_string_line(1), "de ");
    }

    #[test]
    fn test_newline_calls_index_and_cr() {
        // Test newline scrolls correctly at bottom of screen (default region)
        let mut term = Term::new(5, 2);
        term.process_bytes(b"111\n222"); // Writes "111", newline, writes "222" -> cursor at (3, 1)
        set_cursor_pos_absolute(&mut term, 3, 1); // Ensure cursor is at (3,1) before newline

        newline(&mut term); // Should scroll line 0 off, add blank line, cursor moves to (0, 1)

        assert_eq!(term.screen_to_string_line(0), "222  ", "Line 0 after scroll");
        assert_eq!(term.screen_to_string_line(1), "     ", "Line 1 after scroll");
        assert_eq!(term.cursor, Cursor{ x: 0, y: 1}, "Cursor pos after newline"); // Cursor moves to start of line
    }

    #[test]
    fn test_carriage_return_moves_home() {
        let mut term = term_with_bytes(5, 1, b"abc"); // Cursor at (3, 0)
        carriage_return(&mut term);
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 0); // Y should not change
    }

    #[test]
    fn test_backspace_stops_at_zero() {
        let mut term = term_with_bytes(5, 1, b"a"); // Cursor at (1, 0)
        backspace(&mut term);
        assert_eq!(term.cursor.x, 0);
        backspace(&mut term); // Try backspace again at column 0
        assert_eq!(term.cursor.x, 0); // Should remain at 0
    }

    #[test]
    fn test_tab_stops() {
         let mut term = Term::new(20, 1); // Tabs at 0, 8, 16
         term.tabs = (0..20).map(|i| i % 8 == 0).collect();

         // From 0, tab to 8
         set_cursor_pos_absolute(&mut term, 0, 0);
         tab(&mut term);
         assert_eq!(term.cursor.x, 8);

         // From 5, tab to 8
         set_cursor_pos_absolute(&mut term, 5, 0);
         tab(&mut term);
         assert_eq!(term.cursor.x, 8);

         // From 8, tab to 16
         set_cursor_pos_absolute(&mut term, 8, 0);
         tab(&mut term);
         assert_eq!(term.cursor.x, 16);

         // From 17, tab to end (19)
         set_cursor_pos_absolute(&mut term, 17, 0);
         tab(&mut term);
         assert_eq!(term.cursor.x, 19); // Moves to last column

         // From last col (19), tab stays
          set_cursor_pos_absolute(&mut term, 19, 0);
          tab(&mut term);
          assert_eq!(term.cursor.x, 19);
    }

    // --- Erase Tests ---
    #[test]
    fn test_erase_line_variants() {
        let mut term = term_with_bytes(5, 1, b"abcde"); // Content "abcde"
        set_cursor_pos_absolute(&mut term, 2, 0); // Cursor at 'c' (index 2)

        // EL 0: Erase to end
        erase_line_to_end(&mut term);
        assert_eq!(term.screen_to_string_line(0), "ab   ");
        assert_eq!(term.cursor.x, 2, "Cursor pos after EL 0"); // Cursor doesn't move

        // Reset
        term = term_with_bytes(5, 1, b"abcde");
        set_cursor_pos_absolute(&mut term, 2, 0);

        // EL 1: Erase to start (inclusive)
        erase_line_to_start(&mut term);
        assert_eq!(term.screen_to_string_line(0), "   de");
        assert_eq!(term.cursor.x, 2, "Cursor pos after EL 1");

        // Reset
        term = term_with_bytes(5, 1, b"abcde");
        set_cursor_pos_absolute(&mut term, 2, 0);

        // EL 2: Erase whole line
        erase_whole_line(&mut term);
        assert_eq!(term.screen_to_string_line(0), "     ");
        assert_eq!(term.cursor.x, 2, "Cursor pos after EL 2");
    }

     #[test]
    fn test_erase_display_variants() {
        let mut term = Term::new(5, 3);
        term.process_bytes(b"111\n222\n333"); // Content "111", "222", "333"
        set_cursor_pos_absolute(&mut term, 1, 1); // Cursor at '2' on line 1 (index 1)

        // ED 0: Erase to end
        erase_display_to_end(&mut term);
        let expected_ed0 = vec![
            "111  ", // Unaffected
            "2    ", // Erased from cursor (index 1) onwards
            "     ", // Erased whole line below
        ];
        assert_eq!(term.screen_to_string_vec(), expected_ed0, "ED 0 failed");
        assert_eq!(term.cursor, Cursor { x: 1, y: 1 }, "Cursor pos after ED 0");

        // Reset
        term = Term::new(5, 3);
        term.process_bytes(b"111\n222\n333");
        set_cursor_pos_absolute(&mut term, 1, 1);

        // ED 1: Erase to start (inclusive)
        erase_display_to_start(&mut term);
         // FIX: Correct the expected output for ED 1
         // Erasing up to and including column 1 on line "222  " should yield "  2  "
         let expected_ed1 = vec![
            "     ", // Erased whole line above
            "  2  ", // Erased up to cursor (index 1 inclusive)
            "333  ", // Unaffected
        ];
        assert_eq!(term.screen_to_string_vec(), expected_ed1, "ED 1 failed");
        assert_eq!(term.cursor, Cursor { x: 1, y: 1 }, "Cursor pos after ED 1");

        // Reset
        term = Term::new(5, 3);
        term.process_bytes(b"111\n222\n333");
        set_cursor_pos_absolute(&mut term, 1, 1);

        // ED 2: Erase whole display
        erase_whole_display(&mut term);
         let expected_ed2 = vec![
            "     ",
            "     ",
            "     ",
        ];
        assert_eq!(term.screen_to_string_vec(), expected_ed2, "ED 2 failed");
        // ED 2 moves cursor home (respecting origin mode, which is off here)
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor pos after ED 2");
    }

     // --- Insert/Delete Chars Tests ---
    #[test]
    fn test_insert_chars() {
        let mut term = term_with_bytes(5, 1, b"abcde");
        set_cursor_pos_absolute(&mut term, 1, 0); // Cursor at 'b' (index 1)

        insert_blank_chars(&mut term, 2); // Insert 2 blanks
        assert_eq!(term.screen_to_string_line(0), "a  bc"); // 'de' shifted off
        assert_eq!(term.cursor.x, 1, "Cursor pos after ICH"); // Cursor doesn't move
    }

    #[test]
    fn test_delete_chars() {
        let mut term = term_with_bytes(5, 1, b"abcde");
        set_cursor_pos_absolute(&mut term, 1, 0); // Cursor at 'b' (index 1)

        delete_chars(&mut term, 2); // Delete 'b' and 'c'
        assert_eq!(term.screen_to_string_line(0), "ade  "); // 'de' shifted left, blanks fill end
        assert_eq!(term.cursor.x, 1, "Cursor pos after DCH"); // Cursor doesn't move
    }

    // --- Alt Screen Tests ---
    #[test]
    fn test_alt_screen_switch() {
        let mut term = Term::new(5, 2);
        term.process_bytes(b"main");
        set_cursor_pos_absolute(&mut term, 1, 1); // Cursor at (1, 1) on main screen
        term.scroll_top = 1; // Set a non-default scroll region
        enable_origin_mode(&mut term); // Enable origin mode on main screen, cursor moves to (0, 1)

        let saved_main_cursor = term.cursor; // Should be (0, 1)
        let saved_main_origin_mode = term.dec_modes.origin_mode; // Should be true
        let saved_main_scroll_top = term.scroll_top; // Should be 1

        // Enter Alt Screen
        enter_alt_screen(&mut term);
        assert!(term.using_alt_screen, "Should be using alt screen");
        assert_eq!(term.screen_to_string_vec(), vec!["     ", "     "], "Alt screen should be cleared");
        // Cursor moves home on alt screen (respecting origin mode, which might be preserved or reset)
        // Let's assume origin mode is NOT preserved across screens by default
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should be home on alt screen");
        // assert_eq!(term.dec_modes.origin_mode, false, "Origin mode should reset on alt screen?"); // Terminal dependent - let's assume it doesn't reset for now
         assert_eq!(term.scroll_top, saved_main_scroll_top, "Scroll top should be preserved on alt screen"); // Scroll region is often preserved

        // Write something to alt screen
        term.process_bytes(b"alt");
        assert_eq!(term.screen_to_string_line(0), "alt  ");

        // Exit Alt Screen
        exit_alt_screen(&mut term);
        assert!(!term.using_alt_screen, "Should be back on main screen");
        assert_eq!(term.screen_to_string_line(0), "main ", "Main screen content should be restored");
        assert_eq!(term.cursor, saved_main_cursor, "Main screen cursor should be restored");
        assert_eq!(term.dec_modes.origin_mode, saved_main_origin_mode, "Main screen origin mode should be restored");
        assert_eq!(term.scroll_top, saved_main_scroll_top, "Main screen scroll top should be restored");
    }

    // --- Reset Test ---
    #[test]
    fn test_reset() {
         let mut term = Term::new(5, 2);
         term.process_bytes(b"abc\ndef");
         term.current_attributes.flags = AttrFlags::BOLD;
         set_scrolling_region(&mut term, 1, 1); // Set region row 0 only
         enable_origin_mode(&mut term); // Enable origin mode
         term.saved_cursor = Cursor { x: 1, y: 1 }; // Set saved cursor

         reset(&mut term);

         assert_eq!(term.width, 5);
         assert_eq!(term.height, 2);
         assert_eq!(term.screen_to_string_vec(), vec!["     ", "     "], "Screen should be cleared");
         assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should be home");
         assert_eq!(term.current_attributes, term.default_attributes, "Attributes should be default");
         assert!(!term.dec_modes.origin_mode, "Origin mode should be off");
         assert_eq!(term.dec_modes, DecModes::default(), "DEC modes should be default");
         assert_eq!(term.scroll_top, 0, "Scroll top should be reset");
         assert_eq!(term.scroll_bot, 1, "Scroll bottom should be reset");
         assert_eq!(term.saved_cursor, Cursor { x: 0, y: 0 }, "Saved cursor should be reset");
         assert!(!term.using_alt_screen, "Should be on main screen");
    }
}


// --- Enhanced Tests for Control Sequence Robustness ---

#[cfg(test)]
mod control_sequences_robustness {
    use super::*;

    // --- Cursor Down (CUD - CSI B) ---
    #[test_log::test]
    fn test_cud_variants() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"X"); // Cursor at (1, 0)

        // Default (1 line)
        term.process_bytes(b"\x1b[B");
        assert_clean_state(&term, Cursor { x: 1, y: 1 }, "CUD default");
        assert_eq!(term.screen_to_string_line(0), "X         ");
        assert_eq!(term.screen_to_string_line(1), "          ");

        // Explicit 1 line
        term.process_bytes(b"\x1b[1B");
        assert_clean_state(&term, Cursor { x: 1, y: 2 }, "CUD explicit 1");

        // Multiple lines
        term.process_bytes(b"\x1b[2B");
        assert_clean_state(&term, Cursor { x: 1, y: 4 }, "CUD explicit 2");

        // Zero lines (should do nothing)
        term.process_bytes(b"\x1b[0B");
        assert_clean_state(&term, Cursor { x: 1, y: 4 }, "CUD zero");

        // Clamp at bottom edge
        term.process_bytes(b"\x1b[10B"); // Try to move 10 down from y=4 (max y=4)
        assert_clean_state(&term, Cursor { x: 1, y: 4 }, "CUD clamp bottom");

        // Sequence followed by printable
        term.process_bytes(b"\x1b[H"); // Go home
        term.process_bytes(b"\x1b[B_"); // CUD then '_'
        assert_eq!(term.cursor, Cursor { x: 1, y: 1 }, "CUD printable cursor");
        assert_eq!(term.screen_to_string_line(1), "_         ", "CUD printable screen");
        assert_clean_state(&term, Cursor { x: 1, y: 1 }, "CUD printable state check");
    }

    // --- Cursor Up (CUU - CSI A) ---
     #[test_log::test]
    fn test_cuu_variants() {
        let mut term = Term::new(10, 5);
        term.cursor = Cursor { x: 5, y: 4 }; // Start at bottom

        // Default (1 line)
        term.process_bytes(b"\x1b[A");
        assert_clean_state(&term, Cursor { x: 5, y: 3 }, "CUU default");

        // Explicit 1 line
        term.process_bytes(b"\x1b[1A");
        assert_clean_state(&term, Cursor { x: 5, y: 2 }, "CUU explicit 1");

        // Multiple lines
        term.process_bytes(b"\x1b[2A");
        assert_clean_state(&term, Cursor { x: 5, y: 0 }, "CUU explicit 2");

        // Zero lines (should do nothing)
        term.process_bytes(b"\x1b[0A");
        assert_clean_state(&term, Cursor { x: 5, y: 0 }, "CUU zero");

        // Clamp at top edge
        term.process_bytes(b"\x1b[10A"); // Try to move 10 up from y=0
        assert_clean_state(&term, Cursor { x: 5, y: 0 }, "CUU clamp top");

        // Sequence followed by printable
        term.cursor = Cursor { x: 1, y: 1 };
        term.process_bytes(b"\x1b[A_"); // CUU then '_'
        assert_eq!(term.cursor, Cursor { x: 1, y: 0 }, "CUU printable cursor");
        assert_eq!(term.screen_to_string_line(0), "_         ", "CUU printable screen");
        assert_clean_state(&term, Cursor { x: 1, y: 0 }, "CUU printable state check");
    }

    // --- Cursor Forward (CUF - CSI C) ---
    #[test_log::test]
    fn test_cuf_variants() {
        let mut term = Term::new(10, 5); // Cursor at (0, 0)

        // Default (1 col)
        term.process_bytes(b"\x1b[C");
        assert_clean_state(&term, Cursor { x: 1, y: 0 }, "CUF default");

        // Explicit 1 col
        term.process_bytes(b"\x1b[1C");
        assert_clean_state(&term, Cursor { x: 2, y: 0 }, "CUF explicit 1");

        // Multiple cols
        term.process_bytes(b"\x1b[5C");
        assert_clean_state(&term, Cursor { x: 7, y: 0 }, "CUF explicit 5");

        // Zero cols (should do nothing)
        term.process_bytes(b"\x1b[0C");
        assert_clean_state(&term, Cursor { x: 7, y: 0 }, "CUF zero");

        // Clamp at right edge (max x = 9)
        term.process_bytes(b"\x1b[10C"); // Try to move 10 right from x=7
        assert_clean_state(&term, Cursor { x: 9, y: 0 }, "CUF clamp right");

        // Sequence followed by printable
        term.process_bytes(b"\x1b[H"); // Go home
        term.process_bytes(b"\x1b[C>"); // CUF then '>'
        assert_eq!(term.cursor, Cursor { x: 2, y: 0 }, "CUF printable cursor");
        assert_eq!(term.screen_to_string_line(0), " >        ", "CUF printable screen");
        assert_clean_state(&term, Cursor { x: 2, y: 0 }, "CUF printable state check");
    }

    // --- Cursor Back (CUB - CSI D) ---
     #[test_log::test]
    fn test_cub_variants() {
        let mut term = Term::new(10, 5);
        term.cursor = Cursor { x: 9, y: 0 }; // Start at right edge

        // Default (1 col)
        term.process_bytes(b"\x1b[D");
        assert_clean_state(&term, Cursor { x: 8, y: 0 }, "CUB default");

        // Explicit 1 col
        term.process_bytes(b"\x1b[1D");
        assert_clean_state(&term, Cursor { x: 7, y: 0 }, "CUB explicit 1");

        // Multiple cols
        term.process_bytes(b"\x1b[5D");
        assert_clean_state(&term, Cursor { x: 2, y: 0 }, "CUB explicit 5");

        // Zero cols (should do nothing)
        term.process_bytes(b"\x1b[0D");
        assert_clean_state(&term, Cursor { x: 2, y: 0 }, "CUB zero");

        // Clamp at left edge (min x = 0)
        term.process_bytes(b"\x1b[10D"); // Try to move 10 left from x=2
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "CUB clamp left");

        // Sequence followed by printable
        term.cursor = Cursor { x: 2, y: 0 };
        term.process_bytes(b"\x1b[D<"); // CUB then '<'
        assert_eq!(term.cursor, Cursor { x: 2, y: 0 }, "CUB printable cursor"); // Moves left to 1, then prints '<' at 1
        assert_eq!(term.screen_to_string_line(0), " <        ", "CUB printable screen");
        assert_clean_state(&term, Cursor { x: 2, y: 0 }, "CUB printable state check");
    }


    // --- Select Graphic Rendition (SGR - CSI m) ---
    #[test_log::test]
    fn test_sgr_robustness() {
        let mut term = Term::new(10, 5);
        let default_attrs = term.default_attributes;

        // Reset (0 or empty)
        term.current_attributes.flags = AttrFlags::BOLD;
        term.process_bytes(b"\x1b[m");
        assert_eq!(term.current_attributes, default_attrs, "SGR empty reset");
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "SGR empty reset state");

        term.current_attributes.flags = AttrFlags::BOLD;
        term.process_bytes(b"\x1b[0m");
        assert_eq!(term.current_attributes, default_attrs, "SGR 0 reset");
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "SGR 0 reset state");

        // Single attribute
        term.process_bytes(b"\x1b[1m"); // Bold
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD), "SGR bold");
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "SGR bold state");

        // Multiple attributes
        term.process_bytes(b"\x1b[4;31m"); // Underline, Red FG
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE), "SGR multi underline");
        assert!(!term.current_attributes.flags.contains(AttrFlags::BOLD), "SGR multi bold off"); // Bold should still be off from reset
        assert_eq!(term.current_attributes.fg, Color::Idx(1), "SGR multi red fg");
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "SGR multi state");

        // Reset specific attribute
        term.process_bytes(b"\x1b[24m"); // Underline off
        assert!(!term.current_attributes.flags.contains(AttrFlags::UNDERLINE), "SGR underline off");
        assert_eq!(term.current_attributes.fg, Color::Idx(1), "SGR red fg still on");
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "SGR underline off state");

        // Combined reset and set
        term.process_bytes(b"\x1b[0;1;32m"); // Reset, then Bold, Green FG
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD), "SGR reset+bold");
        assert!(!term.current_attributes.flags.contains(AttrFlags::UNDERLINE), "SGR reset+bold underline off");
        assert_eq!(term.current_attributes.fg, Color::Idx(2), "SGR reset+green fg");
        assert_eq!(term.current_attributes.bg, default_attrs.bg, "SGR reset+bg default");
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "SGR reset+bold+green state");

        // SGR followed by printable
        term.process_bytes(b"\x1b[mXY"); // Reset then XY
        assert_eq!(term.current_attributes, default_attrs, "SGR reset before XY");
        assert_eq!(term.screen_to_string_line(0), "XY        ", "SGR reset print XY");
        assert_eq!(term.get_glyph_test(0, 0).attr, default_attrs, "SGR reset attr X");
        assert_eq!(term.get_glyph_test(1, 0).attr, default_attrs, "SGR reset attr Y");
        assert_clean_state(&term, Cursor { x: 2, y: 0 }, "SGR reset print XY state");

        // Invalid param (should be ignored, processing continues)
        term.process_bytes(b"\x1b[1;999;4mZ"); // Bold, Invalid, Underline, then Z
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD), "SGR invalid skip bold");
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE), "SGR invalid skip underline");
        assert_eq!(term.screen_to_string_line(0), "XYZ       ", "SGR invalid skip print Z");
        assert!(term.get_glyph_test(2, 0).attr.flags.contains(AttrFlags::BOLD | AttrFlags::UNDERLINE), "SGR invalid skip attr Z");
        assert_clean_state(&term, Cursor { x: 3, y: 0 }, "SGR invalid skip state");

        // Too many params (should ignore excess) - MAX_CSI_PARAMS = 16
        let many_params = b"\x1b[1;3;4;7;9;1;3;4;7;9;1;3;4;7;9;1;31;32m"; // 17 params + m
        term.process_bytes(many_params);
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD), "SGR too many bold");
        // Check the *last* validly processed color before overflow/ignore
        // The behavior might vary: does it process up to 16, or stop earlier?
        // Assuming it processes up to 16 params, the 16th is '1' (bold again),
        // the 17th (31) and 18th (32) might be ignored or partially processed.
        // Let's assume it processes the 17th (31 - red) but ignores the 18th.
        assert_eq!(term.current_attributes.fg, Color::Idx(1), "SGR too many red fg"); // 31 should be processed
        assert_eq!(term.current_attributes.bg, default_attrs.bg, "SGR too many bg default");
        assert_clean_state(&term, Cursor { x: 3, y: 0 }, "SGR too many state");
    }

    // --- Malformed Sequences ---
    #[test_log::test]
    fn test_incomplete_csi() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b["); // Incomplete CSI
        assert_eq!(term.parser_state, ParserState::CSIEntry, "Parser should be in CSIEntry");
        term.process_bytes(b"1"); // Add param
        assert_eq!(term.parser_state, ParserState::CSIParam, "Parser should be in CSIParam");
        term.process_bytes(b"A"); // Final byte CUU
        assert_clean_state(&term, Cursor { x: 0, y: 0 }, "Incomplete CSI finished"); // Moves 1 up from 0,0 -> clamps at 0,0
    }

     #[test_log::test]
    fn test_csi_interrupted_by_escape() {
         let mut term = Term::new(10, 5);
         term.process_bytes(b"\x1b[1;"); // Incomplete CSI with param and separator
         assert_eq!(term.parser_state, ParserState::CSIParam, "Parser should be in CSIParam");
         term.process_bytes(b"\x1b"); // ESC received mid-sequence
         assert_eq!(term.parser_state, ParserState::Escape, "Parser should transition to Escape");
         term.process_bytes(b"c"); // RIS (Reset)
         assert_eq!(term.parser_state, ParserState::Ground, "Parser should be Ground after RIS");
         // Check a reset condition
         assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should be home after reset");
     }

    #[test_log::test]
    fn test_csi_invalid_final_byte() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[1;2\x07"); // CSI params followed by BEL (invalid final byte)
        // Behavior depends on strictness. Should ideally ignore and return to ground.
        assert_eq!(term.parser_state, ParserState::Ground, "Parser should return to Ground after invalid final byte");
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should not have moved");
        // Xterm might beep here due to the BEL, but our Term state shouldn't change unexpectedly.
    }

    #[test_log::test]
    fn test_csi_intermediate_byte() {
        let mut term = Term::new(10, 5);
        // Example: CSI ! p (Soft terminal reset - DECSOFT) - uses '!' intermediate
        term.process_bytes(b"\x1b[!p");
        // We don't have DECSOFT implemented, but the parser should handle the intermediate
        // and return to ground without printing '!'.
        assert_eq!(term.parser_state, ParserState::Ground, "Parser should handle intermediate and return to Ground");
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should not move for unknown sequence");
        assert_eq!(term.screen_to_string_line(0), "          ", "Screen should be empty");
    }

    #[test_log::test]
    fn test_csi_private_marker() {
        let mut term = Term::new(10, 5);
        // Example: CSI ? 25 l (Hide cursor - DECTCEM)
        term.process_bytes(b"\x1b[?25l");
        // Check if the mode was set correctly (assuming MODE_HIDE corresponds to DECTCEM)
        // This requires checking term.dec_modes or similar internal state if available,
        // or observing behavior via other means (like lack of cursor in visual tests).
        // For now, just check parser state and screen.
        assert_eq!(term.parser_state, ParserState::Ground, "Parser should handle private marker");
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor position unchanged");
        assert_eq!(term.screen_to_string_line(0), "          ", "Screen empty");
        // TODO: Add assertion for term.dec_modes.hide_cursor or equivalent if implemented
    }
}
