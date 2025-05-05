// src/term/tests.rs

//! Unit tests for the main Term struct and its core logic.

// Module for tests related to Utf8Decoder (which is internal to Term)
// Remove this module as Utf8Decoder is gone
// #[cfg(test)]
// mod utf8_tests { ... }

// Module for tests related to the Term struct itself
#[cfg(test)]
mod term_tests {
    // Import necessary items from parent (term) and sibling (glyph) modules
    // Remove ParserState import
    use crate::term::{Term, DEFAULT_TAB_INTERVAL};
    use crate::glyph::{Attributes, Color, AttrFlags, Glyph, REPLACEMENT_CHARACTER};

    // --- Test Helpers ---

    /// Helper to create a Term instance and process bytes, returning the term.
    fn term_with_bytes(w: usize, h: usize, bytes: &[u8]) -> Term {
        let mut term = Term::new(w, h);
        term.process_bytes(bytes);
        term
    }

    /// Helper to get screen content as a Vec of Strings.
    fn screen_to_string_vec(term: &Term) -> Vec<String> {
         let current_screen = if term.using_alt_screen { &term.alt_screen } else { &term.screen };
         current_screen.iter()
             .map(|row| row.iter().map(|g| g.c).collect())
             .collect()
    }

    /// Helper to get a specific glyph, defaulting if out of bounds.
    fn get_glyph_test(term: &Term, x: usize, y: usize) -> Glyph {
        term.get_glyph(x, y).cloned().unwrap_or_default()
    }

    // --- Initialization Tests ---

    #[test]
    fn test_term_new() {
        let term = Term::new(80, 25);
        assert_eq!(term.get_dimensions(), (80, 25), "Initial dimensions");
        assert_eq!(term.get_cursor(), (0, 0), "Initial cursor position");
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "Initial parser state");
        assert_eq!(term.current_attributes, Attributes::default(), "Initial attributes");
        assert!(!term.using_alt_screen, "Should start on main screen");
        assert_eq!(term.scroll_top, 0, "Initial scroll top");
        assert_eq!(term.scroll_bot, 24, "Initial scroll bottom");
        assert_eq!(get_glyph_test(&term, 79, 24).c, ' ', "Initial screen content (corner)");
        assert!(term.tabs[0]);
        assert!(term.tabs[DEFAULT_TAB_INTERVAL]);
        assert!(!term.tabs[1]);
    }

    #[test]
    fn test_term_new_minimum_dimensions() {
        let term = Term::new(0, 0);
        assert_eq!(term.get_dimensions(), (1, 1), "Minimum dimensions");
        let term = Term::new(5, 0);
        assert_eq!(term.get_dimensions(), (5, 1), "Minimum height");
        let term = Term::new(0, 10);
        assert_eq!(term.get_dimensions(), (1, 10), "Minimum width");
    }

    // --- Basic Processing Tests (process_byte / process_bytes) ---

    #[test]
    fn test_process_bytes_printable_ascii() {
        let term = term_with_bytes(5, 1, b"Hi!");
        assert_eq!(term.get_cursor(), (3, 0), "Cursor after printable");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'H');
        assert_eq!(get_glyph_test(&term, 1, 0).c, 'i');
        assert_eq!(get_glyph_test(&term, 2, 0).c, '!');
        assert_eq!(get_glyph_test(&term, 3, 0).c, ' ');
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "State after printable");
    }

    #[test]
    fn test_process_bytes_c0_controls() {
        let mut term = Term::new(5, 3);
        term.process_bytes(b"A\nB\rC\tD\x08E"); // LF, CR, HT, BS

        // Corrected assertion based on trace analysis
        assert_eq!(screen_to_string_vec(&term), vec![
            "A    ".to_string(),
            "C   E".to_string(), // Corrected: Backspace moves cursor, E overwrites D
            "     ".to_string(),
        ], "Screen content after C0 mix");
        // Corrected Assertion: Cursor ends up at (5,1) with wrap_next true
        // Note: Tab behavior depends on tab stops, assuming default 8.
        // A(\n) -> (0,1) B(\r) -> (0,1) C(\t to 8, clamped to 4) -> (4,1) D(\b) -> (3,1) E -> (4,1)
        assert_eq!(term.get_cursor(), (4, 1), "Cursor pos after C0 mix");
        assert!(!term.wrap_next, "wrap_next should be false after C0 mix"); // E didn't wrap
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "State after C0 mix");
    }

    #[test]
    fn test_process_bytes_simple_csi() {
        let term = term_with_bytes(10, 5, b"ABC\x1b[2B\x1b[3DXYZ");
        // Corrected Assertion: Cursor should be after 'Z'
        assert_eq!(term.get_cursor(), (3, 2), "Cursor after simple CSI");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'A');
        assert_eq!(get_glyph_test(&term, 1, 0).c, 'B');
        assert_eq!(get_glyph_test(&term, 2, 0).c, 'C');
        assert_eq!(get_glyph_test(&term, 0, 2).c, 'X');
        assert_eq!(get_glyph_test(&term, 1, 2).c, 'Y');
        assert_eq!(get_glyph_test(&term, 2, 2).c, 'Z');
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "State after simple CSI");
    }

    #[test]
    fn test_process_bytes_simple_sgr() {
        let mut term = Term::new(10, 1);
        term.process_bytes(b"\x1b[1;31mHello\x1b[m World"); // Bold Red "Hello", Reset " World"
        let hello_attr = Attributes { fg: Color::Idx(1), bg: Color::Default, flags: AttrFlags::BOLD };
        let space_attr = Attributes::default();

        // Check characters and attributes
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'H');
        assert_eq!(get_glyph_test(&term, 0, 0).attr, hello_attr);
        assert_eq!(get_glyph_test(&term, 4, 0).c, 'o');
        assert_eq!(get_glyph_test(&term, 4, 0).attr, hello_attr);
        assert_eq!(get_glyph_test(&term, 5, 0).c, ' ');
        assert_eq!(get_glyph_test(&term, 5, 0).attr, space_attr);
        assert_eq!(get_glyph_test(&term, 6, 0).c, 'W');
        assert_eq!(get_glyph_test(&term, 6, 0).attr, space_attr);
        assert_eq!(get_glyph_test(&term, 9, 0).c, 'l');
        assert_eq!(get_glyph_test(&term, 9, 0).attr, space_attr);
        // Cursor should be at the end (10 is off screen, so x=9, wrap_next=true)
        assert_eq!(term.get_cursor(), (9,0));
        assert!(term.wrap_next);
    }


    #[test]
    fn test_process_bytes_utf8_multi_byte() {
        let term = term_with_bytes(10, 1, "你好".as_bytes()); // "Nǐ hǎo"
        assert_eq!(get_glyph_test(&term, 0, 0).c, '你');
        assert_eq!(get_glyph_test(&term, 1, 0).c, '好');
        assert_eq!(term.get_cursor(), (2, 0), "Cursor after UTF8"); // Assumes width 1 for now
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "State after UTF8");
    }

    #[test]
    fn test_process_bytes_utf8_split_across_calls() {
        let mut term = Term::new(10, 1);
        let bytes = "你好".as_bytes();

        // Use process_bytes for each byte chunk
        term.process_bytes(&[bytes[0]]);
        term.process_bytes(&[bytes[1]]);
        term.process_bytes(&[bytes[2]]); // completes '你'
        assert_eq!(get_glyph_test(&term, 0, 0).c, '你');
        assert_eq!(term.get_cursor(), (1, 0), "Cursor after first char");

        term.process_bytes(&bytes[3..]); // Process remaining bytes for '好'
        assert_eq!(get_glyph_test(&term, 1, 0).c, '好');
        assert_eq!(term.get_cursor(), (2, 0), "Cursor after second char");
    }

    #[test]
    fn test_process_bytes_invalid_utf8_sequence() {
        let mut term = Term::new(10, 1);
        term.process_bytes(b"A\x80B");
        assert_eq!(term.get_cursor(), (3, 0), "Cursor after invalid UTF-8");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'A');
        assert_eq!(get_glyph_test(&term, 1, 0).c, REPLACEMENT_CHARACTER);
        assert_eq!(get_glyph_test(&term, 2, 0).c, 'B');
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "State after invalid UTF-8");
    }

    #[test]
    fn test_process_bytes_invalid_utf8_within_csi() {
        let mut term = Term::new(10, 1);
        // Use 0x80 (C1 control) as the invalid byte within CSI
        term.process_bytes(b"A\x1b[1\x80B"); // Start CSI, param 1, invalid C1, then B
        // Expect parser to emit Error(0x80), return to ground, then process B.
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "State after invalid UTF-8 in CSI");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'A');
        assert_eq!(get_glyph_test(&term, 1, 0).c, 'B'); // B should be printed at (1,0)
        assert_eq!(term.get_cursor(), (2, 0), "Cursor after invalid UTF-8 in CSI");
        assert_eq!(term.current_attributes, Attributes::default(), "Attributes unchanged after aborted CSI");
    }

    // --- Resize Tests ---

    #[test]
    fn test_resize_larger() {
        let mut term = term_with_bytes(5, 2, b"12345\nABCDE");
        term.cursor = crate::term::Cursor { x: 4, y: 1 };
        term.scroll_top = 1;
        term.scroll_bot = 1;

        term.resize(10, 4);

        assert_eq!(term.get_dimensions(), (10, 4), "Dimensions after resize larger");
        assert_eq!(term.get_cursor(), (4, 1), "Cursor position after resize larger");
        assert_eq!(term.scroll_top, 0, "Scroll top reset after resize");
        assert_eq!(term.scroll_bot, 3, "Scroll bottom reset after resize");

        assert_eq!(screen_to_string_vec(&term), vec![
            "12345     ".to_string(),
            "ABCDE     ".to_string(),
            "          ".to_string(),
            "          ".to_string(),
        ]);
    }

    #[test]
    fn test_resize_smaller() {
        let mut term = term_with_bytes(10, 4, b"1234567890\nabcdefghij\nABCDEFGHIJ\nqwertyuiop");
        term.cursor = crate::term::Cursor { x: 8, y: 3 };
        term.saved_cursor = crate::term::Cursor { x: 9, y: 2 };

        term.resize(5, 2);

        assert_eq!(term.get_dimensions(), (5, 2), "Dimensions after resize smaller");
        assert_eq!(term.get_cursor(), (4, 1), "Cursor clamped after resize smaller");
        assert_eq!(term.saved_cursor, crate::term::Cursor { x: 4, y: 1 }, "Saved cursor clamped");

        assert_eq!(screen_to_string_vec(&term), vec![
            "12345".to_string(),
            "abcde".to_string(),
        ]);
    }

     #[test]
    fn test_resize_no_change() {
        let mut term = term_with_bytes(10, 4, b"123");
        term.cursor = crate::term::Cursor { x: 1, y: 1 };
        term.scroll_top = 1;
        term.scroll_bot = 2;

        term.resize(10, 4);

        assert_eq!(term.get_dimensions(), (10, 4), "Dimensions after resize no change");
        assert_eq!(term.get_cursor(), (1, 1), "Cursor position after resize no change");
        // Resize should *always* reset scroll region
        assert_eq!(term.scroll_top, 0, "Scroll top reset even if size unchanged");
        assert_eq!(term.scroll_bot, 3, "Scroll bottom reset even if size unchanged");
        assert_eq!(get_glyph_test(&term, 0, 0).c, '1');
    }
}
