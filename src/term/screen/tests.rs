// src/term/screen/tests.rs

//! Tests for screen manipulation logic in screen.rs

#[cfg(test)]
mod basic_tests {
    // Use super to access items from the parent module (screen.rs)
    use crate::term::screen::*;
    // Import necessary items from sibling modules or parent crate root
    // Remove ParserState import
    use crate::term::{Term, Cursor};
    use crate::glyph::{Glyph, Attributes};

    // Helper to get screen content as a Vec of Strings for assertion checking.
    fn screen_to_string_vec(term: &Term) -> Vec<String> {
        let current_screen = if term.using_alt_screen { &term.alt_screen } else { &term.screen };
        current_screen.iter()
            .map(|row| row.iter().map(|g| g.c).collect())
            .collect()
    }

    // Helper to get a specific glyph, defaulting if out of bounds.
    // Useful for checking specific cell content after operations.
    fn get_glyph_test(term: &Term, x: usize, y: usize) -> Glyph {
        term.get_glyph(x, y).cloned().unwrap_or_default()
    }

    // --- Original Tests (Adapted) ---

    #[test]
    fn test_backspace_stops_at_zero() {
        let mut term = Term::new(10, 1);
        term.cursor = Cursor { x: 5, y: 0 };
        backspace(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 4);
        term.cursor.x = 0;
        backspace(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 0);
    }

    #[test]
    fn test_carriage_return_moves_home() {
        let mut term = Term::new(10, 1);
        term.cursor = Cursor { x: 5, y: 0 };
        carriage_return(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 0);
    }

     #[test]
    fn test_newline_calls_index_and_cr() {
        let mut term = Term::new(5, 3);
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 3, y: 1 };

        newline(&mut term); // Use function from parent module

        assert_eq!(term.cursor.x, 0, "Cursor X after newline");
        assert_eq!(term.cursor.y, 2, "Cursor Y after newline");

        // Test newline at bottom margin (should scroll)
        term.cursor = Cursor { x: 3, y: 2 };
        newline(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 0, "Cursor X after scroll");
        assert_eq!(term.cursor.y, 2, "Cursor Y after scroll");
        assert_eq!(screen_to_string_vec(&term), vec![
            "22222".to_string(), // Line 0 scrolled up
            "33333".to_string(), // Line 1 scrolled up
            "     ".to_string(), // New blank line
        ], "Screen content after scroll");
    }


    #[test]
    fn test_index_scrolls_at_bottom_margin() {
        let mut term = Term::new(5, 4);
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[3] = "44444".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        set_scrolling_region(&mut term, 2, 3); // Use function from parent module
        term.cursor = Cursor { x: 0, y: 2 }; // Cursor on line 2 (333), which is scroll_bot

        index(&mut term); // Use function from parent module

        assert_eq!(term.cursor.y, 2, "Cursor Y remains at bottom after scroll");
        assert_eq!(screen_to_string_vec(&term), vec![
            "11111".to_string(), // Unaffected
            "33333".to_string(), // Scrolled up from line 2
            "     ".to_string(), // New blank line at bottom of region
            "44444".to_string(), // Unaffected
        ]);
    }

    #[test]
    fn test_reverse_index_scrolls_at_top_margin() {
        let mut term = Term::new(5, 4);
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[3] = "44444".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        set_scrolling_region(&mut term, 2, 3); // Use function from parent module
        term.cursor = Cursor { x: 0, y: 1 }; // Cursor on line 1 (222), which is scroll_top

        reverse_index(&mut term); // Use function from parent module

        assert_eq!(term.cursor.y, 1, "Cursor Y remains at top after scroll");
         assert_eq!(screen_to_string_vec(&term), vec![
            "11111".to_string(), // Unaffected
            "     ".to_string(), // New blank line at top of region
            "22222".to_string(), // Scrolled down from line 1
            "44444".to_string(), // Unaffected
        ]);
    }

    #[test]
    fn test_origin_mode_cursor_movement_relative() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Use function from parent module
        enable_origin_mode(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 1);

        index(&mut term); // Use function from parent module
        assert_eq!(term.cursor.y, 2, "Index within region");
        index(&mut term); // Use function from parent module
        assert_eq!(term.cursor.y, 3, "Index at region bottom");
        index(&mut term); // Use function from parent module
        assert_eq!(term.cursor.y, 3, "Index scrolls at region bottom");
        reverse_index(&mut term); // Use function from parent module
        assert_eq!(term.cursor.y, 2, "Reverse Index within region");
        reverse_index(&mut term); // Use function from parent module
        assert_eq!(term.cursor.y, 1, "Reverse Index at region top");
        reverse_index(&mut term); // Use function from parent module
        assert_eq!(term.cursor.y, 1, "Reverse Index scrolls at region top");
    }

     #[test]
    fn test_origin_mode_cursor_addressing_cup() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Use function from parent module
        term.cursor = Cursor{ x: 5, y: 2 };

        enable_origin_mode(&mut term); // Use function from parent module

        set_cursor_pos(&mut term, 1, 1); // Use function from parent module
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 1, "CUP 1,1 relative");

        set_cursor_pos(&mut term, 3, 3); // Use function from parent module
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 3, "CUP 3,3 relative");

        set_cursor_pos(&mut term, 1, 10); // Use function from parent module
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 3, "CUP 1,10 relative clamped");

        disable_origin_mode(&mut term); // Use function from parent module

        set_cursor_pos(&mut term, 2, 3); // Use function from parent module
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.cursor.y, 2, "CUP 2,3 absolute");
    }

    #[test]
    fn test_origin_mode_enable_disable() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Use function from parent module
        term.cursor = Cursor { x: 5, y: 2 };

        enable_origin_mode(&mut term); // Use function from parent module
        assert!(term.dec_modes.origin_mode);
        assert_eq!(term.cursor.x, 0, "Cursor X after enable");
        assert_eq!(term.cursor.y, 1, "Cursor Y after enable (region top)");

        term.cursor = Cursor { x: 3, y: 3 }; // Move cursor within region again

        disable_origin_mode(&mut term); // Use function from parent module
        assert!(!term.dec_modes.origin_mode);
        assert_eq!(term.cursor.x, 0, "Cursor X after disable");
        assert_eq!(term.cursor.y, 0, "Cursor Y after disable");
    }

    #[test]
    fn test_deprecated_set_origin_mode() {
        // Test integration via process_bytes
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Use function from parent module
        term.process_bytes(b"\x1b[?6h"); // Enable origin mode
        assert!(term.dec_modes.origin_mode, "Origin mode enabled via CSI");
        assert_eq!(term.cursor.y, 1, "Cursor Y after enable via CSI");

        term.process_bytes(b"\x1b[?6l"); // Disable origin mode
        assert!(!term.dec_modes.origin_mode, "Origin mode disabled via CSI");
        assert_eq!(term.cursor.y, 0, "Cursor Y after disable via CSI");
    }


    #[test]
    fn test_tab_stops() {
        let mut term = Term::new(20, 1);
        assert!(term.tabs[0]);
        assert!(term.tabs[8]);
        assert!(term.tabs[16]);
        assert!(!term.tabs[1]);
        assert!(!term.tabs[9]);

        term.cursor.x = 0;
        tab(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 8, "Tab from 0");

        term.cursor.x = 5;
        tab(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 8, "Tab from 5");

        term.cursor.x = 8;
        tab(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 16, "Tab from 8");

        term.cursor.x = 17;
        tab(&mut term); // Use function from parent module
        assert_eq!(term.cursor.x, 19, "Tab from 17 (to end)");
    }

    #[test]
    fn test_insert_chars() {
        let mut term = Term::new(10,1);
        term.process_bytes(b"abcdefghij");
        term.cursor = Cursor { x: 2, y: 0 }; // Cursor at 'c' (index 2)

        insert_blank_chars(&mut term, 3); // Use function from parent module

        assert_eq!(screen_to_string_vec(&term), vec!["ab   cdefg".to_string()]);
        assert_eq!(term.cursor.x, 2, "Cursor unchanged after ICH");
    }

    #[test]
    fn test_delete_chars() {
        let mut term = Term::new(10,1);
        term.process_bytes(b"abcdefghij");
        term.cursor = Cursor { x: 2, y: 0 }; // Cursor at 'c' (index 2)

        delete_chars(&mut term, 3); // Use function from parent module

        assert_eq!(screen_to_string_vec(&term), vec!["abfghij   ".to_string()]);
        assert_eq!(term.cursor.x, 2, "Cursor unchanged after DCH");
    }

    #[test]
    fn test_insert_delete_lines_respect_region() {
        let mut term = Term::new(5, 5);
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[3] = "44444".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[4] = "55555".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        set_scrolling_region(&mut term, 2, 4); // Use function from parent module (sets region 1-3 0-based)
        term.cursor = Cursor { x: 0, y: 1 }; // Cursor on line 1 (content: 222) - Top of region

        insert_blank_lines(&mut term, 1); // Use function from parent module
        assert_eq!(screen_to_string_vec(&term), vec![
            "11111".to_string(), // Unaffected
            "     ".to_string(), // Inserted blank line at cursor
            "22222".to_string(), // Shifted down
            "33333".to_string(), // Shifted down
            "55555".to_string(), // Unaffected (below region)
        ], "IL failed");
        assert_eq!(term.cursor.x, 0, "Cursor X after IL"); // Cursor moves to col 0

        term.cursor = Cursor { x: 0, y: 1 }; // Cursor on the new blank line

        delete_lines(&mut term, 2); // Use function from parent module
         assert_eq!(screen_to_string_vec(&term), vec![
            "11111".to_string(), // Unaffected
            "33333".to_string(), // Line 3 shifted up to line 1
            "     ".to_string(), // New blank line from scroll_up
            "     ".to_string(), // New blank line from scroll_up
            "55555".to_string(), // Unaffected (below region)
        ], "DL failed");
        assert_eq!(term.cursor.x, 0, "Cursor X after DL"); // Cursor moves to col 0
    }

    #[test]
    fn test_erase_line_variants() {
        let mut term = Term::new(5,1);
        term.process_bytes(b"abcde");
        term.cursor = Cursor { x: 2, y: 0 };

        erase_line_to_end(&mut term); // Use function from parent module
        assert_eq!(screen_to_string_vec(&term), vec!["ab   ".to_string()], "EL 0 failed");

        term.screen[0] = "ab de".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 2, y: 0 };
        erase_line_to_start(&mut term); // Use function from parent module
        assert_eq!(screen_to_string_vec(&term), vec!["   de".to_string()], "EL 1 failed");

        term.screen[0] = "abcde".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 2, y: 0 };
        erase_whole_line(&mut term); // Use function from parent module
        assert_eq!(screen_to_string_vec(&term), vec!["     ".to_string()], "EL 2 failed");
    }

    #[test]
    fn test_erase_display_variants() {
        let mut term = Term::new(5, 3);
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 1, y: 1 };

        erase_display_to_end(&mut term); // Use function from parent module
        assert_eq!(screen_to_string_vec(&term), vec![
            "11111".to_string(),
            "2    ".to_string(),
            "     ".to_string(),
        ], "ED 0 failed");

        // Reset state
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 1, y: 1 };
        erase_display_to_start(&mut term); // Use function from parent module
        assert_eq!(screen_to_string_vec(&term), vec![
            "     ".to_string(),
            "  222".to_string(),
            "33333".to_string(),
        ], "ED 1 failed");

        // Reset state
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 1, y: 1 };
        erase_whole_display(&mut term); // Use function from parent module
        assert_eq!(screen_to_string_vec(&term), vec![
            "     ".to_string(),
            "     ".to_string(),
            "     ".to_string(),
        ], "ED 2 failed");
    }

    #[test]
    fn test_reset() {
        let mut term = Term::new(10, 5);
        // Modify state
        term.process_bytes(b"hello\x1b[1mworld\x1b[?6h"); // Print, set bold, enable origin
        term.cursor = Cursor { x: 3, y: 2 };
        term.scroll_top = 1;
        term.scroll_bot = 3;
        term.using_alt_screen = true;

        reset(&mut term); // Use function from parent module

        assert_eq!(term.get_cursor(), (0, 0), "Cursor after reset");
        assert_eq!(term.current_attributes, Attributes::default(), "Attributes after reset");
        // Remove parser state check
        // assert_eq!(term.parser_state, ParserState::Ground, "Parser state after reset");
        assert!(!term.dec_modes.origin_mode, "Origin mode after reset");
        assert_eq!(term.scroll_top, 0, "Scroll top after reset");
        assert_eq!(term.scroll_bot, 4, "Scroll bot after reset");
        assert!(!term.using_alt_screen, "Screen after reset");
        assert_eq!(screen_to_string_vec(&term)[0], "          ", "Screen content after reset");
    }

    #[test]
    fn test_alt_screen_switch() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"main");
        term.cursor = Cursor { x: 1, y: 1 }; // Cursor at (1, 1) on main screen
        save_cursor(&mut term); // Use function from parent module

        enter_alt_screen(&mut term); // Use function from parent module
        assert!(term.using_alt_screen, "Entered alt screen");
        assert_eq!(term.get_cursor(), (0, 0), "Cursor home on alt screen");
        assert_eq!(screen_to_string_vec(&term)[0], "          ", "Alt screen initially blank");

        term.process_bytes(b"alt");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'a', "Content on alt screen");
        term.cursor = Cursor { x: 5, y: 2 }; // Move cursor on alt screen

        exit_alt_screen(&mut term); // Use function from parent module
        assert!(!term.using_alt_screen, "Exited alt screen");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'm', "Main screen content restored");
    }
}

#[cfg(test)]
mod robustness_tests {
    // Use super to access items from the parent module (screen.rs)
    // Remove ParserState import
    use crate::term::{Term, Cursor};
    use crate::glyph::{Attributes, Color, AttrFlags}; // Glyph already imported via super::*

    #[test]
    
    #[test]
    fn test_cub_variants() {
        let mut term = Term::new(10, 1);
        term.process_bytes(b"abcdefgh"); // Cursor at x=8
        // CUB 0 -> Should move left 1 (param 0 defaults to 1)
        term.process_bytes(b"\x1b[0D");
        assert_eq!(term.cursor.x, 7, "CUB zero (moves 1)"); // Corrected assertion
        // CUB 1 (default) -> Move left 1
        term.process_bytes(b"\x1b[D");
        assert_eq!(term.cursor.x, 6, "CUB default"); // Corrected expected value
        // CUB 3 -> Move left 3
        term.process_bytes(b"\x1b[3D");
        assert_eq!(term.cursor.x, 3, "CUB 3"); // Corrected expected value
        // CUB 10 -> Move left 10 (clamps to 0)
        term.process_bytes(b"\x1b[10D");
        assert_eq!(term.cursor.x, 0, "CUB clamp left");
    }

     #[test]
    fn test_cuf_variants() {
        let mut term = Term::new(10, 1);
        term.cursor.x = 0;
        // CUF 0 -> Should move right 1 (param 0 defaults to 1)
        term.process_bytes(b"\x1b[0C");
        assert_eq!(term.cursor.x, 1, "CUF zero (moves 1)"); // Corrected assertion
        // CUF 1 (default) -> Move right 1
        term.process_bytes(b"\x1b[C");
        assert_eq!(term.cursor.x, 2, "CUF default"); // Corrected expected value
        // CUF 3 -> Move right 3
        term.process_bytes(b"\x1b[3C");
        assert_eq!(term.cursor.x, 5, "CUF 3"); // Corrected expected value
        // CUF 10 -> Move right 10 (clamps to 9)
        term.process_bytes(b"\x1b[10C");
        assert_eq!(term.cursor.x, 9, "CUF clamp right"); // Expected value is correct
    }

     #[test]
    fn test_cuu_variants() {
        let mut term = Term::new(10, 5);
        term.cursor = Cursor { x: 5, y: 3 };
        // CUU 0 -> Should move up 1 (param 0 defaults to 1)
        term.process_bytes(b"\x1b[0A");
        assert_eq!(term.cursor.y, 2, "CUU zero (moves 1)"); // Corrected assertion
        // CUU 1 (default) -> Move up 1
        term.process_bytes(b"\x1b[A");
        assert_eq!(term.cursor.y, 1, "CUU default"); // Corrected expected value
        // CUU 2 -> Move up 2 (from y=1)
        term.process_bytes(b"\x1b[2A");
        assert_eq!(term.cursor.y, 0, "CUU 2"); // Corrected expected value (was already 0)
        // CUU 10 -> Move up 10 (clamps to 0)
        term.process_bytes(b"\x1b[10A");
        assert_eq!(term.cursor.y, 0, "CUU clamp top");

        // Test interaction with printable
        term.cursor = Cursor { x: 0, y: 1 };
        term.process_bytes(b"\x1b[A"); // Move to (0,0)
        term.process_bytes(b"X"); // Print X at (0,0), cursor moves to (1,0)
        assert_eq!(term.get_cursor(), (1, 0), "CUU printable cursor");
    }

    #[test]
    fn test_cud_variants() {
        let mut term = Term::new(10, 5);
        term.cursor = Cursor { x: 5, y: 1 };
        // CUD 0 -> Should move down 1 (param 0 defaults to 1)
        term.process_bytes(b"\x1b[0B");
        assert_eq!(term.cursor.y, 2, "CUD zero (moves 1)"); // Corrected assertion
        // CUD 1 (default) -> Move down 1
        term.process_bytes(b"\x1b[B");
        assert_eq!(term.cursor.y, 3, "CUD default"); // Corrected expected value
        // CUD 1 -> Move down 1
        term.process_bytes(b"\x1b[1B");
        assert_eq!(term.cursor.y, 4, "CUD 1"); // Corrected expected value
        // CUD 10 -> Move down 10 (clamps to 4)
        term.process_bytes(b"\x1b[10B");
        assert_eq!(term.cursor.y, 4, "CUD clamp bottom"); // Expected value is correct
    }

    #[test]
    fn test_sgr_robustness() {
        let mut term = Term::new(10, 1);
        let default_attr = Attributes::default();
        let bold_attr = Attributes { flags: AttrFlags::BOLD, ..Default::default() };
        let red_attr = Attributes { fg: Color::Idx(1), ..Default::default() };
        let bold_red_attr = Attributes { fg: Color::Idx(1), flags: AttrFlags::BOLD, ..Default::default() };

        term.process_bytes(b"\x1b[1m"); // Bold
        assert_eq!(term.current_attributes, bold_attr, "SGR bold");
        term.process_bytes(b"\x1b[31m"); // Red
        assert_eq!(term.current_attributes, bold_red_attr, "SGR bold red");
        term.process_bytes(b"\x1b[m"); // Reset
        assert_eq!(term.current_attributes, default_attr, "SGR reset");

        term.process_bytes(b"\x1b[31;1m"); // Red then Bold
        assert_eq!(term.current_attributes, bold_red_attr, "SGR red;bold");
        term.process_bytes(b"\x1b[0;31m"); // Reset then Red
        assert_eq!(term.current_attributes, red_attr, "SGR 0;red");
        term.process_bytes(b"\x1b[1m"); // Bold
        assert_eq!(term.current_attributes, bold_red_attr, "SGR bold (additive)");
        term.process_bytes(b"\x1b[22m"); // Normal intensity (bold off)
        assert_eq!(term.current_attributes, red_attr, "SGR bold off");
        term.process_bytes(b"\x1b[1;39m"); // Bold, default FG
        assert_eq!(term.current_attributes, bold_attr, "SGR bold; default fg");
    }

    // --- Robustness against invalid/interrupted sequences ---

    #[test]
    fn test_csi_interrupted_by_escape() {
        let mut term = Term::new(10, 1);
        term.process_bytes(b"A\x1b[1;"); // Enter CSIParam state
        // Check internal state via side-effects or if AnsiProcessor exposes it (not ideal)
        // For now, verify subsequent processing works correctly.
        term.process_bytes(b"\x1b[31mB"); // Interrupt with ESC, start new sequence, print B
        assert_eq!(term.get_cursor(), (2, 0), "Cursor after interrupted CSI");
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A', "Char before interrupt");
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'B', "Char after interrupt");
        assert_eq!(term.get_glyph(1, 0).unwrap().attr.fg, Color::Idx(1), "Color after interrupt");
    }

    #[test]
    fn test_csi_invalid_final_byte() {
        let mut term = Term::new(10, 1);
        term.process_bytes(b"A\x1b[1\x07B"); // CSI with param, then BEL instead of final byte, then B
        // Parser should emit Error(BEL), return to ground, then process B
        assert_eq!(term.get_cursor(), (2, 0), "Cursor after invalid final byte");
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'B'); // B should be printed
    }

     #[test]
    fn test_csi_private_marker() {
        let mut term = Term::new(10,1);
        term.process_bytes(b"\x1b[?25h"); // Private sequence (Show cursor)
        // Check side effect (if we had a way to query cursor visibility)
        // For now, just ensure it parses without crashing and subsequent chars work
        term.process_bytes(b"C");
        assert_eq!(term.get_cursor(), (1, 0));
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'C');
    }
}

// Module specific to tests that failed and were fixed or adjusted
#[cfg(test)]
mod fixed_tests {
    // Use super to access items from the parent module (screen.rs)
    use crate::term::Term;
    use crate::glyph::Glyph; // Import necessary items

     // Helper to get a specific glyph, defaulting if out of bounds.
    fn get_glyph_test(term: &Term, x: usize, y: usize) -> Glyph {
        term.get_glyph(x, y).cloned().unwrap_or_default()
    }

    // Test with corrected assertion
    #[test]
    fn test_process_bytes_simple_csi_fixed() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"ABC\x1b[2B\x1b[3DXYZ");
        // Corrected assertion: Expect cursor position *after* printing XYZ
        assert_eq!(term.get_cursor(), (3, 2), "Cursor after simple CSI");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'A');
        assert_eq!(get_glyph_test(&term, 1, 0).c, 'B');
        assert_eq!(get_glyph_test(&term, 2, 0).c, 'C');
        assert_eq!(get_glyph_test(&term, 0, 2).c, 'X');
        assert_eq!(get_glyph_test(&term, 1, 2).c, 'Y');
        assert_eq!(get_glyph_test(&term, 2, 2).c, 'Z');
        // Remove parser state check
        // assert_eq!(term.parser_state, crate::term::ParserState::Ground, "State after simple CSI");
    }
}

