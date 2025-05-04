// src/term/screen/tests.rs

//! Tests for screen manipulation logic in screen.rs

#[cfg(test)]
mod original_tests {
    use crate::term::{Term, Cursor, ParserState, screen::*}; // Import necessary items from parent
    use crate::glyph::{Glyph, Attributes}; // Import necessary glyph items

    // Helper to get screen content as a Vec of Strings
    fn screen_to_string_vec(term: &Term) -> Vec<String> {
        let current_screen = if term.using_alt_screen { &term.alt_screen } else { &term.screen };
        current_screen.iter()
            .map(|row| row.iter().map(|g| g.c).collect())
            .collect()
    }

     // Helper to get screen content with attributes for debugging
     #[allow(dead_code)]
     fn screen_to_debug_vec(term: &Term) -> Vec<Vec<String>> {
         let current_screen = if term.using_alt_screen { &term.alt_screen } else { &term.screen };
         current_screen.iter().map(|row| {
             row.iter().map(|g| format!("'{}'({:?}|{:?}|{:?})", g.c, g.attr.fg, g.attr.bg, g.attr.flags)).collect()
         }).collect()
     }


    // Helper to get a specific glyph, defaulting if out of bounds.
    fn get_glyph_test(term: &Term, x: usize, y: usize) -> Glyph {
        term.get_glyph(x, y).cloned().unwrap_or_default()
    }

    // --- Original Tests (Adapted) ---

    #[test]
    fn test_backspace_stops_at_zero() {
        let mut term = Term::new(10, 1);
        term.cursor = Cursor { x: 5, y: 0 };
        backspace(&mut term);
        assert_eq!(term.cursor.x, 4);
        term.cursor.x = 0;
        backspace(&mut term);
        assert_eq!(term.cursor.x, 0);
    }

    #[test]
    fn test_carriage_return_moves_home() {
        let mut term = Term::new(10, 1);
        term.cursor = Cursor { x: 5, y: 0 };
        carriage_return(&mut term);
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

        newline(&mut term);

        assert_eq!(term.cursor.x, 0, "Cursor X after newline");
        assert_eq!(term.cursor.y, 2, "Cursor Y after newline");

        // Test newline at bottom margin (should scroll)
        term.cursor = Cursor { x: 3, y: 2 };
        newline(&mut term);
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
        set_scrolling_region(&mut term, 2, 3); // Region lines 1-2 (0-based)
        term.cursor = Cursor { x: 0, y: 2 }; // Cursor on line 2 (333), which is scroll_bot

        index(&mut term); // Should scroll region up

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
        set_scrolling_region(&mut term, 2, 3); // Region lines 1-2 (0-based)
        term.cursor = Cursor { x: 0, y: 1 }; // Cursor on line 1 (222), which is scroll_top

        reverse_index(&mut term); // Should scroll region down

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
        set_scrolling_region(&mut term, 2, 4); // Region lines 1-3 (0-based)
        enable_origin_mode(&mut term); // Enable origin mode, cursor moves to (0, 1)
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 1); // Should be at top of region (1)

        index(&mut term);
        assert_eq!(term.cursor.y, 2, "Index within region");
        index(&mut term);
        assert_eq!(term.cursor.y, 3, "Index at region bottom");
        index(&mut term);
        assert_eq!(term.cursor.y, 3, "Index scrolls at region bottom");
        reverse_index(&mut term);
        assert_eq!(term.cursor.y, 2, "Reverse Index within region");
        reverse_index(&mut term);
        assert_eq!(term.cursor.y, 1, "Reverse Index at region top");
        reverse_index(&mut term);
        assert_eq!(term.cursor.y, 1, "Reverse Index scrolls at region top");
    }

     #[test]
    fn test_origin_mode_cursor_addressing_cup() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Region lines 1-3 (0-based)
        term.cursor = Cursor{ x: 5, y: 2 };

        enable_origin_mode(&mut term); // Cursor moves to (0, 1)

        set_cursor_pos(&mut term, 1, 1); // CUP 1,1 (relative)
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 1, "CUP 1,1 relative");

        set_cursor_pos(&mut term, 3, 3); // CUP 3,3 (relative) -> absolute (2, 3)
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 3, "CUP 3,3 relative");

        set_cursor_pos(&mut term, 1, 10); // CUP 1,10 (relative, > region height) -> clamps to (0, 3)
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 3, "CUP 1,10 relative clamped");

        disable_origin_mode(&mut term); // Cursor moves to (0, 0)

        set_cursor_pos(&mut term, 2, 3); // CUP 2, 3 (absolute) -> (1, 2)
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.cursor.y, 2, "CUP 2,3 absolute");
    }

    #[test]
    fn test_origin_mode_enable_disable() {
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4); // Region 1-3
        term.cursor = Cursor { x: 5, y: 2 };

        enable_origin_mode(&mut term); // Should set mode and move cursor to region home (0,1)
        assert!(term.dec_modes.origin_mode);
        assert_eq!(term.cursor.x, 0, "Cursor X after enable");
        assert_eq!(term.cursor.y, 1, "Cursor Y after enable (region top)");

        term.cursor = Cursor { x: 3, y: 3 }; // Move cursor within region again

        disable_origin_mode(&mut term); // Should unset mode and move cursor to absolute home (0,0)
        assert!(!term.dec_modes.origin_mode);
        assert_eq!(term.cursor.x, 0, "Cursor X after disable");
        assert_eq!(term.cursor.y, 0, "Cursor Y after disable");
    }

    #[test]
    fn test_deprecated_set_origin_mode() {
        // Test integration via process_bytes
        let mut term = Term::new(10, 5);
        set_scrolling_region(&mut term, 2, 4);
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
        tab(&mut term);
        assert_eq!(term.cursor.x, 8, "Tab from 0");

        term.cursor.x = 5;
        tab(&mut term);
        assert_eq!(term.cursor.x, 8, "Tab from 5");

        term.cursor.x = 8;
        tab(&mut term);
        assert_eq!(term.cursor.x, 16, "Tab from 8");

        term.cursor.x = 17;
        tab(&mut term);
        assert_eq!(term.cursor.x, 19, "Tab from 17 (to end)");
    }

    #[test]
    fn test_insert_chars() {
        let mut term = Term::new(10,1);
        term.process_bytes(b"abcdefghij");
        term.cursor = Cursor { x: 2, y: 0 }; // Cursor at 'c' (index 2)

        insert_blank_chars(&mut term, 3); // ICH 3

        assert_eq!(screen_to_string_vec(&term), vec!["ab   cdefg".to_string()]);
        assert_eq!(term.cursor.x, 2, "Cursor unchanged after ICH");
    }

    #[test]
    fn test_delete_chars() {
        let mut term = Term::new(10,1);
        term.process_bytes(b"abcdefghij");
        term.cursor = Cursor { x: 2, y: 0 }; // Cursor at 'c' (index 2)

        delete_chars(&mut term, 3); // DCH 3

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
        set_scrolling_region(&mut term, 2, 4); // Region lines 1-3 (0-based)
        term.cursor = Cursor { x: 0, y: 2 }; // Cursor on line 2 (content: 333)

        insert_blank_lines(&mut term, 1); // IL 1
        assert_eq!(screen_to_string_vec(&term), vec![
            "11111".to_string(), // Unaffected
            "22222".to_string(), // Unaffected (top of region)
            "     ".to_string(), // Inserted blank line
            "33333".to_string(), // Shifted down
            "55555".to_string(), // Unaffected (below region)
        ], "IL failed");
        assert_eq!(term.cursor.x, 0, "Cursor X after IL");

        term.cursor = Cursor { x: 0, y: 2 }; // Cursor on the new blank line

        delete_lines(&mut term, 2); // DL 2
         assert_eq!(screen_to_string_vec(&term), vec![
            "11111".to_string(), // Unaffected
            "22222".to_string(), // Unaffected (top of region)
            "33333".to_string(), // Shifted up
            "     ".to_string(), // New blank line
            "55555".to_string(), // Unaffected (below region)
        ], "DL failed");
        assert_eq!(term.cursor.x, 0, "Cursor X after DL");
    }

    #[test]
    fn test_erase_line_variants() {
        let mut term = Term::new(5,1);
        term.process_bytes(b"abcde");
        term.cursor = Cursor { x: 2, y: 0 };

        erase_line_to_end(&mut term); // EL 0
        assert_eq!(screen_to_string_vec(&term), vec!["ab   ".to_string()], "EL 0 failed");

        term.screen[0] = "ab de".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 2, y: 0 };
        erase_line_to_start(&mut term); // EL 1
        assert_eq!(screen_to_string_vec(&term), vec!["   de".to_string()], "EL 1 failed");

        term.screen[0] = "abcde".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 2, y: 0 };
        erase_whole_line(&mut term); // EL 2
        assert_eq!(screen_to_string_vec(&term), vec!["     ".to_string()], "EL 2 failed");
    }

    #[test]
    fn test_erase_display_variants() {
        let mut term = Term::new(5, 3);
        term.screen[0] = "11111".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[1] = "22222".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.screen[2] = "33333".chars().map(|c| Glyph { c, ..Default::default() }).collect();
        term.cursor = Cursor { x: 1, y: 1 };

        erase_display_to_end(&mut term); // ED 0
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
        erase_display_to_start(&mut term); // ED 1
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
        erase_whole_display(&mut term); // ED 2
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

        reset(&mut term); // RIS

        assert_eq!(term.get_cursor(), (0, 0), "Cursor after reset");
        assert_eq!(term.current_attributes, Attributes::default(), "Attributes after reset");
        assert_eq!(term.parser_state, ParserState::Ground, "Parser state after reset");
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
        save_cursor(&mut term); // Save cursor for main screen

        enter_alt_screen(&mut term);
        assert!(term.using_alt_screen, "Entered alt screen");
        assert_eq!(term.get_cursor(), (0, 0), "Cursor home on alt screen");
        assert_eq!(screen_to_string_vec(&term)[0], "          ", "Alt screen initially blank");

        term.process_bytes(b"alt");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'a', "Content on alt screen");
        term.cursor = Cursor { x: 5, y: 2 }; // Move cursor on alt screen

        exit_alt_screen(&mut term);
        assert!(!term.using_alt_screen, "Exited alt screen");
        assert_eq!(get_glyph_test(&term, 0, 0).c, 'm', "Main screen content restored");
    }
}

#[cfg(test)]
mod control_sequences_robustness {
    use crate::term::{Term, Cursor, ParserState, screen::*}; // Import necessary items
    use crate::glyph::{Glyph, Attributes, Color, AttrFlags};

    // Helper to get screen content as a Vec of Strings
    fn screen_to_string_vec(term: &Term) -> Vec<String> {
        let current_screen = if term.using_alt_screen { &term.alt_screen } else { &term.screen };
        current_screen.iter()
            .map(|row| row.iter().map(|g| g.c).collect())
            .collect()
    }
     // Helper to get a specific glyph, defaulting if out of bounds.
    fn get_glyph_test(term: &Term, x: usize, y: usize) -> Glyph {
        term.get_glyph(x, y).cloned().unwrap_or_default()
    }

    #[test]
    fn test_cub_variants() {
        let mut term = Term::new(10, 1);
        term.process_bytes(b"abcdefgh"); // Cursor at x=8
        term.process_bytes(b"\x1b[0D"); // CUB 0 -> No move
        assert_eq!(term.cursor.x, 8, "CUB zero");
        term.process_bytes(b"\x1b[D"); // CUB 1 (default) -> Move left 1
        assert_eq!(term.cursor.x, 7, "CUB default");
        term.process_bytes(b"\x1b[3D"); // CUB 3 -> Move left 3
        assert_eq!(term.cursor.x, 4, "CUB 3");
        term.process_bytes(b"\x1b[10D"); // CUB 10 -> Move left 10 (clamps to 0)
        assert_eq!(term.cursor.x, 0, "CUB clamp left");
    }

     #[test]
    fn test_cuf_variants() {
        let mut term = Term::new(10, 1);
        term.cursor.x = 0;
        term.process_bytes(b"\x1b[0C"); // CUF 0 -> No move
        assert_eq!(term.cursor.x, 0, "CUF zero");
        term.process_bytes(b"\x1b[C"); // CUF 1 (default) -> Move right 1
        assert_eq!(term.cursor.x, 1, "CUF default");
        term.process_bytes(b"\x1b[3C"); // CUF 3 -> Move right 3
        assert_eq!(term.cursor.x, 4, "CUF 3");
        term.process_bytes(b"\x1b[10C"); // CUF 10 -> Move right 10 (clamps to 9)
        assert_eq!(term.cursor.x, 9, "CUF clamp right");
    }

     #[test]
    fn test_cuu_variants() {
        let mut term = Term::new(10, 5);
        term.cursor = Cursor { x: 5, y: 3 };
        term.process_bytes(b"\x1b[0A"); // CUU 0 -> No move
        assert_eq!(term.cursor.y, 3, "CUU zero");
        term.process_bytes(b"\x1b[A"); // CUU 1 (default) -> Move up 1
        assert_eq!(term.cursor.y, 2, "CUU default");
        term.process_bytes(b"\x1b[2A"); // CUU 2 -> Move up 2
        assert_eq!(term.cursor.y, 0, "CUU 2");
        term.process_bytes(b"\x1b[10A"); // CUU 10 -> Move up 10 (clamps to 0)
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
        term.process_bytes(b"\x1b[0B"); // CUD 0 -> No move
        assert_eq!(term.cursor.y, 1, "CUD zero");
        term.process_bytes(b"\x1b[B"); // CUD 1 (default) -> Move down 1
        assert_eq!(term.cursor.y, 2, "CUD default");
        term.process_bytes(b"\x1b[1B"); // CUD 1 -> Move down 1
        assert_eq!(term.cursor.y, 3, "CUD 1");
        term.process_bytes(b"\x1b[10B"); // CUD 10 -> Move down 10 (clamps to 4)
        assert_eq!(term.cursor.y, 4, "CUD clamp bottom");
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
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_bytes(b"\x1b"); // Interrupt with ESC
        assert_eq!(term.parser_state, ParserState::Escape, "Parser should transition to Escape");
        assert!(term.csi_params.is_empty(), "Params should be cleared on ESC interrupt");
    }

    #[test]
    fn test_csi_invalid_final_byte() {
        let mut term = Term::new(10, 1);
        term.process_bytes(b"\x1b[1\x07"); // CSI with param, then BEL instead of final byte
        assert_eq!(term.parser_state, ParserState::Ground, "Parser should return to Ground after invalid final byte");
        assert!(term.csi_params.is_empty(), "Params should be cleared after invalid final byte");
    }

    #[test]
    fn test_csi_intermediate_byte() {
        let mut term = Term::new(10,1);
        term.process_bytes(b"\x1b[1;2!>"); // Params, intermediates
        assert_eq!(term.parser_state, ParserState::CSIIntermediate, "CSIIntermediate byte state");
        assert_eq!(term.csi_params, vec![1, 2], "CSIIntermediate byte params");
        assert_eq!(term.csi_intermediates, vec!['!', '>'], "CSIIntermediate byte intermediates");
    }

     #[test]
    fn test_csi_private_marker() {
        let mut term = Term::new(10,1);
        term.process_bytes(b"\x1b[?25h"); // Private sequence
        assert_eq!(term.parser_state, ParserState::Ground, "State after private sequence");
        assert!(term.csi_intermediates.is_empty(), "Intermediates cleared after private sequence");
    }
}
