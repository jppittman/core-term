// src/term/tests.rs

//! Unit tests for the main TerminalEmulator struct and its core logic.

// Module for tests related to the Term struct itself
#[cfg(test)]
mod term_tests {
    // Import necessary items from parent (term) and sibling (glyph) modules
    use crate::term::{
        TerminalEmulator, EmulatorInput, EmulatorAction,
        DecPrivateModes, // Needed for ScreenContext if used for assertions
        // cursor::ScreenContext, // ScreenContext might not be needed if cursor set via ANSI
    };
    use crate::glyph::{Attributes, Color, AttrFlags, Glyph, NamedColor};
    use crate::ansi::commands::{AnsiCommand, C0Control, CsiCommand, Attribute, Color as AnsiColor};
    use crate::backends::BackendEvent; // For EmulatorInput::User

    // Use test_log::test for tests that need logging output
    use test_log::test;


    // --- Test Helpers ---

    /// Helper to create a `TerminalEmulator` instance.
    fn new_term(cols: usize, rows: usize) -> TerminalEmulator {
        TerminalEmulator::new(cols, rows, 100) // Default scrollback for tests
    }

    /// Helper to process a slice of byte-encoded ANSI sequences.
    /// It parses them into AnsiCommands and then feeds them to the emulator.
    fn process_ansi_bytes(term: &mut TerminalEmulator, bytes: &[u8]) -> Vec<EmulatorAction> {
        let mut processor = crate::ansi::AnsiProcessor::new();
        let commands = processor.process_bytes(bytes);
        let mut actions = Vec::new();
        for cmd in commands {
            if let Some(action) = term.interpret_input(EmulatorInput::Ansi(cmd)) {
                actions.push(action);
            }
        }
        // After processing, always request redraw if dirty, mimicking orchestrator
        if !term.take_dirty_lines().is_empty() && !actions.iter().any(|a| matches!(a, EmulatorAction::RequestRedraw)) {
             actions.push(EmulatorAction::RequestRedraw);
        }
        actions
    }

    /// Helper to process a single AnsiCommand or other EmulatorInput.
    fn process_input(term: &mut TerminalEmulator, input: EmulatorInput) -> Option<EmulatorAction> {
        let action = term.interpret_input(input);
        // Simulate orchestrator requesting redraw if lines are dirty and no explicit redraw was actioned
        if action.is_none() && !term.take_dirty_lines().is_empty() {
            return Some(EmulatorAction::RequestRedraw);
        }
        action
    }


    /// Helper to get screen content as a Vec of Strings.
    fn screen_to_string_vec(term: &TerminalEmulator) -> Vec<String> {
        let (cols, rows) = term.dimensions();
        let mut result = Vec::with_capacity(rows);
        for y in 0..rows {
            let line: String = (0..cols).map(|x| term.get_glyph(x, y).c).collect();
            result.push(line);
        }
        result
    }

    /// Helper to get a specific glyph.
    fn get_glyph_at(term: &TerminalEmulator, x: usize, y: usize) -> Glyph {
        term.get_glyph(x, y)
    }

    /// Helper to assert logical cursor position.
    fn assert_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        assert_eq!(term.cursor_pos(), (x, y), "Logical cursor position check: {}", message);
    }

    /// Helper to assert absolute screen cursor position.
    fn assert_screen_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        assert_eq!(term.get_screen_cursor_pos(), (x,y), "Screen cursor position check: {}", message);
    }

    // The get_screen_context helper might be removed if not used for assertions,
    // as cursor setup will now be via ANSI commands.
    // fn get_screen_context(term: &TerminalEmulator) -> ScreenContext {
    //     let (cols, rows) = term.dimensions();
    //     ScreenContext {
    //         width: cols,
    //         height: rows,
    //         scroll_top: term.screen.scroll_top(),
    //         scroll_bot: term.screen.scroll_bot(),
    //         origin_mode_active: term.dec_modes.origin_mode,
    //     }
    // }


    // --- Initialization Tests ---

    #[test]
    fn test_new_terminal_initial_state() {
        let term = new_term(80, 24);
        assert_eq!(term.dimensions(), (80, 24), "Initial dimensions");
        assert_cursor_pos(&term, 0, 0, "Initial logical cursor position");
        assert_screen_cursor_pos(&term, 0, 0, "Initial screen cursor position");
        assert!(term.is_cursor_visible(), "Cursor initially visible");
        assert!(!term.is_alt_screen_active(), "Initially not on alt screen");

        let expected_initial_attrs = Attributes::default();
        assert_eq!(get_glyph_at(&term, 79, 23), Glyph { c: ' ', attr: expected_initial_attrs }, "Initial screen content (corner)");

        let mut term_mut = new_term(80, 24);
        let dirty_lines = term_mut.take_dirty_lines();
        assert_eq!(dirty_lines.len(), 24, "All lines initially dirty");
        assert_eq!(dirty_lines, (0..24).collect::<Vec<usize>>(), "Correct dirty line indices");
    }

    #[test]
    fn test_new_terminal_minimum_dimensions() {
        let term = new_term(0, 0);
        assert_eq!(term.dimensions().0, 1, "Minimum width clamped to 1");
        assert_eq!(term.dimensions().1, 1, "Minimum height clamped to 1");

        let term = new_term(5, 0);
        assert_eq!(term.dimensions(), (5, 1), "Minimum height clamped");

        let term = new_term(0, 10);
        assert_eq!(term.dimensions(), (1, 10), "Minimum width clamped");
    }

    // --- Basic Printable Character Tests ---
    #[test]
    fn test_print_ascii_chars() {
        let mut term = new_term(10, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('H')));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('i')));

        assert_cursor_pos(&term, 2, 0, "Cursor after 'Hi'");
        assert_eq!(get_glyph_at(&term, 0, 0).c, 'H', "Char H");
        assert_eq!(get_glyph_at(&term, 1, 0).c, 'i', "Char i");
        assert_eq!(get_glyph_at(&term, 2, 0).c, ' ', "Empty char after Hi");
    }

    #[test]
    fn test_print_chars_with_line_wrap() {
        let mut term = new_term(5, 2);
        process_ansi_bytes(&mut term, b"Hello");
        assert_cursor_pos(&term, 5, 0, "Cursor after 'Hello' before wrap");

        process_ansi_bytes(&mut term, b"W");
        assert_cursor_pos(&term, 1, 1, "Cursor after 'W' post-wrap");
        let screen = screen_to_string_vec(&term);
        assert_eq!(screen[0], "Hello", "Line 0 after wrap");
        assert_eq!(screen[1], "W    ", "Line 1 after wrap");
    }

    #[test]
    fn test_print_chars_with_scroll() {
        let mut term = new_term(5, 2);
        process_ansi_bytes(&mut term, b"Line1");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // Moves to (0,1)
        process_ansi_bytes(&mut term, b"Line2");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // Moves to (0,1) on scrolled screen
        process_ansi_bytes(&mut term, b"New");

        assert_cursor_pos(&term, 3, 1, "Cursor after scroll and 'New'");
        let screen = screen_to_string_vec(&term);
        assert_eq!(screen[0], "Line2", "Line 0 after scroll (was Line2)");
        assert_eq!(screen[1], "New  ", "Line 1 after scroll (is New)");
    }

    #[test]
    fn test_print_utf8_multibyte_char() {
        let mut term = new_term(5, 1);
        process_ansi_bytes(&mut term, "你好".as_bytes());
        assert_eq!(get_glyph_at(&term, 0, 0).c, '你', "First char 你");
        assert_eq!(get_glyph_at(&term, 1, 0).c, '好', "Second char 好");
        let width_ni = crate::term::unicode::get_char_display_width('你');
        let width_hao = crate::term::unicode::get_char_display_width('好');
        assert_cursor_pos(&term, width_ni + width_hao, 0, "Cursor after '你好'");
    }

    #[test]
    fn test_print_wide_character_cjk() {
        let mut term = new_term(5, 1);
        let wide_char = '世';
        let char_width = crate::term::unicode::get_char_display_width(wide_char);
        if char_width != 2 {
            log::warn!("Wide character '世' is not treated as width 2 in this environment (width: {}). Skipping full wide char assertion.", char_width);
        }

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print(wide_char)));

        assert_eq!(get_glyph_at(&term, 0, 0).c, wide_char, "Wide char printed");
        assert_cursor_pos(&term, char_width, 0, "Cursor after wide char");

        if char_width == 2 {
            assert_eq!(get_glyph_at(&term, 1, 0).c, '\0', "Wide char placeholder");
        }
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        assert_eq!(get_glyph_at(&term, char_width, 0).c, 'A', "Next char after wide char");
        assert_cursor_pos(&term, char_width + 1, 0, "Cursor after A");
    }

    #[test]
    fn test_print_wide_character_at_edge_of_line() {
        let mut term = new_term(2, 2);
        let wide_char = '世';
        let char_width = crate::term::unicode::get_char_display_width(wide_char);
        if char_width != 2 {
             log::warn!("Wide character '世' is not treated as width 2 in this environment (width: {}). Skipping wide char edge test.", char_width);
            return;
        }

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print(wide_char)));
        assert_eq!(get_glyph_at(&term, 0, 0).c, wide_char, "Wide char on line 0");
        assert_eq!(get_glyph_at(&term, 1, 0).c, '\0', "Placeholder on line 0");
        assert_cursor_pos(&term, 0, 1, "Cursor wrapped to next line");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'A', "Char 'A' on line 1");
        assert_cursor_pos(&term, 1, 1, "Cursor after 'A'");
    }


    // --- C0 Control Tests ---
    #[test]
    fn test_c0_bs_backspace() {
        let mut term = new_term(5, 1);
        process_ansi_bytes(&mut term, b"ABC");
        assert_cursor_pos(&term, 3, 0, "Cursor after ABC");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)));
        assert_cursor_pos(&term, 2, 0, "Cursor after BS");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)));
        assert_cursor_pos(&term, 0, 0, "Cursor at start after multiple BS");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)));
        assert_cursor_pos(&term, 0, 0, "Cursor does not go past start");
    }

    #[test]
    fn test_c0_ht_horizontal_tab() {
        let mut term = new_term(20, 1);

        assert_cursor_pos(&term, 0, 0, "Initial");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 8, 0, "Tab from 0 to 8");

        // Move cursor to (col 5, row 0) using CUP (1-based: row 1, col 6)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1, 6))));
        assert_cursor_pos(&term, 5, 0, "Cursor at (5,0) after CUP");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 8, 0, "Tab from 5 to 8");

        // Move cursor to (col 8, row 0) using CUP (1-based: row 1, col 9)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1, 9))));
        assert_cursor_pos(&term, 8, 0, "Cursor at (8,0) after CUP");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 16, 0, "Tab from 8 to 16");

        // Move cursor to (col 17, row 0) using CUP (1-based: row 1, col 18)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1, 18))));
        assert_cursor_pos(&term, 17, 0, "Cursor at (17,0) after CUP");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 19, 0, "Tab from 17 to end of line (19)");
    }

    #[test]
    fn test_c0_lf_line_feed() {
        let mut term = new_term(5,2);
        process_ansi_bytes(&mut term, b"ABC");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
        assert_cursor_pos(&term, 0, 1, "Cursor after LF (moves to col 0 of next line)");
        let screen = screen_to_string_vec(&term);
        assert_eq!(screen[0], "ABC  ", "Original line content after LF");
    }

    #[test]
    fn test_c0_cr_carriage_return() {
        let mut term = new_term(5,1);
        process_ansi_bytes(&mut term, b"ABC");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
        assert_cursor_pos(&term, 0, 0, "Cursor after CR (moves to col 0 of same line)");
    }

    #[test]
    fn test_c0_bel_produces_action() {
        let mut term = new_term(5,1);
        let actions = process_ansi_bytes(&mut term, b"\x07"); // BEL
        assert!(actions.contains(&EmulatorAction::RingBell), "BEL should produce RingBell action");
    }

    // --- CSI Sequence Tests ---

    #[test]
    fn test_csi_cup_cursor_position() {
        let mut term = new_term(10, 5);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,1))));
        assert_cursor_pos(&term, 0, 0, "CUP default (1,1)");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(3,4))));
        assert_cursor_pos(&term, 3, 2, "CUP R=3, C=4");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(5,1))));
        assert_cursor_pos(&term, 0, 4, "CUP R=5 (col defaults to 1)");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(100,100))));
        assert_cursor_pos(&term, 9, 4, "CUP out of bounds (100,100 -> 9,4)");
    }

    #[test]
    fn test_csi_sgr_set_graphics_rendition() {
        let mut term = new_term(10,1);
        let default_attrs = Attributes::default();

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Bold]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('B')));
        assert_eq!(get_glyph_at(&term, 0, 0).attr.flags, AttrFlags::BOLD, "SGR Bold");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(AnsiColor::Red)]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('R')));
        assert_eq!(get_glyph_at(&term, 1, 0).attr.fg, Color::Named(NamedColor::Red), "SGR Red FG");
        assert_eq!(get_glyph_at(&term, 1, 0).attr.flags, AttrFlags::BOLD, "SGR Bold still active");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('N')));
        assert_eq!(get_glyph_at(&term, 2, 0).attr, default_attrs, "SGR Reset");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Underline,
            Attribute::Background(AnsiColor::Blue),
        ]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('X')));
        let attrs_x = get_glyph_at(&term, 3, 0).attr;
        assert!(attrs_x.flags.contains(AttrFlags::UNDERLINE), "SGR Underline");
        assert_eq!(attrs_x.bg, Color::Named(NamedColor::Blue), "SGR Blue BG");
        assert_eq!(attrs_x.fg, default_attrs.fg, "SGR FG remains default after multi-attr");
    }

    #[test]
    fn test_csi_ed_erase_in_display() {
        let mut term = new_term(5,3);
        process_ansi_bytes(&mut term, b"11111\n22222\n33333");

        // Move cursor to (col 2, row 1) using CUP (1-based: row 2, col 3)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2, 3))));
        assert_cursor_pos(&term, 2, 1, "Cursor at (2,1) before ED 0");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(0))));
        let screen0 = screen_to_string_vec(&term);
        assert_eq!(screen0[0], "11111", "ED 0: Line 0 unaffected");
        assert_eq!(screen0[1], "22   ", "ED 0: Line 1 erased from cursor");
        assert_eq!(screen0[2], "     ", "ED 0: Line 2 erased fully");

        // Reset screen and cursor for next ED test
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2)))); // ED 2 (Clear all)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,1)))); // Home cursor
        process_ansi_bytes(&mut term, b"11111\n22222\n33333");
        // Move cursor to (col 2, row 1) using CUP (1-based: row 2, col 3)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2, 3))));
        assert_cursor_pos(&term, 2, 1, "Cursor at (2,1) before ED 1");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(1))));
        let screen1 = screen_to_string_vec(&term);
        assert_eq!(screen1[0], "     ", "ED 1: Line 0 erased fully");
        assert_eq!(screen1[1], "   22", "ED 1: Line 1 erased to cursor"); // Note: cursor is at (2,1), so cells 0,1,2 erased
        assert_eq!(screen1[2], "33333", "ED 1: Line 2 unaffected");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))));
        let screen2 = screen_to_string_vec(&term);
        assert_eq!(screen2[0], "     ", "ED 2: Line 0 erased");
        assert_eq!(screen2[1], "     ", "ED 2: Line 1 erased");
        assert_eq!(screen2[2], "     ", "ED 2: Line 2 erased");
    }

    #[test]
    fn test_dec_mode_origin_decom() {
        let mut term = new_term(10, 5);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetScrollingRegion{top: 2, bottom: 4}))); // 0-indexed: (1,3)
        assert_cursor_pos(&term, 0, 0, "Cursor home after DECSTBM (origin mode off)");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(6)))); // DECOM On
        assert!(term.dec_modes.origin_mode, "DECOM should be active");
        assert_cursor_pos(&term, 0, 0, "Cursor home (logical 0,0) after DECOM on");
        assert_screen_cursor_pos(&term, 0, 1, "Screen cursor at margin top (phys 1) after DECOM on");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,1))));
        assert_cursor_pos(&term, 0, 0, "CUP 1,1 (logical) with DECOM");
        assert_screen_cursor_pos(&term, 0, 1, "CUP 1,1 (screen phys 1) with DECOM");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(3,3)))); // Logical (2,2) relative to margin top
        assert_cursor_pos(&term, 2, 2, "CUP 3,3 (logical 2,2) with DECOM");
        assert_screen_cursor_pos(&term, 2, 3, "CUP 3,3 (screen phys 1+2=3) with DECOM");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(6)))); // DECOM Off
        assert!(!term.dec_modes.origin_mode, "DECOM should be inactive");
        assert_cursor_pos(&term, 0, 0, "Cursor home after DECOM off");
        assert_screen_cursor_pos(&term, 0, 0, "Screen cursor home after DECOM off");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2,3)))); // Absolute (2,1)
        assert_cursor_pos(&term, 2, 1, "CUP 2,3 absolute after DECOM off");
    }

    // --- Resize Tests ---
    #[test]
    fn test_resize_larger_clamps_cursor_correctly() {
        let mut term = new_term(5, 2);
        process_ansi_bytes(&mut term, b"12345\nAB");
        assert_cursor_pos(&term, 2, 1, "Cursor before resize");

        term.resize(10, 4);

        assert_eq!(term.dimensions(), (10, 4), "Dimensions after resize larger");
        assert_cursor_pos(&term, 2, 1, "Cursor position maintained after resize larger");
        let screen = screen_to_string_vec(&term);
        assert_eq!(screen[0], "12345     ", "Line 0 content after resize");
        assert_eq!(screen[1], "AB        ", "Line 1 content after resize");
    }

    #[test]
    fn test_resize_smaller_clamps_cursor_correctly() {
        let mut term = new_term(10, 4);
        process_ansi_bytes(&mut term, b"1234567890\nabcdefghij\nABCDEFGHIJ\nqwerty");
        assert_cursor_pos(&term, 6, 3, "Cursor before resize");

        term.resize(5, 2);

        assert_eq!(term.dimensions(), (5, 2), "Dimensions after resize smaller");
        assert_cursor_pos(&term, 4, 1, "Cursor clamped after resize smaller");
        let screen = screen_to_string_vec(&term);
        assert_eq!(screen[0], "12345", "Line 0 content after resize smaller");
        assert_eq!(screen[1], "abcde", "Line 1 content after resize smaller");
    }

    // --- User Input Tests ---
    #[test]
    fn test_user_input_printable_key() {
        let mut term = new_term(5,1);
        let event = BackendEvent::Key { keysym: 'A' as u32, text: "A".to_string() };
        let action = term.interpret_input(EmulatorInput::User(event));

        if let Some(EmulatorAction::WritePty(bytes)) = action {
            assert_eq!(bytes, b"A", "WritePty for 'A'");
        } else {
            panic!("Expected WritePty action for printable key");
        }
    }

    #[test]
    fn test_user_input_enter_key() {
        let mut term = new_term(5,1);
        let event = BackendEvent::Key { keysym: 0xFF0D, text: "\r".to_string() };
        let action = term.interpret_input(EmulatorInput::User(event));

        if let Some(EmulatorAction::WritePty(bytes)) = action {
            assert_eq!(bytes, b"\r", "WritePty for Enter key");
        } else {
            panic!("Expected WritePty action for Enter key");
        }
    }

    #[test]
    fn test_user_input_arrow_key_normal_mode() {
        let mut term = new_term(5,1);
        term.dec_modes.cursor_keys_app_mode = false; // Ensure normal mode

        let event_up = BackendEvent::Key { keysym: 0xFF52, text: "".to_string() };
        let action_up = term.interpret_input(EmulatorInput::User(event_up));
        if let Some(EmulatorAction::WritePty(bytes)) = action_up {
            assert_eq!(bytes, b"\x1b[A", "WritePty for Up Arrow (Normal)");
        } else {
            panic!("Expected WritePty for Up Arrow");
        }
    }

    #[test]
    fn test_user_input_arrow_key_application_mode() {
        let mut term = new_term(5,1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(1))));
        assert!(term.dec_modes.cursor_keys_app_mode, "DECCKM should be enabled");

        let event_down = BackendEvent::Key { keysym: 0xFF54, text: "".to_string() };
        let action_down = term.interpret_input(EmulatorInput::User(event_down));
        if let Some(EmulatorAction::WritePty(bytes)) = action_down {
            assert_eq!(bytes, b"\x1bOB", "WritePty for Down Arrow (Application)");
        } else {
            panic!("Expected WritePty for Down Arrow (Application)");
        }
    }
}

