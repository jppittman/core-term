// src/term/tests.rs

//! Unit tests for the main TerminalEmulator struct and its core logic.

// Module for tests related to the Term struct itself
#[cfg(test)]
mod term_tests {
    // Import necessary items from parent (term) and sibling (glyph) modules
    use crate::term::{
        TerminalEmulator, EmulatorInput, EmulatorAction,
        // DecPrivateModes, CharacterSet, Mode, // For checking mode changes - Marked as unused
        // DEFAULT_TAB_INTERVAL, // Marked as unused
    };
    // Using std::char::REPLACEMENT_CHARACTER directly
    use crate::glyph::{Attributes, Color, AttrFlags, Glyph, NamedColor};
    // EscCommand was marked as unused
    use crate::ansi::commands::{AnsiCommand, C0Control, CsiCommand, Attribute, Color as AnsiColor};
    use crate::backends::BackendEvent; // For EmulatorInput::User
    use std::char::REPLACEMENT_CHARACTER; // Correct import for REPLACEMENT_CHARACTER

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

    /// Helper to process a single AnsiCommand.
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

    /// Helper to assert cursor position.
    fn assert_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        assert_eq!(term.cursor_pos(), (x, y), "Cursor position check: {}", message);
    }

    /// Helper to assert absolute screen cursor position.
    fn assert_screen_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        assert_eq!(term.get_screen_cursor_pos(), (x,y), "Screen cursor position check: {}", message);
    }

    /// Helper to assert current SGR attributes.
    fn assert_attributes(_term: &TerminalEmulator, expected_attrs: Attributes, message: &str) {
        // This requires adding a public accessor for current_attributes or testing via glyphs.
        // For now, we'll test by printing a char and checking its attributes.
        // This is an indirect way if direct accessor isn't available/desired.
        // If `current_attributes` becomes pub(crate) or has a getter, use that.
        // Based on current `term/mod.rs`, `current_attributes` is private.
        // We test its effect by checking the attributes of the *next* char printed.
        // This is a valid public API test.
        let mut temp_term = new_term(1,1); // Create a tiny temp term
        temp_term.current_attributes = expected_attrs; // Manually set for comparison basis
                                                       // (or copy from `term` if it had a getter)

        // To test `term`'s current attributes, we'd have to print a character in `term`
        // and then check `get_glyph_at(term, term.cursor_pos().0 -1, term.cursor_pos().1).attr`
        // This is complex for a generic assert helper.
        // For now, this helper is more of a placeholder for how one *would* assert attributes
        // if a direct getter existed or if tests are structured to print and check.
        // Let's assume for robust SGR tests, we will print a char and check its glyph.
        // So, this specific helper might not be used directly, but the principle applies.
        log::warn!("assert_attributes helper is conceptual; actual SGR tests will print and check glyphs. Message: {}", message);
    }


    // --- Initialization Tests ---

    #[test]
    fn test_new_terminal_initial_state() {
        let term = new_term(80, 24);
        assert_eq!(term.dimensions(), (80, 24), "Initial dimensions");
        assert_cursor_pos(&term, 0, 0, "Initial cursor position");
        assert_screen_cursor_pos(&term, 0, 0, "Initial screen cursor position");
        assert!(term.is_cursor_visible(), "Cursor initially visible");
        assert!(!term.is_alt_screen_active(), "Initially not on alt screen");

        // Check a corner glyph
        let default_glyph_attrs = Attributes {
            fg: Color::Named(NamedColor::White), // Default from glyph.rs
            bg: Color::Named(NamedColor::Black), // Default from glyph.rs
            flags: AttrFlags::empty(),
        };
        assert_eq!(get_glyph_at(&term, 79, 23), Glyph { c: ' ', attr: default_glyph_attrs }, "Initial screen content (corner)");

        // Check initial dirty state - all lines should be dirty
        let mut term_mut = new_term(80, 24); // mutable for take_dirty_lines
        let dirty_lines = term_mut.take_dirty_lines();
        assert_eq!(dirty_lines.len(), 24, "All lines initially dirty");
        assert_eq!(dirty_lines, (0..24).collect::<Vec<usize>>(), "Correct dirty line indices");
    }

    #[test]
    fn test_new_terminal_minimum_dimensions() {
        let term = new_term(0, 0); // Should be clamped by Screen::new
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
        process_ansi_bytes(&mut term, b"Hello"); // Prints "Hello", cursor at (5,0) -> next char will wrap
        assert_cursor_pos(&term, 5, 0, "Cursor after 'Hello' before wrap");

        process_ansi_bytes(&mut term, b"W"); // Prints "W" at (0,1)
        assert_cursor_pos(&term, 1, 1, "Cursor after 'W' post-wrap");
        let screen = screen_to_string_vec(&term);
        assert_eq!(screen[0], "Hello", "Line 0 after wrap");
        assert_eq!(screen[1], "W    ", "Line 1 after wrap");
    }

    #[test]
    fn test_print_chars_with_scroll() {
        let mut term = new_term(5, 2);
        process_ansi_bytes(&mut term, b"Line1");
        process_ansi_bytes(&mut term, b"\n"); // Moves to (0,1)
        process_ansi_bytes(&mut term, b"Line2"); // Fills line 1

        // Next print will cause scroll
        process_ansi_bytes(&mut term, b"\n"); // Moves to (0,1) on scrolled screen
        process_ansi_bytes(&mut term, b"New");

        assert_cursor_pos(&term, 3, 1, "Cursor after scroll and 'New'");
        let screen = screen_to_string_vec(&term);
        assert_eq!(screen[0], "Line2", "Line 0 after scroll (was Line2)");
        assert_eq!(screen[1], "New  ", "Line 1 after scroll (is New)");
    }

    #[test]
    fn test_print_utf8_multibyte_char() {
        let mut term = new_term(5, 1);
        process_ansi_bytes(&mut term, "你好".as_bytes()); // "Nǐ hǎo"
        assert_eq!(get_glyph_at(&term, 0, 0).c, '你', "First char 你");
        assert_eq!(get_glyph_at(&term, 1, 0).c, '好', "Second char 好");
        // Assuming these are single-width for now by wcwidth default for CJK in some locales
        // If they are wide, cursor would be at 4,0.
        // The exact cursor position depends on `get_char_display_width` behavior.
        // Let's assume they are treated as width 1 for this basic test.
        // A dedicated wide char test is needed.
        let width_ni = crate::term::unicode::get_char_display_width('你');
        let width_hao = crate::term::unicode::get_char_display_width('好');
        assert_cursor_pos(&term, width_ni + width_hao, 0, "Cursor after '你好'");
    }

    #[test]
    fn test_print_wide_character_cjk() {
        let mut term = new_term(5, 1);
        // '世' (U+4E16) is typically a wide character.
        let wide_char = '世';
        let char_width = crate::term::unicode::get_char_display_width(wide_char);
        // This test's success depends heavily on the test environment's locale for wcwidth.
        // If it's not a CJK locale, it might return 1.
        // Forcing a known wide char like a full-width space might be more portable if needed.
        // Or, we accept that this test might behave differently based on `wcwidth`'s locale.
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
        let mut term = new_term(2, 2); // Width 2
        let wide_char = '世'; // Assume width 2
        let char_width = crate::term::unicode::get_char_display_width(wide_char);
        if char_width != 2 {
             log::warn!("Wide character '世' is not treated as width 2 in this environment (width: {}). Skipping wide char edge test.", char_width);
            return;
        }

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print(wide_char)));
        // Should print '世' at (0,0) and '\0' at (1,0). Cursor moves to (0,1) due to wrap.
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
        let mut term = new_term(20, 1); // Default tabs at 0, 8, 16
        assert_cursor_pos(&term, 0, 0, "Initial");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 8, 0, "Tab from 0 to 8");
        term.cursor.x = 5; // Manually move cursor for test
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 8, 0, "Tab from 5 to 8");
        term.cursor.x = 8;
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 16, 0, "Tab from 8 to 16");
        term.cursor.x = 17;
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
    // (A selection of important ones; more can be added)

    #[test]
    fn test_csi_cup_cursor_position() {
        let mut term = new_term(10, 5);
        // CUP with no params (defaults to 1,1)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,1))));
        assert_cursor_pos(&term, 0, 0, "CUP default (1,1)");

        // CUP with row and col
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(3,4)))); // Moves to (2,3) 0-indexed
        assert_cursor_pos(&term, 3, 2, "CUP R=3, C=4");

        // CUP with row only (col defaults to 1)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(5,1))));
        assert_cursor_pos(&term, 0, 4, "CUP R=5 (col defaults to 1)");

        // CUP out of bounds (should clamp)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(100,100))));
        assert_cursor_pos(&term, 9, 4, "CUP out of bounds (100,100 -> 9,4)");
    }

    #[test]
    fn test_csi_sgr_set_graphics_rendition() {
        let mut term = new_term(10,1);
        let default_attrs = Attributes::default();

        // Test Bold
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Bold]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('B')));
        assert_eq!(get_glyph_at(&term, 0, 0).attr.flags, AttrFlags::BOLD, "SGR Bold");

        // Test Red Foreground
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(AnsiColor::Red)]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('R'))); // Prints at (1,0)
        assert_eq!(get_glyph_at(&term, 1, 0).attr.fg, Color::Named(NamedColor::Red), "SGR Red FG");
        assert_eq!(get_glyph_at(&term, 1, 0).attr.flags, AttrFlags::BOLD, "SGR Bold still active"); // Bold should persist

        // Test Reset
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('N'))); // Prints at (2,0)
        assert_eq!(get_glyph_at(&term, 2, 0).attr, default_attrs, "SGR Reset");

        // Test multiple attributes
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Underline,
            Attribute::Background(AnsiColor::Blue),
        ]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('X'))); // Prints at (3,0)
        let attrs_x = get_glyph_at(&term, 3, 0).attr;
        assert!(attrs_x.flags.contains(AttrFlags::UNDERLINE), "SGR Underline");
        assert_eq!(attrs_x.bg, Color::Named(NamedColor::Blue), "SGR Blue BG");
        assert_eq!(attrs_x.fg, default_attrs.fg, "SGR FG remains default after multi-attr");
    }

    #[test]
    fn test_csi_ed_erase_in_display() {
        let mut term = new_term(5,3);
        process_ansi_bytes(&mut term, b"11111\n22222\n33333");
        // Cursor is now at (0,2) after the implicit LF from last print line.
        // Let's explicitly move it for the test.
        term.cursor.x = 2; term.cursor.y = 1; // Cursor at (2,1) on '2' in "22222"
        term.sync_emulator_state_to_screen_for_processing(); // Sync logical to screen
        term.sync_screen_state_to_emulator_after_processing();

        // ED 0: Erase from cursor to end of screen
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(0))));
        let screen0 = screen_to_string_vec(&term);
        assert_eq!(screen0[0], "11111", "ED 0: Line 0 unaffected");
        assert_eq!(screen0[1], "22   ", "ED 0: Line 1 erased from cursor"); // "22" then 3 spaces
        assert_eq!(screen0[2], "     ", "ED 0: Line 2 erased fully");

        // Reset screen content
        process_ansi_bytes(&mut term, b"\x1b[2J"); // Clear screen
        process_ansi_bytes(&mut term, b"11111\n22222\n33333");
        term.cursor.x = 2; term.cursor.y = 1;
        term.sync_emulator_state_to_screen_for_processing();
        term.sync_screen_state_to_emulator_after_processing();

        // ED 1: Erase from start of screen to cursor
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(1))));
        let screen1 = screen_to_string_vec(&term);
        assert_eq!(screen1[0], "     ", "ED 1: Line 0 erased fully");
        assert_eq!(screen1[1], "   22", "ED 1: Line 1 erased to cursor"); // 3 spaces then "22"
        assert_eq!(screen1[2], "33333", "ED 1: Line 2 unaffected");

        // ED 2: Erase entire screen
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))));
        let screen2 = screen_to_string_vec(&term);
        assert_eq!(screen2[0], "     ", "ED 2: Line 0 erased");
        assert_eq!(screen2[1], "     ", "ED 2: Line 1 erased");
        assert_eq!(screen2[2], "     ", "ED 2: Line 2 erased");
        // ED 2 often moves cursor to home, but our current impl doesn't explicitly do that for ED itself.
        // The cursor remains where it was unless the command specifically moves it.
    }

    #[test]
    fn test_dec_mode_origin_decom() {
        let mut term = new_term(10, 5);
        // Set scrolling region: CSI 2;4 r  (0-indexed: top=1, bot=3)
        // This command is `CSI Pn ; Pn r` for DECSTBM
        // In our CsiCommand enum, this needs to be represented.
        // Assuming a variant like `SetScrollingRegion { top: u16, bottom: u16 }`
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetScrollingRegion{top: 2, bottom: 4})));
        assert_cursor_pos(&term, 0, 0, "Cursor home after DECSTBM (origin mode off)");

        // Enable origin mode: CSI ?6h
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(6))));
        assert_cursor_pos(&term, 0, 0, "Cursor home (logical 0,0) after DECOM on");
        // Screen cursor should be at absolute (0,1) because logical (0,0) is top of margin
        assert_screen_cursor_pos(&term, 0, 1, "Screen cursor at margin top after DECOM on");

        // CUP 1,1 (relative to margin) -> should go to absolute (0,1)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,1))));
        assert_cursor_pos(&term, 0, 0, "CUP 1,1 (logical) with DECOM");
        assert_screen_cursor_pos(&term, 0, 1, "CUP 1,1 (screen) with DECOM");

        // CUP 3,3 (relative) -> screen (2, 1+2=3)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(3,3))));
        assert_cursor_pos(&term, 2, 2, "CUP 3,3 (logical) with DECOM"); // (col 2, row 2 relative to margin)
        assert_screen_cursor_pos(&term, 2, 3, "CUP 3,3 (screen) with DECOM"); // (col 2, abs row 3)

        // Disable origin mode: CSI ?6l
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(6))));
        assert_cursor_pos(&term, 0, 0, "Cursor home after DECOM off");
        assert_screen_cursor_pos(&term, 0, 0, "Screen cursor home after DECOM off");

        // CUP 2,3 (absolute)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2,3))));
        assert_cursor_pos(&term, 2, 1, "CUP 2,3 absolute after DECOM off");
    }

    // --- Resize Tests ---
    #[test]
    fn test_resize_larger_clamps_cursor_correctly() {
        let mut term = new_term(5, 2);
        process_ansi_bytes(&mut term, b"12345\nAB"); // Cursor at (2,1)
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
        process_ansi_bytes(&mut term, b"1234567890\nabcdefghij\nABCDEFGHIJ\nqwerty"); // Cursor (6,3)
        assert_cursor_pos(&term, 6, 3, "Cursor before resize");

        term.resize(5, 2);

        assert_eq!(term.dimensions(), (5, 2), "Dimensions after resize smaller");
        assert_cursor_pos(&term, 4, 1, "Cursor clamped after resize smaller"); // Max (4,1)
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
        let event = BackendEvent::Key { keysym: 0xFF0D, text: "\r".to_string() }; // Typical Enter
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

        let event_up = BackendEvent::Key { keysym: 0xFF52, text: "".to_string() }; // Up arrow
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
        // Enable DECCKM (Cursor Keys Application Mode)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(1))));
        assert!(term.dec_modes.cursor_keys_app_mode, "DECCKM should be enabled");

        let event_down = BackendEvent::Key { keysym: 0xFF54, text: "".to_string() }; // Down arrow
        let action_down = term.interpret_input(EmulatorInput::User(event_down));
        if let Some(EmulatorAction::WritePty(bytes)) = action_down {
            assert_eq!(bytes, b"\x1bOB", "WritePty for Down Arrow (Application)");
        } else {
            panic!("Expected WritePty for Down Arrow (Application)");
        }
    }

    // TODO: Many more tests:
    // - More C0 controls (VT, FF, SO, SI)
    // - More ESC sequences (RIS, charsets thoroughly)
    // - More CSI sequences:
    //   - Scrolling (SU, SD) with regions
    //   - Editing (ICH, DCH, IL, DL) with regions, wide chars
    //   - Tabulation (TBC, HTS via ESC H)
    //   - All SGR attributes and their combinations, including 256-color and RGB.
    //   - All DEC private modes and their interactions (DECAWM, mouse modes if added)
    //   - Status reports (DSR variants)
    // - OSC commands beyond title setting.
    // - Interactions between modes (e.g., printing in origin mode with auto-wrap)
    // - UTF-8 sequences, including invalid/incomplete ones.
    // - Behavior of `take_dirty_lines()` after specific operations.
    // - `get_glyph()` for cells affected by wide chars, reverse video, etc.
    // - `is_alt_screen_active()` after 1047/1049 sequences.
    // - Test sequences with maximum parameters, empty parameters, zero parameters.
}

