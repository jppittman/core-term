// src/term/tests.rs

//! Unit tests for the main TerminalEmulator struct and its core logic.
//! These tests aim to verify the internal state changes and actions produced
//! by the TerminalEmulator based on various inputs, adhering to its public API.

// Module for tests related to the Term struct itself
#[cfg(test)]
mod term_tests {
    use crate::term::{
        TerminalEmulator, EmulatorInput, EmulatorAction,
        DecModeConstant, 
        TerminalInterface, // Import the trait to bring its methods into scope
    };
    use crate::glyph::{Attributes, Color, AttrFlags, Glyph, NamedColor};
    use crate::ansi::commands::{AnsiCommand, C0Control, CsiCommand, Attribute, Color as AnsiColor};
    use crate::backends::BackendEvent;

    use test_log::test;


    // --- Test Helpers ---

    fn new_term(cols: usize, rows: usize) -> TerminalEmulator {
        TerminalEmulator::new(cols, rows, 100) 
    }

    fn process_input(term: &mut TerminalEmulator, input: EmulatorInput) -> Option<EmulatorAction> {
        let action = term.interpret_input(input);
        // Accessing take_dirty_lines via the TerminalInterface trait method
        if action.is_none() && !TerminalInterface::take_dirty_lines(term).is_empty() {
            return Some(EmulatorAction::RequestRedraw);
        }
        action
    }
    
    fn process_commands(term: &mut TerminalEmulator, commands: Vec<AnsiCommand>) -> Vec<EmulatorAction> {
        let mut actions = Vec::new();
        for cmd in commands {
            if let Some(action) = term.interpret_input(EmulatorInput::Ansi(cmd)) {
                actions.push(action);
            }
        }
        // Accessing take_dirty_lines via the TerminalInterface trait method
        if !TerminalInterface::take_dirty_lines(term).is_empty() && !actions.iter().any(|a| matches!(a, EmulatorAction::RequestRedraw)) {
             actions.push(EmulatorAction::RequestRedraw);
        }
        actions
    }

    fn screen_to_string_vec(term: &TerminalEmulator) -> Vec<String> {
        // Accessing dimensions and get_glyph via the TerminalInterface trait methods
        let (cols, rows) = TerminalInterface::dimensions(term);
        let mut result = Vec::with_capacity(rows);
        for y in 0..rows {
            let line: String = (0..cols).map(|x| TerminalInterface::get_glyph(term, x, y).c).collect();
            result.push(line);
        }
        result
    }

    fn get_glyph_at(term: &TerminalEmulator, x: usize, y: usize) -> Glyph {
        // Accessing get_glyph via the TerminalInterface trait method
        TerminalInterface::get_glyph(term, x, y)
    }

    fn assert_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        assert_eq!(term.cursor_pos(), (x, y), "Logical cursor position check: {}", message);
    }

    fn assert_screen_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        // Accessing get_screen_cursor_pos via the TerminalInterface trait method
        assert_eq!(TerminalInterface::get_screen_cursor_pos(term), (x,y), "Screen cursor position check: {}", message);
    }

    // --- Initialization Tests ---

    #[test]
    fn test_new_terminal_initial_state() {
        let term = new_term(80, 24);
        // Accessing dimensions and is_cursor_visible via the TerminalInterface trait methods
        assert_eq!(TerminalInterface::dimensions(&term), (80, 24), "Initial dimensions");
        assert_cursor_pos(&term, 0, 0, "Initial logical cursor position");
        assert_screen_cursor_pos(&term, 0, 0, "Initial screen cursor position");
        assert!(TerminalInterface::is_cursor_visible(&term), "Cursor initially visible (DECTCEM is on by default)");
        assert!(!term.is_alt_screen_active(), "Initially not on alt screen");

        let expected_initial_attrs = Attributes::default(); 
        assert_eq!(get_glyph_at(&term, 0, 0).attr, expected_initial_attrs, "Initial cell attributes at (0,0)");
        assert_eq!(get_glyph_at(&term, 79, 23).attr, expected_initial_attrs, "Initial cell attributes at (79,23)");

        let mut term_mut_for_print = new_term(1,1);
        process_input(&mut term_mut_for_print, EmulatorInput::Ansi(AnsiCommand::Print('T')));
        assert_eq!(get_glyph_at(&term_mut_for_print, 0,0).attr, expected_initial_attrs, "Initial cursor attributes for printing");

        let mut term_mut_dirty = new_term(80, 24);
        // Accessing take_dirty_lines via the TerminalInterface trait method
        let dirty_lines = TerminalInterface::take_dirty_lines(&mut term_mut_dirty);
        assert_eq!(dirty_lines.len(), 24, "All lines initially dirty");
        assert_eq!(dirty_lines, (0..24).collect::<Vec<usize>>(), "Correct dirty line indices");
    }
    
    #[test]
    fn test_new_terminal_minimum_dimensions() {
        let term = new_term(0, 0);
        assert_eq!(TerminalInterface::dimensions(&term).0, 1, "Minimum width clamped to 1");
        assert_eq!(TerminalInterface::dimensions(&term).1, 1, "Minimum height clamped to 1");
    }


    // --- SGR and Attribute Tests (Focus for "White Box" RCA) ---

    #[test]
    fn test_sgr_reset_restores_default_attributes_and_applies_to_new_chars() {
        let mut term = new_term(10, 1);
        let default_attrs = Attributes::default();

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(AnsiColor::Red),
            Attribute::Background(AnsiColor::Blue),
            Attribute::Bold,
        ]))));
        
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('X')));
        let glyph_x = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph_x.c, 'X');
        assert_eq!(glyph_x.attr.fg, Color::Named(NamedColor::Red), "X FG should be Red");
        assert_eq!(glyph_x.attr.bg, Color::Named(NamedColor::Blue), "X BG should be Blue");
        assert!(glyph_x.attr.flags.contains(AttrFlags::BOLD), "X flag should contain BOLD");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]))));

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('Y')));
        let glyph_y = get_glyph_at(&term, 1, 0); 
        assert_eq!(glyph_y.c, 'Y');
        assert_eq!(glyph_y.attr.fg, default_attrs.fg, "Y FG should be Default after SGR Reset");
        assert_eq!(glyph_y.attr.bg, default_attrs.bg, "Y BG should be Default after SGR Reset");
        assert_eq!(glyph_y.attr.flags, default_attrs.flags, "Y Flags should be Default after SGR Reset");
        
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,3)))); 
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0)))); 
        // Accessing dimensions via the TerminalInterface trait method
        for x_idx in 2..TerminalInterface::dimensions(&term).0 { 
            let erased_glyph = get_glyph_at(&term, x_idx, 0);
            assert_eq!(erased_glyph.attr, default_attrs, "Erased cell at ({},0) should have default attributes after SGR Reset", x_idx);
        }
    }

    #[test]
    fn test_reverse_attribute_impact_on_printed_char() {
        let mut term = new_term(5, 1);
        let default_attrs = Attributes::default();

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        let glyph_a = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph_a.attr.fg, default_attrs.fg, "A FG should be Default");
        assert_eq!(glyph_a.attr.bg, default_attrs.bg, "A BG should be Default");
        assert!(!glyph_a.attr.flags.contains(AttrFlags::REVERSE), "A should not have REVERSE flag initially");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reverse]))));
        
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('B')));
        let glyph_b = get_glyph_at(&term, 1, 0);
        assert_eq!(glyph_b.attr.fg, default_attrs.fg, "B FG should be Default (renderer will swap for display)");
        assert_eq!(glyph_b.attr.bg, default_attrs.bg, "B BG should be Default (renderer will swap for display)");
        assert!(glyph_b.attr.flags.contains(AttrFlags::REVERSE), "B should have REVERSE flag set");
        assert!(!glyph_b.attr.flags.contains(AttrFlags::BOLD), "B should not have BOLD flag implicitly");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::NoReverse]))));
        
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('C')));
        let glyph_c = get_glyph_at(&term, 2, 0);
        assert_eq!(glyph_c.attr.fg, default_attrs.fg, "C FG should be Default");
        assert_eq!(glyph_c.attr.bg, default_attrs.bg, "C BG should be Default");
        assert!(!glyph_c.attr.flags.contains(AttrFlags::REVERSE), "C should not have REVERSE flag after SGR 27");
    }

    #[test]
    fn test_explicit_fg_bg_with_reverse() {
        let mut term = new_term(5, 1);
        let red_fg = Color::Named(NamedColor::Red);
        let blue_bg = Color::Named(NamedColor::Blue);

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(AnsiColor::Red),
            Attribute::Background(AnsiColor::Blue),
        ]))));

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reverse]))));
        
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('R')));
        let glyph_r = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph_r.attr.fg, red_fg, "R FG should be Red (renderer will swap for display)");
        assert_eq!(glyph_r.attr.bg, blue_bg, "R BG should be Blue (renderer will swap for display)");
        assert!(glyph_r.attr.flags.contains(AttrFlags::REVERSE), "R should have REVERSE flag set");
        assert!(!glyph_r.attr.flags.contains(AttrFlags::BOLD), "R should not have BOLD flag implicitly");
    }
    
    #[test]
    fn test_screen_default_attributes_updated_by_sgr_for_clearing() {
        let mut term = new_term(5, 1);
        let default_attrs = Attributes::default();
        let red_fg = Color::Named(NamedColor::Red);
        let blue_bg = Color::Named(NamedColor::Blue);

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,3)))); 
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0)))); 
        assert_eq!(get_glyph_at(&term, 3, 0).attr, default_attrs, "Initial erase operation should use system default attributes");

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(AnsiColor::Red),
            Attribute::Background(AnsiColor::Blue),
        ]))));
        
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1,1)))); 
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(2)))); 
        
        let expected_erase_attrs = Attributes { fg: red_fg, bg: blue_bg, flags: AttrFlags::empty() };
        assert_eq!(get_glyph_at(&term, 0, 0).attr, expected_erase_attrs, "Erase after SGR (Red/Blue) should use Red FG, Blue BG for cleared cells");
        assert_eq!(get_glyph_at(&term, 1, 0).attr, expected_erase_attrs, "Another erased cell should also have Red FG, Blue BG");
    }

    #[test]
    fn test_initial_screen_attributes_are_default() {
        let term = new_term(10, 5);
        let glyph = get_glyph_at(&term, 0, 0);
        let expected_attrs = Attributes::default(); 
        
        assert_eq!(glyph.c, ' ', "Initial char should be space");
        assert_eq!(glyph.attr.fg, expected_attrs.fg, "Initial FG should be Color::Default");
        assert_eq!(glyph.attr.bg, expected_attrs.bg, "Initial BG should be Color::Default");
        assert_eq!(glyph.attr.flags, expected_attrs.flags, "Initial flags should be empty");
    }

    #[test]
    fn test_erase_in_line_uses_current_sgr_attributes() {
        let mut term = new_term(10, 2);
        let default_attrs = Attributes::default();

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(AnsiColor::Green), 
            Attribute::Background(AnsiColor::Magenta),
        ]))));
        process_commands(&mut term, vec![AnsiCommand::Print('A'), AnsiCommand::Print('B'), AnsiCommand::Print('C')]); 
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0)))); 

        let expected_attrs_magenta_bg = Attributes {
            fg: Color::Named(NamedColor::Green),
            bg: Color::Named(NamedColor::Magenta),
            flags: AttrFlags::empty(),
        };

        for x in 3..10 {
            let glyph = get_glyph_at(&term, x, 0);
            assert_eq!(glyph.c, ' ', "EL(0) char at ({},0) should be space", x);
            assert_eq!(glyph.attr, expected_attrs_magenta_bg, "EL(0) attrs at ({},0) should be Green FG / Magenta BG", x);
        }

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2,1)))); 
        process_commands(&mut term, vec![AnsiCommand::Print('D'), AnsiCommand::Print('E'), AnsiCommand::Print('F')]); 
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0)))); 

        for x in 3..10 {
            let glyph = get_glyph_at(&term, x, 1);
            assert_eq!(glyph.c, ' ', "EL(0) char at ({},1) after SGR Reset should be space", x);
            assert_eq!(glyph.attr, default_attrs, "EL(0) BG at ({},1) after SGR Reset should be Default", x);
        }
    }

    #[test]
    fn test_erase_in_display_uses_current_sgr_attributes() {
        let mut term = new_term(5, 3);
        let default_attrs = Attributes::default();

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(AnsiColor::Cyan),
            Attribute::Background(AnsiColor::Yellow),
        ]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2)))); 

        let expected_attrs_yellow_bg = Attributes {
            fg: Color::Named(NamedColor::Cyan),
            bg: Color::Named(NamedColor::Yellow),
            flags: AttrFlags::empty(),
        };

        for y in 0..3 {
            for x in 0..5 {
                let glyph = get_glyph_at(&term, x, y);
                assert_eq!(glyph.c, ' ', "ED(2) char at ({},{}) should be space", x, y);
                assert_eq!(glyph.attr, expected_attrs_yellow_bg, "ED(2) attrs at ({},{}) should be Cyan FG / Yellow BG", x,y);
            }
        }

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2)))); 

        for y in 0..3 {
            for x in 0..5 {
                let glyph = get_glyph_at(&term, x, y);
                assert_eq!(glyph.c, ' ', "ED(2) char at ({},{}) after SGR Reset should be space", x, y);
                assert_eq!(glyph.attr, default_attrs, "ED(2) attrs at ({},{}) after SGR Reset should be Default", x,y);
            }
        }
    }
    
    // --- Restructured Placeholder Tests ---
    // Each test now has a clear body and a basic assertion.

    #[test]
    fn test_c0_bs_backspace_moves_cursor_left() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Cursor at (1,0)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)));
        assert_cursor_pos(&term, 0, 0, "BS should move cursor from (1,0) to (0,0)");
    }

    #[test]
    fn test_c0_ht_horizontal_tab_moves_to_next_tab_stop() {
        let mut term = new_term(20, 1); // Default tab stop at 8
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)));
        assert_cursor_pos(&term, 8, 0, "HT from column 0 should move to column 8");
    }

    #[test]
    fn test_c0_lf_line_feed_moves_down_and_to_col0() {
        let mut term = new_term(5, 2);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Cursor at (1,0)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)));
        assert_cursor_pos(&term, 0, 1, "LF from (1,0) should move to (0,1)");
    }

    #[test]
    fn test_c0_cr_carriage_return_moves_to_col0() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Cursor at (1,0)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)));
        assert_cursor_pos(&term, 0, 0, "CR from (1,0) should move to (0,0)");
    }

    #[test]
    fn test_c0_bel_produces_ring_bell_action() {
        let mut term = new_term(5, 1);
        let action = process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BEL)));
        assert_eq!(action, Some(EmulatorAction::RingBell), "BEL should produce RingBell action");
    }

    #[test]
    fn test_csi_cup_cursor_position_moves_cursor() {
        let mut term = new_term(10, 5);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(3, 4)))); // 1-based: row 3, col 4
        assert_cursor_pos(&term, 3, 2, "CUP(3,4) should move cursor to (3,2) (0-based)");
    }

    #[test]
    fn test_csi_sgr_set_graphics_rendition_bold_applies_to_glyph() {
        let mut term = new_term(10, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Bold]))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        let glyph = get_glyph_at(&term, 0, 0);
        assert!(glyph.attr.flags.contains(AttrFlags::BOLD), "Printed char 'A' should have BOLD attribute");
    }

    #[test]
    fn test_csi_ed_erase_in_display_all_clears_screen() {
        let mut term = new_term(5, 3);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Put something on screen
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2)))); // Erase All
        assert_eq!(get_glyph_at(&term, 0, 0).c, ' ', "Cell (0,0) should be space after ED(2)");
        assert_eq!(get_glyph_at(&term, 0, 0).attr, Attributes::default(), "Cell (0,0) should have default attrs after ED(2)");
    }

    #[test]
    fn test_dec_mode_origin_decom_set_moves_cursor_to_margin() {
        let mut term = new_term(10, 5);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetScrollingRegion{top: 2, bottom: 4}))); // Scroll region 1-3 (0-based)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(DecModeConstant::Origin as u16))));
        assert_screen_cursor_pos(&term, 0, 1, "Cursor after DECOM set should be at physical (0,1) (top of margin)");
    }

    #[test]
    fn test_resize_larger_maintains_cursor_logical_pos() {
        let mut term = new_term(5, 2);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2, 2)))); // Cursor at (1,1)
        term.resize(10, 4);
        assert_cursor_pos(&term, 1, 1, "Cursor logical pos (1,1) should be maintained after resize larger");
    }

    #[test]
    fn test_resize_smaller_clamps_cursor_logical_pos() {
        let mut term = new_term(10, 4);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(8, 4)))); // Cursor at (7,3)
        term.resize(5, 2); // New max (4,1)
        assert_cursor_pos(&term, 4, 1, "Cursor logical pos (7,3) should be clamped to (4,1) after resize smaller");
    }

    #[test]
    fn test_user_input_printable_key_generates_write_pty_action() {
        let mut term = new_term(5, 1);
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key{keysym: 'A' as u32, text: "A".to_string()}));
        assert_eq!(action, Some(EmulatorAction::WritePty(b"A".to_vec())), "Printable key 'A' should produce WritePty('A')");
    }

    #[test]
    fn test_user_input_enter_key_generates_write_pty_action() {
        let mut term = new_term(5, 1);
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key{keysym: 0xFF0D, text: "\r".to_string()}));
        assert_eq!(action, Some(EmulatorAction::WritePty(b"\r".to_vec())), "Enter key should produce WritePty('\\r')");
    }

    #[test]
    fn test_user_input_arrow_key_normal_mode_sends_csi_sequence() {
        let mut term = new_term(5, 1);
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key{keysym: 0xFF52, text: "".to_string()})); // Up Arrow
        assert_eq!(action, Some(EmulatorAction::WritePty(b"\x1b[A".to_vec())), "Up Arrow in normal mode should send CSI A");
    }

    #[test]
    fn test_user_input_arrow_key_application_mode_sends_ss3_sequence() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(DecModeConstant::CursorKeys as u16)))); // Enable DECCKM
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key{keysym: 0xFF54, text: "".to_string()})); // Down Arrow
        assert_eq!(action, Some(EmulatorAction::WritePty(b"\x1bOB".to_vec())), "Down Arrow in app mode should send SS3 B");
    }

    #[test]
    fn test_dec_private_mode_12_att610_cursor_blink_is_processed() {
        let mut term = new_term(5, 1);
        // Just ensure it doesn't panic and is processed. Actual blink state is visual.
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(12))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(12))));
    }

    #[test]
    fn test_dec_private_mode_2004_bracketed_paste_is_set_and_reset() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(2004))));
        assert!(term.is_bracketed_paste_mode_active(), "Bracketed paste mode should be active after ?2004h");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(2004))));
        assert!(!term.is_bracketed_paste_mode_active(), "Bracketed paste mode should be inactive after ?2004l");
    }

    #[test]
    fn test_dec_private_mouse_mode_1000_is_set_and_reset() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(1000)))); // X10 Mouse
        assert!(term.is_mouse_mode_active(1000), "Mouse mode 1000 should be active");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(1000))));
        assert!(!term.is_mouse_mode_active(1000), "Mouse mode 1000 should be inactive");
    }

    #[test]
    fn test_dec_private_mode_focus_event_1004_is_set_and_reset() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(1004))));
        assert!(term.is_focus_event_mode_active(), "Focus event mode 1004 should be active");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(1004))));
        assert!(!term.is_focus_event_mode_active(), "Focus event mode 1004 should be inactive");
    }

    #[test]
    fn test_dec_private_mode_7727_is_processed() {
        let mut term = new_term(5, 1);
        // Ensure it doesn't panic. Specific behavior for 7727 is often undefined or ignored.
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(7727))));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(7727))));
    }

    #[test]
    fn test_csi_sp_q_decscusr_set_cursor_style_updates_shape() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetCursorStyle{shape: 4}))); // Underline cursor
        assert_eq!(term.get_cursor_shape(), 4, "Cursor shape should be updated by DECSCUSR");
    }

    #[test]
    fn test_csi_t_window_manipulation_report_chars_produces_action() {
        let mut term = new_term(80, 24);
        let actions = process_commands(&mut term, vec![AnsiCommand::Csi(CsiCommand::WindowManipulation{ps1: 18, ps2: None, ps3: None})]);
        let expected_response = format!("\x1b[8;{};{}t", 24, 80);
        assert!(actions.contains(&EmulatorAction::WritePty(expected_response.into_bytes())), "CSI 18 t should report char dimensions");
    }

    #[test]
    fn test_csi_t_window_manipulation_unsupported_is_graceful() {
        let mut term = new_term(80, 24);
        // Ensure no panic for an unsupported window manipulation command.
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::WindowManipulation{ps1: 99, ps2: Some(0), ps3: Some(0)})));
    }

    #[test]
    fn test_print_ascii_chars_updates_grid() {
        let mut term = new_term(10, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('H')));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('i')));
        assert_eq!(get_glyph_at(&term, 0, 0).c, 'H', "First char 'H' should be at (0,0)");
        assert_eq!(get_glyph_at(&term, 1, 0).c, 'i', "Second char 'i' should be at (1,0)");
        assert_cursor_pos(&term, 2, 0, "Cursor should be at (2,0) after 'Hi'");
    }

    #[test]
    fn test_print_chars_with_line_wrap_moves_to_next_line() {
        let mut term = new_term(2, 2); // Narrow terminal
        process_commands(&mut term, vec![AnsiCommand::Print('A'), AnsiCommand::Print('B')]); // Fills line 0
        assert_cursor_pos(&term, 2, 0, "Cursor at end of line 0 before wrap");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('C'))); // This char should wrap
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'C', "Char 'C' should be on line 1 after wrap");
        assert_cursor_pos(&term, 1, 1, "Cursor should be at (1,1) after 'C' wrapped");
    }

    #[test]
    fn test_print_chars_with_scroll_moves_content_up() {
        let mut term = new_term(1, 2); // 1 col, 2 rows
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Line 0: A
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // Cursor to (0,1)
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('B'))); // Line 1: B
        assert_eq!(get_glyph_at(&term, 0, 0).c, 'A', "Line 0 should be 'A' before scroll");
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'B', "Line 1 should be 'B' before scroll");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF))); // Scroll up
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('C'))); // Line 1: C (new)
        assert_eq!(get_glyph_at(&term, 0, 0).c, 'B', "Line 0 should be 'B' after scroll");
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'C', "Line 1 should be 'C' after scroll");
    } 

    #[test]
    fn test_print_utf8_multibyte_char_occupies_correct_cells() {
        let mut term = new_term(5, 1);
        process_commands(&mut term, vec![AnsiCommand::Print('你'), AnsiCommand::Print('好')]);
        assert_eq!(get_glyph_at(&term, 0, 0).c, '你', "First CJK char '你' at (0,0)");
        assert_eq!(get_glyph_at(&term, 1, 0).c, '\0', "Placeholder for '你' at (1,0)"); // Assuming CJK are wide
        assert_eq!(get_glyph_at(&term, 2, 0).c, '好', "Second CJK char '好' at (2,0)");
        assert_eq!(get_glyph_at(&term, 3, 0).c, '\0', "Placeholder for '好' at (3,0)");
        assert_cursor_pos(&term, 4, 0, "Cursor after '你好'");
    }

    #[test]
    fn test_print_wide_character_cjk_advances_cursor_by_width() {
        let mut term = new_term(5, 1);
        let wide_char = '世';
        let char_width = crate::term::unicode::get_char_display_width(wide_char);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print(wide_char)));
        assert_eq!(get_glyph_at(&term, 0, 0).c, wide_char, "Wide char '世' should be at (0,0)");
        if char_width == 2 {
            assert_eq!(get_glyph_at(&term, 1, 0).c, '\0', "Placeholder for '世' should be at (1,0)");
        }
        assert_cursor_pos(&term, char_width, 0, "Cursor should advance by char_width");
    }

    #[test]
    fn test_print_wide_character_at_edge_of_line_wraps_correctly() {
        let mut term = new_term(2, 2); // Width of 2
        let wide_char = '世'; // Assumed width 2
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print(wide_char))); // Fills line 0
        assert_cursor_pos(&term, 2, 0, "Cursor at end of line 0 before wrap for wide char");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A'))); // Should wrap
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'A', "Char 'A' should be on line 1 after wide char wrap");
        assert_cursor_pos(&term, 1, 1, "Cursor should be at (1,1) after 'A' wrapped");
    }
}

